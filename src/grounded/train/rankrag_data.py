"""M-6.6 — RankRAG rerank training examples from holdout reserve."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from grounded.eval.prompts_build import outline_from_body
from grounded.retrieve.rerank import MockReranker, rerank_chunks
from grounded.utils.incremental_jsonl import append_row, mark_processed


def load_id_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def holdout_reserve_ids(holdout_path: Path, eval_grid_path: Path) -> list[str]:
    holdout = set(load_id_list(holdout_path))
    grid = set(load_id_list(eval_grid_path))
    return sorted(holdout - grid)


def load_paper_meta(parsed_dir: Path, arxiv_id: str) -> dict[str, Any] | None:
    path = parsed_dir / f"{arxiv_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def index_chunks_by_paper(chunks_parquet: Path) -> dict[str, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in pq.read_table(
        chunks_parquet,
        columns=["chunk_id", "paper_id", "text"],
    ).to_pylist():
        by_paper.setdefault(row["paper_id"], []).append(row)
    return by_paper


def _query_terms(query: str) -> set[str]:
    import re

    return {tok for tok in re.findall(r"[a-z0-9]{4,}", query.lower())}


def _candidate_overlap(query: str, text: str) -> float:
    q_terms = _query_terms(query)
    if not q_terms:
        return 0.0
    text_terms = _query_terms(text)
    if not text_terms:
        return 0.0
    return len(q_terms & text_terms) / len(q_terms)


def _label_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    n_positives: int,
) -> list[int]:
    ranked = sorted(
        candidates,
        key=lambda row: _candidate_overlap(query, str(row.get("text", ""))),
        reverse=True,
    )
    positives = [
        row for row in ranked if _candidate_overlap(query, str(row.get("text", ""))) >= 0.12
    ][:n_positives]
    if not positives:
        positives = rerank_chunks(query, candidates, MockReranker(), top_k=min(n_positives, len(candidates)))
    positive_ids = {str(row.get("chunk_id", "")) for row in positives}
    return [1 if str(c.get("chunk_id", "")) in positive_ids else 0 for c in candidates]
    top_ids = {str(row.get("chunk_id", "")) for row in fallback}
    return [1 if str(c.get("chunk_id", "")) in top_ids else 0 for c in candidates]


def build_rankrag_examples(
    reserve_ids: list[str],
    parsed_dir: Path,
    chunks_by_paper: dict[str, list[dict[str, Any]]],
    index_paper_ids: set[str],
    *,
    n_candidates: int = 10,
    n_positives: int = 3,
    seed: int = 1337,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_index_chunks: list[dict[str, Any]] = []
    for pid in index_paper_ids:
        all_index_chunks.extend(chunks_by_paper.get(pid, []))

    ids = reserve_ids[:limit] if limit else reserve_ids
    examples: list[dict[str, Any]] = []
    for arxiv_id in ids:
        paper = load_paper_meta(parsed_dir, arxiv_id)
        if not paper or not paper.get("title") or not paper.get("abstract"):
            continue
        neg_pool = [c for c in all_index_chunks if c["paper_id"] != arxiv_id]
        if len(neg_pool) < n_candidates:
            continue
        query = f"{paper['title']}\n\n{outline_from_body(paper)}"
        sampled = rng.sample(neg_pool, min(len(neg_pool), max(n_candidates * 4, n_candidates)))
        candidates = sorted(
            sampled,
            key=lambda row: _candidate_overlap(query, str(row.get("text", ""))),
            reverse=True,
        )[:n_candidates]
        rng.shuffle(candidates)
        labels = _label_candidates(query, candidates, n_positives=n_positives)
        if not any(labels) or all(labels):
            continue
        examples.append(
            {
                "arxiv_id": arxiv_id,
                "query": query,
                "candidates": [c["text"][:1500] for c in candidates],
                "labels": labels,
            }
        )
    return examples


def build_rankrag_examples_to_jsonl(
    reserve_ids: list[str],
    parsed_dir: Path,
    chunks_by_paper: dict[str, list[dict[str, Any]]],
    index_paper_ids: set[str],
    out_path: Path,
    processed_path: Path,
    *,
    completed_ids: set[str] | None = None,
    n_candidates: int = 10,
    n_positives: int = 3,
    seed: int = 1337,
) -> dict[str, Any]:
    rng = random.Random(seed)
    completed = completed_ids or set()
    all_index_chunks: list[dict[str, Any]] = []
    for pid in index_paper_ids:
        all_index_chunks.extend(chunks_by_paper.get(pid, []))

    written = 0
    skipped = 0
    for arxiv_id in reserve_ids:
        if arxiv_id in completed:
            continue
        paper = load_paper_meta(parsed_dir, arxiv_id)
        if not paper or not paper.get("title") or not paper.get("abstract"):
            skipped += 1
            mark_processed(processed_path, arxiv_id)
            continue
        neg_pool = [c for c in all_index_chunks if c["paper_id"] != arxiv_id]
        if len(neg_pool) < n_candidates:
            skipped += 1
            mark_processed(processed_path, arxiv_id)
            continue
        query = f"{paper['title']}\n\n{outline_from_body(paper)}"
        sampled = rng.sample(neg_pool, min(len(neg_pool), max(n_candidates * 4, n_candidates)))
        candidates = sorted(
            sampled,
            key=lambda row: _candidate_overlap(query, str(row.get("text", ""))),
            reverse=True,
        )[:n_candidates]
        rng.shuffle(candidates)
        labels = _label_candidates(query, candidates, n_positives=n_positives)
        if not any(labels) or all(labels):
            skipped += 1
            mark_processed(processed_path, arxiv_id)
            continue
        append_row(
            out_path,
            {
                "arxiv_id": arxiv_id,
                "query": query,
                "candidates": [c["text"][:1500] for c in candidates],
                "labels": labels,
            },
        )
        mark_processed(processed_path, arxiv_id)
        written += 1
    return {"written": written, "skipped": skipped}


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
