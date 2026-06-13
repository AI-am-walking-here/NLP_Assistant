"""M-5.1 — build (title + outline [+ retrieval]) → abstract SFT JSONL."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Literal

from grounded.eval.factscore import SENTENCE_SPLIT
from grounded.eval.prompts_build import load_paper, outline_from_abstract, outline_from_body
from grounded.generate.prompts import render_abstract_prompt, render_sft_prompt
from grounded.utils.incremental_jsonl import append_row, mark_processed

logger = logging.getLogger(__name__)

OutlineSource = Literal["abstract", "body"]
PromptMode = Literal["no_retrieval", "with_retrieval", "mixed"]


def format_chat_example(system: str, user: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _use_retrieval_for_id(arxiv_id: str, mode: PromptMode, retrieval_fraction: float) -> bool:
    if mode == "with_retrieval":
        return True
    if mode == "no_retrieval":
        return False
    h = int(hashlib.sha256(arxiv_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) / 10_000.0 < retrieval_fraction


def _make_outline(paper: dict, outline_source: OutlineSource) -> str:
    if outline_source == "body":
        return outline_from_body(paper)
    return outline_from_abstract((paper.get("abstract") or "").strip())


_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def evidence_supported_target(
    abstract: str,
    retrieved: str,
    *,
    min_overlap: float = 0.18,
    max_sentences: int = 6,
) -> str:
    """Keep only target sentences with lexical support in retrieved cross-paper evidence."""
    evidence_tokens = _tokens(retrieved)
    if not evidence_tokens:
        return ""
    kept: list[str] = []
    for sent in SENTENCE_SPLIT.split(abstract.strip()):
        sent = re.sub(r"\s+", " ", sent).strip()
        if len(sent) < 30:
            continue
        sent_tokens = _tokens(sent)
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & evidence_tokens) / len(sent_tokens)
        if overlap >= min_overlap:
            kept.append(sent)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept)


def build_sft_row(
    aid: str,
    paper: dict,
    *,
    max_target_chars: int = 8000,
    outline_source: OutlineSource = "body",
    prompt_mode: PromptMode = "mixed",
    retrieval_fraction: float = 0.3,
    retrieval_top_k: int = 8,
) -> dict[str, Any] | None:
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    if not title or not abstract:
        return None
    if len(abstract) > max_target_chars:
        abstract = abstract[:max_target_chars]

    outline = _make_outline(paper, outline_source)
    use_retrieval = _use_retrieval_for_id(aid, prompt_mode, retrieval_fraction)

    if use_retrieval:
        from grounded.train.sft_retrieval import format_training_retrieval_block

        retrieved = format_training_retrieval_block(
            title, outline, aid, top_k=retrieval_top_k
        )
        supported = evidence_supported_target(abstract, retrieved)
        if not supported:
            logger.warning("Skip SFT %s: target abstract unsupported by retrieved evidence", aid)
            return None
        abstract = supported
        system, user = render_abstract_prompt(title, outline, retrieved)
        prompt_style = "with_retrieval"
    else:
        system, user = render_sft_prompt(title, outline)
        prompt_style = "no_retrieval"

    chat = format_chat_example(system, user, abstract)
    return {
        "arxiv_id": aid,
        "title": title,
        "outline": outline,
        "outline_source": outline_source,
        "prompt_style": prompt_style,
        "target_abstract": abstract,
        **chat,
    }


def build_sft_examples(
    arxiv_ids: list[str],
    parsed_dir: Path,
    *,
    max_target_chars: int = 8000,
    outline_source: OutlineSource = "body",
    prompt_mode: PromptMode = "mixed",
    retrieval_fraction: float = 0.3,
    retrieval_top_k: int = 8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, _skipped in iter_sft_examples(
        arxiv_ids,
        parsed_dir,
        max_target_chars=max_target_chars,
        outline_source=outline_source,
        prompt_mode=prompt_mode,
        retrieval_fraction=retrieval_fraction,
        retrieval_top_k=retrieval_top_k,
    ):
        if row:
            rows.append(row)
    return rows


def iter_sft_examples(
    arxiv_ids: list[str],
    parsed_dir: Path,
    *,
    max_target_chars: int = 8000,
    outline_source: OutlineSource = "body",
    prompt_mode: PromptMode = "mixed",
    retrieval_fraction: float = 0.3,
    retrieval_top_k: int = 8,
) -> Iterator[tuple[dict[str, Any] | None, str | None]]:
    for aid in arxiv_ids:
        paper = load_paper(parsed_dir, aid)
        if not paper:
            logger.warning("Skip SFT %s: missing parsed JSON", aid)
            yield None, aid
            continue
        row = build_sft_row(
            aid,
            paper,
            max_target_chars=max_target_chars,
            outline_source=outline_source,
            prompt_mode=prompt_mode,
            retrieval_fraction=retrieval_fraction,
            retrieval_top_k=retrieval_top_k,
        )
        if row:
            yield row, None
        else:
            logger.warning("Skip SFT %s: missing title/abstract or unsupported target", aid)
            yield None, aid


def build_sft_examples_to_jsonl(
    arxiv_ids: list[str],
    parsed_dir: Path,
    out_path: Path,
    processed_path: Path,
    *,
    completed_ids: set[str] | None = None,
    on_progress: Callable[[str, bool], None] | None = None,
    max_target_chars: int = 8000,
    outline_source: OutlineSource = "body",
    prompt_mode: PromptMode = "mixed",
    retrieval_fraction: float = 0.3,
    retrieval_top_k: int = 8,
) -> dict[str, Any]:
    completed = completed_ids or set()
    written = 0
    skipped: list[str] = []
    for aid in arxiv_ids:
        if aid in completed:
            continue
        paper = load_paper(parsed_dir, aid)
        if not paper:
            skipped.append(aid)
            mark_processed(processed_path, aid)
            if on_progress:
                on_progress(aid, False)
            continue
        row = build_sft_row(
            aid,
            paper,
            max_target_chars=max_target_chars,
            outline_source=outline_source,
            prompt_mode=prompt_mode,
            retrieval_fraction=retrieval_fraction,
            retrieval_top_k=retrieval_top_k,
        )
        if row:
            append_row(out_path, row)
            written += 1
        else:
            skipped.append(aid)
            logger.warning("Skip SFT %s: missing title/abstract or unsupported target", aid)
        mark_processed(processed_path, aid)
        if on_progress:
            on_progress(aid, row is not None)
    return {"written": written, "skipped_unbuildable": skipped}


def holdout_overlap_report(
    sft_ids: list[str],
    *,
    eval_grid_path: Path,
    eval_holdout_path: Path | None = None,
) -> dict[str, Any]:
    sft_set = set(sft_ids)
    grid = {
        ln.strip()
        for ln in eval_grid_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    overlap_grid = sorted(sft_set & grid)
    overlap_holdout: list[str] = []
    if eval_holdout_path and eval_holdout_path.is_file():
        holdout = {
            ln.strip()
            for ln in eval_holdout_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        overlap_holdout = sorted(sft_set & holdout)
    return {
        "sft_count": len(sft_set),
        "eval_grid_overlap": overlap_grid,
        "eval_holdout_overlap": overlap_holdout,
        "ok": not overlap_grid and not overlap_holdout,
    }


def split_train_val_rows(
    rows: list[dict[str, Any]],
    *,
    val_fraction: float = 0.05,
    seed: int = 1337,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_fraction <= 0 or not rows:
        return rows, []
    import random

    rng = random.Random(seed)
    ids = sorted({r["arxiv_id"] for r in rows})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_set = set(ids[:n_val])
    train = [r for r in rows if r["arxiv_id"] not in val_set]
    val = [r for r in rows if r["arxiv_id"] in val_set]
    return train, val


def write_sft_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split_ids(split_path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in split_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def load_train_jsonl_ids(train_path: Path) -> set[str]:
    if not train_path.is_file():
        return set()
    ids: set[str] = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids.add(json.loads(line)["arxiv_id"])
    return ids


def sft_coverage_report(
    split_path: Path,
    train_path: Path,
    val_path: Path | None = None,
) -> dict[str, Any]:
    """Compare ``sft.txt`` to train (+ optional val) JSONL (split is source of truth)."""
    split_ids = load_split_ids(split_path)
    split_set = set(split_ids)
    train_set = load_train_jsonl_ids(train_path)
    if val_path and val_path.is_file():
        train_set |= load_train_jsonl_ids(val_path)
    missing = sorted(split_set - train_set)
    extra = sorted(train_set - split_set)
    coverage_pct = (
        round(100.0 * len(train_set & split_set) / len(split_set), 2) if split_set else 100.0
    )
    return {
        "split_count": len(split_set),
        "train_count": len(train_set),
        "overlap": len(split_set & train_set),
        "missing_from_train": missing,
        "extra_in_train": extra,
        "coverage_pct": coverage_pct,
        "ok": not missing and not extra and len(train_set) == len(split_set),
    }


def refresh_sft_train_jsonl(
    split_path: Path,
    train_path: Path,
    parsed_dir: Path,
    *,
    val_path: Path | None = None,
    outline_source: OutlineSource = "body",
    prompt_mode: PromptMode = "mixed",
    retrieval_fraction: float = 0.3,
    retrieval_top_k: int = 8,
    val_fraction: float = 0.05,
    val_seed: int = 1337,
    eval_grid_path: Path | None = None,
    eval_holdout_path: Path | None = None,
) -> dict[str, Any]:
    """Rebuild train (and optional val) JSONL from ``sft.txt``."""
    arxiv_ids = load_split_ids(split_path)
    holdout: dict[str, Any] | None = None
    if eval_grid_path:
        holdout = holdout_overlap_report(
            arxiv_ids,
            eval_grid_path=eval_grid_path,
            eval_holdout_path=eval_holdout_path,
        )
        if not holdout["ok"]:
            logger.warning(
                "SFT/eval leakage: grid=%d holdout=%d",
                len(holdout["eval_grid_overlap"]),
                len(holdout["eval_holdout_overlap"]),
            )
    rows = build_sft_examples(
        arxiv_ids,
        parsed_dir,
        outline_source=outline_source,
        prompt_mode=prompt_mode,
        retrieval_fraction=retrieval_fraction,
        retrieval_top_k=retrieval_top_k,
    )
    train_rows, val_rows = split_train_val_rows(
        rows, val_fraction=val_fraction, seed=val_seed
    )
    write_sft_jsonl(train_rows, train_path)
    if val_path and val_rows:
        write_sft_jsonl(val_rows, val_path)
    skipped = [aid for aid in arxiv_ids if aid not in {r["arxiv_id"] for r in rows}]
    report = sft_coverage_report(split_path, train_path, val_path)
    report["built"] = len(train_rows)
    report["val_built"] = len(val_rows)
    report["skipped_unbuildable"] = skipped
    report["with_retrieval"] = sum(1 for r in train_rows if r.get("prompt_style") == "with_retrieval")
    report["no_retrieval"] = sum(1 for r in train_rows if r.get("prompt_style") == "no_retrieval")
    report["outline_source"] = outline_source
    report["prompt_mode"] = prompt_mode
    if eval_grid_path:
        report["holdout_check"] = holdout_overlap_report(
            [r["arxiv_id"] for r in train_rows],
            eval_grid_path=eval_grid_path,
            eval_holdout_path=eval_holdout_path,
        )
    if skipped:
        logger.warning(
            "SFT build skipped %d ids (missing parsed/title/abstract): %s",
            len(skipped),
            skipped[:5],
        )
    return report
