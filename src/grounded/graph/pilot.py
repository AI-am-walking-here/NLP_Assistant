"""M-6.2 — pilot paper sampling and chunk selection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any


def sample_pilot_ids(
    candidate_ids: list[str],
    *,
    n: int,
    seed: int,
) -> list[str]:
    if n >= len(candidate_ids):
        return list(candidate_ids)
    rng = random.Random(seed)
    ids = list(candidate_ids)
    rng.shuffle(ids)
    return sorted(ids[:n])


def write_id_list(path: Path, arxiv_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(arxiv_ids) + "\n", encoding="utf-8")


def load_chunks_for_papers(
    chunks_parquet: Path,
    paper_ids: set[str],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        chunks_parquet,
        columns=["chunk_id", "paper_id", "section_heading", "text", "token_count"],
    )
    rows = table.to_pylist()
    return [r for r in rows if r["paper_id"] in paper_ids]


def count_chunks_by_paper(chunk_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in chunk_rows:
        pid = row["paper_id"]
        counts[pid] = counts.get(pid, 0) + 1
    return counts
