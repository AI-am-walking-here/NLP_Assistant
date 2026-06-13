"""M-2.5 (v3.1) — stratified index / holdout / eval grid / SFT splits."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

YEAR_BINS = ("2016-2019", "2020-2022", "2023-2025")
EVAL_GRID_SIZE = 80
HOLDOUT_FRACTION = 0.10
SFT_CITATION_MIN = 0
RANDOM_SEED = 1337


def normalize_source(source: str) -> str:
    if source in ("arxiv_s3", "latex_s3"):
        return "latex_s3"
    return source


def year_bin(year: int | None) -> str:
    if year is None:
        return "unknown"
    if year <= 2019:
        return "2016-2019"
    if year <= 2022:
        return "2020-2022"
    return "2023-2025"


def load_paper_meta(parsed_dir: Path, arxiv_id: str) -> dict[str, Any]:
    path = parsed_dir / f"{arxiv_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Corrupt parsed JSON for {arxiv_id}: {exc}") from exc


def drop_unreadable_ids(arxiv_ids: list[str], parsed_dir: Path) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for aid in arxiv_ids:
        path = parsed_dir / f"{aid}.json"
        try:
            json.loads(path.read_text(encoding="utf-8"))
            kept.append(aid)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            dropped.append(aid)
    return kept, dropped


def stratified_holdout_split(
    arxiv_ids: list[str],
    parsed_dir: Path,
    *,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for aid in arxiv_ids:
        meta = load_paper_meta(parsed_dir, aid)
        key = (normalize_source(meta.get("source", "")), year_bin(meta.get("year")))
        strata[key].append(aid)

    rng = random.Random(seed)
    index_ids: list[str] = []
    holdout_ids: list[str] = []
    for key, ids in strata.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_hold = max(1, int(round(len(ids) * holdout_fraction))) if len(ids) > 1 else 0
        if len(ids) == 1:
            index_ids.extend(ids)
            continue
        holdout_ids.extend(ids[:n_hold])
        index_ids.extend(ids[n_hold:])
    return index_ids, holdout_ids


def build_sft_ids(index_ids: list[str], parsed_dir: Path) -> list[str]:
    out: list[str] = []
    for aid in index_ids:
        meta = load_paper_meta(parsed_dir, aid)
        cc = meta.get("citation_count")
        if cc is not None and cc < SFT_CITATION_MIN:
            continue
        if not meta.get("title") or not meta.get("abstract"):
            continue
        out.append(aid)
    return out


def write_split_files(
    splits_dir: Path,
    index_ids: list[str],
    holdout_ids: list[str],
    eval_grid_ids: list[str],
    sft_ids: list[str],
) -> dict[str, int]:
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, ids: list[str]) -> None:
        path = splits_dir / name
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    write("index.txt", index_ids)
    write("eval_holdout.txt", holdout_ids)
    write("eval_grid_80.txt", eval_grid_ids)
    write("sft.txt", sft_ids)
    return {
        "index": len(index_ids),
        "eval_holdout": len(holdout_ids),
        "eval_grid_80": len(eval_grid_ids),
        "sft": len(sft_ids),
    }


def build_splits_v31(
    valid_ids: list[str],
    parsed_dir: Path,
    splits_dir: Path,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    index_ids, holdout_ids = stratified_holdout_split(
        valid_ids, parsed_dir, seed=seed
    )
    rng = random.Random(seed + 1)
    holdout_shuffled = list(holdout_ids)
    rng.shuffle(holdout_shuffled)
    eval_grid_ids = holdout_shuffled[:EVAL_GRID_SIZE]
    sft_ids = build_sft_ids(index_ids, parsed_dir)
    counts = write_split_files(splits_dir, index_ids, holdout_ids, eval_grid_ids, sft_ids)

    overlap = set(index_ids) & set(holdout_ids)
    if overlap:
        raise ValueError(f"index/holdout overlap: {len(overlap)} ids")

    stats = {
        "valid_input": len(valid_ids),
        "counts": counts,
        "holdout_reserve": len(holdout_ids) - len(eval_grid_ids),
        "seed": seed,
    }
    logger.info("Splits built: %s", stats)
    return stats
