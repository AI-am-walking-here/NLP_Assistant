#!/usr/bin/env python3
"""M-6.6 — build RankRAG rerank JSONL from holdout reserve (excludes eval grid)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from grounded.config import load_config, resolve_path
from grounded.train.rankrag_data import (
    build_rankrag_examples,
    build_rankrag_examples_to_jsonl,
    holdout_reserve_ids,
    index_chunks_by_paper,
    write_jsonl,
)
from grounded.utils.incremental_jsonl import load_processed_ids
from grounded.utils.phase_resume import phase_input_fingerprint, stable_hash

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--limit", type=int, default=500, help="Max training examples.")
@click.option("--seed", type=int, default=1337)
@click.option("--ids-file", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--shard-out", type=click.Path(path_type=Path), default=None)
@click.option("--processed-out", type=click.Path(path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(
    limit: int,
    seed: int,
    ids_file: Path | None,
    shard_out: Path | None,
    processed_out: Path | None,
    out: Path | None,
) -> int:
    data_cfg = load_config("data")
    retr_cfg = load_config("retrieval")
    rr_cfg = load_config("rankrag")
    splits_dir = resolve_path(data_cfg.paths.splits_dir)
    parsed_dir = resolve_path(data_cfg.paths.parsed_dir)
    chunks_path = resolve_path(retr_cfg.paths.chunks_parquet)
    out_path = resolve_path(out or rr_cfg.paths.train_jsonl)

    reserve = holdout_reserve_ids(
        splits_dir / "eval_holdout.txt",
        splits_dir / "eval_grid_80.txt",
    )
    index_ids = set(
        ln.strip()
        for ln in (splits_dir / "index.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    )
    logger.info("Holdout reserve: %d papers (eval grid excluded)", len(reserve))

    chunks_by_paper = index_chunks_by_paper(chunks_path)
    if ids_file is not None:
        if shard_out is None or processed_out is None:
            raise click.ClickException("--ids-file requires --shard-out and --processed-out")
        ids = [
            line.strip()
            for line in ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = build_rankrag_examples_to_jsonl(
            ids,
            parsed_dir,
            chunks_by_paper,
            index_ids,
            shard_out,
            processed_out,
            completed_ids=load_processed_ids(processed_out),
            n_candidates=rr_cfg.rerank_candidates,
            n_positives=min(3, rr_cfg.rerank_top_k),
            seed=seed,
        )
        summary.update({"ids": len(ids), "shard_out": str(shard_out)})
        click.echo(json.dumps(summary, indent=2))
        return 0

    examples = build_rankrag_examples(
        reserve,
        parsed_dir,
        chunks_by_paper,
        index_ids,
        n_candidates=rr_cfg.rerank_candidates,
        n_positives=min(3, rr_cfg.rerank_top_k),
        seed=seed,
        limit=limit,
    )
    write_jsonl(examples, out_path)
    (out_path.parent / "merge_manifest.json").write_text(
        json.dumps(
            {
                "inputs_fingerprint": phase_input_fingerprint("build_rankrag_data"),
                "schedule_fingerprint": stable_hash(
                    {
                        "limit": limit,
                        "seed": seed,
                        "mode": "sequential",
                    }
                ),
                "reserve_count": len(reserve),
                "requested_count": len(reserve[:limit] if limit else reserve),
                "written_count": len(examples),
                "processed_count": len(reserve[:limit] if limit else reserve),
                "shard_count": 1,
                "worker_count": 1,
                "path": str(out_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {"reserve_papers": len(reserve), "written": len(examples), "path": str(out_path)}
    click.echo(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
