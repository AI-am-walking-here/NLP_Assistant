#!/usr/bin/env python3
"""M-5.1 — build data/sft/train.jsonl from data/splits/sft.txt."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config, log_metric, resolve_path
from grounded.train.sft_data import (
    build_sft_examples,
    build_sft_examples_to_jsonl,
    refresh_sft_train_jsonl,
    write_sft_jsonl,
)
from grounded.progress import CountProgressReporter
from grounded.utils.incremental_jsonl import load_processed_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--split",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Paper ID list (default: data/splits/sft.txt).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSONL (default: data/sft/train.jsonl).",
)
@click.option("--limit", type=int, default=None)
@click.option(
    "--outline-source",
    type=click.Choice(["body", "abstract"]),
    default=None,
    help="Outline heuristic (default: configs/sft.yaml).",
)
@click.option(
    "--prompt-mode",
    type=click.Choice(["mixed", "with_retrieval", "no_retrieval"]),
    default=None,
)
@click.option("--retrieval-fraction", type=float, default=None)
@click.option("--no-retrieval", is_flag=True, help="Shortcut for --prompt-mode no_retrieval.")
@click.option("--with-retrieval", is_flag=True, help="Shortcut for --prompt-mode with_retrieval.")
@click.option(
    "--ids-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Shard worker input IDs. Enables append/resume worker mode.",
)
@click.option(
    "--shard-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Shard worker output JSONL.",
)
@click.option(
    "--processed-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Shard worker processed-id ledger.",
)
@click.option("--no-val-split", is_flag=True, help="Worker mode emits train rows only.")
def main(
    split: Path | None,
    out: Path | None,
    limit: int | None,
    outline_source: str | None,
    prompt_mode: str | None,
    retrieval_fraction: float | None,
    no_retrieval: bool,
    with_retrieval: bool,
    ids_file: Path | None,
    shard_out: Path | None,
    processed_out: Path | None,
    no_val_split: bool,
) -> int:
    data_cfg = load_config("data")
    sft_cfg = load_config("sft")
    split_path = split or resolve_path(sft_cfg.paths.split_list)
    out_path = out or resolve_path(sft_cfg.paths.train_jsonl)
    val_path = resolve_path(sft_cfg.paths.val_jsonl)
    parsed_dir = resolve_path(data_cfg.paths.parsed_dir)

    o_src = outline_source or sft_cfg.outline_source
    p_mode = prompt_mode or sft_cfg.prompt_mode
    if no_retrieval:
        p_mode = "no_retrieval"
    if with_retrieval:
        p_mode = "with_retrieval"
    r_frac = (
        retrieval_fraction
        if retrieval_fraction is not None
        else sft_cfg.retrieval_fraction
    )

    ctx = init_run("seg5", "build_sft_data", tags=["m-5.1", p_mode, o_src])

    if ids_file is not None:
        if shard_out is None or processed_out is None:
            raise click.ClickException("--ids-file requires --shard-out and --processed-out")
        if not no_val_split:
            logger.warning("Worker mode ignores validation split; pass --no-val-split for clarity")
        arxiv_ids = [
            ln.strip()
            for ln in ids_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        allowed = set(arxiv_ids)
        completed = {aid for aid in load_processed_ids(processed_out) if aid in allowed}
        reporter = CountProgressReporter(
            "build_sft_data",
            total=len(arxiv_ids),
            unit="papers",
            min_interval_s=5.0,
        )
        if completed:
            reporter.done_count = min(len(completed), len(arxiv_ids))

        def _tick(aid: str, wrote: bool) -> None:
            reporter.update(1, detail=f"{aid} {'written' if wrote else 'skipped'}")

        stats = build_sft_examples_to_jsonl(
            arxiv_ids,
            parsed_dir,
            shard_out,
            processed_out,
            completed_ids=completed,
            on_progress=_tick,
            outline_source=o_src,  # type: ignore[arg-type]
            prompt_mode=p_mode,  # type: ignore[arg-type]
            retrieval_fraction=r_frac,
            retrieval_top_k=sft_cfg.retrieval_top_k,
        )
        reporter.finish(detail=str(shard_out))
        stats.update(
            {
                "input_ids": len(arxiv_ids),
                "processed": len(load_processed_ids(processed_out)),
                "output": str(shard_out),
                "processed_out": str(processed_out),
            }
        )
    elif limit is not None:
        arxiv_ids = [
            ln.strip()
            for ln in split_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ][:limit]
        rows = build_sft_examples(
            arxiv_ids,
            parsed_dir,
            outline_source=o_src,  # type: ignore[arg-type]
            prompt_mode=p_mode,  # type: ignore[arg-type]
            retrieval_fraction=r_frac,
            retrieval_top_k=sft_cfg.retrieval_top_k,
        )
        write_sft_jsonl(rows, out_path)
        stats = {"input_ids": len(arxiv_ids), "written": len(rows), "output": str(out_path)}
    else:
        splits_dir = resolve_path(data_cfg.paths.splits_dir)
        stats = refresh_sft_train_jsonl(
            split_path,
            out_path,
            parsed_dir,
            val_path=val_path,
            outline_source=o_src,  # type: ignore[arg-type]
            prompt_mode=p_mode,  # type: ignore[arg-type]
            retrieval_fraction=r_frac,
            retrieval_top_k=sft_cfg.retrieval_top_k,
            val_fraction=sft_cfg.val_fraction,
            val_seed=sft_cfg.val_seed,
            eval_grid_path=splits_dir / "eval_grid_80.txt",
            eval_holdout_path=splits_dir / "eval_holdout.txt",
        )
        stats["output"] = str(out_path)
        stats["val_output"] = str(val_path)

    log_metric(ctx, "sft_examples", float(stats.get("written", stats.get("built", 0))))
    (ctx.run_dir / "sft_build_summary.json").write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    finish_run(ctx)
    n_written = int(stats.get("written", stats.get("built", 0)))
    logger.info("Wrote %d SFT train examples to %s", n_written, out_path)
    click.echo(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
