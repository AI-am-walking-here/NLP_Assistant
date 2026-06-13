#!/usr/bin/env python3
"""M-5.2 — QLoRA SFT on data/sft/train.jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config
from grounded.config import RunContext
from grounded.train.sft_train import preflight_sft, run_sft_training
from grounded.utils.hf_network import enforce_no_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@click.command()
@click.option("--dry-run", is_flag=True, help="Validate data + config only.")
@click.option(
    "--max-steps",
    type=int,
    default=None,
    help="Cap training steps (smoke / debug). Default: full epoch(s).",
)
@click.option(
    "--smoke-train",
    is_flag=True,
    help="Train for 20 steps only (requires 8B weights on disk).",
)
@click.option(
    "--no-eval",
    is_flag=True,
    help="Skip validation during training (saves GPU memory at epoch boundaries).",
)
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Resume/write training in this existing run directory.",
)
def main(
    dry_run: bool,
    max_steps: int | None,
    smoke_train: bool,
    no_eval: bool,
    run_dir: Path | None,
) -> int:
    enforce_no_hub_download()
    sft_cfg = load_config("sft")
    if smoke_train:
        max_steps = max_steps if max_steps is not None else 20

    ctx = init_run("seg5", "sft_train", tags=["m-5.2"])
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            segment=ctx.segment,
            purpose=ctx.purpose,
            run_dir=run_dir,
            meta_path=run_dir / "meta.json",
            log_path=run_dir / "log.txt",
        )
        if not ctx.meta_path.is_file():
            ctx.meta_path.write_text(json.dumps({"status": "running"}, indent=2) + "\n", encoding="utf-8")
        if not ctx.log_path.is_file():
            ctx.log_path.write_text("", encoding="utf-8")
    try:
        report = run_sft_training(
            sft_cfg,
            run_dir=ctx.run_dir,
            max_steps=max_steps,
            dry_run=dry_run,
            eval_during_train=not no_eval,
        )
    except FileNotFoundError as exc:
        finish_run(ctx, status="failed")
        raise click.ClickException(str(exc)) from exc
    except RuntimeError as exc:
        finish_run(ctx, status="failed")
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(report, indent=2))
    finish_run(ctx, status=report.get("status", "finished"))
    if not dry_run:
        click.echo(report.get("adapter_dir", ctx.run_dir / "adapter"))
    return 0 if report.get("status") == "trained" else 1


if __name__ == "__main__":
    raise SystemExit(main())
