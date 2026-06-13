#!/usr/bin/env python3
"""Train QLoRA DPO adapter from FActScore preference pairs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config
from grounded.train.dpo_train import run_dpo_training
from grounded.utils.hf_network import enforce_no_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@click.command()
@click.option("--dry-run", is_flag=True)
@click.option("--max-steps", type=int, default=None)
@click.option(
    "--init-adapter",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Warm-start LoRA (default: latest seg5_sft_train_*/adapter if init_from_latest_sft).",
)
@click.option(
    "--no-init-sft",
    is_flag=True,
    help="Train fresh LoRA instead of warm-starting from SFT adapter.",
)
def main(
    dry_run: bool,
    max_steps: int | None,
    init_adapter: Path | None,
    no_init_sft: bool,
) -> int:
    enforce_no_hub_download()
    sft_cfg = load_config("sft")
    dpo_cfg = sft_cfg.dpo
    if no_init_sft:
        dpo_cfg = dpo_cfg.model_copy(update={"init_from_latest_sft": False})

    ctx = init_run("seg5", "dpo_train", tags=["m-5.5", "dpo"])
    try:
        report = run_dpo_training(
            sft_cfg,
            dpo_cfg,
            run_dir=ctx.run_dir,
            max_steps=max_steps,
            dry_run=dry_run,
            init_adapter_path=init_adapter,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        finish_run(ctx, status="failed")
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(report, indent=2))
    finish_run(ctx, status=report.get("status", "finished"))
    if not dry_run:
        click.echo(report.get("adapter_dir", ctx.run_dir / "adapter"))
    return 0 if report.get("status") == "trained" else 1


if __name__ == "__main__":
    raise SystemExit(main())
