#!/usr/bin/env python3
"""M-4.1 — build data/eval_set/prompts.jsonl from eval_grid_80.txt."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from grounded.config import load_config, resolve_path
from grounded.eval.prompts_build import build_eval_prompts, write_prompts_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--grid",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Eval grid ID list (default: data/splits/eval_grid_80.txt).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSONL (default: configs eval_prompts_path).",
)
def main(grid: Path | None, out: Path | None) -> int:
    eval_cfg = load_config("eval")
    data_cfg = load_config("data")
    grid_path = grid or resolve_path(data_cfg.paths.splits_dir) / "eval_grid_80.txt"
    out_path = out or resolve_path(eval_cfg.eval_prompts_path)

    arxiv_ids = [ln.strip() for ln in grid_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    parsed_dir = resolve_path(data_cfg.paths.parsed_dir)
    rows = build_eval_prompts(arxiv_ids, parsed_dir)
    write_prompts_jsonl(rows, out_path)
    logger.info("Wrote %d prompts to %s", len(rows), out_path)
    click.echo(json.dumps({"count": len(rows), "path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
