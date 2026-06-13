#!/usr/bin/env python3
"""M-8.4 — write report/report.md from data/eval_set/grid_runs.json."""

from __future__ import annotations

from pathlib import Path

import click

from grounded.config import resolve_path
from grounded.eval.report_build import build_report_markdown


@click.command()
@click.option(
    "--grid",
    "grid_path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Grid JSON (default: data/eval_set/grid_runs.json).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output markdown (default: report/report.md).",
)
def main(grid_path: Path | None, out: Path | None) -> None:
    out_path = out or resolve_path("report/report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_report_markdown(grid_path=grid_path)
    out_path.write_text(text, encoding="utf-8")
    click.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
