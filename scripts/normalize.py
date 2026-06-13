#!/usr/bin/env python3
"""Segment 2 — M-2.0 CLI: normalize archive → data/parsed/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grounded.config import append_log, init_run, write_json  # noqa: E402
from grounded.data.normalize import normalize_all  # noqa: E402


@click.command()
@click.option(
    "--archive-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=ROOT / "data" / "archive",
    show_default=True,
    help="Root of the read-only archive containing source-specific subdirs.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=ROOT / "data" / "parsed",
    show_default=True,
    help="Where unified <arxiv_id>.json files land.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=ROOT / "data" / "parsed_manifest.jsonl",
    show_default=True,
    help="One-row-per-paper manifest (status, lengths, notes).",
)
@click.option(
    "--source",
    "sources",
    type=click.Choice(["unarxive", "latex_s3"]),
    multiple=True,
    help="Restrict to a subset of sources (default: all).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap papers per source (smoke testing).",
)
def main(
    archive_root: Path,
    output_dir: Path,
    manifest_path: Path,
    sources: tuple[str, ...],
    limit: int | None,
) -> None:
    """Run two ingress → one egress normalization. See build_plan_v3.md M-2.0."""
    run_dir = init_run(
        segment="seg2",
        purpose="normalize",
        config_snapshot={
            "archive_root": str(archive_root),
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            "sources": list(sources) or "all",
            "limit_per_source": limit,
        },
    )
    append_log(run_dir, f"start: archive={archive_root} output={output_dir}")

    stats = normalize_all(
        archive_root=archive_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        sources=tuple(sources) if sources else None,
        limit_per_source=limit,
        show_progress=True,
    )

    write_json(run_dir, "results.json", stats.as_dict())
    append_log(run_dir, f"done: {json.dumps(stats.as_dict())}")
    click.echo(f"run: {run_dir}")
    click.echo(json.dumps(stats.as_dict(), indent=2))


if __name__ == "__main__":
    main()
