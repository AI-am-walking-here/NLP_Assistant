#!/usr/bin/env python3
"""Re-export data/parsed/*.json from frozen corpus/papers.jsonl.gz."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from grounded.data.corpus_export import DEFAULT_CORPUS, export_corpus, export_corpus_ids
from grounded.data.filter import apply_quality_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@click.command()
@click.option(
    "--corpus",
    type=click.Path(path_type=Path, exists=True),
    default=DEFAULT_CORPUS,
    show_default=True,
    help="Source papers.jsonl.gz (read-only).",
)
@click.option(
    "--parsed-dir",
    type=click.Path(path_type=Path),
    default=PROJECT_ROOT / "data" / "parsed",
    show_default=True,
)
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    default=PROJECT_ROOT / "data" / "parsed_manifest.jsonl",
    show_default=True,
)
@click.option(
    "--no-preserve-sections",
    is_flag=True,
    help="Do not reuse section structure from existing valid parsed JSON.",
)
@click.option("--limit", type=int, default=None, help="Export only the first N corpus rows.")
@click.option(
    "--ids",
    multiple=True,
    help="Re-export specific arxiv IDs only (skips full manifest rewrite unless --rebuild-manifest).",
)
@click.option(
    "--rebuild-manifest",
    is_flag=True,
    help="Rebuild parsed_manifest.jsonl from on-disk parsed JSON after export.",
)
@click.option(
    "--write-valid",
    is_flag=True,
    help="After export, run M-2.3 quality filter to refresh data/parsed_valid.json.",
)
@click.option("--min-body-len", type=int, default=4000)
@click.option("--min-citation-keys", type=int, default=0)
def main(
    corpus: Path,
    parsed_dir: Path,
    manifest: Path,
    no_preserve_sections: bool,
    limit: int | None,
    ids: tuple[str, ...],
    rebuild_manifest: bool,
    write_valid: bool,
    min_body_len: int,
    min_citation_keys: int,
) -> None:
    preserve_sections = False if not ids else not no_preserve_sections
    if ids:
        stats = export_corpus_ids(
            corpus,
            parsed_dir,
            set(ids),
            preserve_sections=preserve_sections,
        )
    else:
        stats = export_corpus(
            corpus,
            parsed_dir,
            manifest,
            preserve_sections=preserve_sections,
            limit=limit,
        )
    if rebuild_manifest:
        from grounded.data.corpus_export import rebuild_manifest_from_parsed

        stats["manifest"] = rebuild_manifest_from_parsed(parsed_dir, manifest)
    logger.info("Export complete: %s", json.dumps(stats, indent=2))

    if write_valid:
        valid_path = PROJECT_ROOT / "data" / "parsed_valid.json"
        filter_stats = apply_quality_filter(
            manifest,
            valid_path,
            min_body_len=min_body_len,
            min_citation_keys=min_citation_keys,
        )
        logger.info("Quality filter: %s", json.dumps(filter_stats, indent=2))

    click.echo(json.dumps(stats, indent=2))


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as exc:
        logger.error("%s", exc)
        sys.exit(1)
