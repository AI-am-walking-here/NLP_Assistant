#!/usr/bin/env python3
"""M-1.5 — unarxive manifest: rebuild stub from parsed index or fix Windows paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

import click

from grounded.config import load_config, project_root, resolve_path

WINDOWS_ABS = re.compile(r"^[A-Za-z]:\\")


def relativize(value: str, root: Path) -> str:
    if not WINDOWS_ABS.match(value) and not value.startswith("\\\\"):
        return value
    normalized = value.replace("\\", "/")
    for marker in ("unarxive_extracted/", "data/", "LLM assistant Final/"):
        idx = normalized.lower().find(marker.lower())
        if idx != -1:
            return normalized[idx:]
    return normalized.split("/")[-1]


def fix_manifest(manifest_path: Path, root: Path, *, dry_run: bool) -> dict:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = 0
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            for key in ("path", "extracted_path", "paper_json", "source_path"):
                if key in row and isinstance(row[key], str):
                    new = relativize(row[key], root)
                    if new != row[key]:
                        row[key] = new
                        changed += 1
    elif isinstance(raw, dict):
        for key, val in list(raw.items()):
            if isinstance(val, str):
                new = relativize(val, root)
                if new != val:
                    raw[key] = new
                    changed += 1
    if not dry_run:
        backup = manifest_path.with_suffix(".json.bak")
        if not backup.is_file():
            backup.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "paths_fixed": changed, "dry_run": dry_run}


def write_manifest_from_parsed(parsed_manifest: Path, out: Path, extracted_root: Path) -> dict:
    rows: list[dict] = []
    with parsed_manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            meta = json.loads(line)
            if meta.get("source") != "unarxive":
                continue
            arxiv_id = meta["arxiv_id"]
            rel = f"unarxive_extracted/{arxiv_id}/paper.json"
            rows.append(
                {
                    "id": arxiv_id,
                    "arxiv_id": arxiv_id,
                    "paper_json": rel,
                    "path": rel,
                    "extracted_path": str(extracted_root / arxiv_id),
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return {"written": str(out), "unarxive_rows": len(rows)}


@click.command()
@click.option("--dry-run", is_flag=True, help="Report path fixes without writing.")
@click.option(
    "--rebuild-from-parsed",
    is_flag=True,
    help="Write manifest from parsed_manifest.jsonl (recovery host; relative paths).",
)
def main(dry_run: bool, rebuild_from_parsed: bool) -> int:
    data_cfg = load_config("data")
    manifest = resolve_path(
        data_cfg.paths.unarxive_manifest or "data/archive/metadata/unarxive_manifest.json"
    )
    if rebuild_from_parsed:
        parsed_manifest = resolve_path(data_cfg.paths.parsed_manifest)
        extracted_root = resolve_path(
            data_cfg.paths.unarxive_extracted or "data/archive/unarxive_extracted"
        )
        stats = write_manifest_from_parsed(parsed_manifest, manifest, extracted_root)
        click.echo(json.dumps(stats, indent=2))
        return 0
    if not manifest.is_file():
        raise click.ClickException(
            f"Manifest not found: {manifest}. Use --rebuild-from-parsed on recovery hosts."
        )
    stats = fix_manifest(manifest, project_root(), dry_run=dry_run)
    click.echo(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
