"""Parse and select `runs/seg4_eval_*` directories."""

from __future__ import annotations

import re
from pathlib import Path

_SEG4_EVAL = re.compile(r"^seg4_eval_(.+)_(\d{4}-\d{2}-\d{2}-\d{4})$")


def parse_seg4_eval_name(name: str) -> tuple[str, str] | None:
    """Return (system, timestamp) for a run folder name, or None."""
    m = _SEG4_EVAL.match(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def list_seg4_eval_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    out: list[Path] = []
    for path in runs_dir.iterdir():
        if path.is_dir() and parse_seg4_eval_name(path.name):
            out.append(path)
    return sorted(out, key=lambda p: p.name)


def latest_seg4_eval_dirs(runs_dir: Path) -> dict[str, Path]:
    """Newest run folder per eval system name."""
    buckets: dict[str, list[Path]] = {}
    for path in list_seg4_eval_dirs(runs_dir):
        system, _ts = parse_seg4_eval_name(path.name)  # type: ignore[misc]
        buckets.setdefault(system, []).append(path)
    return {system: max(paths, key=lambda p: p.stat().st_mtime) for system, paths in buckets.items()}
