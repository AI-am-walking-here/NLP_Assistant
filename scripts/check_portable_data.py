#!/usr/bin/env python3
"""Verify portable data artifacts exist and are not gitignored (upload-ready)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded.data.portable_paths import (  # noqa: E402
    PORTABLE_REL_PATHS,
    archived_per_prompt_dir,
)


def _git_ignored(repo_root: Path, rel_from_repo: str) -> bool:
    """True only if git would skip the path (negation rules => not ignored)."""
    proc = subprocess.run(
        ["git", "check-ignore", "-q", rel_from_repo],
        cwd=repo_root,
        capture_output=True,
    )
    return proc.returncode == 0


def _rel_in_repo(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    missing: list[str] = []
    ignored: list[str] = []
    rows: list[dict[str, object]] = []

    for rel in PORTABLE_REL_PATHS:
        path = PROJECT_ROOT / rel
        repo_rel = f"llm-assistant-final/{rel}"
        exists = path.is_file()
        ign = _git_ignored(REPO_ROOT, repo_rel)
        if not exists:
            missing.append(rel)
        if ign:
            ignored.append(rel)
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "path": rel,
                "exists": exists,
                "bytes": size,
                "gitignored": bool(ign),
            }
        )

    arch_dir = archived_per_prompt_dir(PROJECT_ROOT)
    arch_files = sorted(arch_dir.glob("*.jsonl")) if arch_dir.is_dir() else []
    for f in arch_files:
        rel = _rel_in_repo(f)
        repo_rel = rel if rel.startswith("llm-assistant-final/") else f"llm-assistant-final/{rel}"
        ign = _git_ignored(REPO_ROOT, repo_rel)
        rows.append(
            {
                "path": rel,
                "exists": True,
                "bytes": f.stat().st_size,
                "gitignored": ign,
            }
        )
        if ign:
            ignored.append(rel)

    print(json.dumps({"files": rows}, indent=2))
    print()

    if missing:
        print("Missing (generate before upload):")
        for m in missing:
            print(f"  - {m}")
    if ignored:
        print("Still gitignored (fix .gitignore):")
        for line in ignored:
            print(f"  - {line}")

    if not missing and not ignored:
        print("OK: portable artifacts present and trackable.")
        print()
        print("Upload (from repo root):")
        print("  cd /data/team1")
        print("  git add llm-assistant-final/data/sft/*.jsonl \\")
        print("          llm-assistant-final/data/splits/*.txt \\")
        print("          llm-assistant-final/data/rankrag/*.jsonl \\")
        print("          llm-assistant-final/data/parsed_valid.json \\")
        print("          llm-assistant-final/data/papers_enriched.jsonl \\")
        print("          llm-assistant-final/data/eval_set/")
        print("  git status")
        print("See llm-assistant-final/docs/DATA_UPLOAD.md")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
