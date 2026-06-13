"""Small opt-in pre-overwrite backup helper for hardened rebuilds."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from grounded.config import project_root


def _include_path(path: Path) -> bool:
    if os.environ.get("BACKUP_LARGE", "0") == "1":
        return True
    return path.suffix in {".json", ".jsonl", ".txt", ".yaml", ".yml"}


def backup_paths(phase: str, paths: list[Path], *, run_id: str) -> dict:
    root = project_root() / "data" / "backups" / "hardened" / run_id / phase / "pre"
    copied: list[dict[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file() or not _include_path(path):
            continue
        rel = path.relative_to(project_root()) if path.is_relative_to(project_root()) else Path(path.name)
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied.append({"path": str(path), "backup": str(dest)})
    report = {"phase": phase, "run_id": run_id, "copied": copied, "created_at": time.time()}
    if copied:
        root.mkdir(parents=True, exist_ok=True)
        (root / "backup_log.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def phase_backup_paths(phase: str) -> list[Path]:
    root = project_root()
    mapping = {
        "build_sft_data": [
            root / "data/sft/train.jsonl",
            root / "data/sft/val.jsonl",
            root / "data/sft/merge_manifest.json",
        ],
        "build_index": [
            root / "data/indices/index_meta.json",
            root / "data/indices/faiss.index",
            root / "data/indices/embeddings.npy",
        ],
        "build_rankrag_data": [root / "data/rankrag/train.jsonl"],
        "build_eval_prompts": [root / "data/eval_set/prompts.jsonl"],
    }
    return mapping.get(phase, [])
