"""M-1.5 — manifest path relativization."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fix_unarxive_manifest_paths import fix_manifest


def test_fix_windows_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "unarxive_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "1601.00001",
                    "path": r"C:\Users\Lud\LLM assistant Final\data\unarxive_extracted\1601.00001\paper.json",
                },
            ]
        ),
        encoding="utf-8",
    )
    stats = fix_manifest(manifest, tmp_path, dry_run=False)
    assert stats["paths_fixed"] == 1
    row = json.loads(manifest.read_text())[0]
    assert row["path"].startswith("unarxive_extracted/")
