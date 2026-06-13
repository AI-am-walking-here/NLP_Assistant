from __future__ import annotations

from grounded.utils.artifact_backup import backup_paths


def test_backup_paths_copies_small_files(tmp_path):
    src = tmp_path / "train.jsonl"
    src.write_text('{"x": 1}\n', encoding="utf-8")

    report = backup_paths("build_sft_data", [src], run_id="test-run")

    assert report["copied"]
    backup = report["copied"][0]["backup"]
    assert backup.endswith("train.jsonl")
