from __future__ import annotations

import json
import time

import pytest

from grounded.utils.phase_resume import (
    adapter_complete,
    content_fingerprint,
    phase_input_fingerprint,
    stable_hash,
    validate_phase,
)


def test_stable_hash_order_independent_for_dicts():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_adapter_complete_requires_files(tmp_path):
    run = tmp_path / "seg5_sft_train_test"
    (run / "adapter").mkdir(parents=True)
    assert adapter_complete(run) is False
    (run / "adapter" / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
    assert adapter_complete(run) is True


def test_content_fingerprint_ignores_mtime(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    first = content_fingerprint(path)
    time.sleep(0.01)
    path.write_text('{"a": 1}\n', encoding="utf-8")
    second = content_fingerprint(path)
    assert first == second


def test_build_sft_data_fingerprint_ignores_cuda_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SFT_CUDA_DEVICES", raising=False)
    without = phase_input_fingerprint("build_sft_data")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("SFT_CUDA_DEVICES", "3")
    with_cuda = phase_input_fingerprint("build_sft_data")
    assert without == with_cuda


def test_validate_sft_train_prefers_complete_over_empty_partial(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr("grounded.utils.phase_resume.project_root", lambda: tmp_path)

    complete = runs / "seg5_sft_train_2026-06-05-0617"
    (complete / "adapter").mkdir(parents=True)
    (complete / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (complete / "run_meta.json").write_text(
        json.dumps({"status": "trained", "inputs_fingerprint": "legacy"}),
        encoding="utf-8",
    )

    partial = runs / "seg5_sft_train_2026-06-06-0252"
    partial.mkdir()
    (partial / "meta.json").write_text("{}", encoding="utf-8")
    time.sleep(0.01)
    partial.touch()

    status = validate_phase("sft_train")
    assert status.state == "ok"
    assert "2026-06-05-0617" in status.detail
