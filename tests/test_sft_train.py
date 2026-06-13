"""M-5.2 — SFT training preflight."""

from __future__ import annotations

from grounded.config import load_config
from grounded.train.sft_train import preflight_sft, run_sft_training


def test_preflight_sft() -> None:
    cfg = load_config("sft")
    report = preflight_sft(cfg)
    assert report["train_examples"] >= 900
    assert "messages" not in report


def test_run_sft_dry_run(tmp_path) -> None:
    cfg = load_config("sft")
    report = run_sft_training(cfg, run_dir=tmp_path, dry_run=True)
    assert report["status"] == "dry_run_ok"
