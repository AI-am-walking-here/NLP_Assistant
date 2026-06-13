"""SFT split vs train.jsonl sync."""

from __future__ import annotations

import json
from pathlib import Path

from grounded.train.sft_data import refresh_sft_train_jsonl, sft_coverage_report


def test_sft_coverage_ok(tmp_path: Path) -> None:
    split = tmp_path / "sft.txt"
    train = tmp_path / "train.jsonl"
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    split.write_text("1111.00001\n1111.00002\n", encoding="utf-8")
    for aid in ("1111.00001", "1111.00002"):
        (parsed / f"{aid}.json").write_text(
            json.dumps(
                {
                    "arxiv_id": aid,
                    "title": "T",
                    "abstract": "We show results. They are good.",
                }
            ),
            encoding="utf-8",
        )
    val = tmp_path / "val.jsonl"
    refresh_sft_train_jsonl(
        split,
        train,
        parsed,
        val_path=val,
        val_fraction=0.5,
        prompt_mode="no_retrieval",
    )
    report = sft_coverage_report(split, train, val)
    assert report["ok"] is True
    assert report["split_count"] == 2
    assert report["train_count"] == 2
