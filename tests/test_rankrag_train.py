from __future__ import annotations

import json

from scripts.rankrag_train import _flatten_rankrag_rows, _stats, rankrag_score_probe


def test_flatten_rankrag_rows_produces_binary_label_examples(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    row = {
        "query": "title\nabstract",
        "candidates": ["positive candidate", "negative candidate"],
        "labels": [1, 0],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    flat = _flatten_rankrag_rows(path)

    assert len(flat) == 2
    assert flat[0]["label_text"] == "1"
    assert flat[1]["label_text"] == "0"
    assert "positive candidate" in flat[0]["prompt"]
    assert "negative candidate" in flat[1]["prompt"]
    assert flat[0]["prompt"].endswith("[0]\npositive candidate\nlabel=")


def test_rankrag_stats_reports_grouped_and_flattened_counts(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"query": "q1", "candidates": ["a", "b", "c"], "labels": [1, 0, 0]},
        {"query": "q2", "candidates": ["d", "e"], "labels": [0, 1]},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    stats = _stats(path)

    assert stats == {
        "grouped_examples": 2,
        "flattened_examples": 5,
        "positive_labels": 2,
    }


def test_rankrag_score_probe_reports_positive_margin() -> None:
    class _FakeReranker:
        def score(self, query, candidates):
            del query
            return [0.9 if "good" in c else 0.1 for c in candidates]

    rows = [
        {"query": "q1", "candidates": ["good pos", "bad neg"], "labels": [1, 0]},
        {"query": "q2", "candidates": ["bad neg", "good pos"], "labels": [0, 1]},
    ]

    probe = rankrag_score_probe(_FakeReranker(), rows)

    assert probe["rows_checked"] == 2
    assert probe["avg_positive_score"] > probe["avg_negative_score"]
    assert probe["positive_beats_best_negative_rate"] == 1.0
