from __future__ import annotations

from grounded.utils.incremental_jsonl import append_row, load_processed_ids, mark_processed, read_jsonl


def test_incremental_jsonl_append_and_processed_resume(tmp_path):
    jsonl = tmp_path / "rows.jsonl"
    processed = tmp_path / "processed.txt"

    append_row(jsonl, {"arxiv_id": "1", "value": 1})
    mark_processed(processed, "1")
    mark_processed(processed, "2")

    assert read_jsonl(jsonl) == [{"arxiv_id": "1", "value": 1}]
    assert load_processed_ids(processed) == {"1", "2"}
