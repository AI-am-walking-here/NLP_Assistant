"""Parsed-paper schema validation against on-disk samples."""

from __future__ import annotations

import json
from pathlib import Path

from grounded.data.schema import Paper


def test_paper_roundtrip_sample(sample_parsed_path: Path) -> None:
    raw = json.loads(sample_parsed_path.read_text(encoding="utf-8"))
    paper = Paper.model_validate(raw)
    assert paper.arxiv_id == "1601.02539"
    assert paper.parse_status in ("ok", "partial", "failed")
    assert paper.model_dump()["title"]
