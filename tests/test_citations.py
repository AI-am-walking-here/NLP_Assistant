"""Parsed-JSON enrichment patches (M-2.4)."""

from __future__ import annotations

import json
from pathlib import Path

from grounded.data.citations import (
    apply_meta_to_paper,
    prune_enriched_valid_ids,
    read_parsed_record,
)


def test_apply_meta_patches_without_pydantic(tmp_path: Path) -> None:
    path = tmp_path / "1234.56789.json"
    path.write_text(
        json.dumps(
            {
                "arxiv_id": "1234.56789",
                "title": "T",
                "abstract": "A",
                "citation_count": None,
            }
        ),
        encoding="utf-8",
    )
    assert apply_meta_to_paper(path, {"citation_count": 42, "venue": "ACL"})
    record = read_parsed_record(path)
    assert record is not None
    assert record["citation_count"] == 42
    assert record["venue"] == "ACL"
    assert record["title"] == "T"


def test_prune_enriched_valid_ids(tmp_path: Path) -> None:
    parsed = tmp_path
    good = parsed / "1.json"
    bad_cc = parsed / "2.json"
    bad_json = parsed / "3.json"
    good.write_text(json.dumps({"citation_count": 1}), encoding="utf-8")
    bad_cc.write_text(json.dumps({"citation_count": None}), encoding="utf-8")
    bad_json.write_text("{not json", encoding="utf-8")

    kept, excluded = prune_enriched_valid_ids(["1", "2", "3"], parsed)
    assert kept == ["1"]
    assert excluded["missing_citation_count"] == ["2"]
    assert excluded["unreadable"] == ["3"]
