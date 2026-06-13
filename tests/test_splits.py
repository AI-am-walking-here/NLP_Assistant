"""M-2.5 split helpers."""

from __future__ import annotations

import json

from grounded.data.splits import SFT_CITATION_MIN, build_sft_ids, normalize_source, year_bin


def test_normalize_source_latex_alias() -> None:
    assert normalize_source("arxiv_s3") == "latex_s3"
    assert normalize_source("latex_s3") == "latex_s3"
    assert normalize_source("unarxive") == "unarxive"


def test_year_bins() -> None:
    assert year_bin(2018) == "2016-2019"
    assert year_bin(2021) == "2020-2022"
    assert year_bin(2024) == "2023-2025"
    assert year_bin(None) == "unknown"


def test_sft_citation_min_allows_low_citation_papers(tmp_path) -> None:
    assert SFT_CITATION_MIN == 0
    paper = {
        "title": "Low Citation Paper",
        "abstract": "This paper is still useful for domain SFT.",
        "citation_count": 0,
    }
    (tmp_path / "p1.json").write_text(json.dumps(paper), encoding="utf-8")

    assert build_sft_ids(["p1"], tmp_path) == ["p1"]
