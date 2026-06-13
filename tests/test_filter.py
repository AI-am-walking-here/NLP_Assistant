"""M-2.3 quality filter."""

from __future__ import annotations

from grounded.data.filter import passes_quality_filter


def test_passes_when_ok() -> None:
    row = {
        "parse_status": "ok",
        "body_len": 5000,
        "num_citation_keys": 10,
    }
    assert passes_quality_filter(row) is True


def test_drops_failed() -> None:
    row = {"parse_status": "failed", "body_len": 9000, "num_citation_keys": 20}
    assert passes_quality_filter(row) is False


def test_drops_short_body() -> None:
    row = {"parse_status": "ok", "body_len": 1000, "num_citation_keys": 20}
    assert passes_quality_filter(row, min_body_len=4000) is False


def test_drops_few_cites() -> None:
    row = {"parse_status": "ok", "body_len": 9000, "num_citation_keys": 2}
    assert passes_quality_filter(row, min_citation_keys=5) is False
