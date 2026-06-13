"""Unit tests for RankRAG prompt formatting (no GPU)."""

from __future__ import annotations

from grounded.retrieve.rankrag_reranker import _format_candidate_prompt


def test_format_candidate_prompt() -> None:
    p = _format_candidate_prompt("title\noutline", "passage text")
    assert "Query:" in p
    assert "[0]" in p
    assert "\nlabel=" in p
    assert "passage text" in p


def test_format_candidate_prompt_includes_metadata_when_available() -> None:
    p = _format_candidate_prompt(
        "title\noutline",
        "passage text",
        paper_id="p1",
        section_heading="Methods",
    )
    assert "paper=p1" in p
    assert "section=Methods" in p
