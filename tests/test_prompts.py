"""Prompt template tests (v3.1 — no [CITE])."""

from __future__ import annotations

from grounded.generate.prompts import render_abstract_prompt, sanitize_generated_abstract


def test_prompt_has_no_cite_instruction() -> None:
    system, user = render_abstract_prompt("T", "outline", "[1] evidence")
    assert "[CITE]" not in system
    assert "[CITE]" not in user
    assert "citation markers" in user.lower()


def test_sanitize_generated_abstract_removes_labels_and_citations() -> None:
    text = "Abstract: We present a method {{cite:abc}} that improves results [1]."

    cleaned = sanitize_generated_abstract(text)

    assert cleaned == "We present a method that improves results."
