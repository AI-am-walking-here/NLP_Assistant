"""Eval prompt construction (M-4.1)."""

from __future__ import annotations

from grounded.eval.prompts_build import build_eval_prompts, outline_from_abstract


def test_outline_from_abstract() -> None:
    abstract = "We propose a new method. It improves accuracy. Results are strong."
    outline = outline_from_abstract(abstract, max_bullets=3)
    assert outline.startswith("- ")
    assert "method" in outline.lower()


def test_build_eval_prompts_prefers_body_outline(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "p1.json").write_text(
        '{"title":"Paper","abstract":"Abstract sentence one. Abstract sentence two.","body_text":"Long body sentence about methods and analysis. Long body sentence about methods and analysis. Long body sentence about methods and analysis.","sections":[{"heading":"Method","paragraphs":["Method paragraph with enough detail to be used."]}]}',
        encoding="utf-8",
    )

    rows = build_eval_prompts(["p1"], parsed_dir)

    assert len(rows) == 1
    assert rows[0]["outline"].startswith("- Method")
    assert "Abstract sentence one" not in rows[0]["outline"]


def test_outline_from_body_never_falls_back_to_abstract(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "p2.json").write_text(
        '{"title":"Paper","abstract":"Gold abstract sentence should stay out.","body_text":"Body sentence with enough detail to seed the outline. Body sentence with enough detail to seed the outline. Body sentence with enough detail to seed the outline.","sections":[]}',
        encoding="utf-8",
    )

    rows = build_eval_prompts(["p2"], parsed_dir)

    assert len(rows) == 1
    assert "Gold abstract sentence should stay out" not in rows[0]["outline"]
    assert "Body sentence" in rows[0]["outline"]


def test_build_eval_prompts_falls_back_when_title_missing(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "2505.20977.json").write_text(
        '{"title":"","abstract":"Abstract is present.","body_text":"Long enough body text to form an outline. Long enough body text to form an outline. Long enough body text to form an outline.","sections":[{"heading":"Intro","paragraphs":["Paragraph with enough detail to be used as outline text."]}]}',
        encoding="utf-8",
    )

    rows = build_eval_prompts(["2505.20977"], parsed_dir)

    assert len(rows) == 1
    assert rows[0]["title"] == "arXiv 2505.20977"
