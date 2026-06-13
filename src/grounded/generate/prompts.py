"""M-3.4 — locked abstract-generation prompt (v3.1: no [CITE] in output)."""

from __future__ import annotations

import re

ABSTRACT_SYSTEM = (
    "You are writing the abstract section of an NLP research paper. "
    "Write in formal cs.CL style (~150–300 words). "
    "Ground claims in the supporting evidence; do not invent results, datasets, "
    "metrics, citations, or architecture names."
)

ABSTRACT_USER_TEMPLATE = """Title: {title}

Contributions outline:
{outline}

Supporting evidence from prior work:
{retrieved_chunks}

Write the abstract in 4-6 sentences.
- Anchor each major sentence to the title, outline, or evidence above.
- Prefer concrete datasets, tasks, methods, and findings when supported.
- If a number, dataset, benchmark, or score is not present in the evidence, omit it.
- Ignore contradictory or low-signal passages.
- Output only the abstract text; do not prefix it with Title, Abstract, or section labels.
- Do not use inline citation markers, bracketed cite placeholders, or raw citation keys."""


def format_retrieved_chunks(hits: list[dict]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        heading = hit.get("section_heading") or "Body"
        paper_id = hit.get("paper_id", "?")
        text = (hit.get("text") or "").strip()
        if len(text) > 800:
            text = text[:800] + "…"
        score = hit.get("rerank_score", hit.get("score"))
        score_text = f"{float(score):.3f}" if score is not None else "n/a"
        blocks.append(
            f"[{i}] paper={paper_id} section={heading} score={score_text}\n{text}"
        )
    return "\n\n".join(blocks) if blocks else "(no retrieved passages)"


SFT_USER_TEMPLATE = """Title: {title}

Contributions outline:
{outline}

Write the abstract for this paper. Do not use inline citation markers."""

_PREFIX_RE = re.compile(r"^\s*(?:#+\s*)?(?:title|abstract)\s*:?\s*", re.IGNORECASE)
_RAW_CITE_RE = re.compile(r"\{\{cite:[^}]+\}\}")
_BRACKET_CITE_RE = re.compile(r"\s*\[(?:\d+(?:\s*,\s*\d+)*|CITE|cite:[^\]]+)\]")
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_generated_abstract(text: str) -> str:
    """Normalize model output into abstract-only text."""
    text = text.strip()
    for _ in range(3):
        text = _PREFIX_RE.sub("", text).strip()
    text = _RAW_CITE_RE.sub("", text)
    text = _BRACKET_CITE_RE.sub("", text)
    text = text.replace("{{", "").replace("}}", "")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def render_sft_prompt(title: str, outline: str) -> tuple[str, str]:
    """SFT / zero-shot input: title + outline only (no retrieval block)."""
    user = SFT_USER_TEMPLATE.format(title=title.strip(), outline=outline.strip())
    return ABSTRACT_SYSTEM, user


def render_abstract_prompt(title: str, outline: str, retrieved_chunks: str) -> tuple[str, str]:
    user = ABSTRACT_USER_TEMPLATE.format(
        title=title.strip(),
        outline=outline.strip(),
        retrieved_chunks=retrieved_chunks,
    )
    return ABSTRACT_SYSTEM, user
