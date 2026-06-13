"""Ingress 1 of 2: unarXive → unified `Paper` schema.

Pure functions only. Takes raw unarXive output (the `paper.json` dict from
`data/archive/unarxive_extracted/<id>/`) and emits a `Paper`. No I/O of the
output (the orchestrator does that).

unarXive ships papers in `raw.body_text` as a list of section dicts, each
already carrying `{{cite:HASH}}` markers whose hashes resolve via
`raw.bib_entries`. This parser is mostly a re-shape.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from grounded.data.schema import BibEntry, Paper, Section


_CITE_HASH_RE = re.compile(r"\{\{cite:([a-f0-9]+)\}\}")
_NEW_STYLE_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _year_from_arxiv_id(arxiv_id: str) -> int | None:
    """`2106.05707` → 2021; old-style ids return None."""
    match = _NEW_STYLE_ID_RE.match(arxiv_id)
    if not match:
        return None
    yy = int(match.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def _extract_citation_keys(text: str) -> list[str]:
    return _CITE_HASH_RE.findall(text)


def _build_sections(body_text: list[dict[str, Any]]) -> tuple[list[Section], str, set[str]]:
    """Group consecutive body_text entries by section heading.

    Returns (sections, joined_body_text, set_of_citation_keys).
    """
    sections: list[Section] = []
    body_chunks: list[str] = []
    keys: set[str] = set()

    current_heading: str | None = None
    current_paragraphs: list[str] = []

    def flush() -> None:
        if current_heading is not None:
            sections.append(
                Section(heading=current_heading, level=1, paragraphs=current_paragraphs[:])
            )

    for entry in body_text:
        heading = (entry.get("section") or "body").strip() or "body"
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        keys.update(_extract_citation_keys(text))
        if heading != current_heading:
            flush()
            current_heading = heading
            current_paragraphs = []
        current_paragraphs.append(text)
        body_chunks.append(text)

    flush()
    return sections, "\n\n".join(body_chunks), keys


def _build_bibliography(bib_entries: dict[str, Any]) -> dict[str, BibEntry]:
    """Map unarXive's `bib_entries` (sha-keyed dicts) to our `BibEntry`."""
    out: dict[str, BibEntry] = {}
    for key, entry in (bib_entries or {}).items():
        if not isinstance(entry, dict):
            continue
        raw_text = entry.get("bib_entry_raw", "") or ""
        year_match = _YEAR_RE.search(raw_text)
        year = int(year_match.group()) if year_match else None
        out[key] = BibEntry(citation_key=key, raw_entry=raw_text, year=year)
    return out


def parse_unarxive(paper_json: dict[str, Any]) -> Paper:
    """unarXive `paper.json` dict → `Paper`. Pure."""
    raw = paper_json.get("raw") or {}
    metadata = raw.get("metadata") or {}

    arxiv_id = (paper_json.get("id") or metadata.get("id") or "").strip()
    if not arxiv_id:
        raise ValueError("unarXive record missing arxiv_id")

    title = (paper_json.get("title") or metadata.get("title") or "").strip()
    abstract_obj = raw.get("abstract") or {}
    abstract = (
        paper_json.get("abstract") or abstract_obj.get("text") or ""
    ).strip()

    sections, body_text, keys_in_body = _build_sections(raw.get("body_text") or [])
    if not body_text:
        body_text = (paper_json.get("text") or "").strip()
        if body_text and not keys_in_body:
            keys_in_body = set(_extract_citation_keys(body_text))

    keys_in_body.update(_extract_citation_keys(abstract))

    bibliography = _build_bibliography(raw.get("bib_entries") or {})

    notes: list[str] = []
    if not sections and body_text:
        notes.append("no structured sections; collapsed to single body block")
        sections = [Section(heading="body", level=1, paragraphs=[body_text])]

    if not title:
        notes.append("missing title")
    if not abstract:
        notes.append("missing abstract")
    if not bibliography:
        notes.append("missing bibliography")

    if not title or not abstract or not body_text:
        status = "failed" if not body_text else "partial"
    elif notes:
        status = "partial"
    else:
        status = "ok"

    return Paper(
        arxiv_id=arxiv_id,
        source="unarxive",
        parse_status=status,
        title=title,
        abstract=abstract,
        sections=sections,
        body_text=body_text,
        bibliography=bibliography,
        citation_keys_in_body=sorted(keys_in_body),
        year=_year_from_arxiv_id(arxiv_id),
        notes=notes,
    )


def parse_unarxive_dir(paper_dir: Path) -> Paper:
    """Convenience wrapper: read `paper.json` from disk, then `parse_unarxive`."""
    paper_json_path = paper_dir / "paper.json"
    if not paper_json_path.exists():
        raise FileNotFoundError(f"paper.json not found in {paper_dir}")
    data = json.loads(paper_json_path.read_text(encoding="utf-8"))
    return parse_unarxive(data)
