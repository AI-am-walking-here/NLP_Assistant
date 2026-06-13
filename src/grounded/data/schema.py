"""Unified pydantic schemas — the inter-segment contracts.

These types are the boundary between data acquisition, parsing, retrieval,
generation, and evaluation. Two ingress parsers (`unarxive_parse.py`,
`latex_parse.py`) both produce `Paper`; everything downstream reads `Paper`
and never knows the source.

Keep this file dependency-light (only pydantic + stdlib). Bigger objects
(embeddings, model outputs) belong in artifact files, not these schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt


Source = Literal["unarxive", "latex_s3"]
ParseStatus = Literal["ok", "partial", "failed"]


class Section(BaseModel):
    """A top-level section of a paper.

    `level=1` is `\\section{}`, `level=2` is `\\subsection{}`, etc. unarXive
    inputs that lack explicit section markers collapse to a single
    `Section(heading="body", level=1, ...)`.
    """

    heading: str
    level: NonNegativeInt = 1
    paragraphs: list[str] = Field(default_factory=list)


class ReferenceBundle(BaseModel):
    """Structured references from unarXive / LaTeX parse (optional on Paper JSON)."""

    bib_entries: dict[str, Any] = Field(default_factory=dict)
    ref_entries: dict[str, Any] = Field(default_factory=dict)
    citation_spans: list[Any] = Field(default_factory=list)
    reference_spans: list[Any] = Field(default_factory=list)


class BibEntry(BaseModel):
    """One bibliography record resolved against the paper's `.bib` file.

    `s2_paper_id` is populated by the Semantic Scholar enrichment step
    (M-2.4) when a title match succeeds; missing for unmatched entries
    and for unarXive papers where we don't reconstruct `.bib`.
    """

    citation_key: str
    raw_entry: str | None = None
    title: str | None = None
    year: int | None = None
    s2_paper_id: str | None = None


class Paper(BaseModel):
    """Canonical normalized paper. One JSON file per paper in `data/parsed/`."""

    arxiv_id: str
    source: Source
    parse_status: ParseStatus = "ok"

    title: str
    abstract: str
    sections: list[Section] = Field(default_factory=list)
    body_text: str = ""

    bibliography: dict[str, BibEntry] = Field(default_factory=dict)
    citation_keys_in_body: list[str] = Field(default_factory=list)

    year: int | None = None
    venue: str | None = None
    citation_count: NonNegativeInt | None = None
    s2_paper_id: str | None = None

    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: str | Path) -> Paper:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class Chunk(BaseModel):
    """A 512-token (with 64 overlap), section-aware slice of a paper.

    `text` retains `{{cite:KEY}}` markers from the source paper's body —
    these are retrieval/FActScore signals, never user-visible output.
    """

    chunk_id: str
    paper_id: str
    section_heading: str
    chunk_idx: NonNegativeInt
    text: str
    token_count: NonNegativeInt
    citation_keys: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A resolved citation produced by retrieval (NOT by the generator).

    Used internally by FActScore + the demo's "supporting passages"
    sidebar; not embedded in generated abstracts (per v3.1).
    """

    citation_key: str
    paper_id: str | None = None
    s2_paper_id: str | None = None
    evidence_chunk_id: str | None = None
    score: float | None = None


class RetrievalResult(BaseModel):
    """Output of a single retriever call."""

    query: str
    chunks: list[Chunk]
    scores: list[float] = Field(default_factory=list)
    retriever_name: str


class Claim(BaseModel):
    """An atomic claim extracted from a generated abstract for FActScore."""

    claim_id: str
    text: str
    abstract_id: str
    support: Literal["supported", "partial", "not_supported", "unknown"] = "unknown"
    supporting_chunks: list[str] = Field(default_factory=list)
    verifier_reasoning: str | None = None


class GenerationResult(BaseModel):
    """The output of a full pipeline run for one prompt."""

    prompt_id: str
    title: str
    outline: str
    abstract_text: str
    supporting_chunks: list[Chunk] = Field(default_factory=list)
    system_name: str
    generation_params: dict[str, float | int | str | bool] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Aggregated metrics for one (system, prompt-set) pair."""

    system_name: str
    prompt_set_name: str
    num_prompts: NonNegativeInt
    factscore: NonNegativeFloat | None = None
    ragas_faithfulness: NonNegativeFloat | None = None
    ragas_context_relevance: NonNegativeFloat | None = None
    paired_bootstrap_ci: tuple[float, float] | None = None
    notes: list[str] = Field(default_factory=list)
