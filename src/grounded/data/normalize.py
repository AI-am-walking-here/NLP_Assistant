"""Segment 2 orchestrator: two ingress paths → one unified `data/parsed/`.

Walks `data/archive/{unarxive_extracted, tex_extracted}/`, dispatches each
paper directory to the right parser, writes `data/parsed/<arxiv_id>.json`,
and emits a one-row-per-paper manifest at `data/parsed_manifest.jsonl`.

Pure orchestration — all parsing logic lives in `unarxive_parse.py` and
`latex_parse.py`. Adding a third source = adding one entry to `_PARSERS`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from grounded.data.latex_parse import parse_latex_dir
from grounded.data.schema import Paper
from grounded.data.unarxive_parse import parse_unarxive_dir
from grounded.progress import CountProgressReporter


ParserFn = Callable[[Path], Paper]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    archive_subdir: str
    parser: ParserFn


_PARSERS: tuple[SourceSpec, ...] = (
    SourceSpec("unarxive", "unarxive_extracted", parse_unarxive_dir),
    SourceSpec("latex_s3", "tex_extracted", parse_latex_dir),
)


@dataclass
class NormalizeStats:
    total: int = 0
    ok: int = 0
    partial: int = 0
    failed: int = 0
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, source: str, status: str) -> None:
        self.total += 1
        if status == "ok":
            self.ok += 1
        elif status == "partial":
            self.partial += 1
        else:
            self.failed += 1
        bucket = self.per_source.setdefault(source, {"ok": 0, "partial": 0, "failed": 0})
        bucket[status] = bucket.get(status, 0) + 1

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "ok": self.ok,
            "partial": self.partial,
            "failed": self.failed,
            "per_source": self.per_source,
        }


def _manifest_row(paper: Paper) -> dict:
    return {
        "arxiv_id": paper.arxiv_id,
        "source": paper.source,
        "parse_status": paper.parse_status,
        "year": paper.year,
        "num_sections": len(paper.sections),
        "num_citation_keys": len(paper.citation_keys_in_body),
        "num_bib_entries": len(paper.bibliography),
        "abstract_len": len(paper.abstract),
        "body_len": len(paper.body_text),
        "notes": paper.notes,
    }


def _write_paper(output_dir: Path, paper: Paper) -> Path:
    path = output_dir / f"{paper.arxiv_id}.json"
    path.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
    return path


def _failure_paper(arxiv_id: str, source: str, reason: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        source=source,  # type: ignore[arg-type]
        parse_status="failed",
        title="",
        abstract="",
        sections=[],
        body_text="",
        notes=[f"exception: {reason}"],
    )


def normalize_source(
    spec: SourceSpec,
    archive_root: Path,
    output_dir: Path,
    manifest_handle,
    limit: int | None = None,
    show_progress: bool = True,
) -> NormalizeStats:
    """Normalize all papers under one source. Streams manifest rows."""
    stats = NormalizeStats()
    src_dir = archive_root / spec.archive_subdir
    if not src_dir.exists():
        return stats

    paper_dirs = sorted(p for p in src_dir.iterdir() if p.is_dir())
    if limit is not None:
        paper_dirs = paper_dirs[:limit]

    reporter = (
        CountProgressReporter(f"normalize-{spec.name}", total=len(paper_dirs), unit="papers")
        if show_progress and paper_dirs
        else None
    )

    for pdir in paper_dirs:
        try:
            paper = spec.parser(pdir)
        except Exception as exc:  # parser-level safety net
            paper = _failure_paper(pdir.name, spec.name, repr(exc))

        _write_paper(output_dir, paper)
        manifest_handle.write(json.dumps(_manifest_row(paper)) + "\n")
        stats.record(spec.name, paper.parse_status)

        if reporter:
            reporter.update(1, detail=f"ok={stats.ok} partial={stats.partial} failed={stats.failed}")

    if reporter:
        reporter.finish(
            detail=f"ok={stats.ok} partial={stats.partial} failed={stats.failed}"
        )
    return stats


def normalize_all(
    archive_root: Path,
    output_dir: Path,
    manifest_path: Path,
    sources: tuple[str, ...] | None = None,
    limit_per_source: int | None = None,
    show_progress: bool = True,
) -> NormalizeStats:
    """Walk all configured sources, write one JSON per paper + a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    allowed = set(sources) if sources else {spec.name for spec in _PARSERS}
    combined = NormalizeStats()

    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for spec in _PARSERS:
            if spec.name not in allowed:
                continue
            source_stats = normalize_source(
                spec,
                archive_root=archive_root,
                output_dir=output_dir,
                manifest_handle=manifest_handle,
                limit=limit_per_source,
                show_progress=show_progress,
            )
            combined.total += source_stats.total
            combined.ok += source_stats.ok
            combined.partial += source_stats.partial
            combined.failed += source_stats.failed
            if spec.name in source_stats.per_source:
                combined.per_source[spec.name] = dict(source_stats.per_source[spec.name])

    return combined
