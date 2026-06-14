"""Re-export parsed papers from frozen corpus/papers.jsonl.gz (Segment 1 recovery)."""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CITE_RE = re.compile(r"\{\{cite:([^}]+)\}\}")
COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?")
COMMENT_RE = re.compile(r"(?<!\\)%.*")
SPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
LATEX_SECTION_RE = re.compile(
    r"\\(section|subsection|subsubsection)\*?\s*\{([^{}]+)\}"
)

DEFAULT_CORPUS = Path("data/corpus/papers.jsonl.gz")


def paper_year(arxiv_id: str) -> int | None:
    if not arxiv_id or len(arxiv_id) < 2 or not arxiv_id[:2].isdigit():
        return None
    yy = int(arxiv_id[:2])
    return 2000 + yy if yy <= 99 else yy


def map_source(source: str) -> str:
    if source == "arxiv_s3":
        return "latex_s3"
    return source or "unknown"


def safe_parsed_filename(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_").replace(":", "_") + ".json"


def read_braced(text: str, command: str) -> str:
    marker = "\\" + command
    i = text.find(marker)
    while i != -1:
        j = i + len(marker)
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] == "[":
            depth = 1
            j += 1
            while j < len(text) and depth:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
            while j < len(text) and text[j].isspace():
                j += 1
        if j < len(text) and text[j] == "{":
            start = j + 1
            depth = 1
            j += 1
            while j < len(text) and depth:
                ch = text[j]
                prev = text[j - 1] if j else ""
                if ch == "{" and prev != "\\":
                    depth += 1
                elif ch == "}" and prev != "\\":
                    depth -= 1
                j += 1
            if depth == 0:
                return text[start : j - 1]
        i = text.find(marker, i + 1)
    return ""


def read_environment(text: str, env: str) -> str:
    m = re.search(r"\\begin\s*\{" + re.escape(env) + r"\}", text)
    if not m:
        return ""
    n = re.search(r"\\end\s*\{" + re.escape(env) + r"\}", text[m.end() :])
    if not n:
        return ""
    return text[m.end() : m.end() + n.start()]


def clean_latex_text(value: str) -> str:
    if not value:
        return ""
    value = COMMENT_RE.sub("", value)
    value = value.replace("\\n", " ")
    value = re.sub(r"\{\{(?:formula|figure|table|cite):[^}]+\}\}", " ", value)
    value = value.replace("REF", " ")
    value = re.sub(
        r"\\(textbf|textit|emph|texttt|url|href)\s*\{([^{}]*)\}",
        r"\2",
        value,
    )
    value = COMMAND_RE.sub(" ", value)
    value = value.replace("{", " ").replace("}", " ")
    value = value.replace("~", " ")
    return SPACE_RE.sub(" ", value).strip()


def split_latex_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    matches = list(LATEX_SECTION_RE.finditer(text))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        level = {"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)]
        body = clean_latex_text(text[start:end])
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        sections.append(
            {
                "heading": clean_latex_text(m.group(2)),
                "level": level,
                "paragraphs": paragraphs[:50],
            }
        )
    return sections


def extract_year_from_raw(raw: str | None) -> int | None:
    if not raw:
        return None
    for match in YEAR_RE.finditer(raw):
        year = int(match.group(0))
        if 1900 <= year <= 2099:
            return year
    return None


def convert_bib_entry(key: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "citation_key": key,
            "raw_entry": str(value) if value is not None else None,
            "title": None,
            "year": None,
            "s2_paper_id": None,
        }
    raw = value.get("bib_entry_raw") or value.get("raw_entry")
    entry: dict[str, Any] = {
        "citation_key": key,
        "raw_entry": raw,
        "title": value.get("title"),
        "year": value.get("year") or extract_year_from_raw(raw),
        "s2_paper_id": value.get("s2_paper_id"),
    }
    if value.get("discipline"):
        entry["discipline"] = value["discipline"]
    if isinstance(value.get("ids"), dict):
        entry["ids"] = value["ids"]
    return entry


def normalize_bibliography(
    raw_bib: Any,
    text: str,
) -> dict[str, dict[str, Any]]:
    if isinstance(raw_bib, dict) and raw_bib:
        return {key: convert_bib_entry(key, val) for key, val in raw_bib.items()}
    keys: list[str] = []
    seen: set[str] = set()
    for match in CITE_RE.finditer(text or ""):
        key = match.group(1).strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return {
        key: {
            "citation_key": key,
            "raw_entry": None,
            "title": None,
            "year": None,
            "s2_paper_id": None,
        }
        for key in keys
    }


def citation_keys_in_body(text: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for match in CITE_RE.finditer(text or ""):
        key = match.group(1).strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def reference_bundle(raw_references: Any, raw_bibliography: Any) -> dict[str, Any]:
    refs = raw_references if isinstance(raw_references, dict) else {}
    bib_entries = refs.get("bib_entries")
    if not isinstance(bib_entries, dict) or not bib_entries:
        bib_entries = raw_bibliography if isinstance(raw_bibliography, dict) else {}
    return {
        "bib_entries": bib_entries,
        "ref_entries": refs.get("ref_entries", {}) if isinstance(refs.get("ref_entries"), dict) else {},
        "citation_spans": refs.get("citation_spans", []) if isinstance(refs.get("citation_spans"), list) else [],
        "reference_spans": refs.get("reference_spans", []) if isinstance(refs.get("reference_spans"), list) else [],
    }


def parse_status_for(title: str, abstract: str, body_text: str) -> str:
    if title and abstract and body_text:
        return "ok"
    if title or abstract:
        return "partial"
    return "failed"


_ENRICHMENT_KEYS = ("citation_count", "venue", "s2_paper_id", "year")


def merge_preserved_enrichment(
    paper: dict[str, Any],
    parsed_path: Path | None,
) -> dict[str, Any]:
    """Keep S2/OpenAlex fields when re-exporting from corpus."""
    if parsed_path is None or not parsed_path.is_file():
        return paper
    try:
        existing = json.loads(parsed_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return paper
    for key in _ENRICHMENT_KEYS:
        if existing.get(key) is not None and paper.get(key) is None:
            paper[key] = existing[key]
    return paper


def load_preserved_sections(parsed_path: Path) -> list[dict[str, Any]] | None:
    if not parsed_path.is_file():
        return None
    try:
        data = json.loads(parsed_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    sections = data.get("sections")
    if isinstance(sections, list) and sections:
        return sections
    return None


def corpus_record_to_paper(
    raw: dict[str, Any],
    *,
    preserve_sections_path: Path | None = None,
) -> dict[str, Any]:
    arxiv_id = str(raw.get("id", "")).strip()
    text = raw.get("text") or ""
    title = (raw.get("title") or "").strip()
    abstract = (raw.get("abstract") or "").strip()
    corpus_source = raw.get("source") or "unknown"
    source = map_source(corpus_source)

    if not title and corpus_source == "arxiv_s3":
        title = clean_latex_text(read_braced(text, "title"))
    if not abstract and corpus_source == "arxiv_s3":
        abstract = clean_latex_text(read_environment(text, "abstract"))

    raw_references = raw.get("references") if isinstance(raw.get("references"), dict) else {}
    raw_bibliography = raw.get("bibliography")
    if not raw_bibliography and isinstance(raw_references.get("bib_entries"), dict):
        raw_bibliography = raw_references["bib_entries"]

    bibliography = normalize_bibliography(raw_bibliography, text)
    cite_keys = citation_keys_in_body(text)

    sections: list[dict[str, Any]] = []
    if preserve_sections_path is not None:
        preserved = load_preserved_sections(preserve_sections_path)
        # Only reuse sections when the on-disk file is fully valid JSON.
        if preserved:
            sections = preserved
    if not sections and source == "latex_s3":
        sections = split_latex_sections(text)

    return {
        "arxiv_id": arxiv_id,
        "source": source,
        "parse_status": parse_status_for(title, abstract, text),
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "bibliography": bibliography,
        "body_text": clean_latex_text(text),
        "citation_keys_in_body": cite_keys,
        "references": reference_bundle(raw_references, raw_bibliography),
        "year": paper_year(arxiv_id),
        "venue": None,
        "citation_count": None,
        "s2_paper_id": None,
        "notes": [],
    }


def manifest_row(paper: dict[str, Any]) -> dict[str, Any]:
    body = paper.get("body_text") or ""
    abstract = paper.get("abstract") or ""
    bibliography = paper.get("bibliography") or {}
    return {
        "arxiv_id": paper["arxiv_id"],
        "source": paper["source"],
        "parse_status": paper.get("parse_status", "ok"),
        "year": paper.get("year"),
        "num_sections": len(paper.get("sections") or []),
        "num_citation_keys": len(paper.get("citation_keys_in_body") or []),
        "num_bib_entries": len(bibliography) if isinstance(bibliography, dict) else 0,
        "abstract_len": len(abstract),
        "body_len": len(body),
        "notes": paper.get("notes") or [],
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    json.loads(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def iter_corpus(corpus_path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(corpus_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def rebuild_manifest_from_parsed(
    parsed_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Rebuild parsed_manifest.jsonl from on-disk Paper JSON (accurate body/cite counts)."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in sorted(parsed_dir.glob("*.json")):
        try:
            paper = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            skipped += 1
            continue
        if not paper.get("arxiv_id"):
            paper["arxiv_id"] = path.stem
        rows.append(manifest_row(paper))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "manifest_path": str(manifest_path),
        "rows_written": len(rows),
        "skipped_unreadable": skipped,
    }


def export_corpus_ids(
    corpus_path: Path,
    parsed_dir: Path,
    arxiv_ids: set[str],
    *,
    preserve_sections: bool = False,
) -> dict[str, Any]:
    """Re-export a subset of corpus rows (recovery for corrupt parsed JSON)."""
    stats: Counter[str] = Counter()
    wanted = set(arxiv_ids)
    for raw in iter_corpus(corpus_path):
        arxiv_id = str(raw.get("id", "")).strip()
        if arxiv_id not in wanted:
            continue
        out_path = parsed_dir / safe_parsed_filename(arxiv_id)
        preserve_path = out_path if preserve_sections else None
        paper = corpus_record_to_paper(raw, preserve_sections_path=preserve_path)
        paper = merge_preserved_enrichment(paper, out_path)
        atomic_write_json(out_path, paper)
        stats["written"] += 1
        wanted.discard(arxiv_id)
        if not wanted:
            break
    if wanted:
        stats["not_in_corpus"] = len(wanted)
        logger.warning("IDs not found in corpus: %s", sorted(wanted)[:10])
    return {"written": stats["written"], "not_in_corpus": len(wanted)}


def export_corpus(
    corpus_path: Path,
    parsed_dir: Path,
    manifest_path: Path,
    *,
    preserve_sections: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []

    parsed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, raw in enumerate(iter_corpus(corpus_path)):
        if limit is not None and idx >= limit:
            break
        arxiv_id = str(raw.get("id", "")).strip()
        if not arxiv_id:
            stats["missing_id"] += 1
            continue

        out_path = parsed_dir / safe_parsed_filename(arxiv_id)
        preserve_path = out_path if preserve_sections else None
        paper = corpus_record_to_paper(raw, preserve_sections_path=preserve_path)
        paper = merge_preserved_enrichment(paper, out_path)
        atomic_write_json(out_path, paper)
        manifest_rows.append(manifest_row(paper))

        stats["written"] += 1
        stats[f"source:{paper['source']}"] += 1
        stats[f"parse_status:{paper['parse_status']}"] += 1
        if paper.get("sections"):
            stats["with_sections"] += 1
        if paper.get("bibliography"):
            stats["with_bibliography"] += 1
        if paper.get("references"):
            stats["with_references"] += 1

    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "corpus_path": str(corpus_path),
        "parsed_dir": str(parsed_dir),
        "manifest_path": str(manifest_path),
        "written": stats["written"],
        "with_sections": stats["with_sections"],
        "with_bibliography": stats["with_bibliography"],
        "with_references": stats["with_references"],
        "source_counts": {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("source:")},
        "parse_status_counts": {
            k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("parse_status:")
        },
        "missing_id": stats["missing_id"],
    }
