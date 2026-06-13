"""M-3.1 — section-aware chunking (512 tok, 64 overlap)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from grounded.index.tokenizer import TokenizerLike, count_tokens, load_tokenizer

logger = logging.getLogger(__name__)


def _split_paragraph(paragraph: str, tokenizer: TokenizerLike, chunk_size: int) -> list[str]:
    tokens = tokenizer.encode(paragraph)
    if len(tokens) <= chunk_size:
        return [paragraph] if paragraph.strip() else []
    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        slice_tokens = tokens[start:end]
        parts.append(tokenizer.decode(slice_tokens))
        if end >= len(tokens):
            break
        start = end - 0  # overlap handled at merge level
    return [p.strip() for p in parts if p.strip()]


def _chunk_text_units(
    units: list[str],
    *,
    tokenizer: TokenizerLike,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Pack text units into chunks without crossing unit boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
        current = []
        current_tokens = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        unit_tokens = count_tokens(unit, tokenizer)
        if unit_tokens > chunk_size:
            flush()
            for piece in _split_paragraph(unit, tokenizer, chunk_size):
                chunks.append(piece)
            continue
        if current_tokens + unit_tokens > chunk_size and current:
            flush()
            if chunk_overlap > 0 and chunks:
                overlap_text = chunks[-1]
                overlap_toks = tokenizer.encode(overlap_text)
                tail = overlap_toks[-chunk_overlap:] if len(overlap_toks) > chunk_overlap else overlap_toks
                if hasattr(tokenizer, "decode"):
                    current = [tokenizer.decode(tail)]
                    current_tokens = len(tail)
                else:
                    current = []
                    current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    flush()
    return chunks


def chunk_paper(
    paper: dict[str, Any],
    *,
    tokenizer: TokenizerLike,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Chunk one parsed paper dict; never merge across sections."""
    arxiv_id = paper["arxiv_id"]
    records: list[dict[str, Any]] = []
    sections = paper.get("sections") or []
    if sections:
        for sec_idx, section in enumerate(sections):
            heading = (section.get("heading") or "").strip()
            paragraphs = section.get("paragraphs") or []
            units = [p for p in paragraphs if isinstance(p, str) and p.strip()]
            if not units:
                continue
            for chunk_idx, text in enumerate(
                _chunk_text_units(
                    units,
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            ):
                records.append(
                    {
                        "chunk_id": f"{arxiv_id}:{sec_idx}:{chunk_idx}",
                        "paper_id": arxiv_id,
                        "section_heading": heading,
                        "chunk_idx": chunk_idx,
                        "text": text,
                        "token_count": count_tokens(text, tokenizer),
                    }
                )
        return records

    body = paper.get("body_text") or ""
    if not body.strip():
        return records
    units = [p.strip() for p in body.split("\n\n") if p.strip()]
    for chunk_idx, text in enumerate(
        _chunk_text_units(
            units,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ):
        records.append(
            {
                "chunk_id": f"{arxiv_id}:0:{chunk_idx}",
                "paper_id": arxiv_id,
                "section_heading": "",
                "chunk_idx": chunk_idx,
                "text": text,
                "token_count": count_tokens(text, tokenizer),
            }
        )
    return records


def iter_paper_paths(parsed_dir: Path, arxiv_ids: list[str]) -> Iterator[tuple[str, Path]]:
    for aid in arxiv_ids:
        path = parsed_dir / f"{aid}.json"
        if path.is_file():
            yield aid, path


def chunk_papers_to_records(
    parsed_dir: Path,
    arxiv_ids: list[str],
    *,
    tokenizer_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_name)
    all_records: list[dict[str, Any]] = []
    skipped = 0
    for aid, path in iter_paper_paths(parsed_dir, arxiv_ids):
        try:
            paper = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning("Skip chunking %s: %s", aid, exc)
            skipped += 1
            continue
        if not paper.get("arxiv_id"):
            paper["arxiv_id"] = aid
        all_records.extend(
            chunk_paper(
                paper,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    logger.info(
        "Chunked %d papers → %d chunks (%d skipped)",
        len(arxiv_ids) - skipped,
        len(all_records),
        skipped,
    )
    return all_records


def write_chunks_parquet(records: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)
