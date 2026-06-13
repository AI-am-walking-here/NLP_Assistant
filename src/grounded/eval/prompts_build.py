"""M-4.1 — build eval prompt set from holdout grid + parsed papers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def outline_from_body(paper: dict, *, max_bullets: int = 5) -> str:
    """Heuristic outline from intro/body (no gold abstract sentences)."""
    sections = paper.get("sections") or []
    bullets: list[str] = []
    body_backfill: list[str] = []
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            heading = (sec.get("heading") or sec.get("title") or "").strip()
            if heading and len(heading) > 3:
                bullets.append(heading[:200])
            for para in sec.get("paragraphs") or []:
                if not isinstance(para, str):
                    continue
                clean = re.sub(r"\s+", " ", para).strip()
                if len(clean) > 80:
                    body_backfill.append(clean[:240])
            if len(bullets) >= max_bullets and len(body_backfill) >= max_bullets:
                break

    if len(bullets) < max_bullets:
        body = (paper.get("body_text") or "").strip()
        body = re.sub(r"\s+", " ", body)
        if body:
            body_backfill.extend(
                s.strip()
                for s in SENTENCE_SPLIT.split(body[:4000])
                if len(s.strip()) > 50
            )
    for snippet in body_backfill:
        if len(bullets) >= max_bullets:
            break
        bullets.append(snippet[:240])
    if not bullets:
        return "- (insufficient body text for outline)"
    return "\n".join(f"- {b}" for b in bullets[:max_bullets])


def outline_from_abstract(abstract: str, *, max_bullets: int = 5) -> str:
    """Heuristic outline (no LLM) when HF is unavailable."""
    text = re.sub(r"\s+", " ", abstract.strip())
    if not text:
        return "- (no abstract available)"
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 40]
    if not sentences:
        sentences = [text[:300]]
    bullets = sentences[:max_bullets]
    return "\n".join(f"- {s}" for s in bullets)


def load_paper(parsed_dir: Path, arxiv_id: str) -> dict | None:
    path = parsed_dir / f"{arxiv_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.warning("Skip %s: %s", arxiv_id, exc)
        return None


def build_eval_prompts(
    arxiv_ids: list[str],
    parsed_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    for aid in arxiv_ids:
        paper = load_paper(parsed_dir, aid)
        if not paper:
            continue
        abstract = (paper.get("abstract") or "").strip()
        title = (paper.get("title") or "").strip()
        if not title:
            title = f"arXiv {aid}"
        if not title or not abstract:
            logger.warning("Skip %s: missing title or abstract", aid)
            continue
        rows.append(
            {
                "arxiv_id": aid,
                "title": title,
                "outline": outline_from_body(paper),
                "gold_abstract": abstract,
                "year": paper.get("year"),
                "source": paper.get("source"),
            }
        )
    return rows


def write_prompts_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
