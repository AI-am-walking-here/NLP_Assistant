"""M-2.4 — paper metadata enrichment (Semantic Scholar + OpenAlex fallback)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "paperId,externalIds,title,year,venue,citationCount"
S2_BATCH_SIZE = 500
OPENALEX_BATCH_SIZE = 25
OPENALEX_URL = "https://api.openalex.org/works"
MAX_RETRIES = 8

EnrichmentProvider = Literal["s2", "openalex", "auto"]


def read_parsed_record(path: Path) -> dict[str, Any] | None:
    """Load a parsed paper JSON, or None if unreadable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.debug("Unreadable parsed JSON %s: %s", path.name, exc)
        return None


def write_parsed_record(path: Path, record: dict[str, Any]) -> None:
    """Write parsed paper JSON (raw dict; no Pydantic round-trip)."""
    from grounded.data.corpus_export import atomic_write_json

    atomic_write_json(path, record)


def arxiv_to_doi(arxiv_id: str) -> str:
    return f"10.48550/arXiv.{arxiv_id}"


def arxiv_from_openalex_work(work: dict[str, Any]) -> str | None:
    ids = work.get("ids") or {}
    arxiv = ids.get("arxiv")
    if arxiv:
        return arxiv.rstrip("/").split("/")[-1]
    doi = (ids.get("doi") or "").lower()
    for marker in ("arxiv.", "arxiv/"):
        if marker in doi:
            return doi.split(marker, 1)[-1].rstrip("/")
    return None


def _patch_record(record: dict[str, Any], meta: dict[str, Any]) -> bool:
    changed = False
    if meta.get("s2_paper_id"):
        record["s2_paper_id"] = meta["s2_paper_id"]
        changed = True
    if "citation_count" in meta and meta["citation_count"] is not None:
        record["citation_count"] = meta["citation_count"]
        changed = True
    if meta.get("venue"):
        record["venue"] = meta["venue"]
        changed = True
    if meta.get("year") is not None and not record.get("year"):
        record["year"] = meta["year"]
        changed = True
    return changed


def apply_meta_to_paper(parsed_path: Path, meta: dict[str, Any]) -> bool:
    """Patch enrichment fields in place without rewriting via Pydantic."""
    record = read_parsed_record(parsed_path)
    if record is None:
        logger.warning("Skipping unreadable parsed JSON %s", parsed_path.name)
        return False
    _patch_record(record, meta)
    write_parsed_record(parsed_path, record)
    return True


class MetadataCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    @staticmethod
    def _cache_row_rank(row: dict[str, Any]) -> tuple[int, int]:
        """Prefer real enrichment over trailing not_found zero-fill rows."""
        provider = row.get("provider")
        if provider in (None, "not_found", "?"):
            return (0, int(row.get("citation_count") or 0))
        cc = row.get("citation_count")
        if cc is None:
            return (1, 0)
        return (2, int(cc))

    def _load_cache(self) -> None:
        if not self.cache_path.is_file():
            return
        with self.cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row.get("arxiv_id")
                if not key:
                    continue
                prev = self._cache.get(key)
                if prev is None or self._cache_row_rank(row) > self._cache_row_rank(prev):
                    self._cache[key] = row

    def get(self, arxiv_id: str) -> dict[str, Any] | None:
        return self._cache.get(arxiv_id)

    def needs_remote_fetch(self, arxiv_id: str, *, refresh_not_found: bool = False) -> bool:
        row = self._cache.get(arxiv_id)
        if row is None:
            return True
        if not refresh_not_found:
            return False
        provider = row.get("provider")
        return provider in (None, "not_found", "?")

    def append(self, row: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        key = row.get("arxiv_id")
        if key:
            self._cache[key] = row


class S2Client:
    def __init__(self, cache: MetadataCache, api_key: str | None = None):
        self.cache = cache
        self.api_key = api_key or os.getenv("S2_API_KEY")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    @property
    def min_interval_s(self) -> float:
        return 0.15 if self.api_key else 3.0

    def _post_batch(self, ids: list[str]) -> list[Any]:
        delay = self.min_interval_s
        last_status = 0
        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                S2_BATCH_URL,
                params={"fields": S2_FIELDS},
                headers=self._headers(),
                json={"ids": ids},
                timeout=120,
            )
            last_status = resp.status_code
            if resp.status_code == 429:
                retry_after = float(
                    resp.headers.get("Retry-After", delay * (attempt + 2))
                )
                logger.warning(
                    "S2 rate limit (429); sleeping %.1fs (attempt %d/%d)",
                    retry_after,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(retry_after)
                delay = min(delay * 2, 60.0)
                continue
            if resp.status_code >= 500:
                time.sleep(delay * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        raise requests.HTTPError(
            f"S2 batch failed after {MAX_RETRIES} retries (last status {last_status})"
        )

    def fetch_batch(
        self,
        arxiv_ids: list[str],
        *,
        refresh_not_found: bool = False,
    ) -> dict[str, dict[str, Any]]:
        missing = [
            aid
            for aid in arxiv_ids
            if self.cache.needs_remote_fetch(aid, refresh_not_found=refresh_not_found)
        ]
        results: dict[str, dict[str, Any]] = {
            aid: meta
            for aid in arxiv_ids
            if (meta := self.cache.get(aid)) is not None
            and not self.cache.needs_remote_fetch(aid, refresh_not_found=refresh_not_found)
        }
        for i in range(0, len(missing), S2_BATCH_SIZE):
            chunk = missing[i : i + S2_BATCH_SIZE]
            ids = [f"ARXIV:{aid}" for aid in chunk]
            payload = self._post_batch(ids)
            if not isinstance(payload, list):
                logger.warning("Unexpected S2 batch response: %s", type(payload))
                continue
            for paper in payload:
                if not paper:
                    continue
                ext = paper.get("externalIds") or {}
                arxiv = ext.get("ArXiv") or ext.get("arXiv")
                if not arxiv:
                    continue
                row = {
                    "arxiv_id": arxiv,
                    "provider": "s2",
                    "s2_paper_id": paper.get("paperId"),
                    "citation_count": paper.get("citationCount"),
                    "venue": paper.get("venue"),
                    "year": paper.get("year"),
                    "title": paper.get("title"),
                }
                self.cache.append(row)
                results[arxiv] = row
            time.sleep(self.min_interval_s)
        return results


class OpenAlexClient:
    def __init__(self, cache: MetadataCache, mailto: str | None = None):
        self.cache = cache
        self.mailto = mailto or os.getenv("OPENALEX_MAILTO", "grounded-poc@example.com")

    def fetch_batch(
        self,
        arxiv_ids: list[str],
        *,
        refresh_not_found: bool = False,
    ) -> dict[str, dict[str, Any]]:
        missing = [
            aid
            for aid in arxiv_ids
            if self.cache.needs_remote_fetch(aid, refresh_not_found=refresh_not_found)
        ]
        results: dict[str, dict[str, Any]] = {
            aid: meta
            for aid in arxiv_ids
            if (meta := self.cache.get(aid)) is not None
            and not self.cache.needs_remote_fetch(aid, refresh_not_found=refresh_not_found)
        }
        for i in range(0, len(missing), OPENALEX_BATCH_SIZE):
            chunk = missing[i : i + OPENALEX_BATCH_SIZE]
            doi_filter = "|".join(arxiv_to_doi(aid) for aid in chunk)
            resp = requests.get(
                OPENALEX_URL,
                params={
                    "mailto": self.mailto,
                    "filter": f"doi:{doi_filter}",
                    "per_page": OPENALEX_BATCH_SIZE,
                },
                timeout=120,
            )
            resp.raise_for_status()
            for work in resp.json().get("results") or []:
                arxiv = arxiv_from_openalex_work(work)
                if not arxiv:
                    continue
                venue = None
                pl = work.get("primary_location") or {}
                src = pl.get("source") or {}
                if src.get("display_name"):
                    venue = src["display_name"]
                cited = work.get("cited_by_count")
                row = {
                    "arxiv_id": arxiv,
                    "provider": "openalex",
                    "s2_paper_id": (work.get("ids") or {}).get("openalex"),
                    "citation_count": 0 if cited is None else cited,
                    "venue": venue,
                    "year": work.get("publication_year"),
                    "title": work.get("display_name"),
                }
                self.cache.append(row)
                results[arxiv] = row
            time.sleep(0.12)
        return results


def apply_cached_to_parsed(
    arxiv_ids: list[str],
    parsed_dir: Path,
    cache: MetadataCache,
) -> int:
    updated = 0
    for aid in arxiv_ids:
        meta = cache.get(aid)
        if not meta:
            continue
        path = parsed_dir / f"{aid}.json"
        if apply_meta_to_paper(path, meta):
            updated += 1
    return updated


def ids_missing_citation_count(
    arxiv_ids: list[str],
    parsed_dir: Path,
) -> list[str]:
    missing: list[str] = []
    for aid in arxiv_ids:
        record = read_parsed_record(parsed_dir / f"{aid}.json")
        if record is None:
            continue
        if record.get("citation_count") is None:
            missing.append(aid)
    return missing


def mark_not_found_zero(
    arxiv_ids: list[str],
    parsed_dir: Path,
    cache: MetadataCache,
) -> int:
    row = {"provider": "not_found", "citation_count": 0}
    marked = 0
    for aid in arxiv_ids:
        cache.append({**row, "arxiv_id": aid})
        path = parsed_dir / f"{aid}.json"
        if apply_meta_to_paper(path, row):
            marked += 1
    return marked


def resolve_provider(requested: EnrichmentProvider) -> str:
    if requested == "auto":
        return "s2" if os.getenv("S2_API_KEY") else "openalex"
    return requested


def _apply_batch_meta(
    batch: list[str],
    meta_map: dict[str, dict[str, Any]],
    parsed_dir: Path,
) -> tuple[int, int]:
    enriched = 0
    missing_meta = 0
    for aid in batch:
        meta = meta_map.get(aid)
        path = parsed_dir / f"{aid}.json"
        if not path.is_file():
            missing_meta += 1
            continue
        if not meta:
            missing_meta += 1
            continue
        if apply_meta_to_paper(path, meta):
            enriched += 1
    return enriched, missing_meta


def enrich_papers(
    arxiv_ids: list[str],
    parsed_dir: Path,
    cache_path: Path,
    *,
    provider: EnrichmentProvider = "auto",
    api_key: str | None = None,
    fetch_remote: bool = True,
    refresh_not_found: bool = False,
) -> dict[str, Any]:
    cache = MetadataCache(cache_path)
    chosen = resolve_provider(provider)
    enriched = apply_cached_to_parsed(arxiv_ids, parsed_dir, cache)

    if fetch_remote:
        if chosen == "s2":
            client: S2Client | OpenAlexClient = S2Client(cache, api_key=api_key)
            batch_size = S2_BATCH_SIZE
        else:
            client = OpenAlexClient(cache)
            batch_size = OPENALEX_BATCH_SIZE

        missing_meta = 0
        still_missing = ids_missing_citation_count(arxiv_ids, parsed_dir)

        for i in range(0, len(still_missing), batch_size):
            batch = still_missing[i : i + batch_size]
            try:
                meta_map = client.fetch_batch(batch, refresh_not_found=refresh_not_found)
            except requests.HTTPError as exc:
                if chosen == "s2" and provider == "auto":
                    logger.warning("S2 failed (%s); falling back to OpenAlex", exc)
                    client = OpenAlexClient(cache)
                    batch_size = OPENALEX_BATCH_SIZE
                    chosen = "openalex"
                    meta_map = client.fetch_batch(batch, refresh_not_found=refresh_not_found)
                else:
                    raise
            n, miss = _apply_batch_meta(batch, meta_map, parsed_dir)
            enriched += n
            missing_meta += miss

        still_missing = ids_missing_citation_count(arxiv_ids, parsed_dir)
        marked_zero = mark_not_found_zero(still_missing, parsed_dir, cache)
        enriched += marked_zero
        if marked_zero:
            logger.info(
                "Set citation_count=0 for %d papers not found via API",
                marked_zero,
            )
    else:
        missing_meta = 0
        marked_zero = 0

    final_missing = ids_missing_citation_count(arxiv_ids, parsed_dir)

    return {
        "requested": len(arxiv_ids),
        "enriched_files": enriched,
        "missing_meta": missing_meta if fetch_remote else 0,
        "still_missing_citation_count": len(final_missing),
        "marked_not_found_zero": marked_zero if fetch_remote else 0,
        "provider": chosen,
        "refresh_not_found": refresh_not_found,
        "cache_path": str(cache_path),
    }


def prune_enriched_valid_ids(
    arxiv_ids: list[str],
    parsed_dir: Path,
) -> tuple[list[str], dict[str, list[str]]]:
    """Keep only IDs with readable JSON and a set citation_count."""
    kept: list[str] = []
    excluded: dict[str, list[str]] = {
        "unreadable": [],
        "missing_citation_count": [],
    }
    for aid in arxiv_ids:
        record = read_parsed_record(parsed_dir / f"{aid}.json")
        if record is None:
            excluded["unreadable"].append(aid)
            continue
        if record.get("citation_count") is None:
            excluded["missing_citation_count"].append(aid)
            continue
        kept.append(aid)
    return kept, excluded
