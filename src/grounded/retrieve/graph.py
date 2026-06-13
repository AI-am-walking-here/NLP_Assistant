"""M-6.5 — graph retriever over community summaries."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_CAND_EMBED = int(os.environ.get("GROUNDED_GRAPH_MAX_CAND_EMBED", "48"))


def _lexical_prefilter(query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(candidates) <= limit:
        return candidates
    q_tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    if not q_tokens:
        return candidates[:limit]

    def score(row: dict[str, Any]) -> int:
        t_tokens = set(re.findall(r"[a-z0-9]{3,}", str(row.get("text", "")).lower()))
        return len(q_tokens & t_tokens)

    return sorted(candidates, key=score, reverse=True)[:limit]


class GraphRetriever:
    def __init__(
        self,
        communities: list[dict[str, Any]],
        chunks_by_paper: dict[str, list[dict[str, Any]]],
        embedder: Any,
    ) -> None:
        self._communities = communities
        self._chunks_by_paper = chunks_by_paper
        self._embedder = embedder
        self._summaries = [c["summary"] for c in communities]
        self._emb_matrix: np.ndarray | None = None
        if self._summaries:
            self._emb_matrix = np.asarray(
                embedder.encode(self._summaries, show_progress=False),
                dtype=np.float32,
            )

    @classmethod
    def from_parquet(
        cls,
        communities_path: Path,
        chunks_parquet: Path,
        embedder: Any,
        *,
        paper_filter: set[str] | None = None,
    ) -> GraphRetriever:
        import pyarrow.parquet as pq

        comm_rows = pq.read_table(communities_path).to_pylist()
        chunk_table = pq.read_table(
            chunks_parquet,
            columns=["chunk_id", "paper_id", "section_heading", "text", "token_count"],
        )
        chunks_by_paper: dict[str, list[dict[str, Any]]] = {}
        for row in chunk_table.to_pylist():
            pid = row["paper_id"]
            if paper_filter and pid not in paper_filter:
                continue
            chunks_by_paper.setdefault(pid, []).append(row)
        return cls(comm_rows, chunks_by_paper, embedder)

    def search(
        self,
        query: str,
        *,
        k_communities: int = 5,
        k_chunks_per_community: int = 4,
    ) -> list[dict[str, Any]]:
        if not self._communities or self._emb_matrix is None:
            return []
        q = np.asarray(self._embedder.encode([query])[0], dtype=np.float32)
        scores = self._emb_matrix @ q
        order = np.argsort(-scores)[:k_communities]
        hits: list[dict[str, Any]] = []
        for rank, idx in enumerate(order):
            comm = self._communities[int(idx)]
            paper_ids = json.loads(comm["paper_ids_json"])
            score = float(scores[int(idx)])
            candidates: list[dict[str, Any]] = []
            for pid in paper_ids:
                candidates.extend(self._chunks_by_paper.get(pid, []))
            if not candidates:
                continue
            candidates = _lexical_prefilter(query, candidates, _MAX_CAND_EMBED)
            cand_texts = [c.get("text", "") for c in candidates]
            cand_embs = np.asarray(
                self._embedder.encode(cand_texts, show_progress=False),
                dtype=np.float32,
            )
            cand_scores = cand_embs @ q
            cand_order = np.argsort(-cand_scores)[:k_chunks_per_community]
            for cand_idx in cand_order:
                chunk = candidates[int(cand_idx)]
                row = dict(chunk)
                row["score"] = float(cand_scores[int(cand_idx)])
                row["community_score"] = score
                row["rank"] = len(hits)
                row["community_id"] = comm["community_id"]
                hits.append(row)
        return hits
