"""Retrieve passages for retrieval-conditioned SFT (exclude self-paper chunks)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from grounded.generate.prompts import format_retrieved_chunks

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_generation_stack() -> tuple[Any, Any, Any, Any]:
    from grounded.config import load_config
    from grounded.eval.runner import load_graph_retriever, load_retrieval_stack
    from grounded.generate.pipeline import resolve_latest_adapter
    from grounded.retrieve.rerank import load_reranker

    retr_cfg = load_config("retrieval")
    rank_path = resolve_latest_adapter("seg6_rankrag_*")
    if rank_path is None:
        raise FileNotFoundError(
            "RankRAG adapter required for retrieval-conditioned SFT. "
            "Train with scripts/rankrag_train.py first."
        )
    store, embedder, _meta = load_retrieval_stack(retr_cfg, require_real_index=True)
    graph = load_graph_retriever(retr_cfg)
    reranker = load_reranker(rank_path, mock=False)
    return store, embedder, graph, reranker


def retrieve_training_passages(
    title: str,
    outline: str,
    paper_id: str,
    *,
    top_k: int = 8,
    n_vector: int = 15,
    n_graph: int = 15,
    n_candidates: int = 30,
) -> list[dict[str, Any]]:
    from grounded.generate.baselines import _merge_candidates
    from grounded.retrieve.rerank import rerank_chunks

    store, embedder, graph_retriever, reranker = _load_generation_stack()
    query = f"{title.strip()}\n\n{outline.strip()}"
    vector_hits = store.search_text(query, embedder, n_vector)
    per_comm = max(1, n_graph // 5)
    graph_hits = graph_retriever.search(
        query,
        k_communities=5,
        k_chunks_per_community=per_comm,
    )
    graph_cap = max(0, n_candidates - min(len(vector_hits), n_candidates))
    pool = _merge_candidates(
        vector_hits,
        graph_hits,
        max_candidates=n_candidates,
        per_list_cap=[n_candidates, graph_cap],
    )
    filtered = [h for h in pool if str(h.get("paper_id", "")) != paper_id]
    if len(filtered) < top_k:
        logger.warning(
            "SFT retrieval for %s only found %d cross-paper passages; returning without self-paper backfill.",
            paper_id,
            len(filtered),
        )
    return rerank_chunks(query, filtered, reranker, top_k=top_k)


def format_training_retrieval_block(
    title: str,
    outline: str,
    paper_id: str,
    *,
    top_k: int = 8,
) -> str:
    hits = retrieve_training_passages(title, outline, paper_id, top_k=top_k)
    return format_retrieved_chunks(hits)
