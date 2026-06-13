"""Reranker smoke tests."""

from __future__ import annotations

from grounded.retrieve.rerank import MockReranker, rerank_chunks


def test_rerank_prefers_relevant_chunk() -> None:
    query = "transformer neural machine translation"
    chunks = [
        {"chunk_id": "a", "text": "We study cooking recipes and baking."},
        {"chunk_id": "b", "text": "Transformers improve neural machine translation quality."},
    ]
    ranked = rerank_chunks(query, chunks, MockReranker(), top_k=1)
    assert ranked[0]["chunk_id"] == "b"
