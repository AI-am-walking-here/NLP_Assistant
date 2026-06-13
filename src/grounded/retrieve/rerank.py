"""M-6.8 — rerank retrieved candidates (RankRAG LoRA or dev lexical mock)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, candidates: list[str]) -> list[float]: ...


class MockReranker:
    """Lexical overlap — dev/CI only when --mock-gen or no RankRAG adapter."""

    def score(self, query: str, candidates: list[str]) -> list[float]:
        q_tokens = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
        if not q_tokens:
            return [0.0] * len(candidates)
        scores: list[float] = []
        for text in candidates:
            t_tokens = set(re.findall(r"[a-z0-9]{4,}", text.lower()))
            if not t_tokens:
                scores.append(0.0)
                continue
            scores.append(len(q_tokens & t_tokens) / len(q_tokens))
        return scores


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker: Reranker,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    if hasattr(reranker, "score_rows"):
        scores = reranker.score_rows(query, chunks)  # type: ignore[attr-defined]
    else:
        texts = [c.get("text", "") for c in chunks]
        scores = reranker.score(query, texts)
    ranked = sorted(
        zip(chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for chunk, score in ranked[:top_k]:
        row = dict(chunk)
        row["rerank_score"] = float(score)
        out.append(row)
    return out


def load_reranker(
    adapter_path: Path | None = None,
    *,
    mock: bool = False,
    base_model: str | None = None,
    cuda_device: int = 0,
) -> Reranker:
    """Load RankRAG LoRA reranker; mock only when explicitly requested."""
    if mock:
        return MockReranker()
    if adapter_path is None or not adapter_path.is_dir():
        raise FileNotFoundError(
            "RankRAG adapter required. Train with scripts/rankrag_train.py "
            "or pass --mock-gen for dev runs."
        )
    from grounded.config import load_config
    from grounded.retrieve.rankrag_reranker import load_lora_rankrag_reranker

    rr_cfg = load_config("rankrag")
    hub = base_model or rr_cfg.base_model
    return load_lora_rankrag_reranker(adapter_path, hub, cuda_device=cuda_device)
