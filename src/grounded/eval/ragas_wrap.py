"""M-4.5 — RAGAS-style faithfulness + context relevance (local BGE + 70B verifier)."""

from __future__ import annotations

import logging
import re
from typing import Any

from grounded.eval.factscore import (
    ClaimVerifier,
    compute_factscore,
    faithfulness_from_details,
)

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]{4,}")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def compute_lexical_ragas(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """Offline fallback when embedder/verifier unavailable."""
    q_tokens = _token_set(question)
    a_tokens = _token_set(answer)
    ctx_blob = " ".join(contexts)
    c_tokens = _token_set(ctx_blob)
    if not a_tokens or not c_tokens:
        faithfulness = 0.0
    else:
        faithfulness = len(a_tokens & c_tokens) / len(a_tokens)
    if not c_tokens or not q_tokens:
        context_relevance = 0.0
    else:
        context_relevance = len(c_tokens & q_tokens) / len(c_tokens)
    return {
        "faithfulness": round(faithfulness, 4),
        "context_relevance": round(context_relevance, 4),
        "ragas_backend": "lexical",
    }


def compute_ragas_grounded(
    question: str,
    answer: str,
    contexts: list[str],
    *,
    embedder: Any,
    verifier: ClaimVerifier,
    factscore_details: list[dict[str, Any]] | None = None,
    max_claims: int = 8,
) -> dict[str, float | str]:
    """
    Diagnostic metrics aligned with v3.1:
    - faithfulness: FActScore-style claim support vs retrieved contexts (70B verifier)
    - context_relevance: mean cosine(query, chunk) with BGE embeddings
    """
    if not contexts:
        return {
            "faithfulness": 0.0,
            "context_relevance": 0.0,
            "ragas_backend": "grounded_empty_context",
        }

    if factscore_details is not None:
        faithfulness = faithfulness_from_details(
            factscore_details,
            max_claims=max_claims,
        )
    else:
        fs = compute_factscore(answer, contexts, verifier, max_claims=max_claims)
        faithfulness = float(fs["factscore"])

    import numpy as np

    q_vec = np.asarray(embedder.encode([question])[0], dtype=np.float32)
    ctx_vecs = embedder.encode(contexts)
    sims = []
    for cv in ctx_vecs:
        c = np.asarray(cv, dtype=np.float32)
        denom = float(np.linalg.norm(q_vec) * np.linalg.norm(c)) or 1.0
        sims.append(float(np.dot(q_vec, c) / denom))
    context_relevance = float(sum(sims) / len(sims)) if sims else 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "context_relevance": round(context_relevance, 4),
        "ragas_backend": "grounded_bge+verifier",
    }


def compute_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    *,
    prefer_lexical: bool = False,
    embedder: Any | None = None,
    verifier: ClaimVerifier | None = None,
    factscore_details: list[dict[str, Any]] | None = None,
    max_claims: int = 8,
) -> dict[str, float | str | None]:
    if prefer_lexical or not contexts:
        out = compute_lexical_ragas(question, answer, contexts)
        return {
            "faithfulness": out["faithfulness"],
            "context_relevance": out["context_relevance"],
            "ragas_backend": out.get("ragas_backend"),
        }

    if embedder is not None and verifier is not None:
        return compute_ragas_grounded(
            question,
            answer,
            contexts,
            embedder=embedder,
            verifier=verifier,
            factscore_details=factscore_details,
            max_claims=max_claims,
        )

    # Optional upstream ragas package (may be broken on some installs)
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_relevance, faithfulness

        ds = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )
        result = evaluate(ds, metrics=[faithfulness, context_relevance])
        row = result.to_pandas().iloc[0]
        return {
            "faithfulness": float(row.get("faithfulness", 0)),
            "context_relevance": float(row.get("context_relevance", 0)),
            "ragas_backend": "ragas",
        }
    except Exception as exc:
        logger.info("ragas package path unavailable (%s); using lexical fallback", exc)
        out = compute_lexical_ragas(question, answer, contexts)
        return {
            "faithfulness": out["faithfulness"],
            "context_relevance": out["context_relevance"],
            "ragas_backend": "lexical",
        }
