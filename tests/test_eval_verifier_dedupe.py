from __future__ import annotations

from typing import Any

from grounded.eval.factscore import MockClaimVerifier, compute_factscore
from grounded.eval.ragas_wrap import compute_ragas


class CountingVerifier(MockClaimVerifier):
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, claim: str, passages: list[str]) -> dict[str, Any]:
        self.calls += 1
        return super().verify(claim, passages)

    def verify_batch(self, items: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
        return [self.verify(claim, passages) for claim, passages in items]


class TinyEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def test_ragas_uses_factscore_details_without_extra_verifier_calls() -> None:
    answer = (
        "Transformer models improve natural language processing benchmarks. "
        "Attention mechanisms support scientific summarization tasks."
    )
    contexts = [
        "Transformer attention models improve natural language processing benchmarks.",
        "Scientific summarization tasks use attention mechanisms.",
    ]
    verifier = CountingVerifier()
    fs = compute_factscore(answer, contexts, verifier, max_claims=12)
    calls_after_factscore = verifier.calls

    scores = compute_ragas(
        "Title\nOutline",
        answer,
        contexts,
        embedder=TinyEmbedder(),
        verifier=verifier,
        factscore_details=fs["details"],
        max_claims=8,
    )

    assert verifier.calls == calls_after_factscore
    assert scores["ragas_backend"] == "grounded_bge+verifier"
    assert scores["faithfulness"] is not None
