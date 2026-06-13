"""M-4.5 — RAGAS lexical diagnostic."""

from grounded.eval.factscore import compute_factscore, MockClaimVerifier
from grounded.eval.ragas_wrap import (
    compute_lexical_ragas,
    compute_ragas,
    faithfulness_from_details,
)


def test_lexical_ragas_bounded() -> None:
    scores = compute_lexical_ragas(
        "Title\nOutline about neural NLP.",
        "We propose a neural method for NLP.",
        ["Neural networks for natural language processing."],
    )
    assert 0.0 <= scores["faithfulness"] <= 1.0
    assert 0.0 <= scores["context_relevance"] <= 1.0


def test_compute_ragas_falls_back_without_llm() -> None:
    scores = compute_ragas("Q", "Answer with neural NLP method.", ["NLP neural context."])
    assert scores["faithfulness"] is not None
    assert scores.get("ragas_backend") in ("lexical", "ragas")


def test_faithfulness_from_details_matches_eight_claim_path() -> None:
    answer = (
        "Transformer models improve natural language processing benchmarks. "
        "Attention mechanisms support scientific summarization tasks."
    )
    contexts = [
        "Transformer attention models improve natural language processing benchmarks.",
        "Scientific summarization tasks use attention mechanisms.",
    ]
    fs = compute_factscore(answer, contexts, MockClaimVerifier(), max_claims=8)
    assert faithfulness_from_details(fs["details"], max_claims=8) == fs["factscore"]
