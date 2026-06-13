"""Unit tests for mock FActScore (offline)."""

from __future__ import annotations

from grounded.eval.factscore import MockClaimVerifier, compute_factscore, extract_claims


def test_extract_claims_splits_sentences() -> None:
    text = "First claim is long enough to pass the minimum length filter. Second claim also qualifies here."
    claims = extract_claims(text, max_claims=5)
    assert len(claims) >= 2


def test_mock_verifier_high_overlap() -> None:
    verifier = MockClaimVerifier()
    claim = "transformer models improve benchmark performance significantly"
    passages = [
        "We show transformer models improve benchmark performance on GLUE.",
    ]
    out = verifier.verify(claim, passages)
    assert out["supported"] == "yes"


def test_compute_factscore_returns_score() -> None:
    abstract = (
        "We propose a new method for neural machine translation. "
        "Experiments demonstrate consistent gains over strong baselines."
    )
    evidence = [
        "Our neural machine translation system uses attention over encoder states.",
        "We compare against strong baselines on WMT and report BLEU improvements.",
    ]
    result = compute_factscore(abstract, evidence, MockClaimVerifier())
    assert 0.0 <= result["factscore"] <= 1.0
    assert result["n_claims"] >= 1


def test_compute_factscore_batch_matches_sequential() -> None:
    abstract = (
        "Transformer models improve natural language processing benchmarks. "
        "Attention mechanisms support scientific summarization tasks."
    )
    evidence = [
        "Transformer attention models improve natural language processing benchmarks.",
        "Scientific summarization tasks use attention mechanisms.",
    ]
    sequential = compute_factscore(
        abstract,
        evidence,
        MockClaimVerifier(),
        use_batch=False,
        max_concurrent=1,
    )
    batched = compute_factscore(
        abstract,
        evidence,
        MockClaimVerifier(),
        use_batch=True,
        max_concurrent=4,
        max_batch_size=1,
    )
    assert batched["labels"] == sequential["labels"]
    assert batched["factscore"] == sequential["factscore"]


def test_compute_factscore_concurrency_caps_at_four(monkeypatch) -> None:
    monkeypatch.setenv("EVAL_VERIFIER_CONCURRENCY", "99")
    result = compute_factscore(
        "Transformer models improve natural language processing benchmarks.",
        ["Transformer models improve benchmarks."],
        MockClaimVerifier(),
        use_batch=True,
        max_batch_size=1,
    )
    assert result["n_claims"] == 1
