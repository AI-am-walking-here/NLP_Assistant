from __future__ import annotations

from grounded.train.sft_data import (
    _use_retrieval_for_id,
    build_sft_examples,
    evidence_supported_target,
    format_chat_example,
)


def test_format_chat_example() -> None:
    ex = format_chat_example("sys", "user", "assistant text")
    assert len(ex["messages"]) == 3
    assert ex["messages"][-1]["role"] == "assistant"


def test_build_sft_examples_minimal(tmp_path) -> None:
    aid = "9999.00002"
    paper = {
        "arxiv_id": aid,
        "title": "Test Paper",
        "abstract": "We propose X. It works well. Results are strong.",
    }
    (tmp_path / f"{aid}.json").write_text(
        __import__("json").dumps(paper),
        encoding="utf-8",
    )
    rows = build_sft_examples(
        [aid], tmp_path, prompt_mode="no_retrieval", outline_source="abstract"
    )
    assert len(rows) == 1
    assert rows[0]["target_abstract"] == paper["abstract"]
    assert rows[0]["prompt_style"] == "no_retrieval"
    assert "[CITE]" not in rows[0]["target_abstract"]


def test_use_retrieval_for_id_mixed_deterministic() -> None:
    assert _use_retrieval_for_id("1234.56789", "with_retrieval", 0.0) is True
    assert _use_retrieval_for_id("1234.56789", "no_retrieval", 1.0) is False
    a = _use_retrieval_for_id("1706.03762", "mixed", 0.3)
    b = _use_retrieval_for_id("1706.03762", "mixed", 0.3)
    assert a == b


def test_evidence_supported_target_filters_unsupported_sentences() -> None:
    abstract = (
        "The model uses transformer attention for spelling correction. "
        "It achieves 99.9 percent accuracy on a secret benchmark."
    )
    retrieved = "A transformer attention model can be used for spelling correction."

    target = evidence_supported_target(abstract, retrieved, min_overlap=0.18)

    assert "transformer attention" in target
    assert "99.9 percent" not in target
