from __future__ import annotations

import json
from pathlib import Path

from grounded.eval.factscore import MockClaimVerifier
from grounded.train.faithfulness_dpo import (
    build_preference_pair,
    generic_weak_abstract,
    prompt_messages_for_row,
    unfaithful_variant,
    write_dpo_jsonl,
)


def test_unfaithful_scores_lower_than_gold() -> None:
    gold = (
        "We introduce a transformer model for language understanding. "
        "Experiments on GLUE show improvements over baselines. "
        "Analysis confirms the approach captures syntactic structure."
    )
    passages = [
        "Transformers use self-attention for language understanding tasks.",
        "GLUE benchmark evaluates natural language understanding systems.",
    ]
    verifier = MockClaimVerifier()
    from grounded.train.faithfulness_dpo import score_abstract

    assert score_abstract(gold, passages, verifier) > score_abstract(
        unfaithful_variant(gold), passages, verifier
    )


def test_build_preference_pair_mock(tmp_path: Path) -> None:
    row = {
        "arxiv_id": "1234.56789",
        "title": "Attention Is All You Need",
        "outline": "- We propose transformers\n- We evaluate on MT",
        "target_abstract": (
            "We propose the transformer architecture based on self-attention. "
            "Machine translation experiments show quality and training efficiency gains."
        ),
    }
    # Monkeypatch retrieval to avoid FAISS in unit test
    import grounded.train.faithfulness_dpo as fd

    def _fake_hits(*_a: object, **_k: object) -> list[dict]:
        return [{"text": "Self-attention replaces recurrence in sequence models.", "paper_id": "p2"}]

    fd.retrieve_training_passages = _fake_hits  # type: ignore[method-assign]
    pair = build_preference_pair(
        row, MockClaimVerifier(), min_margin=0.01, seed=1, prompt_style="with_retrieval"
    )
    assert pair is not None
    assert pair.chosen_factscore >= pair.rejected_factscore
    assert len(pair.prompt_messages) == 2
    msgs = prompt_messages_for_row(
        row["title"],
        row["outline"],
        prompt_style="with_retrieval",
        retrieved_block="[1] supporting evidence",
    )
    assert msgs[0]["role"] == "system"
    assert "Title:" in msgs[1]["content"]
    assert "supporting evidence" in msgs[1]["content"]


def test_write_dpo_jsonl(tmp_path: Path) -> None:
    from grounded.train.faithfulness_dpo import PreferencePair

    pair = PreferencePair(
        arxiv_id="1",
        prompt_messages=[{"role": "user", "content": "hi"}],
        chosen="good",
        rejected="bad",
        chosen_factscore=0.9,
        rejected_factscore=0.2,
        chosen_source="gold",
        rejected_source="generic",
        n_passages=1,
    )
    out = tmp_path / "pairs.jsonl"
    write_dpo_jsonl([pair], out)
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["chosen"][0]["content"] == "good"
    assert row["prompt"][0]["role"] == "user"
