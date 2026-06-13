from __future__ import annotations

from grounded.eval.runner import run_eval
from grounded.eval.factscore import MockClaimVerifier
from grounded.generate.baselines import MockGenerator


def test_run_eval_skips_completed_prompt_ids():
    prompts = [
        {"arxiv_id": "a", "title": "A", "outline": "One", "gold_abstract": "Gold A"},
        {"arxiv_id": "b", "title": "B", "outline": "Two", "gold_abstract": "Gold B"},
    ]
    per_prompt, aggregate = run_eval(
        "zero_shot",
        prompts,
        store=None,
        embedder=None,
        generator=MockGenerator(),
        top_k=1,
        verifier=MockClaimVerifier(),
        skip_arxiv_ids={"a"},
    )

    assert [row["arxiv_id"] for row in per_prompt] == ["b"]
    assert aggregate["n_prompts"] == 1
