from __future__ import annotations


def retrieval_stats(
    baseline_ids: list[list[str]],
    candidate_ids: list[list[str]],
    gold_ids: list[str],
) -> dict[str, float]:
    assert len(baseline_ids) == len(candidate_ids) == len(gold_ids)
    overlap = []
    contains_gold = 0
    top1_gold = 0
    avg_docs = 0.0
    for base, cand, gold in zip(baseline_ids, candidate_ids, gold_ids):
        sb = set(base)
        sc = set(cand)
        if sc and gold in sc:
            contains_gold += 1
        if cand and cand[0] == gold:
            top1_gold += 1
        if sb or sc:
            overlap.append(len(sb & sc) / max(1, len(sb | sc)))
        avg_docs += len(cand)
    n = len(gold_ids)
    return {
        "contains_gold_rate": contains_gold / n,
        "top1_gold_rate": top1_gold / n,
        "avg_jaccard_vs_baseline": (sum(overlap) / len(overlap)) if overlap else 0.0,
        "avg_docs": avg_docs / n,
    }


def test_retrieval_stats_capture_regression_pattern() -> None:
    baseline = [["gold", "a"], ["gold", "b"], ["gold", "c"]]
    candidate = [["x", "gold"], ["y", "z"], ["gold", "q"]]
    gold = ["gold", "gold", "gold"]

    stats = retrieval_stats(baseline, candidate, gold)

    assert stats["contains_gold_rate"] == 2 / 3
    assert stats["top1_gold_rate"] == 1 / 3
    assert 0.0 <= stats["avg_jaccard_vs_baseline"] <= 1.0
    assert stats["avg_docs"] == 2.0
