"""M-4.6 — paired bootstrap for metric differences."""

from __future__ import annotations

import random
from typing import Callable


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 1337,
    stat_fn: Callable[[list[float]], float] | None = None,
) -> dict[str, float]:
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("scores_a and scores_b must be same non-zero length")
    stat_fn = stat_fn or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    diffs: list[float] = []
    n = len(scores_a)
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [scores_a[i] for i in idx]
        b = [scores_b[i] for i in idx]
        diffs.append(stat_fn(a) - stat_fn(b))
    diffs.sort()
    alpha = (1.0 - ci) / 2.0
    lo = diffs[int(alpha * n_resamples)]
    hi = diffs[int((1 - alpha) * n_resamples) - 1]
    mean_diff = stat_fn(scores_a) - stat_fn(scores_b)
    return {"mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "n": n}
