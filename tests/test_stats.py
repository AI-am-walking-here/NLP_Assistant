"""Paired bootstrap CI smoke test."""

from __future__ import annotations

from grounded.eval.stats import paired_bootstrap_ci


def test_bootstrap_ci_contains_mean_diff() -> None:
    a = [0.4, 0.5, 0.6, 0.45, 0.55]
    b = [0.2, 0.25, 0.3, 0.22, 0.28]
    ci = paired_bootstrap_ci(a, b, n_resamples=500, seed=42)
    assert ci["mean_diff"] > 0
    assert ci["ci_low"] <= ci["mean_diff"] <= ci["ci_high"]
