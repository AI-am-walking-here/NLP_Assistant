#!/usr/bin/env python3
"""Paired bootstrap on per_prompt.jsonl from two eval runs (same arxiv_id order)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from grounded.eval.stats import paired_bootstrap_ci


def load_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                scores[row["arxiv_id"]] = float(row["factscore"])
    return scores


def _per_prompt_path(arg: Path) -> Path:
    if arg.is_file() and arg.suffix == ".jsonl":
        return arg
    path = arg / "per_prompt.jsonl"
    if not path.is_file():
        raise click.ClickException(f"Missing {path} (pass a run dir or a .jsonl file)")
    return path


@click.command()
@click.argument("run_a", type=click.Path(path_type=Path, exists=True))
@click.argument("run_b", type=click.Path(path_type=Path, exists=True))
def main(run_a: Path, run_b: Path) -> None:
    a_path = _per_prompt_path(run_a)
    b_path = _per_prompt_path(run_b)
    sa, sb = load_scores(a_path), load_scores(b_path)
    ids = sorted(set(sa) & set(sb))
    scores_a = [sa[i] for i in ids]
    scores_b = [sb[i] for i in ids]
    ci = paired_bootstrap_ci(scores_a, scores_b, n_resamples=10_000)
    out = {
        "n": len(ids),
        "mean_a": sum(scores_a) / len(scores_a),
        "mean_b": sum(scores_b) / len(scores_b),
        "bootstrap": ci,
        "run_a": str(a_path),
        "run_b": str(b_path),
    }
    click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
