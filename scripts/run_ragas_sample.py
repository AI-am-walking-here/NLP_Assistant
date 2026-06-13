#!/usr/bin/env python3
"""M-4.5 — compute lexical (and optional LLM) RAGAS on a few eval prompts."""

from __future__ import annotations

import json
from pathlib import Path

import click

from grounded.config import load_config, resolve_path
from grounded.eval.ragas_wrap import compute_lexical_ragas, compute_ragas
from grounded.eval.verifier_client import load_claim_verifier
from grounded.eval.runner import load_retrieval_stack


@click.command()
@click.option("--n", default=5, show_default=True, help="Number of prompts.")
@click.option("--lexical-only", is_flag=True, help="Skip ragas LLM path.")
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(n: int, lexical_only: bool, out: Path | None) -> None:
    prompts_path = resolve_path("data/eval_set/prompts.jsonl")
    rows: list[dict] = []
    with prompts_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            if line.strip():
                rows.append(json.loads(line))

    embedder = verifier = None
    if not lexical_only:
        eval_cfg = load_config("eval")
        retr_cfg = load_config("retrieval")
        try:
            _, embedder, _ = load_retrieval_stack(retr_cfg, require_real_index=True)
            verifier = load_claim_verifier(eval_cfg, mock=False, skip_health_check=False)
        except Exception as exc:
            click.echo(f"Grounded RAGAS unavailable ({exc}); use --lexical-only", err=True)
            lexical_only = True

    results: list[dict] = []
    for row in rows:
        contexts = [row.get("gold_abstract", "")[:800]]
        question = f"{row['title']}\n{row['outline']}"
        answer = row.get("gold_abstract", "")[:500]
        if lexical_only:
            scores = compute_lexical_ragas(question, answer, contexts)
        else:
            scores = compute_ragas(
                question,
                answer,
                contexts,
                embedder=embedder,
                verifier=verifier,
            )
        results.append({"arxiv_id": row["arxiv_id"], **scores})

    payload = {"n": len(results), "rows": results}
    text = json.dumps(payload, indent=2)
    click.echo(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
