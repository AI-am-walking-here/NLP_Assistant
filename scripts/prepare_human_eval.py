#!/usr/bin/env python3
"""M-8.1 — export blinded human-eval template from eval prompts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import click

from grounded.config import resolve_path
from grounded.eval.runner import load_prompts


@click.command()
@click.option("--n", type=int, default=20, show_default=True)
@click.option("--seed", type=int, default=42)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSONL (default: data/eval_set/human_eval_template.jsonl).",
)
def main(n: int, seed: int, out: Path | None) -> int:
    prompts_path = resolve_path("data/eval_set/prompts.jsonl")
    out_path = out or resolve_path("data/eval_set/human_eval_template.jsonl")
    rows = load_prompts(prompts_path)
    rng = random.Random(seed)
    sample = rows if n >= len(rows) else rng.sample(rows, n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(sample, start=1):
            fh.write(
                json.dumps(
                    {
                        "eval_id": f"HE-{i:03d}",
                        "arxiv_id": row["arxiv_id"],
                        "title": row["title"],
                        "outline": row["outline"],
                        "system_label": "",
                        "generated_abstract": "",
                        "ratings": {
                            "faithfulness": None,
                            "style": None,
                            "overall": None,
                        },
                        "notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    click.echo(json.dumps({"written": len(sample), "path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
