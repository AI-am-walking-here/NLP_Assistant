#!/usr/bin/env python3
"""M-8.2 — sample abstracts for manual FActScore audit vs automatic scores."""

from __future__ import annotations

import json
import random
from pathlib import Path

import click

from grounded.tracking.runs_util import latest_seg4_eval_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_NAIVE = PROJECT_ROOT / "data" / "eval_set" / "archived_per_prompt" / "naive_rag.jsonl"


@click.command()
@click.option("--n", type=int, default=50, show_default=True)
@click.option("--seed", type=int, default=7)
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Eval run with per_prompt.jsonl (default: latest naive_rag).",
)
def _resolve_per_prompt(run_dir: Path | None) -> tuple[Path, str]:
    if run_dir is not None:
        path = run_dir / "per_prompt.jsonl"
        if not path.is_file():
            raise click.ClickException(f"Missing {path}")
        return path, str(run_dir)

    latest = latest_seg4_eval_dirs(PROJECT_ROOT / "runs")
    naive_dir = latest.get("naive_rag")
    if naive_dir is not None:
        path = naive_dir / "per_prompt.jsonl"
        if path.is_file():
            return path, str(naive_dir)

    if ARCHIVED_NAIVE.is_file():
        return ARCHIVED_NAIVE, str(ARCHIVED_NAIVE)

    raise click.ClickException(
        "No naive_rag per_prompt data. Run eval or scripts/archive_runs.py with --keep-per-prompt."
    )


def main(n: int, seed: int, run_dir: Path | None) -> int:
    per_prompt, source = _resolve_per_prompt(run_dir)

    rows: list[dict] = []
    with per_prompt.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    rng = random.Random(seed)
    sample = rows if n >= len(rows) else rng.sample(rows, n)

    out_path = PROJECT_ROOT / "data" / "eval_set" / "factscore_audit.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(sample, start=1):
            fh.write(
                json.dumps(
                    {
                        "audit_id": f"FS-{i:03d}",
                        "arxiv_id": row["arxiv_id"],
                        "auto_factscore": row.get("factscore"),
                        "generated_abstract": row.get("generated_abstract"),
                        "gold_abstract": row.get("gold_abstract"),
                        "manual_factscore": None,
                        "notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    click.echo(
        json.dumps(
            {
                "written": len(sample),
                "source_run": source,
                "path": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
