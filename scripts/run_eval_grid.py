#!/usr/bin/env python3
"""M-7.3 — run eval grid over wired systems (real stack by default)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

from grounded.config import load_config
from grounded.eval.verifier_client import check_verifier_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_SYSTEMS = (
    "zero_shot",
    "naive_rag",
    "graph_only",
    "rankrag_only",
    "naive_rag_with_sft",
    "full",
    "full_minus_graph",
    "full_minus_rerank",
    "full_minus_sft",
)


@click.command()
@click.option(
    "--mock-gen/--no-mock-gen",
    default=False,
    help="Dev only: mock 8B generation.",
)
@click.option(
    "--mock-verifier/--no-mock-verifier",
    default=False,
    help="Dev only: lexical FActScore.",
)
@click.option("--limit", type=int, default=None)
@click.option(
    "--systems",
    multiple=True,
    default=GRID_SYSTEMS,
    help="Subset of systems to run.",
)
@click.option(
    "--skip-verifier-check",
    is_flag=True,
    help="Skip verifier /health before grid.",
)
def main(
    mock_gen: bool,
    mock_verifier: bool,
    limit: int | None,
    systems: tuple[str, ...],
    skip_verifier_check: bool,
) -> int:
    eval_cfg = load_config("eval")
    if not mock_verifier and not skip_verifier_check:
        check_verifier_server(eval_cfg.verifier_server_url)

    runs: list[dict[str, object]] = []
    total_systems = len(systems)
    for idx, system in enumerate(systems, start=1):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_eval.py"),
            "--system",
            system,
        ]
        if mock_gen:
            cmd.append("--mock-gen")
        if mock_verifier:
            cmd.append("--mock-verifier")
        if skip_verifier_check:
            cmd.append("--skip-verifier-check")
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        click.echo(f"[{idx}/{total_systems}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        pattern = f"seg4_eval_{system}_*"
        candidates = sorted(
            (PROJECT_ROOT / "runs").glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            results_path = candidates[-1] / "results.json"
            if results_path.is_file():
                runs.append(json.loads(results_path.read_text(encoding="utf-8")))

    summary = {
        "systems": runs,
        "mock_generation": mock_gen,
        "mock_verifier": mock_verifier,
        "verifier_url": None if mock_verifier else eval_cfg.verifier_server_url,
    }
    out = PROJECT_ROOT / "data" / "eval_set" / "grid_runs.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    click.echo(f"Wrote {out}")
    click.echo(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
