#!/usr/bin/env python3
"""M-5.4 — fair SFT smoke: matched-prompt vs RAG-prompt systems."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fair SFT-on-training-prompt comparisons + legacy RAG-conditioned SFT.
SMOKE_SYSTEMS = (
    "zero_shot",
    "zero_shot_with_sft",
    "naive_rag_sft_prompt",
    "naive_rag",
    "naive_rag_with_sft",
)


def _latest_run_dir(system: str, limit: int) -> Path | None:
    runs_dir = PROJECT_ROOT / "runs"
    run_re = re.compile(
        rf"^seg4_eval_{re.escape(system)}_\d{{4}}-\d{{2}}-\d{{2}}-\d{{4}}$"
    )
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and run_re.match(p.name)),
        key=lambda p: p.stat().st_mtime,
    )
    for cand in reversed(candidates):
        agg_path = cand / "results.json"
        if not agg_path.is_file():
            continue
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        if int(agg.get("n_prompts", 0)) == limit:
            return cand
    return candidates[-1] if candidates else None


@click.command()
@click.option("--limit", default=10, show_default=True)
@click.option(
    "--mock-gen/--no-mock-gen",
    default=False,
    help="Dev only: skip 8B load.",
)
@click.option(
    "--mock-verifier/--no-mock-verifier",
    default=False,
    help="Dev only: lexical FActScore.",
)
@click.option(
    "--skip-verifier-check",
    is_flag=True,
    help="Do not GET /health before eval.",
)
@click.option(
    "--systems",
    multiple=True,
    default=SMOKE_SYSTEMS,
    help="Eval systems to run (subset).",
)
@click.option(
    "--adapter-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="LoRA adapter for SFT systems (e.g. runs/seg5_dpo_train_*/adapter).",
)
def main(
    limit: int,
    mock_gen: bool,
    mock_verifier: bool,
    skip_verifier_check: bool,
    systems: tuple[str, ...],
    adapter_path: Path | None,
) -> int:
    if not mock_verifier and not skip_verifier_check:
        from grounded.config import load_config
        from grounded.eval.verifier_client import check_verifier_server

        eval_cfg = load_config("eval")
        check_verifier_server(eval_cfg.verifier_server_url)

    for system in systems:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_eval.py"),
            "--system",
            system,
            "--limit",
            str(limit),
        ]
        if mock_gen:
            cmd.append("--mock-gen")
        else:
            cmd.append("--no-mock-gen")
        if mock_verifier:
            cmd.append("--mock-verifier")
        if skip_verifier_check:
            cmd.append("--skip-verifier-check")
        if adapter_path is not None and system in (
            "zero_shot_with_sft",
            "naive_rag_sft_prompt",
            "naive_rag_with_sft",
        ):
            cmd.extend(["--adapter-path", str(adapter_path)])
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    rows: dict[str, float] = {}
    meta: dict[str, object] = {}
    for system in systems:
        run_dir = _latest_run_dir(system, limit)
        if run_dir is None:
            continue
        agg = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        rows[system] = float(agg["factscore_mean"])
        meta[system] = {
            "mock_generation": agg.get("mock_generation"),
            "mock_verifier": agg.get("mock_verifier"),
            "run_dir": run_dir.name,
            "n_prompts": agg.get("n_prompts"),
            "sft_adapter_dir": agg.get("sft_adapter_dir"),
            "sft_run_dir": agg.get("sft_run_dir"),
            "generation_temperature": agg.get("generation_temperature"),
        }

    summary = {
        "limit": limit,
        "mock_generation": mock_gen,
        "mock_verifier": mock_verifier,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "factscore_mean": rows,
        "per_system": meta,
        "notes": {
            "zero_shot_with_sft": "SFT on training prompt (title+outline only)",
            "naive_rag_sft_prompt": "Retrieve for FActScore; generate with SFT prompt",
            "naive_rag_with_sft": "Retrieve in prompt + SFT (train/eval mismatch)",
        },
    }
    out = PROJECT_ROOT / "data/eval_set/sft_smoke_comparison.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
