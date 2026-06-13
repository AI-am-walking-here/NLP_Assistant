#!/usr/bin/env python3
"""Build FActScore-ranked DPO pairs from data/sft/train.jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config, log_metric, resolve_path
from grounded.eval.verifier_client import load_claim_verifier
from grounded.train.faithfulness_dpo import build_preferences_from_jsonl, write_dpo_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--limit", type=int, default=None, help="Process first N train rows only.")
@click.option(
    "--mock-verifier/--no-mock-verifier",
    default=False,
    help="Lexical verifier (dev only). Default: HTTP 70B.",
)
@click.option("--skip-verifier-check", is_flag=True)
@click.option("--verifier-url", default=None)
@click.option(
    "--min-margin",
    type=float,
    default=None,
    help="Min FActScore gap chosen−rejected (default: configs/sft.yaml dpo).",
)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option(
    "--full-candidates",
    is_flag=True,
    help="Score 4 candidates per row (default: gold vs unfaithful only).",
)
def main(
    limit: int | None,
    mock_verifier: bool,
    skip_verifier_check: bool,
    verifier_url: str | None,
    min_margin: float | None,
    out: click.Path | None,
    full_candidates: bool,
) -> int:
    sft_cfg = load_config("sft")
    dpo_cfg = sft_cfg.dpo
    eval_cfg = load_config("eval")
    train_path = resolve_path(sft_cfg.paths.train_jsonl)
    out_path = resolve_path(out or dpo_cfg.pairs_jsonl)
    cache = resolve_path(dpo_cfg.verifier_cache_path)
    margin = min_margin if min_margin is not None else dpo_cfg.min_factscore_margin

    if not train_path.is_file():
        raise click.ClickException(f"Missing {train_path}. Run scripts/build_sft_data.py first.")

    if not mock_verifier and not skip_verifier_check:
        from grounded.eval.verifier_client import check_verifier_server

        check_verifier_server(verifier_url or eval_cfg.verifier_server_url)

    verifier = load_claim_verifier(
        eval_cfg,
        mock=mock_verifier,
        server_url=verifier_url,
        cache_path=cache,
        skip_health_check=skip_verifier_check,
    )

    ctx = init_run("seg5", "build_sft_dpo_data", tags=["m-5.5", "dpo"])
    pairs, stats = build_preferences_from_jsonl(
        train_path,
        verifier,
        limit=limit,
        retrieval_top_k=dpo_cfg.retrieval_top_k,
        min_margin=margin,
        prompt_style=dpo_cfg.prompt_style,  # type: ignore[arg-type]
        quick_candidates=not full_candidates,
    )
    write_dpo_jsonl(pairs, out_path)
    stats["output"] = str(out_path)
    stats["mock_verifier"] = mock_verifier
    stats["min_margin"] = margin

    log_metric(ctx, "dpo_pairs", float(stats["pairs_built"]))
    (ctx.run_dir / "dpo_build_summary.json").write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    finish_run(ctx)
    logger.info("Wrote %d DPO pairs to %s", stats["pairs_built"], out_path)
    click.echo(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
