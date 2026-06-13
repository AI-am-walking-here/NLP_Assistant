#!/usr/bin/env python3
"""M-4.2 — start FActScore verifier HTTP server."""

from __future__ import annotations

import json
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config, resolve_path
from grounded.eval.verifier_server import (
    _apply_worker_build_env,
    create_app,
    run_acceptance_smoke,
)


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option(
    "--backend",
    type=click.Choice(["mock", "awq", "vllm"]),
    default=None,
    help="mock = lexical dev; awq/vllm = 70B AWQ (default from configs/eval.yaml verifier_default_backend).",
)
@click.option(
    "--cache",
    "cache_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Verifier cache JSONL (default: runs/verifier_cache.jsonl).",
)
@click.option(
    "--smoke",
    is_flag=True,
    help="In-process acceptance smoke and exit.",
)
@click.option("--n-claims", type=int, default=10, show_default=True)
@click.option(
    "--max-seconds",
    type=float,
    default=None,
    help="Pass threshold in seconds (default: 30 mock, 600 vllm).",
)
@click.option(
    "--no-preload",
    is_flag=True,
    help="vLLM smoke: skip explicit preload (still loads on first claim).",
)
def main(
    host: str,
    port: int,
    backend: str,
    cache_path: Path | None,
    smoke: bool,
    n_claims: int,
    max_seconds: float | None,
    no_preload: bool,
) -> None:
    if backend is None:
        backend = load_config("eval").verifier_default_backend
    if backend in ("vllm", "awq", "awq_hf"):
        _apply_worker_build_env()
    if smoke:
        if max_seconds is not None:
            limit = max_seconds
        elif backend == "mock":
            limit = 30.0
        elif backend == "vllm":
            limit = 600.0
        else:
            limit = 900.0  # awq: includes first load
        report = run_acceptance_smoke(
            backend=backend,
            n_claims=n_claims,
            max_seconds=limit,
            warmup=backend in ("vllm", "awq", "awq_hf") and not no_preload,
        )
        click.echo(json.dumps(report, indent=2))
        if not report["pass"]:
            raise SystemExit(1)
        return

    if cache_path is None:
        cache_path = resolve_path("runs/verifier_cache.jsonl")
    ctx = init_run("eval", "verifier_server", tags=[f"backend={backend}"], use_wandb=False)
    click.echo(f"Run dir: {ctx.run_dir}")
    if backend in ("vllm", "awq", "awq_hf"):
        click.echo("Preloading 70B AWQ verifier (first start may take 1–2 minutes)…")
    app = create_app(
        backend=backend,
        cache_path=cache_path,
        preload=backend in ("vllm", "awq", "awq_hf"),
    )
    import uvicorn

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        finish_run(ctx)


if __name__ == "__main__":
    main()
