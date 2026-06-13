#!/usr/bin/env python3
"""Fail-closed smoke gate for accelerated FActScore evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import click

from grounded.config import load_config
from grounded.eval.factscore import MockClaimVerifier, compute_factscore
from grounded.eval.ragas_wrap import compute_ragas
from grounded.eval.verifier_client import check_verifier_server
from grounded.utils.list_gpus import discover_worker_gpus, verifier_reserved_gpus


class CountingVerifier(MockClaimVerifier):
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, claim: str, passages: list[str]) -> dict[str, Any]:
        self.calls += 1
        return super().verify(claim, passages)

    def verify_batch(self, items: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
        return [self.verify(claim, passages) for claim, passages in items]


class TinyEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def _sample_text() -> tuple[str, list[str]]:
    abstract = (
        "Transformer models improve natural language processing benchmark performance. "
        "The method uses attention over encoder states for scientific summarization. "
        "Experiments report consistent gains over strong baselines."
    )
    passages = [
        "Transformer models with attention improve NLP benchmark performance.",
        "The scientific summarization method uses attention over encoder states.",
        "Experiments report consistent gains over strong baselines.",
    ]
    return abstract, passages


def _assert_mock_parity(eval_cfg: Any) -> None:
    abstract, passages = _sample_text()
    baseline = compute_factscore(
        abstract,
        passages,
        MockClaimVerifier(),
        max_claims=eval_cfg.factscore_max_claims,
        max_concurrent=1,
        use_batch=False,
        max_batch_size=eval_cfg.verifier_vllm.max_batch_size,
    )
    fast = compute_factscore(
        abstract,
        passages,
        MockClaimVerifier(),
        max_claims=eval_cfg.factscore_max_claims,
        max_concurrent=4,
        use_batch=True,
        max_batch_size=eval_cfg.verifier_vllm.max_batch_size,
    )
    if baseline["factscore"] != fast["factscore"] or baseline["labels"] != fast["labels"]:
        raise click.ClickException("mock FActScore parity failed")


def _assert_dedupe(eval_cfg: Any) -> None:
    abstract, passages = _sample_text()
    verifier = CountingVerifier()
    fs = compute_factscore(
        abstract,
        passages,
        verifier,
        max_claims=eval_cfg.factscore_max_claims,
        max_concurrent=1,
        use_batch=False,
        max_batch_size=eval_cfg.verifier_vllm.max_batch_size,
    )
    calls_after_factscore = verifier.calls
    scores = compute_ragas(
        "Title\nOutline",
        abstract,
        passages,
        embedder=TinyEmbedder(),
        verifier=verifier,
        factscore_details=fs["details"],
        max_claims=eval_cfg.ragas_max_claims,
    )
    if verifier.calls != calls_after_factscore:
        raise click.ClickException("RAGAS dedupe smoke made extra verifier calls")
    if scores["faithfulness"] is None:
        raise click.ClickException("RAGAS dedupe smoke returned no faithfulness")


def _assert_eval_gpu_isolation() -> None:
    overlap = set(discover_worker_gpus(
        devices_env="EVAL_CUDA_DEVICES",
        exclude=verifier_reserved_gpus(),
    )) & verifier_reserved_gpus()
    if overlap and os.environ.get("ALLOW_VERIFIER_GPU_OVERLAP", "0") != "1":
        raise click.ClickException(
            f"eval GPUs overlap verifier GPUs: {','.join(sorted(overlap))}"
        )


def _eval_subprocess_env() -> dict[str, str]:
    """Pin live smoke to one eval GPU (same contract as parallel workers)."""
    from grounded.utils.list_gpus import gpus_with_capacity

    env = os.environ.copy()
    gpus = discover_worker_gpus(
        devices_env="EVAL_CUDA_DEVICES",
        exclude=verifier_reserved_gpus(),
    )
    ready = gpus_with_capacity(gpus)
    if ready:
        env["CUDA_VISIBLE_DEVICES"] = ready[0]
    return env


def _run_live_eval(limit: int, verifier_url: str) -> None:
    with tempfile.TemporaryDirectory(prefix="eval_verifier_smoke_") as tmp:
        run_dir = Path(tmp) / "run"
        cmd = [
            sys.executable,
            "scripts/run_eval.py",
            "--system",
            "rankrag_only",
            "--limit",
            str(limit),
            "--run-dir",
            str(run_dir),
            "--verifier-url",
            verifier_url.replace("/health", ""),
        ]
        if os.environ.get("EVAL_GENERATOR_4BIT", "1") == "1":
            cmd.append("--generator-4bit")
        env = _eval_subprocess_env()
        result = subprocess.run(cmd, text=True, capture_output=True, env=env)
        if result.returncode != 0:
            raise click.ClickException(
                "live eval smoke failed:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        per_prompt = run_dir / "per_prompt.jsonl"
        if not per_prompt.is_file():
            raise click.ClickException("live eval smoke wrote no per_prompt.jsonl")
        rows = [line for line in per_prompt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != limit:
            raise click.ClickException(
                f"live eval smoke expected {limit} rows, got {len(rows)}"
            )


@click.command()
@click.option("--verifier-url", default=None)
@click.option("--limit", type=int, default=None)
def main(verifier_url: str | None, limit: int | None) -> int:
    eval_cfg = load_config("eval")
    if eval_cfg.factscore_max_claims != 12 or eval_cfg.ragas_max_claims != 8:
        raise click.ClickException("headline eval claim limits must remain factscore=12 ragas=8")

    host = os.environ.get("VERIFIER_HOST", "127.0.0.1")
    if host != "127.0.0.1" and os.environ.get("EVAL_VERIFIER_ALLOW_REMOTE", "0") != "1":
        raise click.ClickException("verifier smoke requires localhost verifier binding")

    health_url = verifier_url or os.environ.get("VERIFIER_URL") or f"{eval_cfg.verifier_server_url.rstrip('/')}/health"
    health = check_verifier_server(
        health_url.replace("/health", ""),
        expect_backend=eval_cfg.verifier_default_backend,
    )
    click.echo(f"verifier health ok backend={health.get('backend')}")

    _assert_mock_parity(eval_cfg)
    _assert_dedupe(eval_cfg)
    _assert_eval_gpu_isolation()

    if os.environ.get("EVAL_VERIFIER_SKIP_LIVE_SMOKE", "0") != "1":
        _run_live_eval(
            limit or int(os.environ.get("EVAL_VERIFIER_SMOKE_LIMIT", "3")),
            health_url,
        )
    click.echo("eval verifier smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
