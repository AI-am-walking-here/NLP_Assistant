#!/usr/bin/env python3
"""M-4.7 — run evaluation for a system on the eval prompt set."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

from grounded.config import RunContext, finish_run, init_run, load_config, resolve_path
from grounded.eval.runner import (
    GRAPH_SYSTEMS,
    RERANK_SYSTEMS,
    RETRIEVAL_SYSTEMS,
    SUPPORTED_SYSTEMS,
    evaluate_one_prompt,
    load_graph_retriever,
    load_prompts,
    load_retrieval_stack,
    run_eval,
)
from grounded.eval.verifier_client import load_claim_verifier
from grounded.generate.pipeline import SFT_GENERATOR_SYSTEMS
from grounded.generate.baselines import build_generator
from grounded.generate.pipeline import (
    IncompleteSftTrainingError,
    load_text_generator,
    require_latest_adapter,
    require_sft_adapter,
)
from grounded.retrieve.rerank import load_reranker
from grounded.utils.model_paths import model_weights_ready, resolve_model_path
from grounded.utils.incremental_jsonl import append_row, jsonl_ids, load_processed_ids, mark_processed, read_jsonl
from grounded.utils.cuda_devices import configure_eval_worker_cuda
from grounded.utils.phase_resume import phase_input_fingerprint
from grounded.progress import update_run_progress

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _require_generator_weights(base_model: str) -> None:
    path = Path(resolve_model_path(base_model, role="generator_8b"))
    if not model_weights_ready(path):
        raise click.ClickException(
            f"8B generator weights not ready at {path}. "
            "Rsync meta-llama/Llama-3.1-8B-Instruct and run scripts/check_models.py"
        )


def _generator_4bit_enabled(flag: bool) -> bool:
    """CLI override; shell may also set EVAL_GENERATOR_4BIT=1."""
    if flag:
        return True
    return os.environ.get("EVAL_GENERATOR_4BIT", "1") == "1"


def _progress_logger() -> callable:
    seen_scores: list[float] = []

    def _callback(idx: int, total: int, row: dict, result_row: dict) -> None:
        seen_scores.append(float(result_row["factscore"]))
        running_mean = sum(seen_scores) / len(seen_scores)
        logger.info(
            "[%d/%d] %s factscore=%.4f running_mean=%.4f claims=%s",
            idx,
            total,
            row.get("arxiv_id", "?"),
            float(result_row["factscore"]),
            running_mean,
            result_row.get("n_claims"),
        )

    return _callback


@click.command()
@click.option(
    "--system",
    type=click.Choice(sorted(SUPPORTED_SYSTEMS)),
    default="naive_rag",
    help="System to evaluate.",
)
@click.option("--limit", type=int, default=None, help="Evaluate first N prompts only.")
@click.option(
    "--mock-gen/--no-mock-gen",
    default=False,
    help="Use mock text generator (dev only). Default: real 8B.",
)
@click.option(
    "--mock-verifier/--no-mock-verifier",
    default=False,
    help="Use lexical FActScore mock (dev only). Default: HTTP 70B verifier server.",
)
@click.option(
    "--skip-verifier-check",
    is_flag=True,
    help="Skip GET /health before eval (verifier assumed running).",
)
@click.option(
    "--verifier-url",
    default=None,
    help="Override configs/eval.yaml verifier_server_url.",
)
@click.option(
    "--adapter-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="SFT LoRA adapter dir (auto-detect latest seg5 run if omitted).",
)
@click.option(
    "--rerank-adapter",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="RankRAG LoRA adapter (required for rerank systems when --no-mock-gen).",
)
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Resume/write eval outputs in this existing run directory.",
)
@click.option(
    "--ids-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Worker mode: evaluate only prompt IDs listed here.",
)
@click.option(
    "--shard-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Worker mode output JSONL.",
)
@click.option(
    "--processed-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Worker mode processed prompt-id ledger.",
)
@click.option(
    "--disable-verifier-cache-write",
    is_flag=True,
    help="Disable client verifier cache writes (safe for parallel workers).",
)
@click.option(
    "--generator-4bit",
    is_flag=True,
    default=False,
    help="Load base 8B generator in 4-bit NF4 (eval VRAM override; default fp16).",
)
def main(
    system: str,
    limit: int | None,
    mock_gen: bool,
    mock_verifier: bool,
    skip_verifier_check: bool,
    verifier_url: str | None,
    adapter_path: Path | None,
    rerank_adapter: Path | None,
    run_dir: Path | None,
    ids_file: Path | None,
    shard_out: Path | None,
    processed_out: Path | None,
    disable_verifier_cache_write: bool,
    generator_4bit: bool,
) -> int:
    if not mock_gen:
        configure_eval_worker_cuda()

    eval_cfg = load_config("eval")
    retr_cfg = load_config("retrieval")
    sft_cfg = load_config("sft")
    prompts_path = resolve_path(eval_cfg.eval_prompts_path)

    if not prompts_path.is_file():
        raise click.ClickException(
            f"Missing {prompts_path}. Run: python3 scripts/build_eval_prompts.py"
        )

    if not mock_gen:
        _require_generator_weights(retr_cfg.generation.base_model)

    use_generator_4bit = _generator_4bit_enabled(generator_4bit)
    if use_generator_4bit and not mock_gen:
        logger.info("Generator 4-bit NF4 override enabled (rankrag_only / full_minus_sft base path)")

    if system in RERANK_SYSTEMS and not mock_gen:
        try:
            rank_path = rerank_adapter or require_latest_adapter("seg6_rankrag_*")
        except (IncompleteSftTrainingError, FileNotFoundError) as exc:
            raise click.ClickException(
                f"System {system!r} requires a complete RankRAG LoRA adapter. {exc}"
            ) from exc

    verifier = load_claim_verifier(
        eval_cfg,
        mock=mock_verifier,
        server_url=verifier_url,
        cache_read=True,
        cache_write=not disable_verifier_cache_write,
        skip_health_check=skip_verifier_check,
    )

    prompts = load_prompts(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]
    if ids_file is not None:
        wanted = {
            line.strip()
            for line in ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        prompts = [row for row in prompts if str(row.get("arxiv_id")) in wanted]

    store = embedder = None
    graph_retriever = None
    meta: dict = {}
    if system in GRAPH_SYSTEMS:
        graph_retriever = load_graph_retriever(retr_cfg)
    if system in RETRIEVAL_SYSTEMS:
        store, embedder, meta = load_retrieval_stack(
            retr_cfg, require_real_index=not mock_gen
        )

    rank_path = rerank_adapter
    if system in RERANK_SYSTEMS and not mock_gen:
        try:
            rank_path = rerank_adapter or require_latest_adapter("seg6_rankrag_*")
        except (IncompleteSftTrainingError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
    rerank_mock = mock_gen or system not in RERANK_SYSTEMS
    reranker = load_reranker(rank_path, mock=rerank_mock) if system in RERANK_SYSTEMS or rank_path else None

    sft_provenance: dict[str, str | None] = {}
    use_sft = system in SFT_GENERATOR_SYSTEMS and system != "full_minus_sft"
    if use_sft:
        adapter = None
        if not mock_gen:
            try:
                resolution = require_sft_adapter(explicit=adapter_path)
            except IncompleteSftTrainingError as exc:
                raise click.ClickException(str(exc)) from exc
            except FileNotFoundError as exc:
                raise click.ClickException(str(exc)) from exc
            adapter = resolution.adapter_dir
            sft_provenance = {
                "sft_adapter_dir": str(adapter),
                "sft_run_dir": str(resolution.run_dir) if resolution.run_dir else None,
                "sft_adapter_source": resolution.source,
            }
        elif adapter_path is not None:
            adapter = adapter_path
            sft_provenance = {
                "sft_adapter_dir": str(adapter),
                "sft_run_dir": None,
                "sft_adapter_source": "explicit",
            }
        sft_temp = getattr(sft_cfg, "inference_temperature", retr_cfg.generation.temperature)
        generator = load_text_generator(
            sft_cfg.base_model,
            adapter,
            max_new_tokens=retr_cfg.generation.max_new_tokens,
            temperature=sft_temp,
            mock=mock_gen or adapter is None,
        )
        sft_provenance["generation_temperature"] = str(sft_temp)
    else:
        generator = build_generator(
            model_name=retr_cfg.generation.base_model,
            max_new_tokens=retr_cfg.generation.max_new_tokens,
            temperature=retr_cfg.generation.temperature,
            mock=mock_gen,
            fail_on_error=not mock_gen,
            load_in_4bit=use_generator_4bit,
        )

    if ids_file is not None:
        if shard_out is None or processed_out is None:
            raise click.ClickException(
                "--ids-file worker mode requires --shard-out and --processed-out"
            )
        completed = load_processed_ids(processed_out)
        progress_cb = _progress_logger()
        total = len(prompts)
        for idx, row in enumerate(prompts, start=1):
            aid = str(row.get("arxiv_id"))
            if aid in completed:
                continue
            result_row = evaluate_one_prompt(
                system,
                row,
                store=store,
                embedder=embedder,
                generator=generator,
                top_k=retr_cfg.top_k,
                verifier=verifier,
                graph_retriever=graph_retriever,
                reranker=reranker,
                factscore_max_claims=eval_cfg.factscore_max_claims,
                ragas_max_claims=eval_cfg.ragas_max_claims,
                verifier_max_batch_size=eval_cfg.verifier_vllm.max_batch_size,
            )
            append_row(shard_out, result_row)
            mark_processed(processed_out, aid)
            progress_cb(idx, total, row, result_row)
        click.echo(
            json.dumps(
                {
                    "system": system,
                    "processed": len(load_processed_ids(processed_out)),
                    "shard_out": str(shard_out),
                },
                indent=2,
            )
        )
        return 0

    ctx = init_run(
        "seg4",
        f"eval_{system}",
        tags=["m-4.7", system],
        config_snapshot={"system": system, "mock_gen": mock_gen, "generator_4bit": use_generator_4bit, **sft_provenance},
    )
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            segment=ctx.segment,
            purpose=ctx.purpose,
            run_dir=run_dir,
            meta_path=run_dir / "meta.json",
            log_path=run_dir / "log.txt",
        )
        if not ctx.meta_path.is_file():
            ctx.meta_path.write_text(json.dumps({"status": "running"}, indent=2) + "\n", encoding="utf-8")
        if not ctx.log_path.is_file():
            ctx.log_path.write_text("", encoding="utf-8")
    out_dir = ctx.run_dir
    per_prompt_path = out_dir / "per_prompt.jsonl"
    completed_ids = jsonl_ids(per_prompt_path)
    progress_cb = _progress_logger()

    def _append_progress(idx: int, total: int, row: dict, result_row: dict) -> None:
        append_row(per_prompt_path, result_row)
        update_run_progress(f"eval_{system}", done=idx, total=total, unit="prompts")
        progress_cb(idx, total, row, result_row)

    per_prompt, aggregate = run_eval(
        system,
        prompts,
        store=store,
        embedder=embedder,
        generator=generator,
        top_k=retr_cfg.top_k,
        index_meta=meta,
        verifier=verifier,
        graph_retriever=graph_retriever,
        reranker=reranker,
        on_progress=_append_progress,
        skip_arxiv_ids=completed_ids,
        factscore_max_claims=eval_cfg.factscore_max_claims,
        ragas_max_claims=eval_cfg.ragas_max_claims,
        verifier_max_batch_size=eval_cfg.verifier_vllm.max_batch_size,
    )
    if completed_ids:
        existing = read_jsonl(per_prompt_path)
        per_prompt = existing + per_prompt
        factscores = [float(row["factscore"]) for row in per_prompt]
        aggregate.update(
            {
                "n_prompts": len(per_prompt),
                "factscore_mean": sum(factscores) / len(factscores) if factscores else 0.0,
            }
        )
    aggregate["verifier_url"] = (
        None if mock_verifier else (verifier_url or eval_cfg.verifier_server_url)
    )
    aggregate.update(sft_provenance)
    aggregate["generator_4bit"] = use_generator_4bit
    aggregate["input_fingerprint"] = phase_input_fingerprint(f"eval_{system}")

    (out_dir / "results.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Wrote %s", out_dir / "results.json")
    finish_run(ctx)
    click.echo(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
