#!/usr/bin/env python3
"""M-6.7 — RankRAG LoRA training on data/rankrag/train.jsonl (requires HF + GPU)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from grounded.retrieve.rankrag_reranker import _format_candidate_prompt

import click

from grounded.config import RunContext, finish_run, init_run, load_config, resolve_path
from grounded.progress import update_run_progress
from grounded.utils.phase_resume import phase_input_fingerprint, training_content_key
from grounded.utils.hf_network import (
    enforce_no_hub_download,
    local_files_only_kwargs,
    require_local_model_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _flatten_rankrag_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            query = row.get("query")
            candidates = row.get("candidates") or []
            labels = row.get("labels") or []
            if not isinstance(query, str):
                continue
            for cand, label in zip(candidates, labels):
                if not isinstance(cand, str):
                    continue
                label_text = "1" if int(label) == 1 else "0"
                prompt = _format_candidate_prompt(query, cand)
                rows.append({"prompt": prompt, "label_text": label_text})
    return rows


def _tokenize_rankrag_rows(
    rows: list[dict[str, str]],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, list[int]]]:
    out: list[dict[str, list[int]]] = []
    for row in rows:
        prompt_ids = tokenizer(
            row["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
        )["input_ids"]
        label_ids = tokenizer(row["label_text"], add_special_tokens=False)["input_ids"]
        if not label_ids:
            continue
        label_ids = label_ids[:1]
        input_ids = prompt_ids + label_ids
        labels = ([-100] * len(prompt_ids)) + label_ids
        out.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return out


def _collate_rankrag_batch(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(f["input_ids"]) for f in features)

    def _pad(values: list[int], fill: int) -> list[int]:
        return values + ([fill] * (max_len - len(values)))

    return {
        "input_ids": torch.tensor(
            [_pad(f["input_ids"], pad_token_id) for f in features],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [_pad(f["attention_mask"], 0) for f in features],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [_pad(f["labels"], -100) for f in features],
            dtype=torch.long,
        ),
    }


def _stats(path: Path) -> dict:
    grouped_examples = 0
    positive_labels = 0
    flattened_examples = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped_examples += 1
            labels = row.get("labels", [])
            positive_labels += sum(labels)
            flattened_examples += min(len(row.get("candidates", [])), len(labels))
    return {
        "grouped_examples": grouped_examples,
        "flattened_examples": flattened_examples,
        "positive_labels": positive_labels,
    }


def rankrag_score_probe(
    reranker: Any,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    checked = 0
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    row_wins = 0
    for row in rows:
        scores = reranker.score(row["query"], row["candidates"])
        pos = [float(s) for s, label in zip(scores, row["labels"]) if int(label) == 1]
        neg = [float(s) for s, label in zip(scores, row["labels"]) if int(label) == 0]
        if not pos or not neg:
            continue
        checked += 1
        positive_scores.extend(pos)
        negative_scores.extend(neg)
        if max(pos) > max(neg):
            row_wins += 1
    return {
        "rows_checked": float(checked),
        "avg_positive_score": (
            sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
        ),
        "avg_negative_score": (
            sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
        ),
        "positive_beats_best_negative_rate": (row_wins / checked) if checked else 0.0,
    }


@click.command()
@click.option("--dry-run", is_flag=True, help="Validate data + config only.")
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Resume/write training in this existing run directory.",
)
@click.option("--max-steps", type=int, default=None, help="Cap training steps (smoke/debug).")
def main(dry_run: bool, run_dir: Path | None, max_steps: int | None) -> int:
    enforce_no_hub_download()
    rr_cfg = load_config("rankrag")
    train_path = resolve_path(rr_cfg.paths.train_jsonl)
    if not train_path.is_file():
        raise click.ClickException(
            f"Missing {train_path}. Run: python3 scripts/build_rankrag_data.py"
        )

    stats = _stats(train_path)

    if dry_run:
        out = {**stats, "status": "dry_run_ok"}
        click.echo(json.dumps(out, indent=2))
        return 0

    ctx = init_run("seg6", "rankrag_train", tags=["m-6.7"])
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

    model_path = require_local_model_path(
        "RankRAG LoRA (Llama-3.1-8B-Instruct)",
        hub_id=rr_cfg.base_model,
        role="generator_8b",
    )
    local_kw = local_files_only_kwargs()

    try:
        import peft  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise click.ClickException("Install train stack: pip install -e '.[train]'") from exc

    adapter_dir = ctx.run_dir / "adapter"
    flat_rows = _flatten_rankrag_rows(train_path)
    if not flat_rows:
        raise click.ClickException(f"No flattened RankRAG examples in {train_path}")
    ds = Dataset.from_list(flat_rows)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, **local_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        **local_kw,
    )
    lora = LoraConfig(
        r=rr_cfg.lora_rank,
        lora_alpha=rr_cfg.lora_alpha,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    tokenized_rows = _tokenize_rankrag_rows(
        flat_rows,
        tokenizer,
        max_length=2048,
    )
    if not tokenized_rows:
        raise click.ClickException("No tokenized RankRAG examples after label masking.")
    ds = Dataset.from_list(tokenized_rows)

    training_args = TrainingArguments(
        output_dir=str(ctx.run_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
        max_steps=max_steps if max_steps is not None else -1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=lambda features: _collate_rankrag_batch(
            features,
            tokenizer.pad_token_id,
        ),
    )
    try:
        from transformers import TrainerCallback

        class _ProgressCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
                update_run_progress(
                    "rankrag_train",
                    done=int(state.global_step),
                    total=int(state.max_steps or 0),
                    unit="steps",
                )

        trainer.add_callback(_ProgressCallback())
    except Exception:
        pass
    checkpoints = sorted(ctx.run_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) in (-1, 0)
    if is_main_process:
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        (ctx.run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "status": "trained",
                    "adapter_dir": str(adapter_dir),
                    "inputs_fingerprint": phase_input_fingerprint("rankrag_train"),
                    "train_content_key": training_content_key("rankrag_train"),
                    "distributed": local_rank >= 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        finish_run(ctx)
    click.echo(str(adapter_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
