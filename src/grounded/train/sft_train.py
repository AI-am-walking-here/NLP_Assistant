"""M-5.2 — QLoRA SFT on Llama-3.1-8B-Instruct (build_plan v3.1)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from grounded.config import SftConfig, resolve_path
from grounded.train.sft_data import sft_coverage_report
from grounded.utils.phase_resume import phase_input_fingerprint, training_content_key
from grounded.progress import update_run_progress
from grounded.utils.hf_network import (
    enforce_no_hub_download,
    local_files_only_kwargs,
    require_local_model_path,
)
from grounded.utils.model_paths import resolve_model_path

logger = logging.getLogger(__name__)


def count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def preflight_sft(sft_cfg: SftConfig) -> dict[str, Any]:
    """Validate data paths and base model availability (no GPU)."""
    train_path = resolve_path(sft_cfg.paths.train_jsonl)
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Missing {train_path}. Run: python scripts/build_sft_data.py"
        )
    split_path = resolve_path(sft_cfg.paths.split_list)
    val_path = resolve_path(sft_cfg.paths.val_jsonl)
    n = count_jsonl_rows(train_path)
    model_path = resolve_model_path(sft_cfg.base_model, role="generator_8b")
    coverage = sft_coverage_report(split_path, train_path, val_path)
    merge_manifest_path = train_path.parent / "merge_manifest.json"
    merge_manifest: dict[str, Any] = {}
    if merge_manifest_path.is_file():
        merge_manifest = json.loads(merge_manifest_path.read_text(encoding="utf-8"))
    skipped = set(merge_manifest.get("skipped_unbuildable", []))
    missing = set(coverage["missing_from_train"])
    coverage_ok = coverage["ok"] or (
        merge_manifest
        and not coverage["extra_in_train"]
        and missing <= skipped
        and merge_manifest.get("built_count", 0) + len(skipped) == coverage["split_count"]
    )
    return {
        "train_examples": n,
        "train_jsonl": str(train_path),
        "split_list": str(split_path),
        "split_count": coverage["split_count"],
        "sft_coverage_ok": bool(coverage_ok),
        "sft_coverage_pct": coverage["coverage_pct"],
        "missing_from_train": len(coverage["missing_from_train"]),
        "extra_in_train": len(coverage["extra_in_train"]),
        "skipped_unbuildable": len(skipped),
        "merge_manifest": str(merge_manifest_path) if merge_manifest else None,
        "base_model_hub": sft_cfg.base_model,
        "base_model_resolved": model_path,
        "lora_rank": sft_cfg.lora_rank,
        "num_epochs": sft_cfg.num_epochs,
        "max_seq_len": sft_cfg.max_seq_len,
    }


def run_sft_training(
    sft_cfg: SftConfig,
    *,
    run_dir: Path,
    max_steps: int | None = None,
    dry_run: bool = False,
    eval_during_train: bool = True,
    resume_from_checkpoint: bool = True,
) -> dict[str, Any]:
    """
    Train a LoRA adapter; writes ``run_dir/adapter/``.

    Raises if base weights are missing unless ``dry_run=True``.
    """
    report = preflight_sft(sft_cfg)
    if dry_run:
        report["status"] = "dry_run_ok"
        return report

    if not report.get("sft_coverage_ok"):
        raise RuntimeError(
            "SFT train.jsonl is out of sync with data/splits/sft.txt "
            f"({report.get('missing_from_train')} missing, "
            f"{report.get('extra_in_train')} stale). "
            "Run: python scripts/build_sft_data.py "
            "(or: python scripts/seg2_bookkeeping.py --only splits)"
        )

    enforce_no_hub_download()
    model_path = require_local_model_path(
        "QLoRA SFT (Llama-3.1-8B-Instruct)",
        hub_id=sft_cfg.base_model,
        role="generator_8b",
    )
    local_kw = local_files_only_kwargs()
    train_path = Path(report["train_jsonl"])
    val_path = resolve_path(sft_cfg.paths.val_jsonl)

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig as TrlSFTConfig
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Install train stack: pip install peft trl datasets bitsandbytes accelerate"
        ) from exc

    adapter_dir = run_dir / "adapter"
    data_files: dict[str, str] = {"train": str(train_path)}
    if val_path.is_file() and count_jsonl_rows(val_path) > 0:
        data_files["validation"] = str(val_path)
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"]
    eval_ds = ds["validation"] if "validation" in ds else None

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

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        **local_kw,
    )
    lora = LoraConfig(
        r=sft_cfg.lora_rank,
        lora_alpha=sft_cfg.lora_alpha,
        lora_dropout=sft_cfg.lora_dropout,
        target_modules=sft_cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    def format_row(row: dict) -> str:
        msgs = row["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(msgs, tokenize=False)
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    report_to = "wandb" if os.environ.get("WANDB_PROJECT") else "none"
    training_args = TrlSFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=sft_cfg.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=sft_cfg.learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=sft_cfg.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy=(
            "epoch" if eval_ds is not None and eval_during_train else "no"
        ),
        per_device_eval_batch_size=1,
        eval_accumulation_steps=8,
        bf16=True,
        report_to=report_to,
        max_steps=max_steps if max_steps is not None else -1,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=format_row,
    )
    logger.info(
        "Starting SFT: model=%s examples=%s max_steps=%s",
        model_path,
        report["train_examples"],
        max_steps,
    )
    try:
        callbacks = []
        try:
            from transformers import TrainerCallback

            class _ProgressCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
                    total = int(state.max_steps or 0)
                    update_run_progress(
                        "sft_train",
                        done=int(state.global_step),
                        total=total,
                        unit="steps",
                    )

            callbacks.append(_ProgressCallback())
            for callback in callbacks:
                trainer.add_callback(callback)
        except Exception:
            pass
        ckpt = None
        if resume_from_checkpoint:
            checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
            ckpt = str(checkpoints[-1]) if checkpoints else None
        trainer.train(resume_from_checkpoint=ckpt)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise RuntimeError(f"SFT training failed: {exc}") from exc

    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) in (-1, 0)
    if is_main_process:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info("Saved adapter to %s", adapter_dir)

    report["status"] = "trained"
    report["adapter_dir"] = str(adapter_dir)
    report["inputs_fingerprint"] = phase_input_fingerprint("sft_train")
    report["train_content_key"] = training_content_key("sft_train")
    if is_main_process:
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "status": "trained",
                    "adapter_dir": str(adapter_dir),
                    "inputs_fingerprint": report["inputs_fingerprint"],
                    "train_content_key": report["train_content_key"],
                    "distributed": local_rank >= 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report
