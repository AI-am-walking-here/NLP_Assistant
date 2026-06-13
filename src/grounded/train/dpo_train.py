"""M-5.x — QLoRA DPO on FActScore preference pairs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from grounded.config import DpoConfig, SftConfig, resolve_path
from grounded.generate.pipeline import require_sft_adapter
from grounded.utils.hf_network import (
    enforce_no_hub_download,
    local_files_only_kwargs,
    require_local_model_path,
)
from grounded.utils.model_paths import resolve_model_path

logger = logging.getLogger(__name__)


def count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def preflight_dpo(sft_cfg: SftConfig, dpo_cfg: DpoConfig) -> dict[str, Any]:
    pairs_path = resolve_path(dpo_cfg.pairs_jsonl)
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Missing {pairs_path}. Run: python scripts/build_sft_dpo_data.py"
        )
    n = count_jsonl_rows(pairs_path)
    model_path = resolve_model_path(sft_cfg.base_model, role="generator_8b")
    init = None
    if dpo_cfg.init_from_latest_sft:
        res = require_sft_adapter()
        init = str(res.adapter_dir) if res else None
    return {
        "pairs_count": n,
        "pairs_jsonl": str(pairs_path),
        "base_model_hub": sft_cfg.base_model,
        "base_model_resolved": model_path,
        "init_adapter": init,
        "dpo_beta": dpo_cfg.beta,
        "num_epochs": dpo_cfg.num_epochs,
    }


def run_dpo_training(
    sft_cfg: SftConfig,
    dpo_cfg: DpoConfig,
    *,
    run_dir: Path,
    max_steps: int | None = None,
    dry_run: bool = False,
    init_adapter_path: Path | None = None,
) -> dict[str, Any]:
    report = preflight_dpo(sft_cfg, dpo_cfg)
    if dry_run:
        report["status"] = "dry_run_ok"
        return report

    if report["pairs_count"] < 10:
        raise RuntimeError(
            f"DPO needs more pairs (got {report['pairs_count']}). "
            "Run build_sft_dpo_data.py with verifier up or lower min_factscore_margin."
        )

    enforce_no_hub_download()
    model_path = require_local_model_path(
        "QLoRA DPO (Llama-3.1-8B-Instruct)",
        hub_id=sft_cfg.base_model,
        role="generator_8b",
    )
    local_kw = local_files_only_kwargs()
    pairs_path = Path(report["pairs_jsonl"])

    try:
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Install train stack: pip install peft trl datasets bitsandbytes accelerate"
        ) from exc

    adapter_dir = run_dir / "adapter"
    ds = load_dataset("json", data_files=str(pairs_path), split="train")

    def _normalize_dpo_row(row: dict[str, Any]) -> dict[str, Any]:
        chosen = row["chosen"]
        rejected = row["rejected"]
        if isinstance(chosen, str):
            chosen = [{"role": "assistant", "content": chosen}]
            rejected = [{"role": "assistant", "content": rejected}]
        return {"prompt": row["prompt"], "chosen": chosen, "rejected": rejected}

    ds = ds.map(_normalize_dpo_row, remove_columns=ds.column_names)

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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        **local_kw,
    )

    init_path = init_adapter_path
    if init_path is None and dpo_cfg.init_from_latest_sft:
        res = require_sft_adapter()
        init_path = res.adapter_dir
        report["init_adapter"] = str(init_path)

    if init_path is not None and Path(init_path).is_dir():
        logger.info("Warm-start DPO from adapter %s", init_path)
        model = PeftModel.from_pretrained(model, str(init_path), is_trainable=True)
    else:
        lora = LoraConfig(
            r=sft_cfg.lora_rank,
            lora_alpha=sft_cfg.lora_alpha,
            lora_dropout=sft_cfg.lora_dropout,
            target_modules=sft_cfg.target_modules,
            task_type="CAUSAL_LM",
        )
        from peft import get_peft_model

        model = get_peft_model(model, lora)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    report_to = "wandb" if os.environ.get("WANDB_PROJECT") else "none"
    training_args = DPOConfig(
        output_dir=str(run_dir),
        beta=dpo_cfg.beta,
        num_train_epochs=dpo_cfg.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        learning_rate=dpo_cfg.learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=dpo_cfg.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=report_to,
        max_steps=max_steps if max_steps is not None else -1,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    logger.info(
        "Starting DPO: model=%s pairs=%s init=%s",
        model_path,
        report["pairs_count"],
        report.get("init_adapter"),
    )
    try:
        trainer.train()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise RuntimeError(f"DPO training failed: {exc}") from exc

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved DPO adapter to %s", adapter_dir)

    report["status"] = "trained"
    report["adapter_dir"] = str(adapter_dir)
    return report
