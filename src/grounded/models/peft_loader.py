"""Load 4-bit base + optional LoRA adapter for inference (matches training stack)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grounded.utils.cuda_devices import hf_model_device_map
from grounded.utils.hf_network import local_files_only_kwargs, require_local_model_path
from grounded.utils.model_paths import is_local_model_path, resolve_model_path


def load_peft_causal_lm(
    hub_id: str,
    adapter_path: Path | None = None,
    *,
    role: str = "generator_8b",
    cuda_device: int = 0,
) -> tuple[Any, Any]:
    """Return (model, tokenizer). Base is NF4 on CUDA; fp32 on CPU."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = require_local_model_path(
        f"model {hub_id}",
        hub_id=hub_id,
        role=role,  # type: ignore[arg-type]
    )
    local_kw = local_files_only_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, **local_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map=hf_model_device_map(cuda_device),
            **local_kw,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            **local_kw,
        )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    else:
        model = base
    model.eval()
    return model, tokenizer
