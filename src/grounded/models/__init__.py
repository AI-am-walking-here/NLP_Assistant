"""Shared local HF model loaders (4-bit QLoRA inference)."""

from grounded.models.peft_loader import load_peft_causal_lm

__all__ = ["load_peft_causal_lm"]
