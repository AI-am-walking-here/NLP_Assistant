"""Segment 3 — generation baselines and prompts."""

from grounded.generate.baselines import GenerationResult, naive_rag, zero_shot
from grounded.generate.prompts import (
    format_retrieved_chunks,
    render_abstract_prompt,
    render_sft_prompt,
)

__all__ = [
    "GenerationResult",
    "format_retrieved_chunks",
    "naive_rag",
    "zero_shot",
    "render_abstract_prompt",
    "render_sft_prompt",
]
