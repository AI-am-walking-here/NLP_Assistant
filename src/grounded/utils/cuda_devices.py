"""CUDA pinning helpers for eval / inference workers."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def parse_cuda_visible_devices() -> list[str] | None:
    """Return explicit CUDA_VISIBLE_DEVICES indices, or None if unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def local_cuda_device_map() -> dict[str, int] | str:
    """
    Pin HF/accelerate loads to the worker's primary visible GPU.

    When CUDA_VISIBLE_DEVICES selects one card, that card is always cuda:0
    inside the process — use device_map={"": 0} instead of "auto" so weights
    never spill onto verifier GPUs.
    """
    try:
        import torch
    except ImportError:
        return "auto"

    if not torch.cuda.is_available():
        return "auto"

    # Always pin inference to cuda:0 (physical GPU chosen via CUDA_VISIBLE_DEVICES).
    # device_map="auto" with multiple visible devices spreads weights across cards
    # and caused OOM against the verifier on GPU 0/1.
    return {"": 0}


def assert_eval_worker_cuda_isolation() -> None:
    """
    Fail fast when an eval worker can see verifier GPUs.

    Unset CUDA_VISIBLE_DEVICES lets device_map="auto" spread the 8B stack
    across all cards and OOM against the 70B verifier on GPU 0/1.
    """
    if os.environ.get("ALLOW_VERIFIER_GPU_OVERLAP", "0") == "1":
        return

    from grounded.utils.list_gpus import verifier_reserved_gpus

    reserved = verifier_reserved_gpus()
    if not reserved:
        return

    visible = parse_cuda_visible_devices()
    if visible is None:
        raise RuntimeError(
            "Eval worker must set CUDA_VISIBLE_DEVICES to GPUs disjoint from "
            f"verifier ({','.join(sorted(reserved))}); unset means all GPUs are "
            "visible and device_map=auto can OOM on verifier cards."
        )

    overlap = sorted(set(visible) & reserved)
    if overlap:
        raise RuntimeError(
            f"Eval worker CUDA_VISIBLE_DEVICES={','.join(visible)} overlaps "
            f"verifier GPUs ({','.join(sorted(reserved))})."
        )


def configure_eval_worker_cuda() -> None:
    """Pin the current process to cuda:0 and validate verifier isolation."""
    assert_eval_worker_cuda_isolation()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    visible = parse_cuda_visible_devices()
    logger.info(
        "Eval worker CUDA pin: CUDA_VISIBLE_DEVICES=%s device_count=%s device_map=%s",
        ",".join(visible) if visible is not None else "(all)",
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        local_cuda_device_map(),
    )


def hf_model_device_map(cuda_device: int = 0) -> dict[str, int] | str:
    """Pin a model to a local CUDA index within CUDA_VISIBLE_DEVICES."""
    try:
        import torch
    except ImportError:
        return "auto"
    if not torch.cuda.is_available():
        return "auto"
    count = torch.cuda.device_count()
    idx = max(0, min(int(cuda_device), count - 1))
    return {"": idx}

