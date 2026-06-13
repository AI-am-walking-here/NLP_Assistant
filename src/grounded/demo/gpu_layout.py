"""Discover demo GPU placement across verifier-disjoint cards."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from grounded.utils.list_gpus import GpuInfo, list_gpu_status, verifier_reserved_gpus

logger = logging.getLogger(__name__)

_LLM_MIN_FREE_MB = 6000
_EMBED_MIN_FREE_MB = 2500


@dataclass(frozen=True)
class DemoGpuLayout:
    mode: Literal["parallel", "sequential"]
    physical_gpus: tuple[str, ...]
    embed_local: int | None
    rankrag_local: int
    generator_local: int

    @property
    def embed_device(self) -> str:
        if self.embed_local is None:
            return "cpu"
        return f"cuda:{self.embed_local}"

    def physical_for_local(self, local: int) -> str | None:
        if 0 <= local < len(self.physical_gpus):
            return self.physical_gpus[local]
        return None

    def summary(self) -> str:
        parts = [f"mode={self.mode}", f"visible={','.join(self.physical_gpus)}"]
        if self.embed_local is not None:
            parts.append(f"embed=cuda:{self.embed_local}(gpu{self.physical_for_local(self.embed_local)})")
        else:
            parts.append("embed=cpu")
        parts.append(
            f"rankrag=cuda:{self.rankrag_local}(gpu{self.physical_for_local(self.rankrag_local)})"
        )
        parts.append(
            f"generator=cuda:{self.generator_local}(gpu{self.physical_for_local(self.generator_local)})"
        )
        return " ".join(parts)


def _eligible_gpus(reserved: set[str]) -> list[GpuInfo]:
    rows = list_gpu_status(min_free_mb=0, respect_cuda_visible=False)
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    allowed = (
        {part.strip() for part in visible_raw.split(",") if part.strip()}
        if visible_raw
        else None
    )
    out: list[GpuInfo] = []
    for gpu in rows:
        if gpu.index in reserved:
            continue
        if allowed is not None and gpu.index not in allowed:
            continue
        out.append(gpu)
    out.sort(key=lambda g: (-g.memory_free_mb, g.index))
    return out


def discover_demo_gpu_layout(*, reserved: set[str] | None = None) -> DemoGpuLayout:
    """
    Assign embedder / RankRAG / generator to disjoint local CUDA indices.

    - 2+ LLM-capable GPUs (>=6 GB free): parallel — rankrag + generator stay loaded.
    - Optional third GPU for BGE; otherwise embedder uses CPU.
    - 1 LLM-capable GPU: sequential swap (rankrag then generator).
    """
    reserved = reserved or verifier_reserved_gpus() or {"0", "1"}
    rows = _eligible_gpus(reserved)
    if not rows:
        raise RuntimeError(
            "No demo GPUs found outside verifier reservation "
            f"({','.join(sorted(reserved))}). Set CUDA_VISIBLE_DEVICES or free a card."
        )

    llm_ready = [g for g in rows if g.memory_free_mb >= _LLM_MIN_FREE_MB]
    if len(llm_ready) >= 2:
        rankrag_gpu = llm_ready[1]
        generator_gpu = llm_ready[0]
        embed_gpu: GpuInfo | None = None
        used = {rankrag_gpu.index, generator_gpu.index}
        for gpu in rows:
            if gpu.index in used:
                continue
            if gpu.memory_free_mb >= _EMBED_MIN_FREE_MB:
                embed_gpu = gpu
                break

        if embed_gpu is not None:
            physical = (embed_gpu.index, rankrag_gpu.index, generator_gpu.index)
            layout = DemoGpuLayout(
                mode="parallel",
                physical_gpus=physical,
                embed_local=0,
                rankrag_local=1,
                generator_local=2,
            )
        else:
            physical = (rankrag_gpu.index, generator_gpu.index)
            layout = DemoGpuLayout(
                mode="parallel",
                physical_gpus=physical,
                embed_local=None,
                rankrag_local=0,
                generator_local=1,
            )
        logger.info("Demo GPU layout (parallel): %s", layout.summary())
        return layout

    best = llm_ready[0] if llm_ready else rows[0]
    if best.memory_free_mb < _LLM_MIN_FREE_MB:
        logger.warning(
            "Best demo GPU %s has only %d MiB free (< %d); sequential loads may OOM",
            best.index,
            best.memory_free_mb,
            _LLM_MIN_FREE_MB,
        )
    layout = DemoGpuLayout(
        mode="sequential",
        physical_gpus=(best.index,),
        embed_local=None,
        rankrag_local=0,
        generator_local=0,
    )
    logger.info("Demo GPU layout (sequential): %s", layout.summary())
    return layout


def apply_demo_cuda_visible(layout: DemoGpuLayout) -> str:
    """Set CUDA_VISIBLE_DEVICES from layout unless user already exported a list."""
    explicit = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if explicit:
        return explicit
    joined = ",".join(layout.physical_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = joined
    return joined
