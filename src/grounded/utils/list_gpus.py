"""GPU/CPU discovery for parallel pipeline phases."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: str
    memory_free_mb: int
    selected: bool = False


def _visible_from_env() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _min_free_mb(min_free_mb: int | None) -> int:
    return (
        int(os.environ.get("MIN_GPU_FREE_MB", "8000"))
        if min_free_mb is None
        else min_free_mb
    )


def list_gpu_status(
    *,
    min_free_mb: int | None = None,
    respect_cuda_visible: bool = True,
) -> list[GpuInfo]:
    """Return every GPU with free memory and whether it passes the capacity gate."""
    min_free = _min_free_mb(min_free_mb)
    visible = _visible_from_env() if respect_cuda_visible else None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        fallback = visible or ["0"]
        return [GpuInfo(index=v, memory_free_mb=0, selected=False) for v in fallback]

    allowed = set(visible) if visible is not None else None
    out: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        idx, free = [part.strip() for part in line.split(",", 1)]
        if allowed is not None and idx not in allowed:
            continue
        free_mb = int(float(free))
        out.append(
            GpuInfo(
                index=idx,
                memory_free_mb=free_mb,
                selected=free_mb >= min_free,
            )
        )
    return out


def list_available_gpus(
    *,
    min_free_mb: int | None = None,
    respect_cuda_visible: bool = True,
    exclude: set[str] | None = None,
) -> list[GpuInfo]:
    reserved = exclude or set()
    return [
        GpuInfo(index=gpu.index, memory_free_mb=gpu.memory_free_mb)
        for gpu in list_gpu_status(
            min_free_mb=min_free_mb,
            respect_cuda_visible=respect_cuda_visible,
        )
        if gpu.selected and gpu.index not in reserved
    ]


def verifier_reserved_gpus() -> set[str]:
    if os.environ.get("ALLOW_VERIFIER_GPU_OVERLAP", "0") == "1":
        return set()
    raw = os.environ.get("VERIFIER_CUDA_DEVICES", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def discover_worker_gpus(
    *,
    gpus: str | None = None,
    devices_env: str = "SFT_CUDA_DEVICES",
    exclude: set[str] | None = None,
    respect_cuda_visible: bool = False,
    min_free_mb: int | None = None,
) -> list[str]:
    """Usable GPU indices for one-worker-per-GPU shard queues."""
    reserved = exclude or set()

    def _filter(chosen: list[str]) -> list[str]:
        return [gpu for gpu in chosen if gpu not in reserved]

    if gpus:
        return _filter([part.strip() for part in gpus.split(",") if part.strip()])
    explicit = os.environ.get(devices_env)
    if explicit:
        return _filter([part.strip() for part in explicit.split(",") if part.strip()])
    return _filter(
        [
            gpu.index
            for gpu in list_gpu_status(
                min_free_mb=min_free_mb,
                respect_cuda_visible=respect_cuda_visible,
            )
            if gpu.selected
        ]
    )


def discover_cpu_workers(
    *,
    workers: int | None,
    n_shards: int,
    env_key: str = "RANKRAG_DATA_WORKERS",
) -> int:
    if workers is not None:
        cap = workers
    else:
        raw = os.environ.get(env_key, "").strip()
        cap = int(raw) if raw else (os.cpu_count() or 4)
    return max(1, min(cap, n_shards))


def gpus_with_capacity(
    gpus: list[str],
    *,
    min_free_mb: int | None = None,
) -> list[str]:
    """Keep only GPUs that currently pass the free-memory gate."""
    if not gpus:
        return []
    min_free = _min_free_mb(min_free_mb)
    free_by_idx = {
        row.index: row.memory_free_mb
        for row in list_gpu_status(min_free_mb=0, respect_cuda_visible=False)
    }
    return [gpu for gpu in gpus if free_by_idx.get(gpu, 0) >= min_free]


def active_worker_gpus(
    procs: list[tuple[object, str, subprocess.Popen]],
) -> set[str]:
    return {gpu for _, gpu, proc in procs if proc.poll() is None}


def format_cuda_visible(
    *,
    min_free_mb: int | None = None,
    respect_cuda_visible: bool = False,
    devices_env: str = "SFT_CUDA_DEVICES",
    exclude: set[str] | None = None,
) -> str:
    return ",".join(
        discover_worker_gpus(
            devices_env=devices_env,
            exclude=exclude,
            respect_cuda_visible=respect_cuda_visible,
            min_free_mb=min_free_mb,
        )
    )


def format_cuda_visible_report(
    *,
    min_free_mb: int | None = None,
    respect_cuda_visible: bool = False,
    devices_env: str = "SFT_CUDA_DEVICES",
    exclude: set[str] | None = None,
    label: str = "workers",
) -> str:
    explicit = os.environ.get(devices_env)
    if explicit:
        return f"using explicit {devices_env}={explicit}"
    min_free = _min_free_mb(min_free_mb)
    reserved = exclude or set()
    rows = list_gpu_status(
        min_free_mb=min_free_mb,
        respect_cuda_visible=respect_cuda_visible,
    )
    if not rows:
        return f"no GPUs from nvidia-smi (min_free_mb={min_free})"
    parts: list[str] = []
    for gpu in rows:
        if gpu.index in reserved:
            state = "reserved"
        elif gpu.selected:
            state = "usable"
        else:
            state = f"skipped (<{min_free} MiB free)"
        parts.append(f"gpu{gpu.index} free={gpu.memory_free_mb}MiB {state}")
    selected = discover_worker_gpus(
        devices_env=devices_env,
        exclude=exclude,
        respect_cuda_visible=respect_cuda_visible,
        min_free_mb=min_free_mb,
    )
    return "; ".join(parts) + f" -> {label} on [{','.join(selected)}]"
