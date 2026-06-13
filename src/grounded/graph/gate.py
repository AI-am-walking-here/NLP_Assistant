"""M-6.3 — extrapolate graph-extraction GPU hours from pilot stats."""

from __future__ import annotations

from typing import Any


def extrapolate_gpu_hours(
    *,
    pilot_chunks: int,
    pilot_seconds: float,
    target_chunks: int,
    gate_hours: float,
) -> dict[str, Any]:
    if pilot_chunks <= 0:
        return {
            "pilot_chunks": 0,
            "pilot_gpu_hours": 0.0,
            "target_chunks": target_chunks,
            "projected_gpu_hours": float("inf"),
            "gate_hours": gate_hours,
            "keep_graph": False,
            "note": "no pilot chunks",
        }
    pilot_hours = (pilot_chunks * pilot_seconds) / 3600.0
    projected = pilot_hours * (target_chunks / pilot_chunks)
    return {
        "pilot_chunks": pilot_chunks,
        "pilot_gpu_hours": round(pilot_hours, 3),
        "seconds_per_chunk": pilot_seconds,
        "target_chunks": target_chunks,
        "projected_gpu_hours": round(projected, 2),
        "gate_hours": gate_hours,
        "keep_graph": projected <= gate_hours,
    }
