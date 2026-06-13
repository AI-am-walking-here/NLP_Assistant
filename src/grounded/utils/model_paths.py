"""Resolve Hugging Face hub IDs to on-disk weights under GROUNDED_MODELS_ROOT."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ModelRole = Literal["embedder", "generator_8b", "verifier_70b_awq"]

_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "pytorch_model.bin")


def models_root() -> Path:
    raw = os.environ.get("GROUNDED_MODELS_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    try:
        from grounded.config import load_config

        cfg = load_config("models")
        return Path(cfg.root).expanduser().resolve()
    except Exception:
        return Path("/data/team1/models")


def _role_for_hub_id(hub_id: str) -> ModelRole | None:
    lowered = hub_id.lower()
    if "bge" in lowered:
        return "embedder"
    if "70b" in lowered and ("awq" in lowered or "quant" in lowered):
        return "verifier_70b_awq"
    if "8b" in lowered or "llama-3.1" in lowered:
        return "generator_8b"
    return None


def _spec_for_role(role: ModelRole) -> dict[str, Any] | None:
    try:
        from grounded.config import load_config

        cfg = load_config("models")
        return getattr(cfg, role).model_dump()
    except Exception:
        return None


def local_dir_for_role(role: ModelRole) -> Path | None:
    spec = _spec_for_role(role)
    if not spec:
        return None
    rel = spec.get("local_dir") or spec.get("local")
    if not rel:
        return None
    path = models_root() / str(rel)
    return path if path.is_dir() else None


def model_weights_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    for pattern in _WEIGHT_GLOBS:
        if any(path.glob(pattern)):
            return True
    if any(path.glob("model-*.safetensors")):
        return True
    return False


def resolve_model_path(
    hub_id: str,
    *,
    role: ModelRole | None = None,
) -> str:
    """
    Return a local directory path when weights are present, else the hub_id string.

    Callers pass the result to ``from_pretrained`` / ``SentenceTransformer``.
    """
    role = role or _role_for_hub_id(hub_id)
    if role:
        local = local_dir_for_role(role)
        if local and model_weights_ready(local):
            logger.debug("Using local weights for %s: %s", hub_id, local)
            return str(local)
    # Direct path or non-standard layout under models root
    candidate = models_root() / hub_id
    if candidate.is_dir() and model_weights_ready(candidate):
        return str(candidate)
    return hub_id


def is_local_model_path(model_id: str) -> bool:
    return Path(model_id).is_dir()


def model_status() -> dict[str, Any]:
    """Summary for preflight / CLI (no GPU load)."""
    out: dict[str, Any] = {"root": str(models_root()), "models": {}}
    for role in ("embedder", "generator_8b", "verifier_70b_awq"):
        spec = _spec_for_role(role) or {}
        hub_id = spec.get("hub_id", "")
        local = local_dir_for_role(role)  # type: ignore[arg-type]
        ready = bool(local and model_weights_ready(local))
        out["models"][role] = {
            "hub_id": hub_id,
            "local_dir": str(local) if local else None,
            "weights_ready": ready,
        }
    return out
