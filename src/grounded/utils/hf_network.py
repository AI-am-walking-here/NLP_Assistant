"""HuggingFace Hub policy — local weights under GROUNDED_MODELS_ROOT need no Hub pull."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from grounded.utils.model_paths import (
    ModelRole,
    is_local_model_path,
    model_weights_ready,
    resolve_model_path,
)

_ALLOW_ENV = "GROUNDED_ALLOW_MODEL_DOWNLOAD"


def allow_model_download() -> bool:
    """True only when ``GROUNDED_ALLOW_MODEL_DOWNLOAD=1`` (professor / ops opt-in)."""
    return os.environ.get(_ALLOW_ENV, "").strip().lower() in ("1", "true", "yes")


def require_model_download(
    purpose: str,
    *,
    hub_id: str | None = None,
    role: ModelRole | None = None,
) -> None:
    """
    Raise before any Hub pull unless weights exist locally or download is explicitly allowed.
    """
    if hub_id:
        resolved = resolve_model_path(hub_id, role=role)
        if is_local_model_path(resolved) and model_weights_ready(Path(resolved)):
            return
    if allow_model_download():
        return
    raise RuntimeError(
        f"Refusing to download models for: {purpose}. "
        f"Place weights under GROUNDED_MODELS_ROOT (see configs/models.yaml) "
        f"or set {_ALLOW_ENV}=1 to allow Hub downloads."
    )


def enforce_no_hub_download() -> None:
    """
    Block Hugging Face Hub pulls for this process (university host policy).

    Skipped when ``GROUNDED_ALLOW_MODEL_DOWNLOAD=1`` is set.
    """
    if allow_model_download():
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def local_files_only_kwargs() -> dict[str, bool]:
    """Kwargs for ``from_pretrained`` when weights must come from disk."""
    return {"local_files_only": True}


def require_local_model_path(
    purpose: str,
    *,
    hub_id: str,
    role: ModelRole | None = None,
) -> str:
    """Return a local directory path or raise (never triggers a Hub download)."""
    enforce_no_hub_download()
    resolved = resolve_model_path(hub_id, role=role)
    if is_local_model_path(resolved) and model_weights_ready(Path(resolved)):
        return resolved
    raise RuntimeError(
        f"Refusing to download models for: {purpose}. "
        f"Missing local weights for {hub_id!r} (role={role!r}). "
        f"Rsync to GROUNDED_MODELS_ROOT — see docs/OFFLINE_MODEL_TRANSFER.md. "
        f"Do not set {_ALLOW_ENV}=1 on the university host unless explicitly approved."
    )


def configure_hf_hub() -> dict[str, Any]:
    """Return Hub env snapshot (does not download)."""
    return {
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "GROUNDED_MODELS_ROOT": os.environ.get("GROUNDED_MODELS_ROOT"),
        "allow_model_download": allow_model_download(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }
