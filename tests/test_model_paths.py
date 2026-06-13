"""Local model path resolution."""

from __future__ import annotations

from pathlib import Path

from grounded.utils.hf_network import require_model_download
from grounded.utils.model_paths import model_weights_ready, resolve_model_path


def test_bge_resolves_to_local() -> None:
    path = resolve_model_path("BAAI/bge-large-en-v1.5", role="embedder")
    root = Path(path)
    if root.is_dir() and model_weights_ready(root):
        assert "bge" in path.lower() or path.startswith("/")
        require_model_download("embedder", hub_id="BAAI/bge-large-en-v1.5", role="embedder")


def test_8b_hub_id_when_weights_missing() -> None:
    path = resolve_model_path("meta-llama/Llama-3.1-8B-Instruct", role="generator_8b")
    if not (Path(path).is_dir() and model_weights_ready(Path(path))):
        assert path == "meta-llama/Llama-3.1-8B-Instruct"
