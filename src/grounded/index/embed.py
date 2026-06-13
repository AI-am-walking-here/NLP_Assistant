"""M-3.2 — batch embed chunks with sentence-transformers (L2-normalized)."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from grounded.utils.hf_network import require_model_download
from grounded.utils.model_paths import is_local_model_path, resolve_model_path

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda"):
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class MockEmbedder:
    """Deterministic unit vectors for smoke tests (no HF download)."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def encode(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        del show_progress
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = hash(text) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dimension).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            out[i] = v
        return out


class ChunkEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        resolved = resolve_model_path(model_name, role="embedder")
        require_model_download(f"embedder {model_name}", hub_id=model_name, role="embedder")
        from sentence_transformers import SentenceTransformer

        self.model_name = resolved
        self.device = resolve_device(device)
        self.normalize = normalize
        self.batch_size = batch_size
        local = is_local_model_path(resolved)
        logger.info(
            "Loading embedder %s on %s (local=%s)",
            resolved,
            self.device,
            local,
        )
        self._model = SentenceTransformer(
            resolved,
            device=self.device,
            local_files_only=local,
        )

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)
