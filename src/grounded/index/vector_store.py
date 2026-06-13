"""M-3.3 — FAISS IndexFlatIP over L2-normalized embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from grounded.index.embed import ChunkEmbedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        index: Any,
        chunk_rows: list[dict[str, Any]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._index = index
        self._rows = chunk_rows
        self.meta = meta or {}

    @property
    def size(self) -> int:
        return len(self._rows)

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        chunk_rows: list[dict[str, Any]],
        *,
        index_type: str = "IndexFlatIP",
    ) -> VectorStore:
        import faiss

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        if index_type != "IndexFlatIP":
            raise ValueError(f"Unsupported index_type: {index_type}")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info("Built FAISS %s: %d vectors, dim=%d", index_type, embeddings.shape[0], dim)
        return cls(index, chunk_rows, meta={"index_type": index_type, "dim": dim, "count": len(chunk_rows)})

    def search(self, query_emb: np.ndarray, k: int) -> list[dict[str, Any]]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = np.asarray(query_emb, dtype=np.float32)
        scores, indices = self._index.search(query_emb, k)
        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            row = dict(self._rows[int(idx)])
            row["score"] = float(score)
            row["rank"] = rank
            results.append(row)
        return results

    def search_text(
        self,
        query: str,
        embedder: ChunkEmbedder,
        k: int,
    ) -> list[dict[str, Any]]:
        query_emb = embedder.encode([query])[0]
        return self.search(query_emb, k)

    def save(self, index_path: Path, meta_path: Path) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        meta_path.write_text(
            json.dumps({**self.meta, "chunk_count": len(self._rows)}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_path: Path,
        chunk_rows: list[dict[str, Any]],
        meta_path: Path | None = None,
    ) -> VectorStore:
        import faiss

        index = faiss.read_index(str(index_path))
        meta: dict[str, Any] = {}
        if meta_path and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(index, chunk_rows, meta=meta)


def load_chunk_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))
