"""Segment 3 — chunking, embedding, and FAISS retrieval."""

from grounded.index.chunker import chunk_paper, chunk_papers_to_records
from grounded.index.embed import ChunkEmbedder
from grounded.index.vector_store import VectorStore

__all__ = [
    "ChunkEmbedder",
    "VectorStore",
    "chunk_paper",
    "chunk_papers_to_records",
]
