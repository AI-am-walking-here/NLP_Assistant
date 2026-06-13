"""M-7.1 / M-5.3 — full pipeline and adapter resolution."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from grounded.generate.baselines import MockGenerator
from grounded.generate.pipeline import (
    FullPipelineConfig,
    IncompleteSftTrainingError,
    config_for_system,
    full_pipeline,
    require_latest_adapter,
    resolve_latest_adapter,
    resolve_sft_adapter,
)
from grounded.index.embed import MockEmbedder
from grounded.index.vector_store import VectorStore
from grounded.retrieve.rerank import MockReranker


def test_config_for_ablations() -> None:
    g = config_for_system("full_minus_graph")
    assert g is not None and g.use_graph is False
    r = config_for_system("full_minus_rerank")
    assert r is not None and r.use_rerank is False


def test_resolve_latest_adapter(tmp_path: Path) -> None:
    run = tmp_path / "seg5_sft_train_2026"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    found = resolve_latest_adapter("seg5_sft_train_*", runs_dir=tmp_path)
    assert found == adapter


def test_resolve_sft_adapter_strict_latest_incomplete(tmp_path: Path) -> None:
    old = tmp_path / "seg5_sft_train_old"
    (old / "adapter").mkdir(parents=True)
    (old / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    time.sleep(0.02)
    incomplete = tmp_path / "seg5_sft_train_new"
    incomplete.mkdir()
    with pytest.raises(IncompleteSftTrainingError):
        resolve_sft_adapter("seg5_sft_train_*", runs_dir=tmp_path, strict_latest=True)
    loose = resolve_sft_adapter("seg5_sft_train_*", runs_dir=tmp_path, strict_latest=False)
    assert loose is not None
    assert loose.adapter_dir == old / "adapter"


def test_require_latest_adapter_rejects_incomplete_newest(tmp_path: Path) -> None:
    old = tmp_path / "seg6_rankrag_train_old"
    (old / "adapter").mkdir(parents=True)
    (old / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    time.sleep(0.02)
    newest = tmp_path / "seg6_rankrag_train_new"
    newest.mkdir()
    with pytest.raises(IncompleteSftTrainingError):
        require_latest_adapter("seg6_rankrag_train_*", runs_dir=tmp_path)


def test_full_pipeline_mock_stack(tmp_path: Path, monkeypatch) -> None:
    """Smoke full_pipeline with tiny in-memory index."""
    import numpy as np

    records = [
        {
            "chunk_id": "p1:0:0",
            "paper_id": "p1",
            "text": "Neural networks improve natural language processing benchmarks.",
            "section_heading": "Intro",
            "chunk_idx": 0,
        },
        {
            "chunk_id": "p2:0:0",
            "paper_id": "p2",
            "text": "Graph methods for knowledge retrieval in NLP.",
            "section_heading": "Methods",
            "chunk_idx": 0,
        },
    ]
    emb = np.vstack([MockEmbedder(16).encode([r["text"]])[0] for r in records])
    store = VectorStore.build(emb, records)
    embedder = MockEmbedder(16)
    gen = MockGenerator()
    result = full_pipeline(
        "Neural NLP",
        "We study benchmark improvements.",
        store,
        embedder,
        MockReranker(),
        gen,
        config=FullPipelineConfig(use_graph=False, top_k=1, n_vector=2, n_candidates=2),
    )
    assert result.abstract_text
    assert len(result.retrieved_chunks) <= 1
