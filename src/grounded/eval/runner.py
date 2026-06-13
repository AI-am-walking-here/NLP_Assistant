"""M-4.7 / M-7.2 — run a named system on eval prompts and aggregate metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from grounded.eval.factscore import ClaimVerifier, MockClaimVerifier, compute_factscore
from grounded.eval.ragas_wrap import compute_ragas
from grounded.generate.baselines import (
    GenerationResult,
    graph_rag,
    naive_rag,
    naive_rag_sft_prompt,
    rankrag_rag,
    zero_shot,
)
from grounded.generate.pipeline import (
    FULL_PIPELINE_SYSTEMS,
    FullPipelineConfig,
    SFT_GENERATOR_SYSTEMS,
    config_for_system,
    full_pipeline,
)
from grounded.index.embed import ChunkEmbedder, MockEmbedder
from grounded.index.vector_store import VectorStore, load_chunk_rows
from grounded.retrieve.rerank import MockReranker, Reranker

logger = logging.getLogger(__name__)

BASE_SYSTEMS = frozenset(
    {
        "naive_rag",
        "zero_shot",
        "zero_shot_with_sft",
        "naive_rag_with_sft",
        "naive_rag_sft_prompt",
        "graph_only",
        "rankrag_only",
    }
)

SUPPORTED_SYSTEMS = BASE_SYSTEMS | FULL_PIPELINE_SYSTEMS

RETRIEVAL_SYSTEMS = SUPPORTED_SYSTEMS - {"zero_shot", "zero_shot_with_sft"}
GRAPH_SYSTEMS = frozenset({"graph_only"}) | FULL_PIPELINE_SYSTEMS


def _require_reranker(reranker: Reranker | None, system_name: str) -> Reranker:
    if reranker is None:
        raise ValueError(
            f"{system_name} requires a RankRAG reranker; pass load_reranker(adapter) "
            "or use --mock-gen for dev."
        )
    return reranker

RERANK_SYSTEMS = frozenset(
    {
        "rankrag_only",
        "full",
        "full_minus_graph",
        "full_minus_sft",
    }
)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_retrieval_stack(
    retr_cfg,
    *,
    require_real_index: bool = False,
) -> tuple[VectorStore, Any, dict]:
    from grounded.config import resolve_path

    rpaths = retr_cfg.paths
    chunks_path = resolve_path(rpaths.chunks_parquet)
    faiss_path = resolve_path(rpaths.faiss_index)
    meta_path = resolve_path(rpaths.index_meta)
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_rows = load_chunk_rows(chunks_path)
    store = VectorStore.load(faiss_path, chunk_rows, meta_path)
    if meta.get("mock_embed"):
        if require_real_index:
            raise RuntimeError(
                "FAISS index was built with mock_embed=true. "
                "Run: python scripts/build_index.py --only-missing --prune-to-index"
            )
        embedder: Any = MockEmbedder(dimension=int(meta.get("dim", 64)))
    else:
        embedder = ChunkEmbedder(
            retr_cfg.embedder,
            device=retr_cfg.embed_device,
            normalize=retr_cfg.normalize_embeddings,
            batch_size=retr_cfg.embed_batch_size,
        )
    return store, embedder, meta


def load_graph_retriever(retr_cfg) -> Any:
    from grounded.config import resolve_path
    from grounded.index.embed import ChunkEmbedder, MockEmbedder
    from grounded.retrieve.graph import GraphRetriever

    communities_path = resolve_path("data/graph/communities.parquet")
    if not communities_path.is_file():
        raise FileNotFoundError(
            f"Missing {communities_path}. Run: python scripts/build_graph_communities.py"
        )
    chunks_path = resolve_path(retr_cfg.paths.chunks_parquet)
    meta_path = resolve_path(retr_cfg.paths.index_meta)
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("mock_embed"):
        raise RuntimeError(
            "Graph retrieval cannot run with mock_embed=true. "
            "Rebuild the index with real embeddings first."
        )
    else:
        embedder = ChunkEmbedder(
            retr_cfg.embedder,
            device=retr_cfg.embed_device,
            normalize=retr_cfg.normalize_embeddings,
            batch_size=retr_cfg.embed_batch_size,
        )
    index_ids = {
        ln.strip()
        for ln in (
            resolve_path("data/splits/index.txt").read_text(encoding="utf-8").splitlines()
        )
        if ln.strip()
    }
    return GraphRetriever.from_parquet(
        communities_path,
        chunks_path,
        embedder,
        paper_filter=index_ids,
    )


def generate_for_system(
    system_name: str,
    row: dict[str, Any],
    *,
    store: VectorStore | None,
    embedder: Any,
    generator: Any,
    top_k: int,
    graph_retriever: Any | None = None,
    reranker: Reranker | None = None,
) -> GenerationResult:
    if system_name in ("zero_shot", "zero_shot_with_sft"):
        return zero_shot(row["title"], row["outline"], generator)
    if system_name == "graph_only":
        if graph_retriever is None:
            raise ValueError("graph_only requires a loaded GraphRetriever")
        return graph_rag(
            row["title"],
            row["outline"],
            graph_retriever,
            generator,
            k=top_k,
        )
    if system_name == "rankrag_only":
        if store is None or embedder is None:
            raise ValueError("rankrag_only requires vector store and embedder")
        return rankrag_rag(
            row["title"],
            row["outline"],
            store,
            embedder,
            _require_reranker(reranker, system_name),
            generator,
            graph_retriever=graph_retriever,
            k=top_k,
        )
    if system_name in FULL_PIPELINE_SYSTEMS:
        if store is None or embedder is None:
            raise ValueError(f"{system_name} requires vector store and embedder")
        cfg = config_for_system(system_name)
        if cfg is None:
            raise ValueError(f"No pipeline config for {system_name}")
        graph = graph_retriever if cfg.use_graph else None
        effective_reranker = (
            _require_reranker(reranker, system_name) if cfg.use_rerank else MockReranker()
        )
        result = full_pipeline(
            row["title"],
            row["outline"],
            store,
            embedder,
            effective_reranker,
            generator,
            graph_retriever=graph,
            config=cfg,
        )
        if not cfg.use_rerank:
            result.prompt_system = f"{result.prompt_system}\n[meta] reranker=mock_lexical"
        return result
    if system_name == "naive_rag_sft_prompt":
        if store is None or embedder is None:
            raise ValueError("naive_rag_sft_prompt requires vector store and embedder")
        return naive_rag_sft_prompt(
            row["title"],
            row["outline"],
            store,
            embedder,
            generator,
            k=top_k,
        )
    if system_name in ("naive_rag", "naive_rag_with_sft"):
        if store is None or embedder is None:
            raise ValueError("naive_rag requires a loaded vector store and embedder")
        return naive_rag(
            row["title"],
            row["outline"],
            store,
            embedder,
            generator,
            k=top_k,
        )
    raise NotImplementedError(f"Unknown system: {system_name}")


def run_eval(
    system_name: str,
    prompts: list[dict[str, Any]],
    *,
    store: VectorStore | None,
    embedder: Any,
    generator: Any,
    top_k: int,
    index_meta: dict | None = None,
    verifier: ClaimVerifier,
    graph_retriever: Any | None = None,
    reranker: Reranker | None = None,
    on_progress: Callable[[int, int, dict[str, Any], dict[str, Any]], None] | None = None,
    skip_arxiv_ids: set[str] | None = None,
    factscore_max_claims: int = 12,
    ragas_max_claims: int = 8,
    verifier_max_concurrent: int | None = None,
    verifier_use_batch: bool | None = None,
    verifier_max_batch_size: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if system_name not in SUPPORTED_SYSTEMS:
        raise NotImplementedError(f"System {system_name!r} not in {sorted(SUPPORTED_SYSTEMS)}")

    mock_verifier = isinstance(verifier, MockClaimVerifier)
    per_prompt: list[dict[str, Any]] = []
    factscores: list[float] = []

    total = len(prompts)
    for idx, row in enumerate(prompts, start=1):
        if skip_arxiv_ids and str(row.get("arxiv_id")) in skip_arxiv_ids:
            continue
        result_row = evaluate_one_prompt(
            system_name,
            row,
            store=store,
            embedder=embedder,
            generator=generator,
            top_k=top_k,
            verifier=verifier,
            graph_retriever=graph_retriever,
            reranker=reranker,
            mock_verifier=mock_verifier,
            factscore_max_claims=factscore_max_claims,
            ragas_max_claims=ragas_max_claims,
            verifier_max_concurrent=verifier_max_concurrent,
            verifier_use_batch=verifier_use_batch,
            verifier_max_batch_size=verifier_max_batch_size,
        )
        per_prompt.append(result_row)
        factscores.append(float(result_row["factscore"]))
        if on_progress is not None:
            on_progress(idx, total, row, result_row)

    aggregate: dict[str, Any] = {
        "system": system_name,
        "n_prompts": len(per_prompt),
        "factscore_mean": sum(factscores) / len(factscores) if factscores else 0.0,
        "mock_generation": all(p.get("mock_generation") for p in per_prompt),
        "mock_verifier": mock_verifier,
        "mock_reranker": system_name == "full_minus_rerank",
        "reference_overlap_mean": (
            sum(float(p.get("reference_overlap", 0.0)) for p in per_prompt) / len(per_prompt)
            if per_prompt
            else 0.0
        ),
        "specificity_ratio_mean": (
            sum(float(p.get("specificity_ratio", 0.0)) for p in per_prompt) / len(per_prompt)
            if per_prompt
            else 0.0
        ),
    }
    if index_meta is not None:
        aggregate["index_mock_embed"] = bool(index_meta.get("mock_embed"))
    return per_prompt, aggregate


def evaluate_one_prompt(
    system_name: str,
    row: dict[str, Any],
    *,
    store: VectorStore | None,
    embedder: Any,
    generator: Any,
    top_k: int,
    verifier: ClaimVerifier,
    graph_retriever: Any | None = None,
    reranker: Reranker | None = None,
    mock_verifier: bool | None = None,
    factscore_max_claims: int = 12,
    ragas_max_claims: int = 8,
    verifier_max_concurrent: int | None = None,
    verifier_use_batch: bool | None = None,
    verifier_max_batch_size: int = 8,
) -> dict[str, Any]:
    gen = generate_for_system(
        system_name,
        row,
        store=store,
        embedder=embedder,
        generator=generator,
        top_k=top_k,
        graph_retriever=graph_retriever,
        reranker=reranker,
    )
    passages = [h.get("text", "") for h in gen.retrieved_chunks]
    fs = compute_factscore(
        gen.abstract_text,
        passages,
        verifier,
        max_claims=factscore_max_claims,
        max_concurrent=verifier_max_concurrent,
        use_batch=verifier_use_batch,
        max_batch_size=verifier_max_batch_size,
    )
    gold_tokens = set(gen.abstract_text.lower().split())
    ref_tokens = set(str(row["gold_abstract"]).lower().split())
    overlap = len(gold_tokens & ref_tokens) / max(len(gold_tokens), 1)
    specificity = len({tok for tok in gold_tokens if len(tok) >= 6}) / max(len(gold_tokens), 1)
    ragas = compute_ragas(
        f"{row['title']}\n{row['outline']}",
        gen.abstract_text,
        passages,
        embedder=embedder,
        verifier=verifier,
        prefer_lexical=isinstance(verifier, MockClaimVerifier),
        factscore_details=fs["details"],
        max_claims=ragas_max_claims,
    )
    return {
        "arxiv_id": row["arxiv_id"],
        "title": row["title"],
        "generated_abstract": gen.abstract_text,
        "gold_abstract": row["gold_abstract"],
        "factscore": fs["factscore"],
        "n_claims": fs["n_claims"],
        "mock_verifier": (
            isinstance(verifier, MockClaimVerifier)
            if mock_verifier is None
            else mock_verifier
        ),
        "mock_generation": gen.mock,
        "mock_reranker": system_name == "full_minus_rerank",
        "retrieved_paper_ids": list(
            dict.fromkeys(h.get("paper_id") for h in gen.retrieved_chunks)
        ),
        "ragas_faithfulness": ragas.get("faithfulness"),
        "ragas_context_relevance": ragas.get("context_relevance"),
        "ragas_backend": ragas.get("ragas_backend"),
        "reference_overlap": round(overlap, 4),
        "specificity_ratio": round(specificity, 4),
    }
