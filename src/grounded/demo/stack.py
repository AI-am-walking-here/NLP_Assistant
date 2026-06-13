"""Load real retrieval + generation stack for the demo API."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from grounded.config import load_config, resolve_path
from grounded.demo.gpu_layout import DemoGpuLayout, apply_demo_cuda_visible, discover_demo_gpu_layout
from grounded.generate.baselines import GenerationResult, build_generator
from grounded.generate.pipeline import (
    FULL_PIPELINE_SYSTEMS,
    FullPipelineConfig,
    SFT_GENERATOR_SYSTEMS,
    load_text_generator,
    require_latest_adapter,
    resolve_latest_adapter,
)
from grounded.index.embed import ChunkEmbedder
from grounded.index.vector_store import VectorStore, load_chunk_rows
from grounded.retrieve.rerank import Reranker, load_reranker
from grounded.utils.cuda_devices import configure_eval_worker_cuda
from grounded.utils.list_gpus import verifier_reserved_gpus

logger = logging.getLogger(__name__)


def demo_uses_mock() -> bool:
    return os.environ.get("GROUNDED_DEMO_MOCK", "").lower() in ("1", "true", "yes")


def demo_fast_mode() -> bool:
    return os.environ.get("GROUNDED_DEMO_FAST", "").lower() in ("1", "true", "yes")


def _demo_max_new_tokens(retr_cfg) -> int:
    raw = os.environ.get("GROUNDED_DEMO_MAX_TOKENS", "").strip()
    if raw.isdigit():
        return max(64, int(raw))
    if demo_fast_mode():
        return 256
    return int(retr_cfg.generation.max_new_tokens)


def _demo_temperature(retr_cfg) -> float:
    if demo_fast_mode():
        return 0.0
    return float(retr_cfg.generation.temperature)


def _demo_pipeline_cfg(retr_cfg) -> FullPipelineConfig:
    if demo_fast_mode():
        top_k = min(6, int(retr_cfg.top_k))
        return FullPipelineConfig(
            top_k=top_k,
            n_vector=8,
            n_graph=4,
            n_candidates=12,
            k_communities=2,
        )
    return FullPipelineConfig(
        top_k=retr_cfg.top_k,
        n_vector=12,
        n_graph=9,
        n_candidates=20,
        k_communities=3,
    )


def _demo_reserved_gpus() -> set[str]:
    reserved = verifier_reserved_gpus()
    if reserved:
        return reserved
    return {"0", "1"}


def _demo_embed_device(*, mock: bool, layout: DemoGpuLayout | None) -> str:
    explicit = os.environ.get("GROUNDED_DEMO_EMBED_DEVICE", "").strip()
    if explicit:
        return explicit
    if mock:
        return "cpu"
    if layout is not None:
        return layout.embed_device
    return "cpu"


def _load_demo_graph(embedder: ChunkEmbedder, retr_cfg) -> Any:
    from grounded.retrieve.graph import GraphRetriever

    communities_path = resolve_path("data/graph/communities.parquet")
    if not communities_path.is_file():
        raise FileNotFoundError(
            f"Missing {communities_path}. Run: python scripts/build_graph_communities.py"
        )
    chunks_path = resolve_path(retr_cfg.paths.chunks_parquet)
    index_ids = {
        ln.strip()
        for ln in resolve_path("data/splits/index.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    return GraphRetriever.from_parquet(
        communities_path,
        chunks_path,
        embedder,
        paper_filter=index_ids,
    )


def _load_base_generator(stack: dict[str, Any]) -> Any:
    layout: DemoGpuLayout = stack["layout"]
    retr_cfg = stack["retr_cfg"]
    logger.info(
        "Loading base 8B generator (4-bit) on cuda:%d (gpu %s)",
        layout.generator_local,
        layout.physical_for_local(layout.generator_local),
    )
    return build_generator(
        model_name=retr_cfg.generation.base_model,
        max_new_tokens=_demo_max_new_tokens(retr_cfg),
        temperature=_demo_temperature(retr_cfg),
        mock=False,
        fail_on_error=True,
        load_in_4bit=True,
        cuda_device=layout.generator_local,
    )


def _load_reranker_model(stack: dict[str, Any]) -> Reranker:
    layout: DemoGpuLayout = stack["layout"]
    adapter_raw = stack.get("rank_adapter")
    if not adapter_raw:
        from grounded.retrieve.rerank import MockReranker

        logger.warning("No RankRAG adapter; using lexical rerank fallback")
        return MockReranker()
    logger.info(
        "Loading RankRAG reranker (4-bit) on cuda:%d (gpu %s)",
        layout.rankrag_local,
        layout.physical_for_local(layout.rankrag_local),
    )
    return load_reranker(
        Path(adapter_raw),
        mock=False,
        cuda_device=layout.rankrag_local,
    )


def _clear_generator_vram(stack: dict[str, Any]) -> None:
    """Unload generator/SFT only; keep RankRAG when parallel."""
    for key in ("generator", "sft_generator"):
        obj = stack.get(key)
        if obj is not None:
            model = getattr(obj, "model", None)
            if model is not None:
                try:
                    import torch

                    model.cpu()
                except Exception:
                    pass
                del model
            del obj
        stack[key] = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _park_model_holder(holder: Any) -> None:
    model = getattr(holder, "model", None)
    if model is None:
        return
    try:
        import torch

        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unpark_model_holder(holder: Any, cuda_device: int) -> None:
    model = getattr(holder, "model", None)
    if model is None:
        return
    try:
        import torch

        model.to(f"cuda:{cuda_device}")
    except Exception:
        pass


def _unload_reranker(stack: dict[str, Any]) -> None:
    reranker = stack.get("reranker")
    if reranker is not None:
        _park_model_holder(reranker)
        stack["reranker"] = None
    try:
        from grounded.retrieve.rankrag_reranker import _cached_rankrag_model

        _cached_rankrag_model.cache_clear()
    except ImportError:
        pass
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _clear_all_llm_vram(stack: dict[str, Any]) -> None:
    """Unload every demo LLM (sequential single-GPU path)."""
    for key in ("generator", "sft_generator", "reranker"):
        obj = stack.get(key)
        if obj is not None:
            model = getattr(obj, "model", None)
            if model is not None:
                try:
                    import torch

                    model.cpu()
                except Exception:
                    pass
                del model
            del obj
        stack[key] = None
    try:
        from grounded.retrieve.rankrag_reranker import _cached_rankrag_model

        _cached_rankrag_model.cache_clear()
    except ImportError:
        pass
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _parallel_mode(stack: dict[str, Any]) -> bool:
    layout: DemoGpuLayout = stack["layout"]
    return layout.mode == "parallel"


@lru_cache(maxsize=1)
def load_demo_stack() -> dict[str, Any]:
    """BGE index + 8B generator (+ optional SFT/RankRAG adapters)."""
    mock = demo_uses_mock()
    layout: DemoGpuLayout | None = None
    cuda_pin: str | None = None

    if not mock:
        layout = discover_demo_gpu_layout(reserved=_demo_reserved_gpus())
        cuda_pin = apply_demo_cuda_visible(layout)
        configure_eval_worker_cuda()

    embed_device = _demo_embed_device(mock=mock, layout=layout)

    retr_cfg = load_config("retrieval")
    sft_cfg = load_config("sft")
    rr_cfg = load_config("rankrag")

    rpaths = retr_cfg.paths
    meta_path = resolve_path(rpaths.index_meta)
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("mock_embed") and not mock:
        raise RuntimeError(
            "Index has mock_embed=true. Rebuild with scripts/build_index.py "
            "or set GROUNDED_DEMO_MOCK=1"
        )

    chunk_rows = load_chunk_rows(resolve_path(rpaths.chunks_parquet))
    store = VectorStore.load(
        resolve_path(rpaths.faiss_index), chunk_rows, meta_path
    )
    embedder = ChunkEmbedder(
        retr_cfg.embedder,
        device=embed_device,
        normalize=retr_cfg.normalize_embeddings,
        batch_size=retr_cfg.embed_batch_size,
    )

    sft_adapter = resolve_latest_adapter("seg5_sft_train_*")
    rank_adapter = None if mock else require_latest_adapter("seg6_rankrag_*")

    generator = None
    sft_generator = None
    reranker: Reranker | None = None

    if mock:
        generator = build_generator(
            model_name=retr_cfg.generation.base_model,
            max_new_tokens=_demo_max_new_tokens(retr_cfg),
            temperature=_demo_temperature(retr_cfg),
            mock=True,
            fail_on_error=False,
        )
        from grounded.retrieve.rerank import MockReranker

        reranker = MockReranker()
    elif layout is not None:
        stack_stub = {
            "layout": layout,
            "retr_cfg": retr_cfg,
            "rank_adapter": str(rank_adapter) if rank_adapter else None,
        }
        if layout.mode == "parallel":
            reranker = _load_reranker_model(stack_stub)
            generator = _load_base_generator(stack_stub)
        elif demo_fast_mode() or os.environ.get("GROUNDED_DEMO_PRELOAD_GENERATOR", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            generator = _load_base_generator(stack_stub)
            _park_model_holder(generator)

    graph = _load_demo_graph(embedder, retr_cfg)

    return {
        "store": store,
        "embedder": embedder,
        "generator": generator,
        "sft_generator": sft_generator,
        "reranker": reranker,
        "graph": graph,
        "top_k": retr_cfg.top_k,
        "pipeline_cfg": _demo_pipeline_cfg(retr_cfg),
        "mock": mock,
        "demo_fast": demo_fast_mode(),
        "layout": layout,
        "cuda_visible_devices": cuda_pin,
        "embed_device": embed_device,
        "sft_adapter": str(sft_adapter) if sft_adapter else None,
        "rank_adapter": str(rank_adapter) if rank_adapter else None,
        "rankrag_base": rr_cfg.base_model,
        "sft_cfg": sft_cfg,
        "retr_cfg": retr_cfg,
    }


def _ensure_base_generator(stack: dict[str, Any]) -> Any:
    cached = stack.get("generator")
    if cached is not None:
        if not _parallel_mode(stack):
            layout: DemoGpuLayout = stack["layout"]
            _unpark_model_holder(cached, layout.generator_local)
        return cached
    if _parallel_mode(stack):
        stack["generator"] = _load_base_generator(stack)
        return stack["generator"]
    _unload_reranker(stack)
    gen = _load_base_generator(stack)
    stack["generator"] = gen
    return gen


def _ensure_reranker(stack: dict[str, Any]) -> Reranker:
    cached = stack.get("reranker")
    if cached is not None:
        return cached
    if _parallel_mode(stack):
        stack["reranker"] = _load_reranker_model(stack)
        return stack["reranker"]
    for key in ("generator", "sft_generator"):
        obj = stack.get(key)
        if obj is not None:
            _park_model_holder(obj)
    _unload_reranker(stack)
    reranker = _load_reranker_model(stack)
    stack["reranker"] = reranker
    return reranker


def _lazy_sft_generator(stack: dict[str, Any]) -> Any | None:
    if stack.get("mock"):
        return None
    cached = stack.get("sft_generator")
    if cached is not None:
        return cached
    adapter_raw = stack.get("sft_adapter")
    if not adapter_raw:
        return None
    _clear_generator_vram(stack)
    layout: DemoGpuLayout = stack["layout"]
    adapter_path = Path(adapter_raw)
    sft_cfg = stack["sft_cfg"]
    retr_cfg = stack["retr_cfg"]
    logger.info(
        "Loading SFT generator (4-bit) on cuda:%d from %s",
        layout.generator_local,
        adapter_path,
    )
    gen = load_text_generator(
        sft_cfg.base_model,
        adapter_path,
        max_new_tokens=_demo_max_new_tokens(retr_cfg),
        temperature=_demo_temperature(retr_cfg),
        mock=False,
        cuda_device=layout.generator_local,
    )
    stack["sft_generator"] = gen
    return gen


def generator_for_system(stack: dict[str, Any], system_name: str) -> Any:
    """Pick base 8B vs SFT LoRA generator (mirrors run_eval.py)."""
    if stack.get("mock"):
        use_sft = system_name in SFT_GENERATOR_SYSTEMS and system_name != "full_minus_sft"
        if use_sft:
            sft_gen = _lazy_sft_generator(stack)
            if sft_gen is not None:
                return sft_gen
        return stack["generator"]
    use_sft = system_name in SFT_GENERATOR_SYSTEMS and system_name != "full_minus_sft"
    if use_sft:
        sft_gen = _lazy_sft_generator(stack)
        if sft_gen is not None:
            return sft_gen
    return _ensure_base_generator(stack)


def _stage(
    stage_id: str,
    label: str,
    *,
    detail: str = "",
    count: int | None = None,
) -> "PipelineStageRecord":
    from grounded.demo.result import PipelineStageRecord

    return PipelineStageRecord(id=stage_id, label=label, detail=detail, count=count)


def _stages_for_system(system_name: str) -> list["PipelineStageRecord"]:
    """Fallback stage list when using generate_for_system (mock path)."""
    from grounded.generate.pipeline import FULL_PIPELINE_SYSTEMS
    from grounded.eval.runner import RERANK_SYSTEMS

    stages = [_stage("compose", "Compose prompt", detail="title + outline")]
    if system_name in ("zero_shot", "zero_shot_with_sft"):
        stages.append(_stage("generate", "8B generate", detail="no retrieval"))
        return stages
    if system_name == "graph_only":
        stages.extend([
            _stage("graph", "Graph communities"),
            _stage("generate", "8B generate"),
        ])
        return stages
    if system_name in ("naive_rag", "naive_rag_with_sft"):
        stages.extend([
            _stage("vector", "BGE + FAISS"),
            _stage("generate", "8B generate"),
        ])
        return stages
    if system_name in RERANK_SYSTEMS or system_name in FULL_PIPELINE_SYSTEMS:
        stages.extend([
            _stage("vector", "BGE + FAISS"),
            _stage("graph", "Graph communities"),
            _stage("merge", "Merge candidates"),
            _stage("rerank", "RankRAG rerank"),
            _stage("generate", "8B generate"),
        ])
        return stages
    stages.append(_stage("generate", "8B generate"))
    return stages


def _wrap_generation(
    gen: GenerationResult,
    stages: list["PipelineStageRecord"],
    *,
    passages_pre_rerank: list[dict[str, Any]] | None = None,
) -> "DemoRunResult":
    from grounded.demo.result import DemoRunResult

    return DemoRunResult.from_generation(
        gen,
        stages,
        passages_pre_rerank=passages_pre_rerank,
    )


def _generate_from_hits(
    title: str,
    outline: str,
    hits: list[dict[str, Any]],
    generator: Any,
    stages: list["PipelineStageRecord"] | None = None,
    *,
    passages_pre_rerank: list[dict[str, Any]] | None = None,
) -> "DemoRunResult":
    from grounded.generate.baselines import MockGenerator
    from grounded.generate.prompts import (
        format_retrieved_chunks,
        render_abstract_prompt,
        sanitize_generated_abstract,
    )

    retrieved = format_retrieved_chunks(hits)
    system, user = render_abstract_prompt(title, outline, retrieved)
    run_stages = list(stages or [])
    run_stages.append(
        _stage("generate", "8B generate", detail="Llama-3.1-8B", count=len(hits)),
    )
    abstract = sanitize_generated_abstract(generator.generate(system, user))
    gen = GenerationResult(
        abstract_text=abstract,
        retrieved_chunks=hits,
        prompt_system=system,
        prompt_user=user,
        mock=isinstance(generator, MockGenerator),
    )
    return _wrap_generation(
        gen,
        run_stages,
        passages_pre_rerank=passages_pre_rerank,
    )


def demo_generate(
    stack: dict[str, Any],
    system_name: str,
    row: dict[str, Any],
    *,
    top_k: int,
) -> "DemoRunResult":
    """Run one demo request; parallel mode keeps rankrag + generator resident."""
    from grounded.eval.runner import RERANK_SYSTEMS, generate_for_system
    from grounded.generate.baselines import (
        _merge_candidates,
        graph_rag,
        naive_rag,
        zero_shot,
    )
    from grounded.demo.result import PipelineStageRecord
    from grounded.generate.pipeline import config_for_system
    from grounded.retrieve.rerank import MockReranker, rerank_chunks

    if stack.get("mock"):
        gen = generate_for_system(
            system_name,
            row,
            store=stack["store"],
            embedder=stack["embedder"],
            generator=generator_for_system(stack, system_name),
            top_k=top_k,
            graph_retriever=stack["graph"],
            reranker=stack["reranker"],
        )
        return _wrap_generation(gen, _stages_for_system(system_name))

    parallel = _parallel_mode(stack)
    title = row["title"]
    outline = row["outline"]
    store = stack["store"]
    embedder = stack["embedder"]
    graph = stack["graph"]
    cfg = stack["pipeline_cfg"]
    stages: list[PipelineStageRecord] = [
        _stage("compose", "Compose prompt", detail="title + outline"),
    ]

    if system_name in ("zero_shot", "zero_shot_with_sft"):
        gen = zero_shot(title, outline, generator_for_system(stack, system_name))
        stages.append(_stage("generate", "8B generate", detail="no retrieval"))
        return _wrap_generation(gen, stages)

    if system_name == "graph_only":
        stages.append(_stage("graph", "Graph communities"))
        gen = graph_rag(
            title, outline, graph, generator_for_system(stack, system_name), k=top_k
        )
        stages.append(_stage("generate", "8B generate", count=len(gen.retrieved_chunks)))
        return _wrap_generation(gen, stages)

    if system_name in ("naive_rag", "naive_rag_with_sft"):
        query = f"{title}\n\n{outline}"
        vector_hits = store.search_text(query, embedder, top_k)
        stages.append(_stage("vector", "BGE + FAISS", count=len(vector_hits)))
        gen = _generate_from_hits(
            title,
            outline,
            vector_hits[:top_k],
            generator_for_system(stack, system_name),
            stages,
        )
        return gen

    query = f"{title}\n\n{outline}"

    if system_name == "rankrag_only":
        if demo_fast_mode():
            vector_k, graph_kc, graph_kpc, merge_cap = 8, 2, 2, 12
        else:
            vector_k, graph_kc, graph_kpc, merge_cap = 15, 5, 3, 30
        vector_hits = store.search_text(query, embedder, vector_k)
        stages.append(_stage("vector", "BGE + FAISS", count=len(vector_hits)))
        graph_hits = graph.search(
            query, k_communities=graph_kc, k_chunks_per_community=graph_kpc
        )
        stages.append(_stage("graph", "Graph communities", count=len(graph_hits)))
        pool = _merge_candidates(vector_hits, graph_hits, max_candidates=merge_cap)
        stages.append(_stage("merge", "Merge candidates", count=len(pool)))
        reranker = _ensure_reranker(stack)
        hits = rerank_chunks(query, pool, reranker, top_k=top_k)
        stages.append(_stage("rerank", "RankRAG rerank", count=len(hits)))
        if not parallel:
            _unload_reranker(stack)
        return _generate_from_hits(
            title,
            outline,
            hits,
            generator_for_system(stack, system_name),
            stages,
            passages_pre_rerank=pool[:top_k],
        )

    if system_name in FULL_PIPELINE_SYSTEMS:
        pipe_cfg = config_for_system(system_name) or cfg
        vector_hits = store.search_text(query, embedder, pipe_cfg.n_vector)
        stages.append(_stage("vector", "BGE + FAISS", count=len(vector_hits)))
        graph_hits: list[dict[str, Any]] = []
        if pipe_cfg.use_graph:
            per = max(1, pipe_cfg.n_graph // pipe_cfg.k_communities)
            graph_hits = graph.search(
                query,
                k_communities=pipe_cfg.k_communities,
                k_chunks_per_community=per,
            )
            stages.append(_stage("graph", "Graph communities", count=len(graph_hits)))
        graph_cap = max(0, pipe_cfg.n_candidates - min(len(vector_hits), pipe_cfg.n_candidates))
        pool = _merge_candidates(
            vector_hits,
            graph_hits,
            max_candidates=pipe_cfg.n_candidates,
            per_list_cap=[pipe_cfg.n_candidates, graph_cap],
        )
        stages.append(_stage("merge", "Merge candidates", count=len(pool)))
        if pipe_cfg.use_rerank:
            reranker = _ensure_reranker(stack)
            hits = rerank_chunks(query, pool, reranker, top_k=top_k or pipe_cfg.top_k)
            stages.append(_stage("rerank", "RankRAG rerank", count=len(hits)))
            if not parallel:
                _unload_reranker(stack)
        else:
            from grounded.generate.pipeline import _top_by_retrieval_score

            hits = _top_by_retrieval_score(pool, top_k or pipe_cfg.top_k)
            stages.append(_stage("rerank", "Lexical rerank", count=len(hits)))
        return _generate_from_hits(
            title,
            outline,
            hits,
            generator_for_system(stack, system_name),
            stages,
            passages_pre_rerank=pool[: top_k or pipe_cfg.top_k],
        )

    reranker = _ensure_reranker(stack) if system_name in RERANK_SYSTEMS else MockReranker()
    gen = generate_for_system(
        system_name,
        row,
        store=store,
        embedder=embedder,
        generator=generator_for_system(stack, system_name),
        top_k=top_k,
        graph_retriever=graph,
        reranker=reranker,
    )
    return _wrap_generation(gen, _stages_for_system(system_name))
