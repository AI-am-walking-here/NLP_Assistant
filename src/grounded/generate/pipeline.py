"""M-5.3 / M-7.1 — load adapters and run the full retrieval + rerank + generate pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from grounded.generate.baselines import (
    GenerationResult,
    TextGenerator,
    _merge_candidates,
)
from grounded.generate.prompts import (
    format_retrieved_chunks,
    render_abstract_prompt,
    sanitize_generated_abstract,
)
from grounded.index.embed import ChunkEmbedder
from grounded.index.vector_store import VectorStore
from grounded.retrieve.rerank import Reranker, rerank_chunks

logger = logging.getLogger(__name__)


def load_text_generator(
    base_model: str,
    adapter_path: Path | None = None,
    *,
    max_new_tokens: int = 384,
    temperature: float = 0.7,
    mock: bool = False,
    cuda_device: int = 0,
) -> TextGenerator:
    if mock or adapter_path is None:
        if adapter_path is None and not mock:
            logger.warning("No adapter_path; using mock generator")
        from grounded.generate.baselines import MockGenerator

        return MockGenerator()
    from grounded.utils.hf_network import require_model_download

    from grounded.utils.model_paths import is_local_model_path, resolve_model_path

    resolved = resolve_model_path(base_model, role="generator_8b")
    require_model_download(
        f"SFT adapter on {base_model}",
        hub_id=base_model,
        role="generator_8b",
    )
    local = is_local_model_path(resolved)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from grounded.generate.baselines import HfCausalGenerator
        from grounded.utils.cuda_devices import hf_model_device_map

        device_map = hf_model_device_map(cuda_device)
        tokenizer = AutoTokenizer.from_pretrained(
            resolved, use_fast=True, local_files_only=local
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Match QLoRA training (sft_train.py): adapter was saved from a 4-bit base.
        load_kw: dict[str, Any] = {"local_files_only": local}
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                resolved,
                quantization_config=bnb_config,
                device_map=device_map,
                **load_kw,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                resolved,
                torch_dtype=torch.float32,
                **load_kw,
            )
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)

        class _PeftGenerator:
            def __init__(self) -> None:
                self.model = model
                self.tokenizer = tokenizer
                self.max_new_tokens = max_new_tokens
                self.temperature = temperature

            def generate(self, system: str, user: str) -> str:
                gen = HfCausalGenerator.__new__(HfCausalGenerator)
                gen.model = self.model
                gen.tokenizer = self.tokenizer
                gen.max_new_tokens = self.max_new_tokens
                gen.temperature = self.temperature
                return HfCausalGenerator.generate(gen, system, user)

        return _PeftGenerator()
    except ImportError as exc:
        raise ImportError(
            "Install train extras: pip install -e '.[train]' (inside project .venv)"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SFT adapter from {adapter_path} on {resolved}: {exc}"
        ) from exc


class IncompleteSftTrainingError(RuntimeError):
    """Newest SFT run directory exists but has no saved ``adapter/``."""


@dataclass(frozen=True)
class AdapterResolution:
    adapter_dir: Path
    run_dir: Path | None
    source: Literal["explicit", "latest_complete"]


def _adapter_usable(adapter_dir: Path) -> bool:
    return adapter_dir.is_dir() and any(adapter_dir.iterdir())


def resolve_sft_adapter(
    run_glob: str = "seg5_sft_train_*",
    *,
    runs_dir: Path | None = None,
    explicit: Path | None = None,
    strict_latest: bool = False,
) -> AdapterResolution | None:
    """
    Resolve a LoRA adapter directory.

    When ``strict_latest`` is True and the newest matching run has no usable
    ``adapter/``, raises :class:`IncompleteSftTrainingError` instead of falling
    back to an older run (prevents silent use of stale weights after a failed train).
    """
    if explicit is not None:
        if not _adapter_usable(explicit):
            raise FileNotFoundError(f"SFT adapter path is missing or empty: {explicit}")
        run_dir = explicit.parent if explicit.name == "adapter" else None
        return AdapterResolution(
            adapter_dir=explicit,
            run_dir=run_dir,
            source="explicit",
        )

    root = runs_dir or Path(__file__).resolve().parents[3] / "runs"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob(run_glob), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None

    newest = candidates[0]
    if strict_latest and not _adapter_usable(newest / "adapter"):
        raise IncompleteSftTrainingError(
            f"Newest SFT run {newest.name} has no saved adapter/ "
            "(training incomplete). Finish scripts/sft_train.py or pass --adapter-path."
        )

    for run_dir in candidates:
        adapter = run_dir / "adapter"
        if _adapter_usable(adapter):
            return AdapterResolution(
                adapter_dir=adapter,
                run_dir=run_dir,
                source="latest_complete",
            )
    return None


def require_sft_adapter(
    *,
    runs_dir: Path | None = None,
    explicit: Path | None = None,
    run_glob: str = "seg5_sft_train_*",
) -> AdapterResolution:
    """Strict adapter resolution for eval (no silent fallback past incomplete runs)."""
    res = resolve_sft_adapter(
        run_glob,
        runs_dir=runs_dir,
        explicit=explicit,
        strict_latest=True,
    )
    if res is None:
        raise IncompleteSftTrainingError(
            f"No usable SFT adapter under {run_glob!r}. "
            "Run scripts/sft_train.py or pass --adapter-path."
        )
    return res


def resolve_latest_adapter(
    run_glob: str,
    *,
    runs_dir: Path | None = None,
) -> Path | None:
    """Return the newest usable ``runs/<glob>/adapter/`` (may skip incomplete newest run)."""
    res = resolve_sft_adapter(run_glob, runs_dir=runs_dir, strict_latest=False)
    return res.adapter_dir if res else None


def require_latest_adapter(
    run_glob: str,
    *,
    runs_dir: Path | None = None,
) -> Path:
    """Return newest usable adapter and fail if the latest matching run is incomplete."""
    res = resolve_sft_adapter(run_glob, runs_dir=runs_dir, strict_latest=True)
    if res is None:
        raise FileNotFoundError(f"No usable adapter under {run_glob!r}.")
    return res.adapter_dir


@dataclass(frozen=True)
class FullPipelineConfig:
    """Feature flags for M-7.1 / ablations (v3.1: no citation resolver)."""

    use_graph: bool = True
    use_rerank: bool = True
    top_k: int = 8
    n_vector: int = 15
    n_graph: int = 15
    n_candidates: int = 30
    k_communities: int = 5


def config_for_system(system_name: str) -> FullPipelineConfig | None:
    """Map eval system name → pipeline flags."""
    presets: dict[str, FullPipelineConfig] = {
        "full": FullPipelineConfig(),
        "full_minus_graph": FullPipelineConfig(use_graph=False),
        "full_minus_rerank": FullPipelineConfig(use_rerank=False),
        "full_minus_sft": FullPipelineConfig(),
    }
    return presets.get(system_name)


FULL_PIPELINE_SYSTEMS = frozenset(
    {
        "full",
        "full_minus_graph",
        "full_minus_rerank",
        "full_minus_sft",
    }
)

SFT_GENERATOR_SYSTEMS = frozenset(
    {
        "full",
        "full_minus_graph",
        "full_minus_rerank",
        "naive_rag_with_sft",
        "naive_rag_sft_prompt",
        "zero_shot_with_sft",
    }
)


def _top_by_retrieval_score(chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    vector = [row for row in chunks if "community_id" not in row]
    graph = [row for row in chunks if "community_id" in row]
    vector_sorted = sorted(vector, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    graph_sorted = sorted(
        graph,
        key=lambda row: (
            float(row.get("community_score", 0.0)),
            float(row.get("score", 0.0)),
        ),
        reverse=True,
    )
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in vector_sorted + graph_sorted:
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id and chunk_id in seen:
            continue
        if chunk_id:
            seen.add(chunk_id)
        merged.append(row)
        if len(merged) >= top_k:
            break
    return merged


def full_pipeline(
    title: str,
    outline: str,
    store: VectorStore,
    embedder: ChunkEmbedder,
    reranker: Reranker,
    generator: TextGenerator,
    *,
    graph_retriever: Any | None = None,
    config: FullPipelineConfig | None = None,
) -> GenerationResult:
    """
    M-7.1 — vector (+ optional graph) retrieve → merge → rerank → generate.

    v3.1: no ``[CITE]`` markers or citation_resolver post-processing.
    """
    from grounded.generate.baselines import MockGenerator

    cfg = config or FullPipelineConfig()
    query = f"{title}\n\n{outline}"

    vector_hits = store.search_text(query, embedder, cfg.n_vector)
    graph_hits: list[dict[str, Any]] = []
    if cfg.use_graph and graph_retriever is not None:
        per = max(1, cfg.n_graph // cfg.k_communities)
        graph_hits = graph_retriever.search(
            query,
            k_communities=cfg.k_communities,
            k_chunks_per_community=per,
        )
    elif cfg.use_graph:
        logger.warning("full pipeline requested graph retrieval but no GraphRetriever was loaded")

    graph_cap = max(0, cfg.n_candidates - min(len(vector_hits), cfg.n_candidates))
    pool = _merge_candidates(
        vector_hits,
        graph_hits,
        max_candidates=cfg.n_candidates,
        per_list_cap=[cfg.n_candidates, graph_cap],
    )
    logger.info(
        "full_pipeline pool: vector=%d graph=%d pooled=%d rerank=%s",
        len(vector_hits),
        len(graph_hits),
        len(pool),
        cfg.use_rerank,
    )
    if cfg.use_rerank:
        hits = rerank_chunks(query, pool, reranker, top_k=cfg.top_k)
    else:
        hits = _top_by_retrieval_score(pool, cfg.top_k)

    retrieved = format_retrieved_chunks(hits)
    system, user = render_abstract_prompt(title, outline, retrieved)
    abstract = sanitize_generated_abstract(generator.generate(system, user))
    return GenerationResult(
        abstract_text=abstract,
        retrieved_chunks=hits,
        prompt_system=system,
        prompt_user=user,
        mock=isinstance(generator, MockGenerator),
    )
