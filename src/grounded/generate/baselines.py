"""M-3.5 — naive RAG baseline (v3.1: no citation resolver)."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from pydantic import BaseModel, Field

from grounded.generate.prompts import (
    format_retrieved_chunks,
    render_abstract_prompt,
    render_sft_prompt,
    sanitize_generated_abstract,
)
from grounded.index.embed import ChunkEmbedder
from grounded.index.vector_store import VectorStore
from grounded.retrieve.rerank import MockReranker, Reranker, rerank_chunks

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    abstract_text: str
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    prompt_system: str = ""
    prompt_user: str = ""
    mock: bool = False


class TextGenerator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class MockGenerator:
    """Retrieval-grounded stub when the 8B model is unavailable."""

    def generate(self, system: str, user: str) -> str:
        del system
        lines = [ln for ln in user.splitlines() if ln.strip()]
        title = next((ln.replace("Title:", "").strip() for ln in lines if ln.startswith("Title:")), "This work")
        return (
            f"We present {title}. "
            "Building on prior work summarized in the retrieved passages, "
            "we study the outlined contributions using standard NLP methodology. "
            "Experiments on representative benchmarks show consistent improvements. "
            "Analysis indicates that the proposed approach captures salient patterns "
            "relevant to the task while remaining efficient at scale."
        )


class HfCausalGenerator:
    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        load_in_4bit: bool = False,
        cuda_device: int = 0,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from grounded.utils.cuda_devices import hf_model_device_map
        from grounded.utils.model_paths import is_local_model_path, resolve_model_path

        resolved = resolve_model_path(model_name, role="generator_8b")
        device_map = hf_model_device_map(cuda_device)
        local = is_local_model_path(resolved)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved, use_fast=True, local_files_only=local
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kw: dict[str, Any] = {"local_files_only": local}
        if torch.cuda.is_available() and load_in_4bit:
            logger.info("Loading generator %s in 4-bit NF4 (eval override)", resolved)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved,
                quantization_config=bnb,
                device_map=device_map,
                **load_kw,
            )
        elif torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved,
                torch_dtype=torch.float16,
                device_map=device_map,
                **load_kw,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved,
                torch_dtype=torch.float32,
                **load_kw,
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_4bit = load_in_4bit

    def generate(self, system: str, user: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system}\n\n{user}\n\nAbstract:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_generator(
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    mock: bool,
    fail_on_error: bool = False,
    load_in_4bit: bool = False,
    cuda_device: int = 0,
) -> TextGenerator:
    from pathlib import Path

    from grounded.utils.model_paths import model_weights_ready, resolve_model_path

    resolved = resolve_model_path(model_name, role="generator_8b")
    weights_ok = Path(resolved).is_dir() and model_weights_ready(Path(resolved))
    if mock or not weights_ok:
        if not mock:
            logger.warning(
                "8B weights not found at %s; using mock generator",
                resolved,
            )
        return MockGenerator()
    from grounded.utils.hf_network import require_model_download

    require_model_download(
        f"generator {model_name}", hub_id=model_name, role="generator_8b"
    )
    try:
        return HfCausalGenerator(
            model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            load_in_4bit=load_in_4bit,
            cuda_device=cuda_device,
        )
    except Exception as exc:
        if fail_on_error:
            raise RuntimeError(f"Failed to load generator {model_name}: {exc}") from exc
        logger.warning("HF generator unavailable (%s); using mock", exc)
        return MockGenerator()


def zero_shot(
    title: str,
    outline: str,
    generator: TextGenerator,
) -> GenerationResult:
    """Title + outline only (no retrieval)."""
    system, user = render_sft_prompt(title, outline)
    abstract = sanitize_generated_abstract(generator.generate(system, user))
    return GenerationResult(
        abstract_text=abstract,
        retrieved_chunks=[],
        prompt_system=system,
        prompt_user=user,
        mock=isinstance(generator, MockGenerator),
    )


def graph_rag(
    title: str,
    outline: str,
    graph_retriever: Any,
    generator: TextGenerator,
    *,
    k: int = 8,
    k_communities: int = 5,
) -> GenerationResult:
    """Retrieve via graph communities, then generate (v3.1: no [CITE])."""
    query = f"{title}\n\n{outline}"
    per_comm = max(1, k // k_communities)
    hits = graph_retriever.search(
        query,
        k_communities=k_communities,
        k_chunks_per_community=per_comm,
    )
    retrieved = format_retrieved_chunks(hits[:k])
    system, user = render_abstract_prompt(title, outline, retrieved)
    abstract = sanitize_generated_abstract(generator.generate(system, user))
    return GenerationResult(
        abstract_text=abstract,
        retrieved_chunks=hits[:k],
        prompt_system=system,
        prompt_user=user,
        mock=isinstance(generator, MockGenerator),
    )


def _merge_candidates(
    *chunk_lists: list[dict[str, Any]],
    max_candidates: int,
    per_list_cap: list[int] | None = None,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for idx, chunks in enumerate(chunk_lists):
        taken = 0
        cap = per_list_cap[idx] if per_list_cap and idx < len(per_list_cap) else None
        for row in chunks:
            if cap is not None and taken >= cap:
                break
            cid = str(row.get("chunk_id", ""))
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            merged.append(row)
            taken += 1
            if len(merged) >= max_candidates:
                return merged
    return merged


def rankrag_rag(
    title: str,
    outline: str,
    store: VectorStore,
    embedder: ChunkEmbedder,
    reranker: Reranker,
    generator: TextGenerator,
    *,
    graph_retriever: Any | None = None,
    k: int = 8,
    n_candidates: int = 30,
    n_vector: int = 15,
    n_graph: int = 15,
) -> GenerationResult:
    """Vector (+ optional graph) pool → rerank → generate."""
    query = f"{title}\n\n{outline}"
    vector_hits = store.search_text(query, embedder, n_vector)
    graph_hits: list[dict[str, Any]] = []
    if graph_retriever is not None:
        per = max(1, n_graph // 5)
        graph_hits = graph_retriever.search(
            query,
            k_communities=5,
            k_chunks_per_community=per,
        )
    graph_cap = max(0, n_candidates - min(len(vector_hits), n_candidates))
    pool = _merge_candidates(
        vector_hits,
        graph_hits,
        max_candidates=n_candidates,
        per_list_cap=[n_candidates, graph_cap],
    )
    hits = rerank_chunks(query, pool, reranker, top_k=k)
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


def naive_rag(
    title: str,
    outline: str,
    store: VectorStore,
    embedder: ChunkEmbedder,
    generator: TextGenerator,
    *,
    k: int = 8,
) -> GenerationResult:
    query = f"{title}\n\n{outline}"
    hits = store.search_text(query, embedder, k)
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


def naive_rag_sft_prompt(
    title: str,
    outline: str,
    store: VectorStore,
    embedder: ChunkEmbedder,
    generator: TextGenerator,
    *,
    k: int = 8,
) -> GenerationResult:
    """Retrieve and generate with retrieval-aligned prompt while preserving SFT-style naming."""
    query = f"{title}\n\n{outline}"
    hits = store.search_text(query, embedder, k)
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
