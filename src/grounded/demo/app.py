"""M-8.3 — demo API: title + outline → abstract + supporting passages."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from grounded.demo.stack import (
    demo_fast_mode,
    demo_generate,
    demo_uses_mock,
    load_demo_stack,
)
from grounded.eval.runner import RERANK_SYSTEMS, SUPPORTED_SYSTEMS
logger = logging.getLogger(__name__)

_stack_ready = False
_stack_error: str | None = None
_preload_task: asyncio.Task[None] | None = None

_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

_SYSTEM_META: dict[str, dict[str, str]] = {
    "zero_shot": {
        "label": "Zero-shot",
        "description": "Base 8B, no retrieval.",
        "group": "baselines",
    },
    "zero_shot_with_sft": {
        "label": "Zero-shot + SFT",
        "description": "SFT LoRA only, no retrieval.",
        "group": "baselines",
    },
    "naive_rag": {
        "label": "Naive RAG",
        "description": "BGE vector retrieval + base 8B.",
        "group": "retrieval",
    },
    "naive_rag_with_sft": {
        "label": "Naive RAG + SFT",
        "description": "Vector retrieval + SFT LoRA writer.",
        "group": "retrieval",
    },
    "graph_only": {
        "label": "Graph only",
        "description": "Pilot Graph RAG communities + base 8B.",
        "group": "retrieval",
    },
    "rankrag_only": {
        "label": "RankRAG only",
        "description": "Vector + RankRAG reranker + base 8B.",
        "group": "retrieval",
    },
    "full": {
        "label": "Full stack",
        "description": "Vector + graph + RankRAG + SFT LoRA.",
        "group": "full",
    },
    "full_minus_sft": {
        "label": "Full pipeline",
        "description": "Vector + graph + RankRAG, base 8B (eval champion).",
        "group": "full",
    },
    "full_minus_graph": {
        "label": "Full pipeline · no graph",
        "description": "Vector + RankRAG + SFT, no graph retrieval.",
        "group": "ablation",
    },
    "full_minus_rerank": {
        "label": "Full pipeline · no rerank",
        "description": "Vector + graph + SFT, lexical rerank fallback.",
        "group": "ablation",
    },
    "naive_rag_sft_prompt": {
        "label": "Vector RAG + prompt tuning",
        "description": "BGE retrieval with SFT-style prompts (research baseline).",
        "group": "other",
    },
}


async def _preload_stack() -> None:
    global _stack_ready, _stack_error
    mode = "MOCK (template generation)" if demo_uses_mock() else "REAL (8B + RankRAG + SFT)"
    fast = " FAST" if demo_fast_mode() else ""
    logger.info("Preloading demo stack [%s%s] — BGE + FAISS + graph + generator...", mode, fast)
    try:
        await asyncio.to_thread(load_demo_stack)
        _stack_ready = True
        _stack_error = None
        stack = load_demo_stack()
        if stack.get("mock"):
            logger.warning(
                "Demo running in MOCK mode — abstracts are template text, not 8B output. "
                "Unset GROUNDED_DEMO_MOCK for live presentation."
            )
        else:
            layout = stack.get("layout")
            logger.info(
                "Demo stack ready (real inference): %s embed=%s rank=%s sft=%s",
                layout.summary() if layout is not None else f"cuda={stack.get('cuda_visible_devices')}",
                stack.get("embed_device"),
                stack.get("rank_adapter"),
                stack.get("sft_adapter"),
            )
    except Exception as exc:
        _stack_ready = False
        _stack_error = str(exc)
        logger.exception("Demo stack preload failed")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _preload_task
    # Yield immediately so Uvicorn binds :8080; /health reports loading until preload finishes.
    _preload_task = asyncio.create_task(_preload_stack())
    yield
    if _preload_task is not None:
        _preload_task.cancel()
        with suppress(asyncio.CancelledError):
            await _preload_task
        _preload_task = None


app = FastAPI(title="NILS-JENS Abstract Demo", version="0.3.1", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    title: str = Field(min_length=3, max_length=500)
    outline: str = Field(min_length=10, max_length=8000)
    top_k: int = Field(default=8, ge=1, le=20)
    system: str = Field(default="full_minus_sft")


class PassageOut(BaseModel):
    paper_id: str
    section_heading: str
    score: float
    text: str


class PipelineStageOut(BaseModel):
    id: str
    label: str
    detail: str = ""
    count: int | None = None


class GenerateResponse(BaseModel):
    abstract: str
    passages: list[PassageOut]
    passages_pre_rerank: list[PassageOut] = Field(default_factory=list)
    stages: list[PipelineStageOut] = Field(default_factory=list)
    mock_generation: bool
    mock_rerank: bool
    backend: str
    system: str


class VerifyRequest(BaseModel):
    abstract: str = Field(min_length=20, max_length=8000)
    passages: list[str] = Field(default_factory=list)


class ClaimDetailOut(BaseModel):
    claim: str
    supported: str
    reasoning: str = ""


class VerifyResponse(BaseModel):
    factscore: float
    n_claims: int
    labels: list[str]
    details: list[ClaimDetailOut]
    verifier: str


class SystemOut(BaseModel):
    id: str
    label: str
    description: str
    group: str
    factscore_mean: float | None = None


def _health_payload() -> dict[str, object]:
    mock = demo_uses_mock()
    info: dict[str, object] = {
        "status": "ok" if _stack_ready else ("degraded" if _stack_error else "loading"),
        "mock_mode": mock,
        "demo_fast": demo_fast_mode(),
        "stack_ready": _stack_ready,
    }
    if _stack_error:
        info["error"] = _stack_error
    if _stack_ready:
        try:
            stack = load_demo_stack()
            info["sft_adapter"] = stack.get("sft_adapter")
            info["rank_adapter"] = stack.get("rank_adapter")
            info["cuda_visible_devices"] = stack.get("cuda_visible_devices")
            info["embed_device"] = stack.get("embed_device")
            layout = stack.get("layout")
            if layout is not None:
                info["gpu_mode"] = layout.mode
                info["gpu_layout"] = layout.summary()
        except Exception as exc:
            info["status"] = "degraded"
            info["error"] = str(exc)
    return info


@app.get("/health")
def health() -> dict[str, object]:
    return _health_payload()


@app.get("/api/systems", response_model=list[SystemOut])
def list_systems() -> list[SystemOut]:
    scores = _load_eval_factscores()
    systems: list[SystemOut] = []
    for system_id in sorted(SUPPORTED_SYSTEMS):
        meta = _SYSTEM_META.get(
            system_id,
            {"label": system_id, "description": "", "group": "other"},
        )
        systems.append(
            SystemOut(
                id=system_id,
                label=meta["label"],
                description=meta["description"],
                group=meta["group"],
                factscore_mean=scores.get(system_id),
            )
        )
    return systems


def _load_eval_factscores() -> dict[str, float]:
    from pathlib import Path

    candidates = [
        Path("/data/team1/mock_presentation/data/eval_grid_jun2026.json"),
        Path("/data/team1/llm-assistant-final/data/eval_set/grid_runs.json"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("systems") or payload.get("runs") or []
            return {
                str(row["system"]): float(row["factscore_mean"])
                for row in rows
                if row.get("system") and row.get("factscore_mean") is not None
            }
        except Exception:
            logger.exception("Failed to load eval scores from %s", path)
    return {}


def _hits_to_passages(hits: list[dict[str, object]]) -> list[PassageOut]:
    return [
        PassageOut(
            paper_id=str(h.get("paper_id", "")),
            section_heading=str(h.get("section_heading", "")),
            score=float(h.get("rerank_score", h.get("score", 0.0))),
            text=str(h.get("text", ""))[:1200],
        )
        for h in hits
    ]


def _generate_sync(body: GenerateRequest) -> GenerateResponse:
    if not _stack_ready:
        detail = _stack_error or "Demo stack is still loading; retry in a minute."
        raise HTTPException(status_code=503, detail=detail)

    stack = load_demo_stack()
    row = {"title": body.title, "outline": body.outline}
    k = body.top_k or stack["top_k"]
    result = demo_generate(stack, body.system, row, top_k=k)

    passages = _hits_to_passages(result.retrieved_chunks)
    pre_rerank = _hits_to_passages(result.passages_pre_rerank)
    stages = [
        PipelineStageOut(
            id=s.id,
            label=s.label,
            detail=s.detail,
            count=s.count,
        )
        for s in result.stages
    ]
    mock_rerank = body.system == "full_minus_rerank" or (
        body.system in RERANK_SYSTEMS and bool(stack.get("mock"))
    )

    return GenerateResponse(
        abstract=result.abstract_text,
        passages=passages,
        passages_pre_rerank=pre_rerank,
        stages=stages,
        mock_generation=result.mock,
        mock_rerank=mock_rerank,
        backend=body.system,
        system=body.system,
    )


def _verify_sync(body: VerifyRequest) -> VerifyResponse:
    import os

    from grounded.eval.factscore import (
        HttpClaimVerifier,
        MockClaimVerifier,
        compute_factscore,
    )

    verifier_url = os.environ.get("GROUNDED_VERIFIER_URL", "http://127.0.0.1:8765").rstrip(
        "/"
    )
    verifier_name = "mock"
    verifier: HttpClaimVerifier | MockClaimVerifier
    try:
        import requests

        health = requests.get(f"{verifier_url}/health", timeout=3)
        if health.ok:
            verifier = HttpClaimVerifier(verifier_url, cache_read=True, cache_write=False)
            verifier_name = "http"
        else:
            verifier = MockClaimVerifier()
    except Exception:
        verifier = MockClaimVerifier()

    evidence = [p.strip() for p in body.passages if p.strip()][:8]
    payload = compute_factscore(body.abstract, evidence, verifier, max_claims=10)
    details = [
        ClaimDetailOut(
            claim=str(row.get("claim", "")),
            supported=str(row.get("supported", "no")),
            reasoning=str(row.get("reasoning", "")),
        )
        for row in payload.get("details", [])
    ]
    return VerifyResponse(
        factscore=float(payload.get("factscore", 0.0)),
        n_claims=int(payload.get("n_claims", 0)),
        labels=[str(label) for label in payload.get("labels", [])],
        details=details,
        verifier=verifier_name,
    )


@app.post("/api/verify", response_model=VerifyResponse)
async def verify(body: VerifyRequest) -> VerifyResponse:
    try:
        return await asyncio.to_thread(_verify_sync, body)
    except Exception as exc:
        logger.exception("Verification failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    if body.system not in SUPPORTED_SYSTEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown system {body.system!r}. Choose from {sorted(SUPPORTED_SYSTEMS)}",
        )
    try:
        return await asyncio.to_thread(_generate_sync, body)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Generation failed for system=%s", body.system)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
