"""M-4.2 — FActScore verifier HTTP server (mock lexical or vLLM-backed)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from fastapi import HTTPException, Request

from grounded.eval.factscore import MockClaimVerifier, SupportLabel

logger = logging.getLogger(__name__)

SupportResponse = dict[str, Any]

_VLLM_ENGINE: Any = None
_AWQ_MODEL: Any = None
_AWQ_TOKENIZER: Any = None


class VerifyRequest(BaseModel):
    claim: str
    passages: list[str] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    supported: Literal["yes", "no", "partial"]
    reasoning: str = ""


class VerifyBatchRequest(BaseModel):
    items: list[VerifyRequest] = Field(default_factory=list)


class VerifyBatchResponse(BaseModel):
    results: list[VerifyResponse]


def _eval_cfg() -> Any:
    from grounded.config import load_config

    return load_config("eval")


def _verifier_model_path() -> str:
    from grounded.utils.hf_network import require_local_model_path

    eval_cfg = _eval_cfg()
    return require_local_model_path(
        "FActScore verifier (70B AWQ)",
        hub_id=eval_cfg.verifier_model,
        role="verifier_70b_awq",
    )


def _vllm_llm_kwargs() -> dict[str, Any]:
    from grounded.utils.model_paths import is_local_model_path
    from vllm.config import CompilationConfig, CompilationMode

    eval_cfg = _eval_cfg()
    vcfg = eval_cfg.verifier_vllm
    model_path = _verifier_model_path()
    kwargs: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "gpu_memory_utilization": vcfg.gpu_memory_utilization,
        "max_model_len": vcfg.max_model_len,
        "tensor_parallel_size": vcfg.tensor_parallel_size,
        "enforce_eager": vcfg.enforce_eager,
        "compilation_config": CompilationConfig(mode=CompilationMode.NONE),
    }
    if is_local_model_path(model_path) or "awq" in model_path.lower():
        # awq_marlin is faster and slightly leaner when vLLM detects support
        kwargs["quantization"] = "awq_marlin"
    return kwargs


def _python_header_paths() -> list[Path] | None:
    """Include dirs from scripts/setup_python_dev_headers.sh (libpython3.10-dev)."""
    from grounded.config import project_root

    base = project_root() / ".tmp" / "py310dev" / "usr" / "include"
    py_inc = base / "python3.10"
    if not (py_inc / "Python.h").is_file():
        return None
    return [py_inc, base]


def _ensure_python_dev_headers() -> None:
    """Extract libpython3.10-dev into .tmp/py310dev when system python3.10-dev is absent."""
    if _python_header_paths() is not None:
        return
    import subprocess

    from grounded.config import project_root

    script = project_root() / "scripts" / "setup_python_dev_headers.sh"
    if script.is_file():
        subprocess.run(["bash", str(script)], check=False, cwd=project_root())


def _apply_worker_build_env() -> None:
    """Env for vLLM spawn workers: Python headers, ninja on PATH, no FlashInfer JIT."""
    import os
    import sys

    from grounded.config import project_root

    _ensure_python_dev_headers()
    root = project_root()
    tool_bins = [
        str(Path(sys.executable).resolve().parent),
        str(root / ".tmp" / "bin"),
    ]
    root.joinpath(".tmp", "bin").mkdir(parents=True, exist_ok=True)
    ninja_src = Path(sys.executable).resolve().parent / "ninja"
    ninja_link = root / ".tmp" / "bin" / "ninja"
    if ninja_src.is_file() and not ninja_link.exists():
        try:
            ninja_link.symlink_to(ninja_src)
        except OSError:
            pass
    path = os.environ.get("PATH", "")
    prefix = ":".join(b for b in tool_bins if b not in path.split(":"))
    if prefix:
        os.environ["PATH"] = f"{prefix}:{path}"
    # FlashInfer JIT needs ninja in worker PATH; disable sampler JIT for smoke/server.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    paths = _python_header_paths()
    if not paths:
        return
    inc_flags = " ".join(f"-I{p}" for p in paths)
    cpath = ":".join(str(p) for p in paths)
    for key in ("CFLAGS", "CXXFLAGS"):
        os.environ[key] = inc_flags
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        os.environ[key] = cpath


def preload_vllm_engine() -> Any:
    """Load the shared vLLM engine (idempotent)."""
    global _VLLM_ENGINE
    if _VLLM_ENGINE is not None:
        return _VLLM_ENGINE
    _apply_worker_build_env()
    model_path = _verifier_model_path()
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Run: pip install vllm (see environment.yml)."
        ) from exc

    kwargs = _vllm_llm_kwargs()
    logger.info("Loading vLLM verifier from %s (%s)", model_path, kwargs)
    t0 = time.perf_counter()
    _VLLM_ENGINE = LLM(**kwargs)
    logger.info("vLLM verifier ready in %.1fs", time.perf_counter() - t0)
    return _VLLM_ENGINE


def _build_verifier_prompt(claim: str, passages: list[str]) -> str:
    context = "\n\n".join(passages[:3]) or "(no passages provided)"
    return (
        "You are a strict fact-checker for scientific abstracts. "
        "Given PASSAGES and a CLAIM, reply with exactly one word: yes, no, or partial.\n"
        "yes = fully supported; no = contradicted or unsupported; partial = partly supported.\n\n"
        f"PASSAGES:\n{context}\n\nCLAIM: {claim}\n\nAnswer (one word only):"
    )


def _parse_support_label(text: str) -> SupportLabel:
    lowered = text.strip().lower()
    if lowered.startswith("yes") or lowered == "y":
        return "yes"
    if lowered.startswith("no") or lowered == "n":
        return "no"
    if "partial" in lowered or "partly" in lowered:
        return "partial"
    if "yes" in lowered and "no" not in lowered:
        return "yes"
    if "no" in lowered:
        return "no"
    return "partial"


def preload_awq_hf_engine() -> tuple[Any, Any]:
    """Load 70B AWQ via AutoAWQ (2×GPU friendly; no vLLM compile step)."""
    global _AWQ_MODEL, _AWQ_TOKENIZER
    if _AWQ_MODEL is not None and _AWQ_TOKENIZER is not None:
        return _AWQ_MODEL, _AWQ_TOKENIZER
    from grounded.utils.hf_network import local_files_only_kwargs

    model_path = _verifier_model_path()
    local_kw = local_files_only_kwargs()
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "AutoAWQ not installed. Run: pip install autoawq"
        ) from exc

    logger.info("Loading AWQ verifier (HF) from %s", model_path)
    t0 = time.perf_counter()
    _AWQ_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        **local_kw,
    )
    _AWQ_MODEL = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        **local_kw,
    )
    logger.info("AWQ verifier ready in %.1fs", time.perf_counter() - t0)
    return _AWQ_MODEL, _AWQ_TOKENIZER


def _verify_awq_hf(claim: str, passages: list[str]) -> SupportResponse:
    import torch

    model, tokenizer = preload_awq_hf_engine()
    vcfg = _eval_cfg().verifier_vllm
    prompt = _build_verifier_prompt(claim, passages)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=vcfg.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    label = _parse_support_label(text)
    return {"supported": label, "reasoning": f"awq_hf:{text[:120]}"}


def _verify_vllm(claim: str, passages: list[str]) -> SupportResponse:
    from vllm import SamplingParams

    llm = preload_vllm_engine()
    vcfg = _eval_cfg().verifier_vllm
    prompt = _build_verifier_prompt(claim, passages)
    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=vcfg.max_tokens, temperature=0.0),
    )
    text = (outputs[0].outputs[0].text if outputs else "").strip()
    label = _parse_support_label(text)
    return {"supported": label, "reasoning": f"vllm:{text[:120]}"}


def _verify_vllm_batch(items: list[VerifyRequest]) -> list[SupportResponse]:
    from vllm import SamplingParams

    llm = preload_vllm_engine()
    vcfg = _eval_cfg().verifier_vllm
    prompts = [_build_verifier_prompt(item.claim, item.passages) for item in items]
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=vcfg.max_tokens, temperature=0.0),
    )
    results: list[SupportResponse] = []
    for out in outputs:
        text = (out.outputs[0].text if out.outputs else "").strip()
        label = _parse_support_label(text)
        results.append({"supported": label, "reasoning": f"vllm:{text[:120]}"})
    return results


def verify_claim(claim: str, passages: list[str], *, backend: str = "mock") -> SupportResponse:
    """Route a single claim verification to the configured backend."""
    if backend == "mock":
        return MockClaimVerifier().verify(claim, passages)
    if backend in ("awq", "awq_hf"):
        return _verify_awq_hf(claim, passages)
    if backend == "vllm":
        return _verify_vllm(claim, passages)
    raise ValueError(f"Unknown backend: {backend!r}")


def verify_claim_batch(
    items: list[VerifyRequest],
    *,
    backend: str = "mock",
) -> list[SupportResponse]:
    """Route batch verification while preserving single-claim semantics."""
    if backend == "mock":
        verifier = MockClaimVerifier()
        return [verifier.verify(item.claim, item.passages) for item in items]
    if backend in ("awq", "awq_hf"):
        return [_verify_awq_hf(item.claim, item.passages) for item in items]
    if backend == "vllm":
        return _verify_vllm_batch(items)
    raise ValueError(f"Unknown backend: {backend!r}")


class VerifierCache:
    """Append-only JSONL cache keyed by claim + passage hash."""

    def __init__(self, path: Path | None):
        self.path = path
        self._mem: dict[str, SupportResponse] = {}
        if path and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._mem[row["key"]] = row["result"]

    @staticmethod
    def cache_key(claim: str, passages: list[str]) -> str:
        payload = json.dumps(
            {"claim": claim, "passages": passages[:3]},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, claim: str, passages: list[str]) -> SupportResponse | None:
        return self._mem.get(self.cache_key(claim, passages))

    def put(self, claim: str, passages: list[str], result: SupportResponse) -> None:
        key = self.cache_key(claim, passages)
        self._mem[key] = result
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"key": key, "claim": claim, "result": result}, ensure_ascii=False)
                + "\n"
            )


def create_app(
    *,
    backend: str = "mock",
    cache_path: Path | None = None,
    preload: bool = True,
) -> Any:
    """Build a FastAPI app exposing POST /verify."""
    from fastapi import FastAPI

    cache = VerifierCache(cache_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        if backend == "vllm" and preload:
            preload_vllm_engine()
        elif backend in ("awq", "awq_hf") and preload:
            preload_awq_hf_engine()
        yield

    app = FastAPI(title="Grounded FActScore Verifier", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "backend": backend,
            "vllm_loaded": str(_VLLM_ENGINE is not None).lower(),
        }

    @app.post("/verify", response_model=VerifyResponse)
    def verify_endpoint(req: VerifyRequest) -> VerifyResponse:
        cached = cache.get(req.claim, req.passages)
        if cached is not None:
            label = cached.get("supported", "no")
            if label not in ("yes", "no", "partial"):
                label = "no"
            return VerifyResponse(
                supported=label,  # type: ignore[arg-type]
                reasoning=str(cached.get("reasoning", "cache_hit")),
            )
        raw = verify_claim(req.claim, req.passages, backend=backend)
        label = raw.get("supported", "no")
        if label not in ("yes", "no", "partial"):
            label = "no"
        result: SupportResponse = {
            "supported": label,
            "reasoning": str(raw.get("reasoning", "")),
        }
        cache.put(req.claim, req.passages, result)
        return VerifyResponse(
            supported=label,  # type: ignore[arg-type]
            reasoning=result["reasoning"],
        )

    @app.post("/verify_batch", response_model=VerifyBatchResponse)
    async def verify_batch_endpoint(
        req: VerifyBatchRequest,
        request: Request,
    ) -> VerifyBatchResponse:
        raw_body = await request.body()
        if len(raw_body) > 256 * 1024:
            raise HTTPException(status_code=413, detail="verify_batch payload too large")
        max_batch = int(_eval_cfg().verifier_vllm.max_batch_size)
        if not req.items:
            return VerifyBatchResponse(results=[])
        if len(req.items) > max_batch:
            raise HTTPException(status_code=413, detail=f"verify_batch max size is {max_batch}")

        results: list[VerifyResponse | None] = [None] * len(req.items)
        misses: list[tuple[int, VerifyRequest]] = []
        for idx, item in enumerate(req.items):
            claim = item.claim.strip()
            if not claim:
                raise HTTPException(status_code=400, detail="claim must be non-empty")
            if len(claim.encode("utf-8")) > 8192:
                raise HTTPException(status_code=413, detail="claim too large")
            passages = item.passages[:5]
            cached = cache.get(claim, passages)
            if cached is not None:
                label = cached.get("supported", "no")
                if label not in ("yes", "no", "partial"):
                    label = "no"
                results[idx] = VerifyResponse(
                    supported=label,  # type: ignore[arg-type]
                    reasoning=str(cached.get("reasoning", "cache_hit")),
                )
            else:
                misses.append((idx, VerifyRequest(claim=claim, passages=passages)))

        if misses:
            started = time.perf_counter()
            raw_results = verify_claim_batch(
                [item for _, item in misses],
                backend=backend,
            )
            logger.info(
                "verify_batch backend=%s misses=%d elapsed=%.2fs",
                backend,
                len(misses),
                time.perf_counter() - started,
            )
            for (idx, item), raw in zip(misses, raw_results, strict=True):
                label = raw.get("supported", "no")
                if label not in ("yes", "no", "partial"):
                    label = "no"
                result: SupportResponse = {
                    "supported": label,
                    "reasoning": str(raw.get("reasoning", "")),
                }
                cache.put(item.claim, item.passages, result)
                results[idx] = VerifyResponse(
                    supported=label,  # type: ignore[arg-type]
                    reasoning=result["reasoning"],
                )

        if any(result is None for result in results):
            raise HTTPException(status_code=500, detail="verify_batch incomplete results")
        return VerifyBatchResponse(results=[r for r in results if r is not None])

    return app


def run_acceptance_smoke(
    *,
    backend: str = "mock",
    n_claims: int = 10,
    max_seconds: float = 30.0,
    warmup: bool = True,
) -> dict[str, Any]:
    """Acceptance: verify n_claims in-process (mock fast; vllm includes model load)."""
    passages = [
        "We introduce a neural architecture for natural language processing tasks.",
        "Experiments on benchmark datasets show consistent improvements over baselines.",
    ]
    claims = [
        "The proposed method improves NLP benchmark performance.",
        "Results are worse than all prior baselines on every dataset.",
        "The model uses a transformer with attention.",
    ]
    if backend in ("awq", "awq_hf") and warmup:
        preload_awq_hf_engine()
    elif backend == "vllm" and warmup:
        preload_vllm_engine()

    t0 = time.perf_counter()
    results: list[SupportResponse] = []
    for i in range(n_claims):
        claim = claims[i % len(claims)]
        results.append(verify_claim(claim, passages, backend=backend))
    elapsed = time.perf_counter() - t0
    labels = [r.get("supported") for r in results]
    return {
        "backend": backend,
        "n_claims": n_claims,
        "elapsed_sec": round(elapsed, 3),
        "elapsed_per_claim_sec": round(elapsed / max(n_claims, 1), 3),
        "labels_sample": labels[:5],
        "pass": elapsed < max_seconds,
        "model_path": _verifier_model_path() if backend == "vllm" else None,
    }
