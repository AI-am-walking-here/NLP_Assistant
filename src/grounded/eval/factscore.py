"""M-4.3 — FActScore claim verification via HTTP 70B server (see verifier_client.py)."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SupportLabel = Literal["yes", "no", "partial"]


class ClaimVerifier(Protocol):
    def verify(self, claim: str, passages: list[str]) -> dict[str, Any]: ...

    def verify_batch(
        self,
        items: list[tuple[str, list[str]]],
    ) -> list[dict[str, Any]]: ...


class MockClaimVerifier:
    """Lexical overlap heuristic — dev/CI only; not for headline FActScore."""

    def verify(self, claim: str, passages: list[str]) -> dict[str, Any]:
        claim_tokens = set(re.findall(r"[a-z0-9]{4,}", claim.lower()))
        if not claim_tokens:
            return {"supported": "partial", "reasoning": "empty claim"}
        best = 0.0
        for passage in passages:
            pt = set(re.findall(r"[a-z0-9]{4,}", passage.lower()))
            if not pt:
                continue
            overlap = len(claim_tokens & pt) / len(claim_tokens)
            best = max(best, overlap)
        if best >= 0.35:
            label: SupportLabel = "yes"
        elif best >= 0.15:
            label = "partial"
        else:
            label = "no"
        return {"supported": label, "reasoning": f"token_overlap={best:.2f}"}

    def verify_batch(
        self,
        items: list[tuple[str, list[str]]],
    ) -> list[dict[str, Any]]:
        return [self.verify(claim, passages) for claim, passages in items]


class HttpClaimVerifier:
    """POST /verify on the M-4.2 FastAPI server (vLLM or AWQ 70B backend)."""

    def __init__(
        self,
        base_url: str,
        cache_path: Path | None = None,
        *,
        cache_namespace: str = "unknown",
        cache_read: bool = True,
        cache_write: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_path = cache_path
        self.cache_namespace = cache_namespace
        self.cache_read = cache_read
        self.cache_write = cache_write
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if cache_read and cache_path and cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = row.get("key")
                    if key:
                        self._cache[key] = row.get("result", row)

    @staticmethod
    def _cache_key(claim: str, passages: list[str], namespace: str) -> str:
        import hashlib

        payload = json.dumps(
            {"namespace": namespace, "claim": claim, "passages": passages[:5]},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
        label = raw.get("supported", "no")
        if label not in ("yes", "no", "partial"):
            label = "no"
        return {"supported": label, "reasoning": str(raw.get("reasoning", ""))}

    def verify(self, claim: str, passages: list[str]) -> dict[str, Any]:
        import requests

        key = self._cache_key(claim, passages, self.cache_namespace)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return self._normalize_result(cached)
        resp = requests.post(
            f"{self.base_url}/verify",
            json={"claim": claim, "passages": passages[:5]},
            timeout=120,
        )
        resp.raise_for_status()
        result = self._normalize_result(resp.json())
        self._store_cache_result(key, claim, result)
        return result

    def verify_batch(
        self,
        items: list[tuple[str, list[str]]],
    ) -> list[dict[str, Any]]:
        import requests

        if not items:
            return []

        results: list[dict[str, Any] | None] = [None] * len(items)
        misses: list[tuple[int, str, str, list[str]]] = []
        with self._lock:
            for idx, (claim, passages) in enumerate(items):
                key = self._cache_key(claim, passages, self.cache_namespace)
                cached = self._cache.get(key)
                if cached is not None:
                    results[idx] = self._normalize_result(cached)
                else:
                    misses.append((idx, key, claim, passages))

        if misses:
            payload = {
                "items": [
                    {"claim": claim, "passages": passages[:5]}
                    for _, _, claim, passages in misses
                ]
            }
            try:
                resp = requests.post(
                    f"{self.base_url}/verify_batch",
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 404:
                    raise RuntimeError("verifier batch endpoint unavailable")
                resp.raise_for_status()
                body = resp.json()
                raw_results = body.get("results", body)
                if not isinstance(raw_results, list) or len(raw_results) != len(misses):
                    raise RuntimeError("verifier batch endpoint returned invalid results")
                for (idx, key, claim, _), raw in zip(misses, raw_results, strict=True):
                    result = self._normalize_result(raw)
                    results[idx] = result
                    self._store_cache_result(key, claim, result)
            except Exception as exc:
                logger.warning(
                    "Verifier batch request failed (%s); falling back to sequential /verify.",
                    exc,
                )
                for idx, _, claim, passages in misses:
                    results[idx] = self.verify(claim, passages)

        return [self._normalize_result(result or {}) for result in results]

    def _store_cache_result(
        self,
        key: str,
        claim: str,
        result: dict[str, Any],
    ) -> None:
        with self._lock:
            self._cache[key] = result
            if not (self.cache_write and self.cache_path):
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "key": key,
                            "claim": claim,
                            "cache_namespace": self.cache_namespace,
                            "result": result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def extract_claims(abstract: str, *, max_claims: int = 12) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(abstract.strip()) if len(s.strip()) > 30]
    return sentences[:max_claims]


def score_labels(labels: list[SupportLabel]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for label in labels:
        if label == "yes":
            total += 1.0
        elif label == "partial":
            total += 0.5
    return total / len(labels)


def _normalize_label(label: Any) -> SupportLabel:
    if label not in ("yes", "no", "partial"):
        return "no"
    return label


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _bounded_concurrency(value: int | None) -> int:
    if value is None:
        raw = os.environ.get("EVAL_VERIFIER_CONCURRENCY", "4")
        try:
            value = int(raw)
        except ValueError:
            value = 4
    return max(1, min(int(value), 4))


def _verify_claims(
    claims: list[str],
    evidence_passages: list[str],
    verifier: ClaimVerifier,
    *,
    max_concurrent: int,
    use_batch: bool,
    max_batch_size: int,
) -> list[dict[str, Any]]:
    if not claims:
        return []
    if not use_batch:
        return [verifier.verify(claim, evidence_passages) for claim in claims]

    verifier_batch = getattr(verifier, "verify_batch", None)
    if verifier_batch is None:
        return [verifier.verify(claim, evidence_passages) for claim in claims]

    batch_size = max(1, int(max_batch_size))
    chunks = [
        claims[idx : idx + batch_size]
        for idx in range(0, len(claims), batch_size)
    ]

    def _run(chunk: list[str]) -> list[dict[str, Any]]:
        return verifier_batch([(claim, evidence_passages) for claim in chunk])

    workers = _bounded_concurrency(max_concurrent)
    if workers == 1 or len(chunks) == 1:
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            out.extend(_run(chunk))
        return out

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result_chunk in pool.map(_run, chunks):
            out.extend(result_chunk)
    return out


def faithfulness_from_details(
    details: list[dict[str, Any]],
    *,
    max_claims: int = 8,
) -> float:
    labels = [
        _normalize_label(row.get("supported", "no"))
        for row in details[:max_claims]
    ]
    return score_labels(labels)


def compute_factscore(
    generated_abstract: str,
    evidence_passages: list[str],
    verifier: ClaimVerifier,
    *,
    max_claims: int = 12,
    max_concurrent: int | None = None,
    use_batch: bool | None = None,
    max_batch_size: int = 8,
) -> dict[str, Any]:
    claims = extract_claims(generated_abstract, max_claims=max_claims)
    results = _verify_claims(
        claims,
        evidence_passages,
        verifier,
        max_concurrent=_bounded_concurrency(max_concurrent),
        use_batch=_env_bool("EVAL_VERIFIER_USE_BATCH", True)
        if use_batch is None
        else use_batch,
        max_batch_size=max_batch_size,
    )
    labels = [_normalize_label(result.get("supported", "no")) for result in results]
    details = [
        {"claim": claim, **result}
        for claim, result in zip(claims, results, strict=True)
    ]
    return {
        "factscore": score_labels(labels),
        "n_claims": len(claims),
        "labels": labels,
        "details": details,
    }
