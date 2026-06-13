"""M-4.2 / M-4.3 — load FActScore verifier (HTTP 70B server or explicit mock)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from grounded.eval.factscore import ClaimVerifier, HttpClaimVerifier, MockClaimVerifier

logger = logging.getLogger(__name__)


def check_verifier_server(
    base_url: str,
    *,
    timeout: float = 10.0,
    expect_backend: str | None = None,
) -> dict[str, Any]:
    """GET /health; raise if server is down or backend mismatches."""
    import requests

    url = base_url.rstrip("/")
    resp = requests.get(f"{url}/health", timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"Verifier unhealthy at {url}: {payload}")
    backend = str(payload.get("backend", ""))
    if expect_backend and backend and backend != expect_backend:
        raise RuntimeError(
            f"Verifier at {url} runs backend={backend!r}, expected {expect_backend!r}. "
            "Restart serve_verifier.py with the matching --backend."
        )
    if backend == "mock":
        logger.warning(
            "Verifier server is running mock backend; FActScore will be lexical, not 70B."
        )
    return payload


def load_claim_verifier(
    eval_cfg: Any,
    *,
    mock: bool = False,
    server_url: str | None = None,
    cache_path: Path | None = None,
    use_cache: bool | None = None,
    cache_read: bool = True,
    cache_write: bool = True,
    skip_health_check: bool = False,
    expect_backend: str | None = None,
) -> ClaimVerifier:
    """
    Real eval (mock=False): HTTP client to M-4.2 verifier daemon (vLLM/AWQ 70B).
    Dev/CI (mock=True): lexical MockClaimVerifier only — never used for headline numbers.
    """
    if mock:
        return MockClaimVerifier()

    url = (server_url or eval_cfg.verifier_server_url).rstrip("/")
    if not skip_health_check:
        payload = check_verifier_server(
            url,
            expect_backend=expect_backend or getattr(
                eval_cfg, "verifier_default_backend", None
            ),
        )
        cache_namespace = str(payload.get("backend", expect_backend or "unknown"))
    else:
        expected = expect_backend or getattr(eval_cfg, "verifier_default_backend", None)
        if not expected:
            raise RuntimeError(
                "Skipping verifier health check requires an explicit expected backend."
            )
        cache_namespace = str(expected)
    if use_cache is not None:
        cache_read = use_cache
        cache_write = use_cache
    if not (cache_read or cache_write):
        cache_path = None
    elif cache_path is None:
        from grounded.config import resolve_path

        cache_path = resolve_path(
            getattr(eval_cfg, "verifier_cache_path", "runs/verifier_cache.jsonl")
        )
    return HttpClaimVerifier(
        url,
        cache_path=cache_path,
        cache_namespace=cache_namespace,
        cache_read=cache_read,
        cache_write=cache_write,
    )
