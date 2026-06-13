"""M-4.2 — verifier server smoke."""

from __future__ import annotations

from pathlib import Path

from grounded.eval.verifier_server import VerifierCache, run_acceptance_smoke


def test_acceptance_smoke_under_30s() -> None:
    report = run_acceptance_smoke(backend="mock", n_claims=10, max_seconds=30.0)
    assert report["pass"] is True
    assert report["n_claims"] == 10
    assert report["backend"] == "mock"


def test_parse_support_label_examples() -> None:
    from grounded.eval.verifier_server import _parse_support_label

    assert _parse_support_label("yes") == "yes"
    assert _parse_support_label("No.") == "no"
    assert _parse_support_label("partially supported") == "partial"


def test_verifier_cache_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    cache = VerifierCache(cache_path)
    cache.put("c1", ["p1"], {"supported": "yes", "reasoning": "ok"})
    loaded = VerifierCache(cache_path)
    assert loaded.get("c1", ["p1"]) == {"supported": "yes", "reasoning": "ok"}


def test_http_verifier_with_testclient(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from grounded.eval.verifier_server import create_app

    cache_path = tmp_path / "v.jsonl"
    client = TestClient(create_app(backend="mock", cache_path=cache_path))
    resp = client.post(
        "/verify",
        json={"claim": "Neural methods improve NLP benchmarks.", "passages": ["NLP neural benchmarks."]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["supported"] in ("yes", "no", "partial")
    assert cache_path.is_file()


def test_http_verifier_batch_with_testclient(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from grounded.eval.verifier_server import create_app

    cache_path = tmp_path / "v.jsonl"
    client = TestClient(create_app(backend="mock", cache_path=cache_path))
    resp = client.post(
        "/verify_batch",
        json={
            "items": [
                {
                    "claim": "Neural methods improve NLP benchmarks.",
                    "passages": ["NLP neural benchmarks."],
                },
                {
                    "claim": "Transformers use attention mechanisms.",
                    "passages": ["Transformers use attention."],
                },
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert all(row["supported"] in ("yes", "no", "partial") for row in body["results"])


def test_http_verifier_batch_rejects_oversized_batch(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from grounded.eval.verifier_server import create_app

    client = TestClient(create_app(backend="mock", cache_path=tmp_path / "v.jsonl"))
    resp = client.post(
        "/verify_batch",
        json={
            "items": [
                {
                    "claim": f"Claim {idx} is long enough to pass validation.",
                    "passages": ["context"],
                }
                for idx in range(32)
            ]
        },
    )
    assert resp.status_code == 413
