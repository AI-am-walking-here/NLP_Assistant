"""Tests for M-4.2 HTTP verifier client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grounded.eval.factscore import HttpClaimVerifier, MockClaimVerifier
from grounded.eval.verifier_client import check_verifier_server, load_claim_verifier


def test_load_claim_verifier_mock() -> None:
    v = load_claim_verifier(MagicMock(), mock=True)
    assert isinstance(v, MockClaimVerifier)


@patch("requests.get")
def test_check_verifier_server_ok(mock_get: MagicMock) -> None:
    mock_get.return_value.json.return_value = {"status": "ok", "backend": "vllm"}
    mock_get.return_value.raise_for_status = MagicMock()
    out = check_verifier_server("http://127.0.0.1:8765", expect_backend="vllm")
    assert out["backend"] == "vllm"


@patch("requests.get")
def test_check_verifier_server_backend_mismatch(mock_get: MagicMock) -> None:
    mock_get.return_value.json.return_value = {"status": "ok", "backend": "mock"}
    mock_get.return_value.raise_for_status = MagicMock()
    with pytest.raises(RuntimeError, match="expected"):
        check_verifier_server("http://127.0.0.1:8765", expect_backend="vllm")


def test_http_claim_verifier_cache_key_stable() -> None:
    k1 = HttpClaimVerifier._cache_key("claim a", ["p1", "p2"], "vllm")
    k2 = HttpClaimVerifier._cache_key("claim a", ["p1", "p2"], "vllm")
    assert k1 == k2
    assert len(k1) == 64


def test_http_claim_verifier_cache_key_depends_on_namespace() -> None:
    k1 = HttpClaimVerifier._cache_key("claim a", ["p1", "p2"], "vllm")
    k2 = HttpClaimVerifier._cache_key("claim a", ["p1", "p2"], "awq")
    assert k1 != k2


def test_http_claim_verifier_read_cache_without_write(tmp_path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    key = HttpClaimVerifier._cache_key("claim a", ["p1"], "vllm")
    cache_path.write_text(
        (
            '{"key": "%s", "result": {"supported": "yes", "reasoning": "cached"}}\n'
            % key
        ),
        encoding="utf-8",
    )
    verifier = HttpClaimVerifier(
        "http://127.0.0.1:1",
        cache_path=cache_path,
        cache_namespace="vllm",
        cache_read=True,
        cache_write=False,
    )
    before = cache_path.read_text(encoding="utf-8")
    assert verifier.verify("claim a", ["p1"])["supported"] == "yes"
    assert cache_path.read_text(encoding="utf-8") == before
