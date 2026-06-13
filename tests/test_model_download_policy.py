"""No large Hub downloads unless GROUNDED_ALLOW_MODEL_DOWNLOAD=1."""

from __future__ import annotations

import pytest

from grounded.utils.hf_network import allow_model_download, require_model_download


def test_download_blocked_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GROUNDED_ALLOW_MODEL_DOWNLOAD", raising=False)
    assert allow_model_download() is False
    with pytest.raises(RuntimeError, match="Refusing to download"):
        require_model_download("test")


def test_download_allowed_when_flag_set(monkeypatch) -> None:
    monkeypatch.setenv("GROUNDED_ALLOW_MODEL_DOWNLOAD", "1")
    assert allow_model_download() is True
    require_model_download("test")
