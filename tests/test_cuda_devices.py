from __future__ import annotations

import pytest

from grounded.utils import cuda_devices as cd


def test_local_cuda_device_map_single_visible_gpu(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert cd.local_cuda_device_map() == {"": 0}


def test_local_cuda_device_map_pins_multi_visible_to_primary(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert cd.local_cuda_device_map() == {"": 0}


def test_assert_eval_worker_rejects_unset_cuda_visible(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VERIFIER_CUDA_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        cd.assert_eval_worker_cuda_isolation()


def test_assert_eval_worker_rejects_verifier_overlap(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("VERIFIER_CUDA_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="overlaps verifier"):
        cd.assert_eval_worker_cuda_isolation()


def test_assert_eval_worker_allows_disjoint_gpu(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("VERIFIER_CUDA_DEVICES", "0,1")
    cd.assert_eval_worker_cuda_isolation()
