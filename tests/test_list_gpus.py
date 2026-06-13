from __future__ import annotations

import grounded.utils.list_gpus as lg


def test_list_gpu_status_marks_selected_by_threshold(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MIN_GPU_FREE_MB", "8000")

    def fake_run(*_args, **_kwargs):
        class Result:
            stdout = "0, 16000\n1, 3000\n2, 12000\n"

        return Result()

    monkeypatch.setattr(lg.subprocess, "run", fake_run)
    rows = lg.list_gpu_status(respect_cuda_visible=False)
    assert [r.index for r in rows if r.selected] == ["0", "2"]
    assert rows[1].selected is False


def test_format_cuda_visible_ignores_parent_cuda_visible_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("SFT_CUDA_DEVICES", raising=False)
    monkeypatch.setenv("MIN_GPU_FREE_MB", "8000")

    def fake_run(*_args, **_kwargs):
        class Result:
            stdout = "0, 16000\n1, 3000\n2, 12000\n"

        return Result()

    monkeypatch.setattr(lg.subprocess, "run", fake_run)
    assert lg.discover_worker_gpus() == ["0", "2"]
    assert lg.format_cuda_visible() == "0,2"


def test_discover_worker_gpus_excludes_verifier(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VERIFIER_CUDA_DEVICES", "0,1")
    monkeypatch.setenv("MIN_GPU_FREE_MB", "8000")

    def fake_run(*_args, **_kwargs):
        class Result:
            stdout = "0, 16000\n1, 16000\n2, 12000\n"

        return Result()

    monkeypatch.setattr(lg.subprocess, "run", fake_run)
    assert lg.discover_worker_gpus(devices_env="EVAL_CUDA_DEVICES", exclude=lg.verifier_reserved_gpus()) == [
        "2"
    ]


def test_discover_worker_gpus_no_fallback_to_reserved_zero(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VERIFIER_CUDA_DEVICES", "0,1")
    monkeypatch.setenv("MIN_GPU_FREE_MB", "8000")

    def fake_run(*_args, **_kwargs):
        class Result:
            stdout = "0, 1000\n1, 1000\n2, 1000\n"

        return Result()

    monkeypatch.setattr(lg.subprocess, "run", fake_run)
    assert lg.discover_worker_gpus(devices_env="EVAL_CUDA_DEVICES", exclude=lg.verifier_reserved_gpus()) == []


def test_gpus_with_capacity_filters_by_free_memory(monkeypatch) -> None:
    monkeypatch.setenv("MIN_GPU_FREE_MB", "8000")

    def fake_run(*_args, **_kwargs):
        class Result:
            stdout = "2, 12000\n3, 3000\n"

        return Result()

    monkeypatch.setattr(lg.subprocess, "run", fake_run)
    assert lg.gpus_with_capacity(["2", "3"]) == ["2"]
    assert lg.gpus_with_capacity(["3"]) == []


def test_discover_cpu_workers_caps_to_shards() -> None:
    assert lg.discover_cpu_workers(workers=32, n_shards=5) == 5
