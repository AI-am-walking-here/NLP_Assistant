from __future__ import annotations

import threading
import time

from grounded.utils.gpu_poll import AsyncGpuPoller, gpu_poll_interval_s


def test_async_gpu_poller_invokes_spawn_on_interval() -> None:
    calls: list[int] = []
    lock = threading.Lock()
    saw_second = threading.Event()

    def spawn() -> None:
        calls.append(1)
        if len(calls) >= 2:
            saw_second.set()

    with AsyncGpuPoller(spawn, interval_s=0.05, lock=lock):
        assert saw_second.wait(timeout=2.0)
    assert len(calls) >= 2


def test_gpu_poll_interval_s_defaults_to_ten(monkeypatch) -> None:
    monkeypatch.delenv("GPU_POLL_INTERVAL_S", raising=False)
    assert gpu_poll_interval_s() == 10.0
