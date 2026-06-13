"""Background GPU capacity polling for parallel shard coordinators."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import ContextManager


def gpu_poll_interval_s() -> float:
    raw = os.environ.get("GPU_POLL_INTERVAL_S", "10").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 10.0


def worker_status_poll_s() -> float:
    """How often the main coordinator re-polls subprocess exit codes."""
    raw = os.environ.get("WORKER_STATUS_POLL_S", "2").strip()
    try:
        return max(0.5, float(raw))
    except ValueError:
        return 2.0


class AsyncGpuPoller(ContextManager["AsyncGpuPoller"]):
    """Fire ``spawn_fn`` on a fixed interval in a daemon thread."""

    def __init__(
        self,
        spawn_fn: Callable[[], None],
        *,
        interval_s: float | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        self._spawn_fn = spawn_fn
        self._interval_s = gpu_poll_interval_s() if interval_s is None else max(1.0, interval_s)
        self._lock = lock or threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def start(self) -> None:
        if self._thread is not None:
            return

        def _run() -> None:
            while not self._stop.is_set():
                with self._lock:
                    self._spawn_fn()
                if self._stop.wait(self._interval_s):
                    break

        self._thread = threading.Thread(
            target=_run,
            name="gpu-capacity-poll",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 2.0)
            self._thread = None

    def __enter__(self) -> AsyncGpuPoller:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
