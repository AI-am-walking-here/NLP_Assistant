from __future__ import annotations

import os
import sys
import time
import json
import tempfile
from collections.abc import Callable
from pathlib import Path


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def format_rate(bytes_per_sec: float) -> str:
    return f"{format_bytes(int(bytes_per_sec))}/s"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class StatusReporter:
    """Periodic status lines for long-running steps."""

    def __init__(
        self,
        label: str,
        *,
        min_interval_s: float = 1.0,
        stream=None,
        use_bar: bool | None = None,
    ) -> None:
        self.label = label
        self.min_interval_s = min_interval_s
        self._last_emit = 0.0
        self._started = time.time()
        self._stream = stream or sys.stdout
        progress_mode = os.environ.get("HARDENED_PROGRESS", "").lower()
        if use_bar is None:
            use_bar = progress_mode != "log" and getattr(self._stream, "isatty", lambda: False)()
        self.use_bar = progress_mode != "quiet" and use_bar

    def status(self, message: str, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_emit < self.min_interval_s:
            return
        self._last_emit = now
        print(f"[{self.label}] {message}", file=self._stream, flush=True)

    def done(self, message: str) -> None:
        print(f"[{self.label}] {message}", file=self._stream, flush=True)


def progress_bar(done: int, total: int, *, width: int = 20) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = min(width, max(0, int(width * done / total)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def update_run_progress(
    phase: str,
    *,
    done: int,
    total: int,
    unit: str,
    eta_s: float | None = None,
    detail: str | None = None,
    status: dict | None = None,
) -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    rank = os.environ.get("RANK")
    if local_rank not in (None, "", "-1", "0") or rank not in (None, "", "-1", "0"):
        return
    root = Path(__file__).resolve().parents[2]
    path = root / "runs" / "hardened_rebuild_state" / "RUN_PROGRESS.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"phase_total": 0, "phases": []}
    payload["current_phase"] = phase
    payload["sub_progress"] = {
        "done": done,
        "total": total,
        "unit": unit,
        "eta_s": eta_s,
        "detail": detail,
    }
    if status is not None:
        payload["status"] = status
    payload["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp = Path(fh.name)
        fh.write(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


class ByteProgressReporter(StatusReporter):
    def __init__(self, label: str, total_bytes: int | None = None, **kwargs) -> None:
        super().__init__(label, **kwargs)
        self.total_bytes = total_bytes
        self.done_bytes = 0

    def update(self, chunk_size: int) -> None:
        self.done_bytes += chunk_size
        now = time.time()
        if now - self._last_emit < self.min_interval_s:
            return
        self._last_emit = now
        elapsed = max(now - self._started, 0.001)
        rate = self.done_bytes / elapsed
        if self.total_bytes:
            pct = min(self.done_bytes / self.total_bytes * 100, 100.0)
            remaining = max(self.total_bytes - self.done_bytes, 0)
            eta = remaining / rate if rate > 0 else None
            eta_text = format_duration(eta) if eta is not None else "unknown"
            self.status(
                f"{format_bytes(self.done_bytes)} / {format_bytes(self.total_bytes)} "
                f"({pct:.1f}%) @ {format_rate(rate)} eta={eta_text}",
                force=True,
            )
        else:
            self.status(f"{format_bytes(self.done_bytes)} @ {format_rate(rate)}", force=True)

    def finish(self) -> None:
        elapsed = max(time.time() - self._started, 0.001)
        rate = self.done_bytes / elapsed
        self.done(
            f"Complete: {format_bytes(self.done_bytes)} in {format_duration(elapsed)} "
            f"(avg {format_rate(rate)})"
        )


class CountProgressReporter(StatusReporter):
    def __init__(
        self,
        label: str,
        total: int | None = None,
        unit: str = "items",
        *,
        on_update: Callable[[dict], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(label, **kwargs)
        self.total = total
        self.unit = unit
        self.done_count = 0
        self.on_update = on_update

    def update(self, step: int = 1, *, detail: str | None = None) -> None:
        self.done_count += step
        now = time.time()
        if now - self._last_emit < self.min_interval_s:
            return
        self._last_emit = now
        elapsed = max(now - self._started, 0.001)
        rate = self.done_count / elapsed
        eta = None
        if self.total:
            pct = min(self.done_count / self.total * 100, 100.0)
            remaining = max(self.total - self.done_count, 0)
            eta = remaining / rate if rate > 0 else None
            eta_text = format_duration(eta) if eta is not None else "unknown"
            message = (
                f"{self.done_count}/{self.total} {self.unit} ({pct:.1f}%) "
                f"@ {rate:.1f}/s eta={eta_text}"
            )
        else:
            message = f"{self.done_count} {self.unit} @ {rate:.1f}/s"
        if detail:
            message = f"{message} — {detail}"
        if self.on_update is not None:
            self.on_update(
                {
                    "done": self.done_count,
                    "total": self.total,
                    "unit": self.unit,
                    "rate": rate,
                    "eta_s": eta,
                    "detail": detail,
                }
            )
        if self.use_bar and self.total:
            bar = progress_bar(self.done_count, self.total)
            self._stream.write(f"\r[{self.label}] {bar} {message}")
            self._stream.flush()
            return
        self.status(message, force=True)

    def finish(self, *, detail: str | None = None) -> None:
        elapsed = max(time.time() - self._started, 0.001)
        message = f"Complete: {self.done_count} {self.unit} in {format_duration(elapsed)}"
        if detail:
            message = f"{message} — {detail}"
        if self.use_bar and self.total:
            self._stream.write("\n")
        self.done(message)
