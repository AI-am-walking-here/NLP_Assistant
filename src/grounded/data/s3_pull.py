from __future__ import annotations

import json
import os
import signal
import threading
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from concurrent.futures import thread as _thread_module
from concurrent.futures.thread import _worker
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from grounded.config import S3TransferConfig, append_log, write_json


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """Exit promptly on Ctrl+C even if worker threads are mid-download."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            thread = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
            )
            thread.daemon = True
            thread.start()
            self._threads.add(thread)
            _thread_module._threads_queues[thread] = self._work_queue


@dataclass(frozen=True)
class TarballPipelineOptions:
    target_ids: set[str]
    output_dir: Path
    extracted_ledger_path: Path
    skip_extensions: set[str]
    extract_after_download: bool = True
    delete_tarball_after_extract: bool = True


@dataclass
class CostTracker:
    egress_per_gb_usd: float
    get_request_usd: float
    hard_cap_usd: float
    gb_downloaded: float = 0.0
    requests: int = 0
    total_cost_usd: float = 0.0
    completed_keys: list[str] = field(default_factory=list)
    failed_keys: list[str] = field(default_factory=list)

    def estimate_cost(self, size_bytes: int) -> float:
        gb = size_bytes / (1024**3)
        return gb * self.egress_per_gb_usd + self.get_request_usd

    def would_exceed_cap(self, size_bytes: int) -> bool:
        return (self.total_cost_usd + self.estimate_cost(size_bytes)) > self.hard_cap_usd

    def record(self, key: str, size_bytes: int) -> None:
        gb = size_bytes / (1024**3)
        self.gb_downloaded += gb
        self.requests += 1
        self.total_cost_usd += gb * self.egress_per_gb_usd + self.get_request_usd
        self.completed_keys.append(key)

    def to_dict(self) -> dict:
        return {
            "gb_downloaded": round(self.gb_downloaded, 4),
            "requests": self.requests,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "hard_cap_usd": self.hard_cap_usd,
            "num_completed": len(self.completed_keys),
            "num_failed": len(self.failed_keys),
        }


def load_ledger(ledger_path: Path) -> set[str]:
    if not ledger_path.exists():
        return set()
    return {
        line.strip()
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_ledger(ledger_path: Path, key: str) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{key}\n")
        handle.flush()
        os.fsync(handle.fileno())


def tarball_local_path(raw_dir: Path, s3_key: str) -> Path:
    return raw_dir / Path(s3_key).name


def _describe_download_error(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = error.get("Code", "Unknown")
        message = error.get("Message", str(exc))
        return f"{code}: {message}"
    return f"{type(exc).__name__}: {exc}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_progress_line(
    *,
    total: int,
    already_done: int,
    completed: int,
    failed: int,
    skipped: int,
    in_flight: int,
    gb_downloaded: float,
    cost_usd: float,
    hard_cap_usd: float,
    started_at: float,
) -> str:
    accounted = already_done + completed + failed + skipped
    percent = (accounted / total * 100) if total else 100.0
    elapsed = max(time.time() - started_at, 0.001)
    tarballs_per_min = completed / elapsed * 60
    mb_per_sec = gb_downloaded * 1024 / elapsed
    remaining = max(total - accounted - in_flight, 0)
    eta = None if tarballs_per_min <= 0 else remaining / (tarballs_per_min / 60)
    return (
        f"[download] {accounted}/{total} ({percent:.1f}%) "
        f"done={already_done + completed} this_run={completed} "
        f"failed={failed} skipped={skipped} active={in_flight} "
        f"{gb_downloaded:.2f} GB ${cost_usd:.2f}/${hard_cap_usd:.2f} "
        f"rate={tarballs_per_min:.1f} tarballs/min {mb_per_sec:.1f} MB/s "
        f"eta={_format_duration(eta)}"
    )


def _megabytes(n: int) -> int:
    return n * 1024 * 1024


def build_transfer_config(transfer: S3TransferConfig) -> TransferConfig:
    return TransferConfig(
        multipart_threshold=_megabytes(transfer.multipart_threshold_mb),
        multipart_chunksize=_megabytes(transfer.multipart_chunksize_mb),
        max_concurrency=transfer.max_concurrency,
        use_threads=True,
    )


def build_botocore_config(transfer: S3TransferConfig) -> BotoConfig:
    return BotoConfig(max_pool_connections=transfer.max_pool_connections)


class _StopTransferCallback:
    def __init__(self, stop: threading.Event | None) -> None:
        self._stop = stop

    def __call__(self, bytes_amount: int) -> None:
        if self._stop and self._stop.is_set():
            raise KeyboardInterrupt


def _head_object_size(client, bucket: str, key: str, request_payer: str) -> int | None:
    try:
        return int(
            client.head_object(Bucket=bucket, Key=key, RequestPayer=request_payer)["ContentLength"]
        )
    except (ClientError, BotoCoreError):
        return None


def _size_is_valid(actual: int, manifest_size: int | None, s3_size: int | None) -> bool:
    """Manifest sizes can lag S3 slightly; trust HeadObject or a small tolerance."""
    if s3_size is not None and actual == s3_size:
        return True
    if manifest_size is None:
        return actual > 0
    tolerance = max(2 * 1024 * 1024, int(manifest_size * 0.01))
    return abs(actual - manifest_size) <= tolerance


def _cleanup_partial_downloads(raw_dir: Path) -> None:
    if not raw_dir.exists():
        return
    for path in raw_dir.iterdir():
        name = path.name
        if ".part" in name or name.endswith(".part"):
            try:
                path.unlink()
            except OSError:
                pass


def _download_one(
    client,
    bucket: str,
    key: str,
    dest: Path,
    request_payer: str,
    max_retries: int,
    transfer_config: TransferConfig,
    expected_size: int | None = None,
    stop: threading.Event | None = None,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if stop and stop.is_set():
        raise KeyboardInterrupt

    last_error: Exception | None = None
    s3_size = _head_object_size(client, bucket, key, request_payer)

    if dest.exists() and dest.stat().st_size > 0:
        size = dest.stat().st_size
        if _size_is_valid(size, expected_size, s3_size):
            return size

    part = dest.with_name(f"{dest.name}.part")
    extra_args = {"RequestPayer": request_payer}
    callback = _StopTransferCallback(stop)
    for attempt in range(max_retries):
        if stop and stop.is_set():
            raise KeyboardInterrupt
        try:
            if part.exists():
                part.unlink()
            client.download_file(
                bucket,
                key,
                str(part),
                ExtraArgs=extra_args,
                Config=transfer_config,
                Callback=callback,
            )
            size = part.stat().st_size
            if not _size_is_valid(size, expected_size, s3_size):
                raise RuntimeError(
                    f"Downloaded size mismatch for {key}: got {size}, "
                    f"manifest={expected_size}, s3={s3_size}"
                )
            part.replace(dest)
            return size
        except KeyboardInterrupt:
            if part.exists():
                part.unlink()
            raise
        except (ClientError, BotoCoreError, OSError, RuntimeError) as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            if stop and stop.is_set():
                raise KeyboardInterrupt from exc
            time.sleep(2**attempt)
    if part.exists():
        part.unlink()
    detail = _describe_download_error(last_error) if last_error else "unknown error"
    raise RuntimeError(f"Failed to download {key}: {detail}") from last_error


class S3TarballDownloader:
    def __init__(
        self,
        bucket: str,
        region: str,
        request_payer: str,
        raw_dir: Path,
        ledger_path: Path,
        cost: CostTracker,
        transfer: S3TransferConfig,
        max_workers: int = 4,
        max_retries: int = 3,
    ):
        self.bucket = bucket
        self.request_payer = request_payer
        self.raw_dir = raw_dir
        self.ledger_path = ledger_path
        self.cost = cost
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._transfer_config = build_transfer_config(transfer)
        self.client = boto3.client(
            "s3",
            region_name=region,
            config=build_botocore_config(transfer),
        )

    def download_tarballs(
        self,
        keys: list[str],
        key_sizes: dict[str, int] | None = None,
        run_dir: Path | None = None,
        progress_interval_s: float = 5.0,
        pipeline: TarballPipelineOptions | None = None,
    ) -> CostTracker:
        key_sizes = key_sizes or {}
        ledger = load_ledger(self.ledger_path)
        extracted_ledger = (
            load_ledger(pipeline.extracted_ledger_path)
            if pipeline and pipeline.extract_after_download
            else set()
        )
        pending: list[str] = []
        already_done = 0
        skipped = 0
        pipeline_mode = pipeline is not None and pipeline.extract_after_download

        for key in keys:
            if pipeline_mode and key in extracted_ledger:
                already_done += 1
                continue

            dest = tarball_local_path(self.raw_dir, key)
            expected_size = key_sizes.get(key)
            has_valid_local = dest.exists() and dest.stat().st_size > 0 and _size_is_valid(
                dest.stat().st_size, expected_size, None
            )

            if not pipeline_mode:
                if key in ledger and has_valid_local:
                    already_done += 1
                    continue
                if has_valid_local:
                    append_ledger(self.ledger_path, key)
                    already_done += 1
                    continue
                if key in ledger and run_dir:
                    append_log(run_dir, f"Redownloading {key}: ledgered file missing locally")
                elif dest.exists() and dest.stat().st_size > 0 and run_dir:
                    append_log(run_dir, f"Redownloading {key}: local size does not match manifest")
            elif has_valid_local and run_dir:
                append_log(run_dir, f"Extracting existing tarball {key}")
            elif dest.exists() and dest.stat().st_size > 0 and run_dir:
                append_log(run_dir, f"Redownloading {key}: local size does not match manifest")

            if expected_size is not None and self.cost.would_exceed_cap(expected_size):
                skipped += 1
                if run_dir:
                    append_log(run_dir, f"Skipping {key}: cost cap would be exceeded")
                continue
            pending.append(key)

        if run_dir:
            mode = "download+extract+delete" if pipeline_mode else "download"
            append_log(run_dir, f"Pending tarballs ({mode}): {len(pending)} / {len(keys)}")
        started_at = time.time()
        last_progress_at = 0.0

        def _print_progress(force: bool = False) -> None:
            nonlocal last_progress_at
            now = time.time()
            if not force and now - last_progress_at < progress_interval_s:
                return
            last_progress_at = now
            print(
                _format_progress_line(
                    total=len(keys),
                    already_done=already_done,
                    completed=len(self.cost.completed_keys),
                    failed=len(self.cost.failed_keys),
                    skipped=skipped,
                    in_flight=len(futures),
                    gb_downloaded=self.cost.gb_downloaded,
                    cost_usd=self.cost.total_cost_usd,
                    hard_cap_usd=self.cost.hard_cap_usd,
                    started_at=started_at,
                ),
                flush=True,
            )

        def _task(key: str) -> tuple[str, int, bool]:
            from grounded.data.filter import process_tarball

            if stop.is_set():
                raise KeyboardInterrupt

            dest = tarball_local_path(self.raw_dir, key)
            expected_size = key_sizes.get(key)
            downloaded_this_run = False
            if dest.exists() and dest.stat().st_size > 0 and _size_is_valid(
                dest.stat().st_size, expected_size, None
            ):
                size = dest.stat().st_size
            else:
                size = _download_one(
                    self.client,
                    self.bucket,
                    key,
                    dest,
                    self.request_payer,
                    self.max_retries,
                    self._transfer_config,
                    expected_size,
                    stop=stop,
                )
                downloaded_this_run = True

            if stop.is_set():
                raise KeyboardInterrupt

            if pipeline_mode and pipeline is not None:
                extracted = process_tarball(
                    dest,
                    key,
                    target_ids=pipeline.target_ids,
                    output_dir=pipeline.output_dir,
                    extracted_ledger_path=pipeline.extracted_ledger_path,
                    skip_extensions=pipeline.skip_extensions,
                    delete_tarball=pipeline.delete_tarball_after_extract,
                )
                if run_dir:
                    append_log(run_dir, f"Processed {key}: extracted {len(extracted)} papers")
            return key, size, downloaded_this_run

        pool = _DaemonThreadPoolExecutor(max_workers=self.max_workers)
        futures: dict[Future[tuple[str, int, bool]], str] = {}
        key_iter = iter(pending)
        stop = threading.Event()
        previous_sigint = signal.getsignal(signal.SIGINT)

        def _request_stop(*_args) -> None:
            if stop.is_set():
                os._exit(130)
            stop.set()
            if run_dir:
                append_log(run_dir, "Stop requested (Ctrl+C); cancelling pending downloads")

        signal.signal(signal.SIGINT, _request_stop)

        def _submit_next() -> None:
            if stop.is_set():
                return
            try:
                key = next(key_iter)
            except StopIteration:
                return
            futures[pool.submit(_task, key)] = key

        for _ in range(min(self.max_workers, len(pending))):
            _submit_next()

        interrupted = False
        _print_progress(force=True)
        try:
            while futures:
                if stop.is_set():
                    raise KeyboardInterrupt
                done, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    key = futures.pop(future)
                    try:
                        _, size, downloaded_this_run = future.result()
                        if downloaded_this_run:
                            if self.cost.would_exceed_cap(size):
                                raise RuntimeError(
                                    f"Cost cap ${self.cost.hard_cap_usd:.2f} would be exceeded "
                                    f"after downloading {key}"
                                )
                            self.cost.record(key, size)
                            append_ledger(self.ledger_path, key)
                            if run_dir:
                                append_log(run_dir, f"Downloaded {key} ({size} bytes)")
                        elif run_dir and pipeline_mode:
                            append_log(run_dir, f"Extracted {key} from existing local tarball")
                        if run_dir:
                            write_json(run_dir, "cost.json", self.cost.to_dict())
                    except KeyboardInterrupt:
                        stop.set()
                        raise
                    except Exception as exc:
                        self.cost.failed_keys.append(key)
                        if run_dir:
                            append_log(run_dir, f"FAILED {key}: {exc}")
                            write_json(run_dir, "cost.json", self.cost.to_dict())
                    if not stop.is_set():
                        _submit_next()
                    _print_progress()
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            if run_dir:
                append_log(run_dir, "Interrupted by user; completed tarballs are ledgered and resumable")
                write_json(run_dir, "cost.json", self.cost.to_dict())
            _print_progress(force=True)
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            _cleanup_partial_downloads(self.raw_dir)
            raise
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            if not interrupted:
                pool.shutdown(wait=True, cancel_futures=True)

        if run_dir:
            write_json(run_dir, "cost.json", self.cost.to_dict())
        _print_progress(force=True)
        return self.cost
