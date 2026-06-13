from __future__ import annotations

import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from grounded.progress import ByteProgressReporter, CountProgressReporter, StatusReporter, format_bytes


class IncompleteDownloadError(RuntimeError):
    """Raised when the remote connection closes before the full file is received."""


def count_shards(root_dir: Path, shard_glob: str) -> int:
    if not root_dir.exists():
        return 0
    return sum(1 for path in root_dir.glob(shard_glob) if path.is_file())


def shards_ready(root_dir: Path, shard_glob: str, *, min_shards: int = 1) -> bool:
    return count_shards(root_dir, shard_glob) >= min_shards


def _remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "grounded-seg1/0.1"})
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def _is_non_retriable_download_error(exc: IncompleteDownloadError) -> bool:
    message = str(exc)
    return "Server rejected resume" in message


def _download_file(
    url: str,
    dest: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
    label: str = "download",
    show_progress: bool = True,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = _remote_content_length(url)
    resume_from = dest.stat().st_size if dest.exists() else 0

    if total_bytes and resume_from >= total_bytes:
        if show_progress:
            StatusReporter(label).done(f"Already complete ({format_bytes(total_bytes)})")
        return total_bytes

    reporter = ByteProgressReporter(label, total_bytes=total_bytes) if show_progress else None
    if reporter and resume_from:
        reporter.done_bytes = resume_from

    if show_progress:
        if resume_from and total_bytes:
            StatusReporter(label).status(
                f"Resuming from {format_bytes(resume_from)} / {format_bytes(total_bytes)}",
                force=True,
            )
        elif total_bytes:
            StatusReporter(label).status(f"Starting download ({format_bytes(total_bytes)} total)", force=True)
        else:
            StatusReporter(label).status("Starting download (total size unknown)", force=True)

    headers = {"User-Agent": "grounded-seg1/0.1"}
    write_mode = "wb"
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        write_mode = "ab"

    request = urllib.request.Request(url, headers=headers)
    downloaded = resume_from
    try:
        with urllib.request.urlopen(request) as response, dest.open(write_mode) as handle:
            if total_bytes is None:
                length = response.headers.get("Content-Length")
                if length:
                    total_bytes = int(length) + resume_from
                    if reporter:
                        reporter.total_bytes = total_bytes
            if resume_from > 0 and getattr(response, "status", 200) not in {200, 206}:
                raise IncompleteDownloadError(
                    f"Server rejected resume at byte {resume_from}; delete {dest} and retry."
                )
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if reporter:
                    reporter.update(len(chunk))
    except IncompleteDownloadError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise IncompleteDownloadError(f"Download interrupted at {format_bytes(downloaded)}: {exc}") from exc

    if total_bytes and downloaded < total_bytes:
        raise IncompleteDownloadError(
            f"Download incomplete: got {format_bytes(downloaded)} of {format_bytes(total_bytes)}. "
            "Re-run the same command to resume."
        )

    if reporter:
        reporter.finish()
    return downloaded


def _extract_archive(archive_path: Path, extract_dir: Path, *, label: str = "extract", show_progress: bool = True) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if show_progress:
        StatusReporter(label).status(f"Opening {archive_path.name}", force=True)

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        reporter = (
            CountProgressReporter(label, total=len(members), unit="files") if show_progress else None
        )
        if show_progress:
            StatusReporter(label).status(f"Extracting {len(members)} files", force=True)
        extract_filter = "data" if "filter" in tarfile.TarFile.extract.__code__.co_varnames else None
        for index, member in enumerate(members, start=1):
            if extract_filter:
                archive.extract(member, path=extract_dir, filter=extract_filter)
            else:
                archive.extract(member, path=extract_dir)
            if reporter:
                reporter.done_count = index
                reporter.update(0, detail=member.name)
        if reporter:
            reporter.finish()


def _flatten_jsonl_shards(
    extract_dir: Path,
    root_dir: Path,
    *,
    label: str = "flatten",
    show_progress: bool = True,
) -> int:
    root_dir.mkdir(parents=True, exist_ok=True)
    sources = [path for path in sorted(extract_dir.rglob("*.jsonl")) if path.is_file()]
    reporter = CountProgressReporter(label, total=len(sources), unit="shards") if show_progress else None
    if show_progress:
        StatusReporter(label).status(f"Moving {len(sources)} JSONL shards into {root_dir}", force=True)

    moved = 0
    for index, source in enumerate(sources, start=1):
        destination = root_dir / source.name
        if not destination.exists():
            shutil.move(str(source), str(destination))
            moved += 1
        if reporter:
            reporter.done_count = index
            reporter.update(0, detail=source.name)
    if reporter:
        reporter.finish(detail=f"{moved} new shards placed")
    return moved


def download_unarxive_shards(
    root_dir: Path,
    *,
    url: str,
    archive_name: str,
    shard_glob: str,
    min_shards: int = 1,
    force: bool = False,
    keep_archive: bool = False,
    show_progress: bool = True,
    max_resume_attempts: int | None = None,
    retry_delay_s: float = 5.0,
    max_retry_delay_s: float = 300.0,
) -> dict[str, int | bool]:
    """Download the unarXive open subset from Zenodo and extract JSONL shards."""
    label = "unarxive"
    if not force and shards_ready(root_dir, shard_glob, min_shards=min_shards):
        shard_count = count_shards(root_dir, shard_glob)
        if show_progress:
            StatusReporter(label).done(f"Skipped download; {shard_count} JSONL shards already present")
        return {
            "skipped": True,
            "downloaded_bytes": 0,
            "extracted_shards": 0,
            "shard_count": shard_count,
        }

    cache_dir = root_dir / "_cache"
    archive_path = cache_dir / archive_name
    extract_dir = cache_dir / "extracted"

    if force:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        if archive_path.exists():
            archive_path.unlink()

    expected_size = _remote_content_length(url)
    archive_complete = (
        archive_path.exists()
        and archive_path.stat().st_size > 0
        and (expected_size is None or archive_path.stat().st_size >= expected_size)
    )

    if not archive_complete:
        last_error: IncompleteDownloadError | None = None
        attempt = 1
        delay_s = retry_delay_s
        while max_resume_attempts is None or attempt <= max_resume_attempts:
            try:
                downloaded_bytes = _download_file(url, archive_path, label=label, show_progress=show_progress)
                last_error = None
                break
            except IncompleteDownloadError as exc:
                last_error = exc
                if _is_non_retriable_download_error(exc):
                    break
                should_retry = max_resume_attempts is None or attempt < max_resume_attempts
                if show_progress and should_retry:
                    StatusReporter(label).status(
                        f"Attempt {attempt}"
                        + (f"/{max_resume_attempts}" if max_resume_attempts is not None else "")
                        + f" interrupted; retrying in {delay_s:.0f}s",
                        force=True,
                    )
                if should_retry:
                    time.sleep(delay_s)
                    delay_s = min(max_retry_delay_s, delay_s * 2)
                    attempt += 1
                    continue
                break
        if last_error is not None:
            raise FileNotFoundError(str(last_error)) from last_error
    else:
        downloaded_bytes = archive_path.stat().st_size
        if show_progress:
            StatusReporter(label).done(f"Using cached archive ({format_bytes(downloaded_bytes)})")

    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        _extract_archive(archive_path, extract_dir, label=label, show_progress=show_progress)
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise FileNotFoundError(
            f"Failed to extract {archive_path.name} ({format_bytes(archive_path.stat().st_size)}). "
            "The archive may be incomplete — re-run to resume the download."
        ) from exc
    extracted_shards = _flatten_jsonl_shards(extract_dir, root_dir, label=label, show_progress=show_progress)
    shutil.rmtree(extract_dir)

    if not keep_archive and archive_path.exists():
        if show_progress:
            StatusReporter(label).status("Removing downloaded archive", force=True)
        archive_path.unlink()

    shard_count = count_shards(root_dir, shard_glob)
    if shard_count < min_shards:
        raise FileNotFoundError(
            f"Expected at least {min_shards} JSONL shards under {root_dir} "
            f"matching {shard_glob!r}, found {shard_count}."
        )

    if show_progress:
        StatusReporter(label).done(
            f"Ready: {shard_count} JSONL shards ({extracted_shards} newly placed this run)"
        )

    return {
        "skipped": False,
        "downloaded_bytes": downloaded_bytes,
        "extracted_shards": extracted_shards,
        "shard_count": shard_count,
    }
