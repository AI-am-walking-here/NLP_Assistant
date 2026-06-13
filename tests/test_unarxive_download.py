import json
import tarfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from grounded.data.unarxive_download import (
    IncompleteDownloadError,
    _download_file,
    count_shards,
    download_unarxive_shards,
    shards_ready,
)


class _RangeFileHandler(BaseHTTPRequestHandler):
    payload = b"hello-world-payload"

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        start = 0
        if self.headers.get("Range", "").startswith("bytes="):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
        body = self.payload[start:]
        if start:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def _serve(handler_cls, host: str = "127.0.0.1", port: int = 0) -> tuple[str, ThreadingHTTPServer]:
    server = ThreadingHTTPServer((host, port), handler_cls)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://{host}:{server.server_address[1]}/file.bin", server


def test_download_file_resumes_partial(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"hello-")

    class Handler(_RangeFileHandler):
        payload = b"hello-world-payload"

    url, server = _serve(Handler)
    try:
        size = _download_file(url, dest, show_progress=False)
    finally:
        server.shutdown()

    assert size == len(Handler.payload)
    assert dest.read_bytes() == Handler.payload


def test_download_file_raises_when_truncated(tmp_path):
    dest = tmp_path / "file.bin"

    class TruncatedHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"short")

        def log_message(self, format: str, *args) -> None:
            return

    url, server = _serve(TruncatedHandler)
    try:
        try:
            _download_file(url, dest, show_progress=False)
        except IncompleteDownloadError as exc:
            assert "incomplete" in str(exc).lower()
        else:
            raise AssertionError("expected IncompleteDownloadError")
    finally:
        server.shutdown()


def _write_sample_archive(archive_path, *, inner_dir: str = "unarXive_subset") -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"paper_id": "2212.11867", "abstract": "Sample", "body_text": "Body"}
    jsonl_bytes = (json.dumps(record) + "\n").encode("utf-8")
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo(name=f"{inner_dir}/sample.jsonl")
        info.size = len(jsonl_bytes)
        archive.addfile(info, fileobj=__import__("io").BytesIO(jsonl_bytes))


def test_download_unarxive_shards_extracts_jsonl(tmp_path):
    root_dir = tmp_path / "unarxive"
    archive_path = root_dir / "_cache" / "sample.tar.gz"
    _write_sample_archive(archive_path)

    stats = download_unarxive_shards(
        root_dir,
        url=f"file://{archive_path.as_posix()}",
        archive_name="sample.tar.gz",
        shard_glob="**/*.jsonl",
        min_shards=1,
        show_progress=False,
    )

    assert stats["skipped"] is False
    assert stats["extracted_shards"] == 1
    assert count_shards(root_dir, "**/*.jsonl") == 1
    assert shards_ready(root_dir, "**/*.jsonl")


def test_download_unarxive_shards_skips_when_present(tmp_path):
    root_dir = tmp_path / "unarxive"
    root_dir.mkdir(parents=True)
    (root_dir / "existing.jsonl").write_text("{}\n", encoding="utf-8")

    stats = download_unarxive_shards(
        root_dir,
        url="file:///missing",
        archive_name="missing.tar.gz",
        shard_glob="**/*.jsonl",
        min_shards=1,
        show_progress=False,
    )

    assert stats["skipped"] is True
    assert stats["shard_count"] == 1
