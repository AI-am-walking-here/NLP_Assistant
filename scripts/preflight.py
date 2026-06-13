#!/usr/bin/env python3
"""Check GPU, data artifacts, and HuggingFace reachability before long jobs."""

from __future__ import annotations

import json
import socket
import sys
import urllib.request
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_hf(timeout: float = 5.0) -> dict:
    for url in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return {"ok": True, "endpoint": url}
        except OSError:
            continue
    return {"ok": False, "error": "huggingface.co and hf-mirror.com unreachable"}


def _check_cuda() -> dict:
    try:
        import torch

        return {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        return {"available": False, "error": "torch not installed"}


def _artifact_status() -> dict:
    checks = {
        "parsed_valid": PROJECT_ROOT / "data/parsed_valid.json",
        "index_split": PROJECT_ROOT / "data/splits/index.txt",
        "chunks": PROJECT_ROOT / "data/chunks/chunks.parquet",
        "faiss": PROJECT_ROOT / "data/indices/faiss.index",
        "eval_prompts": PROJECT_ROOT / "data/eval_set/prompts.jsonl",
        "sft_train": PROJECT_ROOT / "data/sft/train.jsonl",
    }
    out: dict[str, object] = {}
    for name, path in checks.items():
        row: dict[str, object] = {"path": str(path), "exists": path.is_file()}
        if path.is_file() and path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            row["count"] = len(data) if isinstance(data, list) else len(data)
        elif path.is_file() and path.suffix == ".txt":
            row["count"] = sum(1 for ln in path.read_text().splitlines() if ln.strip())
        elif path.is_file() and path.suffix == ".jsonl":
            row["count"] = sum(1 for ln in path.open() if ln.strip())
        out[name] = row
    meta = PROJECT_ROOT / "data/indices/index_meta.json"
    if meta.is_file():
        out["index_meta"] = json.loads(meta.read_text(encoding="utf-8"))
    return out


@click.command()
@click.option("--json-out", type=click.Path(path_type=Path), default=None)
def main(json_out: Path | None) -> int:
    from grounded.config import load_dotenv_project
    from grounded.utils.model_paths import model_status

    load_dotenv_project()
    models = model_status()
    report = {
        "hostname": socket.gethostname(),
        "hf": _check_hf(),
        "cuda": _check_cuda(),
        "artifacts": _artifact_status(),
        "local_models": models,
        "model_download_policy": (
            "Loads from GROUNDED_MODELS_ROOT when weights are present. "
            "Hub download only if GROUNDED_ALLOW_MODEL_DOWNLOAD=1."
        ),
        "commands": {
            "check_models": "python scripts/check_models.py",
            "real_embed_index": (
                "python scripts/build_index.py --embed-only --prune-to-index"
            ),
            "verifier_server": (
                "python scripts/serve_verifier.py --backend vllm"
            ),
        },
    }
    text = json.dumps(report, indent=2)
    click.echo(text)
    if json_out:
        json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["hf"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
