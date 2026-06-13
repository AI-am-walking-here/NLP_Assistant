"""Phase fingerprints and lightweight validation for hardened rebuilds."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from grounded.config import load_config, project_root, resolve_path

PhaseState = Literal["ok", "stale", "missing", "partial"]


@dataclass(frozen=True)
class PhaseStatus:
    state: PhaseState
    detail: str
    output_paths: tuple[Path, ...] = ()
    resume_run_dir: Path | None = None


SMALL_HASH_LIMIT = 32 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    if path.is_file() and stat.st_size <= SMALL_HASH_LIMIT:
        digest = sha256_file(path)
    elif path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as fh:
            h.update(fh.read(64 * 1024))
        digest = h.hexdigest()
    else:
        digest = "dir"
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "sha256": digest,
    }


def content_fingerprint(path: Path) -> dict[str, Any]:
    """Mtime-independent fingerprint for skip/resume checks."""
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    if not path.is_file():
        return {"path": str(path), "size": stat.st_size, "sha256": "dir"}
    return {
        "path": str(path),
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def split_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config_hash(name: str) -> str:
    path = project_root() / "configs" / f"{name}.yaml"
    return sha256_file(path) if path.is_file() else ""


def training_content_key(phase: str) -> str:
    sft_cfg = load_config("sft")
    rr_cfg = load_config("rankrag")
    if phase == "sft_train":
        payload: dict[str, Any] = {
            "phase": phase,
            "config": _config_hash("sft"),
            "train": content_fingerprint(resolve_path(sft_cfg.paths.train_jsonl)),
            "val": content_fingerprint(resolve_path(sft_cfg.paths.val_jsonl)),
        }
    elif phase == "rankrag_train":
        payload = {
            "phase": phase,
            "config": _config_hash("rankrag"),
            "train": content_fingerprint(resolve_path(rr_cfg.paths.train_jsonl)),
        }
    else:
        raise ValueError(f"unsupported training phase {phase}")
    return stable_hash(payload)


def phase_input_fingerprint(phase: str) -> str:
    data_cfg = load_config("data")
    sft_cfg = load_config("sft")
    retr_cfg = load_config("retrieval")
    rr_cfg = load_config("rankrag")
    root = project_root()
    splits = resolve_path(data_cfg.paths.splits_dir)
    payload: dict[str, Any] = {"phase": phase}
    if phase in {"ensure_splits", "regenerate_splits"}:
        payload.update(
            {
                "data_config": _config_hash("data"),
                "parsed_valid": file_fingerprint(resolve_path(data_cfg.paths.parsed_valid)),
            }
        )
    elif phase == "build_index":
        payload.update(
            {
                "index": file_fingerprint(splits / "index.txt"),
                "retrieval_config": _config_hash("retrieval"),
            }
        )
    elif phase == "build_sft_data":
        payload.update(
            {
                "sft_split": content_fingerprint(resolve_path(sft_cfg.paths.split_list)),
                "sft_config": _config_hash("sft"),
                "index_meta": content_fingerprint(resolve_path(retr_cfg.paths.index_meta)),
            }
        )
    elif phase == "build_rankrag_data":
        payload.update(
            {
                "rankrag_config": _config_hash("rankrag"),
                "eval_holdout": file_fingerprint(splits / "eval_holdout.txt"),
                "eval_grid": file_fingerprint(splits / "eval_grid_80.txt"),
                "limit": os.environ.get("RANKRAG_DATA_LIMIT", ""),
                "seed": os.environ.get("RANKRAG_DATA_SEED", ""),
            }
        )
    elif phase == "build_graph_extract":
        graph_cfg = load_config("graph")
        payload.update(
            {
                "graph_config": _config_hash("graph"),
                "index": file_fingerprint(splits / "index.txt"),
                "chunks": file_fingerprint(resolve_path(retr_cfg.paths.chunks_parquet)),
                "extractor": os.environ.get("GRAPH_EXTRACTOR", "llm"),
                "full_corpus": os.environ.get("GRAPH_FULL_CORPUS", "0"),
                "pilot_size": str(getattr(graph_cfg, "pilot_size", "")),
            }
        )
    elif phase in {"rankrag_train", "sft_train"}:
        return training_content_key(phase)
    elif phase == "build_eval_prompts":
        payload.update({"eval_grid": file_fingerprint(splits / "eval_grid_80.txt")})
    elif phase.startswith("eval_"):
        eval_cfg = load_config("eval")
        payload.update(
            {
                "eval_config": _config_hash("eval"),
                "prompts": file_fingerprint(resolve_path(eval_cfg.eval_prompts_path)),
                "system": phase.removeprefix("eval_"),
                "rankrag_adapter": os.environ.get("RERANK_ADAPTER", ""),
                "sft_adapter": os.environ.get("SFT_ADAPTER", ""),
                "verifier": os.environ.get("VERIFIER_URL", ""),
            }
        )
    else:
        payload.update({"root": str(root)})
    return stable_hash(payload)


def adapter_complete(run_dir: Path) -> bool:
    adapter = run_dir / "adapter"
    return adapter.is_dir() and any(adapter.iterdir())


def newest_run(prefix: str) -> Path | None:
    runs = project_root() / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(runs.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_train_runs(prefix: str) -> tuple[Path | None, Path | None]:
    """Return (newest complete run, newest incomplete run)."""
    runs = project_root() / "runs"
    if not runs.is_dir():
        return None, None
    candidates = sorted(runs.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    complete: Path | None = None
    incomplete: Path | None = None
    for run in candidates:
        if adapter_complete(run):
            if complete is None:
                complete = run
        elif incomplete is None:
            incomplete = run
    return complete, incomplete


def _sft_data_coverage_ok(meta: dict[str, Any], split_list: Path, train: Path, val: Path) -> bool:
    split_count = len(split_ids(split_list))
    return (
        int(meta.get("built_count", 0)) == split_count
        and line_count(train) == int(meta.get("train_count", 0))
        and line_count(val) == int(meta.get("val_count", 0))
        and line_count(train) > 0
    )


def _training_run_ok(phase: str, run: Path, content_key: str, input_fp: str) -> PhaseStatus | None:
    meta = read_json(run / "run_meta.json")
    if meta.get("train_content_key") == content_key or meta.get("inputs_fingerprint") == input_fp:
        return PhaseStatus("ok", f"{phase} adapter {run.name}", (run / "adapter",))
    stored_key = meta.get("train_content_key")
    if stored_key is not None and stored_key != content_key:
        return None
    if meta.get("status") == "trained":
        return PhaseStatus("ok", f"{phase} adapter {run.name} (legacy complete)", (run / "adapter",))
    return None


def newest_matching_eval(system: str, input_fp: str) -> PhaseStatus:
    runs = project_root() / "runs"
    candidates = sorted(
        runs.glob(f"seg4_eval_{system}_*") if runs.is_dir() else [],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        results = run_dir / "results.json"
        if results.is_file():
            meta = read_json(results)
            if meta.get("input_fingerprint") == input_fp:
                return PhaseStatus("ok", f"complete eval {run_dir.name}", (results,))
        if (run_dir / "per_prompt.jsonl").is_file():
            return PhaseStatus("partial", f"resume eval {run_dir.name}", resume_run_dir=run_dir)
    return PhaseStatus("missing", f"no eval run for {system}")


def validate_phase(phase: str) -> PhaseStatus:
    data_cfg = load_config("data")
    sft_cfg = load_config("sft")
    retr_cfg = load_config("retrieval")
    rr_cfg = load_config("rankrag")
    splits = resolve_path(data_cfg.paths.splits_dir)
    input_fp = phase_input_fingerprint(phase)
    if phase in {"ensure_splits", "regenerate_splits"}:
        paths = tuple(splits / name for name in ("index.txt", "sft.txt", "eval_grid_80.txt"))
        if not all(path.is_file() for path in paths):
            return PhaseStatus("missing", "one or more split files missing", paths)
        minimum = int(os.environ.get("SFT_SPLIT_MIN", "7000"))
        if line_count(splits / "sft.txt") < minimum:
            return PhaseStatus("stale", "sft split below minimum", paths)
        return PhaseStatus("ok", "split files present", paths)
    if phase == "build_graph_communities":
        path = resolve_path("data/graph/communities.parquet")
        return PhaseStatus("ok", "graph communities exist", (path,)) if path.is_file() else PhaseStatus("missing", "graph communities missing", (path,))
    if phase == "build_graph_extract":
        graph_cfg = load_config("graph")
        path = resolve_path(graph_cfg.paths.pilot_triples)
        manifest = path.parent / "triples_merge_manifest.json"
        if not path.is_file() or not manifest.is_file():
            return PhaseStatus("missing", "graph triples or manifest missing", (path, manifest))
        meta = read_json(manifest)
        if meta.get("inputs_fingerprint") != input_fp:
            return PhaseStatus("stale", "graph extraction input fingerprint mismatch", (path, manifest))
        if int(meta.get("written_triples", 0)) <= 0:
            return PhaseStatus("partial", "graph extraction has no triples", (path, manifest))
        return PhaseStatus("ok", "graph extraction complete", (path, manifest))
    if phase == "build_index":
        paths = (
            resolve_path(retr_cfg.paths.chunks_parquet),
            resolve_path(retr_cfg.paths.faiss_index),
            resolve_path(retr_cfg.paths.index_meta),
        )
        if not all(path.is_file() for path in paths):
            return PhaseStatus("missing", "index artifact missing", paths)
        meta = read_json(paths[-1])
        expected = len(set(split_ids(splits / "index.txt")))
        if int(meta.get("paper_count", 0)) != expected:
            return PhaseStatus("stale", "index paper count mismatch", paths)
        return PhaseStatus("ok", "index artifacts present", paths)
    if phase == "build_sft_data":
        train = resolve_path(sft_cfg.paths.train_jsonl)
        val = resolve_path(sft_cfg.paths.val_jsonl)
        manifest = train.parent / "merge_manifest.json"
        if not train.is_file() or not manifest.is_file():
            return PhaseStatus("missing", "sft train or merge manifest missing", (train, val, manifest))
        meta = read_json(manifest)
        split_list = resolve_path(sft_cfg.paths.split_list)
        if meta.get("inputs_fingerprint") != input_fp:
            if _sft_data_coverage_ok(meta, split_list, train, val):
                return PhaseStatus("ok", "sft data complete (coverage-validated)", (train, val, manifest))
            return PhaseStatus("stale", "sft input fingerprint mismatch", (train, val, manifest))
        if meta.get("built_count", 0) + len(meta.get("skipped_unbuildable", [])) != len(
            split_ids(split_list)
        ):
            return PhaseStatus("partial", "sft merge did not process all ids", (train, val, manifest))
        return PhaseStatus("ok", "sft data complete", (train, val, manifest))
    if phase == "build_rankrag_data":
        path = resolve_path(rr_cfg.paths.train_jsonl)
        manifest = path.parent / "merge_manifest.json"
        if not path.is_file() or not manifest.is_file():
            return PhaseStatus("missing", "rankrag data or merge manifest missing", (path, manifest))
        meta = read_json(manifest)
        if meta.get("inputs_fingerprint") != input_fp:
            return PhaseStatus("stale", "rankrag data input fingerprint mismatch", (path, manifest))
        if int(meta.get("written_count", 0)) <= 0:
            return PhaseStatus("partial", "rankrag data has no rows", (path, manifest))
        return PhaseStatus("ok", "rankrag data complete", (path, manifest))
    if phase == "rankrag_train":
        content_key = training_content_key(phase)
        complete, incomplete = resolve_train_runs("seg6_rankrag_train")
        if complete is not None:
            ok = _training_run_ok(phase, complete, content_key, input_fp)
            if ok is not None:
                return ok
            return PhaseStatus("stale", "rankrag adapter fingerprint mismatch")
        if incomplete is not None:
            return PhaseStatus("partial", f"resume {incomplete.name}", resume_run_dir=incomplete)
        return PhaseStatus("missing", "no rankrag train run")
    if phase == "sft_train":
        content_key = training_content_key(phase)
        complete, incomplete = resolve_train_runs("seg5_sft_train")
        if complete is not None:
            ok = _training_run_ok(phase, complete, content_key, input_fp)
            if ok is not None:
                return ok
            return PhaseStatus("stale", "sft adapter fingerprint mismatch")
        if incomplete is not None:
            return PhaseStatus("partial", f"resume {incomplete.name}", resume_run_dir=incomplete)
        return PhaseStatus("missing", "no sft train run")
    if phase == "build_eval_prompts":
        eval_cfg = load_config("eval")
        path = resolve_path(eval_cfg.eval_prompts_path)
        if not path.is_file():
            return PhaseStatus("missing", "eval prompts missing", (path,))
        expected = int(getattr(eval_cfg, "eval_grid_size", 80))
        if line_count(path) != expected:
            return PhaseStatus("stale", "eval prompt count mismatch", (path,))
        return PhaseStatus("ok", "eval prompts present", (path,))
    if phase.startswith("eval_"):
        return newest_matching_eval(phase.removeprefix("eval_"), input_fp)
    return PhaseStatus("missing", f"unknown phase {phase}")


def record_phase(phase: str, *, state_dir: Path | None = None, extra: dict[str, Any] | None = None) -> Path:
    state = state_dir or (project_root() / "runs" / "hardened_rebuild_state")
    state.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "inputs_fingerprint": phase_input_fingerprint(phase),
        "recorded_at": time.time(),
        **(extra or {}),
    }
    out = state / f"{phase}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
