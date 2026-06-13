"""YAML config loading, run tracking, and path resolution."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


RUNS_DIR = project_root() / "runs"


# --- Segment 1 / data ---


class CostConfig(BaseModel):
    egress_per_gb_usd: float = 0.09
    get_request_usd: float = 0.0004
    hard_cap_usd: float = 0.0


class S3TransferConfig(BaseModel):
    multipart_threshold_mb: int = 8
    multipart_chunksize_mb: int = 16
    max_concurrency: int = 10
    max_pool_connections: int = 32


class S3PipelineConfig(BaseModel):
    extract_after_download: bool = True
    delete_tarball_after_extract: bool = True


class ArxivS3Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    year_min: int = 2023
    year_max: int = 2026
    paper_fraction: float = 0.1
    random_seed: int = 1337
    cost: CostConfig = Field(default_factory=CostConfig)
    bucket: str = "arxiv"
    manifest_key: str = "src/arXiv_src_manifest.xml"
    region: str = "us-east-1"
    request_payer: str = "requester"
    max_workers: int = 4
    max_retries: int = 3
    transfer: S3TransferConfig = Field(default_factory=S3TransferConfig)
    pipeline: S3PipelineConfig = Field(default_factory=S3PipelineConfig)


class UnarxiveDownloadConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str = ""
    archive_name: str = ""
    min_shards: int = 1
    keep_archive: bool = False
    resume_forever: bool = True
    retry_delay_s: float = 5.0
    max_retry_delay_s: float = 300.0


class UnarxiveConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    year_min: int = 2016
    year_max: int = 2022
    paper_fraction: float = 0.9
    random_seed: int = 1337
    root_dir: str = "data/archive/unarxive_extracted"
    shard_glob: str = "**/*.jsonl"
    id_fields: list[str] = Field(default_factory=list)
    title_fields: list[str] = Field(default_factory=list)
    abstract_fields: list[str] = Field(default_factory=list)
    text_fields: list[str] = Field(default_factory=list)
    download: UnarxiveDownloadConfig = Field(default_factory=UnarxiveDownloadConfig)
    delete_shards_after_materialize: bool = False


class SourcesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    arxiv_s3: ArxivS3Config = Field(default_factory=ArxivS3Config)
    unarxive: UnarxiveConfig = Field(default_factory=UnarxiveConfig)


class DataPaths(BaseModel):
    data_dir: str = "data"
    parsed_dir: str = "data/parsed"
    parsed_manifest: str = "data/parsed_manifest.jsonl"
    parsed_valid: str = "data/parsed_valid.json"
    splits_dir: str = "data/splits"
    s2_cache: str = "data/s2_cache.jsonl"
    papers_enriched: str = "data/papers_enriched.jsonl"
    kaggle_metadata: str = "data/archive/metadata/arxiv-metadata-oai-snapshot.json"
    cs_cl_ids: str = "data/archive/metadata/cs_cl_ids.json"
    manifest_xml: str = "data/archive/metadata/arXiv_src_manifest.xml"
    manifest_filtered: str = "data/archive/metadata/manifest_filtered.json"
    raw_tarballs: str = "data/archive/raw_tarballs"
    tex_extracted: str = "data/archive/tex_extracted"
    downloaded_ledger: str = "data/archive/metadata/ledgers/downloaded.txt"
    extracted_ledger: str = "data/archive/metadata/ledgers/extracted.txt"
    unarxive_manifest: str = "data/archive/metadata/unarxive_manifest.json"
    unarxive_extracted: str = "data/archive/unarxive_extracted"
    unarxive_materialized_ledger: str = (
        "data/archive/metadata/ledgers/unarxive_materialized.txt"
    )


class KaggleConfig(BaseModel):
    dataset: str = "Cornell-University/arxiv"


class Seg2Config(BaseModel):
    min_body_len: int = 4000
    min_citation_keys: int = 0
    holdout_fraction: float = 0.10
    eval_grid_size: int = 80
    sft_citation_min: int = 3
    random_seed: int = 1337


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str = "cs.CL"
    category_match: str = "primary"
    paper_count_target: int | None = 50000
    paths: DataPaths = Field(default_factory=DataPaths)
    seg2: Seg2Config = Field(default_factory=Seg2Config)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    cleaning: dict[str, Any] = Field(default_factory=dict)
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)


# --- Segment 3 / retrieval ---


class GenerationConfig(BaseModel):
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_new_tokens: int = 384
    temperature: float = 0.7
    mock_generation: bool = True


class RetrievalPaths(BaseModel):
    chunks_parquet: str = "data/chunks/chunks.parquet"
    chunk_ids_parquet: str = "data/indices/chunk_ids.parquet"
    embeddings_npy: str = "data/indices/embeddings.npy"
    faiss_index: str = "data/indices/faiss.index"
    index_meta: str = "data/indices/index_meta.json"


class RetrievalConfig(BaseModel):
    embedder: str = "BAAI/bge-large-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 8
    index_type: str = "IndexFlatIP"
    normalize_embeddings: bool = True
    embed_batch_size: int = 32
    embed_device: str = "auto"
    tokenizer: str = "cl100k_base"
    paths: RetrievalPaths = Field(default_factory=RetrievalPaths)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# --- Segment 5 / SFT ---


class SftPaths(BaseModel):
    train_jsonl: str = "data/sft/train.jsonl"
    val_jsonl: str = "data/sft/val.jsonl"
    split_list: str = "data/splits/sft.txt"


class DpoConfig(BaseModel):
    """FActScore-aligned DPO (preference pairs from build_sft_dpo_data.py)."""

    pairs_jsonl: str = "data/sft/dpo_pairs.jsonl"
    beta: float = 0.1
    learning_rate: float = 5.0e-5
    num_epochs: int = 1
    max_seq_len: int = 2048
    max_prompt_length: int = 1536
    gradient_accumulation_steps: int = 8
    min_factscore_margin: float = 0.08
    retrieval_top_k: int = 8
    init_from_latest_sft: bool = True
    verifier_cache_path: str = "runs/verifier_cache_dpo.jsonl"
    prompt_style: str = "sft"


class SftConfig(BaseModel):
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 1.0e-4
    num_epochs: int = 2
    max_seq_len: int = 2048
    citation_count_min: int = 3
    outline_source: str = "body"
    prompt_mode: str = "mixed"
    retrieval_fraction: float = 0.3
    retrieval_top_k: int = 8
    val_fraction: float = 0.05
    val_seed: int = 1337
    inference_temperature: float = 0.4
    paths: SftPaths = Field(default_factory=SftPaths)
    dpo: DpoConfig = Field(default_factory=DpoConfig)


# --- Segment 6 ---


class RankragPaths(BaseModel):
    train_jsonl: str = "data/rankrag/train.jsonl"


class RankragConfig(BaseModel):
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    rerank_candidates: int = 30
    rerank_top_k: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    paths: RankragPaths = Field(default_factory=RankragPaths)


class GraphPaths(BaseModel):
    pilot_ids: str = "data/splits/graph_pilot_500.txt"
    pilot_triples: str = "data/graph/pilot_triples.parquet"
    communities: str = "data/graph/communities.parquet"


class GraphConfig(BaseModel):
    pilot_size: int = 500
    gpu_hour_gate: float = 50.0
    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    seconds_per_chunk_estimate: float = 2.5
    paths: GraphPaths = Field(default_factory=GraphPaths)


# --- Segment 4 / eval ---


class VerifierVllmConfig(BaseModel):
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 2048
    tensor_parallel_size: int = 2
    max_tokens: int = 16
    max_batch_size: int = 8
    enforce_eager: bool = True


class EvalConfig(BaseModel):
    eval_prompts_path: str = "data/eval_set/prompts.jsonl"
    eval_grid_size: int = 80
    factscore_max_claims: int = 12
    ragas_max_claims: int = 8
    verifier_server_url: str = "http://127.0.0.1:8765"
    verifier_cache_path: str = "runs/verifier_cache.jsonl"
    verifier_default_backend: str = "vllm"
    verifier_model: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    verifier_quantization: str = "awq"
    verifier_vllm: VerifierVllmConfig = Field(default_factory=VerifierVllmConfig)
    bootstrap_resamples: int = 10000
    systems: list[str] = Field(default_factory=list)


# --- Local model paths ---


class ModelSpec(BaseModel):
    hub_id: str
    local_dir: str


class ModelsConfig(BaseModel):
    root: str = "/data/team1/models"
    embedder: ModelSpec = Field(
        default_factory=lambda: ModelSpec(
            hub_id="BAAI/bge-large-en-v1.5",
            local_dir="BAAI/bge-large-en-v1.5",
        )
    )
    generator_8b: ModelSpec = Field(
        default_factory=lambda: ModelSpec(
            hub_id="meta-llama/Llama-3.1-8B-Instruct",
            local_dir="meta-llama/Llama-3.1-8B-Instruct",
        )
    )
    verifier_70b_awq: ModelSpec = Field(
        default_factory=lambda: ModelSpec(
            hub_id="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            local_dir="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        )
    )


_CONFIG_TYPES: dict[str, type[BaseModel]] = {
    "data": DataConfig,
    "retrieval": RetrievalConfig,
    "sft": SftConfig,
    "rankrag": RankragConfig,
    "graph": GraphConfig,
    "eval": EvalConfig,
    "models": ModelsConfig,
}


def load_config(name: str = "data") -> Any:
    config_path = project_root() / "configs" / f"{name}.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_cls = _CONFIG_TYPES.get(name)
    if model_cls is None:
        raise ValueError(f"Unknown config: {name!r}")
    return model_cls.model_validate(raw)


def load_dotenv_project(path: Path | None = None, override: bool = False) -> None:
    env_path = path or (project_root() / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def resolve_path(rel: str | Path) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    return project_root() / path


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root(),
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


@dataclass
class RunContext:
    segment: str
    purpose: str
    run_dir: Path
    meta_path: Path
    log_path: Path


def init_run(
    segment: str,
    purpose: str,
    *,
    tags: list[str] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    use_wandb: bool = True,
) -> RunContext:
    load_dotenv_project()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    run_dir = RUNS_DIR / f"{segment}_{purpose}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "segment": segment,
        "purpose": purpose,
        "timestamp": ts,
        "tags": tags or [],
        "git_hash": _git_hash(),
        "config": config_snapshot or {},
        "status": "running",
    }
    meta_path = run_dir / "meta.json"
    log_path = run_dir / "log.txt"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path.write_text("", encoding="utf-8")

    if use_wandb and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb.init(
                project="grounded",
                name=f"{segment}_{purpose}_{ts}",
                tags=tags or [],
                dir=str(run_dir),
                config=config_snapshot,
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("wandb init failed: %s", exc)

    return RunContext(
        segment=segment,
        purpose=purpose,
        run_dir=run_dir,
        meta_path=meta_path,
        log_path=log_path,
    )


def log_metric(ctx: RunContext, name: str, value: float) -> None:
    meta = json.loads(ctx.meta_path.read_text(encoding="utf-8"))
    metrics = meta.setdefault("metrics", {})
    metrics[name] = value
    ctx.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({name: value})
    except Exception:
        pass


def finish_run(ctx: RunContext, *, status: str = "finished") -> None:
    meta = json.loads(ctx.meta_path.read_text(encoding="utf-8"))
    meta["status"] = status
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    ctx.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def append_log(run_dir: Path, message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}\n"
    with (run_dir / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line)


def write_json(run_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# Back-compat alias
load_dotenv = load_dotenv_project
