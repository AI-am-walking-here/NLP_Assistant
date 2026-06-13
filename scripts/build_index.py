#!/usr/bin/env python3
"""M-3.1–3.3 — chunk index papers, embed, build FAISS."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np

from grounded.config import finish_run, init_run, load_config, log_metric, resolve_path
from grounded.progress import update_run_progress
from grounded.utils.hf_network import require_model_download
from grounded.index.chunker import chunk_papers_to_records, write_chunks_parquet
from grounded.index.embed import ChunkEmbedder, MockEmbedder
from grounded.index.vector_store import VectorStore, load_chunk_rows, save_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_EMBED_DIM = 64


def load_index_ids(path: Path, limit: int | None) -> list[str]:
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit is not None:
        ids = ids[:limit]
    return ids


def embed_texts(
    texts: list[str],
    *,
    mock_embed: bool,
    retr_cfg,
) -> tuple[np.ndarray, int, str]:
    if mock_embed:
        embedder = MockEmbedder(dimension=MOCK_EMBED_DIM)
        vectors = embedder.encode(texts, show_progress=True)
        return vectors, MOCK_EMBED_DIM, "mock-hash"
    embedder = ChunkEmbedder(
        retr_cfg.embedder,
        device=retr_cfg.embed_device,
        normalize=retr_cfg.normalize_embeddings,
        batch_size=retr_cfg.embed_batch_size,
    )
    vectors = embedder.encode(texts, show_progress=True)
    return vectors, embedder.dimension, retr_cfg.embedder


def write_chunk_id_table(records: list[dict], chunk_ids_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    id_rows = [
        {
            "chunk_id": r["chunk_id"],
            "paper_id": r["paper_id"],
            "section_heading": r.get("section_heading", ""),
            "chunk_idx": r["chunk_idx"],
        }
        for r in records
    ]
    chunk_ids_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(id_rows), chunk_ids_path)


@click.command()
@click.option("--limit", type=int, default=None, help="Only index the first N papers (smoke/pilot).")
@click.option(
    "--index-list",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Paper ID list (default: data/splits/index.txt).",
)
@click.option("--mock-embed", is_flag=True, help="Hash-based unit vectors (no HF download).")
@click.option(
    "--embed-only",
    is_flag=True,
    help="Skip chunking; embed + FAISS from existing chunks.parquet.",
)
@click.option(
    "--skip-chunk",
    is_flag=True,
    help="Reuse chunks.parquet if present (same as --embed-only).",
)
@click.option(
    "--only-missing",
    is_flag=True,
    help="Chunk papers in index list absent from chunks.parquet, merge, re-embed.",
)
@click.option(
    "--prune-to-index",
    is_flag=True,
    help="Drop chunks whose paper_id is not in the index list before embed.",
)
@click.option(
    "--allow-stale-chunks",
    is_flag=True,
    help="Allow embed-only / skip-chunk reuse without verifying that chunks.parquet matches the index list.",
)
@click.option(
    "--skip-embed-if-valid",
    is_flag=True,
    help="Skip embedding if index_meta.json already matches the current inputs fingerprint.",
)
def main(
    limit: int | None,
    index_list: Path | None,
    mock_embed: bool,
    embed_only: bool,
    skip_chunk: bool,
    only_missing: bool,
    prune_to_index: bool,
    allow_stale_chunks: bool,
    skip_embed_if_valid: bool,
) -> int:
    data_cfg = load_config("data")
    retr_cfg = load_config("retrieval")
    paths = data_cfg.paths
    rpaths = retr_cfg.paths

    parsed_dir = resolve_path(paths.parsed_dir)
    list_path = index_list or resolve_path(paths.splits_dir) / "index.txt"
    chunks_path = resolve_path(rpaths.chunks_parquet)
    chunk_ids_path = resolve_path(rpaths.chunk_ids_parquet)
    emb_path = resolve_path(rpaths.embeddings_npy)
    faiss_path = resolve_path(rpaths.faiss_index)
    meta_path = resolve_path(rpaths.index_meta)

    arxiv_ids = load_index_ids(list_path, limit)
    if not arxiv_ids:
        raise click.ClickException(f"No IDs in {list_path}")

    if not mock_embed:
        require_model_download(
            "BGE index rebuild (--embed-only without --mock-embed)",
            hub_id=retr_cfg.embedder,
            role="embedder",
        )
    ctx = init_run("seg3", "build_index", tags=["m-3.1", "m-3.2", "m-3.3"])
    reuse_chunks = (embed_only or skip_chunk) and chunks_path.is_file() and not only_missing

    if only_missing:
        existing: list[dict] = []
        if chunks_path.is_file():
            existing = load_chunk_rows(chunks_path)
        have = {r["paper_id"] for r in existing}
        missing_ids = [aid for aid in arxiv_ids if aid not in have]
        if not missing_ids:
            logger.info("No missing papers in %s", chunks_path)
            records = existing
            paper_count = len(have)
        else:
            logger.info("Chunking %d missing papers (have %d)", len(missing_ids), len(have))
            new_records = chunk_papers_to_records(
                parsed_dir,
                missing_ids,
                tokenizer_name=retr_cfg.tokenizer,
                chunk_size=retr_cfg.chunk_size,
                chunk_overlap=retr_cfg.chunk_overlap,
            )
            records = existing + new_records
            write_chunks_parquet(records, chunks_path)
            paper_count = len({r["paper_id"] for r in records})
            logger.info("Merged → %d chunks from %d papers", len(records), paper_count)
    elif reuse_chunks:
        logger.info("Loading existing chunks from %s", chunks_path)
        records = load_chunk_rows(chunks_path)
        index_set = set(arxiv_ids)
        extra_ids = sorted({r["paper_id"] for r in records if r["paper_id"] not in index_set})
        missing_ids = sorted(index_set - {r["paper_id"] for r in records})
        if (extra_ids or missing_ids) and not (prune_to_index or allow_stale_chunks):
            raise click.ClickException(
                "chunks.parquet does not match the requested index list. "
                "Re-run with --prune-to-index, rebuild chunks, or pass --allow-stale-chunks explicitly."
            )
        paper_count = len({r["paper_id"] for r in records})
        logger.info("Loaded %d chunks from %d papers", len(records), paper_count)
    else:
        logger.info("Chunking %d papers…", len(arxiv_ids))
        records = chunk_papers_to_records(
            parsed_dir,
            arxiv_ids,
            tokenizer_name=retr_cfg.tokenizer,
            chunk_size=retr_cfg.chunk_size,
            chunk_overlap=retr_cfg.chunk_overlap,
        )
        if not records:
            raise click.ClickException("No chunks produced")
        logger.info("Writing %d chunks to %s", len(records), chunks_path)
        write_chunks_parquet(records, chunks_path)
        paper_count = len(arxiv_ids)

    if prune_to_index:
        index_set = set(arxiv_ids)
        before = len(records)
        records = [r for r in records if r["paper_id"] in index_set]
        paper_count = len({r["paper_id"] for r in records})
        logger.info(
            "Pruned to index list: %d → %d chunks (%d papers)",
            before,
            len(records),
            paper_count,
        )
        write_chunks_parquet(records, chunks_path)

    log_metric(ctx, "chunk_count", float(len(records)))
    log_metric(ctx, "paper_count", float(paper_count))

    from grounded.utils.phase_resume import phase_input_fingerprint

    inputs_fp = phase_input_fingerprint("build_index")
    if skip_embed_if_valid and meta_path.is_file() and faiss_path.is_file() and emb_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("inputs_fingerprint") == inputs_fp and int(meta.get("paper_count", 0)) == paper_count:
            logger.info("resume: skip embed; index metadata matches current inputs")
            summary = {
                "papers": paper_count,
                "chunks": len(records),
                "mock_embed": bool(meta.get("mock_embed", mock_embed)),
                "chunks_path": str(chunks_path),
                "faiss_path": str(faiss_path),
                "skipped_embed": True,
            }
            (ctx.run_dir / "build_index_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            finish_run(ctx)
            click.echo(json.dumps(summary, indent=2))
            return 0

    texts = [r["text"] for r in records]
    logger.info("Embedding %d chunks (mock=%s)…", len(texts), mock_embed)
    update_run_progress("build_index", done=0, total=len(texts), unit="chunks")
    try:
        embeddings, dim, embedder_name = embed_texts(
            texts, mock_embed=mock_embed, retr_cfg=retr_cfg
        )
    except Exception as exc:
        if mock_embed:
            raise
        logger.warning("Real embedder failed (%s); retry with --mock-embed", exc)
        raise click.ClickException(
            "Embedding failed (often no HuggingFace network). "
            "Re-run with --mock-embed or --embed-only --mock-embed."
        ) from exc
    update_run_progress("build_index", done=len(texts), total=len(texts), unit="chunks", eta_s=0)

    save_embeddings(emb_path, embeddings)
    write_chunk_id_table(records, chunk_ids_path)

    store = VectorStore.build(
        embeddings,
        records,
        index_type=retr_cfg.index_type,
    )
    store.meta.update(
        {
            "embedder": embedder_name,
            "chunk_size": retr_cfg.chunk_size,
            "chunk_overlap": retr_cfg.chunk_overlap,
            "paper_count": paper_count,
            "mock_embed": mock_embed,
            "dim": dim,
            "inputs_fingerprint": inputs_fp,
        }
    )
    store.save(faiss_path, meta_path)

    summary = {
        "papers": paper_count,
        "chunks": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "mock_embed": mock_embed,
        "chunks_path": str(chunks_path),
        "faiss_path": str(faiss_path),
    }
    (ctx.run_dir / "build_index_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Done: %s", json.dumps(summary, indent=2))
    finish_run(ctx)
    click.echo(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
