#!/usr/bin/env python3
"""M-6.2 + M-6.3 — sample graph pilot, mock-extract triples, extrapolate GPU gate."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config, resolve_path
from grounded.graph.gate import extrapolate_gpu_hours
from grounded.graph.extract import get_extractor
from grounded.graph.pilot import (
    load_chunks_for_papers,
    sample_pilot_ids,
    write_id_list,
)
from grounded.utils.phase_resume import phase_input_fingerprint, stable_hash

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SECONDS_PER_CHUNK = 2.5


def triples_to_rows(triples: list) -> list[dict]:
    rows: list[dict] = []
    for t in triples:
        rows.append(
            {
                "chunk_id": t.chunk_id,
                "paper_id": t.paper_id,
                "extractor": t.extractor,
                "entities_json": json.dumps([e.model_dump() for e in t.entities]),
                "relations_json": json.dumps([r.model_dump() for r in t.relations]),
                "n_entities": len(t.entities),
                "n_relations": len(t.relations),
            }
        )
    return rows


def write_triples_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@click.command()
@click.option(
    "--source-list",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Paper IDs to sample from (default: data/splits/index.txt).",
)
@click.option("--pilot-size", type=int, default=None, help="Override configs/graph.yaml pilot_size.")
@click.option(
    "--seconds-per-chunk",
    type=float,
    default=DEFAULT_SECONDS_PER_CHUNK,
    show_default=True,
    help="Assumed 8B structured-extract seconds/chunk for gate extrapolation.",
)
@click.option(
    "--target-list",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Extrapolation target paper list (default: data/splits/sft.txt).",
)
@click.option(
    "--extractor",
    type=click.Choice(["mock", "llm"]),
    default="llm",
    show_default=True,
    help="mock=regex heuristics; llm=local Llama-3.1-8B JSON extraction.",
)
@click.option(
    "--max-chunks",
    type=int,
    default=None,
    help="Cap chunks processed (debug/smoke).",
)
def main(
    source_list: Path | None,
    pilot_size: int | None,
    seconds_per_chunk: float,
    target_list: Path | None,
    extractor: str,
    max_chunks: int | None,
) -> int:
    graph_cfg = load_config("graph")
    data_cfg = load_config("data")
    retr_cfg = load_config("retrieval")

    n_pilot = pilot_size or graph_cfg.pilot_size
    splits_dir = resolve_path(data_cfg.paths.splits_dir)
    source_path = source_list or splits_dir / "index.txt"
    pilot_ids_path = splits_dir / f"graph_pilot_{n_pilot}.txt"
    out_triples = resolve_path("data/graph/pilot_triples.parquet")
    chunks_path = resolve_path(retr_cfg.paths.chunks_parquet)
    target_path = target_list or splits_dir / "sft.txt"

    candidates = [
        ln.strip()
        for ln in source_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    pilot_ids = sample_pilot_ids(
        candidates,
        n=n_pilot,
        seed=data_cfg.seg2.random_seed,
    )
    write_id_list(pilot_ids_path, pilot_ids)
    logger.info("Wrote %d pilot IDs to %s", len(pilot_ids), pilot_ids_path)

    pilot_set = set(pilot_ids)
    chunk_rows = load_chunks_for_papers(chunks_path, pilot_set)
    if max_chunks is not None:
        chunk_rows = chunk_rows[:max_chunks]
    logger.info("Pilot chunks: %d from %d papers", len(chunk_rows), len(pilot_set))

    extract_fn = get_extractor(extractor)  # type: ignore[arg-type]
    if extractor == "llm":
        seconds_per_chunk = seconds_per_chunk or graph_cfg.seconds_per_chunk_estimate

    t0 = time.perf_counter()
    triples = [
        extract_fn(r["chunk_id"], r["paper_id"], r["text"])
        for r in chunk_rows
    ]
    elapsed = time.perf_counter() - t0
    observed_sec = elapsed / max(len(chunk_rows), 1)

    rows = triples_to_rows(triples)
    write_triples_parquet(rows, out_triples)
    (out_triples.parent / "triples_merge_manifest.json").write_text(
        json.dumps(
            {
                "inputs_fingerprint": phase_input_fingerprint("build_graph_extract"),
                "schedule_fingerprint": stable_hash(
                    {
                        "source": str(source_path),
                        "pilot_size": n_pilot,
                        "extractor": extractor,
                        "max_chunks": max_chunks,
                    }
                ),
                "target_papers": len(pilot_ids),
                "total_chunks": len(chunk_rows),
                "written_triples": len(rows),
                "path": str(out_triples),
                "pilot_is_subset": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    target_ids = [
        ln.strip() for ln in target_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    target_chunks = load_chunks_for_papers(chunks_path, set(target_ids))

    gate = extrapolate_gpu_hours(
        pilot_chunks=len(chunk_rows),
        pilot_seconds=seconds_per_chunk,
        target_chunks=len(target_chunks),
        gate_hours=graph_cfg.gpu_hour_gate,
    )
    gate["extractor"] = extractor
    gate["extract_wall_s"] = round(elapsed, 2)
    gate["extract_sec_per_chunk"] = round(observed_sec, 4)
    gate["pilot_ids_path"] = str(pilot_ids_path)
    gate["pilot_triples_path"] = str(out_triples)
    gate["target_papers"] = len(target_ids)

    ctx = init_run("seg6", "graph_pilot", tags=["m-6.2", "m-6.3"])
    summary_path = ctx.run_dir / "graph_pilot_gate.json"
    summary_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    finish_run(ctx)

    click.echo(json.dumps(gate, indent=2))
    if gate["keep_graph"]:
        logger.info("Gate PASS: projected %.1fh <= %.1fh", gate["projected_gpu_hours"], gate["gate_hours"])
    else:
        logger.warning(
            "Gate FAIL: projected %.1fh > %.1fh — drop graph per build plan",
            gate["projected_gpu_hours"],
            gate["gate_hours"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
