#!/usr/bin/env python3
"""M-3.6 — end-to-end naive RAG smoke test (5 hand-picked prompts)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config, resolve_path
from grounded.generate.baselines import build_generator, naive_rag
from grounded.index.embed import ChunkEmbedder, MockEmbedder
from grounded.index.vector_store import VectorStore, load_chunk_rows

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SMOKE_PROMPTS = [
    {
        "title": "AdaptSum: Towards Low-Resource Domain Adaptation for Abstractive Summarization",
        "outline": (
            "- Benchmark six domains for low-resource abstractive summarization\n"
            "- Study second-phase pre-training (SDPT, DAPT, TAPT)\n"
            "- Apply RecAdam to reduce catastrophic forgetting"
        ),
        "source_paper": "2103.11332",
    },
    {
        "title": "TAT-QA: A Question Answering Benchmark on Tabular and Textual Content",
        "outline": (
            "- Hybrid table+text QA benchmark with numerical reasoning\n"
            "- Baselines and human performance analysis"
        ),
        "source_paper": "2105.07624",
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "outline": (
            "- Combine parametric and non-parametric memories\n"
            "- End-to-end fine-tuning with differentiable retrieval"
        ),
        "source_paper": None,
    },
    {
        "title": "Parameter-Efficient Fine-Tuning of Large Language Models",
        "outline": (
            "- Survey PEFT methods (adapters, prefix tuning, LoRA)\n"
            "- Trade-offs between efficiency and downstream performance"
        ),
        "source_paper": None,
    },
    {
        "title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision",
        "outline": (
            "- Decompose generations into atomic facts\n"
            "- Verify facts against a knowledge source with an LLM"
        ),
        "source_paper": None,
    },
]


@click.command()
@click.option(
    "--mock-gen/--no-mock-gen",
    default=None,
    help="Override configs/retrieval.yaml generation.mock_generation.",
)
@click.option(
    "--mock-embed/--no-mock-embed",
    default=None,
    help="Use hash embedder (no HF download); auto when index_meta.mock_embed is true.",
)
def main(mock_gen: bool | None, mock_embed: bool | None) -> int:
    retr_cfg = load_config("retrieval")
    rpaths = retr_cfg.paths
    chunks_path = resolve_path(rpaths.chunks_parquet)
    faiss_path = resolve_path(rpaths.faiss_index)
    meta_path = resolve_path(rpaths.index_meta)

    if not faiss_path.is_file():
        raise click.ClickException(
            f"Missing {faiss_path}. Run: python3 scripts/build_index.py --limit 200"
        )

    chunk_rows = load_chunk_rows(chunks_path)
    store = VectorStore.load(faiss_path, chunk_rows, meta_path)
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    use_mock_embed = mock_embed if mock_embed is not None else bool(meta.get("mock_embed"))
    if use_mock_embed:
        embedder = MockEmbedder(dimension=int(meta.get("dim", 64)))
    else:
        embedder = ChunkEmbedder(
            retr_cfg.embedder,
            device=retr_cfg.embed_device,
            normalize=retr_cfg.normalize_embeddings,
            batch_size=retr_cfg.embed_batch_size,
        )
    use_mock = (
        mock_gen
        if mock_gen is not None
        else retr_cfg.generation.mock_generation
    )
    generator = build_generator(
        model_name=retr_cfg.generation.base_model,
        max_new_tokens=retr_cfg.generation.max_new_tokens,
        temperature=retr_cfg.generation.temperature,
        mock=use_mock,
    )

    ctx = init_run("seg3", "smoke", tags=["m-3.6"])
    out_dir = ctx.run_dir / "smoke_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(SMOKE_PROMPTS, start=1):
        result = naive_rag(
            sample["title"],
            sample["outline"],
            store,
            embedder,
            generator,
            k=retr_cfg.top_k,
        )
        payload = {
            "sample_id": i,
            **sample,
            "abstract_text": result.abstract_text,
            "mock": result.mock,
            "retrieved": [
                {
                    "chunk_id": h.get("chunk_id"),
                    "paper_id": h.get("paper_id"),
                    "section_heading": h.get("section_heading"),
                    "score": h.get("score"),
                    "text_preview": (h.get("text") or "")[:240],
                }
                for h in result.retrieved_chunks
            ],
        }
        out_path = out_dir / f"sample_{i}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("Wrote %s (%d chars)", out_path, len(result.abstract_text))

    readme = out_dir / "README.txt"
    readme.write_text(
        "Inspect sample_*.json. Each file has abstract_text and top-k retrieved chunks.\n"
        "v3.1: no [CITE] markers in prompts or outputs.\n",
        encoding="utf-8",
    )
    finish_run(ctx)
    click.echo(f"Smoke outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
