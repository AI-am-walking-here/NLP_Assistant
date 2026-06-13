#!/usr/bin/env python3
"""Segment 2 closeout: M-2.3 filter, M-2.4 enrichment, M-2.5 v3.1 splits."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from grounded.config import finish_run, init_run, load_config, log_metric, resolve_path
from grounded.data.citations import (
    MetadataCache,
    apply_cached_to_parsed,
    enrich_papers,
    ids_missing_citation_count,
    mark_not_found_zero,
    prune_enriched_valid_ids,
    read_parsed_record,
)
from grounded.data.corpus_export import rebuild_manifest_from_parsed
from grounded.data.filter import apply_quality_filter
from grounded.data.splits import build_splits_v31, drop_unreadable_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def write_enriched_summary(
    valid_ids: list[str],
    parsed_dir: Path,
    out_path: Path,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for aid in valid_ids:
            record = read_parsed_record(parsed_dir / f"{aid}.json")
            if record is None:
                logger.warning("Skip summary for unreadable %s", aid)
                continue
            row = {
                "arxiv_id": aid,
                "source": record.get("source"),
                "year": record.get("year"),
                "venue": record.get("venue"),
                "citation_count": record.get("citation_count"),
                "s2_paper_id": record.get("s2_paper_id"),
                "parse_status": record.get("parse_status"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _persist_valid(valid_path: Path, valid_ids: list[str]) -> None:
    valid_path.write_text(json.dumps(valid_ids, indent=2) + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--only",
    type=click.Choice(["filter", "s2", "splits", "all"]),
    default="all",
    help="Run a single stage or the full pipeline.",
)
@click.option(
    "--skip-s2",
    is_flag=True,
    help="Skip remote API calls (apply cache + zero-fill only).",
)
@click.option(
    "--provider",
    type=click.Choice(["auto", "s2", "openalex"]),
    default="auto",
    help="Enrichment backend (auto: S2 if S2_API_KEY else OpenAlex).",
)
@click.option(
    "--rebuild-manifest/--no-rebuild-manifest",
    default=True,
    help="Refresh parsed_manifest.jsonl from on-disk parsed JSON before filter.",
)
@click.option(
    "--refresh-not-found",
    is_flag=True,
    help="Re-query OpenAlex/S2 for cache rows marked not_found or stale.",
)
def main(
    only: str,
    skip_s2: bool,
    provider: str,
    rebuild_manifest: bool,
    refresh_not_found: bool,
) -> int:
    data_cfg = load_config("data")
    paths = data_cfg.paths
    manifest = resolve_path(paths.parsed_manifest)
    parsed_dir = resolve_path(paths.parsed_dir)
    valid_path = resolve_path(getattr(paths, "parsed_valid", "data/parsed_valid.json"))
    cache_path = resolve_path(paths.s2_cache)
    splits_dir = resolve_path(paths.splits_dir)
    enriched_path = resolve_path(getattr(paths, "papers_enriched", "data/papers_enriched.jsonl"))

    seg2 = data_cfg.seg2
    min_body = seg2.min_body_len
    min_cites = seg2.min_citation_keys

    ctx = init_run("seg2", "bookkeeping", tags=["m-2.3", "m-2.4", "m-2.5"])
    stats: dict[str, object] = {}

    try:
        if only in ("filter", "all"):
            if rebuild_manifest:
                stats["manifest_rebuild"] = rebuild_manifest_from_parsed(
                    parsed_dir, manifest
                )
                logger.info("Rebuilt manifest: %s", stats["manifest_rebuild"])
            elif not manifest.is_file():
                raise FileNotFoundError(f"Missing manifest: {manifest}")
            stats["filter"] = apply_quality_filter(
                manifest,
                valid_path,
                min_body_len=min_body,
                min_citation_keys=min_cites,
            )
            log_metric(ctx, "filter_kept", float(stats["filter"]["kept_count"]))  # type: ignore[index]
        elif not valid_path.is_file():
            raise FileNotFoundError(
                f"Missing {valid_path}; run with --only filter or --only all first."
            )

        valid_ids = json.loads(valid_path.read_text(encoding="utf-8"))
        valid_ids, unreadable = drop_unreadable_ids(valid_ids, parsed_dir)
        if unreadable:
            stats["unreadable_dropped"] = unreadable
            logger.warning("Dropped %d unreadable parsed JSON files", len(unreadable))

        if only in ("s2", "all") and not skip_s2:
            stats["enrichment"] = enrich_papers(
                valid_ids,
                parsed_dir,
                cache_path,
                provider=provider,  # type: ignore[arg-type]
                fetch_remote=True,
                refresh_not_found=refresh_not_found,
            )
            log_metric(
                ctx,
                "enriched_files",
                float(stats["enrichment"]["enriched_files"]),  # type: ignore[index]
            )
        elif only in ("s2", "all"):
            stats["enrichment"] = enrich_papers(
                valid_ids,
                parsed_dir,
                cache_path,
                provider=provider,  # type: ignore[arg-type]
                fetch_remote=False,
            )

        if only in ("s2", "all"):
            still_missing = ids_missing_citation_count(valid_ids, parsed_dir)
            if still_missing:
                marked = mark_not_found_zero(
                    still_missing, parsed_dir, MetadataCache(cache_path)
                )
                logger.info(
                    "Zero-filled citation_count for %d / %d papers still missing metadata",
                    marked,
                    len(still_missing),
                )
            valid_ids, excluded = prune_enriched_valid_ids(valid_ids, parsed_dir)
            if any(excluded.values()):
                stats["excluded_after_enrichment"] = {
                    k: len(v) for k, v in excluded.items()
                }
                stats["excluded_ids_sample"] = {
                    k: v[:5] for k, v in excluded.items() if v
                }
                logger.warning("Excluded after enrichment: %s", stats["excluded_after_enrichment"])
            _persist_valid(valid_path, valid_ids)

        if only in ("splits", "all"):
            valid_ids, unreadable_late = drop_unreadable_ids(valid_ids, parsed_dir)
            if unreadable_late:
                stats["unreadable_dropped_late"] = unreadable_late
                logger.warning(
                    "Dropped %d late unreadable parsed JSON before splits",
                    len(unreadable_late),
                )
            still_missing = ids_missing_citation_count(valid_ids, parsed_dir)
            if still_missing and not skip_s2:
                marked = mark_not_found_zero(
                    still_missing, parsed_dir, MetadataCache(cache_path)
                )
                logger.info(
                    "Pre-split zero-fill: citation_count=0 for %d / %d papers",
                    marked,
                    len(still_missing),
                )
            elif still_missing and skip_s2:
                cache = MetadataCache(cache_path)
                restored = apply_cached_to_parsed(still_missing, parsed_dir, cache)
                still_missing = ids_missing_citation_count(valid_ids, parsed_dir)
                logger.info(
                    "Pre-split cache restore: updated %d parsed files; "
                    "%d papers still missing citation_count",
                    restored,
                    len(still_missing),
                )
            valid_ids, excluded = prune_enriched_valid_ids(valid_ids, parsed_dir)
            if excluded["unreadable"] or excluded["missing_citation_count"]:
                raise RuntimeError(
                    "Cannot build splits: "
                    f"{len(excluded['unreadable'])} unreadable, "
                    f"{len(excluded['missing_citation_count'])} missing citation_count. "
                    "Re-run with --only s2 or fix parsed JSON."
                )
            _persist_valid(valid_path, valid_ids)
            logger.info("parsed_valid.json: %d papers", len(valid_ids))

            stats["splits"] = build_splits_v31(valid_ids, parsed_dir, splits_dir)
            stats["enriched_rows"] = write_enriched_summary(
                valid_ids, parsed_dir, enriched_path
            )
            from grounded.config import load_config as _load_config
            from grounded.train.sft_data import refresh_sft_train_jsonl

            sft_cfg = _load_config("sft")
            train_path = resolve_path(sft_cfg.paths.train_jsonl)
            val_path = resolve_path(sft_cfg.paths.val_jsonl)
            stats["sft_train_jsonl"] = refresh_sft_train_jsonl(
                splits_dir / "sft.txt",
                train_path,
                parsed_dir,
                val_path=val_path,
                outline_source=sft_cfg.outline_source,  # type: ignore[arg-type]
                prompt_mode=sft_cfg.prompt_mode,  # type: ignore[arg-type]
                retrieval_fraction=sft_cfg.retrieval_fraction,
                retrieval_top_k=sft_cfg.retrieval_top_k,
                val_fraction=sft_cfg.val_fraction,
                val_seed=sft_cfg.val_seed,
                eval_grid_path=splits_dir / "eval_grid_80.txt",
                eval_holdout_path=splits_dir / "eval_holdout.txt",
            )
            log_metric(ctx, "split_index", float(stats["splits"]["counts"]["index"]))  # type: ignore[index]
            log_metric(ctx, "sft_examples", float(stats["sft_train_jsonl"]["built"]))  # type: ignore[index]

        elif only == "filter":
            _persist_valid(valid_path, valid_ids)
            logger.info("parsed_valid.json: %d papers", len(valid_ids))

        summary_path = ctx.run_dir / "seg2_summary.json"
        summary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", summary_path)
        finish_run(ctx)
        return 0
    except Exception:
        logger.exception("Segment 2 bookkeeping failed")
        finish_run(ctx)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
