#!/usr/bin/env python3
"""Segment 1 CLI — thin entrypoints for M-1.1 through M-1.4."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import boto3
import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grounded.config import append_log, init_run, load_config, load_dotenv, resolve_path, write_json  # noqa: E402
from grounded.data.filter import (  # noqa: E402
    build_filtered_manifest,
    build_unarxive_manifest,
    compute_source_target,
    download_kaggle_metadata,
    download_manifest,
    extract_all_tarballs,
    filter_cs_cl_records,
    load_cs_cl_records,
    load_manifest_filtered,
    materialize_unarxive_records,
    parse_manifest_xml,
    sample_records_for_source,
    select_records_for_year_range,
    write_cs_cl_ids,
    write_manifest_filtered,
    write_unarxive_manifest,
)
from grounded.data.s3_pull import (  # noqa: E402
    CostTracker,
    S3TarballDownloader,
    TarballPipelineOptions,
    load_ledger,
    tarball_local_path,
    _size_is_valid,
)
from grounded.data.unarxive_download import download_unarxive_shards  # noqa: E402


def _unarxive_root_dir(config) -> Path:
    root = Path(config.sources.unarxive.root_dir)
    return root if root.is_absolute() else ROOT / root


def _s3_target_ids(config) -> set[str]:
    source_records = select_records_for_year_range(
        load_cs_cl_records(resolve_path(config, "cs_cl_ids")),
        config.sources.arxiv_s3.year_min,
        config.sources.arxiv_s3.year_max,
    )
    source_records, _ = sample_records_for_source(
        source_records,
        overall_target=config.paper_count_target,
        source_fraction=config.sources.arxiv_s3.paper_fraction,
        random_seed=config.sources.arxiv_s3.random_seed,
    )
    return {rec["id"] for rec in source_records}


def _s3_cleaning_options(config) -> set[str]:
    return {ext.lower() for ext in config.cleaning.skip_extensions}


def _s3_pipeline_options(config) -> TarballPipelineOptions | None:
    pipeline = config.sources.arxiv_s3.pipeline
    if not pipeline.extract_after_download:
        return None
    return TarballPipelineOptions(
        target_ids=_s3_target_ids(config),
        output_dir=resolve_path(config, "tex_extracted"),
        extracted_ledger_path=resolve_path(config, "extracted_ledger"),
        skip_extensions=_s3_cleaning_options(config),
        extract_after_download=True,
        delete_tarball_after_extract=pipeline.delete_tarball_after_extract,
    )


@click.group()
def cli() -> None:
    """arXiv LaTeX data acquisition (Segment 1)."""
    load_dotenv()


def _aws_credential_status() -> str:
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        return "not found"
    return f"found via {credentials.method}"


def _enabled_source_year_span(config) -> tuple[int, int]:
    year_mins: list[int] = []
    year_maxs: list[int] = []
    if config.sources.arxiv_s3.enabled:
        year_mins.append(config.sources.arxiv_s3.year_min)
        year_maxs.append(config.sources.arxiv_s3.year_max)
    if config.sources.unarxive.enabled:
        year_mins.append(config.sources.unarxive.year_min)
        year_maxs.append(config.sources.unarxive.year_max)
    if not year_mins:
        raise click.ClickException("No acquisition sources are enabled in configs/data.yaml.")
    return min(year_mins), max(year_maxs)


def _enabled_source_fraction_sum(config) -> float:
    total = 0.0
    if config.sources.unarxive.enabled:
        total += config.sources.unarxive.paper_fraction
    if config.sources.arxiv_s3.enabled:
        total += config.sources.arxiv_s3.paper_fraction
    return total


def _warn_fraction_sum(config) -> None:
    total = _enabled_source_fraction_sum(config)
    if config.paper_count_target is not None and abs(total - 1.0) > 0.01:
        click.echo(
            click.style(
                f"WARNING: enabled source paper_fraction values sum to {total:.2f}, not 1.0. "
                "Each source still receives paper_count_target × its fraction.",
                fg="yellow",
            )
        )


def _print_allocation(config, source_name: str, source_fraction: float, kept: int, pool: int) -> None:
    target = compute_source_target(config.paper_count_target, source_fraction)
    click.echo(
        f"Allocation [{source_name}]: target={target} "
        f"({source_fraction:.0%} of overall {config.paper_count_target}), "
        f"kept {kept} / {pool} available in year window"
    )


@cli.command("filter-metadata")
@click.option("--download", is_flag=True, help="Download metadata from Kaggle first.")
def filter_metadata(download: bool) -> None:
    """M-1.1 — Filter Kaggle metadata to cs.CL 2020–2026."""
    config = load_config("data")
    metadata_path = resolve_path(config, "kaggle_metadata")
    output_path = resolve_path(config, "cs_cl_ids")
    year_min, year_max = _enabled_source_year_span(config)

    if not _needs_filter_metadata(config, download_metadata=download):
        records = load_cs_cl_records(output_path)
        click.echo(f"Skipping filter; using {len(records)} IDs from {output_path}")
        return

    if download or not metadata_path.exists():
        click.echo(f"Downloading Kaggle dataset {config.kaggle.dataset} ...")
        metadata_path = download_kaggle_metadata(config.kaggle.dataset, metadata_path)

    click.echo(f"Filtering {metadata_path} ...")
    records = filter_cs_cl_records(
        metadata_path,
        category=config.category,
        category_match=config.category_match,
        year_min=year_min,
        year_max=year_max,
    )
    write_cs_cl_ids(output_path, records)
    click.echo(f"Wrote {len(records)} IDs to {output_path}")


@cli.command("build-manifest")
@click.option("--download-manifest", "fetch_manifest", is_flag=True, help="Pull manifest from S3.")
def build_manifest_cmd(fetch_manifest: bool) -> None:
    """M-1.2 — Cross-reference IDs with arXiv S3 manifest."""
    config = load_config("data")
    if not config.sources.arxiv_s3.enabled:
        raise click.ClickException("arxiv_s3 is disabled in configs/data.yaml.")
    cs_cl_path = resolve_path(config, "cs_cl_ids")
    manifest_path = resolve_path(config, "manifest_xml")
    output_path = resolve_path(config, "manifest_filtered")

    if not cs_cl_path.exists():
        raise click.ClickException(f"Missing {cs_cl_path}. Run filter-metadata first.")

    if fetch_manifest or not manifest_path.exists():
        click.echo("Downloading arXiv source manifest from S3 ...")
        download_manifest(
            bucket=config.sources.arxiv_s3.bucket,
            manifest_key=config.sources.arxiv_s3.manifest_key,
            output_path=manifest_path,
            region=config.sources.arxiv_s3.region,
            request_payer=config.sources.arxiv_s3.request_payer,
        )
    else:
        click.echo(f"Using local manifest {manifest_path}")

    _warn_fraction_sum(config)
    source_records = select_records_for_year_range(
        load_cs_cl_records(cs_cl_path),
        config.sources.arxiv_s3.year_min,
        config.sources.arxiv_s3.year_max,
    )
    total_source_records = len(source_records)
    source_records, source_target = sample_records_for_source(
        source_records,
        overall_target=config.paper_count_target,
        source_fraction=config.sources.arxiv_s3.paper_fraction,
        random_seed=config.sources.arxiv_s3.random_seed,
    )
    _print_allocation(
        config, "arxiv_s3", config.sources.arxiv_s3.paper_fraction, len(source_records), total_source_records
    )
    payload = build_filtered_manifest(
        parse_manifest_xml(manifest_path),
        {rec["id"] for rec in source_records},
        hard_cap_usd=config.sources.arxiv_s3.cost.hard_cap_usd,
        egress_per_gb_usd=config.sources.arxiv_s3.cost.egress_per_gb_usd,
        get_request_usd=config.sources.arxiv_s3.cost.get_request_usd,
        random_seed=config.sources.arxiv_s3.random_seed,
    )
    write_manifest_filtered(output_path, payload)

    stats = payload["stats"]
    click.echo(f"Mapped {stats['num_mapped_ids']} / {stats['num_cs_cl_ids']} IDs")
    click.echo(
        f"Candidate tarballs: {stats['num_candidate_tarballs']} "
        f"(~{stats['candidate_estimated_download_gb']} GB, "
        f"${stats['candidate_estimated_cost_usd']:.2f})"
    )
    click.echo(
        f"Selected tarballs: {stats['num_tarballs']} "
        f"(~{stats['estimated_download_gb']} GB, "
        f"${stats['estimated_cost_usd']:.2f}, {stats['num_selected_ids']} mapped IDs)"
    )
    if stats["selection_mode"] == "random_budget_sample":
        click.echo(
            click.style(
                "Full eligible corpus exceeds the hard cap; wrote a deterministic random budget sample.",
                fg="yellow",
            )
        )
    click.echo(f"Wrote {output_path}")


@cli.command("build-unarxive-manifest")
def build_unarxive_manifest_cmd() -> None:
    """Build a local manifest of matching unarXive records."""
    config = load_config("data")
    if not config.sources.unarxive.enabled:
        raise click.ClickException("unarxive is disabled in configs/data.yaml.")
    cs_cl_path = resolve_path(config, "cs_cl_ids")
    if not cs_cl_path.exists():
        raise click.ClickException(f"Missing {cs_cl_path}. Run filter-metadata first.")

    _warn_fraction_sum(config)
    source_records = select_records_for_year_range(
        load_cs_cl_records(cs_cl_path),
        config.sources.unarxive.year_min,
        config.sources.unarxive.year_max,
    )
    total_source_records = len(source_records)
    source_records, _ = sample_records_for_source(
        source_records,
        overall_target=config.paper_count_target,
        source_fraction=config.sources.unarxive.paper_fraction,
        random_seed=config.sources.unarxive.random_seed,
    )
    _print_allocation(
        config, "unarxive", config.sources.unarxive.paper_fraction, len(source_records), total_source_records
    )
    root_dir = _unarxive_root_dir(config)
    try:
        payload = build_unarxive_manifest(
            root_dir=root_dir,
            shard_glob=config.sources.unarxive.shard_glob,
            allowed_ids={rec["id"] for rec in source_records},
            id_fields=config.sources.unarxive.id_fields,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    output_path = resolve_path(config, "unarxive_manifest")
    write_unarxive_manifest(output_path, payload)
    stats = payload["stats"]
    click.echo(
        f"unarXive manifest: matched {stats['num_matched_ids']} / {stats['num_allowed_ids']} IDs "
        f"from {stats['num_scanned_files']} files"
    )
    click.echo(f"Wrote {output_path}")


@cli.command("download-unarxive")
@click.option("--force", is_flag=True, help="Re-download and re-extract even if shards exist.")
def download_unarxive_cmd(force: bool) -> None:
    """Download unarXive JSONL shards from Zenodo and extract to data/unarxive/."""
    config = load_config("data")
    if not config.sources.unarxive.enabled:
        raise click.ClickException("unarxive is disabled in configs/data.yaml.")
    if (
        not force
        and config.sources.unarxive.delete_shards_after_materialize
        and _unarxive_materialization_ready(config)
    ):
        click.echo("Skipped unarXive download; materialized outputs already exist.")
        return

    root_dir = _unarxive_root_dir(config)
    download_cfg = config.sources.unarxive.download
    click.echo(f"unarXive shards directory: {root_dir}")
    click.echo(f"Source: {download_cfg.url}")

    try:
        stats = download_unarxive_shards(
            root_dir,
            url=download_cfg.url,
            archive_name=download_cfg.archive_name,
            shard_glob=config.sources.unarxive.shard_glob,
            min_shards=download_cfg.min_shards,
            force=force,
            keep_archive=download_cfg.keep_archive,
            max_resume_attempts=None if download_cfg.resume_forever else 5,
            retry_delay_s=download_cfg.retry_delay_s,
            max_retry_delay_s=download_cfg.max_retry_delay_s,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    if stats["skipped"]:
        click.echo(f"Skipped download; found {stats['shard_count']} existing JSONL shards.")
        return

    downloaded_gb = stats["downloaded_bytes"] / (1024**3)
    click.echo(
        f"Downloaded {downloaded_gb:.2f} GB, extracted {stats['extracted_shards']} shards "
        f"({stats['shard_count']} total under {root_dir})."
    )


@cli.command("download")
@click.option("--dry-run", is_flag=True, help="Print plan without downloading.")
@click.option("--limit", type=int, default=None, help="Download at most N tarballs.")
def download(dry_run: bool, limit: int | None) -> None:
    """M-1.3 — Download LaTeX source tarballs from S3."""
    config = load_config("data")
    if not config.sources.arxiv_s3.enabled:
        raise click.ClickException("arxiv_s3 is disabled in configs/data.yaml.")
    manifest_path = resolve_path(config, "manifest_filtered")
    if not manifest_path.exists():
        raise click.ClickException(f"Missing {manifest_path}. Run build-manifest first.")

    payload = load_manifest_filtered(manifest_path)
    keys = payload["tarballs"][:limit] if limit else payload["tarballs"]
    key_sizes = payload.get("tarball_sizes_bytes", {})
    stats = payload.get("stats", {})
    estimated_cost = sum(
        (
            key_sizes.get(key, 0) / (1024**3)
        )
        * config.sources.arxiv_s3.cost.egress_per_gb_usd
        + config.sources.arxiv_s3.cost.get_request_usd
        for key in keys
    )
    estimated_gb = sum(key_sizes.get(key, 0) for key in keys) / (1024**3)
    if estimated_cost > config.sources.arxiv_s3.cost.hard_cap_usd:
        raise click.ClickException(
            f"Selected download is estimated at ${estimated_cost:.2f}, "
            f"above cap ${config.sources.arxiv_s3.cost.hard_cap_usd:.2f}. Rebuild the manifest with a lower budget or use --limit."
        )
    if not limit and stats.get("estimated_cost_usd", 0) > config.sources.arxiv_s3.cost.hard_cap_usd:
        click.echo(
            click.style(
                f"Estimated cost (${stats['estimated_cost_usd']:.2f}) exceeds "
                f"cap (${config.sources.arxiv_s3.cost.hard_cap_usd:.2f}). Pass --limit for a pilot.",
                fg="yellow",
            )
        )

    raw_dir = resolve_path(config, "raw_tarballs")
    downloaded_ledger = resolve_path(config, "downloaded_ledger")
    extracted_ledger = resolve_path(config, "extracted_ledger")
    pipeline = _s3_pipeline_options(config)
    ledger = load_ledger(extracted_ledger if pipeline else downloaded_ledger)
    already_done = 0
    for key in keys:
        dest = tarball_local_path(raw_dir, key)
        expected_size = key_sizes.get(key)
        if key in ledger:
            if pipeline or (dest.exists() and dest.stat().st_size > 0 and _size_is_valid(
                dest.stat().st_size, expected_size, None
            )):
                already_done += 1

    remaining = len(keys) - already_done
    if pipeline:
        peak_gb = config.sources.arxiv_s3.max_workers * 0.5
        click.echo(
            f"Pipeline: extract after each download, "
            f"delete tarball={'yes' if pipeline.delete_tarball_after_extract else 'no'} "
            f"(peak raw disk ~{peak_gb:.1f} GB with {config.sources.arxiv_s3.max_workers} workers)"
        )
    click.echo(
        f"Download plan: {len(keys)} tarballs, {estimated_gb:.2f} GB, "
        f"~${estimated_cost:.2f} estimated, cap ${config.sources.arxiv_s3.cost.hard_cap_usd:.2f}"
    )
    click.echo(f"Resume status: {already_done} complete, {remaining} remaining")
    click.echo(f"AWS credentials: {_aws_credential_status()}")
    if dry_run:
        return

    cost = CostTracker(
        egress_per_gb_usd=config.sources.arxiv_s3.cost.egress_per_gb_usd,
        get_request_usd=config.sources.arxiv_s3.cost.get_request_usd,
        hard_cap_usd=config.sources.arxiv_s3.cost.hard_cap_usd,
    )
    run_dir = init_run("seg1", "s3_pull", {"data": "configs/data.yaml"})
    append_log(run_dir, f"Starting download of {len(keys)} tarballs")

    try:
        transfer = config.sources.arxiv_s3.transfer
        click.echo(
            f"S3 transfer: multipart >= {transfer.multipart_threshold_mb} MB, "
            f"chunk {transfer.multipart_chunksize_mb} MB, "
            f"{transfer.max_concurrency} parts/file, "
            f"pool={transfer.max_pool_connections}"
        )
        S3TarballDownloader(
            bucket=config.sources.arxiv_s3.bucket,
            region=config.sources.arxiv_s3.region,
            request_payer=config.sources.arxiv_s3.request_payer,
            raw_dir=raw_dir,
            ledger_path=downloaded_ledger,
            cost=cost,
            transfer=transfer,
            max_workers=config.sources.arxiv_s3.max_workers,
            max_retries=config.sources.arxiv_s3.max_retries,
        ).download_tarballs(
            keys,
            key_sizes=key_sizes,
            run_dir=run_dir,
            pipeline=pipeline,
        )
    except KeyboardInterrupt as exc:
        ledger_path = extracted_ledger if pipeline else downloaded_ledger
        click.echo(
            f"Interrupted. Completed tarballs were written to {ledger_path} "
            "and the next run will resume from there.",
            err=True,
        )
        os._exit(130)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Downloaded {cost.requests} tarballs ({cost.gb_downloaded:.2f} GB, ${cost.total_cost_usd:.2f})")
    click.echo(f"Cost log: {run_dir / 'cost.json'}")


@cli.command("extract")
def extract() -> None:
    """M-1.4 — Extract cs.CL papers from downloaded tarballs."""
    config = load_config("data")
    if not config.sources.arxiv_s3.enabled:
        raise click.ClickException("arxiv_s3 is disabled in configs/data.yaml.")
    manifest_path = resolve_path(config, "manifest_filtered")
    raw_dir = resolve_path(config, "raw_tarballs")
    if not manifest_path.exists():
        raise click.ClickException(f"Missing {manifest_path}. Run build-manifest first.")
    if not raw_dir.exists():
        raise click.ClickException(f"Missing {raw_dir}. Run download first.")

    payload = load_manifest_filtered(manifest_path)
    skip_extensions = _s3_cleaning_options(config)
    run_dir = init_run("seg1", "extract", {"data": "configs/data.yaml"})
    stats = extract_all_tarballs(
        raw_dir=raw_dir,
        tarball_keys=payload["tarballs"],
        target_ids=_s3_target_ids(config),
        output_dir=resolve_path(config, "tex_extracted"),
        ledger_path=resolve_path(config, "extracted_ledger"),
        skip_extensions=skip_extensions,
        delete_tarball_after_extract=config.sources.arxiv_s3.pipeline.delete_tarball_after_extract,
    )
    write_json(run_dir, "results.json", stats)
    click.echo(f"Paper directories: {stats['total_paper_dirs']}")
    click.echo(f"Extracted this run: {stats['papers_extracted_this_run']}")
    if stats.get("failed_tarballs"):
        click.echo(click.style(f"Skipped failed tarballs: {len(stats['failed_tarballs'])}", fg="yellow"))
    if stats.get("interrupted"):
        click.echo(
            click.style(
                f"Interrupted. Completed tarballs were written to {resolve_path(config, 'extracted_ledger')} "
                "and the next extract run will resume.",
                fg="yellow",
            )
        )


@cli.command("materialize-unarxive")
def materialize_unarxive_cmd() -> None:
    """Materialize matched unarXive papers into per-paper artifacts."""
    config = load_config("data")
    if not config.sources.unarxive.enabled:
        raise click.ClickException("unarxive is disabled in configs/data.yaml.")
    manifest_path = resolve_path(config, "unarxive_manifest")
    if not manifest_path.exists():
        raise click.ClickException(f"Missing {manifest_path}. Run build-unarxive-manifest first.")

    run_dir = init_run("seg1", "unarxive", {"data": "configs/data.yaml"})
    stats = materialize_unarxive_records(
        manifest_path=manifest_path,
        output_dir=resolve_path(config, "unarxive_extracted"),
        ledger_path=resolve_path(config, "unarxive_materialized_ledger"),
        id_fields=config.sources.unarxive.id_fields,
        title_fields=config.sources.unarxive.title_fields,
        abstract_fields=config.sources.unarxive.abstract_fields,
        text_fields=config.sources.unarxive.text_fields,
        delete_shards_after_materialize=config.sources.unarxive.delete_shards_after_materialize,
    )
    write_json(run_dir, "results.json", stats)
    click.echo(f"unarXive paper directories: {stats['total_paper_dirs']}")
    click.echo(f"Materialized this run: {stats['materialized_this_run']}")


def _needs_filter_metadata(config, *, download_metadata: bool) -> bool:
    if download_metadata:
        return True
    output_path = resolve_path(config, "cs_cl_ids")
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime
    metadata_path = resolve_path(config, "kaggle_metadata")
    config_path = ROOT / "configs" / "data.yaml"
    if metadata_path.exists() and metadata_path.stat().st_mtime > output_mtime:
        return True
    if config_path.exists() and config_path.stat().st_mtime > output_mtime:
        return True
    return False


def _unarxive_materialization_ready(config) -> bool:
    output_dir = resolve_path(config, "unarxive_extracted")
    ledger_path = resolve_path(config, "unarxive_materialized_ledger")
    if not output_dir.exists() or not ledger_path.exists():
        return False
    if not load_ledger(ledger_path):
        return False
    return any(path.is_dir() for path in output_dir.iterdir())


def _run_all_steps(
    *,
    download_metadata: bool,
    dry_run_download: bool,
    limit: int | None,
    skip_extract: bool,
    force_unarxive_download: bool,
    force_download_manifest: bool,
) -> list[list[str]]:
    """Build the full acquisition pipeline for every enabled source."""
    config = load_config("data")
    if not config.sources.unarxive.enabled and not config.sources.arxiv_s3.enabled:
        raise click.ClickException("No acquisition sources are enabled in configs/data.yaml.")

    script = str(Path(__file__).resolve())
    steps: list[list[str]] = []

    if _needs_filter_metadata(config, download_metadata=download_metadata):
        metadata_path = resolve_path(config, "kaggle_metadata")
        needs_metadata_download = download_metadata or not metadata_path.exists()
        steps.append(
            [sys.executable, script, "filter-metadata"] + (["--download"] if needs_metadata_download else [])
        )

    if config.sources.unarxive.enabled:
        if not (
            not force_unarxive_download
            and config.sources.unarxive.delete_shards_after_materialize
            and _unarxive_materialization_ready(config)
        ):
            download_unarxive_cmd = [sys.executable, script, "download-unarxive"]
            if force_unarxive_download:
                download_unarxive_cmd.append("--force")
            steps.extend(
                [
                    download_unarxive_cmd,
                    [sys.executable, script, "build-unarxive-manifest"],
                    [sys.executable, script, "materialize-unarxive"],
                ]
            )

    if config.sources.arxiv_s3.enabled:
        manifest_path = resolve_path(config, "manifest_xml")
        build_manifest_cmd = [sys.executable, script, "build-manifest"]
        if force_download_manifest or not manifest_path.exists():
            build_manifest_cmd.append("--download-manifest")
        steps.append(build_manifest_cmd)
        steps.append(
            [sys.executable, script, "download"]
            + (["--dry-run"] if dry_run_download else [])
            + (["--limit", str(limit)] if limit else [])
        )
        pipeline = config.sources.arxiv_s3.pipeline
        inline_extract = pipeline.extract_after_download and not dry_run_download
        if not skip_extract and not inline_extract and not dry_run_download:
            steps.append([sys.executable, script, "extract"])

    return steps


@cli.command("run-all")
@click.option("--download-metadata", is_flag=True, help="Force Kaggle metadata download.")
@click.option("--force-unarxive-download", is_flag=True, help="Re-download unarXive shards from Zenodo.")
@click.option("--force-download-manifest", is_flag=True, help="Re-download arXiv S3 manifest from AWS.")
@click.option("--dry-run-download", is_flag=True, help="Print S3 download plan without downloading.")
@click.option("--limit", type=int, default=None, help="Download at most N S3 tarballs.")
@click.option("--skip-extract", is_flag=True, help="Skip S3 tarball extraction.")
def run_all(
    download_metadata: bool,
    force_unarxive_download: bool,
    force_download_manifest: bool,
    dry_run_download: bool,
    limit: int | None,
    skip_extract: bool,
) -> None:
    """Run the full end-to-end pipeline for all enabled sources."""
    steps = _run_all_steps(
        download_metadata=download_metadata,
        dry_run_download=dry_run_download,
        limit=limit,
        skip_extract=skip_extract,
        force_unarxive_download=force_unarxive_download,
        force_download_manifest=force_download_manifest,
    )

    for cmd in steps:
        click.echo(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    click.echo("\nSegment 1 complete.")


if __name__ == "__main__":
    cli()
