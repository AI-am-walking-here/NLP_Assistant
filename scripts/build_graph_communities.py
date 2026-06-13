#!/usr/bin/env python3
"""M-6.4 — community detection + mock summaries from pilot triples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from grounded.config import finish_run, init_run, resolve_path
from grounded.graph.communities import (
    build_communities_from_triples,
    read_triples_parquet,
    write_communities_parquet,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--triples", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(triples: Path | None, out: Path | None) -> int:
    triples_path = resolve_path(triples or "data/graph/pilot_triples.parquet")
    out_path = resolve_path(out or "data/graph/communities.parquet")

    rows = read_triples_parquet(triples_path)
    communities = build_communities_from_triples(rows)
    write_communities_parquet(communities, out_path)

    ctx = init_run("seg6", "graph_communities", tags=["m-6.4"])
    summary = {"n_communities": len(communities), "out": str(out_path)}
    (ctx.run_dir / "communities_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    finish_run(ctx)
    click.echo(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
