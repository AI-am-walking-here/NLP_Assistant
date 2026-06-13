#!/usr/bin/env python3
"""Segment 0 acceptance: init_run + config load + metric log."""

from __future__ import annotations

import json
import sys

from grounded.config import finish_run, init_run, load_config, log_metric, resolve_path
from grounded.data.schema import Paper


def main() -> int:
    data_cfg = load_config("data")
    print(f"category={data_cfg.category}")

    manifest = resolve_path(data_cfg.paths.parsed_manifest)
    if not manifest.is_file():
        print(f"WARN: missing {manifest}", file=sys.stderr)
    else:
        print(f"parsed_manifest={manifest}")

    ctx = init_run("test", "bootstrap", tags=["segment0"])
    log_metric(ctx, "bootstrap_ok", 1.0)
    finish_run(ctx)

    meta = json.loads(ctx.meta_path.read_text(encoding="utf-8"))
    print(f"run_dir={ctx.run_dir}")
    print(f"meta.json written; wandb_url={meta.get('wandb_url')}")

    sample = resolve_path(data_cfg.paths.parsed_dir) / "1601.02539.json"
    if sample.is_file():
        paper = Paper.from_json_file(str(sample))
        print(f"schema_ok arxiv_id={paper.arxiv_id} sections={len(paper.sections)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
