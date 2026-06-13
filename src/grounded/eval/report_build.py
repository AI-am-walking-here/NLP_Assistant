"""M-8.4 — build report markdown from eval grid JSON."""

from __future__ import annotations

import json
from pathlib import Path

from grounded.config import resolve_path


def load_grid(grid_path: Path | None = None) -> dict:
    path = grid_path or resolve_path("data/eval_set/grid_runs.json")
    if not path.is_file():
        return {"systems": [], "note": "Run scripts/run_eval_grid.py first."}
    return json.loads(path.read_text(encoding="utf-8"))


def factscore_table(grid: dict) -> str:
    systems = grid.get("systems") or grid.get("results") or []
    if isinstance(systems, dict):
        systems = [{"system": k, **v} for k, v in systems.items()]
    lines = [
        "| System | FActScore (mean) | Ref overlap | Specificity | Mock gen | Mock rerank |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in systems:
        name = row.get("system") or row.get("name", "?")
        fs = row.get("factscore_mean") or row.get("factscore") or "—"
        ref_overlap = row.get("reference_overlap_mean", "—")
        specificity = row.get("specificity_ratio_mean", "—")
        mock = row.get("mock_generation", row.get("mock", "—"))
        mock_rerank = row.get("mock_reranker", "—")
        lines.append(f"| {name} | {fs} | {ref_overlap} | {specificity} | {mock} | {mock_rerank} |")
    return "\n".join(lines)


def build_report_markdown(*, grid_path: Path | None = None) -> str:
    grid = load_grid(grid_path)
    table = factscore_table(grid)
    meta = grid.get("meta", {})
    grid_rel = grid_path or resolve_path("data/eval_set/grid_runs.json")
    return "\n".join(
        [
            "# Grounded PoC — Results (draft)",
            "",
            "Headline metric: **FActScore** (70B verifier via HTTP; see docs/EVAL_WORKFLOW.md).",
            "",
            "## Eval grid",
            "",
            table,
            "",
            "## Notes",
            "",
            f"- Grid source: `{grid_rel.name}` under `data/eval_set/`",
            f"- Index mock embed: `{meta.get('index_mock_embed', 'see data/indices/index_meta.json')}`",
            "- Citation P/R removed per v3.1.",
            "- No large Hub model downloads; see STATUS.md model policy.",
            "",
        ]
    )
