"""Canonical portable data paths (safe to commit / upload via git)."""

from __future__ import annotations

from pathlib import Path

# Relative to llm-assistant-final project root.
PORTABLE_REL_PATHS: tuple[str, ...] = (
    "data/sft/train.jsonl",
    "data/sft/val.jsonl",
    "data/sft/dpo_pairs.jsonl",
    "data/splits/index.txt",
    "data/splits/eval_holdout.txt",
    "data/splits/eval_grid_80.txt",
    "data/splits/sft.txt",
    "data/rankrag/train.jsonl",
    "data/parsed_valid.json",
    "data/papers_enriched.jsonl",
    "data/eval_set/prompts.jsonl",
    "data/eval_set/grid_runs.json",
    "data/eval_set/runs_archive.json",
    "data/eval_set/factscore_audit.jsonl",
    "data/eval_set/human_eval_template.jsonl",
    "data/eval_set/sft_smoke_comparison.json",
)


def portable_paths(project_root: Path) -> list[Path]:
    return [project_root / rel for rel in PORTABLE_REL_PATHS]


def archived_per_prompt_dir(project_root: Path) -> Path:
    return project_root / "data/eval_set/archived_per_prompt"
