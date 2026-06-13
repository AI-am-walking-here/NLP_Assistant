"""Run directory naming helpers."""

from __future__ import annotations

from grounded.tracking.runs_util import latest_seg4_eval_dirs, parse_seg4_eval_name


def test_parse_seg4_eval_name() -> None:
    assert parse_seg4_eval_name("seg4_eval_naive_rag_2026-05-26-1402") == (
        "naive_rag",
        "2026-05-26-1402",
    )
    assert parse_seg4_eval_name("seg3_smoke_2026-05-26-1005") is None


def test_latest_seg4_eval_dirs(repo_root) -> None:
    runs = repo_root / "runs"
    if not runs.is_dir():
        return
    latest = latest_seg4_eval_dirs(runs)
    for system, path in latest.items():
        assert path.name.startswith(f"seg4_eval_{system}_")
