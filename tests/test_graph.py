"""Graph pilot schema and gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.graph.gate import extrapolate_gpu_hours
from grounded.graph.mock_extract import extract_triple_mock
from grounded.graph.pilot import sample_pilot_ids


def test_sample_pilot_ids_deterministic() -> None:
    ids = [f"id{i}" for i in range(100)]
    a = sample_pilot_ids(ids, n=10, seed=1)
    b = sample_pilot_ids(ids, n=10, seed=1)
    assert a == b


def test_gate_pass_when_under_cap() -> None:
    gate = extrapolate_gpu_hours(
        pilot_chunks=1000,
        pilot_seconds=2.0,
        target_chunks=2000,
        gate_hours=50.0,
    )
    assert gate["keep_graph"] is True
    assert gate["projected_gpu_hours"] < 50.0


def test_holdout_reserve_disjoint_from_eval_grid(repo_root: Path) -> None:
    from grounded.train.rankrag_data import holdout_reserve_ids, load_id_list

    holdout_path = repo_root / "data/splits/eval_holdout.txt"
    grid_path = repo_root / "data/splits/eval_grid_80.txt"
    if not holdout_path.is_file() or not grid_path.is_file():
        pytest.skip("split files missing")

    reserve = holdout_reserve_ids(holdout_path, grid_path)
    grid = set(load_id_list(grid_path))
    assert reserve
    assert not (set(reserve) & grid)


def test_mock_extract_finds_entities() -> None:
    text = (
        "We propose a transformer model for machine translation on the WMT dataset "
        "and improve over BERT baselines."
    )
    triple = extract_triple_mock("p:0:0", "p", text)
    assert len(triple.entities) >= 1
