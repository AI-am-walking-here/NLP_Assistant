"""Shared pytest fixtures for the grounded test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.config import project_root


@pytest.fixture
def repo_root() -> Path:
    return project_root()


@pytest.fixture
def sample_parsed_path(repo_root: Path) -> Path:
    path = repo_root / "data/parsed/1601.02539.json"
    if not path.is_file():
        pytest.skip("sample parsed paper missing under data/parsed/")
    return path
