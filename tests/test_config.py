"""Config loading and path resolution."""

from __future__ import annotations

from pathlib import Path

from grounded.config import load_config, project_root, resolve_path


def test_project_root_exists() -> None:
    assert (project_root() / "configs" / "data.yaml").is_file()


def test_load_required_configs() -> None:
    for name in ("data", "retrieval", "sft", "graph", "rankrag", "eval"):
        assert load_config(name) is not None


def test_load_data_config() -> None:
    cfg = load_config("data")
    assert cfg.category == "cs.CL"
    assert cfg.paths.parsed_dir == "data/parsed"


def test_load_retrieval_config() -> None:
    cfg = load_config("retrieval")
    assert cfg.chunk_size == 512
    assert cfg.top_k == 8


def test_resolve_path_relative() -> None:
    p = resolve_path("data/parsed")
    assert p.is_dir() or p.parent.is_dir()


def test_all_config_yamls_parse() -> None:
    configs = Path(project_root()) / "configs"
    for path in configs.glob("*.yaml"):
        load_config(path.stem)
