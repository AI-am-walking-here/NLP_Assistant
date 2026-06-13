"""Segment 0 acceptance: layout, schemas, tracking."""

from __future__ import annotations

import json
from pathlib import Path

from grounded.config import init_run, resolve_path
from grounded.data.schema import Chunk, EvalResult, Paper


def test_paper_schema_roundtrip(tmp_path: Path) -> None:
    paper = Paper(
        arxiv_id="1601.00001",
        source="unarxive",
        title="Test",
        abstract="Abstract text.",
    )
    path = tmp_path / "1601.00001.json"
    path.write_text(paper.model_dump_json(indent=2) + "\n", encoding="utf-8")
    loaded = Paper.from_json_file(str(path))
    assert loaded.arxiv_id == "1601.00001"


def test_chunk_and_eval_result_models() -> None:
    chunk = Chunk(
        chunk_id="x:0:0",
        paper_id="x",
        section_heading="intro",
        chunk_idx=0,
        text="body",
        token_count=10,
    )
    assert chunk.paper_id == "x"
    er = EvalResult(
        system_name="naive_rag",
        prompt_set_name="grid",
        num_prompts=1,
        factscore=0.5,
    )
    assert er.system_name == "naive_rag"


def test_init_run_writes_meta(tmp_path: Path, monkeypatch) -> None:
    from grounded import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "RUNS_DIR", tmp_path)
    ctx = init_run("test", "seg0", tags=["m-0.3"], use_wandb=False)
    meta = json.loads(ctx.meta_path.read_text(encoding="utf-8"))
    assert meta["segment"] == "test"
    assert ctx.log_path.is_file() or ctx.run_dir.is_dir()


def test_project_layout(repo_root: Path) -> None:
    for rel in (
        "src/grounded",
        "configs/data.yaml",
        "scripts/bootstrap_test.py",
        "data/parsed",
        "references/graphrag",
        "references/factscore",
        ".env.example",
    ):
        assert (repo_root / rel).exists(), rel


def test_resolve_path_under_repo(repo_root: Path) -> None:
    parsed = resolve_path("data/parsed")
    assert parsed == repo_root / "data/parsed"
