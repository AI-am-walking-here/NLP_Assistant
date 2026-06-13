import json

from grounded.data.filter import (
    _should_keep_member,
    build_unarxive_manifest,
    compute_source_target,
    materialize_unarxive_records,
    normalize_arxiv_id,
    sample_records_by_fraction,
    sample_records_for_source,
)


def test_normalize_arxiv_id():
    assert normalize_arxiv_id("arXiv:2401.12345v2") == "2401.12345"
    assert normalize_arxiv_id("2401.12345") == "2401.12345"
    assert normalize_arxiv_id("not-an-id") is None


def test_should_keep_member():
    skip_extensions = {".png", ".pdf", ".jpg", ".eps", ".svg"}
    assert _should_keep_member("paper/main.tex", skip_extensions)
    assert _should_keep_member("paper/figure.png", skip_extensions) is False
    assert _should_keep_member("paper/Makefile", skip_extensions)
    assert _should_keep_member("paper/data.bib", skip_extensions)


def test_compute_source_target():
    assert compute_source_target(50000, 0.9) == 45000
    assert compute_source_target(50000, 0.1) == 5000
    assert compute_source_target(None, 0.5) is None


def test_sample_records_for_source_uses_overall_target():
    records = [{"id": f"2401.{i:05d}", "year": 2024} for i in range(100)]
    sampled, target = sample_records_for_source(
        records, overall_target=50000, source_fraction=0.1, random_seed=7
    )
    assert target == 5000
    assert len(sampled) == 100  # pool smaller than target → keep all


def test_sample_records_by_fraction_is_deterministic():
    records = [{"id": f"2401.{i:05d}", "year": 2024} for i in range(10)]
    sample_a = sample_records_by_fraction(records, paper_fraction=0.3, random_seed=7)
    sample_b = sample_records_by_fraction(records, paper_fraction=0.3, random_seed=7)
    assert [rec["id"] for rec in sample_a] == [rec["id"] for rec in sample_b]
    assert len(sample_a) == 3


def test_paper_count_target_overrides_fraction():
    records = [{"id": f"2401.{i:05d}", "year": 2024} for i in range(10)]
    sample = sample_records_by_fraction(records, paper_fraction=0.1, random_seed=7, paper_count_target=6)
    assert len(sample) == 6


def test_paper_count_target_none_uses_fraction():
    records = [{"id": f"2401.{i:05d}", "year": 2024} for i in range(10)]
    sample = sample_records_by_fraction(records, paper_fraction=0.4, random_seed=7, paper_count_target=None)
    assert len(sample) == 4


def test_unarxive_manifest_and_materialize(tmp_path):
    root = tmp_path / "unarxive"
    root.mkdir()
    shard = root / "sample.jsonl"
    records = [
        {
            "paper_id": "arXiv:2401.00001v1",
            "title": "First",
            "abstract": "Abstract one",
            "text": "Body one",
        },
        {
            "paper_id": "2401.00002",
            "title": "Second",
            "abstract": "Abstract two",
            "text": "Body two",
        },
    ]
    shard.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    manifest = build_unarxive_manifest(
        root_dir=root,
        shard_glob="**/*.jsonl",
        allowed_ids={"2401.00001"},
        id_fields=["paper_id"],
    )
    manifest_path = tmp_path / "unarxive_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stats = materialize_unarxive_records(
        manifest_path=manifest_path,
        output_dir=tmp_path / "materialized",
        ledger_path=tmp_path / "materialized.txt",
        id_fields=["paper_id"],
        title_fields=["title"],
        abstract_fields=["abstract"],
        text_fields=["text"],
    )

    paper_dir = tmp_path / "materialized" / "2401.00001"
    assert manifest["stats"]["num_matched_ids"] == 1
    assert stats["materialized_this_run"] == 1
    assert (paper_dir / "paper.json").exists()
    assert (paper_dir / "paper.txt").read_text(encoding="utf-8") == "First\n\nAbstract one\n\nBody one"
