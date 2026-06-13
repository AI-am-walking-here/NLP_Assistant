import io
import json
import tarfile

from grounded.data.filter import process_tarball


def _write_sample_tarball(path, arxiv_id: str = "2401.00001") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex_bytes = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
    with tarfile.open(path, mode="w:") as archive:
        info = tarfile.TarInfo(name=f"{arxiv_id}/main.tex")
        info.size = len(tex_bytes)
        archive.addfile(info, fileobj=io.BytesIO(tex_bytes))
        noise = tarfile.TarInfo(name="2401.99999/main.tex")
        noise.size = len(tex_bytes)
        archive.addfile(noise, fileobj=io.BytesIO(tex_bytes))


def _write_arxiv_bulk_tarball(path, arxiv_id: str = "2401.00001") -> None:
    """Simulate arXiv S3 bulk layout: outer tar with {id}.gz inner tar.gz bundles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tex_bytes = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
    png_bytes = b"PNG"

    inner_buf = io.BytesIO()
    with tarfile.open(fileobj=inner_buf, mode="w:gz") as inner:
        tex = tarfile.TarInfo(name="main.tex")
        tex.size = len(tex_bytes)
        inner.addfile(tex, fileobj=io.BytesIO(tex_bytes))
        fig = tarfile.TarInfo(name="fig.png")
        fig.size = len(png_bytes)
        inner.addfile(fig, fileobj=io.BytesIO(png_bytes))

    gz_payload = inner_buf.getvalue()
    with tarfile.open(path, mode="w:") as outer:
        gz_member = tarfile.TarInfo(name=f"{arxiv_id}.gz")
        gz_member.size = len(gz_payload)
        outer.addfile(gz_member, fileobj=io.BytesIO(gz_payload))


def test_process_tarball_extracts_target_and_keeps_archive(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "extracted"
    ledger = tmp_path / "extracted.txt"
    tarball = raw_dir / "src_arXiv_src_2401_001.tar"
    _write_sample_tarball(tarball)
    skip_extensions = {".png", ".pdf"}

    extracted = process_tarball(
        tarball,
        "src/arXiv_src_2401_001.tar",
        target_ids={"2401.00001"},
        output_dir=output_dir,
        extracted_ledger_path=ledger,
        skip_extensions=skip_extensions,
        delete_tarball=False,
    )

    assert extracted == {"2401.00001"}
    assert (output_dir / "2401.00001" / "main.tex").exists()
    assert tarball.exists()
    assert "src/arXiv_src_2401_001.tar" in ledger.read_text(encoding="utf-8")


def test_process_tarball_extracts_arxiv_gz_bundle_and_skips_figures(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "extracted"
    ledger = tmp_path / "extracted.txt"
    tarball = raw_dir / "src_arXiv_src_2401_001.tar"
    _write_arxiv_bulk_tarball(tarball)
    skip_extensions = {".png", ".pdf", ".jpg"}

    extracted = process_tarball(
        tarball,
        "src/arXiv_src_2401_001.tar",
        target_ids={"2401.00001"},
        output_dir=output_dir,
        extracted_ledger_path=ledger,
        skip_extensions=skip_extensions,
        delete_tarball=False,
    )

    assert extracted == {"2401.00001"}
    assert (output_dir / "2401.00001" / "main.tex").exists()
    assert not (output_dir / "2401.00001" / "fig.png").exists()


def test_materialize_unarxive_deletes_shard_when_configured(tmp_path):
    from grounded.data.filter import materialize_unarxive_records

    root = tmp_path / "unarxive"
    root.mkdir()
    shard = root / "sample.jsonl"
    record = {
        "paper_id": "2401.00001",
        "title": "First",
        "abstract": "Abstract one",
        "text": "Body one",
    }
    shard.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = {
        "matched_ids": {"2401.00001": str(shard)},
        "stats": {"num_matched_ids": 1},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stats = materialize_unarxive_records(
        manifest_path=manifest_path,
        output_dir=tmp_path / "materialized",
        ledger_path=tmp_path / "materialized.txt",
        id_fields=["paper_id"],
        title_fields=["title"],
        abstract_fields=["abstract"],
        text_fields=["text"],
        delete_shards_after_materialize=True,
    )

    assert stats["materialized_this_run"] == 1
    assert not shard.exists()
