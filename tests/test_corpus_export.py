"""Tests for corpus → parsed export."""

from __future__ import annotations

import json

from grounded.data.corpus_export import (
    atomic_write_json,
    citation_keys_in_body,
    convert_bib_entry,
    corpus_record_to_paper,
    extract_year_from_raw,
)


def test_convert_bib_entry_from_corpus_shape() -> None:
    entry = convert_bib_entry(
        "abc123",
        {
            "bib_entry_raw": "Jane Doe. 2019. A paper title. In ACL.",
            "discipline": "CS",
            "ids": {"doi": "10.1/example"},
        },
    )
    assert entry["citation_key"] == "abc123"
    assert entry["raw_entry"].startswith("Jane Doe")
    assert entry["year"] == 2019
    assert entry["discipline"] == "CS"


def test_extract_year_from_raw() -> None:
    assert extract_year_from_raw("Foo. 2021. Bar.") == 2021
    assert extract_year_from_raw("no year here") is None


def test_citation_keys_in_body_order() -> None:
    text = "A {{cite:key2}} and {{cite:key1}} and {{cite:key2}}."
    assert citation_keys_in_body(text) == ["key2", "key1"]


def test_corpus_record_to_paper_maps_source_and_body() -> None:
    raw = {
        "id": "2103.11332",
        "source": "unarxive",
        "title": "T",
        "abstract": "A",
        "text": "Body with {{cite:abc}}.",
        "bibliography": {
            "abc": {"bib_entry_raw": "Author. 2020. Title."},
        },
        "references": {
            "bib_entries": {},
            "ref_entries": {},
            "citation_spans": [],
            "reference_spans": [],
        },
    }
    paper = corpus_record_to_paper(raw)
    assert paper["source"] == "unarxive"
    assert paper["body_text"] == "Body with ."
    assert paper["citation_keys_in_body"] == ["abc"]
    assert paper["bibliography"]["abc"]["raw_entry"].startswith("Author")
    assert paper["references"]["citation_spans"] == []


def test_corpus_record_to_paper_cleans_placeholders() -> None:
    raw = {
        "id": "2103.11332",
        "source": "unarxive",
        "title": "T",
        "abstract": "A",
        "text": "Body with {{formula:x}} {{figure:y}} REF {{cite:abc}}.",
        "bibliography": {},
        "references": {},
    }
    paper = corpus_record_to_paper(raw)
    assert "{{formula:" not in paper["body_text"]
    assert "{{figure:" not in paper["body_text"]


def test_atomic_write_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "x.json"
    data = {"arxiv_id": "1.2.3", "body_text": "ok"}
    atomic_write_json(path, data)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data
