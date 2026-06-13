"""Graph extraction validation (no GPU)."""

from __future__ import annotations

from grounded.graph.extract import get_extractor
from grounded.graph.llm_extract import _parse_llm_json
from grounded.graph.schema import GraphTriple


def test_mock_extractor() -> None:
    fn = get_extractor("mock")
    t: GraphTriple = fn("c1", "p1", "We evaluate on SQuAD dataset using a transformer.")
    assert t.extractor == "mock"
    assert any(e.type == "dataset" for e in t.entities)


def test_llm_json_parser_repairs_truncated_object() -> None:
    raw = (
        '{"entities": [{"name": "BERT", "type": "method"}], '
        '"relations": [{"head": "BERT", "relation": "uses", "tail": "GLUE"'
    )

    data = _parse_llm_json(raw)

    assert data["entities"][0]["name"] == "BERT"
    assert data["relations"][0]["tail"] == "GLUE"


def test_llm_json_parser_uses_first_balanced_object() -> None:
    raw = (
        'Here is the JSON: {"entities": [], "relations": []}\n'
        'extra text {"bad": true}'
    )

    assert _parse_llm_json(raw) == {"entities": [], "relations": []}


def test_llm_json_parser_recovers_missing_colons() -> None:
    raw = (
        '{"entities": [{"name" "BERT", "type": "method"}], '
        '"relations": [{"head": "BERT", "relation" "uses", "tail": "GLUE"}]}'
    )

    data = _parse_llm_json(raw)

    assert data["entities"] == [{"name": "BERT", "type": "method"}]
    assert data["relations"] == [{"head": "BERT", "relation": "uses", "tail": "GLUE"}]
