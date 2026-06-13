"""Graph triple extraction backends (mock vs local 8B)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from grounded.graph.mock_extract import extract_triple_mock
from grounded.graph.schema import GraphTriple

ExtractorName = Literal["mock", "llm"]
ExtractFn = Callable[[str, str, str], GraphTriple]

_llm_singleton = None


def get_extractor(name: ExtractorName, *, base_model: str | None = None) -> ExtractFn:
    if name == "mock":
        return extract_triple_mock
    if name == "llm":
        global _llm_singleton
        if _llm_singleton is None:
            from grounded.config import load_config
            from grounded.graph.llm_extract import load_llm_extractor

            cfg = load_config("sft")
            hub = base_model or cfg.base_model
            _llm_singleton = load_llm_extractor(hub)
        return _llm_singleton.extract
    raise ValueError(f"Unknown extractor: {name!r}")
