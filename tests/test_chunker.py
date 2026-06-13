"""Unit tests for section-aware chunking (no ML deps)."""

from __future__ import annotations

from grounded.index.chunker import chunk_paper
from grounded.index.tokenizer import TiktokenWrapper

import tiktoken


def test_chunk_paper_respects_sections() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    tok = TiktokenWrapper(enc)
    paper = {
        "arxiv_id": "9999.00001",
        "sections": [
            {
                "heading": "Intro",
                "paragraphs": ["Alpha " * 200, "Beta " * 200],
            },
            {
                "heading": "Methods",
                "paragraphs": ["Gamma " * 200],
            },
        ],
    }
    chunks = chunk_paper(paper, tokenizer=tok, chunk_size=80, chunk_overlap=10)
    assert chunks
    assert all(c["paper_id"] == "9999.00001" for c in chunks)
    headings = {c["section_heading"] for c in chunks}
    assert "Intro" in headings
    assert "Methods" in headings


def test_tiktoken_allows_endoftext_literal() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    tok = TiktokenWrapper(enc)
    n = len(tok.encode("prefix <|endoftext|> suffix"))
    assert n > 0
