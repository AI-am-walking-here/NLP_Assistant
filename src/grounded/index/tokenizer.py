"""Token counting for chunk sizing (Llama-3.1 preferred, tiktoken fallback)."""

from __future__ import annotations

from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


class TiktokenWrapper:
    def __init__(self, encoding: Any) -> None:
        self._enc = encoding

    def encode(self, text: str) -> list[int]:
        # Corpus text may contain literal <|endoftext|> strings from LaTeX dumps.
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


def load_tokenizer(model_id: str) -> TokenizerLike:
    import tiktoken

    if model_id in ("cl100k_base", "tiktoken"):
        return TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))

    try:
        from transformers import AutoTokenizer

        # Prefer cache-only so chunking never blocks on HF when offline.
        return AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True)
    except Exception:
        return TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))


def count_tokens(text: str, tokenizer: TokenizerLike) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))
