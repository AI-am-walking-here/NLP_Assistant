"""M-6.8 — RankRAG LoRA reranker inference (label log-prob scoring)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_CAND_CHARS = 800


def _format_candidate_prompt(
    query: str,
    candidate: str,
    *,
    paper_id: str | None = None,
    section_heading: str | None = None,
) -> str:
    text = candidate[:_MAX_CAND_CHARS]
    meta_bits = []
    if paper_id:
        meta_bits.append(f"paper={paper_id}")
    if section_heading:
        meta_bits.append(f"section={section_heading}")
    meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
    return f"Query: {query}\nCandidates:\n[0]{meta}\n{text}\nlabel="


class LoraRankReranker:
    """Scores passages with P(label=1) from the RankRAG LoRA head."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._label_ids: tuple[int | None, int | None] | None = None

    def _label_token_ids(self) -> tuple[int, int]:
        if self._label_ids is not None and self._label_ids[0] is not None:
            return self._label_ids[0], self._label_ids[1]  # type: ignore[return-value]
        ids0 = self.tokenizer.encode("0", add_special_tokens=False)
        ids1 = self.tokenizer.encode("1", add_special_tokens=False)
        if not ids0 or not ids1:
            raise RuntimeError("Tokenizer missing single-token ids for 0/1")
        self._label_ids = (ids0[-1], ids1[-1])
        return self._label_ids

    def _score_prompts(self, prompts: list[str]) -> list[float]:
        import torch

        if not prompts:
            return []
        id0, id1 = self._label_token_ids()
        scores: list[float] = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                lp0 = float(log_probs[id0].item())
                lp1 = float(log_probs[id1].item())
                scores.append(float(torch.sigmoid(torch.tensor(lp1 - lp0)).item()))
        return scores

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        return self._score_prompts([_format_candidate_prompt(query, cand) for cand in candidates])

    def score_rows(self, query: str, candidates: list[dict[str, Any]]) -> list[float]:
        prompts = [
            _format_candidate_prompt(
                query,
                str(row.get("text", "")),
                paper_id=str(row.get("paper_id", "")) or None,
                section_heading=str(row.get("section_heading", "")) or None,
            )
            for row in candidates
        ]
        return self._score_prompts(prompts)


@lru_cache(maxsize=4)
def _cached_rankrag_model(
    adapter_path: str,
    base_model: str,
    cuda_device: int,
) -> LoraRankReranker:
    from grounded.models.peft_loader import load_peft_causal_lm

    model, tokenizer = load_peft_causal_lm(
        base_model,
        Path(adapter_path),
        role="generator_8b",
        cuda_device=cuda_device,
    )
    return LoraRankReranker(model, tokenizer)


def load_lora_rankrag_reranker(
    adapter_path: Path,
    base_model: str,
    *,
    cuda_device: int = 0,
) -> LoraRankReranker:
    logger.info("Loading RankRAG LoRA from %s on cuda:%d", adapter_path, cuda_device)
    return _cached_rankrag_model(str(adapter_path.resolve()), base_model, cuda_device)
