"""FActScore-aligned DPO preference construction (M-5.x)."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from grounded.eval.factscore import ClaimVerifier, compute_factscore
from grounded.generate.prompts import (
    format_retrieved_chunks,
    render_abstract_prompt,
    render_sft_prompt,
)
from grounded.train.sft_data import format_chat_example
from grounded.train.sft_retrieval import retrieve_training_passages

logger = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PromptStyle = Literal["sft", "with_retrieval"]

_HALLUCINATION_PREFIX = (
    "We establish new state-of-the-art results on every benchmark without comparison. "
    "Our method requires no data and guarantees perfect generalization. "
)


@dataclass(frozen=True)
class ScoredCandidate:
    text: str
    factscore: float
    source: str


@dataclass(frozen=True)
class PreferencePair:
    arxiv_id: str
    prompt_messages: list[dict[str, str]]
    chosen: str
    rejected: str
    chosen_factscore: float
    rejected_factscore: float
    chosen_source: str
    rejected_source: str
    n_passages: int

    def to_json_row(self) -> dict[str, Any]:
        # TRL chat DPO: prompt = messages prefix; chosen/rejected = assistant turns only.
        return {
            "arxiv_id": self.arxiv_id,
            "prompt": self.prompt_messages,
            "chosen": [{"role": "assistant", "content": self.chosen}],
            "rejected": [{"role": "assistant", "content": self.rejected}],
            "chosen_factscore": self.chosen_factscore,
            "rejected_factscore": self.rejected_factscore,
            "chosen_source": self.chosen_source,
            "rejected_source": self.rejected_source,
            "n_passages": self.n_passages,
        }


def passage_texts(hits: list[dict[str, Any]]) -> list[str]:
    return [str(h.get("text", "")).strip() for h in hits if str(h.get("text", "")).strip()]


def score_abstract(
    abstract: str,
    passages: list[str],
    verifier: ClaimVerifier,
    *,
    max_claims: int = 8,
) -> float:
    if not abstract.strip():
        return 0.0
    evidence = passages
    if not evidence:
        return 0.0
    return float(
        compute_factscore(abstract, evidence, verifier, max_claims=max_claims)["factscore"]
    )


def generic_weak_abstract(title: str) -> str:
    t = title.strip() or "this topic"
    return (
        f"We study {t}. "
        "The proposed approach is novel and effective. "
        "Experiments demonstrate strong performance across settings. "
        "Our analysis confirms the method is useful for future work."
    )


def unfaithful_variant(abstract: str) -> str:
    """Cheap negative: hallucinated lead + truncated gold."""
    body = re.sub(r"\s+", " ", abstract.strip())
    if len(body) > 400:
        body = body[:400].rsplit(" ", 1)[0] + "."
    return _HALLUCINATION_PREFIX + body


def shuffled_variant(abstract: str, *, rng: random.Random) -> str:
    sents = [s.strip() for s in SENTENCE_SPLIT.split(abstract.strip()) if len(s.strip()) > 20]
    if len(sents) < 3:
        return unfaithful_variant(abstract)
    rng.shuffle(sents)
    return " ".join(sents)


def build_candidates(
    title: str,
    gold_abstract: str,
    *,
    rng: random.Random,
    quick: bool = False,
) -> list[tuple[str, str]]:
    """Return (text, source_tag) candidate variants."""
    base = [
        (gold_abstract.strip(), "gold"),
        (unfaithful_variant(gold_abstract), "unfaithful"),
    ]
    if quick:
        return base
    return base + [
        (shuffled_variant(gold_abstract, rng=rng), "shuffled"),
        (generic_weak_abstract(title), "generic"),
    ]


def prompt_messages_for_row(
    title: str,
    outline: str,
    *,
    prompt_style: PromptStyle = "sft",
    retrieved_block: str | None = None,
) -> list[dict[str, str]]:
    """DPO prompt aligned with the intended eval prompt family."""
    if prompt_style == "with_retrieval":
        system, user = render_abstract_prompt(
            title,
            outline,
            retrieved_block or "(no retrieved passages)",
        )
    else:
        system, user = render_sft_prompt(title, outline)
    return format_chat_example(system, user, "")["messages"][:2]


def build_preference_pair(
    row: dict[str, Any],
    verifier: ClaimVerifier,
    *,
    retrieval_top_k: int = 8,
    min_margin: float = 0.08,
    seed: int = 0,
    prompt_style: PromptStyle = "sft",
    quick_candidates: bool = False,
) -> PreferencePair | None:
    """
    Build one (chosen, rejected) pair from an SFT JSONL row.

    Uses retrieved passages for FActScore only; prompt matches fair SFT eval.
    """
    aid = str(row["arxiv_id"])
    title = str(row.get("title", "")).strip()
    outline = str(row.get("outline", "")).strip()
    gold = str(row.get("target_abstract", "")).strip()
    if not title or not outline or not gold:
        return None

    hits = retrieve_training_passages(title, outline, aid, top_k=retrieval_top_k)
    passages = passage_texts(hits)
    if not passages:
        logger.warning("Skip DPO %s: no retrieval passages", aid)
        return None

    rng = random.Random(f"{seed}:{aid}")
    raw_candidates = build_candidates(title, gold, rng=rng, quick=quick_candidates)
    scored: list[ScoredCandidate] = []
    for text, source in raw_candidates:
        fs = score_abstract(text, passages, verifier)
        scored.append(ScoredCandidate(text=text, factscore=fs, source=source))

    scored.sort(key=lambda c: c.factscore, reverse=True)
    best, worst = scored[0], scored[-1]
    if best.factscore - worst.factscore < min_margin:
        logger.debug(
            "Skip DPO %s: margin %.3f < %.3f (best=%s %.3f worst=%s %.3f)",
            aid,
            best.factscore - worst.factscore,
            min_margin,
            best.source,
            best.factscore,
            worst.source,
            worst.factscore,
        )
        return None

    retrieved_block = None
    if prompt_style == "with_retrieval":
        retrieved_block = format_retrieved_chunks(hits)
    msgs = prompt_messages_for_row(
        title,
        outline,
        prompt_style=prompt_style,
        retrieved_block=retrieved_block,
    )
    return PreferencePair(
        arxiv_id=aid,
        prompt_messages=msgs,
        chosen=best.text,
        rejected=worst.text,
        chosen_factscore=best.factscore,
        rejected_factscore=worst.factscore,
        chosen_source=best.source,
        rejected_source=worst.source,
        n_passages=len(passages),
    )


def build_preferences_from_jsonl(
    train_path: Path,
    verifier: ClaimVerifier,
    *,
    limit: int | None = None,
    retrieval_top_k: int = 8,
    min_margin: float = 0.08,
    seed: int = 1337,
    prompt_style: PromptStyle = "sft",
    quick_candidates: bool = True,
) -> tuple[list[PreferencePair], dict[str, Any]]:
    pairs: list[PreferencePair] = []
    skipped = 0
    input_rows = 0
    for i, line in enumerate(train_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if limit is not None and input_rows >= limit:
            break
        input_rows += 1
        row = json.loads(line)
        pair = build_preference_pair(
            row,
            verifier,
            retrieval_top_k=retrieval_top_k,
            min_margin=min_margin,
            seed=seed + i,
            prompt_style=prompt_style,
            quick_candidates=quick_candidates,
        )
        if pair is None:
            skipped += 1
        else:
            pairs.append(pair)
    stats = {
        "input_rows": input_rows,
        "pairs_built": len(pairs),
        "skipped": skipped,
        "mean_margin": (
            sum(p.chosen_factscore - p.rejected_factscore for p in pairs) / len(pairs)
            if pairs
            else 0.0
        ),
    }
    return pairs, stats


def write_dpo_jsonl(pairs: list[PreferencePair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_json_row(), ensure_ascii=False) + "\n")
