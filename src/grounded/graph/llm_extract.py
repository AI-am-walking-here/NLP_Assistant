"""M-6.2 — structured entity/relation extraction with local Llama-3.1-8B."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from grounded.graph.schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    GraphEntity,
    GraphRelation,
    GraphTriple,
)

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = (
    "You extract a knowledge graph from an NLP paper chunk. "
    f"Entity types: {list(ENTITY_TYPES)}. "
    f"Relation types: {list(RELATION_TYPES)}. "
    "Respond with one complete valid JSON object only. "
    "Use double quotes for every JSON string. "
    "Do not include markdown, comments, trailing commas, or text after the JSON object."
)

_EXTRACT_USER = """Extract entities and relations from this chunk.

Schema:
{{"entities": [{{"name": "...", "type": "method|task|dataset|finding"}}],
  "relations": [{{"head": "...", "relation": "proposes|improves_on|uses|evaluated_on", "tail": "..."}}]}}

Return exactly this top-level shape:
{{"entities": [], "relations": []}}

Chunk:
{text}
"""

def _balanced_json_prefix(raw: str) -> str:
    start = raw.find("{")
    if start < 0:
        return raw.strip()
    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return raw[start:].strip()


def _repair_json_object(text: str) -> str:
    """Repair common LLM JSON truncation without inventing entities."""
    text = text.strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]") and stack and stack[-1] == ch:
            stack.pop()
    if in_string:
        text += '"'
    text = re.sub(r",\s*$", "", text)
    while stack:
        text = re.sub(r",\s*$", "", text)
        text += stack.pop()
    return text


def _parse_llm_json(raw: str) -> dict[str, Any]:
    text = _balanced_json_prefix(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_json_object(text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return _parse_jsonish_graph(text)


def _parse_jsonish_graph(text: str) -> dict[str, Any]:
    """Extract graph rows from near-JSON when punctuation is malformed."""
    entities: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []

    entity_pattern = re.compile(
        r'"name"\s*:?\s*"(?P<name>[^"]+)"[\s,]+'
        r'"type"\s*:?\s*"(?P<type>[^"]+)"',
        re.I,
    )
    for match in entity_pattern.finditer(text):
        name = match.group("name").strip()
        etype = match.group("type").strip().lower()
        if name and etype in ENTITY_TYPES:
            entities.append({"name": name, "type": etype})

    relation_pattern = re.compile(
        r'"head"\s*:?\s*"(?P<head>[^"]+)"[\s,]+'
        r'"relation"\s*:?\s*"(?P<relation>[^"]+)"[\s,]+'
        r'"tail"\s*:?\s*"(?P<tail>[^"]+)"',
        re.I,
    )
    for match in relation_pattern.finditer(text):
        relation = match.group("relation").strip().lower()
        if relation in RELATION_TYPES:
            relations.append(
                {
                    "head": match.group("head").strip(),
                    "relation": relation,
                    "tail": match.group("tail").strip(),
                }
            )
    if not entities and not relations:
        raise json.JSONDecodeError("could not recover graph JSON", text, 0)
    return {"entities": entities, "relations": relations}


def _validate_triple(
    chunk_id: str,
    paper_id: str,
    data: dict[str, Any],
) -> GraphTriple:
    entities: list[GraphEntity] = []
    for row in data.get("entities") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        etype = str(row.get("type", "")).strip().lower()
        if name and etype in ENTITY_TYPES:
            entities.append(GraphEntity(name=name, type=etype))  # type: ignore[arg-type]

    name_set = {e.name.lower() for e in entities}
    relations: list[GraphRelation] = []
    for row in data.get("relations") or []:
        if not isinstance(row, dict):
            continue
        head = str(row.get("head", "")).strip()
        tail = str(row.get("tail", "")).strip()
        rel = str(row.get("relation", "")).strip().lower()
        if (
            head
            and tail
            and rel in RELATION_TYPES
            and head.lower() in name_set
            and tail.lower() in name_set
        ):
            relations.append(
                GraphRelation(head=head, relation=rel, tail=tail)  # type: ignore[arg-type]
            )

    return GraphTriple(
        chunk_id=chunk_id,
        paper_id=paper_id,
        entities=entities[:12],
        relations=relations[:12],
        extractor="llm_8b",
    )


class LlmGraphExtractor:
    """8B instruct extractor (one chunk per generate call)."""

    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int = 384) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def extract(self, chunk_id: str, paper_id: str, text: str) -> GraphTriple:
        import torch

        user = _EXTRACT_USER.format(text=text[:2000])
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{_EXTRACT_SYSTEM}\n\n{user}\n"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        try:
            data = _parse_llm_json(raw)
            return _validate_triple(chunk_id, paper_id, data)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("LLM graph parse failed for %s: %s", chunk_id, exc)
            return GraphTriple(
                chunk_id=chunk_id,
                paper_id=paper_id,
                entities=[],
                relations=[],
                extractor="llm_8b_failed",
            )


def load_llm_extractor(base_model: str) -> LlmGraphExtractor:
    from grounded.models.peft_loader import load_peft_causal_lm

    model, tokenizer = load_peft_causal_lm(base_model, adapter_path=None, role="generator_8b")
    return LlmGraphExtractor(model, tokenizer)
