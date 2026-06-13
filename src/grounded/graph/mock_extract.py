"""Heuristic triple extractor for offline pipeline tests (no LLM)."""

from __future__ import annotations

import re

from grounded.graph.schema import GraphEntity, GraphRelation, GraphTriple

DATASET_RE = re.compile(
    r"\b(SQuAD|SNLI|MNLI|GLUE|WMT|ImageNet|COCO|WikiText|CoNLL|"
    r"MS MARCO|Natural Questions|SuperGLUE|LAMBADA|PTB|"
    r"[A-Z][A-Za-z0-9\-]{2,20})\s+(dataset|corpus|benchmark)\b",
    re.I,
)
TASK_RE = re.compile(
    r"\b(machine translation|question answering|summarization|"
    r"language modeling|sentiment analysis|named entity recognition|"
    r"text classification|dialogue|parsing)\b",
    re.I,
)
METHOD_RE = re.compile(
    r"\b(transformer|BERT|GPT|LSTM|attention|graph neural network|"
    r"reinforcement learning|contrastive learning|pre-train\w*)\b",
    re.I,
)


def extract_triple_mock(
    chunk_id: str,
    paper_id: str,
    text: str,
    *,
    max_entities: int = 8,
) -> GraphTriple:
    entities: list[GraphEntity] = []
    seen: set[str] = set()

    def add(name: str, etype: str) -> None:
        key = name.lower()
        if key in seen or len(entities) >= max_entities:
            return
        seen.add(key)
        entities.append(GraphEntity(name=name.strip(), type=etype))  # type: ignore[arg-type]

    for m in DATASET_RE.finditer(text):
        add(m.group(1), "dataset")
    for m in TASK_RE.finditer(text):
        add(m.group(1).title(), "task")
    for m in METHOD_RE.finditer(text):
        add(m.group(1), "method")
    if "improve" in text.lower() and len(entities) >= 2:
        add("main finding", "finding")

    relations: list[GraphRelation] = []
    methods = [e.name for e in entities if e.type == "method"]
    tasks = [e.name for e in entities if e.type == "task"]
    datasets = [e.name for e in entities if e.type == "dataset"]
    if methods and tasks:
        relations.append(
            GraphRelation(head=methods[0], relation="proposes", tail=tasks[0])
        )
    if methods and datasets:
        relations.append(
            GraphRelation(head=methods[0], relation="evaluated_on", tail=datasets[0])
        )
    if len(methods) >= 2:
        relations.append(
            GraphRelation(head=methods[0], relation="improves_on", tail=methods[1])
        )

    return GraphTriple(
        chunk_id=chunk_id,
        paper_id=paper_id,
        entities=entities,
        relations=relations,
        extractor="mock",
    )
