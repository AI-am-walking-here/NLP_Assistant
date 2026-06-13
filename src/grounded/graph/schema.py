"""M-6.1 — entity/relation schema (locked in configs/graph.yaml)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["method", "task", "dataset", "finding"]
RelationType = Literal["proposes", "improves_on", "uses", "evaluated_on"]

ENTITY_TYPES: tuple[str, ...] = ("method", "task", "dataset", "finding")
RELATION_TYPES: tuple[str, ...] = ("proposes", "improves_on", "uses", "evaluated_on")


class GraphEntity(BaseModel):
    name: str
    type: EntityType


class GraphRelation(BaseModel):
    head: str
    relation: RelationType
    tail: str


class GraphTriple(BaseModel):
    chunk_id: str
    paper_id: str
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    extractor: str = "mock"
