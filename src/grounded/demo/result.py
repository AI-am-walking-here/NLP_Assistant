"""Demo generation result with pipeline stage provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from grounded.generate.baselines import GenerationResult


@dataclass
class PipelineStageRecord:
    id: str
    label: str
    detail: str = ""
    count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "label": self.label, "detail": self.detail}
        if self.count is not None:
            out["count"] = self.count
        return out


@dataclass
class DemoRunResult:
    abstract_text: str
    retrieved_chunks: list[dict[str, Any]]
    mock: bool
    stages: list[PipelineStageRecord] = field(default_factory=list)
    passages_pre_rerank: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_generation(
        cls,
        gen: GenerationResult,
        stages: list[PipelineStageRecord],
        *,
        passages_pre_rerank: list[dict[str, Any]] | None = None,
    ) -> DemoRunResult:
        return cls(
            abstract_text=gen.abstract_text,
            retrieved_chunks=gen.retrieved_chunks,
            mock=gen.mock,
            stages=stages,
            passages_pre_rerank=passages_pre_rerank or [],
        )
