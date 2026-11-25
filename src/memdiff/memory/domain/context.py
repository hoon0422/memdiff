"""Context items handed to downstream QA, and ledger footprint statistics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from .belief import Belief, BeliefStatus


class MemoryContextItem(BaseModel):
    """Official LongMemEval-V2 memory context item format."""

    type: Literal["text", "image"]
    value: str


class MemoryStats(BaseModel):
    """Footprint and lifecycle telemetry for budget-matched comparisons."""

    total_ingested_sources: int = 0
    active_beliefs: int = 0
    superseded_beliefs: int = 0
    refuted_beliefs: int = 0
    forgotten_beliefs: int = 0
    raw_chunks: int = 0
    estimated_active_tokens: int = 0
    revision_counts: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        beliefs: Iterable[Belief],
        raw_chunks: int,
        revision_counts: dict[str, int],
        total_ingested_sources: int,
    ) -> MemoryStats:
        rows = list(beliefs)
        by_status = {s: sum(1 for b in rows if b.status == s) for s in BeliefStatus}
        return cls(
            total_ingested_sources=total_ingested_sources,
            active_beliefs=by_status[BeliefStatus.ACTIVE],
            superseded_beliefs=by_status[BeliefStatus.SUPERSEDED],
            refuted_beliefs=by_status[BeliefStatus.REFUTED],
            forgotten_beliefs=by_status[BeliefStatus.FORGOTTEN],
            raw_chunks=raw_chunks,
            estimated_active_tokens=sum(
                max(1, len(b.claim) // 4) for b in rows if b.status == BeliefStatus.ACTIVE
            ),
            revision_counts=dict(revision_counts),
        )
