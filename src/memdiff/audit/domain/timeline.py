from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from memdiff.memory.domain.revision import RevisionOp

# Belief fields CONSOLIDATE rewrites; the event keeps their pre-consolidate values so time travel can rewind them.
CONSOLIDATE_PREVIOUS_FIELDS = {"claim", "closed_set_items", "confidence", "evidence_count", "provenance"}


class BeliefEdge(BaseModel):
    """Directed relationship between two beliefs in the revision graph."""

    source_id: str
    target_id: str
    relation: Literal["SUPERSEDES", "CONSOLIDATES", "REFUTES"]
    step: int
    reason: str = ""


class TimelineEvent(BaseModel):
    """Immutable audit log entry powering the Memory Time Machine slider."""

    id: int | None = None
    step: int
    source_id: str
    op: RevisionOp
    belief_id: str | None = None
    target_belief_id: str | None = None
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
