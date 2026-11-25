"""Belief entity, its lifecycle states, and the Provenance value object."""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class BeliefType(StrEnum):
    STATIC_SURFACE = "static_surface"
    DYNAMIC_STATE = "dynamic_state"
    WORKFLOW = "workflow"
    GOTCHA = "gotcha"
    DURABLE_RULE = "durable_rule"
    TEMPORARY = "temporary"


class BeliefStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REFUTED = "refuted"
    FORGOTTEN = "forgotten"


class Provenance(BaseModel):
    """Exact trajectory/session and step reference backing a belief."""

    source_id: str
    step_index: int | None = None
    url: str | None = None
    outcome: Literal["success", "failure", "unknown"] = "unknown"
    excerpt: str | None = None


def new_belief_id() -> str:
    return f"b_{uuid.uuid4().hex[:10]}"


class Belief(BaseModel):
    """Structured memory unit tracked across an explicit revision lifecycle.

    Rows are never deleted: transitions return a copy with a new status.
    """

    id: str
    type: BeliefType
    surface: str = Field(
        default="general",
        description="Scoped UI page/module/entity key (e.g. 'ServiceNow > Incidents > Filters')",
    )
    claim: str
    closed_set_items: list[str] = Field(
        default_factory=list,
        description="Optional exhaustive list of visible controls/options on a scoped surface",
    )
    status: BeliefStatus = BeliefStatus.ACTIVE
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    durability: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence_count: int = Field(default=1, ge=1)
    created_at_step: int
    updated_at_step: int
    superseded_by: str | None = None
    provenance: list[Provenance] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        id: str,
        type: BeliefType,
        surface: str,
        claim: str,
        step: int,
        source_id: str,
        provenance: list[Provenance] | None = None,
        closed_set_items: list[str] | None = None,
        confidence: float = 0.8,
        durability: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> Belief:
        return cls(
            id=id,
            type=type,
            surface=surface,
            claim=claim,
            closed_set_items=list(closed_set_items or []),
            confidence=confidence,
            durability=durability,
            created_at_step=step,
            updated_at_step=step,
            provenance=provenance or [Provenance(source_id=source_id)],
            metadata=dict(metadata or {}),
        )

    def mark_superseded(self, *, by: str, step: int) -> Belief:
        return self.model_copy(
            update={"status": BeliefStatus.SUPERSEDED, "superseded_by": by, "updated_at_step": step}
        )

    def mark_refuted(self, *, step: int) -> Belief:
        return self.model_copy(update={"status": BeliefStatus.REFUTED, "updated_at_step": step})

    def mark_forgotten(self, *, step: int) -> Belief:
        return self.model_copy(update={"status": BeliefStatus.FORGOTTEN, "updated_at_step": step})
