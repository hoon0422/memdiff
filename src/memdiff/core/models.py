from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, TypedDict

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


class RevisionOp(StrEnum):
    ADD = "ADD"
    SUPERSEDE = "SUPERSEDE"
    CONSOLIDATE = "CONSOLIDATE"
    REFUTE = "REFUTE"
    IGNORE = "IGNORE"
    FORGET = "FORGET"


class MemoryContextItem(TypedDict):
    """Official LongMemEval-V2 memory context item format."""

    type: Literal["text", "image"]
    value: str


class Provenance(BaseModel):
    """Exact trajectory/session and step reference backing a belief."""

    source_id: str
    step_index: int | None = None
    url: str | None = None
    outcome: Literal["success", "failure", "unknown"] = "unknown"
    excerpt: str | None = None


class Belief(BaseModel):
    """Structured memory unit tracked across an explicit revision lifecycle."""

    id: str
    type: BeliefType
    surface: str = Field(
        default="general",
        description="Scoped UI page/module/entity key (e.g. 'ServiceNow > Incidents > Filters')",
    )
    claim: str
    closed_set_items: list[str] = Field(
        default_factory=list,
        description="Optional exhaustive list of visible controls/options on a scoped surface for premise awareness",
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
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


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


class ReplayQuestionComparison(BaseModel):
    """Side-by-side Q&A comparison record for the static React replay UI."""

    question_id: str
    category: str
    question: str
    gold_answer: str
    baseline_policy: str
    baseline_answer: str
    baseline_correct: bool
    baseline_retrieved: list[str] = Field(default_factory=list)
    proposed_policy: str
    proposed_answer: str
    proposed_correct: bool
    proposed_retrieved_belief_ids: list[str] = Field(default_factory=list)


class ReplayBundle(BaseModel):
    """Self-contained JSON payload consumed by the web Memory Time Machine viewer."""

    schema_version: str = "1.0"
    run_id: str
    benchmark: str
    policy_name: str
    total_steps: int
    beliefs: list[Belief]
    edges: list[BeliefEdge]
    events: list[TimelineEvent]
    stats: MemoryStats
    comparisons: list[ReplayQuestionComparison] = Field(default_factory=list)
