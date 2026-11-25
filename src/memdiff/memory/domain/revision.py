"""Revision decisions and the domain events emitted while the ledger changes."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from .belief import BeliefType


class RevisionOp(StrEnum):
    ADD = "ADD"
    SUPERSEDE = "SUPERSEDE"
    CONSOLIDATE = "CONSOLIDATE"
    REFUTE = "REFUTE"
    IGNORE = "IGNORE"
    FORGET = "FORGET"


class RevisionDecision(BaseModel):
    """Reconciler verdict for one candidate belief against its active neighbors."""

    op: RevisionOp
    target_belief_id: str | None = None
    merged_claim: str | None = None
    reason: str = ""


class RevisionEvent(BaseModel):
    step: int
    source_id: str


class BeliefAdded(RevisionEvent):
    belief_id: str
    belief_type: BeliefType
    surface: str
    claim: str


class BeliefSuperseded(RevisionEvent):
    """Replaces the old belief; the new belief gets no separate BeliefAdded."""

    old_belief_id: str
    old_claim: str
    new_belief_id: str
    new_claim: str
    reason: str = ""


class BeliefConsolidated(RevisionEvent):
    belief_id: str
    surface: str
    evidence_count: int


class BeliefRefuted(RevisionEvent):
    belief_id: str
    reason: str


class BeliefForgotten(RevisionEvent):
    belief_id: str
    reason: str


class CandidateIgnored(RevisionEvent):
    summary: str
