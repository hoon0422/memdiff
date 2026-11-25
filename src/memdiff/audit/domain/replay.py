from __future__ import annotations

from pydantic import BaseModel, Field

from memdiff.audit.domain.timeline import BeliefEdge, TimelineEvent
from memdiff.memory.domain.belief import Belief
from memdiff.memory.domain.context import MemoryStats


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
