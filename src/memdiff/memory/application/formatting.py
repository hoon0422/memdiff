"""Context-string formatting. Output must stay byte-identical to the legacy policies."""

from __future__ import annotations

from memdiff.memory.domain.belief import Belief
from memdiff.memory.domain.trajectory import Trajectory, TrajectoryState


def belief_embedding_text(belief: Belief) -> str:
    return f"[{belief.type.value}] {belief.surface}: {belief.claim}"


def format_state_chunk(trajectory: Trajectory, state: TrajectoryState, max_chunk_chars: int) -> str:
    header = (
        f"[Trajectory {trajectory.id or 'unknown'} | outcome={trajectory.outcome} | "
        f"step={state.state_index}]\n"
        f"Goal: {trajectory.goal[:240]}\n"
        f"URL: {state.url}\n"
        f"Action: {state.action or '<none>'}\n"
    )
    budget = max(200, max_chunk_chars - len(header))
    return header + state.content[:budget]


def format_structured_belief(belief: Belief) -> str:
    prov = ", ".join(
        f"{p.source_id}@step{p.step_index}" if p.step_index is not None else p.source_id
        for p in belief.provenance[:3]
    )
    closed = (
        f"\nClosed-set visible items: {', '.join(belief.closed_set_items)}"
        if belief.closed_set_items
        else ""
    )
    return (
        f"[{belief.id} | {belief.type.value} | status={belief.status.value} | "
        f"surface={belief.surface}]\n"
        f"{belief.claim}{closed}\n"
        f"Provenance: {prov}"
    )


def format_revised_belief(belief: Belief, score: float) -> str:
    prov = ", ".join(
        f"{p.source_id}@step{p.step_index}({p.outcome})"
        if p.step_index is not None
        else f"{p.source_id}({p.outcome})"
        for p in belief.provenance[:4]
    )
    closed = (
        f"\nClosed-set observed items: {', '.join(belief.closed_set_items)}"
        if belief.closed_set_items
        else ""
    )
    return (
        f"[{belief.id} | {belief.type.value} | surface={belief.surface} | "
        f"evidence={belief.evidence_count} | sim={score:.3f}]\n"
        f"{belief.claim}{closed}\n"
        f"Provenance: {prov}"
    )
