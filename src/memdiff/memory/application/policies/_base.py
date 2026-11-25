from __future__ import annotations

from memdiff.memory.application.ports import RevisionEventSink
from memdiff.memory.domain.belief import Belief
from memdiff.memory.domain.revision import BeliefAdded
from memdiff.memory.domain.trajectory import Trajectory


def materialize(
    cand: Belief, *, id: str, claim: str, surface: str, step: int, source_id: str
) -> Belief:
    """The stored belief for an extracted candidate (candidates carry placeholder ids)."""
    return Belief.new(
        id=id,
        type=cand.type,
        surface=surface,
        claim=claim,
        step=step,
        source_id=source_id,
        provenance=cand.provenance,
        closed_set_items=cand.closed_set_items,
        confidence=cand.confidence,
        durability=cand.durability,
        metadata=cand.metadata,
    )


def emit_added(events: RevisionEventSink, belief: Belief, *, step: int, source_id: str) -> None:
    events.emit(
        BeliefAdded(
            step=step,
            source_id=source_id,
            belief_id=belief.id,
            belief_type=belief.type,
            surface=belief.surface,
            claim=belief.claim,
        )
    )


class SteppedPolicy:
    """Owns the ingest-step counter; it is only changed via `resume_from`."""

    def __init__(self) -> None:
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def resume_from(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self._step = step

    def _next_source_id(self, trajectory: Trajectory) -> str:
        self._step += 1
        return trajectory.id or f"traj_{self._step}"
