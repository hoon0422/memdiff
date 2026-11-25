from __future__ import annotations

from collections.abc import Iterable

from memdiff.audit.domain.timeline import TimelineEvent
from memdiff.memory.domain.belief import Belief, BeliefStatus
from memdiff.memory.domain.revision import RevisionOp

_STATUS_BY_OP = {
    RevisionOp.SUPERSEDE: BeliefStatus.SUPERSEDED,
    RevisionOp.REFUTE: BeliefStatus.REFUTED,
    RevisionOp.FORGET: BeliefStatus.FORGOTTEN,
}


def reconstruct_state_at_step(
    beliefs: Iterable[Belief], events: Iterable[TimelineEvent], step: int
) -> list[Belief]:
    """Every belief created by `step`, with the status the events up to `step` gave it.

    Same rule as `computeStatusAtStep()` in web/src/App.tsx. `events` must be in log order.
    """
    visible = {b.id: b for b in beliefs if b.created_at_step <= step}
    status = {bid: BeliefStatus.ACTIVE for bid in visible}
    superseded_by: dict[str, str | None] = dict.fromkeys(visible)
    for ev in events:
        if ev.step > step or ev.op not in _STATUS_BY_OP or ev.target_belief_id not in visible:
            continue
        status[ev.target_belief_id] = _STATUS_BY_OP[ev.op]
        if ev.op == RevisionOp.SUPERSEDE:
            superseded_by[ev.target_belief_id] = ev.belief_id
    return [
        b.model_copy(update={"status": status[bid], "superseded_by": superseded_by[bid]})
        for bid, b in visible.items()
    ]
