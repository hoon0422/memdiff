from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from memdiff.audit.domain.timeline import TimelineEvent
from memdiff.memory.domain.belief import Belief, BeliefStatus, Provenance
from memdiff.memory.domain.revision import RevisionOp

_STATUS_BY_OP = {
    RevisionOp.SUPERSEDE: BeliefStatus.SUPERSEDED,
    RevisionOp.REFUTE: BeliefStatus.REFUTED,
    RevisionOp.FORGET: BeliefStatus.FORGOTTEN,
}


def reconstruct_state_at_step(
    beliefs: Iterable[Belief], events: Iterable[TimelineEvent], step: int
) -> list[Belief]:
    """Every belief created by `step`, with the status and consolidated content as of `step`.

    A belief's row holds its latest content; the first CONSOLIDATE event after `step` carries the
    values it overwrote, so that event's `previous` is the content at `step`.

    Same rule as `computeBeliefAtStep()` in web/src/App.tsx. `events` must be in log order.
    """
    visible = {b.id: b for b in beliefs if b.created_at_step <= step}
    status = {bid: BeliefStatus.ACTIVE for bid in visible}
    superseded_by: dict[str, str | None] = dict.fromkeys(visible)
    content: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.op == RevisionOp.CONSOLIDATE and ev.step > step and ev.belief_id in visible:
            # replays written before `previous` existed simply keep the latest content
            if "previous" in ev.details:
                content.setdefault(ev.belief_id, ev.details["previous"])
        if ev.step > step or ev.op not in _STATUS_BY_OP or ev.target_belief_id not in visible:
            continue
        status[ev.target_belief_id] = _STATUS_BY_OP[ev.op]
        if ev.op == RevisionOp.SUPERSEDE:
            superseded_by[ev.target_belief_id] = ev.belief_id
    return [
        b.model_copy(update=_content_update(content.get(bid)) | {
            "status": status[bid], "superseded_by": superseded_by[bid],
        })
        for bid, b in visible.items()
    ]


def _content_update(previous: dict[str, Any] | None) -> dict[str, Any]:
    if previous is None:
        return {}
    return {**previous, "provenance": [Provenance.model_validate(p) for p in previous["provenance"]]}
