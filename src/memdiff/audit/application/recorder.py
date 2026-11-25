from __future__ import annotations

from memdiff.audit.application.ports import AuditLogRepository
from memdiff.audit.domain.timeline import CONSOLIDATE_PREVIOUS_FIELDS, BeliefEdge, TimelineEvent
from memdiff.memory.domain.revision import (
    BeliefAdded,
    BeliefConsolidated,
    BeliefForgotten,
    BeliefRefuted,
    BeliefSuperseded,
    CandidateIgnored,
    RevisionEvent,
    RevisionOp,
)


class RevisionAuditRecorder:
    """RevisionEventSink that turns memory's domain events into timeline events and edges."""

    def __init__(self, log: AuditLogRepository) -> None:
        self._log = log

    def op_counts(self) -> dict[str, int]:
        return self._log.count_events_by_op()

    def emit(self, event: RevisionEvent) -> None:
        base = {"step": event.step, "source_id": event.source_id}
        match event:
            case BeliefAdded():
                self._log.add_event(
                    TimelineEvent(
                        **base,
                        op=RevisionOp.ADD,
                        belief_id=event.belief_id,
                        summary=(
                            f"Added [{event.belief_type.value}] on {event.surface}: "
                            f"{event.claim[:100]}"
                        ),
                        details={"surface": event.surface, "type": event.belief_type.value},
                    )
                )
            case BeliefSuperseded():
                self._log.add_edge(
                    BeliefEdge(
                        source_id=event.new_belief_id,
                        target_id=event.old_belief_id,
                        relation="SUPERSEDES",
                        step=event.step,
                        reason=event.reason,
                    )
                )
                self._log.add_event(
                    TimelineEvent(
                        **base,
                        op=RevisionOp.SUPERSEDE,
                        belief_id=event.new_belief_id,
                        target_belief_id=event.old_belief_id,
                        summary=(
                            f"Superseded {event.old_belief_id} -> {event.new_belief_id}: "
                            f"{event.reason or event.new_claim[:80]}"
                        ),
                        details={
                            "old_claim": event.old_claim,
                            "new_claim": event.new_claim,
                            "reason": event.reason,
                        },
                    )
                )
            case BeliefConsolidated():
                self._log.add_event(
                    TimelineEvent(
                        **base,
                        op=RevisionOp.CONSOLIDATE,
                        belief_id=event.belief_id,
                        summary=(
                            f"Consolidated evidence (#{event.evidence_count}) into "
                            f"{event.belief_id} on {event.surface}"
                        ),
                        details={
                            "evidence_count": event.evidence_count,
                            "previous": event.previous.model_dump(
                                mode="json", include=CONSOLIDATE_PREVIOUS_FIELDS
                            ),
                        },
                    )
                )
            case BeliefRefuted():
                self._log.add_event(
                    TimelineEvent(
                        **base,
                        op=RevisionOp.REFUTE,
                        target_belief_id=event.belief_id,
                        summary=f"Refuted {event.belief_id}: {event.reason[:100]}",
                        details={"reason": event.reason},
                    )
                )
            case BeliefForgotten():
                self._log.add_event(
                    TimelineEvent(
                        **base,
                        op=RevisionOp.FORGET,
                        target_belief_id=event.belief_id,
                        summary=f"Forgot {event.belief_id} ({event.reason})",
                        details={"reason": event.reason},
                    )
                )
            case CandidateIgnored():
                self._log.add_event(
                    TimelineEvent(**base, op=RevisionOp.IGNORE, summary=event.summary)
                )
            case _:
                raise TypeError(f"Unhandled revision event: {type(event).__name__}")
