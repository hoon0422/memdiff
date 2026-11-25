from __future__ import annotations

from typing import Protocol

from memdiff.audit.domain.replay import ReplayBundle
from memdiff.audit.domain.timeline import BeliefEdge, TimelineEvent
from memdiff.memory.domain.belief import Belief


class AuditLogRepository(Protocol):
    def add_event(self, event: TimelineEvent) -> None: ...

    def add_edge(self, edge: BeliefEdge) -> None: ...

    def list_events(self) -> list[TimelineEvent]:
        """All events ordered by (step, insertion)."""
        ...

    def list_edges(self) -> list[BeliefEdge]: ...

    def count_events_by_op(self) -> dict[str, int]: ...


class BeliefSource(Protocol):
    """Read side of the belief ledger (memory's BeliefRepository satisfies it)."""

    def list_beliefs(self) -> list[Belief]: ...


class ReplayWriter(Protocol):
    def write(self, bundle: ReplayBundle) -> None: ...
