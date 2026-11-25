from __future__ import annotations

from collections.abc import Iterable

from memdiff.audit.application.ports import AuditLogRepository, BeliefSource, ReplayWriter
from memdiff.audit.domain.replay import ReplayBundle, ReplayQuestionComparison
from memdiff.audit.domain.time_travel import reconstruct_state_at_step
from memdiff.memory.domain.belief import Belief
from memdiff.memory.domain.context import MemoryStats


class ReplayService:
    """Builds replay bundles and time-travel snapshots from the ledger and audit log."""

    def __init__(
        self, *, beliefs: BeliefSource, log: AuditLogRepository, writer: ReplayWriter | None = None
    ) -> None:
        self._beliefs = beliefs
        self._log = log
        self._writer = writer

    def state_at_step(self, step: int) -> list[Belief]:
        return reconstruct_state_at_step(
            self._beliefs.list_beliefs(), self._log.list_events(), step
        )

    def build_bundle(
        self,
        *,
        run_id: str,
        benchmark: str,
        policy_name: str,
        total_steps: int,
        stats: MemoryStats,
        comparisons: Iterable[ReplayQuestionComparison] = (),
    ) -> ReplayBundle:
        return ReplayBundle(
            run_id=run_id,
            benchmark=benchmark,
            policy_name=policy_name,
            total_steps=total_steps,
            beliefs=self._beliefs.list_beliefs(),
            edges=self._log.list_edges(),
            events=self._log.list_events(),
            stats=stats,
            comparisons=list(comparisons),
        )

    def export(self, bundle: ReplayBundle) -> None:
        if self._writer is None:
            raise RuntimeError("ReplayService has no ReplayWriter configured")
        self._writer.write(bundle)
