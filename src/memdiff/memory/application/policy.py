from __future__ import annotations

from typing import Protocol

from memdiff.memory.domain.context import MemoryContextItem, MemoryStats
from memdiff.memory.domain.trajectory import Trajectory


class MemoryPolicy(Protocol):
    """Common interface for the 4 matched-budget memory policies."""

    policy_name: str

    def insert(self, trajectory: Trajectory) -> None: ...

    def query(
        self, query: str, query_image: str | None = None, *, top_k: int = 6
    ) -> list[MemoryContextItem]: ...

    def stats(self) -> MemoryStats: ...

    def resume_from(self, step: int) -> None:
        """Continue numbering ingested sources after `step` (e.g. when reopening a ledger)."""
        ...
