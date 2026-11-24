from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import MemoryContextItem, MemoryStats


class MemoryPolicy(ABC):
    """Common interface for all 4 matched-budget memory policies.

    Compatible with LongMemEval-V2's `insert(trajectory)` and `query(query, query_image)`
    contract while adding introspection hooks for footprint metrics and replay export.
    """

    policy_name: str = "base"

    @abstractmethod
    def insert(self, trajectory: dict[str, Any]) -> None:
        """Ingest one trajectory (or DriftSim session) in chronological order."""

    @abstractmethod
    def query(
        self,
        query: str,
        query_image: str | None = None,
        *,
        top_k: int = 6,
    ) -> list[MemoryContextItem]:
        """Retrieve compact context items for downstream QA."""

    @abstractmethod
    def stats(self) -> MemoryStats:
        """Return memory footprint and revision lifecycle statistics."""
