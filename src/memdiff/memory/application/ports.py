"""Ports the memory policies depend on. Adapters (SQL, numpy index, LLM) implement them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memdiff.memory.domain.belief import Belief, BeliefStatus
from memdiff.memory.domain.chunk import Chunk
from memdiff.memory.domain.revision import RevisionDecision, RevisionEvent
from memdiff.memory.domain.trajectory import Trajectory

Embedding = Sequence[float]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[Embedding]: ...


class BeliefExtractor(Protocol):
    def extract(self, trajectory: Trajectory, step: int) -> list[Belief]: ...


class BeliefReconciler(Protocol):
    def decide(
        self, candidate: Belief, neighbors: list[tuple[Belief, float]]
    ) -> RevisionDecision: ...


class BeliefRepository(Protocol):
    def add(self, belief: Belief, embedding: Embedding | None) -> None: ...

    def update(self, belief: Belief, embedding: Embedding | None = None) -> None:
        """Replace the row; keep the stored embedding unless `embedding` is given."""
        ...

    def get(self, belief_id: str) -> Belief | None: ...

    def list_beliefs(
        self, *, status: BeliefStatus | None = None, surface: str | None = None
    ) -> list[Belief]: ...

    def search(
        self, embedding: Embedding, *, top_k: int, active_only: bool
    ) -> list[tuple[Belief, float]]: ...


class ChunkRepository(Protocol):
    def add(self, chunk: Chunk, embedding: Embedding | None) -> None: ...

    def search(
        self, embedding: Embedding, *, top_k: int, recency_decay: float
    ) -> list[tuple[Chunk, float]]: ...

    def count(self) -> int: ...


class RevisionEventSink(Protocol):
    def emit(self, event: RevisionEvent) -> None: ...

    def op_counts(self) -> dict[str, int]:
        """Revision events emitted so far, keyed by op name (feeds MemoryStats)."""
        ...


class UnitOfWork(Protocol):
    def commit(self) -> None: ...

    def rollback(self) -> None: ...
