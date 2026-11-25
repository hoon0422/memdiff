"""In-memory fake ports for the memory/audit cores (offline, deterministic)."""

from __future__ import annotations

import itertools
import zlib
from collections.abc import Callable, Sequence

import numpy as np

from memdiff.audit.domain.timeline import BeliefEdge, TimelineEvent
from memdiff.memory.domain.belief import Belief, BeliefStatus
from memdiff.memory.domain.chunk import Chunk
from memdiff.memory.domain.revision import RevisionDecision, RevisionEvent
from memdiff.memory.domain.trajectory import Trajectory


def _normalized(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _cosine_top(
    query: Sequence[float], rows: list[tuple[str, np.ndarray]], scale: np.ndarray | None = None
) -> list[tuple[str, float]]:
    q = _normalized(query)
    rows = [(rid, v) for rid, v in rows if v.shape == q.shape]
    if not rows:
        return []
    scores = np.vstack([v for _, v in rows]) @ q
    if scale is not None:
        scores = scores * scale
    order = np.argsort(scores)[::-1]
    return [(rows[int(i)][0], float(scores[int(i)])) for i in order]


class InMemoryBeliefRepository:
    def __init__(self) -> None:
        self._rows: dict[str, Belief] = {}
        self._vecs: dict[str, np.ndarray] = {}

    def add(self, belief: Belief, embedding: Sequence[float] | None) -> None:
        self._rows[belief.id] = belief
        if embedding is not None:
            self._vecs[belief.id] = _normalized(embedding)

    def update(self, belief: Belief, embedding: Sequence[float] | None = None) -> None:
        assert belief.id in self._rows
        self._rows[belief.id] = belief
        if embedding is not None:
            self._vecs[belief.id] = _normalized(embedding)

    def get(self, belief_id: str) -> Belief | None:
        return self._rows.get(belief_id)

    def list_beliefs(
        self, *, status: BeliefStatus | None = None, surface: str | None = None
    ) -> list[Belief]:
        rows = [
            b
            for b in self._rows.values()
            if (status is None or b.status == status) and (surface is None or b.surface == surface)
        ]
        return sorted(rows, key=lambda b: (b.created_at_step, b.id))

    def search(
        self, embedding: Sequence[float], *, top_k: int, active_only: bool
    ) -> list[tuple[Belief, float]]:
        if top_k <= 0 or len(embedding) == 0:
            return []
        rows = [
            (bid, v)
            for bid, v in self._vecs.items()
            if not active_only or self._rows[bid].status == BeliefStatus.ACTIVE
        ]
        return [(self._rows[bid], s) for bid, s in _cosine_top(embedding, rows)[:top_k]]


class InMemoryChunkRepository:
    def __init__(self) -> None:
        self._rows: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}

    def add(self, chunk: Chunk, embedding: Sequence[float] | None) -> None:
        self._rows[chunk.id] = chunk
        if embedding is not None:
            self._vecs[chunk.id] = _normalized(embedding)

    def search(
        self, embedding: Sequence[float], *, top_k: int, recency_decay: float
    ) -> list[tuple[Chunk, float]]:
        if top_k <= 0 or len(embedding) == 0 or not self._vecs:
            return []
        rows = list(self._vecs.items())
        scale = None
        if recency_decay > 0.0:
            steps = np.asarray([self._rows[cid].step for cid, _ in rows], dtype=np.int64)
            scale = np.exp(-recency_decay * np.maximum(0, int(steps.max()) - steps))
        return [(self._rows[cid], s) for cid, s in _cosine_top(embedding, rows, scale)[:top_k]]

    def count(self) -> int:
        return len(self._rows)


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class InMemoryAuditLog:
    def __init__(self) -> None:
        self._events: list[TimelineEvent] = []
        self._edges: list[BeliefEdge] = []

    def add_event(self, event: TimelineEvent) -> None:
        self._events.append(event.model_copy(update={"id": len(self._events) + 1}))

    def add_edge(self, edge: BeliefEdge) -> None:
        self._edges.append(edge)

    def list_events(self) -> list[TimelineEvent]:
        return sorted(self._events, key=lambda e: (e.step, e.id or 0))

    def list_edges(self) -> list[BeliefEdge]:
        return sorted(self._edges, key=lambda e: e.step)

    def count_events_by_op(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.op.value] = counts.get(e.op.value, 0) + 1
        return counts


class ListEventSink:
    """RevisionEventSink that just records emitted domain events."""

    def __init__(self) -> None:
        self.events: list[RevisionEvent] = []

    def emit(self, event: RevisionEvent) -> None:
        self.events.append(event)

    def op_counts(self) -> dict[str, int]:
        return {}


class HashEmbedder:
    """Deterministic pseudo-embedding keyed on text (crc32 -> gaussian)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [
            np.random.default_rng(zlib.crc32(t.encode())).standard_normal(self.dim)
            for t in texts
        ]


class ScriptedExtractor:
    def __init__(self, fn: Callable[[Trajectory, int], list[Belief]]) -> None:
        self._fn = fn

    def extract(self, trajectory: Trajectory, step: int) -> list[Belief]:
        return self._fn(trajectory, step)


class ScriptedReconciler:
    def __init__(
        self, fn: Callable[[Belief, list[tuple[Belief, float]]], RevisionDecision]
    ) -> None:
        self._fn = fn

    def decide(
        self, candidate: Belief, neighbors: list[tuple[Belief, float]]
    ) -> RevisionDecision:
        return self._fn(candidate, neighbors)


def counter_ids(prefix: str = "b_") -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"{prefix}{next(counter):010x}"
