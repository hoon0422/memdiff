from __future__ import annotations

from collections.abc import Callable

from memdiff.memory.application.formatting import belief_embedding_text, format_structured_belief
from memdiff.memory.application.ports import (
    BeliefExtractor,
    BeliefRepository,
    ChunkRepository,
    Embedder,
    RevisionEventSink,
    UnitOfWork,
)
from memdiff.memory.domain.belief import new_belief_id
from memdiff.memory.domain.context import MemoryContextItem, MemoryStats
from memdiff.memory.domain.trajectory import Trajectory

from ._base import SteppedPolicy, emit_added, materialize


class StructuredExtractPolicy(SteppedPolicy):
    """Baseline 3: Typed belief extraction without supersession or forgetting."""

    policy_name = "structured_extract"

    def __init__(
        self,
        *,
        beliefs: BeliefRepository,
        chunks: ChunkRepository,
        embedder: Embedder,
        extractor: BeliefExtractor,
        events: RevisionEventSink,
        uow: UnitOfWork,
        new_id: Callable[[], str] = new_belief_id,
    ) -> None:
        super().__init__()
        self._beliefs = beliefs
        self._chunks = chunks
        self._embedder = embedder
        self._extractor = extractor
        self._events = events
        self._uow = uow
        self._new_id = new_id

    def insert(self, trajectory: Trajectory) -> None:
        source_id = self._next_source_id(trajectory)
        extracted = self._extractor.extract(trajectory, self._step)
        if not extracted:
            return
        embeddings = self._embedder.embed([belief_embedding_text(b) for b in extracted])
        for cand, emb in zip(extracted, embeddings, strict=True):
            belief = materialize(
                cand,
                id=self._new_id(),
                claim=cand.claim,
                surface=cand.surface,
                step=self._step,
                source_id=source_id,
            )
            self._beliefs.add(belief, emb)
            emit_added(self._events, belief, step=self._step, source_id=source_id)
        self._uow.commit()

    def query(
        self, query: str, query_image: str | None = None, *, top_k: int = 6
    ) -> list[MemoryContextItem]:
        _ = query_image
        q_emb = self._embedder.embed([query])[0]
        hits = self._beliefs.search(q_emb, top_k=top_k, active_only=False)
        return [
            MemoryContextItem(type="text", value=format_structured_belief(b)) for b, _s in hits
        ]

    def stats(self) -> MemoryStats:
        return MemoryStats.build(
            beliefs=self._beliefs.list_beliefs(),
            raw_chunks=self._chunks.count(),
            revision_counts=self._events.op_counts(),
            total_ingested_sources=self._step,
        )
