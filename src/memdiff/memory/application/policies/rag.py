from __future__ import annotations

from collections.abc import Callable

from memdiff.memory.application.formatting import format_state_chunk
from memdiff.memory.application.ports import ChunkRepository, Embedder, UnitOfWork
from memdiff.memory.domain.chunk import Chunk, new_chunk_id
from memdiff.memory.domain.context import MemoryContextItem, MemoryStats
from memdiff.memory.domain.trajectory import Trajectory

from ._base import SteppedPolicy


class RagPolicy(SteppedPolicy):
    """Baselines 1 and 2: append-only vector RAG over trajectory state chunks.

    `naive_rag` is `recency_decay=0.0`; `recency_rag` weights scores by `exp(-decay * age)`.
    """

    def __init__(
        self,
        *,
        policy_name: str,
        chunks: ChunkRepository,
        embedder: Embedder,
        uow: UnitOfWork,
        new_id: Callable[[], str] = new_chunk_id,
        recency_decay: float = 0.0,
        max_chunk_chars: int = 2400,
    ) -> None:
        super().__init__()
        self.policy_name = policy_name
        self._recency_decay = recency_decay
        self._chunks = chunks
        self._embedder = embedder
        self._uow = uow
        self._new_id = new_id
        self.max_chunk_chars = max_chunk_chars

    def insert(self, trajectory: Trajectory) -> None:
        source_id = self._next_source_id(trajectory)
        texts = [format_state_chunk(trajectory, s, self.max_chunk_chars) for s in trajectory.states]
        if not texts:
            return
        embeddings = self._embedder.embed(texts)
        for idx, (text, emb) in enumerate(zip(texts, embeddings, strict=True)):
            chunk = Chunk(
                id=self._new_id(),
                source_id=source_id,
                step=self._step,
                state_index=idx,
                text=text,
                metadata={"outcome": trajectory.outcome},
            )
            self._chunks.add(chunk, emb)
        self._uow.commit()

    def query(
        self, query: str, query_image: str | None = None, *, top_k: int = 6
    ) -> list[MemoryContextItem]:
        _ = query_image
        q_emb = self._embedder.embed([query])[0]
        hits = self._chunks.search(q_emb, top_k=top_k, recency_decay=self._recency_decay)
        return [MemoryContextItem(type="text", value=chunk.text) for chunk, _score in hits]

    def stats(self) -> MemoryStats:
        return MemoryStats.build(
            beliefs=[],
            raw_chunks=self._chunks.count(),
            revision_counts={},
            total_ingested_sources=self._step,
        )
