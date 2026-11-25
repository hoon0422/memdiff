from __future__ import annotations

from collections.abc import Callable

from memdiff.memory.application.ports import ChunkRepository, Embedder, UnitOfWork
from memdiff.memory.domain.chunk import new_chunk_id

from .naive_rag import NaiveRagPolicy


class RecencyRagPolicy(NaiveRagPolicy):
    """Baseline 2: Time-decay weighted vector RAG over raw chunks."""

    policy_name = "recency_rag"

    def __init__(
        self,
        *,
        chunks: ChunkRepository,
        embedder: Embedder,
        uow: UnitOfWork,
        new_id: Callable[[], str] = new_chunk_id,
        recency_decay: float = 0.05,
        max_chunk_chars: int = 2400,
    ) -> None:
        super().__init__(
            chunks=chunks,
            embedder=embedder,
            uow=uow,
            new_id=new_id,
            max_chunk_chars=max_chunk_chars,
        )
        self._recency_decay = recency_decay
