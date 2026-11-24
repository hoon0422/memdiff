from __future__ import annotations

from collections.abc import Callable

import numpy as np

from memdiff.core.models import MemoryContextItem
from memdiff.core.store import BeliefStore

from .naive_rag import NaiveRagPolicy


class RecencyRagPolicy(NaiveRagPolicy):
    """Baseline 2: Time-decay weighted vector RAG over raw chunks."""

    policy_name = "recency_rag"

    def __init__(
        self,
        *,
        store: BeliefStore,
        embed_fn: Callable[[list[str]], list[np.ndarray]],
        recency_decay: float = 0.05,
        max_chunk_chars: int = 2400,
    ) -> None:
        super().__init__(store=store, embed_fn=embed_fn, max_chunk_chars=max_chunk_chars)
        self.recency_decay = recency_decay

    def query(
        self,
        query: str,
        query_image: str | None = None,
        *,
        top_k: int = 6,
    ) -> list[MemoryContextItem]:
        _ = query_image
        q_emb = self.embed_fn([query])[0]
        hits = self.store.search_chunks(
            q_emb,
            top_k=top_k,
            recency_decay=self.recency_decay,
        )
        return [{"type": "text", "value": chunk["text"]} for chunk, _score in hits]
