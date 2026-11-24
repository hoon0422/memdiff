from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from memdiff.core.models import MemoryContextItem, MemoryStats
from memdiff.core.protocol import MemoryPolicy
from memdiff.core.store import BeliefStore


class NaiveRagPolicy(MemoryPolicy):
    """Baseline 1: Append-only vector RAG over trajectory state chunks."""

    policy_name = "naive_rag"

    def __init__(
        self,
        *,
        store: BeliefStore,
        embed_fn: Callable[[list[str]], list[np.ndarray]],
        max_chunk_chars: int = 2400,
    ) -> None:
        self.store = store
        self.embed_fn = embed_fn
        self.max_chunk_chars = max_chunk_chars
        self._step = 0

    def _format_state_chunk(self, trajectory: dict[str, Any], state: dict[str, Any]) -> str:
        tid = str(trajectory.get("id", "unknown"))
        goal = str(trajectory.get("goal", ""))
        outcome = str(trajectory.get("outcome", "unknown"))
        url = str(state.get("url", ""))
        action = state.get("action") or "<none>"
        axtree = str(state.get("accessibility_tree", state.get("text", "")))
        header = (
            f"[Trajectory {tid} | outcome={outcome} | step={state.get('state_index', 0)}]\n"
            f"Goal: {goal[:240]}\n"
            f"URL: {url}\n"
            f"Action: {action}\n"
        )
        budget = max(200, self.max_chunk_chars - len(header))
        return header + axtree[:budget]

    def insert(self, trajectory: dict[str, Any]) -> None:
        self._step += 1
        source_id = str(trajectory.get("id", f"traj_{self._step}"))
        states = trajectory.get("states", [])
        if not isinstance(states, list) or not states:
            return
        chunks = [self._format_state_chunk(trajectory, s) for s in states if isinstance(s, dict)]
        if not chunks:
            return
        embeddings = self.embed_fn(chunks)
        for idx, (text, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            self.store.add_chunk(
                source_id=source_id,
                step=self._step,
                state_index=idx,
                text=text,
                metadata={"outcome": trajectory.get("outcome")},
                embedding=emb,
            )

    def query(
        self,
        query: str,
        query_image: str | None = None,
        *,
        top_k: int = 6,
    ) -> list[MemoryContextItem]:
        _ = query_image
        q_emb = self.embed_fn([query])[0]
        hits = self.store.search_chunks(q_emb, top_k=top_k, recency_decay=0.0)
        return [{"type": "text", "value": chunk["text"]} for chunk, _score in hits]

    def stats(self) -> MemoryStats:
        return self.store.stats(total_ingested_sources=self._step)
