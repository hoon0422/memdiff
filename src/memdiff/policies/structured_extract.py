from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from memdiff.core.models import Belief, MemoryContextItem, MemoryStats
from memdiff.core.protocol import MemoryPolicy
from memdiff.core.store import BeliefStore


class StructuredExtractPolicy(MemoryPolicy):
    """Baseline 3: Typed belief extraction without supersession or forgetting.

    Appends extracted beliefs directly into the BeliefStore without checking for
    contradictions, consolidation, or low-durability eviction.
    """

    policy_name = "structured_extract"

    def __init__(
        self,
        *,
        store: BeliefStore,
        embed_fn: Callable[[list[str]], list[np.ndarray]],
        extractor_fn: Callable[[dict[str, Any], int], list[Belief]],
    ) -> None:
        self.store = store
        self.embed_fn = embed_fn
        self.extractor_fn = extractor_fn
        self._step = 0

    def insert(self, trajectory: dict[str, Any]) -> None:
        self._step += 1
        source_id = str(trajectory.get("id", f"traj_{self._step}"))
        extracted = self.extractor_fn(trajectory, self._step)
        if not extracted:
            return
        texts = [f"[{b.type.value}] {b.surface}: {b.claim}" for b in extracted]
        embeddings = self.embed_fn(texts)
        for belief, emb in zip(extracted, embeddings, strict=True):
            self.store.add_belief(
                belief_type=belief.type,
                surface=belief.surface,
                claim=belief.claim,
                step=self._step,
                source_id=source_id,
                provenance=belief.provenance,
                closed_set_items=belief.closed_set_items,
                confidence=belief.confidence,
                durability=belief.durability,
                metadata=belief.metadata,
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
        hits = self.store.search_beliefs(q_emb, top_k=top_k, active_only=False)
        return [
            {
                "type": "text",
                "value": self._format_belief(belief),
            }
            for belief, _score in hits
        ]

    @staticmethod
    def _format_belief(belief: Belief) -> str:
        prov = ", ".join(
            f"{p.source_id}@step{p.step_index}" if p.step_index is not None else p.source_id
            for p in belief.provenance[:3]
        )
        closed = (
            f"\nClosed-set visible items: {', '.join(belief.closed_set_items)}"
            if belief.closed_set_items
            else ""
        )
        return (
            f"[{belief.id} | {belief.type.value} | status={belief.status.value} | surface={belief.surface}]\n"
            f"{belief.claim}{closed}\n"
            f"Provenance: {prov}"
        )

    def stats(self) -> MemoryStats:
        return self.store.stats(total_ingested_sources=self._step)
