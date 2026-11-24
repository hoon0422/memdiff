from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from memdiff.core.models import (
    Belief,
    BeliefStatus,
    BeliefType,
    MemoryContextItem,
    MemoryStats,
    Provenance,
    RevisionOp,
)
from memdiff.core.protocol import MemoryPolicy
from memdiff.core.store import BeliefStore


class RevisionDecision:
    """Decision returned by a reconciler comparing a candidate belief against active beliefs."""

    def __init__(
        self,
        op: RevisionOp,
        *,
        target_belief_id: str | None = None,
        merged_claim: str | None = None,
        reason: str = "",
    ) -> None:
        self.op = op
        self.target_belief_id = target_belief_id
        self.merged_claim = merged_claim
        self.reason = reason


class BeliefRevisionPolicy(MemoryPolicy):
    """Policy 4 (Proposed — MemDiff): Provenance tracking, explicit supersession, consolidation & forgetting."""

    policy_name = "belief_revision"

    def __init__(
        self,
        *,
        store: BeliefStore,
        embed_fn: Callable[[list[str]], list[np.ndarray]],
        extractor_fn: Callable[[dict[str, Any], int], list[Belief]],
        reconciler_fn: Callable[[Belief, list[tuple[Belief, float]]], RevisionDecision],
        candidate_top_k: int = 5,
        forgetting_durability_threshold: float = 0.25,
        forgetting_max_age: int = 5,
    ) -> None:
        self.store = store
        self.embed_fn = embed_fn
        self.extractor_fn = extractor_fn
        self.reconciler_fn = reconciler_fn
        self.candidate_top_k = candidate_top_k
        self.forgetting_durability_threshold = forgetting_durability_threshold
        self.forgetting_max_age = forgetting_max_age
        self._step = 0

    def insert(self, trajectory: dict[str, Any]) -> None:
        self._step += 1
        source_id = str(trajectory.get("id", f"traj_{self._step}"))
        candidates = self.extractor_fn(trajectory, self._step)

        for cand in candidates:
            if cand.type == BeliefType.TEMPORARY and cand.durability < self.forgetting_durability_threshold:
                self.store.log_ignore(
                    step=self._step,
                    source_id=source_id,
                    summary=f"Ignored temporary noise on {cand.surface}: {cand.claim[:80]}",
                )
                continue

            cand_text = f"[{cand.type.value}] {cand.surface}: {cand.claim}"
            cand_emb = self.embed_fn([cand_text])[0]
            neighbors = self.store.search_beliefs(
                cand_emb,
                top_k=self.candidate_top_k,
                active_only=True,
            )
            decision = self.reconciler_fn(cand, neighbors)

            if decision.op == RevisionOp.IGNORE:
                self.store.log_ignore(
                    step=self._step,
                    source_id=source_id,
                    summary=decision.reason or f"Ignored candidate on {cand.surface}",
                )
            elif decision.op == RevisionOp.SUPERSEDE and decision.target_belief_id:
                self.store.supersede_belief(
                    old_belief_id=decision.target_belief_id,
                    new_claim=decision.merged_claim or cand.claim,
                    step=self._step,
                    source_id=source_id,
                    reason=decision.reason,
                    belief_type=cand.type,
                    surface=cand.surface,
                    provenance=cand.provenance,
                    closed_set_items=cand.closed_set_items,
                    confidence=cand.confidence,
                    durability=cand.durability,
                    metadata=cand.metadata,
                    embedding=cand_emb,
                )
            elif decision.op == RevisionOp.CONSOLIDATE and decision.target_belief_id:
                prov = cand.provenance[0] if cand.provenance else Provenance(source_id=source_id)
                self.store.consolidate_belief(
                    belief_id=decision.target_belief_id,
                    step=self._step,
                    source_id=source_id,
                    new_provenance=prov,
                    merged_claim=decision.merged_claim,
                    closed_set_items=cand.closed_set_items,
                    embedding=cand_emb if decision.merged_claim else None,
                )
            elif decision.op == RevisionOp.REFUTE and decision.target_belief_id:
                self.store.refute_belief(
                    belief_id=decision.target_belief_id,
                    step=self._step,
                    source_id=source_id,
                    reason=decision.reason or cand.claim,
                )
            else:
                self.store.add_belief(
                    belief_type=cand.type,
                    surface=cand.surface,
                    claim=cand.claim,
                    step=self._step,
                    source_id=source_id,
                    provenance=cand.provenance,
                    closed_set_items=cand.closed_set_items,
                    confidence=cand.confidence,
                    durability=cand.durability,
                    metadata=cand.metadata,
                    embedding=cand_emb,
                )

        self._run_forgetting_pass(source_id=source_id)

    def _run_forgetting_pass(self, *, source_id: str) -> None:
        for belief in self.store.list_beliefs(status=BeliefStatus.ACTIVE):
            age = self._step - belief.updated_at_step
            if (
                belief.type == BeliefType.TEMPORARY
                and age >= self.forgetting_max_age
                and belief.durability <= 0.5
            ):
                self.store.forget_belief(
                    belief_id=belief.id,
                    step=self._step,
                    source_id=source_id,
                    reason=f"temporary_age_eviction(age={age})",
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
        hits = self.store.search_beliefs(q_emb, top_k=top_k, active_only=True)
        items: list[MemoryContextItem] = []
        for belief, score in hits:
            prov = ", ".join(
                f"{p.source_id}@step{p.step_index}({p.outcome})"
                if p.step_index is not None
                else f"{p.source_id}({p.outcome})"
                for p in belief.provenance[:4]
            )
            closed = (
                f"\nClosed-set observed items: {', '.join(belief.closed_set_items)}"
                if belief.closed_set_items
                else ""
            )
            items.append(
                {
                    "type": "text",
                    "value": (
                        f"[{belief.id} | {belief.type.value} | surface={belief.surface} | "
                        f"evidence={belief.evidence_count} | sim={score:.3f}]\n"
                        f"{belief.claim}{closed}\n"
                        f"Provenance: {prov}"
                    ),
                }
            )
        return items

    def stats(self) -> MemoryStats:
        return self.store.stats(total_ingested_sources=self._step)
