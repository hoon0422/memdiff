from __future__ import annotations

from collections.abc import Callable

from memdiff.memory.application.formatting import belief_embedding_text, format_revised_belief
from memdiff.memory.application.ports import (
    BeliefExtractor,
    BeliefReconciler,
    BeliefRepository,
    ChunkRepository,
    Embedder,
    Embedding,
    RevisionEventSink,
    UnitOfWork,
)
from memdiff.memory.domain.belief import Belief, BeliefStatus, Provenance, new_belief_id
from memdiff.memory.domain.context import MemoryContextItem, MemoryStats
from memdiff.memory.domain.revision import (
    BeliefAdded,
    BeliefConsolidated,
    BeliefForgotten,
    BeliefRefuted,
    BeliefSuperseded,
    CandidateIgnored,
    RevisionDecision,
    RevisionOp,
)
from memdiff.memory.domain.rules import consolidate, is_noise, should_forget
from memdiff.memory.domain.trajectory import Trajectory

from ._base import SteppedPolicy


class BeliefRevisionPolicy(SteppedPolicy):
    """Policy 4 (MemDiff): provenance, explicit supersession, consolidation, forgetting."""

    policy_name = "belief_revision"

    def __init__(
        self,
        *,
        beliefs: BeliefRepository,
        chunks: ChunkRepository,
        embedder: Embedder,
        extractor: BeliefExtractor,
        reconciler: BeliefReconciler,
        events: RevisionEventSink,
        uow: UnitOfWork,
        new_id: Callable[[], str] = new_belief_id,
        candidate_top_k: int = 5,
        forgetting_durability_threshold: float = 0.25,
        forgetting_max_age: int = 5,
    ) -> None:
        super().__init__()
        self._beliefs = beliefs
        self._chunks = chunks
        self._embedder = embedder
        self._extractor = extractor
        self._reconciler = reconciler
        self._events = events
        self._uow = uow
        self._new_id = new_id
        self.candidate_top_k = candidate_top_k
        self.forgetting_durability_threshold = forgetting_durability_threshold
        self.forgetting_max_age = forgetting_max_age

    def insert(self, trajectory: Trajectory) -> None:
        source_id = self._next_source_id(trajectory)
        try:
            for cand in self._extractor.extract(trajectory, self._step):
                self._reconcile(cand, source_id)
            self._forget_stale(source_id)
        except Exception:
            self._uow.rollback()
            raise
        self._uow.commit()

    def _reconcile(self, cand: Belief, source_id: str) -> None:
        step = self._step
        if is_noise(cand, threshold=self.forgetting_durability_threshold):
            self._events.emit(
                CandidateIgnored(
                    step=step,
                    source_id=source_id,
                    summary=f"Ignored temporary noise on {cand.surface}: {cand.claim[:80]}",
                )
            )
            return

        emb = self._embedder.embed([belief_embedding_text(cand)])[0]
        neighbors = self._beliefs.search(emb, top_k=self.candidate_top_k, active_only=True)
        decision = self._reconciler.decide(cand, neighbors)
        target = decision.target_belief_id

        if decision.op == RevisionOp.IGNORE:
            self._events.emit(
                CandidateIgnored(
                    step=step,
                    source_id=source_id,
                    summary=decision.reason or f"Ignored candidate on {cand.surface}",
                )
            )
        elif decision.op == RevisionOp.SUPERSEDE and target:
            self._supersede(cand, decision, target, emb, source_id)
        elif decision.op == RevisionOp.CONSOLIDATE and target:
            self._consolidate(cand, decision, target, emb, source_id)
        elif decision.op == RevisionOp.REFUTE and target:
            existing = self._require(target)
            reason = decision.reason or cand.claim
            self._beliefs.update(existing.mark_refuted(step=step))
            self._events.emit(
                BeliefRefuted(step=step, source_id=source_id, belief_id=existing.id, reason=reason)
            )
        else:
            belief = self._materialize(cand, cand.claim, source_id, surface=cand.surface)
            self._beliefs.add(belief, emb)
            self._events.emit(
                BeliefAdded(
                    step=step,
                    source_id=source_id,
                    belief_id=belief.id,
                    belief_type=belief.type,
                    surface=belief.surface,
                    claim=belief.claim,
                )
            )

    def _materialize(self, cand: Belief, claim: str, source_id: str, *, surface: str) -> Belief:
        return Belief.new(
            id=self._new_id(),
            type=cand.type,
            surface=surface,
            claim=claim,
            step=self._step,
            source_id=source_id,
            provenance=cand.provenance,
            closed_set_items=cand.closed_set_items,
            confidence=cand.confidence,
            durability=cand.durability,
            metadata=cand.metadata,
        )

    def _require(self, belief_id: str) -> Belief:
        belief = self._beliefs.get(belief_id)
        if belief is None:
            raise KeyError(f"Unknown belief id: {belief_id}")
        return belief

    def _supersede(
        self,
        cand: Belief,
        decision: RevisionDecision,
        target: str,
        emb: Embedding,
        source_id: str,
    ) -> None:
        old = self._require(target)
        new_claim = decision.merged_claim or cand.claim
        new = self._materialize(cand, new_claim, source_id, surface=cand.surface or old.surface)
        self._beliefs.add(new, emb)
        self._beliefs.update(old.mark_superseded(by=new.id, step=self._step))
        # SUPERSEDE replaces the ADD event: the new belief is not announced separately.
        self._events.emit(
            BeliefSuperseded(
                step=self._step,
                source_id=source_id,
                old_belief_id=old.id,
                old_claim=old.claim,
                new_belief_id=new.id,
                new_claim=new_claim,
                reason=decision.reason,
            )
        )

    def _consolidate(
        self,
        cand: Belief,
        decision: RevisionDecision,
        target: str,
        emb: Embedding,
        source_id: str,
    ) -> None:
        existing = self._require(target)
        updated = consolidate(
            existing,
            step=self._step,
            provenance=cand.provenance[0] if cand.provenance else Provenance(source_id=source_id),
            merged_claim=decision.merged_claim,
            closed_set_items=cand.closed_set_items,
        )
        self._beliefs.update(updated, emb if decision.merged_claim else None)
        self._events.emit(
            BeliefConsolidated(
                step=self._step,
                source_id=source_id,
                belief_id=updated.id,
                surface=updated.surface,
                evidence_count=updated.evidence_count,
                previous=existing,
            )
        )

    def _forget_stale(self, source_id: str) -> None:
        for belief in self._beliefs.list_beliefs(status=BeliefStatus.ACTIVE):
            if should_forget(belief, current_step=self._step, max_age=self.forgetting_max_age):
                age = self._step - belief.updated_at_step
                self._beliefs.update(belief.mark_forgotten(step=self._step))
                self._events.emit(
                    BeliefForgotten(
                        step=self._step,
                        source_id=source_id,
                        belief_id=belief.id,
                        reason=f"temporary_age_eviction(age={age})",
                    )
                )

    def query(
        self, query: str, query_image: str | None = None, *, top_k: int = 6
    ) -> list[MemoryContextItem]:
        _ = query_image
        q_emb = self._embedder.embed([query])[0]
        hits = self._beliefs.search(q_emb, top_k=top_k, active_only=True)
        return [
            MemoryContextItem(type="text", value=format_revised_belief(b, score))
            for b, score in hits
        ]

    def stats(self) -> MemoryStats:
        return MemoryStats.build(
            beliefs=self._beliefs.list_beliefs(),
            raw_chunks=self._chunks.count(),
            revision_counts=self._events.op_counts(),
            total_ingested_sources=self._step,
        )
