"""The four policies on fake ports: same trajectories, same retrieval budget, auditable ledger."""

from __future__ import annotations

from typing import Any

import pytest

from memdiff.audit.application.recorder import RevisionAuditRecorder
from memdiff.audit.application.replay import ReplayService
from memdiff.memory.application.policies import (
    BeliefRevisionPolicy,
    NaiveRagPolicy,
    RecencyRagPolicy,
    StructuredExtractPolicy,
)
from memdiff.memory.domain.belief import Belief, BeliefType, Provenance
from memdiff.memory.domain.revision import RevisionDecision, RevisionOp
from memdiff.memory.domain.trajectory import Trajectory
from memory_fakes import (
    FakeUnitOfWork,
    HashEmbedder,
    InMemoryAuditLog,
    InMemoryBeliefRepository,
    InMemoryChunkRepository,
    ScriptedExtractor,
    ScriptedReconciler,
    counter_ids,
)


def _cand(btype: BeliefType, surface: str, claim: str, *, op: str = "ADD", **kw: Any) -> Belief:
    meta = {"op": op, **kw.pop("metadata", {})}
    return Belief.new(
        id="cand", type=btype, surface=surface, claim=claim, step=0, source_id="x",
        metadata=meta, **kw,
    )


_D = BeliefType.DYNAMIC_STATE
_SCRIPT: dict[str, list[Belief]] = {
    "t1": [
        _cand(_D, "billing.region", "region is us-east-1",
              provenance=[Provenance(source_id="t1", step_index=0, outcome="success")]),
        _cand(BeliefType.DURABLE_RULE, "rules.deploy", "never deploy on friday", durability=0.95),
        _cand(BeliefType.TEMPORARY, "staging.build", "build 17 queued", durability=0.4),
        _cand(BeliefType.TEMPORARY, "debug.ping", "one-off ping", durability=0.1),
        _cand(BeliefType.STATIC_SURFACE, "ui.filters", "filters visible", closed_set_items=["a", "b"]),
        _cand(BeliefType.GOTCHA, "ui.save", "save button is hidden"),
    ],
    "t2": [
        _cand(_D, "billing.region", "region is eu-central-1", op="SUPERSEDE",
              metadata={"reason": "migrated"}),
        _cand(BeliefType.STATIC_SURFACE, "ui.filters", "filters seen", op="CONSOLIDATE",
              closed_set_items=["c", "a"]),
    ],
    "t3": [
        _cand(BeliefType.STATIC_SURFACE, "ui.filters", "filters again", op="CONSOLIDATE",
              closed_set_items=["d"], metadata={"merged": "filters: a b c d"}),
        _cand(BeliefType.GOTCHA, "ui.save", "save works via menu", op="REFUTE"),
        _cand(_D, "billing.region", "dup", op="IGNORE"),
        _cand(_D, "billing.region", "region is eu-central-1 (again)", op="IGNORE",
              metadata={"reason": "redundant"}),
    ],
    "t4": [
        _cand(_D, "svc.a", "a is 1"),
        # no merged claim: the stored embedding must stay untouched
        _cand(BeliefType.STATIC_SURFACE, "ui.filters", "filters third", op="CONSOLIDATE",
              closed_set_items=["e"]),
    ],
    "t5": [_cand(_D, "svc.b", "b is 2")],
    "t6": [_cand(_D, "svc.c", "c is 3")],
    "t7": [_cand(_D, "svc.d", "d is 4")],
}
_TRAJECTORIES = [
    {
        "id": tid,
        "goal": f"goal {tid}",
        "outcome": "success",
        "states": [
            {"state_index": i, "url": f"http://x/{tid}/{i}", "action": None if i == 0 else "click",
             "accessibility_tree": f"tree {tid} {i} " + "x" * 50}
            for i in range(2)
        ],
    }
    for tid in _SCRIPT
]
_QUERIES = ["billing region", "filters", "save button", "build queued", "svc c"]


def _extract(trajectory: Trajectory, step: int) -> list[Belief]:
    return [c.model_copy(update={"created_at_step": step, "updated_at_step": step})
            for c in _SCRIPT[trajectory.id or ""]]


def _decide(cand: Any, neighbors: list[Any]) -> tuple[str, str | None, str | None, str]:
    op = cand.metadata["op"]
    same = [b.id for b, _ in neighbors if b.surface == cand.surface]
    if op == "ADD" or not same:
        return "ADD", None, None, ""
    return op, same[0], cand.metadata.get("merged"), cand.metadata.get("reason", "")


def _new_stack(name: str) -> tuple[Any, ReplayService, InMemoryBeliefRepository]:
    beliefs, chunks, log, uow = (
        InMemoryBeliefRepository(), InMemoryChunkRepository(), InMemoryAuditLog(), FakeUnitOfWork(),
    )
    sink, embedder, extractor = RevisionAuditRecorder(log), HashEmbedder(), ScriptedExtractor(_extract)
    ids = counter_ids("b_")
    if name == "naive_rag":
        policy: Any = NaiveRagPolicy(chunks=chunks, embedder=embedder, uow=uow)
    elif name == "recency_rag":
        policy = RecencyRagPolicy(chunks=chunks, embedder=embedder, uow=uow)
    elif name == "structured_extract":
        policy = StructuredExtractPolicy(
            beliefs=beliefs, chunks=chunks, embedder=embedder, extractor=extractor,
            events=sink, uow=uow, new_id=ids,
        )
    else:
        def decide(cand: Belief, neighbors: list[tuple[Belief, float]]) -> RevisionDecision:
            op, target, merged, reason = _decide(cand, neighbors)
            return RevisionDecision(op=RevisionOp(op), target_belief_id=target,
                                    merged_claim=merged, reason=reason)

        policy = BeliefRevisionPolicy(
            beliefs=beliefs, chunks=chunks, embedder=embedder, extractor=extractor,
            reconciler=ScriptedReconciler(decide), events=sink, uow=uow, new_id=ids,
        )
    return policy, ReplayService(beliefs=beliefs, log=log), beliefs


@pytest.mark.parametrize(
    "name", ["naive_rag", "recency_rag", "structured_extract", "belief_revision"]
)
def test_every_policy_answers_every_query_within_the_budget(name: str) -> None:
    policy, replay, _ = _new_stack(name)
    for traj in _TRAJECTORIES:
        policy.insert(Trajectory.model_validate(traj))

    for q in _QUERIES:
        for top_k in (2, 6):
            items = policy.query(q, top_k=top_k)
            assert 0 < len(items) <= top_k

    bundle = replay.build_bundle(
        run_id="r", benchmark="b", policy_name=name, total_steps=7, stats=policy.stats()
    )
    if name == "belief_revision":
        assert set(policy.stats().revision_counts) == {
            "ADD", "SUPERSEDE", "CONSOLIDATE", "REFUTE", "IGNORE", "FORGET"
        }
        assert "SUPERSEDES" in {e.relation for e in bundle.edges}
    if name in ("structured_extract", "belief_revision"):
        final = {b.id: b.status for b in replay.state_at_step(8)}
        assert final == {b.id: b.status for b in bundle.beliefs}
        assert replay.state_at_step(0) == []


def test_resume_from_sets_step_explicitly() -> None:
    policy, _, _ = _new_stack("belief_revision")
    policy.resume_from(10)
    policy.insert(Trajectory.model_validate(_TRAJECTORIES[3]))
    assert policy.step == 11 and policy.stats().total_ingested_sources == 11
    with pytest.raises(ValueError):
        policy.resume_from(-1)
