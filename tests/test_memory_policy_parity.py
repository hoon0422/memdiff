"""New memory/audit cores (fake ports) must match the legacy BeliefStore + policies exactly."""

from __future__ import annotations

import types
from typing import Any

import pytest

from memdiff.audit.application.recorder import RevisionAuditRecorder
from memdiff.audit.application.replay import ReplayService
from memdiff.core.models import Belief as LegacyBelief
from memdiff.core.models import RevisionOp as LegacyOp
from memdiff.core.store import BeliefStore
from memdiff.memory.application.policies import (
    BeliefRevisionPolicy,
    NaiveRagPolicy,
    RecencyRagPolicy,
    StructuredExtractPolicy,
)
from memdiff.memory.domain.belief import Belief, BeliefType, Provenance
from memdiff.memory.domain.revision import RevisionDecision, RevisionOp
from memdiff.memory.domain.trajectory import Trajectory
from memdiff.policies.belief_revision import BeliefRevisionPolicy as LegacyRevision
from memdiff.policies.belief_revision import RevisionDecision as LegacyDecision
from memdiff.policies.naive_rag import NaiveRagPolicy as LegacyNaive
from memdiff.policies.recency_rag import RecencyRagPolicy as LegacyRecency
from memdiff.policies.structured_extract import StructuredExtractPolicy as LegacyStructured
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


@pytest.fixture
def legacy_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = counter_ids("")
    monkeypatch.setattr(
        "memdiff.core.store.uuid.uuid4", lambda: types.SimpleNamespace(hex=gen())
    )


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


def _legacy_stack(name: str, store: BeliefStore) -> Any:
    embed = HashEmbedder().embed

    def extractor_fn(traj: dict[str, Any], step: int) -> list[LegacyBelief]:
        cands = _extract(Trajectory.model_validate(traj), step)
        return [LegacyBelief.model_validate(c.model_dump()) for c in cands]

    def reconciler_fn(cand: Any, neighbors: Any) -> LegacyDecision:
        op, target, merged, reason = _decide(cand, neighbors)
        return LegacyDecision(LegacyOp(op), target_belief_id=target, merged_claim=merged, reason=reason)

    return {
        "naive_rag": lambda: LegacyNaive(store=store, embed_fn=embed),
        "recency_rag": lambda: LegacyRecency(store=store, embed_fn=embed),
        "structured_extract": lambda: LegacyStructured(
            store=store, embed_fn=embed, extractor_fn=extractor_fn),
        "belief_revision": lambda: LegacyRevision(
            store=store, embed_fn=embed, extractor_fn=extractor_fn, reconciler_fn=reconciler_fn),
    }[name]()


@pytest.mark.parametrize(
    "name", ["naive_rag", "recency_rag", "structured_extract", "belief_revision"]
)
def test_policy_matches_legacy_byte_for_byte(name: str, legacy_ids: None) -> None:
    new, replay, _ = _new_stack(name)
    with BeliefStore(":memory:") as store:
        old = _legacy_stack(name, store)
        for traj in _TRAJECTORIES:
            old.insert(traj)
            new.insert(Trajectory.model_validate(traj))

        for q in _QUERIES:
            for top_k in (2, 6):
                got = [i.model_dump() for i in new.query(q, top_k=top_k)]
                assert got == old.query(q, top_k=top_k)
                assert got  # non-vacuous
        assert new.stats().model_dump() == old.stats().model_dump()

        if name in ("structured_extract", "belief_revision"):
            assert [b.model_dump() for b in replay.build_bundle(
                run_id="r", benchmark="b", policy_name=name, total_steps=7, stats=new.stats(),
            ).beliefs] == [b.model_dump() for b in store.list_beliefs()]
            bundle = replay.build_bundle(
                run_id="r", benchmark="b", policy_name=name, total_steps=7, stats=new.stats())
            strip = lambda es: [  # noqa: E731
                {k: v for k, v in e.model_dump().items() if k not in ("id", "created_at_utc")}
                for e in es
            ]
            assert strip(bundle.events) == strip(store.list_events())
            assert [e.model_dump() for e in bundle.edges] == [e.model_dump() for e in store.list_edges()]
            for step in range(0, 9):
                assert [b.model_dump() for b in replay.state_at_step(step)] == [
                    b.model_dump() for b in store.get_state_at_step(step)
                ]

    if name == "belief_revision":
        assert set(new.stats().revision_counts) == {
            "ADD", "SUPERSEDE", "CONSOLIDATE", "REFUTE", "IGNORE", "FORGET"
        }


def test_resume_from_sets_step_explicitly() -> None:
    policy, _, _ = _new_stack("belief_revision")
    policy.resume_from(10)
    policy.insert(Trajectory.model_validate(_TRAJECTORIES[3]))
    assert policy.step == 11 and policy.stats().total_ingested_sources == 11
    with pytest.raises(ValueError):
        policy.resume_from(-1)
