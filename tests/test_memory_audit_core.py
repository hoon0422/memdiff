"""test_store.py scenarios replayed through the memory/audit cores with in-memory fake ports."""

from __future__ import annotations

import pytest

from memdiff.audit.application.recorder import RevisionAuditRecorder
from memdiff.audit.application.replay import ReplayService
from memdiff.audit.domain.replay import ReplayBundle
from memdiff.audit.domain.time_travel import reconstruct_state_at_step
from memdiff.audit.domain.timeline import TimelineEvent
from memdiff.memory.application.policies import BeliefRevisionPolicy
from memdiff.memory.domain.belief import Belief, BeliefStatus, BeliefType, Provenance
from memdiff.memory.domain.revision import BeliefAdded, RevisionDecision, RevisionOp
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

TEMP = BeliefType.TEMPORARY


def _cand(btype: BeliefType, surface: str, claim: str, **kw: object) -> Belief:
    return Belief.new(id="cand", type=btype, surface=surface, claim=claim, step=0, source_id="x", **kw)  # type: ignore[arg-type]


class _Rig:
    def __init__(self, script: dict[str, list[Belief]], decide, *, extractor=None) -> None:  # type: ignore[no-untyped-def]
        self.beliefs, self.log, self.uow = InMemoryBeliefRepository(), InMemoryAuditLog(), FakeUnitOfWork()
        self.policy = BeliefRevisionPolicy(
            beliefs=self.beliefs,
            chunks=InMemoryChunkRepository(),
            embedder=HashEmbedder(),
            extractor=extractor or ScriptedExtractor(lambda t, step: script[t.id or ""]),
            reconciler=ScriptedReconciler(decide),
            events=RevisionAuditRecorder(self.log),
            uow=self.uow,
            new_id=counter_ids("b_"),
        )
        self.replay = ReplayService(beliefs=self.beliefs, log=self.log)

    def insert(self, tid: str) -> None:
        self.policy.insert(Trajectory(id=tid))


def _first_neighbor(op: RevisionOp, **kw: object):  # type: ignore[no-untyped-def]
    def decide(cand: Belief, neighbors: list[tuple[Belief, float]]) -> RevisionDecision:
        same = [b.id for b, _ in neighbors if b.surface == cand.surface]
        if not same:
            return RevisionDecision(op=RevisionOp.ADD)
        return RevisionDecision(op=op, target_belief_id=same[0], **kw)  # type: ignore[arg-type]

    return decide


def test_lifecycle_supersede_consolidate_forget_and_time_travel() -> None:
    script = {
        "s001": [
            _cand(BeliefType.DYNAMIC_STATE, "billing.region", "region is us-east-1"),
            _cand(TEMP, "staging.build", "build #123 queued", durability=0.4),
        ],
        "s002": [_cand(BeliefType.DYNAMIC_STATE, "billing.region", "region is eu-central-1")],
        "s003": [
            _cand(BeliefType.DYNAMIC_STATE, "billing.region", "eu confirmed", closed_set_items=["eu-central-1"])
        ],
        **{f"s00{i}": [] for i in range(4, 7)},
    }
    ops = {"s002": RevisionOp.SUPERSEDE, "s003": RevisionOp.CONSOLIDATE}
    current = {"tid": "s001"}

    def decide(cand, neighbors):  # type: ignore[no-untyped-def]
        op = ops.get(current["tid"], RevisionOp.ADD)
        return _first_neighbor(op, reason="Region migrated to EU")(cand, neighbors)

    rig = _Rig(script, decide)
    for tid in script:
        current["tid"] = tid
        rig.insert(tid)

    b1, b_tmp, b2 = "b_0000000001", "b_0000000002", "b_0000000003"
    live = {b.id: b for b in rig.beliefs.list_beliefs()}
    assert live[b2].evidence_count == 2 and len(live[b2].provenance) == 2
    assert live[b1].status == BeliefStatus.SUPERSEDED and live[b1].superseded_by == b2
    assert live[b_tmp].status == BeliefStatus.FORGOTTEN  # temporary, age 5 at step 6

    hits = rig.beliefs.search(HashEmbedder().embed(["anything"])[0], top_k=5, active_only=True)
    assert [b.id for b, _ in hits] == [b2]

    snap = lambda step: {b.id: b.status for b in rig.replay.state_at_step(step)}  # noqa: E731
    assert snap(1) == {b1: BeliefStatus.ACTIVE, b_tmp: BeliefStatus.ACTIVE}
    assert snap(2) == {b1: BeliefStatus.SUPERSEDED, b_tmp: BeliefStatus.ACTIVE, b2: BeliefStatus.ACTIVE}
    assert snap(5)[b_tmp] == BeliefStatus.ACTIVE
    assert snap(6)[b_tmp] == BeliefStatus.FORGOTTEN

    bundle = rig.replay.build_bundle(
        run_id="r", benchmark="drift_sim", policy_name="belief_revision", total_steps=6,
        stats=rig.policy.stats(),
    )
    assert isinstance(ReplayBundle.model_validate(bundle.model_dump()), ReplayBundle)
    assert [e.relation for e in bundle.edges] == ["SUPERSEDES"]
    assert (bundle.stats.active_beliefs, bundle.stats.superseded_beliefs,
            bundle.stats.forgotten_beliefs) == (1, 1, 1)


def test_low_durability_temporary_noise_is_ignored() -> None:
    rig = _Rig(
        {"t1": [_cand(TEMP, "ephemeral", "One-off debug ping", durability=0.1)]},
        lambda c, n: RevisionDecision(op=RevisionOp.ADD),
    )
    rig.insert("t1")
    stats = rig.policy.stats()
    assert stats.active_beliefs == 0 and stats.revision_counts == {"IGNORE": 1}


def test_event_column_rules_per_op() -> None:
    script = {
        "t1": [_cand(BeliefType.DYNAMIC_STATE, "a", "one"), _cand(BeliefType.GOTCHA, "g", "trap")],
        "t2": [_cand(BeliefType.DYNAMIC_STATE, "a", "two"), _cand(BeliefType.GOTCHA, "g", "gone")],
    }
    rig = _Rig(script, lambda c, n: RevisionDecision(op=RevisionOp.ADD))
    rig.insert("t1")
    rig.policy._reconciler = ScriptedReconciler(  # swap decision for step 2
        lambda c, n: _first_neighbor(
            RevisionOp.SUPERSEDE if c.surface == "a" else RevisionOp.REFUTE, reason="why"
        )(c, n)
    )
    rig.insert("t2")
    by_op = {}
    for e in rig.log.list_events():
        by_op.setdefault(e.op, []).append(e)

    assert [e.belief_id for e in by_op[RevisionOp.ADD]] == ["b_0000000001", "b_0000000002"]  # no ADD for superseder
    sup = by_op[RevisionOp.SUPERSEDE][0]
    assert (sup.belief_id, sup.target_belief_id) == ("b_0000000003", "b_0000000001")
    ref = by_op[RevisionOp.REFUTE][0]
    assert (ref.belief_id, ref.target_belief_id) == (None, "b_0000000002")
    assert {b.id: b.status for b in rig.beliefs.list_beliefs()}["b_0000000002"] == BeliefStatus.REFUTED


def test_failed_insert_rolls_back_unit_of_work() -> None:
    def boom(t: Trajectory, step: int) -> list[Belief]:
        raise RuntimeError("extractor down")

    rig = _Rig({}, lambda c, n: RevisionDecision(op=RevisionOp.ADD), extractor=ScriptedExtractor(boom))
    with pytest.raises(RuntimeError):
        rig.insert("t1")
    assert (rig.uow.commits, rig.uow.rollbacks) == (0, 1)


def test_time_travel_pure_function_ignores_later_events_and_uncreated_beliefs() -> None:
    mk = lambda i, step: Belief.new(  # noqa: E731
        id=i, type=BeliefType.WORKFLOW, surface="s", claim=i, step=step, source_id="x",
        provenance=[Provenance(source_id="x")],
    )
    beliefs = [mk("a", 1), mk("b", 3)]
    ev = lambda step, op, **kw: TimelineEvent(step=step, source_id="x", op=op, summary="", **kw)  # noqa: E731
    events = [
        ev(2, RevisionOp.REFUTE, target_belief_id="a"),
        ev(3, RevisionOp.SUPERSEDE, belief_id="b", target_belief_id="a"),
    ]
    assert reconstruct_state_at_step(beliefs, events, 0) == []
    s2 = reconstruct_state_at_step(beliefs, events, 2)
    assert [(b.id, b.status) for b in s2] == [("a", BeliefStatus.REFUTED)]
    s3 = {b.id: b for b in reconstruct_state_at_step(beliefs, events, 3)}
    assert (s3["a"].status, s3["a"].superseded_by, s3["b"].status) == (
        BeliefStatus.SUPERSEDED, "b", BeliefStatus.ACTIVE,
    )


def test_recorder_rejects_unknown_event_type() -> None:
    from memdiff.memory.domain.revision import RevisionEvent

    with pytest.raises(TypeError):
        RevisionAuditRecorder(InMemoryAuditLog()).emit(RevisionEvent(step=1, source_id="x"))


def test_recorder_emits_added_summary() -> None:
    log = InMemoryAuditLog()
    RevisionAuditRecorder(log).emit(
        BeliefAdded(step=1, source_id="s", belief_id="b", belief_type=BeliefType.GOTCHA,
                    surface="ui", claim="c" * 150)
    )
    (e,) = log.list_events()
    assert e.summary == f"Added [gotcha] on ui: {'c' * 100}" and e.id == 1


def test_time_travel_restores_claim_and_provenance_across_consolidates() -> None:
    script = {
        "s001": [_cand(BeliefType.GOTCHA, "ui", "use dropdown", closed_set_items=["a"])],
        "s002": [_cand(BeliefType.GOTCHA, "ui", "x", closed_set_items=["b"],
                       provenance=[Provenance(source_id="s002", outcome="success")])],
        "s003": [_cand(BeliefType.GOTCHA, "ui", "y", provenance=[Provenance(source_id="s003")])],
    }
    merged = {"s002": "use funnel icon", "s003": "use funnel icon (confirmed)"}
    current = {"tid": "s001"}

    def decide(cand, neighbors):  # type: ignore[no-untyped-def]
        if current["tid"] == "s001" or not neighbors:
            return RevisionDecision(op=RevisionOp.ADD)
        return RevisionDecision(
            op=RevisionOp.CONSOLIDATE, target_belief_id=neighbors[0][0].id, merged_claim=merged[current["tid"]]
        )

    rig = _Rig(script, decide)
    for tid in script:
        current["tid"] = tid
        rig.insert(tid)

    def at(step: int) -> Belief:
        (b,) = rig.replay.state_at_step(step)
        return b

    sources = lambda b: [p.source_id for p in b.provenance]  # noqa: E731
    assert (at(1).claim, sources(at(1)), at(1).closed_set_items) == ("use dropdown", ["x"], ["a"])
    assert (at(1).evidence_count, at(1).confidence) == (1, 0.8)
    assert (at(2).claim, sources(at(2)), at(2).closed_set_items) == ("use funnel icon", ["x", "s002"], ["a", "b"])
    assert (at(2).evidence_count, at(2).confidence) == (2, pytest.approx(0.85))
    assert (at(3).claim, sources(at(3))) == ("use funnel icon (confirmed)", ["x", "s002", "s003"])
    assert at(3) == rig.beliefs.list_beliefs()[0]  # latest step matches the live row
    assert len(rig.beliefs.list_beliefs()) == 1  # ledger rows are never deleted
