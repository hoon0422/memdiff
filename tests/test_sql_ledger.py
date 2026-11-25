"""SQLAlchemy ledger + numpy index: search ranking, transactions, schema and lifecycle."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import inspect, text

from memdiff.adapters.persistence.sqlalchemy.ledger import (
    SqlAuditLog,
    SqlBeliefRepository,
    SqlChunkRepository,
    SqlUnitOfWork,
    make_ledger_engine,
)
from memdiff.audit.application.recorder import RevisionAuditRecorder
from memdiff.audit.application.replay import ReplayService
from memdiff.audit.domain.timeline import BeliefEdge, TimelineEvent
from memdiff.memory.application.policies import BeliefRevisionPolicy
from memdiff.memory.domain.belief import Belief, BeliefStatus, BeliefType, Provenance
from memdiff.memory.domain.chunk import Chunk
from memdiff.memory.domain.revision import RevisionDecision, RevisionOp
from memdiff.memory.domain.trajectory import Trajectory
from memory_fakes import (
    HashEmbedder,
    ScriptedExtractor,
    ScriptedReconciler,
    counter_ids,
)


class Ledger:
    """The adapters sharing one engine/UoW, as the container will wire them."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.engine = make_ledger_engine(db_path)
        self.uow = SqlUnitOfWork(self.engine)
        self.beliefs = SqlBeliefRepository(self.uow)
        self.chunks = SqlChunkRepository(self.uow)
        self.log = SqlAuditLog(self.uow)

    def close(self) -> None:
        self.uow.close()


@pytest.fixture
def ledger():  # type: ignore[no-untyped-def]
    led = Ledger()
    yield led
    led.close()


def _belief(i: str, step: int = 1, **kw: object) -> Belief:
    return Belief.new(
        id=i, type=BeliefType.DYNAMIC_STATE, surface="s", claim=f"claim {i}", step=step,
        source_id="src", **kw,  # type: ignore[arg-type]
    )


def _unit(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    return arr / np.linalg.norm(arr)


# ── search ranking (expected values recorded from the pre-refactor store) ─────────────


def _vectors() -> list[list[float]]:
    rng = np.random.default_rng(7)
    vecs = [rng.standard_normal(3).tolist() for _ in range(6)]
    vecs.append(vecs[0])  # exact tie with the first vector
    vecs += [rng.standard_normal(4).tolist() for _ in range(2)]  # dim mismatch
    return vecs


QUERIES = [[1.0, 0.2, -0.3], [0.1, 0.9, 0.4, 0.0], [0.0, 0.0, 0.0], []]

BELIEF_EXPECTED: dict[tuple[bool, float, int], list[tuple[str, float]]] = {
    (True, -1.0, 0): [('b5', 0.335423),
                   ('b6', 0.332287),
                   ('b0', 0.332287),
                   ('b2', 0.313253),
                   ('b3', -0.682826)],
 (True, -1.0, 1): [('b8', 0.137934), ('b7', -0.713863)],
 (True, -1.0, 2): [('b6', 0.0), ('b5', 0.0), ('b3', 0.0), ('b2', 0.0), ('b0', 0.0)],
 (True, -1.0, 3): [],
 (True, 0.0, 0): [('b5', 0.335423), ('b6', 0.332287), ('b0', 0.332287), ('b2', 0.313253)],
 (True, 0.0, 1): [('b8', 0.137934)],
 (True, 0.0, 2): [('b6', 0.0), ('b5', 0.0), ('b3', 0.0), ('b2', 0.0), ('b0', 0.0)],
 (True, 0.0, 3): [],
 (True, 0.5, 0): [],
 (True, 0.5, 1): [],
 (True, 0.5, 2): [],
 (True, 0.5, 3): [],
 (False, -1.0, 0): [('b5', 0.335423),
                    ('b6', 0.332287),
                    ('b0', 0.332287),
                    ('b2', 0.313253),
                    ('b4', -0.072199)],
 (False, -1.0, 1): [('b8', 0.137934), ('b7', -0.713863)],
 (False, -1.0, 2): [('b6', 0.0), ('b5', 0.0), ('b4', 0.0), ('b3', 0.0), ('b2', 0.0)],
 (False, -1.0, 3): [],
 (False, 0.0, 0): [('b5', 0.335423), ('b6', 0.332287), ('b0', 0.332287), ('b2', 0.313253)],
 (False, 0.0, 1): [('b8', 0.137934)],
 (False, 0.0, 2): [('b6', 0.0), ('b5', 0.0), ('b4', 0.0), ('b3', 0.0), ('b2', 0.0)],
 (False, 0.0, 3): [],
 (False, 0.5, 0): [],
 (False, 0.5, 1): [],
 (False, 0.5, 2): [],
 (False, 0.5, 3): []}
CHUNK_EXPECTED: dict[tuple[float, int], list[tuple[str, float]]] = {(0.0, 0): [('c5', 0.335423), ('c6', 0.332287), ('c0', 0.332287), ('c2', 0.313253)],
 (0.0, 1): [('c8', 0.137934), ('c7', -0.713863)],
 (0.3, 0): [('c6', 0.332287), ('c5', 0.248488), ('c2', 0.171917), ('c0', 0.135098)],
 (0.3, 1): [('c8', 0.137934), ('c7', -0.528843)]}


def _fill_beliefs(ledger: Ledger) -> None:
    for n, vec in enumerate(_vectors()):
        ledger.beliefs.add(_belief(f"b{n}"), vec)
    for bid in ("b1", "b4"):
        old = ledger.beliefs.get(bid)
        assert old is not None
        ledger.beliefs.update(old.mark_refuted(step=2))
    ledger.uow.commit()


@pytest.mark.parametrize("active_only", [True, False])
@pytest.mark.parametrize("min_score", [-1.0, 0.0, 0.5])
@pytest.mark.parametrize("query_index", range(len(QUERIES)))
def test_belief_search_ranking(ledger, active_only, min_score, query_index) -> None:  # type: ignore[no-untyped-def]
    _fill_beliefs(ledger)
    got = ledger.beliefs.search(
        QUERIES[query_index], top_k=5, active_only=active_only, min_score=min_score
    )
    assert [(b.id, pytest.approx(s, abs=1e-6)) for b, s in got] == BELIEF_EXPECTED[
        (active_only, min_score, query_index)
    ]
    assert [b for b, _ in got] == [Belief.model_validate(b.model_dump()) for b, _ in got]


def test_belief_search_guards(ledger) -> None:  # type: ignore[no-untyped-def]
    ledger.beliefs.add(_belief("a"), [1.0, 0.0])
    assert ledger.beliefs.search([1.0, 0.0], top_k=0, active_only=True) == []
    assert ledger.beliefs.search([1.0, 0.0, 0.0], top_k=3, active_only=True) == []


def test_belief_search_sees_writes_without_commit_and_after_rollback(ledger) -> None:  # type: ignore[no-untyped-def]
    ledger.beliefs.add(_belief("a"), [1.0, 0.0])
    assert [b.id for b, _ in ledger.beliefs.search([1.0, 0.0], top_k=5, active_only=True)] == ["a"]
    ledger.uow.commit()
    ledger.beliefs.add(_belief("b"), [0.9, 0.1])  # cache was warm: must be invalidated
    ids = [b.id for b, _ in ledger.beliefs.search([1.0, 0.0], top_k=5, active_only=True)]
    assert ids == ["a", "b"]
    ledger.uow.rollback()
    ids = [b.id for b, _ in ledger.beliefs.search([1.0, 0.0], top_k=5, active_only=True)]
    assert ids == ["a"]


def test_update_keeps_embedding_unless_given(ledger) -> None:  # type: ignore[no-untyped-def]
    ledger.beliefs.add(_belief("a"), [1.0, 0.0])
    a = ledger.beliefs.get("a")
    assert a is not None
    ledger.beliefs.update(a.model_copy(update={"claim": "new", "confidence": 0.95}))
    (hit,) = ledger.beliefs.search([1.0, 0.0], top_k=1, active_only=True)
    assert (hit[0].claim, hit[0].confidence, round(hit[1], 5)) == ("new", 0.95, 1.0)
    ledger.beliefs.update(hit[0], [0.0, 1.0])
    assert ledger.beliefs.search([1.0, 0.0], top_k=1, active_only=True)[0][1] == pytest.approx(0)


def test_belief_json_columns_round_trip(ledger) -> None:  # type: ignore[no-untyped-def]
    b = _belief(
        "a", closed_set_items=["café", "b"], metadata={"k": [1, {"x": "é"}]},
        provenance=[Provenance(source_id="s", step_index=3, outcome="failure", excerpt="é")],
    )
    ledger.beliefs.add(b, None)
    assert ledger.beliefs.get("a") == b
    assert ledger.beliefs.get("missing") is None


def test_list_beliefs_filters_and_orders(ledger) -> None:  # type: ignore[no-untyped-def]
    for i, step in (("b", 2), ("a", 2), ("c", 1)):
        ledger.beliefs.add(_belief(i, step), None)
    ledger.beliefs.update(_belief("a", 2).mark_forgotten(step=3))
    assert [b.id for b in ledger.beliefs.list_beliefs()] == ["c", "a", "b"]
    assert [b.id for b in ledger.beliefs.list_beliefs(status=BeliefStatus.ACTIVE)] == ["c", "b"]
    assert [b.id for b in ledger.beliefs.list_beliefs(surface="nope")] == []


# ── chunk search ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("decay", [0.0, 0.3])
@pytest.mark.parametrize("query_index", [0, 1])
def test_chunk_search_ranking(ledger, decay, query_index) -> None:  # type: ignore[no-untyped-def]
    for n, vec in enumerate(_vectors()):
        meta = {"n": n, "t": "é"}
        ledger.chunks.add(
            Chunk(id=f"c{n}", source_id="s", step=n // 2, state_index=n, text=f"t{n}", metadata=meta),
            vec,
        )
    ledger.chunks.add(Chunk(id="noemb", source_id="s", step=9, text="x"), None)
    ledger.uow.commit()
    got = ledger.chunks.search(QUERIES[query_index], top_k=4, recency_decay=decay)
    assert ledger.chunks.count() == 10
    assert [(c.id, pytest.approx(s, abs=1e-6)) for c, s in got] == CHUNK_EXPECTED[(decay, query_index)]
    assert got[0][0].metadata["t"] == "é"


def test_chunk_search_sees_new_chunk(ledger) -> None:  # type: ignore[no-untyped-def]
    ledger.chunks.add(Chunk(id="a", source_id="s", step=1, text="a"), [1.0, 0.0])
    assert [c.id for c, _ in ledger.chunks.search([1.0, 0.0], top_k=5, recency_decay=0.0)] == ["a"]
    ledger.chunks.add(Chunk(id="b", source_id="s", step=2, text="b"), [0.9, 0.1])
    assert [c.id for c, _ in ledger.chunks.search([1.0, 0.0], top_k=5, recency_decay=0.0)] == [
        "a", "b",
    ]


# ── audit log ─────────────────────────────────────────────────────────────────────────────


def test_audit_log_orders_events_and_counts_ops_sorted(ledger) -> None:  # type: ignore[no-untyped-def]
    ev = lambda step, op: TimelineEvent(step=step, source_id="s", op=op, summary=op.value)  # noqa: E731
    for step, op in ((2, RevisionOp.SUPERSEDE), (1, RevisionOp.ADD), (2, RevisionOp.ADD)):
        ledger.log.add_event(ev(step, op))
    ledger.log.add_edge(BeliefEdge(source_id="n", target_id="o", relation="SUPERSEDES", step=2))
    ledger.log.add_edge(BeliefEdge(source_id="m", target_id="p", relation="REFUTES", step=1))
    assert [(e.id, e.step, e.op) for e in ledger.log.list_events()] == [
        (2, 1, RevisionOp.ADD), (1, 2, RevisionOp.SUPERSEDE), (3, 2, RevisionOp.ADD),
    ]
    assert [e.source_id for e in ledger.log.list_edges()] == ["m", "n"]
    counts = ledger.log.count_events_by_op()
    assert list(counts.items()) == [("ADD", 2), ("SUPERSEDE", 1)]


# ── unit of work ──────────────────────────────────────────────────────────────────────────


def test_failure_after_belief_write_rolls_back_belief_edge_and_event(ledger) -> None:  # type: ignore[no-untyped-def]
    def decide(cand: Belief, neighbors: list[tuple[Belief, float]]) -> RevisionDecision:
        if cand.claim == "second":
            raise RuntimeError("reconciler down")  # after the first candidate's belief + event
        return RevisionDecision(op=RevisionOp.ADD)

    policy = BeliefRevisionPolicy(
        beliefs=ledger.beliefs, chunks=ledger.chunks, embedder=HashEmbedder(),
        extractor=ScriptedExtractor(
            lambda t, step: [_cand(BeliefType.GOTCHA, "a", "first"),
                             _cand(BeliefType.GOTCHA, "b", "second")]
        ),
        reconciler=ScriptedReconciler(decide), events=RevisionAuditRecorder(ledger.log),
        uow=ledger.uow, new_id=counter_ids("b_"),
    )
    ledger.log.add_edge(BeliefEdge(source_id="x", target_id="y", relation="REFUTES", step=0))
    ledger.uow.rollback()
    with pytest.raises(RuntimeError):
        policy.insert(Trajectory(id="t1"))
    assert ledger.beliefs.list_beliefs() == []
    assert ledger.log.list_events() == [] and ledger.log.list_edges() == []
    assert ledger.beliefs.search([1.0] * 8, top_k=5, active_only=False) == []  # index invalidated
    ledger.chunks.add(Chunk(id="c", source_id="s", step=1, text="t"), [1.0, 0.0])
    ledger.uow.rollback()
    assert ledger.chunks.count() == 0


def test_commit_persists_and_file_db_reopens(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "nested" / "ledger.db")
    led = Ledger(path)
    led.beliefs.add(_belief("a"), [1.0, 0.0])
    led.log.add_event(TimelineEvent(step=1, source_id="s", op=RevisionOp.IGNORE, summary="x"))
    led.uow.commit()
    led.close()
    again = Ledger(path)
    try:
        assert [b.id for b in again.beliefs.list_beliefs()] == ["a"]
        assert [(e.id, e.summary) for e in again.log.list_events()] == [(1, "x")]
        assert again.beliefs.search([1.0, 0.0], top_k=1, active_only=True)[0][1] == pytest.approx(1)
    finally:
        again.close()


def test_schema_table_and_column_names(ledger) -> None:  # type: ignore[no-untyped-def]
    insp = inspect(ledger.engine)
    cols = {t: [c["name"] for c in insp.get_columns(t)] for t in insp.get_table_names()}
    assert cols["timeline_events"] == [
        "id", "step", "source_id", "op", "belief_id", "target_belief_id", "summary",
        "details_json", "created_at_utc",
    ]
    assert cols["belief_edges"] == ["id", "source_id", "target_id", "relation", "step", "reason"]
    assert cols["chunks"] == [
        "id", "source_id", "step", "state_index", "text", "metadata_json", "embedding",
    ]
    assert "closed_set_json" in cols["beliefs"] and "provenance_json" in cols["beliefs"]
    with ledger.engine.connect() as conn:  # the hint printed by scripts/demo_memory.py
        conn.execute(text("SELECT step, op, summary FROM timeline_events"))


# ── lifecycle through the real policy on the SQL adapters ────────────────────────────────


def _cand(btype: BeliefType, surface: str, claim: str, **kw: object) -> Belief:
    return Belief.new(id="cand", type=btype, surface=surface, claim=claim, step=0, source_id="x", **kw)  # type: ignore[arg-type]


def test_lifecycle_supersede_consolidate_forget_and_time_travel_on_sql(ledger) -> None:  # type: ignore[no-untyped-def]
    script = {
        "s001": [
            _cand(BeliefType.DYNAMIC_STATE, "billing.region", "region is us-east-1"),
            _cand(BeliefType.TEMPORARY, "staging.build", "build #123 queued", durability=0.4),
        ],
        "s002": [_cand(BeliefType.DYNAMIC_STATE, "billing.region", "region is eu-central-1")],
        "s003": [_cand(BeliefType.DYNAMIC_STATE, "billing.region", "eu confirmed",
                       closed_set_items=["eu-central-1"])],
        **{f"s00{i}": [] for i in range(4, 7)},
    }
    ops = {"s002": RevisionOp.SUPERSEDE, "s003": RevisionOp.CONSOLIDATE}
    current = {"tid": "s001"}

    def decide(cand: Belief, neighbors: list[tuple[Belief, float]]) -> RevisionDecision:
        same = [b.id for b, _ in neighbors if b.surface == cand.surface]
        op = ops.get(current["tid"], RevisionOp.ADD)
        if not same or op == RevisionOp.ADD:
            return RevisionDecision(op=RevisionOp.ADD)
        return RevisionDecision(op=op, target_belief_id=same[0], reason="Region migrated to EU")

    policy = BeliefRevisionPolicy(
        beliefs=ledger.beliefs, chunks=ledger.chunks, embedder=HashEmbedder(),
        extractor=ScriptedExtractor(lambda t, step: script[t.id or ""]),
        reconciler=ScriptedReconciler(decide), events=RevisionAuditRecorder(ledger.log),
        uow=ledger.uow, new_id=counter_ids("b_"),
    )
    for tid in script:
        current["tid"] = tid
        policy.insert(Trajectory(id=tid))

    b1, b_tmp, b2 = "b_0000000001", "b_0000000002", "b_0000000003"
    live = {b.id: b for b in ledger.beliefs.list_beliefs()}
    assert live[b2].evidence_count == 2 and len(live[b2].provenance) == 2
    assert live[b2].closed_set_items == ["eu-central-1"]
    assert live[b1].status == BeliefStatus.SUPERSEDED and live[b1].superseded_by == b2
    assert live[b_tmp].status == BeliefStatus.FORGOTTEN

    replay = ReplayService(beliefs=ledger.beliefs, log=ledger.log)
    snap = lambda step: {b.id: b.status for b in replay.state_at_step(step)}  # noqa: E731
    assert snap(1) == {b1: BeliefStatus.ACTIVE, b_tmp: BeliefStatus.ACTIVE}
    assert snap(2) == {b1: BeliefStatus.SUPERSEDED, b_tmp: BeliefStatus.ACTIVE, b2: BeliefStatus.ACTIVE}
    assert snap(6)[b_tmp] == BeliefStatus.FORGOTTEN
    stats = policy.stats()
    assert (stats.active_beliefs, stats.superseded_beliefs, stats.forgotten_beliefs) == (1, 1, 1)
    assert stats.revision_counts == {"ADD": 2, "CONSOLIDATE": 1, "FORGET": 1, "SUPERSEDE": 1}
    assert [e.relation for e in ledger.log.list_edges()] == ["SUPERSEDES"]
