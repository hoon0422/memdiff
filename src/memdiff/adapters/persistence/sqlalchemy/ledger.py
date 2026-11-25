"""SQLAlchemy adapters for the memory/audit ports. One session (= one transaction) per UoW."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from memdiff.adapters.persistence.sqlalchemy.ledger_models import (
    BeliefEdgeRow,
    BeliefRow,
    ChunkRow,
    LedgerBase,
    TimelineEventRow,
)
from memdiff.adapters.vector.numpy_index import NumpyVectorIndex, encode_embedding
from memdiff.audit.domain.timeline import BeliefEdge, TimelineEvent
from memdiff.memory.domain.belief import Belief, BeliefStatus, BeliefType, Provenance
from memdiff.memory.domain.chunk import Chunk
from memdiff.memory.domain.revision import RevisionOp

def make_ledger_engine(db_path: str | Path = ":memory:") -> Engine:
    """Open (creating parent dirs and tables) the ledger DB; `:memory:` is one shared connection."""
    if str(db_path) == ":memory:":
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}")
    LedgerBase.metadata.create_all(engine)
    return engine


class SqlUnitOfWork:
    """`UnitOfWork` over one lazily-opened Session shared by every repository built on it."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._session: Session | None = None
        self._indexes: list[NumpyVectorIndex] = []

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = Session(self._engine, expire_on_commit=False)
        return self._session

    def watch(self, index: NumpyVectorIndex) -> None:
        """Index to drop whenever the transaction ends (rollback undoes cached writes)."""
        self._indexes.append(index)

    def commit(self) -> None:
        self.session.commit()
        self._invalidate()

    def rollback(self) -> None:
        self.session.rollback()
        self._invalidate()

    def close(self) -> None:
        """Roll back anything uncommitted and release the DB file (needed to reopen on --resume)."""
        if self._session is not None:
            self._session.close()
            self._session = None
        self._invalidate()
        self._engine.dispose()

    def _invalidate(self) -> None:
        for index in self._indexes:
            index.invalidate()


def make_belief_index(uow: SqlUnitOfWork) -> NumpyVectorIndex:
    """Index over active/all belief embeddings, dropped whenever `uow`'s transaction ends.

    Storage order (`seq`) reproduces the pre-refactor store's tie-breaking between equal scores.
    """

    def load(active_only: object) -> list[tuple[str, int, bytes | None]]:
        stmt = select(BeliefRow.id, BeliefRow.embedding).where(BeliefRow.embedding.is_not(None))
        if active_only:
            stmt = stmt.where(BeliefRow.status == BeliefStatus.ACTIVE.value)
        return [(i, 0, e) for i, e in uow.session.execute(stmt.order_by(BeliefRow.seq))]

    index = NumpyVectorIndex(load)
    uow.watch(index)
    return index


def make_chunk_index(uow: SqlUnitOfWork) -> NumpyVectorIndex:
    """Index over chunk embeddings (storage order, like `make_belief_index`)."""

    def load(_key: object) -> list[tuple[str, int, bytes | None]]:
        stmt = (
            select(ChunkRow.id, ChunkRow.step, ChunkRow.embedding)
            .where(ChunkRow.embedding.is_not(None))
            .order_by(ChunkRow.seq)
        )
        return [tuple(r) for r in uow.session.execute(stmt)]  # type: ignore[misc]

    index = NumpyVectorIndex(load)
    uow.watch(index)
    return index


def _belief(row: BeliefRow) -> Belief:
    return Belief(
        id=row.id,
        type=BeliefType(row.type),
        surface=row.surface,
        claim=row.claim,
        closed_set_items=row.closed_set_json,
        status=BeliefStatus(row.status),
        confidence=row.confidence,
        durability=row.durability,
        evidence_count=row.evidence_count,
        created_at_step=row.created_at_step,
        updated_at_step=row.updated_at_step,
        superseded_by=row.superseded_by,
        provenance=[Provenance.model_validate(p) for p in row.provenance_json],
        metadata=row.metadata_json,
    )


def _fill(row: BeliefRow, b: Belief) -> BeliefRow:
    row.type, row.surface, row.claim = b.type.value, b.surface, b.claim
    row.closed_set_json = list(b.closed_set_items)
    row.status, row.confidence, row.durability = b.status.value, b.confidence, b.durability
    row.evidence_count = b.evidence_count
    row.created_at_step, row.updated_at_step = b.created_at_step, b.updated_at_step
    row.superseded_by = b.superseded_by
    row.provenance_json = [p.model_dump() for p in b.provenance]
    row.metadata_json = dict(b.metadata)
    return row


class SqlBeliefRepository:
    def __init__(self, uow: SqlUnitOfWork, index: NumpyVectorIndex) -> None:
        self._uow = uow
        self._index = index

    def _row(self, belief_id: str) -> BeliefRow | None:
        return self._uow.session.scalar(select(BeliefRow).where(BeliefRow.id == belief_id))

    def add(self, belief: Belief, embedding: Sequence[float] | None) -> None:
        row = _fill(BeliefRow(id=belief.id), belief)
        row.embedding = encode_embedding(embedding)
        self._uow.session.add(row)
        self._index.invalidate()

    def update(self, belief: Belief, embedding: Sequence[float] | None = None) -> None:
        row = self._row(belief.id)
        assert row is not None, f"unknown belief {belief.id}"
        _fill(row, belief)
        if embedding is not None:
            row.embedding = encode_embedding(embedding)
        self._index.invalidate()

    def get(self, belief_id: str) -> Belief | None:
        row = self._row(belief_id)
        return _belief(row) if row else None

    def list_beliefs(
        self, *, status: BeliefStatus | None = None, surface: str | None = None
    ) -> list[Belief]:
        stmt = select(BeliefRow).order_by(BeliefRow.created_at_step, BeliefRow.id)
        if status is not None:
            stmt = stmt.where(BeliefRow.status == status.value)
        if surface is not None:
            stmt = stmt.where(BeliefRow.surface == surface)
        return [_belief(r) for r in self._uow.session.scalars(stmt)]

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        active_only: bool,
        min_score: float = -1.0,
    ) -> list[tuple[Belief, float]]:
        top = self._index.top(embedding, key=active_only, top_k=top_k, min_score=min_score)
        rows = self._uow.session.scalars(
            select(BeliefRow).where(BeliefRow.id.in_([i for i, _ in top]))
        )
        by_id = {r.id: _belief(r) for r in rows}
        return [(by_id[i], s) for i, s in top if i in by_id]


class SqlChunkRepository:
    def __init__(self, uow: SqlUnitOfWork, index: NumpyVectorIndex) -> None:
        self._uow = uow
        self._index = index

    def add(self, chunk: Chunk, embedding: Sequence[float] | None) -> None:
        self._uow.session.add(
            ChunkRow(
                id=chunk.id,
                source_id=chunk.source_id,
                step=chunk.step,
                state_index=chunk.state_index,
                text=chunk.text,
                metadata_json=dict(chunk.metadata),
                embedding=encode_embedding(embedding),
            )
        )
        self._index.invalidate()

    def search(
        self, embedding: Sequence[float], *, top_k: int, recency_decay: float
    ) -> list[tuple[Chunk, float]]:
        top = self._index.top(embedding, top_k=top_k, recency_decay=recency_decay)
        rows = self._uow.session.scalars(
            select(ChunkRow).where(ChunkRow.id.in_([i for i, _ in top]))
        )
        by_id = {
            r.id: Chunk(
                id=r.id,
                source_id=r.source_id,
                step=r.step,
                state_index=r.state_index,
                text=r.text,
                metadata=r.metadata_json,
            )
            for r in rows
        }
        return [(by_id[i], s) for i, s in top if i in by_id]

    def count(self) -> int:
        return self._uow.session.scalar(select(func.count()).select_from(ChunkRow)) or 0


class SqlAuditLog:
    """`AuditLogRepository`: events and edges join the same transaction as the belief writes."""

    def __init__(self, uow: SqlUnitOfWork) -> None:
        self._uow = uow

    def add_event(self, event: TimelineEvent) -> None:
        self._uow.session.add(
            TimelineEventRow(
                step=event.step,
                source_id=event.source_id,
                op=event.op.value,
                belief_id=event.belief_id,
                target_belief_id=event.target_belief_id,
                summary=event.summary,
                details_json=dict(event.details),
                created_at_utc=event.created_at_utc,
            )
        )

    def add_edge(self, edge: BeliefEdge) -> None:
        self._uow.session.add(BeliefEdgeRow(**edge.model_dump()))

    def list_events(self) -> list[TimelineEvent]:
        stmt = select(TimelineEventRow).order_by(TimelineEventRow.step, TimelineEventRow.id)
        return [
            TimelineEvent(
                id=r.id,
                step=r.step,
                source_id=r.source_id,
                op=RevisionOp(r.op),
                belief_id=r.belief_id,
                target_belief_id=r.target_belief_id,
                summary=r.summary,
                details=r.details_json,
                created_at_utc=r.created_at_utc,
            )
            for r in self._uow.session.scalars(stmt)
        ]

    def list_edges(self) -> list[BeliefEdge]:
        stmt = select(BeliefEdgeRow).order_by(BeliefEdgeRow.step, BeliefEdgeRow.id)
        return [
            BeliefEdge(
                source_id=r.source_id,
                target_id=r.target_id,
                relation=r.relation,  # type: ignore[arg-type]
                step=r.step,
                reason=r.reason,
            )
            for r in self._uow.session.scalars(stmt)
        ]

    def count_events_by_op(self) -> dict[str, int]:
        stmt = (
            select(TimelineEventRow.op, func.count())
            .group_by(TimelineEventRow.op)
            .order_by(TimelineEventRow.op)
        )
        return {op: n for op, n in self._uow.session.execute(stmt)}
