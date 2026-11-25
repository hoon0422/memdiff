"""ORM rows of the belief ledger. Table and column names are user-visible (sqlite3 CLI, docs)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Float, Index, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class LedgerBase(DeclarativeBase):
    """Separate metadata from the LLM cache DB (`CacheBase`)."""


class BeliefRow(LedgerBase):
    __tablename__ = "beliefs"
    __table_args__ = (
        Index("idx_beliefs_status", "status"),
        Index("idx_beliefs_surface", "surface"),
        Index("idx_beliefs_type", "type"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[str] = mapped_column(String, unique=True)
    type: Mapped[str] = mapped_column(String)
    surface: Mapped[str] = mapped_column(String)
    claim: Mapped[str] = mapped_column(Text)
    closed_set_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, default="active")
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    durability: Mapped[float] = mapped_column(Float, default=0.8)
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at_step: Mapped[int] = mapped_column(Integer)
    updated_at_step: Mapped[int] = mapped_column(Integer)
    superseded_by: Mapped[str | None] = mapped_column(String, nullable=True)
    provenance_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # insertion sequence: breaks search-score ties in storage order
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)


class BeliefEdgeRow(LedgerBase):
    __tablename__ = "belief_edges"
    __table_args__ = (
        Index("idx_edges_source", "source_id"),
        Index("idx_edges_target", "target_id"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String)
    target_id: Mapped[str] = mapped_column(String)
    relation: Mapped[str] = mapped_column(String)
    step: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(Text, default="")


class TimelineEventRow(LedgerBase):
    __tablename__ = "timeline_events"
    __table_args__ = (Index("idx_timeline_step", "step"), {"sqlite_autoincrement": True})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step: Mapped[int] = mapped_column(Integer)
    source_id: Mapped[str] = mapped_column(String)
    op: Mapped[str] = mapped_column(String)
    belief_id: Mapped[str | None] = mapped_column(String, nullable=True)
    target_belief_id: Mapped[str | None] = mapped_column(String, nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at_utc: Mapped[str] = mapped_column(String)


class ChunkRow(LedgerBase):
    __tablename__ = "chunks"
    __table_args__ = (Index("idx_chunks_source", "source_id"), {"sqlite_autoincrement": True})

    id: Mapped[str] = mapped_column(String, unique=True)
    source_id: Mapped[str] = mapped_column(String)
    step: Mapped[int] = mapped_column(Integer)
    state_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
