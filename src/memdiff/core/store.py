from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    Belief,
    BeliefEdge,
    BeliefStatus,
    BeliefType,
    MemoryStats,
    Provenance,
    RevisionOp,
    TimelineEvent,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    surface TEXT NOT NULL,
    claim TEXT NOT NULL,
    closed_set_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 0.8,
    durability REAL NOT NULL DEFAULT 0.8,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at_step INTEGER NOT NULL,
    updated_at_step INTEGER NOT NULL,
    superseded_by TEXT,
    provenance_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status);
CREATE INDEX IF NOT EXISTS idx_beliefs_surface ON beliefs(surface);
CREATE INDEX IF NOT EXISTS idx_beliefs_type ON beliefs(type);

CREATE TABLE IF NOT EXISTS belief_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    step INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON belief_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON belief_edges(target_id);

CREATE TABLE IF NOT EXISTS timeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    op TEXT NOT NULL,
    belief_id TEXT,
    target_belief_id TEXT,
    summary TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_timeline_step ON timeline_events(step);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    state_index INTEGER,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
"""


def _vec_to_bytes(vec: list[float] | np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


def _bytes_to_vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class BeliefStore:
    """SQLite-backed versioned Belief Ledger with vector retrieval and time-travel queries."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> BeliefStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Belief lifecycle operations
    # ------------------------------------------------------------------

    def add_belief(
        self,
        *,
        belief_type: BeliefType,
        surface: str,
        claim: str,
        step: int,
        source_id: str,
        provenance: list[Provenance] | None = None,
        closed_set_items: list[str] | None = None,
        confidence: float = 0.8,
        durability: float = 0.8,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | np.ndarray | None = None,
        belief_id: str | None = None,
        log_event: bool = True,
    ) -> Belief:
        bid = belief_id or f"b_{uuid.uuid4().hex[:10]}"
        prov_list = provenance or [Provenance(source_id=source_id)]
        belief = Belief(
            id=bid,
            type=belief_type,
            surface=surface,
            claim=claim,
            closed_set_items=list(closed_set_items or []),
            status=BeliefStatus.ACTIVE,
            confidence=confidence,
            durability=durability,
            evidence_count=1,
            created_at_step=step,
            updated_at_step=step,
            superseded_by=None,
            provenance=prov_list,
            metadata=dict(metadata or {}),
        )
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, type, surface, claim, closed_set_json, status,
                confidence, durability, evidence_count,
                created_at_step, updated_at_step, superseded_by,
                provenance_json, metadata_json, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                belief.id,
                belief.type.value,
                belief.surface,
                belief.claim,
                json.dumps(belief.closed_set_items),
                belief.status.value,
                belief.confidence,
                belief.durability,
                belief.evidence_count,
                belief.created_at_step,
                belief.updated_at_step,
                belief.superseded_by,
                json.dumps([p.model_dump() for p in belief.provenance]),
                json.dumps(belief.metadata),
                _vec_to_bytes(embedding),
            ),
        )
        if log_event:
            self._log_event(
                step=step,
                source_id=source_id,
                op=RevisionOp.ADD,
                belief_id=belief.id,
                summary=f"Added [{belief.type.value}] on {belief.surface}: {belief.claim[:100]}",
                details={"surface": belief.surface, "type": belief.type.value},
            )
        self._conn.commit()
        return belief

    def supersede_belief(
        self,
        *,
        old_belief_id: str,
        new_claim: str,
        step: int,
        source_id: str,
        reason: str = "",
        belief_type: BeliefType | None = None,
        surface: str | None = None,
        provenance: list[Provenance] | None = None,
        closed_set_items: list[str] | None = None,
        confidence: float = 0.9,
        durability: float | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | np.ndarray | None = None,
    ) -> Belief:
        old = self.get_belief(old_belief_id)
        if old is None:
            raise KeyError(f"Unknown belief id: {old_belief_id}")

        new_belief = self.add_belief(
            belief_type=belief_type or old.type,
            surface=surface or old.surface,
            claim=new_claim,
            step=step,
            source_id=source_id,
            provenance=provenance or [Provenance(source_id=source_id)],
            closed_set_items=closed_set_items if closed_set_items is not None else old.closed_set_items,
            confidence=confidence,
            durability=durability if durability is not None else old.durability,
            metadata=metadata or {},
            embedding=embedding,
            log_event=False,
        )
        self._conn.execute(
            """
            UPDATE beliefs
            SET status = ?, superseded_by = ?, updated_at_step = ?
            WHERE id = ?
            """,
            (BeliefStatus.SUPERSEDED.value, new_belief.id, step, old.id),
        )
        self._conn.execute(
            """
            INSERT INTO belief_edges (source_id, target_id, relation, step, reason)
            VALUES (?, ?, 'SUPERSEDES', ?, ?)
            """,
            (new_belief.id, old.id, step, reason),
        )
        self._log_event(
            step=step,
            source_id=source_id,
            op=RevisionOp.SUPERSEDE,
            belief_id=new_belief.id,
            target_belief_id=old.id,
            summary=f"Superseded {old.id} -> {new_belief.id}: {reason or new_claim[:80]}",
            details={"old_claim": old.claim, "new_claim": new_claim, "reason": reason},
        )
        self._conn.commit()
        return new_belief

    def consolidate_belief(
        self,
        *,
        belief_id: str,
        step: int,
        source_id: str,
        new_provenance: Provenance,
        merged_claim: str | None = None,
        closed_set_items: list[str] | None = None,
        confidence_boost: float = 0.05,
        embedding: list[float] | np.ndarray | None = None,
    ) -> Belief:
        existing = self.get_belief(belief_id)
        if existing is None:
            raise KeyError(f"Unknown belief id: {belief_id}")

        updated_prov = [*existing.provenance, new_provenance]
        updated_claim = merged_claim or existing.claim
        updated_closed = (
            sorted(set(existing.closed_set_items) | set(closed_set_items))
            if closed_set_items
            else existing.closed_set_items
        )
        updated_conf = min(1.0, existing.confidence + confidence_boost)
        updated_count = existing.evidence_count + 1

        if embedding is not None:
            self._conn.execute(
                """
                UPDATE beliefs
                SET claim = ?, closed_set_json = ?, confidence = ?,
                    evidence_count = ?, updated_at_step = ?,
                    provenance_json = ?, embedding = ?
                WHERE id = ?
                """,
                (
                    updated_claim,
                    json.dumps(updated_closed),
                    updated_conf,
                    updated_count,
                    step,
                    json.dumps([p.model_dump() for p in updated_prov]),
                    _vec_to_bytes(embedding),
                    existing.id,
                ),
            )
        else:
            self._conn.execute(
                """
                UPDATE beliefs
                SET claim = ?, closed_set_json = ?, confidence = ?,
                    evidence_count = ?, updated_at_step = ?,
                    provenance_json = ?
                WHERE id = ?
                """,
                (
                    updated_claim,
                    json.dumps(updated_closed),
                    updated_conf,
                    updated_count,
                    step,
                    json.dumps([p.model_dump() for p in updated_prov]),
                    existing.id,
                ),
            )
        self._log_event(
            step=step,
            source_id=source_id,
            op=RevisionOp.CONSOLIDATE,
            belief_id=existing.id,
            summary=f"Consolidated evidence (#{updated_count}) into {existing.id} on {existing.surface}",
            details={"evidence_count": updated_count},
        )
        self._conn.commit()
        refreshed = self.get_belief(existing.id)
        assert refreshed is not None
        return refreshed

    def refute_belief(
        self,
        *,
        belief_id: str,
        step: int,
        source_id: str,
        reason: str,
    ) -> Belief:
        existing = self.get_belief(belief_id)
        if existing is None:
            raise KeyError(f"Unknown belief id: {belief_id}")

        self._conn.execute(
            """
            UPDATE beliefs
            SET status = ?, updated_at_step = ?
            WHERE id = ?
            """,
            (BeliefStatus.REFUTED.value, step, existing.id),
        )
        self._log_event(
            step=step,
            source_id=source_id,
            op=RevisionOp.REFUTE,
            target_belief_id=existing.id,
            summary=f"Refuted {existing.id}: {reason[:100]}",
            details={"reason": reason},
        )
        self._conn.commit()
        refreshed = self.get_belief(existing.id)
        assert refreshed is not None
        return refreshed

    def forget_belief(
        self,
        *,
        belief_id: str,
        step: int,
        source_id: str,
        reason: str = "low_durability_eviction",
    ) -> Belief:
        existing = self.get_belief(belief_id)
        if existing is None:
            raise KeyError(f"Unknown belief id: {belief_id}")

        self._conn.execute(
            """
            UPDATE beliefs
            SET status = ?, updated_at_step = ?
            WHERE id = ?
            """,
            (BeliefStatus.FORGOTTEN.value, step, existing.id),
        )
        self._log_event(
            step=step,
            source_id=source_id,
            op=RevisionOp.FORGET,
            target_belief_id=existing.id,
            summary=f"Forgot {existing.id} ({reason})",
            details={"reason": reason},
        )
        self._conn.commit()
        refreshed = self.get_belief(existing.id)
        assert refreshed is not None
        return refreshed

    def log_ignore(self, *, step: int, source_id: str, summary: str) -> None:
        self._log_event(
            step=step,
            source_id=source_id,
            op=RevisionOp.IGNORE,
            summary=summary,
        )
        self._conn.commit()

    def _log_event(
        self,
        *,
        step: int,
        source_id: str,
        op: RevisionOp,
        summary: str,
        belief_id: str | None = None,
        target_belief_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = TimelineEvent(
            step=step,
            source_id=source_id,
            op=op,
            belief_id=belief_id,
            target_belief_id=target_belief_id,
            summary=summary,
            details=details or {},
        )
        self._conn.execute(
            """
            INSERT INTO timeline_events (
                step, source_id, op, belief_id, target_belief_id,
                summary, details_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.step,
                event.source_id,
                event.op.value,
                event.belief_id,
                event.target_belief_id,
                event.summary,
                json.dumps(event.details),
                event.created_at_utc,
            ),
        )

    # ------------------------------------------------------------------
    # Raw chunk storage (for Naive RAG & Recency RAG baselines)
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        *,
        source_id: str,
        step: int,
        text: str,
        state_index: int | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | np.ndarray | None = None,
        chunk_id: str | None = None,
    ) -> str:
        cid = chunk_id or f"c_{uuid.uuid4().hex[:10]}"
        self._conn.execute(
            """
            INSERT INTO chunks (id, source_id, step, state_index, text, metadata_json, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                source_id,
                step,
                state_index,
                text,
                json.dumps(metadata or {}),
                _vec_to_bytes(embedding),
            ),
        )
        self._conn.commit()
        return cid

    def search_chunks(
        self,
        query_embedding: list[float] | np.ndarray,
        *,
        top_k: int = 6,
        recency_decay: float = 0.0,
    ) -> list[tuple[dict[str, Any], float]]:
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm

        rows = self._conn.execute(
            "SELECT id, source_id, step, state_index, text, metadata_json, embedding FROM chunks"
        ).fetchall()
        if not rows:
            return []

        max_step = max(int(r["step"]) for r in rows)
        scored: list[tuple[dict[str, Any], float]] = []
        for r in rows:
            vec = _bytes_to_vec(r["embedding"])
            if vec is None or vec.shape != q.shape:
                continue
            sim = float(np.dot(q, vec))
            if recency_decay > 0.0:
                age = max(0, max_step - int(r["step"]))
                sim *= float(np.exp(-recency_decay * age))
            scored.append(
                (
                    {
                        "id": r["id"],
                        "source_id": r["source_id"],
                        "step": int(r["step"]),
                        "state_index": r["state_index"],
                        "text": r["text"],
                        "metadata": json.loads(r["metadata_json"]),
                    },
                    sim,
                )
            )
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Belief retrieval & Time-Travel queries
    # ------------------------------------------------------------------

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        return Belief(
            id=row["id"],
            type=BeliefType(row["type"]),
            surface=row["surface"],
            claim=row["claim"],
            closed_set_items=json.loads(row["closed_set_json"]),
            status=BeliefStatus(row["status"]),
            confidence=float(row["confidence"]),
            durability=float(row["durability"]),
            evidence_count=int(row["evidence_count"]),
            created_at_step=int(row["created_at_step"]),
            updated_at_step=int(row["updated_at_step"]),
            superseded_by=row["superseded_by"],
            provenance=[Provenance.model_validate(p) for p in json.loads(row["provenance_json"])],
            metadata=json.loads(row["metadata_json"]),
        )

    def get_belief(self, belief_id: str) -> Belief | None:
        row = self._conn.execute("SELECT * FROM beliefs WHERE id = ?", (belief_id,)).fetchone()
        return self._row_to_belief(row) if row is not None else None

    def list_beliefs(
        self,
        *,
        status: BeliefStatus | None = None,
        surface: str | None = None,
    ) -> list[Belief]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if surface is not None:
            clauses.append("surface = ?")
            params.append(surface)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM beliefs {where} ORDER BY created_at_step ASC, id ASC",
            params,
        ).fetchall()
        return [self._row_to_belief(r) for r in rows]

    def search_beliefs(
        self,
        query_embedding: list[float] | np.ndarray,
        *,
        top_k: int = 6,
        active_only: bool = True,
        min_score: float = -1.0,
    ) -> list[tuple[Belief, float]]:
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm

        sql = "SELECT * FROM beliefs"
        params: list[Any] = []
        if active_only:
            sql += " WHERE status = ?"
            params.append(BeliefStatus.ACTIVE.value)

        rows = self._conn.execute(sql, params).fetchall()
        scored: list[tuple[Belief, float]] = []
        for row in rows:
            vec = _bytes_to_vec(row["embedding"])
            if vec is None or vec.shape != q.shape:
                continue
            score = float(np.dot(q, vec))
            if score >= min_score:
                scored.append((self._row_to_belief(row), score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def list_edges(self) -> list[BeliefEdge]:
        rows = self._conn.execute(
            "SELECT source_id, target_id, relation, step, reason FROM belief_edges ORDER BY step ASC, id ASC"
        ).fetchall()
        return [
            BeliefEdge(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation=r["relation"],
                step=int(r["step"]),
                reason=r["reason"],
            )
            for r in rows
        ]

    def list_events(self, *, max_step: int | None = None) -> list[TimelineEvent]:
        if max_step is not None:
            rows = self._conn.execute(
                "SELECT * FROM timeline_events WHERE step <= ? ORDER BY step ASC, id ASC",
                (max_step,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM timeline_events ORDER BY step ASC, id ASC"
            ).fetchall()
        return [
            TimelineEvent(
                id=int(r["id"]),
                step=int(r["step"]),
                source_id=r["source_id"],
                op=RevisionOp(r["op"]),
                belief_id=r["belief_id"],
                target_belief_id=r["target_belief_id"],
                summary=r["summary"],
                details=json.loads(r["details_json"]),
                created_at_utc=r["created_at_utc"],
            )
            for r in rows
        ]

    def get_state_at_step(self, step: int) -> list[Belief]:
        """Reconstruct every belief's status as of `step` for the Memory Time Machine."""
        all_beliefs = self.list_beliefs()
        events = self.list_events(max_step=step)

        status_at_step: dict[str, BeliefStatus] = {}
        superseded_by_at_step: dict[str, str | None] = {}
        for belief in all_beliefs:
            if belief.created_at_step <= step:
                status_at_step[belief.id] = BeliefStatus.ACTIVE
                superseded_by_at_step[belief.id] = None

        for ev in events:
            if ev.op == RevisionOp.SUPERSEDE and ev.target_belief_id in status_at_step:
                status_at_step[ev.target_belief_id] = BeliefStatus.SUPERSEDED
                superseded_by_at_step[ev.target_belief_id] = ev.belief_id
            elif ev.op == RevisionOp.REFUTE and ev.target_belief_id in status_at_step:
                status_at_step[ev.target_belief_id] = BeliefStatus.REFUTED
            elif ev.op == RevisionOp.FORGET and ev.target_belief_id in status_at_step:
                status_at_step[ev.target_belief_id] = BeliefStatus.FORGOTTEN

        snapshot: list[Belief] = []
        for belief in all_beliefs:
            if belief.id not in status_at_step:
                continue
            snapshot.append(
                belief.model_copy(
                    update={
                        "status": status_at_step[belief.id],
                        "superseded_by": superseded_by_at_step[belief.id],
                    }
                )
            )
        return snapshot

    def stats(self, *, total_ingested_sources: int = 0) -> MemoryStats:
        status_counts = dict(
            self._conn.execute(
                "SELECT status, COUNT(*) FROM beliefs GROUP BY status"
            ).fetchall()
        )
        op_counts = dict(
            self._conn.execute(
                "SELECT op, COUNT(*) FROM timeline_events GROUP BY op"
            ).fetchall()
        )
        chunk_count = int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        active_claims = self._conn.execute(
            "SELECT claim FROM beliefs WHERE status = ?",
            (BeliefStatus.ACTIVE.value,),
        ).fetchall()
        est_tokens = sum(max(1, len(row["claim"]) // 4) for row in active_claims)
        return MemoryStats(
            total_ingested_sources=total_ingested_sources,
            active_beliefs=int(status_counts.get(BeliefStatus.ACTIVE.value, 0)),
            superseded_beliefs=int(status_counts.get(BeliefStatus.SUPERSEDED.value, 0)),
            refuted_beliefs=int(status_counts.get(BeliefStatus.REFUTED.value, 0)),
            forgotten_beliefs=int(status_counts.get(BeliefStatus.FORGOTTEN.value, 0)),
            raw_chunks=chunk_count,
            estimated_active_tokens=est_tokens,
            revision_counts={str(k): int(v) for k, v in op_counts.items()},
        )
