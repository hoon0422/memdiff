"""One memory workspace = one policy over its own ledger. Runners open one per policy/group."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, SkipValidation

from memdiff.audit.application.ports import AuditLogRepository
from memdiff.audit.application.replay import ReplayService
from memdiff.memory.application.policy import MemoryPolicy
from memdiff.memory.application.ports import BeliefRepository


class WorkspaceSpec(BaseModel):
    """What differs between workspaces; `think` is a per-run flag, not a Settings field."""

    policy_name: str
    db_path: Path | str = ":memory:"
    recency_decay: float = 0.05
    max_chunk_chars: int = 2400
    candidate_top_k: int = 5
    think: bool | None = None


class MemoryWorkspace(BaseModel):
    """Handles over one open ledger. Fields are ports (Protocols), so pydantic only holds them."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    policy: SkipValidation[MemoryPolicy]
    # read handles over the same ledger, valid until the `with` exits
    beliefs: SkipValidation[BeliefRepository]
    log: SkipValidation[AuditLogRepository]
    replay: ReplayService  # build_bundle / state_at_step


class MemoryWorkspaceFactory(Protocol):
    def open(self, spec: WorkspaceSpec) -> AbstractContextManager[MemoryWorkspace]:
        """Open (creating, never truncating) the ledger; leaving the `with` releases the DB."""
        ...
