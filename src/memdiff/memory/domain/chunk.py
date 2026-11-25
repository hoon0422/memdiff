from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


def new_chunk_id() -> str:
    return f"c_{uuid.uuid4().hex[:10]}"


class Chunk(BaseModel):
    """Raw trajectory-state text kept by the RAG baselines."""

    id: str
    source_id: str
    step: int
    state_index: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
