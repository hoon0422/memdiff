from __future__ import annotations

import numpy as np
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from memdiff.adapters.persistence.sqlalchemy.cache_models import (
    CacheBase,
    ChatCacheRow,
    EmbeddingCacheRow,
)


class SqlLLMResponseCache:
    """`LLMResponseCache` on SQLAlchemy ORM. One short session per operation."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        CacheBase.metadata.create_all(engine)

    def get_chat(self, key: str) -> str | None:
        with Session(self._engine) as session:
            row = session.get(ChatCacheRow, key)
            return row.response_text if row else None

    def put_chat(self, key: str, text: str) -> None:
        with Session(self._engine) as session, session.begin():
            session.merge(ChatCacheRow(cache_key=key, response_text=text))

    def get_embedding(self, key: str) -> np.ndarray | None:
        with Session(self._engine) as session:
            row = session.get(EmbeddingCacheRow, key)
            return np.frombuffer(row.vector, dtype=np.float32).copy() if row else None

    def put_embeddings(self, items: dict[str, np.ndarray]) -> None:
        with Session(self._engine) as session, session.begin():
            for key, vec in items.items():
                session.merge(
                    EmbeddingCacheRow(cache_key=key, vector=vec.astype(np.float32).tobytes())
                )
