from __future__ import annotations

from sqlalchemy import LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class CacheBase(DeclarativeBase):
    """Separate metadata from the ledger DB: the LLM cache lives in its own SQLite file."""


class ChatCacheRow(CacheBase):
    __tablename__ = "chat_cache"

    cache_key: Mapped[str] = mapped_column(String, primary_key=True)
    response_text: Mapped[str] = mapped_column(Text)


class EmbeddingCacheRow(CacheBase):
    __tablename__ = "embedding_cache"

    cache_key: Mapped[str] = mapped_column(String, primary_key=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
