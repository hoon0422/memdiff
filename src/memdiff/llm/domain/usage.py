from __future__ import annotations

from pydantic import BaseModel


class UsageStats(BaseModel):
    chat_calls: int = 0
    chat_cache_hits: int = 0
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            **self.model_dump(),
            "total_tokens": self.prompt_tokens + self.completion_tokens + self.embedding_tokens,
        }
