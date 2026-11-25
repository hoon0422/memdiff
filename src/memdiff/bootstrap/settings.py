from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default: local/SSH-tunneled Ollama OpenAI-compatible endpoint
    llm_base_url: str | None = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "ollama"
    llm_timeout_seconds: float = 600.0
    embedding_batch_size: int = 32

    # Optional OpenAI fallbacks
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # 12GB VRAM RTX 3060 defaults (Qwen3.5-9B + Qwen3-Embedding-0.6B co-resident in VRAM)
    memdiff_chat_model: str = "qwen3.5:9b"
    memdiff_embedding_model: str = "qwen3-embedding:0.6b"
    memdiff_judge_model: str = "qwen3.5:9b"

    memdiff_data_dir: Path = Path("data/longmemeval-v2")
    memdiff_cache_dir: Path = Path(".cache/llm")

    @property
    def effective_base_url(self) -> str | None:
        if self.llm_base_url and self.llm_base_url.strip():
            return self.llm_base_url.strip()
        if self.openai_base_url and self.openai_base_url.strip():
            return self.openai_base_url.strip()
        return None

    @property
    def effective_api_key(self) -> str | None:
        if self.effective_base_url:
            return self.llm_api_key or self.openai_api_key or "ollama"
        return self.openai_api_key or (self.llm_api_key if self.llm_api_key != "ollama" else None)

    @property
    def ollama_root_url(self) -> str | None:
        base = self.effective_base_url
        if not base:
            return None
        return base[:-3] if base.endswith("/v1") else base.rstrip("/")


def get_settings() -> Settings:
    return Settings()
