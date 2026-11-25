from __future__ import annotations

import httpx


class OllamaModelLister:
    """`ModelLister` backed by Ollama's `/api/tags` endpoint."""

    def __init__(self, root_url: str | None, *, timeout: float = 5.0) -> None:
        self._root_url = root_url
        self._timeout = timeout

    def list_models(self) -> list[str]:
        if not self._root_url:
            return []
        resp = httpx.get(f"{self._root_url}/api/tags", timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()
        return [
            str(m["name"]) for m in payload.get("models", []) if isinstance(m, dict) and "name" in m
        ]
