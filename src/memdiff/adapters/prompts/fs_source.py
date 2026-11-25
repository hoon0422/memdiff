from __future__ import annotations

from pathlib import Path

from memdiff.llm.domain.prompt import PromptError


class FilesystemPromptSource:
    def __init__(self, root: Path) -> None:
        self._root = root

    def read(self, name: str) -> str:
        path = self._root / f"{name}.md"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise PromptError(f"prompt {name}: file missing: {path}") from exc
