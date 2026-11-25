from __future__ import annotations

from pathlib import Path

from memdiff.audit.domain.replay import ReplayBundle


class JsonFileReplayWriter:
    """`ReplayWriter` for the Memory Time Machine viewer: one pretty-printed JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, bundle: ReplayBundle) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
