"""Demo wiring. Passed to `build_container(settings, DemoProvider(replay_path))` by the demo CLI."""

from __future__ import annotations

from pathlib import Path

from dishka import Provider, Scope, provide

from memdiff.adapters.replay.json_file import JsonFileReplayWriter
from memdiff.audit.application.ports import ReplayWriter


class DemoProvider(Provider):
    scope = Scope.APP

    def __init__(self, replay_path: Path) -> None:
        super().__init__()
        self._replay_path = replay_path

    @provide
    def replay_writer(self) -> ReplayWriter:
        return JsonFileReplayWriter(self._replay_path)
