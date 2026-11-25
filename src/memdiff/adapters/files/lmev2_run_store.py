"""`LMEV2RunStore`: one output directory per run, file names as in the original runner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memdiff.adapters.replay.json_file import JsonFileReplayWriter
from memdiff.audit.domain.replay import ReplayBundle


class JsonLMEV2RunStore:
    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self._dir.name

    def fresh_db_path(self, policy_name: str, group_idx: int) -> Path:
        path = self._dir / f"{policy_name}_group{group_idx}.sqlite"
        path.unlink(missing_ok=True)
        return path

    def write_replay(self, policy_name: str, bundle: ReplayBundle) -> None:
        JsonFileReplayWriter(self._dir / f"replay_{policy_name}.json").write(bundle)

    def write_predictions(self, policy_name: str, records: list[dict[str, Any]]) -> None:
        with (self._dir / f"{policy_name}_predictions.jsonl").open("w", encoding="utf-8") as fp:
            for rec in records:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_summary(self, policy_name: str, summary: Mapping[str, Any]) -> None:
        (self._dir / f"{policy_name}_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
