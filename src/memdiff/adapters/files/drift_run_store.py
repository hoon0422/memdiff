"""`DriftRunStore` on disk: `<runs_dir>/<run_id>/{policy}.sqlite, {policy}_predictions.jsonl, ...`."""

from __future__ import annotations

import json
from pathlib import Path

from memdiff.adapters.replay.json_file import JsonFileReplayWriter
from memdiff.audit.domain.replay import ReplayBundle
from memdiff.benchmarks.drift_sim.domain.prediction import DriftPrediction


class FileDriftRunStore:
    def __init__(self, run_dir: Path) -> None:
        self._dir = run_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_dir.name

    def _db(self, policy: str) -> Path:
        return self._dir / f"{policy}.sqlite"

    def _predictions(self, policy: str) -> Path:
        return self._dir / f"{policy}_predictions.jsonl"

    def db_path(self, policy: str) -> Path:
        return self._db(policy)

    def db_exists(self, policy: str) -> bool:
        return self._db(policy).exists()

    def discard_db(self, policy: str) -> None:
        self._db(policy).unlink(missing_ok=True)

    def load_predictions(self, policy: str) -> list[DriftPrediction] | None:
        path = self._predictions(policy)
        if not path.exists():
            return None
        return [
            DriftPrediction.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def save_predictions(self, policy: str, predictions: list[DriftPrediction]) -> None:
        with self._predictions(policy).open("w", encoding="utf-8") as fp:
            for pred in predictions:
                fp.write(json.dumps(pred.model_dump(), ensure_ascii=False) + "\n")

    def save_replay(self, policy: str, bundle: ReplayBundle) -> None:
        JsonFileReplayWriter(self._dir / f"replay_{policy}.json").write(bundle)

    def save_summary(self, summary: dict[str, object]) -> None:
        (self._dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
