"""`DriftRunStore` on disk: `<runs_dir>/<run_id>/{policy}.sqlite, {policy}_predictions.jsonl, ...`."""

from __future__ import annotations

import json
from pathlib import Path

from memdiff.adapters.replay.json_file import JsonFileReplayWriter
from memdiff.audit.domain.replay import ReplayBundle
from memdiff.benchmarks.drift_sim.domain.prediction import DriftPrediction


class FileDriftRunStore:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    def _dir(self, run_id: str) -> Path:
        path = self._runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _db(self, run_id: str, policy: str) -> Path:
        return self._dir(run_id) / f"{policy}.sqlite"

    def _predictions(self, run_id: str, policy: str) -> Path:
        return self._dir(run_id) / f"{policy}_predictions.jsonl"

    def db_path(self, run_id: str, policy: str) -> Path:
        return self._db(run_id, policy)

    def db_exists(self, run_id: str, policy: str) -> bool:
        return self._db(run_id, policy).exists()

    def discard_db(self, run_id: str, policy: str) -> None:
        self._db(run_id, policy).unlink(missing_ok=True)

    def load_predictions(self, run_id: str, policy: str) -> list[DriftPrediction] | None:
        path = self._predictions(run_id, policy)
        if not path.exists():
            return None
        return [
            DriftPrediction.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def save_predictions(
        self, run_id: str, policy: str, predictions: list[DriftPrediction]
    ) -> None:
        with self._predictions(run_id, policy).open("w", encoding="utf-8") as fp:
            for pred in predictions:
                fp.write(json.dumps(pred.model_dump(), ensure_ascii=False) + "\n")

    def save_replay(self, run_id: str, policy: str, bundle: ReplayBundle) -> None:
        JsonFileReplayWriter(self._dir(run_id) / f"replay_{policy}.json").write(bundle)

    def save_summary(self, run_id: str, summary: dict[str, object]) -> None:
        (self._dir(run_id) / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
