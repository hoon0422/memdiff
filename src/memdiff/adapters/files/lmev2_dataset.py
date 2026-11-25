"""`LMEV2Dataset` over the fetched jsonl/json layout (see the fetch CLI)."""

from __future__ import annotations

import json
from pathlib import Path

from memdiff.benchmarks.lmev2.domain.dataset import LMEV2Question, trajectory_from_record
from memdiff.memory.domain.trajectory import Trajectory


class JsonlLMEV2Dataset:
    def __init__(
        self, root: str | Path, *, haystack_tier: str = "small", trajectories_file: str | None = None
    ) -> None:
        self._root = Path(root)
        self._tier = haystack_tier
        self._trajectories_file = trajectories_file or "trajectories_small.jsonl"

    def load_questions(self, *, domain: str | None = None) -> list[LMEV2Question]:
        path = self._root / "questions.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `uv run python scripts/fetch_lmev2_small.py` first."
            )
        rows: list[LMEV2Question] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                item = LMEV2Question.model_validate(json.loads(line))
                if domain is None or item.domain == domain:
                    rows.append(item)
        return rows

    def load_haystack(self) -> dict[str, list[str]]:
        path = self._root / "haystacks" / f"lme_v2_{self._tier}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing haystack file: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): [str(tid) for tid in v] for k, v in raw.items()}

    def load_trajectories(self, *, allowed_ids: set[str] | None = None) -> dict[str, Trajectory]:
        path = self._root / self._trajectories_file
        if not path.exists():
            fallback = self._root / "trajectories.jsonl"
            if not fallback.exists():
                raise FileNotFoundError(f"Missing trajectories file: {path}")
            path = fallback
        out: dict[str, Trajectory] = {}
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                record = json.loads(line)
                tid = str(record["id"])
                if allowed_ids is None or tid in allowed_ids:
                    out[tid] = trajectory_from_record(record)
        return out

    def image_path(self, question: LMEV2Question) -> Path | None:
        if not question.image:
            return None
        path = self._root / question.image
        return path if path.exists() else None
