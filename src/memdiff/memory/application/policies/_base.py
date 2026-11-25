from __future__ import annotations

from memdiff.memory.domain.trajectory import Trajectory


class SteppedPolicy:
    """Owns the ingest-step counter; it is only changed via `resume_from`."""

    def __init__(self) -> None:
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def resume_from(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self._step = step

    def _next_source_id(self, trajectory: Trajectory) -> str:
        self._step += 1
        return trajectory.id or f"traj_{self._step}"
