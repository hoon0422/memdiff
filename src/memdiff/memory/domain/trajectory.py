"""Typed trajectory input (LME-V2 trajectories and DriftSim sessions share this shape)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class TrajectoryState(BaseModel):
    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    state_index: int = 0
    url: str = ""
    action: str | None = None
    accessibility_tree: str | None = None
    text: str = ""

    @property
    def content(self) -> str:
        return self.text if self.accessibility_tree is None else self.accessibility_tree


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    id: str | None = None
    goal: str = ""
    outcome: str = "unknown"
    states: list[TrajectoryState] = []

    @field_validator("states", mode="before")
    @classmethod
    def _keep_dict_states(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []
        return [s for s in value if isinstance(s, dict | TrajectoryState)]
