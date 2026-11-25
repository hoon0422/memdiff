"""Parse `name|key=value|...` eval_function specs into a typed value object."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from memdiff.benchmarks.lmev2.domain.scoring.matchers import MATCHERS

JUDGE_EVAL_NAMES = frozenset({"llm_abstention_checker", "llm_gotchas_checker"})


def _parse_option_value(key: str, value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if key in {"separators", "separator"}:
        if not value:
            return []
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return json.loads(stripped)
        return [ch for ch in value if not ch.isspace()]
    return value


class EvalSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    options: dict[str, Any]

    @property
    def is_judge(self) -> bool:
        return self.name in JUDGE_EVAL_NAMES

    @classmethod
    def parse(cls, spec: str) -> EvalSpec:
        parts = [p.strip() for p in spec.split("|") if p.strip()]
        if not parts:
            raise ValueError("Empty eval_function spec")
        name = parts[0]
        if name not in MATCHERS and name not in JUDGE_EVAL_NAMES:
            raise ValueError(f"Unknown eval function: {name}")
        options: dict[str, Any] = {}
        for part in parts[1:]:
            key, value = part.split("=", 1)
            options[key.strip()] = _parse_option_value(key.strip(), value.strip())
        return cls(name=name, options=options)
