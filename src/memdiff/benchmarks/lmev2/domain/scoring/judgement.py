"""Parse the binary label out of an LLM judge reply."""

from __future__ import annotations

import json
import re


def parse_binary_judgement(text: str) -> tuple[int, str]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = payload.get("label")
            if label in {0, 1, "0", "1"}:
                return int(label), str(payload.get("reason", ""))
        except json.JSONDecodeError:
            pass
    label_match = re.search(r'"?label"?\s*[:=]\s*([01])', cleaned, flags=re.IGNORECASE)
    if label_match:
        return int(label_match.group(1)), cleaned
    raise ValueError(f"Could not parse binary LLM judgement: {cleaned!r}")
