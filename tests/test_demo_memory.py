from __future__ import annotations

import runpy
from pathlib import Path

DEMO = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "demo_memory.py"))


def test_offline_demo_supersedes_stale_values_and_answers_from_current(tmp_path: Path) -> None:
    result = DEMO["run_demo"](out_dir=tmp_path, sessions=5, top_k=3)
    assert result["hits"] == result["queries"]
    assert result["SUPERSEDE"] > 0
    assert result["IGNORE"] > 0
    assert (tmp_path / "demo.sqlite").exists()
    assert (tmp_path / "replay_demo.json").exists()
