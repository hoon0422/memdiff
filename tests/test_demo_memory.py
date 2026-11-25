from __future__ import annotations

import itertools
import runpy
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fake_sdk import FakeSDKProvider, RecordingFakeSDK, fake_settings

DEMO = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "demo_memory.py"))
GOLDEN = Path(__file__).parent / "golden" / "demo"


def test_offline_demo_supersedes_stale_values_and_answers_from_current(tmp_path: Path) -> None:
    result = DEMO["run_demo"](
        settings=fake_settings(tmp_path), out_dir=tmp_path, sessions=5, top_k=3
    )
    assert result["hits"] == result["queries"]
    assert result["SUPERSEDE"] > 0
    assert result["IGNORE"] > 0
    assert (tmp_path / "demo.sqlite").exists()
    assert (tmp_path / "replay_demo.json").exists()


def test_live_demo_runs_offline_against_the_fake_sdk(tmp_path: Path) -> None:
    sdk = RecordingFakeSDK()
    result = DEMO["run_demo"](
        settings=fake_settings(tmp_path),
        out_dir=tmp_path / "live",
        sessions=5,
        top_k=3,
        live=True,
        providers=[FakeSDKProvider(sdk)],
    )
    roles = {r.get("role") for r in sdk.requests if r["kind"] == "chat"}
    assert {"extractor", "reconciler"} <= roles  # the LLM path, not the rule-based offline stubs
    assert any(r["kind"] == "embedding" for r in sdk.requests)
    assert result["SUPERSEDE"] > 0
    assert (tmp_path / "live" / "replay_demo.json").exists()


def test_offline_demo_output_and_replay_match_the_pre_refactor_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # goldens were produced by the legacy demo with the same counter ids and ticking clock
    ids = itertools.count(1)
    fake_uuid = types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex=f"{next(ids):010x}"))
    monkeypatch.setattr("memdiff.memory.domain.belief.uuid", fake_uuid)
    monkeypatch.setattr("memdiff.memory.domain.chunk.uuid", fake_uuid)
    ticks = itertools.count()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=next(ticks))

    monkeypatch.setattr("memdiff.audit.domain.timeline.datetime", Clock)

    out = tmp_path / "demo"
    DEMO["run_demo"](settings=fake_settings(tmp_path), out_dir=out, sessions=5, top_k=3)

    stdout = capsys.readouterr().out.replace(str(out), "<OUT>")
    assert stdout == (GOLDEN / "stdout.txt").read_text(encoding="utf-8")
    assert (out / "replay_demo.json").read_text(encoding="utf-8") == (
        GOLDEN / "replay_demo.json"
    ).read_text(encoding="utf-8")
