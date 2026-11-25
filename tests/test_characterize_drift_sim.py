"""DriftSim runner (all 4 policies) through the container against the #2 golden snapshots.

Only the SDK provider is faked; settings, prompts, caches, ledgers and runner are the real ones.
The goldens must never be regenerated for this ticket.
"""

from __future__ import annotations

import itertools
import json
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from dishka import Container, Provider, Scope, provide
from openai import OpenAI

from fake_sdk import RecordingFakeSDK
from memdiff.benchmarks.drift_sim.application.runner import DriftBenchmarkRunner
from memdiff.benchmarks.drift_sim.domain.episode import DriftSimConfig
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.drift_providers import DriftSimProvider
from memdiff.bootstrap.settings import Settings
from snapshot_utils import assert_run_matches_golden

POLICIES = ["naive_rag", "recency_rag", "structured_extract", "belief_revision"]
CONFIG = DriftSimConfig(seed=7, num_sessions=6, num_entities=3, num_durable_rules=2)


@pytest.fixture
def deterministic_ids_and_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter ids and a ticking clock: what the goldens were produced with."""
    ids = itertools.count(1)
    fake_uuid = types.SimpleNamespace(
        uuid4=lambda: types.SimpleNamespace(hex=f"{next(ids):010x}")
    )
    monkeypatch.setattr("memdiff.memory.domain.belief.uuid", fake_uuid)
    monkeypatch.setattr("memdiff.memory.domain.chunk.uuid", fake_uuid)
    ticks = itertools.count()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=next(ticks))

    monkeypatch.setattr("memdiff.audit.domain.timeline.datetime", Clock)


def drift_settings(tmp_path: Path, cache: str = "cache") -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_base_url="http://127.0.0.1:11434/v1",
        memdiff_cache_dir=tmp_path / cache,
        memdiff_chat_model="fake-chat",
        memdiff_embedding_model="fake-embed",
        memdiff_judge_model="fake-judge",
    )


def drift_container(tmp_path: Path, sdk: RecordingFakeSDK, cache: str = "cache") -> Container:
    settings = drift_settings(tmp_path, cache)

    class FakeSDKProvider(Provider):
        scope = Scope.APP

        @provide(override=True)
        def sdk_client(self) -> OpenAI:
            return sdk  # type: ignore[return-value]

    return build_container(settings, DriftSimProvider(tmp_path / "runs"), FakeSDKProvider())


def test_drift_sim_runner_matches_golden(
    tmp_path: Path, deterministic_ids_and_clock: None
) -> None:
    sdk = RecordingFakeSDK()
    container = drift_container(tmp_path, sdk)
    try:
        container.get(DriftBenchmarkRunner).run(
            config=CONFIG,
            policies=POLICIES,
            run_id="run_golden",
            top_k=3,
            think=False,  # exercises the reasoning_effort=none request path
        )
    finally:
        container.close()

    out_dir = tmp_path / "runs" / "run_golden"
    assert_run_matches_golden(
        "drift_sim",
        requests=sdk.requests,
        out_dir=out_dir,
        summary_file="summary.json",
        files={
            **{f"predictions_{p}.jsonl": f"{p}_predictions.jsonl" for p in POLICIES},
            "replay_belief_revision.json": "replay_belief_revision.json",
        },
    )
    assert {p.name for p in out_dir.iterdir()} == {
        "summary.json",
        "replay_belief_revision.json",
        *(f"{p}.sqlite" for p in POLICIES),
        *(f"{p}_predictions.jsonl" for p in POLICIES),
    }


def test_replay_falls_back_to_the_last_policy_when_belief_revision_is_absent(
    tmp_path: Path,
) -> None:
    container = drift_container(tmp_path, RecordingFakeSDK())
    try:
        summary = container.get(DriftBenchmarkRunner).run(
            config=CONFIG, policies=["naive_rag", "recency_rag"], run_id="r", top_k=3
        )
    finally:
        container.close()

    out_dir = tmp_path / "runs" / "r"
    assert summary["benchmark"] == "drift_sim"
    assert set(summary["policies"]) == {"naive_rag", "recency_rag"}
    replay = json.loads((out_dir / "replay_recency_rag.json").read_text(encoding="utf-8"))
    assert replay["run_id"] == "r"
    assert replay["comparisons"][0]["baseline_policy"] == "naive_rag"
    assert replay["comparisons"][0]["proposed_policy"] == "recency_rag"
    assert not (out_dir / "replay_naive_rag.json").exists()


def _run(
    tmp_path: Path, sdk: RecordingFakeSDK, policies: list[str], cache: str = "cache", **kw: Any
) -> dict[str, Any]:
    container = drift_container(tmp_path, sdk, cache)
    try:
        return container.get(DriftBenchmarkRunner).run(
            config=CONFIG, policies=policies, run_id="r", top_k=3, **kw
        )
    finally:
        container.close()


def test_resume_with_db_and_predictions_reuses_them_without_any_llm_call(
    tmp_path: Path,
) -> None:
    first_sdk = RecordingFakeSDK()
    first = _run(tmp_path, first_sdk, ["naive_rag", "belief_revision"])
    pred_path = tmp_path / "runs" / "r" / "naive_rag_predictions.jsonl"
    before = pred_path.read_text(encoding="utf-8")

    sdk = RecordingFakeSDK()
    resumed = _run(tmp_path, sdk, ["naive_rag", "belief_revision"], resume=True)

    assert sdk.requests == []
    assert pred_path.read_text(encoding="utf-8") == before  # predictions are not rewritten
    for name in ("naive_rag", "belief_revision"):
        m = resumed["policies"][name]
        assert m["resumed"] is True
        assert "ingest_seconds" not in m and "query_seconds" not in m
        assert m["memory_stats"]["total_ingested_sources"] == CONFIG.num_sessions
        fresh = first["policies"][name]
        assert m["memory_stats"] == fresh["memory_stats"]
        assert m["overall_accuracy"] == fresh["overall_accuracy"]


def test_resume_with_only_the_db_skips_ingest_and_requeries(tmp_path: Path) -> None:
    first = _run(tmp_path, RecordingFakeSDK(), ["structured_extract"])
    (tmp_path / "runs" / "r" / "structured_extract_predictions.jsonl").unlink()

    sdk = RecordingFakeSDK()
    # fresh LLM cache, so an extractor call could not be hidden behind a cache hit
    resumed = _run(tmp_path, sdk, ["structured_extract"], cache="cache2", resume=True)

    roles = {r.get("role") for r in sdk.requests if r["kind"] == "chat"}
    assert roles == {"drift_reader"}  # no extractor calls: the ledger is reused, not re-ingested
    m = resumed["policies"]["structured_extract"]
    assert "resumed" not in m
    assert m["memory_stats"] == first["policies"]["structured_extract"]["memory_stats"]
    assert m["memory_stats"]["total_ingested_sources"] == CONFIG.num_sessions
    assert (tmp_path / "runs" / "r" / "structured_extract_predictions.jsonl").exists()


def test_without_resume_an_existing_run_is_rebuilt_from_scratch(tmp_path: Path) -> None:
    first = _run(tmp_path, RecordingFakeSDK(), ["naive_rag"])
    sdk = RecordingFakeSDK()
    second = _run(tmp_path, sdk, ["naive_rag"], cache="cache2")

    assert any(r["kind"] == "embedding" for r in sdk.requests)  # it ingested again
    # a truncated ledger, not one with doubled chunks
    assert second["policies"]["naive_rag"]["memory_stats"] == first["policies"]["naive_rag"][
        "memory_stats"
    ]


def test_cli_runs_through_the_container_and_prints_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from memdiff.presentation.cli.drift_sim import main

    sdk = RecordingFakeSDK()
    settings = drift_settings(tmp_path)

    class FakeSDKProvider(Provider):
        scope = Scope.APP

        @provide(override=True)
        def sdk_client(self) -> OpenAI:
            return sdk  # type: ignore[return-value]

    main(
        ["--run-id", "cli", "--runs-dir", str(tmp_path / "out"), "--sessions", "6",
         "--entities", "3", "--durable-rules", "2", "--seed", "7", "--top-k", "3",
         "--no-think"],
        settings=settings,
        overrides=[FakeSDKProvider()],
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "out" / "cli" / "summary.json").read_text(encoding="utf-8"))
    assert printed["run_id"] == saved["run_id"] == "cli"
    assert list(printed["policies"]) == ["naive_rag", "recency_rag"]  # CLI default policies
    assert printed["config"]["num_sessions"] == 6
    assert (tmp_path / "out" / "cli" / "replay_recency_rag.json").exists()
