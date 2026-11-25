"""LME-V2 benchmark through the container (only the SDK provider is faked) against the #2 goldens.

Goldens are never regenerated here: the request log, metrics, predictions and replay JSON were
produced by the pre-refactor runner. Update (after an intended behavior change only):
MEMDIFF_UPDATE_GOLDEN=1 uv run pytest tests/test_characterize_lmev2.py
"""

from __future__ import annotations

import itertools
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from dishka import Container, Provider, Scope, provide
from openai import OpenAI

from fake_sdk import RecordingFakeSDK
from memdiff.audit.domain.replay import ReplayBundle
from memdiff.benchmarks.lmev2.application.runner import LMEV2BenchmarkRunner, LMEV2RunParams
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.lmev2_provider import LMEV2Provider
from memdiff.bootstrap.settings import Settings
from snapshot_utils import assert_run_matches_golden

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "lmev2_synthetic"
POLICIES = ["naive_rag", "recency_rag", "structured_extract", "belief_revision"]


@pytest.fixture
def deterministic_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter belief/chunk ids and a ticking clock: what the goldens were produced with."""
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


def lmev2_container(
    tmp_path: Path, sdk: RecordingFakeSDK, *, data_root: Path = FIXTURE_ROOT, out_dir: Path
) -> Container:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_base_url="http://127.0.0.1:11434/v1",
        memdiff_cache_dir=tmp_path / "cache",
        memdiff_chat_model="fake-chat",
        memdiff_embedding_model="fake-embed",
        memdiff_judge_model="fake-judge",
    )

    class FakeSDKProvider(Provider):
        scope = Scope.APP

        @provide(override=True)
        def sdk_client(self) -> OpenAI:
            return sdk  # type: ignore[return-value]

    return build_container(
        settings, FakeSDKProvider(), LMEV2Provider(data_root=data_root, output_dir=out_dir)
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_lmev2_runner_matches_golden(
    policy: str, tmp_path: Path, deterministic_runtime: None
) -> None:
    sdk = RecordingFakeSDK()
    out_dir = tmp_path / "run_golden"
    container = lmev2_container(tmp_path, sdk, out_dir=out_dir)
    try:
        container.get(LMEV2BenchmarkRunner).run(
            LMEV2RunParams(domain="enterprise", policy_name=policy, top_k=3)
        )
    finally:
        container.close()

    assert_run_matches_golden(
        f"lmev2/{policy}",
        requests=sdk.requests,
        out_dir=out_dir,
        summary_file=f"{policy}_metrics.json",
        files={
            "predictions.jsonl": f"{policy}_predictions.jsonl",
            "replay_group1.json": f"replay_{policy}_group1.json",
            "replay_group2.json": f"replay_{policy}_group2.json",
        },
    )


def test_lmev2_every_group_keeps_its_replay(tmp_path: Path, deterministic_runtime: None) -> None:
    """#14: the fixture has two haystack groups; each must leave its own schema-valid replay."""
    out_dir = tmp_path / "run_groups"
    container = lmev2_container(tmp_path, RecordingFakeSDK(), out_dir=out_dir)
    try:
        container.get(LMEV2BenchmarkRunner).run(
            LMEV2RunParams(domain="enterprise", policy_name="belief_revision", top_k=3)
        )
    finally:
        container.close()

    assert sorted(p.name for p in out_dir.glob("replay_*.json")) == [
        "replay_belief_revision_group1.json",
        "replay_belief_revision_group2.json",
    ]
    bundles = [
        ReplayBundle.model_validate_json(
            (out_dir / f"replay_belief_revision_group{i}.json").read_text(encoding="utf-8")
        )
        for i in (1, 2)
    ]
    assert bundles[0].events != bundles[1].events  # distinct ledgers, not one overwritten copy
