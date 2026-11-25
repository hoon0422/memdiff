"""Memory workspace seam: the container opens one request-scoped policy + ledger per `open()`.

The fake OpenAI SDK is swapped in through a container override, so the request bytes of the
LLM embedder, extractor and reconciler are compared against the #2 golden request log.
"""

from __future__ import annotations

import itertools
import json
import types
from pathlib import Path
from typing import Any

import pytest
from dishka import Container, Provider, Scope, provide
from openai import OpenAI

from fake_sdk import RecordingFakeSDK
from memdiff.benchmarks.common.application.workspace import (
    MemoryWorkspaceFactory,
    WorkspaceSpec,
)
from memdiff.benchmarks.drift_sim.domain.episode import DriftSimConfig, generate_episode
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.settings import Settings
from memdiff.memory.domain.trajectory import Trajectory

GOLDEN = Path(__file__).parent / "golden"
POLICIES = ["naive_rag", "recency_rag", "structured_extract", "belief_revision"]


@pytest.fixture
def counter_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same `b_0000000001` / `c_0000000001` sequence the legacy store gave the golden run."""
    ids = itertools.count(1)
    fake = types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex=f"{next(ids):010x}"))
    monkeypatch.setattr("memdiff.memory.domain.belief.uuid", fake)
    monkeypatch.setattr("memdiff.memory.domain.chunk.uuid", fake)


def fake_sdk_container(tmp_path: Path, sdk: RecordingFakeSDK) -> Container:
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

    return build_container(settings, FakeSDKProvider())


def test_llm_stack_sends_the_golden_embedding_extractor_and_reconciler_requests(
    tmp_path: Path, counter_ids: None
) -> None:
    sdk = RecordingFakeSDK()
    container = fake_sdk_container(tmp_path, sdk)
    episode = generate_episode(
        DriftSimConfig(seed=7, num_sessions=6, num_entities=3, num_durable_rules=2)
    )
    factory = container.get(MemoryWorkspaceFactory)
    try:
        for name in POLICIES:
            spec = WorkspaceSpec(policy_name=name, db_path=tmp_path / f"{name}.sqlite",
                                 recency_decay=0.15, think=False)
            with factory.open(spec) as ws:
                for session in episode.sessions:
                    ws.policy.insert(Trajectory.model_validate(session.to_trajectory_dict()))
                for q in episode.queries:
                    ws.policy.query(q.question, top_k=3)
    finally:
        container.close()

    def comparable(requests: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        # reader chats are the runner's job (#10); everything else is this ticket's
        return [(r["kind"], r["request"]) for r in requests if r.get("role") != "drift_reader"
                and not (r["kind"] == "chat" and r["request"]["messages"][0]["content"]
                         .startswith("You are an operations assistant"))]

    golden = json.loads((GOLDEN / "drift_sim" / "requests.json").read_text(encoding="utf-8"))
    assert comparable(sdk.requests) == comparable(golden)


def test_leaving_the_workspace_releases_the_ledger_and_keeps_it_for_reopening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counter_ids: None
) -> None:
    from memdiff.adapters.persistence.sqlalchemy.ledger import make_ledger_engine

    engines = []

    def spy(db_path: Path | str) -> Any:
        engines.append(make_ledger_engine(db_path))
        return engines[-1]

    monkeypatch.setattr("memdiff.bootstrap.providers.make_ledger_engine", spy)
    container = fake_sdk_container(tmp_path, RecordingFakeSDK())
    factory = container.get(MemoryWorkspaceFactory)
    spec = WorkspaceSpec(policy_name="naive_rag", db_path=tmp_path / "ws" / "naive.sqlite")
    traj = Trajectory(id="t1", states=[{"state_index": 0, "text": "hello world"}])
    try:
        with factory.open(spec) as ws:
            ws.policy.insert(traj)
            assert engines[0].pool.checkedin() == 1  # the ledger connection is held open ...
        assert engines[0].pool.checkedin() == 0  # ... and released (engine disposed) on exit

        with factory.open(spec) as ws:  # reopening keeps the data (resume), never truncates
            assert ws.policy.stats().raw_chunks == 1
    finally:
        container.close()


def test_unknown_policy_name_is_rejected(tmp_path: Path) -> None:
    container = fake_sdk_container(tmp_path, RecordingFakeSDK())
    try:
        with pytest.raises(ValueError, match="Unknown policy 'nope'"):
            with container.get(MemoryWorkspaceFactory).open(WorkspaceSpec(policy_name="nope")):
                pass
    finally:
        container.close()


@pytest.mark.parametrize("policy", ["structured_extract", "belief_revision"])
def test_lmev2_trajectories_produce_the_golden_extractor_and_reconciler_requests(
    policy: str, tmp_path: Path, counter_ids: None
) -> None:
    """Sampled long UI trajectories (domain=enterprise, failure -> success) through the LLM stack."""
    from collections import defaultdict

    from memdiff.adapters.files.lmev2_dataset import JsonlLMEV2Dataset
    from memdiff.benchmarks.lmev2.domain.dataset import needed_trajectory_ids

    root = Path(__file__).parent / "fixtures" / "lmev2_synthetic"
    dataset = JsonlLMEV2Dataset(root)
    questions = dataset.load_questions(domain="enterprise")
    haystack = dataset.load_haystack()
    trajectories = dataset.load_trajectories(allowed_ids=needed_trajectory_ids(questions, haystack))
    groups: dict[tuple[str, ...], list[Any]] = defaultdict(list)
    for q in questions:
        groups[tuple(haystack[q.id])].append(q)

    sdk = RecordingFakeSDK()
    container = fake_sdk_container(tmp_path, sdk)
    try:
        for i, (order, qs) in enumerate(groups.items(), start=1):
            spec = WorkspaceSpec(policy_name=policy, db_path=tmp_path / f"g{i}.sqlite")
            with container.get(MemoryWorkspaceFactory).open(spec) as ws:
                for tid in order:
                    ws.policy.insert(trajectories[tid])
                for q in qs:
                    ws.policy.query(q.question, top_k=3)
    finally:
        container.close()

    def comparable(requests: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        keep = {None, "extractor", "reconciler"}  # None = embeddings; readers/judges are #11's
        return [(r["kind"], r["request"]) for r in requests if r.get("role") in keep]

    golden = json.loads((GOLDEN / "lmev2" / policy / "requests.json").read_text(encoding="utf-8"))
    assert len(comparable(golden)) >= 8  # guards against comparing two empty logs
    assert comparable(sdk.requests) == comparable(golden)


def test_llm_extractor_and_reconciler_parse_replies_and_fall_back_safely(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from memdiff.memory.application.ports import BeliefExtractor, BeliefReconciler
    from memdiff.memory.domain.belief import Belief, BeliefType
    from memdiff.memory.domain.revision import RevisionOp

    replies = iter(
        [
            '```json\n{"beliefs": [{"type": "dynamic_state", "surface": "billing-api.region", '
            '"claim": "billing-api region is eu-central-1", "durability": 0.9, "step_index": 0}]}\n```',
            '{"op": "SUPERSEDE", "target_belief_id": "b_old", "reason": "newer"}',
            '{"op": "SUPERSEDE", "target_belief_id": "b_nonexistent", "reason": "bad id"}',
        ]
    )

    class ScriptedSDK(RecordingFakeSDK):
        def create(self, **kwargs: Any) -> Any:
            text = next(replies)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    container = fake_sdk_container(tmp_path, ScriptedSDK())
    try:
        with container(context={WorkspaceSpec: WorkspaceSpec(policy_name="belief_revision")}) as r:
            extractor, reconciler = r.get(BeliefExtractor), r.get(BeliefReconciler)
            traj = Trajectory.model_validate(
                {"id": "s002", "domain": "drift_sim", "outcome": "success",
                 "states": [{"state_index": 0, "accessibility_tree": "billing-api region"}]}
            )
            (cand,) = extractor.extract(traj, 2)
            assert (cand.id, cand.type, cand.surface) == (
                "cand_s002_0", BeliefType.DYNAMIC_STATE, "billing-api.region")
            assert cand.provenance[0].outcome == "success"

            assert reconciler.decide(cand, []).op == RevisionOp.ADD  # no LLM call (no reply used)
            old = Belief(id="b_old", type=BeliefType.DYNAMIC_STATE, surface="billing-api.region",
                         claim="billing-api region is us-east-1", created_at_step=1,
                         updated_at_step=1)
            hit = reconciler.decide(cand, [(old, 0.92)])
            assert (hit.op, hit.target_belief_id) == (RevisionOp.SUPERSEDE, "b_old")
            bad = reconciler.decide(cand, [(old, 0.91)])
            assert (bad.op, bad.reason) == (RevisionOp.ADD, "invalid_target_fallback")
    finally:
        container.close()
