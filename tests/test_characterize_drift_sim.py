"""Characterization of the DriftSim runner (all 4 policies) against golden snapshots.

Update goldens after an intended behavior change: MEMDIFF_UPDATE_GOLDEN=1 uv run pytest tests/test_characterize_*.py
"""

from __future__ import annotations

from pathlib import Path

from fake_sdk import RecordingFakeSDK
from memdiff.benchmarks.drift_sim.runner import run_drift_sim_benchmark
from memdiff.benchmarks.drift_sim.simulator import DriftSimConfig
from memdiff.config import Settings
from memdiff.llm.openai_client import CachedLLMClient
from snapshot_utils import assert_run_matches_golden

POLICIES = ["naive_rag", "recency_rag", "structured_extract", "belief_revision"]


def test_drift_sim_runner_matches_golden(tmp_path: Path, deterministic_runtime: None) -> None:
    sdk = RecordingFakeSDK()
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_base_url="http://127.0.0.1:11434/v1",
        memdiff_cache_dir=tmp_path / "cache",
        memdiff_chat_model="fake-chat",
        memdiff_embedding_model="fake-embed",
        memdiff_judge_model="fake-judge",
    )
    client = CachedLLMClient(settings=settings, client=sdk)
    out_dir = tmp_path / "run_golden"
    try:
        run_drift_sim_benchmark(
            config=DriftSimConfig(seed=7, num_sessions=6, num_entities=3, num_durable_rules=2),
            policies=POLICIES,
            output_dir=out_dir,
            top_k=3,
            think=False,  # exercises the reasoning_effort=none request path
            client=client,
        )
    finally:
        client.close()

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
