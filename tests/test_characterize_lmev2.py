"""Characterization of the LME-V2 runner (4 policies) on a synthetic fixture against golden snapshots.

Update goldens after an intended behavior change: MEMDIFF_UPDATE_GOLDEN=1 uv run pytest tests/test_characterize_*.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fake_sdk import RecordingFakeSDK
from memdiff.benchmarks.lmev2.runner import run_lmev2_policy
from memdiff.config import Settings
from memdiff.llm.openai_client import CachedLLMClient
from snapshot_utils import assert_run_matches_golden

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "lmev2_synthetic"


@pytest.mark.parametrize(
    "policy", ["naive_rag", "recency_rag", "structured_extract", "belief_revision"]
)
def test_lmev2_runner_matches_golden(
    policy: str, tmp_path: Path, deterministic_runtime: None
) -> None:
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
        run_lmev2_policy(
            data_root=FIXTURE_ROOT,
            domain="enterprise",
            policy_name=policy,
            output_dir=out_dir,
            top_k=3,
            settings=settings,
            client=client,
        )
    finally:
        client.close()

    assert_run_matches_golden(
        f"lmev2/{policy}",
        requests=sdk.requests,
        out_dir=out_dir,
        summary_file=f"{policy}_metrics.json",
        files={
            "predictions.jsonl": f"{policy}_predictions.jsonl",
            "replay.json": f"replay_{policy}.json",
        },
    )
