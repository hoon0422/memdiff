from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memdiff.benchmarks.drift_sim.runner import run_drift_sim_benchmark
from memdiff.benchmarks.drift_sim.simulator import DriftSimConfig
from memdiff.benchmarks.lmev2.domain.dataset import LMEV2Question, select_balanced_questions
from memdiff.config import Settings
from memdiff.core.models import Belief, BeliefType, RevisionOp
from memdiff.llm.openai_client import CachedLLMClient
from memdiff.policies.llm_ops import make_llm_extractor, make_llm_reconciler


class _DeterministicFakeSDK:
    """Offline fake OpenAI SDK that returns boxed answers matching the latest chunk."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.embeddings = self

    def create(self, **kwargs: Any) -> Any:
        if "input" in kwargs:
            inputs: list[str] = kwargs["input"]
            data = []
            for text in inputs:
                # Simple bag-of-characters projection into 8 dims for deterministic retrieval
                vec = [0.0] * 8
                for ch in text.lower():
                    vec[ord(ch) % 8] += 1.0
                data.append(SimpleNamespace(embedding=vec))
            return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=len(inputs) * 4))

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="\\boxed{us-east-1}"))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5),
        )


def test_drift_sim_runner_end_to_end_offline(tmp_path: Path) -> None:
    settings = Settings(
        llm_base_url="http://127.0.0.1:11434/v1",
        memdiff_cache_dir=tmp_path / "cache",
    )
    client = CachedLLMClient(settings=settings, client=_DeterministicFakeSDK())
    try:
        cfg = DriftSimConfig(seed=7, num_sessions=6, num_entities=3, num_durable_rules=2)
        out_dir = tmp_path / "run_001"
        summary = run_drift_sim_benchmark(
            config=cfg,
            policies=["naive_rag", "recency_rag"],
            output_dir=out_dir,
            top_k=3,
            client=client,
        )
        assert summary["benchmark"] == "drift_sim"
        assert set(summary["policies"].keys()) == {"naive_rag", "recency_rag"}
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "naive_rag_predictions.jsonl").exists()
        assert (out_dir / "recency_rag_predictions.jsonl").exists()
        assert (out_dir / "replay_recency_rag.json").exists()
    finally:
        client.close()


def test_select_balanced_questions_caps_per_type() -> None:
    qs = [
        LMEV2Question(
            id=f"q_{qtype}_{i}",
            domain="enterprise",
            environment="workarena",
            question_type=qtype,
            question="?",
            answer="a",
            eval_function="norm_phrase_set_match",
        )
        for qtype in ("static-environment", "procedure", "errors-gotchas")
        for i in range(5)
    ]
    picked = select_balanced_questions(qs, per_type=2)
    assert len(picked) == 6


def test_llm_extractor_and_reconciler_json_parsing(tmp_path: Path) -> None:
    responses = [
        # Extractor output
        (
            '```json\n{"beliefs": [{"type": "dynamic_state", "surface": "billing-api.region", '
            '"claim": "billing-api region is eu-central-1", "closed_set_items": [], '
            '"confidence": 0.9, "durability": 0.85, "step_index": 0, "excerpt": "eu-central-1"}]}\n```'
        ),
        # Reconciler output pointing to valid existing belief
        (
            '{"op": "SUPERSEDE", "target_belief_id": "b_old", '
            '"merged_claim": "billing-api region is eu-central-1", "reason": "region changed"}'
        ),
        # Reconciler output pointing to unknown belief -> safe fallback to ADD
        '{"op": "SUPERSEDE", "target_belief_id": "b_nonexistent", "reason": "bad id"}',
    ]

    class _SeqSDK:
        def __init__(self) -> None:
            self.idx = 0
            self.chat = SimpleNamespace(completions=self)
            self.embeddings = self

        def create(self, **_: Any) -> Any:
            text = responses[self.idx]
            self.idx += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
            )

    settings = Settings(llm_base_url="http://127.0.0.1:11435/v1", memdiff_cache_dir=tmp_path / "c")
    client = CachedLLMClient(settings=settings, client=_SeqSDK())
    try:
        extractor = make_llm_extractor(client, think=False)
        reconciler = make_llm_reconciler(client, think=False)

        extracted = extractor(
            {
                "id": "s002",
                "domain": "drift_sim",
                "outcome": "success",
                "states": [{"state_index": 0, "accessibility_tree": "billing-api region -> eu-central-1"}],
            },
            2,
        )
        assert len(extracted) == 1
        cand = extracted[0]
        assert cand.type == BeliefType.DYNAMIC_STATE
        assert cand.surface == "billing-api.region"

        # Fast-path ADD when no similar neighbors exist
        assert reconciler(cand, []).op == RevisionOp.ADD

        existing = Belief(
            id="b_old",
            type=BeliefType.DYNAMIC_STATE,
            surface="billing-api.region",
            claim="billing-api region is us-east-1",
            created_at_step=1,
            updated_at_step=1,
        )
        dec1 = reconciler(cand, [(existing, 0.92)])
        assert dec1.op == RevisionOp.SUPERSEDE
        assert dec1.target_belief_id == "b_old"

        # Invalid target id safely falls back to ADD
        dec2 = reconciler(cand, [(existing, 0.91)])
        assert dec2.op == RevisionOp.ADD
    finally:
        client.close()
