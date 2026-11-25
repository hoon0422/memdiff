from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memdiff.benchmarks.drift_sim.runner import _answer_drift_query
from memdiff.benchmarks.drift_sim.simulator import DriftQuery
from memdiff.benchmarks.lmev2.loader import LMEV2Question
from memdiff.benchmarks.lmev2.metrics import llm_abstention_checker, llm_gotchas_checker
from memdiff.benchmarks.lmev2.runner import _build_reader_messages
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.settings import Settings
from memdiff.core.models import Belief, BeliefType
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.domain.prompt import PromptError, PromptId
from memdiff.policies.llm_ops import make_llm_extractor, make_llm_reconciler

REPO_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


class _RecordingClient:
    """Stands in for CachedLLMClient: records the messages the real code builds."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> str:
        self.messages = messages
        return self.reply


@pytest.fixture
def catalog() -> Iterator[PromptCatalog]:
    container = build_container(Settings(_env_file=None))
    try:
        yield container.get(PromptCatalog)
    finally:
        container.close()


def _belief(id_: str, surface: str, claim: str, closed: list[str] | None = None) -> Belief:
    return Belief(
        id=id_,
        type=BeliefType.DYNAMIC_STATE,
        surface=surface,
        claim=claim,
        closed_set_items=closed or [],
        confidence=0.9,
        durability=0.9,
        created_at_step=3,
        updated_at_step=3,
    )


def test_every_prompt_id_has_a_md_file_under_root_prompts() -> None:
    files = {p.relative_to(REPO_PROMPTS).as_posix() for p in REPO_PROMPTS.rglob("*.md")}
    assert files == {f"{pid.value}.md" for pid in PromptId}
    assert {p.split("/")[0] + "/" + p.split("/")[1] for p in files if "/" in p} >= {
        "memory/belief_extractor.system.md",
        "benchmarks/drift_sim",
        "benchmarks/lmev2",
    }


def test_extractor_and_reconciler_prompts_match_code(catalog: PromptCatalog) -> None:
    ext = _RecordingClient('{"beliefs": []}')
    make_llm_extractor(ext)({"id": "t1", "task": "x", "steps": []}, 0)  # type: ignore[arg-type]
    assert ext.messages[0]["content"] == catalog.render(PromptId.EXTRACTOR_SYSTEM)

    rec = _RecordingClient('{"op": "ADD", "reason": "r"}')
    cand = _belief("b_new", "svc.region", "region is eu-west-1")
    old = _belief("b_old", "svc.region", "region is us-east-1", ["us-east-1", "eu-west-1"])
    make_llm_reconciler(rec)(cand, [(old, 0.9)])  # type: ignore[arg-type]
    assert rec.messages[0]["content"] == catalog.render(PromptId.RECONCILER_SYSTEM)
    candidate = "- type=dynamic_state | surface=svc.region | step=3 | claim=region is eu-west-1"
    neighbors = (
        "- id=b_old | sim=0.900 | type=dynamic_state | surface=svc.region | step=3 | "
        "claim=region is us-east-1 | closed_set=['us-east-1', 'eu-west-1']"
    )
    assert rec.messages[1]["content"] == catalog.render(
        PromptId.RECONCILER_USER, candidate=candidate, neighbors=neighbors
    )
    assert rec.messages[1]["content"] == (
        f"NEW CANDIDATE BELIEF:\n{candidate}\n\nEXISTING ACTIVE BELIEFS:\n{neighbors}"
    )


def test_drift_reader_prompts_match_code(catalog: PromptCatalog) -> None:
    client = _RecordingClient("\\boxed{x}")
    query = DriftQuery(
        id="q1",
        query_type="dynamic_state",
        entity="svc",
        question="Which region is svc in?",
        gold_answer="eu-west-1",
        stale_answers=["us-east-1"],
    )
    _answer_drift_query(client, query, [{"type": "text", "value": "note A"}])  # type: ignore[arg-type]
    assert client.messages[0]["content"] == catalog.render(PromptId.DRIFT_READER_SYSTEM)
    assert client.messages[1]["content"] == catalog.render(
        PromptId.DRIFT_READER_USER, memory="note A", question="Which region is svc in?"
    )


@pytest.mark.parametrize(
    ("domain", "pid"),
    [
        ("enterprise", PromptId.LMEV2_READER_SYSTEM_ENTERPRISE),
        ("web", PromptId.LMEV2_READER_SYSTEM_WEB),
    ],
)
def test_lmev2_reader_prompts_match_code(
    catalog: PromptCatalog, domain: str, pid: PromptId
) -> None:
    question = LMEV2Question.model_validate(
        {
            "id": "q1",
            "domain": "enterprise",
            "environment": "servicenow",
            "question_type": "static",
            "question": "What is on the page?",
            "answer": "a",
            "eval_function": "x",
        }
    )
    msgs = _build_reader_messages(
        domain=domain,
        question=question,
        context_items=[{"type": "text", "value": "m1"}, {"type": "text", "value": "m2"}],
        data_root=Path("."),
    )
    assert msgs[0]["content"] == catalog.render(pid)
    assert msgs[1]["content"] == catalog.render(
        PromptId.LMEV2_READER_USER, memory="m1\n\n---\n\nm2", question="What is on the page?"
    )


@pytest.mark.parametrize(
    ("checker", "system_id", "user_id"),
    [
        (llm_abstention_checker, PromptId.JUDGE_ABSTENTION_SYSTEM, PromptId.JUDGE_ABSTENTION_USER),
        (llm_gotchas_checker, PromptId.JUDGE_GOTCHAS_SYSTEM, PromptId.JUDGE_GOTCHAS_USER),
    ],
)
def test_lmev2_judge_prompts_match_code(
    catalog: PromptCatalog, checker: Any, system_id: PromptId, user_id: PromptId
) -> None:
    client = _RecordingClient('{"label": 1, "reason": "ok"}')
    checker(
        "full response",
        "reference",
        question_text="the question",
        parsed_prediction="final",
        llm_client=client,
    )
    assert client.messages[0]["content"] == catalog.render(system_id)
    assert client.messages[1]["content"] == catalog.render(
        user_id,
        question="the question",
        reference="reference",
        response="full response",
        final_answer="final",
    )


def test_render_is_strict(catalog: PromptCatalog) -> None:
    with pytest.raises(PromptError):
        catalog.render(PromptId.DRIFT_READER_USER, memory="m")
    with pytest.raises(PromptError):
        catalog.render(PromptId.DRIFT_READER_USER, memory="m", question="q", extra="x")


def test_only_one_trailing_newline_is_stripped(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    shutil.copytree(REPO_PROMPTS, root)
    (root / "benchmarks/drift_sim/reader.system.md").write_text("fixed\n\n")
    container = build_container(Settings(_env_file=None, memdiff_prompts_dir=root))
    try:
        got = container.get(PromptCatalog).render(PromptId.DRIFT_READER_SYSTEM)
    finally:
        container.close()
    assert got == "fixed\n"


def _broken_prompts(tmp_path: Path) -> Path:
    root = tmp_path / "prompts"
    shutil.copytree(REPO_PROMPTS, root)
    return root


def test_container_fails_on_missing_prompt_file(tmp_path: Path) -> None:
    root = _broken_prompts(tmp_path)
    (root / "memory/belief_reconciler.user.md").unlink()
    with pytest.raises(PromptError, match=r"memory/belief_reconciler\.user.*missing"):
        build_container(Settings(_env_file=None, memdiff_prompts_dir=root))


def test_container_fails_on_missing_placeholder(tmp_path: Path) -> None:
    root = _broken_prompts(tmp_path)
    (root / "benchmarks/drift_sim/reader.user.md").write_text("Memory:\n$memory\n")
    with pytest.raises(PromptError, match=r"drift_sim/reader\.user.*missing.*question"):
        build_container(Settings(_env_file=None, memdiff_prompts_dir=root))


def test_container_fails_on_unknown_placeholder(tmp_path: Path) -> None:
    root = _broken_prompts(tmp_path)
    (root / "benchmarks/drift_sim/reader.user.md").write_text("$memory $question $bogus\n")
    with pytest.raises(PromptError, match=r"drift_sim/reader\.user.*unknown.*bogus"):
        build_container(Settings(_env_file=None, memdiff_prompts_dir=root))
