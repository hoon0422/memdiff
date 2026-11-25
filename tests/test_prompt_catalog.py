from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memdiff.adapters.llm.judges import build_judges
from memdiff.adapters.llm.lmev2_reader import LMEV2Reader
from memdiff.benchmarks.drift_sim.runner import _answer_drift_query
from memdiff.benchmarks.drift_sim.simulator import DriftQuery
from memdiff.benchmarks.lmev2.domain.dataset import LMEV2Question
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.settings import Settings
from memdiff.core.models import Belief, BeliefType
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.domain.messages import ChatCompletion, ChatMessage, ChatRequest
from memdiff.llm.domain.prompt import PromptError, PromptId
from memdiff.memory.domain.context import MemoryContextItem
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


class _RecordingChat:
    """A `ChatModel` that records what the adapter sends."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[ChatMessage] = []
        self.request: ChatRequest | None = None

    def complete(self, messages: list[ChatMessage], request: ChatRequest) -> ChatCompletion:
        self.messages, self.request = messages, request
        return ChatCompletion(text=self.reply)


@pytest.mark.parametrize(
    ("domain", "pid"),
    [
        ("enterprise", PromptId.LMEV2_READER_SYSTEM_ENTERPRISE),
        ("web", PromptId.LMEV2_READER_SYSTEM_WEB),
        ("unknown-domain", PromptId.LMEV2_READER_SYSTEM_ENTERPRISE),
    ],
)
def test_lmev2_reader_prompts_match_upstream_bytes(
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
    chat = _RecordingChat("\\boxed{a}")
    reader = LMEV2Reader(chat, catalog, model="m")
    context = [MemoryContextItem(type="text", value=v) for v in ("m1", "m2")]
    reader.answer(domain=domain, question=question, context=context, image_path=None, think=False)

    system, user = chat.messages
    assert system.content == catalog.render(pid)
    assert user.content == (
        "Retrieved Memory:\nm1\n\n---\n\nm2\n\nQuestion:\nWhat is on the page?\n\n"
        "Answer based strictly on your memory above, and place your final answer inside "
        "\\boxed{...}."
    )
    assert chat.request == ChatRequest(model="m", temperature=0.0, max_tokens=16384, think=False)


@pytest.mark.parametrize(
    ("name", "system_id", "user_id", "head"),
    [
        (
            "llm_abstention_checker",
            PromptId.JUDGE_ABSTENTION_SYSTEM,
            PromptId.JUDGE_ABSTENTION_USER,
            "Evaluate whether the model correctly handled a flawed-premise question.\n\n",
        ),
        (
            "llm_gotchas_checker",
            PromptId.JUDGE_GOTCHAS_SYSTEM,
            PromptId.JUDGE_GOTCHAS_USER,
            "Evaluate whether the model answer captures the gotcha insight.\n\n",
        ),
    ],
)
def test_lmev2_judge_prompts_match_upstream_bytes(
    catalog: PromptCatalog, name: str, system_id: PromptId, user_id: PromptId, head: str
) -> None:
    chat = _RecordingChat('{"label": 1, "reason": "ok"}')
    judge = build_judges(chat, catalog, model="judge-m")[name]
    assert judge.grade("full response", "reference", question="the question", parsed_answer="final")

    system, user = chat.messages
    assert system.content == catalog.render(system_id)
    assert user.content.startswith(
        head
        + "Question:\nthe question\n\n"
        + ("Reference answer (the expected flaw-aware conclusion):\n" if "abstention" in name
           else "Reference answer (insight points):\n")
        + "reference\n\nModel full response:\nfull response\n\n"
        "Model extracted final answer:\nfinal\n\nScoring rubric:\n"
    )
    assert user.content.endswith('Output JSON only:\n{"label": 0 or 1, "reason": "short rationale"}')
    assert chat.request == ChatRequest(model="judge-m", temperature=0.0, max_tokens=16384)


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
