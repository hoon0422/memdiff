"""LME-V2 context pieces behind their ports: dataset rules, Reader image part, offline fetcher."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from memdiff.adapters.files.lmev2_dataset import JsonlLMEV2Dataset
from memdiff.adapters.http.lmev2_fetcher import HuggingFaceLMEV2Fetcher
from memdiff.adapters.llm.lmev2_reader import LMEV2Reader
from memdiff.adapters.prompts.fs_source import FilesystemPromptSource
from memdiff.benchmarks.lmev2.domain.dataset import (
    LMEV2Question,
    group_by_haystack,
    select_balanced_questions,
    select_questions,
    trajectory_from_record,
)
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.domain.messages import ChatCompletion, ChatMessage, ChatRequest
from memdiff.memory.application.formatting import format_state_chunk

FIXTURE = Path(__file__).parent / "fixtures" / "lmev2_synthetic"


def _q(qid: str, qtype: str = "static-environment", image: str | None = None) -> LMEV2Question:
    return LMEV2Question(
        id=qid,
        domain="enterprise",
        environment="servicenow",
        question_type=qtype,
        question=f"question {qid}",
        image=image,
        answer="a",
        eval_function="mc_choice_match",
    )


def test_balanced_selection_is_per_sorted_type_and_wins_over_limit() -> None:
    qs = [_q("a1", "b"), _q("a2", "b"), _q("a3", "a"), _q("a4", "a")]
    assert [q.id for q in select_balanced_questions(qs, per_type=1)] == ["a3", "a1"]
    assert [q.id for q in select_questions(qs, balanced_per_type=1, limit=3)] == ["a3", "a1"]
    assert [q.id for q in select_questions(qs, balanced_per_type=None, limit=3)] == [
        "a1",
        "a2",
        "a3",
    ]


def test_groups_follow_first_occurrence_and_missing_haystack_id_raises() -> None:
    haystack = {"q1": ["t1", "t2"], "q2": ["t2", "t1"], "q3": ["t1", "t2"]}
    groups = group_by_haystack([_q("q1"), _q("q2"), _q("q3")], haystack)
    assert [(order, [q.id for q in qs]) for order, qs in groups] == [
        (("t1", "t2"), ["q1", "q3"]),
        (("t2", "t1"), ["q2"]),
    ]
    with pytest.raises(KeyError):
        group_by_haystack([_q("missing")], haystack)


def test_null_outcome_and_tree_render_like_the_legacy_raw_dict_policies() -> None:
    record = {
        "id": "t1",
        "goal": "g",
        "outcome": None,
        "states": [{"state_index": 3, "url": "u", "action": None, "accessibility_tree": None}],
    }
    traj = trajectory_from_record(record)
    # legacy: str(None) for a present-but-null key; a missing outcome stays "unknown"
    assert format_state_chunk(traj, traj.states[0], 2400) == (
        "[Trajectory t1 | outcome=None | step=3]\nGoal: g\nURL: u\nAction: <none>\nNone"
    )
    assert trajectory_from_record({"id": "t2"}).outcome == "unknown"


def test_dataset_adapter_reads_the_fixture_and_resolves_images() -> None:
    ds = JsonlLMEV2Dataset(FIXTURE)
    questions = ds.load_questions(domain="enterprise")
    assert len(questions) == 4
    assert ds.load_questions(domain="web") == []
    haystack = ds.load_haystack()
    assert haystack["q_gotcha_01"] == ["traj_b", "traj_a"]
    assert set(ds.load_trajectories(allowed_ids={"traj_a"})) == {"traj_a"}
    (dynamic,) = [q for q in questions if q.image]
    assert ds.image_path(dynamic) == FIXTURE / "screenshots/q_dynamic_01.png"
    assert ds.image_path(_q("x", image="screenshots/absent.png")) is None
    assert ds.image_path(_q("x")) is None


def test_dataset_adapter_reports_missing_files(tmp_path: Path) -> None:
    ds = JsonlLMEV2Dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_lmev2_small.py"):
        ds.load_questions()
    with pytest.raises(FileNotFoundError, match="Missing trajectories file"):
        ds.load_trajectories()
    (tmp_path / "trajectories.jsonl").write_text('{"id": "t9"}\n\n', encoding="utf-8")
    assert set(ds.load_trajectories()) == {"t9"}  # falls back to the full file


class _Chat:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def complete(self, messages: list[ChatMessage], request: ChatRequest) -> ChatCompletion:
        self.messages = messages
        return ChatCompletion(text="\\boxed{x}")


def test_reader_sends_the_screenshot_as_a_data_url_part(tmp_path: Path) -> None:
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG-bytes")
    chat = _Chat()
    prompts = PromptCatalog(FilesystemPromptSource(Settings(_env_file=None).memdiff_prompts_dir))
    LMEV2Reader(chat, prompts, model="m").answer(
        domain="enterprise", question=_q("q"), context=[], image_path=img
    )
    parts = chat.messages[1].content
    assert isinstance(parts, list) and parts[0]["type"] == "text"
    assert "(no retrieved memory)" in parts[0]["text"]
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"\x89PNG-bytes").decode()},
    }


# ── fetcher against a fake HTTP transport ────────────────────────────────────────────────
_QUESTIONS = [
    {"id": "q1", "domain": "enterprise", "image": "screenshots/q1.png"},
    {"id": "q2", "domain": "web", "image": None},
]
_REMOTE: dict[str, bytes] = {
    "SCHEMA.md": b"schema",
    "DATA_CARD.md": b"card",
    "questions.jsonl": "".join(json.dumps(q) + "\n" for q in _QUESTIONS).encode(),
    "haystacks/lme_v2_small.json": json.dumps({"q1": ["t1"], "q2": ["t2"]}).encode(),
    "screenshots/q1.png": b"png",
    "trajectories.jsonl": b'{"id": "t1"}\n{"id": "t2"}\n{"id": "t3"}\n\n',
}


def _fetcher(hits: list[str]) -> HuggingFaceLMEV2Fetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        rel = request.url.path.removeprefix("/ds/")
        hits.append(rel)
        return httpx.Response(200, content=_REMOTE[rel])

    return HuggingFaceLMEV2Fetcher(
        httpx.Client(transport=httpx.MockTransport(handler)), base_url="https://example.invalid/ds"
    )


def test_fetch_keeps_the_original_file_layout_and_summary(tmp_path: Path) -> None:
    hits: list[str] = []
    summary: dict[str, Any] = _fetcher(hits).fetch(
        data_root=tmp_path, domain="enterprise", include_question_screenshots=True
    )
    assert list(summary.items()) == [
        ("questions_selected", 1),
        ("trajectories_needed", 1),
        ("trajectories_saved", 1),
        ("question_screenshots_checked", 1),
    ]
    assert (tmp_path / "screenshots/q1.png").read_bytes() == b"png"
    assert (tmp_path / "trajectories_small_enterprise.jsonl").read_text() == '{"id": "t1"}\n'
    alias = tmp_path / "trajectories_small.jsonl"
    assert alias.is_symlink() and str(alias.readlink()) == "trajectories_small_enterprise.jsonl"

    # second run: nothing is downloaded again, the saved trajectory count is re-read
    hits.clear()
    again = _fetcher(hits).fetch(
        data_root=tmp_path, domain="enterprise", include_question_screenshots=False
    )
    assert hits == []
    assert again["trajectories_saved"] == 1 and again["question_screenshots_checked"] == 0


def test_fetch_all_domains_keeps_every_needed_trajectory(tmp_path: Path) -> None:
    summary = _fetcher([]).fetch(data_root=tmp_path, domain="all", include_question_screenshots=False)
    assert summary["questions_selected"] == 2 and summary["trajectories_saved"] == 2
