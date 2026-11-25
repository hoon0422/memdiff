from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memdiff.adapters.llm.judges import build_judges
from memdiff.adapters.prompts.fs_source import FilesystemPromptSource
from memdiff.benchmarks.lmev2.application.scorer import Scorer
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.domain.messages import ChatCompletion, ChatMessage, ChatRequest


class FakeChat:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[list[ChatMessage], ChatRequest]] = []

    def complete(self, messages: list[ChatMessage], request: ChatRequest) -> ChatCompletion:
        self.calls.append((messages, request))
        return ChatCompletion(text=self.reply)


def _scorer(reply: str, model: str = "m") -> tuple[Scorer, FakeChat]:
    chat = FakeChat(reply)
    prompts = PromptCatalog(FilesystemPromptSource(Settings(_env_file=None).memdiff_prompts_dir))
    return Scorer(build_judges(chat, prompts, model=model)), chat


def _q(answer: str = "ans") -> Any:
    return SimpleNamespace(question="Q?", answer=answer)


def test_matcher_gets_parsed_answer() -> None:
    scorer, chat = _scorer("")
    assert scorer.score("mc_choice_match", "long raw", "C", _q("C"))
    assert not chat.calls


def test_judge_gets_raw_response_and_pinned_call() -> None:
    scorer, chat = _scorer('{"label": 1, "reason": "r"}', model="judge-m")
    assert scorer.score("llm_gotchas_checker", "raw text", "parsed", _q())
    ((messages, request),) = chat.calls
    assert request == ChatRequest(model="judge-m", temperature=0.0, max_tokens=16384)
    assert request.think is None
    assert "Model full response:\nraw text" in messages[1].content
    assert "Model extracted final answer:\nparsed" in messages[1].content


def test_judge_skips_call_on_empty_input() -> None:
    scorer, chat = _scorer('{"label": 1}')
    assert not scorer.score("llm_abstention_checker", "", "p", _q())
    assert not chat.calls


def test_unparseable_judge_reply_raises() -> None:
    scorer, _ = _scorer("???")
    with pytest.raises(ValueError):
        scorer.score("llm_abstention_checker", "r", "p", _q())
