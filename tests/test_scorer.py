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


def test_full_spec_strings_dispatch_to_the_deterministic_matchers() -> None:
    scorer, chat = _scorer("")

    def eval_from_spec(spec: str, prediction: str, answer: str) -> bool:
        return scorer.score(spec, prediction, prediction, _q(answer))

    assert eval_from_spec(
        "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true",
        "My Open Incidents, Incident Mobile, and Incident Portal",
        "Incident Mobile, Incident Portal, My Open Incidents",
    )
    assert eval_from_spec(
        "norm_phrase_set_match_ordered|lower=true|normalize_hyphen=true|strip_punct=true|separators=;|require_non_empty=true",
        "First open Reports, then click Problems",
        "Reports;Problems",
    )
    assert not eval_from_spec(
        "norm_phrase_set_match_ordered|lower=true|normalize_hyphen=true|strip_punct=true|separators=;|require_non_empty=true",
        "First open Problems, then Reports",
        "Reports;Problems",
    )
    assert eval_from_spec("mc_choice_match|require_non_empty=true", "Option C.", "C")
    assert eval_from_spec("mc_choice_set_match|require_non_empty=true", "Choices A and C", "CA")
    assert not chat.calls
