from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memdiff.benchmarks.lmev2.application.scorer import Scorer


class FakeJudge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return self.reply


def _q(answer: str = "ans") -> Any:
    return SimpleNamespace(question="Q?", answer=answer)


def test_matcher_gets_parsed_answer() -> None:
    judge = FakeJudge("")
    scorer = Scorer(judge, judge_model="m")
    assert scorer.score("mc_choice_match", "long raw", "C", _q("C"))
    assert not judge.calls


def test_judge_gets_raw_response_and_pinned_call() -> None:
    judge = FakeJudge('{"label": 1, "reason": "r"}')
    scorer = Scorer(judge, judge_model="judge-m")
    assert scorer.score("llm_gotchas_checker", "raw text", "parsed", _q())
    (call,) = judge.calls
    assert call["model"] == "judge-m"
    assert call["temperature"] == 0.0
    assert call["max_completion_tokens"] == 16384
    assert "think" not in call
    assert "Model full response:\nraw text" in call["messages"][1]["content"]
    assert "Model extracted final answer:\nparsed" in call["messages"][1]["content"]


def test_judge_skips_call_on_empty_input() -> None:
    judge = FakeJudge('{"label": 1}')
    assert not Scorer(judge, judge_model=None).score("llm_abstention_checker", "", "p", _q())
    assert not judge.calls


def test_unparseable_judge_reply_raises() -> None:
    with pytest.raises(ValueError):
        Scorer(FakeJudge("???"), judge_model=None).score("llm_abstention_checker", "r", "p", _q())
