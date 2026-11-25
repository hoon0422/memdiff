"""LLM judges for flawed-premise (abstention) and gotchas questions: catalog prompts + `ChatModel`."""

from __future__ import annotations

from typing import Any

from memdiff.benchmarks.lmev2.domain.scoring.judgement import parse_binary_judgement
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel
from memdiff.llm.domain.messages import ChatMessage, ChatRequest
from memdiff.llm.domain.prompt import PromptId

_MAX_COMPLETION_TOKENS = 16384


class LlmJudge:
    def __init__(
        self,
        chat: ChatModel,
        prompts: PromptCatalog,
        *,
        system: PromptId,
        user: PromptId,
        model: str,
    ) -> None:
        self._chat = chat
        self._prompts = prompts
        self._system = system
        self._user = user
        self._request = ChatRequest(model=model, temperature=0.0, max_tokens=_MAX_COMPLETION_TOKENS)

    def grade(
        self,
        response: str,
        reference: str,
        *,
        question: str,
        parsed_answer: str,
        require_non_empty: bool = True,
        **_: Any,
    ) -> bool:
        pred = (response or "").strip()
        ans = (reference or "").strip()
        if require_non_empty and (not pred or not ans):
            return False
        messages = [
            ChatMessage(role="system", content=self._prompts.render(self._system)),
            ChatMessage(
                role="user",
                content=self._prompts.render(
                    self._user,
                    question=question,
                    reference=ans,
                    response=pred,
                    final_answer=(parsed_answer or pred).strip(),
                ),
            ),
        ]
        label, _ = parse_binary_judgement(self._chat.complete(messages, self._request).text)
        return label == 1


def build_judges(chat: ChatModel, prompts: PromptCatalog, *, model: str) -> dict[str, LlmJudge]:
    """Judges by eval_function name."""
    return {
        "llm_abstention_checker": LlmJudge(
            chat,
            prompts,
            system=PromptId.JUDGE_ABSTENTION_SYSTEM,
            user=PromptId.JUDGE_ABSTENTION_USER,
            model=model,
        ),
        "llm_gotchas_checker": LlmJudge(
            chat,
            prompts,
            system=PromptId.JUDGE_GOTCHAS_SYSTEM,
            user=PromptId.JUDGE_GOTCHAS_USER,
            model=model,
        ),
    }
