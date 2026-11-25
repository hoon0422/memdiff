"""LLM judges for flawed-premise (abstention) and gotchas questions: catalog prompts + `ChatModel`."""

from __future__ import annotations

from typing import Any

from memdiff.adapters.llm.chat import ask
from memdiff.benchmarks.lmev2.domain.scoring.judgement import parse_binary_judgement
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel
from memdiff.llm.domain.prompt import PromptId


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
        self._model = model

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
        user_prompt = self._prompts.render(
            self._user,
            question=question,
            reference=ans,
            response=pred,
            final_answer=(parsed_answer or pred).strip(),
        )
        text = ask(self._chat, self._prompts.render(self._system), user_prompt, model=self._model)
        label, _ = parse_binary_judgement(text)
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
