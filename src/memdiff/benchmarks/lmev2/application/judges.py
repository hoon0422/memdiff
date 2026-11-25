"""LLM judges for abstention and gotchas questions; the client is always injected."""

from __future__ import annotations

from typing import Any, Protocol

from memdiff.benchmarks.lmev2.domain.scoring.judgement import parse_binary_judgement


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None,
        temperature: float,
        max_completion_tokens: int,
    ) -> str: ...


_ABSTENTION_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for flawed-premise (abstention) questions. "
    "Judge whether a model answer correctly identifies that the question premise is wrong, "
    "consistent with the reference answer. "
    "If the model follows the flawed premise and gives a concrete answer under that premise, "
    "it must be graded 0. "
    "If the model's final answer is just UNKNOWN / cannot determine without identifying the flaw, grade 0. "
    "If the model is contradictory (both rejects premise and also gives a concrete premise-following answer), grade 0. "
    "Paraphrases are allowed when they preserve the same core flaw described by the reference answer."
)
_GOTCHAS_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for gotchas-style insight questions. "
    "The reference answer describes the key insight(s). "
    "Grade 1 if the model response includes at least one correct insight point from the reference answer "
    "(paraphrase allowed), and does not contradict any reference point. "
    "If the model's direction is wrong, or it contains contradictions against any reference point, grade 0. "
    "If the model gives multiple points, partial coverage is enough for 1 as long as no contradictions appear."
)


def llm_abstention_checker(
    prediction: str | None,
    answer: str | None,
    *,
    question_text: str = "",
    parsed_prediction: str | None = None,
    llm_client: ChatClient,
    judge_model: str | None = None,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    pred = (prediction or "").strip()
    ans = (answer or "").strip()
    if require_non_empty and (not pred or not ans):
        return False
    final_ans = (parsed_prediction or pred).strip()
    user_prompt = (
        "Evaluate whether the model correctly handled a flawed-premise question.\n\n"
        f"Question:\n{question_text}\n\n"
        f"Reference answer (the expected flaw-aware conclusion):\n{ans}\n\n"
        f"Model full response:\n{pred}\n\n"
        f"Model extracted final answer:\n{final_ans}\n\n"
        "Scoring rubric:\n"
        "- Label 1 only if the model clearly identifies the flawed premise and reaches a conclusion equivalent to the reference answer.\n"
        "- Label 0 if the model follows the flawed premise, gives a generic UNKNOWN without naming the flaw, or contradicts itself.\n\n"
        'Output JSON only:\n{"label": 0 or 1, "reason": "short rationale"}'
    )
    raw = llm_client.chat(
        [
            {"role": "system", "content": _ABSTENTION_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=judge_model,
        temperature=0.0,
        max_completion_tokens=16384,
    )
    label, _ = parse_binary_judgement(raw)
    return label == 1


def llm_gotchas_checker(
    prediction: str | None,
    answer: str | None,
    *,
    question_text: str = "",
    parsed_prediction: str | None = None,
    llm_client: ChatClient,
    judge_model: str | None = None,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    pred = (prediction or "").strip()
    ans = (answer or "").strip()
    if require_non_empty and (not pred or not ans):
        return False
    final_ans = (parsed_prediction or pred).strip()
    user_prompt = (
        "Evaluate whether the model answer captures the gotcha insight.\n\n"
        f"Question:\n{question_text}\n\n"
        f"Reference answer (insight points):\n{ans}\n\n"
        f"Model full response:\n{pred}\n\n"
        f"Model extracted final answer:\n{final_ans}\n\n"
        "Scoring rubric:\n"
        "- Label 1 if the model includes at least one correct insight point from the reference answer (paraphrase acceptable) without contradicting any reference point.\n"
        "- Label 0 if direction is wrong, contradictory, or generic without insight.\n\n"
        'Output JSON only:\n{"label": 0 or 1, "reason": "short rationale"}'
    )
    raw = llm_client.chat(
        [
            {"role": "system", "content": _GOTCHAS_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=judge_model,
        temperature=0.0,
        max_completion_tokens=16384,
    )
    label, _ = parse_binary_judgement(raw)
    return label == 1
