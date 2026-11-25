"""Faithful port of official LongMemEval-V2 evaluation metrics and judge rubrics.

Source reference: xiaowu0162/LongMemEval-V2 `evaluation/qa_eval_metrics.py` & `evaluation/harness.py`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from memdiff.llm.openai_client import CachedOpenAIClient

DEFAULT_SEPARATORS: Sequence[str] = (",", ";")

CATEGORY_MAP: dict[str, str] = {
    "static-environment": "static",
    "static-environment-abs": "static-abs",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic-abs",
    "procedure": "procedure",
    "procedure-abs": "procedure-abs",
    "errors-gotchas": "gotchas",
}
NON_ABSTENTION_CATEGORIES = ["static", "dynamic", "procedure", "gotchas"]
ABSTENTION_CATEGORIES = ["static-abs", "dynamic-abs", "procedure-abs"]

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

_MULTI_SELECT_FILLER_WORDS = {
    "AND",
    "ANSWER",
    "ANSWERS",
    "CHOICE",
    "CHOICES",
    "FINAL",
    "LETTER",
    "LETTERS",
    "OPTION",
    "OPTIONS",
}


def normalize_phrase(
    text: str | None,
    *,
    lower: bool = True,
    normalize_hyphen: bool = True,
    strip_punct: bool = True,
) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if lower:
        text = text.lower()
    if normalize_hyphen:
        text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[,;]", " ", text)
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_phrases(
    text: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    **normalize_kwargs: bool,
) -> list[str]:
    if text is None:
        return []
    separator_list = list(separators)
    if not separator_list:
        normalized = normalize_phrase(text, **normalize_kwargs)
        return [normalized] if normalized else []
    pattern = "|".join(re.escape(sep) for sep in separator_list)
    parts = re.split(pattern, text)
    normalized_parts = [normalize_phrase(part, **normalize_kwargs) for part in parts]
    return [part for part in normalized_parts if part]


def norm_phrase_set_match(
    prediction: str | None,
    answer: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs: bool,
) -> bool:
    normalized_pred = normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    for phrase in set(answer_phrases):
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, normalized_pred) is None:
            return False
    return True


def norm_phrase_set_match_ordered(
    prediction: str | None,
    answer: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs: bool,
) -> bool:
    normalized_pred = normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    start = 0
    for phrase in answer_phrases:
        pattern = rf"\b{re.escape(phrase)}\b"
        match = re.search(pattern, normalized_pred[start:])
        if match is None:
            return False
        start += match.end()
    return True


def mc_choice_match(
    prediction: str | None,
    answer: str | None,
    *,
    strip_chars: str = ".",
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    if prediction is None or answer is None:
        return False
    pred_str = str(prediction)
    ans_str = str(answer)
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", pred_str.lower())
    candidate = boxed_match.group(1) if boxed_match else pred_str
    cleaned = re.sub(r"\b(choice|option)\b", "", candidate, flags=re.IGNORECASE)
    for ch in strip_chars:
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip().upper()
    expected = ans_str.strip().upper()
    if require_non_empty and (not cleaned or not expected):
        return False
    return cleaned == expected


def mc_choice_set_match(
    prediction: str | None,
    answer: str | None,
    *,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    def _letters(val: str | None) -> list[str]:
        if val is None:
            return []
        out: list[str] = []
        for chunk in re.findall(r"[A-Z]+", str(val).upper()):
            if chunk not in _MULTI_SELECT_FILLER_WORDS:
                out.extend(list(chunk))
        return out

    pred_letters = _letters(prediction)
    ans_letters = _letters(answer)
    if require_non_empty and (not pred_letters or not ans_letters):
        return False
    return set(pred_letters) == set(ans_letters)


def extract_boxed_answer(text: str) -> str:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    i = idx + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    parsed = "".join(out).strip()
    return parsed if parsed else text.strip()


def is_unknown(parsed_answer: str) -> bool:
    return parsed_answer.strip().lower() == "unknown"


def _parse_binary_judgement(text: str) -> tuple[int, str]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = payload.get("label")
            if label in {0, 1, "0", "1"}:
                return int(label), str(payload.get("reason", ""))
        except json.JSONDecodeError:
            pass
    label_match = re.search(r'"?label"?\s*[:=]\s*([01])', cleaned, flags=re.IGNORECASE)
    if label_match:
        return int(label_match.group(1)), cleaned
    raise ValueError(f"Could not parse binary LLM judgement: {cleaned!r}")


def llm_abstention_checker(
    prediction: str | None,
    answer: str | None,
    *,
    question_text: str = "",
    parsed_prediction: str | None = None,
    llm_client: CachedOpenAIClient | None = None,
    judge_model: str | None = None,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    pred = (prediction or "").strip()
    ans = (answer or "").strip()
    if require_non_empty and (not pred or not ans):
        return False
    client = llm_client or CachedOpenAIClient()
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
    raw = client.chat(
        [
            {"role": "system", "content": _ABSTENTION_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=judge_model,
        temperature=0.0,
        max_completion_tokens=16384,
    )
    label, _ = _parse_binary_judgement(raw)
    return label == 1


def llm_gotchas_checker(
    prediction: str | None,
    answer: str | None,
    *,
    question_text: str = "",
    parsed_prediction: str | None = None,
    llm_client: CachedOpenAIClient | None = None,
    judge_model: str | None = None,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    pred = (prediction or "").strip()
    ans = (answer or "").strip()
    if require_non_empty and (not pred or not ans):
        return False
    client = llm_client or CachedOpenAIClient()
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
    raw = client.chat(
        [
            {"role": "system", "content": _GOTCHAS_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=judge_model,
        temperature=0.0,
        max_completion_tokens=16384,
    )
    label, _ = _parse_binary_judgement(raw)
    return label == 1


_EVAL_REGISTRY: dict[str, Callable[..., bool]] = {
    "norm_phrase_set_match": norm_phrase_set_match,
    "norm_phrase_set_match_ordered": norm_phrase_set_match_ordered,
    "mc_choice_match": mc_choice_match,
    "mc_choice_set_match": mc_choice_set_match,
    "llm_abstention_checker": llm_abstention_checker,
    "llm_gotchas_checker": llm_gotchas_checker,
}


def _parse_option_value(key: str, value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if key in {"separators", "separator"}:
        if not value:
            return []
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return json.loads(stripped)
        return [ch for ch in value if not ch.isspace()]
    return value


def eval_from_spec(spec: str, prediction: str | None, answer: str | None, **overrides: Any) -> bool:
    parts = [p.strip() for p in spec.split("|") if p.strip()]
    if not parts:
        raise ValueError("Empty eval_function spec")
    name = parts[0]
    if name not in _EVAL_REGISTRY:
        raise ValueError(f"Unknown eval function: {name}")
    kwargs: dict[str, Any] = {}
    for part in parts[1:]:
        key, value = part.split("=", 1)
        kwargs[key.strip()] = _parse_option_value(key.strip(), value.strip())
    kwargs.update(overrides)
    return bool(_EVAL_REGISTRY[name](prediction, answer, **kwargs))


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate accuracy, wrong-answer rate, and unknown rate by category."""
    if not records:
        return {"overall": {"count_all_questions": 0}}

    def _mean(rows: list[dict[str, Any]]) -> float | None:
        return sum(1.0 if r["score_bool"] else 0.0 for r in rows) / len(rows) if rows else None

    def _breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"count": 0, "pct_correct": None, "pct_answered_wrong": None, "pct_unknown": None}
        unknown_n = sum(1 for r in rows if r["is_unknown"])
        correct_n = sum(1 for r in rows if r["score_bool"] and not r["is_unknown"])
        wrong_n = n - correct_n - unknown_n
        return {
            "count": n,
            "pct_correct": correct_n / n,
            "pct_answered_wrong": wrong_n / n,
            "pct_unknown": unknown_n / n,
        }

    non_abst = [r for r in records if not r.get("is_abstention_problem")]
    abst = [r for r in records if r.get("is_abstention_problem")]
    return {
        "overall": {
            "overall_full_set": _mean(records),
            "overall_non_abstention_only": _mean(non_abst),
            "overall_abstention_only": _mean(abst),
            "count_all_questions": len(records),
            "count_non_abstention": len(non_abst),
            "count_abstention": len(abst),
        },
        "non_abstention_by_category": {
            cat: _breakdown([r for r in non_abst if r["category"] == cat])
            for cat in NON_ABSTENTION_CATEGORIES
        },
        "abstention_by_category": {
            cat: _breakdown([r for r in abst if r["category"] == cat])
            for cat in ABSTENTION_CATEGORIES
        },
    }
