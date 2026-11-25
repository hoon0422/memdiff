"""Deterministic LME-V2 matchers (upstream port; signatures keep `**kwargs` for spec options)."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from memdiff.benchmarks.common.domain.answer import (
    DEFAULT_SEPARATORS,
    norm_phrase_set_match,
    normalize_phrase,
    split_phrases,
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


MATCHERS: dict[str, Callable[..., bool]] = {
    "norm_phrase_set_match": norm_phrase_set_match,
    "norm_phrase_set_match_ordered": norm_phrase_set_match_ordered,
    "mc_choice_match": mc_choice_match,
    "mc_choice_set_match": mc_choice_set_match,
}
