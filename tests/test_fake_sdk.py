from __future__ import annotations

from typing import Any

from fake_sdk import RecordingFakeSDK


def _chat(sdk: RecordingFakeSDK, system: str, user: str) -> str:
    resp = sdk.create(
        model="m",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return str(resp.choices[0].message.content)


def test_records_requests_in_order_and_tags_chat_roles() -> None:
    sdk = RecordingFakeSDK()
    sdk.create(model="e", input=["a", "b"])
    out = _chat(sdk, "You are a strict grader for gotchas-style insight questions.",
                "Reference answer (insight points):\nUse moderate risk\n\n"
                "Model full response:\nset moderate risk\n\nModel extracted final answer:\nx")
    assert [(r["seq"], r["kind"], r.get("role")) for r in sdk.requests] == [
        (0, "embedding", None),
        (1, "chat", "judge_gotchas"),
    ]
    assert sdk.requests[0]["request"]["input"] == ["a", "b"]
    assert '"label": 1' in out


def test_embeddings_are_deterministic_and_input_dependent() -> None:
    sdk = RecordingFakeSDK()
    first: Any = sdk.create(model="e", input=["alpha beta", "gamma"])
    again: Any = sdk.create(model="e", input=["alpha beta", "gamma"])
    assert [d.embedding for d in first.data] == [d.embedding for d in again.data]
    assert first.data[0].embedding != first.data[1].embedding


def test_reader_answers_from_retrieved_memory_and_judge_follows_overlap() -> None:
    sdk = RecordingFakeSDK()
    reader = "You are an operations assistant answering from memory."
    memory = "[STATE UPDATE] billing-api deployment_region changed from a to us-west-2 (replaces previous setting)."
    q = "What is `deployment_region` for `billing-api`?"
    assert "\\boxed{us-west-2}" in _chat(sdk, reader, f"Retrieved Memory:\n{memory}\n\nQuestion:\n{q}\n\nbye")
    assert "UNKNOWN" in _chat(sdk, reader, f"Retrieved Memory:\n(no retrieved memory)\n\nQuestion:\n{q}\n\nbye")


def test_unrecognized_system_prompt_fails_loudly() -> None:
    sdk = RecordingFakeSDK()
    try:
        _chat(sdk, "Something else", "hi")
    except AssertionError as exc:
        assert "Unrecognized system prompt" in str(exc)
    else:
        raise AssertionError("expected AssertionError")
