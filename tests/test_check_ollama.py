from __future__ import annotations

import pytest

from memdiff.bootstrap.settings import Settings
from memdiff.presentation.cli.check_ollama import run_check


class FakeModelLister:
    def __init__(self, models: list[str] | None = None, error: Exception | None = None) -> None:
        self.models = models or []
        self.error = error

    def list_models(self) -> list[str]:
        if self.error:
            raise self.error
        return self.models


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="http://127.0.0.1:11434/v1",
        memdiff_chat_model="chat:1",
        memdiff_embedding_model="embed",
        memdiff_judge_model="chat:1",
    )


def test_all_models_installed_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_check(_settings(), FakeModelLister(["chat:1", "embed:latest"]))

    assert code == 0
    assert capsys.readouterr().out == (
        "Checking Ollama endpoint: http://127.0.0.1:11434\n"
        "Connected! Installed models (2): ['chat:1', 'embed:latest']\n"
        "All configured models are installed and ready!\n"
    )


def test_missing_models_exit_2_and_list_pull_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_check(_settings(), FakeModelLister(["chat:1"]))

    assert code == 2
    out = capsys.readouterr().out
    assert "\n[MISSING MODELS] Run on your Ollama host:\n  ollama pull embed\n" in out
    assert "chat:1\n" not in out.split("[MISSING MODELS]")[1]


def test_unreachable_exits_1_with_tunnel_hint(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_check(_settings(), FakeModelLister(error=ConnectionError("boom")))

    assert code == 1
    out = capsys.readouterr().out
    assert "[ERROR] Could not reach Ollama at http://127.0.0.1:11434: boom" in out
    assert "ssh -fN -L 11434:127.0.0.1:11434 <gpu-host>" in out
