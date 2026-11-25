"""Preflight check for Ollama connectivity and required chat/embedding models."""

from __future__ import annotations

from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.settings import Settings, get_settings
from memdiff.llm.application.ports import ModelLister


def _model_present(target: str, installed: list[str]) -> bool:
    if target in installed:
        return True
    if ":" not in target and f"{target}:latest" in installed:
        return True
    return False


def run_check(settings: Settings, lister: ModelLister) -> int:
    print(f"Checking Ollama endpoint: {settings.ollama_root_url}")
    try:
        installed = lister.list_models()
    except Exception as exc:
        print(f"[ERROR] Could not reach Ollama at {settings.ollama_root_url}: {exc}")
        print("\nTo connect to a remote Ollama GPU host:")
        print("  1) Start headless Ollama server on the GPU host: `ollama serve`")
        print("  2) Open an SSH tunnel: `ssh -fN -L 11434:127.0.0.1:11434 <gpu-host>`")
        print("     (Or set OLLAMA_HOST=0.0.0.0 on the host and point LLM_BASE_URL=http://<gpu-host>:11434/v1)")
        return 1

    print(f"Connected! Installed models ({len(installed)}): {installed}")
    required = {
        "chat": settings.memdiff_chat_model,
        "embedding": settings.memdiff_embedding_model,
        "judge": settings.memdiff_judge_model,
    }
    missing = [m for m in dict.fromkeys(required.values()) if not _model_present(m, installed)]
    if missing:
        print("\n[MISSING MODELS] Run on your Ollama host:")
        for model in missing:
            print(f"  ollama pull {model}")
        return 2

    print("All configured models are installed and ready!")
    return 0


def main() -> int:
    settings = get_settings()
    container = build_container(settings)
    try:
        return run_check(settings, container.get(ModelLister))
    finally:
        container.close()
