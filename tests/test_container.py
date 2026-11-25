from __future__ import annotations

from memdiff.adapters.llm.openai_compat import OllamaModelLister
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.ports import ModelLister


def test_container_injects_settings_as_context_and_builds_model_lister() -> None:
    settings = Settings(_env_file=None, llm_base_url="http://example.invalid:1234/v1")
    container = build_container(settings)
    try:
        assert container.get(Settings) is settings
        lister = container.get(ModelLister)
        assert isinstance(lister, OllamaModelLister)
        assert lister._root_url == "http://example.invalid:1234"
    finally:
        container.close()
