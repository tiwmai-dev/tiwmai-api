from openai import AsyncOpenAI

from app.core.config import Settings
from app.services.chat_service import ChatService
from app.services.llm_params import build_llm_extra_params


def test_openrouter_defaults_are_openai_compatible():
    settings = Settings(secret_key="test")

    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert hasattr(settings, "openrouter_api_key")
    assert not hasattr(settings, "litellm_api_key")


def test_chat_service_uses_openrouter_settings(monkeypatch):
    settings = Settings(
        secret_key="test",
        openrouter_api_key="or-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/test-model",
    )
    monkeypatch.setattr("app.services.chat_service.get_settings", lambda: settings)

    service = ChatService()

    assert isinstance(service._aclient, AsyncOpenAI)
    assert service.settings.openrouter_api_key == "or-test"
    assert service.settings.openrouter_model == "openai/test-model"


def test_llm_extra_params_keeps_reasoning_effort_provider_safe():
    assert build_llm_extra_params("openai/test", reasoning_effort="minimal") == {
        "reasoning_effort": "minimal"
    }
    assert build_llm_extra_params("anthropic/claude-3.5", reasoning_effort="minimal") == {}
