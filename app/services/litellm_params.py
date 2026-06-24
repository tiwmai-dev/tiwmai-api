"""Helpers for provider-specific LiteLLM request parameters."""

from typing import Any, Dict, Optional


def supports_reasoning_effort(model_name: Optional[str]) -> bool:
    """Return whether LiteLLM reasoning_effort should be forwarded to a model."""
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return True

    # Anthropic's thinking budget has provider-specific constraints. LiteLLM can
    # map reasoning_effort=minimal to an invalid budget for Claude models.
    if normalized.startswith("anthropic/") or "claude" in normalized:
        return False

    return True


def supports_response_format(model_name: Optional[str]) -> bool:
    """Return whether provider structured response_format should be forwarded."""
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return True

    # DeepSeek currently rejects the json_schema response_format used by quiz
    # generation, while the prompts and parser already support JSON-only fallback.
    if normalized.startswith("deepseek/"):
        return False

    return True


def build_litellm_extra_params(
    model_name: Optional[str],
    *,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    extra_params: Dict[str, Any] = {}
    if reasoning_effort and supports_reasoning_effort(model_name):
        extra_params["reasoning_effort"] = reasoning_effort
    return extra_params
