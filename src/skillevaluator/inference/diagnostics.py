# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded LLM failure diagnostics that never echo prompts or provider bodies."""

from __future__ import annotations

import re

from skillevaluator.inference.types import LLMClientError

# Provider error bodies are untrusted. Only these known protocol values are
# useful to callers; arbitrary messages, URLs, IDs, and headers stay private.
_ERROR_CODES = frozenset(
    {
        "model_not_found",
        "model_retired",
        "model_deprecated",
        "invalid_api_key",
        "authentication_error",
        "permission_denied",
        "insufficient_quota",
        "rate_limit_exceeded",
        "rate_limit_error",
        "context_length_exceeded",
        "unsupported_parameter",
        "unsupported_value",
        "invalid_parameter",
        "invalid_request_error",
        "server_error",
        "overloaded_error",
    }
)
_ERROR_PARAMETERS = frozenset(
    {"model", "temperature", "max_tokens", "max_completion_tokens", "response_format", "messages", "stream", "top_p"}
)
_PROVIDERS = frozenset({"openai", "anthropic", "nv_build", "bedrock", "openai-compatible"})


def safe_llm_labels(provider: object, model: object) -> tuple[str, str]:
    """Keep configured identifiers readable without emitting URLs or control text."""
    safe_provider = provider if isinstance(provider, str) and provider in _PROVIDERS else "unresolved"
    safe_model = "unresolved"
    if (
        isinstance(model, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}", model)
        and "://" not in model
        and not model.startswith(("sk-", "nvapi-"))
    ):
        safe_model = model
    return safe_provider, safe_model


def llm_failure_diagnostic(exc: Exception) -> str:
    """Explain SDK failures without importing optional SDKs or formatting exceptions."""
    status = getattr(exc, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else None
        fields = []
        if isinstance(error, dict):
            for key, allowed in (("code", _ERROR_CODES), ("type", _ERROR_CODES), ("param", _ERROR_PARAMETERS)):
                value = error.get(key)
                if isinstance(value, str) and value in allowed:
                    fields.append(f"{key}={value}")
        context = f" ({', '.join(fields)})" if fields else ""
        if status in (401, 403):
            remedy = "Check the selected LLM provider credentials and model access."
        elif status in (404, 410):
            remedy = (
                "Model or endpoint unavailable. Check SKILL_EVAL_LLM_MODEL against the provider's available models."
            )
        elif status == 429:
            remedy = "The LLM provider rejected the request due to rate or quota limits. Check quota and retry later."
        elif status in (400, 422):
            remedy = "The LLM provider rejected the request. Check the selected model and supported request parameters."
        elif status >= 500:
            remedy = "The LLM provider could not complete the request. Retry later."
        else:
            remedy = "The LLM request failed. Check the selected provider and model configuration."
        return f"HTTP {status}{context}: {remedy}"

    # These names are shared by the supported SDKs; inspecting the hierarchy
    # also recognizes SDK subclasses without loading optional dependencies.
    kinds = {kind.__name__ for kind in type(exc).__mro__}
    if kinds & {"APITimeoutError", "TimeoutException", "TimeoutError"}:
        return "LLM request timed out. Check provider availability and retry."
    if kinds & {"APIConnectionError", "ConnectError", "ConnectionError"}:
        return "LLM connection failed. Check provider connectivity and endpoint configuration."
    if isinstance(exc, LLMClientError) or kinds & {"JSONDecodeError", "ValueError", "TypeError", "AttributeError"}:
        return (
            "LLM configuration or response was invalid. Check configuration and the model's structured-output support."
        )
    return "Unexpected LLM request error. Check the selected provider and model configuration, then retry."
