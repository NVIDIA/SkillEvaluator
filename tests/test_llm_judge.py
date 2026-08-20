# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM judge request-payload compatibility tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.provider_config import CHAT_CHEAP_OPENAI, CHAT_DEFAULT_OPENAI
from skillevaluator.tier3.eval_core import llm_judge


@pytest.mark.parametrize(
    "model",
    [
        CHAT_DEFAULT_OPENAI,
        f"openai/{CHAT_DEFAULT_OPENAI}",
        f"openai/openai/{CHAT_DEFAULT_OPENAI}",
    ],
)
def test_native_openai_gpt5_uses_max_completion_tokens_without_temperature(model: str) -> None:
    payload = llm_judge._chat_completion_payload(
        model=model,
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.25,
        provider="openai",
        request_url=llm_judge.OPENAI_CHAT_URL,
    )

    assert payload == {
        "model": model,
        "max_completion_tokens": 321,
        "messages": [{"role": "user", "content": "Judge this response"}],
    }
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.parametrize(
    "model",
    [
        CHAT_DEFAULT_OPENAI,
        CHAT_CHEAP_OPENAI,
        f"openai/{CHAT_DEFAULT_OPENAI}",
        f"openai/openai/{CHAT_DEFAULT_OPENAI}",
    ],
)
def test_gpt5_family_rejects_custom_temperature(model: str) -> None:
    assert not llm_judge._supports_custom_temperature(model)


def test_older_models_accept_custom_temperature() -> None:
    assert llm_judge._supports_custom_temperature("gpt-4.1-mini")
    assert llm_judge._supports_custom_temperature("claude-opus-4-6")
    assert llm_judge._supports_custom_temperature("claude-opus-4-20250514")
    assert llm_judge._supports_custom_temperature("claude-3-5-sonnet-20241022")


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-7",
        "claude-opus-4-8",
        "anthropic/claude-opus-4-8",
        "azure/anthropic/claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
        "bedrock/us.anthropic.claude-opus-4-8",
        "claude-opus-5",
        "anthropic/claude-opus-5",
        "us.anthropic.claude-opus-5",
        "bedrock/us.anthropic.claude-opus-5",
        "claude-sonnet-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "anthropic/claude-mythos-preview",
    ],
)
def test_newer_claude_models_reject_custom_temperature(model: str) -> None:
    assert not llm_judge._supports_custom_temperature(model)


def test_call_public_llm_uses_production_gpt5_payload_without_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

    with patch("openai.OpenAI", return_value=mock_openai):
        content, error = llm_judge.call_public_llm("Judge this response", max_tokens=4096, temperature=0.0)

    assert (content, error) == ("Done", None)
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_completion_tokens"] == 4096
    assert "temperature" not in call_kwargs


@pytest.mark.parametrize(
    ("provider", "request_url"),
    [
        ("nv_build", llm_judge.NVIDIA_BUILD_CHAT_URL),
        ("openai", "https://openai-compatible.example/v1/chat/completions"),
    ],
)
def test_non_native_gpt5_requests_keep_max_tokens(provider: str, request_url: str) -> None:
    payload = llm_judge._chat_completion_payload(
        model=CHAT_DEFAULT_OPENAI,
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
        provider=provider,
        request_url=request_url,
    )

    assert payload["max_tokens"] == 321
    assert "max_completion_tokens" not in payload
    assert "temperature" not in payload


def test_native_openai_non_gpt5_keeps_max_tokens() -> None:
    payload = llm_judge._chat_completion_payload(
        model="gpt-4.1-mini",
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
        provider="openai",
        request_url=llm_judge.OPENAI_CHAT_URL,
    )

    assert payload["max_tokens"] == 321
    assert payload["temperature"] == 0.0
    assert "max_completion_tokens" not in payload


def test_completion_token_payload_resolves_provider_and_url_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)

    payload = llm_judge._chat_completion_payload(
        model=CHAT_DEFAULT_OPENAI,
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
    )

    assert payload["max_completion_tokens"] == 321
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.parametrize(
    "request_url",
    [
        "https://api.openai.com/v1/chat/completions",
        "https://api.openai.com/v1/chat/completions/",
        "HTTPS://API.OPENAI.COM/v1/chat/completions",
        "https://api.openai.com:443/v1/chat/completions",
    ],
)
def test_native_openai_completion_token_url_accepts_only_canonical_variants(request_url: str) -> None:
    assert llm_judge._is_native_openai_chat_url("OPENAI", request_url)


@pytest.mark.parametrize(
    ("provider", "request_url"),
    [
        ("nv_build", "https://api.openai.com/v1/chat/completions"),
        ("openai-compatible", "https://api.openai.com/v1/chat/completions"),
        ("openai", "http://api.openai.com/v1/chat/completions"),
        ("openai", "https://api.openai.com.evil.example/v1/chat/completions"),
        ("openai", "https://user@api.openai.com/v1/chat/completions"),
        ("openai", "https://api.openai.com/v1/chat/completions?route=proxy"),
        ("openai", "https://api.openai.com/v1/chat/completions?"),
        ("openai", "https://api.openai.com/v1/chat/completions#fragment"),
        ("openai", "https://api.openai.com/v1/chat/completions#"),
        ("openai", "https://api.openai.com/v1/chat/completions;proxy"),
        ("openai", "https://api.openai.com/v1/chat/completions;"),
        ("openai", "https://api.openai.com/v1/chat/completionsbeta"),
        ("openai", "https://api.openai.com:444/v1/chat/completions"),
        ("openai", "https://api.openai.com:/v1/chat/completions"),
        ("openai", "https://api.openai.com:invalid/v1/chat/completions"),
        ("openai", "https://api.openai.com\r/v1/chat/completions"),
        ("openai", "https://api.openai.com\n/v1/chat/completions"),
        ("openai", "https://api.openai.com\t/v1/chat/completions"),
        ("openai", " https://api.openai.com/v1/chat/completions"),
        ("openai", "https://api.openai.com/v1/chat/completions "),
    ],
)
def test_deceptive_openai_urls_keep_max_tokens(provider: str, request_url: str) -> None:
    assert not llm_judge._is_native_openai_chat_url(provider, request_url)
