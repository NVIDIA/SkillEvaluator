# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM judge request-payload compatibility tests."""

from __future__ import annotations

import pytest

from skillevaluator.tier3.eval_core import llm_judge


def test_native_openai_gpt5_uses_max_completion_tokens() -> None:
    payload = llm_judge._chat_completion_payload(
        model="gpt-5.4-mini",
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.25,
        provider="openai",
        request_url=llm_judge.OPENAI_CHAT_URL,
    )

    assert payload == {
        "model": "gpt-5.4-mini",
        "max_completion_tokens": 321,
        "messages": [{"role": "user", "content": "Judge this response"}],
        "temperature": 0.25,
    }
    assert "max_tokens" not in payload


@pytest.mark.parametrize(
    ("provider", "request_url"),
    [
        ("nv_build", llm_judge.NVIDIA_BUILD_CHAT_URL),
        ("openai", "https://openai-compatible.example/v1/chat/completions"),
    ],
)
def test_non_native_gpt5_requests_keep_max_tokens(provider: str, request_url: str) -> None:
    payload = llm_judge._chat_completion_payload(
        model="gpt-5.4-mini",
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
        provider=provider,
        request_url=request_url,
    )

    assert payload["max_tokens"] == 321
    assert "max_completion_tokens" not in payload


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
    assert "max_completion_tokens" not in payload


def test_completion_token_payload_resolves_provider_and_url_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)

    payload = llm_judge._chat_completion_payload(
        model="gpt-5.4-mini",
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
    )

    assert payload["max_completion_tokens"] == 321
    assert "max_tokens" not in payload


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
