# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLMClient transparent retry and rate limit handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from skillevaluator.inference.client import LLMClient


class MockRateLimitError(Exception):
    """Mock rate limit error with HTTP status 429 and headers."""

    def __init__(self, message: str = "Rate limited", retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = 429
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = MagicMock(headers=headers)


class MockAuthError(Exception):
    """Mock auth error with HTTP status 401."""

    def __init__(self, message: str = "Unauthorized") -> None:
        super().__init__(message)
        self.status_code = 401


def test_llm_client_retries_on_rate_limit_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LLMClient.completions retries on RateLimitError and succeeds on subsequent attempt."""
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    mock_client = MagicMock()
    calls = 0

    def mock_create(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MockRateLimitError("Rate limit exceeded")
        choice = MagicMock()
        choice.message.content = "recovered response"
        return MagicMock(choices=[choice])

    mock_client.chat.completions.create = mock_create

    client = LLMClient(
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        max_retries=3,
        retry_base_delay=0.1,
    )
    client._client = mock_client

    result = client.completions("System prompt", "User prompt")
    assert result == "recovered response"
    assert calls == 2
    assert len(slept) == 1


def test_llm_client_fails_fast_on_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LLMClient.completions raises immediately without retry on AuthenticationError."""
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    mock_client = MagicMock()
    calls = 0

    def mock_create(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise MockAuthError("Invalid credentials")

    mock_client.chat.completions.create = mock_create

    client = LLMClient(
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        max_retries=3,
        retry_base_delay=0.1,
    )
    client._client = mock_client

    with pytest.raises(MockAuthError):
        client.completions("System prompt", "User prompt")

    assert calls == 1
    assert len(slept) == 0


def test_llm_client_exhausts_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LLMClient.completions exhausts retries and raises when 429 persists."""
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    mock_client = MagicMock()
    calls = 0

    def mock_create(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise MockRateLimitError("Quota exhausted")

    mock_client.chat.completions.create = mock_create

    client = LLMClient(
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        max_retries=2,
        retry_base_delay=0.1,
    )
    client._client = mock_client

    with pytest.raises(MockRateLimitError):
        client.completions("System prompt", "User prompt")

    assert calls == 3  # 1 initial + 2 retries
    assert len(slept) == 2


def test_llm_client_respects_custom_retry_params() -> None:
    """Verify LLMClient stores and exposes custom retry configuration parameters."""
    client = LLMClient(
        model="gpt-5.6-sol",
        api_key="test-key",
        max_retries=5,
        retry_base_delay=2.5,
        retry_max_delay=45.0,
    )
    assert client.max_retries == 5
    assert client.retry_base_delay == 2.5
    assert client.retry_max_delay == 45.0


@pytest.mark.parametrize(("max_retries", "expected_requests"), [(0, 1), (3, 4)])
def test_openai_transport_level_request_count(
    monkeypatch: pytest.MonkeyPatch,
    max_retries: int,
    expected_requests: int,
) -> None:
    """Verify OpenAI SDK executes exactly expected number of requests on rate limits."""
    import openai

    monkeypatch.setattr("time.sleep", lambda _: None)
    request_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(429, json={"error": {"message": "Rate limit reached", "type": "rate_limit_error"}})

    client = LLMClient(
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
        api_key="test-openai-key",
        max_retries=max_retries,
        retry_base_delay=0.01,
        http_client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )
    with pytest.raises(openai.RateLimitError):
        client.completions("system", "user")

    assert request_count == expected_requests


@pytest.mark.parametrize(("max_retries", "expected_requests"), [(0, 1), (3, 4)])
def test_anthropic_transport_level_request_count(
    monkeypatch: pytest.MonkeyPatch,
    max_retries: int,
    expected_requests: int,
) -> None:
    """Verify Anthropic SDK executes exactly expected number of requests on rate limits."""
    import anthropic

    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    request_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}},
        )

    client = LLMClient(
        model="claude-3-opus",
        max_retries=max_retries,
        retry_base_delay=0.01,
        http_client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )
    with pytest.raises(anthropic.RateLimitError):
        client.completions("system", "user")

    assert request_count == expected_requests
