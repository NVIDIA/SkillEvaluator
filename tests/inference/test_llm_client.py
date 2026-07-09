# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.inference.client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.inference import LLMClient, LLMClientError, LLMVerdict


class TestLLMVerdict:
    def test_stores_fields(self) -> None:
        v = LLMVerdict(verdict="DUPLICATE", confidence=0.9, reasoning="Same content", suggestion="Remove one")
        assert v.verdict == "DUPLICATE"
        assert v.confidence == 0.9
        assert v.reasoning == "Same content"
        assert v.suggestion == "Remove one"


class TestLLMClientInit:
    def test_defaults_follow_selected_public_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

        client = LLMClient()

        assert client.model == "openai/gpt-oss-120b"
        assert client._client is None

    def test_custom_params(self) -> None:
        client = LLMClient(model="custom/model", base_url="https://custom.api", api_key="key123")
        assert client.model == "custom/model"
        assert client.base_url == "https://custom.api"
        assert client.api_key == "key123"


class TestLLMClientGetClient:
    def test_openai_provider_uses_public_openai_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_openai = MagicMock()

        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            client = LLMClient()
            assert client._get_client() is mock_openai

        mock_cls.assert_called_once_with(api_key="test-key", base_url="https://api.openai.com/v1")

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(LLMClientError, match="OPENAI_API_KEY"):
            LLMClient()._get_client()

    def test_constructs_openai_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            client = LLMClient()
            result = client._get_client()
        mock_cls.assert_called_once()
        assert result is mock_openai

    def test_lazy_caches_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            client = LLMClient()
            first = client._get_client()
            second = client._get_client()
        assert first is second
        mock_cls.assert_called_once()

    def test_explicit_api_key_used(self) -> None:
        mock_openai = MagicMock()
        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient(api_key="explicit-key")
            client._get_client()


class TestCompletions:
    def test_returns_message_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Hello world"))]
        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient()
            result = client.completions("system", "user")
        assert result == "Hello world"


class TestExtractJsonFromResponse:
    def test_parses_plain_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"key": "value"}'))
        ]
        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient()
            result = client.extract_json_from_response("system", "user")
        assert result == {"key": "value"}

    def test_strips_markdown_fences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='```json\n{"key": "value"}\n```'))
        ]
        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient()
            result = client.extract_json_from_response("system", "user")
        assert result == {"key": "value"}

    def test_invalid_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="not json at all"))
        ]
        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient()
            with pytest.raises(LLMClientError, match="invalid JSON"):
                client.extract_json_from_response("system", "user")
