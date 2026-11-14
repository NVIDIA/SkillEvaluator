# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.inference.client."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from skillevaluator.constants import RUBRIC_CRITERIA
from skillevaluator.inference import (
    EmptyLLMResponseError,
    FindingVerifier,
    LLMClient,
    LLMClientError,
    LLMClientRefusedError,
    LLMClientTruncatedError,
    LLMVerdict,
)
from skillevaluator.inference import client as client_mod
from skillevaluator.inference.client import _is_native_openai_endpoint, _token_limit_kwargs
from skillevaluator.inference.json_span import MAX_JSON_NESTING
from skillevaluator.provider_config import (
    ANTHROPIC_REFUSAL_FALLBACK_MODEL,
    ANTHROPIC_SERVER_SIDE_FALLBACK_BETA,
    CHAT_DEFAULT_ANTHROPIC,
    CHAT_DEFAULT_BEDROCK,
    CHAT_DEFAULT_GATEWAY,
    CHAT_DEFAULT_OPENAI,
    MAX_COMPLETION_TOKENS,
    OPENAI_BASE_URL,
    PUBLIC_NVIDIA_BUILD_BASE_URL,
    REASONING_MAX_COMPLETION_TOKENS,
    ProviderConfig,
)
from skillevaluator.validators.rubric_eval import RUBRIC_JSON_SCHEMA, RubricJudge


def _gpt5_config(
    *,
    provider: str = "openai",
    base_url: str = OPENAI_BASE_URL,
    model: str = "gpt-5.4-mini",
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key="test-key",
        base_url=base_url,
        litellm_model=f"openai/{model}",
    )


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
        monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)

        client = LLMClient()

        assert client.model == "nvidia/nemotron-3-super-120b-a12b"
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
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        mock_openai = MagicMock()

        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            client = LLMClient()
            assert client._get_client() is mock_openai

        mock_cls.assert_called_once_with(api_key="test-key", base_url="https://api.openai.com/v1", max_retries=0)

    @pytest.mark.parametrize(
        ("provider", "key_env", "sdk_class"),
        [("openai", "OPENAI_API_KEY", "openai.OpenAI"), ("anthropic", "ANTHROPIC_API_KEY", "anthropic.Anthropic")],
    )
    def test_explicit_timeout_reaches_the_sdk(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, key_env: str, sdk_class: str
    ) -> None:
        # The SDKs replace an http_client timeout equal to httpx's 5-second
        # default with their own 600 seconds, so the bound must be explicit.
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
        monkeypatch.setenv(key_env, "test-key")
        for name in ("SKILL_EVAL_LLM_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        http_client = object()

        with patch(sdk_class, return_value=MagicMock()) as mock_cls:
            LLMClient(http_client=http_client, timeout=5.0)._get_client()

        assert mock_cls.call_args.kwargs["timeout"] == 5.0
        assert mock_cls.call_args.kwargs["http_client"] is http_client

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

    def test_explicit_api_key_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        mock_openai = MagicMock()
        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient(api_key="explicit-key")
            client._get_client()

    @pytest.mark.parametrize(
        ("explicit_base_url", "ambient_base_url", "expected_base_url", "expected_provider"),
        [
            (None, None, OPENAI_BASE_URL, "openai"),
            (OPENAI_BASE_URL, "https://ambient.example/v1", OPENAI_BASE_URL, "openai"),
            (None, "HTTPS://API.OPENAI.COM:443/v1/", "HTTPS://API.OPENAI.COM:443/v1/", "openai"),
            (
                "https://explicit.example/v1",
                OPENAI_BASE_URL,
                "https://explicit.example/v1",
                "openai-compatible",
            ),
            (None, "https://ambient.example/v1", "https://ambient.example/v1", "openai-compatible"),
            (None, "", "", "openai-compatible"),
        ],
    )
    def test_explicit_credentials_classify_effective_endpoint_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        explicit_base_url: str | None,
        ambient_base_url: str | None,
        expected_base_url: str,
        expected_provider: str,
    ) -> None:
        if ambient_base_url is None:
            monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("OPENAI_BASE_URL", ambient_base_url)

        config = LLMClient(
            model="gpt-5.4-mini",
            api_key="test-key",
            base_url=explicit_base_url,
        )._resolved_config()

        assert (config.base_url, config.provider) == (expected_base_url, expected_provider)

    @pytest.mark.parametrize(
        "base_url",
        [
            pytest.param("https://api.openai。com/v1", id="idna-dot-host"),
            pytest.param("https://api.openai.com:0443/v1", id="zero-padded-default-port"),
            pytest.param(f"{OPENAI_BASE_URL}#fragment", id="fragment-stripped-by-sdk"),
        ],
    )
    def test_rejects_noncanonical_aliases_normalized_to_native_openai_before_request(self, base_url: str) -> None:
        # This exercises SDK URL normalization without using a host proxy or making requests.
        transport = httpx.MockTransport(lambda _request: pytest.fail("Endpoint validation must precede requests"))
        with httpx.Client(transport=transport, trust_env=False) as http_client:
            client = LLMClient(
                model="gpt-5.4-mini",
                api_key="test-key",
                base_url=base_url,
                max_tokens=512,
                http_client=http_client,
            )

            with pytest.raises(LLMClientError, match="noncanonical alias for the native OpenAI endpoint"):
                client._get_client()

            assert client._client is None

    @pytest.mark.parametrize(
        ("provider", "api_key_env", "base_url_env", "base_url"),
        [
            ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", f"{OPENAI_BASE_URL}#fragment"),
            (
                "openai-compatible",
                "SKILL_EVAL_LLM_API_KEY",
                "SKILL_EVAL_LLM_BASE_URL",
                "https://api.openai.com:0443/v1",
            ),
        ],
    )
    def test_rejects_ambient_provider_aliases_normalized_to_native_openai(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        api_key_env: str,
        base_url_env: str,
        base_url: str,
    ) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
        monkeypatch.setenv(api_key_env, "test-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "gpt-5.4-mini")
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv(base_url_env, base_url)
        client = LLMClient(max_tokens=512)

        with pytest.raises(LLMClientError, match="noncanonical alias for the native OpenAI endpoint"):
            client._get_client()

        assert client._client is None

    def test_nvidia_build_ambient_base_url_cannot_redirect_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "https://api.openai。com/v1")
        mock_openai = MagicMock()

        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            assert LLMClient()._get_client() is mock_openai

        mock_cls.assert_called_once_with(api_key="test-key", base_url=PUBLIC_NVIDIA_BUILD_BASE_URL, max_retries=0)

    def test_accepted_canonical_openai_url_constructs_real_sdk_client(self) -> None:
        client = LLMClient(
            model="gpt-5.4-mini",
            api_key="test-key",
            base_url="HTTPS://API.OPENAI.COM:443/v1/",
            max_tokens=512,
        )

        sdk_client = client._get_client()

        assert client._client is sdk_client
        sdk_client.close()


class TestNativeOpenAIEndpoint:
    @pytest.mark.parametrize("provider", ["nv_build", "openai-compatible", "anthropic", "bedrock", " openai"])
    def test_requires_openai_provider_intent(self, provider: str) -> None:
        config = _gpt5_config(provider=provider)

        assert _is_native_openai_endpoint(config) is False
        assert _token_limit_kwargs(config, 512) == {"max_tokens": REASONING_MAX_COMPLETION_TOKENS}

    @pytest.mark.parametrize(
        "model",
        [
            CHAT_DEFAULT_OPENAI,
            f"openai/{CHAT_DEFAULT_OPENAI}",
            f"openai/openai/{CHAT_DEFAULT_OPENAI}",
        ],
    )
    def test_provider_prefixed_gpt5_models_use_completion_tokens(self, model: str) -> None:
        config = _gpt5_config(model=model)

        assert _token_limit_kwargs(config, 512) == {"max_completion_tokens": REASONING_MAX_COMPLETION_TOKENS}

    @pytest.mark.parametrize(
        "base_url",
        [
            OPENAI_BASE_URL,
            f"{OPENAI_BASE_URL}/",
            "HTTPS://API.OPENAI.COM/v1",
            "https://api.openai.com:443/v1",
            "HTTPS://API.OPENAI.COM:443/v1/",
        ],
    )
    def test_accepts_only_canonical_openai_url_forms(self, base_url: str) -> None:
        config = _gpt5_config(provider="OPENAI", base_url=base_url)

        assert _is_native_openai_endpoint(config) is True
        assert _token_limit_kwargs(config, 512) == {"max_completion_tokens": REASONING_MAX_COMPLETION_TOKENS}

    @pytest.mark.parametrize(
        "base_url",
        [
            pytest.param(f" {OPENAI_BASE_URL}", id="leading-space"),
            pytest.param(f"{OPENAI_BASE_URL} ", id="trailing-space"),
            pytest.param(f"\t{OPENAI_BASE_URL}", id="leading-tab"),
            pytest.param(f"{OPENAI_BASE_URL}\t", id="trailing-tab"),
            pytest.param(f"\r{OPENAI_BASE_URL}", id="leading-cr"),
            pytest.param(f"{OPENAI_BASE_URL}\r", id="trailing-cr"),
            pytest.param(f"\n{OPENAI_BASE_URL}", id="leading-lf"),
            pytest.param(f"{OPENAI_BASE_URL}\n", id="trailing-lf"),
            pytest.param(f"\f{OPENAI_BASE_URL}", id="leading-form-feed"),
            pytest.param(f"{OPENAI_BASE_URL}\v", id="trailing-vertical-tab"),
            pytest.param("https://api.openai.com/v\r1", id="embedded-cr"),
            pytest.param("https://api.openai.com/v\n1", id="embedded-lf"),
            pytest.param("https://api.openai.com/v\t1", id="embedded-tab"),
            pytest.param(f"{OPENAI_BASE_URL}\x00", id="nul-control"),
            pytest.param(f"{OPENAI_BASE_URL}\x1f", id="unit-separator-control"),
            pytest.param(f"{OPENAI_BASE_URL}\x7f", id="delete-control"),
            pytest.param("https://api.openai.com.evil.test/v1", id="suffix-host"),
            pytest.param("https://api.openai.com%2eevil.test/v1", id="percent-dot-host"),
            pytest.param("https://%61pi.openai.com/v1", id="percent-host"),
            pytest.param("https://api.openai.com\\@evil.test/v1", id="backslash-host"),
            pytest.param("https://api.openai.com./v1", id="trailing-dot-host"),
            pytest.param("https://api.openai。com/v1", id="idna-dot-host"),
            pytest.param("https://api.open\u0430i.com/v1", id="unicode-lookalike-host"),
            pytest.param("https://user@api.openai.com/v1", id="userinfo"),
            pytest.param("https://user:pass@api.openai.com/v1", id="password"),
            pytest.param("https://api.openai.com@evil.test/v1", id="userinfo-host-deception"),
            pytest.param(f"{OPENAI_BASE_URL}?route=proxy", id="query"),
            pytest.param(f"{OPENAI_BASE_URL}?", id="empty-query"),
            pytest.param(f"{OPENAI_BASE_URL}#fragment", id="fragment"),
            pytest.param(f"{OPENAI_BASE_URL}#", id="empty-fragment"),
            pytest.param(f"{OPENAI_BASE_URL};transport=proxy", id="params"),
            pytest.param(f"{OPENAI_BASE_URL}/chat/completions", id="other-path"),
            pytest.param("https://api.openai.com/v1beta", id="path-prefix"),
            pytest.param("https://api.openai.com/v%31", id="percent-path"),
            pytest.param("http://api.openai.com/v1", id="http"),
            pytest.param("https://api.openai.com:444/v1", id="other-port"),
            pytest.param("https://api.openai.com:invalid/v1", id="malformed-port"),
            pytest.param("https://api.openai.com:/v1", id="empty-port"),
            pytest.param("https://api.openai.com:0443/v1", id="noncanonical-port-spelling"),
            pytest.param("https://api.openai.com:65536/v1", id="out-of-range-port"),
        ],
    )
    def test_noncanonical_raw_url_is_not_classified_as_native_openai(self, base_url: str) -> None:
        config = _gpt5_config(base_url=base_url)

        assert _is_native_openai_endpoint(config) is False
        assert _token_limit_kwargs(config, 512) == {"max_tokens": REASONING_MAX_COMPLETION_TOKENS}


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

    def test_openai_compatible_empty_response_uses_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=""))]

        with (
            patch("openai.OpenAI", return_value=mock_openai),
            pytest.raises(EmptyLLMResponseError, match="empty response"),
        ):
            LLMClient().completions("system", "user")

    def test_anthropic_empty_response_uses_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value.content = []

        with (
            patch("anthropic.Anthropic", return_value=mock_anthropic),
            pytest.raises(EmptyLLMResponseError, match="empty response"),
        ):
            LLMClient().completions("system", "user")

    def test_bedrock_empty_response_uses_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])

        with (
            patch("litellm.completion", return_value=response),
            pytest.raises(EmptyLLMResponseError, match="empty response"),
        ):
            LLMClient().completions("system", "user")

    def test_none_temperature_is_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class NoTemperatureClient(LLMClient):
            default_temperature = None

        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            NoTemperatureClient(model="gpt-4.1-mini", api_key="test-key").completions("system", "user")

        assert "temperature" not in mock_openai.chat.completions.create.call_args.kwargs

    def test_openai_gpt5_uses_max_completion_tokens_without_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", CHAT_DEFAULT_OPENAI)
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            LLMClient(max_tokens=512).completions("system", "user")

        mock_openai.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == REASONING_MAX_COMPLETION_TOKENS
        assert "temperature" not in call_kwargs

    def test_nvidia_build_uses_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            LLMClient(max_tokens=512).completions("system", "user")

        mock_openai.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert call_kwargs["stream"] is False
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temperature"] == 0.0
        assert "max_completion_tokens" not in call_kwargs

    @pytest.mark.parametrize(
        ("provider", "api_key_env", "api_key"),
        [
            ("nv_build", "NVIDIA_API_KEY", "test-nvidia-key"),
            ("openai-compatible", "SKILL_EVAL_LLM_API_KEY", "test-compatible-key"),
        ],
    )
    def test_non_openai_provider_at_canonical_endpoint_uses_max_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        api_key_env: str,
        api_key: str,
    ) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
        monkeypatch.setenv(api_key_env, api_key)
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "gpt-5.4-mini")
        monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", OPENAI_BASE_URL)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            LLMClient(max_tokens=512).completions("system", "user")

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == REASONING_MAX_COMPLETION_TOKENS
        assert "max_completion_tokens" not in call_kwargs
        assert "temperature" not in call_kwargs

    @pytest.mark.parametrize(
        ("max_tokens", "expected_max_tokens", "temperature"),
        [(None, MAX_COMPLETION_TOKENS, 0.0), (512, 512, 0.3)],
    )
    def test_anthropic_opus5_preserves_token_limit_and_omits_temperature(
        self,
        monkeypatch: pytest.MonkeyPatch,
        max_tokens: int | None,
        expected_max_tokens: int,
        temperature: float,
    ) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value.content = [SimpleNamespace(type="text", text="Done")]

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            content = LLMClient(max_tokens=max_tokens, temperature=temperature).completions("system", "user")

        assert content == "Done"
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["model"] == CHAT_DEFAULT_ANTHROPIC
        assert call_kwargs["max_tokens"] == expected_max_tokens
        assert "max_completion_tokens" not in call_kwargs
        assert "temperature" not in call_kwargs

    def test_older_anthropic_model_preserves_custom_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "claude-3-5-sonnet-20241022")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value.content = [SimpleNamespace(type="text", text="Done")]

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            LLMClient(temperature=0.2).completions("system", "user")

        assert mock_anthropic.messages.create.call_args.kwargs["temperature"] == 0.2

    def test_anthropic_mythos_preview_omits_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "claude-mythos-preview")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value.content = [SimpleNamespace(type="text", text="Done")]

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            LLMClient(temperature=0.2).completions("system", "user")

        assert "temperature" not in mock_anthropic.messages.create.call_args.kwargs

    @pytest.mark.parametrize("max_tokens", [None, 0])
    def test_bedrock_opus5_preserves_token_limits_and_omits_temperature(
        self,
        monkeypatch: pytest.MonkeyPatch,
        max_tokens: int | None,
    ) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
        monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Done"))])

        with patch("litellm.completion", return_value=response) as completion:
            content = LLMClient(max_tokens=max_tokens, temperature=0.2).completions("system", "user")

        assert content == "Done"
        call_kwargs = completion.call_args.kwargs
        assert call_kwargs["model"] == f"bedrock/{CHAT_DEFAULT_BEDROCK}"
        assert "temperature" not in call_kwargs
        if max_tokens is None:
            assert {"max_tokens", "max_completion_tokens"}.isdisjoint(call_kwargs)
        else:
            assert call_kwargs["max_tokens"] == 0
            assert "max_completion_tokens" not in call_kwargs

    @pytest.mark.parametrize(("timeout", "expected"), [(None, {}), (5.0, {"timeout": 5.0})])
    def test_bedrock_forwards_only_an_explicit_timeout(
        self, monkeypatch: pytest.MonkeyPatch, timeout: float | None, expected: dict
    ) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Done"))])

        with patch("litellm.completion", return_value=response) as completion:
            LLMClient(timeout=timeout).completions("system", "user")

        assert {key: value for key, value in completion.call_args.kwargs.items() if key == "timeout"} == expected

    def test_older_bedrock_model_preserves_custom_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Done"))])

        with patch("litellm.completion", return_value=response) as completion:
            LLMClient(temperature=0.2).completions("system", "user")

        assert completion.call_args.kwargs["temperature"] == 0.2

    @pytest.mark.parametrize("base_url_env", ["SKILL_EVAL_LLM_BASE_URL", "OPENAI_BASE_URL"])
    def test_openai_provider_custom_base_url_uses_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch, base_url_env: str
    ) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "gpt-5.4-mini")
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv(base_url_env, "https://example.test/v1")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            LLMClient(max_tokens=512).completions("system", "user")

        mock_cls.assert_called_once_with(api_key="test-key", base_url="https://example.test/v1", max_retries=0)
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == REASONING_MAX_COMPLETION_TOKENS
        assert "max_completion_tokens" not in call_kwargs

    def test_api_key_only_gpt5_uses_max_completion_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            client = LLMClient(model="gpt-5.4-mini", api_key="test-key", max_tokens=512)
            client.completions("system", "user")

        assert (client.base_url, mock_cls.call_args.kwargs["base_url"]) == (OPENAI_BASE_URL, OPENAI_BASE_URL)
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == REASONING_MAX_COMPLETION_TOKENS

    def test_api_key_only_gpt5_honors_ambient_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_base_url = "https://example.test/v1"
        monkeypatch.setenv("OPENAI_BASE_URL", custom_base_url)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            client = LLMClient(model="gpt-5.4-mini", api_key="test-key", base_url=None, max_tokens=512)
            client.completions("system", "user")

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert (
            client.base_url,
            mock_cls.call_args.kwargs["base_url"],
            call_kwargs.get("max_tokens"),
            call_kwargs.get("max_completion_tokens"),
        ) == (custom_base_url, custom_base_url, REASONING_MAX_COMPLETION_TOKENS, None)

    @pytest.mark.parametrize(
        ("base_url", "expected_key"),
        [
            (
                OPENAI_BASE_URL.replace("https://", "HTTPS://").replace("api.openai.com", "API.OPENAI.COM"),
                "max_completion_tokens",
            ),
            (OPENAI_BASE_URL.replace("/v1", ":443/v1"), "max_completion_tokens"),
            (f"{OPENAI_BASE_URL}/", "max_completion_tokens"),
            (OPENAI_BASE_URL.replace("/v1", ".evil.test/v1"), "max_tokens"),
            (OPENAI_BASE_URL.replace("https://", "http://"), "max_tokens"),
            (f"{OPENAI_BASE_URL}?route=proxy", "max_tokens"),
            (OPENAI_BASE_URL.replace("https://", "https://user@"), "max_tokens"),
            (f"{OPENAI_BASE_URL}beta", "max_tokens"),
            (OPENAI_BASE_URL.replace("/v1", ":444/v1"), "max_tokens"),
            (OPENAI_BASE_URL.replace("/v1", ":invalid/v1"), "max_tokens"),
        ],
    )
    def test_endpoint_url_controls_gpt5_token_key(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str, expected_key: str
    ) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        unexpected_key = "max_tokens" if expected_key == "max_completion_tokens" else "max_completion_tokens"
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient(model="gpt-5.4-mini", api_key="test-key", base_url=base_url, max_tokens=512)
            client.completions("system", "user")

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs[expected_key] == REASONING_MAX_COMPLETION_TOKENS
        assert unexpected_key not in call_kwargs

    @pytest.mark.parametrize("base_url", [None, "https://example.test/v1"])
    def test_none_omits_token_limit_keys(self, monkeypatch: pytest.MonkeyPatch, base_url: str | None) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient(model="gpt-5.4-mini", api_key="test-key", base_url=base_url, max_tokens=None)
            client.completions("system", "user")

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert {"max_tokens", "max_completion_tokens"}.isdisjoint(call_kwargs)

    @pytest.mark.parametrize(
        ("base_url", "expected_key", "unexpected_key"),
        [
            (None, "max_completion_tokens", "max_tokens"),
            ("https://example.test/v1", "max_tokens", "max_completion_tokens"),
        ],
    )
    def test_zero_token_limit_uses_endpoint_appropriate_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        base_url: str | None,
        expected_key: str,
        unexpected_key: str,
    ) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient(model="gpt-5.4-mini", api_key="test-key", base_url=base_url, max_tokens=0)
            client.completions("system", "user")

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs[expected_key] == REASONING_MAX_COMPLETION_TOKENS
        assert unexpected_key not in call_kwargs

    def test_custom_gpt5_endpoint_uses_max_tokens(self) -> None:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient(
                model="gpt-5-custom",
                api_key="test-key",
                base_url="https://example.test/v1",
                max_tokens=512,
            )
            client.completions("system", "user")

        mock_openai.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == REASONING_MAX_COMPLETION_TOKENS
        assert "max_completion_tokens" not in call_kwargs

    def test_vertex_openapi_refreshes_token_on_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Refresh Google access token and retry on 401 AuthenticationError for Vertex OpenAPI endpoints."""
        from openai import AuthenticationError

        vertex_base_url = (
            "https://us-central1-aiplatform.googleapis.com/v1beta1/"
            "projects/my-proj/locations/us-central1/endpoints/openapi"
        )
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
        monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", vertex_base_url)
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "google/gemini-3.8-flash")
        monkeypatch.setattr(
            "skillevaluator.provider_config._get_google_access_token", lambda **_kwargs: "mock-refreshed-token"
        )

        mock_openai = MagicMock()
        mock_auth_err = AuthenticationError("Unauthorized", response=MagicMock(status_code=401), body=None)
        mock_success_response = MagicMock()
        mock_success_response.choices = [MagicMock(message=MagicMock(content="Refreshed response"))]
        mock_openai.chat.completions.create.side_effect = [mock_auth_err, mock_success_response]

        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient()
            result = client.completions("system", "user")

        assert result == "Refreshed response"
        assert mock_openai.chat.completions.create.call_count == 2
        assert mock_openai.api_key == "mock-refreshed-token"
        assert client.api_key == "mock-refreshed-token"
        assert client._resolved_config().api_key == "mock-refreshed-token"

    def test_vertex_openapi_does_not_refresh_token_when_explicit_api_key_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not overwrite user-provided API key with ADC token when 401 occurs."""
        from openai import AuthenticationError

        vertex_base_url = (
            "https://us-central1-aiplatform.googleapis.com/v1beta1/"
            "projects/my-proj/locations/us-central1/endpoints/openapi"
        )
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
        monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", vertex_base_url)
        monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "explicit-user-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "google/gemini-3.8-flash")
        monkeypatch.setattr(
            "skillevaluator.provider_config._get_google_access_token", lambda **_kwargs: "host-adc-token"
        )

        mock_openai = MagicMock()
        mock_auth_err = AuthenticationError("Unauthorized", response=MagicMock(status_code=401), body=None)
        mock_openai.chat.completions.create.side_effect = mock_auth_err

        with patch("openai.OpenAI", return_value=mock_openai):
            client = LLMClient()
            with pytest.raises(AuthenticationError):
                client.completions("system", "user")

        assert mock_openai.chat.completions.create.call_count == 1
        assert client.api_key == "explicit-user-key"
        assert client._resolved_config().api_key == "explicit-user-key"


def _nested_object(depth: int) -> str:
    return '{"a":' * (depth - 1) + '{"a": 1}' + "}" * (depth - 1)


def _extract_json_reply(monkeypatch: pytest.MonkeyPatch, content: str) -> dict:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=content))]
    with patch("openai.OpenAI", return_value=mock_openai):
        return LLMClient().extract_json_from_response("system", "user")


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

    def test_recovers_one_object_surrounded_by_prose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = (
            "The chunks overlap, but one summarizes what the other explains.\n\n"
            '{"verdict": "INTENTIONAL_DETAIL", "confidence": 0.82, '
            '"rationale": "Summary and detail.", "suggestion": "Keep both sections."}'
            "\n\nLet me know if you need more detail."
        )

        assert _extract_json_reply(monkeypatch, content) == {
            "verdict": "INTENTIONAL_DETAIL",
            "confidence": 0.82,
            "rationale": "Summary and detail.",
            "suggestion": "Keep both sections.",
        }

    def test_recovers_fenced_object_after_prose_preamble(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = 'Here is the result:\n```json\n{"key": "value"}\n```'

        assert _extract_json_reply(monkeypatch, content) == {"key": "value"}

    def test_recovers_single_line_fenced_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _extract_json_reply(monkeypatch, '```{"verdict": "DUPLICATE"}```') == {"verdict": "DUPLICATE"}

    def test_bracketed_citations_are_not_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = 'Chunks [1] and [2] repeat the same steps.\n{"key": "value"}'

        assert _extract_json_reply(monkeypatch, content) == {"key": "value"}

    def test_nested_object_is_not_counted_as_a_second_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _extract_json_reply(monkeypatch, 'Result:\n{"outer": {"key": "value"}}') == {"outer": {"key": "value"}}

    def test_prose_braces_that_are_not_json_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = 'Compare {skill_name} with ${HOME} first.\n{"key": "value"}'

        assert _extract_json_reply(monkeypatch, content) == {"key": "value"}

    def test_multiple_embedded_objects_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(LLMClientError, match="found 2 JSON objects; expected exactly one"):
            _extract_json_reply(monkeypatch, 'First: {"key": "one"}\nSecond: {"key": "two"}')

    @pytest.mark.parametrize(
        "container",
        [
            pytest.param(
                '{"verdict": "RELATED_BUT_DISTINCT", "detail": {"verdict": "DUPLICATE"} "rationale": "x"}',
                id="missing-comma-after-nested-object",
            ),
            pytest.param(
                '{"verdict": "RELATED_BUT_DISTINCT" "detail": {"verdict": "DUPLICATE"}}',
                id="missing-comma-before-nested-key",
            ),
            pytest.param(
                '{"verdict": "RELATED_BUT_DISTINCT", {"verdict": "DUPLICATE"}}',
                id="nested-object-where-key-expected",
            ),
            pytest.param('{verdict: {"verdict": "DUPLICATE"}}', id="unquoted-key"),
            pytest.param('{"verdict": "RELATED_BUT_DISTINCT", "detail": {"verdict": "DUPLICATE"}', id="never-closes"),
            pytest.param('[oops, {"verdict": "DUPLICATE"}]', id="malformed-array"),
            pytest.param('[{"verdict": "DUPLICATE"}]', id="valid-array"),
        ],
    )
    def test_object_inside_another_container_is_not_promoted(
        self, monkeypatch: pytest.MonkeyPatch, container: str
    ) -> None:
        with pytest.raises(LLMClientError, match="invalid JSON"):
            _extract_json_reply(monkeypatch, "Result: " + container)

    @pytest.mark.parametrize("prefix", [pytest.param("", id="whole-reply"), pytest.param("Result: ", id="in-prose")])
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(_nested_object(MAX_JSON_NESTING + 1), id="nesting-just-beyond-limit"),
            pytest.param(_nested_object(100_000), id="nesting-beyond-recursion-limit"),
            pytest.param('{"a": ' + "1" * 5_000 + "}", id="integer-beyond-digit-limit"),
        ],
    )
    def test_undecodable_object_fails_closed(self, monkeypatch: pytest.MonkeyPatch, prefix: str, payload: str) -> None:
        # The depth bound is checked before decoding, so the result does not
        # depend on the Python version or thread stack size (Python 3.14 can
        # decode 100,000 levels when the C stack allows it).
        with pytest.raises(LLMClientError, match="invalid JSON"):
            _extract_json_reply(monkeypatch, prefix + payload)

    @pytest.mark.parametrize("prefix", [pytest.param("", id="whole-reply"), pytest.param("Result: ", id="in-prose")])
    def test_object_at_nesting_limit_is_accepted(self, monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
        payload = _nested_object(MAX_JSON_NESTING)

        assert _extract_json_reply(monkeypatch, prefix + payload) == json.loads(payload)

    @pytest.mark.parametrize("prefix", [pytest.param("", id="whole-reply"), pytest.param("Result: ", id="in-prose")])
    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            pytest.param(
                '{"verdict": "DUPLICATE", "confidence": 0.9, "verdict": "RELATED_BUT_DISTINCT"}',
                "Duplicate JSON object member",
                id="duplicate-key",
            ),
            pytest.param(
                '{"verdict": "DUPLICATE", "detail": {"k": 1, "k": 2}}',
                "Duplicate JSON object member",
                id="nested-duplicate-key",
            ),
            pytest.param('{"verdict": "DUPLICATE", "confidence": NaN}', "Non-standard JSON constant", id="nan"),
            pytest.param(
                '{"verdict": "DUPLICATE", "confidence": -Infinity}', "Non-standard JSON constant", id="infinity"
            ),
            pytest.param('{"verdict": "DUPLICATE", "confidence": 1e400}', "non-finite", id="float-overflow"),
        ],
    )
    def test_ambiguous_object_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, prefix: str, payload: str, reason: str
    ) -> None:
        with pytest.raises(LLMClientError, match="invalid JSON") as excinfo:
            _extract_json_reply(monkeypatch, prefix + payload)

        if not prefix:
            # In prose the strict whole-reply error is reported; the scan found no usable object.
            assert reason in str(excinfo.value)

    def test_ambiguous_object_does_not_hide_another_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = 'Draft: {"verdict": "DUPLICATE", "verdict": "UNIQUE"}\nFinal: {"verdict": "RELATED_BUT_DISTINCT"}'

        assert _extract_json_reply(monkeypatch, content) == {"verdict": "RELATED_BUT_DISTINCT"}

    def test_undecodable_object_blocks_a_later_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _nested_object(100_000)

        with pytest.raises(LLMClientError, match="invalid JSON"):
            _extract_json_reply(monkeypatch, "Result: " + payload + ' {"key": "value"}')

    def test_many_malformed_spans_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _extract_json_reply(monkeypatch, "{x} " * 5_000 + '{"key": "value"}') == {"key": "value"}

    def test_bedrock_prompt_only_reply_with_preamble_is_recovered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from skillevaluator.deduplication.intra_skill.llm_analyzer import VERDICT_JSON_SCHEMA

        _use_provider(monkeypatch, "bedrock", CHAT_DEFAULT_BEDROCK)
        reply = (
            "Both chunks list the same setup steps.\n"
            '{"verdict": "DUPLICATE", "confidence": 0.9, "rationale": "Same steps.", "suggestion": "Keep one copy."}'
        )

        with patch("litellm.completion", return_value=_openai_choice_response(reply)) as completion:
            result = LLMClient().extract_json_from_response(
                "system", "user", response_schema=VERDICT_JSON_SCHEMA, schema_name="dedup_verdict"
            )

        assert result["verdict"] == "DUPLICATE"
        assert "response_format" not in completion.call_args.kwargs

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


def _openai_choice_response(content: str | None, *, finish_reason: str = "stop", refusal: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content, refusal=refusal))
        ]
    )


def _anthropic_response(text: str, *, stop_reason: str = "end_turn"):
    content = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _use_provider(monkeypatch: pytest.MonkeyPatch, provider: str, model: str) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", model)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    for name in ("SKILL_EVAL_LLM_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


class TestReasoningModelRequests:
    @pytest.mark.parametrize(("model", "effort"), [(CHAT_DEFAULT_OPENAI, "high"), ("gpt-5.6-sol", None)])
    def test_finding_verifier_sends_high_effort_only_to_gpt6(
        self, monkeypatch: pytest.MonkeyPatch, model: str, effort: str | None
    ) -> None:
        _use_provider(monkeypatch, "openai", model)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response("Done")

        with patch("openai.OpenAI", return_value=mock_openai):
            FindingVerifier().completions("system", "user")

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("reasoning_effort") == effort
        assert call_kwargs["max_completion_tokens"] == REASONING_MAX_COMPLETION_TOKENS
        assert "temperature" not in call_kwargs

    @pytest.mark.parametrize("role", ["dedup", "dimension_judge", "insights_judge"])
    def test_medium_effort_roles_send_medium_to_gpt6(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
    ) -> None:
        from skillevaluator.deduplication.intra_skill import intra_skill_validator
        from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
        from skillevaluator.deduplication.utils.chunker import ContentChunk
        from skillevaluator.evaluation.dimension_judge import DimensionJudge
        from skillevaluator.evaluation.insights_judge import InsightsJudge

        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response(
            '{"verdict": "DISTINCT", "confidence": 0.9, "rationale": "Different steps.", "suggestion": ""}'
        )

        with patch("openai.OpenAI", return_value=mock_openai):
            if role == "dedup":
                skill_dir = tmp_path / "skill"
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
                members = [
                    ContentChunk("SKILL.md", "## A", 1, 2, "a" * 100, "markdown"),
                    ContentChunk("SKILL.md", "## B", 3, 4, "b" * 100, "markdown"),
                ]
                with (
                    patch.object(intra_skill_validator, "EmbeddingClient") as embedding_client,
                    patch.object(
                        intra_skill_validator,
                        "build_clusters",
                        return_value=[ContentCluster(members, 1.0, 1.0, False, {"markdown"})],
                    ),
                ):
                    embedding_client.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
                    intra_skill_validator.IntraSkillValidator().validate(skill_dir)
            else:
                judge = DimensionJudge() if role == "dimension_judge" else InsightsJudge()
                judge.completions("system", "user")

        mock_openai.chat.completions.create.assert_called_once()
        assert mock_openai.chat.completions.create.call_args.kwargs["reasoning_effort"] == "medium"

    def test_claude_5_5_rubric_gets_schema_effort_and_server_side_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        reply = {
            "overall_pass": True,
            "score": 90,
            "summary": "ok",
            "checks": [
                {"id": criterion["id"], "criterion": criterion["criterion"], "pass": True, "score": 9, "notes": "ok"}
                for criterion in RUBRIC_CRITERIA
            ],
        }
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_response(json.dumps(reply))

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            report = RubricJudge().process(skill_name="demo", skill_content="# Demo")

        assert report == reply
        assert mock_anthropic.messages.create.call_count == 1
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == MAX_COMPLETION_TOKENS
        assert call_kwargs["output_config"] == {
            "format": {"type": "json_schema", "schema": RUBRIC_JSON_SCHEMA},
            "effort": "medium",
        }
        assert call_kwargs["extra_headers"] == {"anthropic-beta": ANTHROPIC_SERVER_SIDE_FALLBACK_BETA}
        assert call_kwargs["extra_body"] == {"fallbacks": [{"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}]}

    def test_claude_opus_5_keeps_model_default_effort_and_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_response("Done")

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            FindingVerifier().completions("system", "user")

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert {"output_config", "extra_headers", "extra_body"}.isdisjoint(call_kwargs)

    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            (None, [{"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}]),
            ("", [{"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}]),
            (" claude-opus-5 ", [{"model": "claude-opus-5"}]),
            ("none", None),
            ("OFF", None),
            ("disabled", None),
            ("claude-sonnet-5-5", None),
        ],
    )
    def test_refusal_fallback_model_is_configurable(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: list[dict[str, str]] | None
    ) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-sonnet-5-5")
        monkeypatch.delenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", raising=False)
        if env_value is not None:
            monkeypatch.setenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", env_value)
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_response("Done")

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            FindingVerifier().completions("system", "user")

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        if expected is None:
            assert {"extra_headers", "extra_body"}.isdisjoint(call_kwargs)
        else:
            assert call_kwargs["extra_headers"] == {"anthropic-beta": ANTHROPIC_SERVER_SIDE_FALLBACK_BETA}
            assert call_kwargs["extra_body"] == {"fallbacks": expected}

    def test_anthropic_gateway_gets_effort_without_server_side_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_response("Done")

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            FindingVerifier().completions("system", "user")

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {"effort": "high"}
        assert {"extra_headers", "extra_body"}.isdisjoint(call_kwargs)

    def test_schema_downgrade_keeps_anthropic_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class SchemaRejected(Exception):
            status_code = 400

        _use_provider(monkeypatch, "anthropic", "claude-sonnet-5-5")
        monkeypatch.setattr(client_mod, "_SCHEMA_UNSUPPORTED_TARGETS", set())
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            SchemaRejected("output_config.format: Extra inputs are not permitted"),
            _anthropic_response("{}"),
        ]

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            LLMClient(reasoning_effort="medium").completions("system", "user", response_schema={"type": "object"})

        first, second = (call.kwargs for call in mock_anthropic.messages.create.call_args_list)
        assert first["output_config"]["format"]["schema"] == {"type": "object"}
        assert second["output_config"] == {"effort": "medium"}

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("model", ["claude-opus-4-1-20250805", "claude-opus-4-0"])
    def test_anthropic_max_tokens_stay_under_the_sdk_non_streaming_limit(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """The real SDK refuses Opus 4/4.1 non-streaming requests above 8,192 tokens before sending."""
        import httpx

        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": "Done"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        _use_provider(monkeypatch, "anthropic", model)
        verifier = FindingVerifier(http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        assert verifier.completions("system", "user") == "Done"
        assert requests[0]["max_tokens"] == 8192

    @pytest.mark.parametrize(
        ("base_url", "provider_default"),
        [(None, CHAT_DEFAULT_OPENAI), ("https://gateway.example/v1", CHAT_DEFAULT_GATEWAY)],
    )
    def test_explicit_credentials_use_the_provider_default_model(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str | None, provider_default: str
    ) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        assert LLMClient(api_key="test-key", base_url=base_url).model == provider_default


class TestRefusalAndTruncation:
    @pytest.mark.parametrize(
        "response",
        [
            _openai_choice_response("", finish_reason="content_filter"),
            _openai_choice_response(None, refusal="I can't help with that."),
        ],
    )
    def test_openai_refusal_is_a_typed_error_and_not_retried(self, monkeypatch: pytest.MonkeyPatch, response) -> None:
        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = response

        with patch("openai.OpenAI", return_value=mock_openai), pytest.raises(LLMClientRefusedError):
            LLMClient().completions("system", "user")

        assert mock_openai.chat.completions.create.call_count == 1

    def test_anthropic_refusal_is_a_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_response("", stop_reason="refusal")

        with patch("anthropic.Anthropic", return_value=mock_anthropic), pytest.raises(LLMClientRefusedError):
            LLMClient().completions("system", "user")

    def test_bedrock_content_filter_is_a_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "bedrock", "us.anthropic.claude-opus-5-5")

        with (
            patch("litellm.completion", return_value=_openai_choice_response("", finish_reason="content_filter")),
            pytest.raises(LLMClientRefusedError),
        ):
            LLMClient().completions("system", "user")

    @pytest.mark.parametrize(
        ("provider", "model", "response"),
        [
            ("openai", CHAT_DEFAULT_OPENAI, _openai_choice_response('{"partial": ', finish_reason="length")),
            ("anthropic", CHAT_DEFAULT_ANTHROPIC, _anthropic_response('{"partial": ', stop_reason="max_tokens")),
        ],
    )
    def test_truncation_is_a_typed_error_that_keeps_partial_text(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, model: str, response
    ) -> None:
        _use_provider(monkeypatch, provider, model)
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = response
        sdk.messages.create.return_value = response

        with (
            patch("openai.OpenAI", return_value=sdk),
            patch("anthropic.Anthropic", return_value=sdk),
            pytest.raises(LLMClientTruncatedError) as exc_info,
        ):
            LLMClient().completions("system", "user")

        assert exc_info.value.content == '{"partial":'

    def test_process_records_the_refusal_cause(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response("", finish_reason="content_filter")
        verifier = FindingVerifier()

        with patch("openai.OpenAI", return_value=mock_openai):
            assert verifier.process(findings=[], skill_path=tmp_path) == {}

        assert verifier.last_failure is not None
        assert "declined the request" in verifier.last_failure

    @pytest.mark.parametrize(
        ("provider", "model", "make_response"),
        [
            ("openai", CHAT_DEFAULT_OPENAI, lambda text: _openai_choice_response(text, finish_reason="length")),
            ("anthropic", CHAT_DEFAULT_ANTHROPIC, lambda text: _anthropic_response(text, stop_reason="max_tokens")),
        ],
    )
    def test_verifier_keeps_complete_verdicts_from_a_truncated_reply(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str, model: str, make_response
    ) -> None:
        _use_provider(monkeypatch, provider, model)
        reply = (
            '{"index": 0, "verdict": "false_positive", "confidence": "high", "rationale": "Test card."}\n'
            '{"index": 1, "verdict": "true_positive", "confidence": "high", "rationale": "Real key."}\n'
            '{"index": 2, "verdict": "false_pos'
        )
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = make_response(reply)
        sdk.messages.create.return_value = make_response(reply)
        verifier = FindingVerifier()

        with patch("openai.OpenAI", return_value=sdk), patch("anthropic.Anthropic", return_value=sdk):
            verdicts = verifier.process(findings=[], skill_path=tmp_path)

        assert sorted(verdicts) == [0, 1]
        assert verdicts[0]["verdict"] == "false_positive"
        assert verifier.last_failure is not None
        assert "output-token limit" in verifier.last_failure

    def test_truncated_reply_uses_the_fallback_without_salvage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Client(LLMClient):
            def get_system_prompt(self) -> str:
                return "system"

            def create_user_prompt(self, **_kwargs) -> str:
                return "user"

            def parse_response(self, response_text: str, **_kwargs) -> str:
                return response_text

            def get_fallback_response(self, **_kwargs) -> str:
                return "fallback"

        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response("partial", finish_reason="length")

        with patch("openai.OpenAI", return_value=mock_openai):
            client = Client()
            assert client.process() == "fallback"

        assert client.last_failure is not None


_CHAT_HISTORY = [
    {"role": "system", "content": "Use run evidence"},
    {"role": "user", "content": "Reduce tokens"},
    {"role": "assistant", "content": "Check case 1"},
    {"role": "user", "content": "What about runtime?"},
]


class TestChat:
    def test_empty_history_is_rejected_without_creating_client(self) -> None:
        client = LLMClient()
        with (
            patch.object(client, "_get_client") as get_client,
            pytest.raises(LLMClientError, match="at least one message"),
        ):
            client.chat([])
        get_client.assert_not_called()

    @pytest.mark.parametrize(
        ("messages", "match"),
        [
            ([{"role": "system", "content": "only system"}], "user or assistant"),
            ([{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}], "must come before"),
            ([{"role": "tool", "content": "result"}], "system, user or assistant"),
            ([{"role": "user", "content": None}], "must be a string"),
            (["hello"], "must be a string"),
        ],
        ids=["system-only", "late-system", "unknown-role", "non-string-content", "not-a-mapping"],
    )
    def test_invalid_history_is_rejected_without_creating_client(self, messages, match: str) -> None:
        client = LLMClient()
        with patch.object(client, "_get_client") as get_client, pytest.raises(LLMClientError, match=match):
            client.chat(messages)
        get_client.assert_not_called()

    def test_openai_compatible_history_is_sent_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response("ok")
        messages = [dict(message) for message in _CHAT_HISTORY]

        with patch("openai.OpenAI", return_value=mock_openai):
            assert LLMClient().chat(messages) == "ok"

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == _CHAT_HISTORY
        assert call_kwargs["stream"] is False
        assert messages == _CHAT_HISTORY

    def test_anthropic_receives_the_system_text_separately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_response("ok")

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            assert LLMClient().chat(_CHAT_HISTORY) == "ok"

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Use run evidence"
        assert call_kwargs["messages"] == _CHAT_HISTORY[1:]
        # Server-side refusal fallback still applies to chat on the native endpoint.
        assert call_kwargs["extra_headers"] == {"anthropic-beta": ANTHROPIC_SERVER_SIDE_FALLBACK_BETA}
        assert call_kwargs["extra_body"] == {"fallbacks": [{"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}]}

    def test_bedrock_history_is_sent_through_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "bedrock", "us.anthropic.claude-opus-5-5")

        with patch("litellm.completion", return_value=_openai_choice_response("ok")) as completion:
            assert LLMClient().chat(_CHAT_HISTORY) == "ok"

        assert completion.call_args.kwargs["messages"] == _CHAT_HISTORY

    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_history_without_system_text_sends_none(self, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
        _use_provider(monkeypatch, provider, CHAT_DEFAULT_ANTHROPIC if provider == "anthropic" else CHAT_DEFAULT_OPENAI)
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = _openai_choice_response("ok")
        sdk.messages.create.return_value = _anthropic_response("ok")
        turns = [{"role": "user", "content": "hello"}]

        with patch("openai.OpenAI", return_value=sdk), patch("anthropic.Anthropic", return_value=sdk):
            assert LLMClient().chat(turns) == "ok"

        if provider == "anthropic":
            call_kwargs = sdk.messages.create.call_args.kwargs
            assert "system" not in call_kwargs
        else:
            call_kwargs = sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == turns

    def test_completions_still_sends_one_system_and_one_user_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response("ok")

        with patch("openai.OpenAI", return_value=mock_openai):
            assert LLMClient().completions("", "user") == "ok"

        assert mock_openai.chat.completions.create.call_args.kwargs["messages"] == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "user"},
        ]

    def test_feedback_chat_truncation_is_a_typed_error_and_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from skillevaluator.evaluation.tier3_feedback_chat import Tier3FeedbackAssistant

        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response("partial", finish_reason="length")

        with patch("openai.OpenAI", return_value=mock_openai), pytest.raises(LLMClientTruncatedError):
            Tier3FeedbackAssistant().chat([{"role": "user", "content": "hello"}])

        assert mock_openai.chat.completions.create.call_count == 1

    def test_empty_choices_is_a_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = SimpleNamespace(choices=[])

        with patch("openai.OpenAI", return_value=mock_openai), pytest.raises(EmptyLLMResponseError):
            LLMClient().chat([{"role": "user", "content": "hello"}])

        assert mock_openai.chat.completions.create.call_count == 1


def _anthropic_fallback_response(text: str, served_by: object):
    """Return a reply whose server-side fallback block names the model that served it."""
    response = _anthropic_response(text)
    response.content.insert(0, SimpleNamespace(type="fallback", to=served_by))
    return response


def _anthropic_truncated_fallback_response(text: str):
    """Return a reply cut off at the token limit, written by the server-side fallback model."""
    response = _anthropic_response(text, stop_reason="max_tokens")
    response.content.insert(0, SimpleNamespace(type="fallback", to={"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}))
    return response


class TestModelProvenance:
    def test_model_used_names_the_configured_model_after_a_success_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = [
            _openai_choice_response("Done"),
            _openai_choice_response("", finish_reason="content_filter"),
        ]
        client = LLMClient()

        with patch("openai.OpenAI", return_value=mock_openai):
            assert client.model_used is None
            client.completions("system", "user")
            assert client.model_used == CHAT_DEFAULT_OPENAI
            with pytest.raises(LLMClientRefusedError):
                client.completions("system", "user")

        assert client.model_used is None

    @pytest.mark.parametrize(
        "served_by",
        [{"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}, SimpleNamespace(model=ANTHROPIC_REFUSAL_FALLBACK_MODEL)],
        ids=["untyped-block", "typed-block"],
    )
    def test_model_used_names_the_server_side_fallback_model(
        self, monkeypatch: pytest.MonkeyPatch, served_by: object
    ) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_fallback_response("Done", served_by)
        client = LLMClient()

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            assert client.completions("system", "user") == "Done"

        assert client.model == "claude-opus-5-5"
        assert client.model_used == ANTHROPIC_REFUSAL_FALLBACK_MODEL

    @pytest.mark.parametrize(
        ("base_url", "expected_model"),
        [(None, ANTHROPIC_REFUSAL_FALLBACK_MODEL), ("https://gateway.example", "claude-opus-5-5")],
        ids=["native-sticky-turn", "gateway-without-fallback"],
    )
    def test_model_used_reads_the_reply_model_only_when_fallback_was_requested(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str | None, expected_model: str
    ) -> None:
        """A sticky-served turn has no fallback block; its top-level model names the model that wrote it."""
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        if base_url:
            monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
        sent: list[dict] = []

        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": ANTHROPIC_REFUSAL_FALLBACK_MODEL,
                    "content": [{"type": "text", "text": "Done"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "iterations": [{"type": "fallback_message", "model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}],
                    },
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = LLMClient(http_client=http_client)
            assert client.completions("system", "user") == "Done"

        assert ("fallbacks" in sent[0]) is (base_url is None)
        assert client.model_used == expected_model

    @pytest.mark.parametrize(
        "base_url", [None, "https://gateway.example"], ids=["native-with-fallback", "gateway-without-fallback"]
    )
    def test_model_used_reads_the_fallback_block_before_the_reply_model(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str | None
    ) -> None:
        """The block wins over a different top-level model, and a block is honored on any Anthropic endpoint."""
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        if base_url:
            monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
        sent: list[dict] = []

        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": [
                        {
                            "type": "fallback",
                            "from": {"model": "claude-opus-5-5"},
                            "to": {"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL},
                        },
                        {"type": "text", "text": "Done"},
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = LLMClient(http_client=http_client)
            assert client.completions("system", "user") == "Done"

        assert ("fallbacks" in sent[0]) is (base_url is None)
        assert client.model_used == ANTHROPIC_REFUSAL_FALLBACK_MODEL

    def test_model_used_is_kept_per_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _openai_choice_response("Done")
        client = LLMClient()
        with patch("openai.OpenAI", return_value=mock_openai):
            client.completions("system", "user")
        seen: list[str | None] = []

        thread = threading.Thread(target=lambda: seen.append(client.model_used))
        thread.start()
        thread.join()

        assert seen == [None]
        assert client.model_used == CHAT_DEFAULT_OPENAI

    def test_chat_names_the_model_that_answered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_fallback_response(
            "Done", {"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}
        )
        client = LLMClient()

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            assert client.chat([{"role": "user", "content": "hello"}]) == "Done"
            assert client.model_used == ANTHROPIC_REFUSAL_FALLBACK_MODEL
            with pytest.raises(LLMClientError, match="at least one message"):
                client.chat([])

        assert client.model_used is None

    def test_llm_insights_name_the_model_that_wrote_them(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from skillevaluator.evaluation.insights_judge import InsightsJudge

        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        claim = {"claim_type": "general", "severity": "warn", "evidence_case_ids": ["positive-001"]}
        reply = json.dumps(
            {
                "conclusions": [{**claim, "title": "Slow", "message": "positive-001 took longer."}],
                "recommendations": [{**claim, "title": "Clarify", "message": "Clarify positive-001's output."}],
            }
        )
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_fallback_response(
            reply, {"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}
        )
        canonical = {"dataset": [{"id": "positive-001", "expected_skill": "calculator"}]}

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            insights = InsightsJudge().process(canonical=canonical, deterministic={})

        entries = insights["conclusions"] + insights["recommendations"]
        assert len(entries) == 2
        assert {entry["model"] for entry in entries} == {ANTHROPIC_REFUSAL_FALLBACK_MODEL}

    def test_only_llm_written_dimensions_name_a_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from skillevaluator.constants import DIMENSION_MAPPING
        from skillevaluator.evaluation.dimension_judge import compute_dimensions

        _use_provider(monkeypatch, "openai", CHAT_DEFAULT_OPENAI)
        reply = json.dumps(
            {
                "dimensions": [
                    {"id": dimension, "score": 0.8, "verdict": "pass", "explanation": "Strong evidence."}
                    for dimension in DIMENSION_MAPPING
                ]
            }
        )
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = [
            _openai_choice_response(reply),
            _openai_choice_response("not json"),
        ]
        evaluators = {"accuracy": {"with_skill": 0.8, "baseline": 0.6}}

        with patch("openai.OpenAI", return_value=mock_openai):
            written = compute_dimensions(evaluators, [], 0.2)
            deterministic = compute_dimensions(evaluators, [], 0.2)

        assert {dimension["model"] for dimension in written} == {CHAT_DEFAULT_OPENAI}
        assert not any("model" in dimension for dimension in deterministic)

    @pytest.mark.parametrize(
        ("provider", "model", "make_response", "expected"),
        [
            (
                "openai",
                CHAT_DEFAULT_OPENAI,
                lambda: _openai_choice_response('{"partial": ', finish_reason="length"),
                CHAT_DEFAULT_OPENAI,
            ),
            (
                "anthropic",
                "claude-opus-5-5",
                lambda: _anthropic_truncated_fallback_response('{"partial": '),
                ANTHROPIC_REFUSAL_FALLBACK_MODEL,
            ),
        ],
        ids=["openai-configured", "anthropic-fallback"],
    )
    def test_truncation_error_names_the_model_that_wrote_the_partial_reply(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, model: str, make_response, expected: str
    ) -> None:
        _use_provider(monkeypatch, provider, model)
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = make_response()
        sdk.messages.create.return_value = make_response()
        client = LLMClient()

        with (
            patch("openai.OpenAI", return_value=sdk),
            patch("anthropic.Anthropic", return_value=sdk),
            pytest.raises(LLMClientTruncatedError) as exc_info,
        ):
            client.completions("system", "user")

        assert exc_info.value.model_used == expected
        # The call itself failed, so the client still names no model for it.
        assert client.model_used is None

    def test_verdicts_kept_from_a_truncated_reply_name_the_model_that_wrote_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        reply = (
            '{"index": 0, "verdict": "false_positive", "confidence": "high", "rationale": "Test card."}\n'
            '{"index": 1, "verdict": "true_pos'
        )
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_truncated_fallback_response(reply)
        verifier = FindingVerifier()

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            verdicts = verifier.process(findings=[], skill_path=tmp_path)

        assert sorted(verdicts) == [0]
        assert verifier.last_failure is not None
        assert "output-token limit" in verifier.last_failure
        assert verifier.model == "claude-opus-5-5"
        assert verifier.model_used == ANTHROPIC_REFUSAL_FALLBACK_MODEL

    @pytest.mark.parametrize("salvage", [False, True], ids=["salvage-off", "salvage-fails"])
    def test_a_truncated_reply_that_is_not_kept_names_no_model(
        self, monkeypatch: pytest.MonkeyPatch, salvage: bool
    ) -> None:
        class Client(LLMClient):
            salvage_truncated_response = salvage

            def get_system_prompt(self) -> str:
                return "system"

            def create_user_prompt(self, **_kwargs) -> str:
                return "user"

            def parse_response(self, _response_text: str, **_kwargs) -> str:
                raise ValueError("no complete entries")

            def get_fallback_response(self, **_kwargs) -> str:
                return "fallback"

        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_truncated_fallback_response("partial")

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            client = Client()
            assert client.process() == "fallback"

        assert client.last_failure is not None
        assert client.model_used is None


@pytest.mark.parametrize("reply", ['{"index": 0, "verdict": "false_pos', "partial", "{}", "[]"])
def test_truncated_verifier_reply_without_complete_verdicts_names_no_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: str
) -> None:
    _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
    sdk = MagicMock()
    sdk.messages.create.return_value = _anthropic_truncated_fallback_response(reply)
    verifier = FindingVerifier()

    with patch("anthropic.Anthropic", return_value=sdk):
        assert verifier.verify([], tmp_path) == {}

    assert verifier.last_failure is not None
    assert "output-token limit" in verifier.last_failure
    assert verifier.model_used is None


def test_generic_truncated_parser_can_keep_a_complete_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client(LLMClient):
        salvage_truncated_response = True

        def get_system_prompt(self) -> str:
            return "system"

        def create_user_prompt(self, **_kwargs) -> str:
            return "user"

        def parse_response(self, _response_text: str, **_kwargs) -> dict:
            return {}

        def get_fallback_response(self, **_kwargs) -> str:
            return "fallback"

    _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
    sdk = MagicMock()
    sdk.messages.create.return_value = _anthropic_truncated_fallback_response("{}")
    with patch("anthropic.Anthropic", return_value=sdk):
        client = Client()
        assert client.process() == {}

    assert client.last_failure is not None
    assert client.model_used == ANTHROPIC_REFUSAL_FALLBACK_MODEL
