# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.inference.client."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
        client = LLMClient(
            model="gpt-5.4-mini",
            api_key="test-key",
            base_url=base_url,
            max_tokens=512,
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

    def test_claude_5_5_rubric_gets_schema_effort_and_server_side_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_provider(monkeypatch, "anthropic", "claude-opus-5-5")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _anthropic_response('{"checks": []}')

        with patch("anthropic.Anthropic", return_value=mock_anthropic):
            report = RubricJudge().process(skill_name="demo", skill_content="# Demo")

        assert report == {"checks": []}
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
