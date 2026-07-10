# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public provider-configuration behavior."""

from __future__ import annotations

import pytest

from skillevaluator.provider_config import ProviderConfigurationError, resolve_embedding_provider, resolve_llm_provider


def test_openai_provider_uses_standard_openai_credentials() -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
        }
    )

    assert config.provider == "openai"
    assert config.api_key == "test-openai-key"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.litellm_model.startswith("openai/")


def test_nvidia_build_uses_public_build_endpoint() -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "nv_build",
            "NVIDIA_API_KEY": "test-nvidia-key",
        }
    )

    assert config.provider == "nv_build"
    assert config.base_url == "https://integrate.api.nvidia.com/v1"
    assert config.api_key == "test-nvidia-key"
    assert config.model == "meta/llama-3.1-8b-instruct"


def test_nvidia_build_is_inferred_from_its_public_key() -> None:
    config = resolve_llm_provider({"NVIDIA_API_KEY": "test-nvidia-key"})

    assert config.provider == "nv_build"


def test_openai_compatible_provider_requires_explicit_model() -> None:
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "openai-compatible",
                "SKILL_EVAL_LLM_API_KEY": "test-key",
            }
        )
    assert str(exc_info.value) == "SKILL_EVAL_LLM_MODEL is required for openai-compatible providers."


def test_anthropic_requires_explicit_embedding_provider() -> None:
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_embedding_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
            }
        )
    assert (
        str(exc_info.value)
        == "SKILL_EVAL_EMBEDDING_PROVIDER is required because anthropic does not provide embeddings."
    )


def test_llm_provider_requires_public_credential_when_not_configured() -> None:
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_llm_provider({})

    assert str(exc_info.value) == "SKILL_EVAL_LLM_PROVIDER is required when no public provider credential is configured."


def test_llm_provider_error_lists_supported_choices() -> None:
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_llm_provider({"SKILL_EVAL_LLM_PROVIDER": "private-hub"})

    assert (
        str(exc_info.value)
        == "SKILL_EVAL_LLM_PROVIDER must be one of: anthropic, bedrock, nv_build, openai, openai-compatible."
    )


def test_explicit_openai_embedding_provider_uses_standard_openai_credentials() -> None:
    config = resolve_embedding_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "SKILL_EVAL_EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
        }
    )

    assert config.provider == "openai"
    assert config.api_key == "test-openai-key"
    assert config.model == "text-embedding-3-small"
