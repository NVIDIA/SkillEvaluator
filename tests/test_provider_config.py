# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public provider-configuration behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.provider_config import ProviderConfigurationError, resolve_embedding_provider, resolve_llm_provider

PROVIDER_CONTRACT = Path(__file__).parent / "fixtures" / "public_provider_contract.json"


def test_openai_provider_uses_standard_openai_credentials() -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
        }
    )

    assert config.provider == "openai"
    assert config.api_key == "test-openai-key"
    assert config.credential_env == "OPENAI_API_KEY"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.litellm_model.startswith("openai/")
    assert config.child_environment() == {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
    }


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
    assert config.credential_env == "NVIDIA_API_KEY"
    assert config.model == "openai/gpt-oss-120b"
    assert config.child_environment() == {"NVIDIA_API_KEY": "test-nvidia-key"}


def test_public_provider_matches_shared_contract_fixture() -> None:
    assert PROVIDER_CONTRACT.is_file(), "shared public-provider contract fixture is required"
    contract = json.loads(PROVIDER_CONTRACT.read_text(encoding="utf-8"))
    config = resolve_llm_provider({contract["credential_env"]: "nvapi-contract-test"})

    assert contract == {
        "contract_version": "2026-07-08.1",
        "provider": "nv_build",
        "credential_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "credential_owner": "operator",
        "allowed_editions": ["external", "internal"],
    }
    assert config.provider == contract["provider"]
    assert config.credential_env == contract["credential_env"]
    assert config.base_url == contract["base_url"]


def test_nvidia_build_ignores_unrelated_credential_names() -> None:
    legacy_name = "NVI" + "DIA" + "_INFERENCE_KEY"

    with pytest.raises(ProviderConfigurationError, match="NVIDIA_API_KEY"):
        resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "nv_build",
                legacy_name: "must-not-be-consumed",
            }
        )


def test_nvidia_build_is_inferred_from_its_public_key() -> None:
    config = resolve_llm_provider({"NVIDIA_API_KEY": "test-nvidia-key"})

    assert config.provider == "nv_build"


def test_nvidia_build_normalizes_the_key_forwarded_to_children() -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "nv_build",
            "NVIDIA_API_KEY": "  normalized-key\n",
        }
    )

    assert config.api_key == "normalized-key"
    assert config.child_environment() == {"NVIDIA_API_KEY": "normalized-key"}


def test_nvidia_build_endpoint_cannot_be_redirected() -> None:
    llm = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "nv_build",
            "SKILL_EVAL_LLM_BASE_URL": "https://redirect.example/v1",
            "NVIDIA_API_KEY": "test-nvidia-key",
        }
    )
    embedding = resolve_embedding_provider(
        {
            "SKILL_EVAL_EMBEDDING_PROVIDER": "nv_build",
            "SKILL_EVAL_EMBEDDING_BASE_URL": "https://redirect.example/v1",
            "NVIDIA_API_KEY": "test-nvidia-key",
        }
    )

    assert llm.base_url == "https://integrate.api.nvidia.com/v1"
    assert embedding.base_url == "https://integrate.api.nvidia.com/v1"


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
    assert config.credential_env == "OPENAI_API_KEY"
    assert config.model == "text-embedding-3-small"


def test_openai_compatible_embedding_child_environment_preserves_embedding_variables() -> None:
    config = resolve_embedding_provider(
        {
            "SKILL_EVAL_EMBEDDING_PROVIDER": "openai-compatible",
            "SKILL_EVAL_EMBEDDING_MODEL": "local-embedding-model",
            "SKILL_EVAL_EMBEDDING_API_KEY": "local-embedding-key",
            "SKILL_EVAL_EMBEDDING_BASE_URL": "http://localhost:11434/v1",
            "SKILL_EVAL_LLM_API_KEY": "unused-llm-key",
            "SKILL_EVAL_LLM_BASE_URL": "http://localhost:9000/v1",
        }
    )

    assert config.child_environment() == {
        "SKILL_EVAL_EMBEDDING_API_KEY": "local-embedding-key",
        "SKILL_EVAL_EMBEDDING_BASE_URL": "http://localhost:11434/v1",
    }


def test_openai_compatible_embedding_fallback_key_does_not_reassign_chat_endpoint() -> None:
    config = resolve_embedding_provider(
        {
            "SKILL_EVAL_EMBEDDING_PROVIDER": "openai-compatible",
            "SKILL_EVAL_EMBEDDING_MODEL": "local-embedding-model",
            "SKILL_EVAL_EMBEDDING_BASE_URL": "http://localhost:11434/v1",
            "SKILL_EVAL_LLM_API_KEY": "fallback-llm-key",
            "SKILL_EVAL_LLM_BASE_URL": "http://localhost:9000/v1",
        }
    )

    assert config.child_environment() == {
        "SKILL_EVAL_LLM_API_KEY": "fallback-llm-key",
        "SKILL_EVAL_EMBEDDING_BASE_URL": "http://localhost:11434/v1",
    }


@pytest.mark.parametrize(
    ("provider", "credential_name"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("nv_build", "NVIDIA_API_KEY"),
    ],
)
def test_standard_provider_rejects_blank_explicit_chat_model(provider: str, credential_name: str) -> None:
    with pytest.raises(ProviderConfigurationError, match="SKILL_EVAL_LLM_MODEL"):
        resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": provider,
                "SKILL_EVAL_LLM_MODEL": "   ",
                credential_name: "test-key",
            }
        )


def test_explicit_chat_model_is_trimmed() -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "nv_build",
            "SKILL_EVAL_LLM_MODEL": "  openai/gpt-oss-120b  ",
            "NVIDIA_API_KEY": "test-key",
        }
    )

    assert config.model == "openai/gpt-oss-120b"
    assert config.litellm_model == "openai/openai/gpt-oss-120b"
