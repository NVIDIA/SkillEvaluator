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


def test_openai_provider_uses_gpt_5_5_by_default() -> None:
    config = resolve_llm_provider({"SKILL_EVAL_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-openai-key"})

    assert config.model == "gpt-5.6-sol"
    assert config.litellm_model == "openai/gpt-5.6-sol"


def test_anthropic_provider_uses_claude_opus_5_by_default() -> None:
    config = resolve_llm_provider({"SKILL_EVAL_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-anthropic-key"})

    assert config.model == "claude-opus-5"
    assert config.litellm_model == "anthropic/claude-opus-5"


def test_bedrock_provider_uses_claude_opus_5_us_profile_by_default() -> None:
    config = resolve_llm_provider({"SKILL_EVAL_LLM_PROVIDER": "bedrock"})

    assert config.model == "us.anthropic.claude-opus-5"
    assert config.litellm_model == "bedrock/us.anthropic.claude-opus-5"


def test_chat_default_models_are_the_single_source_of_truth() -> None:
    from skillevaluator.constants import DIMENSION_JUDGE_MODEL
    from skillevaluator.provider_config import (
        CHAT_CHEAP_OPENAI,
        CHAT_DEFAULT_ANTHROPIC,
        CHAT_DEFAULT_BEDROCK,
        CHAT_DEFAULT_MODELS,
        CHAT_DEFAULT_OPENAI,
    )
    from skillevaluator.tier3.eval_core.llm_judge import DEFAULT_JUDGE_MODEL

    assert CHAT_DEFAULT_MODELS["openai"] == CHAT_DEFAULT_OPENAI == "gpt-5.6-sol"
    assert CHAT_DEFAULT_MODELS["anthropic"] == CHAT_DEFAULT_ANTHROPIC == "claude-opus-5"
    assert CHAT_DEFAULT_MODELS["bedrock"] == CHAT_DEFAULT_BEDROCK == "us.anthropic.claude-opus-5"
    assert CHAT_CHEAP_OPENAI == "gpt-5.4-mini"
    assert DIMENSION_JUDGE_MODEL == CHAT_DEFAULT_OPENAI
    assert DEFAULT_JUDGE_MODEL == CHAT_DEFAULT_OPENAI


@pytest.mark.parametrize(
    ("configured_base_url", "expected_base_url"),
    [
        ("https://gateway.example", "https://gateway.example"),
        ("https://gateway.example/", "https://gateway.example"),
        ("https://gateway.example/v1", "https://gateway.example"),
        ("https://gateway.example/v1/", "https://gateway.example"),
        ("https://gateway.example/team/v1/", "https://gateway.example/team"),
        ("https://localhost/v1", "https://localhost"),
        ("http://gateway.internal:8080/v1", "http://gateway.internal:8080"),
        ("http://anthropic_proxy:8000/v1", "http://anthropic_proxy:8000"),
        ("https://xn--bcher-kva.example:8443/team/v1", "https://xn--bcher-kva.example:8443/team"),
        ("https://bücher.example/v1", "https://xn--bcher-kva.example"),
        ("https://faß.de/v1", "https://xn--fa-hia.de"),
        ("https://οδός.example/v1", "https://xn--pxavk3b.example"),
        ("https://bücher.example.:8443/team/v1", "https://xn--bcher-kva.example.:8443/team"),
        ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080"),
        ("http://[::1]:8080/team/v1", "http://[::1]:8080/team"),
        ("http://[fe80::1%25eth0]:8080/team/v1", "http://[fe80::1%25eth0]:8080/team"),
        ("http://127.0.0.1:8080/bücher/v1", "http://127.0.0.1:8080/b%C3%BCcher"),
        ("https://gateway.example/caf%C3%A9/v1", "https://gateway.example/caf%C3%A9"),
        ("https://gateway.example/caf%c3%a9/v1", "https://gateway.example/caf%C3%A9"),
        ("https://gateway.example/opaque%ff/v1", "https://gateway.example/opaque%FF"),
        ("https://gateway.example/tenant%25west/v1", "https://gateway.example/tenant%25west"),
        ("https://gateway.example/100%25/v1", "https://gateway.example/100%25"),
        ("https://gateway.example/team/%76%31", "https://gateway.example/team"),
        ("https://gateway.example/team/v1///", "https://gateway.example/team"),
        (
            "https://gateway.example/teams;v=1/@me+you/v1",
            "https://gateway.example/teams;v=1/@me+you",
        ),
    ],
)
def test_anthropic_base_url_is_normalized_to_an_unversioned_root(
    configured_base_url: str,
    expected_base_url: str,
) -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "ANTHROPIC_BASE_URL": configured_base_url,
        }
    )

    assert config.base_url == expected_base_url


def test_anthropic_base_url_remains_unset_for_the_official_default() -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        }
    )

    assert config.base_url is None
    assert config.child_environment() == {"ANTHROPIC_API_KEY": "test-anthropic-key"}


def test_anthropic_skill_eval_base_url_takes_precedence_and_is_normalized() -> None:
    config = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "SKILL_EVAL_LLM_BASE_URL": "https://selected.example/team/v1/",
            "ANTHROPIC_BASE_URL": "https://ignored.example/v1",
        }
    )

    assert config.base_url == "https://selected.example/team"
    assert config.child_environment()["ANTHROPIC_BASE_URL"] == "https://selected.example/team"


@pytest.mark.parametrize(
    "configured_base_url",
    [
        "gateway.example/v1",
        "file:///etc/passwd",
        "https:///team/v1",
        "https://:443/v1",
        "https://url-user:url-secret@gateway.example/v1",
        "https://gateway.example/v1?token=url-secret",
        "https://gateway.example/v1#token=url-secret",
        "https://gateway.example/v1/messages",
        "https://gateway.example\\team\\v1",
        "https://gateway.example/v1\n",
        "https://gateway.example:not-a-port/v1",
        "https://gateway.example:/v1",
        "https://gateway example/v1",
        "https://gateway%2eexample/v1",
        "https://gateway.example|evil/v1",
        "https://999.1.1.1/v1",
        "https://[v1.not-ipv6]/v1",
        "https://gateway.example/%76%31/%6dessages",
        "https://gateway.example/team%2Fv1",
        "https://gateway.example/team%5cv1",
        "https://gateway.example/team%0av1",
        "https://gateway.example/team%",
        "https://gateway.example/team%2",
        "https://gateway.example/team%GG",
        "https://gateway.example/../team/v1",
        "https://gateway.example/%2e%2e/team/v1",
        "https://gateway.example/%2576%2531/%256dessages",
        "https://gateway.example/%252e%252e/team/v1",
        "https://gateway.example/team%252Fv1",
        "https://gateway.example/team%255Cv1",
        "https://gateway.example/team%250Av1",
        "https://gateway.example/%25%37%36%25%33%31/%25%36%64essages",
        "https://gateway.example/team%25%32%46v1",
        "https://gateway.example/team//v1",
        "https://☃.example/v1",
    ],
)
def test_anthropic_base_url_rejects_unsafe_or_non_root_values(configured_base_url: str) -> None:
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "ANTHROPIC_BASE_URL": configured_base_url,
            }
        )

    message = str(exc_info.value)
    assert "ANTHROPIC_BASE_URL" in message
    assert configured_base_url not in message
    assert "url-user" not in message
    assert "url-secret" not in message


def test_anthropic_base_url_error_names_the_selected_precedence_variable_without_credentials() -> None:
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "SKILL_EVAL_LLM_BASE_URL": "https://url-user:url-secret@selected.example/v1",
                "ANTHROPIC_BASE_URL": "https://valid-fallback.example/v1",
            }
        )

    message = str(exc_info.value)
    assert "SKILL_EVAL_LLM_BASE_URL" in message
    assert "ANTHROPIC_BASE_URL" not in message
    assert "url-user" not in message
    assert "url-secret" not in message


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
    assert config.model == "nvidia/nemotron-3-nano-30b-a3b"
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
    legacy_name = "NVI" + "DIA" + "_INFERENCE_KEY"  # oss-boundary-anchor: provider-retired-credential

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


@pytest.mark.parametrize(
    "credentials",
    [
        {"NVIDIA_API_KEY": "nvidia-key", "OPENAI_API_KEY": "openai-key"},
        {"NVIDIA_API_KEY": "nvidia-key", "ANTHROPIC_API_KEY": "anthropic-key"},
        {"OPENAI_API_KEY": "openai-key", "ANTHROPIC_API_KEY": "anthropic-key"},
    ],
)
def test_multiple_public_credentials_require_explicit_provider(credentials: dict[str, str]) -> None:
    with pytest.raises(ProviderConfigurationError, match=r"SKILL_EVAL_LLM_PROVIDER.*multiple"):
        resolve_llm_provider(credentials)


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
    with pytest.raises(ProviderConfigurationError, match="SKILL_EVAL_EMBEDDING_PROVIDER"):
        resolve_embedding_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
            }
        )


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


def test_no_credential_llm_error_lists_every_setup_option() -> None:
    # The error is the setup documentation for a fresh environment: every
    # variable it names must actually work as claimed.
    with pytest.raises(ProviderConfigurationError) as err:
        resolve_llm_provider({})
    message = str(err.value)
    assert "NVIDIA_API_KEY" in message
    assert "build.nvidia.com" in message
    assert "OPENAI_API_KEY" in message
    assert "ANTHROPIC_API_KEY" in message
    # openai-compatible needs all three extras, including the model.
    assert "SKILL_EVAL_LLM_BASE_URL" in message
    assert "SKILL_EVAL_LLM_API_KEY" in message
    assert "SKILL_EVAL_LLM_MODEL" in message


def test_no_credential_embedding_error_omits_non_embedding_providers() -> None:
    # Anthropic/Bedrock are rejected by the embedding resolver, so the
    # no-credential message must not recommend them (or the ANTHROPIC key).
    with pytest.raises(ProviderConfigurationError) as err:
        resolve_embedding_provider({})
    message = str(err.value)
    assert "ANTHROPIC_API_KEY" not in message
    assert "anthropic" not in message
    assert "bedrock" not in message
    assert "NVIDIA_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert "SKILL_EVAL_EMBEDDING_MODEL" in message


def test_embedding_rejection_of_llm_only_providers_names_the_fix() -> None:
    with pytest.raises(ProviderConfigurationError, match=r"nv_build\|openai\|openai-compatible"):
        resolve_embedding_provider({"ANTHROPIC_API_KEY": "test-anthropic-key"})
