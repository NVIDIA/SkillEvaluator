# SPDX-License-Identifier: Apache-2.0
"""CI model routing keeps native models and provider credentials isolated."""

import pytest

from ci import run_tiers


def test_gateway_chat_and_build_embeddings_use_only_selected_credentials():
    child = run_tiers.child_environment(
        {
            "PATH": "/bin",
            "SKILL_EVAL_PROVIDER": "nv_build",
            "SKILL_EVAL_MODEL": "nvidia/build-model",
            "SKILL_EVAL_TIER3_PROVIDER": "openai-compatible",
            "SKILL_EVAL_TIER3_MODEL": "gateway/judge",
            "SKILL_EVAL_TIER3_BASE_URL": "https://gateway.example/api/v1/",
            "SKILL_EVAL_LLM_API_KEY": "gateway-secret",
            "NVIDIA_API_KEY": "embedding-secret",
            "OPENAI_API_KEY": "ambient-secret",
            "ANTHROPIC_API_KEY": "ambient-secret",
            "OPENAI_BASE_URL": "https://ambient.example/v1",
            "GITHUB_TOKEN": "feedback-secret",
            "CI_JOB_TOKEN": "job-secret",
        },
        tier=3,
    )
    assert child["SKILL_EVAL_LLM_PROVIDER"] == "openai-compatible"
    assert child["SKILL_EVAL_LLM_MODEL"] == "gateway/judge"
    assert child["LLM_JUDGE_MODEL"] == "gateway/judge"
    assert child["SKILL_EVAL_LLM_BASE_URL"] == "https://gateway.example/api/v1"
    assert child["SKILL_EVAL_LLM_API_KEY"] == "gateway-secret"
    assert not any(key.startswith("SKILL_EVAL_EMBEDDING_") for key in child)
    assert "NVIDIA_API_KEY" not in child
    assert not set(child) & {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "GITHUB_TOKEN", "CI_JOB_TOKEN"}


def test_changed_tier3_provider_does_not_inherit_build_model_or_api_root():
    from ci.model_config import model_environment

    child = model_environment(
        {
            "SKILL_EVAL_PROVIDER": "nv_build",
            "SKILL_EVAL_MODEL": "nvidia/build-model",
            "SKILL_EVAL_BASE_URL": "https://build.example/v1",
            "SKILL_EVAL_TIER3_PROVIDER": "openai",
            "SKILL_EVAL_TIER3_MODEL": "",
            "SKILL_EVAL_TIER3_BASE_URL": "",
        },
        3,
    )
    assert child["SKILL_EVAL_LLM_PROVIDER"] == "openai"
    assert "SKILL_EVAL_LLM_MODEL" not in child
    assert "LLM_JUDGE_MODEL" not in child
    assert "SKILL_EVAL_LLM_BASE_URL" not in child
    assert "OPENAI_BASE_URL" not in child


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_empty_overrides_inherit_same_provider_and_models(tier):
    from ci.model_config import model_environment

    child = model_environment(
        {
            "SKILL_EVAL_PROVIDER": "openai-compatible",
            "SKILL_EVAL_MODEL": "catalog/exact-model",
            "SKILL_EVAL_BASE_URL": "https://gateway.example/v1",
            "SKILL_EVAL_TIER3_PROVIDER": "",
            "SKILL_EVAL_TIER3_MODEL": "",
            "SKILL_EVAL_TIER3_BASE_URL": "",
            "SKILL_EVAL_EMBEDDING_PROVIDER": "openai-compatible",
            "SKILL_EVAL_EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "SKILL_EVAL_EMBEDDING_API_KEY": "embedding-secret",
        },
        tier,
    )
    assert child["SKILL_EVAL_LLM_MODEL"] == "catalog/exact-model"
    assert child["SKILL_EVAL_LLM_BASE_URL"] == "https://gateway.example/v1"
    if tier == 2:
        assert child["SKILL_EVAL_EMBEDDING_BASE_URL"] == "https://embedding.example/v1"
        assert child["SKILL_EVAL_EMBEDDING_API_KEY"] == "embedding-secret"
    else:
        assert not any(key.startswith("SKILL_EVAL_EMBEDDING_") for key in child)
    assert "SKILL_EVAL_LLM_API_KEY" not in child


@pytest.mark.parametrize(
    "url",
    [
        "relative/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/#secret",
        "https://example.com/\nsecret",
        "file:///secret",
    ],
)
def test_malformed_api_roots_have_fixed_secret_free_errors(url):
    from ci.model_config import model_environment

    with pytest.raises(ValueError) as error:
        model_environment({"SKILL_EVAL_PROVIDER": "openai-compatible", "SKILL_EVAL_BASE_URL": url}, 2)
    assert "secret" not in str(error.value)
    assert url not in str(error.value)


def test_unselected_gateway_credentials_and_settings_are_removed():
    from ci.model_config import model_environment

    child = model_environment(
        {
            "SKILL_EVAL_PROVIDER": "nv_build",
            "NVIDIA_API_KEY": "selected",
            "SKILL_EVAL_LLM_API_KEY": "ambient",
            "SKILL_EVAL_EMBEDDING_API_KEY": "ambient",
            "SKILL_EVAL_EMBEDDING_BASE_URL": "invalid-ignored",
            "ANTHROPIC_BASE_URL": "invalid-ignored",
        },
        2,
    )
    assert child["NVIDIA_API_KEY"] == "selected"
    assert not set(child) & {
        "SKILL_EVAL_LLM_API_KEY",
        "SKILL_EVAL_EMBEDDING_API_KEY",
        "SKILL_EVAL_EMBEDDING_BASE_URL",
        "ANTHROPIC_BASE_URL",
    }


def test_agent_models_preserve_native_catalog_ids():
    from ci.model_config import agent_model_overrides

    assert agent_model_overrides(
        {
            "SKILL_EVAL_AGENT_MODELS": "claude-code=aws/anthropic/bedrock-claude-opus-5-5\ncodex=openai/openai/gpt-6.1-sol"
        }
    ) == (
        "claude-code=aws/anthropic/bedrock-claude-opus-5-5",
        "codex=openai/openai/gpt-6.1-sol",
    )
    assert agent_model_overrides({"SKILL_EVAL_AGENT_MODELS": "codex=Exact-ID,opencode=vendor/model"}) == (
        "codex=Exact-ID",
        "opencode=vendor/model",
    )


@pytest.mark.parametrize(
    "value", ["unknown=foo", "codex=foo,codex=bar", "codex=", "codex=foo\x1b[31m", "codex=" + "x" * 257, "x" * 4097]
)
def test_invalid_agent_models_are_rejected_without_echo(value):
    from ci.model_config import agent_model_overrides

    with pytest.raises(ValueError) as error:
        agent_model_overrides({"SKILL_EVAL_AGENT_MODELS": value})
    assert value not in str(error.value)


def test_tier1_specific_provider_uses_its_default_model():
    from ci.model_config import model_environment

    child = model_environment(
        {
            "SKILL_EVAL_PROVIDER": "nv_build",
            "SKILL_EVAL_MODEL": "tier2-model",
            "SKILL_EVAL_TIER1_PROVIDER": "openai",
            "SKILL_EVAL_TIER1_MODEL": "",
        },
        1,
    )
    assert child["SKILL_EVAL_LLM_PROVIDER"] == "openai"
    assert "SKILL_EVAL_LLM_MODEL" not in child


@pytest.mark.parametrize("tier", [1, 3])
@pytest.mark.parametrize(
    "embedding_settings",
    [
        {},
        {"SKILL_EVAL_EMBEDDING_PROVIDER": "unsupported"},
        {
            "SKILL_EVAL_EMBEDDING_PROVIDER": "openai-compatible",
            "SKILL_EVAL_EMBEDDING_BASE_URL": "invalid-unused-url",
            "SKILL_EVAL_EMBEDDING_MODEL": "invalid unused model",
            "SKILL_EVAL_EMBEDDING_API_KEY": "embedding-only-secret",
        },
    ],
)
def test_chat_only_tiers_do_not_require_or_forward_unused_embedding_routes(tier, embedding_settings):
    from ci.model_config import model_environment

    child = model_environment(
        {
            "SKILL_EVAL_PROVIDER": "openai-compatible",
            f"SKILL_EVAL_TIER{tier}_PROVIDER": "openai",
            f"SKILL_EVAL_TIER{tier}_MODEL": "",
            "OPENAI_API_KEY": "selected-chat-secret",
            **embedding_settings,
        },
        tier,
    )
    assert child["SKILL_EVAL_LLM_PROVIDER"] == "openai"
    assert child["OPENAI_API_KEY"] == "selected-chat-secret"
    assert not any(key.startswith("SKILL_EVAL_EMBEDDING_") for key in child)
    assert "SKILL_EVAL_LLM_API_KEY" not in child


def test_tier2_still_requires_a_valid_selected_embedding_route():
    from ci.model_config import model_environment

    with pytest.raises(ValueError, match="Selected provider configuration is incomplete or invalid"):
        model_environment({"SKILL_EVAL_PROVIDER": "openai", "SKILL_EVAL_EMBEDDING_PROVIDER": "openai-compatible"}, 2)


@pytest.mark.parametrize("tier", [1, 3])
def test_chat_only_tiers_keep_shared_chat_key_but_exclude_embedding_only_key(tier):
    from ci.model_config import model_environment

    child = model_environment(
        {
            "SKILL_EVAL_PROVIDER": "nv_build",
            "SKILL_EVAL_EMBEDDING_PROVIDER": "openai",
            "NVIDIA_API_KEY": "chat-key",
            "OPENAI_API_KEY": "embedding-only-key",
        },
        tier,
    )
    assert child["NVIDIA_API_KEY"] == "chat-key"
    assert "OPENAI_API_KEY" not in child
