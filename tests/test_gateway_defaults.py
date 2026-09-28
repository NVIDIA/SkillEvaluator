# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A gateway needs only a provider, URL and key; model overrides remain exact."""

import pytest

from skillevaluator.provider_config import (
    ProviderConfigurationError,
    resolve_embedding_provider,
    resolve_llm_provider,
)
from skillevaluator.tier3.harbor.runner import _judge_model_config, _model_for_agent, _resolve_agent_runtime_plan


@pytest.fixture
def gateway(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "SKILL_EVAL_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return {
        "SKILL_EVAL_LLM_PROVIDER": "openai-compatible",
        "SKILL_EVAL_LLM_BASE_URL": "https://gateway.example/team/v1",
        "SKILL_EVAL_LLM_API_KEY": "gateway-key",
    }


def test_gateway_defaults_need_only_provider_url_and_key(gateway):
    chat = resolve_llm_provider(gateway)
    embedding = resolve_embedding_provider(gateway)
    assert chat.model == "nvidia/nvidia/nemotron-3-super-120b-long-ctx"
    assert embedding.model == "nvidia/nvidia/nemotron-3-embed-1b"
    assert chat.api_key == embedding.api_key == "gateway-key"
    assert chat.base_url == embedding.base_url == gateway["SKILL_EVAL_LLM_BASE_URL"]
    assert _judge_model_config(chat, {}, "default")["model"] == chat.model


def test_gateway_judge_override_precedence(gateway):
    provider = resolve_llm_provider({**gateway, "SKILL_EVAL_LLM_MODEL": "catalog/evaluator"})
    assert _judge_model_config(provider, {}, "default")["model"] == "catalog/evaluator"
    aliases = {"SKILL_EVAL_JUDGE_MODEL": "catalog/judge"}
    assert _judge_model_config(provider, aliases, "default")["model"] == "catalog/judge"
    aliases["LLM_JUDGE_MODEL"] = "catalog/priority-judge"
    assert _judge_model_config(provider, aliases, "default")["model"] == "catalog/priority-judge"
    assert _judge_model_config(provider, aliases, "custom_only") == {"enabled": False}


def test_gateway_separate_embedding_route_keeps_chat_settings(gateway):
    embedding = resolve_embedding_provider(
        {
            **gateway,
            "SKILL_EVAL_EMBEDDING_BASE_URL": "https://embeddings.example/tenant/v1",
            "SKILL_EVAL_EMBEDDING_API_KEY": "embedding-only-key",
        }
    )
    assert embedding.model == "nvidia/nvidia/nemotron-3-embed-1b"
    assert embedding.base_url == "https://embeddings.example/tenant/v1"
    assert embedding.api_key == "embedding-only-key"
    assert "SKILL_EVAL_LLM_API_KEY" not in embedding.child_environment()
    assert resolve_llm_provider(gateway).api_key == "gateway-key"


@pytest.mark.parametrize(
    "resolver,variable",
    [
        (resolve_llm_provider, "SKILL_EVAL_LLM_MODEL"),
        (resolve_embedding_provider, "SKILL_EVAL_EMBEDDING_MODEL"),
    ],
)
def test_gateway_preserves_catalog_model_overrides(gateway, resolver, variable):
    config = resolver({**gateway, variable: "publisher/catalog/exact-model"})
    assert config.model == "publisher/catalog/exact-model"
    assert config.litellm_model == "openai/publisher/catalog/exact-model"


@pytest.mark.parametrize(
    "resolver,variable",
    [
        (resolve_llm_provider, "SKILL_EVAL_LLM_MODEL"),
        (resolve_embedding_provider, "SKILL_EVAL_EMBEDDING_MODEL"),
    ],
)
def test_gateway_rejects_blank_model_overrides(gateway, resolver, variable):
    with pytest.raises(ProviderConfigurationError, match=variable + " must be a non-empty"):
        resolver({**gateway, variable: " "})


@pytest.mark.parametrize(
    "agent,expected",
    [
        ("codex", "openai/openai/gpt-5.6-sol"),
        ("claude-code", "aws/anthropic/bedrock-claude-opus-5"),
        ("opencode", "openai/nvidia/nvidia/nemotron-3-super-120b-long-ctx"),
    ],
)
def test_gateway_uses_harness_specific_defaults(gateway, agent, expected):
    provider = resolve_llm_provider({**gateway, "SKILL_EVAL_LLM_MODEL": "custom/evaluator"})
    assert _model_for_agent(agent, cli_model=None, config_agents={}, provider=provider) == (
        expected,
        "openai-compatible agent default",
    )
    assert _model_for_agent(agent, cli_model="exact/cli/model", config_agents={}, provider=provider) == (
        "exact/cli/model",
        "CLI",
    )
    assert _model_for_agent(
        agent,
        cli_model=None,
        config_agents={agent: {"model": "exact/config/model"}},
        provider=provider,
    ) == ("exact/config/model", "evals/config.yml")


def plan(gateway):
    provider = resolve_llm_provider(gateway)
    agents = ["codex", "claude-code", "opencode"]
    selected = {agent: _model_for_agent(agent, cli_model=None, config_agents={}, provider=provider) for agent in agents}
    return _resolve_agent_runtime_plan(
        provider=provider,
        agents=agents,
        models={a: v[0] for a, v in selected.items()},
        model_sources={a: v[1] for a, v in selected.items()},
        configured_runtime_env={},
        env_mode="docker",
    )


def test_gateway_shared_credentials_are_scoped_per_agent(gateway):
    plans = plan(gateway)
    claude = plans["claude-code"]
    assert claude.provider.api_key == "gateway-key"
    assert claude.provider.base_url == "https://gateway.example/team"
    assert claude.subprocess_env["ANTHROPIC_API_KEY"] == "gateway-key"
    assert claude.subprocess_env["ANTHROPIC_BASE_URL"] == "https://gateway.example/team"
    assert set(claude.staged_env) == {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"}
    for agent in ("codex", "opencode"):
        assert plans[agent].provider.base_url == gateway["SKILL_EVAL_LLM_BASE_URL"]
        assert set(plans[agent].staged_env) == {"OPENAI_API_KEY", "OPENAI_BASE_URL"}


@pytest.mark.parametrize(
    "base,key,expected_base,expected_key",
    [
        (None, "native-key", None, "native-key"),
        ("https://claude.example/prefix/v1", "other-key", "https://claude.example/prefix", "other-key"),
        ("https://gateway.example/anthropic/v1", None, "https://gateway.example/anthropic", "gateway-key"),
    ],
)
def test_explicit_claude_route_takes_precedence(gateway, monkeypatch, base, key, expected_base, expected_key):
    if base:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", base)
    if key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    plans = plan(gateway)
    assert plans["claude-code"].provider.base_url == expected_base
    assert plans["claude-code"].provider.api_key == expected_key
    assert plans["codex"].provider.api_key == "gateway-key"


def test_independent_native_claude_key_uses_native_model_default(gateway, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "native-key")
    provider = resolve_llm_provider(gateway)
    assert _model_for_agent("claude-code", cli_model=None, config_agents={}, provider=provider) == (
        "claude-opus-5",
        "native Anthropic agent default",
    )
    assert _model_for_agent("claude-code", cli_model="exact/override", config_agents={}, provider=provider) == (
        "exact/override",
        "CLI",
    )


def test_invalid_claude_gateway_root_is_rejected_before_execution(gateway):
    gateway["SKILL_EVAL_LLM_BASE_URL"] = "https://user:secret@gateway.example/v1"
    with pytest.raises(ProviderConfigurationError, match="SKILL_EVAL_LLM_BASE_URL") as exc:
        plan(gateway)
    assert "secret" not in str(exc.value)
