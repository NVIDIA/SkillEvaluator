# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-aware model routing contracts for Harbor agent harnesses."""

from __future__ import annotations

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor.runner import (
    _OPERATOR_OWNED_AGENT_ENV,
    _independent_anthropic_agent_credentials,
    _is_operator_owned_runtime_name,
    _model_for_agent,
    _validate_agent_provider_credentials,
)


def _provider(name: str, model: str) -> ProviderConfig:
    base_url = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "nv_build": "https://integrate.api.nvidia.com/v1",
    }[name]
    prefix = "anthropic" if name == "anthropic" else "openai"
    return ProviderConfig(
        provider=name,
        model=model,
        api_key="provider-key",
        base_url=base_url,
        litellm_model=f"{prefix}/{model}",
    )


@pytest.mark.parametrize(
    ("provider", "agent", "expected"),
    [
        (_provider("openai", "gpt-5.5"), "codex", "gpt-5.5"),
        (_provider("openai", "gpt-5.5"), "opencode", "openai/gpt-5.5"),
        (_provider("anthropic", "claude-opus-5"), "claude-code", "claude-opus-5"),
        (
            _provider("nv_build", "nvidia/nemotron-3-nano-30b-a3b"),
            "opencode",
            "nvidia/nvidia/nemotron-3-nano-30b-a3b",
        ),
        (
            _provider("nv_build", "nvidia/nemotron-3-nano-30b-a3b"),
            "codex",
            "nvidia/nemotron-3-super-120b-a12b",
        ),
        (
            _provider("nv_build", "nvidia/nemotron-3-nano-30b-a3b"),
            "claude-code",
            "nvidia/nemotron-3-super-120b-a12b",
        ),
    ],
)
def test_public_provider_defaults_are_normalized_for_each_agent(
    provider: ProviderConfig,
    agent: str,
    expected: str,
) -> None:
    assert _model_for_agent(agent, cli_model=None, config_agents={}, provider=provider) == (
        expected,
        "public provider default",
    )


@pytest.mark.parametrize(
    ("cli_model", "config_agents", "expected", "source"),
    [
        ("vendor/raw-model", {}, "vendor/raw-model", "CLI"),
        (None, {"opencode": {"model": "custom/exact-model"}}, "custom/exact-model", "evals/config.yml"),
    ],
)
def test_explicit_opencode_models_are_not_rewritten(
    cli_model: str | None,
    config_agents: dict,
    expected: str,
    source: str,
) -> None:
    assert _model_for_agent(
        "opencode",
        cli_model=cli_model,
        config_agents=config_agents,
        provider=_provider("nv_build", "nvidia/nemotron-3-nano-30b-a3b"),
    ) == (expected, source)


@pytest.mark.parametrize("agent", ["codex", "claude-code"])
def test_docker_build_bridges_accept_the_provider_default(agent: str) -> None:
    assert (
        _validate_agent_provider_credentials(
            _provider("nv_build", "nvidia/nemotron-3-nano-30b-a3b"),
            [agent],
            {},
            {agent: "public provider default"},
            env_mode="docker",
            agent_models={agent: "nvidia/nemotron-3-nano-30b-a3b"},
        )
        == []
    )


@pytest.mark.parametrize("env_mode", ["docker", "local"])
@pytest.mark.parametrize(
    ("agent", "model"),
    [
        ("opencode", "nvidia/gpt-5"),
        ("codex", "gpt-5"),
        ("claude-code", "claude-opus-4-8"),
    ],
)
def test_build_agents_reject_native_model_names_that_would_be_misrouted(
    env_mode: str,
    agent: str,
    model: str,
) -> None:
    errors = _validate_agent_provider_credentials(
        _provider("nv_build", "nvidia/nemotron-3-nano-30b-a3b"),
        [agent],
        {},
        {agent: "CLI"},
        env_mode=env_mode,
        agent_models={agent: model},
    )

    assert errors and "NVIDIA Build catalog model ID" in errors[0]
    assert "publisher/model" in errors[0]


@pytest.mark.parametrize(
    ("agent", "runtime_env", "expected_fragment"),
    [
        ("codex", {}, "OPENAI_API_KEY + OPENAI_BASE_URL"),
        ("claude-code", {}, "ANTHROPIC_API_KEY"),
    ],
)
def test_cloud_build_vendor_clis_require_independent_native_credentials(
    agent: str,
    runtime_env: dict[str, str],
    expected_fragment: str,
) -> None:
    errors = _validate_agent_provider_credentials(
        _provider("nv_build", "nvidia/nemotron-3-nano-30b-a3b"),
        [agent],
        runtime_env,
        {agent: "public provider default"},
        env_mode="e2b",
    )
    assert errors and expected_fragment in errors[0]


def test_openai_provider_rejects_claude_without_an_independent_credential() -> None:
    errors = _validate_agent_provider_credentials(
        _provider("openai", "gpt-5.5"),
        ["claude-code"],
        {},
        {"claude-code": "public provider default"},
        env_mode="docker",
    )
    assert errors == [
        "claude-code with the OpenAI evaluator provider requires an independent ANTHROPIC_API_KEY "
        "in the operator host environment."
    ]


def test_openai_provider_rejects_the_default_gpt_model_for_claude() -> None:
    errors = _validate_agent_provider_credentials(
        _provider("openai", "gpt-5.5"),
        ["claude-code"],
        {"ANTHROPIC_API_KEY": "anthropic-key"},
        {"claude-code": "public provider default"},
        env_mode="docker",
    )
    assert errors and "needs an explicit Anthropic model" in errors[0]


def test_openai_provider_accepts_an_explicit_independent_claude_route() -> None:
    assert (
        _validate_agent_provider_credentials(
            _provider("openai", "gpt-5.5"),
            ["claude-code"],
            {"ANTHROPIC_API_KEY": "anthropic-key"},
            {"claude-code": "CLI"},
            env_mode="docker",
        )
        == []
    )


def test_openai_provider_accepts_mixed_routes_for_per_agent_isolation() -> None:
    assert (
        _validate_agent_provider_credentials(
            _provider("openai", "gpt-5.5"),
            ["codex", "claude-code"],
            {"ANTHROPIC_API_KEY": "anthropic-key"},
            {"codex": "public provider default", "claude-code": "CLI"},
            env_mode="docker",
        )
        == []
    )


def test_openai_provider_rejects_an_anthropic_opencode_override() -> None:
    errors = _validate_agent_provider_credentials(
        _provider("openai", "gpt-5.5"),
        ["codex", "opencode"],
        {"ANTHROPIC_API_KEY": "anthropic-key"},
        {"codex": "public provider default", "opencode": "CLI"},
        env_mode="docker",
        agent_models={"codex": "gpt-5.5", "opencode": "anthropic/claude-opus-4-8"},
    )
    assert errors and "must match the evaluator provider" in errors[0]
    assert "ANTHROPIC_API_KEY" not in errors[0]


def test_anthropic_provider_rejects_default_raw_model_for_opencode() -> None:
    errors = _validate_agent_provider_credentials(
        _provider("anthropic", "claude-opus-4-8"),
        ["opencode"],
        {},
        {"opencode": "public provider default"},
        env_mode="docker",
    )
    assert errors and "explicit provider-qualified model" in errors[0]


@pytest.mark.parametrize(
    ("runtime_env", "model_source", "expected_fragment"),
    [
        ({}, "public provider default", "OPENAI_API_KEY and OPENAI_BASE_URL"),
        ({"OPENAI_API_KEY": "openai-key"}, "public provider default", "OPENAI_API_KEY and OPENAI_BASE_URL"),
        (
            {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
            "public provider default",
            "explicit OpenAI-compatible model",
        ),
    ],
)
def test_anthropic_provider_rejects_incomplete_or_default_codex_routes(
    runtime_env: dict[str, str],
    model_source: str,
    expected_fragment: str,
) -> None:
    errors = _validate_agent_provider_credentials(
        _provider("anthropic", "claude-opus-4-8"),
        ["codex"],
        runtime_env,
        {"codex": model_source},
        env_mode="docker",
    )
    assert errors and expected_fragment in errors[0]


def test_anthropic_provider_accepts_explicit_codex_and_opencode_routes() -> None:
    assert (
        _validate_agent_provider_credentials(
            _provider("anthropic", "claude-opus-4-8"),
            ["codex", "opencode"],
            {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
            {"codex": "CLI", "opencode": "CLI"},
            env_mode="docker",
            agent_models={"codex": "gpt-5.5", "opencode": "anthropic/claude-opus-4-8"},
        )
        == []
    )


def test_anthropic_provider_accepts_explicit_single_agent_routes() -> None:
    provider = _provider("anthropic", "claude-opus-4-8")
    assert (
        _validate_agent_provider_credentials(
            provider,
            ["codex"],
            {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
            {"codex": "CLI"},
            env_mode="docker",
        )
        == []
    )
    assert (
        _validate_agent_provider_credentials(
            provider,
            ["opencode"],
            {},
            {"opencode": "CLI"},
            env_mode="docker",
            agent_models={"opencode": "anthropic/claude-opus-4-8"},
        )
        == []
    )


def test_anthropic_provider_rejects_an_openai_opencode_override() -> None:
    errors = _validate_agent_provider_credentials(
        _provider("anthropic", "claude-opus-4-8"),
        ["claude-code", "opencode"],
        {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
        {"claude-code": "public provider default", "opencode": "CLI"},
        env_mode="docker",
        agent_models={"claude-code": "claude-opus-4-8", "opencode": "openai/gpt-5.5"},
    )
    assert errors and "must match the evaluator provider" in errors[0]
    assert "OPENAI_API_KEY" not in errors[0]


def test_openai_provider_accepts_claude_with_vertex_backend() -> None:
    assert (
        _validate_agent_provider_credentials(
            _provider("openai", "gpt-5.5"),
            ["claude-code"],
            {"CLAUDE_CODE_USE_VERTEX": "1", "ANTHROPIC_VERTEX_PROJECT_ID": "test-project"},
            {"claude-code": "CLI"},
            env_mode="gke",
        )
        == []
    )


def test_openai_provider_rejects_default_gpt_model_for_claude_with_vertex() -> None:
    errors = _validate_agent_provider_credentials(
        _provider("openai", "gpt-5.5"),
        ["claude-code"],
        {"CLAUDE_CODE_USE_VERTEX": "1", "ANTHROPIC_VERTEX_PROJECT_ID": "test-project"},
        {"claude-code": "public provider default"},
        env_mode="gke",
    )
    assert errors and "needs an explicit Anthropic model" in errors[0]


def test_independent_anthropic_agent_credentials_resolves_vertex_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-project")
    monkeypatch.setenv("CLOUD_ML_REGION", "us-central1")

    creds = _independent_anthropic_agent_credentials()
    assert creds.get("CLAUDE_CODE_USE_VERTEX") == "1"
    assert creds.get("ANTHROPIC_VERTEX_PROJECT_ID") == "test-project"
    assert creds.get("CLOUD_ML_REGION") == "us-central1"
    assert "ANTHROPIC_API_KEY" not in creds


def test_operator_owned_agent_env_includes_vertex_variables() -> None:
    assert "CLAUDE_CODE_USE_VERTEX" in _OPERATOR_OWNED_AGENT_ENV
    assert "ANTHROPIC_VERTEX_PROJECT_ID" in _OPERATOR_OWNED_AGENT_ENV
    assert "GOOGLE_CLOUD_PROJECT" in _OPERATOR_OWNED_AGENT_ENV
    assert "GCP_PROJECT" in _OPERATOR_OWNED_AGENT_ENV
    assert "CLOUD_ML_REGION" in _OPERATOR_OWNED_AGENT_ENV
    assert "GOOGLE_APPLICATION_CREDENTIALS" in _OPERATOR_OWNED_AGENT_ENV
    assert _is_operator_owned_runtime_name("CLAUDE_CODE_USE_VERTEX")
    assert _is_operator_owned_runtime_name("ANTHROPIC_VERTEX_PROJECT_ID")
    assert _is_operator_owned_runtime_name("GOOGLE_CLOUD_PROJECT")
    assert _is_operator_owned_runtime_name("GCP_PROJECT")
    assert _is_operator_owned_runtime_name("CLOUD_ML_REGION")
    assert _is_operator_owned_runtime_name("GOOGLE_APPLICATION_CREDENTIALS")


def test_independent_anthropic_agent_credentials_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to GOOGLE_CLOUD_PROJECT, default region us-east5, strip GAC in GKE mode, strip API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)
    monkeypatch.delenv("GCP_REGION", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fallback-gcp-proj")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/host/key.json")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")

    # In GKE mode:
    creds_gke = _independent_anthropic_agent_credentials(env_mode="gke")
    assert creds_gke["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert creds_gke["ANTHROPIC_VERTEX_PROJECT_ID"] == "fallback-gcp-proj"
    assert creds_gke["CLOUD_ML_REGION"] == "us-east5"
    assert "ANTHROPIC_API_KEY" not in creds_gke
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in creds_gke

    # In non-GKE mode:
    creds_cloud = _independent_anthropic_agent_credentials(env_mode="cloud")
    assert creds_cloud.get("GOOGLE_APPLICATION_CREDENTIALS") == "/path/to/host/key.json"

