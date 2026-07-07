# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security regressions for the environment passed to the Harbor subprocess."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.adapter import _verifier_env_vars

if TYPE_CHECKING:
    import pytest


def _provider(provider: str = "openai") -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
        region="us-west-2" if provider == "bedrock" else None,
    )


def test_harbor_subprocess_environment_excludes_arbitrary_host_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    host_environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path / "tmp"),
        "LANG": "C.UTF-8",
        "DOCKER_HOST": "unix:///safe/docker.sock",
        "HOST_DATABASE_PASSWORD": "must-not-leak",
        "SSH_AUTH_SOCK": "/private/agent.sock",
        "ANTHROPIC_API_KEY": "unrelated-provider-secret",
        "E2B_API_KEY": "unselected-backend-secret",
    }
    monkeypatch.setattr(runner.os, "environ", host_environment)
    provider = _provider()
    provider_env = runner._provider_environment(provider)

    environment = runner._harbor_subprocess_environment(
        env_mode="docker",
        provider=provider,
        configured_runtime_env={"SERVICE_TOKEN": "declared-runtime-secret"},
        provider_env=provider_env,
    )

    assert environment == {
        "PATH": host_environment["PATH"],
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path / "tmp"),
        "LANG": "C.UTF-8",
        "DOCKER_HOST": "unix:///safe/docker.sock",
        "SKILLEVALUATOR_TELEMETRY_DISABLED": "true",
        "SERVICE_TOKEN": "declared-runtime-secret",
        **provider_env,
    }


def test_harbor_subprocess_environment_includes_only_selected_backend_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "E2B_API_KEY": "selected-key",
            "MODAL_TOKEN_ID": "unselected-id",
            "MODAL_TOKEN_SECRET": "unselected-secret",
        },
    )
    provider = _provider()

    environment = runner._harbor_subprocess_environment(
        env_mode="e2b",
        provider=provider,
        configured_runtime_env={},
        provider_env=runner._provider_environment(provider),
    )

    assert environment["E2B_API_KEY"] == "selected-key"
    assert "MODAL_TOKEN_ID" not in environment
    assert "MODAL_TOKEN_SECRET" not in environment


def test_daytona_subprocess_environment_preserves_jwt_auth_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "DAYTONA_JWT_TOKEN": "jwt-token",
            "DAYTONA_ORGANIZATION_ID": "organization-id",
        },
    )
    provider = _provider()

    environment = runner._harbor_subprocess_environment(
        env_mode="daytona",
        provider=provider,
        configured_runtime_env={},
        provider_env=runner._provider_environment(provider),
    )

    assert environment["DAYTONA_JWT_TOKEN"] == "jwt-token"
    assert environment["DAYTONA_ORGANIZATION_ID"] == "organization-id"


def test_config_declared_substitution_is_resolved_without_leaking_source_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {"PATH": "/usr/bin", "HOME": "/home/test", "HOST_AGENT_TOKEN": "runtime-key", "UNRELATED_SECRET": "no"},
    )
    configured_runtime_env, errors = runner._resolve_runtime_env({"AGENT_TOKEN": "${HOST_AGENT_TOKEN}"})
    provider = _provider()

    environment = runner._harbor_subprocess_environment(
        env_mode="docker",
        provider=provider,
        configured_runtime_env=configured_runtime_env,
        provider_env=runner._provider_environment(provider),
    )

    assert errors == []
    assert environment["AGENT_TOKEN"] == "runtime-key"
    assert "HOST_AGENT_TOKEN" not in environment
    assert "UNRELATED_SECRET" not in environment


def test_runtime_env_rejects_host_process_control_names() -> None:
    unsafe_names = (
        "PATH",
        "PYTHONPATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "BASH_ENV",
        "NODE_OPTIONS",
        "DOCKER_HOST",
        "COMPOSE_FILE",
        "HARBOR_HOME",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "HTTPS_PROXY",
        "E2B_API_KEY",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    )

    for name in unsafe_names:
        resolved, errors = runner._resolve_runtime_env({name: "attacker-controlled"})
        assert name not in resolved
        assert any(name in error and "host process" in error for error in errors)


def test_bedrock_subprocess_environment_keeps_only_explicit_aws_provider_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "AWS_ACCESS_KEY_ID": "access-key",
            "AWS_SECRET_ACCESS_KEY": "secret-key",
            "AWS_SESSION_TOKEN": "session-token",
            "AWS_PROFILE": "evaluation",
            "AWS_BEARER_TOKEN_BEDROCK": "bearer-token",
            "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/evaluator",
            "AWS_ROLE_SESSION_NAME": "evaluation-session",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/secrets/token",
            "OTHER_CLOUD_SECRET": "must-not-leak",
        },
    )
    provider = _provider("bedrock")
    provider_env = runner._provider_environment(provider)

    environment = runner._harbor_subprocess_environment(
        env_mode="docker",
        provider=provider,
        configured_runtime_env={},
        provider_env=provider_env,
    )

    assert provider_env["AWS_ACCESS_KEY_ID"] == "access-key"
    assert provider_env["AWS_BEARER_TOKEN_BEDROCK"] == "bearer-token"
    assert provider_env["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/var/run/secrets/token"
    assert {
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }.issubset(_verifier_env_vars(provider_env))
    assert environment["AWS_ACCESS_KEY_ID"] == "access-key"
    assert environment["AWS_SECRET_ACCESS_KEY"] == "secret-key"
    assert environment["AWS_SESSION_TOKEN"] == "session-token"
    assert environment["AWS_PROFILE"] == "evaluation"
    assert environment["AWS_BEARER_TOKEN_BEDROCK"] == "bearer-token"
    assert environment["AWS_ROLE_ARN"].endswith(":role/evaluator")
    assert environment["AWS_ROLE_SESSION_NAME"] == "evaluation-session"
    assert environment["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/var/run/secrets/token"
    assert "OTHER_CLOUD_SECRET" not in environment


def test_local_nvidia_provider_preserves_explicit_codex_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": str(tmp_path),
            "SKILLEVALUATOR_RUNTIME_DIR": str(tmp_path / "runtimes"),
        },
    )
    provider = _provider("nv_build")

    environment = runner._harbor_subprocess_environment(
        env_mode="local",
        provider=provider,
        configured_runtime_env={
            "OPENAI_API_KEY": "codex-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
        provider_env=runner._provider_environment(provider),
    )

    assert environment["NVIDIA_API_KEY"] == "provider-key"
    assert environment["OPENAI_API_KEY"] == "codex-key"
    assert environment["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
