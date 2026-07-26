# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security regressions for the environment passed to the Harbor subprocess."""

from __future__ import annotations

import os

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.adapter import _verifier_env_vars


def _provider(provider: str = "openai") -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
        region="us-west-2" if provider == "bedrock" else None,
    )


def test_docker_opencode_nv_build_uses_operator_key_without_skill_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider("nv_build")
    monkeypatch.setattr(runner.os, "environ", {"PATH": "/usr/bin", "NVIDIA_API_KEY": "nvapi-test"})

    plans = runner._resolve_agent_runtime_plan(
        provider=provider,
        agents=["opencode"],
        models={"opencode": "nvidia/meta/llama-3.1-8b-instruct"},
        configured_runtime_env={},
        env_mode="docker",
    )

    plan = plans["opencode"]
    assert plan.staged_env == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    assert plan.subprocess_env["NVIDIA_API_KEY"] == "provider-key"
    assert "OPENAI_API_KEY" not in plan.subprocess_env


@pytest.mark.parametrize(
    ("provider_name", "runtime_model", "catalog_model", "litellm_prefix"),
    [
        ("nv_build", "nvidia/meta/llama-3.1-8b-instruct", "meta/llama-3.1-8b-instruct", "openai"),
        ("openai", "openai/gpt-4.1-mini", "gpt-4.1-mini", "openai"),
        ("openai-compatible", "openai/custom-model", "custom-model", "openai"),
        ("anthropic", "anthropic/claude-sonnet-test", "claude-sonnet-test", "anthropic"),
    ],
)
def test_opencode_runtime_plan_separates_runtime_and_catalog_model_names(
    provider_name: str,
    runtime_model: str,
    catalog_model: str,
    litellm_prefix: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model=catalog_model,
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{litellm_prefix}/{catalog_model}",
    )

    plan = runner._resolve_agent_runtime_plan(
        provider=provider,
        agents=["opencode"],
        models={"opencode": runtime_model},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"opencode": "CLI"},
    )["opencode"]

    assert plan.model == runtime_model
    assert plan.provider.model == catalog_model
    assert plan.provider.litellm_model == f"{litellm_prefix}/{catalog_model}"


@pytest.mark.parametrize(
    ("provider_name", "agent", "model", "litellm_prefix"),
    [
        ("openai-compatible", "codex", "openai/gpt-oss", "openai"),
        ("anthropic", "claude-code", "anthropic/claude-custom", "anthropic"),
    ],
)
def test_non_opencode_runtime_plan_preserves_slash_prefixed_raw_model_ids(
    provider_name: str,
    agent: str,
    model: str,
    litellm_prefix: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model=model,
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{litellm_prefix}/{model}",
    )

    plan = runner._resolve_agent_runtime_plan(
        provider=provider,
        agents=[agent],
        models={agent: model},
        configured_runtime_env={},
        env_mode="docker",
    )[agent]

    assert plan.model == model
    assert plan.provider.model == model
    assert plan.provider.litellm_model == f"{litellm_prefix}/{model}"


@pytest.mark.parametrize("agent", ["codex", "claude-code"])
def test_nvidia_build_docker_bridge_plan_keeps_provider_key_out_of_task_env(agent: str) -> None:
    plan = runner._resolve_agent_runtime_plan(
        provider=_provider("nv_build"),
        agents=[agent],
        models={agent: "openai/gpt-oss-120b"},
        configured_runtime_env={},
        env_mode="docker",
    )[agent]

    assert plan.staged_env == {}
    assert plan.subprocess_env["NVIDIA_API_KEY"] == "provider-key"
    assert plan.provider.provider == "nv_build"
    assert plan.provider.model == "openai/gpt-oss-120b"


@pytest.mark.parametrize(
    "owned_name",
    [
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ],
)
def test_skill_config_cannot_override_operator_owned_agent_credentials(owned_name: str) -> None:
    with pytest.raises(ValueError, match=rf"operator-owned.*{owned_name}|{owned_name}.*operator-owned"):
        runner._resolve_agent_runtime_plan(
            provider=_provider("nv_build"),
            agents=["opencode"],
            models={"opencode": "nvidia/model"},
            configured_runtime_env={owned_name: "attacker-value"},
            env_mode="docker",
        )


@pytest.mark.parametrize(
    "source_name",
    [
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "AWS_SECRET_ACCESS_KEY",
        "E2B_API_KEY",
        "DOCKER_HOST",
    ],
)
def test_skill_config_cannot_alias_operator_owned_credentials(
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
) -> None:
    monkeypatch.setenv(source_name, "operator-secret")

    resolved, errors = runner._resolve_runtime_env({"INNOCENT_NAME": f"${{{source_name}}}"})

    assert resolved == {}
    assert errors and "operator-owned" in errors[0]
    assert source_name in errors[0]


@pytest.mark.parametrize(
    "name",
    ["SKILL_EVAL_LLM_BASE_URL", "SKILL_EVAL_LLM_PROVIDER", "SKILL_EVAL_LLM_API_KEY"],
)
def test_skill_config_cannot_control_evaluator_provider_routing(name: str) -> None:
    resolved, errors = runner._resolve_runtime_env({name: "https://attacker.example/v1"})

    assert resolved == {}
    assert errors and "host process" in errors[0]


def test_mixed_agents_receive_disjoint_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "NVIDIA_API_KEY": "nvapi-test",
            "ANTHROPIC_API_KEY": "anthropic-test",
            "OPENAI_API_KEY": "openai-test",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("nv_build"),
        agents=["opencode", "claude-code", "codex"],
        models={
            "opencode": "nvidia/nvidia/nemotron-3-nano-30b-a3b",
            "claude-code": "nvidia/nemotron-3-super-120b-a12b",
            "codex": "nvidia/nemotron-3-nano-30b-a3b",
        },
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"opencode": "provider", "claude-code": "CLI", "codex": "CLI"},
    )

    assert plans["opencode"].staged_env == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    assert plans["claude-code"].staged_env == {}
    assert plans["codex"].staged_env == {}
    assert plans["opencode"].provider.provider == "nv_build"
    assert plans["claude-code"].provider.provider == "nv_build"
    assert plans["codex"].provider.provider == "nv_build"
    assert "ANTHROPIC_API_KEY" not in plans["opencode"].subprocess_env
    assert "OPENAI_API_KEY" not in plans["opencode"].subprocess_env
    assert "NVIDIA_API_KEY" in plans["claude-code"].subprocess_env  # evaluator verifier only
    assert "OPENAI_API_KEY" not in plans["claude-code"].subprocess_env
    assert "ANTHROPIC_API_KEY" not in plans["codex"].subprocess_env
    with pytest.raises(TypeError):
        plans["opencode"].staged_env["MUTATE"] = "forbidden"  # type: ignore[index]


@pytest.mark.parametrize(
    ("provider_name", "agent"),
    [
        ("bedrock", "codex"),
        ("bedrock", "opencode"),
    ],
)
def test_incompatible_provider_agent_pairs_fail_before_harbor(provider_name: str, agent: str) -> None:
    with pytest.raises(ValueError, match="does not support"):
        runner._resolve_agent_runtime_plan(
            provider=_provider(provider_name),
            agents=[agent],
            models={agent: "test-model"},
            configured_runtime_env={},
            env_mode="docker",
            model_sources={agent: "CLI"},
        )


def test_openai_mixed_agents_receive_isolated_native_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "anthropic-agent-key",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("openai"),
        agents=["codex", "claude-code"],
        models={"codex": "test-model", "claude-code": "anthropic/claude-sonnet-4-5"},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"codex": "public provider default", "claude-code": "CLI"},
    )

    assert plans["codex"].staged_env == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    }
    assert plans["claude-code"].staged_env == {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
    assert plans["codex"].provider.provider == "openai"
    assert plans["claude-code"].provider.provider == "anthropic"
    assert plans["claude-code"].provider.model == "claude-sonnet-4-5"
    assert "ANTHROPIC_API_KEY" not in plans["codex"].subprocess_env
    assert plans["claude-code"].subprocess_env["ANTHROPIC_API_KEY"] == "anthropic-agent-key"
    assert plans["claude-code"].subprocess_env["OPENAI_API_KEY"] == "provider-key"


def test_anthropic_mixed_agents_receive_isolated_native_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "openai-agent-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("anthropic"),
        agents=["claude-code", "codex"],
        models={"claude-code": "test-model", "codex": "openai/gpt-5.4-mini"},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"claude-code": "public provider default", "codex": "CLI"},
    )

    assert plans["claude-code"].staged_env == {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL}",
    }
    assert plans["codex"].staged_env == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    }
    assert plans["claude-code"].provider.provider == "anthropic"
    assert plans["codex"].provider.provider == "openai-compatible"
    assert plans["codex"].provider.model == "gpt-5.4-mini"
    assert "OPENAI_API_KEY" not in plans["claude-code"].subprocess_env
    assert plans["codex"].subprocess_env["OPENAI_API_KEY"] == "openai-agent-key"
    assert plans["codex"].subprocess_env["ANTHROPIC_API_KEY"] == "provider-key"


def test_openai_compatible_codex_stages_selected_provider_pair() -> None:
    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("openai-compatible"),
        agents=["codex"],
        models={"codex": "responses-model"},
        configured_runtime_env={},
        env_mode="docker",
    )

    assert plans["codex"].staged_env == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    }


def test_local_anthropic_opencode_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="local mode"):
        runner._resolve_agent_runtime_plan(
            provider=_provider("anthropic"),
            agents=["opencode"],
            models={"opencode": "anthropic/claude-test"},
            configured_runtime_env={},
            env_mode="local",
        )


def test_local_bedrock_agent_is_rejected_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    with pytest.raises(ValueError, match="local mode"):
        runner._resolve_agent_runtime_plan(
            provider=_provider("bedrock"),
            agents=["claude-code"],
            models={"claude-code": "us.anthropic.model"},
            configured_runtime_env={},
            env_mode="local",
        )


def test_docker_bedrock_claude_activates_bedrock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "AWS_ACCESS_KEY_ID": "access",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_REGION": "us-west-2",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("bedrock"),
        agents=["claude-code"],
        models={"claude-code": "us.anthropic.model"},
        configured_runtime_env={},
        env_mode="docker",
    )

    plan = plans["claude-code"]
    assert plan.staged_env["CLAUDE_CODE_USE_BEDROCK"] == "${CLAUDE_CODE_USE_BEDROCK}"
    assert plan.subprocess_env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert plan.staged_env["AWS_ACCESS_KEY_ID"] == "${AWS_ACCESS_KEY_ID}"


def test_run_harbor_eval_stages_per_agent_credential_trees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    provider = _provider("nv_build")
    emitted: list[tuple[str, bool, dict[str, str]]] = []
    launched: dict[str, tuple[str, str, dict[str, str]]] = {}

    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"task_source": "evals_json"}}, None),
    )
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    def emit(_skill, target, *, with_skill, runtime_env, **_kwargs):
        task = target / "case-001"
        task.mkdir(parents=True)
        emitted.append((str(target.relative_to(tmp_path / "results")), with_skill, dict(runtime_env)))
        return [task]

    def launch(**kwargs):
        launched[kwargs["agent"]] = (
            str(kwargs["with_skill"].relative_to(tmp_path / "results")),
            str(kwargs["baseline"].relative_to(tmp_path / "results")),
            dict(kwargs["run_env"]),
        )
        return []

    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    monkeypatch.setattr(runner, "_run_agent_pair", launch)
    monkeypatch.setattr(
        runner,
        "collect_harbor_results",
        lambda **_kwargs: {"execution_status": "complete", "execution_errors": [], "metrics": [], "agents": {}},
    )
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")
    result = runner.run_harbor_eval(
        skill,
        ["opencode", "claude-code"],
        agent_models={
            "opencode": "nvidia/nvidia/nemotron-3-nano-30b-a3b",
            "claude-code": "nvidia/nemotron-3-super-120b-a12b",
        },
        output_dir=tmp_path / "results",
        env_mode="docker",
        keep_harbor_jobs=True,
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert {(path.split("/")[-2], path.split("/")[-1], with_skill) for path, with_skill, _env in emitted} == {
        ("opencode", "with", True),
        ("opencode", "without", False),
        ("claude-code", "with", True),
        ("claude-code", "without", False),
    }
    staged = {(path.split("/")[-2], path.split("/")[-1]): env for path, _with_skill, env in emitted}
    assert staged[("opencode", "with")] == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    assert staged[("claude-code", "with")] == {}
    assert launched["opencode"][0].endswith("_harbor-tasks/opencode/with")
    assert launched["claude-code"][1].endswith("_harbor-tasks/claude-code/without")
    assert "ANTHROPIC_API_KEY" not in launched["opencode"][2]
    assert "OPENAI_API_KEY" not in launched["opencode"][2]
    assert launched["claude-code"][2]["NVIDIA_API_KEY"] == "provider-key"


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
