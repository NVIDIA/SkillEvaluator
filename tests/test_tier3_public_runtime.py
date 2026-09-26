# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Tier 3 runtime boundaries."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.model_catalog import ModelCatalogFailureKind
from skillevaluator.provider_config import ProviderConfig, resolve_llm_provider
from skillevaluator.tier3 import commands as tier3_commands
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.harbor.adapter import (
    _EVALUATOR_MANAGED_RUNTIME_ENV,
    _generate_harbor_tasks_into,
    _stage_native_harbor_tasks_into,
    _write_task_toml,
    generate_harbor_tasks,
    stage_native_harbor_tasks,
)
from skillevaluator.tier3.harbor.runner import (
    _model_for_agent,
    _nvidia_build_agent_import_path,
    _provider_environment,
    _validate_agent_provider_credentials,
    build_harbor_run_command,
)
from skillevaluator.tier3.harbor.runtime_preflight import ModelProbeResult


def _load_verifier_template():
    template_path = Path(__file__).resolve().parents[1] / "src/skillevaluator/tier3/harbor/templates/eval.py"
    spec = importlib.util.spec_from_file_location("skillevaluator_public_verifier_template", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_eval_exposes_only_harbor_native_environments() -> None:
    result = CliRunner().invoke(cli, ["evaluate", "--help"])

    assert result.exit_code == 0
    assert "docker" in result.output
    assert "e2b" in result.output
    assert "modal" in result.output
    assert "harbor-environment" not in result.output
    assert "k8s-sandbox" not in result.output
    assert "local" in result.output
    assert "base-image-mode" not in result.output
    assert "--agent-runtime-preflight" in result.output


def test_public_config_accepts_runtime_controls(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(
        "schema_version: 1\n"
        "harbor:\n"
        "  base_image_mode: rebuild\n"
        "  n_attempts: 3\n"
        "  stop_on_pass: true\n"
        "  agent_runtime_preflight: false\n",
        encoding="utf-8",
    )

    config, _ = load_evals_config(tmp_path)

    assert config["harbor"]["base_image_mode"] == "rebuild"
    assert config["harbor"]["stop_on_pass"] is True
    assert config["harbor"]["agent_runtime_preflight"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [("base_image_mode", "sometimes"), ("stop_on_pass", "'yes'"), ("agent_runtime_preflight", "1")],
)
def test_public_config_validates_runtime_control_values(tmp_path: Path, key: str, value: str) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(f"schema_version: 1\nharbor:\n  {key}: {value}\n", encoding="utf-8")

    with pytest.raises(EvalsConfigError, match=rf"harbor\.{key}"):
        load_evals_config(tmp_path)


def test_public_config_still_rejects_sandbox_policy(tmp_path: Path) -> None:
    """The public engine has no consumer for a config-level sandbox policy."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  sandbox:\n    template: harbor-eval\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalsConfigError, match="unknown harbor key"):
        load_evals_config(tmp_path)


def test_native_environment_is_forwarded_to_harbor() -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="public-env-test",
        env_mode="e2b",
    )

    assert command[1] == "run"
    assert command[command.index("--env") + 1] == "e2b"
    assert "--environment-import-path" not in command


def test_judge_model_overrides_are_forwarded_only_as_harbor_verifier_env() -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="judge-model-test",
        env_mode="docker",
        verifier_env={
            "LLM_JUDGE_MODEL": "${LLM_JUDGE_MODEL}",
            "SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}",
        },
    )

    forwarded = [command[index + 1] for index, value in enumerate(command) if value == "--verifier-env"]
    assert forwarded == [
        "LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}",
        "SKILL_EVAL_JUDGE_MODEL=${SKILL_EVAL_JUDGE_MODEL}",
    ]
    assert "--agent-env" not in command


def test_nvidia_build_agent_import_selection_includes_local_bridge_agents() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="openai/gpt-oss-120b",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/openai/gpt-oss-120b",
    )

    assert _nvidia_build_agent_import_path(provider, "codex", "docker") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"
    )
    assert _nvidia_build_agent_import_path(provider, "claude-code", "docker") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode"
    )
    assert _nvidia_build_agent_import_path(provider, "opencode", "docker") is None
    assert _nvidia_build_agent_import_path(provider, "codex", "local") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildCodex"
    )
    assert _nvidia_build_agent_import_path(provider, "claude-code", "local") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildClaudeCode"
    )
    assert _nvidia_build_agent_import_path(provider, "opencode", "local") is None


def test_docker_bridge_command_combines_custom_agent_and_secure_environment() -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"

    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="bridge-test",
        env_mode="docker",
        agent_import_path=import_path,
    )

    assert command[command.index("--agent-import-path") + 1] == import_path
    assert "-a" not in command
    assert "--environment-import-path" in command


def test_local_bridge_command_uses_custom_agent_import_path() -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildCodex"

    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="local-bridge-test",
        env_mode="local",
        agent_import_path=import_path,
    )

    assert command[command.index("--agent-import-path") + 1] == import_path
    assert "--environment-import-path" in command
    assert "-a" not in command


@pytest.mark.parametrize("env_mode", ["e2b", "daytona"])
def test_custom_agent_import_path_preserves_native_cloud_environment(env_mode: str) -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayCodex"
    model = "openai/openai/gpt-5.6-sol"
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="gateway-test",
        env_mode=env_mode,
        agent_import_path=import_path,
        model=model,
    )

    assert command[command.index("--agent-import-path") + 1] == import_path
    assert command[command.index("--env") + 1] == env_mode
    assert command[command.index("--model") + 1] == model
    assert "-a" not in command
    assert "--environment-import-path" not in command


def test_evaluate_forwards_native_environment_without_legacy_sandbox_configuration(monkeypatch, tmp_path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    captured: dict = {}
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4.1-mini",
    )

    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "resolve_results_root", lambda *_args: tmp_path / "results")
    monkeypatch.setattr(tier3_commands, "run_harbor_eval", lambda **kwargs: captured.update(kwargs) or {"ok": True})

    tier3_commands.evaluate(
        skill,
        agents="codex",
        env_mode="e2b",
        skip_baseline=False,
        n_attempts=None,
        pass_threshold=None,
        n_concurrent=None,
        max_agents=None,
        model=None,
        agent_model=(),
        custom_dockerfile_mode=None,
        skill_workspace_mode=None,
        include_skills=(),
        copy_repo=False,
        grading_mode="default_plus_custom",
        results_dir=None,
        harbor_keep_jobs=False,
        timeout_multiplier=None,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert captured["env_mode"] == "e2b"
    assert captured["grading_mode"] == "default_plus_custom"
    assert "sandbox_config" not in captured


def test_evaluate_forwards_claude_alias_as_canonical_agent(monkeypatch, tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    captured: dict = {}
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        litellm_model="anthropic/claude-sonnet-4-5",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "resolve_results_root", lambda *_args: tmp_path / "results")
    monkeypatch.setattr(
        tier3_commands,
        "run_harbor_eval",
        lambda **kwargs: (
            captured.update(kwargs) or {"execution_status": "succeeded", "execution_errors": [], "agents": {}}
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(skill),
            "--agents",
            "claude",
            "--agent-model",
            "claude=anthropic/claude-sonnet-4-5",
            "--progress",
            "off",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["agents"] == ["claude-code"]
    assert captured["agent_models"] == {"claude-code": ["anthropic/claude-sonnet-4-5"]}


def test_evaluate_rejects_repeated_model_override_before_engine(monkeypatch, tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        litellm_model="anthropic/claude-sonnet-4-5",
    )
    engine = Mock()
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "run_harbor_eval", engine)

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(skill),
            "--agents",
            "claude-code",
            "--agent-model",
            "claude-code=first",
            "--agent-model",
            "claude-code=second",
            "--progress",
            "off",
        ],
    )

    assert result.exit_code != 0
    assert "specify only one model for claude-code" in result.output
    engine.assert_not_called()


def test_doctor_rejects_alias_model_collision_consistently(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        litellm_model="anthropic/claude-sonnet-4-5",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])

    result = CliRunner().invoke(
        cli,
        [
            "doctor",
            "--agents",
            "claude",
            "--agent-model",
            "claude=first",
            "--agent-model",
            "claude-code=second",
        ],
    )

    assert result.exit_code == 1
    normalized = " ".join(result.output.split())
    assert "refer to the same agent" in normalized
    assert "specify only one model for claude-code" in normalized


def test_generated_task_stages_public_provider_variables_for_the_verifier(tmp_path) -> None:
    _write_task_toml(
        tmp_path,
        {"id": "provider-test", "expected_skill": "demo"},
        has_skill=True,
        runtime_env={
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
            "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
            "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
        },
    )

    task = (tmp_path / "task.toml").read_text(encoding="utf-8")
    assert 'SKILL_EVAL_LLM_PROVIDER = "${SKILL_EVAL_LLM_PROVIDER}"' in task
    assert 'NVIDIA_API_KEY = "${NVIDIA_API_KEY}"' in task
    assert 'OPENAI_API_KEY = "${OPENAI_API_KEY}"' in task


def test_generated_task_verifier_timeout_covers_all_structured_judge_attempts(tmp_path) -> None:
    _write_task_toml(
        tmp_path,
        {"id": "timeout-test", "expected_skill": "demo"},
        has_skill=True,
        runtime_env={},
    )

    task = tomllib.loads((tmp_path / "task.toml").read_text(encoding="utf-8"))

    # Accuracy, custom goal, and behavior can each make two sequential
    # 90-second provider attempts. Leave a full minute for verifier overhead.
    assert task["verifier"]["timeout_sec"] >= (3 * 2 * 90) + 60


def test_generated_task_keeps_evaluator_provider_variables_out_of_agent_environment(tmp_path) -> None:
    _write_task_toml(
        tmp_path,
        {"id": "provider-test", "expected_skill": "demo"},
        has_skill=True,
        runtime_env={"SERVICE_API_TOKEN": "${SERVICE_API_TOKEN}"},
        verifier_env={
            "LLM_JUDGE_MODEL": "${LLM_JUDGE_MODEL}",
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
            "SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}",
            "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
        },
    )

    task = tomllib.loads((tmp_path / "task.toml").read_text(encoding="utf-8"))
    assert task["verifier"]["env"] == {
        "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
        "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
    }
    assert task["environment"]["env"] == {
        **_EVALUATOR_MANAGED_RUNTIME_ENV,
        "SERVICE_API_TOKEN": "${SERVICE_API_TOKEN}",
    }


def test_nvidia_build_provider_mapping_does_not_supply_an_openai_agent_credential() -> None:
    environment = _provider_environment(
        ProviderConfig(
            provider="nv_build",
            model="meta/llama-3.1-8b-instruct",
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            litellm_model="openai/meta/llama-3.1-8b-instruct",
        )
    )

    assert environment["NVIDIA_API_KEY"] == "test-key"
    assert "OPENAI_API_KEY" not in environment
    assert "OPENAI_BASE_URL" not in environment


def test_doctor_accepts_nvidia_build_codex_without_openai_runtime_credential(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = CliRunner().invoke(cli, ["doctor", "--agents", "codex"])

    assert result.exit_code == 0
    assert "Codex runtime credential" in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_doctor_nvidia_build_codex_ignores_incomplete_openai_runtime_credential(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("OPENAI_API_KEY", "openai-runtime-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = CliRunner().invoke(cli, ["doctor", "--agents", "codex"])

    assert result.exit_code == 0
    assert "Codex runtime credential" in result.output
    assert "OPENAI_API_KEY + OPENAI_BASE_URL" not in result.output


def test_doctor_build_codex_ignores_native_pair_and_accepts_build_model(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("OPENAI_API_KEY", "openai-runtime-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    result = CliRunner().invoke(
        cli,
        [
            "doctor",
            "--agents",
            "codex",
            "--agent-model",
            "codex=nvidia/nemotron-3-super-120b-a12b",
        ],
    )

    assert result.exit_code == 0
    assert "Codex runtime credential" in result.output
    assert "pass" in result.output


def test_doctor_verify_models_warns_when_catalog_success_does_not_verify_credentials(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    probe = Mock(
        return_value=ModelProbeResult(
            True,
            "nv_build",
            "meta/llama-3.1-8b-instruct",
            "model is available",
        )
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", "opencode", "--env-mode", "docker", "--verify-models"],
    )

    assert result.exit_code == 0
    assert "warn" in result.output
    assert "does not verify runtime credentials" in " ".join(result.output.split()).lower()
    probe.assert_called_once()
    probed_provider = probe.call_args.args[0]
    assert probed_provider.provider == "nv_build"
    assert probed_provider.model == "meta/llama-3.1-8b-instruct"


@pytest.mark.parametrize(
    ("failure_kind", "http_status", "expected_status", "expected_exit_code"),
    [
        (ModelCatalogFailureKind.AUTHORIZATION, 403, "warn", 0),
        (ModelCatalogFailureKind.AUTHENTICATION, 401, "fail", 1),
    ],
)
def test_doctor_verify_models_uses_probe_disposition_for_catalog_failures(
    monkeypatch,
    failure_kind: ModelCatalogFailureKind,
    http_status: int,
    expected_status: str,
    expected_exit_code: int,
) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider="openai",
        model="gpt-test",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-test",
    )
    probe = Mock(
        return_value=ModelProbeResult(
            False,
            "openai",
            "gpt-test",
            f"model catalog returned HTTP {http_status}",
            failure_kind=failure_kind,
            http_status=http_status,
        )
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", "codex", "--env-mode", "docker", "--verify-models"],
    )

    assert result.exit_code == expected_exit_code
    assert expected_status in result.output


@pytest.mark.parametrize(
    ("provider_name", "base_url", "agent"),
    [
        ("openai", "https://gateway.example/v1", "codex"),
        ("anthropic", "https://gateway.example/v1", "claude-code"),
    ],
)
def test_doctor_verify_models_warns_for_custom_endpoint_catalog_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    base_url: str,
    agent: str,
) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider=provider_name,
        model="model-test",
        api_key="provider-key",
        base_url=base_url,
        litellm_model=f"{provider_name}/model-test",
    )
    probe = Mock(
        return_value=ModelProbeResult(
            False,
            provider_name,
            "model-test",
            "model catalog returned HTTP 401",
            failure_kind=ModelCatalogFailureKind.AUTHENTICATION,
            http_status=401,
        )
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", agent, "--env-mode", "docker", "--verify-models"],
    )

    assert result.exit_code == 0
    assert "warn" in result.output
    assert "HTTP 401" in result.output


def test_doctor_reports_missing_independent_cross_provider_credential(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4.1-mini",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", "claude-code", "--verify-models"],
        terminal_width=240,
    )

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_nvidia_build_docker_codex_uses_the_compatibility_bridge() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["codex"], {}, env_mode="docker") == []


def test_nvidia_build_rejects_agents_without_a_credential_contract() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="openai/gpt-oss-120b",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/openai/gpt-oss-120b",
    )

    errors = _validate_agent_provider_credentials(provider, ["cursor-cli"], {})

    assert errors and "does not support live agent" in errors[0]


def test_nvidia_build_local_codex_uses_the_compatibility_bridge() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["codex"], {}, env_mode="local") == []


def test_nvidia_build_local_claude_uses_the_compatibility_bridge() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["claude-code"], {}, env_mode="local") == []


@pytest.mark.parametrize("agent", ["opencode", "codex", "claude-code"])
def test_nvidia_build_local_agents_require_network_access(
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="nvidia/nemotron-3-nano-30b-a3b",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/nvidia/nemotron-3-nano-30b-a3b",
    )
    monkeypatch.setenv("SKILLEVALUATOR_LOCAL_ALLOW_NET", "0")

    errors = _validate_agent_provider_credentials(provider, [agent], {}, env_mode="local")

    assert errors and "network" in errors[0].lower()
    assert "SKILLEVALUATOR_LOCAL_ALLOW_NET" in errors[0]


def test_nvidia_build_claude_accepts_explicit_anthropic_credential_and_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert (
        _validate_agent_provider_credentials(
            provider,
            ["claude-code"],
            {"ANTHROPIC_API_KEY": "anthropic-key"},
            {"claude-code": "CLI"},
            env_mode="local",
        )
        == []
    )


def test_nvidia_build_opencode_default_model_is_prefixed_for_local_runtime() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _model_for_agent(
        "opencode",
        cli_model=None,
        config_agents={},
        provider=provider,
    ) == ("nvidia/meta/llama-3.1-8b-instruct", "public provider default")


@pytest.mark.parametrize(
    ("provider_name", "expected"),
    [
        ("openai", "openai/test-model"),
        ("openai-compatible", "openai/nvidia/nvidia/nemotron-3-super-120b-long-ctx"),
        ("anthropic", "anthropic/test-model"),
    ],
)
def test_opencode_default_model_is_provider_qualified(provider_name: str, expected: str) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model="test-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{provider_name}/test-model",
    )

    assert _model_for_agent("opencode", cli_model=None, config_agents={}, provider=provider) == (
        expected,
        "openai-compatible agent default" if provider_name == "openai-compatible" else "public provider default",
    )


@pytest.mark.parametrize(
    ("provider_name", "raw_model", "expected"),
    [
        ("nv_build", "nvidia/llama-test", "nvidia/nvidia/llama-test"),
        ("openai", "openai/vendor/model", "openai/openai/vendor/model"),
        ("anthropic", "anthropic/vendor/model", "anthropic/anthropic/vendor/model"),
    ],
)
def test_opencode_provider_default_preserves_raw_ids_that_begin_with_runtime_namespace(
    provider_name: str,
    raw_model: str,
    expected: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model=raw_model,
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model=f"openai/{raw_model}",
    )

    assert _model_for_agent("opencode", cli_model=None, config_agents={}, provider=provider) == (
        expected,
        "public provider default",
    )


@pytest.mark.parametrize(
    ("provider_name", "cli_model", "config_agents", "expected", "source"),
    [
        ("nv_build", "meta/llama-3.1-8b-instruct", {}, "meta/llama-3.1-8b-instruct", "CLI"),
        ("nv_build", "openai/gpt-oss-120b", {}, "openai/gpt-oss-120b", "CLI"),
        ("nv_build", "nvidia/openai/gpt-oss-120b", {}, "nvidia/openai/gpt-oss-120b", "CLI"),
        ("openai", "gpt-4.1-mini", {}, "gpt-4.1-mini", "CLI"),
        (
            "anthropic",
            None,
            {"opencode": {"model": "claude-sonnet-test"}},
            "claude-sonnet-test",
            "evals/config.yml",
        ),
        (
            "openai-compatible",
            None,
            {"opencode": {"model": "vendor/custom-model"}},
            "vendor/custom-model",
            "evals/config.yml",
        ),
    ],
)
def test_opencode_explicit_model_is_preserved_exactly(
    provider_name: str,
    cli_model: str | None,
    config_agents: dict,
    expected: str,
    source: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model="provider-default",
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{provider_name}/provider-default",
    )

    assert _model_for_agent(
        "opencode",
        cli_model=cli_model,
        config_agents=config_agents,
        provider=provider,
    ) == (expected, source)


def test_doctor_explicit_opencode_runtime_model_probes_raw_catalog_id(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider="nv_build",
        model="provider-default",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/provider-default",
    )
    probe = Mock(return_value=ModelProbeResult(True, "nv_build", "openai/gpt-oss-120b", "available"))
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        [
            "doctor",
            "--agents",
            "opencode",
            "--env-mode",
            "docker",
            "--verify-models",
            "--agent-model",
            "opencode=nvidia/openai/gpt-oss-120b",
        ],
    )

    assert result.exit_code == 0
    probed_provider = probe.call_args.args[0]
    assert probed_provider.model == "openai/gpt-oss-120b"
    assert probed_provider.litellm_model == "openai/openai/gpt-oss-120b"


def test_nvidia_build_docker_opencode_uses_selected_provider_key() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    errors = _validate_agent_provider_credentials(provider, ["opencode"], {}, env_mode="docker")

    assert errors == []


def test_nvidia_build_local_opencode_uses_evaluator_provider_mapping() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["opencode"], {}, env_mode="local") == []


def test_nvidia_build_local_codex_uses_the_provider_default_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["codex"], {}, env_mode="local") == []


def test_nvidia_build_codex_accepts_explicit_independent_credential_and_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert (
        _validate_agent_provider_credentials(
            provider,
            ["codex"],
            {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
            {"codex": "CLI"},
            env_mode="local",
        )
        == []
    )


def test_generated_verifier_rejects_non_http_provider_base_urls(monkeypatch) -> None:
    verifier = _load_verifier_template()
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "file:///etc/passwd")

    with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
        verifier._resolve_url("openai")
    with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
        verifier._anthropic_url()


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://gateway.example", "https://gateway.example/v1/messages"),
        ("https://gateway.example/", "https://gateway.example/v1/messages"),
        ("https://gateway.example/v1", "https://gateway.example/v1/messages"),
        ("https://gateway.example/team", "https://gateway.example/team/v1/messages"),
        ("https://gateway.example/team/v1", "https://gateway.example/team/v1/messages"),
        ("https://gateway.example:8443/team/v1/", "https://gateway.example:8443/team/v1/messages"),
        ("http://gateway.internal:8080/v1", "http://gateway.internal:8080/v1/messages"),
        ("http://anthropic_proxy:8000/v1", "http://anthropic_proxy:8000/v1/messages"),
        ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1/messages"),
        ("http://[::1]:8080/team/v1", "http://[::1]:8080/team/v1/messages"),
        (
            "http://[fe80::1%25eth0]:8080/team/v1",
            "http://[fe80::1%25eth0]:8080/team/v1/messages",
        ),
        ("https://xn--bcher-kva.example/v1", "https://xn--bcher-kva.example/v1/messages"),
        ("https://bücher.example/v1", "https://xn--bcher-kva.example/v1/messages"),
        ("https://faß.de/v1", "https://xn--fa-hia.de/v1/messages"),
        ("https://οδός.example/v1", "https://xn--pxavk3b.example/v1/messages"),
        (
            "https://bücher.example.:8443/bücher/v1",
            "https://xn--bcher-kva.example.:8443/b%C3%BCcher/v1/messages",
        ),
        ("https://gateway.example/caf%C3%A9/v1", "https://gateway.example/caf%C3%A9/v1/messages"),
        ("https://gateway.example/caf%c3%a9/v1", "https://gateway.example/caf%C3%A9/v1/messages"),
        ("https://gateway.example/opaque%ff/v1", "https://gateway.example/opaque%FF/v1/messages"),
        (
            "https://gateway.example/tenant%25west/v1",
            "https://gateway.example/tenant%25west/v1/messages",
        ),
        ("https://gateway.example/100%25/v1", "https://gateway.example/100%25/v1/messages"),
        ("https://gateway.example/team/%76%31", "https://gateway.example/team/v1/messages"),
        ("https://gateway.example/team/v1///", "https://gateway.example/team/v1/messages"),
        (
            "https://gateway.example/teams;v=1/@me+you/v1",
            "https://gateway.example/teams;v=1/@me+you/v1/messages",
        ),
    ],
)
def test_generated_verifier_builds_exactly_one_anthropic_native_messages_path(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    expected_url: str,
) -> None:
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)

    assert verifier._anthropic_url() == expected_url


@pytest.mark.parametrize(
    ("variable", "base_url"),
    [
        ("SKILL_EVAL_LLM_BASE_URL", "https://gateway.example/v1/messages"),
        ("ANTHROPIC_BASE_URL", "https://url-user:url-secret@gateway.example/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/v1?token=url-secret"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/v1#token=url-secret"),
        ("ANTHROPIC_BASE_URL", "https://:443/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example:not-a-port/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example:/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example\\team\\v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/v1\n"),
        ("ANTHROPIC_BASE_URL", "https:///team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway example/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway%2eexample/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example|evil/v1"),
        ("ANTHROPIC_BASE_URL", "https://999.1.1.1/v1"),
        ("ANTHROPIC_BASE_URL", "https://[v1.not-ipv6]/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%76%31/%6dessages"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%2Fv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%5cv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%0av1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%2"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%GG"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/../team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%2e%2e/team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%2576%2531/%256dessages"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%252e%252e/team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%252Fv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%255Cv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%250Av1"),
        (
            "ANTHROPIC_BASE_URL",
            "https://gateway.example/%25%37%36%25%33%31/%25%36%64essages",
        ),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%25%32%46v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team//v1"),
        ("ANTHROPIC_BASE_URL", "https://☃.example/v1"),
    ],
)
def test_generated_verifier_rejects_unsafe_anthropic_api_roots_without_echoing_credentials(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    base_url: str,
) -> None:
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv(variable, base_url)

    with pytest.raises(ValueError) as exc_info:
        verifier._anthropic_url()

    message = str(exc_info.value)
    assert variable in message
    assert base_url not in message
    assert "url-user" not in message
    assert "url-secret" not in message


def test_generated_verifier_uses_the_official_anthropic_messages_url_when_base_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    assert verifier._anthropic_url() == "https://api.anthropic.com/v1/messages"


@pytest.mark.parametrize(
    ("configured_path", "expected_path"),
    [
        ("bücher/v1", "/b%C3%BCcher/v1/messages"),
        ("opaque%FF/v1", "/opaque%FF/v1/messages"),
        ("tenant%25west/v1", "/tenant%25west/v1/messages"),
        ("100%25/v1", "/100%25/v1/messages"),
        ("team/v1///", "/team/v1/messages"),
    ],
)
def test_anthropic_sdk_and_bundled_verifier_use_the_same_ascii_path(
    monkeypatch: pytest.MonkeyPatch,
    configured_path: str,
    expected_path: str,
) -> None:
    from anthropic import Anthropic

    requested_paths: list[str] = []

    class RecordingHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requested_paths.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return None

    verifier = _load_verifier_template()
    with ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        provider = resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}/{configured_path}",
            }
        )
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url or "")
        try:
            with Anthropic(
                api_key="test-anthropic-key",
                base_url=provider.base_url,
                max_retries=0,
                timeout=5.0,
            ) as client:
                sdk_response = client.messages.create(
                    model="claude-test",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "hello"}],
                )
            verifier_response, verifier_error = verifier._call_anthropic("hello", "claude-test", 16, 0.0)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert sdk_response.content[0].text == "ok"
    assert verifier_response == "ok"
    assert verifier_error is None
    assert requested_paths == [expected_path, expected_path]


def test_anthropic_sdk_and_bundled_verifier_prepare_the_same_scoped_ipv6_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthropic import Anthropic

    sdk_urls: list[str] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        sdk_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    provider = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "ANTHROPIC_BASE_URL": "http://[fe80::1%25eth0]:8080/bücher/v1",
        }
    )
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url or "")

    with (
        httpx.Client(transport=httpx.MockTransport(handle_request)) as http_client,
        Anthropic(
            api_key="test-anthropic-key",
            base_url=provider.base_url,
            http_client=http_client,
        ) as client,
    ):
        client.messages.create(
            model="claude-test",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )

    verifier_url = verifier._anthropic_url()
    verifier_request = urllib.request.Request(verifier_url)
    expected_url = "http://[fe80::1%25eth0]:8080/b%C3%BCcher/v1/messages"
    assert sdk_urls == [expected_url]
    assert verifier_request.full_url == expected_url
    assert verifier_request.selector == "/b%C3%BCcher/v1/messages"


@pytest.mark.parametrize(
    ("unicode_host", "ascii_host"),
    [
        ("faß.de", "xn--fa-hia.de"),
        ("οδός.example", "xn--pxavk3b.example"),
    ],
)
def test_anthropic_idna_matches_httpx_sdk_and_bundled_verifier(
    monkeypatch: pytest.MonkeyPatch,
    unicode_host: str,
    ascii_host: str,
) -> None:
    from anthropic import Anthropic

    sdk_urls: list[str] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        sdk_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    configured_url = f"https://{unicode_host}/team/v1"
    provider = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "ANTHROPIC_BASE_URL": configured_url,
        }
    )
    expected_url = f"https://{ascii_host}/team/v1/messages"
    assert str(httpx.URL(configured_url)) == f"https://{ascii_host}/team/v1"

    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url or "")
    with (
        httpx.Client(transport=httpx.MockTransport(handle_request)) as http_client,
        Anthropic(
            api_key="test-anthropic-key",
            base_url=provider.base_url,
            http_client=http_client,
        ) as client,
    ):
        client.messages.create(
            model="claude-test",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )

    assert sdk_urls == [expected_url]
    assert verifier._anthropic_url() == expected_url


@pytest.mark.parametrize(
    ("has_skill", "arm_suffix", "expected_task_name"),
    [
        (True, "", "nvidia/skillevaluator-case-001"),
        (True, "-with-skill", "nvidia/skillevaluator-case-001-with-skill"),
        (False, "-without-skill", "nvidia/skillevaluator-case-001-without-skill"),
    ],
)
def test_write_task_toml_dual_arm_suffix(
    tmp_path: Path,
    has_skill: bool,
    arm_suffix: str,
    expected_task_name: str,
) -> None:
    """Verify that arm suffix is appended only when provided for dual-arm runs."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_task_toml(
        case_dir,
        {"id": "case-001", "expected_skill": "demo"},
        has_skill=has_skill,
        arm_suffix=arm_suffix,
    )
    task = tomllib.loads((case_dir / "task.toml").read_text(encoding="utf-8"))
    assert task["task"]["name"] == expected_task_name


def test_write_task_toml_type_safety(tmp_path: Path) -> None:
    """Verify that _write_task_toml rejects non-string arm_suffix values."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    with pytest.raises(TypeError, match="arm_suffix must be a string"):
        _write_task_toml(case_dir, {"id": "case-001"}, has_skill=True, arm_suffix=123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("case-001-with-skill", "case-001"),
        ("case-001-without-skill", "case-001"),
        ("case-001-with", "case-001-with"),
        ("case-001-without", "case-001-without"),
        ("case-001", "case-001"),
    ],
)
def test_strip_arm_suffix(value: str, expected: str) -> None:
    """Verify _strip_arm_suffix trims exact dual-arm suffixes."""
    from skillevaluator.tier3.harbor.collector import _strip_arm_suffix

    assert _strip_arm_suffix(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("case-001-with-skill-attempt1", "case-001"),
        ("case-001-without-skill_attempt2", "case-001"),
        ("case-001-attempt1-with-skill", "case-001"),
        ("case-001_attempt2-without-skill", "case-001"),
        ("case-001-with-skill", "case-001"),
        ("case-001-without-skill", "case-001"),
        ("case-001-attempt1", "case-001"),
        ("case-001", "case-001"),
    ],
)
def test_strip_arm_and_attempt_suffixes(value: str, expected: str) -> None:
    """Verify _strip_arm_and_attempt_suffixes removes both suffixes regardless of ordering."""
    from skillevaluator.tier3.harbor.collector import _strip_arm_and_attempt_suffixes

    assert _strip_arm_and_attempt_suffixes(value) == expected


@pytest.mark.parametrize(
    ("raw_input", "expected_ids", "expected_output"),
    [
        # Bare case IDs without suffixes
        ("case-001", None, "case-001"),
        ("skillevaluator-case-001", None, "case-001"),
        ("skillevaluator-case-001", {"case-001", "case-002"}, "case-001"),
        # Dual-arm suffixes without expected_case_ids (fallback)
        ("case-001-with-skill", None, "case-001"),
        ("case-001-without-skill", None, "case-001"),
        ("skillevaluator-case-001-with-skill", None, "case-001"),
        ("skillevaluator-case-001-without-skill", None, "case-001"),
        # Dual-arm suffixes with expected_case_ids matching
        ("case-001-with-skill", {"case-001", "case-002"}, "case-001"),
        ("case-001-without-skill", {"case-001", "case-002"}, "case-001"),
        ("skillevaluator-case-001-with-skill", {"case-001", "case-002"}, "case-001"),
        ("skillevaluator-case-001-without-skill", {"case-001", "case-002"}, "case-001"),
        # Namespaced task names (e.g., nvidia/...)
        ("nvidia/skillevaluator-case-001", None, "case-001"),
        ("nvidia/skillevaluator-case-001", {"case-001"}, "case-001"),
        ("nvidia/skillevaluator-case-001-with-skill", None, "case-001"),
        ("nvidia/skillevaluator-case-001-with-skill", {"case-001"}, "case-001"),
        ("nvidia/skillevaluator-case-001-without-skill", None, "case-001"),
        ("nvidia/skillevaluator-case-001-without-skill", {"case-001"}, "case-001"),
        ("custom/repo/skillevaluator-case-002-with-skill", {"case-002"}, "case-002"),
        # Attempt suffixes combined with dual-arm suffixes in both orderings
        ("case-001-with-skill-attempt1", None, "case-001"),
        ("case-001-with-skill-attempt1", {"case-001"}, "case-001"),
        ("case-001-without-skill_attempt2", None, "case-001"),
        ("case-001-without-skill_attempt2", {"case-001"}, "case-001"),
        ("case-001-attempt1-with-skill", None, "case-001"),
        ("case-001-attempt1-with-skill", {"case-001"}, "case-001"),
        ("case-001_attempt2-without-skill", None, "case-001"),
        ("case-001_attempt2-without-skill", {"case-001"}, "case-001"),
        ("skillevaluator-case-001-with-skill-attempt1", None, "case-001"),
        ("skillevaluator-case-001-with-skill-attempt1", {"case-001"}, "case-001"),
        ("skillevaluator-case-001-attempt1-with-skill", None, "case-001"),
        ("skillevaluator-case-001-attempt1-with-skill", {"case-001"}, "case-001"),
        ("nvidia/skillevaluator-case-001-with-skill-attempt3", {"case-001"}, "case-001"),
        ("nvidia/skillevaluator-case-001-attempt3-with-skill", {"case-001"}, "case-001"),
        ("nvidia/skillevaluator-case-001-without-skill_attempt4", None, "case-001"),
        ("nvidia/skillevaluator-case-001_attempt4-without-skill", None, "case-001"),
        # Legitimate case IDs ending with -with or -without preserved when in expected_case_ids
        ("case-with", {"case-with"}, "case-with"),
        ("case-without", {"case-without"}, "case-without"),
        ("nvidia/case-with", {"case-with"}, "case-with"),
        ("nvidia/case-without", {"case-without"}, "case-without"),
        ("skillevaluator-case-with", {"case-with"}, "case-with"),
        ("skillevaluator-case-without", {"case-without"}, "case-without"),
        ("case-with-attempt1", {"case-with"}, "case-with"),
        ("case-without-attempt2", {"case-without"}, "case-without"),
        # Dual-arm runs on legitimate -with / -without IDs
        ("case-with-with-skill", {"case-with"}, "case-with"),
        ("case-without-without-skill", {"case-without"}, "case-without"),
        ("skillevaluator-case-with-with-skill", {"case-with"}, "case-with"),
        ("skillevaluator-case-without-without-skill", {"case-without"}, "case-without"),
        ("nvidia/skillevaluator-case-with-with-skill", {"case-with"}, "case-with"),
        ("nvidia/skillevaluator-case-without-without-skill", {"case-without"}, "case-without"),
        ("case-with-with-skill-attempt1", {"case-with"}, "case-with"),
        ("case-with-attempt1-with-skill", {"case-with"}, "case-with"),
        ("case-without-without-skill_attempt2", {"case-without"}, "case-without"),
        ("case-without_attempt2-without-skill", {"case-without"}, "case-without"),
        # Case IDs that retain the skillevaluator- prefix in expected_case_ids
        ("skillevaluator-case-001-with-skill", {"skillevaluator-case-001"}, "skillevaluator-case-001"),
        ("skillevaluator-case-001-without-skill", {"skillevaluator-case-001"}, "skillevaluator-case-001"),
        ("nvidia/skillevaluator-case-001-with-skill", {"skillevaluator-case-001"}, "skillevaluator-case-001"),
        ("skillevaluator-case-001-with-skill-attempt1", {"skillevaluator-case-001"}, "skillevaluator-case-001"),
        ("skillevaluator-case-001-attempt1-with-skill", {"skillevaluator-case-001"}, "skillevaluator-case-001"),
        # Empty and blank strings
        ("", None, ""),
        ("   ", None, ""),
        ("", {"case-001"}, ""),
    ],
)
def test_canonical_case_id_arm_stripping(
    raw_input: str,
    expected_ids: set[str] | None,
    expected_output: str,
) -> None:
    """Verify _canonical_case_id normalizes task identifiers across naming and attempt variants."""
    from skillevaluator.tier3.harbor.collector import _canonical_case_id

    assert _canonical_case_id(raw_input, expected_ids) == expected_output


@pytest.mark.parametrize(
    ("arm_suffix", "expected_task_name"),
    [
        ("-with-skill", "nvidia/case-001-with-skill"),
        ("-without-skill", "nvidia/case-001-without-skill"),
    ],
)
def test_stage_native_harbor_tasks_dual_arm_suffix(
    tmp_path: Path,
    arm_suffix: str,
    expected_task_name: str,
) -> None:
    """Verify that stage_native_harbor_tasks appends arm suffix to native task.toml name."""
    skill_dir = tmp_path / "target-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Target Skill\n", encoding="utf-8")
    evals_dir = skill_dir / "evals" / "harbor"
    task_dir = evals_dir / "case-001"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Instruction\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[environment]\n',
        encoding="utf-8",
    )
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    out_dir = tmp_path / f"out{arm_suffix}"
    staged = stage_native_harbor_tasks(
        skill_dir,
        out_dir,
        grading_mode="custom_only",
        arm_suffix=arm_suffix,
    )[0]
    task = tomllib.loads((staged / "task.toml").read_text(encoding="utf-8"))
    assert task["task"]["name"] == expected_task_name


def test_stage_native_harbor_tasks_type_safety(tmp_path: Path) -> None:
    """Verify that native staging functions reject non-string arm_suffix values."""
    skill_dir = tmp_path / "target-skill"
    skill_dir.mkdir()

    with pytest.raises(TypeError, match="arm_suffix must be a string"):
        stage_native_harbor_tasks(skill_dir, tmp_path / "err", arm_suffix=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="arm_suffix must be a string"):
        _stage_native_harbor_tasks_into(
            skill_dir,
            tmp_path / "err",
            evaluator_skill_path=skill_dir,
            arm_suffix=123,  # type: ignore[arg-type]
        )


def test_generate_harbor_tasks_dual_arm_suffix(tmp_path: Path) -> None:
    """Verify that generate_harbor_tasks propagates arm suffix into task.toml name."""
    skill_dir = tmp_path / "gen-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Gen Skill\n", encoding="utf-8")
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir()
    (evals_dir / "evals.json").write_text(
        json.dumps([{"id": "case-001", "prompt": "test prompt"}]),
        encoding="utf-8",
    )

    out = tmp_path / "gen_out"
    staged = generate_harbor_tasks(skill_dir, out, arm_suffix="-with-skill")[0]
    task = tomllib.loads((staged / "task.toml").read_text(encoding="utf-8"))
    assert task["task"]["name"] == "nvidia/skillevaluator-case-001-with-skill"

    with pytest.raises(TypeError, match="arm_suffix must be a string"):
        generate_harbor_tasks(skill_dir, tmp_path / "err", arm_suffix=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="arm_suffix must be a string"):
        _generate_harbor_tasks_into(
            skill_dir,
            tmp_path / "err",
            evaluator_skill_path=skill_dir,
            arm_suffix=123,  # type: ignore[arg-type]
        )

