# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Tier 3 runtime boundaries."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3 import commands as tier3_commands
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.harbor.adapter import _write_task_toml
from skillevaluator.tier3.harbor.runner import (
    _model_for_agent,
    _provider_environment,
    _validate_agent_provider_credentials,
    build_harbor_run_command,
)


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
    assert "local" not in result.output
    assert "base-image-mode" not in result.output
    assert "agent-runtime-preflight" not in result.output


@pytest.mark.parametrize("key", ["base_image_mode", "agent_runtime_preflight"])
def test_public_config_rejects_retired_runtime_controls(tmp_path: Path, key: str) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(f"schema_version: 1\nharbor:\n  {key}: true\n", encoding="utf-8")

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


def test_generated_task_keeps_evaluator_provider_variables_out_of_agent_environment(tmp_path) -> None:
    _write_task_toml(
        tmp_path,
        {"id": "provider-test", "expected_skill": "demo"},
        has_skill=True,
        runtime_env={"SERVICE_API_TOKEN": "${SERVICE_API_TOKEN}"},
        verifier_env={
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
            "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
        },
    )

    task = tomllib.loads((tmp_path / "task.toml").read_text(encoding="utf-8"))
    assert task["verifier"]["env"] == {
        "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
        "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
    }
    assert task["environment"]["env"] == {"SERVICE_API_TOKEN": "${SERVICE_API_TOKEN}"}


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


def test_doctor_rejects_nvidia_build_codex_without_openai_runtime_credential(monkeypatch) -> None:
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

    assert result.exit_code == 1
    assert "Codex runtime credential" in result.output
    assert "codex requires a full OpenAI Responses API" in result.output
    assert "--agent-model" not in result.output
    assert "harbor.agents.codex.model" not in result.output


def test_doctor_reports_only_the_nvidia_build_codex_runtime_credential(monkeypatch) -> None:
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

    assert result.exit_code == 1
    assert "Codex runtime credential" in result.output
    assert "OPENAI_API_KEY + OPENAI_BASE_URL" in result.output
    assert "--agent-model" not in result.output
    assert "harbor.agents.codex.model" not in result.output


def test_nvidia_build_requires_an_independent_codex_credential() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    errors = _validate_agent_provider_credentials(provider, ["codex"], {})

    assert len(errors) == 1
    assert "codex requires a full OpenAI Responses API credential" in errors[0]
    assert "does not support codex's tool schema" in errors[0]
    assert "OPENAI_API_KEY" in errors[0]


def test_nvidia_build_codex_rejects_an_openai_key_without_base_url() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    errors = _validate_agent_provider_credentials(
        provider,
        ["codex"],
        {"OPENAI_API_KEY": "openai-key"},
    )

    assert errors and "OPENAI_API_KEY + OPENAI_BASE_URL" in errors[0]


def test_nvidia_build_claude_requires_an_independent_anthropic_credential() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    errors = _validate_agent_provider_credentials(provider, ["claude-code"], {}, env_mode="local")

    assert errors and "ANTHROPIC_API_KEY" in errors[0]


def test_nvidia_build_claude_accepts_explicit_anthropic_credential_and_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(
        provider,
        ["claude-code"],
        {"ANTHROPIC_API_KEY": "anthropic-key"},
        {"claude-code": "CLI"},
        env_mode="local",
    ) == []


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


def test_nvidia_build_docker_opencode_requires_explicit_runtime_key() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    errors = _validate_agent_provider_credentials(provider, ["opencode"], {}, env_mode="docker")

    assert errors == [
        "opencode with NVIDIA Build requires NVIDIA_API_KEY in harbor.runtime_env so the agent container receives a credential."
    ]


def test_nvidia_build_local_opencode_uses_evaluator_provider_mapping() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["opencode"], {}, env_mode="local") == []


def test_nvidia_build_requires_an_explicit_codex_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    errors = _validate_agent_provider_credentials(
        provider,
        ["codex"],
        {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
    )

    assert errors == [
        "codex needs an explicit OpenAI-compatible model when NVIDIA Build is the evaluator provider; set --agent-model codex=MODEL or harbor.agents.codex.model."
    ]


def test_nvidia_build_codex_accepts_explicit_independent_credential_and_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(
        provider,
        ["codex"],
        {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
        {"codex": "CLI"},
    ) == []


def test_generated_verifier_rejects_non_http_provider_base_urls(monkeypatch) -> None:
    verifier = _load_verifier_template()
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "file:///etc/passwd")

    with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
        verifier._resolve_url("openai")
    with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
        verifier._anthropic_url()
