# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog-only public model discovery CLI contract."""

from __future__ import annotations

import json

from click.testing import CliRunner

from skillevaluator import model_commands
from skillevaluator.cli import cli
from skillevaluator.model_catalog import ModelCatalogError, ModelRecord
from skillevaluator.provider_config import ProviderConfig

_PROVIDER_ENV = (
    "SKILL_EVAL_LLM_PROVIDER",
    "SKILL_EVAL_LLM_MODEL",
    "SKILL_EVAL_LLM_API_KEY",
    "SKILL_EVAL_LLM_BASE_URL",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "AWS_REGION",
)


def _clear_provider_env(monkeypatch) -> None:
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _provider(*, model: str = "nvidia/nemotron-3-nano-30b-a3b") -> ProviderConfig:
    return ProviderConfig(
        provider="nv_build",
        model=model,
        api_key="top-secret-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model=f"openai/{model}",
        credential_env="NVIDIA_API_KEY",
    )


def test_models_help_exposes_catalog_options_without_runtime_verification() -> None:
    result = CliRunner().invoke(cli, ["models", "--help"])

    assert result.exit_code == 0
    assert "--limit" in result.output
    assert "--json" in result.output
    assert "--verify" not in result.output
    assert "--agents" not in result.output
    normalized_output = " ".join(result.output.split()).lower()
    assert "catalog reachability" in normalized_output
    assert "authenticated" not in normalized_output


def test_models_missing_provider_uses_canonical_configuration_error(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)

    result = CliRunner().invoke(cli, ["models"])

    assert result.exit_code == 1
    assert "No provider is configured" in result.output
    assert "NVIDIA_API_KEY" in result.output


def test_models_requires_explicit_provider_when_credentials_are_ambiguous(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    result = CliRunner().invoke(cli, ["models"])

    assert result.exit_code == 1
    assert "SKILL_EVAL_LLM_PROVIDER is required" in result.output
    assert "multiple public provider credentials" in result.output
    assert "nvidia-secret" not in result.output
    assert "openai-secret" not in result.output


def test_models_text_output_marks_only_canonical_configured_model(monkeypatch) -> None:
    config = _provider(model="minimaxai/minimax-m3")
    monkeypatch.setattr(model_commands, "resolve_llm_provider", lambda: config)
    monkeypatch.setattr(
        model_commands,
        "fetch_model_records",
        lambda _config: (
            ModelRecord("nvidia/nemotron-3-nano-30b-a3b", 10),
            ModelRecord("minimaxai/minimax-m3", 1),
        ),
    )

    result = CliRunner().invoke(cli, ["models"])

    assert result.exit_code == 0, result.output
    assert "Provider: nv_build" in result.output
    assert "Configured model: minimaxai/minimax-m3" in result.output
    assert "* minimaxai/minimax-m3" in result.output
    assert "  nvidia/nemotron-3-nano-30b-a3b" in result.output
    assert "top-secret-key" not in result.output


def test_models_json_output_is_copyable_and_bounded(monkeypatch) -> None:
    config = _provider()
    monkeypatch.setattr(model_commands, "resolve_llm_provider", lambda: config)
    monkeypatch.setattr(
        model_commands,
        "fetch_model_records",
        lambda _config: (
            ModelRecord(config.model, 2),
            ModelRecord("openai/gpt-oss-120b", 1),
            ModelRecord("another/chat-model", 3),
        ),
    )

    result = CliRunner().invoke(cli, ["models", "--limit", "2", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "configured_model": "nvidia/nemotron-3-nano-30b-a3b",
        "endpoint": "https://integrate.api.nvidia.com",
        "models": [
            {"created": 2, "id": "nvidia/nemotron-3-nano-30b-a3b", "is_configured": True},
            {"created": 1, "id": "openai/gpt-oss-120b", "is_configured": False},
        ],
        "provider": "nv_build",
    }


def test_models_supports_explicit_openai_compatible_provider(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "local-model")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "local-secret")
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(model_commands, "fetch_model_records", lambda _config: (ModelRecord("local-model"),))

    result = CliRunner().invoke(cli, ["models", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "openai-compatible"
    assert payload["models"][0]["id"] == "local-model"
    assert "local-secret" not in result.output


def test_models_surfaces_only_safe_catalog_errors(monkeypatch) -> None:
    monkeypatch.setattr(model_commands, "resolve_llm_provider", _provider)
    monkeypatch.setattr(
        model_commands,
        "fetch_model_records",
        lambda _config: (_ for _ in ()).throw(ModelCatalogError("model catalog returned HTTP 401")),
    )

    result = CliRunner().invoke(cli, ["models"])

    assert result.exit_code == 1
    assert result.output.strip() == "Error: model catalog returned HTTP 401"
    assert "top-secret-key" not in result.output


def test_models_limit_is_click_validated() -> None:
    runner = CliRunner()

    assert runner.invoke(cli, ["models", "--limit", "0"]).exit_code == 2
    assert runner.invoke(cli, ["models", "--limit", "101"]).exit_code == 2
