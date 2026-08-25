# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Harbor agent-runtime smoke preflight regressions."""

from __future__ import annotations

import errno
import gzip
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from threading import enumerate as enumerate_threads
from unittest.mock import Mock

import pytest
from botocore import credentials as botocore_credentials
from botocore import exceptions as botocore_exceptions
from botocore import tokens as botocore_tokens
from botocore.exceptions import (
    ApiVersionNotFoundError,
    BaseEndpointResolverError,
    ConfigNotFound,
    ConfigParseError,
    DataNotFoundError,
    EndpointConnectionError,
    EndpointProviderError,
    InvalidConfigError,
    InvalidDefaultsMode,
    InvalidIMDSEndpointError,
    InvalidIMDSEndpointModeError,
    InvalidProxiesConfigError,
    InvalidRegionError,
    InvalidRetryConfigurationError,
    InvalidSTSRegionalEndpointsConfigError,
    MissingDependencyException,
    NoAuthTokenError,
    NoCredentialsError,
    ParamValidationError,
    PartialCredentialsError,
    ProfileNotFound,
    RefreshWithMFAUnsupportedError,
    ServiceNotInRegionError,
    SSOTokenLoadError,
    UnauthorizedSSOTokenError,
    UnknownCredentialError,
    UnknownRegionError,
    UnknownSignatureVersionError,
    UnsupportedSignatureVersionError,
)
from botocore.exceptions import (
    SSLError as BotocoreSSLError,
)
from botocore.session import Session as BotocoreSession

from skillevaluator.model_catalog import ModelCatalogError, ModelRecord
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runtime_preflight
from skillevaluator.tier3.harbor.collector import validate_harbor_job_result


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "tasks"
    (dataset / "case-002").mkdir(parents=True)
    (dataset / "case-001").mkdir()
    (dataset / "case-001" / "task.toml").write_text('[task]\nname = "nvidia/skillevaluator-case-001"\n')
    (dataset / "case-002" / "task.toml").write_text('[task]\nname = "nvidia/skillevaluator-case-002"\n')
    return dataset


def _configure_isolated_aws_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    config: str,
    credentials: str = "",
    profile: str = "test",
) -> None:
    config_path = tmp_path / "aws-config"
    credentials_path = tmp_path / "aws-credentials"
    config_path.write_text(config, encoding="utf-8")
    credentials_path.write_text(credentials, encoding="utf-8")
    for variable in tuple(os.environ):
        if variable.startswith("AWS_"):
            monkeypatch.delenv(variable)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_path))
    monkeypatch.setenv("AWS_PROFILE", profile)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _write_harbor_0132_unscored_result(jobs_dir: Path) -> Path:
    job_dir = jobs_dir / "runtime-preflight-opencode"
    trial_dir = job_dir / "case-001__attempt"
    trial_dir.mkdir(parents=True)
    result_path = job_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {
                        "opencode__model___harbor-tasks": {
                            "n_trials": 0,
                            "n_errors": 0,
                            "reward_stats": {},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "case-001__attempt",
                "agent_result": {
                    "n_input_tokens": 100,
                    "n_cache_tokens": 0,
                    "n_output_tokens": 1,
                    "cost_usd": None,
                },
                "exception_info": None,
            }
        ),
        encoding="utf-8",
    )
    return result_path


def test_runtime_preflight_runs_one_case_once_without_verification(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def build(**kwargs):
        captured.update(kwargs)
        return ["harbor", "run", "--safe"]

    def run(command, **kwargs):
        captured["command"] = command
        captured["run_kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime_preflight, "build_harbor_run_command", build)
    monkeypatch.setattr(runtime_preflight.subprocess, "run", run)
    monkeypatch.setattr(
        runtime_preflight,
        "validate_harbor_agent_only_job_result",
        lambda *_args, **_kwargs: (True, "ok"),
    )

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=_dataset(tmp_path),
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={"NVIDIA_API_KEY": "secret"},
        timeout_multiplier=2.0,
        timeout_seconds=321,
    )

    assert result.ok is True
    assert captured["n_attempts"] == 1
    assert captured["n_concurrent"] == 1
    assert captured["disable_verification"] is True
    assert captured["include_task_names"] == ["case-001"]
    assert captured["timeout_multiplier"] == 2.0
    run_kwargs = captured["run_kwargs"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["timeout"] == 321
    assert run_kwargs["env"] == {"NVIDIA_API_KEY": "secret"}


def test_runtime_preflight_hands_nvidia_build_key_only_over_stdin(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    secret = "nvidia-real-secret-value-for-test"

    monkeypatch.setattr(
        runtime_preflight,
        "build_harbor_run_command",
        lambda **_kwargs: ["harbor", "run", "--safe"],
    )

    def run(command, **kwargs):
        captured["command"] = command
        captured["run_kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime_preflight.subprocess, "run", run)
    monkeypatch.setattr(
        runtime_preflight,
        "validate_harbor_agent_only_job_result",
        lambda *_args, **_kwargs: (True, "ok"),
    )

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=_dataset(tmp_path),
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": secret},
    )

    assert result.ok is True
    run_kwargs = captured["run_kwargs"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["input"] == secret
    environment = run_kwargs["env"]
    assert isinstance(environment, dict)
    assert secret not in environment.values()
    assert environment["NVIDIA_API_KEY"] == "skillevaluator-stdin-backed-nvidia-key"
    assert environment["SKILLEVALUATOR_NVIDIA_API_KEY_STDIN"] == "1"
    assert "SKILLEVALUATOR_NVIDIA_API_KEY_FILE" not in environment


def test_runtime_preflight_accepts_harbor_0132_unscored_agent_success(monkeypatch, tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_harbor_0132_unscored_result(jobs_dir)
    monkeypatch.setattr(runtime_preflight, "build_harbor_run_command", lambda **_kwargs: ["harbor", "run"])
    monkeypatch.setattr(
        runtime_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=_dataset(tmp_path),
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=jobs_dir,
        run_env={},
    )

    assert result.ok is True


def test_agent_only_validation_accepts_harbor_0132_unscored_multistep_agent_success(tmp_path: Path) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    trial_result_path = result_path.parent / "case-001__attempt" / "result.json"
    trial_result_path.write_text(
        json.dumps(
            {
                "trial_name": "case-001__attempt",
                "agent_result": None,
                "exception_info": None,
                "step_results": [
                    {
                        "step_name": "author-skill",
                        "agent_result": {
                            "n_input_tokens": 100,
                            "n_cache_tokens": 0,
                            "n_output_tokens": 10,
                            "cost_usd": None,
                        },
                        "verifier_result": None,
                        "exception_info": None,
                        "agent_execution": {
                            "started_at": "2026-07-08T17:00:00Z",
                            "finished_at": "2026-07-08T17:00:10Z",
                        },
                        "verifier": None,
                    },
                    {
                        "step_name": "reuse-skill",
                        "agent_result": {
                            "n_input_tokens": 200,
                            "n_cache_tokens": 20,
                            "n_output_tokens": 15,
                            "cost_usd": None,
                        },
                        "verifier_result": None,
                        "exception_info": None,
                        "agent_execution": {
                            "started_at": "2026-07-08T17:00:11Z",
                            "finished_at": "2026-07-08T17:00:20Z",
                        },
                        "verifier": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1) == (True, "")


def test_agent_only_validation_rejects_mixed_single_and_multistep_agent_results(tmp_path: Path) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    trial_result_path = result_path.parent / "case-001__attempt" / "result.json"
    trial_result_path.write_text(
        json.dumps(
            {
                "trial_name": "case-001__attempt",
                "agent_result": {"n_input_tokens": 100},
                "exception_info": None,
                "step_results": [
                    {
                        "step_name": "author-skill",
                        "agent_result": {"n_input_tokens": 100},
                        "exception_info": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert "mixed top-level and step agent results" in detail.lower()


@pytest.mark.parametrize(
    ("step_results", "expected"),
    [
        ([], "has no step results"),
        ({}, "has invalid step_results"),
        ([None], "invalid step result 1"),
        ([{"exception_info": None, "agent_result": {}}], "invalid step_name"),
        ([{"step_name": "  ", "exception_info": None, "agent_result": {}}], "invalid step_name"),
        ([{"step_name": "author-skill", "agent_result": {}}], "missing exception_info"),
        (
            [
                {
                    "step_name": "author-skill",
                    "exception_info": {"exception_type": "AgentTimeoutError"},
                    "agent_result": {},
                }
            ],
            "recorded an exception",
        ),
        ([{"step_name": "author-skill", "exception_info": None}], "has no agent result"),
        (
            [
                {"step_name": "author-skill", "exception_info": None, "agent_result": {}},
                {"step_name": "reuse-skill", "exception_info": None, "agent_result": None},
            ],
            "has no agent result",
        ),
    ],
)
def test_agent_only_validation_rejects_empty_malformed_or_failed_multistep_results(
    tmp_path: Path,
    step_results: object,
    expected: str,
) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    trial_result_path = result_path.parent / "case-001__attempt" / "result.json"
    trial_result_path.write_text(
        json.dumps(
            {
                "trial_name": "case-001__attempt",
                "agent_result": None,
                "exception_info": None,
                "step_results": step_results,
            }
        ),
        encoding="utf-8",
    )

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert expected in detail.lower()


def test_scored_job_validation_still_rejects_harbor_0132_unscored_result(tmp_path: Path) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")

    ok, detail = validate_harbor_job_result(result_path, expected_trials=1)

    assert ok is False
    assert "account for 0/1" in detail


@pytest.mark.parametrize(
    ("counter", "expected"),
    [
        ("n_errored_trials", "1 errored"),
        ("n_running_trials", "1 running"),
        ("n_pending_trials", "1 pending"),
        ("n_cancelled_trials", "1 cancelled"),
    ],
)
def test_agent_only_validation_rejects_non_successful_harbor_0132_states(
    tmp_path: Path,
    counter: str,
    expected: str,
) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["stats"][counter] = 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert expected in detail


def test_agent_only_validation_surfaces_first_errored_trial_exception(tmp_path: Path) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["stats"]["n_errored_trials"] = 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    trial_result_path = result_path.parent / "case-001__attempt" / "result.json"
    trial_result_path.write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": (
                        "Command failed (exit 128): git -C /workspace init -q\n"
                        "stderr: fatal: Invalid path '/Users/example': Operation not permitted"
                    ),
                },
                "agent_result": {},
            }
        ),
        encoding="utf-8",
    )

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert "1 errored" in detail
    assert "case-001__attempt" in detail
    assert "NonZeroAgentExitCodeError" in detail
    assert "git -C /workspace init -q" in detail
    assert "Operation not permitted" in detail


def test_agent_only_validation_surfaces_multistep_trial_exception(tmp_path: Path) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["stats"]["n_errored_trials"] = 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    trial_result_path = result_path.parent / "case-001__attempt" / "result.json"
    trial_result_path.write_text(
        json.dumps(
            {
                "exception_info": None,
                "agent_result": None,
                "step_results": [
                    {
                        "step_name": "reuse-skill",
                        "exception_info": {
                            "exception_type": "AgentTimeoutError",
                            "exception_message": "agent step timed out",
                        },
                        "agent_result": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert "AgentTimeoutError" in detail
    assert "agent step timed out" in detail


def test_runtime_preflight_redacts_and_sanitizes_retained_trial_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    result_path = _write_harbor_0132_unscored_result(jobs_dir)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["stats"]["n_errored_trials"] = 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    secret = "nvapi-super-secret"
    trial_result_path = result_path.parent / "case-001__attempt" / "result.json"
    trial_result_path.write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": f"token={secret}\x1b[2J {'detail ' * 300}",
                },
                "agent_result": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_preflight, "build_harbor_run_command", lambda **_kwargs: ["harbor", "run"])
    monkeypatch.setattr(
        runtime_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=_dataset(tmp_path),
        agent="opencode",
        model="nvidia/model",
        env_mode="local",
        jobs_dir=jobs_dir,
        run_env={"NVIDIA_API_KEY": secret},
    )

    assert result.ok is False
    assert "NonZeroAgentExitCodeError" in result.detail
    assert secret not in result.detail
    assert "\x1b" not in result.detail
    assert len(result.detail) <= 2000


def test_agent_only_validation_rejects_incomplete_harbor_0132_state(tmp_path: Path) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["stats"]["n_completed_trials"] = 0
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert "completed 0/1" in detail


@pytest.mark.parametrize("result_text", [None, "{not-json"])
def test_agent_only_validation_rejects_missing_or_malformed_job_result(
    tmp_path: Path,
    result_text: str | None,
) -> None:
    result_path = tmp_path / "jobs" / "runtime-preflight-opencode" / "result.json"
    if result_text is not None:
        result_path.parent.mkdir(parents=True)
        result_path.write_text(result_text, encoding="utf-8")

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert "result.json" in detail


@pytest.mark.parametrize(
    ("trial_payload", "expected"),
    [
        (None, "did not produce 1 trial result"),
        ("{not-json", "unreadable trial result"),
        (json.dumps({"exception_info": {"exception_type": "AgentTimeoutError"}, "agent_result": {}}), "exception"),
        (json.dumps({"exception_info": None, "agent_result": None}), "no agent result"),
        (json.dumps({"exception_info": None}), "no agent result"),
    ],
)
def test_agent_only_validation_rejects_missing_malformed_or_failed_trial_result(
    tmp_path: Path,
    trial_payload: str | None,
    expected: str,
) -> None:
    result_path = _write_harbor_0132_unscored_result(tmp_path / "jobs")
    trial_result_path = result_path.parent / "case-001__attempt" / "result.json"
    if trial_payload is None:
        trial_result_path.unlink()
    else:
        trial_result_path.write_text(trial_payload, encoding="utf-8")

    ok, detail = runtime_preflight.validate_harbor_agent_only_job_result(result_path, expected_trials=1)

    assert ok is False
    assert expected in detail.lower()


def test_runtime_preflight_reports_agent_start_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_preflight, "build_harbor_run_command", lambda **_kwargs: ["harbor", "run"])
    monkeypatch.setattr(
        runtime_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 17, "", "401 Unauthorized"),
    )

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=_dataset(tmp_path),
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={},
    )

    assert result.ok is False
    assert result.agent == "opencode"
    assert "401 Unauthorized" in result.detail


def test_runtime_preflight_timeout_is_bounded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_preflight, "build_harbor_run_command", lambda **_kwargs: ["harbor", "run"])
    secret = "nvidia-real-secret-value-for-test"

    def timeout(*_args, **kwargs):
        assert kwargs["input"] == secret
        assert secret not in kwargs["env"].values()
        raise subprocess.TimeoutExpired(["harbor", "run"], timeout=30)

    monkeypatch.setattr(runtime_preflight.subprocess, "run", timeout)

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=_dataset(tmp_path),
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": secret},
        timeout_seconds=30,
    )

    assert result.ok is False
    assert "timed out after 30s" in result.detail
    assert secret not in result.detail


def test_runtime_preflight_rejects_empty_task_tree(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks"
    dataset.mkdir()

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=dataset,
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={},
    )

    assert result.ok is False
    assert "no staged tasks" in result.detail.lower()


def test_task_timeout_plan_uses_largest_staged_timeout(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import runner

    root = tmp_path / "tasks"
    for name, timeout in (("case-1", 120), ("case-2", 300)):
        task = root / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f"[agent]\ntimeout_sec = {timeout}.0\n")

    assert runner._task_timeout_plan([root], 2.0) == 600.0


def test_model_probe_delegates_to_shared_catalog_client_without_exposing_key(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fetch(provider, *, timeout_seconds):
        captured.update(provider=provider, timeout_seconds=timeout_seconds)
        return (ModelRecord("meta/llama-3.1-8b-instruct"),)

    monkeypatch.setattr(runtime_preflight, "fetch_model_records", fetch)
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvapi-secret",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=4.5)

    assert result.ok is True
    assert captured == {"provider": provider, "timeout_seconds": 4.5}
    assert "nvapi-secret" not in result.detail


def test_model_probe_preserves_raw_catalog_id_that_begins_with_nvidia(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: (ModelRecord("nvidia/llama-test"),),
    )
    provider = ProviderConfig(
        provider="nv_build",
        model="nvidia/llama-test",
        api_key="nvapi-secret",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/nvidia/llama-test",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is True
    assert result.model == "nvidia/llama-test"


def test_model_probe_reports_unlisted_model(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: (ModelRecord("different-model"),),
    )
    provider = ProviderConfig(
        provider="openai-compatible",
        model="requested-model",
        api_key="secret-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/requested-model",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is False
    assert "requested-model" in result.detail
    assert "not listed" in result.detail


def test_model_probe_reports_safe_shared_catalog_error(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelCatalogError(
                "model catalog returned HTTP 401",
                kind="authentication",
                http_status=401,
            )
        ),
    )
    provider = ProviderConfig(
        provider="openai",
        model="gpt-test",
        api_key="secret-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-test",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is False
    assert "HTTP 401" in result.detail
    assert "secret-key" not in result.detail
    assert getattr(result, "failure_kind", None) == "authentication"
    assert getattr(result, "http_status", None) == 401


def test_anthropic_model_probe_resolves_alias_omitted_from_listing(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: (ModelRecord("claude-canonical-20260824"),),
    )

    def resolve_alias(provider, model_id, *, timeout_seconds):
        captured.update(provider=provider, model_id=model_id, timeout_seconds=timeout_seconds)
        return ModelRecord("claude-canonical-20260824")

    monkeypatch.setattr(runtime_preflight, "fetch_anthropic_model_record", resolve_alias)
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-stable-alias",
        api_key="secret-key",
        base_url=None,
        litellm_model="anthropic/claude-stable-alias",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=4.5)

    assert result.ok is True
    assert result.model == "claude-stable-alias"
    assert "claude-canonical-20260824" in result.detail
    assert captured["provider"] == provider
    assert captured["model_id"] == "claude-stable-alias"
    assert 0 < float(captured["timeout_seconds"]) <= 4.5


def test_anthropic_alias_lookup_uses_remaining_catalog_probe_deadline(monkeypatch) -> None:
    captured: dict[str, object] = {}
    clock = iter((100.0, 100.06))
    monkeypatch.setattr(runtime_preflight, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: (ModelRecord("claude-canonical-20260824"),),
    )

    def resolve_alias(provider, model_id, *, timeout_seconds):
        captured.update(provider=provider, model_id=model_id, timeout_seconds=timeout_seconds)
        return ModelRecord("claude-canonical-20260824")

    monkeypatch.setattr(runtime_preflight, "fetch_anthropic_model_record", resolve_alias)
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-stable-alias",
        api_key="secret-key",
        base_url=None,
        litellm_model="anthropic/claude-stable-alias",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=0.1)

    assert result.ok is True
    assert captured["timeout_seconds"] == pytest.approx(0.04)


def test_anthropic_alias_lookup_stops_when_catalog_probe_deadline_is_exhausted(monkeypatch) -> None:
    clock = iter((100.0, 100.1))
    monkeypatch.setattr(runtime_preflight, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: (ModelRecord("claude-canonical-20260824"),),
    )
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_anthropic_model_record",
        lambda *_args, **_kwargs: pytest.fail("alias lookup must not start after the shared deadline"),
    )
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-stable-alias",
        api_key="secret-key",
        base_url=None,
        litellm_model="anthropic/claude-stable-alias",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=0.1)

    assert result.ok is False
    assert result.failure_kind == "unavailable"
    assert "timed out" in result.detail


@pytest.mark.parametrize("timeout_seconds", [0, True, float("nan")])
def test_http_catalog_probe_rejects_invalid_shared_deadline(monkeypatch, timeout_seconds) -> None:
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: pytest.fail("invalid timeout must fail before catalog I/O"),
    )
    provider = ProviderConfig(
        provider="openai",
        model="gpt-test",
        api_key="secret-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-test",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=timeout_seconds)

    assert result.ok is False
    assert result.failure_kind == "invalid_configuration"
    assert "positive number" in result.detail


def test_anthropic_model_probe_treats_native_single_model_404_as_fatal(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_model_records",
        lambda *_args, **_kwargs: (ModelRecord("different-model"),),
    )
    monkeypatch.setattr(
        runtime_preflight,
        "fetch_anthropic_model_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelCatalogError(
                "model catalog returned HTTP 404",
                kind="unsupported",
                http_status=404,
            )
        ),
    )
    provider = ProviderConfig(
        provider="anthropic",
        model="missing-model",
        api_key="secret-key",
        base_url="https://api.anthropic.com:443",
        litellm_model="anthropic/missing-model",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is False
    assert result.failure_kind == "model_not_found"
    assert result.http_status == 404
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"


@pytest.mark.parametrize(
    ("provider", "probe", "expected"),
    [
        (
            ProviderConfig("openai", "gpt-test", "key", "https://api.openai.com/v1", "openai/gpt-test"),
            {"ok": True, "failure_kind": None, "http_status": None},
            "verified",
        ),
        (
            ProviderConfig(
                "nv_build",
                "nvidia/model",
                "invalid-key",
                "https://integrate.api.nvidia.com/v1",
                "openai/nvidia/model",
            ),
            {"ok": True, "failure_kind": None, "http_status": 200},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "invalid-key",
                "https://gateway.example/v1",
                "openai/gpt-test",
            ),
            {"ok": True, "failure_kind": None, "http_status": 200},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "key",
                "https://api.openai.com/v1",
                "openai/gpt-test",
            ),
            {"ok": True, "failure_kind": None, "http_status": 200},
            "verified",
        ),
        (
            ProviderConfig("openai", "gpt-test", "key", "https://api.openai.com/v1", "openai/gpt-test"),
            {"ok": False, "failure_kind": "authentication", "http_status": 401},
            "fatal",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "key",
                "https://gateway.example/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": "authentication", "http_status": 401},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai",
                "gpt-test",
                "key",
                "https://gateway.example/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": "authentication", "http_status": 401},
            "degraded",
        ),
        (
            ProviderConfig(
                "anthropic",
                "claude-test",
                "key",
                "https://gateway.example/v1",
                "anthropic/claude-test",
            ),
            {"ok": False, "failure_kind": "authentication", "http_status": 401},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "key",
                "https://api.openai.com/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": "authentication", "http_status": 401},
            "fatal",
        ),
        (
            ProviderConfig(
                "anthropic",
                "claude-test",
                "key",
                "https://api.anthropic.com/v1",
                "anthropic/claude-test",
            ),
            {"ok": False, "failure_kind": "authentication", "http_status": 401},
            "fatal",
        ),
        (
            ProviderConfig(
                "nv_build",
                "nvidia/model",
                "key",
                "https://integrate.api.nvidia.com/v1",
                "openai/nvidia/model",
            ),
            {"ok": False, "failure_kind": "authentication", "http_status": 401},
            "fatal",
        ),
        (
            ProviderConfig(
                "nv_build",
                "nvidia/model",
                "key",
                "https://integrate.api.nvidia.com/v1",
                "openai/nvidia/model",
            ),
            {"ok": False, "failure_kind": "authorization", "http_status": 403},
            "fatal",
        ),
        (
            ProviderConfig(
                "nv_build",
                "nvidia/model",
                "key",
                "https://integrate.api.nvidia.com/v1",
                "openai/nvidia/model",
            ),
            {"ok": False, "failure_kind": None, "http_status": None},
            "fatal",
        ),
        (
            ProviderConfig("openai", "gpt-test", "key", "https://api.openai.com/v1", "openai/gpt-test"),
            {"ok": False, "failure_kind": "authorization", "http_status": 403},
            "degraded",
        ),
        (
            ProviderConfig("openai", "gpt-test", "key", "https://gateway.example/v1", "openai/gpt-test"),
            {"ok": False, "failure_kind": "authorization", "http_status": 403},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "key",
                "https://api.openai.com/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": "authorization", "http_status": 403},
            "degraded",
        ),
        (
            ProviderConfig("bedrock", "model", None, None, "bedrock/model", region="us-west-2"),
            {"ok": False, "failure_kind": "authorization", "http_status": 403},
            "degraded",
        ),
        (
            ProviderConfig("anthropic", "claude-test", "key", None, "anthropic/claude-test"),
            {"ok": False, "failure_kind": None, "http_status": None},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "key",
                "https://gateway.example/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": None, "http_status": None},
            "degraded",
        ),
        (
            ProviderConfig("openai", "gpt-test", "key", "https://api.openai.com/v1", "openai/gpt-test"),
            {"ok": False, "failure_kind": "unavailable", "http_status": 429},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "key",
                "https://gateway.example/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": "invalid_configuration", "http_status": None},
            "fatal",
        ),
        (
            ProviderConfig(
                "openai-compatible",
                "gpt-test",
                "key",
                "http://your-server:8000/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": "unsupported", "http_status": None},
            "degraded",
        ),
        (
            ProviderConfig(
                "openai",
                "gpt-test",
                "key",
                "https://api.openai.com:443/v1",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": "authorization", "http_status": 403},
            "degraded",
        ),
        (
            ProviderConfig(
                "anthropic",
                "claude-alias",
                "key",
                "https://api.anthropic.com",
                "anthropic/claude-alias",
            ),
            {"ok": False, "failure_kind": None, "http_status": None},
            "degraded",
        ),
        (
            ProviderConfig(
                "anthropic",
                "claude-alias",
                "key",
                "https://api.anthropic.com:443/v1",
                "anthropic/claude-alias",
            ),
            {"ok": False, "failure_kind": "authorization", "http_status": 403},
            "fatal",
        ),
        (
            ProviderConfig(
                "openai",
                "gpt-test",
                "key",
                "https://api.openai.com:443/v1/",
                "openai/gpt-test",
            ),
            {"ok": False, "failure_kind": None, "http_status": None},
            "degraded",
        ),
        (
            ProviderConfig(
                "bedrock",
                "application-inference-profile/model-id",
                None,
                None,
                "bedrock/application-inference-profile/model-id",
                region="us-west-2",
            ),
            {"ok": False, "failure_kind": None, "http_status": None},
            "degraded",
        ),
    ],
)
def test_credential_probe_disposition_is_endpoint_aware(provider, probe: dict[str, object], expected: str) -> None:
    result = type(
        "Probe",
        (),
        {
            **probe,
            "provider": provider.provider,
            "model": provider.model,
            "detail": "safe detail",
        },
    )()

    assert runtime_preflight.credential_probe_disposition(provider, result) == expected


def test_model_probe_rejects_non_http_catalog_url() -> None:
    provider = ProviderConfig(
        provider="openai-compatible",
        model="model",
        api_key="secret-key",
        base_url="file:///etc",
        litellm_model="openai/model",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is False
    assert "HTTP or HTTPS" in result.detail


def test_model_probe_checks_bedrock_foundation_catalog(monkeypatch) -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class Bedrock:
        def list_foundation_models(self):
            return {"modelSummaries": [{"modelId": "anthropic.claude-sonnet-test-v1:0"}]}

    class Session:
        def client(self, service, **kwargs):
            captured.append((service, kwargs))
            return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=4.5)

    assert result.ok is True
    assert [service for service, _kwargs in captured] == ["bedrock", "bedrock-runtime"]
    for _service, kwargs in captured:
        assert kwargs["region_name"] == "us-west-2"
        request_config = kwargs["config"]
        assert request_config.connect_timeout == pytest.approx(4.5, rel=1e-3)
        assert request_config.read_timeout == pytest.approx(4.5, rel=1e-3)
        assert request_config.retries == {"max_attempts": 0}


@pytest.mark.parametrize(
    ("catalog_endpoint", "runtime_endpoint", "expected_disposition"),
    [
        (
            "https://bedrock.us-west-2.amazonaws.com",
            "https://bedrock-runtime.us-west-2.amazonaws.com",
            "verified",
        ),
        (
            "https://bedrock-fips.us-west-2.amazonaws.com",
            "https://bedrock-runtime-fips.us-west-2.amazonaws.com",
            "verified",
        ),
        (
            "https://bedrock.us-west-2.api.aws",
            "https://bedrock-runtime.us-west-2.api.aws",
            "verified",
        ),
        (
            "http://127.0.0.1:18001",
            "https://bedrock-runtime.us-west-2.amazonaws.com",
            "degraded",
        ),
        (
            "https://bedrock.us-west-2.amazonaws.com",
            "http://127.0.0.1:18002",
            "degraded",
        ),
        ("http://127.0.0.1:18001", "http://127.0.0.1:18002", "degraded"),
    ],
)
def test_bedrock_catalog_success_only_verifies_native_catalog_and_runtime_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    catalog_endpoint: str,
    runtime_endpoint: str,
    expected_disposition: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    botocore_session = BotocoreSession()

    class Client:
        def __init__(self, endpoint_url: str) -> None:
            self.meta = type("Meta", (), {"endpoint_url": endpoint_url})()

        def list_foundation_models(self):
            return {"modelSummaries": [{"modelId": "test-model"}]}

    class Session:
        _session = botocore_session

        def client(self, service: str, **_kwargs):
            return Client(catalog_endpoint if service == "bedrock" else runtime_endpoint)

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.ok is True
    assert runtime_preflight.credential_probe_disposition(provider, result) == expected_disposition


def test_bedrock_custom_endpoint_metadata_cannot_make_custom_hosts_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    data_path.mkdir()
    (data_path / "endpoints.json").write_text(
        json.dumps(
            {
                "version": 3,
                "partitions": [
                    {
                        "partition": "aws",
                        "partitionName": "AWS Standard",
                        "dnsSuffix": "example.invalid",
                        "regionRegex": "^(us)-\\w+-\\d+$",
                        "regions": {"us-west-2": {}},
                        "defaults": {
                            "hostname": "{service}.{region}.{dnsSuffix}",
                            "protocols": ["https"],
                            "signatureVersions": ["v4"],
                        },
                        "services": {
                            "bedrock": {"endpoints": {"us-west-2": {"hostname": "catalog.attacker.invalid"}}},
                            "bedrock-runtime": {"endpoints": {"us-west-2": {"hostname": "runtime.attacker.invalid"}}},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    builtin_data_path = Path(BotocoreSession().get_component("data_loader").BUILTIN_DATA_PATH)
    (data_path / "endpoints").symlink_to(builtin_data_path / "endpoints")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    real_session = runtime_preflight.boto3.session.Session()
    clients = {
        service_name: real_session.client(service_name, region_name="us-west-2")
        for service_name in ("bedrock", "bedrock-runtime")
    }
    clients["bedrock"].list_foundation_models = lambda: {"modelSummaries": [{"modelId": "test-model"}]}

    class Session:
        _session = real_session._session

        def get_credentials(self):
            return real_session.get_credentials()

        def client(self, service_name: str, **_kwargs):
            return clients[service_name]

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert clients["bedrock"].meta.endpoint_url == "https://catalog.attacker.invalid"
    assert clients["bedrock-runtime"].meta.endpoint_url == "https://runtime.attacker.invalid"
    assert result.ok is True
    assert result.catalog_authoritative is False
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"


def test_bedrock_custom_endpoint_rules_cannot_make_custom_hosts_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    endpoint_rule_set = {
        "version": "1.0",
        "parameters": {
            "Region": {"builtIn": "AWS::Region", "required": True, "type": "String"},
            "UseDualStack": {
                "builtIn": "AWS::UseDualStack",
                "required": True,
                "default": False,
                "type": "Boolean",
            },
            "UseFIPS": {
                "builtIn": "AWS::UseFIPS",
                "required": True,
                "default": False,
                "type": "Boolean",
            },
        },
        "rules": [
            {
                "conditions": [],
                "type": "endpoint",
                "endpoint": {
                    "url": "https://rules.attacker.invalid",
                    "properties": {},
                    "headers": {},
                },
            }
        ],
    }
    for service_name, api_version in (
        ("bedrock", "2023-04-20"),
        ("bedrock-runtime", "2023-09-30"),
    ):
        rule_path = data_path / service_name / api_version / "endpoint-rule-set-1.json"
        rule_path.parent.mkdir(parents=True)
        rule_path.write_text(json.dumps(endpoint_rule_set), encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    real_session = runtime_preflight.boto3.session.Session()
    clients = {
        service_name: real_session.client(service_name, region_name="us-west-2")
        for service_name in ("bedrock", "bedrock-runtime")
    }
    clients["bedrock"].list_foundation_models = lambda: {"modelSummaries": [{"modelId": "test-model"}]}

    class Session:
        _session = real_session._session

        def get_credentials(self):
            return real_session.get_credentials()

        def client(self, service_name: str, **_kwargs):
            return clients[service_name]

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    routed_urls = {
        clients[service_name]
        ._ruleset_resolver.construct_endpoint(
            clients[service_name].meta.service_model.operation_model(operation_name),
            {},
            {},
        )
        .url
        for service_name, operation_name in (
            ("bedrock", "ListFoundationModels"),
            ("bedrock-runtime", "InvokeModel"),
        )
    }
    assert routed_urls == {"https://rules.attacker.invalid"}
    assert result.ok is True
    assert result.catalog_authoritative is False
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"


@pytest.mark.parametrize(
    ("status_code", "expected_ok", "expected_kind", "expected_disposition"),
    [(200, True, None, "degraded"), (401, False, "authentication", "fatal")],
)
def test_bedrock_bearer_token_only_uses_real_catalog_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status_code: int,
    expected_ok: bool,
    expected_kind: str | None,
    expected_disposition: str,
) -> None:
    auth_schemes = BotocoreSession().get_service_model("bedrock").metadata.get("auth", [])
    if "smithy.api#httpBearerAuth" not in auth_schemes:
        pytest.skip("installed Botocore does not support Bedrock bearer authentication")

    seen_authorization: list[bool] = []

    class CatalogResponse(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen_authorization.append(self.headers.get("Authorization") == "Bearer test-bearer-token")
            body = (
                {"modelSummaries": [{"modelId": "test-model"}]}
                if status_code == 200
                else {"message": "private rejected bearer"}
            )
            encoded = json.dumps(body).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format, *_args) -> None:
            return

    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer-token")
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    with ThreadingHTTPServer(("127.0.0.1", 0), CatalogResponse) as server:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK", endpoint)
        monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", endpoint)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = runtime_preflight.probe_model(provider, timeout_seconds=5)
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert seen_authorization == [True]
    assert result.ok is expected_ok
    assert result.failure_kind == expected_kind
    assert runtime_preflight.credential_probe_disposition(provider, result) == expected_disposition
    assert "test-bearer-token" not in result.detail
    assert "private rejected bearer" not in result.detail


@pytest.mark.parametrize(
    "bearer_token",
    ["bad\nvalue", "bad\rvalue", "bad\x7fvalue", "bad value", "bad\N{LATIN SMALL LETTER E WITH ACUTE}value"],
)
def test_bedrock_model_probe_rejects_unsafe_bearer_token_characters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bearer_token: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "InvalidConfigError" in result.detail
    assert bearer_token not in result.detail


@pytest.mark.parametrize(
    ("error_code", "http_status", "expected_kind", "expected_disposition"),
    [
        ("UnexpectedAuthCode", 401, "authentication", "fatal"),
        ("InvalidClientTokenId", 400, "authentication", "fatal"),
        ("SignatureDoesNotMatch", 403, "authentication", "fatal"),
        ("InvalidAccessKeyId", 403, "authentication", "fatal"),
        ("ExpiredToken", 403, "authentication", "fatal"),
        ("MissingAuthenticationToken", 403, "authentication", "fatal"),
        ("RequestExpired", 400, "invalid_configuration", "fatal"),
        ("AccessDeniedException", 403, "authorization", "degraded"),
        ("ThrottlingException", 429, "unavailable", "degraded"),
        ("InternalServerException", 500, "unavailable", "degraded"),
        ("ServiceUnavailableException", 503, "unavailable", "degraded"),
    ],
)
def test_bedrock_model_probe_classifies_client_errors(
    monkeypatch,
    error_code: str,
    http_status: int,
    expected_kind: str,
    expected_disposition: str,
) -> None:
    class Bedrock:
        def list_foundation_models(self):
            raise runtime_preflight.ClientError(
                {
                    "Error": {"Code": error_code, "Message": "must stay private"},
                    "ResponseMetadata": {"HTTPStatusCode": http_status},
                },
                "ListFoundationModels",
            )

    class Session:
        def client(self, *_args, **_kwargs):
            return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is False
    assert result.failure_kind == expected_kind
    assert result.http_status == http_status
    assert runtime_preflight.credential_probe_disposition(provider, result) == expected_disposition
    assert "must stay private" not in result.detail


@pytest.mark.parametrize(
    ("operation_name", "error_code", "http_status", "expected_kind", "expected_disposition"),
    [
        ("AssumeRole", "AccessDenied", 403, "authentication", "fatal"),
        ("AssumeRoleWithWebIdentity", "InvalidIdentityToken", 400, "authentication", "fatal"),
        ("AssumeRoleWithWebIdentity", "IDPRejectedClaim", 400, "authentication", "fatal"),
        ("GetRoleCredentials", "AccessDeniedException", 403, "authentication", "fatal"),
        ("AssumeRoleWithWebIdentity", "IDPCommunicationError", 400, "unavailable", "degraded"),
        ("AssumeRole", "ThrottlingException", 429, "unavailable", "degraded"),
        ("AssumeRole", "InternalServerError", 500, "unavailable", "degraded"),
    ],
)
def test_bedrock_model_probe_classifies_credential_bootstrap_client_errors(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    error_code: str,
    http_status: int,
    expected_kind: str,
    expected_disposition: str,
) -> None:
    class Session:
        def __init__(self) -> None:
            raise runtime_preflight.ClientError(
                {
                    "Error": {"Code": error_code, "Message": "must stay private"},
                    "ResponseMetadata": {"HTTPStatusCode": http_status},
                },
                operation_name,
            )

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.failure_kind == expected_kind
    assert runtime_preflight.credential_probe_disposition(provider, result) == expected_disposition
    assert result.http_status == http_status
    assert "must stay private" not in result.detail


def test_bedrock_model_probe_degrades_when_active_imds_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED")
    monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "1")
    monkeypatch.setenv("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
    monkeypatch.setenv("NO_PROXY", "*")
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "unavailable"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "127.0.0.1" not in result.detail


@pytest.mark.parametrize("bearer_token", [None, " \t "])
def test_bedrock_model_probe_fails_when_credentials_are_absent_and_imds_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bearer_token: str | None,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    if bearer_token is not None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "authentication"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"


@pytest.mark.parametrize(
    ("status_code", "expected_kind", "expected_disposition"),
    [(500, "unavailable", "degraded"), (404, "authentication", "fatal")],
)
def test_bedrock_model_probe_distinguishes_transient_imds_from_no_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status_code: int,
    expected_kind: str,
    expected_disposition: str,
) -> None:
    requests: list[str] = []

    class MetadataResponse(BaseHTTPRequestHandler):
        def _respond(self) -> None:
            requests.append(self.path)
            self.send_response(status_code)
            self.end_headers()

        def do_GET(self) -> None:
            self._respond()

        def do_PUT(self) -> None:
            self._respond()

        def log_message(self, _format, *_args) -> None:
            return

    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED")
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "1")
    monkeypatch.setenv("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
    monkeypatch.setenv("NO_PROXY", "*")
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    with ThreadingHTTPServer(("127.0.0.1", 0), MetadataResponse) as server:
        monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = runtime_preflight.probe_model(provider, timeout_seconds=5)
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert requests
    assert result.failure_kind == expected_kind
    assert runtime_preflight.credential_probe_disposition(provider, result) == expected_disposition
    assert "127.0.0.1" not in result.detail


@pytest.mark.parametrize("credential_body", [b"{bad", b'{"AccessKeyId":"test"}', b""])
def test_bedrock_model_probe_degrades_malformed_imds_credential_responses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    credential_body: bytes,
) -> None:
    class MetadataResponse(BaseHTTPRequestHandler):
        def _write(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self) -> None:
            self._write(b"token")

        def do_GET(self) -> None:
            if self.path.endswith("/iam/security-credentials/"):
                self._write(b"TestRole")
            else:
                self._write(credential_body)

        def log_message(self, _format, *_args) -> None:
            return

    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED")
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "1")
    monkeypatch.setenv("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
    monkeypatch.setenv("NO_PROXY", "*")
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    with ThreadingHTTPServer(("127.0.0.1", 0), MetadataResponse) as server:
        monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = runtime_preflight.probe_model(provider, timeout_seconds=5)
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert result.failure_kind == "unavailable"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "TestRole" not in result.detail


def test_bedrock_model_probe_degrades_malformed_imds_role_listing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class MetadataResponse(BaseHTTPRequestHandler):
        def _write(self, status_code: int, body: bytes = b"") -> None:
            self.send_response(status_code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self) -> None:
            self._write(200, b"token")

        def do_GET(self) -> None:
            if self.path.endswith("/iam/security-credentials/"):
                self._write(200, b"Role1\nRole2")
            else:
                self._write(404)

        def log_message(self, _format, *_args) -> None:
            return

    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED")
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "1")
    monkeypatch.setenv("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
    monkeypatch.setenv("NO_PROXY", "*")
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    with ThreadingHTTPServer(("127.0.0.1", 0), MetadataResponse) as server:
        monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = runtime_preflight.probe_model(provider, timeout_seconds=5)
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert result.failure_kind == "unavailable"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "Role1" not in result.detail


@pytest.mark.parametrize("proxy_url", ["http://[::1", "http://127.0.0.1:banana"])
def test_bedrock_model_probe_classifies_malformed_effective_proxy_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    proxy_url: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    for variable in ("https_proxy", "all_proxy", "ALL_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("NO_PROXY", "")
    provider = ProviderConfig(
        provider="bedrock",
        model="test-model",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/test-model",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=2)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "HTTPClientError" in result.detail
    assert proxy_url not in result.detail


def test_bedrock_model_probe_degrades_valid_but_unavailable_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    for variable in ("https_proxy", "all_proxy", "ALL_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=2)

    assert result.failure_kind == "unavailable"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "127.0.0.1" not in result.detail


def test_bedrock_model_probe_does_not_blame_proxy_bypassed_by_no_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CatalogResponse(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b'{"modelSummaries":[{"modelId":"test-model"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args) -> None:
            return

    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    for variable in ("http_proxy", "all_proxy", "ALL_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://[::1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    with ThreadingHTTPServer(("127.0.0.1", 0), CatalogResponse) as server:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK", endpoint)
        monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", endpoint)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = runtime_preflight.probe_model(provider, timeout_seconds=5)
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert result.ok is True
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"


@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [
        (NoCredentialsError(), "authentication"),
        (UnauthorizedSSOTokenError(), "authentication"),
        (RefreshWithMFAUnsupportedError(), "authentication"),
        (SSOTokenLoadError(error_msg="private SSO token detail"), "authentication"),
        (NoAuthTokenError(), "authentication"),
        (
            PartialCredentialsError(
                provider="env",
                cred_var="AWS_SECRET_ACCESS_KEY",
            ),
            "invalid_configuration",
        ),
        (ProfileNotFound(profile="missing"), "invalid_configuration"),
        (ConfigParseError(path="/private/config"), "invalid_configuration"),
        (ConfigNotFound(path="/private/config"), "invalid_configuration"),
        (
            ApiVersionNotFoundError(data_path="/private/data", api_version="private version"),
            "invalid_configuration",
        ),
        (DataNotFoundError(data_path="/private/data"), "invalid_configuration"),
        (BaseEndpointResolverError(), "invalid_configuration"),
        (EndpointProviderError(msg="private endpoint detail"), "invalid_configuration"),
        (InvalidConfigError(error_msg="invalid profile"), "invalid_configuration"),
        (InvalidDefaultsMode(mode="private mode", valid_modes=["standard"]), "invalid_configuration"),
        (InvalidIMDSEndpointError(endpoint="private endpoint"), "invalid_configuration"),
        (
            InvalidIMDSEndpointModeError(mode="private mode", valid_modes=["ipv4", "ipv6"]),
            "invalid_configuration",
        ),
        (InvalidProxiesConfigError(), "invalid_configuration"),
        (InvalidRegionError(region_name="not a region"), "invalid_configuration"),
        (
            InvalidRetryConfigurationError(
                retry_config_option="private option",
                valid_options=["mode", "max_attempts"],
            ),
            "invalid_configuration",
        ),
        (
            InvalidSTSRegionalEndpointsConfigError(sts_regional_endpoints_config="private mode"),
            "invalid_configuration",
        ),
        (MissingDependencyException(msg="private dependency"), "invalid_configuration"),
        (ParamValidationError(report="private parameter detail"), "invalid_configuration"),
        (
            ServiceNotInRegionError(service_name="private service", region_name="private region"),
            "invalid_configuration",
        ),
        (UnknownCredentialError(name="private credential provider"), "invalid_configuration"),
        (UnknownRegionError(region_name="private region", error_msg="private detail"), "invalid_configuration"),
        (UnknownSignatureVersionError(signature_version="private signature"), "invalid_configuration"),
        (UnsupportedSignatureVersionError(signature_version="private signature"), "invalid_configuration"),
        (FileNotFoundError("private credential file"), "invalid_configuration"),
        (IsADirectoryError("private credential directory"), "invalid_configuration"),
        (NotADirectoryError("private credential parent"), "invalid_configuration"),
        (PermissionError("private credential file"), "invalid_configuration"),
        *(
            [
                (
                    botocore_exceptions.UnsupportedServiceProtocolsError(
                        botocore_supported_protocols="private protocols",
                        service="private service",
                        service_supported_protocols="private protocols",
                    ),
                    "invalid_configuration",
                )
            ]
            if hasattr(botocore_exceptions, "UnsupportedServiceProtocolsError")
            else []
        ),
        *(
            [
                (
                    botocore_exceptions.InvalidChecksumConfigError(
                        config_key="request_checksum_calculation",
                        valid_options=["when_supported"],
                        config_value="private value",
                    ),
                    "invalid_configuration",
                )
            ]
            if hasattr(botocore_exceptions, "InvalidChecksumConfigError")
            else []
        ),
        *(
            [
                (
                    botocore_exceptions.UnknownTokenProviderError(name="private token provider"),
                    "invalid_configuration",
                )
            ]
            if hasattr(botocore_exceptions, "UnknownTokenProviderError")
            else []
        ),
        *(
            [(botocore_exceptions.LoginRefreshRequired(), "authentication")]
            if hasattr(botocore_exceptions, "LoginRefreshRequired")
            else []
        ),
    ],
)
def test_bedrock_model_probe_fails_on_deterministic_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_kind: str,
) -> None:
    class Session:
        def __init__(self) -> None:
            raise error

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is False
    assert result.failure_kind == expected_kind
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert type(error).__name__ in result.detail
    assert str(error) not in result.detail


@pytest.mark.parametrize("phase", ["session", "catalog"])
def test_bedrock_model_probe_degrades_unattributed_value_errors(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    error = ValueError("private remote response detail")

    class Bedrock:
        def list_foundation_models(self):
            raise error

    class Session:
        def __init__(self) -> None:
            if phase == "session":
                raise error

        def client(self, *_args, **_kwargs):
            return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.failure_kind == ("unknown" if phase == "session" else "invalid_response")
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private remote response detail" not in result.detail


def test_bedrock_model_probe_degrades_transport_sdk_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Session:
        def __init__(self) -> None:
            raise EndpointConnectionError(endpoint_url="https://bedrock.us-west-2.amazonaws.com/private")

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.failure_kind == "unavailable"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private" not in result.detail


def test_bedrock_model_probe_classifies_real_invalid_region_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="not a region",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=1)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "InvalidRegionError" in result.detail
    assert "not a region" not in result.detail


@pytest.mark.parametrize(
    ("environment", "expected_error"),
    [
        ({"AWS_DEFAULTS_MODE": "private invalid mode"}, "InvalidDefaultsMode"),
        ({"AWS_EC2_METADATA_SERVICE_ENDPOINT": "private invalid endpoint"}, "InvalidIMDSEndpointError"),
        ({"AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE": "private invalid mode"}, "InvalidIMDSEndpointModeError"),
        ({"AWS_RETRY_MODE": "private invalid mode"}, "InvalidRetryModeError"),
        ({"AWS_MAX_ATTEMPTS": "private invalid integer"}, "ValueError"),
        ({"AWS_METADATA_SERVICE_TIMEOUT": "private invalid integer"}, "ValueError"),
        ({"AWS_METADATA_SERVICE_TIMEOUT": "0"}, "InvalidConfigError"),
        ({"AWS_METADATA_SERVICE_TIMEOUT": "-1"}, "InvalidConfigError"),
        ({"AWS_METADATA_SERVICE_NUM_ATTEMPTS": "private invalid integer"}, "ValueError"),
        ({"AWS_ENDPOINT_URL": "private invalid endpoint"}, "ValueError"),
        ({"AWS_ENDPOINT_URL_BEDROCK": "private invalid endpoint"}, "ValueError"),
        ({"AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "private invalid endpoint"}, "ValueError"),
        (
            {"AWS_CSM_ENABLED": "true", "AWS_CSM_PORT": "private invalid integer"},
            "ValueError",
        ),
    ],
)
def test_bedrock_model_probe_classifies_real_invalid_aws_environment_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected_error: str,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert expected_error in result.detail
    assert all(value not in result.detail for value in environment.values())


def test_bedrock_model_probe_classifies_real_invalid_profile_api_version_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
region = us-west-2
api_versions =
    bedrock = 1900-01-01
""",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "DataNotFoundError" in result.detail
    assert "1900-01-01" not in result.detail


def test_bedrock_model_probe_classifies_real_invalid_profile_max_attempts_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
region = us-west-2
max_attempts = private invalid integer
""",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "ValueError" in result.detail
    assert "private invalid integer" not in result.detail


@pytest.mark.parametrize("max_attempts", ["0", "-1"])
def test_bedrock_model_probe_classifies_out_of_range_profile_max_attempts_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_attempts: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config=f"""\
[profile test]
region = us-west-2
max_attempts = {max_attempts}
""",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "InvalidMaxRetryAttemptsError" in result.detail
    assert max_attempts not in result.detail


@pytest.mark.parametrize("service_name", ["bedrock", "bedrock-runtime"])
def test_bedrock_model_probe_classifies_real_malformed_local_service_data_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service_name: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    service_model = data_path / service_name / "9999-01-01" / "service-2.json"
    service_model.parent.mkdir(parents=True)
    service_model.write_text("{private malformed service data", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK", "http://127.0.0.1:1")
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "JSONDecodeError" in result.detail
    assert "private malformed service data" not in result.detail


def test_bedrock_model_probe_classifies_invalid_compressed_local_sdk_data_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    endpoint_rules = data_path / "bedrock" / "2023-04-20" / "endpoint-rule-set-1.json.gz"
    endpoint_rules.parent.mkdir(parents=True)
    endpoint_rules.write_text("private invalid gzip data", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "BadGzipFile" in result.detail
    assert "private invalid gzip data" not in result.detail


def test_bedrock_model_probe_does_not_blame_remote_bad_gzip_as_local_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Bedrock:
        def list_foundation_models(self):
            raise gzip.BadGzipFile("private remote response")

    class Session:
        def client(self, *_args, **_kwargs):
            return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "unknown"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private remote response" not in result.detail


@pytest.mark.parametrize("credential_source", ["web_identity", "container"])
def test_bedrock_model_probe_classifies_invalid_local_credential_token_encoding_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    credential_source: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    token_path = tmp_path / "private-token"
    token_path.write_bytes(b"\xff")
    if credential_source == "web_identity":
        monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/Test")
        monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token_path))
    else:
        monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://127.0.0.1:9/credentials")
        monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", str(token_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "UnicodeDecodeError" in result.detail
    assert str(token_path) not in result.detail


def test_bedrock_model_probe_does_not_blame_remote_unicode_error_as_local_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Bedrock:
        def list_foundation_models(self):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "private remote response")

    class Session:
        def client(self, *_args, **_kwargs):
            return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_response"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private remote response" not in result.detail


def test_bedrock_model_probe_classifies_malformed_assume_role_service_data_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
role_arn = arn:aws:iam::123456789012:role/Test
source_profile = source
region = us-west-2
""",
        credentials="""\
[source]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    service_model = data_path / "sts" / "9999-01-01" / "service-2.json"
    service_model.parent.mkdir(parents=True)
    service_model.write_text("{private malformed service data", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "JSONDecodeError" in result.detail
    assert "private malformed service data" not in result.detail


def test_bedrock_model_probe_classifies_malformed_assume_role_endpoint_rules_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
role_arn = arn:aws:iam::123456789012:role/Test
source_profile = source
region = us-west-2
""",
        credentials="""\
[source]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    endpoint_rules = data_path / "sts" / "2011-06-15" / "endpoint-rule-set-1.json"
    endpoint_rules.parent.mkdir(parents=True)
    endpoint_rules.write_text("{private malformed endpoint rules", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "JSONDecodeError" in result.detail
    assert "private malformed endpoint rules" not in result.detail


@pytest.mark.parametrize(
    ("service_name", "api_version"),
    [("bedrock", "2023-04-20"), ("bedrock-runtime", "2023-09-30")],
)
def test_bedrock_model_probe_classifies_malformed_runtime_endpoint_rules_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service_name: str,
    api_version: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    endpoint_rules = data_path / service_name / api_version / "endpoint-rule-set-1.json"
    endpoint_rules.parent.mkdir(parents=True)
    endpoint_rules.write_text("{private malformed endpoint rules", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "JSONDecodeError" in result.detail
    assert "private malformed endpoint rules" not in result.detail


def test_bedrock_model_probe_classifies_malformed_local_endpoints_data_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    data_path = tmp_path / "aws-data"
    data_path.mkdir()
    (data_path / "endpoints.json").write_text("{private malformed endpoints data", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "JSONDecodeError" in result.detail
    assert "private malformed endpoints data" not in result.detail


def test_bedrock_model_probe_classifies_malformed_nested_sso_service_data_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
role_arn = arn:aws:iam::123456789012:role/Test
source_profile = source
region = us-west-2

[profile source]
sso_start_url = https://private.awsapps.example/start
sso_region = us-west-2
sso_account_id = 123456789012
sso_role_name = TestRole
""",
    )
    data_path = tmp_path / "aws-data"
    service_model = data_path / "sso" / "9999-01-01" / "service-2.json"
    service_model.parent.mkdir(parents=True)
    service_model.write_text("{private malformed service data", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "JSONDecodeError" in result.detail
    assert "private malformed service data" not in result.detail


@pytest.mark.parametrize(
    ("expires_at", "expected_error"),
    [
        ("private malformed timestamp", "ParserError"),
        (None, "TypeError"),
        (123, "TypeError"),
        ([], "TypeError"),
        ({}, "TypeError"),
    ],
)
def test_bedrock_model_probe_classifies_malformed_local_sso_token_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    expires_at: object,
    expected_error: str,
) -> None:
    start_url = "https://private.awsapps.example/start"
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config=f"""\
[profile test]
region = us-west-2
sso_start_url = {start_url}
sso_region = us-west-2
sso_account_id = 123456789012
sso_role_name = TestRole
""",
    )
    cache_dir = tmp_path / "sso-cache"
    cache_dir.mkdir()
    cache_key = hashlib.sha1(start_url.encode()).hexdigest()
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps(
            {
                "accessToken": "private-access-token",
                "expiresAt": expires_at,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(botocore_credentials.SSOProvider, "_SSO_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(botocore_tokens.SSOTokenProvider, "_SSO_TOKEN_CACHE_DIR", str(cache_dir))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "authentication"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert expected_error in result.detail or "UnauthorizedSSOTokenError" in result.detail
    assert "private" not in result.detail


def test_bedrock_model_probe_classifies_malformed_local_login_token_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not hasattr(botocore_credentials, "LoginProvider"):
        pytest.skip("installed Botocore has no login credential provider")
    session_name = "private-login-session"
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config=f"""\
[profile test]
region = us-west-2
login_session = {session_name}
""",
    )
    cache_dir = tmp_path / "login-cache"
    cache_dir.mkdir()
    cache_key = hashlib.sha256(session_name.encode()).hexdigest()
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps(
            {
                "accessToken": {
                    "accessKeyId": "test-access-key",
                    "secretAccessKey": "test-secret-key",
                    "sessionToken": "test-token",
                    "accountId": "123456789012",
                    "expiresAt": "private malformed timestamp",
                },
                "refreshToken": "test-refresh-token",
                "dpopKey": "test-key",
                "clientId": "test-client",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_LOGIN_CACHE_DIRECTORY", str(cache_dir))
    monkeypatch.setattr(botocore_credentials, "EC", object(), raising=False)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "authentication"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "ParserError" in result.detail
    assert session_name not in result.detail


def test_bedrock_model_probe_classifies_malformed_credential_process_without_blame_unused_service_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credential_process = tmp_path / "credential-process.py"
    credential_process.write_text("print('{')\n", encoding="utf-8")
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config=f"""\
[profile test]
region = us-west-2
credential_process = "{sys.executable}" "{credential_process}"
""",
    )
    data_path = tmp_path / "aws-data"
    service_model = data_path / "sts" / "9999-01-01" / "service-2.json"
    service_model.parent.mkdir(parents=True)
    service_model.write_text("{\n", encoding="utf-8")
    monkeypatch.setenv("AWS_DATA_PATH", str(data_path))
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "JSONDecodeError" in result.detail
    assert str(credential_process) not in result.detail


@pytest.mark.parametrize("configuration_source", ["environment", "profile"])
@pytest.mark.parametrize("bundle_state", ["missing", "invalid"])
def test_bedrock_model_probe_classifies_proven_invalid_ca_bundle_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bundle_state: str,
    configuration_source: str,
) -> None:
    bundle_path = tmp_path / "private-ca-bundle.pem"
    if bundle_state == "invalid":
        bundle_path.write_text("private invalid CA data", encoding="utf-8")
    if configuration_source == "environment":
        monkeypatch.setenv("AWS_CA_BUNDLE", str(bundle_path))
        real_session = None
    else:
        _configure_isolated_aws_profile(
            monkeypatch,
            tmp_path,
            config=f"""\
[profile test]
region = us-west-2
ca_bundle = {bundle_path}
""",
            credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
        )
        real_session = runtime_preflight.boto3.session.Session()
    error = BotocoreSSLError(endpoint_url="https://private.example", error="private TLS detail")

    class Session:
        def __init__(self) -> None:
            if real_session is None:
                raise error
            self._session = real_session._session

        def client(self, *_args, **_kwargs):
            raise error

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "SSLError" in result.detail
    assert "private" not in result.detail


def test_bedrock_model_probe_degrades_unattributed_tls_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    error = BotocoreSSLError(endpoint_url="https://private.example", error="private TLS detail")

    class Session:
        def __init__(self) -> None:
            raise error

    monkeypatch.delenv("AWS_CA_BUNDLE", raising=False)
    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "unavailable"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private" not in result.detail


def test_bedrock_model_probe_rejects_invalid_ca_bundle_before_https_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "missing-ca-bundle.pem"
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    monkeypatch.setenv("AWS_CA_BUNDLE", str(bundle_path))
    real_session = runtime_preflight.boto3.session.Session()

    class Client:
        meta = type("Meta", (), {"endpoint_url": "https://private.example"})()

        def list_foundation_models(self):
            return {"modelSummaries": [{"modelId": "test-model"}]}

    class Session:
        def __init__(self) -> None:
            self._session = real_session._session

        def get_credentials(self):
            return real_session.get_credentials()

        def client(self, *_args, **_kwargs):
            return Client()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert str(bundle_path) not in result.detail


def test_bedrock_model_probe_classifies_real_invalid_credential_process_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
region = us-west-2
credential_process = private-command "unterminated
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "ValueError" in result.detail
    assert "private-command" not in result.detail


def test_bedrock_model_probe_classifies_real_unexecutable_credential_process_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credential_process = tmp_path / "credential-process"
    credential_process.write_text("private invalid executable", encoding="utf-8")
    credential_process.chmod(0o700)
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config=f"""\
[profile test]
region = us-west-2
credential_process = {credential_process}
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "OSError" in result.detail
    assert str(credential_process) not in result.detail


@pytest.mark.parametrize(
    ("payload", "exit_code", "expected_kind", "expected_disposition"),
    [
        ({"Version": 1}, 0, "invalid_configuration", "fatal"),
        ({"Version": 2}, 0, "invalid_configuration", "fatal"),
        (
            {
                "Version": 1,
                "AccessKeyId": "test-access-key",
                "SecretAccessKey": "test-secret-key",
                "SessionToken": "test-session-token",
                "Expiration": "private malformed timestamp",
            },
            0,
            "invalid_configuration",
            "fatal",
        ),
        (None, 1, "unavailable", "degraded"),
    ],
)
def test_bedrock_model_probe_classifies_structural_credential_process_output_causally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, object] | None,
    exit_code: int,
    expected_kind: str,
    expected_disposition: str,
) -> None:
    credential_process = tmp_path / "credential-process.py"
    if exit_code != 0:
        credential_process.write_text(
            'import sys\nprint("private process failure", file=sys.stderr)\nsys.exit(1)\n',
            encoding="utf-8",
        )
    else:
        credential_process.write_text(f"print({json.dumps(payload)!r})\n", encoding="utf-8")
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config=f"""\
[profile test]
region = us-west-2
credential_process = "{sys.executable}" "{credential_process}"
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == expected_kind
    assert runtime_preflight.credential_probe_disposition(provider, result) == expected_disposition
    assert "private" not in result.detail
    assert str(credential_process) not in result.detail


def test_bedrock_model_probe_degrades_transient_credential_process_spawn_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_init = botocore_credentials.ProcessProvider.__init__

    def failing_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)

        def fail_to_spawn(*_args, **_kwargs):
            raise BlockingIOError(errno.EAGAIN, "private temporary spawn failure")

        self._popen = fail_to_spawn

    monkeypatch.setattr(botocore_credentials.ProcessProvider, "__init__", failing_init)
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
region = us-west-2
credential_process = private-command
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "unavailable"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private" not in result.detail


@pytest.mark.parametrize(
    ("environment", "expected_error"),
    [
        ({"AWS_STS_REGIONAL_ENDPOINTS": "private invalid mode"}, "InvalidSTSRegionalEndpointsConfigError"),
        ({"AWS_RETRY_MODE": "private invalid mode"}, "InvalidRetryModeError"),
        ({"AWS_MAX_ATTEMPTS": "private invalid integer"}, "ValueError"),
        ({"AWS_ENDPOINT_URL_STS": "private invalid endpoint"}, "ValueError"),
    ],
)
def test_bedrock_model_probe_classifies_real_invalid_assume_role_configuration_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment: dict[str, str],
    expected_error: str,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
role_arn = arn:aws:iam::123456789012:role/Test
source_profile = source
region = us-west-2
""",
        credentials="""\
[source]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert expected_error in result.detail
    assert all(value not in result.detail for value in environment.values())


def test_bedrock_model_probe_classifies_real_invalid_assume_role_parameters_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="""\
[profile test]
role_arn = arn:aws:iam::123456789012:role/Test
role_session_name = !
source_profile = source
region = us-west-2
""",
        credentials="""\
[source]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider, timeout_seconds=5)

    assert result.failure_kind == "invalid_configuration"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "fatal"
    assert "ParamValidationError" in result.detail
    assert "role_session_name" not in result.detail


def test_bedrock_model_probe_degrades_malformed_catalog_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lifecycle_shape = BotocoreSession().get_service_model("bedrock").shape_for("FoundationModelLifecycle")
    if "startOfLifeTime" not in lifecycle_shape.members:
        pytest.skip("installed Botocore service model has no Bedrock lifecycle timestamps")
    requests: list[str] = []

    class MalformedCatalog(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            payload = json.dumps(
                {
                    "modelSummaries": [
                        {
                            "modelId": "test-model",
                            "modelLifecycle": {
                                "status": "ACTIVE",
                                "startOfLifeTime": "private malformed timestamp",
                            },
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
        credentials="""\
[test]
aws_access_key_id = test-access-key
aws_secret_access_key = test-secret-key
""",
    )
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with ThreadingHTTPServer(("127.0.0.1", 0), MalformedCatalog) as server:
        monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK", f"http://127.0.0.1:{server.server_port}")
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")
            result = runtime_preflight.probe_model(provider, timeout_seconds=5)
        finally:
            server.shutdown()
            thread.join(timeout=2)

    assert requests == ["/foundation-models"]
    assert result.failure_kind == "invalid_response"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private malformed timestamp" not in result.detail


def test_bedrock_model_probe_degrades_malformed_container_credential_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[str] = []

    class MalformedCredentials(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            payload = json.dumps(
                {
                    "AccessKeyId": "test-access-key",
                    "SecretAccessKey": "test-secret-key",
                    "Token": "test-token",
                    "Expiration": "private malformed timestamp",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    _configure_isolated_aws_profile(
        monkeypatch,
        tmp_path,
        config="[profile test]\nregion = us-west-2\n",
    )
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with ThreadingHTTPServer(("127.0.0.1", 0), MalformedCredentials) as server:
        monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", f"http://127.0.0.1:{server.server_port}/creds")
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = ProviderConfig("bedrock", "test-model", None, None, "bedrock/test-model", region="us-west-2")
            result = runtime_preflight.probe_model(provider, timeout_seconds=5)
        finally:
            server.shutdown()
            thread.join(timeout=2)

    assert requests == ["/creds"]
    assert result.failure_kind == "unknown"
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert "private malformed timestamp" not in result.detail


def test_bedrock_model_probe_uses_a_fresh_session_per_route(monkeypatch) -> None:
    sessions: list[object] = []

    class Bedrock:
        def list_foundation_models(self):
            return {"modelSummaries": []}

    class Session:
        def __init__(self):
            sessions.append(self)

        def client(self, *_args, **_kwargs):
            return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    first = ProviderConfig("bedrock", "model-a", None, None, "bedrock/model-a", region="us-west-2")
    second = ProviderConfig("bedrock", "model-b", None, None, "bedrock/model-b", region="us-west-2")

    runtime_preflight.probe_model(first)
    runtime_preflight.probe_model(second)

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]


def test_bedrock_model_probe_allows_distinct_routes_to_overlap(monkeypatch) -> None:
    rendezvous = Barrier(2)
    counter_lock = Lock()
    active = 0
    max_active = 0

    class Bedrock:
        def list_foundation_models(self):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                rendezvous.wait(timeout=0.5)
                return {"modelSummaries": []}
            finally:
                with counter_lock:
                    active -= 1

    class Session:
        def client(self, *_args, **_kwargs):
            return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    providers = [
        ProviderConfig("bedrock", "model-a", None, None, "bedrock/model-a", region="us-west-2"),
        ProviderConfig("bedrock", "model-b", None, None, "bedrock/model-b", region="us-west-2"),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(runtime_preflight.probe_model, providers))

    assert max_active == 2
    assert all(result.failure_kind is None and "not listed" in result.detail for result in results)


def test_bedrock_model_probe_deadline_covers_blocked_credential_provider(monkeypatch) -> None:
    release = Event()
    credential_provider_finished = Event()
    existing_threads = {id(thread) for thread in enumerate_threads()}

    class Session:
        def __init__(self):
            release.wait(timeout=1)
            credential_provider_finished.set()

        def client(self, *_args, **_kwargs):
            raise AssertionError("client creation must remain inside the deadline worker")

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "model", None, None, "bedrock/model", region="us-west-2")
    try:
        result = runtime_preflight.probe_model(provider, timeout_seconds=0.02)
        credential_provider_finished_before_release = credential_provider_finished.is_set()
    finally:
        release.set()
        for thread in enumerate_threads():
            if id(thread) not in existing_threads and thread.name == "bedrock-model-catalog-probe":
                thread.join(timeout=0.2)

    assert result.ok is False
    assert result.failure_kind == "unavailable"
    assert "timed out" in result.detail
    assert credential_provider_finished_before_release is False


def test_bedrock_model_probe_deadline_caps_blocked_workers(monkeypatch) -> None:
    release = Event()
    existing_threads = {id(thread) for thread in enumerate_threads()}

    class Session:
        def __init__(self):
            release.wait(timeout=1)

        def client(self, *_args, **_kwargs):
            raise AssertionError("client creation must remain inside the deadline worker")

    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    provider = ProviderConfig("bedrock", "model", None, None, "bedrock/model", region="us-west-2")
    try:
        results = [runtime_preflight.probe_model(provider, timeout_seconds=0.005) for _ in range(10)]
        probe_threads = [
            thread
            for thread in enumerate_threads()
            if id(thread) not in existing_threads and thread.name == "bedrock-model-catalog-probe"
        ]
    finally:
        release.set()

    assert all(result.failure_kind == "unavailable" and "timed out" in result.detail for result in results)
    assert len(probe_threads) == 4
    for thread in probe_threads:
        thread.join(timeout=0.2)


@pytest.mark.parametrize("failure_phase", ["constructor", "start"])
def test_bedrock_model_probe_releases_slot_when_worker_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    real_thread = runtime_preflight.Thread
    attempts = 0

    class ThreadFailsOnce:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1 and failure_phase == "constructor":
                raise RuntimeError("private worker-constructor detail")
            self._thread = real_thread(*args, **kwargs)

        def start(self) -> None:
            if attempts == 1 and failure_phase == "start":
                raise RuntimeError("private worker-start detail")
            self._thread.start()

    class Bedrock:
        def list_foundation_models(self):
            return {"modelSummaries": [{"modelId": "model"}]}

    class Session:
        def client(self, *_args, **_kwargs):
            return Bedrock()

    monkeypatch.setattr(runtime_preflight, "Thread", ThreadFailsOnce)
    monkeypatch.setattr(runtime_preflight.boto3.session, "Session", Session)
    monkeypatch.setattr(runtime_preflight, "_BEDROCK_PROBE_SLOT", runtime_preflight.BoundedSemaphore(1))
    provider = ProviderConfig("bedrock", "model", None, None, "bedrock/model", region="us-west-2")

    failed = runtime_preflight.probe_model(provider, timeout_seconds=0.1)
    recovered = runtime_preflight.probe_model(provider, timeout_seconds=0.1)

    assert failed.ok is False
    assert failed.failure_kind == "unknown"
    assert runtime_preflight.credential_probe_disposition(provider, failed) == "degraded"
    assert "RuntimeError" in failed.detail
    assert "private worker" not in failed.detail
    assert recovered.ok is True


def test_runtime_preflight_failure_stops_full_matrix(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import runner

    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvapi-test",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    def emit(_skill, target, **_kwargs):
        task = target / "case-001"
        task.mkdir(parents=True)
        return [task]

    full_matrix = Mock(return_value=[])
    preflight_run_env: dict[str, str] = {}
    monkeypatch.setenv("LLM_JUDGE_MODEL", "host-legacy")
    monkeypatch.delenv("SKILL_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"task_source": "evals_json"}}, None),
    )
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    monkeypatch.setattr(runner, "_run_agent_pair", full_matrix)
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected_provider: runtime_preflight.ModelProbeResult(
            True,
            selected_provider.provider,
            selected_provider.model,
            f"model {selected_provider.model} is available",
        ),
    )

    def fail_preflight(**kwargs):
        preflight_run_env.update(kwargs["run_env"])
        return runtime_preflight.PreflightResult(
            False,
            "opencode",
            "nvidia/meta/llama-3.1-8b-instruct",
            "401 Unauthorized",
            "runtime-preflight-opencode",
        )

    monkeypatch.setattr(runtime_preflight, "run_agent_runtime_preflight", fail_preflight)

    result = runner.run_harbor_eval(
        skill,
        ["opencode"],
        output_dir=tmp_path / "results",
        env_mode="docker",
        agent_runtime_preflight=True,
    )

    assert result["execution_status"] == "failed"
    assert result["execution_errors"] == ["opencode runtime preflight failed: 401 Unauthorized"]
    assert preflight_run_env["LLM_JUDGE_MODEL"] == "host-legacy"
    assert preflight_run_env["SKILL_EVAL_JUDGE_MODEL"] == "host-legacy"
    full_matrix.assert_not_called()
    result_path = Path(result["run_dir"]) / "result.json"
    assert result["result_path"] == str(result_path)
    assert result_path.is_file()
    assert result["harbor_jobs_retained"] is False
    assert result["harbor_jobs_retention_reason"] == "not_retained"
    assert not (Path(result["run_dir"]) / "_harbor-jobs").exists()
    assert not (Path(result["run_dir"]) / "_harbor-tasks").exists()
    assert result["run_config"]["credential_validation"]["status"] == "degraded"
    validation_targets = result["run_config"]["credential_validation"]["targets"]
    assert {target["status"] for target in validation_targets} == {"degraded"}
    assert {target["detail"] for target in validation_targets} == {
        "model catalog access does not verify runtime credentials for this endpoint"
    }
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["run_config"] == result["run_config"]
    run_config_path = Path(result["run_dir"]) / "run_config.json"
    assert json.loads(run_config_path.read_text(encoding="utf-8")) == result["run_config"]


@pytest.mark.parametrize("provider_name", ["openai", "openai-compatible", "anthropic"])
def test_live_custom_catalog_401_is_inconclusive_without_exposing_secret(provider_name: str) -> None:
    requests: list[str] = []
    secret = "loopback-provider-secret"

    class RejectingCatalog(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"rejected {secret}"}).encode())

        def log_message(self, *_args) -> None:
            return None

    with ThreadingHTTPServer(("127.0.0.1", 0), RejectingCatalog) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = ProviderConfig(
                provider=provider_name,
                model="requested-model",
                api_key=secret,
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                litellm_model=f"{'anthropic' if provider_name == 'anthropic' else 'openai'}/requested-model",
            )
            result = runtime_preflight.probe_model(provider)
        finally:
            server.shutdown()
            thread.join(timeout=2)

    assert result.ok is False
    assert result.failure_kind == "authentication"
    assert result.http_status == 401
    assert runtime_preflight.credential_probe_disposition(provider, result) == "degraded"
    assert secret not in result.detail
    assert requests == ["/v1/models"]
