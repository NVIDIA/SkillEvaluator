# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Harbor agent-runtime smoke preflight regressions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

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

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["harbor", "run"], timeout=30)

    monkeypatch.setattr(runtime_preflight.subprocess, "run", timeout)

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=_dataset(tmp_path),
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        timeout_seconds=30,
    )

    assert result.ok is False
    assert "timed out after 30s" in result.detail


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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ModelCatalogError("model catalog returned HTTP 401")),
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
    captured: dict[str, object] = {}

    class Bedrock:
        def list_foundation_models(self):
            return {"modelSummaries": [{"modelId": "anthropic.claude-sonnet-test-v1:0"}]}

    def client(service, **kwargs):
        captured.update({"service": service, **kwargs})
        return Bedrock()

    monkeypatch.setattr(runtime_preflight.boto3, "client", client)
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-test-v1:0",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-sonnet-test-v1:0",
        region="us-west-2",
    )

    result = runtime_preflight.probe_model(provider)

    assert result.ok is True
    assert captured == {"service": "bedrock", "region_name": "us-west-2"}


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
        "run_agent_runtime_preflight",
        lambda **_kwargs: runtime_preflight.PreflightResult(
            False,
            "opencode",
            "nvidia/meta/llama-3.1-8b-instruct",
            "401 Unauthorized",
            "runtime-preflight-opencode",
        ),
    )

    result = runner.run_harbor_eval(
        skill,
        ["opencode"],
        output_dir=tmp_path / "results",
        env_mode="docker",
        agent_runtime_preflight=True,
    )

    assert result["execution_status"] == "failed"
    assert result["execution_errors"] == ["opencode runtime preflight failed: 401 Unauthorized"]
    full_matrix.assert_not_called()
    result_path = Path(result["run_dir"]) / "result.json"
    assert result["result_path"] == str(result_path)
    assert result_path.is_file()
    assert result["harbor_jobs_retained"] is False
    assert result["harbor_jobs_retention_reason"] == "not_retained"
    assert not (Path(result["run_dir"]) / "_harbor-jobs").exists()
    assert not (Path(result["run_dir"]) / "_harbor-tasks").exists()
