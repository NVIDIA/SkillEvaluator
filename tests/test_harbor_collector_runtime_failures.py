# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Harbor agent-runtime failure classification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest

from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET

_HARBOR_022_AGENT_RUNTIME_EXCEPTION_TYPES = (
    "AgentAuthenticationError",
    "ApiConnectionClosedError",
    "ApiInternalServerError",
    "ApiOverloadedError",
    "ApiProviderResourceNotFoundError",
    "ApiRateLimitError",
    "ApiResponseStalledError",
    "ApiUsageLimitError",
    "ContextWindowExceededError",
    "ModelNotFoundError",
    "NetworkConnectionError",
    "OutputTokenExceededError",
    "UnknownApiError",
)


def _expected_typed_runtime_reason(exception_type: str, message: str) -> str:
    if exception_type == "OutputTokenExceededError":
        return "OutputTokenExceededError:<redacted>"
    return f"{exception_type}: {message}"


def _write_actual_harbor_022_result(
    job_dir: Path,
    *,
    reward: float = 1.0,
    verifier_mode: Literal["present", "null", "missing"] = "present",
    exception_type: str | None = None,
    step_rewards: tuple[float, ...] | None = None,
    step_exception_type: str | None = None,
) -> str:
    """Persist a real Harbor 0.22 JobResult and its TrialResult artifact."""
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult

    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    now = datetime(2026, 8, 25, tzinfo=UTC)
    agent_context = {
        "n_input_tokens": 7,
        "n_cache_tokens": 2,
        "n_output_tokens": 3,
    }
    payload: dict[str, object] = {
        "id": UUID(int=2),
        "task_name": "nvidia/skillevaluator-case-001",
        "trial_name": trial_name,
        "trial_uri": trial_dir.as_uri(),
        "task_id": {"path": str(job_dir / "task" / "case-001")},
        "task_checksum": "harbor-0.22-fixture",
        "config": {
            "task": {"path": str(job_dir / "task" / "case-001")},
            "trial_name": trial_name,
        },
        "agent_info": {
            "name": "opencode",
            "version": "test",
            "model_info": {"name": "test-model"},
        },
        "agent_result": agent_context,
        "started_at": now,
        "finished_at": now,
        "step_results": None,
    }
    if verifier_mode == "present":
        payload["verifier_result"] = {"rewards": {"overall": reward}}
    elif verifier_mode == "null":
        payload["verifier_result"] = {"rewards": None}
    if step_rewards is not None:
        payload["agent_result"] = None
        step_results_payload: list[dict[str, object]] = []
        for index, step_reward in enumerate(step_rewards, start=1):
            step_result: dict[str, object] = {
                "step_name": f"step-{index}",
                "agent_result": agent_context,
                "verifier_result": {"rewards": {"overall": step_reward}},
            }
            if index == 1 and step_exception_type is not None:
                step_result["exception_info"] = {
                    "exception_type": step_exception_type,
                    "exception_message": "provider step operation failed",
                    "exception_traceback": "",
                    "occurred_at": now,
                }
            step_results_payload.append(step_result)
        payload["step_results"] = step_results_payload
    if exception_type is not None:
        payload["exception_info"] = {
            "exception_type": exception_type,
            "exception_message": "provider operation failed",
            "exception_traceback": "",
            "occurred_at": now,
        }
    trial_result = TrialResult.model_validate(payload)
    job_result = JobResult(
        id=UUID(int=1),
        started_at=now,
        updated_at=now,
        finished_at=now,
        n_total_trials=1,
        stats=JobStats.from_trial_results([trial_result], n_total_trials=1),
        trial_results=[trial_result],
    )
    (trial_dir / "result.json").write_text(trial_result.model_dump_json(indent=2), encoding="utf-8")
    (job_dir / "result.json").write_text(job_result.model_dump_json(indent=2), encoding="utf-8")
    for index, step_reward in enumerate(step_rewards or (), start=1):
        verifier_dir = trial_dir / "steps" / f"step-{index}" / "verifier"
        verifier_dir.mkdir(parents=True)
        (verifier_dir / "reward.json").write_text(
            json.dumps({"overall": step_reward, "entry_id": "case-001"}),
            encoding="utf-8",
        )
    return trial_name


@pytest.mark.parametrize(
    "exception_type",
    _HARBOR_022_AGENT_RUNTIME_EXCEPTION_TYPES,
)
def test_harbor_022_typed_infrastructure_failure_invalidates_present_reward(
    tmp_path: Path,
    exception_type: str,
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(job_dir, reward=1.0, exception_type=exception_type)

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    opencode = results["agents"]["opencode"]
    assert opencode["num_trials_with"] == 0
    assert opencode["with_skill"] == {}
    assert opencode["agent_runtime_failures"]["with_skill"] == [
        {
            "trial": trial_name,
            "reason": _expected_typed_runtime_reason(exception_type, "provider operation failed"),
        }
    ]


def test_harbor_022_safety_refusal_remains_a_scored_zero_outcome(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(job_dir, reward=0.0)
    agent_dir = job_dir / trial_name / "agent"
    agent_dir.mkdir()
    (agent_dir / "opencode.txt").write_text(
        json.dumps(
            {
                "type": "error",
                "error": {
                    "name": "AgentSafetyRefusalError",
                    "message": "the model declined this request on safety grounds",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "succeeded"
    assert opencode["num_trials_with"] == 1
    assert opencode["agent_runtime_failures"]["with_skill"] == []
    persisted_summary = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted_summary["overall_score"] == 0.0
    persisted_reward = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "trials" / trial_name / "reward.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted_reward["overall"] == 0.0


def test_harbor_022_safety_refusal_exception_is_not_an_infrastructure_failure(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(
        job_dir,
        reward=0.0,
        exception_type="AgentSafetyRefusalError",
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["num_trials_with"] == 0
    assert opencode["agent_runtime_failures"]["with_skill"] == []
    assert opencode["trial_failures"]["with_skill"] == [
        {"trial": trial_name, "reason": "AgentSafetyRefusalError: provider operation failed"}
    ]


def test_actual_harbor_022_single_step_success_serializes_and_scores(tmp_path: Path) -> None:
    from harbor.models.job.result import JobResult
    from harbor.models.trial.result import TrialResult

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(job_dir, reward=1.0)

    job_result = JobResult.model_validate_json((job_dir / "result.json").read_text(encoding="utf-8"))
    trial_result = TrialResult.model_validate_json((job_dir / trial_name / "result.json").read_text(encoding="utf-8"))
    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert job_result.stats.n_completed_trials == 1
    assert job_result.stats.n_errored_trials == 0
    assert trial_result.step_results is None
    assert trial_result.verifier_result is not None
    assert trial_result.verifier_result.rewards == {"overall": 1.0}
    assert results["execution_status"] == "succeeded"
    assert results["agents"]["opencode"]["num_trials_with"] == 1
    summary = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["overall_score"] == 1.0


@pytest.mark.parametrize("verifier_mode", ("null", "missing"))
def test_actual_harbor_022_null_or_missing_reward_is_unscored(
    tmp_path: Path,
    verifier_mode: Literal["null", "missing"],
) -> None:
    from harbor.models.job.result import JobResult
    from harbor.models.trial.result import TrialResult

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(job_dir, verifier_mode=verifier_mode)

    job_result = JobResult.model_validate_json((job_dir / "result.json").read_text(encoding="utf-8"))
    trial_result = TrialResult.model_validate_json((job_dir / trial_name / "result.json").read_text(encoding="utf-8"))
    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert job_result.stats.n_completed_trials == 1
    assert job_result.stats.n_errored_trials == 0
    assert trial_result.step_results is None
    if verifier_mode == "null":
        assert trial_result.verifier_result is not None
        assert trial_result.verifier_result.rewards is None
    else:
        assert trial_result.verifier_result is None
    assert results["execution_status"] == "failed"
    assert results["agents"]["opencode"]["num_trials_with"] == 0
    assert results["agents"]["opencode"]["job_failures"]["with_skill"] == (
        "Harbor evaluation statistics account for 0/1 completed trials"
    )


@pytest.mark.parametrize(
    ("exception_type", "job_failure"),
    (
        ("RuntimeError", "Harbor job did not complete successfully: 1 errored"),
        ("CancelledError", "Harbor job did not complete successfully: 1 cancelled"),
    ),
)
def test_actual_harbor_022_error_or_cancelled_job_suppresses_reward(
    tmp_path: Path,
    exception_type: str,
    job_failure: str,
) -> None:
    from harbor.models.job.result import JobResult

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    _write_actual_harbor_022_result(job_dir, reward=1.0, exception_type=exception_type)

    job_result = JobResult.model_validate_json((job_dir / "result.json").read_text(encoding="utf-8"))
    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert job_result.stats.n_errored_trials == 1
    assert job_result.stats.n_cancelled_trials == (exception_type == "CancelledError")
    assert results["execution_status"] == "failed"
    assert results["agents"]["opencode"]["num_trials_with"] == 0
    assert results["agents"]["opencode"]["job_failures"]["with_skill"] == job_failure


def test_actual_harbor_022_multistep_root_reward_is_authoritative(tmp_path: Path) -> None:
    from harbor.models.trial.result import TrialResult

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(job_dir, reward=0.8, step_rewards=(0.0, 0.2))

    trial_result = TrialResult.model_validate_json((job_dir / trial_name / "result.json").read_text(encoding="utf-8"))
    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert trial_result.agent_result is None
    assert trial_result.step_results is not None
    assert [step.verifier_result.rewards for step in trial_result.step_results if step.verifier_result] == [
        {"overall": 0.0},
        {"overall": 0.2},
    ]
    assert trial_result.verifier_result is not None
    assert trial_result.verifier_result.rewards == {"overall": 0.8}
    assert results["execution_status"] == "succeeded"
    assert results["agents"]["opencode"]["num_trials_with"] == 1
    summary = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["overall_score"] == 0.8


@pytest.mark.parametrize(
    "exception_type",
    _HARBOR_022_AGENT_RUNTIME_EXCEPTION_TYPES,
)
def test_actual_harbor_022_multistep_typed_agent_failure_is_infrastructure_failure(
    tmp_path: Path,
    exception_type: str,
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(
        job_dir,
        reward=1.0,
        step_rewards=(1.0, 1.0),
        step_exception_type=exception_type,
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["num_trials_with"] == 0
    assert opencode["agent_runtime_failures"]["with_skill"] == [
        {
            "trial": trial_name,
            "reason": _expected_typed_runtime_reason(exception_type, "provider step operation failed"),
        }
    ]
    assert opencode["trial_failures"]["with_skill"] == []


def test_actual_harbor_022_multistep_safety_refusal_stays_out_of_infrastructure_failures(
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = _write_actual_harbor_022_result(
        job_dir,
        reward=0.0,
        step_rewards=(0.0, 0.0),
        step_exception_type="AgentSafetyRefusalError",
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["num_trials_with"] == 0
    assert opencode["agent_runtime_failures"]["with_skill"] == []
    assert opencode["trial_failures"]["with_skill"] == [
        {
            "trial": trial_name,
            "reason": (
                "Required judge evaluation failed: collector: Constituent default reward for step step-1 "
                "is incomplete, non-finite, or failed; the authoritative aggregate was not scored"
            ),
        }
    ]


def _write_complete_job_result(job_dir: Path, trial_names: list[str]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(trial_names),
                "stats": {
                    "n_trials": len(trial_names),
                    "n_errors": 0,
                    "evals": {
                        "agent__model___harbor-tasks": {
                            "n_trials": len(trial_names),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.1": trial_names}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_responses_api_not_found_invalidates_agent_trial_rewards(tmp_path: Path) -> None:
    """A failed Codex Responses API request must not become a scored trial."""
    jobs_dir = tmp_path / "jobs"
    trial_dir = jobs_dir / "demo-codex-with" / "case-001__attempt"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": "unexpected status 404 Not Found: https://integrate.api.nvidia.com/v1/responses",
                }
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "agent" / "codex.txt").write_text(
        "ERROR responses_websocket: HTTP error: 405 Method Not Allowed\n"
        "unexpected status 404 Not Found: https://integrate.api.nvidia.com/v1/responses\n",
        encoding="utf-8",
    )
    (trial_dir / "verifier" / "reward.json").write_text(
        json.dumps({"security": 1.0, "accuracy": 1.0, "overall": 1.0}),
        encoding="utf-8",
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["codex"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
    )

    codex = results["agents"]["codex"]
    assert codex["num_trials_with"] == 0
    assert codex["with_skill"] == {}
    failures = codex["agent_runtime_failures"]["with_skill"]
    assert len(failures) == 1
    assert failures[0]["trial"] == "case-001__attempt"
    assert failures[0]["reason"] in {
        "405 Method Not Allowed",
        "404 Not Found: https://integrate.api.nvidia.com/v1/responses",
    }


def test_agent_timeout_invalidates_reward_and_is_reported_as_trial_failure(tmp_path: Path) -> None:
    """A timeout can leave verifier rewards behind, but it is not a valid scored trial."""
    jobs_dir = tmp_path / "jobs"
    trial_dir = jobs_dir / "demo-opencode-with" / "case-001__attempt"
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "AgentTimeoutError",
                    "exception_message": "Agent timed out after 600 seconds",
                }
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "verifier" / "reward.json").write_text(
        json.dumps({"security": 1.0, "accuracy": 1.0, "overall": 1.0}),
        encoding="utf-8",
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
    )

    opencode = results["agents"]["opencode"]
    assert opencode["num_trials_with"] == 0
    assert opencode["with_skill"] == {}
    assert opencode["lift"] == {}
    assert opencode["agent_runtime_failures"]["with_skill"] == [
        {"trial": "case-001__attempt", "reason": "AgentTimeoutError: Agent timed out after 600 seconds"}
    ]
    assert opencode["trial_failures"]["with_skill"] == [
        {"trial": "case-001__attempt", "reason": "AgentTimeoutError: Agent timed out after 600 seconds"}
    ]


def test_errored_job_stats_suppress_rewards_without_trial_exception(tmp_path: Path) -> None:
    """Aggregate Harbor failure state wins even when a trial reward looks valid."""
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_dir = job_dir / "case-001__attempt"
    (trial_dir / "verifier").mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_trials": 1,
                    "n_errors": 1,
                    "evals": {
                        "eval": {
                            "n_trials": 0,
                            "n_errors": 1,
                            "reward_stats": {},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "result.json").write_text(json.dumps({"trial_name": "case-001__attempt"}), encoding="utf-8")
    (trial_dir / "verifier" / "reward.json").write_text(
        json.dumps({"security": 1.0, "accuracy": 1.0, "overall": 1.0}),
        encoding="utf-8",
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
    )

    opencode = results["agents"]["opencode"]
    assert opencode["num_trials_with"] == 0
    assert opencode["with_skill"] == {}
    assert opencode["job_failures"]["with_skill"] == "Harbor job did not complete successfully: 1 errored"


def test_partially_errored_job_preserves_only_completed_trial_coverage(tmp_path: Path) -> None:
    """Known failed trials are excluded without hiding the other completed attempts."""
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    case_ids = ["case-001", "case-002", "case-003", "case-004"]
    trial_names = [f"{case_id}__attempt" for case_id in case_ids]
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 4,
                "stats": {
                    "n_completed_trials": 4,
                    "n_errored_trials": 1,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {},
                },
            }
        ),
        encoding="utf-8",
    )
    for index, (case_id, trial_name) in enumerate(zip(case_ids, trial_names, strict=True), start=1):
        trial_dir = job_dir / trial_name
        (trial_dir / "verifier").mkdir(parents=True)
        result: dict[str, object] = {"trial_name": trial_name}
        if case_id == "case-002":
            result["exception_info"] = {
                "exception_type": "AgentTimeoutError",
                "exception_message": "Agent execution timed out after 300.0 seconds",
            }
        (trial_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (trial_dir / "verifier" / "reward.json").write_text(
            json.dumps({"entry_id": case_id, "overall": index / 10}),
            encoding="utf-8",
        )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=4,
        expected_case_ids=case_ids,
        expected_trials=4,
    )

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert results["expected_attempts"] == 4
    assert results["scored_attempts"] == 3
    assert opencode["num_trials_with"] == 3
    assert opencode["conditions"]["with_skill"]["scored_attempts"] == 3
    assert opencode["trial_failures"]["with_skill"] == [
        {
            "trial": "case-002__attempt",
            "reason": "AgentTimeoutError: Agent execution timed out after 300.0 seconds",
        }
    ]
    assert any("Scored attempt coverage is 3/4" in error for error in results["execution_errors"])


def test_partial_rewards_stay_suppressed_when_not_every_job_error_maps_to_a_trial(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 2,
                "stats": {
                    "n_completed_trials": 2,
                    "n_errored_trials": 2,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {},
                },
            }
        ),
        encoding="utf-8",
    )
    for case_id in ("case-001", "case-002"):
        trial_dir = job_dir / f"{case_id}__attempt"
        (trial_dir / "verifier").mkdir(parents=True)
        result: dict[str, object] = {"trial_name": trial_dir.name}
        if case_id == "case-001":
            result["exception_info"] = {
                "exception_type": "AgentTimeoutError",
                "exception_message": "Agent execution timed out",
            }
        (trial_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (trial_dir / "verifier" / "reward.json").write_text(
            json.dumps({"entry_id": case_id, "overall": 1.0}),
            encoding="utf-8",
        )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=2,
        expected_case_ids=["case-001", "case-002"],
        expected_trials=2,
    )

    assert results["execution_status"] == "failed"
    assert results["scored_attempts"] == 0
    assert results["agents"]["opencode"]["num_trials_with"] == 0


def test_complete_low_score_is_execution_success(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "verifier" / "reward.json").write_text(
        json.dumps({"overall": 0.1, "entry_id": "case-001"}),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        n_attempts=1,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    condition = results["agents"]["opencode"]["conditions"]["with_skill"]
    assert condition == {
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 1,
        "scored_attempts": 1,
    }
    assert results["execution_status"] == "succeeded"
    assert "error" not in results


def test_incomplete_default_reward_is_unscored_and_reported(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = "case-001__attempt"
    verifier = job_dir / trial_name / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(
        json.dumps(
            {
                "metric_set": DEFAULT_METRIC_SET,
                "security": 1.0,
                "entry_id": "case-001",
            }
        ),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])
    results_dir = tmp_path / "results"

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=results_dir,
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert results["scored_attempts"] == 0
    assert opencode["num_trials_with"] == 0
    assert opencode["with_skill"] == {}
    assert opencode["trial_failures"]["with_skill"] == [
        {
            "trial": trial_name,
            "reason": "Reward metrics are incomplete or non-finite; trial was not scored",
        }
    ]
    skill = tmp_path / "demo"
    skill.mkdir()
    report = render_agent_eval_html_report(
        skill,
        results_dir,
        use_llm_judge=False,
    ).read_text(encoding="utf-8")
    assert "Reward metrics are incomplete or non-finite; trial was not scored" in report


def test_missing_job_result_fails_execution_and_preserves_error_alias(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_dir = jobs_dir / "demo-opencode-with" / "case-001__attempt" / "verifier"
    trial_dir.mkdir(parents=True)
    (trial_dir / "reward.json").write_text(json.dumps({"overall": 1.0}), encoding="utf-8")

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert results["execution_status"] == "failed"
    assert results["scored_attempts"] == 0
    assert results["error"] == results["execution_errors"]
    assert "result.json" in results["error"][0]
    persisted = json.loads((tmp_path / "results" / "opencode" / "with-skill" / "summary.json").read_text())
    assert persisted["execution_status"] == "failed"
    assert persisted["scored_attempts"] == 0


def test_native_multistep_rewards_count_as_one_logical_attempt(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = "case-001__attempt"
    for step in ("prepare", "finish"):
        verifier = job_dir / trial_name / "steps" / step / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(
            json.dumps({"overall": 0.8, "entry_id": "case-001"}),
            encoding="utf-8",
        )
    _write_complete_job_result(job_dir, [trial_name])

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert results["execution_status"] == "succeeded"
    assert results["scored_attempts"] == 1
    assert results["agents"]["opencode"]["num_trials_with"] == 2


def test_unexpected_case_fails_execution_coverage(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = "case-evil__attempt"
    verifier = job_dir / trial_name / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(
        json.dumps({"overall": 1.0, "entry_id": "case-evil"}),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert results["execution_status"] == "failed"
    assert any("Unexpected scored cases: case-evil" in error for error in results["execution_errors"])


CASES = ("case-a", "case-b")


def _write_reward(
    jobs_dir: Path,
    *,
    variant: str,
    case_id: str,
    attempt: int,
    score: float = 0.25,
    steps: tuple[str, ...] = (),
    trial_name: str | None = None,
    include_entry_id: bool = True,
    result_task_name: str | None = None,
) -> None:
    trial = jobs_dir / f"demo-opencode-{variant}" / (trial_name or f"{case_id}_attempt{attempt:03d}")
    verifier_dirs = [trial / "steps" / step / "verifier" for step in steps] or [trial / "verifier"]
    reward = {
        "overall": score,
        "security": score,
        "skill_execution": score,
        "skill_efficiency": score,
        "accuracy": score,
        "goal_accuracy": score,
        "behavior_check": score,
    }
    if include_entry_id:
        reward["entry_id"] = case_id
    for verifier_dir in verifier_dirs:
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    if result_task_name is not None:
        (trial / "result.json").write_text(
            json.dumps({"trial_name": trial.name, "task_name": result_task_name}),
            encoding="utf-8",
        )


def _write_variant_job_results(jobs_dir: Path, variants: tuple[str, ...] = ("with", "without")) -> None:
    """Persist a complete Harbor job result covering every staged trial directory."""
    for variant in variants:
        job_dir = jobs_dir / f"demo-opencode-{variant}"
        if not job_dir.is_dir():
            continue
        trial_names = sorted(path.name for path in job_dir.iterdir() if path.is_dir())
        _write_complete_job_result(job_dir, trial_names)


def _collect(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    options: dict[str, object] = {
        "n_attempts": 2,
        "expected_cases": 2,
        "expected_case_ids": list(CASES),
    }
    options.update(kwargs)
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        **options,
    )


def test_stop_on_pass_does_not_report_intentionally_skipped_attempts(tmp_path: Path) -> None:
    for variant in ("with", "without"):
        for case_id in CASES:
            _write_reward(tmp_path / "jobs", variant=variant, case_id=case_id, attempt=1, score=1.0)
    _write_variant_job_results(tmp_path / "jobs")

    result = _collect(tmp_path, n_attempts=3, stop_on_pass=True)

    assert result["execution_status"] == "succeeded"
    assert result["expected_attempts"] == 4
    assert result["scored_attempts"] == 4


def test_complete_ab_run_records_paired_pass_evidence(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=1, score=1.0)
    _write_reward(jobs_dir, variant="with", case_id="case-b", attempt=1, score=1.0)
    _write_reward(jobs_dir, variant="without", case_id="case-a", attempt=1, score=0.0)
    _write_reward(jobs_dir, variant="without", case_id="case-b", attempt=1, score=1.0)
    _write_variant_job_results(jobs_dir)

    result = _collect(tmp_path, n_attempts=1)

    assert result["execution_status"] == "succeeded"
    pass_at_k = result["agents"]["opencode"]["pass_at_k"]
    assert pass_at_k["with_skill"]["rate_interval"]["confidence_level"] == 0.95
    paired = pass_at_k["lift"]["paired_comparison"]
    assert paired["pairing_status"] == "complete"
    assert paired["paired_cases"] == 2
    assert paired["with_skill_only_pass"] == 1
    assert paired["without_skill_only_pass"] == 0
    assert paired["paired_rate_delta"] == 0.5
    assert paired["mcnemar_exact"]["p_value"] == 1.0

    persisted = json.loads((tmp_path / "results/opencode/pass_at_k_lift.json").read_text(encoding="utf-8"))
    assert persisted["delta"] == 0.5
    assert persisted["count_derived_delta"] == 0.5
    assert persisted["paired_comparison"] == paired


def test_result_derived_case_ids_exercise_partial_pairing_through_collector(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_reward(
        jobs_dir,
        variant="with",
        case_id="unused",
        attempt=1,
        score=1.0,
        trial_name="opaque-with-shared",
        include_entry_id=False,
        result_task_name="suite/shared",
    )
    _write_reward(
        jobs_dir,
        variant="with",
        case_id="unused",
        attempt=1,
        score=1.0,
        trial_name="opaque-with-only",
        include_entry_id=False,
        result_task_name="suite/with-only",
    )
    _write_reward(
        jobs_dir,
        variant="without",
        case_id="unused",
        attempt=1,
        score=0.0,
        trial_name="opaque-without-shared",
        include_entry_id=False,
        result_task_name="suite/shared",
    )
    _write_reward(
        jobs_dir,
        variant="without",
        case_id="unused",
        attempt=1,
        score=0.0,
        trial_name="opaque-without-only",
        include_entry_id=False,
        result_task_name="suite/without-only",
    )
    _write_variant_job_results(jobs_dir)

    result = _collect(tmp_path, n_attempts=1, expected_cases=2, expected_case_ids=None)

    assert result["execution_status"] == "succeeded"
    paired = result["agents"]["opencode"]["pass_at_k"]["lift"]["paired_comparison"]
    assert paired["pairing_status"] == "partial"
    assert paired["paired_cases"] == 1
    assert paired["with_skill_unpaired_case_ids"] == ["with-only"]
    assert paired["without_skill_unpaired_case_ids"] == ["without-only"]
    assert "mcnemar_exact" not in paired


def test_stop_on_pass_records_skipped_attempts_in_pass_summary(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=1, score=0.2)
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=2, score=1.0)
    _write_variant_job_results(jobs_dir, variants=("with",))

    result = _collect(
        tmp_path,
        skip_baseline=True,
        n_attempts=3,
        stop_on_pass=True,
        expected_cases=1,
        expected_case_ids=["case-a"],
    )

    pass_at_k = result["agents"]["opencode"]["pass_at_k"]["with_skill"]
    assert pass_at_k["stop_on_pass"] is True
    case = pass_at_k["cases"]["case-a"]
    assert case["passed"] is True
    assert case["first_pass_attempt"] == 2
    assert case["attempts_used"] == 2
    assert case["attempts_skipped"] == 1
    assert case["attempts_missing"] == 0
    assert result["attempt_policy"]["stop_on_pass"] is True
    assert result["execution_status"] == "succeeded"


def test_stop_on_pass_rejects_a_lone_late_attempt(tmp_path: Path) -> None:
    _write_reward(tmp_path / "jobs", variant="with", case_id="case-a", attempt=3, score=1.0)
    _write_variant_job_results(tmp_path / "jobs", variants=("with",))

    result = _collect(
        tmp_path,
        skip_baseline=True,
        n_attempts=3,
        stop_on_pass=True,
        expected_cases=1,
        expected_case_ids=["case-a"],
    )

    assert result["execution_status"] == "failed"
    assert result["expected_attempts"] == 3
    assert result["scored_attempts"] == 1


def test_stop_on_pass_rejects_failed_attempt_before_pass(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=1, score=1.0)
    failed_trial = jobs_dir / "demo-opencode-with/case-a_attempt001"
    (failed_trial / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "TaskFailure",
                    "exception_message": "attempt one crashed",
                }
            }
        ),
        encoding="utf-8",
    )
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=2, score=1.0)
    _write_variant_job_results(jobs_dir, variants=("with",))

    result = _collect(
        tmp_path,
        skip_baseline=True,
        n_attempts=3,
        stop_on_pass=True,
        expected_cases=1,
        expected_case_ids=["case-a"],
    )

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["trial_failures"]["with_skill"]


def test_multistep_stop_on_pass_uses_authoritative_root_reward(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=1, score=1.0, steps=("prepare",))
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=1, score=0.0, steps=("finish",))
    trial = jobs_dir / "demo-opencode-with/case-a_attempt001"
    (trial / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "case-a_attempt001",
                "task_name": "case-a",
                "verifier_result": {"rewards": {"overall": 0.5}},
                "step_results": [
                    {"step_name": "prepare", "verifier_result": {"rewards": {"overall": 1.0}}},
                    {"step_name": "finish", "verifier_result": {"rewards": {"overall": 0.0}}},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_variant_job_results(jobs_dir, variants=("with",))

    result = _collect(
        tmp_path,
        skip_baseline=True,
        n_attempts=2,
        pass_threshold=0.75,
        stop_on_pass=True,
        expected_cases=1,
        expected_case_ids=["case-a"],
    )

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert result["expected_attempts"] == 2
    assert result["scored_attempts"] == 1
    assert agent["pass_at_k"]["with_skill"] == {}


def test_duplicate_logical_attempt_ordinals_fail(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_reward(jobs_dir, variant="with", case_id="case-a", attempt=1)
    _write_reward(
        jobs_dir,
        variant="with",
        case_id="case-a",
        attempt=1,
        trial_name="copy-case-a_attempt001",
    )
    _write_variant_job_results(jobs_dir, variants=("with",))

    result = _collect(
        tmp_path,
        skip_baseline=True,
        n_attempts=2,
        expected_cases=1,
        expected_case_ids=["case-a"],
    )

    assert result["execution_status"] == "failed"
    assert any("duplicate attempt ordinals" in str(error) for error in result["execution_errors"])


def test_structured_opencode_resource_exhaustion_invalidates_no_trajectory_reward(tmp_path: Path) -> None:
    """A provider error event is an agent failure even when Harbor exits cleanly."""
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = "managing-teams-001__attempt"
    trial_dir = job_dir / trial_name
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    (trial_dir / "agent" / "opencode.txt").write_text(
        json.dumps(
            {
                "type": "error",
                "error": {
                    "name": "UnknownError",
                    "data": {"message": '"ResourceExhausted: Worker local total request limit reached (32/32)"'},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (trial_dir / "verifier" / "reward.json").write_text(
        json.dumps(
            {
                "overall": 0.0,
                "entry_id": "skillevaluator-managing-teams-001",
                "error": "No trajectory or reconstructible agent log",
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "nvidia/skillevaluator-managing-teams-001",
                "trial_name": trial_name,
            }
        ),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["managing-teams-001"],
        expected_trials=1,
    )

    opencode = results["agents"]["opencode"]
    assert opencode["num_trials_with"] == 0
    assert opencode["agent_runtime_failures"]["with_skill"] == [
        {
            "trial": trial_name,
            "reason": "ResourceExhausted: Worker local total request limit reached (32/32)",
        }
    ]
    errors = opencode["conditions"]["with_skill"]["execution_errors"]
    assert any("ResourceExhausted: Worker local total request limit reached (32/32)" in error for error in errors)
    assert not any("Unexpected scored cases" in error for error in errors)


def test_expected_case_normalizes_generated_skillevaluator_task_prefix(tmp_path: Path) -> None:
    """Fallback task metadata must not replace the original staged case id."""
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-opencode-with"
    trial_name = "managing-teams-001__attempt"
    verifier_dir = job_dir / trial_name / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "reward.json").write_text(
        json.dumps({"overall": 0.5, "entry_id": "skillevaluator-managing-teams-001"}),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["managing-teams-001"],
        expected_trials=1,
    )

    assert results["execution_status"] == "succeeded"
    assert results["agents"]["opencode"]["pass_at_k"]["with_skill"]["extra_cases"] == []
