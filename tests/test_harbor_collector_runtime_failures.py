# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Harbor agent-runtime failure classification."""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.html_report import generate_html_report
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET


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
    report = generate_html_report("demo", results_dir).read_text(encoding="utf-8")
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
