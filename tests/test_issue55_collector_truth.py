# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end collector truthfulness regressions for issue #55."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.tier3.harbor import collector as collector_module
from skillevaluator.tier3.harbor import report, report_data
from skillevaluator.tier3.harbor.collector import (
    _copy_trial_artifacts,
    _extract_rewards,
    _write_redacted_text_copy,
    collect_harbor_results,
)
from skillevaluator.tier3.harbor.metrics import (
    CUSTOM_ONLY_METRIC_SET,
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    LEGACY_METRIC_SET,
    LEGACY_METRICS,
    overall_score,
)


def _write_complete_job_result(job_dir: Path, trial_names: list[str]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(trial_names),
                "stats": {
                    "n_completed_trials": len(trial_names),
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
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


def _default_reward(entry_id: str, score: float) -> dict[str, object]:
    return {
        "entry_id": entry_id,
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, score),
        "overall": score,
        "details": {
            "goal_accuracy": {
                "score": score,
                "reason": "the requested result was not produced" if score < 0.8 else "goal completed",
            }
        },
    }


def _write_reward(
    job_dir: Path,
    trial_name: str,
    reward: dict[str, object],
    *,
    sidecar: dict[str, object] | None = None,
    custom: dict[str, object] | None = None,
) -> None:
    verifier = job_dir / trial_name / "verifier"
    verifier.mkdir(parents=True, exist_ok=True)
    (verifier / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    if sidecar is not None:
        (verifier / "skill_evaluator_reward.json").write_text(json.dumps(sidecar), encoding="utf-8")
    if custom is not None:
        (verifier / "custom_reward.json").write_text(json.dumps(custom), encoding="utf-8")


def _failed_judge_artifacts(entry_id: str, *, reason: str = "judge request returned HTTP 401") -> tuple[dict, dict]:
    numeric = {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "overall": 0.0,
    }
    rich = {
        "entry_id": entry_id,
        "metric_set": DEFAULT_METRIC_SET,
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": None,
        "goal_accuracy": None,
        "behavior_check": None,
        "evaluation_status": "failed",
        "evaluation_errors": {
            "accuracy": reason,
            "goal_accuracy": "judge response was malformed",
            "behavior_check": "judge timed out",
        },
    }
    return numeric, rich


def _write_authoritative_multistep_result(
    job_dir: Path,
    trial_name: str,
    *,
    aggregate: dict[str, object],
    step_rewards: list[dict[str, object]],
) -> Path:
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "verifier_result": {"rewards": aggregate},
                "step_results": [
                    {"step_name": f"step-{index}", "verifier_result": {"rewards": rewards}}
                    for index, rewards in enumerate(step_rewards, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])
    return trial_dir


def _collect(
    tmp_path: Path,
    *,
    skip_baseline: bool,
    case_ids: list[str],
) -> dict[str, object]:
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=skip_baseline,
        expected_cases=len(case_ids),
        expected_case_ids=case_ids,
        expected_trials=len(case_ids),
    )


def test_failed_judge_sidecar_is_merged_but_never_scored_and_reason_is_safe(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    redaction_marker = "synthetic-redaction-marker-123456"
    numeric, rich = _failed_judge_artifacts(
        "case-001",
        reason=f"Authorization: Bearer {redaction_marker}; \x00\x1b" + ("upstream unavailable " * 100),
    )
    _write_reward(job_dir, trial_name, numeric, sidecar=rich)
    _write_complete_job_result(job_dir, [trial_name])

    extracted = _extract_rewards(job_dir)
    assert extracted[0]["evaluation_status"] == "failed"
    assert extracted[0]["evaluation_errors"]["accuracy"].startswith("Authorization:")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 0
    assert agent["with_skill"] == {}
    assert agent["custom_with_skill"] == {}
    [failure] = agent["trial_failures"]["with_skill"]
    assert failure["trial"] == trial_name
    assert "Required judge evaluation failed" in failure["reason"]
    assert "accuracy" in failure["reason"]
    assert redaction_marker not in failure["reason"]
    assert "<redacted>" in failure["reason"]
    assert len(failure["reason"]) <= 2048
    persisted_files = [path for path in (tmp_path / "results").rglob("*") if path.is_file()]
    assert persisted_files
    assert all(redaction_marker not in path.read_text(encoding="utf-8") for path in persisted_files)
    persisted_reward = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "trials" / trial_name / "reward.json").read_text(
            encoding="utf-8"
        )
    )
    saved_reason = persisted_reward["evaluation_errors"]["accuracy"]
    assert "<redacted>" in saved_reason
    assert len(saved_reason) <= 512
    assert not any(ord(character) < 32 or ord(character) == 127 for character in saved_reason)


@pytest.mark.parametrize("verifier_relative", [Path("verifier"), Path("steps/finish/verifier")])
def test_failed_harbor_trial_preserves_redacted_judge_diagnostics(
    tmp_path: Path,
    verifier_relative: Path,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    verifier_dir = trial_dir / verifier_relative
    redaction_marker = "".join(("AKIA", "IOSFODNN7", "EXAMPLE"))  # noqa: FLY002 - keep scanners quiet
    numeric, rich = _failed_judge_artifacts(
        "case-001",
        reason=f"judge returned HTTP 401 with {redaction_marker}",
    )
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "reward.json").write_text(json.dumps(numeric), encoding="utf-8")
    (verifier_dir / "skill_evaluator_reward.json").write_text(json.dumps(rich), encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "exception_info": {
                    "exception_type": "VerifierError",
                    "exception_message": "verifier exited nonzero",
                },
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "exception.txt").write_text("VerifierError: verifier exited nonzero\n", encoding="utf-8")
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    [failure] = result["agents"]["opencode"]["trial_failures"]["with_skill"]
    assert "HTTP 401" in failure["reason"]
    assert redaction_marker not in failure["reason"]
    assert "<redacted>" in failure["reason"]
    persisted_reward_path = tmp_path / "results" / "opencode" / "with-skill" / "trials" / trial_name / "reward.json"
    persisted_reward = json.loads(persisted_reward_path.read_text(encoding="utf-8"))
    assert persisted_reward["evaluation_status"] == "failed"
    assert overall_score(persisted_reward) is None
    persisted_text = json.dumps(persisted_reward)
    assert "HTTP 401" in persisted_text
    assert redaction_marker not in persisted_text
    assert "<redacted>" in persisted_text


def test_failed_judge_without_sidecar_still_fails_closed_with_generic_reason(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    numeric, _rich = _failed_judge_artifacts("case-001")
    _write_reward(job_dir, trial_name, numeric)
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 0
    assert agent["trial_failures"]["with_skill"] == [
        {
            "trial": trial_name,
            "reason": "Reward metrics are incomplete or non-finite; trial was not scored",
        }
    ]


def test_failed_judge_does_not_depend_on_expected_coverage_options(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    numeric, rich = _failed_judge_artifacts("case-001")
    _write_reward(job_dir, trial_name, numeric, sidecar=rich)
    _write_complete_job_result(job_dir, [trial_name])

    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
    )

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "Required judge evaluation failed" in " ".join(result["execution_errors"])


def test_incomplete_reward_does_not_depend_on_expected_coverage_options(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    numeric, _rich = _failed_judge_artifacts("case-001")
    secret = "sk-incompletejudgecredential"
    numeric["error"] = f"provider echoed {secret} " + ("unavailable " * 1000)
    _write_reward(job_dir, trial_name, numeric)
    _write_complete_job_result(job_dir, [trial_name])

    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
    )

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "Reward metrics are incomplete or non-finite" in " ".join(result["execution_errors"])
    persisted = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "trials" / trial_name / "reward.json").read_text(
            encoding="utf-8"
        )
    )
    assert secret not in persisted["error"]
    assert "<redacted>" in persisted["error"]
    assert len(persisted["error"]) <= 8192


def test_authoritative_multistep_reward_honors_failed_step_sidecar(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    numeric, rich = _failed_judge_artifacts("case-001")
    # A complete aggregate must not hide a failed judge from one constituent step.
    aggregate = _default_reward("case-001", 1.0)
    (trial_dir / "result.json").parent.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "verifier_result": {"rewards": aggregate},
                "step_results": [
                    {"step_name": "prepare", "verifier_result": {"rewards": aggregate}},
                    {"step_name": "finish", "verifier_result": {"rewards": numeric}},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_reward(job_dir, trial_name + "/steps/finish", numeric, sidecar=rich)
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "Required judge evaluation failed" in agent["trial_failures"]["with_skill"][0]["reason"]


@pytest.mark.parametrize(
    ("invalid_step", "case_name"),
    [
        (
            {
                "security": 1.0,
                "skill_execution": 1.0,
                "skill_efficiency": 1.0,
                "overall": 0.0,
            },
            "incomplete",
        ),
        ({**_default_reward("case-001", 1.0), "accuracy": float("nan")}, "non-finite"),
        ({**dict.fromkeys(DEFAULT_METRICS, float("nan")), "overall": 0.0}, "all-non-finite-unversioned"),
        ({**_default_reward("case-001", 1.0), "evaluation_status": "failed"}, "failed-status"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_authoritative_multistep_reward_rejects_invalid_default_constituent_without_sidecar(
    tmp_path: Path,
    invalid_step: dict[str, object],
    case_name: str,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    _write_authoritative_multistep_result(
        job_dir,
        trial_name,
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0), invalid_step],
    )

    [extracted] = _extract_rewards(job_dir)
    assert extracted["evaluation_status"] == "failed", case_name

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 0
    assert agent["with_skill"] == {}
    [failure] = agent["trial_failures"]["with_skill"]
    assert "constituent" in failure["reason"].casefold()
    assert len(failure["reason"]) <= 2048


@pytest.mark.parametrize("status_location", ["result", "verifier", "reward"])
def test_authoritative_root_failed_status_is_never_scored(tmp_path: Path, status_location: str) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    result_path = trial_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if status_location == "result":
        payload["evaluation_status"] = "failed"
    elif status_location == "verifier":
        payload["verifier_result"]["evaluation_status"] = "failed"
    else:
        payload["verifier_result"]["rewards"]["evaluation_status"] = "failed"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "authoritative" in " ".join(result["execution_errors"]).casefold()


@pytest.mark.parametrize(
    "verifier_result",
    [
        {"evaluation_status": "failed"},
        {"evaluation_status": "failed", "rewards": {}},
        {"rewards": {}},
    ],
    ids=("missing-failed-reward", "empty-failed-reward", "empty-default-reward"),
)
def test_authoritative_default_aggregate_rejects_missing_or_empty_step_verifier_reward(
    tmp_path: Path,
    verifier_result: dict[str, object],
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    payload = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    payload["step_results"].append({"step_name": "hidden-failure", "verifier_result": verifier_result})
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "constituent" in " ".join(result["execution_errors"]).casefold()


def test_authoritative_default_aggregate_allows_agent_only_step_without_verifier_result(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 0.8),
        step_rewards=[_default_reward("case-001", 0.8)],
    )
    payload = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    payload["step_results"].append({"step_name": "agent-only", "verifier_result": None})
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "succeeded"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 1


def test_authoritative_default_aggregate_rejects_real_harbor_step_exception_without_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    payload = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    payload["step_results"].append(
        {
            "step_name": "judge-crashed",
            "agent_result": {},
            "verifier_result": None,
            "exception_info": {
                "exception_type": "RuntimeError",
                "exception_message": "judge unavailable",
            },
        }
    )
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "constituent" in " ".join(result["execution_errors"]).casefold()


def test_authoritative_default_aggregate_rejects_present_empty_step_exception(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    payload = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    payload["step_results"].append(
        {"step_name": "invalid-exception", "agent_result": {}, "verifier_result": None, "exception_info": {}}
    )
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0


@pytest.mark.parametrize("invalid_step", [None, "malformed-step"], ids=("null", "string"))
def test_authoritative_default_aggregate_rejects_malformed_step_entry(
    tmp_path: Path,
    invalid_step: object,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    payload = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    payload["step_results"].append(invalid_step)
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "constituent" in " ".join(result["execution_errors"]).casefold()


@pytest.mark.parametrize("invalid_steps", [{}, "malformed-steps"], ids=("mapping", "string"))
def test_authoritative_default_aggregate_rejects_malformed_step_results_container(
    tmp_path: Path,
    invalid_steps: object,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    payload = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    payload["step_results"] = invalid_steps
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "authoritative" in " ".join(result["execution_errors"]).casefold()


def test_v2_root_aggregate_does_not_reclassify_constituent_missing_security_as_legacy(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    undeclared_five_metric_step = dict.fromkeys(LEGACY_METRICS, 1.0)
    _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0), undeclared_five_metric_step],
    )

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    assert "constituent" in " ".join(result["execution_errors"]).casefold()


@pytest.mark.parametrize(
    ("metric_set", "metrics"),
    [(DEFAULT_METRIC_SET, DEFAULT_METRICS), (LEGACY_METRIC_SET, LEGACY_METRICS)],
    ids=("default-v2", "legacy-v1"),
)
def test_nested_standard_step_fallback_preserves_metrics_instead_of_trusting_overall(
    tmp_path: Path,
    metric_set: str,
    metrics: tuple[str, ...],
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    nested_metrics = {metric: {"score": 0.0} for metric in metrics}
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "step_results": [
                    {
                        "step_name": "nested-default",
                        "verifier_result": {
                            "rewards": {
                                "metric_set": metric_set,
                                "metrics": nested_metrics,
                                "overall": 1.0,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["with_skill"] == dict.fromkeys(metrics, 0.0)
    assert agent["custom_with_skill"] == {}
    assert agent["pass_at_k"]["with_skill"]["rate"] == 0.0
    assert agent["pass_at_k"]["with_skill"]["cases"]["case-001"]["best_score"] == 0.0


def test_custom_only_step_fallback_remains_scoreable(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "step_results": [
                    {
                        "step_name": "custom",
                        "verifier_result": {"rewards": {"overall": 0.75, "domain_quality": 0.9}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["with_skill"] == {}
    assert agent["custom_with_skill"] == {"domain_quality": 0.9}
    assert agent["pass_at_k"]["with_skill"]["rate"] == 1.0


def test_custom_only_authoritative_reward_ignores_standard_judge_sidecar_scan_bound(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate={"overall": 0.75, "domain_quality": 0.8},
        step_rewards=[{"overall": 0.75, "domain_quality": 0.8}],
    )
    steps_dir = trial_dir / "steps"
    for index in range(collector_module._MAX_FAILED_JUDGE_SIDECARS):
        (steps_dir / f"custom-{index:03d}" / "verifier").mkdir(parents=True)

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 1
    assert agent["custom_with_skill"] == {"domain_quality": 0.8}


def test_nested_custom_only_step_fallback_preserves_custom_metrics(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "step_results": [
                    {
                        "step_name": "custom",
                        "verifier_result": {
                            "rewards": {
                                "metric_set": CUSTOM_ONLY_METRIC_SET,
                                "metrics": {"domain_quality": {"score": 0.9}},
                                "overall": 0.75,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["with_skill"] == {}
    assert agent["custom_with_skill"] == {"domain_quality": 0.9}
    assert agent["pass_at_k"]["with_skill"]["rate"] == 1.0


def test_step_fallback_aggregates_each_logical_trial_before_cross_trial_average(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_scores = {
        "case-a__attempt": ("case-a", [0.4, 0.4, 1.0]),
        "case-b__attempt": ("case-b", [1.0]),
    }
    for trial_name, (case_id, scores) in trial_scores.items():
        trial_dir = job_dir / trial_name
        trial_dir.mkdir(parents=True)
        step_results: list[dict[str, object]] = []
        for index, score in enumerate(scores, start=1):
            step_name = f"step-{index}"
            reward = _default_reward(case_id, score)
            reward["details"] = {"accuracy": {"reason": "accuracy missed" if score < 0.8 else "accurate result"}}
            verifier_dir = trial_dir / "steps" / step_name / "verifier"
            verifier_dir.mkdir(parents=True)
            (verifier_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
            step_results.append({"step_name": step_name, "verifier_result": {"rewards": reward}})
        (trial_dir / "result.json").write_text(
            json.dumps({"trial_name": trial_name, "task_name": case_id, "step_results": step_results}),
            encoding="utf-8",
        )
    _write_complete_job_result(job_dir, list(trial_scores))

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-a", "case-b"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 2
    # The public count remains the number of persisted reward rows for backward
    # compatibility; scoring and pass@k use one aggregate per logical trial.
    assert agent["num_trials_with"] == 4
    assert agent["with_skill"] == dict.fromkeys(DEFAULT_METRICS, 0.8)
    summary = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["num_trials"] == 4
    assert len(list((tmp_path / "results" / "opencode" / "with-skill" / "trials").glob("*/reward.json"))) == 4

    # Findings use bounded raw rows for evidence, but their score must remain
    # the complete collector summary even when an entire logical trial is past
    # the report-loader limit.
    monkeypatch.setattr(report_data, "_MAX_TRIALS_PER_CONDITION", 3)
    monkeypatch.setattr(report, "_generate_suggestions_structured", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(report, "_passing_skill_suggestions", lambda *_args, **_kwargs: [])
    assert report.display_findings_report(result, "demo", ["opencode"], tmp_path / "results")
    findings_payload = json.loads((tmp_path / "results" / "opencode" / "findings.json").read_text(encoding="utf-8"))
    [accuracy_finding] = [finding for finding in findings_payload["findings"] if finding["metric"] == "accuracy"]
    assert accuracy_finding["score"] == agent["with_skill"]["accuracy"]
    assert accuracy_finding["severity"] == "ok"


def test_saved_fallback_steps_keep_logical_trial_weight_in_findings_and_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_trial_scores = {
        "case-a__attempt": ("case-a", [0.4, 0.4, 1.0]),
        "case-b__attempt": ("case-b", [1.0]),
    }
    without_trial_scores = {
        "case-a__attempt": ("case-a", [0.6]),
        "case-b__attempt": ("case-b", [0.8, 0.8, 0.8]),
    }

    def write_job(variant: str, trial_scores: dict[str, tuple[str, list[float]]]) -> None:
        job_dir = tmp_path / "jobs" / f"demo-opencode-{variant}"
        for trial_name, (case_id, scores) in trial_scores.items():
            trial_dir = job_dir / trial_name
            trial_dir.mkdir(parents=True)
            step_results: list[dict[str, object]] = []
            for index, score in enumerate(scores, start=1):
                step_name = f"step-{index}"
                reward = {
                    "entry_id": case_id,
                    "metric_set": CUSTOM_ONLY_METRIC_SET,
                    "metrics": {"domain_quality": {"score": score}},
                    "overall": score,
                }
                if case_id == "case-a":
                    reward["custom_details"] = {
                        "domain_quality": {
                            "reason": "needs improvement" if score < 0.8 else "quality target met",
                        }
                    }
                verifier_dir = trial_dir / "steps" / step_name / "verifier"
                verifier_dir.mkdir(parents=True)
                (verifier_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
                step_results.append({"step_name": step_name, "verifier_result": {"rewards": reward}})
            (trial_dir / "result.json").write_text(
                json.dumps({"trial_name": trial_name, "task_name": case_id, "step_results": step_results}),
                encoding="utf-8",
            )
        _write_complete_job_result(job_dir, list(trial_scores))

    write_job("with", with_trial_scores)
    write_job("without", without_trial_scores)

    result = _collect(tmp_path, skip_baseline=False, case_ids=["case-a", "case-b"])

    agent = result["agents"]["opencode"]
    assert agent["custom_with_skill"] == {"domain_quality": 0.8}
    assert agent["custom_without_skill"] == {"domain_quality": 0.7}
    assert agent["custom_lift"]["overall"] == {
        "with_skill": 0.8,
        "without_skill": 0.7,
        "delta": 0.1,
        "direction": "up",
    }
    assert agent["pass_at_k"]["with_skill"]["cases"]["case-a"]["best_score"] == 0.6
    assert agent["pass_at_k"]["with_skill"]["cases"]["case-b"]["best_score"] == 1.0
    # Keep all four step artifacts for diagnostics; only their score weighting
    # is logical-trial based.
    reward_paths = list((tmp_path / "results" / "opencode" / "with-skill" / "trials").glob("*/reward.json"))
    assert len(reward_paths) == 4
    persisted_rewards = [json.loads(path.read_text(encoding="utf-8")) for path in reward_paths]
    assert len({reward["trial_id"] for reward in persisted_rewards}) == 2

    monkeypatch.setattr(report_data, "_MAX_TRIALS_PER_CONDITION", 3)
    monkeypatch.setattr(report, "_generate_suggestions_structured", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(report, "_passing_skill_suggestions", lambda *_args, **_kwargs: [])
    assert report.display_findings_report(result, "demo", ["opencode"], tmp_path / "results")
    findings_payload = json.loads((tmp_path / "results" / "opencode" / "findings.json").read_text(encoding="utf-8"))
    [domain_finding] = [finding for finding in findings_payload["findings"] if finding["metric"] == "domain_quality"]
    assert domain_finding["score"] == agent["custom_with_skill"]["domain_quality"]
    assert domain_finding["severity"] == "ok"

    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    report_path = render_agent_eval_html_report(
        skill_dir,
        tmp_path / "results",
        use_llm_judge=False,
    )
    output = report_path.read_text(encoding="utf-8")
    payload_match = re.search(
        r'<script type="application/json" id="tier3-full">(.*?)</script>',
        output,
        re.DOTALL,
    )
    assert payload_match is not None
    report_payload = json.loads(payload_match.group(1))
    assert report_payload["overall_score"] == 0.8
    assert report_payload["overall_lift"] == 0.1
    # Custom-only metrics retain their score and lift, but they do not satisfy
    # the canonical five-dimension publication gate introduced on main.
    assert report_payload["verdict"] == "neutral"
    assert report_payload["agents"]["opencode"]["with_skill"] == 0.8
    assert report_payload["agents"]["opencode"]["baseline"] == 0.7
    assert report_payload["agents"]["opencode"]["lift"] == 0.1
    assert report_payload["agents"]["opencode"]["num_trials"] == 4
    assert report_payload["agents"]["opencode"]["num_trials_baseline"] == 4

    # A legacy summary has no canonical overall. When its raw rows are
    # truncated, the report must show unavailable rather than publish the
    # partial first logical trial as a numeric headline.
    for variant in ("with-skill", "without-skill"):
        summary_path = tmp_path / "results" / "opencode" / variant / "summary.json"
        legacy_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        legacy_summary.pop("overall_score")
        summary_path.write_text(json.dumps(legacy_summary), encoding="utf-8")
    legacy_report = render_agent_eval_html_report(
        skill_dir,
        tmp_path / "results",
        output_path=tmp_path / "results" / "legacy-report.html",
        use_llm_judge=False,
    )
    legacy_output = legacy_report.read_text(encoding="utf-8")
    legacy_payload_match = re.search(
        r'<script type="application/json" id="tier3-full">(.*?)</script>',
        legacy_output,
        re.DOTALL,
    )
    assert legacy_payload_match is not None
    legacy_payload = json.loads(legacy_payload_match.group(1))
    assert legacy_payload["overall_score"] is None
    assert legacy_payload["overall_lift"] is None
    assert legacy_payload["agents"]["opencode"]["with_skill"] is None
    assert legacy_payload["agents"]["opencode"]["baseline"] is None

    # The same fail-closed rule applies below the row cap when a whole saved
    # trial directory is missing: summary count and loaded row count disagree.
    monkeypatch.setattr(report_data, "_MAX_TRIALS_PER_CONDITION", 512)
    reward_paths[0].parent.rename(tmp_path / "removed-saved-trial")
    missing_trial_report = render_agent_eval_html_report(
        skill_dir,
        tmp_path / "results",
        output_path=tmp_path / "results" / "legacy-missing-trial-report.html",
        use_llm_judge=False,
    )
    missing_trial_output = missing_trial_report.read_text(encoding="utf-8")
    missing_trial_payload_match = re.search(
        r'<script type="application/json" id="tier3-full">(.*?)</script>',
        missing_trial_output,
        re.DOTALL,
    )
    assert missing_trial_payload_match is not None
    missing_trial_payload = json.loads(missing_trial_payload_match.group(1))
    assert missing_trial_payload["overall_score"] is None
    assert missing_trial_payload["agents"]["opencode"]["with_skill"] is None


@pytest.mark.parametrize(
    ("aggregate", "step_rewards", "expected_custom_scores"),
    [
        (
            {"overall": 0.75, "domain_quality": 0.8},
            [{"overall": 0.5, "domain_quality": 0.6}, {"overall": 1.0, "domain_quality": 1.0}],
            {"domain_quality": 0.8},
        ),
        (
            _default_reward("case-001", 0.8),
            [_default_reward("case-001", 0.7), {"overall": 0.9, "domain_quality": 1.0}],
            {},
        ),
        (
            _default_reward("case-001", 0.8),
            [_default_reward("case-001", 0.7), _default_reward("case-001", 0.9)],
            {},
        ),
    ],
    ids=("custom-only", "mixed-default-and-custom", "complete-default"),
)
def test_authoritative_multistep_reward_preserves_complete_custom_topologies(
    tmp_path: Path,
    aggregate: dict[str, object],
    step_rewards: list[dict[str, object]],
    expected_custom_scores: dict[str, float],
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=aggregate,
        step_rewards=step_rewards,
    )

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 1
    assert agent["custom_with_skill"] == expected_custom_scores
    assert agent["pass_at_k"]["with_skill"]["rate"] == 1.0


def _create_step_entries(trial_dir: Path, names: list[str], *, directories: bool) -> None:
    steps_dir = trial_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    # Intentionally avoid lexical creation order so correctness cannot depend on
    # whichever directory entry happens to be returned first by the filesystem.
    shuffled_names = names[::2] + list(reversed(names[1::2]))
    for name in shuffled_names:
        path = steps_dir / name
        if directories:
            (path / "verifier").mkdir(parents=True)
        else:
            path.write_text("not a step directory", encoding="utf-8")


@pytest.mark.parametrize(
    ("total_candidate_count", "expected_status"),
    [
        (collector_module._MAX_FAILED_JUDGE_SIDECARS, "succeeded"),
        (collector_module._MAX_FAILED_JUDGE_SIDECARS + 1, "failed"),
    ],
    ids=("exact-candidate-limit", "beyond-candidate-limit"),
)
def test_public_collection_fails_closed_when_sidecar_candidate_limit_is_exceeded(
    tmp_path: Path,
    total_candidate_count: int,
    expected_status: str,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    # Only sidecars that actually exist consume the sidecar bound. Empty
    # verifier directories remain compatible with large native step graphs.
    step_count = total_candidate_count
    _create_step_entries(
        trial_dir,
        [f"candidate-{index:04d}" for index in range(step_count)],
        directories=True,
    )
    for index in range(step_count):
        sidecar = trial_dir / "steps" / f"candidate-{index:04d}" / "verifier" / "skill_evaluator_reward.json"
        sidecar.write_text('{"evaluation_status":"succeeded"}', encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == expected_status
    condition = result["agents"]["opencode"]["conditions"]["with_skill"]
    if expected_status == "succeeded":
        assert condition["scored_attempts"] == 1
    else:
        assert condition["scored_attempts"] == 0
        errors = " ".join(result["execution_errors"])
        assert "sidecar" in errors.casefold()
        assert "limit" in errors.casefold()
        assert len(errors) <= 2048


@pytest.mark.parametrize(
    ("entry_count", "expected_status"),
    [
        (collector_module._MAX_FAILED_JUDGE_STEP_PATHS_SCANNED, "succeeded"),
        (collector_module._MAX_FAILED_JUDGE_STEP_PATHS_SCANNED + 1, "failed"),
    ],
    ids=("exact-scan-limit", "beyond-scan-limit"),
)
def test_public_collection_fails_closed_when_step_entry_scan_limit_is_exceeded(
    tmp_path: Path,
    entry_count: int,
    expected_status: str,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    _create_step_entries(
        trial_dir,
        [f"entry-{index:04d}" for index in range(entry_count)],
        directories=False,
    )

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == expected_status
    condition = result["agents"]["opencode"]["conditions"]["with_skill"]
    if expected_status == "succeeded":
        assert condition["scored_attempts"] == 1
    else:
        assert condition["scored_attempts"] == 0
        errors = " ".join(result["execution_errors"])
        assert "sidecar" in errors.casefold()
        assert "limit" in errors.casefold()
        assert len(errors) <= 2048


def test_public_collection_fails_closed_when_step_sidecars_cannot_be_scanned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    steps_dir = trial_dir / "steps"
    steps_dir.mkdir()
    original_scandir = os.scandir

    def deny_step_scan(path: os.PathLike[str] | str):
        if Path(path) == steps_dir:
            raise PermissionError("synthetic sensitive filesystem detail")
        return original_scandir(path)

    monkeypatch.setattr(collector_module.os, "scandir", deny_step_scan)

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    errors = " ".join(result["execution_errors"])
    assert "could not inspect step artifacts" in errors.casefold()
    assert "sensitive filesystem detail" not in errors
    assert len(errors) <= 2048


@pytest.mark.parametrize("sidecar_kind", ["malformed", "oversized", "symlink"])
def test_public_collection_fails_closed_when_present_sidecar_cannot_be_read(
    tmp_path: Path,
    sidecar_kind: str,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    sidecar = trial_dir / "verifier" / "skill_evaluator_reward.json"
    sidecar.parent.mkdir()
    external_marker = "external-unreadable-sidecar-marker"
    if sidecar_kind == "malformed":
        sidecar.write_text('{"evaluation_status":', encoding="utf-8")
    elif sidecar_kind == "oversized":
        sidecar.write_bytes(b" " * (collector_module.DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES + 1))
    else:
        external = tmp_path / "external-sidecar.json"
        _numeric, rich = _failed_judge_artifacts("case-001", reason=external_marker)
        external.write_text(json.dumps(rich), encoding="utf-8")
        sidecar.symlink_to(external)

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    errors = " ".join(result["execution_errors"])
    assert "sidecar" in errors.casefold()
    assert "could not be read" in errors.casefold()
    assert external_marker not in errors
    assert len(errors) <= 2048


def test_sidecar_scan_limit_diagnostic_precedes_existing_reward_errors(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    reward = {
        **_default_reward("case-001", 1.0),
        "evaluation_status": "failed",
        "evaluation_errors": {f"existing-{index}": f"existing error {index}" for index in range(len(DEFAULT_METRICS))},
    }
    _write_reward(job_dir, trial_name, reward)
    _write_complete_job_result(job_dir, [trial_name])
    trial_dir = job_dir / trial_name
    _create_step_entries(
        trial_dir,
        [f"candidate-{index:04d}" for index in range(collector_module._MAX_FAILED_JUDGE_SIDECARS + 1)],
        directories=True,
    )
    for index in range(collector_module._MAX_FAILED_JUDGE_SIDECARS + 1):
        sidecar = trial_dir / "steps" / f"candidate-{index:04d}" / "verifier" / "skill_evaluator_reward.json"
        sidecar.write_text('{"evaluation_status":"succeeded"}', encoding="utf-8")

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    errors = " ".join(result["execution_errors"])
    assert "sidecar" in errors.casefold()
    assert "limit" in errors.casefold()
    assert len(errors) <= 2048


@pytest.mark.parametrize("linked_component", ["steps", "step-verifier"])
def test_public_collection_fails_closed_on_linked_step_sidecar_boundaries(
    tmp_path: Path,
    linked_component: str,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_dir = _write_authoritative_multistep_result(
        job_dir,
        "case-001__attempt",
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    external_marker = "external-linked-sidecar-must-not-be-read"
    trajectory_marker = "external-linked-trajectory-must-not-be-read"
    _numeric, rich = _failed_judge_artifacts("case-001", reason=external_marker)
    external = tmp_path / "external-step-boundary"
    if linked_component == "steps":
        external_verifier = external / "finish" / "verifier"
        external_verifier.mkdir(parents=True)
        (external_verifier / "skill_evaluator_reward.json").write_text(json.dumps(rich), encoding="utf-8")
        external_agent = external / "finish" / "agent"
        external_agent.mkdir()
        (external_agent / "trajectory.json").write_text(
            json.dumps({"steps": [{"message": trajectory_marker}]}),
            encoding="utf-8",
        )
        (trial_dir / "steps").symlink_to(external, target_is_directory=True)
    else:
        external.mkdir()
        (external / "skill_evaluator_reward.json").write_text(json.dumps(rich), encoding="utf-8")
        verifier_dir = trial_dir / "steps" / "finish" / "verifier"
        verifier_dir.parent.mkdir(parents=True)
        verifier_dir.symlink_to(external, target_is_directory=True)

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0
    errors = " ".join(result["execution_errors"])
    assert "symlink" in errors.casefold()
    assert external_marker not in errors
    assert trajectory_marker not in json.dumps(result)
    persisted_trials = tmp_path / "results" / "opencode" / "with-skill" / "trials"
    assert all(trajectory_marker not in path.read_text(encoding="utf-8") for path in persisted_trials.rglob("*.json"))
    assert len(errors) <= 2048


@pytest.mark.parametrize("verifier_relative", [Path("verifier"), Path("steps/finish/verifier")])
def test_authoritative_reward_fails_closed_on_symlinked_external_judge_sidecar(
    tmp_path: Path,
    verifier_relative: Path,
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    aggregate = _default_reward("case-001", 1.0)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "verifier_result": {"rewards": aggregate},
                "step_results": [],
            }
        ),
        encoding="utf-8",
    )
    external_verifier = tmp_path / "outside-verifier"
    external_verifier.mkdir()
    _numeric, rich = _failed_judge_artifacts("case-001", reason="external sidecar marker")
    (external_verifier / "skill_evaluator_reward.json").write_text(json.dumps(rich), encoding="utf-8")
    verifier_dir = trial_dir / verifier_relative
    verifier_dir.parent.mkdir(parents=True, exist_ok=True)
    verifier_dir.symlink_to(external_verifier, target_is_directory=True)

    [extracted] = _extract_rewards(job_dir)

    assert extracted["evaluation_status"] == "failed"
    assert "external sidecar marker" not in json.dumps(extracted)


def test_discovered_step_sidecar_cannot_be_swapped_to_external_ancestor(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    verifier_dir = trial_dir / "steps" / "step-a" / "verifier"
    verifier_dir.mkdir(parents=True)
    sidecar_path = verifier_dir / "skill_evaluator_reward.json"
    sidecar_path.write_text('{"evaluation_status":"succeeded"}', encoding="utf-8")

    candidates, failure = collector_module._failed_judge_sidecar_paths(trial_dir)
    assert failure == ""
    [(step_name, discovered_path, expected)] = candidates
    assert step_name == "step-a"

    original_step = trial_dir / "steps" / "step-a"
    original_step.rename(trial_dir / "steps" / "step-a-original")
    external_step = tmp_path / "external-step"
    external_verifier = external_step / "verifier"
    external_verifier.mkdir(parents=True)
    external_marker = "EXTERNAL-RACE-MARKER"
    (external_verifier / "skill_evaluator_reward.json").write_text(
        json.dumps({"evaluation_status": "failed", "evaluation_errors": {"accuracy": external_marker}}),
        encoding="utf-8",
    )
    original_step.symlink_to(external_step, target_is_directory=True)

    sidecar, read_failure = collector_module._read_failed_judge_sidecar(
        discovered_path,
        trial_dir=trial_dir,
        expected=expected,
    )

    assert sidecar is None
    assert "could not be read" in read_failure
    assert external_marker not in read_failure


def test_public_collection_rejects_symlinked_trial_root_without_reading_external_artifacts(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    job_dir.mkdir(parents=True)
    trial_name = "case-001__attempt"
    external_trial = tmp_path / "outside-trial"
    _write_authoritative_multistep_result(
        tmp_path / "outside-job",
        external_trial.name,
        aggregate=_default_reward("case-001", 1.0),
        step_rewards=[_default_reward("case-001", 1.0)],
    )
    # Move the generated result into the exact external target while keeping
    # the public job path as a directory link.
    generated = tmp_path / "outside-job" / external_trial.name
    generated.rename(external_trial)
    marker = "EXTERNAL-TRIAL-DIAGNOSTIC-MARKER"
    (external_trial / "trial.log").write_text(marker, encoding="utf-8")
    (job_dir / trial_name).symlink_to(external_trial, target_is_directory=True)
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    condition = result["agents"]["opencode"]["conditions"]["with_skill"]
    assert result["execution_status"] == "failed"
    assert condition["scored_attempts"] == 0
    assert result["agents"]["opencode"]["with_skill"] == {}
    assert "symlink" in " ".join(result["execution_errors"]).casefold()
    copied = tmp_path / "results" / "opencode" / "with-skill" / "trials" / trial_name / "trial.log"
    assert not copied.exists()
    assert marker not in json.dumps(result)


def test_single_step_result_fallback_honors_failed_top_level_sidecar(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    trial_dir = job_dir / trial_name
    _numeric, rich = _failed_judge_artifacts("case-001")
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-001",
                "verifier_result": {"rewards": _default_reward("case-001", 1.0)},
            }
        ),
        encoding="utf-8",
    )
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "skill_evaluator_reward.json").write_text(json.dumps(rich), encoding="utf-8")
    _write_complete_job_result(job_dir, [trial_name])

    [extracted] = _extract_rewards(job_dir)
    assert extracted["evaluation_status"] == "failed"

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])
    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["with_skill"] == {}


def test_failed_sidecar_overrides_conflicting_reward_status_and_errors(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    secret = "sk-authoritativesidecar123"
    reward = {
        **_default_reward("case-001", 1.0),
        "evaluation_status": "succeeded",
        "evaluation_errors": {"accuracy": "stale reward error"},
    }
    _numeric, rich = _failed_judge_artifacts("case-001", reason=f"provider echoed {secret}")
    rich["evaluation_status"] = "error"
    _write_reward(job_dir, trial_name, reward, sidecar=rich)
    _write_complete_job_result(job_dir, [trial_name])

    [extracted] = _extract_rewards(job_dir)
    assert extracted["evaluation_status"] == "failed"
    assert extracted["evaluation_errors"]["accuracy"] == "provider echoed sk-<redacted>"

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])
    assert result["execution_status"] == "failed"
    assert secret not in " ".join(result["execution_errors"])


def test_default_plus_custom_cannot_rescue_an_incomplete_default_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    numeric, rich = _failed_judge_artifacts("case-001")
    _write_reward(
        job_dir,
        trial_name,
        numeric,
        sidecar=rich,
        custom={"overall": 1.0, "domain_quality": 1.0, "details": {"domain_quality": {"reason": "perfect"}}},
    )
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert agent["with_skill"] == {}
    assert agent["custom_with_skill"] == {}
    assert agent["pass_at_k"]["with_skill"] == {}


def test_custom_only_overall_reward_remains_scoreable(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    _write_reward(
        job_dir,
        trial_name,
        {"entry_id": "case-001", "overall": 0.75, "domain_quality": 0.9, "token_efficiency": 0.8},
    )
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["with_skill"] == {}
    assert agent["custom_with_skill"] == {"domain_quality": 0.9, "token_efficiency": 0.8}
    assert agent["pass_at_k"]["with_skill"]["rate"] == 1.0
    assert agent["conditions"]["without_skill"]["execution_status"] == "skipped"
    persisted = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "trials" / trial_name / "reward.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["token_efficiency"] == 0.8


def test_mixed_failed_condition_suppresses_published_quality_and_paired_artifacts(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    with_job = jobs_dir / "demo-opencode-with"
    without_job = jobs_dir / "demo-opencode-without"

    _write_reward(with_job, "case-001__attempt", _default_reward("case-001", 0.9))
    numeric, rich = _failed_judge_artifacts("case-002")
    _write_reward(with_job, "case-002__attempt", numeric, sidecar=rich)
    _write_complete_job_result(with_job, ["case-001__attempt", "case-002__attempt"])

    for case_id in ("case-001", "case-002"):
        _write_reward(without_job, f"{case_id}__attempt", _default_reward(case_id, 0.2))
    _write_complete_job_result(without_job, ["case-001__attempt", "case-002__attempt"])

    result = _collect(tmp_path, skip_baseline=False, case_ids=["case-001", "case-002"])

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert agent["conditions"]["with_skill"]["execution_status"] == "failed"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 1
    assert agent["conditions"]["without_skill"]["execution_status"] == "succeeded"
    assert agent["with_skill"] == {}
    assert agent["custom_with_skill"] == {}
    assert agent["dimensions_with_skill"] == {}
    assert agent["pass_at_k"]["with_skill"] == {}
    assert agent["without_skill"] == dict.fromkeys(DEFAULT_METRICS, 0.2)
    assert agent["dimensions_without_skill"]
    assert agent["pass_at_k"]["without_skill"]["rate"] == 0.0
    assert agent["lift"] == {}
    assert agent["custom_lift"] == {}
    assert agent["pass_at_k"]["lift"] == {}
    assert agent["security_attribution"] == {}

    results_dir = tmp_path / "results" / "opencode"
    persisted = json.loads((results_dir / "with-skill" / "summary.json").read_text(encoding="utf-8"))
    assert persisted["scores"] == {}
    assert persisted["custom_scores"] == {}
    assert persisted["dimensions"] == {}
    assert persisted["pass_at_k"] == {}
    assert not (results_dir / "lift.json").exists()
    assert not (results_dir / "custom_lift.json").exists()
    assert not (results_dir / "pass_at_k_lift.json").exists()
    assert not (results_dir / "security_attribution.json").exists()
    # Both source artifacts remain available as redacted trial diagnostics.
    assert (results_dir / "with-skill" / "trials" / "case-001__attempt" / "reward.json").exists()
    failed_reward = results_dir / "with-skill" / "trials" / "case-002__attempt" / "reward.json"
    assert json.loads(failed_reward.read_text(encoding="utf-8"))["evaluation_status"] == "failed"

    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    html = render_agent_eval_html_report(skill_dir, tmp_path / "results", use_llm_judge=False).read_text(
        encoding="utf-8"
    )
    assert "Evaluation incomplete" in html
    assert "Required judge evaluation failed" in html
    assert re.findall(r'class="t3-dim-score"[^>]*>([^<]+)</span>', html) == ["N/A"] * 5
    assert "return raw === null || raw === undefined ? null : Number(raw);" in html
    payload_match = re.search(
        r'<script type="application/json" id="tier3-full">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert payload_match is not None
    report_payload = json.loads(payload_match.group(1))
    for dimension in report_payload["agents"]["opencode"]["dimensions"]:
        assert dimension["with_skill"] is None
        assert dimension["score"] is None
        assert dimension["verdict"] is None
        assert "0.00" not in str(dimension["explanation"])
        assert all("0.00" not in bullet for bullet in dimension["reasoning_bullets"])


def test_reused_results_remove_stale_generated_quality_but_preserve_unrelated_files(
    tmp_path: Path, monkeypatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    with_job = jobs_dir / "demo-opencode-with"
    without_job = jobs_dir / "demo-opencode-without"
    trial_name = "case-001__attempt"
    with_reward = {**_default_reward("case-001", 0.2), "domain_quality": 0.4}
    without_reward = {**_default_reward("case-001", 0.1), "domain_quality": 0.2}
    _write_reward(with_job, trial_name, with_reward)
    _write_reward(without_job, trial_name, without_reward)
    _write_complete_job_result(with_job, [trial_name])
    _write_complete_job_result(without_job, [trial_name])

    first = _collect(tmp_path, skip_baseline=False, case_ids=["case-001"])
    agent_dir = tmp_path / "results" / "opencode"
    monkeypatch.setattr(report, "_generate_suggestions_structured", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(report, "_passing_skill_suggestions", lambda *_args, **_kwargs: [])
    assert report.display_findings_report(first, "demo", ["opencode"], tmp_path / "results")
    stale_trial = agent_dir / "with-skill" / "trials" / "case-stale__attempt"
    stale_trial.mkdir(parents=True)
    (stale_trial / "reward.json").write_text(json.dumps(_default_reward("case-stale", 1.0)), encoding="utf-8")
    unrelated_agent_file = agent_dir / "user-note.txt"
    unrelated_condition_file = agent_dir / "with-skill" / "user-note.txt"
    unrelated_agent_file.write_text("keep", encoding="utf-8")
    unrelated_condition_file.write_text("keep", encoding="utf-8")
    for artifact in (
        "lift.json",
        "custom_lift.json",
        "pass_at_k_lift.json",
        "security_attribution.json",
        "findings.json",
    ):
        assert (agent_dir / artifact).exists()

    numeric, rich = _failed_judge_artifacts("case-001")
    _write_reward(with_job, trial_name, numeric, sidecar=rich)
    _write_complete_job_result(with_job, [trial_name])
    failed = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert failed["execution_status"] == "failed"
    for artifact in (
        "lift.json",
        "custom_lift.json",
        "pass_at_k_lift.json",
        "security_attribution.json",
        "findings.json",
    ):
        assert not (agent_dir / artifact).exists()
    assert not stale_trial.exists()
    assert not (agent_dir / "without-skill" / "trials").exists()
    assert not (agent_dir / "without-skill" / "summary.json").exists()
    assert unrelated_agent_file.read_text(encoding="utf-8") == "keep"
    assert unrelated_condition_file.read_text(encoding="utf-8") == "keep"


def test_generated_output_cleanup_refuses_symlink_escape(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    outside = tmp_path / "outside"
    results_dir.mkdir()
    outside.mkdir()
    (tmp_path / "jobs").mkdir()
    sentinel = outside / "lift.json"
    sentinel.write_text("do not delete", encoding="utf-8")
    (results_dir / "opencode").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="generated output"):
        collect_harbor_results(
            skill_name="demo",
            agents=["opencode"],
            output_dir=results_dir,
            jobs_dir=tmp_path / "jobs",
            skip_baseline=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "do not delete"


def test_reused_results_remove_omitted_agent_from_report_discovery(tmp_path: Path, monkeypatch) -> None:
    for agent, score in (("opencode", 0.2), ("claude-code", 0.1)):
        job_dir = tmp_path / "jobs" / f"demo-{agent}-with"
        _write_reward(job_dir, "case-001__attempt", _default_reward("case-001", score))
        _write_complete_job_result(job_dir, ["case-001__attempt"])

    first = collect_harbor_results(
        skill_name="demo",
        agents=["opencode", "claude-code"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )
    monkeypatch.setattr(report, "_generate_suggestions_structured", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(report, "_passing_skill_suggestions", lambda *_args, **_kwargs: [])
    assert report.display_findings_report(first, "demo", ["opencode", "claude-code"], tmp_path / "results")
    results_dir = tmp_path / "results"
    omitted_dir = results_dir / "claude-code"
    omitted_note = omitted_dir / "user-note.txt"
    omitted_note.write_text("keep", encoding="utf-8")
    unrelated_dir = results_dir / "user-content"
    unrelated_summary = unrelated_dir / "with-skill" / "summary.json"
    unrelated_summary.parent.mkdir(parents=True)
    unrelated_summary.write_text('{"notes": "not a collector summary"}', encoding="utf-8")
    unrelated_note = unrelated_dir / "notes.txt"
    unrelated_note.write_text("keep", encoding="utf-8")
    outside_agent = tmp_path / "outside-agent"
    outside_summary = outside_agent / "with-skill" / "summary.json"
    outside_summary.parent.mkdir(parents=True)
    outside_summary.write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(DEFAULT_METRICS, 1.0),
                "execution_status": "succeeded",
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    stale_agent_link = results_dir / "stale"
    stale_agent_link.symlink_to(outside_agent, target_is_directory=True)
    assert (results_dir / "comparison.json").exists()
    assert (omitted_dir / "findings.json").exists()

    second = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=results_dir,
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert second["execution_status"] == "succeeded"
    assert set(report_data.load_agent_data(results_dir)) == {"opencode"}
    assert not (results_dir / "comparison.json").exists()
    assert not (omitted_dir / "with-skill" / "summary.json").exists()
    assert not (omitted_dir / "with-skill" / "trials").exists()
    assert not (omitted_dir / "findings.json").exists()
    assert omitted_note.read_text(encoding="utf-8") == "keep"
    assert unrelated_note.read_text(encoding="utf-8") == "keep"
    assert unrelated_summary.read_text(encoding="utf-8") == '{"notes": "not a collector summary"}'
    assert stale_agent_link.is_symlink()
    assert outside_summary.exists()


@pytest.mark.parametrize("summary_kind", ["deep", "huge-integer", "malformed", "unicode-invalid"])
def test_unrelated_invalid_summary_does_not_block_collection(tmp_path: Path, summary_kind: str) -> None:
    unrelated_dir = tmp_path / "results" / "user-content"
    summary = unrelated_dir / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    if summary_kind == "deep":
        summary.write_text("[" * 10_000 + "]" * 10_000, encoding="utf-8")
    elif summary_kind == "huge-integer":
        summary.write_text('{"value": ' + "1" * 10_000 + "}", encoding="utf-8")
    elif summary_kind == "malformed":
        summary.write_text('{"scores": ', encoding="utf-8")
    else:
        summary.write_bytes(b"\xff\xfe\x00")
    note = unrelated_dir / "notes.txt"
    note.write_text("keep", encoding="utf-8")
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    _write_reward(job_dir, "case-001__attempt", _default_reward("case-001", 0.8))
    _write_complete_job_result(job_dir, ["case-001__attempt"])

    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert result["execution_status"] == "succeeded"
    assert note.read_text(encoding="utf-8") == "keep"
    assert summary.exists()


@pytest.mark.parametrize("reward_kind", ["deep", "huge-integer", "unicode-invalid"])
def test_invalid_verifier_reward_fails_collection_without_crashing(tmp_path: Path, reward_kind: str) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    verifier_dir = job_dir / trial_name / "verifier"
    verifier_dir.mkdir(parents=True)
    reward_path = verifier_dir / "reward.json"
    if reward_kind == "deep":
        reward_path.write_text("[" * 10_000 + "]" * 10_000, encoding="utf-8")
    elif reward_kind == "huge-integer":
        reward_path.write_text('{"security": ' + "1" * 10_000 + "}", encoding="utf-8")
    else:
        reward_path.write_bytes(b"\xff\xfe\x00")
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["with_skill"] == {}
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO paths are not available on this platform")
def test_fifo_reward_is_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "reward.json"
    os.mkfifo(fifo)
    script = (
        "from pathlib import Path; "
        "from skillevaluator.tier3.harbor.collector import _read_json; "
        f"assert _read_json(Path({str(fifo)!r})) is None"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("artifact", ["exception", "agent-log"])
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO paths are not available on this platform")
def test_fifo_diagnostic_text_is_rejected_without_blocking(tmp_path: Path, artifact: str) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    if artifact == "exception":
        os.mkfifo(trial_dir / "exception.txt")
        expression = f"_trial_failure_reason(Path({str(trial_dir)!r})) == ''"
        imports = "_trial_failure_reason"
    else:
        (trial_dir / "agent").mkdir()
        os.mkfifo(trial_dir / "agent" / "codex.txt")
        expression = f"_agent_log_runtime_failure_reason(Path({str(trial_dir)!r})) == ''"
        imports = "_agent_log_runtime_failure_reason"
    script = (
        f"from pathlib import Path; from skillevaluator.tier3.harbor.collector import {imports}; assert {expression}"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0, completed.stderr


def test_diagnostic_copy_rejects_final_symlink_and_records_manifest(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_out = tmp_path / "collected"
    trial_dir.mkdir()
    external_marker = "external-diagnostic-marker-must-not-be-copied"
    external = tmp_path / "external.log"
    external.write_text(external_marker, encoding="utf-8")
    (trial_dir / "trial.log").symlink_to(external)
    (trial_dir / "result.json").write_text('{"safe": true}', encoding="utf-8")

    copied = _copy_trial_artifacts(trial_dir, trial_out)

    assert copied == ["result.json"]
    assert json.loads((trial_out / "result.json").read_text(encoding="utf-8")) == {"safe": True}
    assert not (trial_out / "trial.log").exists()
    assert external.read_text(encoding="utf-8") == external_marker
    manifest = json.loads((trial_out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["copied"] == [{"name": "result.json", "size_bytes": 14}]
    assert manifest["skipped"][0]["name"] == "trial.log"
    assert "regular" in manifest["skipped"][0]["reason"]
    assert external_marker not in json.dumps(manifest)


def test_diagnostic_copy_records_dangling_final_symlink_in_manifest(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_out = tmp_path / "collected"
    trial_dir.mkdir()
    trial_out.mkdir()
    (trial_dir / "trial.log").symlink_to(tmp_path / "missing-external.log")

    copied = _copy_trial_artifacts(trial_dir, trial_out)

    assert copied == []
    manifest = json.loads((trial_out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["copied"] == []
    assert manifest["skipped"] == [{"name": "trial.log", "reason": "not_regular_file"}]


def test_diagnostic_copy_rejects_symlinked_agent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial_dir = tmp_path / "trial"
    trial_out = tmp_path / "collected"
    external_agent_dir = tmp_path / "external-agent"
    trial_dir.mkdir()
    trial_out.mkdir()
    external_agent_dir.mkdir()
    external_marker = "external-agent-marker-must-not-be-copied"
    (external_agent_dir / "codex.txt").write_text(external_marker, encoding="utf-8")
    (trial_dir / "agent").symlink_to(external_agent_dir, target_is_directory=True)
    # Windows junctions/reparse points are not reliably reported by
    # Path.is_symlink(); force that observable shape while preserving lstat.
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)

    copied = _copy_trial_artifacts(trial_dir, trial_out)

    assert copied == []
    assert not (trial_out / "codex.txt").exists()
    assert (external_agent_dir / "codex.txt").read_text(encoding="utf-8") == external_marker
    manifest = json.loads((trial_out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["copied"] == []
    assert manifest["skipped"] == [{"name": "agent", "reason": "not_regular_directory"}]
    assert external_marker not in json.dumps(manifest)


def test_diagnostic_copy_preserves_bounded_redacted_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "trial.log"
    destination = tmp_path / "out" / "trial.log"
    secret = "sk-diagnosticcopysecret123"
    source.write_text(f"Authorization: Bearer {secret}\nordinary output\n", encoding="utf-8")

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is True
    assert record == {"name": "trial.log", "size_bytes": source.stat().st_size}
    copied = destination.read_text(encoding="utf-8")
    assert "ordinary output" in copied
    assert secret not in copied
    assert "<redacted>" in copied


@pytest.mark.parametrize(
    "payload",
    (
        "[" * 2_000 + "0" + "]" * 2_000,
        '{"value":' + "9" * 10_000 + "}",
        '{"api_key":"SUPERSECRET12345", broken',
    ),
    ids=("deep", "huge-integer", "malformed-secret"),
)
def test_diagnostic_copy_rejects_invalid_json_without_crashing(tmp_path: Path, payload: str) -> None:
    source = tmp_path / "result.json"
    destination = tmp_path / "out" / "result.json"
    source.write_text(payload, encoding="utf-8")

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is False
    assert record == {"name": "result.json", "reason": "invalid_json", "size_bytes": len(payload)}
    assert not destination.exists()
    assert "SUPERSECRET12345" not in json.dumps(record)


def test_diagnostic_copy_does_not_pretty_print_amplify_nested_json(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    destination = tmp_path / "out" / "result.json"
    source.write_text("[" * 200 + "0" + "]" * 200, encoding="utf-8")

    ok, _record = _write_redacted_text_copy(source, destination)

    assert ok is True
    assert destination.stat().st_size <= source.stat().st_size


def test_diagnostic_copy_rejects_hard_linked_artifact(tmp_path: Path) -> None:
    outside = tmp_path / "outside.log"
    source = tmp_path / "trial.log"
    destination = tmp_path / "out" / "trial.log"
    outside.write_text("outside-marker", encoding="utf-8")
    os.link(outside, source)

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is False
    assert not destination.exists()
    assert record == {"name": "trial.log", "reason": "not_regular_file"}


@pytest.mark.parametrize("destination_kind", ["symlink", "hardlink"])
def test_diagnostic_copy_never_follows_or_replaces_unsafe_destination(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    source = tmp_path / "trial.log"
    destination = tmp_path / "out" / "trial.log"
    external = tmp_path / "external.log"
    source.write_text("safe-source", encoding="utf-8")
    external.write_text("outside-sentinel", encoding="utf-8")
    destination.parent.mkdir()
    if destination_kind == "symlink":
        destination.symlink_to(external)
    else:
        os.link(external, destination)

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is False
    assert record == {"name": "trial.log", "reason": "write_failed", "size_bytes": 11}
    assert external.read_text(encoding="utf-8") == "outside-sentinel"


@pytest.mark.parametrize(("size_bytes", "expected_ok"), [(4, True), (5, False)], ids=("exact", "beyond"))
def test_diagnostic_copy_honors_lower_configured_byte_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size_bytes: int,
    expected_ok: bool,
) -> None:
    source = tmp_path / "trial.log"
    destination = tmp_path / "out" / "trial.log"
    source.write_text("x" * size_bytes, encoding="utf-8")
    monkeypatch.setenv("SKILLEVALUATOR_HARBOR_DIAGNOSTIC_ARTIFACT_MAX_BYTES", "4")

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is expected_ok
    if expected_ok:
        assert destination.read_text(encoding="utf-8") == "x" * size_bytes
        assert record == {"name": "trial.log", "size_bytes": size_bytes}
    else:
        assert not destination.exists()
        assert record == {
            "name": "trial.log",
            "reason": "exceeds_max_bytes",
            "size_bytes": size_bytes,
            "max_bytes": 4,
        }


def test_diagnostic_copy_zero_bound_rejects_nonempty_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trial.log"
    destination = tmp_path / "out" / "trial.log"
    source.write_text("x", encoding="utf-8")
    monkeypatch.setenv("SKILLEVALUATOR_HARBOR_DIAGNOSTIC_ARTIFACT_MAX_BYTES", "0")

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is False
    assert not destination.exists()
    assert record == {
        "name": "trial.log",
        "reason": "exceeds_max_bytes",
        "size_bytes": 1,
        "max_bytes": 0,
    }


def test_diagnostic_copy_clamps_extreme_configured_byte_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "trial.log"
    destination = tmp_path / "out" / "trial.log"
    source.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("SKILLEVALUATOR_HARBOR_DIAGNOSTIC_ARTIFACT_MAX_BYTES", "1" + "0" * 100)

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is True
    assert record == {"name": "trial.log", "size_bytes": 2}
    assert destination.read_text(encoding="utf-8") == "ok"
    assert collector_module._diagnostic_artifact_max_bytes() == collector_module.DIAGNOSTIC_ARTIFACT_HARD_MAX_BYTES


def test_diagnostic_copy_allows_moderate_configured_bound_above_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trial.log"
    destination = tmp_path / "out" / "trial.log"
    size_bytes = collector_module.DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES + 1
    configured_max = size_bytes + 1024
    source.write_bytes(b"x" * size_bytes)
    monkeypatch.setenv("SKILLEVALUATOR_HARBOR_DIAGNOSTIC_ARTIFACT_MAX_BYTES", str(configured_max))

    ok, record = _write_redacted_text_copy(source, destination)

    assert ok is True
    assert collector_module._diagnostic_artifact_max_bytes() == configured_max
    assert record == {"name": "trial.log", "size_bytes": size_bytes}
    assert destination.stat().st_size == size_bytes


def test_successful_collection_does_not_materialize_job_result_as_a_trial(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    trial_name = "case-001__attempt"
    _write_reward(job_dir, trial_name, _default_reward("case-001", 0.8))
    _write_complete_job_result(job_dir, [trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "succeeded"
    trials_dir = tmp_path / "results" / "opencode" / "with-skill" / "trials"
    assert {path.name for path in trials_dir.iterdir()} == {trial_name}


def _create_windows_junction(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junction semantics")
def test_diagnostic_copy_rejects_native_windows_trial_root_junction(tmp_path: Path) -> None:
    external_trial = tmp_path / "external-trial"
    external_trial.mkdir()
    marker = "external-windows-trial-marker"
    (external_trial / "trial.log").write_text(marker, encoding="utf-8")
    trial_junction = tmp_path / "trial-junction"
    _create_windows_junction(trial_junction, external_trial)
    trial_out = tmp_path / "collected"
    trial_out.mkdir()

    copied = _copy_trial_artifacts(trial_junction, trial_out)

    assert copied == []
    assert not (trial_out / "trial.log").exists()
    assert marker not in json.dumps(json.loads((trial_out / "artifact_manifest.json").read_text(encoding="utf-8")))


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junction semantics")
def test_diagnostic_copy_rejects_native_windows_agent_junction(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    external_agent = tmp_path / "external-agent"
    external_agent.mkdir()
    marker = "external-windows-agent-marker"
    (external_agent / "codex.txt").write_text(marker, encoding="utf-8")
    _create_windows_junction(trial_dir / "agent", external_agent)
    trial_out = tmp_path / "collected"
    trial_out.mkdir()

    copied = _copy_trial_artifacts(trial_dir, trial_out)

    assert copied == []
    assert not (trial_out / "codex.txt").exists()
    assert marker not in json.dumps(json.loads((trial_out / "artifact_manifest.json").read_text(encoding="utf-8")))


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junction semantics")
def test_diagnostic_copy_rejects_native_windows_destination_parent_junction(tmp_path: Path) -> None:
    source = tmp_path / "trial.log"
    source.write_text("safe-source", encoding="utf-8")
    external_output = tmp_path / "external-output"
    external_output.mkdir()
    output_junction = tmp_path / "output-junction"
    _create_windows_junction(output_junction, external_output)

    ok, record = _write_redacted_text_copy(source, output_junction / "trial.log")

    assert ok is False
    assert record == {"name": "trial.log", "reason": "write_failed", "size_bytes": 11}
    assert not (external_output / "trial.log").exists()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junction semantics")
def test_judge_sidecar_scan_rejects_native_windows_verifier_junction(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    external_verifier = tmp_path / "external-verifier"
    external_verifier.mkdir()
    marker = "external-windows-sidecar-marker"
    (external_verifier / "skill_evaluator_reward.json").write_text(
        json.dumps({"evaluation_status": "failed", "evaluation_errors": {"accuracy": marker}}),
        encoding="utf-8",
    )
    _create_windows_junction(trial_dir / "verifier", external_verifier)

    diagnostic = collector_module._failed_judge_diagnostic(trial_dir)

    assert diagnostic is not None
    assert diagnostic["evaluation_status"] == "failed"
    assert "symlink" in json.dumps(diagnostic).casefold()
    assert marker not in json.dumps(diagnostic)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO paths are not available on this platform")
def test_diagnostic_copy_rejects_fifo_without_blocking_and_records_manifest(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_out = tmp_path / "collected"
    trial_dir.mkdir()
    trial_out.mkdir()
    os.mkfifo(trial_dir / "trial.log")
    script = (
        "from pathlib import Path; "
        "from skillevaluator.tier3.harbor.collector import _copy_trial_artifacts; "
        f"assert _copy_trial_artifacts(Path({str(trial_dir)!r}), Path({str(trial_out)!r})) == []"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (trial_out / "trial.log").exists()
    manifest = json.loads((trial_out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["copied"] == []
    assert manifest["skipped"][0]["name"] == "trial.log"
    assert "regular" in manifest["skipped"][0]["reason"]


def test_harbor_trial_name_cannot_escape_results_directory(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    physical_trial_name = "case-001__attempt"
    trial_dir = job_dir / physical_trial_name
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "../../../../escaped",
                "task_name": "case-001",
                "verifier_result": {"rewards": _default_reward("case-001", 0.8)},
            }
        ),
        encoding="utf-8",
    )
    _write_complete_job_result(job_dir, [physical_trial_name])

    result = _collect(tmp_path, skip_baseline=True, case_ids=["case-001"])

    assert result["execution_status"] == "succeeded"
    assert not (tmp_path / "escaped" / "reward.json").exists()
    persisted = tmp_path / "results" / "opencode" / "with-skill" / "trials" / physical_trial_name / "reward.json"
    assert persisted.exists()


@pytest.mark.parametrize(
    ("artifact", "agents"),
    (("attempt_policy.json", ["opencode"]), ("comparison.json", ["opencode", "claude-code"])),
)
def test_root_generated_artifact_write_unlinks_symlink_without_following(
    tmp_path: Path,
    artifact: str,
    agents: list[str],
) -> None:
    for agent in agents:
        job_dir = tmp_path / "jobs" / f"demo-{agent}-with"
        _write_reward(job_dir, "case-001__attempt", _default_reward("case-001", 0.8))
        _write_complete_job_result(job_dir, ["case-001__attempt"])
    results_dir = tmp_path / "results"
    outside = tmp_path / "outside"
    results_dir.mkdir()
    outside.mkdir()
    sentinel = outside / artifact
    sentinel.write_text("outside sentinel", encoding="utf-8")
    generated = results_dir / artifact
    generated.symlink_to(sentinel)

    collect_harbor_results(
        skill_name="demo",
        agents=agents,
        output_dir=results_dir,
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )

    assert sentinel.read_text(encoding="utf-8") == "outside sentinel"
    assert generated.is_file()
    assert not generated.is_symlink()


@pytest.mark.parametrize(
    ("artifact", "agents"),
    (("attempt_policy.json", ["opencode"]), ("comparison.json", ["opencode", "claude-code"])),
)
def test_root_generated_artifact_write_rejects_hard_link_replacement_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    agents: list[str],
) -> None:
    for agent in agents:
        job_dir = tmp_path / "jobs" / f"demo-{agent}-with"
        _write_reward(job_dir, "case-001__attempt", _default_reward("case-001", 0.8))
        _write_complete_job_result(job_dir, ["case-001__attempt"])
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sentinel = tmp_path / f"outside-{artifact}"
    sentinel.write_text("outside sentinel", encoding="utf-8")
    generated = results_dir / artifact
    original_prepare = collector_module._prepare_generated_outputs

    def prepare_then_replace(output_root: Path, selected_agents: list[str]) -> None:
        original_prepare(output_root, selected_agents)
        os.link(sentinel, generated)

    monkeypatch.setattr(collector_module, "_prepare_generated_outputs", prepare_then_replace)

    with pytest.raises(ValueError, match="single-link regular file"):
        collect_harbor_results(
            skill_name="demo",
            agents=agents,
            output_dir=results_dir,
            jobs_dir=tmp_path / "jobs",
            skip_baseline=True,
            expected_cases=1,
            expected_case_ids=["case-001"],
            expected_trials=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "outside sentinel"
    assert generated.samefile(sentinel)
    assert generated.stat().st_nlink == 2
    assert not list(results_dir.glob(f".{artifact}.*.tmp"))


def test_findings_skip_failed_single_agent_even_when_stale_rewards_exist(tmp_path: Path, capsys) -> None:
    trial_dir = tmp_path / "opencode" / "with-skill" / "trials" / "case-001__attempt"
    trial_dir.mkdir(parents=True)
    (trial_dir / "reward.json").write_text(json.dumps(_default_reward("case-001", 0.1)), encoding="utf-8")
    stale_findings = tmp_path / "opencode" / "findings.json"
    stale_findings.write_text('{"stale": true}', encoding="utf-8")

    rendered = report.display_findings_report(
        {
            "agents": {
                "opencode": {
                    "execution_status": "failed",
                    "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.1),
                    "conditions": {"with_skill": {"execution_status": "failed"}},
                }
            }
        },
        "demo",
        ["opencode"],
        tmp_path,
    )

    assert rendered == set()
    assert capsys.readouterr().out == ""
    assert not stale_findings.exists()


def test_findings_keep_succeeded_with_skill_when_only_baseline_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = tmp_path / "opencode"
    condition_dir = agent_dir / "with-skill"
    trial_dir = condition_dir / "trials" / "case-001__attempt"
    trial_dir.mkdir(parents=True)
    score = 0.1
    (condition_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "opencode",
                "scores": dict.fromkeys(DEFAULT_METRICS, score),
                "metrics": list(DEFAULT_METRICS),
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "reward.json").write_text(json.dumps(_default_reward("case-001", score)), encoding="utf-8")
    monkeypatch.setattr(report, "_generate_suggestions_structured", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(report, "_passing_skill_suggestions", lambda *_args, **_kwargs: [])

    rendered = report.display_findings_report(
        {
            "agents": {
                "opencode": {
                    "execution_status": "failed",
                    "with_skill": dict.fromkeys(DEFAULT_METRICS, score),
                    "conditions": {
                        "with_skill": {"execution_status": "succeeded"},
                        "without_skill": {"execution_status": "failed"},
                    },
                }
            }
        },
        "demo",
        ["opencode"],
        tmp_path,
    )

    assert rendered
    assert (agent_dir / "findings.json").exists()


def test_findings_do_not_follow_external_reward_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = tmp_path / "opencode"
    condition_dir = agent_dir / "with-skill"
    trial_dir = condition_dir / "trials" / "case-001__attempt"
    trial_dir.mkdir(parents=True)
    score = 0.1
    (condition_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "opencode",
                "scores": dict.fromkeys(DEFAULT_METRICS, score),
                "metrics": list(DEFAULT_METRICS),
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    external_marker = "outside-reward-must-not-reach-suggestion-model"
    outside_reward = tmp_path / "outside-reward.json"
    payload = _default_reward("case-001", score)
    payload["details"]["goal_accuracy"]["reason"] = external_marker
    outside_reward.write_text(json.dumps(payload), encoding="utf-8")
    (trial_dir / "reward.json").symlink_to(outside_reward)
    suggestion_called = False

    def capture_suggestions(*_args, **_kwargs):
        nonlocal suggestion_called
        suggestion_called = True
        return []

    monkeypatch.setattr(report, "_generate_suggestions_structured", capture_suggestions)
    monkeypatch.setattr(report, "_passing_skill_suggestions", lambda *_args, **_kwargs: [])

    rendered = report.display_findings_report(
        {
            "agents": {
                "opencode": {
                    "execution_status": "succeeded",
                    "with_skill": dict.fromkeys(DEFAULT_METRICS, score),
                    "conditions": {"with_skill": {"execution_status": "succeeded"}},
                }
            }
        },
        "demo",
        ["opencode"],
        tmp_path,
    )

    assert rendered == set()
    assert suggestion_called is False
    assert not (agent_dir / "findings.json").exists()


def test_findings_multi_agent_selects_and_writes_only_successful_agent(tmp_path: Path, monkeypatch) -> None:
    for agent, score in (("failed", 1.0), ("succeeded", 0.8)):
        condition_dir = tmp_path / agent / "with-skill"
        trial_dir = condition_dir / "trials" / "case-001__attempt"
        trial_dir.mkdir(parents=True)
        status = "failed" if agent == "failed" else "succeeded"
        (condition_dir / "summary.json").write_text(
            json.dumps(
                {
                    "agent": agent,
                    "scores": dict.fromkeys(DEFAULT_METRICS, score),
                    "metrics": list(DEFAULT_METRICS),
                    "execution_status": status,
                    "execution_errors": ["agent failed"] if status == "failed" else [],
                    "expected_attempts": 1,
                    "scored_attempts": 0 if status == "failed" else 1,
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "reward.json").write_text(json.dumps(_default_reward("case-001", score)), encoding="utf-8")
    monkeypatch.setattr(report, "_generate_suggestions_structured", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(report, "_passing_skill_suggestions", lambda *_args, **_kwargs: [])

    rendered = report.display_findings_report(
        {
            "agents": {
                "failed": {
                    "execution_status": "failed",
                    "with_skill": dict.fromkeys(DEFAULT_METRICS, 1.0),
                    "conditions": {"with_skill": {"execution_status": "failed"}},
                },
                "succeeded": {
                    "execution_status": "succeeded",
                    "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
                    "conditions": {"with_skill": {"execution_status": "succeeded"}},
                },
            }
        },
        "demo",
        ["failed", "succeeded"],
        tmp_path,
    )

    assert rendered
    assert not (tmp_path / "failed" / "findings.json").exists()
    assert (tmp_path / "succeeded" / "findings.json").exists()
