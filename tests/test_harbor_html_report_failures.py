# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Harbor failure details in the HTML report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.collector import _build_comparison, collect_harbor_results
from skillevaluator.tier3.harbor.html_report import generate_html_report
from skillevaluator.tier3.results_location import external_results_root, resolve_latest_results

if TYPE_CHECKING:
    import pytest


def test_report_renders_aggregate_and_trial_failure_details(tmp_path: Path) -> None:
    results_dir = tmp_path / "20260704_220000"
    summary_dir = results_dir / "opencode" / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {},
                "metrics": [],
                "num_trials": 0,
                "job_failure": "Harbor job result is missing trial state counter: n_pending_trials",
                "trial_failures": [
                    {
                        "trial": "case-001__attempt",
                        "reason": "AgentTimeoutError: Agent timed out after <600> seconds",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_summary_dir = results_dir / "opencode" / "without-skill"
    baseline_summary_dir.mkdir(parents=True)
    (baseline_summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {},
                "metrics": [],
                "num_trials": 0,
                "job_failure": "Harbor job did not complete successfully: 1 cancelled",
                "trial_failures": [
                    {"trial": "case-002__attempt", "reason": "HarborTrialError: cancelled by scheduler"}
                ],
            }
        ),
        encoding="utf-8",
    )

    report_path = generate_html_report("demo", results_dir)
    output = report_path.read_text(encoding="utf-8")

    assert "Failure Details" in output
    assert "With skill aggregate job" in output
    assert "missing trial state counter: n_pending_trials" in output
    assert "case-001__attempt" in output
    assert "AgentTimeoutError: Agent timed out after &lt;600&gt; seconds" in output
    assert "Without skill aggregate job" in output
    assert "Harbor job did not complete successfully: 1 cancelled" in output
    assert "case-002__attempt" in output


def test_collector_failure_details_flow_into_generated_report(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_dir = jobs_dir / "demo-opencode-with" / "case-001__attempt"
    trial_dir.mkdir(parents=True)
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
    results_dir = tmp_path / "20260704_220001"

    collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=results_dir,
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
    )
    output = generate_html_report("demo", results_dir).read_text(encoding="utf-8")

    assert "With skill aggregate job" in output
    assert "did not produce" in output
    assert "result.json" in output
    assert "case-001__attempt" in output
    assert "AgentTimeoutError: Agent timed out after 600 seconds" in output


def test_failed_agent_has_no_synthetic_score_in_comparison_or_html(tmp_path: Path) -> None:
    agents = {
        "failed": {
            "execution_status": "failed",
            "with_skill": {},
            "without_skill": {},
            "lift": {},
        },
        "succeeded": {
            "execution_status": "succeeded",
            "with_skill": {"accuracy": 0.8},
            "without_skill": {"accuracy": 0.5},
            "lift": {"accuracy": {"delta": 0.3}},
        },
    }

    comparison = _build_comparison(agents)
    failed_score = comparison["metrics"]["accuracy"]["failed"]
    assert failed_score == {"with_skill": None, "without_skill": None, "lift": None}

    for name, data in agents.items():
        summary_dir = tmp_path / name / "with-skill"
        summary_dir.mkdir(parents=True)
        summary = {
            "scores": data["with_skill"],
            "metrics": list(data["with_skill"]),
            "num_trials": 0 if name == "failed" else 1,
            "execution_status": data["execution_status"],
            "execution_errors": ["agent failed"] if name == "failed" else [],
            "expected_attempts": 1,
            "scored_attempts": 0 if name == "failed" else 1,
        }
        (summary_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    output = generate_html_report("demo", tmp_path).read_text(encoding="utf-8")
    assert '<td class="subtle">NO SCORE</td>' in output


def test_partial_execution_failure_reports_coverage_without_quality_claims(tmp_path: Path) -> None:
    results_dir = tmp_path / "20260708_120000"
    with_skill_dir = results_dir / "opencode" / "with-skill"
    with_skill_dir.mkdir(parents=True)
    (with_skill_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {},
                "metrics": [],
                "num_trials": 0,
                "execution_status": "failed",
                "execution_errors": ["With-skill environment setup failed before trials started"],
                "expected_attempts": 4,
                "scored_attempts": 0,
            }
        ),
        encoding="utf-8",
    )

    baseline_dir = results_dir / "opencode" / "without-skill"
    baseline_trials_dir = baseline_dir / "trials"
    baseline_trials_dir.mkdir(parents=True)
    (baseline_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {"accuracy": 1.0},
                "metrics": ["accuracy"],
                "num_trials": 4,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 4,
                "scored_attempts": 4,
            }
        ),
        encoding="utf-8",
    )

    for index in range(1, 5):
        entry_id = f"case-{index:03d}"
        baseline_trial = baseline_trials_dir / f"{entry_id}__attempt-1"
        baseline_trial.mkdir()
        (baseline_trial / "reward.json").write_text(
            json.dumps({"entry_id": entry_id, "accuracy": 1.0, "overall": 1.0}),
            encoding="utf-8",
        )
        for condition in ("with", "without"):
            staged_entry = results_dir / "_harbor-tasks" / "opencode" / condition / entry_id / "tests"
            staged_entry.mkdir(parents=True)
            (staged_entry / "entry.json").write_text(
                json.dumps({"id": entry_id, "prompt": f"Evaluate case {index}"}),
                encoding="utf-8",
            )

    output = generate_html_report("demo", results_dir).read_text(encoding="utf-8")

    assert "All commands succeeded on first attempt" not in output
    assert "Resolve the execution failures shown in Failure Details" in output
    assert "Expand evals.json" not in output
    assert "currently 0" not in output
    assert "4 AgentSkills eval case(s) in dataset" in output
    assert "0 trials — agent did not produce results" not in output
    assert "No valid with-skill score" in output
    assert "4 of 8 expected attempts scored" in output
    assert "Scored 4 of 8 expected logical attempts" in output


def test_failed_with_skill_condition_ignores_stale_partial_quality_artifacts(tmp_path: Path) -> None:
    results_dir = tmp_path / "20260708_120004"
    metrics = ["skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check"]
    with_skill_dir = results_dir / "opencode" / "with-skill"
    with_trial = with_skill_dir / "trials" / "case-001__attempt-1"
    with_trial.mkdir(parents=True)
    (with_skill_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(metrics, 0.9),
                "metrics": metrics,
                "num_trials": 1,
                "execution_status": "failed",
                "execution_errors": ["one required attempt failed"],
                "expected_attempts": 2,
                "scored_attempts": 1,
                "pass_at_k": {
                    "rate": 1.0,
                    "passed_cases": 1,
                    "total_cases": 1,
                    "attempts_used": 1,
                    "max_attempts_possible": 2,
                    "cases": {
                        "case-001": {
                            "passed": True,
                            "attempts_used": 1,
                            "attempts_missing": 1,
                            "first_pass_attempt": 1,
                            "attempts": [{"attempt": 1, "score": 0.9, "passed": True}],
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (with_trial / "reward.json").write_text(
        json.dumps({"entry_id": "case-001", **dict.fromkeys(metrics, 0.9)}),
        encoding="utf-8",
    )

    baseline_dir = results_dir / "opencode" / "without-skill"
    baseline_trial = baseline_dir / "trials" / "case-001__attempt-1"
    baseline_trial.mkdir(parents=True)
    (baseline_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(metrics, 0.2),
                "metrics": metrics,
                "num_trials": 1,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    (baseline_trial / "reward.json").write_text(
        json.dumps({"entry_id": "case-001", **dict.fromkeys(metrics, 0.2)}),
        encoding="utf-8",
    )
    (results_dir / "opencode" / "lift.json").write_text(
        json.dumps({metric: {"with_skill": 0.9, "without_skill": 0.2, "delta": 0.7} for metric in metrics}),
        encoding="utf-8",
    )
    (results_dir / "attempt_policy.json").write_text(
        json.dumps({"max_attempts": 2, "pass_threshold": 0.5}),
        encoding="utf-8",
    )

    output = generate_html_report("demo", results_dir).read_text(encoding="utf-8")

    assert "NO SCORE" in output
    assert "one required attempt failed" in output
    assert "0.90" not in output
    assert "+0.70" not in output
    assert "passed on attempt" not in output
    assert "1 of 2 scored" in output
    assert "1 not scored" in output
    assert "0 of 0 scored" not in output


def test_baseline_only_failure_keeps_valid_with_skill_score(tmp_path: Path) -> None:
    results_dir = tmp_path / "20260708_120001"
    metrics = ["skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check"]
    with_skill_dir = results_dir / "opencode" / "with-skill"
    with_skill_trials = with_skill_dir / "trials" / "case-001__attempt-1"
    with_skill_trials.mkdir(parents=True)
    (with_skill_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(metrics, 0.8),
                "metrics": metrics,
                "num_trials": 1,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    (with_skill_trials / "reward.json").write_text(
        json.dumps({"entry_id": "case-001", "accuracy": 0.8}),
        encoding="utf-8",
    )

    baseline_dir = results_dir / "opencode" / "without-skill"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {},
                "metrics": metrics,
                "num_trials": 0,
                "execution_status": "failed",
                "execution_errors": ["Baseline environment setup failed"],
                "expected_attempts": 1,
                "scored_attempts": 0,
            }
        ),
        encoding="utf-8",
    )

    output = generate_html_report("demo", results_dir).read_text(encoding="utf-8")

    assert "Baseline environment setup failed" in output
    assert "Baseline execution failed; lift unavailable" in output
    assert "No valid with-skill score" not in output
    assert "No successfully scored agent" not in output
    assert '<div class="agent-score" style="color:#22c55e">0.80</div>' in output
    assert '<td class="subtle">NO SCORE</td>' not in output


def test_mixed_agent_charts_use_null_for_failed_with_skill_scores(tmp_path: Path) -> None:
    metrics = ["skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check"]
    for agent, status, score in (("failed", "failed", None), ("succeeded", "succeeded", 0.8)):
        summary_dir = tmp_path / agent / "with-skill"
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps(
                {
                    "scores": {} if score is None else dict.fromkeys(metrics, score),
                    "metrics": metrics,
                    "num_trials": 0 if score is None else 1,
                    "execution_status": status,
                    "execution_errors": ["agent failed"] if score is None else [],
                    "expected_attempts": 1,
                    "scored_attempts": 0 if score is None else 1,
                    "pass_at_k": {
                        "rate": 0.5 if score is None else 0.75,
                        "passed_cases": 1,
                        "total_cases": 1,
                        "attempts_used": 1,
                        "max_attempts_possible": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "attempt_policy.json").write_text(
        json.dumps({"max_attempts": 2, "pass_threshold": 0.5}),
        encoding="utf-8",
    )

    output = generate_html_report("demo", tmp_path).read_text(encoding="utf-8")

    assert 'label:"failed",data:[null, null, null, null, null]' in output
    assert 'label:"failed",data:[0.0, 0.0, 0.0, 0.0, 0.0]' not in output
    assert 'label:"succeeded",data:[0.8, 0.8, 0.8, 0.8, 0.8]' in output
    assert "<td><b>50%</b></td>" not in output
    assert "<td><b>75%</b></td>" in output


def test_failed_default_run_without_metrics_is_not_labeled_custom_only(tmp_path: Path) -> None:
    results_dir = tmp_path / "20260708_120002"
    summary_dir = results_dir / "opencode" / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {},
                "metrics": [],
                "num_trials": 0,
                "execution_status": "failed",
                "execution_errors": ["agent setup failed"],
                "expected_attempts": 1,
                "scored_attempts": 0,
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "run_config.json").write_text(
        json.dumps({"grading": {"mode": "default"}}),
        encoding="utf-8",
    )

    output = generate_html_report("demo", results_dir).read_text(encoding="utf-8")

    assert "Custom Reward Mode" not in output
    assert "BYOT" not in output
    assert "custom-only user reward overall" not in output
    assert "No scored evaluator metrics are available" in output


def test_successful_custom_only_run_keeps_custom_reward_labels(tmp_path: Path) -> None:
    results_dir = tmp_path / "20260708_120003"
    with_skill_dir = results_dir / "opencode" / "with-skill"
    trial_dir = with_skill_dir / "trials" / "case-001__attempt-1"
    trial_dir.mkdir(parents=True)
    (with_skill_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {},
                "metrics": [],
                "num_trials": 1,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "reward.json").write_text(
        json.dumps({"entry_id": "case-001", "overall": 0.7}),
        encoding="utf-8",
    )
    baseline_dir = results_dir / "opencode" / "without-skill"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": {},
                "metrics": [],
                "num_trials": 0,
                "execution_status": "skipped",
                "execution_errors": [],
                "expected_attempts": 0,
                "scored_attempts": 0,
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "run_config.json").write_text(
        json.dumps({"grading": {"mode": {"value": "custom_only", "source": "CLI"}}}),
        encoding="utf-8",
    )

    output = generate_html_report("demo", results_dir).read_text(encoding="utf-8")

    assert "Custom Reward Mode" in output
    assert "BYOT" in output
    assert "custom-only user reward overall" in output
    assert '<div class="agent-score" style="color:#eab308">0.70</div>' in output
    assert "No Scored Metrics" not in output


def test_pre_job_launch_failure_produces_html_report(tmp_path: Path) -> None:
    results_dir = tmp_path / "20260704_220002"
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=results_dir,
        jobs_dir=jobs_dir,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        launch_errors=["opencode with-skill Harbor run failed: model not found"],
    )
    report = generate_html_report("demo", results_dir)

    assert results["agents"]["opencode"]["job_failures"]["with_skill"] == "model not found"
    output = report.read_text(encoding="utf-8")
    assert "Failure Details" in output
    assert "model not found" in output


def test_html_generation_failure_is_persisted_identically_to_returned_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A report warning is part of the final result contract on disk and in memory."""
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    output_dir = external_results_root(cli_results_dir, skill_path)
    provider = ProviderConfig(
        provider="openai",
        model="test-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )

    def emit_tasks(_skill_path: Path, tasks_dir: Path, **_kwargs: object) -> list[Path]:
        task = tasks_dir / "case-001"
        task.mkdir(parents=True)
        return [task]

    def fail_html(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("template rendering failed")

    def deny_symlink(*_args: object, **_kwargs: object) -> None:
        raise OSError("symbolic links require Developer Mode")

    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _skill_path: ({"harbor": {"task_source": "evals_json"}}, None),
    )
    monkeypatch.setattr(runner, "find_evals_file", lambda _skill_path: skill_path / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "generate_harbor_tasks", emit_tasks)
    monkeypatch.setattr(runner, "_run_agent_pair", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "generate_html_report", fail_html)
    monkeypatch.setattr(runner, "record_agent_eval_summary", lambda **_kwargs: None)
    monkeypatch.setattr(Path, "symlink_to", deny_symlink)

    returned = runner.run_harbor_eval(
        skill_path,
        ["opencode"],
        skip_baseline=True,
        n_attempts=1,
        n_concurrent=1,
        max_agents=1,
        output_dir=output_dir,
        env_mode="docker",
        agent_runtime_preflight=False,
    )

    run_dir = Path(returned["run_dir"])
    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert returned["warnings"] == ["HTML report was not generated: template rendering failed"]
    assert persisted == returned
    assert resolve_latest_results(skill_path, cli_results_dir, environ={}) == run_dir
