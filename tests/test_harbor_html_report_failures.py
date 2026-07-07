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
    )

    run_dir = Path(returned["run_dir"])
    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert returned["warnings"] == ["HTML report was not generated: template rendering failed"]
    assert persisted == returned
    assert resolve_latest_results(skill_path, cli_results_dir, environ={}) == run_dir
