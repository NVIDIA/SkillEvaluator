# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline-only Tier 3 runs (``--skip-with-skill``) stay truthful end to end."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli, evaluate
from skillevaluator.evaluation import EvaluationService
from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory, render_agent_eval_html_report
from skillevaluator.tier3 import commands
from skillevaluator.tier3.harbor import progress, runner
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.result_display import render_result
from skillevaluator.tier3.workflow import build_tier3_workflow

CASE_IDS = ["case-001", "case-002"]


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    path = tmp_path / "demo"
    path.mkdir()
    (path / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Explain how to validate a configuration.\n---\n# Demo\n",
        encoding="utf-8",
    )
    return path


def _reward(score: float, **identity: str) -> dict[str, Any]:
    return {**identity, "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, score), "overall": score}


def _write_job(job_dir: Path, scores: dict[str, float]) -> None:
    for case_id, score in scores.items():
        verifier = job_dir / f"{case_id}__attempt" / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps(_reward(score, entry_id=case_id)), encoding="utf-8")
    _write_job_result(job_dir, [f"{case_id}__attempt" for case_id in scores])


def _write_job_result(job_dir: Path, trial_names: list[str]) -> None:
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


def _collect_baseline_only(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    jobs_dir = tmp_path / "jobs"
    # A with-skill job left in a reused jobs directory must never be adopted.
    _write_job(jobs_dir / "demo-opencode-with", dict.fromkeys(CASE_IDS, 0.99))
    _write_job(jobs_dir / "demo-opencode-without", {"case-001": 0.62, "case-002": 0.72})
    run_dir = tmp_path / "results" / "20260101_000000"
    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=run_dir,
        jobs_dir=jobs_dir,
        skip_with_skill=True,
        expected_cases=len(CASE_IDS),
        expected_case_ids=CASE_IDS,
        expected_trials=len(CASE_IDS),
    )
    return results, run_dir


# ---------------------------------------------------------------------------
# CLI, service and engine entry points
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_options(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    captured: list[Any] = []
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: captured.append(options) or {"execution_status": "succeeded"},
    )
    return captured


def test_cli_forwards_skip_with_skill(skill: Path, captured_options: list[Any]) -> None:
    result = CliRunner().invoke(
        cli,
        ["tier3", "evaluate", str(skill), "--agents", "opencode", "--skip-with-skill", "--progress", "off"],
    )

    assert result.exit_code == 0, result.output
    assert captured_options[0].skip_with_skill is True
    assert captured_options[0].skip_baseline is False


def test_cli_rejects_skipping_both_arms_before_any_work(
    skill: Path, captured_options: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must not generate")
    )
    result = CliRunner().invoke(
        cli,
        ["tier3", "evaluate", str(skill), "--autopilot", "--skip-baseline", "--skip-with-skill", "--progress", "off"],
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
    assert captured_options == []


def test_direct_workflow_rejects_skipping_both_arms_before_dataset_generation(
    skill: Path, captured_options: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must not generate")
    )
    result = CliRunner().invoke(
        build_tier3_workflow(evaluate),
        [str(skill), "--skip-baseline", "--skip-with-skill", "--progress", "off"],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert captured_options == []


def _commands_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "agents": "opencode",
        "env_mode": "docker",
        "skip_baseline": False,
        "n_attempts": None,
        "pass_threshold": None,
        "n_concurrent": None,
        "max_agents": None,
        "model": None,
        "agent_model": (),
        "custom_dockerfile_mode": None,
        "skill_workspace_mode": None,
        "include_skills": (),
        "copy_repo": False,
        "grading_mode": None,
        "results_dir": None,
        "harbor_keep_jobs": False,
        "timeout_multiplier": None,
        "override_cpus": None,
        "override_memory_mb": None,
        "override_storage_mb": None,
    }
    kwargs.update(overrides)
    return kwargs


def test_commands_forward_skip_with_skill_and_reject_both_arms(skill: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))
    monkeypatch.setattr(commands, "run_harbor_eval", lambda **kwargs: calls.append(kwargs) or {})

    commands.evaluate(skill, **_commands_kwargs(skip_with_skill=True))
    assert calls[0]["skip_with_skill"] is True
    assert calls[0]["skip_baseline"] is False

    with pytest.raises(ValueError, match="mutually exclusive"):
        commands.evaluate(skill, **_commands_kwargs(skip_baseline=True, skip_with_skill=True))
    assert len(calls) == 1


def test_engine_rejects_skipping_both_arms(skill: Path) -> None:
    result = runner.run_harbor_eval(skill, ["opencode"], skip_baseline=True, skip_with_skill=True)

    assert result == {"error": ["skip_baseline and skip_with_skill are mutually exclusive (no arm would run)"]}


# ---------------------------------------------------------------------------
# Runner staging and launch
# ---------------------------------------------------------------------------


class _Reporter:
    def __init__(self) -> None:
        self.plans: list[Any] = []
        self.events: list[Any] = []
        self.closed = False

    @property
    def is_active(self) -> bool:
        return bool(self.plans) and not self.closed

    def start(self, plan: Any) -> None:
        self.plans.append(plan)

    def set_secret_values(self, _values: Any) -> None:
        return None

    def emit(self, event: Any) -> None:
        self.events.append(event)

    def heartbeat(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _stub_engine(
    monkeypatch: pytest.MonkeyPatch,
    skill: Path,
    *,
    harbor_config: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    (skill / "evals").mkdir(exist_ok=True)
    evals_file = skill / "evals" / "evals.json"
    evals_file.write_text("{}", encoding="utf-8")
    provider = SimpleNamespace(
        provider="openai", model="gpt-5", api_key="test-provider-key", base_url="https://api.openai.com/v1"
    )
    calls: dict[str, list[Any]] = {"emitted": [], "agent_runs": [], "collected": []}

    def emit_tasks(_skill: Path, output: Path, **kwargs: Any) -> list[Path]:
        calls["emitted"].append({"output": output, **kwargs})
        tasks = [output / case_id for case_id in CASE_IDS]
        for task in tasks:
            task.mkdir(parents=True, exist_ok=True)
        return tasks

    def collect(**kwargs: Any) -> dict[str, Any]:
        calls["collected"].append(kwargs)
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": []}

    def write_report(_skill_path: Path, run_dir: Path, **_kwargs: Any) -> Path:
        path = run_dir / "report.html"
        path.write_text("<html></html>", encoding="utf-8")
        return path

    config = {"task_source": "evals_json", "n_attempts": 1, "n_concurrent": 1, "max_agents": 1}
    config.update(harbor_config or {})
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "load_evals_config", lambda _path: ({"harbor": config}, None))
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: evals_file)
    monkeypatch.setattr(runner, "generate_harbor_tasks", emit_tasks)
    monkeypatch.setattr(runner, "_provider_environment", lambda _provider: {"OPENAI_API_KEY": provider.api_key})
    monkeypatch.setattr(runner, "_resolve_runtime_env", lambda _templates, **_kwargs: ({}, []))
    monkeypatch.setattr(runner, "_harbor_subprocess_environment", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "_run_agent_pair", lambda **kwargs: calls["agent_runs"].append(kwargs) or [])
    monkeypatch.setattr(runner, "collect_harbor_results", collect)
    monkeypatch.setattr(runner, "render_agent_eval_html_report", write_report)
    from skillevaluator.tier3.harbor import runtime_preflight

    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: SimpleNamespace(
            ok=True,
            provider=selected.provider,
            model=selected.model,
            detail="model is available",
            failure_kind=None,
            http_status=None,
        ),
    )
    return calls


def test_runner_stages_and_runs_only_the_baseline_arm(skill: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_engine(monkeypatch, skill)
    reporter = _Reporter()

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        skip_with_skill=True,
        output_dir=skill.parent / "results",
        progress_reporter=reporter,
    )

    assert result["execution_status"] == "succeeded", result
    # Only the baseline arm is staged, with single-arm (unsuffixed) task names
    # that the collector resolves with the same rule.
    assert [call["with_skill"] for call in calls["emitted"]] == [False]
    assert calls["emitted"][0]["output"].name == "without"
    assert calls["emitted"][0]["arm_suffix"] == ""
    [agent_run] = calls["agent_runs"]
    assert agent_run["with_skill"] is None
    assert agent_run["baseline"] == calls["emitted"][0]["output"]
    assert agent_run["task_names"] == CASE_IDS
    [collected] = calls["collected"]
    assert collected["skip_with_skill"] is True
    assert collected["skip_baseline"] is False
    assert collected["expected_case_ids"] == CASE_IDS

    final_plan = reporter.plans[-1]
    assert final_plan.with_skill is False
    assert final_plan.baseline is True
    assert final_plan.matrix_trials == len(CASE_IDS)
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("with-skill-tasks", "skipped") in transitions
    assert ("with-skill-tasks", "running") not in transitions
    assert transitions.index(("baseline-tasks", "running")) < transitions.index(("baseline-tasks", "ready"))
    agent_event = next(event for event in reporter.events if event.stage == "agent:codex" and event.state == "running")
    assert agent_event.detail == "baseline only"


def test_paired_runner_still_suffixes_both_arms(skill: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_engine(monkeypatch, skill)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=skill.parent / "results")

    assert result["execution_status"] == "succeeded", result
    assert [(call["with_skill"], call["arm_suffix"]) for call in calls["emitted"]] == [
        (True, "-with-skill"),
        (False, "-without-skill"),
    ]
    assert calls["agent_runs"][0]["with_skill"] is not None
    assert calls["collected"][0]["skip_with_skill"] is False


def test_baseline_only_keeps_the_baseline_isolation_guard(skill: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_engine(monkeypatch, skill, harbor_config={"pre_agent_setup": ["echo setup"]})

    result = runner.run_harbor_eval(skill, ["codex"], skip_with_skill=True, output_dir=skill.parent / "results")

    assert "pre_agent_setup" in result["error"][0]
    assert calls["emitted"] == []


@pytest.mark.parametrize("stop_on_pass", [False, True])
def test_agent_pair_launches_only_the_baseline_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_on_pass: bool
) -> None:
    launched: list[str] = []
    monkeypatch.setattr(
        runner,
        "_run_harbor",
        lambda **kwargs: launched.append(kwargs["job_name"]) or (True, ""),
    )
    monkeypatch.setattr(
        runner,
        "_run_stop_on_pass_variant",
        lambda **kwargs: launched.append(f"{kwargs['skill_name']}-{kwargs['agent']}-{kwargs['variant']}") or [],
    )

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent="codex",
        model="gpt-5",
        env_mode="docker",
        with_skill=None,
        baseline=tmp_path / "without",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=2,
        n_concurrent=2,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=4,
        stop_on_pass=stop_on_pass,
        task_names=CASE_IDS,
    )

    assert errors == []
    assert launched == ["demo-codex-without"]


# ---------------------------------------------------------------------------
# Collection and reporting
# ---------------------------------------------------------------------------


def test_collector_records_a_skipped_with_skill_arm_and_ignores_stale_jobs(tmp_path: Path) -> None:
    results, run_dir = _collect_baseline_only(tmp_path)

    assert results["execution_status"] == "succeeded", results.get("execution_errors")
    agent = results["agents"]["opencode"]
    assert agent["conditions"]["with_skill"]["execution_status"] == "skipped"
    assert agent["conditions"]["without_skill"]["execution_status"] == "succeeded"
    assert agent["with_skill"] == {}
    assert agent["without_skill"]["goal_accuracy"] == pytest.approx(0.67)
    assert agent["num_trials_with"] == 0
    assert agent["num_trials_without"] == 2
    assert agent["lift"] == {}
    assert agent["pass_at_k"]["lift"] == {}
    assert agent["skill_activation"]["status"] == "not_applicable"
    assert agent["score_policy_with_skill"] is None
    assert results["metric_set"] == DEFAULT_METRIC_SET
    assert results["score_policy"] == agent["score_policy_without_skill"]
    assert not (run_dir / "opencode" / "with-skill" / "trials").exists()
    summary = json.loads((run_dir / "opencode" / "with-skill" / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution_status"] == "skipped"
    assert summary["scores"] == {}
    assert summary["skill_activation"]["status"] == "not_applicable"


def test_collector_resolves_unsuffixed_baseline_only_task_names(tmp_path: Path) -> None:
    # Single-arm runs stage unsuffixed task names. A case ID that happens to end
    # in the paired-run arm suffix must survive collection intact.
    case_id = "probe-without-skill"
    job_dir = tmp_path / "jobs" / "demo-opencode-without"
    trial_dir = job_dir / f"{case_id}__attempt"
    (trial_dir / "verifier").mkdir(parents=True)
    # No entry_id or task path: identity comes from Harbor's persisted task name.
    (trial_dir / "verifier" / "reward.json").write_text(json.dumps(_reward(0.7)), encoding="utf-8")
    (trial_dir / "result.json").write_text(json.dumps({"task_name": case_id}), encoding="utf-8")
    _write_job_result(job_dir, [trial_dir.name])

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_with_skill=True,
        expected_cases=1,
        expected_case_ids=[case_id],
        expected_trials=1,
    )

    assert results["execution_status"] == "succeeded", results.get("execution_errors")
    assert list(results["agents"]["opencode"]["pass_at_k"]["without_skill"]["cases"]) == [case_id]


def test_collector_rejects_skipping_both_arms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        collect_harbor_results(
            skill_name="demo",
            agents=["opencode"],
            output_dir=tmp_path / "results",
            jobs_dir=tmp_path / "jobs",
            skip_baseline=True,
            skip_with_skill=True,
        )


def test_report_presents_baseline_calibration_without_a_skill_verdict(tmp_path: Path, skill: Path) -> None:
    results, run_dir = _collect_baseline_only(tmp_path)

    result = agent_eval_result_from_directory(skill, run_dir, engine_result=results, use_llm_judge=False)

    assert result is not None
    assert result.passed is True
    assert "baseline calibration complete" in result.success_details[0].message
    payload = result.metadata["agent_eval"]
    assert payload["mode"] == payload["summary"]["mode"] == "baseline_only"
    assert payload["execution_status"] == "succeeded"
    assert payload["verdict"] == "neutral"
    assert payload["best_agent"] == "opencode"
    assert payload["overall_score"] == pytest.approx(0.67)
    assert payload["overall_lift"] is None
    agent = payload["agents"]["opencode"]
    assert agent["with_skill"] is None
    assert agent["baseline"] == pytest.approx(0.67)
    assert agent["lift"] is None
    # Unmeasured with-skill values stay unavailable; they never become 0.0.
    assert payload["evaluators"]["goal_accuracy"] == {
        "with_skill": None,
        "baseline": pytest.approx(0.67),
        "lift": None,
        "with_skill_status": "skipped",
        "baseline_status": "scored",
    }
    assert {card["with_skill"] for card in payload["evaluator_cards"]} == {None}
    assert {card["status"] for card in payload["evaluator_cards"]} == {"warn"}
    assert all(dimension["with_skill"] is None and dimension["lift"] is None for dimension in payload["dimensions"])
    assert all(dimension["baseline"] == pytest.approx(0.67) for dimension in payload["dimensions"])
    assert "no skill lift was measured" in payload["dimensions"][0]["explanation"]
    assert payload["conclusions"][0]["title"] == "Baseline calibration complete"
    assert "--skip-with-skill" in payload["suggestions"][0]


def test_paired_report_mode_is_both(tmp_path: Path, skill: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_job(jobs_dir / "demo-opencode-with", {"case-001": 0.8, "case-002": 0.9})
    _write_job(jobs_dir / "demo-opencode-without", {"case-001": 0.6, "case-002": 0.7})
    run_dir = tmp_path / "results" / "20260101_000000"
    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=run_dir,
        jobs_dir=jobs_dir,
        expected_cases=len(CASE_IDS),
        expected_case_ids=CASE_IDS,
        expected_trials=len(CASE_IDS),
    )

    result = agent_eval_result_from_directory(skill, run_dir, engine_result=results, use_llm_judge=False)

    assert result is not None
    payload = result.metadata["agent_eval"]
    assert payload["mode"] == "both"
    assert payload["overall_score"] == pytest.approx(0.85)
    assert payload["overall_lift"] == pytest.approx(0.2)
    assert "Tier 3 evaluation complete" in result.success_details[0].message


def test_html_report_explains_the_baseline_only_run(tmp_path: Path, skill: Path) -> None:
    results, run_dir = _collect_baseline_only(tmp_path)

    report = render_agent_eval_html_report(skill, run_dir, engine_result=results, use_llm_judge=False)
    html = report.read_text(encoding="utf-8")

    assert "Baseline-only calibration:" in html
    assert "Baseline calibration (no with-skill arm)" in html
    assert "No skill verdict: baseline-only run" in html
    assert "Verdict at a glance:" not in html
    assert '<span class="hero-score-label">baseline score</span>' in html
    assert re.search(r"<td><strong>Goal Accuracy</strong></td>\s*<td>skipped</td>\s*<td>0\.67</td>", html)


def test_report_tables_mark_the_with_skill_arm_skipped(tmp_path: Path, skill: Path) -> None:
    from skillevaluator.reporting.cli import CLIReporter
    from skillevaluator.reporting.markdown import MarkdownReporter

    results, run_dir = _collect_baseline_only(tmp_path)
    result = agent_eval_result_from_directory(skill, run_dir, engine_result=results, use_llm_judge=False)
    assert result is not None

    terminal = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render(result))
    assert re.search(r"Goal Accuracy\s*│\s*skipped\s*│\s*0\.67\s*│\s*skipped", terminal)
    assert "NO SCORE" not in terminal
    markdown = MarkdownReporter().render_all([result])
    assert "| Goal Accuracy | skipped | 0.67 | skipped |" in markdown


def test_cli_summary_marks_the_with_skill_arm_skipped(tmp_path: Path) -> None:
    results, _run_dir = _collect_baseline_only(tmp_path)

    rendered = render_result(results)

    assert re.search(r"Goal Accuracy\s+skipped\s+0\.67", rendered)
    # The case-first verdict line names the calibration instead of an unmeasured skill effect.
    verdict_line = rendered.splitlines()[0]
    assert "baseline only" in verdict_line
    assert "no skill effect was measured" in verdict_line
    assert "contracts differ" not in rendered
    # Per-case rows show no effect rather than "inconclusive" for an arm that never ran.
    assert "inconclusive" not in rendered
    assert re.search(r"case-001\s+-\s+1/1 .*\s+-\s*$", rendered, re.MULTILINE)
    security_dimension = next(line for line in rendered.splitlines() if line.strip("│ ").startswith("Security"))
    assert "skipped" in security_dimension


def test_progress_plan_names_the_baseline_only_mode() -> None:
    plan = progress.Tier3RunPlan(
        skill_name="demo", environment="docker", agents=("codex",), baseline=True, with_skill=False
    )
    plain = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=plain, refresh_interval=60)
    reporter.start(plan)
    reporter.close()
    assert "with-skill=no, baseline=yes" in plain.getvalue()

    rich_reporter = progress.RichProgressReporter(stream=io.StringIO(), refresh_interval=60)
    rich_reporter._live_plan = plan
    from rich.console import Console

    rendered = io.StringIO()
    Console(file=rendered, force_terminal=False, width=180).print(rich_reporter._build_live_table())
    assert "Baseline-only evaluation" in rendered.getvalue()
    assert "Skill-lift evaluation" not in rendered.getvalue()
