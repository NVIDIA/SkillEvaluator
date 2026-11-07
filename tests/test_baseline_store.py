# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.tier3 import baseline_store
from skillevaluator.tier3.harbor import collector, runner, runtime_preflight
from skillevaluator.tier3.harbor.metrics import CUSTOM_ONLY_METRIC_SET, CUSTOM_SCORE_POLICY, DEFAULT_METRICS


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _job(path: Path, score: float, *, grading_mode: str = "default") -> None:
    trial = "case-001__attempt"
    _write(
        path / "result.json",
        {
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
                "n_retries": 0,
                "evals": {"eval": {"n_trials": 1, "n_errors": 0, "reward_stats": {"reward": {str(score): [trial]}}}},
            },
        },
    )
    _write(path / trial / "result.json", {"trial_name": trial, "task_id": {"path": str(path / "tasks/case-001")}})
    reward = {"entry_id": "case-001", **dict.fromkeys(DEFAULT_METRICS, score)}
    if grading_mode == "custom_only":
        reward = {
            "entry_id": "case-001",
            "metric_set": CUSTOM_ONLY_METRIC_SET,
            "score_policy": CUSTOM_SCORE_POLICY,
            "overall": score,
        }
    _write(path / trial / "verifier" / "reward.json", reward)


def _identity() -> dict:
    return baseline_store.comparison_identity(
        {
            "agents": {"codex": {"model": "model-a", "source": "explicit"}},
            "harbor": {"environment": "docker"},
            "judge": {"model": "judge-a"},
        },
        {
            "dataset_digest": "sha256:one",
            "expected_case_ids": ["case-001"],
            "evaluator_version": "1",
            "attempt_policy": {"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        },
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("expected_case_ids", ["case-002"], "dataset changed"),
        ("attempt_policy", {"max_attempts": 3}, "attempt policy changed"),
        ("score_policy", "custom", "score policy changed"),
        ("evaluator_version", "2", "evaluator version changed"),
    ],
)
def test_identity_reasons(field: str, value: object, reason: str) -> None:
    old = _identity()
    current = copy.deepcopy(old)
    current[field] = value
    assert any(reason in item for item in baseline_store.comparison_mismatches(current, old))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("agents", {"codex": {"model": "model-b", "source": "explicit"}}, "model changed for codex"),
        ("harbor", {"environment": "local"}, "environment"),
        ("judge", {"model": "judge-b"}, "judge/scorer"),
        ("new_setting", True, "new_setting changed"),
    ],
)
def test_config_identity_reasons(field: str, value: object, reason: str) -> None:
    old = _identity()
    current = copy.deepcopy(old)
    current["config"][field] = value
    assert any(reason in item for item in baseline_store.comparison_mismatches(current, old))


def test_identity_separates_skill_and_ignores_only_operational_settings() -> None:
    config = {"harbor": {"n_concurrent": 1}, "evaluated_source": {"revision": "one"}}
    old = baseline_store.comparison_identity(config, {})
    config["harbor"]["n_concurrent"] = 8
    config["evaluated_source"]["revision"] = "two"
    current = baseline_store.comparison_identity(config, {})
    assert current["skill"] != old["skill"]
    assert baseline_store.comparison_mismatches(current, old) == []


@pytest.mark.parametrize("skip", ["--skip-baseline", "--skip-with-skill"])
def test_cli_mutual_exclusion(tmp_path: Path, skip: str) -> None:
    result = CliRunner().invoke(cli, ["tier3", "evaluate", str(tmp_path), "--reuse-baseline", "latest", skip])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_collector_reuses_same_rows_and_lift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _job(jobs / "demo-codex-with", 0.75)
    _job(jobs / "demo-codex-without", 0.25)
    fresh = collector.collect_harbor_results(
        "demo", ["codex"], tmp_path / "fresh", jobs, expected_case_ids=["case-001"]
    )

    def guard(name):
        original = getattr(collector, name)

        def checked(*args, **kwargs):
            assert "without" not in str(args), name
            return original(*args, **kwargs)

        monkeypatch.setattr(collector, name, checked)

    for name in (
        "_staged_task_entry_id_map",
        "_extract_rewards",
        "_can_preserve_partial_rewards",
        "_extract_agent_runtime_failures",
        "_extract_trial_failures",
        "_observe_private_trials",
        "_recovered_timeout_attempts",
        "_recovered_job_failure",
    ):
        guard(name)
    reused = collector.collect_harbor_results(
        "demo",
        ["codex"],
        tmp_path / "reuse",
        jobs,
        expected_case_ids=["case-001"],
        reused_baseline_dir=tmp_path / "fresh",
    )
    for key in ("without_skill", "lift", "custom_lift", "pass_at_k", "security_attribution", "conditions"):
        assert reused["agents"]["codex"][key] == fresh["agents"]["codex"][key]
    assert reused["agents"]["codex"]["execution_counts"] == {
        "fresh": {"with_skill": 1, "without_skill": 0},
        "reused": {"without_skill": 1},
    }
    summary = json.loads((tmp_path / "reuse/codex/without-skill/summary.json").read_text())
    assert summary["reused"] is True
    assert baseline_store.load_baseline_rows(tmp_path / "reuse", "codex")


@pytest.mark.parametrize(
    "baseline", [{"mode": "fresh"}, {"mode": "reused", "source_run_id": "source", "source_evaluated_at": "time"}]
)
def test_generated_benchmark_reports_baseline(tmp_path: Path, baseline: dict) -> None:
    jobs, output = tmp_path / "jobs", tmp_path / "output"
    _job(jobs / "demo-codex-with", 0.75)
    _job(jobs / "demo-codex-without", 0.25)
    result = collector.collect_harbor_results("demo", ["codex"], output, jobs, expected_case_ids=["case-001"])
    _write(output / "run_config.json", {"baseline": baseline})
    report_result = agent_eval_result_from_directory(
        tmp_path / "demo", output, engine_result=result, use_llm_judge=False
    )
    assert report_result is not None
    report = BenchmarkReporter(skill_name="demo", report_scope="tier3").render(report_result)
    assert baseline_store.baseline_description(baseline) in report


def test_reused_rows_with_conflicting_policy_fail_collection(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _job(jobs / "demo-codex-without", 0.25, grading_mode="custom_only")
    collector.collect_harbor_results(
        "demo",
        ["codex"],
        tmp_path / "source",
        jobs,
        skip_with_skill=True,
        expected_case_ids=["case-001"],
        grading_mode="custom_only",
    )
    _job(jobs / "demo-codex-with", 0.75)
    result = collector.collect_harbor_results(
        "demo",
        ["codex"],
        tmp_path / "reuse",
        jobs,
        expected_case_ids=["case-001"],
        reused_baseline_dir=tmp_path / "source",
    )
    agent = result["agents"]["codex"]
    assert agent["execution_status"] == "failed"
    assert "Conflicting score policies across conditions" in str(agent["execution_errors"])
    assert not agent["lift"]


def test_unobserved_policy_defers_to_collection() -> None:
    current, source = _identity(), _identity()
    source["score_policy"] = CUSTOM_SCORE_POLICY
    assert baseline_store.comparison_mismatches(current, source) == []


def test_execution_counts_include_unscoreable_trials(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _job(jobs / "demo-codex-with", 0.75)
    _job(jobs / "demo-codex-without", 0.25)
    reward = jobs / "demo-codex-with/case-001__attempt/verifier/reward.json"
    _write(reward, {"entry_id": "case-001", "evaluation_status": "failed", "error": "judge unavailable"})
    result = collector.collect_harbor_results(
        "demo", ["codex"], tmp_path / "output", jobs, expected_case_ids=["case-001"]
    )
    agent = result["agents"]["codex"]
    assert agent["execution_status"] == "failed"
    assert agent["execution_counts"]["fresh"] == {"with_skill": 1, "without_skill": 1}


@pytest.mark.parametrize("skip", ["skip_baseline", "skip_with_skill"])
def test_runner_mutual_exclusion_emits_configuration_failure(tmp_path: Path, skip: str) -> None:
    events = []
    result = runner._run_harbor_eval_impl(
        tmp_path,
        ["opencode"],
        reuse_baseline="latest",
        **{skip: True},
        progress_reporter=SimpleNamespace(emit=events.append),
    )
    assert result["error"]
    assert [(event.stage, event.state) for event in events] == [("configuration", "failed")]


def test_latest_empty_falls_back_and_explicit_missing_rejects(tmp_path: Path) -> None:
    result = baseline_store.select_baseline(tmp_path, "latest", ["codex"], _identity())
    assert result["mode"] == "fresh"
    assert result["rejected_candidates"] == []
    with pytest.raises(ValueError, match="missing or corrupt"):
        baseline_store.select_baseline(tmp_path, "missing", ["codex"], _identity())
    with pytest.raises(ValueError, match="run id"):
        baseline_store.select_baseline(tmp_path, "../outside", ["codex"], _identity())


@pytest.mark.parametrize("grading_mode", ["default", "custom_only"])
def test_runner_fresh_reused_and_incompatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, grading_mode: str
) -> None:
    skill = tmp_path / "demo"
    _write(skill / "evals/evals.json", [])
    provider = ProviderConfig(
        provider="openai",
        model="test-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "load_evals_config", lambda _path: ({"harbor": {"task_source": "evals_json"}}, None))
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals/evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: runtime_preflight.ModelProbeResult(True, selected.provider, selected.model, "available"),
    )
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")

    def emit(_skill, target, **_kwargs):
        task = target / "case-001"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            'schema_version = "1.3"\n[task]\nname = "case-001"\n[metadata]\nentry_id = "case-001"\n'
        )
        return [task]

    launched = []

    def launch(**kwargs):
        launched.append(kwargs["baseline"] is not None)
        for condition, suffix, score in (("with_skill", "with", 0.75), ("baseline", "without", 0.25)):
            if kwargs[condition] is not None:
                _job(kwargs["jobs_dir"] / f"demo-opencode-{suffix}", score, grading_mode=grading_mode)
        return []

    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    monkeypatch.setattr(runner, "_run_agent_pair", launch)
    kwargs = {
        "output_dir": tmp_path / "results",
        "env_mode": "docker",
        "agent_runtime_preflight": False,
        "agent_models": {"opencode": "openai/test-model"},
        "grading_mode": grading_mode,
    }
    fresh = runner.run_harbor_eval(skill, ["opencode"], **kwargs)
    assert "error" not in fresh, fresh
    reused = runner.run_harbor_eval(skill, ["opencode"], reuse_baseline=fresh["run_id"], **kwargs)
    assert "error" not in reused, reused
    assert launched == [True, False]
    assert reused["baseline"]["mode"] == "reused"
    assert fresh["comparison_identity"]["score_policy"] == reused["comparison_identity"]["score_policy"]
    assert fresh["agents"]["opencode"]["lift"] == reused["agents"]["opencode"]["lift"]
    assert (
        baseline_store.select_baseline(tmp_path / "results", "latest", ["opencode"], fresh["comparison_identity"])[
            "source_run_id"
        ]
        == fresh["run_id"]
    )
    incompatible = runner.run_harbor_eval(skill, ["opencode"], reuse_baseline=fresh["run_id"], n_attempts=3, **kwargs)
    assert incompatible["error"]
    assert "attempt" in str(incompatible["error"])
    assert launched == [True, False]

    source = Path(fresh["run_dir"])
    identity = fresh["comparison_identity"]
    result_path = source / "result.json"
    original_result = result_path.read_text()
    changed = json.loads(original_result)
    changed["baseline"] = {"mode": "reused"}
    result_path.write_text(json.dumps(changed))
    assert any("reused" in reason for reason in baseline_store.baseline_eligibility(source, ["opencode"], identity)[0])
    result_path.write_text(original_result)
    assert baseline_store.baseline_eligibility(source, ["missing-agent"], identity)[0]
    reward = next((source / "opencode/without-skill/trials").glob("*/reward.json"))
    original_reward = reward.read_text()
    for invalid in ("{", "{}"):
        reward.write_text(invalid)
        assert baseline_store.baseline_eligibility(source, ["opencode"], identity)[0]
    partial = json.loads(original_reward)
    partial["entry_id"] = "unexpected-case"
    reward.write_text(json.dumps(partial))
    assert any(
        "coverage" in reason for reason in baseline_store.baseline_eligibility(source, ["opencode"], identity)[0]
    )
    reward.write_text(original_reward)
    summary = source / "opencode/without-skill/summary.json"
    summary.write_text(json.dumps({"execution_status": "failed"}))
    assert any("failed" in reason for reason in baseline_store.baseline_eligibility(source, ["opencode"], identity)[0])
    fallback = baseline_store.select_baseline(tmp_path / "results", "latest", ["opencode"], identity)
    assert fallback["mode"] == "fresh"
    assert {candidate["run_id"] for candidate in fallback["rejected_candidates"]} == {fresh["run_id"], reused["run_id"]}
    reward.unlink()
    assert baseline_store.baseline_eligibility(source, ["opencode"], identity)[0]


def test_usage_separates_missing_and_measured_counters(tmp_path: Path) -> None:
    assert all(value is None for value in baseline_store.measured_condition_usage(tmp_path).values())
    trial = tmp_path / "trials/one"
    _write(trial / "reward.json", {"trial_id": "one"})
    _write(
        trial / "trajectory.json",
        {"final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 4, "total_cost_usd": 0.05}},
    )
    usage = baseline_store.measured_condition_usage(tmp_path)
    assert usage["total_prompt_tokens"] == 10
    assert usage["total_completion_tokens"] == 4
    assert usage["total_cost_usd"] == 0.05
    assert usage["total_cached_tokens"] is None


def test_provenance_description() -> None:
    assert (
        baseline_store.baseline_description({"mode": "reused", "source_run_id": "run", "source_evaluated_at": "time"})
        == "Baseline: reused from run run (time)"
    )
    assert (
        baseline_store.baseline_description({"mode": "fresh", "reuse_reason": "no match"})
        == "Baseline: fresh (reuse not possible: no match)"
    )
