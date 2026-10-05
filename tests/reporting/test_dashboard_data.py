# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from skillevaluator.reporting.dashboard_data import comparison_rows, load_dashboard_path, parse_dashboard_report


def _report() -> dict:
    def trial(prompt: int, completion: int, seconds: float, cost: float) -> dict:
        return {
            "entry_id": "case-a",
            "trial_id": "case-a__random",
            "overall": 0.8,
            "tokens": {"prompt": prompt, "completion": completion, "cached": 0},
            "duration_seconds": seconds,
            "cost_usd": cost,
        }

    return {
        "schema_version": "2.0",
        "skill_name": "calculator",
        "run_id": "run-1",
        "dataset_digest": "sha256:dataset",
        "attempt_policy": {"max_attempts": 1},
        "agents": {
            "claude-code": {
                "name": "claude-code",
                "model": "model-a",
                "execution_status": "succeeded",
                "with_skill": 0.8,
                "baseline": 0.6,
                "num_trials": 1,
                "num_trials_baseline": 1,
                "conditions": {
                    arm: {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
                    for arm in ("with_skill", "without_skill")
                },
                "trials": [trial(120, 30, 8, 0.02)],
                "trials_baseline": [trial(100, 20, 10, 0.01)],
                "pass_at_k": {"with_skill": {"rate": 1.0}, "without_skill": {"rate": 0.0}},
            }
        },
    }


def test_with_without_differences_and_explicit_features() -> None:
    report = _report()
    report["dashboard_metadata"] = {"harness_features": ["rules", "memory"]}
    data = parse_dashboard_report(report, "report.json")
    assert len(data.rows) == 2
    assert data.rows[0]["features"] == "memory, rules"
    comparison = comparison_rows(data.rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["score_delta"] == pytest.approx(0.2)
    assert comparison["total_tokens_delta"] == 30
    assert comparison["total_tokens_percent"] == 25
    assert comparison["duration_seconds_delta"] == -2
    assert comparison["cost_usd_percent"] == 100


def test_authoritative_validator_results_exclude_top_level_mirror() -> None:
    first, second = _report(), _report()
    second["skill_name"] = "second-skill"
    wrapper = {"tier3": first, "results": [{"tier3": first}, {"metadata": {"agent_eval": second}}]}
    data = parse_dashboard_report(wrapper, "combined.json")
    assert len(data.rows) == 4
    assert {row["skill"] for row in data.rows} == {"calculator", "second-skill"}
    assert len(comparison_rows(data.rows)) == 2
    assert parse_dashboard_report({"skills": [wrapper]}, "combined.json").rows == data.rows


def test_partial_missing_metrics_never_become_zero_or_partial_total() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field in ("trials", "trials_baseline"):
        agent[field][0]["tokens"].pop("completion")
        agent[field][0].pop("cost_usd")
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["total_tokens"] is None
    assert data.rows[0]["cost_usd"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["prompt_tokens_delta"] == 20
    assert comparison["total_tokens_delta"] is None
    assert comparison["cost_usd_delta"] is None
    assert "cost 0/1" in data.rows[0]["coverage"]


@pytest.mark.parametrize("mutation", ["failed", "missing_trial", "truncated", "different_case", "missing_attempt"])
def test_incomplete_or_unmatched_arms_block_all_deltas(mutation: str) -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    if mutation == "failed":
        agent["conditions"]["without_skill"]["execution_status"] = "failed"
    elif mutation == "missing_trial":
        agent["trials_baseline"] = []
    elif mutation == "truncated":
        report["report_truncation"] = {"truncated": True}
    elif mutation == "different_case":
        agent["trials_baseline"][0]["entry_id"] = "case-b"
    elif mutation == "missing_attempt":
        agent["conditions"]["without_skill"]["expected_attempts"] = 2
    comparison = comparison_rows(parse_dashboard_report(report, "same.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["score_delta"] is None
    assert comparison["total_tokens_delta"] is None


def test_disjoint_runs_features_and_duplicate_conditions_cannot_pair() -> None:
    data = parse_dashboard_report(_report(), "first.json")
    rows = copy.deepcopy(data.rows)
    rows[1]["source"] = "second.json"
    assert all(row["comparison_status"] == "missing condition" for row in comparison_rows(rows))
    rows = copy.deepcopy(data.rows)
    rows[1]["features"] = "rules"
    assert all(row["comparison_status"] == "missing condition" for row in comparison_rows(rows))
    duplicate = comparison_rows([*data.rows, data.rows[0]])[0]
    assert duplicate["comparison_status"] == "ambiguous duplicate conditions"
    assert duplicate["score_delta"] is None


def test_zero_baseline_and_nonfinite_values() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["trials_baseline"][0]["cost_usd"] = 0
    agent["trials"][0]["duration_seconds"] = float("inf")
    agent["trials"][0]["tokens"]["cached"] = True
    report["dashboard_metadata"] = {"harness_features": {"invalid": float("nan")}}
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["features"] == "unknown"
    assert data.rows[0]["cached_tokens"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["cost_usd_delta"] == 0.02
    assert comparison["cost_usd_percent"] is None
    assert comparison["duration_seconds_delta"] is None


def test_malformed_status_and_aggregate_overflow_fail_closed() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["execution_status"] = ["succeeded"]
    agent["conditions"] = {}
    assert all(row["status"] == "unknown" for row in parse_dashboard_report(report, "report.json").rows)
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["trials"][0]["tokens"] = {"prompt": 1e308, "completion": 1e308}
    row = parse_dashboard_report(report, "report.json").rows[0]
    assert row["total_tokens"] is None


def test_repeated_attempts_require_recorded_identity_and_match_exact_set() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field, count_field, arm in (
        ("trials", "num_trials", "with_skill"),
        ("trials_baseline", "num_trials_baseline", "without_skill"),
    ):
        agent[field].append(copy.deepcopy(agent[field][0]))
        agent[field][0]["trial_id"] = "case-a__attempt1__abc"
        agent[field][1]["trial_id"] = "case-a__attempt2__def"
        agent[count_field] = 2
        agent["conditions"][arm].update(expected_attempts=2, scored_attempts=2)
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"] == "comparable"
    agent["trials_baseline"][1]["trial_id"] = "case-a__attempt3__def"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None
    for field in ("trials", "trials_baseline"):
        for index, trial in enumerate(agent[field]):
            trial["trial_id"] = f"case-a__unrelated-{index}"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None


def test_multistep_rewards_count_logical_attempt_once_and_sum_resources() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field, count_field in (("trials", "num_trials"), ("trials_baseline", "num_trials_baseline")):
        agent[field].append(copy.deepcopy(agent[field][0]))
        agent[count_field] = 2
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["trials"] == 1
    assert data.rows[0]["total_tokens"] == 300
    assert comparison_rows(data.rows)[0]["score_delta"] == pytest.approx(0.2)


def test_explicit_adverse_row_has_no_invented_baseline_delta() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent.update(adverse=0.2, trials_adverse=copy.deepcopy(agent["trials"]), num_trials_adverse=1)
    agent["conditions"]["adverse"] = {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
    data = parse_dashboard_report(report, "report.json")
    assert [row["condition"] for row in data.rows] == ["with_skill", "without_skill", "adverse"]
    assert data.rows[2]["score"] == 0.2
    assert len(comparison_rows(data.rows)) == 1


def test_bad_input_is_diagnostic_and_directory_skips_symlinks(tmp_path: Path) -> None:
    malformed = tmp_path / "broken.json"
    malformed.write_text("{", encoding="utf-8")
    assert load_dashboard_path(malformed).warnings
    assert load_dashboard_path(tmp_path / "missing").warnings
    good = tmp_path / "report.json"
    good.write_text(json.dumps(_report()), encoding="utf-8")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    data = load_dashboard_path(tmp_path)
    assert len(data.rows) == 2
    assert any("cannot read" in warning for warning in data.warnings)


def test_catalog_index_loads_authoritative_local_reports_with_containment(tmp_path: Path) -> None:
    reports = tmp_path / "calculator"
    reports.mkdir()
    (reports / "current.json").write_text(json.dumps(_report()))
    index = tmp_path / "catalog-summary.json"
    index.write_text(
        json.dumps(
            {
                "skills": [
                    {"report_dir": "calculator", "json_report": "current.json"},
                    {"report_dir": "..", "json_report": "outside.json"},
                ]
            }
        )
    )
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert data.warnings


def _native_run(path: Path) -> None:
    metrics = ["security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check"]
    path.mkdir()
    (path / "result.json").write_text(json.dumps({"skill_name": "native-skill", "run_id": "native-run"}))
    (path / "run_config.json").write_text(json.dumps({"agents": {"opencode": {"agent": "opencode", "model": "m"}}}))
    for variant, input_tokens in (("with-skill", 200), ("without-skill", 100)):
        condition = path / "opencode" / variant
        trial = condition / "trials" / "case-a__random"
        trial.mkdir(parents=True)
        (condition / "summary.json").write_text(
            json.dumps(
                {
                    "agent": "opencode",
                    "model": "m",
                    "scores": dict.fromkeys(metrics, 0.8),
                    "metrics": metrics,
                    "num_trials": 1,
                    "execution_status": "succeeded",
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
            )
        )
        (trial / "reward.json").write_text(
            json.dumps(
                {
                    **dict.fromkeys(metrics, 0.8),
                    "entry_id": "case-a",
                    "trial_id": "case-a__random",
                    "overall": 0.8,
                }
            )
        )
        (trial / "trajectory.json").write_text(
            json.dumps(
                {
                    "steps": [],
                    "final_metrics": {"total_prompt_tokens": input_tokens, "total_completion_tokens": 10},
                }
            )
        )
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "agent_result": {"n_input_tokens": input_tokens, "n_output_tokens": 10, "cost_usd": None},
                    "agent_execution": {"started_at": "2026-10-01T00:00:00Z", "finished_at": "2026-10-01T00:00:10Z"},
                }
            )
        )


def test_native_results_use_real_usage_and_prune_nested_mirrors(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    (run / "report.json").write_text(json.dumps(_report()))
    data = load_dashboard_path(tmp_path)
    assert len(data.rows) == 2
    assert data.rows[0]["skill"] == "native-skill"
    assert data.rows[0]["prompt_tokens"] == 200
    assert data.rows[0]["duration_seconds"] == 10
    assert data.rows[0]["cost_usd"] is None
    assert data.rows[0]["cached_tokens"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["total_tokens_delta"] == 100


def test_native_different_condition_models_cannot_pair(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    baseline_summary = run / "opencode" / "without-skill" / "summary.json"
    summary = json.loads(baseline_summary.read_text())
    summary["model"] = "different-model"
    baseline_summary.write_text(json.dumps(summary))
    comparisons = comparison_rows(load_dashboard_path(run).rows)
    assert len(comparisons) == 2
    assert all(row["comparison_status"] == "missing condition" for row in comparisons)


def test_multistep_native_cost_time_and_tokens_require_every_step(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    trial_result = run / "opencode" / "with-skill" / "trials" / "case-a__random" / "result.json"
    result = json.loads(trial_result.read_text())
    step = {
        "agent_result": {"n_input_tokens": 50, "n_output_tokens": 5, "n_cache_tokens": 0, "cost_usd": 0.1},
        "agent_execution": {"started_at": "2026-10-01T00:00:00Z", "finished_at": "2026-10-01T00:00:03Z"},
    }
    result["step_results"] = [step, copy.deepcopy(step)]
    trial_result.write_text(json.dumps(result))
    row = load_dashboard_path(run).rows[0]
    assert row["prompt_tokens"] == 100
    assert row["duration_seconds"] == 6
    assert row["cost_usd"] == pytest.approx(0.2)
    result["step_results"][1]["agent_result"].pop("n_input_tokens")
    result["step_results"][1]["agent_result"].pop("cost_usd")
    trial_result.write_text(json.dumps(result))
    row = load_dashboard_path(run).rows[0]
    assert row["prompt_tokens"] is None
    assert row["total_tokens"] is None
    assert row["cost_usd"] is None
