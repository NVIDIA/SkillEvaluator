# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation eligibility must survive canonical and retained dashboard loading."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting.dashboard_data import comparison_rows, load_dashboard_path, parse_dashboard_report
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def _report(status: str | None = "never_activated") -> dict:
    agent = {
        "name": "opencode",
        "model": "model-a",
        "execution_status": "succeeded",
        "with_skill": 0.32,
        "baseline": 0.8,
        "num_trials": 1,
        "num_trials_baseline": 1,
        "conditions": {
            arm: {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
            for arm in ("with_skill", "without_skill")
        },
        "trials": [
            {
                "entry_id": "case-a",
                "trial_id": "case-a__random",
                "tokens": {"prompt": 20, "completion": 2, "cached": 0},
                "agent_seconds": 3,
                "cost_usd": 0.02,
            }
        ],
        "trials_baseline": [
            {
                "entry_id": "case-a",
                "trial_id": "case-a__other",
                "tokens": {"prompt": 10, "completion": 1, "cached": 0},
                "agent_seconds": 5,
                "cost_usd": 0.01,
            }
        ],
        "pass_at_k": {"with_skill": {"rate": 0.0}, "without_skill": {"rate": 1.0}},
    }
    if status is not None:
        agent["skill_activation"] = {"status": status}
    return {"skill_name": "calculator", "run_id": "run-a", "agents": {"opencode": agent}}


def test_never_activated_comparison_keeps_diagnostics_without_deltas() -> None:
    rows = parse_dashboard_report(_report(), "report.json").rows
    comparison = comparison_rows(rows)[0]
    assert comparison["comparison_status"] == "inconclusive"
    assert comparison["comparison_reason"] == "Inconclusive: skill never activated in with-skill trials"
    assert comparison["with_score"] == 0.32
    assert comparison["without_score"] == 0.8
    assert comparison["with_pass_rate"] == 0.0
    assert comparison["without_pass_rate"] == 1.0
    assert comparison["with_total_tokens"] == 22
    assert comparison["without_total_tokens"] == 11
    assert comparison["with_duration_seconds"] == 3
    assert comparison["without_duration_seconds"] == 5
    assert comparison["with_cost_usd"] == 0.02
    assert comparison["without_cost_usd"] == 0.01
    assert all(value is None for key, value in comparison.items() if key.endswith(("_delta", "_percent")))
    assert all(row["_complete"] and row["status"] == "succeeded" for row in rows)


@pytest.mark.parametrize("status", [None, "unknown", "activated", "not_applicable"])
def test_legacy_unknown_activated_and_negative_controls_remain_comparable(status: str | None) -> None:
    comparison = comparison_rows(parse_dashboard_report(_report(status), "report.json").rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["comparison_reason"] == ""
    assert comparison["score_delta"] == pytest.approx(-0.48)
    assert comparison["pass_rate_delta"] == -1
    assert comparison["total_tokens_delta"] == 11


def test_mixed_agents_keep_eligible_comparisons_separate() -> None:
    report = _report()
    report["agents"]["activated"] = copy.deepcopy(_report("activated")["agents"]["opencode"])
    report["agents"]["activated"]["with_skill"] = 0.9
    # A top-level best-agent census must not disqualify other agents.
    report["skill_activation"] = {"status": "never_activated"}
    comparisons = {row["_agent_key"]: row for row in comparison_rows(parse_dashboard_report(report, "r").rows)}
    assert comparisons["opencode"]["score_delta"] is None
    assert comparisons["opencode"]["comparison_status"] == "inconclusive"
    assert comparisons["activated"]["score_delta"] == pytest.approx(0.1)
    assert comparisons["activated"]["comparison_status"] == "comparable"


def test_missing_baseline_takes_precedence_over_activation() -> None:
    report = _report()
    agent = report["agents"]["opencode"]
    agent.pop("baseline")
    agent.pop("trials_baseline")
    agent["conditions"].pop("without_skill")
    comparison = comparison_rows(parse_dashboard_report(report, "r").rows)[0]
    assert comparison["comparison_status"] == "missing condition"
    assert comparison["comparison_reason"] == "no without-skill arm for this run and configuration"
    assert comparison["score_delta"] is None


@pytest.mark.parametrize("condition", ["with_skill", "without_skill"])
@pytest.mark.parametrize("failure", ["failed", "incomplete"])
def test_execution_and_coverage_failures_take_precedence_over_activation(condition: str, failure: str) -> None:
    report = _report()
    details = report["agents"]["opencode"]["conditions"][condition]
    if failure == "failed":
        details["execution_status"] = "failed"
    else:
        details["expected_attempts"] = 2
    comparison = comparison_rows(parse_dashboard_report(report, "r").rows)[0]
    label = condition.replace("_", "-")
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["comparison_reason"] == (
        f"{label} arm status failed" if failure == "failed" else f"{label} trial coverage incomplete"
    )
    assert comparison["score_delta"] is None


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _retained_run(path: Path, status: str, *, persist_census: bool) -> None:
    metrics = dict.fromkeys(DEFAULT_METRICS, 0.8)
    _write(path / "run_config.json", {"agents": {"opencode": {"agent": "opencode", "model": "model-a"}}})
    _write(path / "result.json", {"skill_name": "calculator", "run_id": path.name})
    for variant, prompt_tokens in (("with-skill", 20), ("without-skill", 10)):
        condition = path / "opencode" / variant
        summary = {
            "model": "model-a",
            "scores": metrics,
            "metrics": list(DEFAULT_METRICS),
            "num_trials": 1,
            "execution_status": "succeeded",
            "expected_attempts": 1,
            "scored_attempts": 1,
            "pass_at_k": {"k": 1, "rate": 1.0},
        }
        if persist_census and variant == "with-skill":
            summary["skill_activation"] = {"status": status}
        _write(condition / "summary.json", summary)
        details = {
            "skill_execution": {
                "should_trigger": status != "not_applicable",
                "activation": {"passed": status == "activated", "score": float(status == "activated")},
            }
        }
        reward = {**metrics, "entry_id": "case-a", "trial_id": "case-a__random", "details": details}
        if status == "unknown":
            # Custom-only rewards cannot establish activation eligibility.
            reward = {"custom_score": 0.8, "entry_id": "case-a", "trial_id": "case-a__random"}
        trial = condition / "trials" / "case-a__random"
        _write(trial / "reward.json", reward)
        _write(
            trial / "trajectory.json",
            {"steps": [], "final_metrics": {"total_prompt_tokens": prompt_tokens, "total_completion_tokens": 2}},
        )
        _write(
            trial / "result.json",
            {
                "agent_execution": {
                    "started_at": "2026-10-18T00:00:00Z",
                    "finished_at": "2026-10-18T00:00:03Z",
                }
            },
        )


@pytest.mark.parametrize("status", ["never_activated", "activated", "not_applicable", "unknown"])
@pytest.mark.parametrize("persist_census", [False, True])
def test_retained_and_exported_reports_preserve_activation_eligibility(
    tmp_path: Path, status: str, persist_census: bool
) -> None:
    run = tmp_path / "run"
    _retained_run(run, status, persist_census=persist_census)
    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    canonical = result.metadata["agent_eval"]
    assert canonical["agents"]["opencode"]["skill_activation"]["status"] == status
    report_path = tmp_path / "report.json"
    _write(report_path, canonical)
    for source in (run, run / "result.json", report_path):
        comparison = comparison_rows(load_dashboard_path(source).rows)[0]
        if status == "never_activated":
            assert canonical["agents"]["opencode"]["lift"] is None
            assert comparison["comparison_status"] == "inconclusive"
            assert "skill never activated" in comparison["comparison_reason"]
            assert comparison["score_delta"] is None
            assert comparison["pass_rate_delta"] is None
            assert comparison["total_tokens_delta"] is None
        else:
            assert comparison["comparison_status"] == "comparable"
            assert comparison["score_delta"] == pytest.approx(0)
            assert comparison["total_tokens_delta"] == 10
        assert comparison["with_score"] is not None
        assert comparison["without_score"] is not None


def test_dashboard_app_displays_mixed_activation_without_reconstructing_lift(tmp_path: Path) -> None:
    st = pytest.importorskip("streamlit")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from streamlit.testing.v1 import AppTest

    report = _report()
    eligible = copy.deepcopy(_report("activated")["agents"]["opencode"])
    eligible.update({"name": "claude-code", "model": "model-b", "with_skill": 0.9})
    eligible["pass_at_k"] = {"with_skill": {"rate": 1.0}, "without_skill": {"rate": 0.0}}
    report["agents"]["claude-code"] = eligible
    path = tmp_path / "mixed-report.json"
    _write(path, report)
    st.cache_data.clear()
    try:
        app = AppTest.from_string(
            f"from skillevaluator.reporting.dashboard_app import main\nmain([{str(path)!r}])\n",
            default_timeout=60,
        ).run()
        assert not app.exception
        performance = app.dataframe[0].value
        assert performance["score"].tolist() == [0.32, 0.8, 0.9, 0.8]
        for view in ("Quality", "All metrics"):
            app.selectbox(key="comparison_metrics").set_value(view).run()
            assert not app.exception
            comparison = app.dataframe[1].value.set_index("model")
            zero = comparison.loc["model-a"]
            assert zero["comparison_status"] == "inconclusive"
            assert zero["comparison_reason"] == "Inconclusive: skill never activated in with-skill trials"
            assert pd.isna(zero["score_delta"])
            assert pd.isna(zero["pass_rate_delta"])
            assert zero["with_score"] == 0.32
            assert zero["without_score"] == 0.8
            activated = comparison.loc["model-b"]
            assert activated["comparison_status"] == "comparable"
            assert activated["score_delta"] == pytest.approx(0.1)
            assert activated["pass_rate_delta"] == 1
            if view == "All metrics":
                assert pd.isna(zero["total_tokens_delta"])
                assert zero["with_total_tokens"] == 22
                assert activated["total_tokens_delta"] == 11
        assert len(app.get("download_button")) == 2
    finally:
        st.cache_data.clear()
