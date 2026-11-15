# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
from rich.console import Console

from skillevaluator.evaluation import tier3_report
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter
from skillevaluator.tier3.harbor import report_data
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.run_budget import BudgetController, budget_incomplete_reason, budget_line
from skillevaluator.tier3.result_display import render_evaluation_result


def _agents():
    scores = dict.fromkeys(DEFAULT_METRICS, 1.0)
    baseline = dict.fromkeys(DEFAULT_METRICS, 0.5)
    condition = {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
    return {
        "agent": {
            "model": "test-model",
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 2,
            "scored_attempts": 2,
            "with_skill": scores,
            "without_skill": baseline,
            "rewards": [{**scores, "entry_id": "case", "trial_id": "case__attempt1"}],
            "rewards_baseline": [{**baseline, "entry_id": "case", "trial_id": "case__attempt1"}],
            "conditions": {"with_skill": dict(condition), "without_skill": dict(condition)},
            "num_trials": 1,
            "num_trials_baseline": 1,
        }
    }


def _budget(kind):
    budget = BudgetController(max_agent_tokens=None if kind == "no_limits" else 10, planned=4)
    budget.admit(agent="agent", arm="with", case_id="case", attempt=1)
    budget.record_usage({"total_tokens": 20, "judge_calls": 1})
    if kind == "partial":
        budget.admit(agent="agent", arm="with", case_id="case", attempt=2)
    if kind == "unverified":
        budget.record_usage({})
    return budget.metadata()


@pytest.mark.parametrize("kind", ["partial", "overrun", "unverified", "no_limits"])
def test_budget_payload_and_all_reports(kind):
    budget = _budget(kind)
    agents = _agents()
    original = copy.deepcopy(agents)
    payload = tier3_report.build_agent_eval_payload("demo", agents, run_config={"budget": budget}, use_llm_judge=False)
    assert agents == original
    assert payload["budget"] == budget
    assert payload["budget_summary"] == budget_line(budget)
    assert payload["execution_status"] == "succeeded"
    assert payload["agents"]["agent"]["with_skill"] is not None
    if kind == "partial":
        assert payload["verdict"] == payload["summary"]["verdict"] == "incomplete"
        assert payload["summary"]["verdict_reason"] == budget_incomplete_reason(budget)
        assert payload["overall_lift"] is None
    else:
        assert payload["verdict"] == "pass"

    result = tier3_report._validation_result_from_payload(payload)
    for scope in ("publication", "tier3"):
        benchmark = BenchmarkReporter(report_scope=scope, include_timestamp=False).render_all([result])
        assert budget_line(budget) in benchmark
        if kind == "partial":
            assert "> **Tier 3: INCOMPLETE**" in benchmark
            assert "completed-agent result: PASS" not in benchmark
            assert "Skill Lift:" not in benchmark
    html = HTMLReporter(report_scope="tier3", include_timestamp=False).render_all([result])
    if kind != "no_limits":
        assert f'<p class="warning">{budget_line(budget)}</p>' in html
    else:
        assert f'<p class="warning">{budget_line(budget)}</p>' not in html
    if kind == "partial":
        assert "Tier 3 quality: Incomplete" in html
        assert "cover admitted attempts only" in html
    console = Console(record=True, force_terminal=False, width=240)
    render_evaluation_result({"agents": agents, "budget": budget, "execution_status": "succeeded"}, console=console)
    terminal = console.export_text()
    if kind != "no_limits":
        assert budget_line(budget) in terminal
    if kind == "partial":
        assert "RUN QUALITY VERDICT: INCOMPLETE" in terminal
        assert "skill helps" not in terminal
        assert "OVERALL SKILL LIFT   " not in terminal
    cli_console = Console(record=True, force_terminal=False, width=240)
    CLIReporter._print_agent_eval_tables(payload, cli_console)
    assert budget_line(budget) in cli_console.export_text()


@pytest.mark.parametrize("live", [False, True, "missing_budget"])
def test_saved_top_level_budget_overrides_stale_config(monkeypatch, tmp_path, live):
    fresh = _budget("partial")
    stale = BudgetController(max_agent_tokens=10, planned=4).metadata()
    (tmp_path / "run_config.json").write_text(json.dumps({"budget": stale}))
    (tmp_path / "result.json").write_text(json.dumps({"budget": fresh}))
    monkeypatch.setattr(report_data, "load_agent_data", lambda *_args, **_kwargs: _agents())
    result = tier3_report.agent_eval_result_from_directory(
        tmp_path / "demo",
        tmp_path,
        engine_result={} if live == "missing_budget" else {"budget": fresh} if live else None,
        use_llm_judge=False,
    )
    payload = result.metadata["agent_eval"]
    assert payload["budget"] == fresh
    assert payload["summary"]["verdict"] == "incomplete"
    assert payload["budget_summary"] == budget_line(fresh)


def test_saved_complete_run_feedback_cannot_restore_skill_helps(tmp_path):
    payload = tier3_report.build_agent_eval_payload(
        "demo", _agents(), run_config={"budget": _budget("partial")}, use_llm_judge=False
    )
    tier3_report._apply_saved_feedback(
        payload,
        {"conclusions": [{"title": "Skill effect", "message": "skill helps"}]},
        {"agents": _agents()},
        tmp_path,
        tmp_path,
    )
    assert "skill helps" not in json.dumps(payload["conclusions"])
    assert payload["verdict"] == "incomplete"


@pytest.mark.parametrize("metadata", [None, {}, {"not_run": []}, {"not_run": True}, {"not_run": "invalid"}])
def test_empty_budget_has_no_incomplete_reason(metadata):
    assert budget_incomplete_reason(metadata) is None
