# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import EvaluationService
from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.progress import NullProgressReporter
from skillevaluator.tier3.harbor.run_budget import BudgetController


@pytest.fixture
def denied_result():
    budget = BudgetController(max_attempts_total=0, planned=4)
    for arm in ("with", "without"):
        for attempt in (1, 2):
            assert not budget.admit(agent="opencode", arm=arm, case_id="case", attempt=attempt)
    return {"execution_status": "skipped", "execution_errors": [], "budget": budget.metadata()}


def test_budget_denied_run_is_not_an_execution_failure(denied_result):
    assert EvaluationService.failure_reason(denied_result) is None


def test_budget_denied_run_finishes_progress_without_failure(denied_result):
    class Reporter(NullProgressReporter):
        def emit(self, event):
            events.append(event)

    events = []
    lifecycle = runner._RunProgressLifecycle(Reporter())
    lifecycle.finish_result(denied_result)
    assert events[-1].stage == "run-finished"
    assert events[-1].state == "complete"


def test_budget_denied_cli_exits_successfully(monkeypatch, denied_result):
    monkeypatch.setattr(EvaluationService, "evaluate", lambda *_args, **_kwargs: denied_result)
    skill = Path(__file__).parents[1] / "fixtures" / "skills" / "simple"
    result = CliRunner().invoke(cli, ["tier3", "evaluate", str(skill), "--progress", "off"])
    assert result.exit_code == 0, result.output
    assert "RUN QUALITY VERDICT: INCOMPLETE" in result.output
    assert "execution status is skipped" not in result.output


def test_budget_denied_report_is_incomplete_without_execution_error(denied_result):
    condition = {
        "execution_status": "skipped",
        "skip_reason": "budget_exhausted",
        "expected_attempts": 0,
        "scored_attempts": 0,
    }
    payload = build_agent_eval_payload(
        "skill",
        {"opencode": {**denied_result, "conditions": {"with_skill": condition, "without_skill": condition}}},
        run_config={"budget": denied_result["budget"]},
        use_llm_judge=False,
    )
    validation = _validation_result_from_payload(payload)
    assert not validation.errors
    assert payload["verdict"] == "incomplete"
    assert payload["overall_score"] is None
    assert payload["overall_lift"] is None


@pytest.mark.parametrize("change", ["error", "failed", "started", "reason", "partial", "empty", "missing"])
def test_other_incomplete_runs_still_fail_closed(denied_result, change):
    if change == "error":
        denied_result["execution_errors"] = ["real execution failure"]
    elif change == "failed":
        denied_result["execution_status"] = "failed"
    elif change == "started":
        denied_result["budget"]["started_attempts"] = 1
    elif change == "reason":
        denied_result["budget"]["not_run"][0]["reason"] = "execution_failed"
    elif change == "partial":
        denied_result["budget"]["planned_attempts"] = 5
    elif change == "empty":
        denied_result["budget"]["not_run"] = []
    else:
        denied_result.pop("budget")
    assert EvaluationService.failure_reason(denied_result) is not None
