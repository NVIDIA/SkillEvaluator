# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil

import pytest
from tests.tier3.test_baseline_timeouts import SCORES, _arm, _attempt_job

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.tier3.harbor import collector, runner
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.harbor.run_budget import BudgetController


def _collect(tmp_path, agents, budget, **kwargs):
    options = {"n_attempts": 2, "expected_trials": None, "expected_cases": 1, "expected_case_ids": ["case"]}
    options.update(kwargs)
    return collector.collect_harbor_results(
        skill_name="demo",
        agents=agents,
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        case_id_by_task_selector={"case": "case"},
        budget_not_run=budget.not_run,
        **options,
    )


@pytest.mark.parametrize("admit_first", [False, True], ids=["zero-cap", "second-agent-denied"])
def test_stop_on_pass_fully_budget_denied_arms(monkeypatch, tmp_path, admit_first):
    def harbor(*, job_name, jobs_dir, **kwargs):
        source = _attempt_job(tmp_path / "physical", job_name, "with", "case", 1, "pass")
        shutil.copytree(source, jobs_dir / job_name)
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", harbor)
    expected = []
    validate = collector.validate_harbor_job_result

    def record_validation(path, **kwargs):
        if "expected_trials" in kwargs:
            expected.append(kwargs["expected_trials"])
        return validate(path, **kwargs)

    monkeypatch.setattr(collector, "validate_harbor_job_result", record_validation)
    agents = ["first", "denied"] if admit_first else ["denied"]
    budget = BudgetController(max_attempts_total=2 if admit_first else 0, planned=4 * len(agents))
    for agent in agents:
        errors = runner._run_budgeted_variants(
            skill_name="demo",
            agent=agent,
            jobs=[("with", tmp_path / "with", None), ("without", tmp_path / "without", None)],
            task_names=["case"],
            env_mode="docker",
            model="model",
            jobs_dir=tmp_path / "jobs",
            run_env={},
            n_attempts=2,
            pass_threshold=0.5,
            stop_on_pass=True,
            timeout_multiplier=1,
            override_cpus=None,
            override_memory_mb=None,
            override_storage_mb=None,
            budget=budget,
            n_concurrent=1,
        )
        assert not any(errors.values())
    results = _collect(tmp_path, agents, budget, stop_on_pass=True)
    for arm in results["agents"]["denied"]["conditions"].values():
        assert arm["execution_status"] == "skipped"
        assert arm["skip_reason"] == "budget_exhausted"
        assert arm["expected_attempts"] == arm["scored_attempts"] == 0
    assert not results["execution_errors"]
    assert expected == ([None, None] if admit_first else [])
    payload = build_agent_eval_payload(
        "demo", load_agent_data(tmp_path / "results"), run_config={"budget": budget.metadata()}
    )
    assert payload["verdict"] == "incomplete"
    if admit_first:
        assert results["agents"]["first"]["execution_status"] == "succeeded"


def test_reused_baseline_is_not_the_score_of_a_budget_denied_run(tmp_path):
    source = tmp_path / "source"
    for arm in ("with", "without"):
        _arm(source / "jobs", "agent", arm, [("case", "scored")], scores=dict.fromkeys(SCORES, 0.8))
    _collect(source, ["agent"], BudgetController(), n_attempts=1, expected_trials=1)
    budget = BudgetController(max_attempts_total=0, planned=1)
    assert not budget.admit(agent="agent", arm="with", case_id="case", attempt=1)
    _collect(tmp_path, ["agent"], budget, n_attempts=1, expected_trials=1, reused_baseline_dir=source / "results")
    payload = build_agent_eval_payload(
        "demo", load_agent_data(tmp_path / "results"), run_config={"budget": budget.metadata()}
    )
    assert payload["mode"] != "baseline_only"
    assert payload["overall_score"] is None
    assert payload["verdict"] == "incomplete"
    assert payload["agents"]["agent"]["baseline"] == pytest.approx(0.8)


@pytest.mark.parametrize("reason", ["execution_failed", "budget_exhausted", None])
def test_only_budget_or_legacy_omissions_reduce_expected_attempts(tmp_path, reason):
    for arm in ("with", "without"):
        _arm(tmp_path / "jobs", "agent", arm, [("case", "scored")])
    budget = BudgetController()
    for arm in ("with", "without"):
        row = {"agent": "agent", "arm": arm, "case_id": "case", "attempt": 2, "status": "not_run"}
        if reason is not None:
            row["reason"] = reason
        budget.not_run.append(row)
    results = _collect(tmp_path, ["agent"], budget, expected_trials=2)
    assert results["execution_status"] == ("failed" if reason == "execution_failed" else "succeeded")
    condition = results["agents"]["agent"]["conditions"]["with_skill"]
    assert condition["expected_attempts"] == (2 if reason == "execution_failed" else 1)
    if reason == "execution_failed":
        assert any("Missing scored attempts" in error for error in results["execution_errors"])


@pytest.mark.parametrize("attempts", [[1], [1, 1], [1, 3], [0, 2]])
def test_full_budget_denial_requires_every_planned_slot(tmp_path, attempts):
    budget = BudgetController()
    budget.not_run.extend(
        {
            "agent": "agent",
            "arm": arm,
            "case_id": "case",
            "attempt": attempt,
            "status": "not_run",
            "reason": "budget_exhausted",
        }
        for arm in ("with", "without")
        for attempt in attempts
    )
    results = _collect(tmp_path, ["agent"], budget, stop_on_pass=True)
    assert results["execution_status"] == "failed"
    assert all("skip_reason" not in arm for arm in results["agents"]["agent"]["conditions"].values())


def test_full_budget_denial_can_use_the_planned_case_count(tmp_path):
    budget = BudgetController(max_attempts_total=0, planned=4)
    for arm in ("with", "without"):
        for attempt in (1, 2):
            assert not budget.admit(agent="agent", arm=arm, case_id="case", attempt=attempt)
    results = _collect(tmp_path, ["agent"], budget, stop_on_pass=True, expected_case_ids=None)
    assert results["agents"]["agent"]["execution_status"] == "skipped"
    assert not results["execution_errors"]
