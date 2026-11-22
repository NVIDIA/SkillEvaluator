# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run budget must not change when ``--stop-on-pass`` stops an arm."""

import shutil
import threading
from pathlib import Path

import pytest
from tests.tier3.test_baseline_timeouts import _attempt_job

from skillevaluator.tier3.harbor import collector, runner
from skillevaluator.tier3.harbor.run_budget import BudgetController

AGENT = "agent"
N_ATTEMPTS = 3


def _run(monkeypatch, tmp_path, first_pass, *, budget, n_concurrent=1):
    """Run the cases through the real runner and collector; return (starts, results).

    ``first_pass`` maps a case to the attempt each arm first passes on, or ``None``
    if it never passes.
    """
    cases = list(first_pass)
    starts: list[tuple[str, str, int]] = []

    def harbor(*, job_name, jobs_dir, dataset, include_task_names, **_kwargs):
        variant = Path(dataset).name
        (case,) = include_task_names
        attempt = int(job_name.rsplit("-attempt", 1)[1])
        starts.append((case, variant, attempt))
        passes_on = first_pass[case][variant]
        outcome = "pass" if passes_on is not None and attempt >= passes_on else "fail"
        source = _attempt_job(tmp_path / "physical", job_name, variant, case, attempt, outcome)
        shutil.copytree(source, jobs_dir / job_name)
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", harbor)
    errors = runner._run_agent_pair(
        skill_name="demo",
        agent=AGENT,
        model="model",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=N_ATTEMPTS,
        n_concurrent=n_concurrent,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=len(cases) * N_ATTEMPTS,
        stop_on_pass=True,
        pass_threshold=0.5,
        task_names=cases,
        budget_controller=budget,
    )
    assert errors == []
    results = collector.collect_harbor_results(
        skill_name="demo",
        agents=[AGENT],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        n_attempts=N_ATTEMPTS,
        stop_on_pass=True,
        expected_trials=None,
        expected_cases=len(cases),
        expected_case_ids=cases,
        case_id_by_task_selector={case: case for case in cases},
        budget_not_run=budget.not_run if budget is not None else [],
    )
    return starts, results


def _assert_each_arm_stopped_on_its_first_pass(starts, first_pass):
    for case, arms in first_pass.items():
        for variant, passes_on in arms.items():
            attempts = sorted(attempt for started, arm, attempt in starts if (started, arm) == (case, variant))
            assert attempts == list(range(1, (passes_on or N_ATTEMPTS) + 1)), (case, variant)


OUTCOMES = [
    pytest.param({"with": 1, "without": 2}, id="baseline-passes-later"),
    pytest.param({"with": 1, "without": None}, id="baseline-never-passes"),
    pytest.param({"with": 3, "without": 1}, id="with-skill-passes-later"),
]


@pytest.mark.parametrize("budgeted", [False, True], ids=["no-budget", "budget"])
@pytest.mark.parametrize("outcomes", OUTCOMES)
def test_each_arm_stops_after_its_own_first_pass(monkeypatch, tmp_path, outcomes, budgeted):
    # The cap is far above the plan, so the budget denies nothing.
    budget = BudgetController(max_attempts_total=100, planned=2 * N_ATTEMPTS) if budgeted else None
    first_pass = {"case": outcomes}
    starts, results = _run(monkeypatch, tmp_path, first_pass, budget=budget)

    assert not results["execution_errors"]
    assert results["agents"][AGENT]["execution_status"] == "succeeded"
    _assert_each_arm_stopped_on_its_first_pass(starts, first_pass)
    if budget is not None:
        assert budget.not_run == []


def test_concurrent_cases_stop_independently_under_a_budget(monkeypatch, tmp_path):
    first_pass = {
        "case-a": {"with": 1, "without": 3},
        "case-b": {"with": 2, "without": 1},
        "case-c": {"with": 1, "without": 1},
    }
    budget = BudgetController(max_attempts_total=100, planned=2 * N_ATTEMPTS * len(first_pass))
    starts, results = _run(monkeypatch, tmp_path, first_pass, budget=budget, n_concurrent=3)

    assert not results["execution_errors"]
    assert results["agents"][AGENT]["execution_status"] == "succeeded"
    _assert_each_arm_stopped_on_its_first_pass(starts, first_pass)
    assert budget.not_run == []
    assert budget.metadata()["started_attempts"] == len(starts) == 9


def test_budget_denial_after_one_arm_passed_leaves_that_arm_complete(monkeypatch, tmp_path):
    # The budget fits two attempts, the first pair. The with-skill arm passes
    # there, so only the baseline still needs attempts when the budget runs out.
    budget = BudgetController(max_attempts_total=2, planned=2 * N_ATTEMPTS)
    starts, results = _run(monkeypatch, tmp_path, {"case": {"with": 1, "without": None}}, budget=budget)

    assert starts == [("case", "with", 1), ("case", "without", 1)]
    assert [(row["arm"], row["attempt"]) for row in budget.not_run] == [("without", 2), ("without", 3)]
    assert not results["execution_errors"]
    conditions = results["agents"][AGENT]["conditions"]
    assert conditions["with_skill"]["expected_attempts"] == conditions["with_skill"]["scored_attempts"] == 1
    assert conditions["without_skill"]["scored_attempts"] == 1


def test_single_arm_takes_the_last_slot_after_a_pair_is_denied(monkeypatch, tmp_path):
    # Five attempts fit and the first attempt of both cases uses four. After
    # that case-b still needs both arms and case-a only its baseline. The
    # requests are ordered so the pair is denied before the single arm asks.
    first_pass = {
        "case-a": {"with": 1, "without": None},
        "case-b": {"with": None, "without": None},
    }
    budget = BudgetController(max_attempts_total=5, planned=2 * N_ATTEMPTS * len(first_pass))
    admit_many = budget.admit_many
    first_attempts_started = threading.Event()
    pair_denied = threading.Event()

    def pair_before_single_arm(attempts):
        if attempts[0]["attempt"] > 1:
            assert first_attempts_started.wait(timeout=30)
        if len(attempts) == 1:
            assert pair_denied.wait(timeout=30)
        admitted = admit_many(attempts)
        if budget.metadata()["started_attempts"] == 4:
            first_attempts_started.set()
        if len(attempts) == 2 and not admitted:
            pair_denied.set()
        return admitted

    monkeypatch.setattr(budget, "admit_many", pair_before_single_arm)
    starts, results = _run(monkeypatch, tmp_path, first_pass, budget=budget, n_concurrent=2)

    assert sorted(starts) == [
        ("case-a", "with", 1),
        ("case-a", "without", 1),
        ("case-a", "without", 2),
        ("case-b", "with", 1),
        ("case-b", "without", 1),
    ]
    assert sorted((row["case_id"], row["arm"], row["attempt"]) for row in budget.not_run) == [
        ("case-a", "without", 3),
        ("case-b", "with", 2),
        ("case-b", "with", 3),
        ("case-b", "without", 2),
        ("case-b", "without", 3),
    ]
    assert not results["execution_errors"]
