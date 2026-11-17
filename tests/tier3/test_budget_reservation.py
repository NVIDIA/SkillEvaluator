import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.run_budget import (
    BudgetCancelledError,
    BudgetConfigurationError,
    BudgetController,
    Price,
    PriceBook,
    budget_line,
)


def unit(name, size=2):
    return [{"agent": "a", "arm": str(index), "case_id": name, "attempt": 1} for index in range(size)]


def observe_wait(monkeypatch, budget):
    waiting = threading.Event()
    woke = threading.Event()
    original = budget._condition.wait

    def wait(timeout=None):
        waiting.set()
        result = original(timeout)
        woke.set()
        return result

    monkeypatch.setattr(budget._condition, "wait", wait)
    return waiting, woke


@pytest.mark.parametrize("dimension", ["agent_tokens", "judge_calls", "cost_usd"])
def test_reservation_projection_includes_in_flight_and_whole_unit(monkeypatch, dimension):
    kwargs = {f"max_{dimension}": 50}
    if dimension == "cost_usd":
        kwargs["price_book"] = PriceBook({"model": Price(1_000_000, 1_000_000)})
    budget = BudgetController(**kwargs)

    def usage(value):
        return {"input_tokens": value, "output_tokens": 0} if dimension != "judge_calls" else {"judge_calls": value}

    budget.record_usage(usage(10), model="model")
    active = unit("active", 4)
    assert budget.admit_many(active)
    waiting, _ = observe_wait(monkeypatch, budget)
    proposed = unit("proposed", 1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(budget.admit_many, proposed)
        try:
            assert waiting.wait(5)
            assert not future.done()
            assert budget.metadata()["not_run"] == []
            budget.record_usage(usage(0), model="model", attempt=active[0])
            assert future.result(timeout=5)
            metadata = budget.metadata()["reservation"]
            assert metadata["estimate_per_attempt"][dimension] == 10
            assert metadata["in_flight_attempts"] == 4
            assert metadata["waits"] >= 1
        finally:
            budget.cancel()
            for attempt in active + proposed:
                budget.release(attempt)


def test_wait_rechecks_usage_and_only_then_marks_not_run(monkeypatch):
    budget = BudgetController(max_agent_tokens=25)
    budget.record_usage({"total_tokens": 10})
    active = unit("active", 1)
    assert budget.admit_many(active)
    waiting, _ = observe_wait(monkeypatch, budget)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(budget.admit_many, unit("next"))
        try:
            assert waiting.wait(5)
            assert budget.metadata()["status"] == "ok"
            assert budget.metadata()["attempts_not_run"] == 0
            budget.record_usage({"total_tokens": 15}, attempt=active[0])
            assert future.result(timeout=5) is False
            assert budget.metadata()["exhausted_limit"] == "agent_tokens"
            assert budget.metadata()["attempts_not_run"] == 2
        finally:
            budget.cancel()


@pytest.mark.parametrize("dimension", ["agent_tokens", "judge_calls", "cost_usd"])
def test_idle_projection_denial_has_estimate_basis_and_honest_budget_line(dimension):
    kwargs = {f"max_{dimension}": 25}
    if dimension == "cost_usd":
        kwargs["price_book"] = PriceBook({})
    budget = BudgetController(**kwargs, planned=6)
    usage_key = {"agent_tokens": "total_tokens"}.get(dimension, dimension)
    budget.record_usage({usage_key: 10})
    assert not budget.admit_many(unit("next"))
    metadata = budget.metadata()
    assert metadata["exhausted_limit"] == ("cost" if dimension == "cost_usd" else dimension)
    assert metadata["exhausted_basis"] == "estimate"
    assert all(row["basis"] == "estimate" and row["reason"] == "budget_exhausted" for row in metadata["not_run"])
    assert metadata["reservation"]["waits"] == 0
    assert metadata["reservation"]["estimate_denials"] == 1
    assert metadata["observed_overruns"] == {}
    label = {"agent_tokens": "agent tokens", "judge_calls": "judge calls", "cost_usd": "cost"}[dimension]
    used, cap, estimate = ("$10.00", "$25.00", "$20.00") if dimension == "cost_usd" else ("10", "25", "20")
    qualifier = (
        "; agent inference only (user-supplied rates; provider-cost fallback), not billed spend"
        if dimension == "cost_usd"
        else ""
    )
    assert budget_line(metadata) == (
        f"Budget exhausted ({label} {used} of {cap}; next unit estimated {estimate} for 2 attempts)"
        f": 2 of 6 attempts not run; results are partial{qualifier}"
    )
    assert not budget.admit_many(unit("later", 1))
    assert budget.metadata()["reservation"]["estimate_denials"] == 2
    assert budget.metadata()["reservation"]["denied_unit_size"] == 2
    assert budget.metadata()["not_run"][-1]["basis"] == "estimate"


@pytest.mark.parametrize("dimension", ["agent_tokens", "judge_calls", "cost_usd"])
def test_calibration_serializes_whole_unit_and_builds_largest_estimate(monkeypatch, dimension):
    kwargs = {f"max_{dimension}": 100}
    if dimension == "cost_usd":
        kwargs["price_book"] = PriceBook({})
    budget = BudgetController(**kwargs)
    usage_key = {"agent_tokens": "total_tokens"}.get(dimension, dimension)
    active = unit("first")
    assert budget.admit_many(active)
    assert budget.metadata()["reservation"] == {
        "policy": "largest_measured_attempt",
        "estimate_per_attempt": dict.fromkeys(["agent_tokens", "judge_calls", "cost_usd"], "not measured"),
        "in_flight_attempts": 2,
        "waits": 0,
        "unestimated_admissions": 2,
        "calibration_waits": 0,
        "estimate_denials": 0,
        "denied_unit_size": 0,
    }
    waiting, _ = observe_wait(monkeypatch, budget)
    proposed = unit("next")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(budget.admit_many, proposed)
        try:
            assert waiting.wait(5)
            assert budget.metadata()["started_attempts"] == 2
            assert budget.metadata()["not_run"] == []
            waiting.clear()
            budget.record_usage({usage_key: 5}, attempt=active[0])
            assert waiting.wait(5)
            assert not future.done()
            budget.record_usage({usage_key: 10}, attempt=active[1])
            assert future.result(timeout=5)
            for attempt, value in zip(proposed, [0, 7], strict=True):
                budget.record_usage({usage_key: value}, attempt=attempt)
                budget.release(attempt)
        finally:
            budget.cancel()
    metadata = budget.metadata()
    assert metadata["finished_attempts"] == 4
    assert metadata["used"][dimension] == 22
    assert metadata["reservation"]["estimate_per_attempt"][dimension] == 10
    assert metadata["reservation"]["in_flight_attempts"] == 0
    assert metadata["reservation"]["calibration_waits"] >= 1
    assert metadata["reservation"]["waits"] >= metadata["reservation"]["calibration_waits"]
    assert metadata["reservation"]["unestimated_admissions"] == 2


def test_unknown_calibration_usage_wakes_waiter_and_stops_admission(monkeypatch):
    budget = BudgetController(max_agent_tokens=100)
    active = unit("first")
    assert budget.admit_many(active)
    waiting, _ = observe_wait(monkeypatch, budget)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(budget.admit_many, unit("next"))
        try:
            assert waiting.wait(5)
            waiting.clear()
            budget.release(active[0])
            assert waiting.wait(5)
            assert not future.done()
            budget.release(active[1])
            assert future.result(timeout=5) is False
        finally:
            budget.cancel()
    metadata = budget.metadata()
    assert metadata["unmeasured_attempts"]["agent_tokens"] == 2
    assert metadata["reservation"]["estimate_denials"] == 0
    assert all("basis" not in row for row in metadata["not_run"])


@pytest.mark.parametrize("reservation", [None, {}, {"estimate_per_attempt": []}])
def test_estimate_budget_line_remains_honest_with_missing_reservation_details(reservation):
    budget = BudgetController(max_agent_tokens=15)
    budget.record_usage({"total_tokens": 10})
    assert not budget.admit_many(unit("next", 1))
    metadata = budget.metadata()
    assert "next attempt estimated 10" in budget_line(metadata)
    metadata["reservation"] = reservation
    assert "next unit denied using estimated usage" in budget_line(metadata)


@pytest.mark.parametrize("dimension", ["agent_tokens", "judge_calls", "cost_usd"])
def test_unknown_bounded_usage_stops_admission(dimension):
    kwargs = {f"max_{dimension}": 100}
    if dimension == "cost_usd":
        kwargs["price_book"] = PriceBook({})
    budget = BudgetController(**kwargs)
    active = unit("first")
    assert budget.admit_many(active)
    budget.release(active[0])
    budget.release(active[1])
    assert not budget.admit_many(unit("next"))
    assert budget.metadata()["unmeasured_attempts"][dimension] == 2
    assert budget.metadata()["reservation"]["in_flight_attempts"] == 0


@pytest.mark.parametrize("token_cap", [None, 100])
def test_attempt_cap_is_atomic_and_does_not_wait_for_in_flight_attempts(token_cap):
    budget = BudgetController(max_attempts_total=3, max_agent_tokens=token_cap)
    assert budget.admit_many(unit("first"))
    assert not budget.admit_many(unit("next"))
    assert budget.metadata()["started_attempts"] == 2
    assert budget.metadata()["reservation"]["waits"] == 0
    assert budget.metadata()["exhausted_limit"] == "attempts"


def test_each_dimension_keeps_its_own_largest_measurement():
    budget = BudgetController()
    budget.record_usage({"total_tokens": 3, "judge_calls": 5, "cost_usd": 0.4})
    budget.record_usage({"total_tokens": 10, "judge_calls": 2, "cost_usd": 0.1})
    assert budget.metadata()["reservation"]["estimate_per_attempt"] == {
        "agent_tokens": 10,
        "judge_calls": 5,
        "cost_usd": 0.4,
    }


def test_measured_zero_is_an_estimate_not_unknown():
    budget = BudgetController(max_judge_calls=5)
    budget.record_usage({"judge_calls": 0})
    assert budget.admit_many(unit("first"))
    assert budget.admit_many(unit("next"))
    assert budget.metadata()["reservation"]["unestimated_admissions"] == 0
    assert budget.metadata()["reservation"]["estimate_per_attempt"]["judge_calls"] == 0


def test_usage_exception_can_be_released_without_partial_accounting():
    budget = BudgetController(max_cost_usd=1, price_book=PriceBook({}))
    attempt = unit("first", 1)[0]
    assert budget.admit_many([attempt])
    with pytest.raises(BudgetConfigurationError):
        budget.record_usage({"input_tokens": 10, "output_tokens": 10}, model="missing", attempt=attempt)
    budget.release(attempt)
    budget.release(attempt)
    metadata = budget.metadata()
    assert metadata["finished_attempts"] == 1
    assert metadata["used"]["agent_tokens"] == "not measured"
    assert metadata["reservation"]["in_flight_attempts"] == 0
    assert metadata["unmeasured_attempts"]["cost_usd"] == 1


def test_several_waiting_threads_make_progress_without_deadlock(monkeypatch):
    budget = BudgetController(max_agent_tokens=55)
    budget.record_usage({"total_tokens": 10})
    active = unit("first")
    assert budget.admit_many(active)
    waiting, _ = observe_wait(monkeypatch, budget)

    def work(index):
        attempts = unit(str(index))
        if not budget.admit_many(attempts):
            return False
        for attempt in attempts:
            budget.record_usage({"total_tokens": 10}, attempt=attempt)
        return True

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(work, index) for index in range(6)]
        try:
            assert waiting.wait(5)
            for attempt in active:
                budget.record_usage({"total_tokens": 10}, attempt=attempt)
            assert sum(future.result(timeout=5) for future in futures) == 1
        finally:
            budget.cancel()
    assert budget.metadata()["reservation"]["in_flight_attempts"] == 0
    assert budget.metadata()["attempts_not_run"] == 10


def run_pairs(tmp_path, budget, cancellation=None, **kwargs):
    return runner._run_budgeted_variants(
        skill_name="skill",
        agent="a",
        jobs=[("with", tmp_path, None), ("without", tmp_path, None)],
        task_names=kwargs.get("task_names", ["case"]),
        env_mode="docker",
        model="model",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=kwargs.get("n_attempts", 1),
        n_concurrent=2,
        pass_threshold=0.5,
        stop_on_pass=False,
        timeout_multiplier=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        budget=budget,
        cancellation=cancellation,
    )


@pytest.mark.parametrize("path", ["success", "error", "exception", "cancel", "telemetry_exception"])
def test_runner_releases_every_admitted_attempt_once(monkeypatch, tmp_path, path):
    budget = BudgetController(max_agent_tokens=100)
    cancellation = runner._HarborRunCancellation()

    def harbor(**kwargs):
        if path == "exception":
            raise RuntimeError("job failure")
        if path == "cancel":
            cancellation.requested.set()
        return path != "error", "failed" if path == "error" else ""

    def usage(_path):
        if path == "telemetry_exception":
            raise ValueError("unreadable usage")
        return {"total_tokens": 10}

    monkeypatch.setattr(runner, "_run_harbor", harbor)
    monkeypatch.setattr(runner, "_budget_usage", usage)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_args: None)
    monkeypatch.setattr(runner, "_job_passed", lambda *_args: True)
    if path in {"success", "error"}:
        run_pairs(tmp_path, budget, cancellation)
    else:
        error = {
            "exception": RuntimeError,
            "cancel": runner._HarborRunCancelledError,
            "telemetry_exception": ValueError,
        }[path]
        with pytest.raises(error):
            run_pairs(tmp_path, budget, cancellation)
    metadata = budget.metadata()
    assert metadata["started_attempts"] == metadata["finished_attempts"] == 2
    assert metadata["reservation"]["in_flight_attempts"] == 0
    expected_unknown = 2 if path in {"exception", "telemetry_exception"} else int(path == "cancel")
    assert metadata["unmeasured_attempts"]["agent_tokens"] == expected_unknown
    assert cancellation.requested._callbacks == []


def test_cancellation_wakes_waiter_before_in_flight_job_finishes(monkeypatch, tmp_path):
    budget = BudgetController(max_agent_tokens=35)
    budget.record_usage({"total_tokens": 10})
    cancellation = runner._HarborRunCancellation()
    waiting, woke = observe_wait(monkeypatch, budget)
    finish_job = threading.Event()

    def harbor(**_kwargs):
        assert finish_job.wait(5)
        cancellation.raise_if_requested()

    monkeypatch.setattr(runner, "_run_harbor", harbor)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_pairs, tmp_path, budget, cancellation, task_names=["case", "other"])
        try:
            assert waiting.wait(5)
            cancellation.requested.set()
            assert woke.wait(5)
            assert budget.metadata()["reservation"]["in_flight_attempts"] == 2
        finally:
            finish_job.set()
            budget.cancel()
        # Both workers observe cancellation. Either the waiting budget admission or
        # the in-flight Harbor job can surface first through as_completed().
        with pytest.raises((BudgetCancelledError, runner._HarborRunCancelledError)):
            future.result(timeout=5)
    assert budget.metadata()["reservation"]["in_flight_attempts"] == 0
    assert budget.metadata()["not_run"] == []
