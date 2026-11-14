import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from rich.console import Console

from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.run_budget import (
    BudgetConfigurationError,
    BudgetController,
    Price,
    PriceBook,
    budget_line,
    planned_attempts,
)


def test_planned_attempts_and_attempt_cap():
    assert planned_attempts(3, 2, 2) == 24
    budget = BudgetController(max_attempts_total=1, planned=24)
    assert budget.admit(agent="a", arm="with_skill", case_id="c1", attempt=1)
    assert not budget.admit(agent="a", arm="with_skill", case_id="c2", attempt=1)
    assert budget.metadata()["not_run"][0]["reason"] == "budget_exhausted"
    assert "Budget exhausted (attempts 1 of 1)" in budget_line(budget.metadata())


def test_token_and_cost_admission():
    prices = PriceBook({"model": Price(input_per_mtok=1.0, output_per_mtok=2.0)})
    budget = BudgetController(max_agent_tokens=10, max_cost_usd=1.0, price_book=prices, planned=2)
    assert budget.admit(agent="a", arm="with_skill", case_id="c1", attempt=1)
    budget.record_usage({"total_tokens": 10, "total_cost_usd": 1.0})
    assert not budget.admit(agent="a", arm="with_skill", case_id="c2", attempt=1)
    assert budget.metadata()["exhausted_limit"] == "agent_tokens"
    assert (
        budget_line(budget.metadata())
        == "Budget exhausted (agent tokens 10 of 10): 1 of 2 attempts not run; results are partial"
    )


def test_judge_call_and_priced_cost_caps():
    prices = PriceBook({"model": Price(input_per_mtok=1.0, output_per_mtok=2.0)})
    judge_budget = BudgetController(max_judge_calls=1)
    assert judge_budget.admit(agent="a", arm="with_skill", case_id="c1", attempt=1)
    judge_budget.record_usage({"judge_calls": 1})
    assert not judge_budget.admit(agent="a", arm="with_skill", case_id="c2", attempt=1)

    cost_budget = BudgetController(max_cost_usd=0.001, price_book=prices)
    assert cost_budget.admit(agent="a", arm="with_skill", case_id="c1", attempt=1)
    cost_budget.record_usage({"input_tokens": 1000, "output_tokens": 0}, model="model")
    assert not cost_budget.admit(agent="a", arm="with_skill", case_id="c2", attempt=1)


def test_price_file_requires_each_used_model(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"model": {"input_per_mtok": 1, "output_per_mtok": 2}}))
    prices = PriceBook.load(path)
    assert prices.price_for("model").output_per_mtok == 2
    assert prices.price_for("model").cached_input_per_mtok == prices.price_for("model").input_per_mtok
    with pytest.raises(BudgetConfigurationError):
        prices.price_for("other")


def test_relative_configured_price_file_uses_source_skill_directory(tmp_path):
    from skillevaluator.tier3.harbor.runner import _resolve_harbor_price_file

    skill = tmp_path / "skill"
    price_file = skill / "evals" / "prices.json"
    price_file.parent.mkdir(parents=True)
    price_file.write_text("{}")
    assert _resolve_harbor_price_file(None, {"price_file": "evals/prices.json"}, skill) == price_file.resolve()


def test_unknown_usage_is_not_measured():
    budget = BudgetController(max_agent_tokens=10)
    assert budget.admit(agent="a", arm="with_skill", case_id="c1", attempt=1)
    budget.record_usage({})
    assert budget.metadata()["used"]["agent_tokens"] == "not measured"
    assert not budget.admit(agent="a", arm="with_skill", case_id="c2", attempt=1)
    assert budget.metadata()["unmeasured_attempts"]["agent_tokens"] == 1


def test_measured_usage_is_preserved_across_unmeasured_attempt():
    budget = BudgetController(max_agent_tokens=10_000)
    budget.record_usage({"total_tokens": 900})
    budget.record_usage({})
    budget.record_usage({"total_tokens": 100})
    assert budget.metadata()["used"]["agent_tokens"] == 1000
    assert budget.metadata()["unmeasured_attempts"]["agent_tokens"] == 1


def test_cached_input_pricing_overrides_reported_cost():
    prices = PriceBook({"model": Price(2.0, 4.0, 1.0)})
    budget = BudgetController(price_book=prices)
    budget.record_usage(
        {
            "input_tokens": 1000,
            "total_cached_tokens": 400,
            "output_tokens": 100,
            "cost_usd": 99,
        },
        model="model",
    )
    assert budget.metadata()["used"]["cost_usd"] == pytest.approx(0.002)


def test_empty_judge_attempt_artifact_counts_zero_and_allows_more_runs(tmp_path):
    job_dir = tmp_path / "job"
    artifact = job_dir / "trial" / "verifier" / "judge_attempts.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"attempts": []}))
    budget = BudgetController(max_judge_calls=1)
    budget.record_usage(runner._budget_usage(job_dir))
    assert budget.metadata()["used"]["judge_calls"] == 0
    assert budget.admit(agent="a", arm="with", case_id="next", attempt=1)


def test_budget_usage_reads_cached_tokens_and_excludes_salvage(tmp_path):
    trajectory_dir = tmp_path / "trial" / "agent"
    verifier_dir = tmp_path / "trial" / "verifier"
    trajectory_dir.mkdir(parents=True)
    verifier_dir.mkdir()
    (trajectory_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "final_metrics": {
                    "total_prompt_tokens": 1000,
                    "total_cached_tokens": 250,
                    "total_completion_tokens": 100,
                }
            }
        )
    )
    (verifier_dir / "judge_attempts.json").write_text(
        json.dumps({"attempts": [{"method": "llm"}, {"method": "salvage"}, {"method": "ragas"}]})
    )

    usage = runner._budget_usage(tmp_path)

    assert usage == {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cached_input_tokens": 250,
        "judge_calls": 2,
    }


def test_budget_usage_counts_one_judge_call_per_ragas_score_not_per_ragas_request(tmp_path):
    verifier_dir = tmp_path / "trial" / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "judge_attempts.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {"metric": "accuracy", "method": "llm", "status": "accepted", "prompt_sha256": "a" * 64},
                    # Requests RAGAS sent itself, the first one retried by the SDK after a 503.
                    {"metric": "goal_accuracy", "method": "ragas", "status": "error", "prompt_sha256": "b" * 64},
                    {"metric": "goal_accuracy", "method": "ragas", "status": "replied", "prompt_sha256": "b" * 64},
                    # The outcome entry holds the score and has no prompt.
                    {"metric": "goal_accuracy", "method": "ragas", "status": "accepted", "score": 1.0},
                ]
            }
        )
    )

    assert runner._budget_usage(tmp_path) == {"judge_calls": 2}


def test_budgeted_attempts_use_real_cancellation_object(monkeypatch, tmp_path):
    cancellation = runner._HarborRunCancellation()
    cancellation.requested.set()
    monkeypatch.setattr(runner, "_run_harbor", lambda **_kwargs: pytest.fail("must not start"))
    budget = BudgetController()
    with pytest.raises(runner._HarborRunCancelledError):
        runner._run_budgeted_variant(
            skill_name="skill",
            agent="agent",
            variant="with",
            dataset=tmp_path / "dataset",
            task_names=["case"],
            env_mode="docker",
            model="model",
            jobs_dir=tmp_path / "jobs",
            run_env={},
            n_attempts=1,
            pass_threshold=0.5,
            stop_on_pass=False,
            timeout_multiplier=1,
            override_cpus=None,
            override_memory_mb=None,
            override_storage_mb=None,
            budget=budget,
            cancellation=cancellation,
        )


def test_budgeted_attempts_run_concurrently_up_to_limit(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_harbor(*, job_name, jobs_dir, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        (jobs_dir / job_name).mkdir(parents=True)
        with lock:
            active -= 1
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_harbor)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_: None)
    runner._run_budgeted_variant(
        skill_name="skill",
        agent="agent",
        variant="with",
        dataset=tmp_path / "dataset",
        task_names=["one", "two"],
        env_mode="docker",
        model="model",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        pass_threshold=0.5,
        stop_on_pass=False,
        timeout_multiplier=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        budget=BudgetController(),
        n_concurrent=2,
    )
    assert peak == 2


def test_in_flight_agents_are_not_rejected_for_missing_usage(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    starts = []
    lock = threading.Lock()

    def fake_harbor(*, job_name, jobs_dir, **_kwargs):
        with lock:
            starts.append(job_name)
        barrier.wait(timeout=5)
        (jobs_dir / job_name).mkdir(parents=True)
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_harbor)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_: None)
    budget = BudgetController(max_agent_tokens=100_000)

    def run_agent(agent):
        return runner._run_budgeted_variant(
            skill_name="skill",
            agent=agent,
            variant="with",
            dataset=tmp_path / "dataset",
            task_names=["case"],
            env_mode="docker",
            model="model",
            jobs_dir=tmp_path / "jobs",
            run_env={},
            n_attempts=1,
            pass_threshold=0.5,
            stop_on_pass=False,
            timeout_multiplier=1,
            override_cpus=None,
            override_memory_mb=None,
            override_storage_mb=None,
            budget=budget,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run_agent, ["first", "second"]))
    assert len(starts) == 2
    assert budget.metadata()["finished_attempts"] == 2


def test_budget_admits_both_arms_as_one_case_attempt(monkeypatch, tmp_path):
    starts = []

    def fake_harbor(*, job_name, jobs_dir, **_kwargs):
        starts.append(job_name)
        trajectory = jobs_dir / job_name / "trial" / "agent"
        trajectory.mkdir(parents=True)
        (trajectory / "trajectory.json").write_text(
            json.dumps({"final_metrics": {"total_prompt_tokens": 1, "total_completion_tokens": 0}})
        )
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_harbor)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_: None)
    budget = BudgetController(max_agent_tokens=2)
    runner._run_budgeted_variants(
        skill_name="skill",
        agent="agent",
        jobs=[("with", tmp_path / "with", None), ("without", tmp_path / "without", None)],
        task_names=["first", "second"],
        env_mode="docker",
        model="model",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        pass_threshold=0.5,
        stop_on_pass=False,
        timeout_multiplier=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        budget=budget,
        n_concurrent=1,
    )
    assert len(starts) == 2
    assert any("-with-first-" in name for name in starts)
    assert any("-without-first-" in name for name in starts)
    assert budget.metadata()["attempts_not_run"] == 2


def test_partial_budget_run_collects_real_harbor_results(monkeypatch, tmp_path):
    import shutil
    from datetime import UTC, datetime
    from uuid import UUID

    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult

    from skillevaluator.tier3.harbor.collector import _extract_rewards, collect_harbor_results
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS

    jobs_dir = tmp_path / "jobs"
    starts = []
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def write_job_result(job_dir, task_selector, attempt, score):
        trial_name = f"{task_selector}__attempt-{attempt}"
        trial_dir = job_dir / trial_name
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True)
        (trial_dir / "agent").mkdir()
        reward = {
            "entry_id": task_selector,
            "_attempt_ordinal": attempt,
            "metric_set": DEFAULT_METRIC_SET,
            **dict.fromkeys(DEFAULT_METRICS, score),
        }
        numeric_reward = dict.fromkeys(DEFAULT_METRICS, score)
        (verifier_dir / "reward.json").write_text(json.dumps(numeric_reward))
        (verifier_dir / "skill_evaluator_reward.json").write_text(json.dumps(reward))
        (trial_dir / "agent" / "trajectory.json").write_text(
            json.dumps({"final_metrics": {"total_prompt_tokens": 500, "total_completion_tokens": 0}})
        )
        trial = TrialResult.model_validate(
            {
                "id": UUID(int=len(starts) + 1),
                "task_name": task_selector,
                "trial_name": trial_name,
                "trial_uri": trial_dir.as_uri(),
                "task_id": {"path": task_selector},
                "task_checksum": "budget-fixture",
                "config": {"task": {"path": task_selector}, "trial_name": trial_name},
                "agent_info": {
                    "name": "codex",
                    "version": "fixture",
                    "model_info": {"name": "fixture-model"},
                },
                "agent_result": {"n_input_tokens": 500, "n_output_tokens": 0},
                "verifier_result": {"rewards": numeric_reward},
                "started_at": now,
                "finished_at": now,
                "step_results": None,
            }
        )
        (trial_dir / "result.json").write_text(trial.model_dump_json())
        result = JobResult(
            id=UUID(int=len(starts) + 100),
            started_at=now,
            updated_at=now,
            finished_at=now,
            n_total_trials=1,
            stats=JobStats.from_trial_results([trial], n_total_trials=1),
            trial_results=[trial],
        )
        (job_dir / "result.json").write_text(result.model_dump_json())

    def fake_harbor(*, job_name, jobs_dir, **_kwargs):
        starts.append(job_name)
        variant = "with" if "-with-" in job_name else "without"
        task_selector = job_name.split(f"agent-{variant}-", 1)[1].rsplit("-attempt", 1)[0]
        attempt = int(job_name.rsplit("-attempt", 1)[1])
        write_job_result(jobs_dir / job_name, task_selector, attempt, 0.8 if variant == "with" else 0.6)
        return True, ""

    def merge_attempts(source_dirs, aggregate_dir):
        trials = []
        for source_dir in source_dirs:
            source = JobResult.model_validate_json((source_dir / "result.json").read_text())
            trials.extend(source.trial_results or [])
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        for source_dir in source_dirs:
            for trial_dir in source_dir.iterdir():
                if trial_dir.is_dir():
                    shutil.copytree(trial_dir, aggregate_dir / trial_dir.name, dirs_exist_ok=True)
        result = JobResult(
            id=UUID(int=999),
            started_at=now,
            updated_at=now,
            finished_at=now,
            n_total_trials=len(trials),
            stats=JobStats.from_trial_results(trials, n_total_trials=len(trials)),
            trial_results=trials,
        )
        (aggregate_dir / "result.json").write_text(result.model_dump_json())

    monkeypatch.setattr(runner, "_run_harbor", fake_harbor)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", merge_attempts)
    budget = BudgetController(max_agent_tokens=2500, planned=12)
    errors = runner._run_budgeted_variants(
        skill_name="skill",
        agent="agent",
        jobs=[("with", tmp_path / "with", None), ("without", tmp_path / "without", None)],
        task_names=["case-1", "case-2", "case-3"],
        env_mode="docker",
        model="fixture-model",
        jobs_dir=jobs_dir,
        run_env={},
        n_attempts=2,
        pass_threshold=0.5,
        stop_on_pass=False,
        timeout_multiplier=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        budget=budget,
        n_concurrent=1,
    )
    assert not any(errors.values())
    collected = _extract_rewards(
        jobs_dir / "skill-agent-with",
        {case_id: case_id for case_id in ("case-1", "case-2", "case-3")},
    )
    assert len(collected) == 3, collected
    results = collect_harbor_results(
        "skill",
        ["agent"],
        tmp_path / "results",
        jobs_dir,
        n_attempts=2,
        expected_cases=3,
        expected_case_ids=["case-1", "case-2", "case-3"],
        case_id_by_task_selector={case_id: case_id for case_id in ("case-1", "case-2", "case-3")},
        expected_trials=6,
        budget_not_run=budget.not_run,
    )
    agent_result = results["agents"]["agent"]
    assert results["partial"] is True
    assert agent_result["execution_status"] == "succeeded", agent_result.get("execution_errors") or agent_result
    assert agent_result["num_trials_with"] == 3
    assert len(starts) == 6


def test_fake_attempt_runner_stops_after_token_budget(monkeypatch, tmp_path):
    starts = []

    def fake_harbor(*, job_name, jobs_dir, **_kwargs):
        starts.append(job_name)
        job_dir = jobs_dir / job_name
        trajectory_dir = job_dir / "trial" / "agent"
        trajectory_dir.mkdir(parents=True)
        (trajectory_dir / "trajectory.json").write_text(
            json.dumps({"final_metrics": {"total_prompt_tokens": 8, "total_completion_tokens": 4}})
        )
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_harbor)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_: None)
    budget = BudgetController(max_agent_tokens=10, planned=3)
    runner._run_budgeted_variant(
        skill_name="skill",
        agent="agent",
        variant="with",
        dataset=tmp_path / "dataset",
        task_names=["case"],
        env_mode="docker",
        model="model",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=3,
        pass_threshold=0.5,
        stop_on_pass=False,
        timeout_multiplier=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        budget=budget,
    )
    assert len(starts) == 1
    assert budget.metadata()["attempts_not_run"] == 2


def test_evaluate_summary_renders_budget_warning():
    from skillevaluator.tier3.result_display import render_evaluation_result

    output = Console(record=True, force_terminal=False, width=160)
    result = {
        "execution_status": "failed",
        "budget": {
            "limits": {"max_agent_tokens": 1000},
            "used": {"agent_tokens": 1000},
            "status": "exhausted",
            "exhausted_limit": "agent_tokens",
            "planned_attempts": 3,
            "attempts_not_run": 2,
            "not_run": [{}, {}],
        },
    }
    render_evaluation_result(result, console=output)
    assert (
        "Budget exhausted (agent tokens 1,000 of 1,000): 2 of 3 attempts not run; results are partial"
        in output.export_text()
    )


def test_html_overview_renders_budget_warning():
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting import HTMLReporter

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = {
        "execution_status": "failed",
        "budget": {"status": "exhausted", "not_run": [{}, {}]},
        "budget_summary": "Budget exhausted (agent tokens 1,000 of 1,000): 2 of 3 attempts not run; results are partial",
    }
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert "Budget exhausted (agent tokens 1,000 of 1,000): 2 of 3 attempts not run; results are partial" in html
    assert 'class="warning"' in html


def test_final_attempt_overrun_is_reported_without_changing_admission_status():
    # Retained A2 shape: all four attempts finished, but no fifth admission
    # happened to update the admission status after usage exceeded 32,000.
    budget = BudgetController(max_agent_tokens=32_000, max_judge_calls=12, max_attempts_total=4, planned=4)
    for attempt, tokens in enumerate([10_000, 10_000, 10_000, 15_236], start=1):
        assert budget.admit(agent="a", arm="with", case_id="c", attempt=attempt)
        budget.record_usage({"total_tokens": tokens, "judge_calls": 3})

    metadata = budget.metadata()
    assert metadata["status"] == "ok"
    assert metadata["partial"] is False
    assert metadata["exhausted_limit"] is None
    assert metadata["attempts_not_run"] == 0
    assert metadata["observed_overruns"] == {
        "agent_tokens": {"used": 45_236, "limit": 32_000, "excess": 13_236},
    }
    assert budget_line(metadata) == "Budget overrun observed (agent tokens 45,236 of 32,000)"


@pytest.mark.parametrize("cached", [None, {}, {"agent_tokens": {"used": 1, "limit": 0, "excess": 1}}])
def test_legacy_summary_derives_overruns_from_usage_instead_of_cached_fields(cached):
    metadata = {
        "status": "ok",
        "limits": {"max_agent_tokens": 32_000},
        "used": {"agent_tokens": 45_236},
        "observed_overruns": cached,
    }
    assert budget_line(metadata) == "Budget overrun observed (agent tokens 45,236 of 32,000)"
    metadata["used"]["agent_tokens"] = 32_000
    assert budget_line(metadata) == "Budget: within configured limits"


@pytest.mark.parametrize("cap,used", [(0, 0), (10, 10), (10, 9)])
def test_measured_usage_at_or_below_cap_has_no_overrun(cap, used):
    budget = BudgetController(max_agent_tokens=cap)
    budget.record_usage({"total_tokens": used})
    assert budget.metadata()["observed_overruns"] == {}
    assert budget_line(budget.metadata()) == "Budget: within configured limits"


def test_multiple_overruns_include_partial_measurements_and_not_run_state():
    prices = PriceBook({"model": Price(1.0, 2.0)})
    budget = BudgetController(max_agent_tokens=10, max_judge_calls=1, max_cost_usd=0.01, price_book=prices, planned=3)
    assert budget.admit(agent="a", arm="with", case_id="c", attempt=1)
    budget.record_usage({"total_tokens": 12, "judge_calls": 2, "cost_usd": 0.02})
    budget.record_usage({})
    assert not budget.admit(agent="a", arm="with", case_id="c", attempt=3)
    metadata = budget.metadata()
    assert metadata["observed_overruns"] == {
        "agent_tokens": {"used": 12, "limit": 10, "excess": 2},
        "judge_calls": {"used": 2, "limit": 1, "excess": 1},
        "cost_usd": {"used": 0.02, "limit": 0.01, "excess": 0.01},
    }
    line = budget_line(metadata)
    assert "Budget exhausted (agent tokens 12 of 10): 1 of 3 attempts not run; results are partial" in line
    assert "overrun observed (agent tokens 12 of 10; judge calls 2 of 1; cost $0.02 of $0.01)" in line
    assert "usage partially measured" in line


@pytest.mark.parametrize("usage", [None, {}, {"total_tokens": 5}])
def test_bounded_unmeasured_or_partial_usage_is_not_claimed_within_limits(usage):
    budget = BudgetController(max_agent_tokens=10)
    budget.record_usage(usage)
    budget.record_usage({})
    line = budget_line(budget.metadata())
    assert "within configured limits" not in line
    assert "not fully verified" in line
    assert ("partially measured" if usage else "not measured") in line


def test_unconfigured_usage_does_not_imply_a_budget_verification():
    budget = BudgetController()
    budget.record_usage({"total_tokens": 100})
    assert budget.metadata()["observed_overruns"] == {}
    assert budget_line(budget.metadata()) == "Budget: no limits configured"


@pytest.mark.parametrize(
    "key,cap,value,expected",
    [
        ("agent_tokens", 0, 1, "agent tokens 1 of 0"),
        ("judge_calls", 0, 1, "judge calls 1 of 0"),
        ("cost_usd", 0.00001, 0.00002, "cost $2e-05 of $1e-05"),
        ("attempts", 1, 2, "attempts 2 of 1"),
    ],
)
def test_legacy_overrun_boundaries_include_zero_and_subcent_costs(key, cap, value, expected):
    limit = {
        "agent_tokens": "max_agent_tokens",
        "judge_calls": "max_judge_calls",
        "cost_usd": "max_cost_usd",
        "attempts": "max_attempts_total",
    }[key]
    metadata = {"limits": {limit: cap}, "used": {key: value}, "started_attempts": value}
    assert budget_line(metadata) == f"Budget overrun observed ({expected})"


@pytest.mark.parametrize("invalid", [True, False, -1, "[bold]bad[/bold]", float("nan"), float("inf"), [], {}])
@pytest.mark.parametrize("field", ["used", "limits", "unmeasured_attempts"])
def test_malformed_bounded_metadata_never_claims_verified_usage_or_breaks_rich(invalid, field):
    metadata = {"limits": {"max_agent_tokens": 10}, "used": {"agent_tokens": 5}}
    metadata[field] = {"max_agent_tokens" if field == "limits" else "agent_tokens": invalid}
    line = budget_line(metadata)
    output = Console(record=True, force_terminal=False)
    output.print(line)
    assert "within configured limits" not in output.export_text()
    assert "not fully verified" in line


@pytest.mark.parametrize("invalid", [None, True, float("inf"), "[bold]bad[/bold]", []])
def test_sparse_or_malformed_exhausted_metadata_is_safe_to_render(invalid):
    metadata = {
        "status": "exhausted",
        "exhausted_limit": invalid,
        "limits": invalid,
        "used": invalid,
        "planned_attempts": invalid,
        "attempts_not_run": invalid,
    }
    output = Console(record=True, force_terminal=False)
    output.print(budget_line(metadata))
    assert "Budget exhausted" in output.export_text()
    assert "[bold]" not in output.export_text()
    assert "results are partial" not in output.export_text()


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_budget_overrun_reaches_terminal_html_and_json_summary(legacy):
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting import HTMLReporter, JSONReporter
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
    from skillevaluator.tier3.result_display import render_evaluation_result

    budget = BudgetController(max_agent_tokens=32_000, planned=4)
    budget.record_usage({"total_tokens": 45_236})
    metadata = budget.metadata()
    if legacy:
        metadata.pop("observed_overruns", None)
    payload = build_agent_eval_payload(
        "budget-fixture",
        {
            "a": {
                "execution_status": "succeeded",
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 1.0),
            }
        },
        run_config={"budget": metadata},
        use_llm_judge=False,
    )
    assert payload is not None
    expected = "Budget overrun observed (agent tokens 45,236 of 32,000)"
    assert payload["budget_summary"] == expected
    assert payload["summary"]["execution_status"] == "succeeded"
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert f'<p class="warning">{expected}</p>' in html
    report = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    assert report["tier3"]["budget_summary"] == expected
    output = Console(record=True, force_terminal=False, width=160)
    render_evaluation_result({"execution_status": "succeeded", "budget": metadata}, console=output)
    assert expected in output.export_text()
