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
