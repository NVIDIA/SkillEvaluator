import json

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
    with pytest.raises(BudgetConfigurationError):
        prices.price_for("other")


def test_unknown_usage_is_not_measured():
    budget = BudgetController(max_agent_tokens=10)
    assert budget.admit(agent="a", arm="with_skill", case_id="c1", attempt=1)
    budget.record_usage({})
    assert budget.metadata()["used"]["agent_tokens"] == "not measured"
    assert not budget.admit(agent="a", arm="with_skill", case_id="c2", attempt=1)


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
