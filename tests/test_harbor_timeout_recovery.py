# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeout recovery uses native Harbor evidence, never a fabricated reward."""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("harbor")

from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.result import TrialResult

from skillevaluator.evaluation.tier3_report import (
    _compact_agent_conditions,
    agent_eval_result_from_directory,
    render_agent_eval_html_report,
)
from skillevaluator.reporting import HTMLReporter
from skillevaluator.tier3.harbor.collector import (
    _pass_summary,
    _shared_metric_scorer,
    _timeout_recovered_by,
    collect_harbor_results,
    harbor_timeout_job_recovered,
)
from skillevaluator.tier3.harbor.runner import _merge_attempt_jobs
from skillevaluator.tier3.result_display import render_result


@pytest.mark.parametrize(
    ("shared_value", "skill_value", "threshold", "passed"),
    [(0.0, 1.0, 0.5, False), (1.0, 0.0, 0.8, True)],
)
def test_pass_summary_recovery_uses_shared_score_and_physical_ordinal(shared_value, skill_value, threshold, passed):
    shared = {"security", "accuracy", "goal_accuracy", "behavior_check"}
    reward = {
        "entry_id": "case-a",
        "_trial_name": "case-a__attempt2",
        "_attempt_ordinal": 2,
        "security": 1.0,
        "accuracy": shared_value,
        "goal_accuracy": shared_value,
        "behavior_check": shared_value,
        "skill_execution": skill_value,
        "skill_efficiency": skill_value,
    }
    summary = _pass_summary(
        [reward],
        n_attempts=3,
        pass_threshold=threshold,
        stop_on_pass=True,
        expected_cases=1,
        expected_case_ids=["case-a"],
        recovered_attempts=[{"case_id": "case-a", "attempt": 1}],
        scorer=_shared_metric_scorer({"case-a": shared}, shared, {"case-a"}),
    )
    case = summary["cases"]["case-a"]
    assert case["passed"] is passed
    assert case["first_pass_attempt"] == (2 if passed else None)
    assert case["attempts"] == [
        {"attempt": 2, "trial": "case-a__attempt2", "score": 1.0 if passed else 0.3333, "passed": passed}
    ]
    assert case["attempts_used"] == case["attempts_recovered"] == 1
    assert case["attempts_skipped"] == int(passed)
    assert case["attempts_missing"] == int(not passed)


def native_attempt(root: Path, attempt: int, *, timeout: bool = False, reward: float = 1.0) -> Path:
    job = root / f"demo-opencode-with-case-a-attempt{attempt:03d}"
    trial = job / f"case-a__{attempt:07d}"
    trial.mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=attempt)
    finish = start + timedelta(seconds=5)
    result = TrialResult.model_validate(
        {
            "id": uuid4(),
            "task_name": "case-a",
            "trial_name": trial.name,
            "trial_uri": trial.as_uri(),
            "task_id": {"path": str(root / "task" / "case-a")},
            "task_checksum": "public-timeout-example",
            "config": {
                "task": {"path": str(root / "task" / "case-a")},
                "trial_name": trial.name,
                "trials_dir": str(job),
            },
            "agent_info": {"name": "opencode", "version": "test", "model_info": {"name": "test-model"}},
            "agent_result": {},
            "started_at": start,
            "finished_at": finish,
            "agent_execution": {"started_at": start, "finished_at": finish},
            "exception_info": {
                "exception_type": "AgentTimeoutError",
                "exception_message": "Agent execution timed out after 5.0 seconds",
                "exception_traceback": "in _run_agent_phase\nraise AgentTimeoutError(...)",
                "occurred_at": finish,
            }
            if timeout
            else None,
            "verifier_result": None if timeout else {"rewards": {"reward": reward}},
        }
    )
    (trial / "result.json").write_text(result.model_dump_json(indent=2))
    job_result = JobResult(
        id=uuid4(),
        started_at=start,
        updated_at=finish,
        finished_at=finish,
        n_total_trials=1,
        stats=JobStats.from_trial_results([result], n_total_trials=1),
        trial_results=[result],
    )
    (job / "result.json").write_text(job_result.model_dump_json(indent=2))
    return job


def collect(root: Path, *attempt_jobs: Path, stop_on_pass: bool = True):
    _merge_attempt_jobs(list(attempt_jobs), root / "jobs" / "demo-opencode-with")
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=root / "results",
        jobs_dir=root / "jobs",
        skip_baseline=True,
        n_attempts=3,
        expected_cases=1,
        expected_case_ids=["case-a"],
        case_id_by_task_selector={"case-a": "case-a"},
        stop_on_pass=stop_on_pass,
    )


def test_fresh_scored_retry_preserves_timeout_history(tmp_path):
    failed = native_attempt(tmp_path / "attempts", 1, timeout=True)
    passed = native_attempt(tmp_path / "attempts", 2)
    original = next(failed.glob("*/result.json")).read_bytes()
    assert harbor_timeout_job_recovered(failed, passed)
    result = collect(tmp_path, failed, passed)
    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded", result.get("execution_errors")
    assert agent["overall_with_skill"] == 1.0
    condition = agent["conditions"]["with_skill"]
    assert condition["scored_attempts"] == condition["expected_attempts"] == 1
    assert condition["physical_expected_attempts"] == 2
    assert condition["recovered_attempts_total"] == 1
    assert agent["trial_failures"]["with_skill"] == []
    assert agent["recovered_trial_failures"]["with_skill"]
    assert agent["job_failures"]["with_skill"] == ""
    assert agent["historical_job_failures"]["with_skill"]
    assert agent["failure_detail_metadata"]["with_skill_trials"]["details_total"] == 0
    from skillevaluator.tier3.result_display import render_result

    rendered = render_result(result)
    assert "AgentTimeoutError" not in rendered
    assert "errored" not in rendered
    case = agent["pass_at_k"]["with_skill"]["cases"]["case-a"]
    assert case["first_pass_attempt"] == 2
    assert case["attempts_used"] == 1
    assert case["attempts_recovered"] == 1
    assert case["attempts_skipped"] == 1
    assert case["attempts_missing"] == 0
    assert next(failed.glob("*/result.json")).read_bytes() == original
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    loaded = load_agent_data(tmp_path / "results")["opencode"]
    assert loaded["execution_status"] == "succeeded"
    assert loaded["execution_errors"] == []
    assert loaded["conditions"]["with_skill"]["recovered_trial_failures"]


@pytest.mark.parametrize(
    "mutation",
    [
        "type",
        "origin",
        "timing",
        "checksum",
        "model",
        "config",
        "identity",
        "verifier",
        "multistep",
    ],
)
def test_recovery_rejects_ambiguous_or_different_execution(tmp_path, mutation):
    failed_job = native_attempt(tmp_path, 1, timeout=True)
    passed_job = native_attempt(tmp_path, 2)
    failed = json.loads(next(failed_job.glob("*/result.json")).read_text())
    passed = json.loads(next(passed_job.glob("*/result.json")).read_text())
    assert _timeout_recovered_by(failed, passed)
    if mutation == "type":
        failed["exception_info"]["exception_type"] = "VerifierTimeoutError"
    elif mutation == "origin":
        failed["exception_info"]["exception_traceback"] = "AgentTimeoutError in setup"
    elif mutation == "timing":
        passed["started_at"] = failed["started_at"]
    elif mutation == "checksum":
        passed["task_checksum"] = "different"
    elif mutation == "model":
        passed["agent_info"]["model_info"]["name"] = "different"
    elif mutation == "config":
        passed["config"]["timeout_multiplier"] = 2.0
    elif mutation == "identity":
        passed["id"] = failed["id"]
    elif mutation == "verifier":
        passed["exception_info"] = copy.deepcopy(failed["exception_info"])
    elif mutation == "multistep":
        failed["step_results"] = [{"step_name": "step"}]
    assert not _timeout_recovered_by(failed, passed)


def test_fixed_attempt_budget_does_not_silently_discard_timeouts(tmp_path):
    failed = native_attempt(tmp_path / "attempts", 1, timeout=True)
    passed = native_attempt(tmp_path / "attempts", 2)
    result = collect(tmp_path, failed, passed, stop_on_pass=False)
    assert result["execution_status"] == "failed"


def test_unscored_retry_cannot_recover_timeout(tmp_path):
    failed = native_attempt(tmp_path / "attempts", 1, timeout=True)
    failed_again = native_attempt(tmp_path / "attempts", 2, timeout=True)
    assert not harbor_timeout_job_recovered(failed, failed_again)
    assert collect(tmp_path, failed, failed_again)["execution_status"] == "failed"


def test_post_timeout_verifier_reward_does_not_become_a_score(tmp_path):
    failed = native_attempt(tmp_path / "attempts", 1, timeout=True)
    passed = native_attempt(tmp_path / "attempts", 2)
    # Harbor may run the verifier after the deadline. Its reward is not a completed attempt.
    path = next(failed.glob("*/result.json"))
    payload = json.loads(path.read_text())
    payload["verifier_result"] = {"rewards": {"reward": 0.0}}
    path.write_text(json.dumps(payload))
    # Build canonical job statistics from the unchanged exception-bearing trial.
    result = TrialResult.model_validate(payload)
    job_path = failed / "result.json"
    job = JobResult.model_validate_json(job_path.read_text())
    job.stats = JobStats.from_trial_results([result], n_total_trials=1)
    job_path.write_text(job.model_dump_json())
    data = collect(tmp_path, failed, passed)
    assert data["execution_status"] == "succeeded"
    assert data["agents"]["opencode"]["overall_with_skill"] == 1.0
    assert data["scored_attempts"] == 1


def test_runner_keeps_unrelated_launch_errors(tmp_path, monkeypatch):
    from skillevaluator.tier3.harbor import runner

    jobs = tmp_path / "jobs"
    native_attempt(jobs, 1, timeout=True)
    native_attempt(jobs, 2)
    calls = []

    def launch(**kwargs):
        calls.append(kwargs)
        return (False, "output exceeded safety limit") if len(calls) == 1 else (True, "")

    monkeypatch.setattr(runner, "_run_harbor", launch)
    errors = runner._run_stop_on_pass_variant(
        skill_name="demo",
        agent="opencode",
        variant="with",
        dataset=tmp_path,
        task_names=["case-a"],
        env_mode="docker",
        model="test-model",
        jobs_dir=jobs,
        run_env={},
        n_attempts=3,
        pass_threshold=0.5,
        timeout_multiplier=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )
    assert len(calls) == 2
    assert len(errors) == 1 and "safety limit" in errors[0]


def test_runner_does_not_retain_recovered_deadline_as_launch_failure(tmp_path, monkeypatch):
    from skillevaluator.tier3.harbor import runner

    jobs = tmp_path / "jobs"
    native_attempt(jobs, 1, timeout=True)
    native_attempt(jobs, 2)
    calls = []

    def launch(**kwargs):
        calls.append(kwargs)
        return (False, "Harbor job did not complete successfully: 1 errored") if len(calls) == 1 else (True, "")

    monkeypatch.setattr(runner, "_run_harbor", launch)
    errors = runner._run_stop_on_pass_variant(
        skill_name="demo",
        agent="opencode",
        variant="with",
        dataset=tmp_path,
        task_names=["case-a"],
        env_mode="docker",
        model="test-model",
        jobs_dir=jobs,
        run_env={},
        n_attempts=3,
        pass_threshold=0.5,
        timeout_multiplier=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )
    assert len(calls) == 2
    assert errors == []


def test_recovery_ignores_harbor_job_metrics(tmp_path):
    failed = native_attempt(tmp_path / "attempts", 1, timeout=True)
    passed = native_attempt(tmp_path / "attempts", 2)
    for job in (failed, passed):
        path = job / "result.json"
        payload = json.loads(path.read_text())
        for stats in payload["stats"]["evals"].values():
            stats["metrics"] = [{"mean": 0.0}]
            stats["pass_at_k"] = {"1": 0.0}
        path.write_text(json.dumps(payload))
    assert harbor_timeout_job_recovered(failed, passed)
    assert collect(tmp_path, failed, passed)["execution_status"] == "succeeded"


def test_recovery_does_not_depend_on_directory_order(tmp_path, monkeypatch):
    attempts = [
        native_attempt(tmp_path / "attempts", 1, timeout=True),
        native_attempt(tmp_path / "attempts", 2, timeout=True),
        native_attempt(tmp_path / "attempts", 3),
    ]
    _merge_attempt_jobs(attempts, tmp_path / "jobs" / "demo-opencode-with")
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(sorted(original_iterdir(self), reverse=True)))
    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        n_attempts=3,
        expected_cases=1,
        expected_case_ids=["case-a"],
        case_id_by_task_selector={"case-a": "case-a"},
        stop_on_pass=True,
    )
    assert result["execution_status"] == "succeeded", result.get("execution_errors")
    agent = result["agents"]["opencode"]
    assert agent["overall_with_skill"] == 1.0
    assert len(agent["recovered_trial_failures"]["with_skill"]) == 2
    assert agent["trial_failures"]["with_skill"] == []


def attempt_details_rows(html: str) -> list[str]:
    """Return each Attempt Details body row as its visible text."""
    body = html.split('id="t3-attempts-table"', 1)[1].split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    return [" ".join(re.sub(r"<[^>]+>", " ", row).split()) for row in re.findall(r"<tr>(.*?)</tr>", body, re.DOTALL)]


def attempt_details_hint(html: str) -> str:
    """Return the Attempt Details section hint as one line of text."""
    section = html.split('id="tier3-attempt-details"', 1)[1]
    return " ".join(section.split('<p class="t3-section-hint">', 1)[1].split("</p>", 1)[0].split())


def recovery_lines(rendered: str) -> list[str]:
    """Return the terminal lines that report recovered agent timeouts."""
    return [line.strip() for line in rendered.splitlines() if "recovered by a later scored attempt" in line]


def render_html(root: Path) -> str:
    return render_agent_eval_html_report(root, root / "results", use_llm_judge=False).read_text(encoding="utf-8")


def recovered_run_report(root: Path):
    """Collect a timeout on attempt 1 and a pass on attempt 2 of 3, and load it for the HTML reporter."""
    collect(root, native_attempt(root / "attempts", 1, timeout=True), native_attempt(root / "attempts", 2))
    result = agent_eval_result_from_directory(root, root / "results", use_llm_judge=False)
    return result, result.metadata["agent_eval"]["agents"]["opencode"]


HINT_START = "Each row is one eval case for one agent under one condition. Attempt Scores lists the scored attempts"
HINT_THRESHOLD = (
    "a score at or above 0.50 counts as passed for Pass@3. "
    '"Not scored" means no reward was found for that expected attempt; it is excluded from pass/fail.'
)
RECOVERED_HINT = (
    f"{HINT_START} and any recovered timeouts; {HINT_THRESHOLD} "
    '"Timed out, recovered" marks an agent timeout that a later scored attempt of the same case recovered; '
    "the chip shows the attempt number when the report has it, and otherwise a count. "
    "It is not scored, but it used one of the case's attempts, so it counts in "
    "the attempt total under Scored Attempts (the second number), not in the scored count."
)


def test_attempt_details_show_recovered_timeout_in_the_attempt_budget(tmp_path):
    result = collect(
        tmp_path,
        native_attempt(tmp_path / "attempts", 1, timeout=True),
        native_attempt(tmp_path / "attempts", 2),
    )
    assert result["execution_status"] == "succeeded", result.get("execution_errors")

    html = render_html(tmp_path)

    # Attempt 1 timed out, attempt 2 passed, attempt 3 was skipped: three attempts, one scored.
    assert attempt_details_rows(html) == [
        "opencode With skill case-a A1: timed out, recovered A2: 1.00 pass 1 of 3 scored 1.00 passed on attempt 2"
    ]
    assert "passed on scored attempt" not in html
    # The hint lists the recovered chip among the attempt scores and counts it only in the second number.
    assert attempt_details_hint(html) == RECOVERED_HINT
    assert 'title="Includes 1 recovered agent timeout, which is not scored"' in html


def test_attempt_details_place_each_recovered_timeout_at_its_attempt_number(tmp_path):
    result = collect(
        tmp_path,
        native_attempt(tmp_path / "attempts", 1, reward=0.0),
        native_attempt(tmp_path / "attempts", 2, timeout=True),
        native_attempt(tmp_path / "attempts", 3),
    )
    assert result["execution_status"] == "succeeded", result.get("execution_errors")
    case = result["agents"]["opencode"]["pass_at_k"]["with_skill"]["cases"]["case-a"]
    assert (case["attempts_used"], case["attempts_recovered"], case["first_pass_attempt"]) == (2, 1, 3)

    [row] = attempt_details_rows(render_html(tmp_path))

    assert row == (
        "opencode With skill case-a A1: 0.00 fail A2: timed out, recovered A3: 1.00 pass "
        "2 of 3 scored 1.00 passed on attempt 3"
    )


def test_attempt_details_without_recovery_only_reword_the_passing_attempt(tmp_path):
    result = collect(
        tmp_path,
        native_attempt(tmp_path / "attempts", 1, reward=0.0),
        native_attempt(tmp_path / "attempts", 2),
    )
    assert result["execution_status"] == "succeeded", result.get("execution_errors")

    html = render_html(tmp_path)

    assert attempt_details_rows(html) == [
        "opencode With skill case-a A1: 0.00 fail A2: 1.00 pass 2 of 3 scored 1.00 passed on attempt 2"
    ]
    assert "recovered" not in html.split('id="t3-attempts-table"', 1)[1].split("</table>", 1)[0]
    assert attempt_details_hint(html) == f"{HINT_START} only; {HINT_THRESHOLD}"
    assert recovery_lines(render_result(result)) == []


def test_attempt_details_read_recovered_attempt_numbers_from_the_rows_own_arm(tmp_path):
    result, agent = recovered_run_report(tmp_path)
    # Give the baseline arm the same recovered case, then drop the with-skill recovery list
    # (compacted payloads keep the per-case count but not the bounded list).
    agent["conditions"]["without_skill"] = copy.deepcopy(agent["conditions"]["with_skill"])
    agent["pass_at_k"]["without_skill"] = copy.deepcopy(agent["pass_at_k"]["with_skill"])
    del agent["conditions"]["with_skill"]["recovered_attempts"]

    html = HTMLReporter(report_scope="tier3").render(result)

    assert attempt_details_rows(html) == [
        "opencode With skill case-a A2: 1.00 pass 1 timed out, recovered 1 of 3 scored 1.00 passed on attempt 2",
        "opencode Without skill case-a A1: timed out, recovered A2: 1.00 pass 1 of 3 scored 1.00 passed on attempt 2",
    ]


def test_attempt_details_hint_does_not_promise_an_attempt_number_for_a_recovered_count(tmp_path):
    result, agent = recovered_run_report(tmp_path)
    # Compacting a non-best agent's conditions for a large multi-agent report keeps the
    # per-case count but drops the arm's recovery list, so the row can only show a count.
    assert _compact_agent_conditions(agent)
    assert "recovered_attempts" not in agent["conditions"]["with_skill"]

    html = HTMLReporter(report_scope="tier3").render(result)

    assert attempt_details_rows(html) == [
        "opencode With skill case-a A2: 1.00 pass 1 timed out, recovered 1 of 3 scored 1.00 passed on attempt 2"
    ]
    assert attempt_details_hint(html) == RECOVERED_HINT
    assert "keeps its attempt number" not in html


@pytest.mark.parametrize(
    ("attempts_used", "notice"),
    [(1, "scored attempt details omitted"), (0, "no attempts recorded")],
)
def test_attempt_details_say_when_a_recovered_case_has_no_attempt_rows(tmp_path, attempts_used, notice):
    result, agent = recovered_run_report(tmp_path)
    # The collector publishes a bounded number of attempt rows per run, so a later case keeps
    # its counts and recovery but can arrive without rows.
    agent["pass_at_k"]["with_skill"]["cases"]["case-a"].update(attempts=[], attempts_used=attempts_used)

    html = HTMLReporter(report_scope="tier3").render(result)

    assert attempt_details_rows(html) == [
        f"opencode With skill case-a A1: timed out, recovered {notice} "
        f"{attempts_used} of {attempts_used + 2} scored 1.00 passed on attempt 2"
    ]


@pytest.mark.parametrize("attempts_recovered", ["1", -3, True])
def test_attempt_details_ignore_an_invalid_recovered_count(tmp_path, attempts_recovered):
    result, agent = recovered_run_report(tmp_path)
    agent["pass_at_k"]["with_skill"]["cases"]["case-a"]["attempts_recovered"] = attempts_recovered

    html = HTMLReporter(report_scope="tier3").render(result)

    # Read like a case without recovery: no recovered chip, tooltip or hint, and the
    # denominator is the scored and skipped attempts only.
    assert attempt_details_rows(html) == [
        "opencode With skill case-a A2: 1.00 pass 1 of 2 scored 1.00 passed on attempt 2"
    ]
    assert attempt_details_hint(html) == f"{HINT_START} only; {HINT_THRESHOLD}"
    assert "recovered agent timeout" not in html


def test_tier3_evaluate_prints_the_collected_recovery_line(tmp_path):
    result = collect(
        tmp_path,
        native_attempt(tmp_path / "attempts", 1, timeout=True),
        native_attempt(tmp_path / "attempts", 2),
    )

    rendered = render_result(result)

    assert recovery_lines(rendered) == [
        "opencode: 1 agent timeout recovered by a later scored attempt (1 with skill); "
        "the timed-out attempt is not scored and stays in the run history."
    ]
    assert "AgentTimeoutError" not in rendered


def test_tier3_evaluate_prints_one_recovery_line_per_agent_with_recovered_timeouts():
    result = {
        "execution_status": "succeeded",
        "agents": {
            "both": {
                "conditions": {
                    "with_skill": {"recovered_attempts_total": 2},
                    "without_skill": {"recovered_attempts_total": 1},
                }
            },
            "baseline-only": {"conditions": {"without_skill": {"recovered_attempts_total": 1}}},
            "none": {"conditions": {"with_skill": {"recovered_attempts_total": 0}}},
            "boolean": {"conditions": {"with_skill": {"recovered_attempts_total": True}}},
            "text": {"conditions": {"with_skill": {"recovered_attempts_total": "2"}}},
            "no-conditions": {"conditions": None},
        },
    }

    # Both arms add up; zero, boolean, text and missing counts print nothing.
    assert recovery_lines(render_result(result)) == [
        "both: 3 agent timeouts recovered by a later scored attempt (2 with skill, 1 baseline); "
        "timed-out attempts are not scored and stay in the run history.",
        "baseline-only: 1 agent timeout recovered by a later scored attempt (1 baseline); "
        "the timed-out attempt is not scored and stays in the run history.",
    ]
