# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tolerance for baseline-only agent timeouts (``--tolerate-baseline-timeouts``).

The collector runs on synthetic Harbor jobs: arm-level jobs as Harbor writes
them, and stop-on-pass attempt jobs merged by the runner. The default stays
off, so a baseline timeout still fails the agent and the run unless a user
opts in.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from click.testing import CliRunner

from skillevaluator.tier3.harbor import baseline_timeouts
from skillevaluator.tier3.harbor.baseline_timeouts import (
    RECORD_KEY,
    REJECTION_KEY,
    RUN_KEY,
    build_record,
    check_failures,
    select_tolerated,
    settle,
    unexplained_launch_errors,
)
from skillevaluator.tier3.harbor.collector import collect_harbor_results

SKILL = "demo"
CASES = ["case-001", "case-002"]
SCORES = {
    "security": 1.0,
    "skill_execution": 1.0,
    "skill_efficiency": 0.9,
    "accuracy": 0.9,
    "goal_accuracy": 0.8,
    "behavior_check": 1.0,
}
TIMEOUT = "Agent execution timed out after 1200.0 seconds"
CLEAN_LOG = json.dumps({"type": "assistant", "message": "Reading the configuration"}) + "\n"
LAUNCH_ERROR = "{agent} without-skill Harbor run failed: Harbor job did not complete successfully: {count} errored"


# -- synthetic arm-level Harbor jobs ------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _trial_result(name: str, case: str, **fields: Any) -> dict[str, Any]:
    return {
        "trial_name": name,
        "task_name": case,
        "task_id": {"path": f"/tasks/{case}"},
        "config": {"task": {"path": f"/tasks/{case}"}},
        "started_at": "2026-11-01T00:00:00+00:00",
        "finished_at": "2026-11-01T00:20:00+00:00",
        "exception_info": None,
        **fields,
    }


def _scored_trial(job: Path, name: str, case: str) -> None:
    reward = {**SCORES, "entry_id": case}
    _write_json(job / name / "verifier" / "reward.json", reward)
    _write_json(
        job / name / "result.json",
        _trial_result(name, case, agent_result={"n_output_tokens": 500}, verifier_result={"rewards": SCORES}),
    )


def _failed_trial(
    job: Path,
    name: str,
    case: str,
    *,
    exception_type: str = "AgentTimeoutError",
    output_tokens: int | None = 321,
    log: str | None = CLEAN_LOG,
    log_name: str = "opencode.txt",
) -> None:
    trial = job / name
    _write_json(
        trial / "result.json",
        _trial_result(
            name,
            case,
            agent_result={"n_output_tokens": output_tokens} if output_tokens is not None else None,
            exception_info={"exception_type": exception_type, "exception_message": TIMEOUT},
        ),
    )
    (trial / "exception.txt").write_text(f"{exception_type}: {TIMEOUT}\n", encoding="utf-8")
    if log is not None:
        (trial / "agent").mkdir(parents=True, exist_ok=True)
        (trial / "agent" / log_name).write_text(log, encoding="utf-8")


def _job_result(job: Path, *, errored: int) -> None:
    trials = sorted(path.name for path in job.iterdir() if path.is_dir())
    if errored:
        stats: dict[str, Any] = {
            "n_completed_trials": len(trials),
            "n_errored_trials": errored,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
            "evals": {},
        }
    else:
        stats = {
            "n_trials": len(trials),
            "n_errors": 0,
            "evals": {
                "agent__model": {"n_trials": len(trials), "n_errors": 0, "reward_stats": {"reward": {"0.9": trials}}}
            },
        }
    _write_json(job / "result.json", {"n_total_trials": len(trials), "stats": stats})


def _arm(jobs: Path, agent: str, variant: str, outcomes: list[tuple[str, str]], **failed: Any) -> None:
    """Write one arm-level job; each outcome is ``(case, "scored" | "timeout")``."""
    job = jobs / f"{SKILL}-{agent}-{variant}"
    job.mkdir(parents=True, exist_ok=True)
    errored = 0
    for index, (case, outcome) in enumerate(outcomes):
        name = f"{case}__{variant}{index}"
        if outcome == "scored":
            _scored_trial(job, name, case)
        else:
            _failed_trial(job, name, case, **failed)
            errored += 1
    _job_result(job, errored=errored)


def _two_agent_jobs(jobs: Path, *, peer_with: str = "scored", **failed: Any) -> list[str]:
    """opencode's baseline times out on case-002; codex completes both arms. Returns launch errors."""
    _arm(jobs, "opencode", "with", [(case, "scored") for case in CASES])
    _arm(jobs, "opencode", "without", [("case-001", "scored"), ("case-002", "timeout")], **failed)
    _arm(jobs, "codex", "with", [("case-001", "scored"), ("case-002", peer_with)])
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES])
    errors = [LAUNCH_ERROR.format(agent="opencode", count=1)]
    if peer_with != "scored":
        errors.append("codex with-skill Harbor run failed: Harbor job did not complete successfully: 1 errored")
    return errors


def _collect(tmp_path: Path, launch_errors: list[str], *, agents: list[str] | None = None, **kwargs: Any):
    kwargs.setdefault("tolerate_baseline_timeouts", True)
    return collect_harbor_results(
        skill_name=SKILL,
        agents=agents or ["opencode", "codex"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        n_attempts=1,
        expected_cases=len(CASES),
        expected_case_ids=CASES,
        case_id_by_task_selector={case: case for case in CASES},
        expected_trials=len(CASES),
        launch_errors=launch_errors,
        **kwargs,
    )


# -- collector ------------------------------------------------------------------


def test_baseline_timeout_still_fails_the_agent_and_run_by_default(tmp_path: Path) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs")

    results = _collect(tmp_path, launch_errors, tolerate_baseline_timeouts=False)

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["execution_status"] == "failed"
    assert opencode["conditions"]["with_skill"]["execution_status"] == "succeeded"
    assert RECORD_KEY not in opencode
    assert REJECTION_KEY not in opencode
    assert RUN_KEY not in results
    assert "warnings" not in results


def test_tolerated_agent_stays_scored_on_its_with_skill_arm(tmp_path: Path) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs")

    results = _collect(tmp_path, launch_errors)

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "succeeded"
    assert results["execution_errors"] == []
    assert "error" not in results
    assert opencode["execution_status"] == "succeeded"
    assert opencode["execution_errors"] == []
    assert opencode["overall_with_skill"] is not None
    # The baseline stays honest: failed, with its errors, and never scored or compared.
    baseline = opencode["conditions"]["without_skill"]
    assert baseline["execution_status"] == "failed"
    assert (baseline["scored_attempts"], baseline["expected_attempts"]) == (1, 2)
    assert "Scored attempt coverage is 1/2" in baseline["execution_errors"]
    assert opencode["without_skill"] == {}
    assert opencode["overall_without_skill"] is None
    assert opencode["lift"] == {}
    assert opencode["pass_at_k"]["lift"] == {}
    assert opencode["security_attribution"] == {}
    assert (opencode["expected_attempts"], opencode["scored_attempts"]) == (4, 3)
    assert results["comparison"]["metrics"]["accuracy"]["opencode"]["without_skill"] is None
    assert results["comparison"]["metrics"]["accuracy"]["opencode"]["lift"] is None
    assert opencode[RECORD_KEY] == {
        "status": "partial",
        "reason": "agent_execution_timeout",
        "exception_type": "AgentTimeoutError",
        "expected_attempts": 2,
        "scored_attempts": 1,
        "failed_attempts": 1,
        "timeout_seconds": 1200.0,
        "failures": [{"trial": "case-002__without1", "case_id": "case-002", "attempt": None, "output_tokens": 321}],
        "failures_truncated": False,
    }
    assert results[RUN_KEY] == {"tolerated_agents": ["opencode"], "explained_launch_errors": launch_errors}
    assert results["warnings"] == [
        "opencode baseline incomplete: 1 attempt(s) hit the agent execution timeout after 1200 seconds "
        "(1 of 2 scored). The agent stays scored on its with-skill arm (--tolerate-baseline-timeouts); its "
        "baseline and uplift are not reported, and the run is not complete publication evidence."
    ]
    assert results["agents"]["codex"]["execution_status"] == "succeeded"
    assert results["agents"]["codex"]["lift"]


def test_reports_reload_the_tolerated_agent_from_disk(tmp_path: Path) -> None:
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
    from skillevaluator.reporting.base import (
        _agent_eval_attempt_coverage_complete,
        agent_eval_publication_evidence_complete,
    )
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    _collect(tmp_path, _two_agent_jobs(tmp_path / "jobs"))
    record_file = tmp_path / "results" / "opencode" / baseline_timeouts.RECORD_ARTIFACT

    agents = load_agent_data(tmp_path / "results")
    assert agents["opencode"]["execution_status"] == "succeeded"
    assert agents["opencode"]["execution_errors"] == []
    assert agents["opencode"]["conditions"]["without_skill"]["execution_status"] == "failed"
    assert agents["opencode"][RECORD_KEY]["failed_attempts"] == 1
    payload = build_agent_eval_payload(
        SKILL,
        agents,
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5, "stop_on_pass": False},
        use_llm_judge=False,
    )
    assert payload is not None
    assert payload["execution_status"] == "succeeded"
    assert payload["overall_score"] is not None
    assert payload["agents"]["opencode"]["baseline"] is None
    assert payload["agents"]["opencode"]["lift"] is None
    # The missing baseline attempts still keep the run out of publication evidence
    # and short of the attempt coverage a blocking Tier 3 gate requires.
    assert agent_eval_publication_evidence_complete(payload) is False
    assert _agent_eval_attempt_coverage_complete(payload, require_baseline=False) is False

    # A record that no longer matches the baseline arm on disk is ignored.
    record = json.loads(record_file.read_text(encoding="utf-8"))
    record_file.write_text(json.dumps({**record, "scored_attempts": 0}), encoding="utf-8")
    assert load_agent_data(tmp_path / "results")["opencode"]["execution_status"] == "failed"

    # A rerun without the option removes the stale record.
    _collect(tmp_path, _two_agent_jobs(tmp_path / "jobs"), tolerate_baseline_timeouts=False)
    assert not record_file.exists()
    assert load_agent_data(tmp_path / "results")["opencode"]["execution_status"] == "failed"


def test_trajectory_completion_tokens_prove_progress(tmp_path: Path) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs", output_tokens=None)
    trial = tmp_path / "jobs" / f"{SKILL}-opencode-without" / "case-002__without1"
    _write_json(trial / "agent" / "trajectory.json", {"steps": [], "final_metrics": {"total_completion_tokens": 42}})

    results = _collect(tmp_path, launch_errors)

    assert results["execution_status"] == "succeeded"
    assert results["agents"]["opencode"][RECORD_KEY]["failures"][0]["output_tokens"] == 42


@pytest.mark.parametrize(
    ("failed", "code"),
    [
        ({"output_tokens": None}, "no_progress_evidence"),
        ({"output_tokens": 0}, "no_progress_evidence"),
        ({"log": None}, "no_agent_log"),
        (
            {"log": CLEAN_LOG + '{"type":"error","error":{"type":"rate_limit_error"}}\n'},
            "provider_degradation_observed",
        ),
        (
            {"log": CLEAN_LOG + '{"type":"system","subtype":"api_retry","error_status":529}\n'},
            "provider_degradation_observed",
        ),
        ({"log": CLEAN_LOG + "unexpected status: 429 Too Many Requests\n"}, "provider_degradation_observed"),
        ({"log": CLEAN_LOG + "API Error: 529 Overloaded\n"}, "agent_runtime_failure"),
        ({"log": CLEAN_LOG + "Error: invalid_api_key\n"}, "agent_runtime_failure"),
        ({"exception_type": "VerifierTimeoutError"}, "reason_not_tolerated"),
        ({"exception_type": "NonZeroAgentExitCodeError"}, "reason_not_tolerated"),
    ],
)
def test_untrusted_baseline_timeouts_keep_the_agent_failed(tmp_path: Path, failed: dict[str, Any], code: str) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs", **failed)

    results = _collect(tmp_path, launch_errors)

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["execution_status"] == "failed"
    assert RECORD_KEY not in opencode
    assert opencode[REJECTION_KEY] == code
    assert results[RUN_KEY] == {"tolerated_agents": [], "explained_launch_errors": []}
    assert results["warnings"] == [
        f"opencode baseline timeouts were not tolerated: {baseline_timeouts.REJECTION_REASONS[code]} ({code})."
    ]


def test_trial_that_is_neither_scored_nor_timed_out_is_refused(tmp_path: Path) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs")
    (tmp_path / "jobs" / f"{SKILL}-opencode-without" / "case-002__stray" / "agent").mkdir(parents=True)

    results = _collect(tmp_path, launch_errors)

    assert results["agents"]["opencode"][REJECTION_KEY] == "unaccounted_trial"
    assert results["execution_status"] == "failed"


def test_errored_count_without_matching_trials_is_refused(tmp_path: Path) -> None:
    _two_agent_jobs(tmp_path / "jobs")
    job = tmp_path / "jobs" / f"{SKILL}-opencode-without"
    # Harbor's statistics claim a second errored trial that no trial accounts for,
    # so the scored trial is not kept either and the arm holds an untyped trial.
    payload = json.loads((job / "result.json").read_text(encoding="utf-8"))
    payload["stats"]["n_errored_trials"] = 2
    (job / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    results = _collect(tmp_path, [LAUNCH_ERROR.format(agent="opencode", count=2)])

    assert results["agents"]["opencode"][REJECTION_KEY] == "reason_not_tolerated"
    assert results["execution_status"] == "failed"


def _evidence(job: Path, **overrides: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Run the baseline evidence check on the arm-level fixture, with the summary inputs it receives."""
    from skillevaluator.tier3.harbor import collector

    rewards, _invalid = collector._partition_scoreable_rewards(
        collector._extract_rewards(job, {case: case for case in CASES})
    )
    kwargs: dict[str, Any] = {
        "job_dir": job,
        "rewards": rewards,
        "execution": {"expected_attempts": 2, "scored_attempts": 1},
        "slots": {
            "case-001": {"required": 1, "missing": 0, "missing_ordinals": None},
            "case-002": {"required": 1, "missing": 1, "missing_ordinals": None},
        },
        "residual_errors": ["Missing scored attempts for cases: case-002", "Scored attempt coverage is 1/2"],
        "job_failure": "Harbor job did not complete successfully: 1 errored",
        "runtime_failures": collector._extract_agent_runtime_failures(job),
        "trial_failures": collector._extract_trial_failures(job),
        "recovered": [],
        "expected_case_ids": CASES,
        "case_id_by_task_selector": {case: case for case in CASES},
        "arm_suffix": "",
        "task_entry_id_map": None,
    }
    kwargs.update(overrides)
    return collector._baseline_timeout_failures(**kwargs)


def test_baseline_evidence_binds_every_error_to_the_timeouts(tmp_path: Path) -> None:
    _two_agent_jobs(tmp_path / "jobs")
    job = tmp_path / "jobs" / f"{SKILL}-opencode-without"

    failures, code = _evidence(job)
    assert code is None
    assert [(failure["trial"], failure["case_id"], failure["output_tokens"]) for failure in failures] == [
        ("case-002__without1", "case-002", 321)
    ]

    timed_out = {"trial": "case-002__without1", "reason": f"AgentTimeoutError: {TIMEOUT}"}
    other = {"trial": "case-001__without0", "reason": "Reward metrics are incomplete or non-finite"}
    assert _evidence(job, job_failure="Harbor job did not complete successfully: 1 running")[1] == "unbound_error"
    assert _evidence(job, residual_errors=["Unexpected scored cases: case-009"])[1] == "unbound_error"
    assert _evidence(job, trial_failures=[timed_out, other])[1] == "untyped_trial_failure"
    assert _evidence(job, trial_failures=[])[1] == "untyped_trial_failure"
    assert _evidence(job, runtime_failures=[{**timed_out, "reason": "rate limited"}])[1] == "agent_runtime_failure"
    assert _evidence(job, expected_case_ids=[])[1] == "no_case_identity"
    assert _evidence(job, case_id_by_task_selector={"case-001": "case-001"})[1] == "per_case_mismatch"


def test_failed_peer_keeps_every_agent_failed(tmp_path: Path) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs", peer_with="timeout")

    results = _collect(tmp_path, launch_errors)

    assert results["execution_status"] == "failed"
    assert results["agents"]["opencode"]["execution_status"] == "failed"
    assert results["agents"]["opencode"][REJECTION_KEY] == "other_agent_failed"
    # codex's with-skill arm failed, so it was never a candidate.
    assert REJECTION_KEY not in results["agents"]["codex"]


def test_single_agent_has_no_clean_peer(tmp_path: Path) -> None:
    _two_agent_jobs(tmp_path / "jobs")

    results = _collect(tmp_path, [LAUNCH_ERROR.format(agent="opencode", count=1)], agents=["opencode"])

    assert results["execution_status"] == "failed"
    assert results["agents"]["opencode"][REJECTION_KEY] == "no_clean_peer"


@pytest.mark.parametrize(
    "extra",
    [
        "opencode with-skill Harbor run failed: harbor run exited 1",
        "Harbor job directory was not created: demo-opencode-with",
    ],
)
def test_other_launch_errors_keep_the_run_failed(tmp_path: Path, extra: str) -> None:
    launch_errors = [*_two_agent_jobs(tmp_path / "jobs"), extra]

    results = _collect(tmp_path, launch_errors)

    assert results["agents"]["opencode"]["execution_status"] == "failed"
    assert results["agents"]["opencode"][REJECTION_KEY] == "unbound_launch_error"
    assert results[RUN_KEY]["explained_launch_errors"] == []


@pytest.mark.parametrize("kwargs", [{"grading_mode": "custom_only"}, {"skip_baseline": True}])
def test_tolerance_needs_a_baseline_and_dimension_grading(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs")

    results = _collect(tmp_path, launch_errors, **kwargs)

    assert RUN_KEY not in results
    assert all(RECORD_KEY not in agent and REJECTION_KEY not in agent for agent in results["agents"].values())


# -- stop-on-pass attempt jobs merged by the runner ---------------------------


def _attempt_job(root: Path, agent: str, variant: str, case: str, attempt: int, outcome: str) -> Path:
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult

    job = root / "attempts" / f"{SKILL}-{agent}-{variant}-{case}-attempt{attempt:03d}"
    trial = job / f"{case}__{attempt:07d}"
    trial.mkdir(parents=True)
    start = datetime(2026, 11, 1, tzinfo=UTC) + timedelta(minutes=attempt)
    finish = start + timedelta(seconds=30)
    timed_out = outcome == "timeout"
    score = 1.0 if outcome == "pass" else 0.2
    result = TrialResult.model_validate(
        {
            "id": uuid4(),
            "task_name": case,
            "trial_name": trial.name,
            "trial_uri": trial.as_uri(),
            "task_id": {"path": str(root / "task" / case)},
            "task_checksum": "public-tolerance-example",
            "config": {"task": {"path": str(root / "task" / case)}, "trial_name": trial.name, "trials_dir": str(job)},
            "agent_info": {"name": agent, "version": "test", "model_info": {"name": "test-model"}},
            "agent_result": {"n_output_tokens": 77},
            "started_at": start,
            "finished_at": finish,
            "agent_execution": {"started_at": start, "finished_at": finish},
            "exception_info": {
                "exception_type": "AgentTimeoutError",
                "exception_message": TIMEOUT,
                "exception_traceback": "in _run_agent_phase\nraise AgentTimeoutError(...)",
                "occurred_at": finish,
            }
            if timed_out
            else None,
            "verifier_result": None if timed_out else {"rewards": dict.fromkeys(SCORES, score)},
        }
    )
    (trial / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (trial / "agent").mkdir()
    (trial / "agent" / f"{agent}.txt").write_text(CLEAN_LOG, encoding="utf-8")
    job_result = JobResult(
        id=uuid4(),
        started_at=start,
        updated_at=finish,
        finished_at=finish,
        n_total_trials=1,
        stats=JobStats.from_trial_results([result], n_total_trials=1),
        trial_results=[result],
    )
    (job / "result.json").write_text(job_result.model_dump_json(indent=2), encoding="utf-8")
    return job


def test_stop_on_pass_timeouts_are_matched_attempt_by_attempt(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    from skillevaluator.tier3.harbor.runner import _merge_attempt_jobs

    plan = {
        ("opencode", "with"): ["pass"],
        ("opencode", "without"): ["timeout", "timeout"],
        ("codex", "with"): ["pass"],
        ("codex", "without"): ["fail", "fail"],
    }
    for (agent, variant), outcomes in plan.items():
        jobs = [
            _attempt_job(tmp_path, agent, variant, "case-a", attempt, outcome)
            for attempt, outcome in enumerate(outcomes, start=1)
        ]
        _merge_attempt_jobs(jobs, tmp_path / "jobs" / f"{SKILL}-{agent}-{variant}")
    launch_errors = [
        f"opencode without-skill Harbor run failed: case-a attempt {attempt}: "
        "Harbor job did not complete successfully: 1 errored"
        for attempt in (1, 2)
    ]

    def collect(**kwargs: Any) -> dict[str, Any]:
        return collect_harbor_results(
            skill_name=SKILL,
            agents=["opencode", "codex"],
            output_dir=tmp_path / "results",
            jobs_dir=tmp_path / "jobs",
            n_attempts=2,
            stop_on_pass=True,
            expected_cases=1,
            expected_case_ids=["case-a"],
            case_id_by_task_selector={"case-a": "case-a"},
            **kwargs,
        )

    results = collect(launch_errors=launch_errors, tolerate_baseline_timeouts=True)

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "succeeded", results.get("warnings")
    assert opencode["execution_status"] == "succeeded"
    record = opencode[RECORD_KEY]
    assert record["status"] == "unavailable"
    assert (record["scored_attempts"], record["expected_attempts"], record["failed_attempts"]) == (0, 2, 2)
    assert [failure["attempt"] for failure in record["failures"]] == [1, 2]
    assert results[RUN_KEY]["explained_launch_errors"] == launch_errors

    # One attempt's launch error the timeouts cannot explain keeps the run failed.
    wrong = [launch_errors[0], launch_errors[0].replace("attempt 1", "attempt 3")]
    refused = collect(launch_errors=wrong, tolerate_baseline_timeouts=True)
    assert refused["agents"]["opencode"][REJECTION_KEY] == "unbound_launch_error"
    assert refused["execution_status"] == "failed"


# -- pure rules -----------------------------------------------------------------


def _failure(case: str = "case-a", attempt: int | None = None, **fields: Any) -> dict[str, Any]:
    return {
        "trial": f"{case}__{attempt or 0}",
        "case_id": case,
        "task": case,
        "attempt": attempt,
        "exception_types": ["AgentTimeoutError"],
        "output_tokens": 10,
        **fields,
    }


def _slots(**cases: tuple[int, list[int] | None]) -> dict[str, dict[str, Any]]:
    return {
        case.replace("_", "-"): {"required": 2, "missing": missing, "missing_ordinals": ordinals}
        for case, (missing, ordinals) in cases.items()
    }


@pytest.mark.parametrize(
    ("expected", "scored", "slots", "failures", "code"),
    [
        (2, 1, _slots(case_a=(1, None)), [_failure()], None),
        (2, 0, _slots(case_a=(2, [1, 2])), [_failure(attempt=2), _failure(attempt=1)], None),
        (2, 2, _slots(case_a=(0, None)), [], "baseline_not_short"),
        (
            2,
            1,
            _slots(case_a=(1, None)),
            [_failure(exception_types=["AgentSetupTimeoutError"])],
            "reason_not_tolerated",
        ),
        (
            2,
            1,
            _slots(case_a=(1, None)),
            [_failure(exception_types=["AgentTimeoutError", "VerifierTimeoutError"])],
            "reason_not_tolerated",
        ),
        (2, 1, _slots(case_a=(1, None)), [_failure(exception_types=[])], "reason_not_tolerated"),
        (2, 0, _slots(case_a=(2, None)), [_failure()], "failures_do_not_account_for_missing"),
        (2, 1, _slots(case_a=(1, None)), [_failure(case="case-b")], "per_case_mismatch"),
        (2, 1, _slots(case_a=(1, [2])), [_failure(attempt=1)], "ordinal_mismatch"),
        (2, 1, _slots(case_a=(1, [2])), [_failure(attempt=None)], "ordinal_mismatch"),
        (2, 1, _slots(case_a=(1, None)), [_failure(output_tokens=0)], "no_progress_evidence"),
        (2, 1, _slots(case_a=(1, None)), [_failure(output_tokens=True)], "no_progress_evidence"),
    ],
)
def test_check_failures(expected, scored, slots, failures, code) -> None:
    assert check_failures(expected_attempts=expected, scored_attempts=scored, slots=slots, failures=failures) == code


def _agent(status: str = "succeeded", baseline: str = "succeeded") -> dict[str, Any]:
    return {"execution_status": status, "conditions": {"without_skill": {"execution_status": baseline}}}


def test_select_tolerated_is_all_or_nothing() -> None:
    agents = {"opencode": _agent("failed", "failed"), "codex": _agent(), "codex-cli": _agent()}
    candidates = {"opencode": [_failure(attempt=1)]}
    arm_level = LAUNCH_ERROR.format(agent="opencode", count=1)

    assert select_tolerated(agents, candidates, {}, [arm_level]) == (frozenset({"opencode"}), {}, [arm_level])
    # An agent whose name extends a candidate's never lends it a launch error.
    other = "codex-cli without-skill Harbor run failed: harbor run exited 1"
    tolerated, rejected, explained = select_tolerated(
        {**agents, "codex-cli": _agent("failed")}, candidates, {}, [arm_level, other]
    )
    assert (tolerated, explained) == (frozenset(), [])
    assert rejected == {"opencode": "other_agent_failed"}
    # A refused candidate fails the run, so no other candidate is tolerated either.
    agents["claude-code"] = _agent("failed", "failed")
    assert select_tolerated(agents, candidates, {"claude-code": "no_agent_log"}, [arm_level])[:2] == (
        frozenset(),
        {"claude-code": "no_agent_log", "opencode": "other_agent_failed"},
    )


def test_select_tolerated_needs_a_peer_that_completed_both_arms() -> None:
    agents = {"opencode": _agent("failed", "failed"), "codex": _agent(baseline="skipped")}
    tolerated, rejected, _explained = select_tolerated(agents, {"opencode": [_failure()]}, {}, [])
    assert tolerated == frozenset()
    assert rejected == {"opencode": "no_clean_peer"}


@pytest.mark.parametrize(
    "errors",
    [
        [LAUNCH_ERROR.format(agent="opencode", count=2)],
        [LAUNCH_ERROR.format(agent="opencode", count=1)] * 2,
        ["opencode runtime preflight failed: model unavailable"],
        ["Harbor execution failed before any agent ran"],
    ],
)
def test_select_tolerated_refuses_launch_errors_it_cannot_bind(errors: list[str]) -> None:
    agents = {"opencode": _agent("failed", "failed"), "codex": _agent()}
    tolerated, rejected, explained = select_tolerated(agents, {"opencode": [_failure()]}, {}, errors)
    assert (tolerated, explained) == (frozenset(), [])
    assert rejected == {"opencode": "unbound_launch_error"}


def test_settle_and_unexplained_launch_errors() -> None:
    record = build_record(
        expected_attempts=40,
        scored_attempts=0,
        failures=[_failure(attempt=n) for n in range(1, 41)],
        timeout_seconds=None,
    )
    assert record["failed_attempts"] == 40
    assert len(record["failures"]) == baseline_timeouts.FAILURES_PUBLISHED_MAX
    assert record["failures_truncated"] is True
    assert record["status"] == "unavailable"

    agent = {
        "execution_status": "failed",
        "execution_errors": ["Scored attempt coverage is 0/40"],
        "execution_error_details_total": 1,
        REJECTION_KEY: "stale",
        "conditions": {"without_skill": {"execution_status": "failed", "execution_errors": ["kept"]}},
    }
    settle(agent, record)
    assert agent["execution_status"] == "succeeded"
    assert agent["execution_errors"] == []
    assert agent["execution_error_details_total"] == 0
    assert REJECTION_KEY not in agent
    assert agent["conditions"]["without_skill"] == {"execution_status": "failed", "execution_errors": ["kept"]}

    error = LAUNCH_ERROR.format(agent="opencode", count=1)
    results = {RUN_KEY: {"explained_launch_errors": [error]}}
    assert unexplained_launch_errors(results, [error, error, "other"]) == [error, "other"]
    assert unexplained_launch_errors({}, [error]) == [error]


# -- engine and CLI plumbing ----------------------------------------------------


def _provider():
    from skillevaluator.provider_config import ProviderConfig

    return ProviderConfig(
        provider="nv_build",
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )


def _run_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_yaml: str, **engine_kwargs: Any):
    """Run ``run_harbor_eval`` with Harbor faked and the real collector."""
    from skillevaluator.tier3.harbor import runner, runtime_preflight

    skill = tmp_path / SKILL
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(config_yaml, encoding="utf-8")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(runner, "resolve_llm_provider", _provider)
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    def emit(_skill: Path, target: Path, **_kwargs: Any) -> list[Path]:
        tasks = []
        for case_id in CASES:
            task = target / case_id
            (task / "tests").mkdir(parents=True)
            (task / "tests" / "entry.json").write_text(json.dumps({"id": case_id}), encoding="utf-8")
            tasks.append(task)
        return tasks

    def pair(**kwargs: Any) -> list[str]:
        agent, jobs_dir = kwargs["agent"], kwargs["jobs_dir"]
        _arm(jobs_dir, agent, "with", [(case, "scored") for case in CASES])
        if agent == "opencode":
            _arm(jobs_dir, agent, "without", [("case-001", "scored"), ("case-002", "timeout")])
            return [LAUNCH_ERROR.format(agent=agent, count=1)]
        _arm(jobs_dir, agent, "without", [(case, "scored") for case in CASES])
        return []

    def collect(**kwargs: Any) -> dict[str, Any]:
        captured["collect"] = kwargs
        return collect_harbor_results(**kwargs)

    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    monkeypatch.setattr(runner, "_run_agent_pair", pair)
    monkeypatch.setattr(runner, "collect_harbor_results", collect)
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: runtime_preflight.ModelProbeResult(True, selected.provider, selected.model, "available"),
    )
    result = runner.run_harbor_eval(
        skill,
        ["opencode", "codex"],
        output_dir=tmp_path / "results",
        env_mode="docker",
        agent_runtime_preflight=False,
        keep_harbor_jobs=True,
        **engine_kwargs,
    )
    return result, captured


CONFIG = "schema_version: 1\nharbor:\n  task_source: evals_json\n"


def test_engine_default_merges_the_baseline_launch_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, captured = _run_engine(monkeypatch, tmp_path, CONFIG)

    assert captured["collect"]["tolerate_baseline_timeouts"] is False
    assert result["execution_status"] == "failed"
    assert LAUNCH_ERROR.format(agent="opencode", count=1) in result["execution_errors"]
    assert "tolerate_baseline_timeouts" not in result["run_config"]["harbor"]


def test_engine_config_key_tolerates_and_drops_the_explained_launch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, captured = _run_engine(monkeypatch, tmp_path, CONFIG + "  tolerate_baseline_timeouts: true\n")

    assert captured["collect"]["tolerate_baseline_timeouts"] is True
    assert result["execution_status"] == "succeeded", result.get("execution_errors")
    assert "error" not in result
    assert result["execution_errors"] == []
    assert result["agents"]["opencode"]["execution_status"] == "succeeded"
    assert result["run_config"]["harbor"]["tolerate_baseline_timeouts"] is True
    assert any(warning.startswith("opencode baseline incomplete") for warning in result["warnings"])
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert persisted["agents"]["opencode"][RECORD_KEY]["failed_attempts"] == 1


def test_engine_argument_overrides_the_config_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, captured = _run_engine(
        monkeypatch,
        tmp_path,
        CONFIG + "  tolerate_baseline_timeouts: true\n",
        tolerate_baseline_timeouts=False,
    )

    assert captured["collect"]["tolerate_baseline_timeouts"] is False
    assert result["execution_status"] == "failed"


def test_config_key_must_be_a_boolean(tmp_path: Path) -> None:
    from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  tolerate_baseline_timeouts: sometimes\n", encoding="utf-8"
    )
    with pytest.raises(EvalsConfigError, match=r"harbor\.tolerate_baseline_timeouts"):
        load_evals_config(tmp_path)


@pytest.mark.parametrize(
    ("flag", "expected"), [("--tolerate-baseline-timeouts", True), ("--no-tolerate-baseline-timeouts", False)]
)
def test_validate_forwards_the_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str, expected: bool) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    captured: dict[str, Any] = {}

    def tier3(*_args: Any, **kwargs: Any) -> ValidationResult:
        captured.update(kwargs)
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", tier3)
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "skills" / "simple"
    result = CliRunner().invoke(
        cli_module.cli,
        [
            "validate",
            "--no-autopilot",
            str(fixture),
            "--no-llm",
            "--no-tier2",
            "--tier3",
            flag,
            "--checks",
            "schema",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["tolerate_baseline_timeouts"] is expected


def test_evaluation_options_reach_the_engine() -> None:
    import inspect

    from skillevaluator.evaluation import EvaluationOptions
    from skillevaluator.tier3 import commands

    options = EvaluationOptions(skill_path=Path("demo"), tolerate_baseline_timeouts=True)
    assert options.engine_kwargs()["tolerate_baseline_timeouts"] is True
    assert "tolerate_baseline_timeouts" in inspect.signature(commands.evaluate).parameters
