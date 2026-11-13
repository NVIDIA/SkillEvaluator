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
    run_record,
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
# The runner's message when it kills a Harbor job that outlived the Google ADC token on GKE.
ADC_LAUNCH_ERROR = (
    "opencode without-skill Harbor run failed: Harbor job exceeded Google ADC token lifetime (3300s) in gke mode"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _doc_block(name: str, lead: str) -> str:
    """Return the one blank-line separated block of ``name`` that starts with ``lead``, whitespace-normalized."""
    blocks = [" ".join(block.split()) for block in (REPO_ROOT / name).read_text(encoding="utf-8").split("\n\n")]
    [block] = [block for block in blocks if block.startswith(lead)]
    return block


def _changelog_entry(lead: str) -> str:
    """Return the one CHANGELOG.md bullet that starts with ``lead``, whitespace-normalized."""
    entries: list[list[str]] = []
    in_entry = False
    for line in (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            entries.append([line])
            in_entry = True
        elif in_entry and line.startswith("  "):
            entries[-1].append(line)
        else:
            in_entry = False
    [entry] = [text for text in (" ".join(" ".join(lines).split()) for lines in entries) if text.startswith(lead)]
    return entry


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


def _activation(activated: bool) -> dict[str, Any]:
    """Reward details recording whether the with-skill agent activated the skill."""
    score = 1.0 if activated else 0.0
    return {"skill_execution": {"activation": {"supported": True, "score": score, "passed": activated}}}


def _scored_trial(
    job: Path, name: str, case: str, scores: dict[str, float] | None = None, details: dict[str, Any] | None = None
) -> None:
    scores = scores or SCORES
    reward = {**scores, "entry_id": case, **({"details": details} if details is not None else {})}
    _write_json(job / name / "verifier" / "reward.json", reward)
    _write_json(
        job / name / "result.json",
        _trial_result(name, case, agent_result={"n_output_tokens": 500}, verifier_result={"rewards": scores}),
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


def _arm(
    jobs: Path,
    agent: str,
    variant: str,
    outcomes: list[tuple[str, str]],
    *,
    scores: dict[str, float] | None = None,
    details: dict[str, Any] | None = None,
    **failed: Any,
) -> None:
    """Write one arm-level job; each outcome is ``(case, "scored" | "timeout")``."""
    job = jobs / f"{SKILL}-{agent}-{variant}"
    job.mkdir(parents=True, exist_ok=True)
    errored = 0
    for index, (case, outcome) in enumerate(outcomes):
        name = f"{case}__{variant}{index}"
        if outcome == "scored":
            _scored_trial(job, name, case, scores, details)
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
    assert results[RUN_KEY] == {
        "tolerated_agents": ["opencode"],
        "explained_launch_errors": launch_errors,
        "explained_launch_errors_total": 1,
        "explained_launch_errors_truncated": False,
    }
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


def _completion_counter_jobs(tmp_path: Path, source: str, count: Any) -> list[str]:
    """Return the launch errors of jobs whose timed-out baseline attempt reports ``count`` output tokens."""
    launch_errors = _two_agent_jobs(tmp_path / "jobs", output_tokens=count if source == "harbor_result" else None)
    if source == "trajectory":
        trial = tmp_path / "jobs" / f"{SKILL}-opencode-without" / "case-002__without1"
        _write_json(
            trial / "agent" / "trajectory.json",
            {"steps": [], "final_metrics": {"total_completion_tokens": count}},
        )
    return launch_errors


@pytest.mark.parametrize("source", ["harbor_result", "trajectory"])
# Counts past the float range (either sign), past the JSON-safe integer range, and large floats.
@pytest.mark.parametrize(
    "count",
    [10**400, -(10**400), 2**53, 2**70, 1e16, 1e300],
    ids=["int-10pow400", "int-minus-10pow400", "int-2pow53", "int-2pow70", "float-1e16", "float-1e300"],
)
def test_oversized_completion_counter_rejects_tolerance_without_aborting_collection(
    tmp_path: Path, source: str, count: Any
) -> None:
    launch_errors = _completion_counter_jobs(tmp_path, source, count)

    results = _collect(tmp_path, launch_errors)

    assert results["execution_status"] == "failed"
    assert RECORD_KEY not in results["agents"]["opencode"]
    assert results["agents"]["opencode"][REJECTION_KEY] == "no_progress_evidence"


@pytest.mark.parametrize("source", ["harbor_result", "trajectory"])
def test_completion_counter_at_the_json_safe_limit_proves_progress(tmp_path: Path, source: str) -> None:
    launch_errors = _completion_counter_jobs(tmp_path, source, 2**53 - 1)

    results = _collect(tmp_path, launch_errors)

    assert results["execution_status"] == "succeeded"
    assert results["agents"]["opencode"][RECORD_KEY]["failures"][0]["output_tokens"] == 2**53 - 1


def test_trajectory_completion_tokens_prove_progress(tmp_path: Path) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs", output_tokens=None)
    trial = tmp_path / "jobs" / f"{SKILL}-opencode-without" / "case-002__without1"
    _write_json(trial / "agent" / "trajectory.json", {"steps": [], "final_metrics": {"total_completion_tokens": 42}})

    results = _collect(tmp_path, launch_errors)

    assert results["execution_status"] == "succeeded"
    assert results["agents"]["opencode"][RECORD_KEY]["failures"][0]["output_tokens"] == 42


@pytest.mark.parametrize("count", [2**53, 2**70, 1e16], ids=["int-2pow53", "int-2pow70", "float-1e16"])
def test_out_of_range_harbor_counter_falls_back_to_the_trajectory_and_the_changelog_says_so(
    tmp_path: Path, count: Any
) -> None:
    """Harbor's count past the JSON-safe range is no evidence, so an in-range trajectory count still proves progress."""
    launch_errors = _two_agent_jobs(tmp_path / "jobs", output_tokens=count)
    trial = tmp_path / "jobs" / f"{SKILL}-opencode-without" / "case-002__without1"
    _write_json(trial / "agent" / "trajectory.json", {"steps": [], "final_metrics": {"total_completion_tokens": 42}})

    results = _collect(tmp_path, launch_errors)

    assert results["execution_status"] == "succeeded"
    assert results["agents"]["opencode"][RECORD_KEY]["failures"][0]["output_tokens"] == 42
    # The entry refuses the agent only when the attempt has no other count in range.
    entry = _changelog_entry("- `--tolerate-baseline-timeouts` (added in 0.9.0) now follows its own rules")
    assert "with no other in-range count for that attempt the agent is refused as `no_progress_evidence`" in entry


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
        # OpenAI-compatible providers through the OpenAI SDK, and OpenCode's APIError.
        (
            {
                "log": CLEAN_LOG + "Error code: 429 - {'status': 429, 'title': 'Too Many Requests', "
                "'detail': 'Rate limit exceeded'}\n"
            },
            "provider_degradation_observed",
        ),
        (
            {
                "log": CLEAN_LOG + '{"type":"error","error":{"name":"APIError","data":'
                '{"message":"Too Many Requests","statusCode":429}}}\n'
            },
            "provider_degradation_observed",
        ),
        ({"log": CLEAN_LOG + "429 Too Many Requests\n"}, "provider_degradation_observed"),
        ({"log": CLEAN_LOG + '{"error":{"code":"insufficient_quota"}}\n'}, "provider_degradation_observed"),
        ({"log": CLEAN_LOG + '{"error":{"status":"RESOURCE_EXHAUSTED"}}\n'}, "provider_degradation_observed"),
        ({"log": CLEAN_LOG + "API Error: 529 Overloaded\n"}, "agent_runtime_failure"),
        ({"log": CLEAN_LOG + "Error: invalid_api_key\n"}, "agent_runtime_failure"),
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
    assert results[RUN_KEY] == run_record((), [])
    assert results["warnings"] == [
        f"opencode baseline timeouts were not tolerated: {baseline_timeouts.REJECTION_REASONS[code]} ({code})."
    ]


@pytest.mark.parametrize("exception_type", ["VerifierTimeoutError", "NonZeroAgentExitCodeError"])
def test_baseline_without_an_agent_timeout_is_left_as_on_main(tmp_path: Path, exception_type: str) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs", exception_type=exception_type)

    results = _collect(tmp_path, launch_errors)

    # No attempt timed out, so the option has nothing to judge: no rejection
    # code or warning names a timeout that never happened.
    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["execution_status"] == "failed"
    assert RECORD_KEY not in opencode
    assert REJECTION_KEY not in opencode
    assert "warnings" not in results
    assert results[RUN_KEY] == run_record((), [])


def test_another_baseline_failure_beside_a_timeout_is_refused(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _arm(jobs, "opencode", "with", [(case, "scored") for case in CASES])
    _arm(jobs, "opencode", "without", [(case, "timeout") for case in CASES])
    # case-001 failed another way; case-002 still timed out.
    job = jobs / f"{SKILL}-opencode-without"
    _failed_trial(job, "case-001__without0", "case-001", exception_type="NonZeroAgentExitCodeError")
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES])
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES])

    results = _collect(tmp_path, [LAUNCH_ERROR.format(agent="opencode", count=2)])

    assert results["agents"]["opencode"][REJECTION_KEY] == "reason_not_tolerated"
    assert results["execution_status"] == "failed"


@pytest.mark.parametrize(
    ("killed", "code"),
    [
        # The runner killed Harbor while every attempt still ran: nothing timed out.
        ({"case-001": "cancelled", "case-002": "cancelled"}, None),
        # One attempt timed out, then the kill cancelled the other.
        ({"case-001": "cancelled", "case-002": "timeout"}, "reason_not_tolerated"),
        # One attempt timed out, then the kill left the other without a result.
        ({"case-001": "no_result", "case-002": "timeout"}, "unaccounted_trial"),
    ],
)
def test_baseline_killed_by_the_runner_is_never_tolerated(
    tmp_path: Path, killed: dict[str, str], code: str | None
) -> None:
    jobs = tmp_path / "jobs"
    _arm(jobs, "opencode", "with", [(case, "scored") for case in CASES])
    job = jobs / f"{SKILL}-opencode-without"
    for index, (case, outcome) in enumerate(killed.items()):
        name = f"{case}__without{index}"
        if outcome == "no_result":
            (job / name / "agent").mkdir(parents=True)
            (job / name / "agent" / "opencode.txt").write_text(CLEAN_LOG, encoding="utf-8")
        else:
            exception_type = "CancelledError" if outcome == "cancelled" else "AgentTimeoutError"
            _failed_trial(job, name, case, exception_type=exception_type)
    _job_result(job, errored=sum(outcome != "no_result" for outcome in killed.values()))
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES])
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES])

    results = _collect(tmp_path, [ADC_LAUNCH_ERROR])

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["execution_status"] == "failed"
    assert RECORD_KEY not in opencode
    assert opencode.get(REJECTION_KEY) == code
    assert results[RUN_KEY]["explained_launch_errors"] == []


def _multi_step_timeout_trial(job: Path, name: str, case: str) -> None:
    """Write a native multi-step trial as Harbor 0.24 records a step's agent timeout.

    ``MultiStepTrial._run_step_agent`` stores the exception on the step only,
    so the trial has no root exception and Harbor counts it as completed.
    """
    from harbor.models.trial.result import TrialResult

    trial = job / name
    finish = datetime(2026, 11, 1, 0, 20, tzinfo=UTC)
    result = TrialResult.model_validate(
        {
            "id": uuid4(),
            "task_name": case,
            "trial_name": name,
            "trial_uri": trial.as_uri(),
            "task_id": {"path": f"/tasks/{case}"},
            "task_checksum": "public-tolerance-example",
            "config": {"task": {"path": f"/tasks/{case}"}, "trial_name": name, "trials_dir": str(job)},
            "agent_info": {"name": "opencode", "version": "test", "model_info": {"name": "test-model"}},
            "started_at": finish - timedelta(minutes=20),
            "finished_at": finish,
            "exception_info": None,
            "step_results": [
                {
                    "step_name": "build",
                    "agent_result": {"n_output_tokens": 321},
                    "exception_info": {
                        "exception_type": "AgentTimeoutError",
                        "exception_message": TIMEOUT,
                        "exception_traceback": "in _run_agent_phase\nraise AgentTimeoutError(...)",
                        "occurred_at": finish,
                    },
                }
            ],
        }
    )
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (trial / "steps" / "build" / "agent").mkdir(parents=True)
    (trial / "steps" / "build" / "agent" / "opencode.txt").write_text(CLEAN_LOG, encoding="utf-8")


def test_native_multi_step_timeouts_are_refused_by_name(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    jobs = tmp_path / "jobs"
    _arm(jobs, "opencode", "with", [(case, "scored") for case in CASES])
    _arm(jobs, "opencode", "without", [("case-001", "scored")])
    job = jobs / f"{SKILL}-opencode-without"
    _multi_step_timeout_trial(job, "case-002__without1", "case-002")
    # Harbor counts the trial as completed, so the job reports no errored trial
    # and the runner no launch error.
    _job_result(job, errored=0)
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES])
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES])

    results = _collect(tmp_path, [])

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "failed"
    assert opencode["execution_status"] == "failed"
    assert (
        f"Agent runtime failed in case-002__without1: AgentTimeoutError: {TIMEOUT}"
        in opencode["conditions"]["without_skill"]["execution_errors"]
    )
    assert RECORD_KEY not in opencode
    assert opencode[REJECTION_KEY] == "step_timeout_not_supported"
    assert results["warnings"] == [
        "opencode baseline timeouts were not tolerated: a baseline attempt timed out in a step of a native "
        "multi-step task, which the option does not cover (step_timeout_not_supported)."
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


def _render_reports(tmp_path: Path, results: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    """Write ``result.json`` and render the canonical payload, ``tier3-benchmark.md`` and ``report.html``."""
    payload, benchmark, html, _message = _render_reports_and_message(tmp_path, results)
    return payload, benchmark, html


def _render_reports_and_message(
    tmp_path: Path, results: dict[str, Any], *, max_attempts: int = 1
) -> tuple[dict[str, Any], str, str, str]:
    """Like :func:`_render_reports`, and also return the Tier 3 result's ``agent_eval`` message."""
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory, render_agent_eval_html_report

    run_dir = tmp_path / "results"
    results.update(
        {
            "skill_name": SKILL,
            "run_dir": str(run_dir),
            "attempt_policy": {
                **results["attempt_policy"],
                "max_attempts": max_attempts,
                "pass_threshold": 0.5,
                "stop_on_pass": False,
            },
        }
    )
    _write_json(run_dir / "result.json", results)
    skill = tmp_path / SKILL
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: demo skill\n---\nbody\n", encoding="utf-8")
    result = agent_eval_result_from_directory(
        skill, run_dir, env_mode="docker", engine_result=results, use_llm_judge=False
    )
    messages = [detail.message for detail in result.success_details if detail.check_name == "agent_eval"]
    report = render_agent_eval_html_report(
        skill, run_dir, env_mode="docker", engine_result=results, include_benchmark=True, use_llm_judge=False
    )
    benchmark = (run_dir / "tier3-benchmark.md").read_text(encoding="utf-8")
    return result.metadata["agent_eval"], benchmark, report.read_text(encoding="utf-8"), "\n".join(messages)


def _tolerated_line(agent: str) -> str:
    return (
        f"Baseline incomplete for **{agent}** (1 of 2 attempts scored): kept by `--tolerate-baseline-timeouts`, "
        "so its baseline and uplift are not reported."
    )


def test_reports_rank_tolerated_agents_below_their_complete_peer(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.report import _pick_best_agent
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    # Both tolerated agents score higher than their complete peer.
    low = {**dict.fromkeys(SCORES, 0.55), "security": 1.0}
    jobs = tmp_path / "jobs"
    for agent in ("opencode", "claude-code"):
        _arm(jobs, agent, "with", [(case, "scored") for case in CASES])
        _arm(jobs, agent, "without", [("case-001", "scored"), ("case-002", "timeout")])
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES], scores=low)
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES], scores=low)
    launch_errors = [LAUNCH_ERROR.format(agent=agent, count=1) for agent in ("opencode", "claude-code")]
    results = _collect(tmp_path, launch_errors, agents=["opencode", "claude-code", "codex"])
    assert results["execution_status"] == "succeeded"
    assert results[RUN_KEY]["tolerated_agents"] == ["claude-code", "opencode"]

    payload, benchmark, html = _render_reports(tmp_path, results)
    assert payload["execution_status"] == "succeeded"
    assert payload["best_agent"] == "codex"
    assert payload["overall_score"] == payload["agents"]["codex"]["with_skill"]
    assert payload["agents"]["opencode"]["with_skill"] > payload["overall_score"]
    assert payload["overall_lift"] == payload["agents"]["codex"]["lift"]
    # Findings follow the same agent.
    assert _pick_best_agent(load_agent_data(tmp_path / "results"), results["agents"]) == "codex"

    assert "> **Tier 3 completed-agent result: " in benchmark
    assert "Completed agent: **codex**; score **0.6400**" in benchmark
    assert "**Needs attention:** other execution failures remain." in benchmark
    assert f"{_tolerated_line('claude-code')}\n{_tolerated_line('opencode')}\n" in benchmark
    assert (
        "Completed-agent quality result; publication not assessed; other execution failures need attention" in benchmark
    )
    # Each HTML notice is its own block, never run together with the next one.
    assert ".t3-overview-lead .tolerated-baseline { display:block;" in html
    for agent in ("claude-code", "opencode"):
        assert f"Baseline incomplete for <strong>{agent}</strong> (1 of 2 attempts scored)" in html
    assert html.count("<code>--tolerate-baseline-timeouts</code>, so its baseline and uplift are not reported.") == 2
    assert "not reported.</span><span" not in html


def test_reports_never_fall_back_to_a_tolerated_agent(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.report import _pick_best_agent
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    # codex completes both arms but never activates the skill; opencode
    # activates it and is tolerated.
    jobs = tmp_path / "jobs"
    _arm(jobs, "opencode", "with", [(case, "scored") for case in CASES], details=_activation(True))
    _arm(jobs, "opencode", "without", [("case-001", "scored"), ("case-002", "timeout")])
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES], details=_activation(False))
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES])
    results = _collect(tmp_path, [LAUNCH_ERROR.format(agent="opencode", count=1)])
    assert results["execution_status"] == "succeeded"

    payload, benchmark, html = _render_reports(tmp_path, results)
    # The canonical report ranks the complete agent first, then prefers activation.
    assert payload["best_agent"] == "codex"
    assert payload["agents"]["codex"]["skill_activation"]["status"] == "never_activated"
    assert payload["verdict"] == "inconclusive"
    # The findings panel has nothing to show for a never-activated best agent,
    # as on main, and never picks the tolerated agent instead.
    loaded = load_agent_data(tmp_path / "results")
    assert _pick_best_agent(results["agents"], loaded) == ""
    assert _pick_best_agent(loaded, results["agents"]) == ""
    # Both reports name the tolerated agent, also in the HTML's inconclusive overview.
    assert _tolerated_line("opencode") in benchmark
    assert "<strong>Inconclusive: skill never activated.</strong>" in html
    assert "Baseline incomplete for <strong>opencode</strong> (1 of 2 attempts scored)" in html


@pytest.mark.parametrize(
    ("peer", "verdict", "lead"),
    [
        ("failing", "fail", "<strong>Completed-agent result: FAIL.</strong>"),
        ("never-activated", "inconclusive", "<strong>Inconclusive: skill never activated.</strong>"),
    ],
    ids=["failing-peer", "never-activated-peer"],
)
def test_verdict_comes_from_the_complete_peer_not_a_tolerated_agent(
    tmp_path: Path, peer: str, verdict: str, lead: str
) -> None:
    from skillevaluator.evaluation.tier3_report import _agent_quality_verdict
    from skillevaluator.tier3.result_display import render_result

    # opencode activates the skill and passes every dimension and required
    # check (goal accuracy 1.0), but its baseline timed out. codex completes
    # both arms and fails its dimensions or never activates the skill.
    passing = {**SCORES, "goal_accuracy": 1.0}
    failing = {**dict.fromkeys(SCORES, 0.3), "security": 1.0}
    jobs = tmp_path / "jobs"
    _arm(jobs, "opencode", "with", [(case, "scored") for case in CASES], scores=passing, details=_activation(True))
    _arm(jobs, "opencode", "without", [("case-001", "scored"), ("case-002", "timeout")])
    peer_scores = failing if peer == "failing" else None
    peer_activation = _activation(peer != "never-activated")
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES], scores=peer_scores, details=peer_activation)
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES], scores=peer_scores)
    results = _collect(tmp_path, [LAUNCH_ERROR.format(agent="opencode", count=1)])
    assert results["execution_status"] == "succeeded"

    payload, benchmark, html, message = _render_reports_and_message(tmp_path, results)
    assert _agent_quality_verdict(payload["agents"]["opencode"]) == "pass"
    # The verdict comes from the same agent as the best agent and every headline.
    assert payload["best_agent"] == payload["summary"]["best_agent"] == "codex"
    assert payload["verdict"] == payload["summary"]["verdict"] == verdict
    assert message == f"Tier 3 evaluation complete: verdict {verdict.upper()}; best agent codex"
    assert f"> **Tier 3 completed-agent result: {verdict.upper()}**" in benchmark
    assert f"| Tier 3 | Live agent evaluation | **{verdict.upper()}** |" in benchmark
    assert lead in html
    # The HTML overview says once that the tolerated baseline needs attention:
    # in the completed-agent lead, or in its own note when the lead reads
    # inconclusive, as BENCHMARK.md does in both cases.
    overview = html.split('<div class="t3-overview-lead">', 1)[1].split("</div>", 1)[0]
    lead_attention = "Other execution failures need attention."
    tolerated_attention = "<strong>Needs attention:</strong> baseline timeouts were tolerated."
    assert (lead_attention in overview) is (verdict == "fail")
    assert (tolerated_attention in overview) is (verdict == "inconclusive")
    assert "Baseline incomplete for <strong>opencode</strong> (1 of 2 attempts scored)" in overview
    assert "**Needs attention:** other execution failures remain." in benchmark
    if verdict == "inconclusive":
        # The tolerated agent did activate the skill, so Insights names whose trials never did.
        assert payload["conclusions"][0]["message"].startswith(
            "The skill was never activated in any positive with-skill trial of an agent with a complete baseline."
        )
        entry = _changelog_entry("- `--tolerate-baseline-timeouts` (added in 0.9.0) now follows its own rules")
        assert (
            "`report.html` also keeps the note when its overview reads Inconclusive because the complete agent "
            "never activated the skill." in entry
        )
    # The run verdict that `tier3 evaluate` prints reads the same agents.
    terminal = render_result(results)
    assert "Tier 3 Evaluation: DEGRADED" in terminal
    assert [line.strip() for line in terminal.splitlines() if line.startswith("RUN QUALITY VERDICT:")] == [
        f"RUN QUALITY VERDICT: {verdict.upper()}"
    ]
    block = _doc_block("docs/tier3-live-evaluation.mdx", "For a tolerated agent, the baseline condition stays")
    assert (
        "The run's verdict and the `RUN QUALITY VERDICT` line that `tier3 evaluate` prints read the same agents:"
        in block
    )


@pytest.mark.parametrize("field", ["with_skill", "without_skill", "dimensions_with_skill"])
def test_printed_verdict_reads_every_agent_when_the_complete_peer_cannot_rank(tmp_path: Path, field: str) -> None:
    from skillevaluator.tier3.result_display import render_result

    # opencode passes and is tolerated. codex completes both arms, but its
    # persisted scores are not a map, so it cannot rank and does not set the
    # tolerated agent aside: the printed verdict reads every agent, and the
    # summary still renders.
    passing = {**SCORES, "goal_accuracy": 1.0}
    jobs = tmp_path / "jobs"
    _arm(jobs, "opencode", "with", [(case, "scored") for case in CASES], scores=passing, details=_activation(True))
    _arm(jobs, "opencode", "without", [("case-001", "scored"), ("case-002", "timeout")])
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES], details=_activation(True))
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES])
    results = _collect(tmp_path, [LAUNCH_ERROR.format(agent="opencode", count=1)])
    assert results[RUN_KEY]["tolerated_agents"] == ["opencode"]
    results.update({"skill_name": SKILL, "run_dir": str(tmp_path / "results")})
    results["agents"]["codex"][field] = "not a score map"

    terminal = render_result(results)

    assert "Tier 3 Evaluation: DEGRADED" in terminal
    assert [line.strip() for line in terminal.splitlines() if line.startswith("RUN QUALITY VERDICT:")] == [
        "RUN QUALITY VERDICT: PASS"
    ]


def test_verdict_counts_come_from_a_complete_agent_that_set_the_verdict(tmp_path: Path) -> None:
    # claude-code passes everything but its baseline timed out. opencode has the
    # top score but misses a goal (FAIL), so the best agent's own result is not
    # the run verdict; codex passes with a lower score and sets the PASS.
    passing = {**SCORES, "goal_accuracy": 1.0}
    lower_passing = {**dict.fromkeys(SCORES, 0.6), "goal_accuracy": 1.0, "security": 1.0}
    jobs = tmp_path / "jobs"
    _arm(jobs, "claude-code", "with", [(case, "scored") for case in CASES], scores=passing, details=_activation(True))
    _arm(jobs, "claude-code", "without", [("case-001", "scored"), ("case-002", "timeout")])
    for agent, scores in (("codex", lower_passing), ("opencode", None)):
        _arm(jobs, agent, "with", [(case, "scored") for case in CASES], scores=scores, details=_activation(True))
        _arm(jobs, agent, "without", [(case, "scored") for case in CASES], scores=scores)
    results = _collect(
        tmp_path, [LAUNCH_ERROR.format(agent="claude-code", count=1)], agents=["claude-code", "codex", "opencode"]
    )
    assert results[RUN_KEY]["tolerated_agents"] == ["claude-code"]

    payload, _benchmark, _html, _message = _render_reports_and_message(tmp_path, results)

    assert payload["best_agent"] == "opencode"
    assert payload["verdict"] == payload["summary"]["verdict"] == "pass"
    # The PASS counts are codex's, never those of the tolerated agent with no baseline.
    for field in ("correct_answers", "required_checks", "security_clean"):
        assert payload["summary"][field] == payload["agents"]["codex"][field]
    assert payload["agents"]["claude-code"]["correct_answers"] != payload["agents"]["codex"]["correct_answers"]


@pytest.mark.parametrize(
    ("tolerated", "panel_agent", "run_verdict"),
    [
        ((), "opencode", "FAIL"),
        (("opencode",), None, "INCONCLUSIVE"),
        (("opencode", "claude-code"), None, "INCONCLUSIVE"),
    ],
    ids=["none-tolerated", "one-tolerated", "two-tolerated"],
)
def test_terminal_findings_panel_never_shows_a_tolerated_agent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tolerated: tuple[str, ...],
    panel_agent: str | None,
    run_verdict: str,
) -> None:
    from skillevaluator.tier3.result_display import render_result

    # codex completes both arms but never activates the skill, so the terminal
    # summary leaves it out before it picks the agent for the findings panel.
    activated = list(dict.fromkeys(["opencode", *tolerated]))
    jobs = tmp_path / "jobs"
    for agent in activated:
        _arm(jobs, agent, "with", [(case, "scored") for case in CASES], details=_activation(True))
        outcome = "timeout" if agent in tolerated else "scored"
        _arm(jobs, agent, "without", [("case-001", "scored"), ("case-002", outcome)])
    _arm(jobs, "codex", "with", [(case, "scored") for case in CASES], details=_activation(False))
    _arm(jobs, "codex", "without", [(case, "scored") for case in CASES])
    launch_errors = [LAUNCH_ERROR.format(agent=agent, count=1) for agent in tolerated]
    results = _collect(tmp_path, launch_errors, agents=[*activated, "codex"])
    assert results["execution_status"] == "succeeded"
    assert results[RUN_KEY]["tolerated_agents"] == sorted(tolerated)
    results.update({"skill_name": SKILL, "run_dir": str(tmp_path / "results")})

    capsys.readouterr()
    rendered = render_result(results)
    printed = capsys.readouterr().out
    # The run verdict reads only codex once every activated agent is tolerated
    # (codex never activated). Without tolerance, opencode misses its goals
    # (goal accuracy 0.8), so the run fails, as on main.
    assert [line for line in rendered.splitlines() if line.startswith("RUN QUALITY VERDICT:")] == [
        f"RUN QUALITY VERDICT: {run_verdict}"
    ]
    written = sorted(path.parent.name for path in (tmp_path / "results").glob("*/findings.json"))
    if panel_agent is None:
        # The canonical best agent is codex, which has no findings to show. No
        # tolerated agent stands in for it, however many are left.
        assert "Findings" not in printed
        assert written == []
    else:
        # Without tolerance the one activated agent keeps its panel, as on main.
        assert f"{SKILL} / {panel_agent} — Findings" in printed
        assert written == [panel_agent]


@pytest.mark.parametrize("tolerate", [True, False], ids=["tolerated", "default"])
def test_failed_baseline_fails_the_run_unless_tolerated_and_the_docs_say_so(tmp_path: Path, tolerate: bool) -> None:
    """The baseline arm stays failed either way; only the option keeps the agent and the run."""
    results = _collect(tmp_path, _two_agent_jobs(tmp_path / "jobs"), tolerate_baseline_timeouts=tolerate)

    payload, _benchmark, _html = _render_reports(tmp_path, results)

    (notice,) = payload["coverage_notices"]
    assert (notice["agent"], notice["code"], notice["state"]) == ("opencode", "baseline_failed", "incomplete")
    assert payload["execution_status"] == ("succeeded" if tolerate else "failed")
    assert ("The agent and the Tier 3 run are marked failed." in notice["message"]) is not tolerate
    block = _doc_block("docs/reports.mdx", "An arm that ran but failed")
    assert "fails the agent and the run unless `--tolerate-baseline-timeouts` keeps the agent scored" in block
    assert "[Baseline timeouts](/skills/skillevaluator/tier3-live-evaluation#baseline-timeouts)" in block


@pytest.mark.parametrize("kwargs", [{"grading_mode": "custom_only"}, {"skip_baseline": True}])
def test_tolerance_needs_a_baseline_and_dimension_grading(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    launch_errors = _two_agent_jobs(tmp_path / "jobs")

    results = _collect(tmp_path, launch_errors, **kwargs)

    assert RUN_KEY not in results
    assert all(RECORD_KEY not in agent and REJECTION_KEY not in agent for agent in results["agents"].values())


def test_reused_baseline_leaves_the_option_inert(tmp_path: Path) -> None:
    # A run that completed both arms is the retained baseline (--reuse-baseline).
    source_jobs = tmp_path / "source-jobs"
    for agent in ("opencode", "codex"):
        for variant in ("with", "without"):
            _arm(source_jobs, agent, variant, [(case, "scored") for case in CASES])
    source = tmp_path / "source"
    collect_harbor_results(
        skill_name=SKILL,
        agents=["opencode", "codex"],
        output_dir=source / "results",
        jobs_dir=source_jobs,
        n_attempts=1,
        expected_cases=len(CASES),
        expected_case_ids=CASES,
        case_id_by_task_selector={case: case for case in CASES},
        expected_trials=len(CASES),
    )
    jobs = tmp_path / "jobs"
    for agent in ("opencode", "codex"):
        _arm(jobs, agent, "with", [(case, "scored") for case in CASES])

    results = _collect(tmp_path, [], reused_baseline_dir=source / "results")

    # The baseline did not run here, so there are no timeouts to judge.
    assert results["execution_status"] == "succeeded"
    assert all(agent["without_skill"] for agent in results["agents"].values())
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


def test_stop_on_pass_recovered_timeout_beside_unrecovered_ones(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    from skillevaluator.tier3.harbor.runner import _merge_attempt_jobs

    plan = {
        ("opencode", "with"): {"case-a": ["pass"], "case-b": ["pass"]},
        # case-a timed out once and then passed, so main's handling recovers it;
        # case-b timed out on both attempts.
        ("opencode", "without"): {"case-a": ["timeout", "pass"], "case-b": ["timeout", "timeout"]},
        ("codex", "with"): {"case-a": ["pass"], "case-b": ["pass"]},
        ("codex", "without"): {"case-a": ["fail", "fail"], "case-b": ["fail", "fail"]},
    }
    for (agent, variant), cases in plan.items():
        jobs = [
            _attempt_job(tmp_path, agent, variant, case, attempt, outcome)
            for case, outcomes in cases.items()
            for attempt, outcome in enumerate(outcomes, start=1)
        ]
        _merge_attempt_jobs(jobs, tmp_path / "jobs" / f"{SKILL}-{agent}-{variant}")
    # The runner already dropped the recovered attempt's launch error.
    launch_errors = [
        f"opencode without-skill Harbor run failed: case-b attempt {attempt}: "
        "Harbor job did not complete successfully: 1 errored"
        for attempt in (1, 2)
    ]

    results = collect_harbor_results(
        skill_name=SKILL,
        agents=["opencode", "codex"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        n_attempts=2,
        stop_on_pass=True,
        expected_cases=2,
        expected_case_ids=["case-a", "case-b"],
        case_id_by_task_selector={"case-a": "case-a", "case-b": "case-b"},
        launch_errors=launch_errors,
        tolerate_baseline_timeouts=True,
    )

    opencode = results["agents"]["opencode"]
    # Harbor's errored count includes the recovered timeout.
    assert opencode["job_failures"]["without_skill"] == "Harbor job did not complete successfully: 3 errored"
    assert opencode["conditions"]["without_skill"]["recovered_attempts_total"] == 1
    assert results["execution_status"] == "succeeded", results.get("warnings")
    assert opencode["execution_status"] == "succeeded"
    record = opencode[RECORD_KEY]
    assert [(failure["case_id"], failure["attempt"]) for failure in record["failures"]] == [
        ("case-b", 1),
        ("case-b", 2),
    ]
    assert results[RUN_KEY]["explained_launch_errors"] == launch_errors


_COST_LIMIT_LAUNCH_ERROR = (
    "opencode without-skill Harbor run failed: case-a attempt 2: Harbor job did not complete successfully: 1 errored"
)


def _collect_cost_limited(tmp_path: Path, *, codex_short: bool = False) -> dict[str, Any]:
    """Collect attempt jobs a cost limit cut short; opencode's second case-a baseline attempt timed out.

    The limit admitted no attempt of case-b for opencode. With ``codex_short``
    it also left codex's second case-b attempt ``not_run`` in both arms.
    """
    pytest.importorskip("harbor")
    from skillevaluator.tier3.harbor.runner import _merge_attempt_jobs

    codex_b_attempts = 1 if codex_short else 2
    plan = {
        ("opencode", "with"): {"case-a": ["pass", "pass"]},
        ("opencode", "without"): {"case-a": ["pass", "timeout"]},
        ("codex", "with"): {"case-a": ["pass", "pass"], "case-b": ["pass"] * codex_b_attempts},
        ("codex", "without"): {"case-a": ["fail", "fail"], "case-b": ["fail"] * codex_b_attempts},
    }
    for (agent, variant), cases in plan.items():
        jobs = [
            _attempt_job(tmp_path, agent, variant, case, attempt, outcome)
            for case, outcomes in cases.items()
            for attempt, outcome in enumerate(outcomes, start=1)
        ]
        _merge_attempt_jobs(jobs, tmp_path / "jobs" / f"{SKILL}-{agent}-{variant}")
    not_run = [
        {"agent": "opencode", "arm": arm, "case_id": "case-b", "attempt": attempt, "status": "not_run"}
        for arm in ("with", "without")
        for attempt in (1, 2)
    ]
    if codex_short:
        not_run += [
            {"agent": "codex", "arm": arm, "case_id": "case-b", "attempt": 2, "status": "not_run"}
            for arm in ("with", "without")
        ]
    return collect_harbor_results(
        skill_name=SKILL,
        agents=["opencode", "codex"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        n_attempts=2,
        expected_cases=2,
        expected_case_ids=["case-a", "case-b"],
        case_id_by_task_selector={"case-a": "case-a", "case-b": "case-b"},
        expected_trials=4,
        launch_errors=[_COST_LIMIT_LAUNCH_ERROR],
        tolerate_baseline_timeouts=True,
        budget_not_run=not_run,
    )


def test_cost_limit_attempts_that_never_ran_are_not_missing_baseline_attempts(tmp_path: Path) -> None:
    # A cost limit runs every attempt as its own job. It admitted no attempt
    # of case-b for opencode, so neither arm expects case-b; only the timed-out
    # second attempt of case-a is missing from the baseline.
    results = _collect_cost_limited(tmp_path)

    opencode = results["agents"]["opencode"]
    assert results["execution_status"] == "succeeded", results.get("warnings")
    assert results["partial"] is True
    assert REJECTION_KEY not in opencode
    assert opencode["conditions"]["with_skill"]["expected_attempts"] == 2
    record = opencode[RECORD_KEY]
    assert (record["expected_attempts"], record["scored_attempts"], record["failed_attempts"]) == (2, 1, 1)
    assert [(failure["case_id"], failure["attempt"]) for failure in record["failures"]] == [("case-a", 2)]
    assert results[RUN_KEY]["explained_launch_errors"] == [_COST_LIMIT_LAUNCH_ERROR]


@pytest.mark.parametrize("codex_short", [False, True], ids=["peer-complete", "peer-cost-limited"])
def test_reports_name_tolerated_agents_without_a_completed_agent_result(tmp_path: Path, codex_short: bool) -> None:
    from skillevaluator.reporting.tier3_scope import completed_agent_result

    # With codex_short the cost limit also cut the best agent's coverage, so
    # the reports have no completed-agent result; the tolerated baseline still
    # needs attention and is named in both reports.
    results = _collect_cost_limited(tmp_path, codex_short=codex_short)
    assert results["execution_status"] == "succeeded", results.get("warnings")
    assert results[RUN_KEY]["tolerated_agents"] == ["opencode"]

    payload, benchmark, html, _message = _render_reports_and_message(tmp_path, results, max_attempts=2)

    assert payload["best_agent"] == "codex"
    assert (completed_agent_result(payload) is None) is codex_short
    assert _tolerated_line("opencode") in benchmark
    assert benchmark.count("**Needs attention:**") == 1
    assert "Baseline incomplete for <strong>opencode</strong> (1 of 2 attempts scored)" in html
    attention = "**Needs attention:** baseline timeouts were tolerated."
    html_attention = (
        '<span class="tolerated-baseline"><strong>Needs attention:</strong> baseline timeouts were tolerated.</span>'
    )
    if codex_short:
        # Nothing before the note names other failures, so it names the cause.
        assert "Completed agent: **codex**" not in benchmark
        assert f"{attention}\n{_tolerated_line('opencode')}" in benchmark
        assert "other execution failures" not in benchmark.lower()
        assert html_attention in html
        assert "other execution failures" not in html.lower()
        entry = _changelog_entry("- `--tolerate-baseline-timeouts` (added in 0.9.0) now follows its own rules")
        assert (
            "also when a cost limit leaves the best agent's attempts short; both reports used to leave them out "
            "in that case." in entry
        )
    else:
        # The completed-agent lead already says that other failures need attention.
        assert "Completed agent: **codex**" in benchmark
        assert "**Needs attention:** other execution failures remain." in benchmark
        assert "Other execution failures need attention. This is not a whole-run or publication PASS." in html
        assert attention not in benchmark
        assert html_attention not in html


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
        [ADC_LAUNCH_ERROR],
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


def test_run_record_bounds_launch_errors_and_the_runner_drops_only_explained_ones() -> None:
    errors = [
        f"opencode without-skill Harbor run failed: case-{n:03d} attempt 1: "
        "Harbor job did not complete successfully: 1 errored"
        for n in range(40)
    ]
    record = run_record({"opencode"}, errors)
    assert record["tolerated_agents"] == ["opencode"]
    assert record["explained_launch_errors"] == errors[: baseline_timeouts.LAUNCH_ERRORS_PUBLISHED_MAX]
    assert record["explained_launch_errors_total"] == 40
    assert record["explained_launch_errors_truncated"] is True

    agents = {"opencode": {}, "opencode-cli": {}, "codex": {}}
    other = "opencode-cli with-skill Harbor run failed: harbor run exited 1"
    results = {"agents": agents, RUN_KEY: record}
    # Every tolerated agent's error is dropped, even past the published sample.
    assert unexplained_launch_errors(results, [*errors, other]) == [other]
    # A count that no longer matches the record, or no tolerated agent, keeps every error.
    assert unexplained_launch_errors(results, [*errors[:39], other]) == [*errors[:39], other]
    assert unexplained_launch_errors({"agents": agents, RUN_KEY: run_record((), [])}, errors[:1]) == errors[:1]
    assert unexplained_launch_errors({}, errors[:1]) == errors[:1]


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


def test_option_is_not_part_of_the_comparison_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The option changes no execution, so reuse and version comparison still pair runs across it."""
    from skillevaluator.tier3 import baseline_store

    result, _captured = _run_engine(monkeypatch, tmp_path, CONFIG + "  tolerate_baseline_timeouts: true\n")
    run_dir = Path(result["run_dir"])
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["harbor"]["tolerate_baseline_timeouts"] is True
    source = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))["comparison_identity"]
    assert "tolerate_baseline_timeouts" not in source["config"]["harbor"]

    # The same run configured without the option.
    config["harbor"].pop("tolerate_baseline_timeouts")
    current = {**source, "config": baseline_store.comparison_identity(config, {})["config"]}
    assert baseline_store.comparison_mismatches(current, source) == []
    assert baseline_store.comparison_mismatches(source, current) == []
    # The complete peer's baseline is a --reuse-baseline source; the tolerated baseline is not.
    reasons, _result = baseline_store.baseline_eligibility(run_dir, ["codex"], current)
    assert reasons == []
    reasons, _result = baseline_store.baseline_eligibility(run_dir, ["opencode"], current)
    assert "opencode: without-skill condition did not succeed" in reasons
    # Both clean with-skill arms are a --compare-to prior.
    reasons, _result = baseline_store.condition_eligibility(run_dir, ["codex", "opencode"], current, "with_skill")
    assert reasons == []


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
