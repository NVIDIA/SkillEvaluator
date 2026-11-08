# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tolerance for Tier 3 baseline attempts that hit the agent execution timeout.

The Tier 3 verdict reads only with-skill dimensions, so a without-skill
(baseline) attempt that ran out of time says nothing about the skill. With
``--tolerate-baseline-timeouts`` (or ``harbor.tolerate_baseline_timeouts``),
an agent whose with-skill arm completed cleanly stays scored when its baseline
fell short only because attempts timed out while the agent was still working.
Its baseline and uplift are not reported; the baseline condition keeps its
``failed`` status and errors.

The collector gathers the evidence (trial results, logs, attempt slots) and
calls the pure rules here. Every rule must hold:

* T1: the with-skill arm is complete and clean.
* T2: the baseline arm failed short of its expected attempts.
* T3: no agent runtime failure other than the timeout: the timed-out trials'
  logs show no provider overload, rate limit or authentication errors, and each
  trial left an agent log to check.
* T4: every failed baseline trial ended in ``AgentTimeoutError``. Native
  multi-step tasks are not covered: Harbor records their agent timeouts on the
  step and counts the trial as completed, so they are refused.
* T5: the timeouts match the missing attempts one for one, per case and, when
  attempts carry labels, per attempt; every trial in the job is scored or timed out.
* T6: every timed-out attempt shows agent output (output tokens > 0).
* T7: every baseline error is one the timeouts produce.
* T8 (run level): every other agent completed, one of them completed both arms,
  and every runner launch error is a tolerated baseline's own. Tolerance is all
  or nothing, so it never applies to a run that fails for another reason.

A tolerated run is still not complete publication evidence: its attempt
coverage stays short, so publication and required-gate checks treat it as
incomplete. Reports rank a tolerated agent below every agent with a complete
baseline (:func:`kept_with_failed_baseline`), so the run's best agent and
headline score always come from complete evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any

RECORD_KEY = "baseline_timeout_tolerance"
REJECTION_KEY = "baseline_timeout_tolerance_rejection"
RUN_KEY = "baseline_timeout_tolerance"
RECORD_ARTIFACT = "baseline_timeout_tolerance.json"  # beside lift.json in the agent results directory
TOLERATED_EXCEPTION_TYPE = "AgentTimeoutError"
TOLERATED_REASON = "agent_execution_timeout"
JOB_FAILURE = "Harbor job did not complete successfully: {count} errored"
_LAUNCH_PREFIX = "{agent} without-skill Harbor run failed: "
FAILURES_PUBLISHED_MAX = 32
LAUNCH_ERRORS_PUBLISHED_MAX = 32

REJECTION_REASONS = {
    "with_skill_incomplete": "the with-skill arm did not score every attempt",
    "with_skill_not_clean": "the with-skill arm reported failures",
    "custom_only_grading": "custom_only grading has no dimension verdict",
    "baseline_not_short": "the baseline did not fall short of its attempts",
    "no_case_identity": "the run has no expected case identities to match attempts against",
    "agent_runtime_failure": "the baseline logged agent runtime errors",
    "provider_degradation_observed": "a timed-out attempt logged provider overload, rate limit or authentication errors",
    "no_agent_log": "a timed-out attempt left no agent log to check for provider errors",
    "reason_not_tolerated": "a baseline attempt failed with something other than an agent execution timeout",
    "untyped_trial_failure": "a failed baseline trial is not an agent execution timeout",
    "step_timeout_not_supported": "a baseline attempt timed out in a step of a native multi-step task, which the option does not cover",
    "timed_out_trial_scored": "a timed-out baseline trial also has a scored reward",
    "unaccounted_trial": "the baseline job holds a trial that is neither scored nor timed out",
    "failures_do_not_account_for_missing": "the timeouts do not account for every missing attempt",
    "per_case_mismatch": "the timeouts do not match the missing attempts case by case",
    "ordinal_mismatch": "the timeouts do not match the missing attempt numbers",
    "no_progress_evidence": "a timed-out attempt shows no agent output tokens",
    "unbound_error": "the baseline reported other errors",
    "other_agent_failed": "another agent did not complete",
    "no_clean_peer": "no other agent completed both arms",
    "unbound_launch_error": "the run reported other launch errors",
}


def _count(value: Any, *, minimum: int = 0) -> int | None:
    """Return ``value`` when it is a real integer (never bool) at or above ``minimum``."""
    return value if type(value) is int and value >= minimum else None


def baseline_status(scored_attempts: int) -> str:
    """Return ``partial`` when some baseline attempts scored, else ``unavailable``."""
    return "partial" if scored_attempts > 0 else "unavailable"


def check_failures(
    *,
    expected_attempts: int,
    scored_attempts: int,
    slots: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> str | None:
    """Check the typed baseline failures against the missing attempts (T4 to T6).

    ``slots`` maps each expected case to ``{"required", "missing",
    "missing_ordinals"}``; ``missing_ordinals`` is None when the case's attempt
    labels are implicit. Each failure is ``{"case_id", "attempt",
    "exception_types", "output_tokens"}``, where ``attempt`` is the attempt
    ordinal or None and ``exception_types`` lists every exception Harbor
    recorded on the trial and its steps. Returns None when the failures are
    exactly the missing attempts, otherwise the first failing code.
    """
    expected = _count(expected_attempts, minimum=1)
    scored = _count(scored_attempts)
    if expected is None or scored is None or scored >= expected:
        return "baseline_not_short"
    for failure in failures:
        types = failure.get("exception_types")
        if not isinstance(types, (list, tuple)) or not types or set(types) != {TOLERATED_EXCEPTION_TYPE}:
            return "reason_not_tolerated"
    if len(failures) != expected - scored:
        return "failures_do_not_account_for_missing"
    missing_by_case = Counter(
        {str(case): count for case, slot in slots.items() if (count := _count(slot.get("missing"), minimum=1))}
    )
    if Counter(str(failure.get("case_id") or "") for failure in failures) != missing_by_case:
        return "per_case_mismatch"
    for case, slot in slots.items():
        ordinals = slot.get("missing_ordinals")
        if ordinals is None:
            continue
        attempts = [failure.get("attempt") for failure in failures if failure.get("case_id") == case]
        if any(_count(attempt, minimum=1) is None for attempt in attempts) or sorted(attempts) != sorted(ordinals):
            return "ordinal_mismatch"
    if any(_count(failure.get("output_tokens"), minimum=1) is None for failure in failures):
        return "no_progress_evidence"
    return None


def build_record(
    *,
    expected_attempts: int,
    scored_attempts: int,
    failures: Sequence[Mapping[str, Any]],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    """Return the baseline record kept on a tolerated agent.

    Each failure carries the published ``trial`` label, ``case_id``, ``task``
    (the staged task selector), ``attempt`` and ``output_tokens``. Only the
    first :data:`FAILURES_PUBLISHED_MAX` rows are published; ``failed_attempts``
    is the exact count.
    """
    rows = [
        {
            "trial": str(failure.get("trial") or ""),
            "case_id": str(failure.get("case_id") or ""),
            "attempt": failure.get("attempt"),
            "output_tokens": failure.get("output_tokens"),
        }
        for failure in failures
    ]
    return {
        "status": baseline_status(scored_attempts),
        "reason": TOLERATED_REASON,
        "exception_type": TOLERATED_EXCEPTION_TYPE,
        "expected_attempts": expected_attempts,
        "scored_attempts": scored_attempts,
        "failed_attempts": len(rows),
        "timeout_seconds": timeout_seconds,
        "failures": rows[:FAILURES_PUBLISHED_MAX],
        "failures_truncated": len(rows) > FAILURES_PUBLISHED_MAX,
    }


def _bound_launch_errors(agent: str, errors: Sequence[str], failures: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether ``errors`` are exactly the runner messages the timed-out attempts produce.

    An arm-level job reports one ``<agent> without-skill Harbor run failed:
    Harbor job did not complete successfully: <k> errored`` with ``k`` the
    number of timeouts. Stop-on-pass runs each attempt as its own job and
    reports ``... failed: <task> attempt <n>: Harbor job did not complete
    successfully: 1 errored`` once per timed-out attempt.
    """
    if not errors:
        return True
    prefix = _LAUNCH_PREFIX.format(agent=agent)
    arm_level = prefix + JOB_FAILURE.format(count=len(failures))
    if list(errors) == [arm_level]:
        return True
    expected = Counter(
        f"{prefix}{failure.get('task')} attempt {failure.get('attempt')}: {JOB_FAILURE.format(count=1)}"
        for failure in failures
        if failure.get("task") and _count(failure.get("attempt"), minimum=1) is not None
    )
    observed = Counter(errors)
    return all(expected[error] >= count for error, count in observed.items())


def _owner(error: str, agents: Iterable[str]) -> str | None:
    """Return the agent a runner launch error names, or None when it names none."""
    for agent in sorted(agents, key=len, reverse=True):
        if error.startswith(f"{agent} "):
            return agent
    return None


def select_tolerated(
    agents: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    rejected: Mapping[str, str],
    launch_errors: Sequence[str] | None,
) -> tuple[frozenset[str], dict[str, str], list[str]]:
    """Decide the run-level outcome (T8); returns ``(tolerated, rejected, explained_launch_errors)``.

    ``agents`` is the collected agent payloads, ``candidates`` maps each agent
    whose baseline passed T1 to T7 to its full typed failure list, and
    ``rejected`` maps each agent that had a complete with-skill arm and a short
    baseline but failed those rules to its code. Tolerance is all or nothing:
    it applies only when no candidate was rejected, every other agent
    completed, one of them completed both arms, and every launch error is a
    candidate's own timeout message. Otherwise each candidate keeps its failed
    status and gets the run-level code.
    """
    outcome = dict(rejected)
    if not candidates:
        return frozenset(), outcome, []
    others = {name: data for name, data in agents.items() if name not in candidates}
    reason: str | None = None
    if rejected or any(data.get("execution_status") != "succeeded" for data in others.values()):
        reason = "other_agent_failed"
    elif not any(
        ((data.get("conditions") or {}).get("without_skill") or {}).get("execution_status") == "succeeded"
        for data in others.values()
    ):
        reason = "no_clean_peer"
    explained: list[str] = []
    if reason is None:
        by_agent: dict[str, list[str]] = {}
        for error in launch_errors or ():
            owner = _owner(str(error), agents)
            if owner not in candidates:
                reason = "unbound_launch_error"
                break
            by_agent.setdefault(owner, []).append(str(error))
        else:
            for agent, errors in by_agent.items():
                if not _bound_launch_errors(agent, errors, candidates[agent]):
                    reason = "unbound_launch_error"
                    break
                explained.extend(errors)
    if reason is not None:
        outcome.update(dict.fromkeys(candidates, reason))
        return frozenset(), outcome, []
    return frozenset(candidates), outcome, explained


def settle(agent_data: MutableMapping[str, Any], record: Mapping[str, Any]) -> None:
    """Keep a tolerated agent scored on its with-skill arm.

    The agent's status becomes ``succeeded`` with no agent-level errors, and
    ``record`` is attached under :data:`RECORD_KEY`. ``conditions.without_skill``
    keeps its ``failed`` status and errors, and the attempt totals still count
    the missing baseline attempts. Baseline scores, lift and security
    attribution were never computed for the failed baseline, so they stay empty.
    """
    agent_data[RECORD_KEY] = dict(record)
    agent_data.pop(REJECTION_KEY, None)
    agent_data["execution_status"] = "succeeded"
    agent_data["execution_errors"] = []
    agent_data["execution_error_details_total"] = 0
    agent_data["execution_error_details_shown"] = 0
    agent_data["execution_error_details_truncated"] = False


def run_record(tolerated: Iterable[str], explained_launch_errors: Sequence[str]) -> dict[str, Any]:
    """Return the run-level record: the tolerated agents and a bounded sample of the launch errors they explain.

    ``explained_launch_errors_total`` is the exact count; only the first
    :data:`LAUNCH_ERRORS_PUBLISHED_MAX` messages are published, since a
    stop-on-pass baseline that timed out on many cases reports one per attempt.
    """
    explained = [str(error) for error in explained_launch_errors]
    return {
        "tolerated_agents": sorted(tolerated),
        "explained_launch_errors": explained[:LAUNCH_ERRORS_PUBLISHED_MAX],
        "explained_launch_errors_total": len(explained),
        "explained_launch_errors_truncated": len(explained) > LAUNCH_ERRORS_PUBLISHED_MAX,
    }


def unexplained_launch_errors(results: Mapping[str, Any], launch_errors: Sequence[str]) -> list[str]:
    """Return the launch errors that tolerated baselines do not explain, in order.

    Tolerance applies only when every launch error is a tolerated baseline's
    own timeout message, so those are exactly the errors a tolerated agent
    owns. They are dropped only while their count still equals the recorded
    ``explained_launch_errors_total``; otherwise every error is kept.
    """
    run = results.get(RUN_KEY)
    if not isinstance(run, Mapping):
        return list(launch_errors)
    tolerated = {str(agent) for agent in run.get("tolerated_agents") or ()}
    agents = results.get("agents")
    names = [str(name) for name in agents] if isinstance(agents, Mapping) else []
    remaining = [error for error in launch_errors if _owner(str(error), names) not in tolerated]
    if not tolerated or len(launch_errors) - len(remaining) != _count(run.get("explained_launch_errors_total")):
        return list(launch_errors)
    return remaining


def kept_with_failed_baseline(agent: Mapping[str, Any]) -> bool:
    """Return whether ``agent`` stays scored although its baseline condition failed.

    Without tolerance a failed arm always fails its agent, so this shape marks
    an agent kept by ``--tolerate-baseline-timeouts``, both in ``result.json``
    and in the report payload built from it.
    """
    conditions = agent.get("conditions")
    baseline = conditions.get("without_skill") if isinstance(conditions, Mapping) else None
    return (
        agent.get("execution_status") == "succeeded"
        and isinstance(baseline, Mapping)
        and baseline.get("execution_status") == "failed"
    )


def tolerated_baselines(agents: Any) -> list[dict[str, Any]]:
    """Describe each agent kept with a failed baseline: ``{"name", "scored_attempts", "expected_attempts"}``."""
    if not isinstance(agents, Mapping):
        return []
    described: list[dict[str, Any]] = []
    for name in sorted(agents, key=str):
        agent = agents[name]
        if not isinstance(agent, Mapping) or not kept_with_failed_baseline(agent):
            continue
        baseline = agent["conditions"]["without_skill"]
        described.append(
            {
                "name": str(name),
                "scored_attempts": _count(baseline.get("scored_attempts")),
                "expected_attempts": _count(baseline.get("expected_attempts")),
            }
        )
    return described


def tolerance_warning(agent: str, record: Mapping[str, Any]) -> str:
    """Return the run warning that names a tolerated agent and its missing baseline attempts."""
    failed = record.get("failed_attempts")
    expected = record.get("expected_attempts")
    scored = record.get("scored_attempts")
    timeout = record.get("timeout_seconds")
    after = f" after {timeout:g} seconds" if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) else ""
    return (
        f"{agent} baseline incomplete: {failed} attempt(s) hit the agent execution timeout{after} "
        f"({scored} of {expected} scored). The agent stays scored on its with-skill arm "
        "(--tolerate-baseline-timeouts); its baseline and uplift are not reported, and the run is not "
        "complete publication evidence."
    )


def rejection_warning(agent: str, code: str) -> str:
    """Return the run warning for an agent whose baseline timeouts were not tolerated."""
    reason = REJECTION_REASONS.get(code, "the baseline evidence did not meet the tolerance rules")
    return f"{agent} baseline timeouts were not tolerated: {reason} ({code})."
