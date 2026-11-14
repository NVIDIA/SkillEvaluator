# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 token usage per case, per agent and over all agents.

Each Tier 3 attempt records the agent's token counters in its ATIF trajectory.
This module adds them up per case, per agent arm and over all agents, and says
how many attempts the sums cover. It is a measurement only: it changes no
metric, dimension, overall score, verdict or gate.

- Prompt counters include cached tokens (ATIF ``total_prompt_tokens`` does), so
  cached tokens are a part of the prompt and are never added again.
  ``total_tokens`` is prompt plus completion, and ``uncached_prompt_tokens`` is
  prompt minus cached.
- Nothing is estimated. An attempt with a missing or invalid counter adds no
  tokens. Its arm's coverage is then ``partial`` (other attempts were measured)
  or ``unavailable`` (none were), and the totals cover only measured attempts.
- An attempt that failed before it was scored still used tokens. It is counted
  here as an unscored attempt and nowhere else: scores never include it.
- Case rows are validated against the evaluation dataset's own case IDs. Every
  row uses one of them at most once, and an attempt whose only name is Harbor's
  32-character cut of a case name is attributed to that case, so its missing
  telemetry stays with the canonical case. A row that cannot be matched is a
  report-integrity problem: it keeps its recorded name, is marked, and
  ``case_id_validation`` names it, so no report shows it as a dataset case.
- A change (with skill minus baseline) needs complete counts in both arms for
  the same cases and attempts, and both arms must have succeeded: the report
  withholds the scores of an arm that failed or has no known run status.
- A sum above 2**53 - 1, the largest integer JSON keeps exact, is not reported.

``resource_comparison`` keeps its own totals, which need every attempt of an
arm scored and measured; this module does not change them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from skillevaluator.tier3.harbor.resource_usage import _MAX_SAFE_INTEGER, _is_measurement

TOKEN_USAGE_SCHEMA_VERSION = "1"
TOKEN_MEASURES = (
    "prompt_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
# Loaded agent data keeps, per arm, the attempts whose usage no report trial
# carries: attempts without a usable reward (``scored`` false) and scored
# attempts of an arm whose scores are withheld. See ``report_data.load_agent_data``.
USAGE_ONLY_ATTEMPT_KEYS = ("usage_only_attempts", "usage_only_attempts_baseline")
# (payload arm, condition key, report trials key, usage-only attempts key)
_ARMS = (
    ("with_skill", "with_skill", "trials", "usage_only_attempts"),
    ("baseline", "without_skill", "trials_baseline", "usage_only_attempts_baseline"),
)
_ARM_NAMES = {"with_skill": "with-skill run", "baseline": "baseline"}
# Harbor names a trial after its task, cut to 32 characters, and the task after
# its case. A failed attempt without a Harbor result carries only that trial name.
_HARBOR_TRIAL_NAME_CHARS = 32
# A stop-on-pass attempt's trial is named ``<skill>-<agent>-<with|without>-<case>-attemptNNN__<Harbor trial>``.
_STOP_ON_PASS_ATTEMPT = re.compile(r"-attempt0*[1-9][0-9]*__.")
# Run statuses whose scores the report reads (``succeeded``), that never ran
# (``skipped``), or that the coverage notices name (``failed``).
_KNOWN_ARM_STATUSES = {"succeeded", "skipped", "failed"}
# The condition each payload ``mode`` leaves out, as the reports read it.
_UNEXECUTED_CONDITIONS = {
    "with_skill_only": frozenset({"without_skill"}),
    "baseline_only": frozenset({"with_skill"}),
}
_ACCOUNTING = {
    "prompt_includes_cached": True,
    "total_tokens": "prompt_tokens + completion_tokens",
    "uncached_prompt_tokens": "prompt_tokens - cached_tokens",
    "change": "with_skill - baseline total_tokens; percent of the baseline total",
}


def _count(value: object) -> int | None:
    """Return a non-negative whole token count; bools, strings, fractions and NaN are not counts."""
    return int(value) if _is_measurement(value) and float(value).is_integer() else None


def normalize_token_counts(raw: object) -> dict[str, Any] | None:
    """Normalize one attempt's token counters, or return ``None`` when it recorded none.

    ``raw`` is a report trial's ``tokens`` object (``prompt``, ``completion``,
    ``cached``) or ATIF-style counters (``prompt_tokens`` and so on). Harbor
    omits a zero cache counter, so a missing cached count is 0 once the prompt
    count is present. ``complete`` is ``False`` when a counter is missing or
    invalid, or when cached exceeds prompt; ``invalid_counters`` names the
    rejected fields, never their values.
    """
    if not isinstance(raw, Mapping):
        return None
    counts: dict[str, int | None] = {}
    invalid: list[str] = []
    for field, short in (("prompt_tokens", "prompt"), ("completion_tokens", "completion"), ("cached_tokens", "cached")):
        value = raw.get(short, raw.get(field))
        counts[field] = _count(value)
        if value is not None and counts[field] is None:
            invalid.append(field)
    if all(count is None for count in counts.values()) and not invalid:
        return None
    prompt, completion, cached = counts["prompt_tokens"], counts["completion_tokens"], counts["cached_tokens"]
    if cached is None and prompt is not None and "cached_tokens" not in invalid:
        cached = 0
    if prompt is not None and cached is not None and cached > prompt:
        invalid.append("cached_tokens")
        cached = None
    normalized: dict[str, Any] = {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "uncached_prompt_tokens": prompt - cached if prompt is not None and cached is not None else None,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
        "complete": not invalid and all(value is not None for value in (prompt, completion, cached)),
    }
    if invalid:
        normalized["invalid_counters"] = sorted(invalid)
    return normalized


@dataclass(frozen=True)
class _Attempt:
    case: str
    tokens: dict[str, Any] | None
    scored: bool


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _case_label(trial: Mapping[str, Any]) -> str:
    for key in ("entry_id", "trial_id"):
        value = trial.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _stop_on_pass_case(trial_id: str, case: str) -> bool:
    """Whether ``trial_id`` is a stop-on-pass attempt of ``case``: ``...-<with|without>-<case>-attemptNNN__...``."""
    marker = f"-{case}"
    index = trial_id.rfind(marker + "-attempt")
    return (
        index >= 0
        and trial_id[:index].endswith(("-with", "-without"))
        and _STOP_ON_PASS_ATTEMPT.match(trial_id, index + len(marker)) is not None
    )


def _matching_case(label: str, trial_id: str, known_cases: set[str]) -> str:
    """Map an attempt to its case when exactly one case matches.

    Every attempt is resolved against the dataset's IDs, scored or not: a
    scored attempt whose ``entry_id`` fell back to Harbor's cut of the case
    name needs the same canonicalization as an unscored one.

    ``label`` is the case directory the attempt's Harbor result names, or its
    trial name without Harbor's ``__<id>`` suffix. Without a known case of that
    name, a stop-on-pass trial name or Harbor's 32-character cut of the case
    name decides.
    """
    if label in known_cases:
        return label
    for matches in (
        [case for case in known_cases if _stop_on_pass_case(trial_id, case)],
        [case for case in known_cases if case[:_HARBOR_TRIAL_NAME_CHARS].rstrip("_-") == label],
    ):
        if len(matches) == 1:
            return matches[0]
    return label


def _pass_at_k_branch(agent: Mapping[str, Any], condition: str) -> Mapping[str, Any]:
    return _mapping(_mapping(agent.get("pass_at_k")).get(condition))


def _known_cases(agent: Mapping[str, Any], dataset_cases: frozenset[str]) -> set[str]:
    """Return the case names an attempt may be matched to.

    The evaluation dataset decides when the report knows it: its IDs are the
    only canonical ones, and an attempt whose arm died before it was scored
    carries Harbor's cut trial name, which no recorded trial has to spell out.
    Without a dataset the recorded trials and pass@k cases stand in, as before.
    """
    if dataset_cases:
        return set(dataset_cases)
    known = {_case_label(trial) for _arm, _condition, key, _usage in _ARMS for trial in _rows(agent.get(key))}
    for _arm, condition, _key, usage_key in _ARMS:
        known.update(str(case) for case in _mapping(_pass_at_k_branch(agent, condition).get("cases")))
        known.update(_case_label(row) for row in _rows(agent.get(usage_key)) if row.get("scored") is True)
    return known


def _agent_attempts(agent: Mapping[str, Any], known_cases: set[str]) -> dict[str, list[_Attempt]]:
    attempts: dict[str, list[_Attempt]] = {}
    for arm, _condition, key, usage_key in _ARMS:
        attempts[arm] = []
        for trial in _rows(agent.get(key)):
            trial_id = trial.get("trial_id")
            attempts[arm].append(
                _Attempt(
                    _matching_case(_case_label(trial), trial_id if isinstance(trial_id, str) else "", known_cases),
                    normalize_token_counts(trial.get("tokens")),
                    True,
                )
            )
        for row in _rows(agent.get(usage_key)):
            scored = row.get("scored") is True
            # Scored or not, the name an attempt carries is its case directory,
            # its recorded entry_id or its trial name, and any of the three may
            # be Harbor's cut of the case name, so every one is resolved
            # against the dataset's IDs.
            trial_id = row.get("trial_id")
            attempts[arm].append(
                _Attempt(
                    _matching_case(_case_label(row), trial_id if isinstance(trial_id, str) else "", known_cases),
                    normalize_token_counts(row.get("tokens")),
                    scored,
                )
            )
    return attempts


def _condition(agent: Mapping[str, Any], condition: str) -> Mapping[str, Any]:
    return _mapping(_mapping(agent.get("conditions")).get(condition))


def _condition_not_run(agent: Mapping[str, Any], condition: str, *, attempted: bool, unexecuted: bool) -> bool:
    """Whether the arm is known not to have run.

    It was skipped, or it had no planned work and no recorded status. An arm
    without attempts also did not run when the payload ``mode`` leaves it out
    (``unexecuted``) or the agent has no record of it; the score tables show
    such a baseline as not run.
    """
    record = _condition(agent, condition)
    status = record.get("execution_status")
    if status == "skipped":
        return True
    if status in {"succeeded", "failed"} or (_count(record.get("scored_attempts")) or 0) > 0:
        return False
    if not attempted and (unexecuted or condition not in _mapping(agent.get("conditions"))):
        return True
    return _count(record.get("expected_attempts")) == 0


def _arm_expected(agent: Mapping[str, Any], condition: str, observed: int) -> int:
    """Return the attempts the arm should have run: its recorded expected attempts plus recovered ones.

    A timed-out attempt that a later attempt recovered is not expected for
    scoring, but it ran and used tokens. Observed attempts never count as
    missing, so the expected count is at least the observed one.
    """
    record = _condition(agent, condition)
    expected = _count(record.get("expected_attempts"))
    if expected is None:
        return observed
    return max(observed, expected + (_count(record.get("recovered_attempts_total")) or 0))


def _pass_at_k_cases(agent: Mapping[str, Any], condition: str, known_cases: set[str]) -> dict[str, Mapping[str, Any]]:
    """Return an arm's pass@k record per case, under the case names the rows use.

    Two keys can canonicalize onto one case, Harbor's cut of a name beside the
    full name. Today's collector lists a dataset ID before an unmatched cut
    name, so keeping the first would keep the canonical record, but that
    ordering belongs to a neighbouring module and holds only while it has
    expected case IDs. The merge here does not depend on it: a key that is
    already a dataset ID wins, and otherwise the larger ``attempts_skipped``
    does, so the merged row cannot claim attempts that never had to run.
    """
    ranked: dict[str, tuple[tuple[int, int], Mapping[str, Any]]] = {}
    for case, record in _mapping(_pass_at_k_branch(agent, condition).get("cases")).items():
        name = _matching_case(str(case), "", known_cases)
        fields = _mapping(record)
        rank = (int(str(case) in known_cases), _count(fields.get("attempts_skipped")) or 0)
        if name not in ranked or rank > ranked[name][0]:
            ranked[name] = (rank, fields)
    return {name: fields for name, (_rank, fields) in ranked.items()}


def _case_expected(
    agent: Mapping[str, Any],
    condition: str,
    case: str,
    observed: int,
    pass_at_k_cases: Mapping[str, Mapping[str, Any]],
) -> int:
    """Return the attempts a case should have run, from pass@k: ``k`` minus attempts skipped after a pass."""
    branch = _pass_at_k_branch(agent, condition)
    record = _mapping(pass_at_k_cases.get(case))
    k = _count(branch.get("k"))
    if not record or k is None:
        return observed
    return max(observed, k - min(k, _count(record.get("attempts_skipped")) or 0))


def _arm_summary(attempts: list[_Attempt], *, expected: int, not_run: bool) -> dict[str, Any]:
    totals = dict.fromkeys(TOKEN_MEASURES, 0)
    measured = 0
    reasons: set[str] = set()
    for attempt in attempts:
        tokens = attempt.tokens
        if tokens is None or not tokens["complete"]:
            # An incomplete attempt adds none of its counters, so every total
            # keeps total = prompt + completion over the same attempts.
            reasons.add(
                "invalid_token_counters" if tokens and "invalid_counters" in tokens else "missing_token_counters"
            )
            continue
        measured += 1
        for measure in TOKEN_MEASURES:
            totals[measure] += tokens[measure]
    observed = len(attempts)
    # ``total_tokens`` is the largest sum, so it alone can pass the exact JSON range.
    if measured and totals["total_tokens"] > _MAX_SAFE_INTEGER:
        reasons.add("total_out_of_range")
    if "total_out_of_range" in reasons:
        status = "unavailable"
    elif not attempts and expected == 0:
        status = "not_run" if not_run else "unavailable"
        if not not_run:
            reasons.add("no_attempts_recorded")
    elif measured == observed == expected:
        status = "complete"
    else:
        status = "partial" if measured else "unavailable"
    if observed < expected:
        reasons.add("missing_attempts")
    return {
        "totals": totals if measured and "total_out_of_range" not in reasons else None,
        "coverage": {
            "status": status,
            "expected_attempts": expected,
            "observed_attempts": observed,
            "measured_attempts": measured,
            "unscored_attempts": sum(not attempt.scored for attempt in attempts),
            "reason_codes": sorted(reasons),
        },
    }


def _combine(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum arm summaries of several agents; the sum is complete only when every part is.

    A part whose total is out of the exact JSON range makes the sum out of range too.
    """
    totals = dict.fromkeys(TOKEN_MEASURES, 0)
    coverage = dict.fromkeys(("expected_attempts", "observed_attempts", "measured_attempts", "unscored_attempts"), 0)
    reasons: set[str] = set()
    for summary in summaries:
        for measure in TOKEN_MEASURES:
            totals[measure] += (summary["totals"] or {}).get(measure, 0)
        for key in coverage:
            coverage[key] += summary["coverage"][key]
        reasons.update(summary["coverage"]["reason_codes"])
    statuses = {summary["coverage"]["status"] for summary in summaries}
    if totals["total_tokens"] > _MAX_SAFE_INTEGER:
        reasons.add("total_out_of_range")
    if "total_out_of_range" in reasons:
        status = "unavailable"
    elif not summaries or statuses == {"not_run"}:
        status = "not_run"
    elif statuses <= {"complete", "not_run"}:
        status = "complete"
    else:
        status = "partial" if coverage["measured_attempts"] else "unavailable"
    return {
        "totals": totals if coverage["measured_attempts"] and "total_out_of_range" not in reasons else None,
        "coverage": {"status": status, **coverage, "reason_codes": sorted(reasons)},
    }


def _delta(
    with_skill: dict[str, Any],
    baseline: dict[str, Any],
    *,
    blockers: list[tuple[str, str]],
) -> dict[str, Any]:
    """Return with skill minus baseline total tokens, or why there is none.

    ``blockers`` are ``(code, sentence)`` pairs found by the caller, such as a
    failed arm or arms that ran different cases.
    """
    reasons: list[tuple[str, str]] = []
    for arm, summary in (("with_skill", with_skill), ("baseline", baseline)):
        status = summary["coverage"]["status"]
        if status == "not_run":
            reasons.append((f"{arm}_not_run", f"The {_ARM_NAMES[arm]} did not run."))
        elif "total_out_of_range" in summary["coverage"]["reason_codes"]:
            reasons.append(
                (
                    f"{arm}_total_out_of_range",
                    f"{_ARM_NAMES[arm].capitalize()} token total is above {_MAX_SAFE_INTEGER:,}, "
                    "the largest integer JSON keeps exact.",
                )
            )
        elif status != "complete":
            coverage = summary["coverage"]
            reasons.append(
                (
                    f"{arm}_tokens_incomplete",
                    f"{_ARM_NAMES[arm].capitalize()} token counts cover {coverage['measured_attempts']} of "
                    f"{coverage['expected_attempts']} attempts."
                    if coverage["observed_attempts"]
                    else f"No {_ARM_NAMES[arm]} attempts were recorded.",
                )
            )
    reasons.extend(blockers)
    if not reasons and with_skill["coverage"]["observed_attempts"] != baseline["coverage"]["observed_attempts"]:
        reasons.append(("attempt_scope_mismatch", "The arms ran different numbers of attempts."))
    if reasons:
        return {
            "status": "unavailable",
            "total_tokens": None,
            "total_tokens_change_percent": None,
            "reason_codes": [code for code, _sentence in reasons],
            "reason": " ".join(sentence for _code, sentence in reasons),
        }
    difference = with_skill["totals"]["total_tokens"] - baseline["totals"]["total_tokens"]
    baseline_total = baseline["totals"]["total_tokens"]
    return {
        "status": "available",
        "total_tokens": difference,
        "total_tokens_change_percent": round(difference / baseline_total * 100.0, 2) if baseline_total else None,
        "reason_codes": [] if baseline_total else ["zero_baseline"],
        "reason": None if baseline_total else "The baseline used no tokens, so there is no percentage change.",
    }


def _withheld_arm_blockers(agent: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Name each arm whose scores the report withholds: its usage is real but not comparable.

    A failed arm reuses the coverage notice's short form, such as ``No change:
    baseline incomplete (1 of 2 attempts scored).`` An arm that ran attempts
    without a recognized run status (``unknown``) gets ``<arm>_status_unknown``.
    An arm without attempts already has ``<arm>_not_run`` or
    ``<arm>_tokens_incomplete``.
    """
    from skillevaluator.reporting.tier3_coverage import incomplete_arms

    codes = {"with_skill": "with_skill_failed", "without_skill": "baseline_failed"}
    blockers = [(codes[arm.condition], f"No change: {arm}.") for arm in incomplete_arms(dict(agent))]
    for arm, condition, _key, _usage in _ARMS:
        status = _condition(agent, condition).get("execution_status")
        if status not in _KNOWN_ARM_STATUSES and arms[arm]["coverage"]["observed_attempts"]:
            blockers.append((f"{arm}_status_unknown", f"No change: {_ARM_NAMES[arm]} status unknown."))
    return blockers


def _agent_usage(
    agent: Mapping[str, Any], unexecuted: frozenset[str], dataset_cases: frozenset[str]
) -> dict[str, Any] | None:
    known_cases = _known_cases(agent, dataset_cases)
    attempts = _agent_attempts(agent, known_cases)
    pass_at_k = {condition: _pass_at_k_cases(agent, condition, known_cases) for _a, condition, _k, _u in _ARMS}
    arms: dict[str, dict[str, Any]] = {}
    for arm, condition, _key, _unscored in _ARMS:
        not_run = _condition_not_run(
            agent, condition, attempted=bool(attempts[arm]), unexecuted=condition in unexecuted
        )
        arms[arm] = _arm_summary(
            attempts[arm],
            # An arm that did not run expects nothing, whatever its record planned.
            expected=0 if not_run and not attempts[arm] else _arm_expected(agent, condition, len(attempts[arm])),
            not_run=not_run,
        )
    if all(summary["coverage"]["status"] == "not_run" for summary in arms.values()):
        return None
    withheld = _withheld_arm_blockers(agent, arms)
    cases_by_arm = {arm: Counter(attempt.case for attempt in attempts[arm]) for arm in arms}
    # An arm without attempts already says why there is no change.
    scope = (
        [("case_scope_mismatch", "The arms ran different cases or attempts per case.")]
        if all(cases_by_arm.values()) and cases_by_arm["with_skill"] != cases_by_arm["baseline"]
        else []
    )
    case_rows = []
    for case in sorted(
        {*cases_by_arm["with_skill"], *cases_by_arm["baseline"]}
        | {case for cases in pass_at_k.values() for case in cases}
    ):
        case_arms = {}
        for arm, condition, _key, _unscored in _ARMS:
            case_attempts = [attempt for attempt in attempts[arm] if attempt.case == case]
            case_arms[arm] = _arm_summary(
                case_attempts,
                expected=0
                if arms[arm]["coverage"]["status"] == "not_run"
                else _case_expected(agent, condition, case, len(case_attempts), pass_at_k[condition]),
                not_run=arms[arm]["coverage"]["status"] == "not_run",
            )
        row = {
            "entry_id": case,
            **case_arms,
            "delta": _delta(case_arms["with_skill"], case_arms["baseline"], blockers=withheld),
        }
        # A row the dataset does not name is a report-integrity problem, not a
        # case: every renderer marks it instead of showing it as coverage.
        if dataset_cases and case not in dataset_cases:
            row["case_id_in_dataset"] = False
        case_rows.append(row)
    return {
        **arms,
        "delta": _delta(arms["with_skill"], arms["baseline"], blockers=[*withheld, *scope]),
        "cases": case_rows,
    }


def build_token_usage(
    agents: Mapping[str, Mapping[str, Any]],
    *,
    mode: str = "both",
    baseline_reused: bool = False,
    dataset_case_ids: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Return token usage per case, per agent and over all agents, or ``None`` when no attempt was measured.

    ``agents`` maps each agent to its report payload (``trials``,
    ``trials_baseline``, ``conditions``, ``pass_at_k`` and ``execution_status``)
    plus the loader's usage-only attempts under :data:`USAGE_ONLY_ATTEMPT_KEYS`.
    ``mode`` is the payload's ``mode``: an arm that ``with_skill_only`` or
    ``baseline_only`` leaves out is ``not_run`` for every agent without attempts in it.
    ``baseline_reused`` marks a run that reused an earlier run's baseline
    (``--reuse-baseline``): its baseline counts are that run's usage, so the
    result says so with ``baseline_reused``.

    ``dataset_case_ids`` are the evaluation dataset's canonical case IDs. Given
    them, every case row uses one of those IDs, each at most once: an attempt
    whose only name is Harbor's 32-character cut of a case name is attributed
    to that case, so its missing telemetry stays with the canonical case
    instead of becoming a row of its own. A row that still cannot be matched
    keeps its recorded name and is marked ``case_id_in_dataset`` false, and
    ``case_id_validation`` names it, so no report shows it as a dataset case.
    """
    dataset_cases = frozenset(case for case in dataset_case_ids or () if isinstance(case, str) and case)
    unexecuted = _UNEXECUTED_CONDITIONS.get(mode, frozenset())
    usage: dict[str, dict[str, Any]] = {}
    for name in sorted(agents):
        agent_usage = _agent_usage(_mapping(agents[name]), unexecuted, dataset_cases)
        if agent_usage is not None:
            usage[name] = agent_usage
    if not any(
        agent_usage[arm]["coverage"]["measured_attempts"]
        for agent_usage in usage.values()
        for arm in ("with_skill", "baseline")
    ):
        return None
    dataset = {
        arm: _combine([agent_usage[arm] for agent_usage in usage.values()]) for arm in ("with_skill", "baseline")
    }
    agents_without_change = (
        []
        if all(item["delta"]["status"] == "available" for item in usage.values())
        else [("agent_scope_mismatch", "Not every agent has a token change, so there is no change over all agents.")]
    )
    return {
        "schema_version": TOKEN_USAGE_SCHEMA_VERSION,
        "scored": False,
        "accounting": dict(_ACCOUNTING),
        **({"baseline_reused": True} if baseline_reused else {}),
        **_case_id_validation(usage, dataset_cases),
        "agents": usage,
        "dataset": {
            **dataset,
            "delta": _delta(dataset["with_skill"], dataset["baseline"], blockers=agents_without_change),
        },
    }


def _case_id_validation(
    usage: Mapping[str, Mapping[str, Any]], dataset_cases: frozenset[str]
) -> dict[str, dict[str, Any]]:
    """Say whether every case row names a canonical dataset case, and name the rows that do not.

    Nothing is recorded without a dataset to check against: the report then
    cannot tell a canonical case ID from a trial name, and says nothing either way.
    """
    if not dataset_cases:
        return {}
    unmatched = sorted(
        {
            str(row.get("entry_id"))
            for agent in usage.values()
            for row in agent.get("cases", [])
            if isinstance(row, Mapping) and row.get("case_id_in_dataset") is False
        }
    )
    return {
        "case_id_validation": {
            "status": "unverified_case_ids" if unmatched else "validated",
            "dataset_case_ids": len(dataset_cases),
            "unmatched_case_ids": unmatched,
        }
    }


def per_agent_token_usage(usage: object) -> dict[str, Any] | None:
    """Return ``usage`` without its per-case rows, for the terminal summary of a run."""
    if not isinstance(usage, Mapping) or not isinstance(usage.get("agents"), Mapping):
        return None
    agents = {
        name: {key: value for key, value in agent.items() if key != "cases"}
        for name, agent in usage["agents"].items()
        if isinstance(agent, Mapping)
    }
    return {**usage, "agents": agents}


def drop_token_usage_cases(usage: object) -> int:
    """Empty the per-case rows of ``usage`` in place and return how many were dropped.

    The report payload drops them first when it would pass its limits; the
    agent and all-agent sums stay, and ``case_rows_omitted`` counts the rows.
    """
    agents = usage.get("agents") if isinstance(usage, dict) else None
    dropped = 0
    for agent in agents.values() if isinstance(agents, dict) else []:
        if isinstance(agent, dict) and isinstance(agent.get("cases"), list) and agent["cases"]:
            dropped += len(agent["cases"])
            agent["cases"] = []
    if dropped and isinstance(usage, dict):
        usage["case_rows_omitted"] = dropped
    return dropped
