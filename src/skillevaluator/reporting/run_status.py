# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one place a validation run's status is decided.

``resolve_run_status(results)`` returns a :class:`RunStatus`: the headline every report leads with, each tier's
status, each result's row status, the BENCHMARK.md publication verdict, and the reasons behind them. Every
reporter (terminal summary and footer, JSON, Markdown, HTML, SARIF, BENCHMARK.md) renders what it returns and never
works a status out on its own, so two surfaces cannot give one run two verdicts.

The precedence (docs/plugin-evaluation.mdx, "Reports and verdicts"):

1. A real failure outranks missing evidence. A check that found a blocking problem is FAILED even when a scanner in
   the same run, or in the same check, did not complete; the reasons still name what did not complete.
2. Missing evidence outranks a pass. A scanner that did not complete, or a partial Tier 3 plugin run, is INCOMPLETE.
3. A skipped check is not a pass. It is SKIPPED, and a tier where nothing ran is SKIPPED.
4. The Integration block (Bundle lift) and the lift ladder are advisory: an INCONCLUSIVE or negative lift, or a
   ladder that was not measured, never changes a verdict.
5. The headline is the exit gate: a result outside the gate (an advisory tier) never changes it. It is PASSED
   exactly when ``validate`` exits 0, and the advisory problems are named instead.

Tier 3 adds four rules: a FAIL verdict or a confirmed Skill Lift regression is FAILED even for a partial run; a run
that never ran is FAILED when ``--block-on-agent-eval`` makes Tier 3 gate and SKIPPED otherwise; a run whose
verdict is NEUTRAL is NEUTRAL; and a run whose verdict is INCONCLUSIVE (the skill never activated, so no lift verdict
was computed) is INCOMPLETE, never PASSED. When ``--block-on-agent-eval`` makes Tier 3 gate, only a completed PASS
with complete attempt coverage passes the gate (``passes_required_gate``), so a gated NEUTRAL or INCONCLUSIVE run
fails it.

BENCHMARK.md is a publication card and is stricter than the gate: it also wants complete Tier 3 provenance (Tier 3
is publication-required by default), and a skipped required check or any missing scan keeps it INCOMPLETE. It is
never kinder than the headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from skillevaluator.reporting.base import (
    is_advisory_agent_eval_skip,
    is_cleanly_skipped,
    is_partial_plugin_agent_eval,
    passes_required_gate,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from skillevaluator.models import Finding, ValidationResult

__all__ = [
    "TIER3_INCONCLUSIVE_REASON",
    "RunStatus",
    "Status",
    "TierStatus",
    "is_tier2_validator_name",
    "is_tier3_inconclusive",
    "resolve_run_status",
]

# Why an INCONCLUSIVE Tier 3 run is INCOMPLETE, in report words.
TIER3_INCONCLUSIVE_REASON = "Tier 3 is inconclusive: the skill never activated, so no lift verdict was computed"


class Status(StrEnum):
    """A status, as every machine-readable report spells it."""

    PASSED = "passed"
    NEUTRAL = "neutral"  # Tier 3 and the publication verdict only: complete, but not every dimension passes
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    SKIPPED = "skipped"  # a check or tier that did not run
    NOT_RUN = "not_run"  # a tier with no result

    @property
    def verdict(self) -> str:
        """The verdict word a reader sees: ``PASS``, ``NEUTRAL``, ``FAIL``, ``INCOMPLETE``, ``SKIPPED`` or ``NOT RUN``."""
        return _VERDICT_WORDS[self]


_VERDICT_WORDS = {
    Status.PASSED: "PASS",
    Status.NEUTRAL: "NEUTRAL",
    Status.FAILED: "FAIL",
    Status.INCOMPLETE: "INCOMPLETE",
    Status.SKIPPED: "SKIPPED",
    Status.NOT_RUN: "NOT RUN",
}


@dataclass(frozen=True)
class TierStatus:
    """One tier: its status, its stricter BENCHMARK.md row, whether it gates the exit code, and its results.

    ``blocking`` is the gate ``validate`` recorded: ``True`` when the tier can fail the exit code, ``False`` when it
    is advisory, ``None`` when nothing was recorded (results built outside ``validate``).
    """

    tier: int
    status: Status
    publication: Status
    blocking: bool | None
    results: tuple[ValidationResult, ...]


@dataclass(frozen=True)
class RunStatus:
    """What a validation run's reports say about it. Build it with :func:`resolve_run_status`.

    ``headline`` is ``PASSED``, ``FAILED`` or ``INCOMPLETE``, and is ``PASSED`` exactly when the exit gate passed.
    ``publication`` is the BENCHMARK.md verdict (``PASSED``, ``NEUTRAL``, ``FAILED`` or ``INCOMPLETE``), never kinder
    than the headline; ``publication_advisory`` says its FAIL came only from Tier 3 outside the exit gate. ``plugin``
    is the plugin row: the headline, but INCOMPLETE when any evidence is missing, even outside the gate.

    ``failed`` and ``incomplete`` are the results that failed the exit gate, for a real failure or only for missing
    evidence. ``missing`` names what did not complete (every scanner, and Tier 3 when it gated), even when the
    headline is FAILED. ``advisory`` names the tiers outside the gate that did not pass, Tier 3 first
    (``Tier 3 verdict FAIL``, ``Tier 3 INCOMPLETE``, ``Tier 2 FAIL``). ``reasons`` says all of it in report words.
    """

    headline: Status
    publication: Status
    publication_advisory: bool
    plugin: Status
    tiers: Mapping[int, TierStatus]
    failed: tuple[ValidationResult, ...]
    incomplete: tuple[ValidationResult, ...]
    missing: tuple[str, ...]
    advisory: tuple[str, ...]
    reasons: tuple[str, ...]
    _outcomes: Mapping[int, Status] = field(repr=False, compare=False)

    def of(self, result: ValidationResult) -> Status:
        """The row status of one result: ``PASSED``, ``FAILED``, ``INCOMPLETE`` or ``SKIPPED``."""
        outcome = self._outcomes.get(id(result))
        return outcome if outcome is not None else _outcome(result)


def resolve_run_status(results: Iterable[ValidationResult]) -> RunStatus:
    """Decide every status a run's reports show, from its validation results (in any order)."""
    results = list(results)
    outcomes = {id(result): _outcome(result) for result in results}
    by_tier: dict[int, list[ValidationResult]] = {1: [], 2: [], 3: []}
    for result in results:
        by_tier[_tier_of(result)].append(result)

    gate_failures = [result for result in results if not passes_required_gate(result)]
    failed = tuple(result for result in gate_failures if outcomes[id(result)] is not Status.INCOMPLETE)
    incomplete = tuple(result for result in gate_failures if outcomes[id(result)] is Status.INCOMPLETE)
    headline = Status.FAILED if failed else Status.INCOMPLETE if incomplete else Status.PASSED

    evidence = _Tier3Evidence.of(results, by_tier[3])
    tiers = {
        number: TierStatus(
            tier=number,
            status=_tier_status(number, tier_results, outcomes, evidence),
            publication=_publication_tier_status(number, tier_results, outcomes, evidence),
            blocking=_recorded_gate(tier_results),
            results=tuple(tier_results),
        )
        for number, tier_results in by_tier.items()
    }

    missing = list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans))
    if any(_is_tier3(result) for result in incomplete):
        missing.append("Tier 3")
    advisory = _advisory_labels(by_tier, outcomes)
    reasons = [_failure_reason(result) for result in failed]
    tier3_inconclusive = any(is_tier3_inconclusive(result) for result in incomplete)
    reasons += [
        TIER3_INCONCLUSIVE_REASON if name == "Tier 3" and tier3_inconclusive else f"{name} did not complete"
        for name in missing
    ]
    reasons += [f"{label} is advisory" for label in advisory]

    evidence_missing = any(
        result.is_incomplete or is_partial_plugin_agent_eval(result) or outcomes[id(result)] is Status.INCOMPLETE
        for result in results
    )
    publication = _publication(results, headline, by_tier[3], evidence)
    return RunStatus(
        headline=headline,
        publication=publication,
        publication_advisory=bool(
            publication is Status.FAILED and evidence.fails and headline is Status.PASSED and tiers[3].blocking is False
        ),
        plugin=Status.INCOMPLETE if headline is Status.PASSED and evidence_missing else headline,
        tiers=MappingProxyType(tiers),
        failed=failed,
        incomplete=incomplete,
        missing=tuple(missing),
        advisory=advisory,
        reasons=tuple(dict.fromkeys(reasons)),
        _outcomes=MappingProxyType(outcomes),
    )


# ---------------------------------------------------------------------------
# One result
# ---------------------------------------------------------------------------


def _outcome(result: ValidationResult) -> Status:
    """A result's own status, before any gate: the precedence applied to one check."""
    if _is_tier3(result):
        return _tier3_outcome(result)
    if result.is_incomplete:
        # A scanner that did not complete adds warnings or plain errors, never a finding, so a blocking finding
        # in the same result is a real failure of its own.
        return Status.FAILED if _has_blocking_finding(result) else Status.INCOMPLETE
    if _skipped(result):
        return Status.SKIPPED if result.passed else Status.FAILED
    return Status.PASSED if result.passed else Status.FAILED


def _tier3_outcome(result: ValidationResult) -> Status:
    metadata = _mapping(result.metadata)
    payload = _mapping(metadata.get("agent_eval"))
    summary = _mapping(payload.get("summary"))
    if metadata.get("tier3_gate_failures") or _verdict(payload) == "fail":
        # A FAIL verdict or a confirmed Skill Lift regression: the evaluated parts failed, even in a partial run.
        return Status.FAILED
    if _tier3_not_run(result) or is_cleanly_skipped(result):
        # Never ran: a failed gate under --block-on-agent-eval, an advisory skip otherwise.
        return Status.SKIPPED if passes_required_gate(result) else Status.FAILED
    if is_partial_plugin_agent_eval(result):
        return Status.INCOMPLETE
    if result.is_incomplete:
        # A Tier 3 scan that did not complete is missing evidence, unless the result also has a blocking finding.
        return Status.FAILED if _has_blocking_finding(result) else Status.INCOMPLETE
    statuses = {metadata.get("execution_status"), payload.get("execution_status"), summary.get("execution_status")}
    if not result.passed:
        return Status.INCOMPLETE if "skipped" in statuses else Status.FAILED
    execution = str(payload.get("execution_status") or summary.get("execution_status") or "").lower()
    if execution == "failed":
        # A failed execution outranks a pass verdict, as in the publication assessment.
        return Status.FAILED
    if execution == "incomplete":
        return Status.INCOMPLETE
    if execution and execution != "succeeded":
        # Advisory and passed, but nothing was evaluated (no task source): it did not run.
        return Status.SKIPPED
    if _verdict(payload) == "inconclusive":
        # The skill never activated, so no lift verdict was computed: evidence is missing, not a pass.
        return Status.INCOMPLETE
    return Status.PASSED


def is_tier3_inconclusive(result: ValidationResult) -> bool:
    """Whether *result* is a Tier 3 run whose verdict is INCONCLUSIVE (the skill never activated)."""
    return _is_tier3(result) and _verdict(_mapping(_mapping(result.metadata).get("agent_eval"))) == "inconclusive"


def _has_blocking_finding(result: ValidationResult) -> bool:
    """Whether the result records a blocking finding of its own, not only missing evidence."""
    return any(_finding_is_error(finding, result) for finding in result.findings)


def _finding_is_error(finding: Finding, result: ValidationResult) -> bool:
    """CRITICAL and HIGH, or a MEDIUM or LOW that *result* records as an error (``fail_on_medium``)."""
    if finding.severity.value in {"critical", "high"}:
        return True
    # merge_with_prefix lists a bundled skill's error as "[skill] [TAG] msg in path"; recognize every form.
    return any(form in result.errors for form in finding.legacy_string_forms())


def _skipped(result: ValidationResult) -> bool:
    metadata = _mapping(result.metadata)
    return bool(metadata.get("skipped")) or metadata.get("execution_status") == "skipped"


def _tier3_not_run(result: ValidationResult) -> bool:
    """A Tier 3 run that never ran (its payload records an advisory skip), gated or not."""
    if not _is_tier3(result):
        return False
    provenance = _mapping(_mapping(_mapping(result.metadata).get("agent_eval")).get("provenance"))
    return bool(provenance.get("advisory") and provenance.get("reason") == "skipped")


def _failure_reason(result: ValidationResult) -> str:
    if _is_tier3(result):
        labels = _mapping(result.metadata).get("tier3_gate_failures")
        if isinstance(labels, list) and labels:
            return " · ".join(str(label) for label in labels)
        if _tier3_not_run(result):
            return "Tier 3 live evaluation did not run"
        if _verdict(_mapping(_mapping(result.metadata).get("agent_eval"))) == "neutral":
            return "Tier 3 verdict NEUTRAL did not pass the gate"
        return "Tier 3 live evaluation failed"
    return f"{result.validator_name or 'validation'} failed in Tier {_tier_of(result)}"


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

_TIER2_VALIDATOR_MARKERS = ("similarity", "dedup", "context optimization")


def is_tier2_validator_name(validator_name: str | None) -> bool:
    """Return whether a validator name belongs to Tier 2 (deduplication and similarity)."""
    normalized_name = " ".join((validator_name or "").casefold().replace("_", " ").replace("-", " ").split())
    return any(marker in normalized_name for marker in _TIER2_VALIDATOR_MARKERS)


def _is_tier3(result: ValidationResult) -> bool:
    return result.validator_name == "AGENT_EVAL" or bool(_mapping(result.metadata).get("agent_eval"))


def _tier_of(result: ValidationResult) -> int:
    """The tier ``validate`` recorded for the result; otherwise the one its name or payload implies."""
    tier = _mapping(_mapping(result.metadata).get("gating")).get("tier")
    if tier in (1, 2, 3) and not isinstance(tier, bool):
        return int(tier)
    if _is_tier3(result):
        return 3
    if is_tier2_validator_name(result.validator_name) or any(
        finding.category == "CONTENT_DEDUP" for finding in result.findings
    ):
        return 2
    return 1


def _recorded_gate(results: Sequence[ValidationResult]) -> bool | None:
    values = [
        gating["blocking"]
        for result in results
        if isinstance(gating := _mapping(result.metadata).get("gating"), dict)
        and isinstance(gating.get("blocking"), bool)
    ]
    return any(values) if values else None


def _tier_status(
    number: int,
    results: Sequence[ValidationResult],
    outcomes: Mapping[int, Status],
    evidence: _Tier3Evidence,
) -> Status:
    if not results:
        return Status.NOT_RUN
    statuses = [outcomes[id(result)] for result in results]
    if Status.FAILED in statuses:
        return Status.FAILED
    if Status.INCOMPLETE in statuses:
        return Status.INCOMPLETE
    if all(status is Status.SKIPPED for status in statuses):
        return Status.SKIPPED
    if number == 3 and evidence.verdict == "neutral":
        return Status.NEUTRAL
    return Status.PASSED


def _publication_tier_status(
    number: int,
    results: Sequence[ValidationResult],
    outcomes: Mapping[int, Status],
    evidence: _Tier3Evidence,
) -> Status:
    """The BENCHMARK.md Tier Status row: the tier's status, strict about skips and Tier 3 provenance."""
    if not results:
        return Status.NOT_RUN
    if number == 3 and evidence.partial:
        # A partial plugin run is INCOMPLETE unless its evaluated parts already failed.
        return Status.FAILED if evidence.fails else Status.INCOMPLETE
    if any(result.is_incomplete for result in results):
        real_failure = any(
            not passes_required_gate(result) and outcomes[id(result)] is not Status.INCOMPLETE for result in results
        )
        return Status.FAILED if real_failure else Status.INCOMPLETE
    if all(_publication_skipped(result) for result in results):
        optional = all(_optional_skip(result, evidence) for result in results)
        return Status.SKIPPED if optional else Status.INCOMPLETE
    if number == 3 and evidence.payload and evidence.fails:
        return Status.FAILED
    if any(_tier3_not_run(result) and not passes_required_gate(result) for result in results):
        return Status.FAILED
    if any(not result.passed and not _publication_skipped(result) for result in results):
        return Status.FAILED
    if number == 3 and evidence.payload:
        if evidence.execution and evidence.execution != "succeeded":
            return Status.INCOMPLETE
        if not evidence.complete:
            return Status.INCOMPLETE
        if evidence.verdict == "neutral":
            return Status.NEUTRAL
    return Status.PASSED


def _publication_skipped(result: ValidationResult) -> bool:
    """A check BENCHMARK.md counts as skipped: one that recorded a skip and did not fail, or a Tier 3 run outside the
    gate that never ran. A check that recorded a skip but failed is a failure, as on every other surface."""
    skipped = bool(_mapping(result.metadata).get("skipped")) and result.passed
    return skipped or is_advisory_agent_eval_skip(result)


def _optional_skip(result: ValidationResult, evidence: _Tier3Evidence) -> bool:
    """Whether a skipped check may stay skipped on the card: an optional check, an advisory Tier 3 skip, or a tier the
    persisted publication policy does not require (Tier 2 or Tier 3; Tier 1 is always required)."""
    if _mapping(result.metadata).get("optional") or is_advisory_agent_eval_skip(result):
        return True
    tier = _tier_of(result)
    if tier == 3:
        return not evidence.required
    if tier == 2:
        return not evidence.tier2_required
    return False


def _advisory_labels(
    by_tier: Mapping[int, Sequence[ValidationResult]], outcomes: Mapping[int, Status]
) -> tuple[str, ...]:
    """Name each tier outside the exit gate that did not pass, Tier 3 first (as the footer names it)."""
    labels: list[str] = []
    for result in by_tier[3]:
        if not passes_required_gate(result):
            continue
        outcome = outcomes[id(result)]
        if outcome is Status.FAILED:
            gate_labels = _mapping(result.metadata).get("tier3_gate_failures")
            has_labels = isinstance(gate_labels, list) and gate_labels
            labels.append(" · ".join(str(label) for label in gate_labels) if has_labels else "Tier 3 FAIL")
        elif outcome is Status.INCOMPLETE:
            inconclusive = is_tier3_inconclusive(result)
            labels.append(
                "Tier 3 INCOMPLETE (inconclusive: skill never activated)" if inconclusive else "Tier 3 INCOMPLETE"
            )
    for number in (1, 2):
        advisory = [outcomes[id(result)] for result in by_tier[number] if passes_required_gate(result)]
        if Status.FAILED in advisory:
            labels.append(f"Tier {number} FAIL")
        elif Status.INCOMPLETE in advisory:
            labels.append(f"Tier {number} INCOMPLETE")
    return tuple(dict.fromkeys(labels))


# ---------------------------------------------------------------------------
# Tier 3 evidence and the publication verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tier3Evidence:
    """The Tier 3 facts the publication verdict reads, from the run's first ``agent_eval`` payload."""

    payload: Mapping[str, Any]
    partial: bool
    fails: bool  # a FAIL verdict (recomputed from the dimensions) or a confirmed Skill Lift regression
    complete: bool  # a succeeded run with the provenance BENCHMARK.md requires
    verdict: str  # the effective verdict: a PASS whose dimensions are NEUTRAL is NEUTRAL
    execution: str
    required: bool  # the persisted publication policy: Tier 3 is required unless a catalog waived it
    tier2_required: bool  # the same policy for Tier 2

    @classmethod
    def of(cls, results: Sequence[ValidationResult], tier3: Sequence[ValidationResult]) -> _Tier3Evidence:
        from skillevaluator.reporting.base import (
            _agent_eval_rejected_truth_state,
            agent_eval_report_serialization_issue,
        )

        # The payload readers live with the publication card that prints them.
        from skillevaluator.reporting.benchmark import (
            _agent_eval_payload,
            _benchmark_policy,
            _tier3_dimension_verdict,
            _tier3_evidence_complete,
            _tier3_fail_reasons,
        )

        payload = _agent_eval_payload(list(results))
        policy = _benchmark_policy(list(results), payload)
        source = _mapping(payload)
        summary = _mapping(source.get("summary"))
        if isinstance(payload, dict) and agent_eval_report_serialization_issue(payload) is not None:
            # A payload the reports cannot carry whole (too wide or too deep): read only its bounded root truth,
            # never its agents, as the publication assessment does. It is never publication-complete.
            rejected_verdict, rejected_execution, explicit_failure, _nodes = _agent_eval_rejected_truth_state(payload)
            return cls(
                payload=source,
                partial=any(is_partial_plugin_agent_eval(result) for result in tier3),
                fails=explicit_failure,
                complete=False,
                verdict=rejected_verdict,
                execution=rejected_execution,
                required=policy["tier3_required"],
                tier2_required=policy["tier2_required"],
            )
        verdict = _verdict(source)
        if verdict == "pass" and _tier3_dimension_verdict(payload) == "neutral":
            verdict = "neutral"
        return cls(
            payload=source,
            partial=any(is_partial_plugin_agent_eval(result) for result in tier3),
            fails=bool(_tier3_fail_reasons(payload)),
            complete=_tier3_evidence_complete(payload),
            verdict=verdict,
            execution=str(source.get("execution_status") or summary.get("execution_status") or "").lower(),
            required=policy["tier3_required"],
            tier2_required=policy["tier2_required"],
        )


def _publication(
    results: Sequence[ValidationResult],
    headline: Status,
    tier3: Sequence[ValidationResult],
    evidence: _Tier3Evidence,
) -> Status:
    """The BENCHMARK.md verdict: the headline, plus what the card demands before it recommends publication."""
    # A real failure outranks missing evidence on the card too, and there a Tier 3 FAIL counts even when advisory.
    if headline is Status.FAILED or evidence.fails:
        return Status.FAILED
    if evidence.partial:
        # Tier 3 evaluated only part of the plugin: a partial result, never a recommendation.
        return Status.INCOMPLETE
    if headline is Status.INCOMPLETE or any(result.is_incomplete for result in results):
        return Status.INCOMPLETE
    if any(_publication_skipped(result) and not _optional_skip(result, evidence) for result in results):
        return Status.INCOMPLETE  # a required check did not run
    present = bool(tier3) and not all(_publication_skipped(result) for result in tier3)
    if (evidence.required or present) and not evidence.complete:
        return Status.INCOMPLETE
    if evidence.verdict == "neutral" and evidence.execution in {"succeeded", ""}:
        return Status.NEUTRAL
    return Status.PASSED


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("verdict") or _mapping(payload.get("summary")).get("verdict") or "").lower()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}
