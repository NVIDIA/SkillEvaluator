# SPDX-License-Identifier: Apache-2.0
"""Shared presentation of explicit evidence that a check was not applicable."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from skillevaluator.reporting import base as reporting_base
from skillevaluator.reporting.base import get_skip_reason, select_agent_eval_payload
from skillevaluator.reporting.run_status import Status, resolve_run_status, result_status
from skillevaluator.reporting.tier3_scope import validate_report_scope

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.base import PublicationAssessment


def success_is_not_applicable(check: object, message: object, metadata: object = None) -> bool:
    """Use typed producer metadata, with exact legacy evidence for retained reports.

    A message merely mentioning applicability is not proof of a skipped check.
    Keep the compatibility vocabulary in one place for HTML and CI Markdown.
    """
    if isinstance(metadata, dict) and type(metadata.get("not_applicable")) is bool:
        return metadata["not_applicable"]
    evidence = {
        "version_optional": None,
        "lint": "No scripts/ or tools/ directory found",
        "code_file_discovery": "No code files found - code risk analysis is not applicable",
        "dependency_file_discovery": "No dependency files found - vulnerability audit is not applicable",
    }
    if not isinstance(check, str) or check not in evidence:
        return False
    expected = evidence[check]
    return expected is None or message == expected


# These projections contain only immutable display values. They never rewrite
# ValidationResult or the canonical agent payload (including retained reports).


@dataclass(frozen=True)
class TierPresentation:
    status: Status
    requested: bool
    displayed: bool
    passed: int = 0
    failed: int = 0
    incomplete: int = 0
    skipped: int = 0
    not_applicable: int = 0
    assessment_status: str | None = None

    @property
    def display_status(self) -> str:
        if self.assessment_status is not None:
            return {"pass": "passed", "fail": "failed"}.get(self.assessment_status, self.assessment_status)
        if not self.executed and self.not_applicable and not self.skipped:
            return "not_applicable"
        return self.status.value

    @property
    def summary(self) -> str:
        if self.status == Status.NOT_RUN:
            return "Not run" if self.requested else "Not run (not requested)"
        assessment = (
            f"{self.display_status.capitalize()} evidence; "
            if self.assessment_status is not None and self.display_status != self.status.value
            else ""
        )
        return (
            assessment + f"{self.passed} of {self.executed} checks passed; "
            f"{self.failed} failed, {self.incomplete} incomplete, "
            f"{self.skipped} skipped, {self.not_applicable} N/A"
        )

    @property
    def executed(self) -> int:
        return self.passed + self.failed + self.incomplete


@dataclass(frozen=True)
class ConditionCoverage:
    status: str = "unavailable"
    scored: int | None = None
    expected: int | None = None
    observed: int | None = None


@dataclass(frozen=True)
class TrialCoverage:
    with_skill: ConditionCoverage
    baseline: ConditionCoverage
    agents: Mapping[str, Mapping[str, ConditionCoverage]]

    @property
    def details(self) -> tuple[str, ...]:
        lines = []
        for label, arm in (("With skill", self.with_skill), ("Baseline", self.baseline)):
            if arm.status == "skipped":
                lines.append(f"{label}: skipped")
            elif arm.scored is not None and arm.expected is not None:
                lines.append(f"{label}: {arm.scored} of {arm.expected} scored ({arm.status})")
        return tuple(lines)

    @property
    def text(self) -> str:
        arms = (self.with_skill, self.baseline)
        if all(arm.scored is not None or arm.status == "skipped" for arm in arms):
            total = sum(arm.scored or 0 for arm in arms)
            with_text = "with skill skipped" if arms[0].status == "skipped" else f"{arms[0].scored} with skill"
            baseline_text = "baseline skipped" if arms[1].status == "skipped" else f"{arms[1].scored} baseline"
            return f"{total} scored trials: {with_text}, {baseline_text}"
        parts = []
        for label, arm in zip(("with skill", "baseline"), arms, strict=True):
            if arm.status == "skipped":
                parts.append(f"{label} skipped")
            elif arm.scored is not None:
                parts.append(f"{arm.scored} {label} scored trials")
            else:
                parts.append(f"{label} scored trials unavailable")
        for label, arm in zip(("with-skill", "baseline"), arms, strict=True):
            if arm.scored is None and arm.observed is not None and arm.status != "skipped":
                parts.append(f"{arm.observed} retained {label} rows")
        return "; ".join(parts)


@dataclass(frozen=True)
class FinalRenderContext:
    command_status: Status
    command_label: str
    publication_status: Status | None
    publication_label: str
    tiers: Mapping[int, TierPresentation]
    result_statuses: Mapping[int, str]
    score_scope: str
    coverage: TrialCoverage


def result_execution_status(result: ValidationResult, outcome: Status | None = None) -> str:
    """Display explicit N/A evidence without changing gate or JSON status."""
    outcome = result_status(result) if outcome is None else outcome
    if outcome in (Status.FAILED, Status.INCOMPLETE):
        return "fail" if outcome == Status.FAILED else "incomplete"
    if outcome == Status.SKIPPED:
        return "not_applicable" if "not applicable" in get_skip_reason(result).casefold() else "skipped"
    if (
        result.success_details
        and not result.findings
        and all(
            success_is_not_applicable(detail.check_name, detail.message, detail.metadata)
            for detail in result.success_details
        )
    ):
        return "not_applicable"
    return "pass"


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _rows(value: object) -> int | None:
    return (
        sum(isinstance(row, dict) for row in value)
        if isinstance(value, list) and len(value) <= reporting_base.AGENT_EVAL_REPORT_MAX_NODES
        else None
    )


def trial_coverage(payload: object) -> TrialCoverage:
    """Prefer per-condition counters; rows are only retained observations.

    Physical retry rows, preview limits and early-stop policy cannot establish
    the number of scored attempts. Missing/malformed counters stay unknown.
    """
    payload = payload if isinstance(payload, dict) else {}
    raw_agents = payload.get("agents")
    raw_agents = raw_agents if isinstance(raw_agents, dict) else {}
    if len(raw_agents) > reporting_base.AGENT_EVAL_REPORT_MAX_NODES:
        return TrialCoverage(ConditionCoverage(), ConditionCoverage(), MappingProxyType({}))
    agents = {}
    for name, raw in raw_agents.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            continue
        conditions = raw.get("conditions")
        conditions = conditions if isinstance(conditions, dict) else {}
        coverage = {}
        for arm in ("with_skill", "without_skill"):
            evidence = conditions.get(arm)
            evidence = evidence if isinstance(evidence, dict) else {}
            status = evidence.get("execution_status")
            status = status if status in ("succeeded", "failed", "incomplete", "skipped") else "unavailable"
            coverage[arm] = ConditionCoverage(
                status, _count(evidence.get("scored_attempts")), _count(evidence.get("expected_attempts"))
            )
        agents[name] = MappingProxyType(coverage)

    def combine(arm: str) -> ConditionCoverage:
        members = [agent[arm] for agent in agents.values()]
        status = next(
            (
                s
                for s in ("failed", "incomplete", "unavailable", "succeeded", "skipped")
                if any(c.status == s for c in members)
            ),
            "unavailable",
        )

        def total(field: str) -> int | None:
            values = [getattr(c, field) if c.status != "skipped" else 0 for c in members]
            return sum(values) if values and all(value is not None for value in values) else None

        observed = _rows(payload.get("trials")) if arm == "with_skill" else None
        if arm == "without_skill":
            rows = [_rows(raw.get("trials_baseline")) for raw in raw_agents.values() if isinstance(raw, dict)]
            observed = sum(rows) if rows and all(n is not None for n in rows) else None
        return ConditionCoverage(status, total("scored"), total("expected"), observed)

    return TrialCoverage(combine("with_skill"), combine("without_skill"), MappingProxyType(agents))


def build_final_render_context(
    results: Iterable[ValidationResult],
    *,
    requested_tiers: Iterable[int] | None = None,
    displayed_tiers: Iterable[int] | None = None,
    report_scope: str = "publication",
    expected_skill_name: str | None = None,
    agent_eval: dict | None = None,
    publication_assessment: PublicationAssessment | None = None,
) -> FinalRenderContext:
    """Classify once for final human renders; decision scope stays independent."""
    results = list(results)
    validate_report_scope(report_scope, results)
    payload = agent_eval if agent_eval is not None else select_agent_eval_payload(results)
    run = resolve_run_status(results, agent_eval=payload, expected_skill_name=expected_skill_name)
    present = tuple(n for n, tier in run.tiers.items() if tier.results)
    requested = frozenset(present if requested_tiers is None else requested_tiers)
    displayed = frozenset(requested if displayed_tiers is None else displayed_tiers)
    if any(type(n) is not int or n not in (1, 2, 3) for n in requested | displayed):
        raise ValueError("presentation tiers must be 1, 2 or 3")
    outcomes = {id(r): result_execution_status(r, run.of(r)) for r in results}
    assessment = publication_assessment or run.publication_assessment
    tiers = {}
    for n, tier in run.tiers.items():
        counts = {
            status: sum(outcomes[id(r)] == status for r in tier.results)
            for status in ("pass", "fail", "incomplete", "skipped", "not_applicable")
        }
        tiers[n] = TierPresentation(
            tier.status,
            n in requested,
            n in displayed,
            counts["pass"],
            counts["fail"],
            counts["incomplete"],
            counts["skipped"],
            counts["not_applicable"],
            (
                "fail"
                if tier.status == Status.FAILED
                else "incomplete"
                if tier.status == Status.INCOMPLETE
                else "skipped"
                if tier.status == Status.SKIPPED
                else assessment.tier3.status
            )
            if n == 3 and tier.results
            else None,
        )
    publication = (
        None
        if report_scope == "tier3"
        else {"pass": Status.PASSED, "fail": Status.FAILED, "neutral": Status.NEUTRAL, "incomplete": Status.INCOMPLETE}[
            assessment.status
        ]
    )
    return FinalRenderContext(
        run.headline,
        {Status.PASSED: "Passed required checks", Status.FAILED: "Changes required", Status.INCOMPLETE: "Incomplete"}[
            run.headline
        ],
        publication,
        "Not assessed (Tier 3 only)"
        if publication is None
        else {
            Status.PASSED: "Eligible",
            Status.FAILED: "Not eligible",
            Status.INCOMPLETE: "Incomplete",
            Status.NEUTRAL: "Neutral",
        }[publication],
        MappingProxyType(tiers),
        MappingProxyType(outcomes),
        "Best agent score",
        trial_coverage(payload),
    )
