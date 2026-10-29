# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared command and report status, retaining the stricter publication assessment.

A confirmed failure outranks incomplete evidence; incomplete evidence outranks a
pass. Advisory results retain their own status without changing the recorded exit
gate. Publication continues to use the existing provenance and coverage contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting.base import (
    assess_publication,
    is_cleanly_skipped,
    is_tier2_validator_name,
    is_tier3_result,
    passes_required_gate,
)


class Status(StrEnum):
    PASSED = "passed"
    NEUTRAL = "neutral"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"

    @property
    def verdict(self) -> str:
        return {"passed": "PASS", "failed": "FAIL", "not_run": "NOT RUN"}.get(self.value, self.value.upper())


@dataclass(frozen=True)
class TierStatus:
    tier: int
    status: Status
    blocking: bool | None
    results: tuple[ValidationResult, ...]


@dataclass(frozen=True)
class RunStatus:
    headline: Status
    tiers: Mapping[int, TierStatus]
    missing: tuple[str, ...]
    advisory: tuple[str, ...]
    _outcomes: Mapping[int, Status] = field(repr=False)
    _publication: Callable[[], Status] = field(repr=False, compare=False)

    @cached_property
    def publication(self) -> Status:
        """Assess publication once, only on surfaces that need its stricter verdict."""
        return self._publication()

    def of(self, result: ValidationResult) -> Status:
        return self._outcomes.get(id(result), result_status(result))


def _blocking_finding(result: ValidationResult) -> bool:
    for finding in result.findings:
        if finding.severity.is_error() or finding.to_legacy_string() in result.errors:
            return True
        # Folder merges prefix legacy errors and place those same labels inside
        # finding paths. Respect fail-on-medium/low in either representation.
        location, prefix = finding.location, ""
        while location.startswith("[") and "] " in location:
            label, _, location = location.partition("] ")
            prefix += label + "] "
            if f"{prefix}{finding.tag} {finding.message} in {location}" in result.errors:
                return True
    return False


def result_status(result: ValidationResult) -> Status:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if _blocking_finding(result) or metadata.get("confirmed_failure") is True:
        return Status.FAILED
    if is_tier3_result(result):
        payload = metadata.get("agent_eval")
        payload = payload if isinstance(payload, dict) else {}
        summary = payload.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        verdicts = {str(payload.get("verdict") or "").lower(), str(summary.get("verdict") or "").lower()}
        if metadata.get("tier3_gate_failures") or "fail" in verdicts:
            return Status.FAILED
        if "inconclusive" in verdicts:
            return Status.INCOMPLETE
    if result.is_incomplete:
        return Status.FAILED if _blocking_finding(result) else Status.INCOMPLETE
    if is_cleanly_skipped(result):
        return Status.SKIPPED if passes_required_gate(result) else Status.FAILED
    return Status.PASSED if result.passed else Status.FAILED


def _tier(result: ValidationResult) -> int:
    gate = result.metadata.get("gating")
    if isinstance(gate, dict) and type(gate.get("tier")) is int and gate["tier"] in (1, 2, 3):
        return gate["tier"]
    return 3 if is_tier3_result(result) else 2 if is_tier2_validator_name(result.validator_name) else 1


def _combined(outcomes: Iterable[Status]) -> Status:
    outcomes = set(outcomes)
    return next(
        (
            status
            for status in (Status.FAILED, Status.INCOMPLETE, Status.NEUTRAL, Status.PASSED, Status.SKIPPED)
            if status in outcomes
        ),
        Status.NOT_RUN,
    )


def resolve_run_status(results: Iterable[ValidationResult], *, agent_eval=None, expected_skill_name=None) -> RunStatus:
    results = tuple(results)
    outcomes = {id(result): result_status(result) for result in results}
    failing = [outcomes[id(result)] for result in results if not passes_required_gate(result)]
    headline = (
        Status.PASSED
        if not failing
        else Status.INCOMPLETE
        if all(s == Status.INCOMPLETE for s in failing)
        else Status.FAILED
    )
    tiers = {}
    for number in (1, 2, 3):
        members = tuple(result for result in results if _tier(result) == number)
        blocking = [
            r.metadata["gating"].get("blocking", True) for r in members if isinstance(r.metadata.get("gating"), dict)
        ]
        tiers[number] = TierStatus(
            number, _combined(outcomes[id(r)] for r in members), any(blocking) if blocking else None, members
        )

    def publication_status() -> Status:
        assessment = assess_publication(list(results), agent_eval, expected_skill_name=expected_skill_name)
        return {
            "pass": Status.PASSED,
            "neutral": Status.NEUTRAL,
            "fail": Status.FAILED,
            "incomplete": Status.INCOMPLETE,
        }[assessment.status]

    return RunStatus(
        headline,
        MappingProxyType(tiers),
        tuple(dict.fromkeys(tool for result in results for tool in result.incomplete_scans)),
        tuple(
            f"{r.validator_name}: {outcomes[id(r)].value}"
            for r in results
            if isinstance(r.metadata.get("gating"), dict)
            and not r.metadata["gating"].get("blocking", True)
            and outcomes[id(r)] != Status.PASSED
        ),
        MappingProxyType(outcomes),
        publication_status,
    )
