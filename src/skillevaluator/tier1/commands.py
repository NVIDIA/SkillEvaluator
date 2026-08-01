# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 command implementations."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from skillevaluator.constants import (
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_WORKFLOWS,
)
from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting import CLIReporter, HTMLReporter, JSONReporter, MarkdownReporter
from skillevaluator.reporting.html import is_tier2_validator_name
from skillevaluator.reporting.naming import DEFAULT_REPORT_BASENAME
from skillevaluator.validators.base import continue_on_failure_scope
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.hygiene import HygieneValidator
from skillevaluator.validators.license import LicenseValidator
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.policy import ValidationPolicy, apply_policy
from skillevaluator.validators.quality_score import QualityScoreValidator
from skillevaluator.validators.rubric_eval import RubricEvalValidator
from skillevaluator.validators.rules_schema import RulesSchemaValidator
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.script_lint import ScriptLintValidator
from skillevaluator.validators.secrets import SecretsValidator
from skillevaluator.validators.security import SecurityValidator
from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator
from skillevaluator.validators.version import VersionValidator
from skillevaluator.validators.workflows_schema import WorkflowsSchemaValidator

console = Console()

# Per-check progress goes to stderr so piped stdout (reports, JSON) stays
# clean; without it, slow targets print nothing for minutes and look hung.
progress_console = Console(stderr=True)

ValidatorRunner = Callable[[Path], ValidationResult]

DEFAULT_CHECKS = (
    "schema",
    "version",
    "security",
    "pii",
    "license",
    "code-integrity",
    "unicode",
    "quality",
    "lint",
)
# Opt-in checks: recognized by ``--checks`` but excluded from the default run.
# Mirrors Skill Evaluator, where the semantic-version bump check and the pip-audit /
# Safety dependency CVE audit (shipped as the standalone ``dependency-audit``
# command) were opt-in rather than part of the default validate pipeline.
OPTIONAL_CHECKS = ("dependency",)
# Every canonical check name ``run_validation`` understands after alias
# resolution (the default run plus the opt-in checks).
RECOGNIZED_CHECKS = frozenset(DEFAULT_CHECKS) | frozenset(OPTIONAL_CHECKS)
REPORTERS = {
    "json": JSONReporter,
    "html": HTMLReporter,
    "markdown": MarkdownReporter,
}


def _as_result(name: str, description: str, validator: ValidatorRunner, target: Path) -> ValidationResult:
    result = validator(target)
    if not result.validator_name:
        result.validator_name = name
    if not result.validator_description:
        result.validator_description = description
    return result


def _enabled_checks(checks: str | None) -> set[str]:
    if not checks:
        return set(DEFAULT_CHECKS)
    aliases = {
        "code": "code-integrity",
        "code-risk": "code-integrity",
        "scripts": "lint",
        "script-lint": "lint",
        "dependencies": "dependency",
        "deps": "dependency",
        "dependency-audit": "dependency",
        "licence": "license",
        "license-check": "license",
    }
    enabled = set()
    for raw in checks.split(","):
        check = raw.strip().lower()
        if check:
            enabled.add(aliases.get(check, check))
    return enabled


def _schema_validator_for(content_type: str | None, policy: ValidationPolicy | None):
    """Return the schema validator matching the (forced or detected) content type.

    Rules, workflows, and plugins use their dedicated schema validators; skill
    and unknown content fall back to the skill :class:`SchemaValidator` (the
    historical default).
    """
    if content_type == CONTENT_TYPE_RULES:
        return RulesSchemaValidator()
    if content_type == CONTENT_TYPE_WORKFLOWS:
        return WorkflowsSchemaValidator()
    if content_type == CONTENT_TYPE_PLUGIN:
        return PluginSchemaValidator()
    return SchemaValidator(policy=policy)


def enabled_check_lineup(checks: str | None) -> list[str]:
    """Return the resolved check names for a run, in canonical pipeline order.

    Unrecognized names are kept (sorted, at the end) so the printed lineup
    matches what ``run_validation`` was actually asked to do.
    """
    enabled = _enabled_checks(checks)
    ordered = [check for check in (*DEFAULT_CHECKS, *OPTIONAL_CHECKS) if check in enabled]
    return ordered + sorted(enabled - set(ordered))


def run_validation(
    target_path: Path,
    *,
    checks: str | None = None,
    use_llm: bool = False,
    llm_verify: bool = False,
    min_score: int = 70,
    previous_version: str | None = None,
    policy: ValidationPolicy | None = None,
    content_type: str | None = None,
    fail_fast: bool = False,
    continue_on_failure: bool = False,
    on_check: Callable[[str], None] | None = None,
) -> list[ValidationResult]:
    """Run selected Tier 1 validators and return structured results.

    *on_check* is invoked with each canonical check name just before it runs;
    when provided it replaces the stderr ``[n/total]`` progress lines (the
    caller owns presentation, e.g. the quiet pipeline view).

    *content_type* (``skill`` | ``rules`` | ``workflows`` | ``plugin`` |
    ``unknown`` | ``None``) selects the schema validator and gates skill-only
    checks (``quality`` and ``lint`` are skipped for rules, workflows, and
    plugins). When *fail_fast* is set, the
    run stops after the first failing check. *continue_on_failure* overrides
    *fail_fast* and also keeps batch folder validation scanning every skill past
    a CRITICAL finding (parity with Skill Evaluator ``--continue-on-failure``).

    When *policy* is provided, the schema validator applies the policy's
    audience-aware author rules; finalized severities for all validators are
    applied centrally in :func:`emit_reports` via the policy.
    """
    enabled = _enabled_checks(checks)
    results: list[ValidationResult] = []
    skill_like = content_type in (None, CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN)

    def _schema_results() -> list[ValidationResult]:
        v = _schema_validator_for(content_type, policy)
        return [_as_result(v.name, v.description, v.validate, target_path)]

    def _security_results() -> list[ValidationResult]:
        v = SecurityValidator(use_llm=use_llm, verify_llm=llm_verify)
        return [_as_result("Security Scan", v.description, v.validate_security_only, target_path)]

    def _pii_results() -> list[ValidationResult]:
        v = SecurityValidator(use_llm=False, verify_llm=llm_verify)
        return [_as_result("PII Scan", "Detect PII and local identifiers", v.validate_pii_only, target_path)]

    def _code_integrity_results() -> list[ValidationResult]:
        return [
            _as_result(v.name, v.description, v.validate, target_path)
            for v in (CodeRiskValidator(), SecretsValidator(), HygieneValidator())
        ]

    def _unicode_results() -> list[ValidationResult]:
        v = UnicodeSmuggleValidator()
        return [_as_result(v.name, v.description, v.validate, target_path)]

    def _quality_results() -> list[ValidationResult]:
        v = QualityScoreValidator(min_score=min_score)
        return [_as_result(v.name, v.description, v.validate, target_path)]

    def _lint_results() -> list[ValidationResult]:
        v = ScriptLintValidator()
        return [_as_result(v.name, v.description, v.validate, target_path)]

    def _version_results() -> list[ValidationResult]:
        v = VersionValidator(previous_version=previous_version)
        return [_as_result(v.name, v.description, v.validate, target_path)]

    def _license_results() -> list[ValidationResult]:
        v = LicenseValidator()
        return [_as_result(v.name, v.description, v.validate, target_path)]

    def _dependency_results() -> list[ValidationResult]:
        v = DependencySecurityValidator()
        return [_as_result(v.name, v.description, v.validate, target_path)]

    # (check name, builder, applies-to-this-content-type). Quality scoring and
    # script linting are skill-oriented and skipped for rules/workflows. Finding
    # severities (incl. the LICENSE.* / CVE findings added below) are normalized
    # centrally by the active policy in emit_reports, so they honor the
    # selected validation profile.
    steps = (
        ("schema", _schema_results, True),
        ("version", _version_results, skill_like),
        ("security", _security_results, True),
        ("pii", _pii_results, True),
        ("license", _license_results, True),
        ("code-integrity", _code_integrity_results, True),
        ("dependency", _dependency_results, True),
        ("unicode", _unicode_results, True),
        ("quality", _quality_results, skill_like),
        ("lint", _lint_results, skill_like),
    )
    active = [
        (check_name, builder) for check_name, builder, applicable in steps if check_name in enabled and applicable
    ]
    with continue_on_failure_scope(continue_on_failure):
        for step_number, (check_name, builder) in enumerate(active, 1):
            if on_check is not None:
                on_check(check_name)
            else:
                progress_console.print(f"[{step_number}/{len(active)}] {check_name} ...", markup=False, highlight=False)
            started = time.monotonic()
            step_results = builder()
            results.extend(step_results)

            error_count = sum(r.summary.errors for r in step_results)
            warning_count = sum(r.summary.warnings for r in step_results)
            if any(r.is_incomplete for r in step_results):
                scanners = list(dict.fromkeys(tool for r in step_results for tool in r.incomplete_scans))
                outcome = f"incomplete: {', '.join(scanners)}"
            else:
                outcome = "ok" if all(r.passed for r in step_results) else f"{error_count} error(s)"
            if warning_count:
                outcome += f", {warning_count} warning(s)"
            if on_check is None:
                progress_console.print(
                    f"[{step_number}/{len(active)}] {check_name} done in {time.monotonic() - started:.1f}s ({outcome})",
                    markup=False,
                    highlight=False,
                )

            if fail_fast and not continue_on_failure and any(not r.passed for r in results):
                return results

    unknown = enabled - RECOGNIZED_CHECKS
    if unknown:
        result = ValidationResult(
            validator_name="Tier 1 option validation",
            validator_description="Validate requested check names",
        )
        result.add_error(f"Unknown Tier 1 check(s): {', '.join(sorted(unknown))}")
        results.insert(0, result)

    return results


def run_quality_check(target_path: Path, *, min_score: int = 70) -> list[ValidationResult]:
    validator = QualityScoreValidator(min_score=min_score)
    return [_as_result(validator.name, validator.description, validator.validate, target_path)]


def run_rubric_eval(target_path: Path, *, min_score: int = 70) -> list[ValidationResult]:
    validator = RubricEvalValidator(min_score=min_score)
    return [_as_result(validator.name, validator.description, validator.validate, target_path)]


def run_security_scan(
    target_path: Path,
    *,
    use_llm: bool = False,
    llm_verify: bool = False,
) -> list[ValidationResult]:
    validator = SecurityValidator(use_llm=use_llm, verify_llm=llm_verify)
    return [_as_result("Security Scan", validator.description, validator.validate_security_only, target_path)]


def run_pii_scan(target_path: Path, *, llm_verify: bool = False) -> list[ValidationResult]:
    validator = SecurityValidator(use_llm=False, verify_llm=llm_verify)
    return [_as_result("PII Scan", "Detect PII and local identifiers", validator.validate_pii_only, target_path)]


def run_lint_scripts(target_path: Path) -> list[ValidationResult]:
    validator = ScriptLintValidator()
    return [_as_result(validator.name, validator.description, validator.validate, target_path)]


def _is_dedup_result(result: ValidationResult) -> bool:
    """Return True when a result came from a Tier 2 deduplication validator."""
    return is_tier2_validator_name(result.validator_name)


def _derive_html_tabs(results: list[ValidationResult]) -> list[dict[str, str]]:
    """Build the HTML navigation tabs from the tiers present in *results*.

    Tier 1 is included only when a non-Tier-2 result is present; this keeps
    standalone similarity and deduplication reports in their actual tier.
    Tier 3 is added when a live agent-evaluation payload is attached.
    """
    tabs: list[dict[str, str]] = []
    if not results or any(not _is_dedup_result(result) for result in results):
        tabs.append({"id": "tier1", "label": "Tier 1: Security and Static Validation"})
    if any(_is_dedup_result(r) for r in results):
        tabs.append({"id": "tier2", "label": "Tier 2: Deduplication"})
    if any((getattr(r, "metadata", None) or {}).get("agent_eval") for r in results):
        tabs.append({"id": "tier3", "label": "Tier 3: Live Agent Evaluation"})
    return tabs


def emit_reports(
    results: list[ValidationResult],
    *,
    report_formats: tuple[str, ...],
    output_dir: Path,
    basename: str = DEFAULT_REPORT_BASENAME,
    policy: ValidationPolicy | None = None,
    target_path: str | None = None,
    content_label: str = "Skill",
    announce_paths: bool = True,
) -> bool:
    """Render reports and return whether every result passed.

    When *policy* is provided, finding severities are remapped per the active
    profile (and pass/fail recomputed) before rendering, and the active profile
    is stamped onto each result's metadata for reporters.

    *target_path* (the content's repo URL or path) and *content_label* are
    forwarded to the HTML reporter so the report shows a Target link and the
    correct content noun. HTML navigation tabs are derived from the tiers
    present in *results*, matching Skill Evaluator's combined report.
    """
    if policy is not None:
        apply_policy(results, policy)
    if "cli" in report_formats:
        CLIReporter(console=console).print_all(results)

    html_tabs = _derive_html_tabs(results)
    for fmt in report_formats:
        if fmt == "cli":
            continue
        reporter_cls = REPORTERS[fmt]
        if fmt == "html":
            reporter = reporter_cls(target_path=target_path, content_label=content_label, tabs=html_tabs)
        else:
            reporter = reporter_cls()
        output_path = output_dir / f"{basename}{reporter.get_file_extension()}"
        reporter.save(results, output_path)
        if announce_paths:
            console.print(f"[dim]{fmt} report:[/dim] [cyan]{output_path}[/cyan]")

    return all(result.passed for result in results)
