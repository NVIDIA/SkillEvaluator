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
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
)
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting import CLIReporter, HTMLReporter, JSONReporter, MarkdownReporter, SARIFReporter
from skillevaluator.reporting.html import is_tier2_validator_name
from skillevaluator.reporting.naming import DEFAULT_REPORT_BASENAME
from skillevaluator.validators.base import continue_on_failure_scope
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.hygiene import HygieneValidator
from skillevaluator.validators.license import LicenseValidator
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.plugin_tree import plugin_tree_scope
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
# The dependency CVE audit is also available through the standalone
# ``dependency-audit`` command; the version check above runs by default.
OPTIONAL_CHECKS = ("dependency",)
# Every canonical check name ``run_validation`` understands after alias
# resolution (the default run plus the opt-in checks).
RECOGNIZED_CHECKS = frozenset(DEFAULT_CHECKS) | frozenset(OPTIONAL_CHECKS)
# Whole-content checks. For a plugin they scan the entire plugin tree exactly
# once: each bundled skill, then root-owned content with the bundled-skill
# subtrees excluded (see ``validators.plugin_tree``).
PLUGIN_TREE_CHECKS = frozenset({"security", "pii", "license", "code-integrity", "dependency", "unicode"})
REPORTERS = {
    "json": JSONReporter,
    "html": HTMLReporter,
    "markdown": MarkdownReporter,
    "sarif": SARIFReporter,
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


def _schema_validator_for(
    content_type: str | None,
    policy: ValidationPolicy | None,
    repo_root: Path | None = None,
):
    """Return the schema validator matching the (forced or detected) content type.

    Rules, workflows, and plugins use their dedicated schema validators; skill
    and unknown content fall back to the skill :class:`SchemaValidator` (the
    historical default). *repo_root* only affects plugin dependency resolution.
    """
    if content_type == CONTENT_TYPE_RULES:
        return RulesSchemaValidator()
    if content_type == CONTENT_TYPE_WORKFLOWS:
        return WorkflowsSchemaValidator()
    if content_type == CONTENT_TYPE_PLUGIN:
        return PluginSchemaValidator(policy=policy, repo_root=repo_root)
    return SchemaValidator(policy=policy)


def enabled_check_lineup(checks: str | None) -> list[str]:
    """Return the resolved check names for a run, in canonical pipeline order.

    Unrecognized names are kept (sorted, at the end) so the printed lineup
    matches what ``run_validation`` was actually asked to do.
    """
    enabled = _enabled_checks(checks)
    ordered = [check for check in (*DEFAULT_CHECKS, *OPTIONAL_CHECKS) if check in enabled]
    return ordered + sorted(enabled - set(ordered))


def _unsafe_plugin_tree_results(
    target_path: Path,
    exc: ValueError,
    *,
    include_schema: bool,
    policy: ValidationPolicy | None,
    repo_root: Path | None,
) -> list[ValidationResult]:
    """Return the fail-closed results for a plugin tree that cannot be scanned safely."""
    results: list[ValidationResult] = []
    if include_schema:
        validator = PluginSchemaValidator(policy=policy, repo_root=repo_root)
        results.append(_as_result(validator.name, validator.description, validator.validate, target_path))
    code = getattr(exc, "code", None)
    relative_path = getattr(exc, "relative_path", None)
    if code == "path_count_limit":
        reason = f"the plugin tree exceeds the {PLUGIN_TREE_MAX_DISCOVERED_PATHS}-entry limit for whole-plugin scans"
    else:
        reason = str(exc)
    tree_result = ValidationResult(
        validator_name="Plugin Tree Security",
        validator_description="Verify the whole plugin tree is regular, contained, and link-free before scanning it",
    )
    tree_result.add_finding(
        Finding(
            category="PLUGIN_SCHEMA",
            severity=Severity.HIGH,
            check_name="unsafe_plugin_filesystem",
            message=f"Refusing to scan the plugin tree: {reason}",
            file_path=relative_path if relative_path and relative_path != "." else "<plugin-root>",
            suggestion=(
                "Replace linked, hard-linked, reparse-point, or special plugin paths with regular files and "
                "directories contained by the plugin root."
            ),
        )
    )
    tree_result.metadata["security_failure"] = True
    results.append(tree_result)
    return results


def _with_component_attribution(results: list[ValidationResult], content_type: str | None) -> list[ValidationResult]:
    """Recount plugin component ``findings`` across every Tier 1 validator's results."""
    if content_type == CONTENT_TYPE_PLUGIN:
        from skillevaluator.plugin_components import refresh_component_finding_counts

        refresh_component_finding_counts(results)
    return results


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
    repo_root: Path | None = None,
) -> list[ValidationResult]:
    """Run selected Tier 1 validators and return structured results.

    *on_check* is invoked with each canonical check name just before it runs;
    when provided it replaces the stderr ``[n/total]`` progress lines (the
    caller owns presentation, e.g. the quiet pipeline view).

    *content_type* (``skill`` | ``rules`` | ``workflows`` | ``plugin`` |
    ``unknown`` | ``None``) selects the schema validator and gates skill-only
    checks. ``version``, ``quality``, and ``lint`` run for skills and for each
    skill bundled under a plugin's ``skills/`` directory (findings are
    attributed to that skill); they are skipped for rules, workflows, and
    plugins without bundled skills. For plugins, the whole-content checks
    (:data:`PLUGIN_TREE_CHECKS`) scan the entire plugin tree exactly once:
    each bundled skill, then root-owned content with the bundled-skill
    subtrees excluded. The tree is first verified without following links,
    and any linked, hard-linked, or special entry fails the run closed.
    When *fail_fast* is set, the run stops after the first failing check.
    *continue_on_failure* overrides *fail_fast* and also keeps batch folder
    validation scanning every skill past a CRITICAL finding (parity with
    SkillEvaluator ``--continue-on-failure``).

    When *policy* is provided, the schema validator applies the policy's
    audience-aware author rules; finalized severities for all validators are
    applied centrally in :func:`emit_reports` via the policy.

    *repo_root* (``--repo-root``) optionally names the repository root used to
    resolve a bundle-reference plugin's same-repository skill/rule refs.
    """
    enabled = _enabled_checks(checks)
    results: list[ValidationResult] = []
    bundled_skill_dirs: list[Path] = []
    if content_type == CONTENT_TYPE_PLUGIN:
        from skillevaluator.cli_core import resolve_plugin_path
        from skillevaluator.utils.helpers import find_bundled_plugin_skills, verify_plugin_tree

        target_path = resolve_plugin_path(target_path)
        try:
            # Secure, no-follow discovery: any linked or special entry under
            # ``skills/`` fails here before a skill-scoped validator reads it.
            bundled_skill_dirs = find_bundled_plugin_skills(target_path)
        except ValueError as exc:
            if "schema" in enabled:
                validator = PluginSchemaValidator(policy=policy, repo_root=repo_root)
                return [_as_result(validator.name, validator.description, validator.validate, target_path)]
            security_result = ValidationResult(
                validator_name="Plugin Bundle Security",
                validator_description="Securely discover skills bundled inside the plugin",
            )
            security_result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="bundled_skill_path_unsafe",
                    message=f"Could not securely discover bundled skills: {exc}",
                    file_path="<plugin-skills>",
                    suggestion="Replace linked or special bundled-skill paths with regular contained directories.",
                )
            )
            security_result.metadata["security_failure"] = True
            return [security_result]
        if enabled & PLUGIN_TREE_CHECKS:
            # Whole-plugin scanners also read root-owned content (scripts/,
            # hooks/, .mcp.json, ...), so the entire tree must pass the same
            # no-follow verification before any of them runs.
            try:
                verify_plugin_tree(target_path)
            except ValueError as exc:
                return _with_component_attribution(
                    _unsafe_plugin_tree_results(
                        target_path,
                        exc,
                        include_schema="schema" in enabled,
                        policy=policy,
                        repo_root=repo_root,
                    ),
                    content_type,
                )
    # Skill-scoped checks (version/quality/lint) also run for plugins that
    # bundle skills. They target ``<plugin>/skills`` only, so the folder walker
    # validates each bundled skill once and prefixes its findings with the
    # skill's name; root-owned plugin content is not a skill and is never
    # scored or linted as one, so the two scopes cannot double-report.
    skill_like = content_type in (None, CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN) or bool(bundled_skill_dirs)
    skill_target = target_path / "skills" if content_type == CONTENT_TYPE_PLUGIN else target_path

    def _schema_results() -> list[ValidationResult]:
        v = _schema_validator_for(content_type, policy, repo_root)
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
        return [_as_result(v.name, v.description, v.validate, skill_target)]

    def _lint_results() -> list[ValidationResult]:
        v = ScriptLintValidator()
        return [_as_result(v.name, v.description, v.validate, skill_target)]

    def _version_results() -> list[ValidationResult]:
        v = VersionValidator(previous_version=previous_version)
        return [_as_result(v.name, v.description, v.validate, skill_target)]

    def _license_results() -> list[ValidationResult]:
        v = LicenseValidator()
        return [_as_result(v.name, v.description, v.validate, target_path)]

    def _dependency_results() -> list[ValidationResult]:
        v = DependencySecurityValidator()
        return [_as_result(v.name, v.description, v.validate, target_path)]

    # (check name, builder, applies-to-this-content-type). Version, quality
    # scoring, and script linting are skill-oriented: they run for skills and
    # for plugins' bundled skills, and are skipped for rules/workflows. Finding
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
            if content_type == CONTENT_TYPE_PLUGIN and check_name in PLUGIN_TREE_CHECKS:
                with plugin_tree_scope(target_path, bundled_skill_dirs):
                    step_results = builder()
            else:
                step_results = builder()
            results.extend(step_results)

            if any(result.metadata.get("security_failure") for result in step_results):
                return _with_component_attribution(results, content_type)

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
                return _with_component_attribution(results, content_type)

    unknown = enabled - RECOGNIZED_CHECKS
    if unknown:
        result = ValidationResult(
            validator_name="Tier 1 option validation",
            validator_description="Validate requested check names",
        )
        result.add_error(f"Unknown Tier 1 check(s): {', '.join(sorted(unknown))}")
        results.insert(0, result)

    return _with_component_attribution(results, content_type)


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
    sarif_scan_root: Path | None = None,
    sarif_repository_root: Path | None = None,
) -> bool:
    """Render reports and return whether every result passed.

    When *policy* is provided, finding severities are remapped per the active
    profile (and pass/fail recomputed) before rendering, and the active profile
    is stamped onto each result's metadata for reporters.

    *target_path* (the content's repo URL or path) and *content_label* are
    forwarded to the HTML reporter so the report shows a Target link and the
    correct content noun. HTML navigation tabs are derived from the tiers
    present in *results*, matching SkillEvaluator's combined report.
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
        elif fmt == "sarif":
            reporter = reporter_cls(
                workspace_root=sarif_repository_root,
                scan_root=sarif_scan_root,
            )
        else:
            reporter = reporter_cls()
        output_path = output_dir / f"{basename}{reporter.get_file_extension()}"
        reporter.save(results, output_path)
        if announce_paths:
            console.print(f"[dim]{fmt} report:[/dim] [cyan]{output_path}[/cyan]")

    return all(result.passed for result in results)
