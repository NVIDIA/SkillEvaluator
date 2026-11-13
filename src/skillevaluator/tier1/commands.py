# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 command implementations."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from rich.console import Console

from skillevaluator.constants import (
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_WORKFLOWS,
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_MANIFEST_FILES,
    QUALITY_DEFAULT_MIN_SCORE,
    SCAN_EXCLUDED_DIRS,
    SCAN_EXCLUDED_FILES,
    SKILL_MANIFEST_FILE,
    WORKFLOWS_MANIFEST_FILE,
)
from skillevaluator.models.result import ValidationResult
from skillevaluator.publication_evidence import (
    TIER1_UNSCANNED_PATHS_KEY,
    stamp_publication_evidence,
    stamp_tier1_required_checks,
)
from skillevaluator.publication_identity import (
    finalize_publication_target,
    publication_paths_skipped_by_scan,
    publication_target_from_path,
)
from skillevaluator.reporting import CLIReporter, HTMLReporter, JSONReporter, MarkdownReporter, SARIFReporter
from skillevaluator.reporting.naming import DEFAULT_REPORT_BASENAME
from skillevaluator.reporting.presentation import FinalRenderContext, build_final_render_context
from skillevaluator.reporting.run_status import result_tier
from skillevaluator.utils.rich_markup import escape_markup
from skillevaluator.validators.base import continue_on_failure_scope
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.harness import HarnessValidator
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

# emoji=False: ":name:" codes in untrusted text (such as "root:x:0:0:") must print literally.
console = Console(emoji=False)

# Per-check progress goes to stderr so piped stdout (reports, JSON) stays
# clean; without it, slow targets print nothing for minutes and look hung.
progress_console = Console(stderr=True)

ValidatorRunner = Callable[[Path], ValidationResult]

_AGENT_ARTIFACT_CONTENT_TYPES = frozenset(
    {CONTENT_TYPE_SKILL, CONTENT_TYPE_RULES, CONTENT_TYPE_WORKFLOWS, CONTENT_TYPE_PLUGIN}
)

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
    "harness",
)
# Advisory checks never gate and are not part of the publication lineup.
ADVISORY_CHECKS = ("harness",)
# Opt-in checks: recognized by ``--checks`` but excluded from the default run.
# The dependency CVE audit is also available through the standalone
# ``dependency-audit`` command; the version check above runs by default.
OPTIONAL_CHECKS = ("dependency",)
# Every canonical check name ``run_validation`` understands after alias
# resolution (the default run plus the opt-in checks).
RECOGNIZED_CHECKS = frozenset(DEFAULT_CHECKS) | frozenset(OPTIONAL_CHECKS)
# ``--skip`` accepts every check plus the Gitleaks-backed secrets scan inside code-integrity.
SKIPPABLE_CHECKS = RECOGNIZED_CHECKS | {"secrets"}
SKIP_REASON = "skipped with --skip"
_CHECK_ALIASES = {
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
# Validator names recorded for a skipped check, matching the names its run would produce.
_SKIPPED_VALIDATORS = {
    "schema": ("Schema & Repository Governance",),
    "version": ("Semantic Version Validation",),
    "security": ("Security Scan",),
    "pii": ("PII Scan",),
    "license": ("License Compliance",),
    "code-integrity": ("Code Risk Analysis", "Secrets Detection", "Code Integrity & Hygiene"),
    "dependency": ("Dependency Vulnerability Audit",),
    "unicode": ("Unicode Smuggling Detection",),
    "quality": ("Quality Score (4-Dimension Analysis)",),
    "lint": ("Script Lint (Advisory)",),
    "harness": ("Harness Compatibility",),
}
# Checks that read a skill manifest (SKILL.md frontmatter, its version, its
# quality). A target that ``--type auto`` detects as ``unknown`` and that holds
# no artifact manifest entry has no manifest, so these are recorded as not
# applicable rather than run.
_MANIFEST_CHECKS = ("schema", "version", "quality")
# Root manifests that auto-detection looks for, compared in any letter case.
# Detection matches them by exact name with ``exists()``, so a mis-cased name
# or a dangling symlink leaves the target ``unknown``.
_ROOT_MANIFEST_NAMES = frozenset(
    name.casefold() for name in (SKILL_MANIFEST_FILE, WORKFLOWS_MANIFEST_FILE, *PLUGIN_MANIFEST_FILES)
)
UNKNOWN_CONTENT_SKIP_REASON = (
    "not applicable: no supported skill, rules, workflows or plugin artifact detected; use --type to force"
)
# Machine-readable form of the applicability decision. Every result of such a
# run carries it, and the JSON report emits it at the root as
# ``tier1_applicability``.
UNKNOWN_CONTENT_APPLICABILITY = {
    "applicability": "not_applicable",
    "reason_code": "unsupported_content_type",
    "detected_content_type": CONTENT_TYPE_UNKNOWN,
    "supported_content_types": [CONTENT_TYPE_SKILL, CONTENT_TYPE_RULES, CONTENT_TYPE_WORKFLOWS, CONTENT_TYPE_PLUGIN],
}
REPORTERS = {
    "json": JSONReporter,
    "html": HTMLReporter,
    "markdown": MarkdownReporter,
    "sarif": SARIFReporter,
}


def _stamp_tier1_scan_scope(results: list[ValidationResult], target: Path) -> None:
    """Record certified source paths that the Tier 1 scans skipped by name."""
    skipped = publication_paths_skipped_by_scan(target, SCAN_EXCLUDED_DIRS, SCAN_EXCLUDED_FILES)
    for result in results:
        result.metadata[TIER1_UNSCANNED_PATHS_KEY] = list(skipped)


def _entry_names(folder: Path) -> list[str] | None:
    """Return the entry names of *folder*, or ``None`` when it cannot be listed."""
    try:
        with os.scandir(folder) as entries:
            return [entry.name for entry in entries]
    except OSError:
        return None


def _has_manifest_entry(target: Path) -> bool:
    """Return whether *target* is, or directly holds, an artifact manifest entry in any letter case.

    The manifests are ``SKILL.md``, ``workflow-rules.mdc``,
    ``agent_plugin.yaml``/``.yml`` and ``.claude-plugin/plugin.json``. Entry
    names are compared instead of calling ``exists()``, so a dangling symlink
    and a mis-cased name such as ``Skill.md`` count on every filesystem: such
    a folder is a broken artifact, not a target without one. A directory that
    cannot be listed also counts, and so does a ``.claude-plugin`` entry that
    is not a listable directory, so the manifest checks run and report the
    problem instead of being recorded as not applicable. A ``.claude-plugin``
    folder without a ``plugin.json``, such as a marketplace's, does not count.
    """
    contained_dir = PLUGIN_CONTAINED_MANIFEST_DIR.casefold()
    contained_file = PLUGIN_CONTAINED_MANIFEST_FILE.casefold()
    name = target.name.casefold()
    if os.path.lexists(target) and (
        name in _ROOT_MANIFEST_NAMES or (name == contained_file and target.parent.name.casefold() == contained_dir)
    ):
        return True
    if not target.is_dir():
        return False
    names = _entry_names(target)
    if names is None:
        return True
    for entry in names:
        folded = entry.casefold()
        if folded in _ROOT_MANIFEST_NAMES:
            return True
        if folded == contained_dir:
            contained = _entry_names(target / entry)
            if contained is None or any(item.casefold() == contained_file for item in contained):
                return True
    return False


def _stamp_tier1_applicability(results: list[ValidationResult]) -> None:
    """Record on every result that the manifest checks do not apply to the target."""
    for result in results:
        result.metadata["tier1_applicability"] = deepcopy(UNKNOWN_CONTENT_APPLICABILITY)


def _as_result(
    name: str,
    description: str,
    validator: ValidatorRunner,
    target: Path,
    *,
    publication_check_id: str,
    bind_publication_target: bool = True,
    skill_like: bool = True,
) -> ValidationResult:
    initial_target = publication_target_from_path(target) if bind_publication_target else None
    result = validator(target)
    if not result.validator_name:
        result.validator_name = name
    if not result.validator_description:
        result.validator_description = description
    stamp_publication_evidence([result], tier=1, check_id=publication_check_id)
    # Record the whole applicable lineup, so a run narrowed with --checks (or a
    # standalone single-check command) cannot certify Tier 1 by itself.
    stamp_tier1_required_checks([result], skill_like=skill_like)
    if bind_publication_target:
        _stamp_tier1_scan_scope([result], target)
        finalize_publication_target([result], target, initial_target)
    return result


def _parse_checks(value: str | None) -> set[str]:
    return {_CHECK_ALIASES.get(check, check) for raw in (value or "").split(",") if (check := raw.strip().lower())}


def _enabled_checks(checks: str | None) -> set[str]:
    return _parse_checks(checks) if checks else set(DEFAULT_CHECKS)


def skipped_checks(skip: str | None) -> set[str]:
    """Resolve ``--skip`` names (aliases included) to canonical check names."""
    return _parse_checks(skip)


def runs_schema_check(checks: str | None, skip: str | None) -> bool:
    """True when a run with these ``--checks``/``--skip`` values runs the schema check."""
    return "schema" in _enabled_checks(checks) and "schema" not in skipped_checks(skip)


def runs_secrets_scan(checks: str | None, skip: str | None) -> bool:
    """True when a run with these ``--checks``/``--skip`` values executes the Gitleaks scan."""
    return "code-integrity" in _enabled_checks(checks) and not {"code-integrity", "secrets"} & skipped_checks(skip)


def runs_security_scan(checks: str | None, skip: str | None) -> bool:
    """True when a run with these ``--checks``/``--skip`` values executes the SkillSpector scan."""
    return "security" in _enabled_checks(checks) and "security" not in skipped_checks(skip)


def _schema_validator_for(content_type: str | None, policy: ValidationPolicy | None):
    """Return the schema validator matching the (forced or detected) content type.

    Rules, workflows, and plugins use their dedicated schema validators; skill
    content, ``None`` and an ``unknown`` folder that holds an artifact
    manifest entry fall back to the skill :class:`SchemaValidator` (the
    historical default). ``run_validation`` records the schema check as not
    applicable for any other ``unknown`` target instead of calling this.
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
    min_score: int = QUALITY_DEFAULT_MIN_SCORE,
    quality_config: Path | None = None,
    pii_waivers: Path | None = None,
    previous_version: str | None = None,
    policy: ValidationPolicy | None = None,
    content_type: str | None = None,
    target_root: Path | None = None,
    fail_fast: bool = False,
    continue_on_failure: bool = False,
    on_check: Callable[[str], None] | None = None,
    harness: str | None = None,
    harness_profiles: Path | None = None,
    skillspector_baseline: Path | None = None,
    skip: str | None = None,
) -> list[ValidationResult]:
    """Run selected Tier 1 validators and return structured results.

    Checks named in *skip* do not run; each is recorded as a user-skipped
    result, which reports show as skipped and which never certifies Tier 1.

    *on_check* is invoked with each canonical check name just before it runs;
    when provided it replaces the stderr ``[n/total]`` progress lines (the
    caller owns presentation, e.g. the quiet pipeline view).

    *content_type* (``skill`` | ``rules`` | ``workflows`` | ``plugin`` |
    ``unknown`` | ``None``) selects the schema validator and gates skill-only
    checks (``quality`` and ``lint`` are skipped for rules, workflows, and
    plugins). For ``unknown`` (``--type auto`` found no supported artifact) the
    manifest checks ``schema``, ``version`` and ``quality`` are recorded as not
    applicable instead of run, and every result carries the
    ``tier1_applicability`` record, while the content scans still run. A target
    that holds an artifact manifest entry (``SKILL.md``, ``workflow-rules.mdc``,
    ``agent_plugin.yaml``/``.yml`` or ``.claude-plugin/plugin.json``) in any
    letter case, including a dangling symlink, is a broken artifact, so its
    manifest checks still run and fail.
    ``None`` keeps the historical skill behavior. When *fail_fast* is set, the
    run stops after the first failing check. *continue_on_failure* overrides
    *fail_fast* and also keeps batch folder validation scanning every skill past
    a CRITICAL finding (parity with SkillEvaluator ``--continue-on-failure``).

    When *policy* is provided, the schema validator applies the policy's
    audience-aware author rules; finalized severities for all validators are
    applied centrally in :func:`emit_reports` via the policy.

    *pii_waivers* is an optional PII waiver file applied by the ``pii`` check.
    *skillspector_baseline* is an audited SkillSpector suppression baseline
    applied to the ``security`` check; it is never discovered from the target.
    *target_root* is the folder that *target_path* belongs to when a caller
    validates that folder one skill at a time, as ``validate`` does for a
    folder of skills. The ``security`` check then accepts no SkillSpector
    missing reference in the skill, because its instructions may name a file
    elsewhere in the folder that its scan did not inspect.
    """
    enabled = _enabled_checks(checks)
    skipped = skipped_checks(skip)
    results: list[ValidationResult] = []
    initial_target = publication_target_from_path(target_path)
    skill_like = content_type in (None, CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN)
    manifest_not_applicable = content_type == CONTENT_TYPE_UNKNOWN and not _has_manifest_entry(target_path)

    def _run_result(
        check_id: str,
        name: str,
        description: str,
        validator: ValidatorRunner,
    ) -> ValidationResult:
        return _as_result(
            name,
            description,
            validator,
            target_path,
            publication_check_id=check_id,
            bind_publication_target=False,
            skill_like=skill_like,
        )

    def _skipped_result(check_id: str, name: str) -> ValidationResult:
        result = ValidationResult(
            validator_name=name,
            validator_description=SKIP_REASON,
            metadata={"skipped": True, "skipped_by_user": True, "skip_reason": SKIP_REASON},
        )
        stamp_publication_evidence([result], tier=1, check_id=check_id)
        stamp_tier1_required_checks([result], skill_like=skill_like)
        return result

    def _not_applicable_result(check_id: str) -> ValidationResult:
        # Recorded, never executed: a skipped result is not publication
        # evidence, so a run with one of these can never certify Tier 1.
        result = ValidationResult(
            validator_name=_SKIPPED_VALIDATORS[check_id][0],
            validator_description=UNKNOWN_CONTENT_SKIP_REASON,
            metadata={"skipped": True, "skip_reason": UNKNOWN_CONTENT_SKIP_REASON},
        )
        stamp_publication_evidence([result], tier=1, check_id=check_id)
        stamp_tier1_required_checks([result], skill_like=skill_like)
        return result

    def _schema_results() -> list[ValidationResult]:
        v = _schema_validator_for(content_type, policy)
        return [_run_result("schema", v.name, v.description, v.validate)]

    def _security_results() -> list[ValidationResult]:
        # Agent-security findings are scoped to agent control paths only for a
        # detected agent artifact; an unknown layout keeps the unrestricted
        # scan that ``security-scan`` runs.
        v = SecurityValidator(
            use_llm=use_llm,
            verify_llm=llm_verify,
            content_type=content_type if content_type in _AGENT_ARTIFACT_CONTENT_TYPES else None,
            skillspector_reference_root=target_path if content_type == CONTENT_TYPE_PLUGIN else None,
            skillspector_baseline=skillspector_baseline,
            target_root=target_root,
        )
        return [_run_result("security", "Security Scan", v.description, v.validate_security_only)]

    def _pii_results() -> list[ValidationResult]:
        v = SecurityValidator(use_llm=False, verify_llm=llm_verify, pii_waiver_path=pii_waivers)
        return [_run_result("pii", "PII Scan", "Detect PII and local identifiers", v.validate_pii_only)]

    def _code_integrity_results() -> list[ValidationResult]:
        return [
            _skipped_result("code-integrity", v.name)
            if "secrets" in skipped and isinstance(v, SecretsValidator)
            else _run_result("code-integrity", v.name, v.description, v.validate)
            for v in (CodeRiskValidator(), SecretsValidator(), HygieneValidator())
        ]

    def _unicode_results() -> list[ValidationResult]:
        v = UnicodeSmuggleValidator()
        return [_run_result("unicode", v.name, v.description, v.validate)]

    def _quality_results() -> list[ValidationResult]:
        v = QualityScoreValidator(min_score=min_score, quality_config=quality_config)
        return [_run_result("quality", v.name, v.description, v.validate)]

    def _lint_results() -> list[ValidationResult]:
        v = ScriptLintValidator()
        return [_run_result("lint", v.name, v.description, v.validate)]

    def _version_results() -> list[ValidationResult]:
        v = VersionValidator(previous_version=previous_version)
        return [_run_result("version", v.name, v.description, v.validate)]

    def _license_results() -> list[ValidationResult]:
        v = LicenseValidator()
        return [_run_result("license", v.name, v.description, v.validate)]

    def _harness_results() -> list[ValidationResult]:
        v = HarnessValidator(harness, harness_profiles)
        return [_run_result("harness", v.name, v.description, v.validate)]

    def _dependency_results() -> list[ValidationResult]:
        v = DependencySecurityValidator()
        return [_run_result("dependency", v.name, v.description, v.validate)]

    # (check name, builder, applies-to-this-content-type). Quality scoring and
    # script linting are skill-oriented and skipped for rules/workflows. Finding
    # severities (incl. the LICENSE.* and DEPENDENCY.* advisory findings added
    # below) are normalized centrally by the active policy in emit_reports, so
    # they honor the selected validation profile.
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
        ("harness", _harness_results, skill_like),
    )
    active = [
        (check_name, builder) for check_name, builder, applicable in steps if check_name in enabled and applicable
    ]
    for check_name, _builder in [step for step in active if step[0] in skipped]:
        active.remove((check_name, _builder))
        results.extend(_skipped_result(check_name, name) for name in _SKIPPED_VALIDATORS[check_name])
    if manifest_not_applicable:
        # No skill, rules, workflows or plugin layout was detected, so there is
        # no manifest to validate: each manifest check would only report "No
        # skills found". The content scans below keep their unrestricted scope.
        for check_name, _builder in [step for step in active if step[0] in _MANIFEST_CHECKS]:
            active.remove((check_name, _builder))
            results.append(_not_applicable_result(check_name))
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
                if manifest_not_applicable:
                    _stamp_tier1_applicability(results)
                _stamp_tier1_scan_scope(results, target_path)
                finalize_publication_target(results, target_path, initial_target)
                return results

    unknown = (enabled - RECOGNIZED_CHECKS) | (skipped - SKIPPABLE_CHECKS)
    if unknown:
        result = ValidationResult(
            validator_name="Tier 1 option validation",
            validator_description="Validate requested check names",
        )
        result.add_error(f"Unknown Tier 1 check(s): {', '.join(sorted(unknown))}")
        results.insert(0, result)

    if manifest_not_applicable:
        # Recorded whatever --checks/--skip selected, so CI can read the
        # applicability of the target from any run, including one narrowed to
        # the content scans.
        _stamp_tier1_applicability(results)
    _stamp_tier1_scan_scope(results, target_path)
    finalize_publication_target(results, target_path, initial_target)
    return results


def run_quality_check(
    target_path: Path, *, min_score: int = QUALITY_DEFAULT_MIN_SCORE, quality_config: Path | None = None
) -> list[ValidationResult]:
    validator = QualityScoreValidator(min_score=min_score, quality_config=quality_config)
    return [
        _as_result(
            validator.name,
            validator.description,
            validator.validate,
            target_path,
            publication_check_id="quality",
        )
    ]


def run_rubric_eval(target_path: Path, *, min_score: int = QUALITY_DEFAULT_MIN_SCORE) -> list[ValidationResult]:
    validator = RubricEvalValidator(min_score=min_score)
    return [
        _as_result(
            validator.name,
            validator.description,
            validator.validate,
            target_path,
            publication_check_id="rubric",
        )
    ]


def run_security_scan(
    target_path: Path,
    *,
    use_llm: bool = False,
    llm_verify: bool = False,
    skillspector_baseline: Path | None = None,
) -> list[ValidationResult]:
    validator = SecurityValidator(use_llm=use_llm, verify_llm=llm_verify, skillspector_baseline=skillspector_baseline)
    return [
        _as_result(
            "Security Scan",
            validator.description,
            validator.validate_security_only,
            target_path,
            publication_check_id="security",
        )
    ]


def run_pii_scan(
    target_path: Path,
    *,
    llm_verify: bool = False,
    pii_waivers: Path | None = None,
) -> list[ValidationResult]:
    validator = SecurityValidator(use_llm=False, verify_llm=llm_verify, pii_waiver_path=pii_waivers)
    return [
        _as_result(
            "PII Scan",
            "Detect PII and local identifiers",
            validator.validate_pii_only,
            target_path,
            publication_check_id="pii",
        )
    ]


def run_lint_scripts(target_path: Path) -> list[ValidationResult]:
    validator = ScriptLintValidator()
    return [
        _as_result(
            validator.name,
            validator.description,
            validator.validate,
            target_path,
            publication_check_id="lint",
        )
    ]


def _derive_html_tabs(results: list[ValidationResult]) -> list[dict[str, str]]:
    """Build populated tabs with the same tier classifier as the HTML evidence."""
    tiers = {result_tier(result) for result in results} if results else {1}
    labels = {
        1: "Tier 1: Security and Static Validation",
        2: "Tier 2: Deduplication",
        3: "Tier 3: Live Agent Evaluation",
    }
    return [{"id": f"tier{tier}", "label": labels[tier]} for tier in sorted(tiers)]


def emit_reports(
    results: list[ValidationResult],
    *,
    report_formats: tuple[str, ...],
    output_dir: Path,
    basename: str = DEFAULT_REPORT_BASENAME,
    policy: ValidationPolicy | None = None,
    target_path: str | None = None,
    expected_skill_name: str | None = None,
    explicit_skill_name: str | None = None,
    content_label: str = "Skill",
    announce_paths: bool = True,
    sarif_scan_root: Path | None = None,
    sarif_repository_root: Path | None = None,
    requested_tiers: tuple[int, ...] | None = None,
    displayed_tiers: tuple[int, ...] | None = None,
    presentation_context: FinalRenderContext | None = None,
) -> bool:
    """Render reports and return whether every result passed.

    When *policy* is provided, finding severities are remapped per the active
    profile (and pass/fail recomputed) before rendering, and the active profile
    is stamped onto each result's metadata for reporters.

    *target_path* (the content's repo URL or path) and *content_label* are
    forwarded to the HTML reporter so the report shows a Target link and the
    correct content noun. HTML navigation tabs are derived from the tiers
    present in *results*, matching SkillEvaluator's combined report.

    The JSON report records *target_path* too. *explicit_skill_name* names
    the one skill a single-skill run validated, so the JSON ``skills`` list
    names it even when no finding path or quality score does.
    """
    if policy is not None:
        apply_policy(results, policy)
    context = presentation_context or build_final_render_context(
        results,
        requested_tiers=requested_tiers,
        displayed_tiers=displayed_tiers,
        expected_skill_name=expected_skill_name,
    )
    if "cli" in report_formats:
        # The CLI report previews failed Tier 3 source checks only when a file written below lists them all.
        CLIReporter(console=console, report_formats=report_formats, presentation_context=context).print_all(results)

    html_tabs = _derive_html_tabs(results)
    for fmt in report_formats:
        if fmt == "cli":
            continue
        reporter_cls = REPORTERS[fmt]
        if fmt == "html":
            reporter = reporter_cls(
                target_path=target_path,
                content_label=content_label,
                tabs=html_tabs,
                presentation_context=context,
                requested_tiers=requested_tiers,
                displayed_tiers=displayed_tiers,
                expected_skill_name=expected_skill_name,
            )
        elif fmt == "sarif":
            reporter = reporter_cls(
                workspace_root=sarif_repository_root,
                scan_root=sarif_scan_root,
            )
        elif fmt == "json":
            reporter = reporter_cls(
                expected_skill_name=expected_skill_name,
                target_path=target_path,
                explicit_skill_name=explicit_skill_name,
            )
        else:
            reporter = reporter_cls(expected_skill_name=expected_skill_name)
        output_path = output_dir / f"{basename}{reporter.get_file_extension()}"
        reporter.save(results, output_path)
        if announce_paths:
            console.print(f"[dim]{fmt} report:[/dim] [cyan]{escape_markup(str(output_path))}[/cyan]")

    return all(result.passed for result in results)
