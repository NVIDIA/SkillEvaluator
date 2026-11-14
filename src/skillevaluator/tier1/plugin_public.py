# SPDX-License-Identifier: Apache-2.0
"""Public static options over the shared bounded plugin composition."""

from __future__ import annotations

import time
from pathlib import Path

from skillevaluator.models.result import ValidationResult


class _SourceWaivers:
    """Match immutable scan-view paths against their original logical paths."""

    def __init__(self, waivers, view: Path, source: Path):
        self.waivers, self.view, self.source = waivers, view, source

    def cover(self, path: Path) -> None:
        self.waivers.cover(self.source / path.relative_to(self.view))

    def matcher(self, path: Path):
        return self.waivers.matcher(self.source / path.relative_to(self.view))


def run_public_plugin_validation(
    target_path: Path,
    *,
    checks=None,
    skip=None,
    use_llm=False,
    llm_verify=False,
    min_score=70,
    quality_config=None,
    pii_waivers=None,
    previous_version=None,
    policy=None,
    fail_fast=False,
    continue_on_failure=False,
    on_check=None,
    skillspector_baseline=None,
    repo_root=None,
    resolve_endpoints=False,
    claude_validate=False,
) -> list[ValidationResult]:
    from skillevaluator.cli_core import resolve_plugin_path
    from skillevaluator.constants import PLUGIN_CONTAINED_MODE, PLUGIN_MANIFEST_TYPE, PLUGIN_MODE
    from skillevaluator.tier1 import commands as public
    from skillevaluator.tier1.plugin_checks import PrivatePluginChecks, run_plugin_validation
    from skillevaluator.validators.base import continue_on_failure_scope
    from skillevaluator.validators.pii_waivers import PIIWaiverSet, configured_pii_categories

    root = resolve_plugin_path(target_path)
    enabled, skipped = public._enabled_checks(checks), public.skipped_checks(skip)
    unknown = (enabled - public.RECOGNIZED_CHECKS) | (skipped - public.SKIPPABLE_CHECKS)
    if unknown:
        result = ValidationResult(validator_name="Tier 1 option validation")
        result.add_error(f"Unknown Tier 1 check(s): {', '.join(sorted(unknown))}")
        return [result]
    active = enabled - skipped
    selected = active - {"dependency", "harness"}
    if "code-integrity" in active:
        selected |= {"hygiene", "secrets"} - skipped
    selected = frozenset(selected)
    initial_target = public.publication_target_from_path(root)
    loaded_waivers = None

    def tree_factory(source_root, snapshot, views):
        nonlocal loaded_waivers
        from skillevaluator.validators.security import _rewrite_path_prefix

        snapshot_by_view = {view.resolve(): (snapshot / unit).resolve() for unit, view in views.items()}

        class SnapshotSecurityValidator(public.SecurityValidator):
            def _skillspector_issue_scopes(self, issues, components, *, source_root, report_truncated):
                # Scanner inputs are partitioned, but reachability must include
                # references from every component in the immutable full tree.
                if source_root is not None and (scope_root := snapshot_by_view.get(source_root.resolve())) is not None:
                    issues = _rewrite_path_prefix(issues, str(source_root), str(scope_root))
                    components = _rewrite_path_prefix(components, str(source_root), str(scope_root))
                    source_root = scope_root
                return super()._skillspector_issue_scopes(
                    issues, components, source_root=source_root, report_truncated=report_truncated
                )

        def mapped_option(path):
            if path is None:
                return None
            absolute = path.absolute()
            for unit in sorted(views, key=lambda item: len(Path(item).parts), reverse=True):
                source = source_root / unit
                if absolute.is_relative_to(source):
                    return views[unit] / absolute.relative_to(source)
            return path

        security = SnapshotSecurityValidator(
            use_llm=use_llm,
            verify_llm=llm_verify,
            content_type="plugin",
            skillspector_reference_root=snapshot,
            skillspector_baseline=mapped_option(skillspector_baseline),
        )
        pii = public.SecurityValidator(use_llm=False, verify_llm=llm_verify)
        if pii_waivers is not None and "pii" in selected:
            loaded_waivers = PIIWaiverSet.load(pii_waivers, configured_pii_categories(pii.pii_patterns))
        source_by_view = {view: source_root / unit for unit, view in views.items()}

        def verify(result, validator, view):
            if llm_verify and result.findings:
                try:
                    validator._verify_findings_with_llm(result, view)
                except Exception as exc:
                    result.mark_scan_incomplete("llm-verification")
                    result.add_warning(
                        f"Optional verification unavailable ({type(exc).__name__}); static findings retained."
                    )
            return result

        def security_scan(view):
            return verify(security._run_skillspector(view), security, view)

        def pii_scan(view):
            if loaded_waivers is not None:
                pii._pii_waivers = _SourceWaivers(loaded_waivers, view, source_by_view[view])
            try:
                return verify(pii._scan_for_pii(view), pii, view)
            finally:
                pii._pii_waivers = None

        return {
            "security": (security, security_scan),
            "pii": (pii, pii_scan),
            "secrets": (public.SecretsValidator(), None),
            "code-integrity": (public.CodeRiskValidator(), None),
            "hygiene": (public.HygieneValidator(), None),
            "unicode": (public.UnicodeSmuggleValidator(), None),
            "license": (public.LicenseValidator(), None),
        }

    def skill_factory():
        return {
            "schema": public.SchemaValidator(policy=policy),
            "quality": public.QualityScoreValidator(min_score=min_score, quality_config=quality_config),
            "lint": public.ScriptLintValidator(),
            "version": public.VersionValidator(previous_version=previous_version),
        }

    notified = set()
    progress_total, progress_step, progress_started = 0, 0, 0.0

    def plan_progress(total):
        nonlocal progress_total
        progress_total = total

    def notify(check):
        nonlocal progress_step, progress_started
        canonical = "code-integrity" if check in {"hygiene", "secrets"} else check
        if on_check is None:
            progress_step += 1
            progress_started = time.monotonic()
            public.progress_console.print(
                f"[{progress_step}/{progress_total}] {canonical} ...", markup=False, highlight=False
            )
            return
        if canonical in active and canonical not in notified:
            notified.add(canonical)
            on_check(canonical)

    def complete_progress(check, result):
        if on_check is not None:
            return
        canonical = "code-integrity" if check in {"hygiene", "secrets"} else check
        outcome = (
            f"incomplete: {', '.join(result.incomplete_scans)}"
            if result.is_incomplete
            else "ok"
            if result.passed
            else f"{result.summary.errors} error(s)"
        )
        if result.summary.warnings:
            outcome += f", {result.summary.warnings} warning(s)"
        public.progress_console.print(
            f"[{progress_step}/{progress_total}] {canonical} done in {time.monotonic() - progress_started:.1f}s ({outcome})",
            markup=False,
            highlight=False,
        )

    with continue_on_failure_scope(continue_on_failure):
        results = run_plugin_validation(
            target_path,
            options=PrivatePluginChecks(
                checks=selected,
                audit_dependencies="dependency" in active,
                resolve_endpoints=resolve_endpoints or bool(policy and policy.resolve_endpoints),
                claude_validate=claude_validate,
                allowed_private_hosts=policy.mcp_allowed_private_hosts if policy else (),
                hook_allowed_urls=policy.hook_allowed_urls if policy else (),
                repo_root=repo_root,
            ),
            tree_checks_factory=tree_factory,
            skill_checks_factory=skill_factory,
            fail_fast=fail_fast and not continue_on_failure,
            on_check=notify,
            on_check_plan=plan_progress,
            on_check_complete=complete_progress,
        )
    if loaded_waivers is not None:
        waiver_result = ValidationResult(validator_name="PII Waiver Review", metadata={"plugin_check": "pii"})
        for finding, is_error in loaded_waivers.findings(root):
            waiver_result.add_structured_finding(finding, is_error=is_error)
        if loaded_waivers.applied:
            waiver_result.add_message(f"PII waivers suppressed {len(loaded_waivers.applied)} reviewed PII value(s)")
        results.append(waiver_result)
    if "harness" in active:
        # Native-plugin profiles are exposed through harness-check; combined
        # validation does not yet run that separate advisory.
        harness = ValidationResult(validator_name="Harness Compatibility", metadata={"plugin_check": "harness"})
        harness.add_success(
            "harness_applicability",
            "Not applicable in combined validation; use harness-check to assess native plugin loading.",
            not_applicable=True,
        )
        results.append(harness)
    public_names = {
        "Private Manifest and Component Checks": "Plugin Schema & Component Checks",
        "Private Tree Coverage": "Plugin Tree Coverage",
        "Private Endpoint Checks": "Plugin Endpoint Checks",
    }
    # Both tree scans run on SecurityValidator instances, so they share its name.
    # Report each under its skill-run and --skip name; the panel maps checks by name.
    scan_names = {
        "security": ("Security Scan", None),
        "pii": ("PII Scan", "Detect PII and local identifiers"),
    }
    skill_members_present = any(
        member.get("type") == "skill" and member.get("path") and not member.get("problem")
        for result in results
        for member in result.metadata.get("plugin", {}).get("component_inventory", {}).get("components", [])
    )
    for result in results:
        result.validator_name = public_names.get(result.validator_name, result.validator_name)
        scan = scan_names.get(result.metadata.get("plugin_check"))
        if scan is not None and result.validator_name == "Security & PII Scanning":
            result.validator_name = scan[0]
            result.validator_description = scan[1] or result.validator_description
        if result.validator_name == "Plugin Schema & Component Checks" and result.passed and not result.is_incomplete:
            result.add_success("plugin_schema", "Completed bounded plugin manifest and component checks")
        if plugin := result.metadata.get("plugin"):
            chosen = next((row for row in plugin.get("manifest_declarations", []) if row.get("selected")), None)
            if chosen:
                result.metadata["manifest_type"] = chosen["manifest_type"]
                result.metadata["plugin_mode"] = (
                    PLUGIN_MODE if chosen["manifest_type"] == PLUGIN_MANIFEST_TYPE else PLUGIN_CONTAINED_MODE
                )
                plugin.update({key: chosen[key] for key in ("name", "version") if chosen.get(key) is not None})
        check = result.metadata.get("plugin_check", "schema")
        check = "code-integrity" if check in {"hygiene", "secrets"} else check
        if "endpoint_resolution" in result.metadata:
            check = "schema"
        elif result.validator_name == "Dependency Vulnerability Audit":
            check = "dependency"
        public.stamp_publication_evidence([result], tier=1, check_id=check)
        public.stamp_tier1_required_checks([result], skill_like=skill_members_present)
    omitted = [(check, name) for check in enabled & skipped for name in public._SKIPPED_VALIDATORS[check]]
    if "code-integrity" in active and "secrets" in skipped:
        omitted.append(("code-integrity", "Secrets Detection"))
    for check, name in omitted:
        result = ValidationResult(
            validator_name=name,
            metadata={
                "skipped": True,
                "skipped_by_user": True,
                "skip_reason": public.SKIP_REASON,
            },
        )
        public.stamp_publication_evidence([result], tier=1, check_id=check)
        public.stamp_tier1_required_checks([result], skill_like=skill_members_present)
        results.append(result)
    public._stamp_tier1_scan_scope(results, root)
    public.finalize_publication_target(results, root, initial_target)
    return results
