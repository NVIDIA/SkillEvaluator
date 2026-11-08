# SPDX-License-Identifier: Apache-2.0
"""Public static options over the shared bounded plugin composition."""

from __future__ import annotations

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
    selected = active - {"dependency"}
    if "code-integrity" in active:
        selected |= {"hygiene", "secrets"} - skipped
    selected = frozenset(selected)
    initial_target = public.publication_target_from_path(root)
    loaded_waivers = None

    def tree_factory(source_root, snapshot, views):
        nonlocal loaded_waivers

        def mapped_option(path):
            if path is None:
                return None
            absolute = path.absolute()
            for unit in sorted(views, key=lambda item: len(Path(item).parts), reverse=True):
                source = source_root / unit
                if absolute.is_relative_to(source):
                    return views[unit] / absolute.relative_to(source)
            return path

        security = public.SecurityValidator(
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

    def notify(check):
        canonical = "code-integrity" if check in {"hygiene", "secrets"} else check
        if canonical in active and canonical not in notified:
            notified.add(canonical)
            if on_check is not None:
                on_check(canonical)

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
        )
    if loaded_waivers is not None:
        waiver_result = ValidationResult(validator_name="PII Waiver Review", metadata={"plugin_check": "pii"})
        for finding, is_error in loaded_waivers.findings(root):
            waiver_result.add_structured_finding(finding, is_error=is_error)
        if loaded_waivers.applied:
            waiver_result.add_message(f"PII waivers suppressed {len(loaded_waivers.applied)} reviewed PII value(s)")
        results.append(waiver_result)
    public_names = {
        "Private Manifest and Component Checks": "Plugin Schema & Component Checks",
        "Private Tree Coverage": "Plugin Tree Coverage",
        "Private Endpoint Checks": "Plugin Endpoint Checks",
    }
    for result in results:
        result.validator_name = public_names.get(result.validator_name, result.validator_name)
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
        public.stamp_tier1_required_checks([result], skill_like=False)
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
        public.stamp_tier1_required_checks([result], skill_like=False)
        results.append(result)
    public._stamp_tier1_scan_scope(results, root)
    public.finalize_publication_target(results, root, initial_target)
    return results
