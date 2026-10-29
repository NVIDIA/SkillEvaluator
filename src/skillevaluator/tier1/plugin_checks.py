# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private static composition. No CLI, detection, registry or configuration hooks."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_dependency_audit import collect_audit_inputs, collect_endpoint_targets, run_dependency_audits
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.plugin_paths import PluginRootReader
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, discover_secure_files
from skillevaluator.validators.claude_plugin_validate import (
    ClaudePluginValidateParity,
    skillevaluator_manifest_fields,
    skillevaluator_manifest_verdict,
)
from skillevaluator.validators.endpoint_resolution import EndpointChecker
from skillevaluator.validators.private_plugin_schema import VALIDATOR_NAME, validate_inventory_schema

TREE_CHECKS = frozenset({"security", "pii", "secrets", "code-integrity", "hygiene", "unicode", "license"})
SKILL_CHECKS = frozenset({"schema", "quality", "lint", "version"})


@dataclass(frozen=True)
class PrivatePluginChecks:
    checks: frozenset[str] = TREE_CHECKS | SKILL_CHECKS
    audit_dependencies: bool = False
    resolve_endpoints: bool = False
    claude_validate: bool = False
    allowed_private_hosts: tuple[str, ...] = ()
    hook_allowed_urls: tuple[str, ...] = ()


def _tree_checks():
    # Import on use: importing private inventory must not load optional scanners.
    from skillevaluator.validators.code_risk import CodeRiskValidator
    from skillevaluator.validators.hygiene import HygieneValidator
    from skillevaluator.validators.license import LicenseValidator
    from skillevaluator.validators.secrets import SecretsValidator
    from skillevaluator.validators.security import SecurityValidator
    from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator

    security = SecurityValidator(use_llm=False, verify_llm=False)
    return {
        "security": (security, security._run_skillspector),
        "pii": (security, security._scan_for_pii),
        "secrets": (SecretsValidator(), None),
        "code-integrity": (CodeRiskValidator(), None),
        "hygiene": (HygieneValidator(), None),
        "unicode": (UnicodeSmuggleValidator(), None),
        "license": (LicenseValidator(), None),
    }


def _skill_checks():
    from skillevaluator.validators.quality_score import QualityScoreValidator
    from skillevaluator.validators.schema import SchemaValidator
    from skillevaluator.validators.script_lint import ScriptLintValidator
    from skillevaluator.validators.version import VersionValidator

    return {
        "schema": SchemaValidator(),
        "quality": QualityScoreValidator(),
        "lint": ScriptLintValidator(),
        "version": VersionValidator(),
    }


def _rebase(result: ValidationResult, old: Path, new: Path) -> None:
    def rewrite(value):
        if isinstance(value, str):
            return value.replace(str(old), str(new))
        if isinstance(value, BaseModel):
            return value.model_copy(update={key: rewrite(getattr(value, key)) for key in type(value).model_fields})
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, tuple):
            return tuple(rewrite(v) for v in value)
        return value

    for finding in result.findings:
        for name in ("file_path", "message", "suggestion", "line_content"):
            setattr(finding, name, rewrite(getattr(finding, name)))
        finding.metadata = rewrite(finding.metadata)
    result.metadata = rewrite(result.metadata)
    for name in ("errors", "warnings", "messages"):
        setattr(result, name, rewrite(getattr(result, name)))
    for success in result.success_details:
        success.message = rewrite(success.message)
        success.metadata = rewrite(success.metadata)


def _snapshot(root: Path, destination: Path) -> None:
    """One bounded copy with inode-pinned, no-follow reads before scanners run.

    Scanners receive an immutable task-owned input view and cannot follow source
    links or read an entry swapped between discovery and copying. Refusal never
    falls back to scanning the original filesystem.
    """
    files = discover_secure_files(
        root,
        selected=lambda _: True,
        excluded_dirs=SCAN_EXCLUDED_DIRS,
        max_paths=PLUGIN_TREE_MAX_DISCOVERED_PATHS,
        allow_context_alias=False,
        refuse_selected_dirs=False,
    )
    remaining = CONTENT_DEDUP_MAX_TOTAL_BYTES
    with SecureRoot(root) as reader:
        for source in files:
            raw, _ = reader.read_bytes(
                source.relative_path, min(CONTENT_DEDUP_MAX_FILE_BYTES, remaining), expected=source.metadata
            )
            remaining -= len(raw)
            output = destination / source.relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(raw)


def run_plugin_validation(path: Path, *, options: PrivatePluginChecks | None = None) -> list[ValidationResult]:
    """Assemble a private run; network collectors require independent explicit options."""
    options = options or PrivatePluginChecks()
    unknown = options.checks - (TREE_CHECKS | SKILL_CHECKS)
    if unknown:
        raise ValueError(f"Unknown private checks: {sorted(unknown)}")
    inventory = plugin_inventory_for_root(
        path, allowed_private_hosts=options.allowed_private_hosts, hook_allowed_urls=options.hook_allowed_urls
    )
    if inventory is None:
        result = ValidationResult(validator_name=VALIDATOR_NAME)
        result.add_finding(
            Finding(
                category="PLUGIN_SCHEMA",
                severity=Severity.HIGH,
                check_name="manifest_missing_or_unsafe",
                message="No supported manifest could be safely located.",
                file_path=str(path),
            )
        )
        result.mark_scan_incomplete("plugin-inventory")
        return [result]
    root = inventory.root
    # Composition is mandatory even for a narrowed scanner selection: a declared
    # but unreadable tree cannot claim complete coverage.
    schema = validate_inventory_schema(inventory)
    results = [schema]
    scan_checks = options.checks & TREE_CHECKS
    skill_checks = options.checks & SKILL_CHECKS
    if scan_checks or skill_checks:
        snapshot_result = ValidationResult(validator_name="Private Tree Coverage")
        if not inventory.tree_complete:
            snapshot_result.mark_scan_incomplete("plugin-source-tree")
            results.append(snapshot_result)
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="skillevaluator-private-tree-") as scratch:
                    view = Path(scratch) / root.name
                    view.mkdir()
                    _snapshot(root, view)
                    for key, (validator, runner) in _tree_checks().items():
                        if key not in scan_checks:
                            continue
                        if runner is None:
                            runner = (
                                getattr(validator, "_validate_single_skill", None)
                                or getattr(validator, "_validate_single", None)
                                or validator._validate_single_asset
                            )
                        result = runner(view)
                        result.validator_name = validator.name
                        result.validator_description = validator.description
                        result.metadata["plugin_check"] = key
                        result.metadata["plugin_scan_scope"] = "whole_tree"
                        _rebase(result, view, root)
                        results.append(result)
                    # Every audited client skill receives its own quality/lint/
                    # version/schema checks; root-owned files are scanned once above.
                    skills = sorted(
                        {m.path for m in inventory.components if m.type == "skill" and m.path and not m.problem}
                    )
                    validators = _skill_checks()
                    for relative in skills:
                        target = view / relative
                        if not any((target / name).is_file() for name in ("SKILL.md", "skill.md")):
                            continue
                        for key, validator in validators.items():
                            if key not in skill_checks:
                                continue
                            result = validator.validate(target)
                            result.validator_name = validator.name
                            result.validator_description = validator.description
                            result.metadata.update(plugin_check=key, plugin_member=relative)
                            _rebase(result, view, root)
                            results.append(result)
                    snapshot_result.metadata.update(source_files=len(inventory.source_files), complete=True)
                    results.append(snapshot_result)
            except (SecurePathError, OSError):
                snapshot_result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="unsafe_or_bounded_source_tree",
                        message="Could not stage the complete source tree within safe read limits.",
                        file_path=str(root),
                    )
                )
                snapshot_result.mark_scan_incomplete("plugin-source-tree")
                results.append(snapshot_result)
    declarations = inventory.all_mcp_declarations()
    if options.audit_dependencies:
        reader = PluginRootReader(root)
        inputs = collect_audit_inputs(reader, inventory.source_files, mcp=declarations)
        audit = run_dependency_audits(reader, inputs)
        if not inventory.tree_complete:
            audit.mark_scan_incomplete("dependency-source-discovery")
        results.append(audit)
    if options.resolve_endpoints:
        targets = collect_endpoint_targets(
            declarations, inventory.hook_records, allowed_hosts=options.allowed_private_hosts
        )
        summary, findings = EndpointChecker().check(targets)
        result = ValidationResult(validator_name="Private Endpoint Checks")
        for finding in findings:
            result.add_finding(finding)
        result.metadata["endpoint_resolution"] = summary
        if summary.get("incomplete") or not inventory.complete:
            result.mark_scan_incomplete("endpoint-resolution")
        results.append(result)
    if options.claude_validate:
        results.append(
            ClaudePluginValidateParity().validate(
                root,
                skillevaluator_verdict=skillevaluator_manifest_verdict([schema], VALIDATOR_NAME),
                skillevaluator_fields=skillevaluator_manifest_fields([schema], VALIDATOR_NAME),
            )
        )
    inventory.attribute_findings(f for result in results[1:] for f in result.findings)
    schema.metadata["plugin"]["component_inventory"] = inventory.component_inventory()
    return results
