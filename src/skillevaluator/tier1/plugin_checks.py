# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private static composition. No CLI, detection, registry or configuration hooks."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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


def _snapshot(root: Path, destination: Path, skills: list[str]) -> tuple[PurePosixPath, ...]:
    """One bounded copy with inode-pinned, no-follow reads before scanners run.

    Scanners receive an immutable task-owned input view and cannot follow source
    links or read an entry swapped between discovery and copying. Refusal never
    falls back to scanning the original filesystem.
    """
    # Explicit client paths override generic artifact pruning. All reads still
    # share the same path/byte caps, so an oversized loaded tree is incomplete.
    required_names = {part for skill in skills for part in Path(skill).parts}
    files = discover_secure_files(
        root,
        selected=lambda _: True,
        excluded_dirs=SCAN_EXCLUDED_DIRS - {"__pycache__"} - required_names,
        max_paths=PLUGIN_TREE_MAX_DISCOVERED_PATHS,
        allow_context_alias=False,
        refuse_selected_dirs=False,
    )
    remaining = CONTENT_DEDUP_MAX_TOTAL_BYTES
    with SecureRoot(root) as reader:
        for source in files:
            raw, metadata = reader.read_bytes(
                source.relative_path, min(CONTENT_DEDUP_MAX_FILE_BYTES, remaining), expected=source.metadata
            )
            remaining -= len(raw)
            output = destination / source.relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(raw)
            output.chmod(metadata.st_mode & 0o777)
    return tuple(PurePosixPath(file.relative_path.as_posix()) for file in files)


def _scan_units(skills: list[str], excluded: frozenset[str]) -> list[str]:
    units = ["."]
    for skill in sorted(skills, key=lambda value: (len(Path(value).parts), value)):
        target = Path(skill)
        ancestor = max(
            (Path(unit) for unit in units if target.is_relative_to(Path(unit))), key=lambda value: len(value.parts)
        )
        if any(part in excluded for part in target.relative_to(ancestor).parts):
            units.append(skill)
    return units


def _partition_views(
    snapshot: Path, destination: Path, files: tuple[PurePosixPath, ...], skills: list[str]
) -> dict[str, Path]:
    """Give each source file to its deepest skill, or to the plugin root.

    Readers inspect the complete, already verified snapshot once. Partition
    roots retain each skill's own manifest and root-relative scanner semantics,
    while parent scanners cannot rediscover nested skills or borrow their author,
    license, dependencies, or tests.
    """
    units = [".", *sorted(set(skills) - {"."}, key=lambda value: (len(Path(value).parts), value))]
    views = {unit: destination / f"unit-{index}" / (snapshot / unit).name for index, unit in enumerate(units)}
    for target in views.values():
        target.mkdir(parents=True)
    deepest_first = sorted(views, key=lambda unit: len(PurePosixPath(unit).parts), reverse=True)
    remaining = CONTENT_DEDUP_MAX_TOTAL_BYTES
    with SecureRoot(snapshot) as reader:
        for relative in files:
            owner = next(unit for unit in deepest_first if relative.is_relative_to(PurePosixPath(unit)))
            raw, metadata = reader.read_bytes(Path(relative), min(CONTENT_DEDUP_MAX_FILE_BYTES, remaining))
            remaining -= len(raw)
            output = views[owner] / relative.relative_to(PurePosixPath(owner))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(raw)
            output.chmod(metadata.st_mode & 0o777)
    return views


def _attribute_scan_result(result: ValidationResult, unit: str, view: Path, root: Path) -> None:
    """Map partition-local paths and grouped occurrences back to their owner."""
    if unit != ".":
        result.metadata["plugin_member"] = unit
        for finding in result.findings:
            location = Path(finding.file_path)
            if finding.file_path and not location.is_absolute() and not finding.file_path.startswith(("<", "[")):
                finding.file_path = (Path(unit) / location).as_posix()
            occurrences = finding.metadata.get("occurrences", [])
            for occurrence in occurrences if isinstance(occurrences, list) else []:
                if isinstance(occurrence, dict) and isinstance(occurrence.get("file"), str):
                    file = Path(occurrence["file"])
                    if not file.is_absolute():
                        occurrence["file"] = (Path(unit) / file).as_posix()
    _rebase(result, view, root / unit)


def _invoke_scan(runner, target: Path, check: str) -> ValidationResult:
    """A failed scanner leaves its unit incomplete while later units continue."""
    try:
        result = runner(target)
        if not isinstance(result, ValidationResult):
            raise TypeError("scanner returned an unsupported result")
        return result
    except Exception as exc:
        result = ValidationResult()
        result.mark_scan_incomplete(f"plugin-{check}")
        result.add_finding(
            Finding(
                category="PLUGIN_SCHEMA",
                severity=Severity.MEDIUM,
                check_name="private_scan_incomplete",
                message=f"The {check} scanner did not produce a usable result for this scan unit.",
                file_path=str(target),
                metadata={"exception_type": type(exc).__name__, "plugin_check": check},
            )
        )
        return result


def _run_hygiene(validator, snapshot: Path, views: dict[str, Path], skills: list[str], root: Path) -> ValidationResult:
    """Resolve links against the full snapshot; audit each unit's own metadata."""
    result = ValidationResult()
    # Root traversal already covers ordinary skill directories. Only explicitly
    # loaded directories that it prunes need another link pass, still using the
    # complete snapshot so valid links between components remain valid.
    for unit in _scan_units(skills, SCAN_EXCLUDED_DIRS):
        links = _invoke_scan(validator._check_dead_links, snapshot / unit, "hygiene-links")
        _rebase(links, snapshot, root)
        if not links.passed and not links.is_incomplete:
            result.metadata["confirmed_failure"] = True
        result.merge(links)
    for unit, view in views.items():
        dependencies = _invoke_scan(validator._audit_dependencies, view, "hygiene-dependencies")
        tests = _invoke_scan(validator._check_test_presence, view, "hygiene-tests")
        local = ValidationResult()
        for subcheck in (dependencies, tests):
            # Preserve confirmed content failures independently of another
            # subcheck's missing evidence. Scanner exceptions alone do not set it.
            if not subcheck.passed and not subcheck.is_incomplete:
                local.metadata["confirmed_failure"] = True
            local.merge(subcheck)
        _attribute_scan_result(local, unit, view, root)
        if unit != ".":
            local.errors = [f"[{unit}] {error}" for error in local.errors]
            local.warnings = [f"[{unit}] {warning}" for warning in local.warnings]
        for detail in local.success_details:
            detail.metadata["plugin_member"] = unit
        result.merge(local)
    result.metadata.pop("plugin_member", None)
    result.metadata.update(plugin_check="hygiene", plugin_scan_scope="whole_tree")
    return result


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
    source_files = inventory.source_files
    skills = sorted({m.path for m in inventory.components if m.type == "skill" and m.path and not m.problem})
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
                    source_files = _snapshot(root, view, skills)
                    copied_manifest = view / inventory.selected_manifest.manifest_filename
                    if hashlib.sha256(copied_manifest.read_bytes()).hexdigest() != inventory.manifest_sha256:
                        raise SecurePathError("source_changed", "Selected manifest changed during staging.")
                    views = _partition_views(view, Path(scratch) / "scanner-views", source_files, skills)
                    for key, (validator, runner) in _tree_checks().items():
                        if key not in scan_checks:
                            continue
                        if key == "hygiene":
                            result = _run_hygiene(validator, view, views, skills, root)
                            result.validator_name = validator.name
                            result.validator_description = validator.description
                            results.append(result)
                            continue
                        if runner is None:
                            runner = (
                                getattr(validator, "_validate_single_skill", None)
                                or getattr(validator, "_validate_single", None)
                                or validator._validate_single_asset
                            )
                        for unit, scan_view in views.items():
                            result = _invoke_scan(runner, scan_view, key)
                            result.validator_name = validator.name
                            result.validator_description = validator.description
                            result.metadata["plugin_check"] = key
                            result.metadata["plugin_scan_scope"] = "root_owned" if unit == "." else "declared_skill"
                            _attribute_scan_result(result, unit, scan_view, root)
                            results.append(result)
                    # Every audited client skill receives its own quality/lint/
                    # version/schema checks; root-owned files are scanned once above.
                    validators = _skill_checks()
                    for relative in skills:
                        target = view / relative
                        if not any((target / name).is_file() for name in ("SKILL.md", "skill.md")):
                            snapshot_result.mark_scan_incomplete("declared-skill-coverage")
                            snapshot_result.add_finding(
                                Finding(
                                    category="PLUGIN_SCHEMA",
                                    severity=Severity.HIGH,
                                    check_name="declared_skill_not_scanned",
                                    message="An inventoried skill is absent from the safe scan view.",
                                    file_path=str(root / relative),
                                )
                            )
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
                    snapshot_result.metadata.update(
                        source_files=len(source_files),
                        scan_units=list(views),
                        complete=not snapshot_result.is_incomplete,
                    )
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
        inputs = collect_audit_inputs(reader, source_files, mcp=declarations)
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
