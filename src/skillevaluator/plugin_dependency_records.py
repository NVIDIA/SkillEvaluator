# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private offline dependency classifications and the missing-reference gate."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import PLUGIN_CONTAINED_MANIFEST_TYPE, PLUGIN_MANIFEST_TYPE
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_dependencies import (
    DependencyRow,
    classify_plugin_dependencies,
    record_unverified_dependencies,
    resolve_repository_identity,
    status_counts,
)
from skillevaluator.plugin_format_schema import CLAUDE_DEPENDENCY_RE
from skillevaluator.plugin_manifest import (
    CLAUDE_DEPENDENCY_MAX_ENTRIES,
    PluginManifestPathError,
    classify_claude_dependency,
    find_claude_marketplace,
)
from skillevaluator.plugin_refs import normalize_ref
from skillevaluator.utils.helpers import resolve_git_root

if TYPE_CHECKING:
    from skillevaluator.plugin_inventory import PluginInventory
    from skillevaluator.plugin_manifest import PluginManifestFile

VALIDATOR_NAME = "Private Dependency References"


def _finding(
    result: ValidationResult,
    manifest: PluginManifestFile,
    check: str,
    message: str,
    *,
    severity: Severity,
    **metadata: Any,
) -> None:
    result.add_finding(
        Finding(
            category="PLUGIN_SCHEMA",
            severity=severity,
            check_name=check,
            message=message[:1000],
            file_path=str(manifest.declared_path),
            metadata={"declared_by": manifest.manifest_filename, **metadata},
        )
    )


def _annotate(
    inventory: PluginInventory, manifest: PluginManifestFile, section: str, rows: Iterable[DependencyRow]
) -> None:
    kind = {"skills": "skill", "rules": "rule"}.get(section)
    if kind is None:
        return
    for row in rows:
        for member in list(inventory.components):
            if member.type != kind or member.path is not None or member.declared_by != manifest.manifest_filename:
                continue
            if member.name != row.ref and normalize_ref(member.name) != normalize_ref(row.ref):
                continue
            member.dependency_state = row.state
            if row.state == "missing":
                member.problem = "missing"
            elif row.state == "provided" and row.path:
                packaged = next(
                    (other for other in inventory.components if other.type == kind and other.path == row.path), None
                )
                if packaged is not None:
                    packaged.origin = "declared+packaged"
                    packaged.dependency_state = "provided"
                    inventory.components.remove(member)


def _claude_rows(inventory: PluginInventory, data: dict[str, Any], repo_root: Path | None) -> list[DependencyRow]:
    entries = data.get("dependencies")
    if not isinstance(entries, list):
        return []
    stop = None
    for candidate in (repo_root, resolve_git_root(inventory.root)):
        if candidate is not None:
            candidate = candidate.expanduser().resolve()
            if inventory.root.resolve().is_relative_to(candidate):
                stop = candidate
                break
    marketplace = find_claude_marketplace(
        inventory.root, plugin_name=data.get("name") if isinstance(data.get("name"), str) else None, stop=stop
    )
    rows = []
    for entry in entries[:CLAUDE_DEPENDENCY_MAX_ENTRIES]:
        if isinstance(entry, str) and len(entry) <= 1024 and (match := CLAUDE_DEPENDENCY_RE.fullmatch(entry)):
            name, market = match.group("name"), match.group("marketplace")
        elif (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and len(entry["name"]) <= 1024
            and (match := CLAUDE_DEPENDENCY_RE.fullmatch(entry["name"]))
            and not match.group("marketplace")
            and (entry.get("marketplace") is None or isinstance(entry["marketplace"], str))
        ):
            name, market = entry["name"], entry.get("marketplace") or None
            if market is not None and (len(market) > 1024 or not CLAUDE_DEPENDENCY_RE.fullmatch(market)):
                rows.append(
                    DependencyRow(
                        "<invalid dependency>",
                        "unresolved",
                        None,
                        "malformed marketplace dependency",
                        cause="malformed",
                    )
                )
                continue
        else:
            rows.append(
                DependencyRow(
                    "<invalid dependency>", "unresolved", None, "malformed client dependency", cause="malformed"
                )
            )
            continue
        rows.append(classify_claude_dependency(name, market, marketplace))
    return rows


def dependency_records(inventory: PluginInventory, *, repo_root: Path | None = None) -> ValidationResult:
    """Classify all original manifest refs locally; only proven missing targets block."""
    result = ValidationResult(validator_name=VALIDATOR_NAME)
    combined: dict[str, list[dict[str, Any]]] = {"skills": [], "rules": [], "plugins": []}
    all_rows: list[DependencyRow] = []
    manifests = []
    identity = None
    for manifest in inventory.discovery.manifests:
        try:
            data = manifest.parse_for_audit()
        except PluginManifestPathError:
            data = None
        if data is None:
            _finding(
                result,
                manifest,
                "plugin_dependency_unverified",
                "The manifest could not be read for offline dependency classification.",
                severity=Severity.MEDIUM,
                state="unresolved",
            )
            result.mark_scan_incomplete("plugin-dependency-resolution")
            continue
        start = len(result.findings)
        sections: dict[str, list[DependencyRow]] = {}
        if manifest.manifest_type == PLUGIN_MANIFEST_TYPE:
            if identity is None:
                identity = resolve_repository_identity(inventory.root, repo_root)
            if identity.repo_root_ignored:
                result.add_message(
                    "The supplied repository root does not contain this component tree; dependency classification uses its enclosing repository."
                )
            try:
                resolution = classify_plugin_dependencies(
                    data,
                    inventory.root,
                    identity,
                    bundled_skills=(m.path for m in inventory.components if m.type == "skill" and m.path),
                    limit=None,
                )
            except (ValueError, OSError):
                _finding(
                    result,
                    manifest,
                    "plugin_dependency_unverified",
                    "The dependency sections exceeded the offline classifier's syntax or work limits.",
                    severity=Severity.MEDIUM,
                    state="unresolved",
                )
                result.mark_scan_incomplete("plugin-dependency-resolution")
                continue
            sections = {"skills": list(resolution.skills), "rules": list(resolution.rules)}
            record_unverified_dependencies(resolution, str(manifest.declared_path), result)
        elif manifest.manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            sections = {"plugins": _claude_rows(inventory, data, repo_root)}
            entries = data.get("dependencies")
            if isinstance(entries, list) and len(entries) > CLAUDE_DEPENDENCY_MAX_ENTRIES:
                _finding(
                    result,
                    manifest,
                    "plugin_dependency_limit",
                    "Client dependencies exceed the bounded offline classification limit.",
                    severity=Severity.MEDIUM,
                )
                result.mark_scan_incomplete("plugin-dependency-resolution")
        else:
            continue
        by_manifest = {}
        for section, rows in sections.items():
            _annotate(inventory, manifest, section, rows)
            by_manifest[section] = []
            for row in rows:
                metadata = {"ref": row.ref, "state": row.state, "section": section}
                if row.state == "missing":
                    _finding(
                        result,
                        manifest,
                        "plugin_dependency_missing",
                        f"Declared {section} dependency '{row.ref}' is missing: {row.reason}.",
                        severity=Severity.HIGH,
                        **metadata,
                    )
                elif row.state == "external" or (section == "plugins" and row.state == "unresolved"):
                    _finding(
                        result,
                        manifest,
                        "plugin_dependency_unverified",
                        f"Declared {section} dependency '{row.ref}' remains unverified offline: {row.reason}.",
                        severity=Severity.MEDIUM,
                        **metadata,
                    )
                else:
                    result.add_message(f"Dependency '{row.ref}': {row.state} ({row.reason}).")
                output = {**row.to_dict(), "declared_by": manifest.manifest_filename}
                if row.cause:
                    output["cause"] = row.cause
                by_manifest[section].append(output)
                combined[section].append(output)
                all_rows.append(row)
        for finding in result.findings[start:]:
            finding.metadata.setdefault("declared_by", manifest.manifest_filename)
        manifests.append(
            {
                "manifest_filename": manifest.manifest_filename,
                "selected": manifest == inventory.selected_manifest,
                "resolution": by_manifest,
            }
        )
    counts = status_counts(all_rows)
    result.metadata["plugin"] = {
        "dependency_resolution": combined,
        "dependency_status_counts": counts,
        "dependency_manifests": manifests,
    }
    if all_rows and not counts["missing"] and not counts["unresolved"]:
        result.add_success(
            check_name="plugin_dependencies",
            message="Every local dependency target was classified; external declarations remain unverified.",
        )
    inventory.attribute_findings(result.findings)
    return result
