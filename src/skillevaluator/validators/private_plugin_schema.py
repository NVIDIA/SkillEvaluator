# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private manifest schema composition; public basic-bundle validation is unchanged."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from skillevaluator.constants import (
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_MANIFEST_TYPE,
)
from skillevaluator.models.plugin import PluginManifest
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_dependency_records import dependency_records
from skillevaluator.plugin_format_schema import validate_manifest_fields
from skillevaluator.plugin_formats import parse_manifest_text
from skillevaluator.plugin_manifest import PluginManifestPathError
from skillevaluator.utils.structured_data import StructuredDataError

if TYPE_CHECKING:
    from skillevaluator.plugin_inventory import PluginInventory

VALIDATOR_NAME = "Private Manifest and Component Checks"
MAX_SCHEMA_FINDINGS = 100


def validate_inventory_schema(inventory: PluginInventory) -> ValidationResult:
    """Validate every declared format and compose its inventory findings.

    The additional manifests remain audit sources even when another client is
    selected. Parsing or size failures cannot silently become a passing schema.
    """
    result = ValidationResult(validator_name=VALIDATOR_NAME)
    findings = list(inventory.findings)
    declarations = []
    selected_data = None
    for manifest in inventory.discovery.manifests:
        data = manifest.parse_for_audit()
        declarations.append(
            {
                "manifest_type": manifest.manifest_type,
                "manifest_filename": manifest.manifest_filename,
                "selected": manifest == inventory.selected_manifest,
                "name": data.get("name") if data else None,
                "version": data.get("version") if data else None,
            }
        )
        try:
            raw = manifest.read_text(encoding="utf-8-sig")
            strict = parse_manifest_text(manifest.manifest_type, raw)
            if manifest.manifest_type == PLUGIN_MANIFEST_TYPE and isinstance(data, dict):
                value = data.get("version")
                literal = re.search(r"^version[ \t]*:[ \t]*([0-9][0-9._eE+-]*)[ \t]*(?:#.*)?$", raw, re.MULTILINE)
                if type(value) in (int, float) and literal:
                    try:
                        same = float(literal[1].replace("_", "")) == float(value)
                    except ValueError:
                        same = False
                    if same:
                        data = {**data, "version": literal[1]}
                        declarations[-1]["version"] = literal[1]
            if not isinstance(strict, dict) or not strict:
                raise ValueError("manifest must be a nonempty mapping")
        except (PluginManifestPathError, StructuredDataError, ValueError):
            findings.append(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_unreadable",
                    message="Manifest did not pass the bounded syntax/read checks.",
                    file_path=str(manifest.declared_path),
                    metadata={"manifest_type": manifest.manifest_type},
                )
            )
            result.mark_scan_incomplete("manifest-schema")
        if not data:
            continue
        if manifest == inventory.selected_manifest:
            selected_data = data
        if manifest.case_variant:
            findings.append(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_case_variant",
                    message="Manifest spelling differs from the client path.",
                    file_path=str(manifest.declared_path),
                )
            )
        if manifest.manifest_type == PLUGIN_MANIFEST_TYPE:
            try:
                PluginManifest.model_validate(data)
            except ValidationError as exc:
                for error in exc.errors(include_input=False, include_url=False):
                    field = ".".join(map(str, error["loc"])) or "<root>"
                    findings.append(
                        Finding(
                            category="PLUGIN_SCHEMA",
                            severity=Severity.HIGH,
                            check_name=f"schema:{field}:{error['type']}",
                            message=error["msg"][:500],
                            file_path=str(manifest.declared_path),
                            metadata={"field": field, "manifest_type": manifest.manifest_type},
                        )
                    )
        else:
            overlay = (
                manifest.manifest_type == PLUGIN_CODEX_MANIFEST_TYPE
                and inventory.selected_manifest.manifest_type == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE
            )
            for issue in validate_manifest_fields(manifest.manifest_type, data, overlay=overlay):
                findings.append(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity={"error": Severity.HIGH, "warning": Severity.MEDIUM, "note": Severity.LOW}[
                            issue.level
                        ],
                        check_name=f"schema:{issue.field}:{issue.error}",
                        message=issue.message,
                        file_path=str(manifest.declared_path),
                        suggestion=issue.suggestion,
                        metadata={"field": issue.field, "manifest_type": manifest.manifest_type},
                    )
                )
    if selected_data:
        for row in declarations:
            if not row["selected"] and any(
                row[k] is not None and selected_data.get(k) is not None and row[k] != selected_data[k]
                for k in ("name", "version")
            ):
                findings.append(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.MEDIUM,
                        check_name="plugin_manifest_conflict",
                        message="Client manifests declare different names or versions.",
                        file_path=str(inventory.root / row["manifest_filename"]),
                        metadata=row,
                    )
                )
    # Preserve the most severe evidence when reports are bounded.
    order = {severity: i for i, severity in enumerate(Severity)}
    ranked = sorted(enumerate(findings), key=lambda pair: (order[pair[1].severity], pair[0]))
    for _, finding in sorted(ranked[:MAX_SCHEMA_FINDINGS]):
        result.add_finding(finding)
    if len(findings) > MAX_SCHEMA_FINDINGS:
        severity = ranked[MAX_SCHEMA_FINDINGS][1].severity
        result.add_finding(
            Finding(
                category="PLUGIN_SCHEMA",
                severity=severity,
                check_name="schema_findings_truncated",
                message=f"{len(findings) - MAX_SCHEMA_FINDINGS} further findings omitted.",
                file_path=str(inventory.root),
            )
        )
    if not inventory.complete:
        result.mark_scan_incomplete("plugin-inventory")
    result.metadata["plugin"] = {
        "root": str(inventory.root),
        "manifest_sha256": inventory.manifest_sha256,
        "manifest_declarations": declarations,
        "component_inventory": inventory.component_inventory(),
        "mcp": inventory.mcp_summary(),
        "context_cost": inventory.context_cost(),
    }
    inventory_metadata = result.metadata["plugin"]
    dependencies = dependency_records(inventory)
    result.merge(dependencies)
    result.metadata["plugin"] = {**inventory_metadata, **dependencies.metadata.get("plugin", {})}
    result.metadata["plugin"]["component_inventory"] = inventory.component_inventory()
    return result
