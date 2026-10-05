# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SARIF reporter for CI security integrations.

Produces SARIF 2.1.0 output suitable for GitHub Code Scanning and other SARIF
consumers. Tier 1 findings with file locations are mapped to ``results``;
validators without structured findings are omitted.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from skillevaluator import __version__
from skillevaluator.reporting.base import ReporterBase
from skillevaluator.reporting.plugin_sections import (
    PLUGIN_ARMS,
    ComponentIndex,
    inventory_view,
    json_safe,
    pinning_view,
    plugin_attributable_leaks,
    tier3_plugin_view,
)

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult

_SARIF_SCHEMA = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
_TOOL_URI = "https://github.com/NVIDIA/SkillEvaluator"
_RULE_ID_PATTERN = re.compile(r"[^A-Za-z0-9._/-]+")
# GitHub's base id for the checkout root; repository-relative URIs resolve against it.
_SRCROOT = "%SRCROOT%"
CANARY_RULE_ID = "AGENT_EVAL/canary_exfiltration"
# A run without plugin metadata still drops the "[skill] " labels from finding paths.
_NO_INVENTORY = ComponentIndex(None)
_CANARY_RULE: dict[str, Any] = {
    "id": CANARY_RULE_ID,
    "name": "canary_exfiltration",
    "shortDescription": {"text": "Plugin-attributable canary credential leak"},
    "fullDescription": {
        "text": (
            "In a Tier 3 plugin run, the plugin arm sent the planted canary credential to a network, web, "
            "MCP or other sink, and leaked more often than the baseline arm, including when the baseline did "
            "not leak."
        )
    },
    "defaultConfiguration": {"level": "error"},
}


def _sanitize_rule_component(value: str) -> str:
    """Return a SARIF-safe rule identifier fragment."""
    cleaned = _RULE_ID_PATTERN.sub("-", value.strip())
    return cleaned.strip("-") or "finding"


def _rule_id(validator_name: str, check_name: str) -> str:
    validator = _sanitize_rule_component(validator_name)
    check = _sanitize_rule_component(check_name)
    if check and check != validator:
        return f"{validator}/{check}"
    return validator


def _severity_to_level(severity: str) -> str:
    if severity in {"critical", "high"}:
        return "error"
    if severity == "medium":
        return "warning"
    return "note"


def _finding_severity_value(finding: Finding) -> str:
    severity = finding.severity
    if hasattr(severity, "value"):
        return str(severity.value).lower()
    return str(severity).lower()


def _rule_descriptor(finding: Finding, validator_name: str) -> dict[str, Any]:
    rule_id = _rule_id(validator_name, finding.check_name)
    severity = _finding_severity_value(finding)
    descriptor: dict[str, Any] = {
        "id": rule_id,
        "name": finding.check_name,
        "shortDescription": {"text": finding.check_name},
        "defaultConfiguration": {"level": _severity_to_level(severity)},
    }
    if finding.message:
        descriptor["fullDescription"] = {"text": finding.message}
    return descriptor


def _positive_start_line(line_number: Any) -> int | None:
    """Return a SARIF-valid positive integer line number, else ``None``."""
    if isinstance(line_number, bool):
        return None
    if isinstance(line_number, str):
        stripped = line_number.strip()
        if not stripped:
            return None
        try:
            line_number = int(stripped)
        except ValueError:
            return None
    if isinstance(line_number, int) and line_number > 0:
        return line_number
    return None


def _resolve_artifact_path(file_path: str, scan_root: Path | None) -> Path:
    """Resolve a validator file path against the scanned skill directory."""
    normalized = file_path.replace("\\", "/")
    path = Path(normalized)
    if scan_root is not None and not path.is_absolute():
        return (scan_root / path).resolve()
    if path.is_absolute():
        return path.resolve()
    return path


def _normalize_artifact_uri(
    file_path: str,
    workspace_root: Path | None,
    scan_root: Path | None = None,
) -> str:
    """Return a repository-relative, URI-encoded artifact path for SARIF."""
    path = _resolve_artifact_path(file_path, scan_root)
    if workspace_root is not None:
        try:
            # Windows paths can be rooted (``\\workspace\\...``) without a
            # drive, in which case ``is_absolute()`` is false until resolved.
            if path.is_absolute() or path.root:
                relative = path.resolve().relative_to(workspace_root.resolve())
                normalized = relative.as_posix()
            else:
                normalized = path.as_posix()
        except ValueError:
            normalized = path.as_posix()
    elif path.is_absolute():
        normalized = path.as_posix()
    else:
        normalized = path.as_posix()
    return quote(normalized, safe="/:@%")


def _artifact_location(uri: str) -> dict[str, Any]:
    location: dict[str, Any] = {"uri": uri}
    if uri and not uri.startswith("/") and ":" not in uri.split("/", 1)[0]:
        # Repository-relative: resolve against the checkout root (GitHub's convention).
        location["uriBaseId"] = _SRCROOT
    return location


def _physical_location(
    finding: Finding,
    artifact_path: str,
    workspace_root: Path | None,
    scan_root: Path | None = None,
) -> dict[str, Any] | None:
    if not artifact_path:
        return None
    location: dict[str, Any] = {
        "artifactLocation": _artifact_location(_normalize_artifact_uri(artifact_path, workspace_root, scan_root)),
    }
    start_line = _positive_start_line(finding.line_number)
    if start_line is not None:
        region: dict[str, Any] = {"startLine": start_line}
        if finding.line_content:
            region["snippet"] = {"text": finding.line_content}
        location["region"] = region
    return {"physicalLocation": location}


def _plugin_component(artifact_path: str, components: ComponentIndex, scan_root: Path | None) -> dict[str, str] | None:
    """Return the inventory component a finding's file belongs to.

    The plugin block records its root as typed (``.`` for ``validate .``),
    which an absolute finding path cannot match. The scan root is the resolved
    plugin root, so such a path is looked up relative to it instead.
    """
    component = components.component(artifact_path)
    if component is not None or scan_root is None or not Path(artifact_path).is_absolute():
        return component
    try:
        relative = _resolve_artifact_path(artifact_path, scan_root).relative_to(scan_root.resolve())
    except ValueError:
        return None
    return components.component(relative.as_posix())


def _result_from_finding(
    finding: Finding,
    validator_name: str,
    workspace_root: Path | None,
    scan_root: Path | None = None,
    components: ComponentIndex | None = None,
) -> dict[str, Any]:
    """Convert one finding; *components* indexes the plugin inventory of a plugin run."""
    severity = _finding_severity_value(finding)
    result: dict[str, Any] = {
        "ruleId": _rule_id(validator_name, finding.check_name),
        "level": _severity_to_level(severity),
        "message": {"text": finding.message},
    }
    if finding.suggestion:
        result["message"]["markdown"] = f"{finding.message}\n\n**Suggestion:** {finding.suggestion}"
    # Resolve the file the finding points at once, so its location and its plugin component agree. A bundled
    # skill's "[skill] " label is not part of the path: kept, it became "%5Bskill%5D%20/abs/path", which points
    # nowhere and leaks the local path.
    artifact_path = (components or _NO_INVENTORY).artifact_path(finding.file_path) if finding.file_path else ""
    location = _physical_location(finding, artifact_path, workspace_root, scan_root)
    if location is not None:
        result["locations"] = [location]
    properties: dict[str, Any] = {
        "category": finding.category,
        "validator": validator_name,
        "checkName": finding.check_name,
        "severity": severity,
    }
    if finding.metadata:
        properties["metadata"] = finding.metadata
    plugin_component = _plugin_component(artifact_path, components, scan_root) if components is not None else None
    if plugin_component:
        properties["pluginComponent"] = plugin_component
    result["properties"] = properties
    return result


def _plugin_run_properties(plugin: dict[str, Any], results: list[ValidationResult]) -> dict[str, Any]:
    """Summarize Tier 1 (and, when present, Tier 3) plugin context for a SARIF run."""
    properties: dict[str, Any] = {
        "name": plugin.get("name"),
        "manifestType": plugin.get("manifest_type"),
        "pluginMode": plugin.get("plugin_mode"),
    }
    for source_key, target_key in (
        ("declared_dependencies", "declaredDependencies"),
        ("dependency_status_counts", "dependencyStatusCounts"),
    ):
        if isinstance(plugin.get(source_key), dict):
            properties[target_key] = plugin[source_key]
    inventory = inventory_view(plugin.get("component_inventory"))
    if inventory is not None:
        properties["componentCounts"] = {row["type"]: row["count"] for row in inventory["counts"]}
        properties["unsupportedTypesPresent"] = inventory["unsupported_types"]
    mcp = plugin.get("mcp") if isinstance(plugin.get("mcp"), dict) else {}
    pinning = pinning_view(mcp.get("pinning"))
    if pinning is not None:
        properties["mcpPinning"] = {
            key: pinning[key] for key in ("total", "pinned", "unpinned", "not_applicable", "ratio")
        }
    cost = plugin.get("context_cost") if isinstance(plugin.get("context_cost"), dict) else {}
    if cost:
        properties["contextCost"] = {
            "method": cost.get("method"),
            "estimator": cost.get("estimator"),
            "alwaysOnTokens": cost.get("always_on_tokens"),
            "onDemandTokens": cost.get("on_demand_tokens"),
        }
    for source_key, target_key in (
        ("catalog_skill_similarity", "catalogSkillSimilarity"),
        ("inter_plugin_similarity", "interPluginSimilarity"),
    ):
        similarity = plugin.get(source_key) if isinstance(plugin.get(source_key), dict) else {}
        if similarity:
            matches = similarity.get("matches")
            properties[target_key] = {
                "status": similarity.get("status"),
                "catalogEntries": similarity.get("catalog_entries"),
                "matches": len(matches) if isinstance(matches, list) else 0,
                "advisory": True,
            }
    for result in results:
        payload = result.metadata.get("agent_eval") if isinstance(result.metadata, dict) else None
        view = tier3_plugin_view(payload)
        if view is None:
            continue
        properties["evaluationIncomplete"] = view["partial"]
        if view["coverage"] is not None:
            # Staged is not evaluated: report the not-staged count, plus the staged
            # components no plugin trial exercised when trials recorded activation.
            properties["componentsNotStaged"] = view["coverage"]["not_staged"]
            properties["componentsStagedNotObserved"] = view["coverage"]["staged_not_observed"]
        break
    return json_safe({key: value for key, value in properties.items() if value is not None})


def _collect_incomplete_scans(results: list[ValidationResult]) -> list[str]:
    scans: list[str] = []
    for result in results:
        for tool in result.incomplete_scans:
            if tool not in scans:
                scans.append(tool)
    return scans


def _tier3_notifications(results: list[ValidationResult]) -> list[dict[str, Any]]:
    """Return a SARIF notification for each Tier 3 run that produced no complete scored run.

    A ``failed`` run is an error. A run that was skipped or is INCOMPLETE is a
    warning: Tier 3 was requested but gave no full result. That covers an
    engine crash or timeout, which the CLI records as an advisory skip. A
    ``not_applicable`` run (advisory, no task source) already has its own
    result, so it adds no notification.
    """
    notifications: list[dict[str, Any]] = []
    for result in results:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        payload = metadata.get("agent_eval")
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        status = str(payload.get("execution_status") or summary.get("execution_status") or "").lower()
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        if status == "failed":
            errors = [str(error) for error in payload.get("execution_errors") or result.errors if str(error).strip()]
            reason = errors[0] if errors else "Tier 3 evaluation did not produce a complete scored run"
            notifications.append(
                {
                    "descriptor": {"id": "tier3/execution-failed"},
                    "level": "error",
                    "message": {"text": f"Tier 3 live evaluation did not complete: {reason}"},
                }
            )
        elif status == "skipped" and provenance.get("reason") == "skipped":
            message = str(provenance.get("message") or "").strip()
            notifications.append(
                {
                    "descriptor": {"id": "tier3/skipped"},
                    "level": "warning",
                    "message": {"text": message or "Tier 3 live evaluation produced no scored run."},
                }
            )
        elif metadata.get("execution_status") == "skipped":
            reason = str(metadata.get("skip_reason") or "").strip()
            notifications.append(
                {
                    "descriptor": {"id": "tier3/incomplete"},
                    "level": "warning",
                    "message": {"text": reason or "Tier 3 live evaluation is INCOMPLETE."},
                }
            )
    return notifications


def _build_invocation(results: list[ValidationResult]) -> dict[str, Any]:
    incomplete_scans = _collect_incomplete_scans(results)
    notifications = [
        {
            "descriptor": {"id": f"incomplete/{tool}"},
            "level": "error",
            "message": {"text": f"{tool} scan did not complete"},
        }
        for tool in incomplete_scans
    ]
    notifications.extend(_tier3_notifications(results))
    invocation: dict[str, Any] = {
        "executionSuccessful": not any(notification["level"] == "error" for notification in notifications),
        "endTimeUtc": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if notifications:
        invocation["toolExecutionNotifications"] = notifications
    return invocation


def _canary_results(
    results: list[ValidationResult],
    plugin: dict[str, Any] | None,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> list[dict[str, Any]]:
    """One SARIF result per plugin-attributable canary leak (the plugin arm leaked more often than the baseline)."""
    location = None
    root = (plugin or {}).get("root")
    manifest = (plugin or {}).get("manifest_filename")
    if isinstance(root, str) and root and isinstance(manifest, str) and manifest:
        uri = _normalize_artifact_uri(f"{root.rstrip('/')}/{manifest}", workspace_root, scan_root)
        location = {"physicalLocation": {"artifactLocation": _artifact_location(uri)}}
    sarif_results: list[dict[str, Any]] = []
    for result in results:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        for entry in plugin_attributable_leaks(metadata.get("agent_eval")):
            plugin_row = next((row for row in entry["rows"] if row["arm"] in PLUGIN_ARMS), {})
            message = (
                f"{entry['verdict']} ({entry['scope']}: leaked in {plugin_row.get('leaked', 0)} of "
                f"{plugin_row.get('trials') if plugin_row.get('trials') is not None else 'unknown'} trial(s); "
                f"sinks: {plugin_row.get('sinks', 'none')})."
            )
            sarif_result: dict[str, Any] = {
                "ruleId": CANARY_RULE_ID,
                "level": "error",
                "message": {"text": message},
                "properties": {
                    "category": "SECURITY",
                    "validator": result.validator_name or "AGENT_EVAL",
                    "checkName": "canary_exfiltration",
                    "severity": "critical",
                    "agent": entry["scope"],
                },
            }
            if location is not None:
                sarif_result["locations"] = [location]
            sarif_results.append(sarif_result)
    return sarif_results


def merge_catalog_sarif_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-skill SARIF payloads into one upload-ready document.

    GitHub Code Scanning rejects SARIF with more than 20 runs, so child reports
    are folded into a single run with combined rules and results.
    """
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    driver: dict[str, Any] = {
        "name": "SkillEvaluator",
        "informationUri": _TOOL_URI,
    }

    for payload in documents:
        for run in payload.get("runs", []):
            run_driver = run.get("tool", {}).get("driver", {})
            if run_driver.get("name"):
                driver["name"] = run_driver["name"]
            if run_driver.get("informationUri"):
                driver["informationUri"] = run_driver["informationUri"]
            if run_driver.get("version"):
                driver["version"] = run_driver["version"]
            for rule in run_driver.get("rules", []):
                rules[rule["id"]] = rule
            results.extend(run.get("results", []))
            invocations.extend(run.get("invocations", []))

    merged_run: dict[str, Any] = {
        "tool": {
            "driver": {
                **driver,
                "rules": sorted(rules.values(), key=lambda item: item["id"]),
            }
        },
        "results": results,
        "automationDetails": {"id": "/skillevaluator/catalog"},
    }
    if invocations:
        merged_run["invocations"] = invocations
    return {
        "$schema": _SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [merged_run],
    }


class SARIFReporter(ReporterBase):
    """SARIF 2.1.0 export for security scanning integrations."""

    def __init__(
        self,
        *,
        indent: int | None = 2,
        include_timestamp: bool = True,
        workspace_root: Path | None = None,
        scan_root: Path | None = None,
    ) -> None:
        self.indent = indent
        self.include_timestamp = include_timestamp
        self.workspace_root = workspace_root
        self.scan_root = scan_root

    @property
    def name(self) -> str:
        return "sarif"

    @property
    def description(self) -> str:
        return "SARIF 2.1.0 for GitHub Code Scanning and SARIF consumers"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        rules: dict[str, dict[str, Any]] = {}
        sarif_results: list[dict[str, Any]] = []
        workspace_root = self.workspace_root
        scan_root = self.scan_root

        plugin = self._plugin_block_from_results(results)
        components = ComponentIndex(plugin) if plugin is not None else None
        for result in results:
            validator_name = result.validator_name or "UNKNOWN"
            for finding in result.findings:
                rule = _rule_descriptor(finding, validator_name)
                rules[rule["id"]] = rule
                sarif_results.append(
                    _result_from_finding(finding, validator_name, workspace_root, scan_root, components)
                )
        canary_results = _canary_results(results, plugin, workspace_root, scan_root)
        if canary_results:
            rules[CANARY_RULE_ID] = _CANARY_RULE
            sarif_results.extend(canary_results)

        run: dict[str, Any] = {
            "tool": {
                "driver": {
                    "name": "SkillEvaluator",
                    "informationUri": _TOOL_URI,
                    "version": __version__,
                    "rules": sorted(rules.values(), key=lambda item: item["id"]),
                }
            },
            "results": sarif_results,
        }
        if self.include_timestamp or _collect_incomplete_scans(results) or _tier3_notifications(results):
            run["invocations"] = [_build_invocation(results)]
        if plugin is not None:
            run["properties"] = {"plugin": _plugin_run_properties(plugin, results)}

        document: dict[str, Any] = {
            "$schema": _SARIF_SCHEMA,
            "version": "2.1.0",
            "runs": [run],
        }
        return json.dumps(document, indent=self.indent, default=str, allow_nan=False)

    def get_file_extension(self) -> str:
        return ".sarif.json"
