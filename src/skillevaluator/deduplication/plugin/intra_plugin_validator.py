# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check A: intra-plugin manifest dependency-reference deduplication.

Flags duplicate ``skills.refs`` / ``rules.refs`` entries inside a
bundle-reference plugin manifest (``agent_plugin.yaml``/``.yml``). Findings are
advisory (MEDIUM): Tier 2 for plugins warns but never fails the build. ``mcp``
is intentionally untouched -- duplicate MCP ``(name, provider)`` pairs are
already rejected as a blocking schema error by
:meth:`skillevaluator.models.plugin.PluginManifest.check_dependencies_and_mcp`.

This module is deliberately offline: it imports only stdlib + ``yaml`` + the
result/base models, so it runs on a base install without the ``tier2`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skillevaluator.constants import PLUGIN_MANIFEST_TYPE
from skillevaluator.deduplication.plugin.ref_utils import find_duplicate_refs
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_manifest import PluginManifestLocation, PluginManifestPathError, locate_plugin_manifest
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_yaml
from skillevaluator.validators.base import ValidatorBase

# Advisory severity for every Check A finding. Tier 2 plugin dedup is a warning,
# not a gate, so duplicate references never fail the build.
_DUPLICATE_REF_SEVERITY = Severity.MEDIUM


class IntraPluginValidator(ValidatorBase):
    """Detect duplicate skill/rule dependency references within one plugin manifest."""

    @property
    def name(self) -> str:
        return "Plugin Dependency Deduplication"

    @property
    def description(self) -> str:
        return "Detect duplicate skill/rule dependency references within a plugin manifest"

    def validate(self, plugin_root: Path) -> ValidationResult:
        """Run dependency-reference dedup for a bundle-reference plugin."""
        result = ValidationResult(
            validator_name=self.name,
            validator_description=self.description,
        )
        result.metadata["advisory_tier2"] = True

        try:
            located = locate_plugin_manifest(plugin_root)
        except PluginManifestPathError as exc:
            self._mark_security_failure(result, str(exc))
            return result
        if located is None or located.manifest_type != PLUGIN_MANIFEST_TYPE:
            # Contained (.claude-plugin/plugin.json) or manifest-less plugins
            # expose no parsed refs, so Check A simply does not apply.
            result.add_success(
                "plugin_dep_dedup",
                "No bundle-reference manifest; dependency-reference dedup not applicable",
            )
            return result

        manifest_path = located.path

        try:
            data = self._load_manifest(located)
        except PluginManifestPathError as exc:
            self._mark_security_failure(result, str(exc))
            return result
        if data is None:
            # Unparseable YAML (or a manifest that is not a mapping) means Check A
            # never ran. Record an advisory optional skip -- not a bare pass -- so
            # CLI/Markdown/HTML never show a fake "no duplicate references" green.
            self._mark_skipped(
                result,
                "Plugin manifest could not be parsed as a YAML mapping; skipping dependency-reference dedup.",
            )
            return result

        self._check_section(data, "skills", "duplicate_skill_ref", manifest_path, result)
        self._check_section(data, "rules", "duplicate_rule_ref", manifest_path, result)

        if not result.findings:
            result.add_success(
                "plugin_dep_dedup",
                "No duplicate skill/rule dependency references found",
            )
        return result

    @staticmethod
    def _mark_skipped(result: ValidationResult, reason: str) -> None:
        """Record Check A as an advisory optional skip (never a silent pass).

        Used when the manifest cannot be parsed as a mapping. Marking
        ``execution_status="skipped"`` keeps every
        reporter consistent -- CLI/Markdown/HTML all show a non-blocking skip
        instead of a false-green "no duplicate references" pass -- and
        ``optional=True`` keeps it from counting as an incomplete requested run.
        """
        result.add_warning(reason)
        result.metadata.update(
            {
                "execution_status": "skipped",
                "skip_reason": reason,
                "optional": True,
            }
        )

    @staticmethod
    def _mark_security_failure(result: ValidationResult, reason: str) -> None:
        """Keep unsafe input visible as a failed, non-optional per-check result."""
        safe_reason = f"Unsafe plugin manifest refused: {reason}"
        result.add_finding(
            Finding(
                category="PLUGIN_SECURITY",
                severity=Severity.HIGH,
                check_name="unsafe_plugin_manifest",
                message=safe_reason,
                file_path="<plugin-manifest>",
                suggestion="Replace links/hardlinks/special manifests with one regular file inside the plugin root.",
            )
        )
        result.metadata.update(
            {
                "security_failure": True,
                "execution_status": "failed",
                "optional": False,
            }
        )

    @staticmethod
    def _load_manifest(location: PluginManifestLocation) -> dict | None:
        """Parse the manifest YAML into a dict, or ``None`` on any problem."""
        try:
            raw = location.read_text()
            data: Any = load_bounded_yaml(raw)
        except StructuredDataError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _check_section(
        data: dict,
        section: str,
        check_name: str,
        manifest_path: Path,
        result: ValidationResult,
    ) -> None:
        """Emit an advisory finding per duplicate ref in ``<section>.refs``."""
        section_obj = data.get(section)
        if not isinstance(section_obj, dict):
            return
        refs = section_obj.get("refs")
        if not isinstance(refs, list):
            return

        for group in find_duplicate_refs(refs):
            count = len(group.occurrences)
            result.add_finding(
                Finding(
                    category="PLUGIN_DEDUP",
                    severity=_DUPLICATE_REF_SEVERITY,
                    check_name=check_name,
                    message=(
                        f"Duplicate {section} dependency reference '{group.canonical_id}' "
                        f"is declared {count} times in the plugin manifest."
                    ),
                    file_path=str(manifest_path),
                    suggestion=(
                        f"Remove the redundant '{section}.refs' entries so each referenced "
                        "resource is declared once (string and selector forms that point at "
                        "the same resource count as duplicates)."
                    ),
                    metadata={"canonical_id": group.canonical_id, "occurrences": count},
                )
            )
