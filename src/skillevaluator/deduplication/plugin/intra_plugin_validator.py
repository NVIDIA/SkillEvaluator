# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory duplicate-reference validation within one public plugin manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skillevaluator.constants import PLUGIN_MANIFEST_TYPE
from skillevaluator.deduplication.plugin.ref_utils import find_duplicate_refs
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest
from skillevaluator.validators.base import ValidatorBase


class IntraPluginValidator(ValidatorBase):
    """Check A: detect duplicate skill/rule references without network access."""

    @property
    def name(self) -> str:
        return "Plugin Dependency Deduplication"

    @property
    def description(self) -> str:
        return "Detect duplicate skill/rule dependency references within a plugin manifest"

    def validate(self, plugin_root: Path) -> ValidationResult:
        result = ValidationResult(validator_name=self.name, validator_description=self.description)
        try:
            located = locate_plugin_manifest(plugin_root)
        except PluginManifestPathError:
            return self._skip(
                result,
                "Plugin manifest resolves outside the plugin root; refusing to read it.",
            )
        if located is None or located.manifest_type != PLUGIN_MANIFEST_TYPE:
            result.add_success("plugin_dep_dedup", "No bundle-reference manifest; check not applicable")
            return result
        manifest_path = located.path
        data = self._load_manifest(manifest_path)
        if data is None:
            return self._skip(result, "Plugin manifest could not be parsed as a YAML mapping.")

        for section, check_name in (("skills", "duplicate_skill_ref"), ("rules", "duplicate_rule_ref")):
            value = data.get(section)
            refs = value.get("refs") if isinstance(value, dict) else None
            if not isinstance(refs, list):
                continue
            for group in find_duplicate_refs(refs):
                result.add_finding(
                    Finding(
                        category="PLUGIN_DEDUP",
                        severity=Severity.MEDIUM,
                        check_name=check_name,
                        message=(
                            f"Duplicate {section} dependency reference '{group.canonical_id}' "
                            f"is declared {len(group.occurrences)} times."
                        ),
                        file_path=str(manifest_path),
                        suggestion=f"Declare each {section} dependency exactly once.",
                        metadata={"canonical_id": group.canonical_id, "occurrences": len(group.occurrences)},
                    )
                )
        if not result.findings:
            result.add_success("plugin_dep_dedup", "No duplicate skill/rule dependency references found")
        result.metadata["advisory_tier2"] = True
        return result

    @staticmethod
    def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _skip(result: ValidationResult, reason: str) -> ValidationResult:
        result.add_warning(reason)
        result.metadata.update(
            {"execution_status": "skipped", "skip_reason": reason, "optional": True, "advisory_tier2": True}
        )
        return result
