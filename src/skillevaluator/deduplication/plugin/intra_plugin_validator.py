# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory duplicate-reference validation within one public plugin manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skillevaluator.constants import PLUGIN_MANIFEST_FILES
from skillevaluator.deduplication.plugin.ref_utils import find_duplicate_refs
from skillevaluator.models.result import Finding, Severity, ValidationResult
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
        manifest_path = self._locate_manifest(plugin_root)
        if manifest_path is None:
            result.add_success("plugin_dep_dedup", "No bundle-reference manifest; check not applicable")
            return result
        if plugin_root.is_dir() and not self._is_within(manifest_path, plugin_root):
            return self._skip(result, "Plugin manifest resolves outside the plugin root; refusing to read it.")
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
    def _locate_manifest(plugin_root: Path) -> Path | None:
        if plugin_root.is_file():
            return plugin_root if plugin_root.name in PLUGIN_MANIFEST_FILES else None
        return next((plugin_root / name for name in PLUGIN_MANIFEST_FILES if (plugin_root / name).exists()), None)

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        try:
            return candidate.resolve().is_relative_to(root.resolve())
        except OSError:
            return False

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
