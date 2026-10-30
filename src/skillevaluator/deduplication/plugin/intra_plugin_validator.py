# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline duplicate dependency-reference checks for the private bundle workflow."""

from pathlib import Path

from skillevaluator.constants import PLUGIN_MANIFEST_TYPE
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest
from skillevaluator.plugin_refs import find_duplicate_refs
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_yaml
from skillevaluator.validators.base import ValidatorBase


class IntraPluginValidator(ValidatorBase):
    @property
    def name(self):
        return "Plugin Dependency Deduplication"

    @property
    def description(self):
        return "Detect duplicate skill/rule dependency references within a bundle manifest"

    def validate(self, plugin_root: Path) -> ValidationResult:
        result = ValidationResult(validator_name=self.name, validator_description=self.description)
        result.metadata["advisory_tier2"] = True
        try:
            located = locate_plugin_manifest(plugin_root)
            if located is None or located.manifest_type != PLUGIN_MANIFEST_TYPE:
                result.add_success("plugin_dep_dedup", "Dependency references do not apply to this manifest")
                return result
            try:
                data = load_bounded_yaml(located.read_text())
            except StructuredDataError:
                data = None
        except PluginManifestPathError as exc:
            result.add_finding(
                Finding(
                    "PLUGIN_SECURITY", Severity.HIGH, "unsafe_plugin_manifest", str(exc), file_path=exc.relative_path
                )
            )
            result.metadata["security_failure"] = True
            return result
        if not isinstance(data, dict):
            result.metadata.update(scan_status="skipped", optional_skip=True)
            result.add_warning("Manifest could not be parsed; dependency-reference deduplication did not run")
            return result
        for section, check in (("skills", "duplicate_skill_ref"), ("rules", "duplicate_rule_ref")):
            value = data.get(section)
            refs = value.get("refs") if isinstance(value, dict) else None
            for duplicate in find_duplicate_refs(refs if isinstance(refs, list) else None):
                result.add_finding(
                    Finding(
                        "PLUGIN_DEDUP",
                        Severity.MEDIUM,
                        check,
                        f"Dependency '{duplicate.canonical_id}' occurs {len(duplicate.occurrences)} times",
                        file_path=located.manifest_filename,
                        suggestion=f"Keep each {section}.refs dependency once.",
                        metadata={"canonical_id": duplicate.canonical_id, "occurrences": len(duplicate.occurrences)},
                    )
                )
        if not result.findings:
            result.add_success("plugin_dep_dedup", "No duplicate dependency references found")
        return result
