# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin manifest and bundled-skill validation.

Two plugin models are recognized:

* **Bundle-reference** (``agent_plugin.yaml`` / ``agent_plugin.yml``), validated
  in full against :class:`~skillevaluator.models.plugin.PluginManifest`.
* **Contained** (``.claude-plugin/plugin.json``), shallowly validated as a JSON
  object with a non-empty ``name``. Full Claude-plugin schema validation is
  intentionally deferred.

For either model, skills under ``<plugin-root>/skills/`` are discovered and
validated with :class:`~skillevaluator.validators.schema.SchemaValidator`.
Reporting metadata identifies the selected manifest model and summarizes
declared dependencies without resolving or fetching them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from skillevaluator.constants import (
    NAME_MAX_LENGTH,
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MODE,
    PLUGIN_MANIFEST_FILES,
    PLUGIN_MODE,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.plugin import PluginManifest
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_manifest import PluginManifestLocation, PluginManifestPathError, locate_plugin_manifest
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.structured_data import (
    StructuredDataLimitError,
    StructuredDataSyntaxError,
    load_bounded_json,
    load_bounded_yaml,
    require_bounded_string,
)
from skillevaluator.validators.base import ValidatorBase
from skillevaluator.validators.mcp_static import validate_contained_mcp_servers

if TYPE_CHECKING:
    from skillevaluator.validators.policy import ValidationPolicy

logger = get_logger(__name__)
MAX_PLUGIN_SCHEMA_FINDINGS = 100


class PluginSchemaValidator(ValidatorBase):
    """Validate a plugin manifest and any skills bundled by the plugin."""

    def __init__(self, policy: ValidationPolicy | None = None) -> None:
        self.policy = policy

    @property
    def name(self) -> str:
        return "Plugin Schema & Bundle References"

    @property
    def description(self) -> str:
        return "Validate the plugin manifest and any bundled skills"

    def validate(self, path: Path) -> ValidationResult:
        """Validate the plugin manifest located at or under *path*."""
        result = ValidationResult()

        try:
            located = locate_plugin_manifest(path)
        except PluginManifestPathError as exc:
            result.metadata["security_failure"] = True
            selected_manifests = {
                *PLUGIN_MANIFEST_FILES,
                f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}",
            }
            if exc.relative_path in selected_manifests:
                check_name = "manifest_outside_root"
                message = str(exc)
                suggestion = "Replace the manifest symlink with a regular file contained by the plugin root."
            else:
                check_name = "unsafe_plugin_filesystem"
                message = f"Unsafe bundled plugin filesystem path '{exc.relative_path}': {exc}"
                suggestion = (
                    "Replace linked, reparse-point, or special bundled plugin paths with regular files and "
                    "directories contained by the plugin root."
                )
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name=check_name,
                    message=message,
                    file_path=str(path),
                    suggestion=suggestion,
                )
            )
            return result
        if located is None:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_missing",
                    message=(
                        "No plugin manifest found. Expected one of "
                        f"{', '.join(PLUGIN_MANIFEST_FILES)} or "
                        f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE} "
                        "at the plugin root."
                    ),
                    file_path=str(path),
                    suggestion=(
                        "Add an agent_plugin.yaml (or agent_plugin.yml), or a "
                        f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}, "
                        "at the plugin root."
                    ),
                )
            )
            return result

        manifest_type = located.manifest_type
        root = located.root
        self._stamp_manifest_metadata(located.manifest_filename, root, manifest_type, result)

        if manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            self._validate_contained_manifest(located, result)
        else:
            self._validate_bundle_manifest(located, result)

        # A manifest error must not hide problems in skills bundled alongside it.
        self._validate_in_plugin_skills(root, result)
        return result

    @staticmethod
    def _stamp_manifest_metadata(
        manifest_filename: str,
        root: Path,
        manifest_type: str,
        result: ValidationResult,
    ) -> None:
        """Attach manifest metadata even when validation later fails."""
        if manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            mode = PLUGIN_CONTAINED_MODE
        else:
            mode = PLUGIN_MODE

        result.metadata["manifest_type"] = manifest_type
        result.metadata["plugin_mode"] = mode
        result.metadata["plugin"] = {
            "manifest_filename": manifest_filename,
            "root": str(root),
        }

    def _validate_bundle_manifest(self, location: PluginManifestLocation, result: ValidationResult) -> None:
        """Validate a bundle-reference manifest against ``PluginManifest``."""
        manifest_path = location.path
        data = self._load_yaml(location, result)
        if data is None:
            return

        try:
            manifest = PluginManifest(**data)
        except ValidationError as exc:
            self._add_validation_findings(exc, manifest_path, result)
            return

        result.add_message(f"Plugin name: {manifest.name}")
        result.add_message(f"Author: {manifest.author.email}")
        result.add_success(
            check_name="plugin_manifest",
            message=f"Plugin manifest '{manifest.name}' is valid",
        )
        plugin_meta = result.metadata.setdefault("plugin", {})
        plugin_meta["name"] = manifest.name
        plugin_meta["declared_dependencies"] = {
            "skills": len(manifest.skills.refs) if manifest.skills and manifest.skills.refs else 0,
            "rules": len(manifest.rules.refs) if manifest.rules and manifest.rules.refs else 0,
            "mcp": len(manifest.mcp) if manifest.mcp else 0,
        }

    def _load_yaml(self, location: PluginManifestLocation, result: ValidationResult) -> dict | None:
        """Parse manifest YAML, recording a finding on failure."""
        manifest_path = location.path
        try:
            raw = location.read_text()
        except PluginManifestPathError as exc:
            result.metadata["security_failure"] = True
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_unsafe",
                    message=f"Could not securely read plugin manifest: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Replace links/hardlinks/special manifests with one regular file inside the plugin root.",
                )
            )
            return None

        try:
            data = load_bounded_yaml(raw)
        except StructuredDataLimitError as exc:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_complexity_limit",
                    message=f"Plugin manifest exceeds structured-data complexity limits: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Reduce manifest nesting, collection sizes, or YAML aliases.",
                )
            )
            return None
        except StructuredDataSyntaxError as exc:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_invalid_yaml",
                    message=f"Plugin manifest is not valid YAML: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Fix the YAML syntax in the plugin manifest.",
                )
            )
            return None

        if not data or not isinstance(data, dict):
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_not_mapping",
                    message="Plugin manifest must be a non-empty YAML mapping.",
                    file_path=str(manifest_path),
                    suggestion="Populate the manifest with at least name, author, and a dependency.",
                )
            )
            return None

        return data

    def _add_validation_findings(
        self,
        exc: ValidationError,
        manifest_path: Path,
        result: ValidationResult,
    ) -> None:
        """Translate a Pydantic validation error into structured findings."""
        errors = exc.errors()
        for error in errors[:MAX_PLUGIN_SCHEMA_FINDINGS]:
            location = ".".join(str(loc) for loc in error["loc"]) or "<root>"
            error_type = error.get("type", "value_error")
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name=f"schema:{location}:{error_type}",
                    message=f"Field '{location}': {error['msg']}",
                    file_path=str(manifest_path),
                    suggestion=(
                        "Fix the plugin manifest to satisfy the bundle-reference contract "
                        "(allowed fields, required name + author.email, at least one "
                        "dependency, valid selectors/MCP entries)."
                    ),
                )
            )
        if len(errors) > MAX_PLUGIN_SCHEMA_FINDINGS:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="schema_errors_truncated",
                    message=(
                        f"Plugin schema produced {len(errors)} errors; only the first "
                        f"{MAX_PLUGIN_SCHEMA_FINDINGS} are reported."
                    ),
                    file_path=str(manifest_path),
                    suggestion="Fix the reported schema errors, then rerun validation.",
                    metadata={"actual": len(errors), "reported": MAX_PLUGIN_SCHEMA_FINDINGS},
                )
            )

    def _validate_contained_manifest(self, location: PluginManifestLocation, result: ValidationResult) -> None:
        """Shallow-validate a contained ``.claude-plugin/plugin.json`` file."""
        manifest_path = location.path
        try:
            raw = location.read_text(encoding="utf-8-sig")
        except PluginManifestPathError as exc:
            result.metadata["security_failure"] = True
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_unsafe",
                    message=f"Could not securely read plugin manifest: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Replace links/hardlinks/special manifests with one regular file inside the plugin root.",
                )
            )
            return

        try:
            data: Any = load_bounded_json(raw)
        except StructuredDataLimitError as exc:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_complexity_limit",
                    message=f"Contained plugin manifest exceeds structured-data complexity limits: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Reduce JSON nesting or collection sizes in plugin.json.",
                )
            )
            return
        except StructuredDataSyntaxError as exc:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_invalid_json",
                    message=f"Contained plugin manifest is not valid JSON: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Fix the JSON syntax in .claude-plugin/plugin.json.",
                )
            )
            return

        if not isinstance(data, dict) or not data:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_not_object",
                    message="Contained plugin manifest must be a non-empty JSON object.",
                    file_path=str(manifest_path),
                    suggestion="Populate plugin.json with at least a non-empty 'name'.",
                )
            )
            return

        try:
            name = require_bounded_string(data.get("name"), "Contained plugin name", max_chars=NAME_MAX_LENGTH)
        except ValueError:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="schema:name:missing",
                    message="Contained plugin manifest must define a non-empty 'name'.",
                    file_path=str(manifest_path),
                    suggestion="Add a 'name' string to .claude-plugin/plugin.json.",
                )
            )
            return

        # Runnable MCP servers declared in a contained manifest get blocking,
        # network-free static security validation (command/url/transport/env).
        # Provider MCP entries in agent_plugin.yaml are validated by the Pydantic
        # model instead; runnable entries only exist in the contained form.
        mcp_findings = validate_contained_mcp_servers(data.get("mcpServers"), str(manifest_path))
        for finding in mcp_findings[:MAX_PLUGIN_SCHEMA_FINDINGS]:
            result.add_finding(finding)
        if len(mcp_findings) > MAX_PLUGIN_SCHEMA_FINDINGS:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="schema_errors_truncated",
                    message=(
                        f"Contained MCP validation produced {len(mcp_findings)} findings; only the first "
                        f"{MAX_PLUGIN_SCHEMA_FINDINGS} are reported."
                    ),
                    file_path=str(manifest_path),
                    suggestion="Fix the reported MCP declaration errors, then rerun validation.",
                    metadata={"actual": len(mcp_findings), "reported": MAX_PLUGIN_SCHEMA_FINDINGS},
                )
            )

        result.add_message(f"Plugin name: {name}")
        if not mcp_findings:
            result.add_success(
                check_name="plugin_manifest",
                message=(
                    f"Contained plugin manifest '{name}' is valid (name present; full Claude-plugin schema deferred)"
                ),
            )
        plugin_meta = result.metadata.setdefault("plugin", {})
        plugin_meta["name"] = name
        declared = {key: len(value) for key, value in data.items() if isinstance(value, list)}
        if declared:
            plugin_meta["declared_dependencies"] = declared

    def _validate_in_plugin_skills(self, root: Path, result: ValidationResult) -> None:
        """Validate skills bundled under ``<root>/skills/``."""
        skills_dir = root / "skills"
        from skillevaluator.utils.helpers import find_bundled_plugin_skill_manifests
        from skillevaluator.validators.schema import SchemaValidator

        try:
            skill_manifests = find_bundled_plugin_skill_manifests(root)
        except ValueError as exc:
            result.metadata["security_failure"] = True
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="bundled_skill_path_unsafe",
                    message=f"Could not securely discover bundled skills: {exc}",
                    file_path=str(skills_dir),
                    suggestion="Replace linked/junction bundled skill directories with regular contained directories.",
                )
            )
            return
        if not skill_manifests:
            return

        skill_names = [manifest.relative_path.parent.as_posix() for manifest in skill_manifests]
        skill_dirs = [skills_dir / manifest.relative_path.parent for manifest in skill_manifests]
        plugin_meta = result.metadata.setdefault("plugin", {})
        plugin_meta["in_plugin_skills"] = len(skill_dirs)
        plugin_meta["bundled_skills"] = skill_names
        validator = SchemaValidator(policy=self.policy)

        for skill_dir, skill_name, manifest in zip(skill_dirs, skill_names, skill_manifests, strict=True):
            try:
                skill_result = validator.validate_secure_manifest(skill_dir, manifest)
            except SecurePathError as exc:
                result.metadata["security_failure"] = True
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="bundled_skill_path_unsafe",
                        message=f"Bundled skill '{skill_name}' changed or became unsafe after discovery: {exc}",
                        file_path=f"[{skill_name}] {skill_dir}",
                        suggestion="Replace linked, hard-linked, or special manifests with regular contained files.",
                    )
                )
                continue
            except Exception as exc:
                logger.warning("In-plugin skill validation failed for %s: %s", skill_dir, exc)
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="in_plugin_skill_error",
                        message=f"Could not validate bundled skill '{skill_name}': {exc}",
                        file_path=f"[{skill_name}] {skill_dir}",
                        suggestion="Inspect the bundled skill directory; it may be malformed.",
                    )
                )
                continue

            if skill_result.passed:
                result.merge_with_prefix(skill_result, skill_name)
                result.add_success(
                    check_name=skill_name,
                    message=f"Bundled skill '{skill_name}' passed skill schema validation",
                )
            else:
                result.merge_with_prefix(skill_result, skill_name)
