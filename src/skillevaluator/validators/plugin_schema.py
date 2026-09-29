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
declared dependencies. For a valid bundle-reference manifest every declared
skill/rule ref is classified offline (never fetched) by
:mod:`skillevaluator.plugin_dependencies`; a same-repository ref whose target
does not exist is the only blocking outcome (``plugin_dependency_missing``).

Every declared and packaged component is also inventoried statically
(:mod:`skillevaluator.plugin_components`): MCP servers from every documented
``mcpServers`` form and the root ``.mcp.json`` get blocking static checks,
declared component paths must exist inside the plugin root, shipped settings
and ``.env`` files are checked, and ``metadata['plugin']`` gains
``component_inventory``, ``mcp``, and ``context_cost``.
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
from skillevaluator.validators.mcp_static import CATEGORY as MCP_CATEGORY

if TYPE_CHECKING:
    from skillevaluator.validators.policy import ValidationPolicy

logger = get_logger(__name__)
MAX_PLUGIN_SCHEMA_FINDINGS = 100


class PluginSchemaValidator(ValidatorBase):
    """Validate a plugin manifest and any skills bundled by the plugin."""

    def __init__(self, policy: ValidationPolicy | None = None, repo_root: Path | None = None) -> None:
        self.policy = policy
        # Optional ``--repo-root``: repository root for same-repository ref resolution.
        self.repo_root = repo_root

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

        validated_manifest: dict[str, Any] | None = None
        contained_data: dict[str, Any] | None = None
        if manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            contained_data = self._validate_contained_manifest(located, result)
        else:
            validated_manifest = self._validate_bundle_manifest(located, result)

        # A manifest error must not hide problems in skills bundled alongside it.
        self._validate_in_plugin_skills(root, result)
        if validated_manifest is not None:
            self._resolve_dependencies(located, validated_manifest, result)
        self._inventory_components(located, contained_data, result)
        return result

    def _inventory_components(
        self,
        location: PluginManifestLocation,
        contained_data: dict[str, Any] | None,
        result: ValidationResult,
    ) -> None:
        """Inventory declared + packaged components and run the static component checks.

        Adds the MCP (all ``mcpServers`` forms and the root ``.mcp.json``),
        component-path, shipped-settings, and ``.env`` findings, then stamps
        ``component_inventory`` / ``mcp`` / ``context_cost`` into
        ``metadata['plugin']``. Plugin-controlled paths are classified without
        following links and read through bounded, root-anchored reads.
        """
        # Imported lazily: plugin_components imports validators.mcp_static, and the
        # validators package imports this module at package-import time.
        from skillevaluator.plugin_components import attribute_findings, build_plugin_inventory, manifest_rel_for

        contained = location.manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE
        manifest = contained_data if contained else self._bundle_manifest_data(location)
        root = location.secure_file.root
        allowed_hosts = self.policy.mcp_allowed_private_hosts if self.policy is not None else ()
        inventory = build_plugin_inventory(
            root,
            manifest,
            contained=contained,
            manifest_rel=manifest_rel_for(location.path, location.root),
            allowed_private_hosts=allowed_hosts,
        )
        findings = inventory.findings
        for finding in findings[:MAX_PLUGIN_SCHEMA_FINDINGS]:
            result.add_finding(finding)
        if len(findings) > MAX_PLUGIN_SCHEMA_FINDINGS:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="schema_errors_truncated",
                    message=(
                        f"Plugin component validation produced {len(findings)} findings; only the first "
                        f"{MAX_PLUGIN_SCHEMA_FINDINGS} are reported."
                    ),
                    file_path=str(location.path),
                    suggestion="Fix the reported plugin component and MCP declaration errors, then rerun validation.",
                    metadata={"actual": len(findings), "reported": MAX_PLUGIN_SCHEMA_FINDINGS},
                )
            )
        if contained and contained_data is not None:
            blocking_mcp = [
                finding
                for finding in findings
                if finding.category == MCP_CATEGORY and finding.severity in (Severity.CRITICAL, Severity.HIGH)
            ]
            if not blocking_mcp:
                name = result.metadata.get("plugin", {}).get("name", "")
                result.add_success(
                    check_name="plugin_manifest",
                    message=(
                        f"Contained plugin manifest '{name}' is valid (name present; full Claude-plugin schema "
                        "deferred)"
                    ),
                )
        attribute_findings(inventory.components, result.findings, root)
        result.metadata.setdefault("plugin", {}).update(inventory.metadata())

    @staticmethod
    def _bundle_manifest_data(location: PluginManifestLocation) -> dict[str, Any] | None:
        """Best-effort bounded re-parse of ``agent_plugin.yaml`` for the inventory."""
        try:
            data = load_bounded_yaml(location.read_text())
        except (PluginManifestPathError, StructuredDataLimitError, StructuredDataSyntaxError, ValueError):
            return None
        return data if isinstance(data, dict) else None

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

    def _validate_bundle_manifest(
        self, location: PluginManifestLocation, result: ValidationResult
    ) -> dict[str, Any] | None:
        """Validate a bundle-reference manifest against ``PluginManifest``.

        Returns the raw manifest mapping when it satisfies the schema, so its
        refs can be classified, else ``None``.
        """
        manifest_path = location.path
        data = self._load_yaml(location, result)
        if data is None:
            return None

        try:
            manifest = PluginManifest(**data)
        except ValidationError as exc:
            self._add_validation_findings(exc, manifest_path, result)
            return None

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
        return data

    def _resolve_dependencies(
        self,
        location: PluginManifestLocation,
        data: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Classify declared skill/rule refs and gate on missing same-repo targets.

        States are ``provided``/``referenced``/``missing``/``external``/
        ``unresolved`` (see :mod:`skillevaluator.plugin_dependencies`). Only
        ``missing`` blocks. Repository identity fails closed: without a verified
        git ``origin`` slug every ref is ``unresolved`` (advisory), never
        ``referenced`` or ``missing``.
        """
        from skillevaluator.plugin_dependencies import classify_plugin_dependencies, resolve_repository_identity

        plugin_meta = result.metadata.setdefault("plugin", {})
        identity = resolve_repository_identity(location.root, self.repo_root)
        if identity.repo_root_ignored:
            result.add_message("--repo-root ignored for plugin dependency resolution: it does not contain the plugin")
        try:
            resolution = classify_plugin_dependencies(
                data,
                location.root,
                identity,
                bundled_skills=plugin_meta.get("bundled_skills") or (),
            )
        except ValueError as exc:
            result.add_warning(f"Plugin dependency resolution skipped: {exc}")
            return

        manifest_path = str(location.path)
        for section, rows in (("skills", resolution.skills), ("rules", resolution.rules)):
            for row in rows:
                if row.state == "missing":
                    result.add_finding(
                        Finding(
                            category="PLUGIN_SCHEMA",
                            severity=Severity.HIGH,
                            check_name="plugin_dependency_missing",
                            message=f"Declared {section} dependency '{row.ref}' is missing: {row.reason}.",
                            file_path=manifest_path,
                            suggestion=(
                                "Add the referenced component at its repository path, or correct the "
                                "reference (source::owner/repo::kind::name)."
                            ),
                            metadata={"ref": row.ref, "state": row.state, "section": section},
                        )
                    )
                else:
                    location_note = f" at {row.path}" if row.path else ""
                    result.add_message(
                        f"Plugin {section} dependency '{row.ref}': {row.state}{location_note} ({row.reason})"
                    )

        counts = resolution.status_counts()
        if resolution.rows and not counts["missing"]:
            summary = ", ".join(f"{counts[state]} {state}" for state in counts if counts[state])
            advisory = ""
            if counts["unresolved"]:
                advisory = (
                    f"; {counts['unresolved']} unresolved ref(s) are advisory only -- the missing-dependency "
                    "gate could not be evaluated for them (validate from the plugin's git clone with an "
                    "'origin' remote, or pass --repo-root)"
                )
            result.add_success(
                check_name="plugin_dependencies",
                message=f"Declared dependencies: {summary}{advisory}",
                **counts,
            )
        plugin_meta["dependency_resolution"] = resolution.to_metadata()
        plugin_meta["dependency_status_counts"] = counts

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

    def _validate_contained_manifest(
        self, location: PluginManifestLocation, result: ValidationResult
    ) -> dict[str, Any] | None:
        """Shallow-validate a contained ``.claude-plugin/plugin.json`` file.

        Returns the parsed manifest when it is a JSON object with a valid name, for
        the component inventory; ``None`` after any blocking manifest error.
        """
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
            return None

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
            return None
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
            return None

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
            return None

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
            return None

        # Runnable MCP servers (every mcpServers form plus the root .mcp.json) get
        # blocking, network-free static validation in _inventory_components, which
        # also emits the manifest success row once no blocking MCP finding exists.
        result.add_message(f"Plugin name: {name}")
        plugin_meta = result.metadata.setdefault("plugin", {})
        plugin_meta["name"] = name
        declared = {key: len(value) for key, value in data.items() if isinstance(value, list)}
        if declared:
            plugin_meta["declared_dependencies"] = declared
        return data

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
        skill_names = [manifest.relative_path.parent.as_posix() for manifest in skill_manifests]
        skill_dirs = [skills_dir / manifest.relative_path.parent for manifest in skill_manifests]
        plugin_meta = result.metadata.setdefault("plugin", {})
        plugin_meta["in_plugin_skills"] = len(skill_dirs)
        # Plugin-root-relative ids (``skills/<name>``).
        plugin_meta["bundled_skills"] = [f"skills/{name}" for name in skill_names]
        if not skill_manifests:
            return
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
