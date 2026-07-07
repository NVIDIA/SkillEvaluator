# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin manifest and bundled-skill validation.

Validates bundle-reference manifests (``agent_plugin.yaml`` / ``agent_plugin.yml``)
and contained manifests (``.claude-plugin/plugin.json``). Bundle-reference
manifests take precedence when both are present.

Mirrors the structure of
:class:`~skillevaluator.validators.rules_schema.RulesSchemaValidator` but emits
structured :class:`~skillevaluator.models.result.Finding` objects (category
``PLUGIN_SCHEMA``) instead of legacy error strings, and attaches reporting
metadata (``manifest_type``, ``plugin_mode``, ``plugin``) on success.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from skillevaluator.constants import (
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MODE,
    PLUGIN_MANIFEST_FILES,
    PLUGIN_MANIFEST_TYPE,
    PLUGIN_MODE,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.plugin import PluginManifest
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.validators.base import ValidatorBase

logger = get_logger(__name__)


class PluginSchemaValidator(ValidatorBase):
    """Validate a plugin manifest and any skills bundled by the plugin.

    Checks: manifest presence, parseable YAML, and the
    :class:`PluginManifest` contract (allowed top-level fields, required
    ``name`` + ``author.email``, at least one dependency, well-formed
    selectors/MCP entries).
    """

    def __init__(self, policy=None) -> None:
        self.policy = policy

    @property
    def name(self) -> str:
        return "Plugin Schema & Bundle References"

    @property
    def description(self) -> str:
        return "Validate the plugin manifest and any bundled skills"

    def validate(self, path: Path) -> ValidationResult:
        """Validate the plugin manifest located at (or under) ``path``."""
        result = ValidationResult()

        located = self._locate_manifest(path)
        if located is None:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_missing",
                    message=(
                        "No plugin manifest found. Expected one of "
                        f"{', '.join(PLUGIN_MANIFEST_FILES)} or "
                        f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE} at the plugin root."
                    ),
                    file_path=str(path),
                    suggestion=(
                        "Add an agent_plugin.yaml (or agent_plugin.yml), or a "
                        f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}, at the plugin root."
                    ),
                )
            )
            return result

        manifest_path, manifest_type = located
        root = manifest_path.parent.parent if manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE else manifest_path.parent
        self._stamp_manifest_metadata(manifest_path, root, manifest_type, result)

        if manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            self._validate_contained_manifest(manifest_path, result)
        else:
            data = self._load_yaml(manifest_path, result)
            if data is not None:
                try:
                    manifest = PluginManifest(**data)
                except ValidationError as exc:
                    self._add_validation_findings(exc, manifest_path, result)
                else:
                    self._add_success(manifest, manifest_path, result)

        self._validate_in_plugin_skills(root, result)
        return result

    def _locate_manifest(self, path: Path) -> tuple[Path, str] | None:
        """Return ``(manifest_path, manifest_type)`` with bundle precedence."""
        if path.is_file():
            if path.name in PLUGIN_MANIFEST_FILES:
                return path, PLUGIN_MANIFEST_TYPE
            if path.name == PLUGIN_CONTAINED_MANIFEST_FILE and path.parent.name == PLUGIN_CONTAINED_MANIFEST_DIR:
                return path, PLUGIN_CONTAINED_MANIFEST_TYPE
            return None
        if path.is_dir():
            for manifest_name in PLUGIN_MANIFEST_FILES:
                candidate = path / manifest_name
                if candidate.exists():
                    return candidate, PLUGIN_MANIFEST_TYPE
            contained = path / PLUGIN_CONTAINED_MANIFEST_DIR / PLUGIN_CONTAINED_MANIFEST_FILE
            if contained.exists():
                return contained, PLUGIN_CONTAINED_MANIFEST_TYPE
        return None

    @staticmethod
    def _stamp_manifest_metadata(manifest_path: Path, root: Path, manifest_type: str, result: ValidationResult) -> None:
        if manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            mode = PLUGIN_CONTAINED_MODE
            filename = f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}"
        else:
            mode = PLUGIN_MODE
            filename = manifest_path.name
        result.metadata["manifest_type"] = manifest_type
        result.metadata["plugin_mode"] = mode
        result.metadata["plugin"] = {"manifest_filename": filename, "root": str(root)}

    def _load_yaml(self, manifest_path: Path, result: ValidationResult) -> dict | None:
        """Parse the manifest YAML; record a finding and return None on failure."""
        try:
            raw = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_unreadable",
                    message=f"Could not read plugin manifest: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Ensure the manifest file exists and is readable.",
                )
            )
            return None

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
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
        """Translate a pydantic ``ValidationError`` into structured findings."""
        for error in exc.errors():
            location = ".".join(str(loc) for loc in error["loc"]) or "<root>"
            error_type = error.get("type", "value_error")
            check_name = f"schema:{location}:{error_type}"
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name=check_name,
                    message=f"Field '{location}': {error['msg']}",
                    file_path=str(manifest_path),
                    suggestion=(
                        "Fix the plugin manifest to satisfy the bundle-reference contract "
                        "(allowed fields, required name + author.email, at least one "
                        "dependency, valid selectors/MCP entries)."
                    ),
                )
            )

    def _validate_contained_manifest(self, manifest_path: Path, result: ValidationResult) -> None:
        """Shallow-validate a contained ``.claude-plugin/plugin.json`` file."""
        try:
            data: Any = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_unreadable",
                    message=f"Could not read plugin manifest: {exc}",
                    file_path=str(manifest_path),
                    suggestion="Ensure the manifest file exists and is readable.",
                )
            )
            return
        except json.JSONDecodeError as exc:
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
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
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
        result.add_success(
            check_name="plugin_manifest",
            message=f"Contained plugin manifest '{name}' is valid (name present; full schema deferred)",
        )
        plugin = result.metadata.setdefault("plugin", {})
        plugin["name"] = name
        dependencies = {key: len(value) for key, value in data.items() if isinstance(value, list)}
        if dependencies:
            plugin["declared_dependencies"] = dependencies

    def _validate_in_plugin_skills(self, root: Path, result: ValidationResult) -> None:
        """Merge schema results for live skills bundled below ``skills/``."""
        from skillevaluator.utils.helpers import find_bundled_plugin_skills
        from skillevaluator.validators.schema import SchemaValidator

        skills_root = root / "skills"
        skill_dirs = find_bundled_plugin_skills(root)
        if not skill_dirs:
            return
        names = [skill_dir.relative_to(skills_root).as_posix() for skill_dir in skill_dirs]
        plugin = result.metadata.setdefault("plugin", {})
        plugin["in_plugin_skills"] = len(skill_dirs)
        plugin["bundled_skills"] = names
        validator = SchemaValidator(policy=self.policy)
        for skill_dir, name in zip(skill_dirs, names, strict=True):
            try:
                skill_result = validator.validate(skill_dir)
            except Exception as exc:
                logger.warning("In-plugin skill validation failed for %s: %s", skill_dir, exc)
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="in_plugin_skill_error",
                        message=f"Could not validate bundled skill '{name}': {exc}",
                        file_path=f"[{name}] {skill_dir}",
                        suggestion="Inspect the bundled skill directory; it may be malformed.",
                    )
                )
                continue
            if skill_result.passed:
                result.merge_with_prefix(skill_result, name)
                result.add_success(check_name=name, message=f"Bundled skill '{name}' passed skill schema validation")
            else:
                result.merge_with_prefix(skill_result, name)

    def _add_success(
        self,
        manifest: PluginManifest,
        manifest_path: Path,
        result: ValidationResult,
    ) -> None:
        """Record success details and reporting metadata for a valid manifest."""
        result.add_message(f"Plugin name: {manifest.name}")
        result.add_message(f"Author: {manifest.author.email}")
        result.add_success(
            check_name="plugin_manifest",
            message=f"Plugin manifest '{manifest.name}' is valid",
        )

        result.metadata["manifest_type"] = PLUGIN_MANIFEST_TYPE
        result.metadata["plugin_mode"] = PLUGIN_MODE
        result.metadata["plugin"] = {
            "name": manifest.name,
            "manifest_filename": manifest_path.name,
            "root": str(manifest_path.parent),
        }
