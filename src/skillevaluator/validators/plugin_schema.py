# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin manifest and bundled-skill validation.

Two plugin models are recognized:

* **Bundle-reference** (``agent_plugin.yaml`` / ``agent_plugin.yml``), validated
  in full against :class:`~skillevaluator.models.plugin.PluginManifest`.
* **Contained** (``.claude-plugin/plugin.json``), validated against the
  manifest schema Claude Code applies when it loads a plugin.
* **Contained native formats**: Agent Plugins v1 (root ``plugin.json``), Codex
  (``.codex-plugin/plugin.json``), and Cursor (``.cursor-plugin/plugin.json``),
  validated field by field by :mod:`skillevaluator.plugin_formats`.

When a root holds more than one supported manifest, the first one in
precedence order is validated and inventoried; the others are recorded under
``metadata['plugin']['manifest_declarations']``, their components and static
findings are merged into the inventory, and a name or version that differs from
the selected manifest is a MEDIUM ``plugin_manifest_conflict``.

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
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    NAME_MAX_LENGTH,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_TYPES,
    PLUGIN_CONTAINED_MODE,
    PLUGIN_MANIFEST_RELATIVE_PATHS,
    PLUGIN_MODE,
    PLUGIN_NAME_MAX_REPORT_CHARS,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.plugin import PluginManifest
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_formats import (
    agent_plugins_schema_version,
    declared_value_replaces_default,
    manifest_syntax,
    normalized_component_manifest,
    profile_for,
    validate_manifest_fields,
)
from skillevaluator.plugin_manifest import (
    CLAUDE_DEPENDENCY_MAX_ENTRIES,
    ClaudeMarketplace,
    PluginManifestCandidate,
    PluginManifestLocation,
    PluginManifestPathError,
    canonical_manifest_relative,
    classify_claude_dependency,
    find_claude_marketplace,
    locate_plugin_manifest,
    manifest_relative_path,
)
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.structured_data import (
    StructuredDataLimitError,
    StructuredDataSyntaxError,
    load_bounded_json,
    load_bounded_yaml,
)
from skillevaluator.validators.base import ValidatorBase
from skillevaluator.validators.frontmatter_parser import format_validation_location
from skillevaluator.validators.mcp_static import CATEGORY as MCP_CATEGORY

if TYPE_CHECKING:
    from skillevaluator.plugin_components import PluginInventory
    from skillevaluator.validators.policy import ValidationPolicy

logger = get_logger(__name__)
MAX_PLUGIN_SCHEMA_FINDINGS = 100


class PluginSchemaValidator(ValidatorBase):
    """Validate a plugin manifest and any skills bundled by the plugin."""

    def __init__(
        self,
        policy: ValidationPolicy | None = None,
        repo_root: Path | None = None,
        *,
        resolve_endpoints: bool = False,
    ) -> None:
        self.policy = policy
        # Optional ``--repo-root``: repository root for same-repository ref resolution.
        self.repo_root = repo_root
        # Opt-in DNS/redirect checks (``--resolve-endpoints`` or ``endpoints.resolve`` in the policy).
        self.resolve_endpoints = resolve_endpoints or bool(policy is not None and policy.resolve_endpoints)

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
            if canonical_manifest_relative(exc.relative_path) is not None:
                check_name, message, suggestion = self._manifest_link_problem(path, exc)
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
            if self._inventory_manifestless(path, result):
                return result
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_missing",
                    message=(
                        "No plugin manifest found. Expected one of "
                        f"{', '.join(PLUGIN_MANIFEST_RELATIVE_PATHS)} at the plugin root. A root plugin.json is an "
                        "Agent Plugins manifest only when it declares an https://agent-plugins.org/schemas/ $schema."
                    ),
                    file_path=str(path),
                    suggestion=(
                        "Add an agent_plugin.yaml (or agent_plugin.yml), a .claude-plugin/, .codex-plugin/, or "
                        ".cursor-plugin/ plugin.json, or an Agent Plugins root plugin.json, at the plugin root."
                    ),
                )
            )
            return result

        manifest_type = located.manifest_type
        root = located.root
        self._stamp_manifest_metadata(located.manifest_filename, root, manifest_type, result)
        self._report_case_variants(located, result)

        validated_manifest: dict[str, Any] | None = None
        contained_data: dict[str, Any] | None = None
        manifest_valid = True
        if manifest_type in PLUGIN_CONTAINED_MANIFEST_TYPES:
            contained_data, manifest_valid = self._validate_contained_manifest(located, result)
        else:
            validated_manifest = self._validate_bundle_manifest(located, result)

        # A manifest error must not hide problems in skills bundled alongside it.
        replaced_by = self._skills_dir_replaced_by(manifest_type, contained_data)
        self._validate_in_plugin_skills(root, result, replaced_by=replaced_by)
        if validated_manifest is not None:
            self._resolve_dependencies(located, validated_manifest, result)
        elif manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE and contained_data is not None:
            self._resolve_claude_dependencies(located, contained_data, result)
        additional = self._record_manifest_declarations(located, result)
        inventory = self._inventory_components(
            located, contained_data, result, additional, manifest_valid=manifest_valid
        )
        self._validate_inventory_skills(inventory, root, result)
        return result

    @staticmethod
    def _report_case_variants(location: PluginManifestLocation, result: ValidationResult) -> None:
        """HIGH for a client manifest spelled in another case (``.Codex-Plugin/plugin.json``).

        Clients open fixed paths, so on a case-insensitive filesystem (the macOS
        default) they load the variant as their manifest, and on a
        case-sensitive one they ignore it. SkillEvaluator checks it as the
        manifest it stands for.
        """
        for item in (location, *location.additional):
            if not item.case_variant:
                continue
            canonical_path = canonical_manifest_relative(item.secure_file.relative_path)
            canonical = canonical_path.as_posix() if canonical_path is not None else item.manifest_filename
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_case_variant",
                    message=(
                        f"{item.manifest_filename} is a case variant of {canonical}. Clients on a case-insensitive "
                        f"filesystem (the macOS default) load it as {canonical}, and clients on a case-sensitive "
                        "filesystem ignore it, so the plugin differs by platform"
                    ),
                    file_path=str(item.declared_path),
                    suggestion=f"Rename it to exactly {canonical}.",
                    metadata={"manifest_filename": item.manifest_filename, "canonical": canonical},
                )
            )

    def _inventory_components(
        self,
        location: PluginManifestLocation,
        contained_data: dict[str, Any] | None,
        result: ValidationResult,
        additional: list[tuple[str, str, dict[str, Any] | None]] | None = None,
        *,
        manifest_valid: bool = True,
    ) -> PluginInventory:
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

        contained = location.contained
        manifest = contained_data if contained else self._bundle_manifest_data(location)
        root = location.secure_file.root
        allowed_hosts = self.policy.mcp_allowed_private_hosts if self.policy is not None else ()
        hook_allowed_urls = self.policy.hook_allowed_urls if self.policy is not None else ()
        inventory = build_plugin_inventory(
            root,
            manifest,
            contained=contained,
            manifest_rel=manifest_rel_for(location.path, location.root),
            allowed_private_hosts=allowed_hosts,
            hook_allowed_urls=hook_allowed_urls,
            manifest_type=location.manifest_type,
            additional=additional or (),
        )
        findings = list(inventory.findings)
        endpoint_resolution: dict[str, Any] | None = None
        if self.resolve_endpoints:
            endpoint_resolution, endpoint_findings = self._resolve_endpoints(
                inventory, root, allowed_hosts, hook_allowed_urls
            )
            findings.extend(endpoint_findings)
            if endpoint_resolution.get("incomplete"):
                # Endpoints left unchecked (cap, time budget, or no DNS) are missing evidence, never a pass.
                from skillevaluator.validators.endpoint_resolution import INCOMPLETE_SCAN

                result.mark_scan_incomplete(INCOMPLETE_SCAN)
        self._report_component_findings(findings, result, str(location.path))
        if contained and contained_data is not None:
            blocking_mcp = [
                finding
                for finding in findings
                if finding.category == MCP_CATEGORY and finding.severity in (Severity.CRITICAL, Severity.HIGH)
            ]
            if not blocking_mcp and manifest_valid:
                name = result.metadata.get("plugin", {}).get("name", "")
                profile = profile_for(location.manifest_type)
                message = (
                    f"{profile.label} manifest '{name}' ({location.manifest_filename}) passed the required-field checks"
                )
                result.add_success(check_name="plugin_manifest", message=message)
        if not contained:
            self._mark_dependency_states(inventory, result.metadata.get("plugin", {}).get("dependency_resolution"))
        attribute_findings(inventory.components, result.findings, root)
        result.metadata.setdefault("plugin", {}).update(inventory.metadata())
        if endpoint_resolution is not None:
            result.metadata["plugin"]["endpoint_resolution"] = endpoint_resolution
        return inventory

    @staticmethod
    def _report_component_findings(findings: list[Finding], result: ValidationResult, file_path: str) -> None:
        """Add the component findings, at most ``MAX_PLUGIN_SCHEMA_FINDINGS``, keeping every blocking one first.

        The cap bounds report size. It keeps CRITICAL and HIGH findings before
        any MEDIUM, LOW, or INFO one (and truncation notes before other notes
        of their severity), so many advisory findings never push a blocking one
        out, and never fail Tier 1 on their own. The HIGH ``schema_errors_truncated``
        is added only when blocking findings were dropped; dropped advisory
        findings get a LOW ``schema_findings_truncated`` note.
        """
        blocking = {Severity.CRITICAL, Severity.HIGH}
        rank = {Severity.CRITICAL: 0, Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}

        def priority(index: int) -> tuple[int, int, int]:
            finding = findings[index]
            return (rank.get(finding.severity, 3), 0 if finding.check_name.endswith("_truncated") else 1, index)

        kept = set(sorted(range(len(findings)), key=priority)[:MAX_PLUGIN_SCHEMA_FINDINGS])
        for index, finding in enumerate(findings):
            if index in kept:
                result.add_finding(finding)
        dropped = [finding for index, finding in enumerate(findings) if index not in kept]
        if not dropped:
            return
        dropped_blocking = any(finding.severity in blocking for finding in dropped)
        result.add_finding(
            Finding(
                category="PLUGIN_SCHEMA",
                severity=Severity.HIGH if dropped_blocking else Severity.LOW,
                check_name="schema_errors_truncated" if dropped_blocking else "schema_findings_truncated",
                message=(
                    f"Plugin component validation produced {len(findings)} findings; only "
                    f"{MAX_PLUGIN_SCHEMA_FINDINGS} are reported"
                    + ("." if dropped_blocking else f" (the {len(dropped)} not shown are MEDIUM or lower).")
                ),
                file_path=file_path,
                suggestion="Fix the reported plugin component and MCP declaration errors, then rerun validation.",
                metadata={"actual": len(findings), "reported": MAX_PLUGIN_SCHEMA_FINDINGS},
            )
        )

    @staticmethod
    def _mark_dependency_states(inventory: PluginInventory, resolution: Any) -> None:
        """Show each bundle ref's dependency state in the inventory instead of a plain "Evaluated" row.

        A ``missing`` ref is a broken component (its HIGH finding is attributed
        to it); ``external`` and ``unresolved`` refs carry their state. A
        ``provided`` ref that a packaged skill satisfies is that skill, so the
        two rows become one ``declared+packaged`` row.
        """
        if not isinstance(resolution, dict):
            return
        for section, component_type in (("skills", "skill"), ("rules", "rule")):
            rows = resolution.get(section)
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict) or not isinstance(row.get("ref"), str):
                    continue
                ref, state, path = row["ref"], row.get("state"), row.get("path")
                matches = [
                    component
                    for component in inventory.components
                    if component.type == component_type and component.path is None and component.name == ref
                ]
                for component in matches:
                    if state == "missing":
                        component.problem = "missing"
                    elif state in {"external", "unresolved"}:
                        component.dependency_state = state
                    elif state == "provided" and isinstance(path, str):
                        packaged = next(
                            (
                                other
                                for other in inventory.components
                                if other.type == component_type and other.path == path.strip("/")
                            ),
                            None,
                        )
                        if packaged is not None:
                            packaged.origin = "declared+packaged"
                            inventory.components.remove(component)

    def _inventory_manifestless(self, path: Path, result: ValidationResult) -> bool:
        """Inventory a folder without a manifest that Claude Code loads as a plugin; ``False`` if it is not one.

        Claude Code ``--plugin-dir`` reads such a folder's default locations
        (``agents/``, ``commands/``, ``hooks/hooks.json``, ``.mcp.json``, ...)
        and names the plugin after the folder. Codex and Cursor need a
        manifest, so the missing manifest is MEDIUM, and every component Claude
        Code loads gets the usual checks, subagent and command privileges
        included.
        """
        from skillevaluator.cli_core import manifestless_plugin_markers
        from skillevaluator.plugin_components import attribute_findings, build_plugin_inventory

        markers = manifestless_plugin_markers(path)
        if not markers:
            return False
        root = path
        result.metadata["manifest_type"] = PLUGIN_CONTAINED_MANIFEST_TYPE
        result.metadata["plugin_mode"] = PLUGIN_CONTAINED_MODE
        result.metadata["plugin"] = {"manifest_filename": None, "root": str(root), "manifestless": True}
        result.add_finding(
            Finding(
                category="PLUGIN_SCHEMA",
                severity=Severity.MEDIUM,
                check_name="manifest_missing",
                message=(
                    "No plugin manifest found, but the folder ships Claude Code plugin components in their default "
                    f"locations ({', '.join(markers)}). Claude Code --plugin-dir loads it as a plugin named after the "
                    "folder; Codex and Cursor do not load it. Its components are checked as Claude Code loads them"
                ),
                file_path=str(path),
                suggestion=(
                    "Add .claude-plugin/plugin.json (and .codex-plugin/ or .cursor-plugin/ manifests for other "
                    "clients) so the plugin has an explicit name and version."
                ),
                metadata={"markers": markers},
            )
        )
        self._validate_in_plugin_skills(root, result)
        allowed_hosts = self.policy.mcp_allowed_private_hosts if self.policy is not None else ()
        hook_allowed_urls = self.policy.hook_allowed_urls if self.policy is not None else ()
        inventory = build_plugin_inventory(
            root,
            None,
            contained=True,
            manifest_rel=profile_for(PLUGIN_CONTAINED_MANIFEST_TYPE).manifest_path,
            allowed_private_hosts=allowed_hosts,
            hook_allowed_urls=hook_allowed_urls,
            manifestless=True,
        )
        self._report_component_findings(list(inventory.findings), result, str(path))
        attribute_findings(inventory.components, result.findings, root)
        result.metadata["plugin"].update(inventory.metadata())
        self._validate_inventory_skills(inventory, root, result)
        return True

    @staticmethod
    def _resolve_endpoints(
        inventory: Any,
        root: Path,
        allowed_hosts: tuple[str, ...],
        hook_allowed_urls: tuple[str, ...],
    ) -> tuple[dict[str, Any], list[Finding]]:
        """Opt-in DNS + single-HEAD redirect checks for MCP URLs (and OAuth metadata URLs) and HTTP hooks.

        MCP servers come from the selected manifest and from every additional
        manifest (merged into the inventory as components with ``declared_by``),
        because the client that loads an additional manifest connects to its
        servers too. A server is checked once per ``(name, url)``.
        """
        from skillevaluator.plugin_component_risk import hook_allowlist_hosts
        from skillevaluator.plugin_components import PluginRootReader
        from skillevaluator.validators.endpoint_resolution import EndpointChecker, EndpointTarget
        from skillevaluator.validators.mcp_static import expand_url_defaults

        reader = PluginRootReader(root)
        targets: list[EndpointTarget] = []
        declarations = [
            *inventory.mcp.effective,
            *(
                component.mcp
                for component in inventory.components
                if component.type == "mcp" and component.declared_by and component.mcp is not None
            ),
        ]
        seen_servers: set[tuple[str, str]] = set()
        for declaration in declarations:
            config = declaration.config
            if not isinstance(config, dict):
                continue
            oauth = config.get("oauth")
            # The server URL, and the OAuth metadata URL Claude Code fetches before it connects.
            urls = (config.get("url"), oauth.get("authServerMetadataUrl") if isinstance(oauth, dict) else None)
            for url in urls:
                if not isinstance(url, str) or not url.strip() or (declaration.name, url) in seen_servers:
                    continue
                seen_servers.add((declaration.name, url))
                targets.append(
                    EndpointTarget(
                        # The endpoint the plugin ships: '${API_URL:-https://host/mcp}' is checked as its default.
                        url=expand_url_defaults(url),
                        kind="mcp",
                        name=declaration.name,
                        file_path=reader.display(declaration.file),
                        allowed_hosts=tuple(allowed_hosts),
                    )
                )
        hook_hosts = (*allowed_hosts, *hook_allowlist_hosts(hook_allowed_urls))
        for record in inventory.hook_records:
            if record.handler_type == "http" and record.url:
                targets.append(
                    EndpointTarget(
                        url=record.url,
                        kind="hook",
                        name=record.id,
                        file_path=reader.display(record.file),
                        component=("hook", record.source),
                        allowed_hosts=hook_hosts,
                    )
                )
        return EndpointChecker().check(targets)

    @staticmethod
    def _manifest_link_problem(path: Path, exc: PluginManifestPathError) -> tuple[str, str, str]:
        """Check name, message, and suggestion for a manifest that discovery refused as a link.

        The manifest is never read either way. The name says what was found:
        ``manifest_linked`` for a symlink whose target stays inside the plugin
        root, ``manifest_hardlinked`` for a hard link, and
        ``manifest_outside_root`` for a symlink that leaves the root (or
        dangles). Only link metadata is inspected; nothing is opened.
        """
        import os
        import stat as stat_module

        suggestion = "Replace the link with a regular file contained by the plugin root."
        try:
            metadata = path.lstat()
            root = path if stat_module.S_ISDIR(metadata.st_mode) and not stat_module.S_ISLNK(metadata.st_mode) else None
        except OSError:
            root = None
        if root is None:
            relative = manifest_relative_path(path)
            root = (path.parent if len(relative.parts) == 1 else path.parent.parent) if relative else path.parent
        candidate = root / exc.relative_path
        try:
            link_metadata = candidate.lstat()
        except OSError:
            return "manifest_outside_root", str(exc), suggestion
        if stat_module.S_ISLNK(link_metadata.st_mode):
            target = Path(os.path.realpath(candidate))
            if target.is_relative_to(Path(os.path.realpath(root))) and target.is_file():
                return (
                    "manifest_linked",
                    f"{exc.relative_path} is a symlink to {target.relative_to(Path(os.path.realpath(root))).as_posix()} "
                    "inside the plugin root. SkillEvaluator never follows a manifest link, so the plugin is not "
                    f"evaluated: {exc}",
                    "Replace the symlink with the regular file it points to.",
                )
            return (
                "manifest_outside_root",
                f"{exc.relative_path} is a symlink that leaves the plugin root: {exc}",
                suggestion,
            )
        if stat_module.S_ISREG(link_metadata.st_mode) and getattr(link_metadata, "st_nlink", 1) > 1:
            return (
                "manifest_hardlinked",
                f"{exc.relative_path} is hard-linked ({link_metadata.st_nlink} links), so its content can change "
                f"through another path. SkillEvaluator does not read it: {exc}",
                "Replace the hard link with a regular file that has a single link.",
            )
        return "manifest_outside_root", str(exc), suggestion

    @classmethod
    def note_requested_manifest(cls, results: list[ValidationResult], requested: Path) -> None:
        """Tell the user when the manifest file they named is not the one evaluated.

        ``validate`` resolves a manifest path to its plugin root and selects
        the manifest by precedence, so naming ``.codex-plugin/plugin.json`` in a
        root that also has ``.claude-plugin/plugin.json`` evaluates the Claude
        Code manifest. That is documented, but it must not happen silently: an
        INFO ``manifest_named_not_selected`` finding names both files.
        """
        relative = manifest_relative_path(requested)
        if relative is None:
            return
        named = canonical_manifest_relative(relative)
        named_text = (named or relative).as_posix()
        for result in results:
            if result.validator_name != cls().name or not isinstance(result.metadata, dict):
                continue
            selected = (result.metadata.get("plugin") or {}).get("manifest_filename")
            if not isinstance(selected, str):
                return
            canonical_selected = canonical_manifest_relative(selected)
            if (canonical_selected.as_posix() if canonical_selected else selected) == named_text:
                return
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.INFO,
                    check_name="manifest_named_not_selected",
                    message=(
                        f"You named {named_text}, but validate evaluates the plugin root and selected {selected} by "
                        f"precedence. {named_text} is checked as an additional manifest, and Tier 3 stages {selected}."
                    ),
                    file_path=str(requested),
                    suggestion=(
                        "To evaluate the other client's manifest, validate a copy of the plugin without the "
                        "higher-precedence manifest."
                    ),
                    metadata={"named": named_text, "selected": selected},
                )
            )
            return

    @staticmethod
    def _yaml_version_literal(text: str, value: Any) -> str | None:
        """The literal text of a top-level numeric YAML ``version`` (``1.10``, which YAML reads as ``1.1``)."""
        import re

        if not isinstance(value, int | float) or isinstance(value, bool):
            return None
        match = re.search(r"^version[ \t]*:[ \t]*(?P<literal>[0-9][0-9._eE+-]*)[ \t]*(?:#.*)?$", text, re.MULTILINE)
        if match is None:
            return None
        literal = match.group("literal")
        try:
            same = float(literal.replace("_", "")) == float(value)
        except ValueError:
            return None
        return literal if same and literal != str(value) else None

    def _numeric_yaml_version(self, location: PluginManifestLocation, data: dict[str, Any]) -> str | None:
        try:
            return self._yaml_version_literal(location.read_text(), data.get("version"))
        except PluginManifestPathError:
            return None

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
        if manifest_type in PLUGIN_CONTAINED_MANIFEST_TYPES:
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
        literal = self._numeric_yaml_version(location, data)
        if literal is not None:
            data = {**data, "version": literal}
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.LOW,
                    check_name="schema:version:numeric",
                    message=(
                        f"{location.manifest_filename} declares 'version: {literal}' as a YAML number, which YAML "
                        f"readers turn into {float(literal.replace('_', ''))!r}. SkillEvaluator keeps the text "
                        f"'{literal}'."
                    ),
                    file_path=str(manifest_path),
                    suggestion=f'Quote the version: version: "{literal}".',
                    metadata={"field": "version"},
                )
            )

        try:
            manifest = PluginManifest.model_validate(data)
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
        ``referenced`` or ``missing``. Unresolved refs get a MEDIUM
        ``plugin_dependency_unverified`` finding (the gate did not run for
        them), never a passing row. Every ref is classified, past the 256-ref
        staging limit too, and a section over that limit gets a MEDIUM
        ``plugin_dependency_limit`` finding.
        """
        from skillevaluator.plugin_dependencies import (
            classify_plugin_dependencies,
            record_unverified_dependencies,
            resolve_repository_identity,
        )

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
                limit=None,
            )
        except ValueError as exc:
            # The gate could not run at all: never a silent pass.
            result.add_warning(f"Plugin dependency resolution could not run: {exc}")
            result.mark_scan_incomplete("plugin-dependency-resolution")
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
        record_unverified_dependencies(resolution, manifest_path, result)
        if resolution.rows and not counts["missing"] and not counts["unresolved"]:
            summary = ", ".join(f"{counts[state]} {state}" for state in counts if counts[state])
            result.add_success(
                check_name="plugin_dependencies",
                message=f"Declared dependencies: {summary}",
                **counts,
            )
        elif resolution.rows:
            summary = ", ".join(f"{counts[state]} {state}" for state in counts if counts[state])
            result.add_message(f"Declared dependencies: {summary}")
        plugin_meta["dependency_resolution"] = resolution.to_metadata()
        plugin_meta["dependency_status_counts"] = counts

    def _claude_marketplace(self, location: PluginManifestLocation, data: dict[str, Any]) -> ClaudeMarketplace | None:
        """The Claude Code marketplace that lists this plugin (see :func:`find_claude_marketplace`).

        The walk never goes above ``--repo-root`` or the git top-level that
        contains the plugin.
        """
        from skillevaluator.utils.helpers import resolve_git_root

        try:
            root = location.root.expanduser().resolve()
        except OSError:
            return None
        stop: Path | None = None
        for candidate in (self.repo_root, resolve_git_root(root)):
            if candidate is not None and root.is_relative_to(resolved := candidate.expanduser().resolve()):
                stop = resolved
                break
        name = data.get("name") if isinstance(data.get("name"), str) else None
        return find_claude_marketplace(root, plugin_name=name, stop=stop)

    def _resolve_claude_dependencies(
        self, location: PluginManifestLocation, data: dict[str, Any], result: ValidationResult
    ) -> None:
        """Classify a Claude Code manifest's ``dependencies`` the way ``agent_plugin.yaml`` refs are classified.

        Claude Code loads a plugin only after every dependency is installed and
        enabled, and installs a dependency itself only from the plugin's own
        marketplace (or from a marketplace that marketplace allows). Offline,
        that can be proven only through the marketplace manifest:

        * ``referenced``: the plugin's marketplace lists the dependency.
        * ``missing``: the plugin's marketplace lists no plugin of that name,
          or lists it with a local source folder that does not exist. Claude
          Code cannot install it and refuses to load the plugin, so this is HIGH
          ``plugin_dependency_missing``, as for ``agent_plugin.yaml``.
        * ``external``: the dependency names another marketplace.
        * ``unresolved``: no marketplace that lists this plugin was found.

        ``external`` and ``unresolved`` dependencies are MEDIUM
        ``plugin_dependency_unverified``: Claude Code still refuses to load the
        plugin when such a dependency is not installed, but that cannot be
        checked offline. Malformed entries are schema errors, reported by the
        manifest check, and are skipped here.
        """
        from skillevaluator.plugin_dependencies import DependencyRow, status_counts
        from skillevaluator.plugin_formats import CLAUDE_DEPENDENCY_RE

        entries = data.get("dependencies")
        if not isinstance(entries, list) or not entries:
            return
        marketplace = self._claude_marketplace(location, data)
        rows: list[DependencyRow] = []
        for entry in entries[:CLAUDE_DEPENDENCY_MAX_ENTRIES]:
            if isinstance(entry, str) and (match := CLAUDE_DEPENDENCY_RE.match(entry)):
                name, market = match.group("name"), match.group("marketplace")
            elif (
                isinstance(entry, dict)
                and isinstance(entry.get("name"), str)
                and (match := CLAUDE_DEPENDENCY_RE.match(entry["name"])) is not None
                and match.group("marketplace") is None
                and (entry.get("marketplace") is None or isinstance(entry["marketplace"], str))
            ):
                name, market = entry["name"], entry.get("marketplace") or None
            else:
                continue  # a schema error, reported by the manifest check
            rows.append(classify_claude_dependency(name, market, marketplace))

        manifest_path = str(location.path)
        for row in rows:
            if row.state == "missing":
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="plugin_dependency_missing",
                        message=(
                            f"Declared plugin dependency '{row.ref}' is missing: {row.reason}. Claude Code cannot "
                            "install it, so it refuses to load this plugin."
                        ),
                        file_path=manifest_path,
                        suggestion="Add the dependency to the marketplace, or remove it from 'dependencies'.",
                        metadata={"ref": row.ref, "state": row.state, "section": "plugins"},
                    )
                )
            elif row.state in {"external", "unresolved"}:
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.MEDIUM,
                        check_name="plugin_dependency_unverified",
                        message=(
                            f"Claude Code loads this plugin only when dependency '{row.ref}' is installed and "
                            f"enabled, and that cannot be checked offline: {row.reason}."
                        ),
                        file_path=manifest_path,
                        suggestion=(
                            "Publish the plugin in a marketplace that also lists the dependency, or tell users to "
                            "install the dependency first."
                        ),
                        metadata={"ref": row.ref, "state": row.state, "section": "plugins"},
                    )
                )
            else:
                where = f" at {row.path}" if row.path else ""
                result.add_message(f"Plugin dependency '{row.ref}': {row.state}{where} ({row.reason})")
        counts = status_counts(rows)
        if rows and counts["referenced"] == len(rows):
            result.add_success(
                check_name="plugin_dependencies",
                message=f"Declared plugin dependencies: {counts['referenced']} listed in the plugin's marketplace",
                **counts,
            )
        plugin_meta = result.metadata.setdefault("plugin", {})
        plugin_meta["dependency_resolution"] = {"plugins": [row.to_dict() for row in rows]}
        plugin_meta["dependency_status_counts"] = counts

    def _load_yaml(self, location: PluginManifestLocation, result: ValidationResult) -> dict | None:
        """Parse manifest YAML, recording a finding on failure."""
        manifest_path = location.path
        try:
            raw = location.read_text()
        except PluginManifestPathError as exc:
            if exc.content_error:
                # Only SkillEvaluator reads agent_plugin.yaml, so its own limits apply. It is not read
                # leniently, so nothing it declares is checked: a policy override must not turn this
                # finding into a pass, and Tier 1 stops here as for an unsafe manifest.
                result.metadata["security_failure"] = True
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="manifest_unreadable",
                        message=f"Plugin manifest {location.manifest_filename} {self._content_problem(exc)}.",
                        file_path=str(manifest_path),
                        suggestion=(
                            f"Save {location.manifest_filename} as UTF-8 YAML under "
                            f"{CONTENT_DEDUP_MAX_FILE_BYTES} bytes."
                        ),
                        metadata={"manifest_filename": location.manifest_filename, "reason": exc.reason},
                    )
                )
                return None
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
            location = format_validation_location(error) or "<root>"
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

    def _blocks_after_policy(self, check_name: str, severity: Severity) -> bool:
        """Whether a manifest finding blocks once the active policy's severity overrides apply.

        The policy is applied to every finding later (in ``emit_reports``), so
        the manifest success row must not say "passed" next to a finding the
        policy raises to HIGH, nor be withheld for one it lowers.
        """
        if self.policy is not None:
            severity = self.policy.severity_for("PLUGIN_SCHEMA", check_name, severity)
        return severity in (Severity.CRITICAL, Severity.HIGH)

    def _load_contained_json(
        self, location: PluginManifestLocation, result: ValidationResult
    ) -> tuple[dict[str, Any] | None, bool]:
        """Read and parse a contained JSON manifest; record a HIGH finding on failure.

        Returns ``(data, readable)``; ``readable`` is ``False`` when the manifest
        could only be read leniently, so it never gets a clean pass.

        A manifest that is not UTF-8 or is over the manifest size bound is HIGH
        ``manifest_unreadable``. That is a content problem of a regular file, not
        an unsafe path, so the other Tier 1 checks and the parity check still
        run. Clients do not share these limits (Claude Code reads a Latin-1
        manifest, Codex one of any size), so it is parsed leniently and its
        fields and components are still checked. Only when even the lenient
        parse yields nothing is it also a security failure, so a policy override
        cannot pass a plugin whose declared components were never checked.
        """
        manifest_path = location.path
        filename = location.manifest_filename
        try:
            raw = location.read_text(encoding="utf-8-sig")
        except PluginManifestPathError as exc:
            if exc.content_error:
                data, _status = self._parse_unreadable(
                    location, self._content_problem(exc), result, selected=True, reason=exc.reason
                )
                if data is None:
                    result.metadata["security_failure"] = True
                return data, False
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
            return None, False

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
                    suggestion=f"Reduce JSON nesting or collection sizes in {filename}.",
                )
            )
            return None, False
        except StructuredDataSyntaxError as exc:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_invalid_json",
                    message=f"Contained plugin manifest is not valid JSON: {exc}",
                    file_path=str(manifest_path),
                    suggestion=f"Fix the JSON syntax in {filename}.",
                )
            )
            return None, False

        if not isinstance(data, dict) or not data:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_not_object",
                    message="Contained plugin manifest must be a non-empty JSON object.",
                    file_path=str(manifest_path),
                    suggestion=f"Populate {filename} with at least a non-empty 'name'.",
                )
            )
            return None, False
        return data, True

    def _validate_contained_manifest(
        self, location: PluginManifestLocation, result: ValidationResult
    ) -> tuple[dict[str, Any] | None, bool]:
        """Validate a Claude Code, Agent Plugins v1, Codex, or Cursor manifest against its format's rules.

        A Claude Code manifest gets the schema Claude Code itself applies (see
        :func:`skillevaluator.plugin_formats.validate_manifest_fields`), not only
        its name. Returns ``(parsed manifest, valid)``. The parsed manifest is
        returned even after a HIGH field error, so its declared component paths
        and MCP servers are still inventoried and statically checked; ``valid``
        is ``False`` then, and no manifest success row is recorded. ``None``
        means the manifest could not be read or parsed. The same holds for a
        manifest that could only be read leniently (``manifest_unreadable``).
        Runnable MCP servers get their blocking static checks in
        :meth:`_inventory_components`, which also records the success row.
        """
        data, readable = self._load_contained_json(location, result)
        if data is None:
            return None, False
        profile = profile_for(location.manifest_type)
        # A selected manifest read leniently (not UTF-8, or over the size bound) is never a clean pass.
        blocking = not readable
        severities = {"error": Severity.HIGH, "warning": Severity.MEDIUM, "note": Severity.LOW}
        issues = validate_manifest_fields(location.manifest_type, data)
        for issue in issues[:MAX_PLUGIN_SCHEMA_FINDINGS]:
            severity = severities[issue.level]
            check_name = (
                "plugin_manifest_unknown_field"
                if issue.error == "unknown_field" and "." not in issue.field
                else f"schema:{issue.field}:{issue.error}"
            )
            blocking = blocking or self._blocks_after_policy(check_name, severity)
            if self._scalar_component_value(location.manifest_type, issue.field, data):
                # One defect, one finding: the inventory reports a number or boolean in a component field
                # (plugin_component_path_invalid or mcp_servers_not_object) at the client's severity: HIGH
                # where the client refuses the manifest, MEDIUM where Codex drops the value and installs.
                continue
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=severity,
                    check_name=check_name,
                    message=issue.message,
                    file_path=str(location.path),
                    suggestion=issue.suggestion
                    or f"Fix {location.manifest_filename} to satisfy the {profile.reference}.",
                    metadata={"manifest_type": location.manifest_type, "field": issue.field},
                )
            )
        plugin_meta = result.metadata.setdefault("plugin", {})
        name = data.get("name")
        if isinstance(name, str) and name.strip():
            plugin_meta["name"] = name.strip()[:PLUGIN_NAME_MAX_REPORT_CHARS]
            result.add_message(f"Plugin name: {plugin_meta['name']}")
        version = agent_plugins_schema_version(data.get("$schema"))
        if version is not None:
            plugin_meta["manifest_spec_version"] = version
        # Only fields that name other plugins are dependencies. Components the
        # plugin ships (skills, mcpServers, hooks, ...) and metadata lists such as
        # keywords are not. Of the contained formats only Claude Code has one.
        dependencies = data.get("dependencies") if location.manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE else None
        if isinstance(dependencies, list) and dependencies:
            plugin_meta["declared_dependencies"] = {"plugins": len(dependencies)}
        return data, not blocking

    def _record_manifest_declarations(
        self, location: PluginManifestLocation, result: ValidationResult
    ) -> list[tuple[str, str, dict[str, Any] | None]]:
        """Record every supported manifest in the root and flag name/version conflicts.

        Returns the parsed additional manifests for the inventory merge. An
        additional manifest that cannot be parsed or fails its required fields
        is a MEDIUM finding: it does not drive this evaluation, but the client
        that loads it would reject it. A client manifest that is not UTF-8 or is
        over the manifest size bound is HIGH, because clients without those
        limits still load it; it is then read leniently so its components are
        still checked. A linked, special, or changed one is a HIGH security
        failure. A ``.codex-plugin/plugin.json`` beside a root Agent Plugins
        manifest is validated as the documented Codex overlay.
        """
        selected_data = self._best_effort_parse(location.manifest_type, location.read_text)
        rows: list[dict[str, Any]] = [
            self._declaration_row(location.manifest_type, location.manifest_filename, selected_data, selected=True)
        ]
        additional: list[tuple[str, str, dict[str, Any] | None]] = []
        conflicts: list[dict[str, Any]] = []
        has_agent_plugins_root = PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE in {
            location.manifest_type,
            *(candidate.manifest_type for candidate in location.additional),
        }
        for candidate in location.additional:
            overlay = has_agent_plugins_root and candidate.manifest_type == PLUGIN_CODEX_MANIFEST_TYPE
            data, status = self._parse_additional(candidate, result, overlay=overlay)
            row = self._declaration_row(candidate.manifest_type, candidate.manifest_filename, data, selected=False)
            row["status"] = status
            if overlay:
                row["overlay"] = True
            rows.append(row)
            if data is not None:
                additional.append((candidate.manifest_type, candidate.manifest_filename, data))
            for field_name in ("name", "version"):
                selected_value = rows[0].get(field_name)
                other_value = row.get(field_name)
                if selected_value and other_value and selected_value != other_value:
                    conflicts.append(
                        {
                            "field": field_name,
                            "selected": selected_value,
                            "additional": other_value,
                            "manifest_filename": candidate.manifest_filename,
                        }
                    )
                    result.add_finding(
                        Finding(
                            category="PLUGIN_SCHEMA",
                            severity=Severity.MEDIUM,
                            check_name="plugin_manifest_conflict",
                            message=(
                                f"{candidate.manifest_filename} declares {field_name} {other_value!r}, but the "
                                f"selected manifest {location.manifest_filename} declares {selected_value!r}; "
                                "clients that load different manifests see different plugins"
                            ),
                            file_path=str(candidate.declared_path),
                            suggestion=f"Keep the plugin {field_name} identical in every manifest.",
                            metadata={"field": field_name, "manifest_filename": candidate.manifest_filename},
                        )
                    )
        plugin_meta = result.metadata.setdefault("plugin", {})
        plugin_meta["manifest_declarations"] = {
            "selected": location.manifest_filename,
            "precedence": list(PLUGIN_MANIFEST_RELATIVE_PATHS),
            "manifests": rows,
            "conflicts": conflicts,
        }
        if location.additional:
            others = ", ".join(f"{row['manifest_filename']} ({row['status']})" for row in rows[1:])
            result.add_success(
                check_name="plugin_manifests",
                message=(
                    f"Selected {location.manifest_filename} ({location.manifest_type}) by precedence; additional "
                    f"manifests: {others}; {len(conflicts)} name/version conflict(s)"
                ),
            )
        return additional

    @staticmethod
    def _best_effort_parse(manifest_type: str, read_text: Any) -> dict[str, Any] | None:
        try:
            text = read_text(encoding="utf-8-sig")
            data = load_bounded_json(text) if manifest_syntax(manifest_type) == "json" else load_bounded_yaml(text)
        except (PluginManifestPathError, StructuredDataLimitError, StructuredDataSyntaxError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        if manifest_syntax(manifest_type) == "yaml":
            literal = PluginSchemaValidator._yaml_version_literal(text, data.get("version"))
            if literal is not None:
                data = {**data, "version": literal}
        return data

    def _parse_additional(
        self, candidate: PluginManifestCandidate, result: ValidationResult, *, overlay: bool = False
    ) -> tuple[dict[str, Any] | None, str]:
        problem: str | None = None
        data: Any = None
        try:
            text = candidate.read_text(encoding="utf-8-sig")
        except PluginManifestPathError as exc:
            if not exc.content_error:
                result.metadata["security_failure"] = True
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="manifest_unsafe",
                        message=f"Could not securely read additional plugin manifest: {exc}",
                        file_path=str(candidate.declared_path),
                        suggestion=(
                            "Replace links/hardlinks/special manifests with one regular file inside the plugin root."
                        ),
                    )
                )
                return None, "unsafe"
            # A regular file that is not UTF-8 or is over the size bound is invalid content, not an unsafe path.
            problem = self._content_problem(exc)
            if candidate.manifest_type in PLUGIN_CONTAINED_MANIFEST_TYPES:
                return self._parse_unreadable(candidate, problem, result, reason=exc.reason)
        else:
            try:
                data = (
                    load_bounded_json(text)
                    if manifest_syntax(candidate.manifest_type) == "json"
                    else load_bounded_yaml(text)
                )
            except (StructuredDataLimitError, StructuredDataSyntaxError, ValueError) as exc:
                problem = f"could not be parsed: {exc}"
            if isinstance(data, dict) and manifest_syntax(candidate.manifest_type) == "yaml":
                # Keep a numeric YAML version as written (1.10, not 1.1), as for the selected manifest.
                literal = self._yaml_version_literal(text, data.get("version"))
                if literal is not None:
                    data = {**data, "version": literal}
        if problem is None and not isinstance(data, dict):
            problem = "is not a JSON/YAML object"
        if problem is None:
            issues = validate_manifest_fields(candidate.manifest_type, data, overlay=overlay)
            all_errors = [issue for issue in issues if issue.level == "error"]
            # One defect, one finding: a scalar where a component path belongs is already a component
            # path finding (plugin_component_path_invalid, at full severity), so it is not repeated here.
            errors = [
                issue
                for issue in all_errors
                if not self._scalar_component_value(candidate.manifest_type, issue.field, data)
            ]
            if errors:
                problem = "fails required-field checks: " + "; ".join(issue.message.rstrip(".") for issue in errors[:3])
            elif all_errors:
                return data, "invalid"
        if problem is not None:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.MEDIUM,
                    check_name="plugin_manifest_additional_invalid",
                    message=(
                        f"Additional manifest {candidate.manifest_filename} {problem.rstrip('.')}. It is not the "
                        "manifest SkillEvaluator evaluates, but the client that loads it rejects or misreads the "
                        "plugin."
                    ),
                    file_path=str(candidate.declared_path),
                    suggestion=f"Fix {candidate.manifest_filename}, or remove it if the plugin does not target that client.",
                    metadata={"manifest_filename": candidate.manifest_filename},
                )
            )
            return (data if isinstance(data, dict) else None), "invalid"
        return data, "parsed"

    @staticmethod
    def _content_problem(exc: PluginManifestPathError) -> str:
        """Why a safely discovered manifest file cannot be read strictly: its encoding or its size."""
        if exc.reason == "encoding":
            return f"is not valid UTF-8 text ({exc.__cause__ or exc})"
        return f"is larger than the {CONTENT_DEDUP_MAX_FILE_BYTES}-byte manifest read limit"

    @staticmethod
    def _scalar_component_value(manifest_type: str, field: str, data: dict[str, Any]) -> bool:
        """Whether a field error is about a number or boolean in a component field the inventory checks as a path.

        Every component field is resolved as a path or an MCP server map, which
        reports such a value at full severity, except ``settings`` and
        ``experimental`` (and the Agent Plugins ``$schema``/``extensions``).
        ``experimental.monitors`` is resolved too. A top-level Claude Code
        ``monitors`` is not deduplicated: the inventory reports it under
        ``experimental.monitors``, so the field error is the only finding that
        names the field the client reports.
        """
        if field == "experimental.monitors":
            experimental = data.get("experimental")
            value = experimental.get("monitors") if isinstance(experimental, dict) else None
            return isinstance(value, int | float | bool)
        if field in {"settings", "experimental", "$schema", "extensions", "monitors"}:
            return False
        if field not in profile_for(manifest_type).component_fields:
            return False
        value = data.get(field)
        return isinstance(value, int | float | bool)

    @staticmethod
    def _parse_unreadable(
        manifest: PluginManifestLocation | PluginManifestCandidate,
        problem: str,
        result: ValidationResult,
        *,
        selected: bool = False,
        reason: str | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """HIGH for a client manifest over the size bound or not UTF-8, then a lenient parse.

        Clients do not share SkillEvaluator's 1 MiB bound or its strict UTF-8
        decoding: Codex reads a manifest of any size, and Claude Code reads a
        Latin-1 one. So the manifest is read again with a larger bound and
        replacement characters, and what it declares is inventoried and
        checked like any readable manifest. The finding is
        ``manifest_unreadable`` for the selected manifest and
        ``plugin_manifest_additional_unreadable`` for an additional one.
        """
        if selected:
            subject, check_name, alternative = "Plugin manifest", "manifest_unreadable", ""
            # The selected manifest's fields are validated too; an additional one's are not.
            scope = "its fields and the components it declares are"
        else:
            subject, check_name = "Additional manifest", "plugin_manifest_additional_unreadable"
            alternative = ", or remove it if the plugin does not target that client"
            scope = "the components it declares are"
        data: dict[str, Any] | None = None
        try:
            parsed = load_bounded_json(manifest.read_lenient_text().removeprefix("﻿"))
        except PluginManifestPathError as exc:
            if not exc.content_error:
                result.metadata["security_failure"] = True
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="manifest_unsafe",
                        message=f"Could not securely read {subject.lower()}: {exc}",
                        file_path=str(manifest.declared_path),
                        suggestion=(
                            "Replace links/hardlinks/special manifests with one regular file inside the plugin root."
                        ),
                    )
                )
                return None, "unsafe"
        except (StructuredDataLimitError, StructuredDataSyntaxError, ValueError):
            pass
        else:
            data = parsed if isinstance(parsed, dict) else None
        checked = (
            f"It was read leniently, and {scope} checked below"
            if data is not None
            else f"It could not be parsed even leniently (up to {CONTENT_DEDUP_MAX_TOTAL_BYTES} bytes), so the "
            "components it declares are not checked"
        )
        result.add_finding(
            Finding(
                category="PLUGIN_SCHEMA",
                severity=Severity.HIGH,
                check_name=check_name,
                message=(
                    f"{subject} {manifest.manifest_filename} {problem.rstrip('.')}. Clients without this "
                    f"limit still load it. {checked}."
                ),
                file_path=str(manifest.declared_path),
                suggestion=(
                    f"Save {manifest.manifest_filename} as UTF-8 JSON under {CONTENT_DEDUP_MAX_FILE_BYTES} bytes"
                    f"{alternative}."
                ),
                metadata={"manifest_filename": manifest.manifest_filename, "reason": reason},
            )
        )
        return data, "unreadable"

    @staticmethod
    def _declaration_row(
        manifest_type: str, manifest_filename: str, data: dict[str, Any] | None, *, selected: bool
    ) -> dict[str, Any]:
        def _field(key: str) -> str | None:
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, int | float) and not isinstance(value, bool):
                value = str(value)
            return value.strip()[: NAME_MAX_LENGTH * 2] if isinstance(value, str) and value.strip() else None

        row: dict[str, Any] = {
            "manifest_type": manifest_type,
            "manifest_filename": manifest_filename,
            "selected": selected,
            "status": "selected" if selected else "parsed",
            "name": _field("name"),
            "version": _field("version"),
        }
        spec_version = agent_plugins_schema_version(data.get("$schema")) if isinstance(data, dict) else None
        if spec_version is not None:
            row["spec_version"] = spec_version
        return row

    @staticmethod
    def _skills_dir_replaced_by(manifest_type: str, data: dict[str, Any] | None) -> str | None:
        """The format label when the selected manifest's declared ``skills`` replace ``skills/``.

        Cursor and Codex load only the declared skill folders when ``skills`` is
        set (Codex only when it accepts the path). Skills found only in
        ``skills/`` are then not part of that client's plugin.
        """
        if manifest_type not in PLUGIN_CONTAINED_MANIFEST_TYPES or not isinstance(data, dict):
            return None
        from skillevaluator.plugin_components import normalize_declared_path

        profile = profile_for(manifest_type)
        value = (normalized_component_manifest(manifest_type, data) or {}).get("skills")
        if not declared_value_replaces_default(profile, "skills", value):
            return None
        default_dir = profile.default_skills_dir or "skills"
        for raw in value if isinstance(value, list) else [value]:
            if isinstance(raw, str):
                declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
                if declared.rel is not None and declared.rel.as_posix() == default_dir:
                    return None
        return profile.label

    def _validate_in_plugin_skills(
        self, root: Path, result: ValidationResult, *, replaced_by: str | None = None
    ) -> None:
        """Validate skills bundled under ``<root>/skills/``.

        When the selected format's declared ``skills`` replace ``skills/``
        (``replaced_by``), these skills are not loaded by that client, so their
        findings are advisory (MEDIUM at most); Claude Code still loads them
        from any plugin folder. A ``SKILL.md`` inside a folder that Tier 1
        scans skip (``evals/``, ``results/``, ``versions/`` within a skill) is
        HIGH, because clients that search ``skills/`` recursively load it.
        """
        skills_dir = root / "skills"
        from skillevaluator.utils.helpers import (
            find_bundled_plugin_skill_manifests,
            find_unscanned_plugin_skill_manifests,
        )

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
        self._report_unscanned_skills(find_unscanned_plugin_skill_manifests(root), root, result)
        self._report_dependency_folder_skills(root, result)
        if not skill_manifests:
            return
        from skillevaluator.validators.schema import SchemaValidator

        validator = SchemaValidator(policy=self.policy)
        advisory = (
            f"advisory: the {replaced_by} declares its own skill folders and does not load skills/"
            if replaced_by
            else None
        )
        for skill_dir, skill_name, manifest in zip(skill_dirs, skill_names, skill_manifests, strict=True):
            self._validate_one_skill(validator, skill_dir, skill_name, manifest, result, advisory=advisory)

    @staticmethod
    def _report_dependency_folder_skills(root: Path, result: ValidationResult) -> None:
        """HIGH for each ``SKILL.md`` a client loads from ``node_modules/``, ``.venv/``, ``.git/``, or ``__pycache__/``.

        Skill discovery prunes these folders like the whole-tree scans, but
        Claude Code loads ``skills/node_modules/SKILL.md`` and Codex loads
        ``skills/node_modules/pkg/x/SKILL.md``, so such a skill would pass
        unchecked.
        """
        from skillevaluator.plugin_components import (
            PluginRootReader,
            dependency_folder_skill_findings,
            find_dependency_folder_skills,
        )

        found, complete = find_dependency_folder_skills(root)
        for finding in dependency_folder_skill_findings(PluginRootReader(root).display, "skills", found, complete):
            result.add_finding(finding)

    @staticmethod
    def _report_unscanned_skills(unscanned: list[Any], root: Path, result: ValidationResult) -> None:
        """HIGH for each ``SKILL.md`` in a skill's own ``evals/``, ``results/``, or ``versions/`` folder."""
        for rel in unscanned[:MAX_PLUGIN_SCHEMA_FINDINGS]:
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.HIGH,
                    check_name="plugin_skill_in_unscanned_folder",
                    message=(
                        f"'{rel}' is a skill inside a folder that Tier 1 scans skip (evals/, results/, versions/). "
                        "Codex searches skills/ recursively and loads it, but SkillEvaluator does not check it"
                    ),
                    file_path=str(root / str(rel)),
                    suggestion=(
                        "Move evaluation output and version snapshots out of the plugin (for Tier 3 results, use "
                        "--results-dir or SKILLEVALUATOR_RESULTS_DIR), or give a real skill a different folder name."
                    ),
                    metadata={"path": str(rel)},
                )
            )

    @staticmethod
    def _skill_not_utf8_finding(skill_name: str, skill_dir: Path) -> Finding:
        """HIGH for a skill manifest that is not UTF-8 text (its own finding; the other scans still run)."""
        return Finding(
            category="PLUGIN_SCHEMA",
            severity=Severity.HIGH,
            check_name="bundled_skill_not_utf8",
            message=(
                f"Skill '{skill_name}' has a SKILL.md that is not valid UTF-8, so its frontmatter and instructions "
                "cannot be checked. Clients may still load it with replacement characters"
            ),
            file_path=f"[{skill_name}] {skill_dir}",
            suggestion="Save SKILL.md as UTF-8 text.",
        )

    def _validate_one_skill(
        self,
        validator: Any,
        skill_dir: Path,
        skill_name: str,
        manifest: Any,
        result: ValidationResult,
        *,
        advisory: str | None = None,
    ) -> None:
        try:
            skill_result = validator.validate_secure_manifest(skill_dir, manifest)
        except SecurePathError as exc:
            if exc.code == "invalid_text_encoding":
                # Not a filesystem attack: report the encoding and keep the other scans running.
                result.add_finding(self._skill_not_utf8_finding(skill_name, skill_dir))
                return
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
            return
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
            return
        if advisory is not None:
            skill_result = _advisory_result(skill_result, advisory)
        result.merge_with_prefix(skill_result, skill_name)
        if skill_result.passed:
            result.add_success(
                check_name=skill_name,
                message=f"Bundled skill '{skill_name}' passed skill schema validation",
            )

    def _validate_inventory_skills(self, inventory: PluginInventory, root: Path, result: ValidationResult) -> None:
        """Schema-validate every inventoried skill folder outside ``skills/``.

        Cursor, Codex, and Claude Code load the skill folders a manifest
        declares (Cursor's own example is ``"skills": "./my-skills/"``), and
        Cursor loads a root ``SKILL.md``. These get the same skill schema checks
        as the skills in ``skills/``, which :meth:`_validate_in_plugin_skills`
        already covered.
        """
        from skillevaluator.utils.helpers import find_skill_manifest_in
        from skillevaluator.validators.schema import SchemaValidator

        validator: Any = None
        seen: set[str] = set()
        validated: list[str] = []
        for component in inventory.components:
            path = component.path
            if component.type != "skill" or component.problem is not None or path is None:
                continue
            if path == "skills" or path.startswith("skills/") or path in seen:
                continue
            seen.add(path)
            skill_dir = root if path == "." else root / path
            try:
                manifest = find_skill_manifest_in(skill_dir)
            except ValueError as exc:
                result.metadata["security_failure"] = True
                result.add_finding(
                    Finding(
                        category="PLUGIN_SCHEMA",
                        severity=Severity.HIGH,
                        check_name="bundled_skill_path_unsafe",
                        message=f"Could not securely read declared skill '{path}': {exc}",
                        file_path=str(skill_dir),
                        suggestion="Replace linked or special entries in the skill folder with regular files.",
                    )
                )
                continue
            if manifest is None:
                continue
            validator = validator or SchemaValidator(policy=self.policy)
            self._validate_one_skill(validator, skill_dir, path, manifest, result)
            validated.append(path)
        if validated:
            result.metadata.setdefault("plugin", {})["declared_skills"] = validated


def _advisory_result(skill_result: ValidationResult, note: str) -> ValidationResult:
    """Copy a skill result with blocking findings lowered to MEDIUM and errors kept as warnings."""
    advisory = ValidationResult()
    for finding in skill_result.findings:
        blocking = finding.severity in (Severity.CRITICAL, Severity.HIGH)
        advisory.add_finding(
            Finding(
                category=finding.category,
                severity=Severity.MEDIUM if blocking else finding.severity,
                check_name=finding.check_name,
                message=f"{finding.message} ({note})" if blocking else finding.message,
                file_path=finding.file_path,
                line_number=finding.line_number,
                line_content=finding.line_content,
                suggestion=finding.suggestion,
                metadata={**finding.metadata, "advisory": True} if blocking else finding.metadata,
            )
        )
    finding_errors = {finding.to_legacy_string() for finding in skill_result.findings}
    for message in [*skill_result.errors, *skill_result.warnings]:
        if message not in finding_errors:
            advisory.add_warning(message)
    advisory.success_details.extend(skill_result.success_details)
    return advisory
