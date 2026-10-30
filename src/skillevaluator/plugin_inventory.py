# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private composition of bounded discovery, static risk and context estimates.

The selected effective client view is separate from every declaration that an
independent audit must inspect. Nothing here registers a validator or runtime.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    PLUGIN_MANIFEST_PRECEDENCE,
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_component_risk import (
    HookRecord,
    PrivilegeRecord,
    analyze_agent,
    analyze_command,
    analyze_skill,
    hook_risk_summary,
    mcp_server_is_read_only,
    privilege_summary,
)
from skillevaluator.plugin_config_risk import (
    analyze_codex_settings,
    analyze_hook_config,
    analyze_lsp_server,
    analyze_monitor,
    analyze_settings,
    create_hook_analyzer,
)
from skillevaluator.plugin_context_cost import CostRow, context_cost, member_cost_rows
from skillevaluator.plugin_discovery import PluginDiscovery, PluginMember, discover_plugin_members
from skillevaluator.plugin_formats import profile_for
from skillevaluator.plugin_manifest import PluginManifestFile, PluginManifestPathError, _read_secure_manifest_bytes
from skillevaluator.plugin_mcp import McpCollection, McpDeclaration, mcp_pinning_summary
from skillevaluator.plugin_paths import (
    PluginRootReader,
    _plugin_finding,
    _pruned_folder_findings,
    normalize_declared_path,
)
from skillevaluator.utils.secure_fs import SecurePathError, discover_secure_files
from skillevaluator.validators.mcp_static import validate_mcp_server_declaration

COMPONENT_TYPES = ("skill", "rule", "mcp", "hook", "agent", "command", "lsp", "output_style", "monitor", "settings")
_MANIFEST_PATHS = frozenset(path for path, _kind in PLUGIN_MANIFEST_PRECEDENCE)


@dataclass
class PluginInventory:
    """One passive source tree with selected members, audit inputs and static findings."""

    root: Path
    discovery: PluginDiscovery
    selected_manifest: PluginManifestFile
    manifest_sha256: str
    components: list[PluginMember] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    hook_records: tuple[HookRecord, ...] = ()
    privilege_records: tuple[PrivilegeRecord, ...] = ()
    source_files: tuple[PurePosixPath, ...] = ()
    tree_complete: bool = True
    _reader: PluginRootReader = field(kw_only=True, repr=False)
    _cost_rows: tuple[CostRow, ...] = field(default=(), repr=False)

    @property
    def selected_components(self) -> list[PluginMember]:
        """Effective rows of the selected manifest, including declarations it cannot load."""
        selected = self.selected_manifest.manifest_filename
        return [
            member
            for member in self.components
            if member.declared_by == selected and not member.loaded_by and member.effective
        ]

    @property
    def selected_skills(self) -> list[PluginMember]:
        """Readable selected skills with contained source directories; unresolved refs remain inventory rows."""
        return [
            member
            for member in self.selected_components
            if member.type == "skill" and member.path and not member.problem
        ]

    @property
    def complete(self) -> bool:
        """Whether required discovery, source traversal and bounded reads completed."""
        return (
            self.discovery.complete
            and self.tree_complete
            and bool(self.manifest_sha256)
            and not any(
                finding.check_name.endswith(("scan_incomplete", "scan_truncated", "list_truncated"))
                or finding.check_name
                in {"plugin_hook_script_unanalyzed", "plugin_component_unreadable", "plugin_manifest_unreadable"}
                for finding in self.findings
            )
        )

    @property
    def mcp(self) -> McpCollection:
        """Selected effective declarations for later consumers that need a load plan."""
        return McpCollection(declarations=[member.mcp for member in self.selected_components if member.mcp is not None])

    def of_type(self, component_type: str) -> list[PluginMember]:
        return [member for member in self.components if member.type == component_type]

    def all_mcp_declarations(self) -> list[McpDeclaration]:
        """All distinct launch declarations across effective, overridden and additional client views."""
        declarations: list[McpDeclaration] = []
        seen: set[tuple[str, str, str]] = set()
        for member in self.discovery.components:
            declaration = member.mcp
            if declaration is None:
                continue
            key = (declaration.name, declaration.file, repr(declaration.config))
            if key not in seen:
                seen.add(key)
                declarations.append(declaration)
        return declarations

    def component_inventory(self) -> dict[str, Any]:
        counts = dict.fromkeys(COMPONENT_TYPES, 0)
        rows: list[dict[str, Any]] = []
        selected = self.selected_manifest.manifest_filename
        for member in self.components:
            counts[member.type] = counts.get(member.type, 0) + 1
            row = {
                "type": member.type,
                "name": member.name,
                "origin": member.origin,
                "path": member.path,
                "support": member.support,
                "findings": member.findings,
            }
            if member.problem:
                row["problem"] = member.problem
            if member.dependency_state:
                row["dependency"] = member.dependency_state
            if member.declared_by != selected or member.loaded_by:
                row["declared_by"] = member.declared_by
            if member.loaded_by:
                row["loaded_by"] = member.loaded_by
            if not member.effective:
                row["effective"] = False
            rows.append(row)
        return {
            "components": rows,
            "counts": counts,
            "unsupported_types_present": sorted(
                {member.type for member in self.components if member.support == "unsupported"}
            ),
            "broken": sum(bool(member.problem) for member in self.components),
            "complete": self.complete,
        }

    def mcp_summary(self) -> dict[str, Any]:
        declarations = self.mcp.effective
        return {
            "servers": [
                {
                    "name": item.name,
                    "source": item.source,
                    "kind": item.kind,
                    "transport": item.transport,
                    "pinned": item.pinning().pinned,
                    "pin_detail": item.pinning().detail,
                }
                for item in declarations
            ],
            "pinning": mcp_pinning_summary(declarations),
        }

    def context_cost(self, **kwargs: Any) -> dict[str, Any]:
        return context_cost(
            self.selected_components,
            self.hook_records,
            self._reader,
            manifest_type=self.selected_manifest.manifest_type,
            unread_files=self.discovery.unread_files,
            rows=self._cost_rows,
            **kwargs,
        )

    def metadata(self) -> dict[str, Any]:
        """Static values a future result adapter can attach explicitly."""
        data = {
            "component_inventory": self.component_inventory(),
            "mcp": self.mcp_summary(),
            "context_cost": self.context_cost(),
        }
        if self.hook_records:
            data["hook_risk"] = hook_risk_summary(list(self.hook_records))
        if self.privilege_records:
            data["privileges"] = privilege_summary(list(self.privilege_records))
        return data

    def attribute_findings(self, findings: Iterable[Finding] = ()) -> None:
        """Recount static and supplied validator findings against this inventory's rows."""
        attribute_findings(self.components, _unique_findings([*self.findings, *findings]), self.root)


def _unique_findings(findings: Iterable[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[Finding] = []
    for finding in findings:
        key = (finding.category, finding.check_name, str(finding.file_path or ""), finding.message)
        if key not in seen:
            seen.add(key)
            result.append(finding)
    return result


def _inventory_members(discovery: PluginDiscovery) -> list[PluginMember]:
    selected = discovery.manifests[0].manifest_filename
    members: dict[tuple[str, str, str], PluginMember] = {}
    for member in discovery.components:
        key = (member.type, member.path or "", member.name)
        current = members.get(key)
        if current is not None:
            if (
                current.declared_by == member.declared_by
                and not member.loaded_by
                and member.effective
                and not current.effective
            ):
                members[key] = replace(member)
            continue
        copied = replace(member)
        if copied.declared_by != selected or copied.loaded_by:
            copied.support = "static_only" if copied.support == "evaluated" else copied.support
        members[key] = copied
    return list(members.values())


def _source_files(root: Path, reader: PluginRootReader) -> tuple[tuple[PurePosixPath, ...], bool, list[Finding]]:
    """One no-follow bounded listing for independent audits, with exclusions explicitly reported."""
    try:
        files = discover_secure_files(
            root,
            selected=lambda _relative: True,
            excluded_dirs=SCAN_EXCLUDED_DIRS,
            max_paths=PLUGIN_TREE_MAX_DISCOVERED_PATHS,
            allow_context_alias=False,
            refuse_selected_dirs=False,
        )
    except (SecurePathError, OSError) as exc:
        finding = _plugin_finding(
            Severity.HIGH,
            "plugin_source_scan_incomplete",
            f"Authored source files could not be listed safely: {exc}",
            reader.display("."),
            "Use regular contained files and keep the authored tree within its scan limits.",
        )
        return (), False, [finding]
    return tuple(PurePosixPath(item.relative_path.as_posix()) for item in files), True, []


def _environment_findings(discovery: PluginDiscovery, reader: PluginRootReader) -> list[Finding]:
    findings = []
    for rel in discovery.environment_files[:20]:
        findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "plugin_env_file_shipped",
                f"Plugin ships an environment file '{rel.as_posix()}'; these files commonly hold credentials (contents are not printed).",
                reader.display(rel),
                "Ship an .env.example with placeholder values instead.",
            )
        )
    if len(discovery.environment_files) > 20:
        findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "plugin_env_file_shipped",
                f"Plugin ships {len(discovery.environment_files)} environment files; only the first 20 are listed.",
                reader.display("."),
                "Do not package environment files containing credentials.",
            )
        )
    if not discovery.environment_scan_complete:
        findings.append(
            _plugin_finding(
                Severity.LOW,
                "plugin_env_scan_incomplete",
                "The bounded environment-file scan stopped before every directory was inspected.",
                reader.display("."),
                "Keep generated or unreadable directories out of the package.",
            )
        )
    return findings


def _static_analysis(
    inventory: PluginInventory, *, allowed_private_hosts: tuple[str, ...], hook_allowed_urls: tuple[str, ...]
) -> None:
    reader = inventory._reader
    analyzers = {}
    hooks: dict[tuple[str, str], HookRecord] = {}
    privileges: dict[tuple[str, str, str | None], PrivilegeRecord] = {}
    manifest_data = {
        manifest.manifest_filename: manifest.parse_for_audit() or {} for manifest in inventory.discovery.manifests
    }
    analyzed: set[tuple[str, str, str, str, str]] = set()
    for member in inventory.discovery.components:
        profile = profile_for(member.manifest_type)
        key = (
            member.type,
            member.path or "",
            member.name,
            member.manifest_type,
            repr(member.config if member.config is not None else member.frontmatter),
        )
        if key in analyzed:
            continue
        analyzed.add(key)
        if member.manifest_type not in analyzers:
            analyzers[member.manifest_type] = create_hook_analyzer(
                reader, profile, hook_allowed_urls=hook_allowed_urls, allowed_private_hosts=allowed_private_hosts
            )
        analyzer = analyzers[member.manifest_type]
        path = member.path or member.declared_by
        display = reader.display(path)
        findings: list[Finding] = []
        analysis = None
        if member.type == "hook":
            analysis = analyze_hook_config(
                analyzer,
                member.config,
                source=member.name,
                file=path.removesuffix("#hooks"),
                display=display,
                manifest_type=member.manifest_type,
            )
        elif member.type == "monitor":
            analysis = analyze_monitor(analyzer, member.name, member.config, path, display)
        elif member.type in {"agent", "command", "skill"} and not member.problem:
            if member.type == "agent":
                write_mcp = [
                    item.name
                    for item in inventory.discovery.components
                    if item.mcp
                    and item.effective
                    and item.declared_by == member.declared_by
                    and item.manifest_type == member.manifest_type
                    and item.mcp.runnable
                    and not mcp_server_is_read_only(item.config)
                ]
                name = manifest_data.get(member.declared_by, {}).get("name")
                record, findings = analyze_agent(
                    member.name,
                    member.path,
                    member.frontmatter,
                    display,
                    write_capable_mcp=write_mcp,
                    plugin_name=name if isinstance(name, str) else None,
                )
            elif member.type == "command":
                record, findings = analyze_command(
                    member.name,
                    member.path,
                    member.frontmatter,
                    display,
                    entry=member.config if isinstance(member.config, dict) else None,
                    client="codex" if profile.codex_path_rules else None,
                )
            else:
                record, findings = analyze_skill(
                    member.name,
                    member.path,
                    member.frontmatter,
                    display,
                    client="codex" if profile.codex_path_rules else None,
                )
            privileges.setdefault((record.type, record.name, record.path), record)
        elif member.type == "lsp":
            findings = analyze_lsp_server(member.name, member.config, display)
        elif member.type == "settings":
            findings = (
                analyze_codex_settings(member.config, path, display)
                if path == ".codex/config.toml"
                else analyze_settings(member.config, path, display, name=member.name)
            )
        elif member.mcp is not None:
            findings = validate_mcp_server_declaration(
                member.name,
                member.config,
                display,
                allowed_private_hosts=allowed_private_hosts,
                manifest_type=member.manifest_type,
            )
        if analysis is not None:
            findings.extend(analysis.findings)
            for record in analysis.records:
                kept = hooks.setdefault((record.id, record.file), record)
                for flag in record.risk_flags:
                    kept.add_flag(flag)
        if member.loaded_by:
            for finding in findings:
                if profile.codex_path_rules and finding.metadata.get("claude_only_grant"):
                    _inert_grant_finding(finding)
                _cross_client_finding(finding, member.loaded_by)
        inventory.findings.extend(findings)
    inventory.hook_records = tuple(hooks.values())
    inventory.privilege_records = tuple(privileges.values())


def plugin_inventory_for_root(
    path: Path, *, allowed_private_hosts: Iterable[str] = (), hook_allowed_urls: Iterable[str] = ()
) -> PluginInventory | None:
    """Compose every real client declaration and fallback view without public activation."""
    discovery = discover_plugin_members(path, include_cross_client=True)
    if discovery is None or not discovery.manifests:
        return None
    selected = discovery.manifests[0]
    root = selected.secure_file.root
    reader = PluginRootReader(root)
    findings = list(discovery.findings)
    try:
        raw = _read_secure_manifest_bytes(
            selected.secure_file, selected.declared_path, max_bytes=CONTENT_DEDUP_MAX_TOTAL_BYTES
        )
        manifest_hash = hashlib.sha256(raw).hexdigest()
    except PluginManifestPathError as exc:
        manifest_hash = ""
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_manifest_unreadable",
                str(exc),
                str(selected.declared_path),
                "Keep the manifest a stable bounded regular file.",
            )
        )
    files, complete, source_findings = _source_files(root, reader)
    findings.extend(source_findings)
    findings.extend(_environment_findings(discovery, reader))
    findings.extend(_pruned_folder_findings(reader))
    inventory = PluginInventory(
        root=root,
        discovery=discovery,
        selected_manifest=selected,
        manifest_sha256=manifest_hash,
        components=_inventory_members(discovery),
        findings=findings,
        source_files=files,
        tree_complete=complete,
        _reader=reader,
    )
    _static_analysis(
        inventory, allowed_private_hosts=tuple(allowed_private_hosts), hook_allowed_urls=tuple(hook_allowed_urls)
    )
    inventory._cost_rows = tuple(member_cost_rows(inventory.selected_components, inventory.hook_records, reader))
    inventory.findings = _unique_findings(inventory.findings)
    inventory.attribute_findings()
    return inventory


def _finding_rel_path(file_path: str, root: Path) -> str | None:
    """Root-relative POSIX path of a finding location (``[prefix] `` markers stripped)."""
    text = file_path.strip()
    while text.startswith("["):
        closing = text.find("] ")
        if closing < 0:
            break
        text = text[closing + 2 :].strip()
    if not text or text.startswith("<"):
        return None
    root_text = os.path.abspath(os.fspath(root))  # noqa: PTH100 - lexical, never resolved
    absolute = os.path.abspath(text)  # noqa: PTH100 - lexical, never resolved
    # Some scanners (Semgrep, Bandit) report resolved paths, so a plugin reached
    # through a symlinked parent folder is also matched by its resolved root.
    # Only the root is resolved; the finding path itself is never followed.
    for base in dict.fromkeys((root_text, os.path.realpath(root_text))):
        if absolute == base:
            return "."
        if absolute.startswith(base.rstrip(os.sep) + os.sep):
            return Path(os.path.relpath(absolute, base)).as_posix()
    if Path(text).is_absolute():
        return None
    # Validators that report paths relative to the plugin root.
    return PurePosixPath(text.replace("\\", "/")).as_posix()


def attribute_findings(components: list[PluginMember], findings: Iterable[Finding], root: Path) -> None:
    """Set each component's ``findings`` count from finding locations.

    MCP findings carry ``metadata['mcp_server']`` and are attributed to that
    server; every other finding is attributed to the component whose root-
    relative path is the longest prefix of the finding path.
    """
    for component in components:
        component.findings = 0
    # Inline components point at the manifest itself; manifest findings are not theirs.
    by_path = sorted(
        (
            component
            for component in components
            if component.path
            and component.path not in _MANIFEST_PATHS
            and component.path != "."
            and component.type != "mcp"
        ),
        key=lambda component: len(component.path or ""),
        reverse=True,
    )
    mcp_by_key: dict[tuple[str, str], PluginMember] = {}
    for component in components:
        if component.type == "mcp" and component.mcp is not None:
            mcp_by_key[(component.mcp.name, component.mcp.file)] = component
    mcp_by_name = {component.name: component for component in components if component.type == "mcp"}
    by_type_name: dict[tuple[str, str], PluginMember] = {}
    by_name: dict[str, PluginMember] = {}
    by_exact_path: dict[str, PluginMember] = {}
    for component in components:
        by_type_name.setdefault((component.type, component.name), component)
        by_name.setdefault(component.name, component)
        # A broken component (missing, invalid) wins its path over a working one of the same path.
        if component.path:
            current = by_exact_path.get(component.path)
            if current is None or (component.problem and not current.problem):
                by_exact_path[component.path] = component
    for finding in findings:
        rel = _finding_rel_path(str(finding.file_path or ""), root)
        metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
        server = metadata.get("mcp_server")
        if isinstance(server, str):
            target = mcp_by_key.get((server, rel or "")) or mcp_by_name.get(server)
            if target is not None:
                target.findings += 1
            continue
        tagged = metadata.get("plugin_component")
        if isinstance(tagged, dict):
            tagged_key = (str(tagged.get("type")), str(tagged.get("name")))
            candidates = [
                component
                for component in by_path
                if (component.type, component.name) == tagged_key
                and rel
                and (rel == component.path or rel.startswith((component.path or "").rstrip("/") + "/"))
            ]
            target = candidates[0] if candidates else by_type_name.get(tagged_key)
            if target is not None:
                target.findings += 1
                continue
        ref = metadata.get("plugin_component_ref")
        if isinstance(ref, str):
            target = by_name.get(ref) or _component_for_ref(ref, by_exact_path)
            if target is not None:
                target.findings += 1
                continue
        # A bundle-reference dependency finding names its ref and section (skills or rules).
        dependency_type = {"skills": "skill", "rules": "rule"}.get(str(metadata.get("section")))
        if dependency_type is not None and isinstance(metadata.get("ref"), str):
            target = by_type_name.get((dependency_type, metadata["ref"]))
            if target is not None:
                target.findings += 1
                continue
        if rel is None or rel == ".":
            continue
        for component in by_path:
            path = component.path or ""
            if rel == path or rel.startswith(path.rstrip("/") + "/"):
                component.findings += 1
                break


# Root variables any supported format may put in front of a component path.
_ALL_ROOT_VARIABLES = ("${CLAUDE_PLUGIN_ROOT}", "${CURSOR_PLUGIN_ROOT}", "${PLUGIN_ROOT}")


def _component_for_ref(ref: str, by_exact_path: dict[str, PluginMember]) -> PluginMember | None:
    """The component whose root-relative path a declared ref names, compared after normalization.

    ``./agents/x.md``, ``agents/x.md``, and ``${CLAUDE_PLUGIN_ROOT}/agents/x.md``
    all name the component at ``agents/x.md``, whatever its frontmatter name.
    """
    declared = normalize_declared_path(ref, _ALL_ROOT_VARIABLES)
    if declared.rel is None or declared.problem is not None:
        return None
    return by_exact_path.get(declared.rel.as_posix())


# Subagent, command, and skill grant findings (check 6).
_PRIVILEGE_CHECKS = frozenset(
    {
        *(
            f"plugin_agent_{name}"
            for name in (
                "unrestricted_bash",
                "wildcard_tools",
                "inherits_all_tools",
                "bypass_permissions",
                "accept_edits",
                "auto_mode",
            )
        ),
        *(f"plugin_{kind}_{name}" for kind in ("command", "skill") for name in ("unrestricted_bash", "wildcard_tools")),
    }
)
_BENIGN_PRIVILEGE_RECORD_FLAGS = frozenset(
    {"no_frontmatter", "inherits_all_tools", "ignored_hooks", "ignored_mcpServers", "wildcard_ignored", "inert_grant"}
)


def _inert_grant_finding(finding: Finding) -> None:
    """Rewrite a Claude Code-only grant in a file Claude Code does not load: it has no effect in either client.

    Codex loads the file (it reads a Claude Code manifest with its own path rules,
    so a ``commands`` map does not hide ``commands/`` from it) but ignores
    ``allowed-tools``; Claude Code honors ``allowed-tools`` but does not load the
    file. The finding is LOW and says so, instead of warning about Claude Code.
    """
    component = finding.metadata.get("plugin_component") or {}
    kind = str(component.get("type") or "component")
    name = str(component.get("name") or "")
    grant = "a wildcard tool grant" if finding.check_name.endswith("_wildcard_tools") else "unrestricted 'Bash'"
    finding.message = (
        f"{kind} '{name}' pre-approves {grant} in allowed-tools, but the grant has no effect: Codex ignores "
        f"allowed-tools (it has no Bash tool), and Claude Code does not load this {kind} from this folder"
    )
    finding.severity = Severity.LOW
    finding.suggestion = (
        f"Remove the unused {kind}, or declare it in the Claude Code manifest and scope its allowed-tools to the "
        "exact commands it needs."
    )
    finding.metadata["inert_grant"] = True


def _cross_client_finding(finding: Finding, loaded_by: str) -> None:
    """Name the client that loads a cross-client file in its finding, and cap a grant finding at MEDIUM.

    Only a client whose manifest is not the selected one loads such a file
    (Claude Code ``--plugin-dir`` in a Codex or Cursor folder, or Codex through
    another client's manifest). Every such finding says which client loads it.
    Subagent, command, and skill grants (check 6) then never apply to the plugin
    as its author ships it, so they warn and do not fail the plugin; hook, MCP,
    LSP, and other executable findings keep their severity, since that client
    would run the code.
    """
    finding.message = f"loaded only by {loaded_by}: {finding.message}"
    finding.metadata["loaded_by"] = loaded_by
    if finding.check_name in _PRIVILEGE_CHECKS and finding.severity in {Severity.CRITICAL, Severity.HIGH}:
        finding.metadata["cross_client_severity"] = str(getattr(finding.severity, "value", finding.severity))
        finding.severity = Severity.MEDIUM
