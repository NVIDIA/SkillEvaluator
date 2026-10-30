# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a private plugin inventory for a runtime evaluation.

Preparation is passive: it reads only the plugin root through the inventory's
anchored reader, never fetches dependencies, installs packages, or probes an
endpoint. Runnable configuration comes from the selected manifest's effective
rows only; rows of overridden or additional client views stay audit records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES, PLUGIN_CURSOR_MANIFEST_TYPE
from skillevaluator.plugin_config_risk import _nested_hook_groups
from skillevaluator.plugin_formats import profile_for
from skillevaluator.plugin_markdown import parse_markdown
from skillevaluator.plugin_paths import PluginRootReader
from skillevaluator.tier3 import plugin_native as native
from skillevaluator.utils.secure_fs import SecurePathError

if TYPE_CHECKING:
    from skillevaluator.plugin_discovery import PluginMember
    from skillevaluator.plugin_inventory import PluginInventory

_FRONTMATTER_HOOKS = "#hooks"
_TEXT_TYPES = frozenset({"agent", "command", "output_style"})
_RUNTIME_TYPES = frozenset({"skill", "rule", "mcp"})


class PreparationError(ValueError):
    """The inventory cannot be prepared for a runtime evaluation."""


@dataclass(frozen=True)
class SourceIdentity:
    """The analyzed source a prepared package was built from."""

    root: str
    manifest: str
    manifest_sha256: str

    @classmethod
    def of(cls, inventory: PluginInventory) -> SourceIdentity:
        return cls(
            str(inventory.root),
            inventory.selected_manifest.manifest_filename,
            inventory.manifest_sha256,
        )


@dataclass(frozen=True)
class AuditRow:
    """An inventory row that is not runnable configuration, with the reason it is kept."""

    type: str
    name: str
    path: str | None
    reason: str
    declared_by: str = ""


@dataclass(frozen=True)
class PreparedRuntimePackage:
    """One plugin inventory prepared for the runtime arms of an evaluation."""

    identity: SourceIdentity
    plugin_name: str
    description: str
    skills: tuple[PluginMember, ...]
    rules: tuple[PluginMember, ...]
    mcp_servers: tuple[Mapping[str, Any], ...]
    native_source: native.NativePluginSource
    audit: tuple[AuditRow, ...] = ()
    policy_findings: tuple[Any, ...] = ()
    incomplete_reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.incomplete_reasons

    @property
    def runnable(self) -> bool:
        """Whether there is anything a runtime arm can load."""
        return bool(self.skills or self.rules or self.mcp_servers or self.native_source.texts)


def prepare_runtime_package(
    inventory: PluginInventory,
    *,
    expected: SourceIdentity | None = None,
    blocking_findings: Sequence[Any] = (),
    excluded_paths: Sequence[Path] = (),
) -> PreparedRuntimePackage:
    """Build the prepared package from the selected manifest's effective rows.

    ``expected`` is the identity the caller analyzed; a different root,
    manifest, or manifest digest is refused instead of preparing a changed
    source. ``blocking_findings`` are findings the caller's effective policy
    rates blocking; any of them refuses preparation.
    """
    identity = SourceIdentity.of(inventory)
    if expected is not None and identity != expected:
        raise PreparationError(f"plugin source changed after analysis: expected {expected}, found {identity}")
    if blocking_findings:
        names = ", ".join(sorted({str(getattr(finding, "check_name", finding)) for finding in blocking_findings}))
        raise PreparationError(f"plugin policy blocks runtime evaluation: {names}")
    manifest = inventory.selected_manifest.parse_for_audit()
    if not isinstance(manifest, dict):
        raise PreparationError(f"selected manifest is unreadable: {identity.manifest}")
    selected = inventory.selected_components
    selected_ids = {id(member) for member in selected}
    audit = [_audit_row(member) for member in inventory.components if id(member) not in selected_ids]
    skills: list[PluginMember] = []
    rules: list[PluginMember] = []
    mcp: list[Mapping[str, Any]] = []
    for member in selected:
        if member.problem is not None:
            audit.append(AuditRow(member.type, member.name, member.path, f"unresolved: {member.problem}"))
        elif member.type == "skill" and member.path:
            skills.append(member)
        elif member.type == "rule":
            rules.append(member)
        elif member.type == "mcp" and member.mcp is not None:
            if member.mcp.runnable:
                mcp.append({"name": member.mcp.name, **member.mcp.config})
            else:
                audit.append(
                    AuditRow("mcp", member.name, member.path, f"not runnable: {member.mcp.kind or 'no launch'}")
                )
    plugin_name = str(manifest.get("name") or inventory.root.name)
    incomplete = [] if inventory.complete else ["plugin inventory is incomplete"]
    return PreparedRuntimePackage(
        identity=identity,
        plugin_name=plugin_name,
        description=str(manifest.get("description") or ""),
        skills=tuple(skills),
        rules=tuple(rules),
        mcp_servers=tuple(mcp),
        native_source=build_native_source(
            inventory, manifest, plugin_name=plugin_name, excluded_paths=excluded_paths, mcp_servers=mcp
        ),
        audit=tuple(audit),
        policy_findings=tuple(inventory.findings),
        incomplete_reasons=tuple(incomplete),
    )


def verify_source_identity(package: PreparedRuntimePackage, inventory: PluginInventory) -> None:
    """Refuse staging when the source changed since the package was prepared."""
    if SourceIdentity.of(inventory) != package.identity:
        raise PreparationError("plugin source changed between preparation and staging")


def _audit_row(member: PluginMember) -> AuditRow:
    if member.loaded_by:
        reason = f"additional client view ({member.loaded_by})"
    elif not member.effective:
        reason = "overridden by an effective declaration"
    else:
        reason = "declared by an additional manifest"
    return AuditRow(member.type, member.name, member.path, reason, member.declared_by)


def build_native_source(
    inventory: PluginInventory,
    manifest: Mapping[str, Any],
    *,
    plugin_name: str,
    excluded_paths: Sequence[Path] = (),
    mcp_servers: Sequence[Mapping[str, Any]] = (),
) -> native.NativePluginSource:
    """Snapshot what a native adapter can stage from the selected effective rows.

    Text components and hook files are read through the anchored root reader
    (bounded, never following links). A component enabling a permission bypass
    is recorded in ``refusals`` so every adapter that would stage its type
    natively refuses it.
    """
    reader = PluginRootReader(inventory.root)
    manifest_rel = inventory.selected_manifest.manifest_filename
    texts: list[native.NativeTextComponent] = []
    hooks: list[native.NativeHookSource] = []
    other: list[tuple[str, str]] = []
    refusals: list[tuple[str, str, str]] = []
    settings: dict[str, Any] = {}
    lsp_servers: dict[str, Any] = {}
    manifest_type = inventory.selected_manifest.manifest_type
    nest = manifest_type == PLUGIN_CURSOR_MANIFEST_TYPE
    root_prefixes = tuple(profile_for(manifest_type).root_prefixes)
    dialect = "cursor" if nest else "claude"

    def refuse(kind: str, name: str, message: str | None) -> None:
        if message is not None:
            refusals.append((kind, name, message))

    for member in inventory.selected_components:
        if member.problem is not None or member.type in _RUNTIME_TYPES:
            continue
        if member.type in _TEXT_TYPES:
            text = _member_text(reader, manifest, member)
            if text is None:
                continue
            where = f"plugin {member.type} '{member.name}'"
            refuse(
                member.type,
                member.name,
                native._bypass_frontmatter_refusal(parse_markdown(text).frontmatter, where=where),
            )
            texts.append(native.NativeTextComponent(member.type, member.name, member.path, text))
        elif member.type == "hook":
            rel = member.path or ""
            if rel.endswith(_FRONTMATTER_HOOKS):
                config, rel, hook_dialect = member.config, rel.removesuffix(_FRONTMATTER_HOOKS), "claude"
            else:
                config = (
                    native._inline_hook_config(manifest, member.name)
                    if rel == manifest_rel
                    else (native._read_plugin_json(reader, rel, "hooks") if rel else None)
                )
                config, hook_dialect = (_nested_hook_groups(config) if nest else config), dialect
            if not isinstance(config, dict):
                continue
            refuse("hook", member.name, native._bypass_refusal(config, where=f"plugin hooks '{member.name}'"))
            hooks.append(native.NativeHookSource(member.name, rel, config, hook_dialect, root_prefixes))
        else:
            other.append((member.type, member.name))
            if member.type == "settings" and member.path == native._PLUGIN_SETTINGS_FILE:
                applied = native._plugin_settings(native._read_plugin_json(reader, member.path, "settings"))
                refuse("settings", member.name, native._bypass_refusal(applied, where="plugin settings.json"))
                settings.update(applied)
            elif member.type == "lsp" and member.path:
                config = (
                    manifest.get("lspServers")
                    if member.path == manifest_rel
                    else native._read_plugin_json(reader, member.path, "LSP servers")
                )
                server = native._lsp_server_config(config, member.name)
                if server is not None:
                    refuse(
                        "lsp", member.name, native._bypass_refusal(server, where=f"plugin LSP server '{member.name}'")
                    )
                    lsp_servers[member.name] = server
    if reader.kind(PurePosixPath("bin")) == "dir":
        other.append(("bin", "bin"))
    return native.NativePluginSource(
        plugin_name=plugin_name,
        description=str(manifest.get("description") or ""),
        contained=True,
        plugin_root=inventory.root,
        manifest=dict(manifest),
        manifest_rel=manifest_rel,
        member_skills=tuple(inventory.root / member.path for member in inventory.selected_skills if member.path),
        rules=tuple(
            (member.name, member.text or "")
            for member in inventory.selected_components
            if member.type == "rule" and not member.problem
        ),
        mcp_servers=tuple(dict(server) for server in mcp_servers),
        texts=tuple(texts),
        hooks=tuple(hooks),
        other=tuple(other),
        excluded_paths=tuple(excluded_paths),
        refusals=tuple(refusals),
        settings=settings,
        lsp_servers=lsp_servers,
    )


def _member_text(reader: PluginRootReader, manifest: Mapping[str, Any], member: PluginMember) -> str | None:
    if member.text is not None:
        return member.text
    if member.path is None:
        return native._inline_command_content(manifest, member.name) if member.type == "command" else None
    try:
        return reader.read_text(PurePosixPath(member.path), CONTENT_DEDUP_MAX_FILE_BYTES)
    except (SecurePathError, OSError) as exc:
        raise PreparationError(f"Refusing unsafe or unreadable plugin {member.type} '{member.path}': {exc}") from exc
