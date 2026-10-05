# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static, network-free inventory of a plugin's declared and packaged components.

The inventory answers three questions for Tier 1 and Tier 3 reporting:

* **What does the plugin ship?** Components come from the fields declared in
  ``.claude-plugin/plugin.json`` (``skills``, ``commands``, ``agents``,
  ``hooks``, ``mcpServers``, ``lspServers``, ``outputStyles``,
  ``experimental.monitors``, ``settings``) and from the Claude Code default
  locations (``skills/``, ``commands/``, ``agents/``, ``hooks/hooks.json``,
  ``.mcp.json``, ``.lsp.json``, ``output-styles/``, ``monitors/monitors.json``,
  ``settings.json``) plus SkillEvaluator's ``rules/``. For ``agent_plugin.yaml``
  the declared ``skills``/``rules`` refs and ``mcp`` entries are included.
* **What can SkillEvaluator evaluate?** Skills and rules are evaluated, runnable
  MCP servers are evaluated and provider-only ones are static-only; every other
  type is inventoried only (``unsupported``).
* **What always loads into the agent context?** A ``chars_div_4`` static
  estimate of always-on versus on-demand text.

Every path comes from plugin-controlled input. Declared paths must be relative
and contained in the plugin root; existence is checked with ``lstat`` per
component (links are never followed), and every file read goes through
:class:`~skillevaluator.utils.secure_fs.SecureRoot` with a byte bound
(:mod:`skillevaluator.plugin_paths`). MCP server declarations are collected by
:mod:`skillevaluator.plugin_mcp`.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_COMPONENT_MAX_ITEMS,
    PLUGIN_CONFIG_MAX_BYTES,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_TYPES,
    PLUGIN_CURSOR_MANIFEST_TYPE,
    PLUGIN_MANIFEST_RELATIVE_PATHS,
    PLUGIN_MANIFEST_TYPE,
    SCAN_ARTIFACT_DIRS,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.deduplication.plugin.ref_utils import normalize_ref
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_component_risk import (
    CLAUDE_HOOKS,
    MAX_SCRIPT_BYTES,
    MONITOR_EVENT,
    MONITOR_HOOKS,
    PERMISSIVE_PERMISSION_MODES,
    HookAnalyzer,
    HookRecord,
    HookScriptUnreadable,
    PrivilegeRecord,
    analyze_agent,
    analyze_command,
    analyze_skill,
    hook_dialect,
    hook_risk_summary,
    is_broad_allow_rule,
    mcp_server_is_read_only,
    permission_mode_flag_issues,
    privilege_summary,
)
from skillevaluator.plugin_formats import (
    CLAUDE_PROFILE,
    CODEX_PROFILE,
    PROFILES,
    FormatProfile,
    declared_value_replaces_default,
    normalized_component_manifest,
    profile_for,
)

# The MCP collection and the root-bounded reads moved to plugin_mcp and plugin_paths;
# the names marked "re-exported" stay importable from here for existing callers.
from skillevaluator.plugin_mcp import (
    _CODEX_UNAPPLIED_MCP_FIELDS,  # noqa: F401 - re-exported
    _HTTP_TYPE_ALIASES,  # noqa: F401 - re-exported
    _MCP_BUNDLE_SUFFIXES,  # noqa: F401 - re-exported
    McpCollection,
    McpDeclaration,
    McpSource,  # noqa: F401 - re-exported
    _codex_headers,  # noqa: F401 - re-exported
    _collect_path_ref,  # noqa: F401 - re-exported
    _is_inline_secret,  # noqa: F401 - re-exported
    _load_mcp_file,  # noqa: F401 - re-exported
    _normalize_dialect,  # noqa: F401 - re-exported
    collect_mcp_declarations,
    mcp_pinning_summary,  # noqa: F401 - re-exported
    summarize_pinning,
)
from skillevaluator.plugin_paths import (
    _WINDOWS_DRIVE_RE,  # noqa: F401 - re-exported
    PLUGIN_CATEGORY,  # noqa: F401 - re-exported
    DeclaredPath,
    PathKind,  # noqa: F401 - re-exported
    PluginRootReader,
    _in_unscanned_folder,
    _path_problem_finding,
    _plugin_finding,
    _style_finding,
    _unscanned_path_finding,
    normalize_declared_path,
)
from skillevaluator.utils.secure_fs import SecurePathError, stat_is_link_or_reparse
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_json, load_bounded_yaml
from skillevaluator.validators.mcp_static import (
    OverrideIssue,
    env_override_issues,
    permission_bypass_issues,
)
from skillevaluator.validators.mcp_static import (
    _validate_command as mcp_validate_command,
)
from skillevaluator.validators.mcp_static import (
    _validate_pinning as mcp_validate_pinning,
)

if TYPE_CHECKING:
    from skillevaluator.plugin_manifest import PluginManifestLocation

Support = Literal["evaluated", "static_only", "unsupported"]
Origin = Literal["declared", "packaged", "declared+packaged"]
CoverageState = Literal["staged", "not_staged", "unsupported", "unavailable", "invalid", "loaded", "exercised"]

# Types every inventory counts. The format-specific types, Codex apps ("app") and
# Agent Plugins client-extension namespaces ("extension"), are counted only when present.
COMPONENT_TYPES: tuple[str, ...] = (
    "skill",
    "rule",
    "mcp",
    "hook",
    "agent",
    "command",
    "lsp",
    "output_style",
    "monitor",
    "settings",
)
# Tier 3 coverage states. Staging assigns one of COVERAGE_STATES to each component;
# after the run the native load census ("loaded") and runtime evidence ("exercised")
# can raise a row by rank, never lower it. A row in one of these ranked states counts
# as evaluated.
COVERAGE_STATES: tuple[str, ...] = ("staged", "not_staged", "unsupported", "unavailable", "invalid")
COVERAGE_STATE_RANK: dict[str, int] = {"staged": 1, "loaded": 2, "exercised": 3}
EVALUATED_COVERAGE_STATES: frozenset[str] = frozenset(COVERAGE_STATE_RANK)
_TYPE_SUPPORT: dict[str, Support] = {
    "skill": "evaluated",
    "rule": "evaluated",
    "mcp": "evaluated",
    "hook": "unsupported",
    "agent": "unsupported",
    "command": "unsupported",
    "lsp": "unsupported",
    "output_style": "unsupported",
    "monitor": "unsupported",
    "settings": "unsupported",
    "app": "unsupported",
    "extension": "unsupported",
}

MCP_JSON = PurePosixPath(".mcp.json")
_ENV_TEMPLATE_SUFFIXES = frozenset({"example", "sample", "template", "dist", "defaults", "tmpl"})
_MAX_ENV_FILE_FINDINGS = 20
_MANIFEST_PATHS = frozenset(PLUGIN_MANIFEST_RELATIVE_PATHS)
# Agent Plugins client-extension namespace directory names (reverse-domain).
_NAMESPACE_DIR_RE = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")


# --------------------------------------------------------------------------- #
# Records                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class CostRow:
    type: str
    name: str
    always_on_chars: int
    on_demand_chars: int
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "always_on_tokens": estimate_tokens(self.always_on_chars),
            "on_demand_tokens": estimate_tokens(self.on_demand_chars),
            "basis": self.basis,
        }


@dataclass
class Component:
    """One inventoried plugin component."""

    type: str
    name: str
    origin: Origin
    path: str | None
    support: Support
    findings: int = 0
    problem: str | None = None  # missing | escape | unsafe | invalid (Tier 3 coverage "invalid")
    mcp: McpDeclaration | None = None
    bundle: bool = False
    cost: CostRow | None = None
    # Root-relative path of the additional (non-selected) manifest, or of the
    # default file or folder of another client, that declares this component;
    # ``None`` for components of the selected manifest.
    declared_by: str | None = None
    # For a component only another client loads from its default location (the
    # cross-client pass): which clients load it.
    loaded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = {
            "type": self.type,
            "name": self.name,
            "origin": self.origin,
            "path": self.path,
            "support": self.support,
            "findings": self.findings,
        }
        if self.declared_by:
            row["declared_by"] = self.declared_by
        if self.loaded_by:
            row["loaded_by"] = self.loaded_by
        return row


@dataclass
class PluginInventory:
    """Static component inventory, its findings, and the context-cost estimate."""

    components: list[Component] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    mcp: McpCollection = field(default_factory=McpCollection)
    notes: list[str] = field(default_factory=list)
    unread_files: int = 0
    # Static risk records (plugin_component_risk): per-hook handlers and per agent/command grants.
    hook_records: list[HookRecord] = field(default_factory=list)
    privilege_records: list[PrivilegeRecord] = field(default_factory=list)

    def of_type(self, component_type: str) -> list[Component]:
        return [component for component in self.components if component.type == component_type]

    def all_mcp_declarations(self) -> list[McpDeclaration]:
        """Every MCP server some supported client may start from this plugin, once per ``(name, file)``.

        The selected manifest's effective servers come first, then the servers
        only an additional manifest declares, and then those only another
        client's default files hold (the cross-client pass). Audits that must
        cover everything a client can launch (dependency CVEs, endpoint
        checks) read this list; Tier 3 stages only ``mcp.effective``.
        """
        declarations = list(self.mcp.effective)
        seen = {(declaration.name, declaration.file) for declaration in declarations}
        for component in self.components:
            declaration = component.mcp
            if component.type != "mcp" or declaration is None or component.declared_by is None:
                continue
            if (declaration.name, declaration.file) not in seen:
                seen.add((declaration.name, declaration.file))
                declarations.append(declaration)
        return declarations

    def component_inventory(self) -> dict[str, Any]:
        counts = dict.fromkeys(COMPONENT_TYPES, 0)
        for component in self.components:
            counts[component.type] = counts.get(component.type, 0) + 1
        # Format-specific types (apps, extensions) appear only when present.
        unsupported = sorted({c.type for c in self.components if c.support == "unsupported"})
        return {
            "components": [component.to_dict() for component in self.components],
            "counts": counts,
            "unsupported_types_present": unsupported,
        }

    def mcp_summary(self) -> dict[str, Any]:
        effective = self.mcp.effective
        pins = [declaration.pinning() for declaration in effective]
        servers = [
            {
                "name": declaration.name,
                "source": declaration.source,
                "kind": declaration.kind,
                "transport": declaration.transport,
                "pinned": pin.pinned,
                "pin_detail": pin.detail,
            }
            for declaration, pin in zip(effective, pins, strict=True)
        ]
        return {"servers": servers, "pinning": summarize_pinning(pins)}

    def context_cost(self, *, extra_rows: Iterable[CostRow] = (), extra_notes: Iterable[str] = ()) -> dict[str, Any]:
        rows = [component.cost for component in self.components if component.cost is not None]
        rows.extend(extra_rows)
        notes = [*_CONTEXT_COST_NOTES, *self.notes, *extra_notes]
        if self.unread_files:
            notes.append(f"{self.unread_files} component file(s) could not be read safely and are not counted.")
        by_component = [row.to_dict() for row in rows]
        return {
            "method": "static_estimate",
            "estimator": "chars_div_4",
            "always_on_tokens": sum(row["always_on_tokens"] for row in by_component),
            "on_demand_tokens": sum(row["on_demand_tokens"] for row in by_component),
            "by_component": by_component,
            "notes": notes,
        }

    def metadata(self) -> dict[str, Any]:
        """The C1 keys merged into ``ValidationResult.metadata['plugin']``.

        ``hook_risk`` and ``privileges`` are present only when the plugin ships
        hooks, or subagents and commands, respectively.
        """
        data = {
            "component_inventory": self.component_inventory(),
            "mcp": self.mcp_summary(),
            "context_cost": self.context_cost(),
        }
        if self.hook_records:
            data["hook_risk"] = hook_risk_summary(self.hook_records)
        if self.privilege_records:
            data["privileges"] = privilege_summary(self.privilege_records)
        return data


_CONTEXT_COST_NOTES: tuple[str, ...] = (
    "Static estimate: tokens are approximated as characters / 4 (chars_div_4); no tokenizer is run.",
    "Always-on: each skill's frontmatter name + description, each agent's and command's description, and "
    "output styles marked force-for-plugin: true.",
    "On-demand: SKILL.md bodies, rule bodies, agent and command bodies, and output styles the user must select.",
    "In SkillEvaluator's Tier 3 wrapper, plugin rules are embedded in the generated wrapper SKILL.md, so rules "
    "load on demand with the wrapper rather than always-on.",
    "MCP tool schemas are not known statically; MCP servers contribute 0 tokens to this estimate.",
)


def estimate_tokens(chars: int) -> int:
    """``chars_div_4`` estimator (rounded up)."""
    return math.ceil(max(chars, 0) / 4)


# --------------------------------------------------------------------------- #
# Findings helpers                                                            #
# --------------------------------------------------------------------------- #
def _unscanned_skill_finding(reader: PluginRootReader, manifest_rel: PurePosixPath) -> Finding:
    """HIGH for a skill a client loads from a declared skills folder that Tier 1 never scans.

    The skill sits deep in an ``evals/``, ``results/``, or ``versions/`` folder
    inside a skill of the declared folder (a skill's own evaluation output or
    snapshots, which the whole-tree security, secret, and Unicode scans prune).
    Codex searches the folder recursively and loads it. Every client view
    reports it with the same text, so the merged inventory keeps one finding.
    """
    return _plugin_finding(
        Severity.HIGH,
        "plugin_skill_in_unscanned_folder",
        f"'{manifest_rel.as_posix()}' is a skill inside a folder that Tier 1 scans skip (evals/, results/, "
        "versions/). A client loads it from the declared skills folder, but SkillEvaluator never security-scans it",
        reader.display(manifest_rel),
        "Give the skill folder a different name, or move evaluation output and version snapshots out of the "
        "declared skills folder.",
        metadata={"path": manifest_rel.as_posix()},
    )


def _override_findings(
    issues: Iterable[OverrideIssue],
    file_path: str,
    *,
    where: str,
    component: tuple[str, str] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    metadata = {"plugin_component": {"type": component[0], "name": component[1]}} if component else None
    for issue in issues:
        findings.append(
            _plugin_finding(
                issue.severity,
                f"plugin_{issue.concept}",
                f"{where}: {issue.message}",
                file_path,
                issue.suggestion,
                metadata=dict(metadata) if metadata else None,
            )
        )
    return findings


def _lsp_command_findings(name: str, server: dict[str, Any], file_path: str) -> list[Finding]:
    """The MCP stdio command-form checks for one LSP server's ``command`` and ``args``, as ``plugin_lsp_*``.

    Claude Code starts an LSP server like a stdio MCP server (argv, no shell),
    so a shell ``-c`` program, shell metacharacters, inline credentials, and
    unpinned package runners get the same findings.
    """
    if not isinstance(server.get("command"), str):
        return []
    raw: list[Finding] = []
    mcp_validate_command(name, server, file_path, raw)
    mcp_validate_pinning(name, server, file_path, raw)
    prefix = f"mcpServers['{name}']: "
    return [
        _plugin_finding(
            finding.severity,
            f"plugin_lsp_{finding.check_name.removeprefix('mcp_')}",
            f"lspServers['{name}']: {finding.message.removeprefix(prefix)}",
            file_path,
            finding.suggestion,
            metadata={"plugin_component": {"type": "lsp", "name": name}},
        )
        for finding in raw
    ]


# --------------------------------------------------------------------------- #
# Markdown frontmatter                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Markdown:
    name: str | None
    description: str | None
    body: str
    frontmatter: dict[str, Any]


def parse_markdown(text: str) -> _Markdown:
    """Split optional ``---`` YAML frontmatter from a Markdown body (bounded YAML)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return _Markdown(None, None, text.strip(), {})
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            raw = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :]).strip()
            try:
                data = load_bounded_yaml(raw) if raw.strip() else {}
            except (StructuredDataError, ValueError):
                return _Markdown(None, None, body, {})
            if not isinstance(data, dict):
                return _Markdown(None, None, body, {})
            name = data.get("name") if isinstance(data.get("name"), str) else None
            description = data.get("description") if isinstance(data.get("description"), str) else None
            return _Markdown(name, description, body, data)
    return _Markdown(None, None, text.strip(), {})


# --------------------------------------------------------------------------- #
# Inventory builder                                                           #
# --------------------------------------------------------------------------- #
class _JsonSource(NamedTuple):
    """One loaded source of a JSON-config field (hooks, LSP servers, monitors, Codex apps)."""

    name: str  # the file, or "inline" / "inline[<index>]" for a value written in the manifest
    origin: Origin
    rel: str  # root-relative file that holds the config (the manifest for an inline value)
    config: Any  # the parsed config; None when the file could not be read or parsed safely


class _ResolvedPath(NamedTuple):
    """A declared component path that can be loaded (:meth:`_Builder._resolve_component`)."""

    rel: PurePosixPath  # contained root-relative path
    kind: str  # "file" or "dir"


class _Builder:
    """Builds the inventory of one manifest view: its components, findings, and hook and privilege records.

    :meth:`build` runs one phase per component type, and each section below
    holds one phase with its helpers. Declared paths are resolved with the
    view's format profile (:meth:`_resolve_declared`), and every read is
    bounded and never follows links (:class:`PluginRootReader`).
    :func:`build_plugin_inventory` builds a view per manifest and merges them.
    """

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any] | None,
        *,
        contained: bool,
        manifest_rel: str,
        allowed_private_hosts: Iterable[str],
        hook_allowed_urls: Iterable[str] = (),
        profile: FormatProfile = CLAUDE_PROFILE,
    ) -> None:
        self.reader = PluginRootReader(root)
        self.manifest = manifest if isinstance(manifest, dict) else None
        self.contained = contained
        self.profile = profile
        self.manifest_rel = manifest_rel
        self.allowed_private_hosts = tuple(allowed_private_hosts)
        self.hook_allowed_urls = tuple(hook_allowed_urls)
        self.inventory = PluginInventory()
        hook_root_prefixes, relative_hook_scripts = profile.root_prefixes, profile.relative_hook_scripts
        if profile.client_extensions:
            # Agent Plugins hooks live in client-extension namespaces (com.cursor/, ...) that each
            # client runs with its own root placeholder, so every known one is recognized.
            hook_root_prefixes = tuple(dict.fromkeys(p for known in PROFILES.values() for p in known.root_prefixes))
            relative_hook_scripts = any(known.relative_hook_scripts for known in PROFILES.values())
        self._hook_analyzer = HookAnalyzer(
            read_script=self._read_hook_script,
            hook_allowed_urls=self.hook_allowed_urls,
            allowed_private_hosts=self.allowed_private_hosts,
            root_prefixes=hook_root_prefixes,
            relative_scripts=relative_hook_scripts,
        )
        self._keys: dict[tuple[str, str, str], Component] = {}
        self._type_counts: Counter[str] = Counter()  # components listed per type, for the per-type cap
        self._truncated: set[str] = set()
        self._privilege_keys: set[tuple[str, str, str | None]] = set()
        # Runnable MCP servers without a declared read-only mode; mcp() sets them for the subagent checks.
        self._write_capable_servers: list[str] = []

    def build(self) -> PluginInventory:
        self.skills()
        self.rules()
        self.mcp()
        self.hooks()
        self.markdown_components("agent", "agents", self.profile.default_agents_dir)
        self.markdown_components("command", "commands", self.profile.default_commands_dir)
        self.lsp()
        self.markdown_components("output_style", "outputStyles", self.profile.default_output_styles_dir)
        self.monitors()
        self.settings()
        self.apps()
        self.extensions()
        self.openai_extension()
        self.env_files()
        return self.inventory

    # -- inventory bookkeeping --------------------------------------------- #
    @property
    def manifest_display(self) -> str:
        return self.reader.display(self.manifest_rel)

    def _add(self, component: Component) -> Component:
        key = (component.type, component.path or "", component.name)
        existing = self._keys.get(key)
        if existing is not None:
            if existing.origin != component.origin:
                existing.origin = "declared+packaged"
            return existing
        if self._type_counts[component.type] >= PLUGIN_COMPONENT_MAX_ITEMS:
            if component.type not in self._truncated:
                self._truncated.add(component.type)
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.LOW,
                        "plugin_component_scan_truncated",
                        f"More than {PLUGIN_COMPONENT_MAX_ITEMS} {component.type} components; the inventory lists "
                        "only the first ones",
                        self.reader.display("."),
                        "Split very large plugins; the static inventory is bounded.",
                    )
                )
            return component
        self._keys[key] = component
        self._type_counts[component.type] += 1
        self.inventory.components.append(component)
        return component

    def _broken(self, component_type: str, declared: DeclaredPath | None, raw: Any, problem: str) -> None:
        path = declared.rel.as_posix() if declared is not None and declared.rel is not None else None
        name = raw if isinstance(raw, str) else repr(raw)
        self._add(Component(component_type, name, "declared", path, _TYPE_SUPPORT[component_type], problem=problem))

    def _declared_field(self, field_name: str) -> Any:
        if self.manifest is None or not self.contained:
            return None
        return self.manifest.get(field_name)

    def _declared_values(self, field_name: str, value: Any) -> list[Any]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        self._check_item_count(f"'{field_name}'", len(values))
        return values[:PLUGIN_COMPONENT_MAX_ITEMS]

    def _check_item_count(self, what: str, count: int, rel: str | None = None) -> None:
        """HIGH finding when only the first ``PLUGIN_COMPONENT_MAX_ITEMS`` entries get checked."""
        if count <= PLUGIN_COMPONENT_MAX_ITEMS:
            return
        self.inventory.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_component_list_truncated",
                f"{what} has {count} entries; only the first {PLUGIN_COMPONENT_MAX_ITEMS} are inspected, so the "
                "rest are loaded without static checks",
                self.reader.display(rel or self.manifest_rel),
                f"Keep each declared component list or server map to at most {PLUGIN_COMPONENT_MAX_ITEMS} entries.",
            )
        )

    def _declared_replaces(self, field_name: str, value: Any) -> bool:
        """Whether a declared field suppresses default-folder discovery for its type.

        Codex drops a path it does not accept and loads the default location
        instead (:func:`~skillevaluator.plugin_formats.declared_value_replaces_default`).
        """
        return declared_value_replaces_default(self.profile, field_name, value)

    # -- declared paths and bounded reads ---------------------------------- #
    def _resolve_declared(
        self,
        field_name: str,
        raw: Any,
        *,
        kinds: tuple[str, ...] = ("file", "dir"),
        wrong_kind_field: str | None = None,
        style: bool = True,
        profile: FormatProfile | None = None,
    ) -> tuple[DeclaredPath | None, str]:
        """Normalize + classify one declared path and record its findings: ``(path, kind or problem)``.

        The second value is the path's kind when it is one of ``kinds``, and
        otherwise the problem: ``escape``, ``invalid``, ``missing``, or
        ``unsafe``. A file where the field takes a folder (or the reverse) is
        ``invalid``; its finding names ``wrong_kind_field`` when given.
        ``profile`` overrides the builder's format profile (the Codex rules of
        ``extensions["com.openai"]``).
        """
        profile = profile or self.profile
        if not isinstance(raw, str):
            finding = _path_problem_finding(
                self.reader, field_name, DeclaredPath(repr(raw), None), self.manifest_rel, "invalid"
            )
            self.inventory.findings.append(finding)
            return None, "invalid"
        declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
        if declared.problem is not None or declared.rel is None:
            problem = declared.problem if declared.problem == "escape" else "invalid"
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader, field_name, declared, self.manifest_rel, problem, reference=profile.reference
                )
            )
            return declared, problem
        if style and self.contained and profile.require_dot_relative and not declared.dot_relative:
            self.inventory.findings.append(
                _style_finding(self.reader, field_name, declared, self.manifest_rel, profile)
            )
        # Skill folders a client loads are scanned as skill units even in such a folder
        # (client_skill_dirs_outside_tree_scans), so only other components get this.
        if field_name != "skills" and (
            unscanned := _unscanned_path_finding(self.reader, field_name, declared, self.manifest_rel)
        ):
            self.inventory.findings.append(unscanned)
        kind = self.reader.kind(declared.rel)
        if kind == "missing":
            self.inventory.findings.append(
                _path_problem_finding(self.reader, field_name, declared, self.manifest_rel, "missing")
            )
            return declared, "missing"
        if kind in {"link", "special"}:
            self.inventory.findings.append(
                _path_problem_finding(self.reader, field_name, declared, self.manifest_rel, "unsafe", declared.rel)
            )
            return declared, "unsafe"
        if kind not in kinds:
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader, wrong_kind_field or field_name, declared, self.manifest_rel, "invalid"
                )
            )
            return declared, "invalid"
        return declared, kind

    def _resolve_component(
        self,
        component_type: str,
        field_name: str,
        raw: Any,
        *,
        kinds: tuple[str, ...] = ("file", "dir"),
        name: str | None = None,
        wrong_kind_field: str | None = None,
        style: bool = True,
    ) -> _ResolvedPath | None:
        """A declared component path that can be loaded, or ``None`` when it cannot.

        A path that cannot be loaded gets its finding (:meth:`_resolve_declared`)
        and a broken ``component_type`` component named ``name`` (by default
        the declared value) in its place.
        """
        declared, kind = self._resolve_declared(
            field_name, raw, kinds=kinds, wrong_kind_field=wrong_kind_field, style=style
        )
        if kind in kinds and declared is not None and declared.rel is not None:
            return _ResolvedPath(declared.rel, kind)
        self._broken(component_type, declared, raw if name is None else name, kind)
        return None

    def _list_dir(
        self, component_type: str, rel_dir: PurePosixPath, suffixes: tuple[str, ...] | None
    ) -> list[PurePosixPath] | None:
        try:
            return self.reader.list_files(rel_dir, suffixes=suffixes)
        except SecurePathError as exc:
            offending = exc.relative_path if exc.relative_path not in {"", "."} else ""
            location = (rel_dir / offending) if offending else rel_dir
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_unsafe",
                    f"{component_type} directory '{rel_dir.as_posix()}' contains an unsafe entry "
                    f"({exc}); it was not followed",
                    self.reader.display(location),
                    "Replace symlinks, hard links, and special files with regular contained files.",
                )
            )
            self._add(
                Component(
                    component_type,
                    rel_dir.as_posix(),
                    "packaged",
                    rel_dir.as_posix(),
                    _TYPE_SUPPORT[component_type],
                    problem="unsafe",
                )
            )
            return None

    def _read(self, rel: PurePosixPath, max_bytes: int = CONTENT_DEDUP_MAX_FILE_BYTES) -> str | None:
        try:
            return self.reader.read_text(rel, max_bytes)
        except (SecurePathError, OSError):
            self.inventory.unread_files += 1
            return None

    def _load_json(self, rel: PurePosixPath, field_name: str) -> Any:
        try:
            text = self.reader.read_text(rel, PLUGIN_CONFIG_MAX_BYTES, config=True)
            return load_bounded_json(text)
        except (SecurePathError, StructuredDataError, OSError) as exc:
            # Blocking: an unread hooks/LSP/monitor/settings config skips the
            # permission-bypass and env-override checks, so fail closed.
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_unreadable",
                    f"{field_name} config '{rel.as_posix()}' could not be read or parsed safely: {exc}",
                    self.reader.display(rel),
                    f"Keep '{rel.as_posix()}' a regular UTF-8 JSON file under {PLUGIN_CONFIG_MAX_BYTES} bytes.",
                )
            )
            return None

    def _json_sources(
        self,
        component_type: str,
        field_name: str,
        declared_value: Any,
        default: PurePosixPath | None,
        *,
        merge_default: bool = True,
        profile: FormatProfile | None = None,
    ) -> Iterator[_JsonSource]:
        """The loaded sources of a JSON-config field: its default file, inline values, and declared files.

        ``hooks`` and ``lspServers`` merge with their default file; monitors replace it
        (``merge_default=False``) when declared. For Codex, a declared value
        replaces the default only when Codex keeps it; otherwise the default
        file is loaded, as Codex does. A declared path that cannot be loaded
        is not yielded: it gets its finding and a broken ``component_type``
        component without a path. ``profile`` overrides the builder's format
        profile (the Codex rules of ``extensions["com.openai"]``).
        """
        profile = profile or self.profile
        declared_values = self._declared_values(field_name, declared_value)
        explicit: set[PurePosixPath] = set()
        for raw in declared_values:
            if isinstance(raw, str):
                declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
                if declared.rel is not None:
                    explicit.add(declared.rel)
        if merge_default:
            load_default = True
        elif profile.codex_path_rules:
            load_default = not declared_value_replaces_default(profile, field_name, declared_value)
        else:
            load_default = not declared_values
        default_kind = self.reader.kind(default) if load_default and default is not None else "missing"
        if default_kind == "file" and default not in explicit:
            config = self._load_json(default, field_name)
            yield _JsonSource(default.as_posix(), "packaged", default.as_posix(), config)
        elif default_kind in {"link", "special"} and default is not None:
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader,
                    field_name,
                    DeclaredPath(default.as_posix(), default),
                    self.manifest_rel,
                    "unsafe",
                    default,
                )
            )
        for index, raw in enumerate(declared_values):
            if isinstance(raw, dict | list) and not isinstance(raw, str):
                name = f"inline[{index}]" if len(declared_values) > 1 else "inline"
                yield _JsonSource(name, "declared", self.manifest_rel, raw)
                continue
            declared, kind = self._resolve_declared(field_name, raw, kinds=("file",), profile=profile)
            if kind != "file" or declared is None or declared.rel is None:
                self._broken(component_type, None, raw, kind)  # a broken config source has no path
                continue
            origin: Origin = "declared+packaged" if declared.rel == default else "declared"
            config = self._load_json(declared.rel, field_name)
            yield _JsonSource(declared.rel.as_posix(), origin, declared.rel.as_posix(), config)

    # -- skills ------------------------------------------------------------ #
    def skills(self) -> None:
        from skillevaluator.utils.helpers import find_bundled_plugin_skill_manifests

        declared_default = False
        declared_value = self.manifest.get("skills") if self.manifest is not None else None
        default_dir = self.profile.default_skills_dir or "skills"
        if self.contained and self.manifest is not None:
            for raw in self._declared_values("skills", declared_value):
                resolved = self._resolve_component("skill", "skills", raw, kinds=("dir",))
                if resolved is None:
                    continue
                if resolved.rel == PurePosixPath(default_dir):
                    declared_default = True
                    continue
                self._declared_skill_dir(resolved.rel)
        elif self.manifest is not None:
            for ref in _section_refs(self.manifest.get("skills")):
                self._add(Component("skill", _ref_label(ref), "declared", None, "evaluated"))

        if self.contained and self._declared_replaces("skills", declared_value) and not declared_default:
            return  # the declared paths replace skills/ discovery for this format
        try:
            manifests = find_bundled_plugin_skill_manifests(self.reader.root)
        except ValueError:
            manifests = []  # reported by the bundled-skill validator (bundled_skill_path_unsafe)
        if self.profile.skills_immediate_children_only:
            manifests = [manifest for manifest in manifests if len(manifest.relative_path.parts) == 2]
        if (
            not manifests
            and declared_value is None
            and self.profile.root_skill_fallback
            and self.reader.kind(PurePosixPath(default_dir)) == "missing"
        ):
            self._root_skill()
        for manifest in manifests[:PLUGIN_COMPONENT_MAX_ITEMS]:
            skill_dir = PurePosixPath("skills") / manifest.relative_path.parent.as_posix()
            origin: Origin = "declared+packaged" if declared_default else "packaged"
            component = self._add(
                Component("skill", manifest.relative_path.parent.as_posix(), origin, skill_dir.as_posix(), "evaluated")
            )
            self._skill_cost(component, skill_dir / manifest.relative_path.name)

    def _root_skill(self) -> None:
        """A root ``SKILL.md`` makes a single-skill plugin (Cursor) when no skills are declared or bundled."""
        for variant in SKILL_MANIFEST_VARIANTS:
            if self.reader.kind(PurePosixPath(variant)) == "file":
                component = self._add(Component("skill", self.reader.root.name, "packaged", ".", "evaluated"))
                self._skill_cost(component, PurePosixPath(variant))
                return

    def _declared_skill_dir(self, rel_dir: PurePosixPath) -> None:
        """A declared skills dir holds ``SKILL.md`` directly or ``<name>/SKILL.md`` folders.

        Codex searches a declared skills folder at any depth, like ``skills/``.
        A child named ``evals``, ``results``, or ``versions`` is a skill folder
        like any other here (clients load it). The whole-tree walk prunes such
        folder names, so Tier 1 scans these skills as their own units
        (:func:`client_skill_dirs_outside_tree_scans`).
        """
        if self.profile.skills_recursive and str(rel_dir) != ".":
            self._declared_skill_tree(rel_dir)
            return
        candidates: list[PurePosixPath] = []
        for variant in SKILL_MANIFEST_VARIANTS:
            if self.reader.kind(rel_dir / variant) == "file":
                candidates.append(rel_dir / variant)
                break
        if not candidates:
            candidates = self._child_skill_manifests(rel_dir)
        for manifest_rel in candidates:
            skill_dir = manifest_rel.parent
            name = skill_dir.name if str(skill_dir) != "." else self.reader.root.name
            component = self._add(Component("skill", name, "declared", skill_dir.as_posix(), "evaluated"))
            self._skill_cost(component, manifest_rel)

    def _child_skill_manifests(self, rel_dir: PurePosixPath) -> list[PurePosixPath]:
        """The ``SKILL.md`` of each skill folder directly in a declared skills dir.

        A linked child folder or skill manifest is reported and not followed.
        """
        start = self.reader.root if str(rel_dir) == "." else self.reader.root / rel_dir.as_posix()
        try:
            with os.scandir(start) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            entries = []
        manifests: list[PurePosixPath] = []
        for entry in entries[:CONTENT_DEDUP_MAX_DISCOVERED_PATHS]:
            if entry.name.startswith(".") or (
                entry.name in SCAN_EXCLUDED_DIRS and entry.name not in SCAN_ARTIFACT_DIRS
            ):
                continue  # hidden folders, VCS, virtualenv, package, and bytecode caches
            child = PurePosixPath(entry.name) if str(rel_dir) == "." else rel_dir / entry.name
            if str(rel_dir) == "." and entry.name == "skills":
                continue  # the default skills/ scan covers it
            if entry.is_symlink():
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "plugin_component_path_unsafe",
                        f"skills directory entry '{child.as_posix()}' is a symlink; it was not followed",
                        self.reader.display(child),
                        "Replace linked skill directories with regular contained directories.",
                    )
                )
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            for variant in SKILL_MANIFEST_VARIANTS:
                kind = self.reader.kind(child / variant)
                if kind == "file":
                    manifests.append(child / variant)
                    break
                if kind in {"link", "special"}:
                    self.inventory.findings.append(
                        _path_problem_finding(
                            self.reader,
                            "skills",
                            DeclaredPath((child / variant).as_posix(), child / variant),
                            self.manifest_rel,
                            "unsafe",
                            child / variant,
                        )
                    )
                    break
        return manifests

    def _declared_skill_tree(self, rel_dir: PurePosixPath) -> None:
        """Every skill below a declared folder, at any depth (Codex recursive discovery).

        The folder is searched like ``skills/``: an ``evals``, ``results``, or
        ``versions`` child is searched as a skill folder (Tier 1 scans it as its
        own unit), and a ``SKILL.md`` deeper in such a folder, inside a skill's
        own evaluation output or snapshots, is HIGH
        ``plugin_skill_in_unscanned_folder``, as it is under ``skills/``.
        """
        from skillevaluator.utils.helpers import (
            find_bundled_plugin_skill_manifests,
            find_unscanned_plugin_skill_manifests,
        )

        try:
            manifests = find_bundled_plugin_skill_manifests(self.reader.root, rel_dir.as_posix())
        except ValueError as exc:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_unsafe",
                    f"skills directory '{rel_dir.as_posix()}' contains an unsafe entry ({exc}); it was not followed",
                    self.reader.display(rel_dir),
                    "Replace symlinks, hard links, and special files with regular contained files.",
                )
            )
            self._add(
                Component("skill", rel_dir.as_posix(), "declared", rel_dir.as_posix(), "evaluated", problem="unsafe")
            )
            return
        for manifest in manifests[:PLUGIN_COMPONENT_MAX_ITEMS]:
            parent = PurePosixPath(manifest.relative_path.parent.as_posix())
            skill_dir = rel_dir / parent if str(parent) != "." else rel_dir
            name = parent.as_posix() if str(parent) != "." else rel_dir.name
            component = self._add(Component("skill", name, "declared", skill_dir.as_posix(), "evaluated"))
            self._skill_cost(component, skill_dir / manifest.relative_path.name)
        for manifest_rel in find_unscanned_plugin_skill_manifests(self.reader.root, rel_dir.as_posix()):
            self._flag_unscanned_skill(rel_dir, manifest_rel)

    def _flag_unscanned_skill(self, rel_dir: PurePosixPath, manifest_rel: PurePosixPath) -> None:
        """HIGH for a skill deep in a skill's own artifact folder that Codex loads from declared folder ``rel_dir``.

        Skills under ``skills/`` are reported by the bundled-skill validator, so
        they are not repeated here.
        """
        if rel_dir.parts[:1] == ("skills",):
            return
        if _in_unscanned_folder(manifest_rel.parent):
            self.inventory.findings.append(_unscanned_skill_finding(self.reader, manifest_rel))

    def _skill_cost(self, component: Component, manifest_rel: PurePosixPath) -> None:
        text = self._read(manifest_rel)
        if text is None:
            return
        parsed = parse_markdown(text)
        # Claude Code pre-approves a skill's allowed-tools and registers its frontmatter hooks while it is active.
        self._privileges(
            component, parsed.frontmatter, self.reader.display(manifest_rel), source_file=manifest_rel.as_posix()
        )
        always = len(parsed.name or "") + len(parsed.description or "")
        component.cost = CostRow(
            "skill",
            component.name,
            always,
            len(parsed.body),
            "always-on: frontmatter name + description; on-demand: SKILL.md body",
        )

    # -- rules ------------------------------------------------------------- #
    def rules(self) -> None:
        declared_default = False
        section = self.manifest.get("rules") if self.manifest is not None else None
        if self.contained and section is not None:
            for raw in self._declared_values("rules", section):
                if isinstance(raw, str) and "::" in raw:
                    self._add(Component("rule", _ref_label(raw), "declared", None, "evaluated"))
                    continue
                resolved = self._resolve_component("rule", "rules", raw, style=False)
                if resolved is None:
                    continue
                if resolved.rel == PurePosixPath(self.profile.default_rules_dir or "rules"):
                    declared_default = True
                elif resolved.kind == "dir":
                    self._rule_dir(resolved.rel, "declared")
                else:
                    self._rule_file(resolved.rel, "declared")
        elif self.manifest is not None and not self.contained:
            # Bundle-reference refs are inventoried by label only; resolving them
            # (and flagging dangling ones) is the dependency-resolution report's job.
            for ref in _section_refs(section):
                self._add(Component("rule", _ref_label(ref), "declared", None, "evaluated"))
        default_name = self.profile.default_rules_dir
        if default_name is None:
            return
        if self.contained and self._declared_replaces("rules", section) and not declared_default:
            return  # the declared paths replace rules/ discovery for this format
        default_dir = PurePosixPath(default_name)
        if self.reader.kind(default_dir) == "dir":
            self._rule_dir(default_dir, "declared+packaged" if declared_default else "packaged")
        elif self.reader.kind(default_dir) == "link":
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader,
                    "rules",
                    DeclaredPath(default_name, default_dir),
                    self.manifest_rel,
                    "unsafe",
                    default_dir,
                )
            )
            self._add(Component("rule", default_name, "packaged", default_name, "evaluated", problem="unsafe"))

    def _rule_dir(self, rel_dir: PurePosixPath, origin: Origin, *, support: Support = "evaluated") -> None:
        files = self._list_dir("rule", rel_dir, self.profile.rule_suffixes)
        for rel in files or []:
            self._rule_file(rel, origin, name=rel.relative_to(rel_dir).as_posix(), support=support)

    def _rule_file(
        self, rel: PurePosixPath, origin: Origin, *, name: str | None = None, support: Support = "evaluated"
    ) -> None:
        component = self._add(Component("rule", name or rel.name, origin, rel.as_posix(), support))
        text = self._read(rel)
        if text is not None:
            component.cost = CostRow(
                "rule",
                component.name,
                0,
                len(text.strip()),
                "on-demand: rule body (embedded in the Tier 3 wrapper SKILL.md)",
            )

    # -- MCP servers ------------------------------------------------------- #
    def mcp(self) -> None:
        collection = collect_mcp_declarations(
            self.reader,
            self.manifest,
            contained=self.contained,
            manifest_rel=self.manifest_rel,
            allowed_private_hosts=self.allowed_private_hosts,
            profile=self.profile,
        )
        self.inventory.mcp = collection
        self.inventory.findings.extend(collection.findings)
        self.inventory.findings.extend(collection.server_findings)
        sources_by_name: dict[str, set[str]] = {}
        for declaration in collection.declarations:
            sources_by_name.setdefault(declaration.name, set()).add(declaration.source)
        effective = collection.effective
        # Tool lists are unknown statically, so any runnable server without a read-only mode may write.
        self._write_capable_servers = [
            declaration.name
            for declaration in effective
            if declaration.runnable and not mcp_server_is_read_only(declaration.config)
        ]
        for declaration in effective:
            if declaration.source == "mcp_json" and not self.contained:
                support: Support = "static_only"
            else:
                support = "evaluated" if declaration.runnable else "static_only"
            sources = sources_by_name.get(declaration.name, set())
            if "mcp_json" in sources and sources - {"mcp_json"}:
                origin: Origin = "declared+packaged"  # a declared server replaced the .mcp.json one
            else:
                origin = "packaged" if declaration.source == "mcp_json" else "declared"
            component = self._add(
                Component("mcp", declaration.name, origin, declaration.file, support, mcp=declaration)
            )
            component.cost = CostRow(
                "mcp",
                declaration.name,
                0,
                0,
                "tool schemas are not known statically (not counted)",
            )
        for raw, path, problem in collection.broken_sources:
            self._add(Component("mcp", raw, "declared", path, "static_only", problem=problem))
        for raw, path in collection.bundles:
            self._add(Component("mcp", raw, "declared", path, "unsupported", bundle=True))

    # -- hooks ------------------------------------------------------------- #
    def hooks(self) -> None:
        declared_value = self._declared_field("hooks")
        default = PurePosixPath(self.profile.default_hooks_file) if self.profile.default_hooks_file else None
        merge_default = not self.profile.declared_replaces_default
        for source in self._json_sources("hook", "hooks", declared_value, default, merge_default=merge_default):
            self._hook(source)

    def _hook(self, source: _JsonSource, *, profile: FormatProfile | None = None) -> None:
        """One hooks config: a ``hook`` component, its permission-bypass findings, and its risk records.

        ``profile`` overrides the builder's format profile for the hook dialect.
        """
        name, origin, rel, config = source
        profile = profile or self.profile
        self._add(Component("hook", name, origin, rel, "unsupported"))
        if config is not None:
            self.inventory.findings.extend(
                _override_findings(
                    [*permission_bypass_issues(config), *permission_mode_flag_issues(config)],
                    self.reader.display(rel),
                    where=f"hooks ({name})",
                    component=("hook", name),
                )
            )
            # Cursor and Agent Plugins client extensions (com.cursor/, com.github.copilot/)
            # list handlers flat under each event; nested matcher groups pass through. Each
            # client's own event names, approval output, and command keys apply.
            config = _nested_hook_groups(config)
            analysis = self._hook_analyzer.analyze(
                config,
                source=name,
                file=rel,
                display=self.reader.display(rel),
                dialect=hook_dialect(profile.manifest_type, rel),
            )
            self.inventory.hook_records.extend(analysis.records)
            self.inventory.findings.extend(analysis.findings)

    def _read_hook_script(self, rel: PurePosixPath) -> str | None:
        """Text of a script a hook runs from the plugin root (``${CLAUDE_PLUGIN_ROOT}``), for the hook risk analyzer.

        Returns the whole file (at most ``MAX_SCRIPT_BYTES``) of a regular file,
        hard-linked or not, decoded as UTF-8 with replacement characters (so one
        non-UTF-8 byte does not hide the script), or ``None`` when nothing or a
        directory is at ``rel``. Raises :class:`HookScriptUnreadable` when
        something is there that cannot be read safely: a link, a special file, a
        file over the size limit, or an exhausted read budget.
        """
        kind = self.reader.kind(rel, allow_hard_links=True)
        if kind in {"missing", "dir"}:
            return None
        if kind == "link":
            raise HookScriptUnreadable(
                f"'{rel.as_posix()}' is (or passes through) a symlink or reparse point; it was not followed"
            )
        if kind != "file":
            raise HookScriptUnreadable(f"'{rel.as_posix()}' is a special file, or cannot be inspected")
        try:
            raw = self.reader.read_script_bytes(rel, MAX_SCRIPT_BYTES)
        except (SecurePathError, OSError) as exc:
            raise HookScriptUnreadable(f"'{rel.as_posix()}' could not be read safely: {exc}") from exc
        return raw.decode("utf-8-sig", errors="replace")

    # -- markdown component types (agents, commands, output styles) -------- #
    def _suffixes(self, component_type: str) -> tuple[str, ...]:
        if component_type == "agent":
            return self.profile.agent_suffixes
        if component_type == "command":
            return self.profile.command_suffixes
        return (".md",)

    def markdown_components(self, component_type: str, field_name: str, default_dir: str | None) -> None:
        declared_value = self._declared_field(field_name)
        suffixes = self._suffixes(component_type)
        # Codex loads the default folder when it drops the declared value.
        codex_default = self.profile.codex_path_rules and not self._declared_replaces(field_name, declared_value)
        if declared_value is None or codex_default:
            if default_dir is not None and self.reader.kind(PurePosixPath(default_dir)) == "dir":
                self._markdown_dir(component_type, PurePosixPath(default_dir), "packaged")
            elif default_dir is not None and self.reader.kind(PurePosixPath(default_dir)) == "link":
                self._list_dir(component_type, PurePosixPath(default_dir), suffixes)
            if declared_value is None:
                return
        if component_type == "command" and isinstance(declared_value, dict):
            self._command_map(declared_value)
            return
        for raw in self._declared_values(field_name, declared_value):
            resolved = self._resolve_component(component_type, field_name, raw)
            if resolved is None:
                continue
            rel = resolved.rel
            origin: Origin = (
                "declared+packaged"
                if default_dir is not None and (rel.parts[:1] == (default_dir,) or rel == PurePosixPath(default_dir))
                else "declared"
            )
            if resolved.kind == "dir":
                self._markdown_dir(component_type, rel, origin)
            else:
                self._markdown_file(component_type, rel, origin)

    def _markdown_dir(self, component_type: str, rel_dir: PurePosixPath, origin: Origin) -> None:
        for rel in self._list_dir(component_type, rel_dir, self._suffixes(component_type)) or []:
            self._markdown_file(component_type, rel, origin)

    def _markdown_file(self, component_type: str, rel: PurePosixPath, origin: Origin) -> None:
        text = self._read(rel)
        parsed = parse_markdown(text) if text is not None else None
        name = (parsed.name if parsed is not None and parsed.name else None) or rel.stem
        component = self._add(Component(component_type, name, origin, rel.as_posix(), _TYPE_SUPPORT[component_type]))
        if parsed is not None:
            component.cost = _markdown_cost(component_type, component.name, parsed)
            self._privileges(component, parsed.frontmatter, self.reader.display(rel))

    def _command_map(self, commands: dict[str, Any]) -> None:
        """The object form of ``commands``: each entry has a ``source`` Markdown file or inline ``content``."""
        self._check_item_count("'commands' map", len(commands))
        for index, (command_name, entry) in enumerate(commands.items()):
            if index >= PLUGIN_COMPONENT_MAX_ITEMS:
                break
            if not isinstance(entry, dict) or ("source" in entry) == ("content" in entry):
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "plugin_component_path_invalid",
                        f"commands[{command_name!r}] must set exactly one of 'source' or 'content'",
                        self.manifest_display,
                        "Give each command map entry either a 'source' file path or inline 'content'.",
                    )
                )
                self._add(Component("command", str(command_name), "declared", None, "unsupported", problem="invalid"))
            elif "content" in entry:
                self._inline_command(command_name, entry)
            else:
                self._command_file(command_name, entry)

    def _inline_command(self, command_name: str, entry: dict[str, Any]) -> None:
        description = entry.get("description") if isinstance(entry.get("description"), str) else ""
        content = entry.get("content") if isinstance(entry.get("content"), str) else ""
        component = self._add(Component("command", str(command_name), "declared", None, "unsupported"))
        component.cost = CostRow(
            "command",
            component.name,
            len(description),
            len(content),
            "always-on: description; on-demand: inline content",
        )
        self._privileges(component, {}, self.manifest_display, entry=entry)

    def _command_file(self, command_name: str, entry: dict[str, Any]) -> None:
        resolved = self._resolve_component(
            "command",
            f"commands[{command_name!r}].source",
            entry.get("source"),
            kinds=("file",),
            name=str(command_name),
            wrong_kind_field="commands",
        )
        if resolved is None:
            return
        description = entry.get("description") if isinstance(entry.get("description"), str) else ""
        text = self._read(resolved.rel)
        component = self._add(
            Component("command", str(command_name), "declared", resolved.rel.as_posix(), "unsupported")
        )
        if text is not None:
            parsed = parse_markdown(text)
            component.cost = CostRow(
                "command",
                component.name,
                len(description or parsed.description or ""),
                len(parsed.body),
                "always-on: description; on-demand: command body",
            )
            self._privileges(component, parsed.frontmatter, self.reader.display(resolved.rel), entry=entry)
        else:
            self._privileges(component, {}, self.manifest_display, entry=entry)

    # -- tool and permission grants (skills, agents, commands) ------------- #
    def _privileges(
        self,
        component: Component,
        frontmatter: dict[str, Any],
        display: str,
        *,
        source_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        if (component.type, component.name, component.path) in self._privilege_keys:
            return
        if component.type == "agent":
            plugin_name = self.manifest.get("name") if self.manifest is not None else None
            record, findings = analyze_agent(
                component.name,
                component.path,
                frontmatter,
                display,
                write_capable_mcp=self._write_capable_servers,
                plugin_name=plugin_name if isinstance(plugin_name, str) else None,
            )
        elif component.type == "command":
            record, findings = analyze_command(component.name, component.path, frontmatter, display, **kwargs)
        elif component.type == "skill":
            record, findings = analyze_skill(component.name, component.path, frontmatter, display)
        else:
            return
        self.inventory.privilege_records.append(record)
        self._privilege_keys.add((record.type, record.name, record.path))
        self.inventory.findings.extend(findings)
        if component.type in {"command", "skill"} and "hooks" in frontmatter:
            file = source_file or component.path or self.manifest_rel
            self._frontmatter_hooks(frontmatter.get("hooks"), file, display)

    def _frontmatter_hooks(self, config: Any, file: str, display: str) -> None:
        """Hooks in a skill's or command's frontmatter: Claude Code registers them while it is active."""
        if not isinstance(config, dict):
            return
        source = f"{file}#hooks"
        self.inventory.findings.extend(
            _override_findings(
                [*permission_bypass_issues(config), *permission_mode_flag_issues(config)],
                display,
                where=f"hooks ({source})",
                component=("hook", source),
            )
        )
        analysis = self._hook_analyzer.analyze(config, source=source, file=file, display=display, dialect=CLAUDE_HOOKS)
        self.inventory.hook_records.extend(analysis.records)
        self.inventory.findings.extend(analysis.findings)

    # -- LSP servers ------------------------------------------------------- #
    def lsp(self) -> None:
        declared_value = self._declared_field("lspServers")
        default = PurePosixPath(self.profile.default_lsp_file) if self.profile.default_lsp_file else None
        if default is None and declared_value is None:
            return
        for name, origin, rel, config in self._json_sources("lsp", "lspServers", declared_value, default):
            servers = config.get("lspServers", config) if isinstance(config, dict) else None
            if not isinstance(servers, dict) or not servers:
                self._add(Component("lsp", name, origin, rel, "unsupported"))
                continue
            self._check_item_count(f"lspServers map in '{rel}'", len(servers), rel)
            for server_name, server in list(servers.items())[:PLUGIN_COMPONENT_MAX_ITEMS]:
                self._add(Component("lsp", str(server_name), origin, rel, "unsupported"))
                where = f"lspServers['{server_name}']"
                display = self.reader.display(rel)
                component = ("lsp", str(server_name))
                self.inventory.findings.extend(
                    _override_findings(
                        [*permission_bypass_issues(server), *permission_mode_flag_issues(server)],
                        display,
                        where=where,
                        component=component,
                    )
                )
                if isinstance(server, dict):
                    self.inventory.findings.extend(
                        _override_findings(
                            env_override_issues(server.get("env")), display, where=where, component=component
                        )
                    )
                    self.inventory.findings.extend(_lsp_command_findings(str(server_name), server, display))

    # -- monitors ---------------------------------------------------------- #
    def monitors(self) -> None:
        if self.profile.default_monitors_file is None:
            return  # only Claude Code plugins declare monitors
        default = PurePosixPath(self.profile.default_monitors_file)
        declared_value = None
        if self.manifest is not None and self.contained:
            experimental = self.manifest.get("experimental")
            if isinstance(experimental, dict) and "monitors" in experimental:
                declared_value = experimental.get("monitors")
            elif "monitors" in self.manifest:
                declared_value = self.manifest.get("monitors")
        field_name = "experimental.monitors"
        if isinstance(declared_value, list) and all(isinstance(item, dict) for item in declared_value):
            sources: Iterable[_JsonSource] = [_JsonSource("inline", "declared", self.manifest_rel, declared_value)]
        elif declared_value is not None:
            sources = self._json_sources("monitor", field_name, declared_value, default, merge_default=False)
        else:
            sources = self._json_sources("monitor", field_name, None, default)
        for name, origin, rel, config in sources:
            entries = config.get("monitors", config) if isinstance(config, dict) else config
            if not isinstance(entries, list) or not entries:
                self._add(Component("monitor", name, origin, rel, "unsupported"))
                continue
            for index, entry in enumerate(entries[:PLUGIN_COMPONENT_MAX_ITEMS]):
                entry_name = (
                    entry.get("name") if isinstance(entry, dict) and isinstance(entry.get("name"), str) else None
                )
                monitor_name = entry_name or f"{name}[{index}]"
                self._add(Component("monitor", monitor_name, origin, rel, "unsupported"))
                self._monitor_command(monitor_name, entry, rel)
            self.inventory.findings.extend(
                _override_findings(
                    [*permission_bypass_issues(entries), *permission_mode_flag_issues(entries)],
                    self.reader.display(rel),
                    where=f"monitors ({name})",
                )
            )

    def _monitor_command(self, monitor_name: str, entry: Any, rel: str) -> None:
        """A monitor runs its ``command`` unsandboxed for the whole session (like a hook), and every line it
        prints goes to the model, so it gets the hook command checks plus a context-injection note."""
        if not isinstance(entry, dict) or not isinstance(entry.get("command"), str):
            return
        handler: dict[str, Any] = {"type": "command", "command": entry["command"]}
        if isinstance(entry.get("args"), list):
            handler["args"] = entry["args"]
        analysis = self._hook_analyzer.analyze(
            {MONITOR_EVENT: [{"hooks": [handler]}]},
            source=f"monitor:{monitor_name}",
            file=rel,
            display=self.reader.display(rel),
            dialect=MONITOR_HOOKS,
        )
        self.inventory.hook_records.extend(analysis.records)
        self.inventory.findings.extend(analysis.findings)

    # -- shipped settings -------------------------------------------------- #
    def settings(self) -> None:
        for rel in (PurePosixPath(path) for path in self.profile.settings_files):
            kind = self.reader.kind(rel)
            if kind in {"missing", "dir"}:
                continue
            if kind in {"link", "special"}:
                self.inventory.findings.append(
                    _path_problem_finding(
                        self.reader, "settings", DeclaredPath(rel.as_posix(), rel), self.manifest_rel, "unsafe", rel
                    )
                )
                self._add(
                    Component("settings", rel.as_posix(), "packaged", rel.as_posix(), "unsupported", problem="unsafe")
                )
                continue
            self._add(Component("settings", rel.as_posix(), "packaged", rel.as_posix(), "unsupported"))
            config = self._load_json(rel, "settings")
            if config is not None:
                self._settings_checks(config, rel.as_posix(), rel.as_posix())
        if self.contained and self.manifest is not None and "settings" in self.manifest:
            inline = self.manifest.get("settings")
            self._add(Component("settings", "plugin.json#settings", "declared", self.manifest_rel, "unsupported"))
            self._settings_checks(inline, self.manifest_rel, "plugin.json#settings")

    def _settings_checks(self, config: Any, rel: str, component_name: str) -> None:
        if not isinstance(config, dict):
            return
        for finding in _settings_findings(config, rel, self.reader.display(rel)):
            finding.metadata["plugin_component"] = {"type": "settings", "name": component_name}
            self.inventory.findings.append(finding)

    # -- Codex apps (connectors) ------------------------------------------- #
    def apps(self) -> None:
        """Inventory Codex app (connector) declarations: ``apps`` or the default ``.app.json``.

        An app is a hosted connector that SkillEvaluator cannot stage, so every
        declared alias is inventoried as ``unsupported``.
        """
        declared_value = self._declared_field("apps")
        default_name = self.profile.default_apps_file
        if default_name is None and declared_value is None:
            return
        default = PurePosixPath(default_name) if default_name else None
        self._add_apps(self._json_sources("app", "apps", declared_value, default, merge_default=False))

    def _add_apps(self, sources: Iterable[_JsonSource]) -> None:
        """One ``app`` component per alias in each ``.app.json`` ``apps`` map; a file without aliases is one."""
        for name, origin, rel, config in sources:
            apps = config.get("apps") if isinstance(config, dict) else None
            if not isinstance(apps, dict) or not apps:
                self._add(Component("app", name, origin, rel, "unsupported"))
                continue
            for alias in list(apps)[:PLUGIN_COMPONENT_MAX_ITEMS]:
                self._add(Component("app", str(alias), origin, rel, "unsupported"))

    # -- Agent Plugins client extensions ----------------------------------- #
    def extensions(self) -> None:
        """Inventory Agent Plugins client-extension namespaces.

        A namespace is declared in the manifest ``extensions`` object and/or
        ships a top-level reverse-domain directory (spec section 8). Extensions
        have no portable semantics, so each namespace is ``unsupported``. Inside
        a namespace directory, the documented client layouts (``agents/``,
        ``commands/``, ``rules/``, and ``hooks/hooks.json``) are inventoried as
        their own types, all ``unsupported``, and hooks get the bypass checks.
        """
        if not self.profile.client_extensions:
            return
        declared = self.manifest.get("extensions") if self.manifest is not None else None
        namespaces: dict[str, Origin] = {}
        if isinstance(declared, dict):
            for key in list(declared)[:PLUGIN_COMPONENT_MAX_ITEMS]:
                namespaces[str(key)] = "declared"
        try:
            with os.scandir(self.reader.root) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            entries = []
        directories: set[str] = set()
        for entry in entries[:CONTENT_DEDUP_MAX_DISCOVERED_PATHS]:
            if not _NAMESPACE_DIR_RE.match(entry.name) or self.reader.kind(PurePosixPath(entry.name)) != "dir":
                continue
            directories.add(entry.name)
            namespaces[entry.name] = "declared+packaged" if entry.name in namespaces else "packaged"
        for namespace, origin in namespaces.items():
            path = namespace if namespace in directories else self.manifest_rel
            self._add(Component("extension", namespace, origin, path, "unsupported"))
            if namespace in directories:
                self._namespace_components(PurePosixPath(namespace))

    def _namespace_components(self, base: PurePosixPath) -> None:
        """The documented client layouts in a namespace folder, inventoried as their own (unsupported) types."""
        for component_type, folder in (("agent", "agents"), ("command", "commands")):
            if self.reader.kind(base / folder) == "dir":
                self._markdown_dir(component_type, base / folder, "packaged")
        if self.reader.kind(base / "rules") == "dir":
            self._rule_dir(base / "rules", "packaged", support="unsupported")
        hooks_file = base / "hooks" / "hooks.json"
        if self.reader.kind(hooks_file) == "file":
            config = self._load_json(hooks_file, "hooks")
            self._hook(_JsonSource(hooks_file.as_posix(), "packaged", hooks_file.as_posix(), config))

    # -- OpenAI settings of an Agent Plugins manifest ---------------------- #
    def openai_extension(self) -> None:
        """Hooks and apps in ``extensions["com.openai"]`` of an Agent Plugins manifest.

        OpenAI documents this object as the place for Codex hooks (a path, an
        array of paths, an inline hooks object, or an array of them) and app
        mappings (a path to an ``.app.json``), read with the Codex path rules.
        Its hooks get the same bypass and risk checks as any other hooks.
        """
        if not self.profile.client_extensions or self.manifest is None:
            return
        extensions = self.manifest.get("extensions")
        openai = extensions.get("com.openai") if isinstance(extensions, dict) else None
        if not isinstance(openai, dict):
            return
        # Codex path rules, messages, and hook dialect for these fields.
        if openai.get("hooks") is not None:
            for source in self._json_sources("hook", "hooks", openai.get("hooks"), None, profile=CODEX_PROFILE):
                if source.rel == self.manifest_rel:  # an inline hooks object
                    source = source._replace(name=f"extensions.com.openai.hooks:{source.name}")
                self._hook(source, profile=CODEX_PROFILE)
        if openai.get("apps") is not None:
            self._add_apps(self._json_sources("app", "apps", openai.get("apps"), None, profile=CODEX_PROFILE))

    # -- shipped .env files ------------------------------------------------ #
    def env_files(self) -> None:
        hits, complete = _find_env_files(self.reader.root)
        if not complete:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.LOW,
                    "plugin_env_scan_incomplete",
                    f"the shipped .env file scan stopped early (more than {CONTENT_DEDUP_MAX_DISCOVERED_PATHS} "
                    "entries, or a directory could not be opened); later directories were not checked",
                    self.reader.display("."),
                    "Keep generated or unreadable directories out of the plugin package.",
                )
            )
        for rel in hits[:_MAX_ENV_FILE_FINDINGS]:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_env_file_shipped",
                    f"plugin ships an environment file '{rel.as_posix()}'; .env files commonly hold credentials "
                    "(contents are not printed; secret scanning covers the values)",
                    self.reader.display(rel),
                    "Do not package .env files; ship a .env.example with placeholder values instead.",
                )
            )
        if len(hits) > _MAX_ENV_FILE_FINDINGS:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_env_file_shipped",
                    f"plugin ships {len(hits)} .env files; only the first {_MAX_ENV_FILE_FINDINGS} are listed",
                    self.reader.display("."),
                    "Do not package .env files.",
                )
            )


def _settings_findings(config: dict[str, Any], rel: str, display: str) -> list[Finding]:
    """The checks on shipped Claude Code settings: a settings file, or the manifest's inline ``settings``."""
    findings: list[Finding] = []
    permissions = config.get("permissions")
    if isinstance(permissions, dict):
        mode = permissions.get("defaultMode")
        if mode == "bypassPermissions":
            findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_settings_bypass_permissions",
                    f"shipped settings '{rel}' sets permissions.defaultMode to 'bypassPermissions', which "
                    "disables every tool-approval prompt",
                    display,
                    "Remove permissions.defaultMode (Claude Code ignores it from plugins, but shipping it is a "
                    "red flag and it applies if the file is copied into a project).",
                )
            )
        elif isinstance(mode, str) and mode in PERMISSIVE_PERMISSION_MODES:
            findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_settings_permission_mode",
                    f"shipped settings '{rel}' sets permissions.defaultMode to {mode!r}, which approves some "
                    "tool calls (file edits, or what a classifier allows) without a prompt",
                    display,
                    "Remove permissions.defaultMode; let the user choose the permission mode (Claude Code "
                    "ignores it from plugins, but it applies if the file is copied into a project).",
                )
            )
        allow = permissions.get("allow")
        if isinstance(allow, list):
            broad = sorted(
                {
                    str(rule)
                    for rule in allow  # every rule: already bounded by the structured-data limits
                    if isinstance(rule, str) and is_broad_allow_rule(rule)
                }
            )
            if broad:
                findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "plugin_settings_broad_allow",
                        f"shipped settings '{rel}' pre-approves unrestricted tools or interpreter Bash rules "
                        f"that run any command {broad[:8]}",
                        display,
                        "Remove blanket allow rules such as Bash / Bash(*) / Bash(python3:*); scope permissions "
                        "to exact commands.",
                    )
                )
    if config.get("enableAllProjectMcpServers") is True:
        findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "plugin_settings_auto_approve",
                f"shipped settings '{rel}' sets enableAllProjectMcpServers, auto-approving every project MCP server",
                display,
                "Remove enableAllProjectMcpServers; let users approve MCP servers explicitly.",
            )
        )
    findings.extend(_override_findings(env_override_issues(config.get("env")), display, where=rel))
    findings.extend(
        _override_findings(
            [*permission_bypass_issues(config), *permission_mode_flag_issues(config)], display, where=rel
        )
    )
    return findings


def _markdown_cost(component_type: str, name: str, parsed: _Markdown) -> CostRow:
    if component_type == "output_style":
        forced = parsed.frontmatter.get("force-for-plugin") is True
        if forced:
            return CostRow(component_type, name, len(parsed.body), 0, "always-on: force-for-plugin output style body")
        return CostRow(component_type, name, 0, len(parsed.body), "on-demand: output style body when selected")
    label = "agent" if component_type == "agent" else "command"
    basis = (
        "always-on: description; on-demand: subagent prompt (loads in the subagent's own context)"
        if label == "agent"
        else "always-on: description; on-demand: command body"
    )
    return CostRow(component_type, name, len(parsed.description or ""), len(parsed.body), basis)


def _section_refs(section: Any) -> list[Any]:
    if isinstance(section, dict):
        refs = section.get("refs")
    else:
        refs = section
    if not isinstance(refs, list):
        return []
    return refs[:PLUGIN_COMPONENT_MAX_ITEMS]


def _ref_label(ref: Any) -> str:
    """Same label Tier 3 uses for unresolved refs (canonical ID or trailing name)."""
    canonical = normalize_ref(ref)
    if canonical:
        return canonical
    if isinstance(ref, dict) and isinstance(ref.get("path"), str):
        return ref["path"].strip().split("/")[-1] or repr(ref)
    return str(ref)


def is_env_file(name: str) -> bool:
    """Whether *name* is a ``.env`` or ``.env.*`` file, in any letter case, that is not a template.

    Tier 1 flags these names and native Tier 3 staging leaves them out of the
    plugin copy, so both use this one rule.
    """
    folded = name.casefold()
    return folded == ".env" or (folded.startswith(".env.") and folded.rsplit(".", 1)[-1] not in _ENV_TEMPLATE_SUFFIXES)


def _scan_env_entries(
    entries: list[os.DirEntry[str]], rel_dir: PurePosixPath, depth: int, hits: list[PurePosixPath], budget: int
) -> tuple[list[str], int]:
    """Record ``.env`` hits among one directory's entries; return (subdirectories to visit, budget left)."""
    children: list[str] = []
    for entry in entries:
        budget -= 1
        if budget < 0:
            break
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if not is_dir:
            if is_env_file(entry.name):
                hits.append(rel_dir / entry.name)
        elif entry.name not in SCAN_EXCLUDED_DIRS and depth < 32:
            children.append(entry.name)
    return children, budget


def _find_env_files(root: Path) -> tuple[list[PurePosixPath], bool]:
    """Names-only walk for shipped ``.env`` / ``.env.*`` files; files are never opened.

    Returns the hits and whether the walk was complete (``False`` when the entry
    budget ran out or a directory could not be opened or listed).

    On POSIX every directory is opened relative to its parent descriptor with
    ``O_NOFOLLOW``, so a directory swapped for a symlink is never listed. The walk
    is depth-first and opens a subdirectory only after its previous sibling's
    subtree is closed, so open descriptors grow with depth (at most 33), not with
    the number of sibling directories.
    """
    hits: list[PurePosixPath] = []
    budget = CONTENT_DEDUP_MAX_DISCOVERED_PATHS
    complete = True
    if os.name == "posix" and os.open in os.supports_dir_fd and os.scandir in os.supports_fd:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            root_fd = os.open(root, flags)
        except OSError:
            return hits, False
        # Frames: (fd, rel_dir, depth, subdirectories left to visit; None until listed).
        stack: list[tuple[int, PurePosixPath, int, list[str] | None]] = [(root_fd, PurePosixPath(), 0, None)]
        try:
            while stack:
                fd, rel_dir, depth, children = stack[-1]
                if children is None:
                    try:
                        with os.scandir(fd) as iterator:
                            entries = sorted(iterator, key=lambda item: item.name)
                    except OSError:
                        complete = False
                        entries = []
                    found, budget = _scan_env_entries(entries, rel_dir, depth, hits, budget)
                    children = found[::-1]  # pop() visits them in name order
                    stack[-1] = (fd, rel_dir, depth, children)
                if budget < 0 or not children:
                    stack.pop()
                    os.close(fd)
                    continue
                name = children.pop()
                try:
                    child_fd = os.open(name, flags, dir_fd=fd)
                except OSError:
                    complete = False
                    continue
                stack.append((child_fd, rel_dir / name, depth + 1, None))
        finally:
            for fd, *_frame in stack:
                os.close(fd)
        return sorted(hits), complete and budget >= 0

    pending: list[tuple[Path, PurePosixPath, int]] = [(root, PurePosixPath(), 0)]
    while pending and budget >= 0:
        directory, rel_dir, depth = pending.pop()
        try:
            if stat_is_link_or_reparse(directory.lstat()):
                continue
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            complete = False
            continue
        children, budget = _scan_env_entries(entries, rel_dir, depth, hits, budget)
        pending.extend((directory / name, rel_dir / name, depth + 1) for name in children)
    return sorted(hits), complete and budget >= 0


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #
def manifest_rel_for(manifest_path: Path, root: Path) -> str:
    """Root-relative POSIX spelling of a located manifest path."""
    from skillevaluator.plugin_manifest import manifest_relative_path

    relative = manifest_relative_path(manifest_path)
    if relative is not None and len(relative.parts) > 1:
        return relative.as_posix()
    try:
        return manifest_path.relative_to(root).as_posix()
    except ValueError:
        return manifest_path.name


def parsed_additional_manifests(location: PluginManifestLocation) -> list[tuple[str, str, dict[str, Any]]]:
    """Alias of :meth:`~skillevaluator.plugin_manifest.PluginManifestLocation.parsed_additional`, kept for importers."""
    return location.parsed_additional()


def plugin_inventory_for_root(root: Path) -> PluginInventory | None:
    """The component inventory of the plugin at ``root``, read the way the audits read it (``None`` if unsafe).

    The selected manifest and every additional client manifest are parsed
    (an oversize or non-UTF-8 client manifest leniently), then
    :func:`build_plugin_inventory` adds the cross-client views. No policy is
    applied; the schema check reports every finding.
    """
    from skillevaluator.plugin_formats import manifest_syntax
    from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest

    try:
        located = locate_plugin_manifest(root)
    except PluginManifestPathError:
        return None
    additional: list[tuple[str, str, dict[str, Any]]] = []
    if located is None:
        manifest_type, data, manifest_rel = None, None, CLAUDE_PROFILE.manifest_path
    else:
        manifest_type = located.manifest_type
        try:
            text = located.read_text(encoding="utf-8-sig")
            parsed = load_bounded_json(text) if manifest_syntax(manifest_type) == "json" else load_bounded_yaml(text)
        except (PluginManifestPathError, StructuredDataError, ValueError, RecursionError):
            parsed = None
        data = parsed if isinstance(parsed, dict) else None
        manifest_rel = manifest_rel_for(located.path, located.root)
        additional = located.parsed_additional()
    return build_plugin_inventory(
        root,
        data,
        contained=manifest_type in PLUGIN_CONTAINED_MANIFEST_TYPES,
        manifest_rel=manifest_rel,
        manifest_type=manifest_type,
        additional=additional,
    )


def client_skill_dirs_outside_tree_scans(root: Path) -> list[PurePosixPath]:
    """Skill folders a client loads that the whole-tree walk would prune, plugin-root relative.

    Tier 1 whole-tree scans prune ``evals/``, ``results/``, and ``versions/``
    (and dotted forms) everywhere except one level under ``skills/``. A
    declared skills folder can still load a skill from such a folder, for
    example ``my-skills/evals/`` or a declared ``./evals/``. These folders are
    scanned as their own skill units instead (``tier1.commands``). A
    ``SKILL.md`` deep inside a skill's own artifact folder is not listed; it
    stays HIGH ``plugin_skill_in_unscanned_folder``.
    """
    inventory = plugin_inventory_for_root(root)
    if inventory is None:
        return []
    found: dict[str, PurePosixPath] = {}
    for component in inventory.components:
        if component.type != "skill" or component.problem is not None or component.path in {None, "", "."}:
            continue
        rel = PurePosixPath(str(component.path))
        if _in_unscanned_folder(rel):
            found.setdefault(rel.as_posix(), rel)
    return [found[key] for key in sorted(found)]


def build_plugin_inventory(
    root: Path,
    manifest: dict[str, Any] | None,
    *,
    contained: bool,
    manifest_rel: str,
    allowed_private_hosts: Iterable[str] = (),
    hook_allowed_urls: Iterable[str] = (),
    manifest_type: str | None = None,
    additional: Iterable[tuple[str, str, dict[str, Any] | None]] = (),
) -> PluginInventory:
    """Build the static component inventory for a plugin root (never follows links).

    ``hook_allowed_urls`` is the policy's ``hooks.allowed_urls`` allowlist for
    HTTP hook endpoints. ``manifest_type`` selects the contained format's
    profile (Claude Code when omitted). ``additional`` lists the other supported
    manifests in the root as ``(manifest_type, manifest_rel, parsed manifest)``.
    Their components, static findings, and hook and privilege risk records are
    merged in, so an MCP server, hook, subagent, or command that only another
    client loads is still checked. A merged component that the selected manifest
    does not also declare carries ``declared_by`` and is never ``evaluated``. An
    additional bundle-reference manifest (``agent_plugin.yml`` beside
    ``agent_plugin.yaml``) is read like a selected one: its ``skills``/``rules``
    refs and ``mcp`` list, never component paths.

    Last, the cross-client pass (:func:`_cross_client_views`) checks what
    clients load from the same folder through a manifest that is not their
    own: Claude Code ``--plugin-dir`` reads its default locations
    (``hooks/hooks.json``, ``.mcp.json``, ``agents/``, ...) in a folder without
    a ``.claude-plugin/plugin.json``, and Codex reads a Claude Code or Cursor
    manifest with its own path rules and defaults. What only such a view loads
    carries ``declared_by`` (the manifest that client reads, or the default
    location) and ``loaded_by`` (the client), and is checked but not staged.
    """
    allowed_private_hosts = tuple(allowed_private_hosts)
    hook_allowed_urls = tuple(hook_allowed_urls)

    def _inventory(
        data: dict[str, Any] | None, data_type: str | None, rel: str, *, is_contained: bool
    ) -> PluginInventory:
        return _Builder(
            root,
            normalized_component_manifest(data_type or "", data) if is_contained else data,
            contained=is_contained,
            manifest_rel=rel,
            allowed_private_hosts=allowed_private_hosts,
            hook_allowed_urls=hook_allowed_urls,
            profile=profile_for(data_type) if is_contained else CLAUDE_PROFILE,
        ).build()

    inventory = _inventory(manifest, manifest_type, manifest_rel, is_contained=contained)
    for extra_type, extra_rel, extra_manifest in additional:
        is_contained = extra_type in PLUGIN_CONTAINED_MANIFEST_TYPES
        extra = _inventory(extra_manifest, extra_type, extra_rel, is_contained=is_contained)
        _merge_additional(inventory, extra, extra_rel)
    selected_type = manifest_type or (PLUGIN_CONTAINED_MANIFEST_TYPE if contained else PLUGIN_MANIFEST_TYPE)
    loaded: list[tuple[str | None, dict[str, Any] | None, str]] = [(selected_type, manifest, manifest_rel)]
    loaded.extend((extra_type, extra_manifest, extra_rel) for extra_type, extra_rel, extra_manifest in additional)
    for profile, data, rel, loaded_by in _cross_client_views(loaded):
        view = _Builder(
            root,
            normalized_component_manifest(profile.manifest_type, data),
            contained=True,
            manifest_rel=rel,
            allowed_private_hosts=allowed_private_hosts,
            hook_allowed_urls=hook_allowed_urls,
            profile=profile,
        ).build()
        _merge_cross_client(
            inventory,
            view,
            root,
            declared_by=rel if data is not None else None,
            loaded_by=loaded_by,
            manifest_files={loaded_rel for _type, _data, loaded_rel in loaded},
        )
    return inventory


def _cross_client_views(
    loaded: Iterable[tuple[str | None, dict[str, Any] | None, str]],
) -> list[tuple[FormatProfile, dict[str, Any] | None, str, str]]:
    """How clients other than the manifests' own load this folder: ``(profile, manifest, rel, loaded_by)``.

    Each manifest is inventoried with its own client's rules. Two clients also
    load a folder through a manifest that is not their own:

    * Claude Code ``--plugin-dir`` loads any folder. Without a
      ``.claude-plugin/plugin.json`` it reads its default locations
      (``skills/``, ``agents/``, ``commands/``, ``hooks/hooks.json``,
      ``.mcp.json``, ``.lsp.json``, ``output-styles/``, monitors, settings).
    * Codex takes the first of an Agent Plugins root ``plugin.json``,
      ``.codex-plugin/plugin.json``, ``.claude-plugin/plugin.json``, and
      ``.cursor-plugin/plugin.json``. When that is a Claude Code or Cursor
      manifest, Codex reads it with its own path rules and defaults
      (``.mcp.json``, ``hooks/hooks.json``, recursive ``skills/``).
    """
    by_type: dict[str, tuple[dict[str, Any] | None, str]] = {}
    for manifest_type, data, rel in loaded:
        if manifest_type is not None:
            by_type.setdefault(manifest_type, (data, rel))
    views: list[tuple[FormatProfile, dict[str, Any] | None, str, str]] = []
    if PLUGIN_CONTAINED_MANIFEST_TYPE not in by_type and PLUGIN_MANIFEST_TYPE not in by_type:
        views.append(
            (
                _CLAUDE_DEFAULTS_VIEW,
                None,
                CLAUDE_PROFILE.manifest_path,
                "Claude Code (--plugin-dir reads its default locations in a folder without "
                f"{CLAUDE_PROFILE.manifest_path})",
            )
        )
    if not {PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE, PLUGIN_CODEX_MANIFEST_TYPE} & by_type.keys():
        for manifest_type in (PLUGIN_CONTAINED_MANIFEST_TYPE, PLUGIN_CURSOR_MANIFEST_TYPE):
            if manifest_type in by_type:
                data, rel = by_type[manifest_type]
                views.append(
                    (
                        CODEX_PROFILE,
                        data if isinstance(data, dict) else {},
                        rel,
                        f"Codex (reads {rel} with its own path rules and default locations)",
                    )
                )
                break
    return views


# Claude Code's default locations, read as another format's folder. MCP files
# there are often written for Codex (streamable_http, http_headers), so the
# Codex field mapping is applied before the risk checks.
_CLAUDE_DEFAULTS_VIEW = dataclasses.replace(CLAUDE_PROFILE, mcp_dialect="codex")
# SkillEvaluator's rules/ folder is its own concept; other clients do not load it.
_CROSS_CLIENT_SKIPPED_TYPES = frozenset({"rule", "extension"})


def _merge_cross_client(
    inventory: PluginInventory,
    view: PluginInventory,
    root: Path,
    *,
    declared_by: str | None,
    loaded_by: str,
    manifest_files: Iterable[str] = (),
) -> None:
    """Merge what another client's view loads that the inventory has not checked yet.

    Such a component is checked like any other (its findings and hook and
    privilege records are merged) but is never ``evaluated`` and not staged by
    Tier 3. It carries ``declared_by`` (the manifest that client reads, or its
    own default location when it reads none) and ``loaded_by``. Findings,
    hook records, and privilege records about files the inventory already read
    (including the manifests) are dropped: those files were checked with the
    rules of the manifest that loads them, and a second client's format rules
    would only repeat or contradict them.
    """
    covered = _covered_paths(inventory) | _MANIFEST_PATHS | set(manifest_files)
    components = [component for component in view.components if component.type not in _CROSS_CLIENT_SKIPPED_TYPES]
    for component in components:
        component.declared_by = declared_by or component.path or loaded_by
        component.loaded_by = loaded_by
    extra = PluginInventory(
        components=components,
        findings=[
            finding for finding in view.findings if _finding_rel_path(str(finding.file_path or ""), root) not in covered
        ],
        hook_records=[record for record in view.hook_records if record.file not in covered],
        privilege_records=[record for record in view.privilege_records if record.path not in covered],
    )
    _merge_additional(inventory, extra, None)


def _covered_paths(inventory: PluginInventory) -> set[str]:
    """Root-relative files and folders the inventory already read (components, hooks, MCP sources)."""
    covered = {component.path for component in inventory.components if component.path}
    covered.update(record.file for record in inventory.hook_records)
    covered.update(declaration.file for declaration in inventory.mcp.declarations)
    covered.update(path for _raw, path, _problem in inventory.mcp.broken_sources if path)
    covered.update(component.mcp.file for component in inventory.components if component.mcp is not None)
    return covered


def _finding_key(finding: Finding) -> tuple[str, str, str, str]:
    return (finding.category, finding.check_name, str(finding.file_path or ""), finding.message)


def _nested_hook_groups(config: Any) -> Any:
    """Return a hooks config in the nested shape the hook risk analyzer reads.

    Cursor, and Agent Plugins client extensions such as ``com.cursor/`` and
    ``com.github.copilot/``, list handlers directly under each event
    (``{"hooks": {"stop": [{"command": ...}]}}``); Claude Code and Codex nest
    them in matcher groups (``{"matcher": ..., "hooks": [{"type": "command",
    ...}]}``). Each flat handler becomes a one-handler group with its
    ``matcher``; the handler ``type`` defaults to ``command``, as in Cursor. A
    group (an entry with a ``hooks`` list) is left unchanged, so every format's
    config passes through here.
    """
    events = config.get("hooks") if isinstance(config, dict) and isinstance(config.get("hooks"), dict) else config
    if not isinstance(events, dict):
        return config
    nested: dict[str, Any] = {}
    for event, handlers in events.items():
        if not isinstance(handlers, list):
            nested[event] = handlers
            continue
        groups: list[Any] = []
        for handler in handlers:
            if isinstance(handler, dict) and not isinstance(handler.get("hooks"), list):
                inner = {key: value for key, value in handler.items() if key != "matcher"}
                inner.setdefault("type", "command")
                group: dict[str, Any] = {"hooks": [inner]}
                if "matcher" in handler:
                    group["matcher"] = handler["matcher"]
                groups.append(group)
            else:
                groups.append(handler)
        nested[event] = groups
    return {"hooks": nested}


def _merge_key(component: Component) -> tuple[str, str, str]:
    """Identity of a component across manifests.

    A bundle manifest's ``mcp`` entry lives in the manifest file itself, so the
    same server name in ``agent_plugin.yaml`` and ``agent_plugin.yml`` is one
    component.
    """
    if component.mcp is not None and component.mcp.source == "agent_plugin_yaml":
        return (component.type, "agent_plugin_yaml:", component.name)
    return (component.type, component.path or "", component.name)


def _merge_additional(inventory: PluginInventory, extra: PluginInventory, declared_by: str | None) -> None:
    """Merge an additional manifest's inventory without duplicating shared components, findings, or risk records.

    ``declared_by`` names the additional manifest; ``None`` keeps the
    ``declared_by`` each component already carries (the cross-client pass).
    """
    keys = {_merge_key(component) for component in inventory.components}
    for component in extra.components:
        key = _merge_key(component)
        if key in keys:
            continue
        keys.add(key)
        if declared_by is not None:
            component.declared_by = declared_by
        component.cost = None  # the context estimate describes the selected manifest only
        if component.support == "evaluated":
            component.support = "static_only"
        inventory.components.append(component)
    seen = {_finding_key(finding) for finding in inventory.findings}
    for finding in extra.findings:
        key = _finding_key(finding)
        if key not in seen:
            seen.add(key)
            inventory.findings.append(finding)
    hook_keys = {(record.id, record.file) for record in inventory.hook_records}
    for record in extra.hook_records:
        if (record.id, record.file) not in hook_keys:
            hook_keys.add((record.id, record.file))
            inventory.hook_records.append(record)
    privilege_keys = {(record.type, record.name, record.path) for record in inventory.privilege_records}
    for record in extra.privilege_records:
        if (record.type, record.name, record.path) not in privilege_keys:
            privilege_keys.add((record.type, record.name, record.path))
            inventory.privilege_records.append(record)


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
    if absolute == root_text:
        return "."
    if absolute.startswith(root_text.rstrip(os.sep) + os.sep):
        return Path(os.path.relpath(absolute, root_text)).as_posix()
    if Path(text).is_absolute():
        return None
    # Validators that report paths relative to the plugin root.
    return PurePosixPath(text.replace("\\", "/")).as_posix()


def attribute_findings(components: list[Component], findings: Iterable[Finding], root: Path) -> None:
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
    mcp_by_key: dict[tuple[str, str], Component] = {}
    for component in components:
        if component.type == "mcp" and component.mcp is not None:
            mcp_by_key[(component.mcp.name, component.mcp.file)] = component
    mcp_by_name = {component.name: component for component in components if component.type == "mcp"}
    by_type_name: dict[tuple[str, str], Component] = {}
    by_name: dict[str, Component] = {}
    for component in components:
        by_type_name.setdefault((component.type, component.name), component)
        by_name.setdefault(component.name, component)
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
            target = by_type_name.get((str(tagged.get("type")), str(tagged.get("name"))))
            if target is not None:
                target.findings += 1
                continue
        ref = metadata.get("plugin_component_ref")
        if isinstance(ref, str) and ref in by_name:
            by_name[ref].findings += 1
            continue
        if rel is None or rel == ".":
            continue
        for component in by_path:
            path = component.path or ""
            if rel == path or rel.startswith(path.rstrip("/") + "/"):
                component.findings += 1
                break


def refresh_component_finding_counts(results: Iterable[Any]) -> None:
    """Recount per-component findings across every Tier 1 result of a plugin run.

    The plugin schema result carries the inventory in ``metadata['plugin']``;
    other validators (security, secrets, hygiene, ...) report on files inside the
    same plugin root, so their findings are attributed by path prefix as well.
    """
    results = list(results)
    for result in results:
        plugin_meta = result.metadata.get("plugin") if isinstance(result.metadata, dict) else None
        if not isinstance(plugin_meta, dict):
            continue
        inventory = plugin_meta.get("component_inventory")
        root = plugin_meta.get("root")
        if not isinstance(inventory, dict) or not isinstance(root, str):
            continue
        rows = inventory.get("components")
        if not isinstance(rows, list):
            continue
        components = [
            Component(
                row.get("type", ""),
                row.get("name", ""),
                row.get("origin", "packaged"),
                row.get("path"),
                row.get("support", "unsupported"),
            )
            for row in rows
            if isinstance(row, dict)
        ]
        for component, row in zip(components, rows, strict=False):
            if component.type == "mcp" and isinstance(row, dict) and row.get("path"):
                component.mcp = McpDeclaration(component.name, None, "inline", str(row["path"]))
        all_findings = [finding for other in results for finding in getattr(other, "findings", [])]
        attribute_findings(components, all_findings, Path(root))
        for component, row in zip(components, rows, strict=False):
            if isinstance(row, dict):
                row["findings"] = component.findings


# --------------------------------------------------------------------------- #
# Tier 3 coverage                                                             #
# --------------------------------------------------------------------------- #
def coverage_row(component: Component, state: CoverageState, reason: str) -> dict[str, Any]:
    return {
        "type": component.type,
        "name": component.name,
        "origin": component.origin,
        "path": component.path,
        "state": state,
        "reason": reason,
    }


def summarize_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = dict.fromkeys(COVERAGE_STATES, 0)
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {
        "components": rows,
        "counts": counts,
        "not_evaluated": sum(1 for row in rows if row["state"] not in EVALUATED_COVERAGE_STATES),
    }


def problem_reason(component: Component) -> str:
    return {
        "missing": "declared path does not exist",
        "escape": "declared path escapes the plugin root",
        "unsafe": "path is a symlink, hard link, or special file and was not followed",
        "invalid": "declaration is invalid for this component type",
    }.get(component.problem or "", "component could not be loaded")
