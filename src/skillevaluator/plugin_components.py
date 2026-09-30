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
:class:`~skillevaluator.utils.secure_fs.SecureRoot` with a byte bound.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    PLUGIN_COMPONENT_MAX_ITEMS,
    PLUGIN_CONFIG_MAX_BYTES,
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.deduplication.plugin.ref_utils import normalize_ref
from skillevaluator.models.result import Finding, Severity
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, discover_secure_files, stat_is_link_or_reparse
from skillevaluator.utils.structured_data import (
    StructuredDataError,
    StructuredDataLimitError,
    load_bounded_json,
    load_bounded_yaml,
)
from skillevaluator.validators.mcp_static import (
    CATEGORY as MCP_CATEGORY,
)
from skillevaluator.validators.mcp_static import (
    McpPinning,
    OverrideIssue,
    classify_mcp_pinning,
    env_override_issues,
    permission_bypass_issues,
    redacted_url,
    validate_mcp_server_declaration,
)

PLUGIN_CATEGORY = "PLUGIN_SCHEMA"

ComponentType = Literal[
    "skill", "rule", "mcp", "hook", "agent", "command", "lsp", "output_style", "monitor", "settings"
]
Support = Literal["evaluated", "static_only", "unsupported"]
Origin = Literal["declared", "packaged", "declared+packaged"]
McpSource = Literal["inline", "mcp_json", "path_ref", "agent_plugin_yaml"]
CoverageState = Literal["staged", "not_staged", "unsupported", "unavailable", "invalid"]

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
COVERAGE_STATES: tuple[str, ...] = ("staged", "not_staged", "unsupported", "unavailable", "invalid")
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
}
_CONTEXT_COST_TYPES = frozenset({"skill", "rule", "mcp", "agent", "command", "output_style"})

MCP_JSON = PurePosixPath(".mcp.json")
LSP_JSON = PurePosixPath(".lsp.json")
HOOKS_JSON = PurePosixPath("hooks/hooks.json")
MONITORS_JSON = PurePosixPath("monitors/monitors.json")
_SETTINGS_FILES = (
    PurePosixPath("settings.json"),
    PurePosixPath(".claude/settings.json"),
    PurePosixPath(".claude/settings.local.json"),
)
_MCP_BUNDLE_SUFFIXES = (".mcpb", ".dxt")
_PLUGIN_ROOT_PREFIXES = ("${CLAUDE_PLUGIN_ROOT}/", "${CLAUDE_PLUGIN_ROOT}")
_ENV_TEMPLATE_SUFFIXES = frozenset({"example", "sample", "template", "dist", "defaults", "tmpl"})
_BROAD_ALLOW_RULES = frozenset({"bash", "bash(*)", "bash(:*)", "bash(*:*)", "*"})
_MAX_ENV_FILE_FINDINGS = 20
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_MANIFEST_PATHS = frozenset(
    {f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}", "agent_plugin.yaml", "agent_plugin.yml"}
)


# --------------------------------------------------------------------------- #
# Records                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class McpDeclaration:
    """One MCP server declaration and where it came from."""

    name: str
    config: Any
    source: McpSource
    file: str  # root-relative POSIX path of the declaring file

    @property
    def kind(self) -> str | None:
        if not isinstance(self.config, dict):
            return None
        for kind in ("command", "url", "provider"):
            if kind in self.config:
                return kind
        return None

    @property
    def runnable(self) -> bool:
        return isinstance(self.config, dict) and bool(self.config.get("command") or self.config.get("url"))

    @property
    def transport(self) -> str | None:
        if not isinstance(self.config, dict):
            return None
        raw = self.config.get("transport", self.config.get("type"))
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return "stdio" if self.kind == "command" else None

    def pinning(self) -> McpPinning:
        return classify_mcp_pinning(self.config)


@dataclass
class McpCollection:
    """Every MCP declaration a plugin makes, in Claude Code load order."""

    declarations: list[McpDeclaration] = field(default_factory=list)
    # Config-source findings (paths, files, duplicates); per-server static checks are separate.
    findings: list[Finding] = field(default_factory=list)
    server_findings: list[Finding] = field(default_factory=list)
    # Declared config sources that could not be loaded: (raw ref, root-relative path or None, problem).
    broken_sources: list[tuple[str, str | None, str]] = field(default_factory=list)
    # .mcpb/.dxt bundles: (raw ref, root-relative path or None for URLs).
    bundles: list[tuple[str, str | None]] = field(default_factory=list)

    @property
    def effective(self) -> list[McpDeclaration]:
        """Declarations after Claude Code's merge: a later same-name server replaces an earlier one.

        ``agent_plugin.yaml`` ``mcp`` entries are a list keyed by ``(name, provider)``
        and are kept as declared.
        """
        merged: dict[str, McpDeclaration] = {}
        listed: list[McpDeclaration] = []
        for declaration in self.declarations:
            if declaration.source == "agent_plugin_yaml":
                listed.append(declaration)
            else:
                merged[declaration.name] = declaration
        return [*merged.values(), *listed]

    @property
    def blocking_source_findings(self) -> list[Finding]:
        """Blocking config-source findings (the per-server checks run again at staging time)."""
        return [finding for finding in self.findings if finding.severity in (Severity.CRITICAL, Severity.HIGH)]


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "origin": self.origin,
            "path": self.path,
            "support": self.support,
            "findings": self.findings,
        }


@dataclass
class PluginInventory:
    """Static component inventory, its findings, and the context-cost estimate."""

    components: list[Component] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    mcp: McpCollection = field(default_factory=McpCollection)
    notes: list[str] = field(default_factory=list)
    unread_files: int = 0

    def of_type(self, component_type: str) -> list[Component]:
        return [component for component in self.components if component.type == component_type]

    def component_inventory(self) -> dict[str, Any]:
        counts = dict.fromkeys(COMPONENT_TYPES, 0)
        for component in self.components:
            counts[component.type] = counts.get(component.type, 0) + 1
        unsupported = sorted({c.type for c in self.components if c.support == "unsupported"})
        return {
            "components": [component.to_dict() for component in self.components],
            "counts": counts,
            "unsupported_types_present": unsupported,
        }

    def mcp_summary(self) -> dict[str, Any]:
        servers: list[dict[str, Any]] = []
        for declaration in self.mcp.effective:
            pin = declaration.pinning()
            servers.append(
                {
                    "name": declaration.name,
                    "source": declaration.source,
                    "kind": declaration.kind,
                    "transport": declaration.transport,
                    "pinned": pin.pinned,
                    "pin_detail": pin.detail,
                }
            )
        return {"servers": servers, "pinning": mcp_pinning_summary(self.mcp.effective)}

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
        """The C1 keys merged into ``ValidationResult.metadata['plugin']``."""
        return {
            "component_inventory": self.component_inventory(),
            "mcp": self.mcp_summary(),
            "context_cost": self.context_cost(),
        }


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


def mcp_pinning_summary(declarations: Iterable[McpDeclaration]) -> dict[str, Any]:
    """C1 ``mcp.pinning`` / C2 ``mcp_pinning``: counts plus ``pinned / (pinned + unpinned)``.

    ``ratio`` is ``None`` when no declaration runs a package (nothing to pin).
    """
    counts = {"pinned": 0, "unpinned": 0, "not_applicable": 0}
    total = 0
    for declaration in declarations:
        total += 1
        counts[declaration.pinning().status] += 1
    applicable = counts["pinned"] + counts["unpinned"]
    return {
        "total": total,
        **counts,
        "ratio": round(counts["pinned"] / applicable, 4) if applicable else None,
    }


# --------------------------------------------------------------------------- #
# Root-bounded reads                                                          #
# --------------------------------------------------------------------------- #
PathKind = Literal["missing", "file", "dir", "link", "special"]


@dataclass(frozen=True)
class DeclaredPath:
    raw: str
    rel: PurePosixPath | None
    problem: str | None = None  # empty | escape | invalid
    dot_relative: bool = True


def normalize_declared_path(raw: str) -> DeclaredPath:
    """Normalize one manifest path to a contained root-relative POSIX path.

    Claude Code requires ``./``-relative paths (``"."``/``"./"`` names the root).
    ``${CLAUDE_PLUGIN_ROOT}/`` prefixes are treated as the root. Absolute paths,
    home-relative paths, drive letters, and ``..`` segments are escapes.
    """
    text = raw.strip()
    if not text:
        return DeclaredPath(raw, None, "empty")
    if "\x00" in text or len(text) > 4096:
        return DeclaredPath(raw, None, "invalid")
    normalized = text.replace("\\", "/")
    dot_relative = normalized in {".", "./"} or normalized.startswith("./")
    for prefix in _PLUGIN_ROOT_PREFIXES:
        if normalized.startswith(prefix):
            normalized = "./" + normalized[len(prefix) :].lstrip("/")
            dot_relative = True
            break
    if normalized.startswith(("/", "~")) or _WINDOWS_DRIVE_RE.match(normalized):
        return DeclaredPath(raw, None, "escape")
    if "${" in normalized or normalized.startswith("$"):
        return DeclaredPath(raw, None, "invalid")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return DeclaredPath(raw, None, "escape")
    rel = PurePosixPath(*parts) if parts else PurePosixPath(".")
    return DeclaredPath(raw, rel, None, dot_relative)


class PluginRootReader:
    """No-follow classification and bounded reads beneath one plugin root.

    Security-relevant JSON configs (hooks, LSP, monitors, settings, MCP) draw on
    their own read budget, so large skill or rule files read for the context-cost
    estimate cannot starve them.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))  # noqa: PTH100 - lexical, never resolved
        self.bytes_read = 0
        self.config_bytes_read = 0

    def display(self, rel: PurePosixPath | str) -> str:
        rel_text = str(rel)
        return str(self.root) if rel_text in {"", "."} else str(self.root / rel_text)

    def kind(self, rel: PurePosixPath) -> PathKind:
        """Classify ``rel`` with ``lstat`` on every component (links are never followed)."""
        if not rel.parts or str(rel) == ".":
            return "dir"
        current = self.root
        parts = rel.parts
        for index, part in enumerate(parts):
            current = current / part
            try:
                metadata = current.lstat()
            except (FileNotFoundError, NotADirectoryError):
                return "missing"
            except OSError:
                return "special"
            if stat_is_link_or_reparse(metadata):
                return "link"
            if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                return "missing"
            if index == len(parts) - 1:
                if stat.S_ISDIR(metadata.st_mode):
                    return "dir"
                if stat.S_ISREG(metadata.st_mode):
                    return "file" if getattr(metadata, "st_nlink", 1) == 1 else "special"
                return "special"
        return "special"

    def read_text(self, rel: PurePosixPath, max_bytes: int, *, config: bool = False) -> str:
        """Bounded, anchored, no-follow UTF-8 read; raises :class:`SecurePathError`.

        ``config`` charges the read to the separate config budget.
        """
        used = self.config_bytes_read if config else self.bytes_read
        budget = "config" if config else "inventory"
        remaining = CONTENT_DEDUP_MAX_TOTAL_BYTES - used
        if remaining <= 0:
            raise SecurePathError("total_size_limit", f"Plugin {budget} read budget exhausted.")
        try:
            with SecureRoot(self.root) as secure_root:
                raw, _metadata = secure_root.read_bytes(Path(*rel.parts), min(max_bytes, remaining))
        except SecurePathError as exc:
            if exc.code == "file_size_limit" and remaining < max_bytes:
                raise SecurePathError(
                    "total_size_limit", f"Plugin {budget} read budget exhausted.", relative_path=rel.as_posix()
                ) from exc
            raise
        if config:
            self.config_bytes_read += len(raw)
        else:
            self.bytes_read += len(raw)
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SecurePathError(
                "invalid_text_encoding", f"File is not valid UTF-8: {rel.as_posix()}", relative_path=rel.as_posix()
            ) from exc

    def list_files(self, rel_dir: PurePosixPath, *, suffixes: tuple[str, ...] | None = None) -> list[PurePosixPath]:
        """Securely list regular files below a contained directory (raises on links)."""
        start = self.root if str(rel_dir) == "." else self.root / rel_dir.as_posix()

        def _selected(relative: Path) -> bool:
            return suffixes is None or relative.name.lower().endswith(suffixes)

        files = discover_secure_files(
            start,
            selected=_selected,
            excluded_dirs=SCAN_EXCLUDED_DIRS,
            max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
            allow_context_alias=False,
        )
        base = PurePosixPath() if str(rel_dir) == "." else rel_dir
        return [base / file.relative_path.as_posix() for file in files]


# --------------------------------------------------------------------------- #
# Findings helpers                                                            #
# --------------------------------------------------------------------------- #
def _plugin_finding(
    severity: Severity,
    check_name: str,
    message: str,
    file_path: str,
    suggestion: str,
    *,
    category: str = PLUGIN_CATEGORY,
    metadata: dict[str, Any] | None = None,
) -> Finding:
    return Finding(
        category=category,
        severity=severity,
        check_name=check_name,
        message=message,
        file_path=file_path,
        suggestion=suggestion,
        metadata=metadata or {},
    )


def _path_problem_finding(
    reader: PluginRootReader,
    field_name: str,
    declared: DeclaredPath,
    manifest_rel: str,
    problem: str,
    rel: PurePosixPath | None = None,
) -> Finding:
    """HIGH finding for a declared component path that is missing, escapes, or is unsafe."""
    raw = declared.raw
    # Lets the inventory attribute the finding to the (broken) component it names.
    metadata = {"plugin_component_ref": raw}
    if problem == "escape":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_escape",
            f"'{field_name}' path {raw!r} is absolute or escapes the plugin root",
            reader.display(manifest_rel),
            "Use a './'-relative path that stays inside the plugin root (no '..', no absolute or home paths).",
            metadata=metadata,
        )
    if problem == "missing":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_missing",
            f"'{field_name}' path {raw!r} does not exist in the plugin",
            reader.display(manifest_rel),
            "Ship the referenced file/directory inside the plugin root or remove the declaration.",
            metadata=metadata,
        )
    if problem == "unsafe":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_unsafe",
            f"'{field_name}' path {raw!r} is (or passes through) a symlink, hard link, or special file; "
            "it was not followed",
            reader.display(rel if rel is not None else manifest_rel),
            "Replace links with regular files and directories contained in the plugin root.",
            metadata=metadata,
        )
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_invalid",
        f"'{field_name}' entry {raw!r} is not a valid component path for this field",
        reader.display(manifest_rel),
        f"Fix the '{field_name}' value to match the Claude Code plugin manifest reference.",
        metadata=metadata,
    )


def _style_finding(reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str) -> Finding:
    return _plugin_finding(
        Severity.MEDIUM,
        "plugin_component_path_style",
        f"'{field_name}' path {declared.raw!r} does not start with './'; Claude Code rejects such manifest paths",
        reader.display(manifest_rel),
        f"Write the path as './{declared.rel.as_posix() if declared.rel else declared.raw}'.",
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
class _Builder:
    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any] | None,
        *,
        contained: bool,
        manifest_rel: str,
        allowed_private_hosts: Iterable[str],
    ) -> None:
        self.reader = PluginRootReader(root)
        self.manifest = manifest if isinstance(manifest, dict) else None
        self.contained = contained
        self.manifest_rel = manifest_rel
        self.allowed_private_hosts = tuple(allowed_private_hosts)
        self.inventory = PluginInventory()
        self._keys: dict[tuple[str, str, str], Component] = {}
        self._truncated: set[str] = set()

    # -- generic helpers -------------------------------------------------- #
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
        if len(self.inventory.of_type(component.type)) >= PLUGIN_COMPONENT_MAX_ITEMS:
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
        self.inventory.components.append(component)
        return component

    def _read(self, rel: PurePosixPath, max_bytes: int = CONTENT_DEDUP_MAX_FILE_BYTES) -> str | None:
        try:
            return self.reader.read_text(rel, max_bytes)
        except (SecurePathError, OSError):
            self.inventory.unread_files += 1
            return None

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

    def _resolve_declared(self, field_name: str, raw: Any, *, style: bool = True) -> tuple[DeclaredPath | None, str]:
        """Normalize + classify one declared path; record findings. Returns (path, kind|problem)."""
        if not isinstance(raw, str):
            finding = _path_problem_finding(
                self.reader, field_name, DeclaredPath(repr(raw), None), self.manifest_rel, "invalid"
            )
            self.inventory.findings.append(finding)
            return None, "invalid"
        declared = normalize_declared_path(raw)
        if declared.problem is not None or declared.rel is None:
            problem = declared.problem if declared.problem == "escape" else "invalid"
            self.inventory.findings.append(
                _path_problem_finding(self.reader, field_name, declared, self.manifest_rel, problem)
            )
            return declared, problem
        if style and self.contained and not declared.dot_relative:
            self.inventory.findings.append(_style_finding(self.reader, field_name, declared, self.manifest_rel))
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
        return declared, kind

    def _broken(self, component_type: str, declared: DeclaredPath | None, raw: Any, problem: str) -> None:
        path = declared.rel.as_posix() if declared is not None and declared.rel is not None else None
        name = raw if isinstance(raw, str) else repr(raw)
        self._add(Component(component_type, name, "declared", path, _TYPE_SUPPORT[component_type], problem=problem))

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

    # -- skills ----------------------------------------------------------- #
    def skills(self) -> None:
        from skillevaluator.utils.helpers import find_bundled_plugin_skill_manifests

        declared_default = False
        if self.contained and self.manifest is not None:
            for raw in self._declared_values("skills", self.manifest.get("skills")):
                declared, kind = self._resolve_declared("skills", raw)
                if declared is None or declared.rel is None or kind in {"escape", "invalid", "missing", "unsafe"}:
                    self._broken("skill", declared, raw, kind)
                    continue
                if kind != "dir":
                    self.inventory.findings.append(
                        _path_problem_finding(self.reader, "skills", declared, self.manifest_rel, "invalid")
                    )
                    self._broken("skill", declared, raw, "invalid")
                    continue
                if declared.rel == PurePosixPath("skills"):
                    declared_default = True
                    continue
                self._declared_skill_dir(declared.rel)
        elif self.manifest is not None:
            for ref in _section_refs(self.manifest.get("skills")):
                self._add(Component("skill", _ref_label(ref), "declared", None, "evaluated"))

        try:
            manifests = find_bundled_plugin_skill_manifests(self.reader.root)
        except ValueError:
            manifests = []  # reported by the bundled-skill validator (bundled_skill_path_unsafe)
        for manifest in manifests[:PLUGIN_COMPONENT_MAX_ITEMS]:
            skill_dir = PurePosixPath("skills") / manifest.relative_path.parent.as_posix()
            origin: Origin = "declared+packaged" if declared_default else "packaged"
            component = self._add(
                Component("skill", manifest.relative_path.parent.as_posix(), origin, skill_dir.as_posix(), "evaluated")
            )
            self._skill_cost(component, skill_dir / manifest.relative_path.name)

    def _declared_skill_dir(self, rel_dir: PurePosixPath) -> None:
        """A declared skills dir holds ``SKILL.md`` directly or ``<name>/SKILL.md`` folders."""
        candidates: list[PurePosixPath] = []
        for variant in SKILL_MANIFEST_VARIANTS:
            if self.reader.kind(rel_dir / variant) == "file":
                candidates.append(rel_dir / variant)
                break
        if not candidates:
            start = self.reader.root if str(rel_dir) == "." else self.reader.root / rel_dir.as_posix()
            try:
                with os.scandir(start) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError:
                entries = []
            for entry in entries[:CONTENT_DEDUP_MAX_DISCOVERED_PATHS]:
                if entry.name in SCAN_EXCLUDED_DIRS or entry.name.startswith("."):
                    continue
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
                        candidates.append(child / variant)
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
        for manifest_rel in candidates:
            skill_dir = manifest_rel.parent
            name = skill_dir.name if str(skill_dir) != "." else self.reader.root.name
            component = self._add(Component("skill", name, "declared", skill_dir.as_posix(), "evaluated"))
            self._skill_cost(component, manifest_rel)

    def _skill_cost(self, component: Component, manifest_rel: PurePosixPath) -> None:
        text = self._read(manifest_rel)
        if text is None:
            return
        parsed = parse_markdown(text)
        always = len(parsed.name or "") + len(parsed.description or "")
        component.cost = CostRow(
            "skill",
            component.name,
            always,
            len(parsed.body),
            "always-on: frontmatter name + description; on-demand: SKILL.md body",
        )

    # -- rules ------------------------------------------------------------ #
    def rules(self) -> None:
        declared_default = False
        section = self.manifest.get("rules") if self.manifest is not None else None
        if self.contained and section is not None:
            for raw in self._declared_values("rules", section):
                if isinstance(raw, str) and "::" in raw:
                    self._add(Component("rule", _ref_label(raw), "declared", None, "evaluated"))
                    continue
                declared, kind = self._resolve_declared("rules", raw, style=False)
                if declared is None or declared.rel is None or kind in {"escape", "invalid", "missing", "unsafe"}:
                    self._broken("rule", declared, raw, kind)
                    continue
                if declared.rel == PurePosixPath("rules"):
                    declared_default = True
                elif kind == "dir":
                    self._rule_dir(declared.rel, "declared")
                else:
                    self._rule_file(declared.rel, "declared")
        elif self.manifest is not None and not self.contained:
            # Bundle-reference refs are inventoried by label only; resolving them
            # (and flagging dangling ones) is the dependency-resolution report's job.
            for ref in _section_refs(section):
                self._add(Component("rule", _ref_label(ref), "declared", None, "evaluated"))
        if self.reader.kind(PurePosixPath("rules")) == "dir":
            self._rule_dir(PurePosixPath("rules"), "declared+packaged" if declared_default else "packaged")
        elif self.reader.kind(PurePosixPath("rules")) == "link":
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader,
                    "rules",
                    DeclaredPath("rules", PurePosixPath("rules")),
                    self.manifest_rel,
                    "unsafe",
                    PurePosixPath("rules"),
                )
            )
            self._add(Component("rule", "rules", "packaged", "rules", "evaluated", problem="unsafe"))

    def _rule_dir(self, rel_dir: PurePosixPath, origin: Origin) -> None:
        files = self._list_dir("rule", rel_dir, None)
        for rel in files or []:
            self._rule_file(rel, origin, name=rel.relative_to(rel_dir).as_posix())

    def _rule_file(self, rel: PurePosixPath, origin: Origin, *, name: str | None = None) -> None:
        component = self._add(Component("rule", name or rel.name, origin, rel.as_posix(), "evaluated"))
        text = self._read(rel)
        if text is not None:
            component.cost = CostRow(
                "rule",
                component.name,
                0,
                len(text.strip()),
                "on-demand: rule body (embedded in the Tier 3 wrapper SKILL.md)",
            )

    # -- markdown component types (agents, commands, output styles) ------- #
    def markdown_components(self, component_type: str, field_name: str, default_dir: str) -> None:
        declared_value = self._declared_field(field_name)
        if declared_value is None:
            if self.reader.kind(PurePosixPath(default_dir)) == "dir":
                self._markdown_dir(component_type, PurePosixPath(default_dir), "packaged")
            elif self.reader.kind(PurePosixPath(default_dir)) == "link":
                self._list_dir(component_type, PurePosixPath(default_dir), (".md",))
            return
        if component_type == "command" and isinstance(declared_value, dict):
            self._command_map(declared_value)
            return
        for raw in self._declared_values(field_name, declared_value):
            declared, kind = self._resolve_declared(field_name, raw)
            if declared is None or declared.rel is None or kind in {"escape", "invalid", "missing", "unsafe"}:
                self._broken(component_type, declared, raw, kind)
                continue
            origin: Origin = (
                "declared+packaged"
                if declared.rel.parts[:1] == (default_dir,) or declared.rel == PurePosixPath(default_dir)
                else "declared"
            )
            if kind == "dir":
                self._markdown_dir(component_type, declared.rel, origin)
            else:
                self._markdown_file(component_type, declared.rel, origin)

    def _declared_field(self, field_name: str) -> Any:
        if self.manifest is None or not self.contained:
            return None
        return self.manifest.get(field_name)

    def _markdown_dir(self, component_type: str, rel_dir: PurePosixPath, origin: Origin) -> None:
        for rel in self._list_dir(component_type, rel_dir, (".md",)) or []:
            self._markdown_file(component_type, rel, origin)

    def _markdown_file(self, component_type: str, rel: PurePosixPath, origin: Origin) -> None:
        text = self._read(rel)
        parsed = parse_markdown(text) if text is not None else None
        name = (parsed.name if parsed is not None and parsed.name else None) or rel.stem
        component = self._add(Component(component_type, name, origin, rel.as_posix(), _TYPE_SUPPORT[component_type]))
        if parsed is not None:
            component.cost = _markdown_cost(component_type, component.name, parsed)

    def _command_map(self, commands: dict[str, Any]) -> None:
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
                continue
            description = entry.get("description") if isinstance(entry.get("description"), str) else ""
            if "content" in entry:
                content = entry.get("content") if isinstance(entry.get("content"), str) else ""
                component = self._add(Component("command", str(command_name), "declared", None, "unsupported"))
                component.cost = CostRow(
                    "command",
                    component.name,
                    len(description),
                    len(content),
                    "always-on: description; on-demand: inline content",
                )
                continue
            declared, kind = self._resolve_declared(f"commands[{command_name!r}].source", entry.get("source"))
            if declared is None or declared.rel is None or kind != "file":
                if kind == "dir" and declared is not None:
                    self.inventory.findings.append(
                        _path_problem_finding(self.reader, "commands", declared, self.manifest_rel, "invalid")
                    )
                    kind = "invalid"
                self._broken("command", declared, str(command_name), kind)
                continue
            text = self._read(declared.rel)
            component = self._add(
                Component("command", str(command_name), "declared", declared.rel.as_posix(), "unsupported")
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

    # -- JSON-config component types (hooks, lsp, monitors, settings) ----- #
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
        self, field_name: str, declared_value: Any, default: PurePosixPath, *, merge_default: bool = True
    ) -> Iterator[tuple[str, Origin, str, Any]]:
        """Yield (name, origin, root-relative file, parsed config) for a JSON-config field.

        ``hooks`` and ``lspServers`` merge with their default file; monitors replace it
        (``merge_default=False``) when declared.
        """
        declared_values = self._declared_values(field_name, declared_value)
        explicit: set[PurePosixPath] = set()
        for raw in declared_values:
            if isinstance(raw, str):
                declared = normalize_declared_path(raw)
                if declared.rel is not None:
                    explicit.add(declared.rel)
        load_default = merge_default or not declared_values
        default_kind = self.reader.kind(default) if load_default else "missing"
        if default_kind == "file" and default not in explicit:
            config = self._load_json(default, field_name)
            yield default.as_posix(), "packaged", default.as_posix(), config
        elif default_kind in {"link", "special"}:
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
                yield f"inline[{index}]" if len(declared_values) > 1 else "inline", "declared", self.manifest_rel, raw
                continue
            declared, kind = self._resolve_declared(field_name, raw)
            if declared is None or declared.rel is None or kind != "file":
                if kind == "dir" and declared is not None:
                    self.inventory.findings.append(
                        _path_problem_finding(self.reader, field_name, declared, self.manifest_rel, "invalid")
                    )
                    kind = "invalid"
                yield (raw if isinstance(raw, str) else repr(raw)), "declared", "", kind
                continue
            origin: Origin = "declared+packaged" if declared.rel == default else "declared"
            yield declared.rel.as_posix(), origin, declared.rel.as_posix(), self._load_json(declared.rel, field_name)

    def hooks(self) -> None:
        declared_value = self._declared_field("hooks")
        for name, origin, rel, config in self._json_sources("hooks", declared_value, HOOKS_JSON):
            if not rel:
                self._add(Component("hook", name, origin, None, "unsupported", problem=str(config)))
                continue
            self._add(Component("hook", name, origin, rel, "unsupported"))
            if config is not None:
                self.inventory.findings.extend(
                    _override_findings(
                        permission_bypass_issues(config),
                        self.reader.display(rel),
                        where=f"hooks ({name})",
                        component=("hook", name),
                    )
                )

    def lsp(self) -> None:
        declared_value = self._declared_field("lspServers")
        for name, origin, rel, config in self._json_sources("lspServers", declared_value, LSP_JSON):
            if not rel:
                self._add(Component("lsp", name, origin, None, "unsupported", problem=str(config)))
                continue
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
                    _override_findings(permission_bypass_issues(server), display, where=where, component=component)
                )
                if isinstance(server, dict):
                    self.inventory.findings.extend(
                        _override_findings(
                            env_override_issues(server.get("env")), display, where=where, component=component
                        )
                    )

    def monitors(self) -> None:
        declared_value = None
        if self.manifest is not None and self.contained:
            experimental = self.manifest.get("experimental")
            if isinstance(experimental, dict) and "monitors" in experimental:
                declared_value = experimental.get("monitors")
            elif "monitors" in self.manifest:
                declared_value = self.manifest.get("monitors")
        if isinstance(declared_value, list) and all(isinstance(item, dict) for item in declared_value):
            sources: Iterable[tuple[str, Origin, str, Any]] = [
                ("inline", "declared", self.manifest_rel, declared_value)
            ]
        elif declared_value is not None:
            sources = self._json_sources("experimental.monitors", declared_value, MONITORS_JSON, merge_default=False)
        else:
            sources = self._json_sources("experimental.monitors", None, MONITORS_JSON)
        for name, origin, rel, config in sources:
            if not rel:
                self._add(Component("monitor", name, origin, None, "unsupported", problem=str(config)))
                continue
            entries = config.get("monitors", config) if isinstance(config, dict) else config
            if not isinstance(entries, list) or not entries:
                self._add(Component("monitor", name, origin, rel, "unsupported"))
                continue
            for index, entry in enumerate(entries[:PLUGIN_COMPONENT_MAX_ITEMS]):
                entry_name = (
                    entry.get("name") if isinstance(entry, dict) and isinstance(entry.get("name"), str) else None
                )
                self._add(Component("monitor", entry_name or f"{name}[{index}]", origin, rel, "unsupported"))
            self.inventory.findings.extend(
                _override_findings(
                    permission_bypass_issues(entries), self.reader.display(rel), where=f"monitors ({name})"
                )
            )

    def settings(self) -> None:
        for rel in _SETTINGS_FILES:
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
        display = self.reader.display(rel)
        if not isinstance(config, dict):
            return
        start = len(self.inventory.findings)
        self._settings_findings(config, rel, display)
        for finding in self.inventory.findings[start:]:
            finding.metadata["plugin_component"] = {"type": "settings", "name": component_name}

    def _settings_findings(self, config: dict[str, Any], rel: str, display: str) -> None:
        permissions = config.get("permissions")
        if isinstance(permissions, dict):
            if permissions.get("defaultMode") == "bypassPermissions":
                self.inventory.findings.append(
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
            allow = permissions.get("allow")
            if isinstance(allow, list):
                broad = sorted(
                    {
                        str(rule)
                        for rule in allow  # every rule: already bounded by the structured-data limits
                        if isinstance(rule, str) and rule.replace(" ", "").lower() in _BROAD_ALLOW_RULES
                    }
                )
                if broad:
                    self.inventory.findings.append(
                        _plugin_finding(
                            Severity.HIGH,
                            "plugin_settings_broad_allow",
                            f"shipped settings '{rel}' pre-approves unrestricted tools {broad}",
                            display,
                            "Remove blanket allow rules such as Bash / Bash(*); scope permissions to exact commands.",
                        )
                    )
        if config.get("enableAllProjectMcpServers") is True:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_settings_auto_approve",
                    f"shipped settings '{rel}' sets enableAllProjectMcpServers, auto-approving every project MCP "
                    "server",
                    display,
                    "Remove enableAllProjectMcpServers; let users approve MCP servers explicitly.",
                )
            )
        self.inventory.findings.extend(_override_findings(env_override_issues(config.get("env")), display, where=rel))
        self.inventory.findings.extend(_override_findings(permission_bypass_issues(config), display, where=rel))

    # -- MCP -------------------------------------------------------------- #
    def mcp(self) -> None:
        collection = collect_mcp_declarations(
            self.reader,
            self.manifest,
            contained=self.contained,
            manifest_rel=self.manifest_rel,
            allowed_private_hosts=self.allowed_private_hosts,
        )
        self.inventory.mcp = collection
        self.inventory.findings.extend(collection.findings)
        self.inventory.findings.extend(collection.server_findings)
        sources_by_name: dict[str, set[str]] = {}
        for declaration in collection.declarations:
            sources_by_name.setdefault(declaration.name, set()).add(declaration.source)
        for declaration in collection.effective:
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

    def build(self) -> PluginInventory:
        self.skills()
        self.rules()
        self.mcp()
        self.hooks()
        self.markdown_components("agent", "agents", "agents")
        self.markdown_components("command", "commands", "commands")
        self.lsp()
        self.markdown_components("output_style", "outputStyles", "output-styles")
        self.monitors()
        self.settings()
        self.env_files()
        return self.inventory


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


def _is_env_file(name: str) -> bool:
    lowered = name.lower()
    return lowered == ".env" or (
        lowered.startswith(".env.") and lowered.rsplit(".", 1)[-1] not in _ENV_TEMPLATE_SUFFIXES
    )


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
            if _is_env_file(entry.name):
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
# MCP declaration collection                                                  #
# --------------------------------------------------------------------------- #
def collect_mcp_declarations(
    reader: PluginRootReader,
    manifest: dict[str, Any] | None,
    *,
    contained: bool,
    manifest_rel: str,
    allowed_private_hosts: Iterable[str] = (),
    validate_servers: bool = True,
) -> McpCollection:
    """Collect every MCP server a plugin declares, in Claude Code load order.

    Claude Code loads ``.mcp.json`` at the plugin root first, then each declared
    ``mcpServers`` shape in order (inline map, ``.json`` path, or an array mixing
    them); a later same-name server replaces an earlier one. A config file is
    either ``{"mcpServers": {...}}`` or a bare server map (both documented).
    ``.mcpb``/``.dxt`` bundles are recorded but not inspected. For
    ``agent_plugin.yaml`` the ``mcp`` list is used (Pydantic validates it).

    Findings cover config-source problems (escapes, absolute paths, symlinks,
    missing/oversize/invalid files, duplicate names) and, when
    ``validate_servers`` is set, the per-server static checks.
    """
    collection = McpCollection()
    manifest = manifest if isinstance(manifest, dict) else None
    manifest_display = reader.display(manifest_rel)

    if manifest is not None and not contained and isinstance(manifest.get("mcp"), list):
        for entry in manifest["mcp"][:PLUGIN_COMPONENT_MAX_ITEMS]:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                config = {key: value for key, value in entry.items() if key != "name"}
                collection.declarations.append(McpDeclaration(entry["name"], config, "agent_plugin_yaml", manifest_rel))

    declared = manifest.get("mcpServers") if manifest is not None else None
    entries: list[Any]
    if declared is None:
        entries = []
    elif isinstance(declared, dict | str):
        entries = [declared]
    elif isinstance(declared, list):
        entries = declared[:PLUGIN_COMPONENT_MAX_ITEMS]
        if len(declared) > PLUGIN_COMPONENT_MAX_ITEMS:
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_config_file_too_large",
                    f"'mcpServers' has {len(declared)} entries; only {PLUGIN_COMPONENT_MAX_ITEMS} are inspected",
                    manifest_display,
                    "Reduce the number of mcpServers entries.",
                    category=MCP_CATEGORY,
                )
            )
    else:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_servers_not_object",
                "'mcpServers' must be an inline server map, a config-file path string, or an array of those "
                f"(got {type(declared).__name__})",
                manifest_display,
                'Use {"<name>": {"command"|"url"|"provider": ...}}, "./.mcp.json", or an array mixing both.',
                category=MCP_CATEGORY,
            )
        )
        entries = []

    explicit_files: set[PurePosixPath] = set()
    for entry in entries:
        if isinstance(entry, str):
            normalized = normalize_declared_path(entry)
            if normalized.rel is not None:
                explicit_files.add(normalized.rel)

    # 1. The default root .mcp.json (unless the manifest names it explicitly).
    if MCP_JSON not in explicit_files:
        kind = reader.kind(MCP_JSON)
        if kind == "file":
            _load_mcp_file(reader, collection, MCP_JSON, "mcp_json", MCP_JSON.as_posix())
        elif kind in {"link", "special"}:
            collection.findings.append(
                _path_problem_finding(
                    reader, "mcpServers", DeclaredPath(".mcp.json", MCP_JSON), manifest_rel, "unsafe", MCP_JSON
                )
            )
            collection.broken_sources.append((".mcp.json", MCP_JSON.as_posix(), "unsafe"))

    # 2. Declared shapes in order.
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            if len(entry) > PLUGIN_COMPONENT_MAX_ITEMS:
                collection.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "mcp_config_file_too_large",
                        f"inline 'mcpServers' map declares {len(entry)} servers; only "
                        f"{PLUGIN_COMPONENT_MAX_ITEMS} are inspected",
                        manifest_display,
                        "Reduce the number of MCP servers per map.",
                        category=MCP_CATEGORY,
                    )
                )
            for name, config in list(entry.items())[:PLUGIN_COMPONENT_MAX_ITEMS]:
                collection.declarations.append(McpDeclaration(str(name), config, "inline", manifest_rel))
        elif isinstance(entry, str):
            _collect_path_ref(reader, collection, entry, manifest_rel)
        else:
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_servers_entry_invalid",
                    f"mcpServers[{index}] must be a config-file path string or an inline server map "
                    f"(got {type(entry).__name__})",
                    manifest_display,
                    'Use "./path/to/servers.json" or {"<name>": {"command"|"url": ...}} for each array entry.',
                    category=MCP_CATEGORY,
                )
            )

    # 3. Duplicate names across Claude Code sources (later replaces earlier).
    seen: dict[str, McpDeclaration] = {}
    for declaration in collection.declarations:
        if declaration.source == "agent_plugin_yaml":
            continue
        previous = seen.get(declaration.name)
        if previous is not None:
            collection.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "mcp_server_duplicate_name",
                    f"MCP server '{declaration.name}' is declared in both '{previous.file}' ({previous.source}) and "
                    f"'{declaration.file}' ({declaration.source}); Claude Code keeps only the later declaration",
                    reader.display(declaration.file),
                    "Declare each MCP server name once so reviewers see the configuration that actually runs.",
                    category=MCP_CATEGORY,
                    metadata={"mcp_server": declaration.name},
                )
            )
        seen[declaration.name] = declaration

    if validate_servers:
        for declaration in collection.declarations:
            if declaration.source == "agent_plugin_yaml":
                continue  # validated by the PluginManifest model
            collection.server_findings.extend(
                validate_mcp_server_declaration(
                    declaration.name,
                    declaration.config,
                    reader.display(declaration.file),
                    allowed_private_hosts=allowed_private_hosts,
                )
            )
    return collection


def _collect_path_ref(reader: PluginRootReader, collection: McpCollection, raw: str, manifest_rel: str) -> None:
    manifest_display = reader.display(manifest_rel)
    lowered = raw.strip().lower()
    if lowered.startswith(("https://", "http://")):
        shown = redacted_url(raw)  # reports and the inventory never carry userinfo or query credentials
        if lowered.split("?", 1)[0].endswith(_MCP_BUNDLE_SUFFIXES):
            collection.bundles.append((shown, None))
            collection.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "mcp_bundle_not_inspected",
                    f"mcpServers references a remote MCP bundle {shown!r}; it is downloaded and executed at load time "
                    "and its contents cannot be inspected statically",
                    manifest_display,
                    "Vendor the server into the plugin with a pinned version so its configuration can be reviewed.",
                    category=MCP_CATEGORY,
                )
            )
            return
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_path_invalid",
                f"mcpServers URL {shown!r} is not an .mcpb/.dxt bundle; only bundles may be referenced by URL",
                manifest_display,
                "Reference a './'-relative .json config file, an inline server map, or an .mcpb bundle.",
                category=MCP_CATEGORY,
            )
        )
        collection.broken_sources.append((shown, None, "invalid"))
        return

    declared = normalize_declared_path(raw)
    if declared.problem is not None or declared.rel is None:
        problem = "escape" if declared.problem == "escape" else "invalid"
        collection.findings.append(_path_problem_finding(reader, "mcpServers", declared, manifest_rel, problem))
        collection.broken_sources.append((raw, None, problem))
        return
    rel = declared.rel
    if not declared.dot_relative:
        collection.findings.append(_style_finding(reader, "mcpServers", declared, manifest_rel))
    suffix = rel.suffix.lower()
    if suffix not in {".json", *_MCP_BUNDLE_SUFFIXES}:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_path_invalid",
                f"mcpServers path {raw!r} must name a .json config file or an .mcpb/.dxt bundle",
                manifest_display,
                "Point mcpServers at a JSON file such as './.mcp.json'.",
                category=MCP_CATEGORY,
            )
        )
        collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        return
    kind = reader.kind(rel)
    if kind == "missing":
        collection.findings.append(_path_problem_finding(reader, "mcpServers", declared, manifest_rel, "missing"))
        collection.broken_sources.append((raw, rel.as_posix(), "missing"))
        return
    if kind in {"link", "special"}:
        collection.findings.append(_path_problem_finding(reader, "mcpServers", declared, manifest_rel, "unsafe", rel))
        collection.broken_sources.append((raw, rel.as_posix(), "unsafe"))
        return
    if kind != "file":
        collection.findings.append(_path_problem_finding(reader, "mcpServers", declared, manifest_rel, "invalid"))
        collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        return
    if suffix in _MCP_BUNDLE_SUFFIXES:
        collection.bundles.append((raw, rel.as_posix()))
        collection.findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "mcp_bundle_not_inspected",
                f"mcpServers references the MCP bundle '{rel.as_posix()}'; bundle contents are not inspected "
                "statically and are not staged for evaluation",
                reader.display(rel),
                "Declare the server as a .json config (command/url) so it can be validated and evaluated.",
                category=MCP_CATEGORY,
            )
        )
        return
    source: McpSource = "path_ref"
    _load_mcp_file(reader, collection, rel, source, raw)


def _load_mcp_file(
    reader: PluginRootReader, collection: McpCollection, rel: PurePosixPath, source: McpSource, raw: str
) -> None:
    display = reader.display(rel)
    try:
        text = reader.read_text(rel, PLUGIN_CONFIG_MAX_BYTES, config=True)
    except SecurePathError as exc:
        if exc.code in {"file_size_limit", "total_size_limit"}:
            problem = (
                f"exceeds the {PLUGIN_CONFIG_MAX_BYTES}-byte limit"
                if exc.code == "file_size_limit"
                else f"could not be read: {exc}"
            )
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_config_file_too_large",
                    f"MCP config '{rel.as_posix()}' {problem}",
                    display,
                    "Keep MCP config files small; declare only server entries.",
                    category=MCP_CATEGORY,
                )
            )
            collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        elif exc.code == "invalid_text_encoding":
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_config_file_invalid",
                    f"MCP config '{rel.as_posix()}' is not valid UTF-8",
                    display,
                    "Save the MCP config as UTF-8 JSON.",
                    category=MCP_CATEGORY,
                )
            )
            collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        else:
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_unsafe",
                    f"MCP config '{rel.as_posix()}' could not be read safely ({exc}); it was not followed",
                    display,
                    "Replace links and special files with one regular contained JSON file.",
                )
            )
            collection.broken_sources.append((raw, rel.as_posix(), "unsafe"))
        return
    try:
        data = load_bounded_json(text)
    except StructuredDataLimitError as exc:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_file_too_large",
                f"MCP config '{rel.as_posix()}' exceeds structured-data complexity limits: {exc}",
                display,
                "Reduce nesting and collection sizes in the MCP config.",
                category=MCP_CATEGORY,
            )
        )
        collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        return
    except StructuredDataError as exc:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_file_invalid",
                f"MCP config '{rel.as_posix()}' is not valid JSON: {exc}",
                display,
                "Fix the JSON syntax of the MCP config.",
                category=MCP_CATEGORY,
            )
        )
        collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        return
    if isinstance(data, dict) and "mcpServers" in data:
        servers = data["mcpServers"]
        if not isinstance(servers, dict):
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_servers_not_object",
                    f"'mcpServers' in '{rel.as_posix()}' must be an object mapping server names to their config",
                    display,
                    'Use {"mcpServers": {"<name>": {"command"|"url": ...}}}.',
                    category=MCP_CATEGORY,
                )
            )
            collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
            return
    elif isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
        # Documented bare form: servers at the top level without the wrapper.
        servers = data
    else:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_file_invalid",
                f"MCP config '{rel.as_posix()}' is neither {{\"mcpServers\": {{...}}}} nor a map of server objects",
                display,
                'Use {"mcpServers": {"<name>": {"command"|"url": ...}}} (or the same map without the wrapper).',
                category=MCP_CATEGORY,
            )
        )
        collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        return
    if len(servers) > PLUGIN_COMPONENT_MAX_ITEMS:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_file_too_large",
                f"MCP config '{rel.as_posix()}' declares {len(servers)} servers; only "
                f"{PLUGIN_COMPONENT_MAX_ITEMS} are inspected",
                display,
                "Reduce the number of MCP servers per config file.",
                category=MCP_CATEGORY,
            )
        )
    for name, config in list(servers.items())[:PLUGIN_COMPONENT_MAX_ITEMS]:
        collection.declarations.append(McpDeclaration(str(name), config, source, rel.as_posix()))


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #
def manifest_rel_for(manifest_path: Path, root: Path) -> str:
    """Root-relative POSIX spelling of a located manifest path."""
    if (
        manifest_path.name == PLUGIN_CONTAINED_MANIFEST_FILE
        and manifest_path.parent.name == PLUGIN_CONTAINED_MANIFEST_DIR
    ):
        return f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}"
    try:
        return manifest_path.relative_to(root).as_posix()
    except ValueError:
        return manifest_path.name


def build_plugin_inventory(
    root: Path,
    manifest: dict[str, Any] | None,
    *,
    contained: bool,
    manifest_rel: str,
    allowed_private_hosts: Iterable[str] = (),
) -> PluginInventory:
    """Build the static component inventory for a plugin root (never follows links)."""
    return _Builder(
        root,
        manifest,
        contained=contained,
        manifest_rel=manifest_rel,
        allowed_private_hosts=allowed_private_hosts,
    ).build()


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
        "not_evaluated": sum(1 for row in rows if row["state"] != "staged"),
    }


def problem_reason(component: Component) -> str:
    return {
        "missing": "declared path does not exist",
        "escape": "declared path escapes the plugin root",
        "unsafe": "path is a symlink, hard link, or special file and was not followed",
        "invalid": "declaration is invalid for this component type",
    }.get(component.problem or "", "component could not be loaded")
