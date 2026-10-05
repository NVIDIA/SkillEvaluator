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

import dataclasses
import math
import os
import posixpath
import re
import stat
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_COMPONENT_MAX_ITEMS,
    PLUGIN_CONFIG_MAX_BYTES,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_TYPES,
    PLUGIN_CURSOR_MANIFEST_TYPE,
    PLUGIN_MANIFEST_RELATIVE_PATHS,
    PLUGIN_MANIFEST_TYPE,
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    PLUGIN_TREE_PRUNED_DIRS,
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
    agent_plugins_schema_version,
    codex_accepts_path,
    declared_value_replaces_default,
    normalized_component_manifest,
    profile_for,
)
from skillevaluator.utils.secure_fs import (
    MAX_SECURE_DIRECTORY_DEPTH,
    SecurePathError,
    SecureRoot,
    discover_secure_files,
    stat_is_link_or_reparse,
)
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
    looks_like_inline_secret,
    permission_bypass_issues,
    redacted_url,
    validate_mcp_server_declaration,
)
from skillevaluator.validators.mcp_static import (
    _validate_command as mcp_validate_command,
)
from skillevaluator.validators.mcp_static import (
    _validate_pinning as mcp_validate_pinning,
)

PLUGIN_CATEGORY = "PLUGIN_SCHEMA"

ComponentType = Literal[
    "skill", "rule", "mcp", "hook", "agent", "command", "lsp", "output_style", "monitor", "settings", "app", "extension"
]
Support = Literal["evaluated", "static_only", "unsupported"]
Origin = Literal["declared", "packaged", "declared+packaged"]
McpSource = Literal["inline", "mcp_json", "path_ref", "agent_plugin_yaml"]
CoverageState = Literal["staged", "not_staged", "unsupported", "unavailable", "invalid", "loaded", "exercised"]

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
# Types only some manifest formats declare: Codex apps (connectors) and Agent
# Plugins client-extension namespaces. Counted only when present.
EXTRA_COMPONENT_TYPES: tuple[str, ...] = ("app", "extension")
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
    "app": "unsupported",
    "extension": "unsupported",
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
_ENV_TEMPLATE_SUFFIXES = frozenset({"example", "sample", "template", "dist", "defaults", "tmpl"})
_MAX_ENV_FILE_FINDINGS = 20
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_MANIFEST_PATHS = frozenset(PLUGIN_MANIFEST_RELATIVE_PATHS)
# Agent Plugins client-extension namespace directory names (reverse-domain).
_NAMESPACE_DIR_RE = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
# Codex MCP transport aliases and Agent Plugins' streamable-http type both mean http.
_HTTP_TYPE_ALIASES = frozenset({"streamable-http", "streamable_http"})
# Codex MCP fields that carry auth or environment the evaluation runtime does not apply.
_CODEX_UNAPPLIED_MCP_FIELDS = ("env_vars", "env_http_headers", "bearer_token_env_var", "http_headers_helper", "oauth")


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
    # Declared fields the evaluation runtime cannot apply (beyond env/headers).
    unapplied: tuple[str, ...] = ()

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
    # Top-level ``$schema`` of each loaded MCP config file (Agent Plugins checks).
    file_schemas: dict[str, Any] = field(default_factory=dict)

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
    """One component's static context-cost row.

    ``always_on_chars`` and ``on_demand_chars`` are token-weighted characters
    (:func:`cost_chars`) as the component's own harness loads it natively.
    :func:`_view_rows` turns them into the numbers of one harness and load
    mode. ``not_counted`` names always-on text whose size is not known
    statically (MCP tool schemas, hook output). ``traits`` are the frontmatter
    facts the views read: ``model_hidden`` (``disable-model-invocation``),
    ``forced_style`` and ``keeps_coding_instructions`` (output styles), and
    the rule traits of :func:`rule_traits` (``always_apply``, ``paths_rule``,
    ``scoped``, ``requested``).
    """

    type: str
    name: str
    always_on_chars: int
    on_demand_chars: int
    basis: str
    not_counted: str | None = None
    traits: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "always_on_tokens": estimate_tokens(self.always_on_chars),
            "on_demand_tokens": estimate_tokens(self.on_demand_chars),
            "basis": self.basis,
        }
        if self.not_counted:
            row["not_counted"] = self.not_counted
        return row


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
    # For a bundle-reference skill or rule ref: ``external`` or ``unresolved``
    # (``missing`` refs are broken: ``problem == "missing"``).
    dependency_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = {
            "type": self.type,
            "name": self.name,
            "origin": self.origin,
            "path": self.path,
            "support": self.support,
            "findings": self.findings,
        }
        if self.problem:
            # A broken declaration (missing, escape, unsafe, invalid): the client cannot load it.
            row["problem"] = self.problem
        if self.dependency_state:
            row["dependency"] = self.dependency_state
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
    # The harness whose native loading the default context-cost estimate describes.
    cost_harness: str = "claude-code"

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
            "broken": sum(1 for component in self.components if component.problem),
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

    def context_cost(
        self,
        *,
        extra_rows: Iterable[CostRow] = (),
        extra_notes: Iterable[str] = (),
        harness: str | None = None,
        load_mode: str | None = None,
    ) -> dict[str, Any]:
        """The static always-on versus on-demand estimate, for one harness and load mode plus every other view.

        *harness* and *load_mode* pick the headline view (a Tier 3 run's own);
        either defaults to the plugin's own harness (``cost_harness``) and native
        loading, and a pair that is not modeled falls back to that native view.
        ``by_harness`` repeats the totals for each modeled harness and load mode.
        """
        rows = [component.cost for component in self.components if component.cost is not None]
        rows.extend(extra_rows)
        selected = (harness or self.cost_harness, load_mode or COST_NATIVE)
        if selected not in _VIEW_TYPES:
            selected = (self.cost_harness, COST_NATIVE)
        views = list(COST_VIEWS)
        if (self.cost_harness, COST_NATIVE) not in views:
            views.insert(0, (self.cost_harness, COST_NATIVE))  # Cursor: its own view first
        by_component = [row.to_dict() for row in _view_rows(rows, *selected)]
        not_counted = [f"{row['type']} {row['name']}" for row in by_component if row.get("not_counted")]
        notes = [*_CONTEXT_COST_NOTES, *self.notes, *extra_notes]
        if not_counted:
            notes.append(
                "Lower bound: the always-on total leaves out what cannot be sized statically: "
                + ", ".join(not_counted[:5])
                + (f" and {len(not_counted) - 5} more" if len(not_counted) > 5 else "")
                + "."
            )
        if self.unread_files:
            notes.append(f"{self.unread_files} component file(s) could not be read safely and are not counted.")
        return {
            "method": "static_estimate",
            "estimator": COST_ESTIMATOR,
            "harness": selected[0],
            "load_mode": selected[1],
            "always_on_tokens": sum(row["always_on_tokens"] for row in by_component),
            "on_demand_tokens": sum(row["on_demand_tokens"] for row in by_component),
            "lower_bound": bool(not_counted),
            "not_counted": not_counted,
            "by_component": by_component,
            "by_harness": [_view_summary(rows, *view) for view in views],
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
    "Static estimate: tokens are approximated as characters / 4, with each CJK or other full-width character "
    "counted as 1 token (chars_div_4_cjk); no tokenizer is run.",
    "Always-on, for the harness and load mode shown: skill names and descriptions (Claude Code leaves out "
    "disable-model-invocation items), agent and command descriptions, the one forced output style Claude Code "
    "applies (it replaces the default coding instructions unless keep-coding-instructions is set, so it can "
    "shrink the prompt), rules the native adapter stages as always on (not agent-requested or manual rules, and "
    "for Codex not paths- or globs-scoped rules, which it does not stage), and UserPromptSubmit or startup "
    "SessionStart hook output that reads a plugin file.",
    "On-demand: SKILL.md bodies, agent and command bodies, output styles the user must select, and rules in "
    "wrapper mode, where they are embedded in the generated wrapper SKILL.md.",
    "MCP tool schemas are not known statically, so MCP servers are marked not counted, like hook output produced "
    "at run time. Claude Code may shorten long skill descriptions in its listing, depending on its version. "
    "by_harness gives the estimate for each harness and load mode.",
)


def estimate_tokens(chars: int) -> int:
    """Tokens for token-weighted characters (:func:`cost_chars`): divided by 4, rounded away from zero."""
    return math.ceil(chars / 4) if chars >= 0 else -math.ceil(-chars / 4)


COST_ESTIMATOR = "chars_div_4_cjk"
COST_NATIVE = "native"
COST_WRAPPER = "wrapper"
#: The harness and load-mode views every static estimate reports in ``by_harness``.
COST_VIEWS: tuple[tuple[str, str], ...] = (
    ("claude-code", COST_NATIVE),
    ("claude-code", COST_WRAPPER),
    ("codex", COST_NATIVE),
    ("codex", COST_WRAPPER),
)
COST_HARNESS_LABELS = {"claude-code": "Claude Code", "codex": "Codex", "cursor": "Cursor"}
#: The component types each view puts into the model's context. Native Claude Code loads every
#: type; SkillEvaluator's native Codex adapter stages skills, rules (AGENTS.md), and MCP servers;
#: the generated wrapper stages member skills, embeds rules, and starts runnable MCP servers.
_VIEW_TYPES: dict[tuple[str, str], frozenset[str]] = {
    ("claude-code", COST_NATIVE): frozenset({"skill", "rule", "agent", "command", "output_style", "mcp", "hook"}),
    ("claude-code", COST_WRAPPER): frozenset({"skill", "rule", "mcp"}),
    ("codex", COST_NATIVE): frozenset({"skill", "rule", "mcp"}),
    ("codex", COST_WRAPPER): frozenset({"skill", "rule", "mcp"}),
    ("cursor", COST_NATIVE): frozenset({"skill", "rule", "agent", "command", "mcp"}),
}
#: Harnesses that read ``disable-model-invocation`` and keep such skills and commands out of the listing.
#: Codex rejects the field in its own skill validator and lists the skill anyway.
_HIDES_MODEL_DISABLED = frozenset({"claude-code", "cursor"})
#: Characters of Claude Code's default system text ("Doing tasks" coding instructions) that a forced
#: output style replaces unless it sets ``keep-coding-instructions`` (2.1.284 captures: 3,233-3,262).
CLAUDE_CODE_CODING_INSTRUCTIONS_CHARS = 3_250
#: Hook events whose output Claude Code adds to the first model request.
_FIRST_TURN_HOOK_EVENTS = frozenset({"SessionStart", "UserPromptSubmit"})
#: The SessionStart source of a new session; resume, clear, and compact start later.
_SESSION_START_SOURCE = "startup"


def _runs_in_first_request(record: HookRecord) -> bool:
    """Whether a first-turn hook handler runs before the first request of a new session.

    UserPromptSubmit has no matcher. A SessionStart matcher is matched against
    the start source (``startup``, ``resume``, ``clear``, ``compact``) the way
    Claude Code matches tool names, read like ``matcher_scope`` in
    ``plugin_component_risk``: no matcher, ``""`` or ``*`` selects every
    source, a name or ``|`` list must name ``startup``, and anything else is a
    regular expression searched in ``startup``. A matcher the evaluator cannot
    decide counts (fail closed).
    """
    from skillevaluator.plugin_component_risk import (
        _EXACT_MATCHER_RE,
        _NAME_LIST_RE,
        MAX_MATCHER_CHARS,
        _InvalidMatcher,
        _matcher_names,
        _MatcherEvaluator,
        _MatcherParser,
        _UnsupportedMatcher,
    )

    if record.event != "SessionStart" or record.matcher is None or record.matcher.strip() in {"", "*"}:
        return True
    text = record.matcher.strip()
    if _NAME_LIST_RE.fullmatch(text) and _SESSION_START_SOURCE in _matcher_names(text):
        return True
    if _EXACT_MATCHER_RE.fullmatch(text):
        return False
    if len(text) > MAX_MATCHER_CHARS:
        return True
    try:
        return _MatcherEvaluator(_MatcherParser(text).parse()).matches(_SESSION_START_SOURCE)
    except _UnsupportedMatcher:
        return True
    except _InvalidMatcher:
        return False


_PRIMARY_COST_HARNESS = {PLUGIN_CODEX_MANIFEST_TYPE: "codex", PLUGIN_CURSOR_MANIFEST_TYPE: "cursor"}
_TYPE_WORDS = {
    "skill": "skills",
    "rule": "rules",
    "agent": "subagents",
    "command": "commands",
    "output_style": "output styles",
    "mcp": "MCP servers",
    "hook": "plugin hooks",
}


def cost_chars(text: str) -> int:
    """Token-weighted length: 1 per character, 4 per CJK or other full-width character (about 1 token each).

    Characters divided by 4 fits English text. CJK text runs near one token
    per character (a 225-character Chinese description is 172 o200k tokens),
    so wide characters weigh 4.
    """
    if text.isascii():
        return len(text)
    import unicodedata

    wide = sum(1 for char in text if unicodedata.east_asian_width(char) in {"W", "F"})
    return len(text) + 3 * wide


def _claude_flag(value: Any) -> bool:
    """Whether Claude Code reads a frontmatter flag as on.

    A Claude Code 2.1.284 probe forced an output style for ``true``,
    ``"true"``, ``yes`` and ``1``, and not for ``false`` or ``"false"``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def model_hidden_traits(frontmatter: dict[str, Any]) -> frozenset[str]:
    """``{"model_hidden"}`` when ``disable-model-invocation`` is on (as Claude Code reads it), else empty."""
    return frozenset({"model_hidden"}) if _claude_flag(frontmatter.get("disable-model-invocation")) else frozenset()


def _rule_flag(value: Any) -> bool:
    """``alwaysApply`` as the native adapters read it: YAML ``true`` or the string ``"true"`` (any case)."""
    return value is True or (isinstance(value, str) and value.strip().casefold() == "true")


def _rule_has_frontmatter(text: str) -> bool:
    """Whether a rule file opens with a closed ``---`` frontmatter block, as the native adapters test it."""
    lines = text.splitlines()
    return bool(lines) and lines[0].strip() == "---" and any(line.strip() == "---" for line in lines[1:])


def _rule_patterns(scope: Any) -> list[str]:
    """Cursor ``globs`` or Claude ``paths``: a list, or a comma-separated string, of glob patterns."""
    items = scope if isinstance(scope, list) else str(scope).split(",")
    return [text for item in items if (text := str(item).strip())]


def rule_traits(name: str, text: str, frontmatter: dict[str, Any]) -> frozenset[str]:
    """How SkillEvaluator's native adapters stage one rule (``tier3.plugin_native``).

    A rule without frontmatter is always on. With frontmatter:
    ``always_apply`` when ``alwaysApply`` is true; else ``paths_rule`` when
    ``paths`` (or ``globs``) lists a pattern, which Claude Code native loading
    stages as a ``paths`` rule; ``scoped`` when ``globs`` or ``paths`` is set at
    all, which Codex native loading does not stage (it has only an always-on
    rules channel); and ``requested`` for a Cursor agent-requested or manual
    rule (``alwaysApply`` present but not true, or a ``.mdc`` rule), which
    neither adapter stages.
    """
    if not _rule_has_frontmatter(text):
        return frozenset()
    if _rule_flag(frontmatter.get("alwaysApply")):
        return frozenset({"always_apply"})
    traits: set[str] = set()
    if _rule_patterns(frontmatter.get("paths") or frontmatter.get("globs") or []):
        traits.add("paths_rule")
    if frontmatter.get("globs") or frontmatter.get("paths"):
        traits.add("scoped")
    if "alwaysApply" in frontmatter or name.casefold().endswith(".mdc"):
        traits.add("requested")
    return frozenset(traits)


def native_rule_staging(harness: str, traits: frozenset[str]) -> str:
    """``always_on``, ``on_demand`` (a ``paths`` rule), or ``not_staged`` under one harness's native loading."""
    if "always_apply" in traits:
        return "always_on"
    if harness == "codex" and "scoped" in traits:
        return "not_staged"
    if harness != "codex" and "paths_rule" in traits:
        return "on_demand"
    return "not_staged" if "requested" in traits else "always_on"


def _rule_view(row: CostRow, harness: str, load_mode: str) -> CostRow:
    body = row.on_demand_chars + row.always_on_chars
    if load_mode == COST_WRAPPER:
        return CostRow("rule", row.name, 0, body, "on-demand: embedded in the generated wrapper SKILL.md")
    if harness == "cursor":
        if "always_apply" in row.traits:
            return CostRow("rule", row.name, body, 0, "always-on: alwaysApply rule (Cursor adds it to every request)")
        return CostRow("rule", row.name, 0, body, "on-demand: Cursor attaches it by globs, description, or mention")
    label = COST_HARNESS_LABELS.get(harness, harness)
    staging = native_rule_staging(harness, row.traits)
    if staging == "on_demand":
        return CostRow("rule", row.name, 0, body, "on-demand: a paths- or globs-scoped rule loads with matching files")
    if staging == "not_staged" and harness == "codex" and "scoped" in row.traits:
        return CostRow(
            "rule",
            row.name,
            0,
            0,
            f"not loaded: {label} native loading has only an always-on rules channel, so a paths- or "
            "globs-scoped rule is not staged",
        )
    if staging == "not_staged":
        return CostRow(
            "rule",
            row.name,
            0,
            0,
            f"not loaded: {label} native loading does not stage an agent-requested or manual rule "
            "(alwaysApply is not true)",
        )
    where = "appends it to $CODEX_HOME/AGENTS.md" if harness == "codex" else "stages it as a Claude Code user rule"
    return CostRow("rule", row.name, body, 0, f"always-on: native loading {where}")


def _view_rows(rows: Iterable[CostRow], harness: str, load_mode: str) -> list[CostRow]:
    """The rows as one harness and load mode loads them (see :data:`_VIEW_TYPES`).

    Claude Code applies one forced output style (the first in file order) and
    warns about the rest; the applied style replaces the default coding
    instructions unless it sets ``keep-coding-instructions``.
    """
    loads = _VIEW_TYPES[(harness, load_mode)]
    label = f"{COST_HARNESS_LABELS.get(harness, harness)} ({load_mode})"
    applied_style: str | None = None
    out: list[CostRow] = []
    for row in rows:
        if row.type not in loads:
            words = _TYPE_WORDS.get(row.type, f"{row.type} components")
            out.append(CostRow(row.type, row.name, 0, 0, f"not loaded: {label} does not load {words}"))
        elif row.type == "rule":
            out.append(_rule_view(row, harness, load_mode))
        elif "model_hidden" in row.traits and harness in _HIDES_MODEL_DISABLED:
            out.append(
                CostRow(
                    row.type,
                    row.name,
                    0,
                    row.always_on_chars + row.on_demand_chars,
                    "on-demand only: disable-model-invocation keeps it out of the model's listing",
                )
            )
        elif "forced_style" in row.traits and applied_style is not None:
            out.append(
                CostRow(
                    row.type,
                    row.name,
                    0,
                    row.always_on_chars,
                    f"on-demand: forced too, but Claude Code applies only one forced output style ({applied_style})",
                )
            )
        elif "forced_style" in row.traits:
            applied_style = row.name
            header = cost_chars(f"# Output Style: {row.name}\n\n")
            if "keeps_coding_instructions" in row.traits:
                out.append(dataclasses.replace(row, always_on_chars=header + row.always_on_chars))
            else:
                net = header + row.always_on_chars - CLAUDE_CODE_CODING_INSTRUCTIONS_CHARS
                basis = (
                    "always-on: forced output style body; it replaces Claude Code's default coding instructions "
                    f"(about {CLAUDE_CODE_CODING_INSTRUCTIONS_CHARS:,} characters), so the net cost can be negative"
                )
                out.append(dataclasses.replace(row, always_on_chars=net, basis=basis))
        else:
            out.append(row)
    return out


def _view_summary(rows: list[CostRow], harness: str, load_mode: str) -> dict[str, Any]:
    by_component = [row.to_dict() for row in _view_rows(rows, harness, load_mode)]
    not_counted = sum(1 for row in by_component if row.get("not_counted"))
    return {
        "harness": harness,
        "load_mode": load_mode,
        "always_on_tokens": sum(row["always_on_tokens"] for row in by_component),
        "on_demand_tokens": sum(row["on_demand_tokens"] for row in by_component),
        "lower_bound": bool(not_counted),
        "not_counted": not_counted,
    }


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
    problem: str | None = None  # empty | escape | invalid | placeholder
    dot_relative: bool = True
    # The root placeholder (``${CURSOR_PLUGIN_ROOT}``) the path started with, if any.
    root_variable: str | None = None


def normalize_declared_path(raw: str, root_prefixes: Iterable[str] = ("${CLAUDE_PLUGIN_ROOT}",)) -> DeclaredPath:
    """Normalize one manifest path to a contained root-relative POSIX path.

    Claude Code requires ``./``-relative paths (``"."``/``"./"`` names the root).
    A root placeholder prefix in ``root_prefixes`` is treated as the root. The
    inventory passes only the placeholders a format's client expands in
    manifest paths (:attr:`FormatProfile.manifest_path_prefixes`, Cursor's), so
    any other leading ``${...}`` is the ``placeholder`` problem. Absolute
    paths, home-relative paths, drive letters, and ``..`` segments are escapes.
    """
    text = raw.strip()
    if not text:
        return DeclaredPath(raw, None, "empty")
    if "\x00" in text or len(text) > 4096:
        return DeclaredPath(raw, None, "invalid")
    normalized = text.replace("\\", "/")
    dot_relative = normalized in {".", "./"} or normalized.startswith("./")
    root_variable: str | None = None
    for prefix in root_prefixes:
        if normalized.startswith(f"{prefix}/") or normalized == prefix:
            normalized = "./" + normalized[len(prefix) :].lstrip("/")
            dot_relative = True
            root_variable = prefix
            break
    if normalized.startswith(("/", "~")) or _WINDOWS_DRIVE_RE.match(normalized):
        return DeclaredPath(raw, None, "escape")
    if normalized.startswith("${"):
        return DeclaredPath(raw, None, "placeholder")
    if "${" in normalized or normalized.startswith("$"):
        return DeclaredPath(raw, None, "invalid")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return DeclaredPath(raw, None, "escape")
    rel = PurePosixPath(*parts) if parts else PurePosixPath(".")
    return DeclaredPath(raw, rel, None, dot_relative, root_variable)


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

    def kind(self, rel: PurePosixPath, *, allow_hard_links: bool = False) -> PathKind:
        """Classify ``rel`` with ``lstat`` on every component (links are never followed).

        A regular file with more than one hard link is ``special`` unless
        ``allow_hard_links`` (hook scripts, which are only scanned for evidence).
        """
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
                    return "file" if allow_hard_links or getattr(metadata, "st_nlink", 1) == 1 else "special"
                return "special"
        return "special"

    def _read_bytes(self, rel: PurePosixPath, max_bytes: int, *, config: bool = False) -> bytes:
        """Bounded, anchored, no-follow read counted against a read budget; raises :class:`SecurePathError`.

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
        return raw

    def read_text(self, rel: PurePosixPath, max_bytes: int, *, config: bool = False) -> str:
        """Bounded, anchored, no-follow UTF-8 read; raises :class:`SecurePathError`.

        ``config`` charges the read to the separate config budget.
        """
        raw = self._read_bytes(rel, max_bytes, config=config)
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SecurePathError(
                "invalid_text_encoding", f"File is not valid UTF-8: {rel.as_posix()}", relative_path=rel.as_posix()
            ) from exc

    def read_script_bytes(self, rel: PurePosixPath, max_bytes: int) -> bytes:
        """Whole, bounded, anchored, no-follow read of a script a hook runs; hard links are allowed.

        A hook script is only scanned for evidence (no file content leaves the
        hook risk analyzer), so a hard-linked file, such as a ``node_modules``
        file pnpm links to its store, is read like any other regular file. Links
        anywhere in the path, special files, a file over ``max_bytes``, and an
        exhausted read budget raise :class:`SecurePathError`. Platforms without
        descriptor-anchored ``openat`` keep :class:`SecureRoot`'s single-link rule.
        """
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
            return self._read_bytes(rel, max_bytes)
        remaining = CONTENT_DEDUP_MAX_TOTAL_BYTES - self.bytes_read
        if remaining <= 0:
            raise SecurePathError("total_size_limit", "Plugin inventory read budget exhausted.")
        limit = min(max_bytes, remaining)
        shown = rel.as_posix()
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        with SecureRoot(self.root) as secure_root:
            directory = secure_root.duplicate_posix_root_descriptor()
            try:
                for part in rel.parts[:-1]:
                    child = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
                    os.close(directory)
                    directory = child
                descriptor = os.open(
                    rel.parts[-1], flags | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0), dir_fd=directory
                )
            except OSError as exc:
                raise SecurePathError("unsafe_path", f"Cannot open {shown} without following links: {exc}") from exc
            finally:
                os.close(directory)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise SecurePathError("unsafe_path", f"Refusing a path that is not a regular file: {shown}")
            chunks: list[bytes] = []
            total = 0
            while total <= limit:
                chunk = os.read(descriptor, min(65_536, limit + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
        finally:
            os.close(descriptor)
        if total > limit:
            code = "total_size_limit" if remaining < max_bytes else "file_size_limit"
            raise SecurePathError(code, f"{shown} is larger than the {limit}-byte read limit", relative_path=shown)
        self.bytes_read += total
        return b"".join(chunks)

    def list_files(
        self, rel_dir: PurePosixPath, *, suffixes: tuple[str, ...] | None = None, prune: bool = True
    ) -> list[PurePosixPath]:
        """Securely list regular files below a contained directory (raises on links).

        ``prune`` skips the folders Tier 1 whole-tree scans skip
        (:data:`~skillevaluator.constants.SCAN_EXCLUDED_DIRS`). A client that
        loads a folder at any depth (Claude Code ``agents/``, ``commands/``,
        ``output-styles/``) also loads files there, so those listings pass
        ``prune=False`` and report such files instead of hiding them.
        """
        start = self.root if str(rel_dir) == "." else self.root / rel_dir.as_posix()

        def _selected(relative: Path) -> bool:
            return suffixes is None or relative.name.lower().endswith(suffixes)

        files = discover_secure_files(
            start,
            selected=_selected,
            excluded_dirs=SCAN_EXCLUDED_DIRS if prune else frozenset(),
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


# Codex drops a value of the wrong type in these fields, loads the default location, and still installs the
# plugin (plugin_formats: Codex check-1 rules). A wrong-typed 'apps' makes Codex refuse the manifest.
_CODEX_IGNORED_SCALAR_FIELDS = frozenset({"skills", "commands", "hooks"})


def _path_problem_finding(
    reader: PluginRootReader,
    field_name: str,
    declared: DeclaredPath,
    manifest_rel: str,
    problem: str,
    rel: PurePosixPath | None = None,
    *,
    reference: str = CLAUDE_PROFILE.reference,
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
    if declared.problem == "placeholder":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_invalid",
            f"'{field_name}' path {raw!r} starts with a variable. Clients expand root variables such as "
            "${CLAUDE_PLUGIN_ROOT} only in hook commands and MCP server fields, not in manifest component paths, "
            "so the client rejects or ignores this path",
            reader.display(manifest_rel),
            f"Write the path relative to the plugin root with a leading './', as the {reference} requires.",
            metadata=metadata,
        )
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_invalid",
        f"'{field_name}' entry {raw!r} is not a valid component path for this field",
        reader.display(manifest_rel),
        f"Fix the '{field_name}' value to match the {reference}.",
        metadata=metadata,
    )


def _unscanned_folder(rel: PurePosixPath) -> str | None:
    """The first folder of a plugin-root-relative path that Tier 1 whole-tree scans skip, or ``None``.

    The scans prune every :data:`~skillevaluator.constants.SCAN_EXCLUDED_DIRS`
    name at any depth: ``evals/``, ``results/``, and ``versions/`` (and their
    dotted forms), ``node_modules/``, ``.venv/``, ``.git/``, and
    ``__pycache__/``. Only an ``evals``/``results``/``versions`` folder at the
    first level of ``skills/`` is scanned anyway, because bundled-skill
    discovery scans it as a skill.
    """
    parts = rel.parts
    for index, part in enumerate(parts):
        if part not in SCAN_EXCLUDED_DIRS:
            continue
        if part in SCAN_ARTIFACT_DIRS and index == 1 and parts[0] == "skills":
            continue
        return part
    return None


def _in_unscanned_folder(rel: PurePosixPath) -> bool:
    """Whether a plugin-root-relative path is inside a folder that Tier 1 whole-tree scans skip."""
    return _unscanned_folder(rel) is not None


_UNSCANNED_FOLDERS_TEXT = "evals/, results/, versions/, node_modules/, .venv/, .git/, __pycache__/"


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


# Dependency folders, VCS metadata, and bytecode caches: whole-tree scans prune them at any depth.
_DEPENDENCY_DIRS = SCAN_EXCLUDED_DIRS - SCAN_ARTIFACT_DIRS
_DEPENDENCY_DIRS_TEXT = "node_modules/, .venv/, .git/, __pycache__/"


def _loaded_dependency_skill(parts: tuple[str, ...], *, recursive: bool) -> str | None:
    """The dependency folder of a ``SKILL.md`` a client loads from a skills folder, or ``None``.

    ``parts`` is the manifest path relative to the skills folder. Claude Code
    loads every ``<skills folder>/<name>/SKILL.md``, hidden names included, so
    ``skills/node_modules/SKILL.md`` and ``skills/.venv/SKILL.md`` are skills.
    Codex (``recursive``) searches the folder at any depth but skips hidden
    folders, so it also loads ``skills/x/node_modules/pkg/SKILL.md``.
    """
    folders = parts[:-1]
    if len(folders) == 1 and folders[0] in _DEPENDENCY_DIRS:
        return folders[0]
    if not recursive or any(part.startswith(".") for part in folders):
        return None
    return next((part for part in folders if part in _DEPENDENCY_DIRS), None)


def find_dependency_folder_skills(
    root: Path, skills_dir: str = "skills", *, recursive: bool = True
) -> tuple[list[PurePosixPath], bool]:
    """Skills a client loads from a dependency folder inside a skills folder, and whether the search finished.

    Bundled-skill discovery prunes ``node_modules/``, ``.venv/``, ``.git/``,
    and ``__pycache__/`` like the whole-tree scans, so a ``SKILL.md`` there is
    never listed or scanned, yet clients load it
    (:func:`_loaded_dependency_skill`). The paths are plugin-root relative.
    The walk reads names only, never follows links, skips hidden folders
    below the first level (no client loads skills from them), and stops after
    :data:`~skillevaluator.constants.PLUGIN_TREE_MAX_DISCOVERED_PATHS` entries.
    """
    start = Path(root)
    for part in PurePosixPath(skills_dir).parts:
        start = start / part
        try:
            metadata = start.lstat()
        except OSError:
            return [], True
        if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            return [], True  # a linked skills folder is reported by skill discovery
    base = PurePosixPath(skills_dir)
    found: list[PurePosixPath] = []
    budget = PLUGIN_TREE_MAX_DISCOVERED_PATHS
    # Breadth first, so the folders clients load first are searched first.
    # (directory, path relative to the skills folder, whether its subfolders are searched)
    pending: list[tuple[Path, PurePosixPath, bool]] = [(start, PurePosixPath(), True)]
    while pending:
        directory, relative, descend = pending.pop(0)
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            budget -= 1
            if budget < 0:
                return sorted(found), False
            child = relative / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                if entry.name in SKILL_MANIFEST_VARIANTS and _loaded_dependency_skill(child.parts, recursive=recursive):
                    found.append(base / child)
                continue
            if not descend or not stat.S_ISDIR(metadata.st_mode) or stat_is_link_or_reparse(metadata):
                continue
            searched = recursive and not entry.name.startswith(".") and len(child.parts) < MAX_SECURE_DIRECTORY_DEPTH
            if searched or (not relative.parts and entry.name in _DEPENDENCY_DIRS):
                # A first-level dependency folder is a skill folder (its own SKILL.md); only a
                # recursive client searches below it, and never below a hidden one.
                pending.append((Path(entry.path), child, searched))
    return sorted(found), True


def dependency_folder_skill_findings(
    display: Callable[[PurePosixPath], str], skills_dir: str, found: list[PurePosixPath], complete: bool
) -> list[Finding]:
    """HIGH for each skill a client loads from a dependency folder; LOW when the search stopped early.

    ``display`` maps a plugin-root-relative path to the finding's file path.
    Every client view and the bundled-skill check report the same text, so a
    merged inventory keeps one finding per skill.
    """
    findings: list[Finding] = []
    for rel in found[:PLUGIN_COMPONENT_MAX_ITEMS]:
        folder = next(part for part in rel.parts if part in _DEPENDENCY_DIRS)
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_skill_in_unscanned_folder",
                f"'{rel.as_posix()}' is a skill inside '{folder}/', a folder that Tier 1 scans skip "
                f"({_DEPENDENCY_DIRS_TEXT}). A client loads it from '{skills_dir}/' (Claude Code loads every "
                f"'{skills_dir}/<name>/' folder, and Codex searches the folder at any depth), but SkillEvaluator "
                "never lists or security-scans it",
                display(rel),
                "Give the skill folder a different name, or keep dependency folders, VCS metadata, and bytecode "
                "caches out of skills folders.",
                metadata={"path": rel.as_posix()},
            )
        )
    if not complete:
        findings.append(
            _plugin_finding(
                Severity.LOW,
                "plugin_skill_scan_incomplete",
                f"the search for skills inside dependency folders in '{skills_dir}/' stopped after "
                f"{PLUGIN_TREE_MAX_DISCOVERED_PATHS} entries; a client may load skills from the rest",
                display(PurePosixPath(skills_dir)),
                "Keep dependency folders out of skills folders.",
                metadata={"path": skills_dir},
            )
        )
    return findings


def _unscanned_path_finding(
    reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str
) -> Finding | None:
    """HIGH when a declared component lives in a folder that Tier 1 whole-tree scans skip.

    The security, secret, and Unicode scans prune evaluation output and
    version snapshots (``evals/``, ``results/``, ``versions/``), dependency
    folders (``node_modules/``, ``.venv/``), VCS metadata (``.git/``), and
    bytecode caches. A component the manifest loads from there would never be
    scanned. Only ``skills/<name>`` at the first level of ``skills/`` is
    searched, because bundled-skill discovery scans it as a skill.
    """
    rel = declared.rel
    folder = _unscanned_folder(rel) if rel is not None else None
    if folder is None:
        return None
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_unscanned",
        f"'{field_name}' path {declared.raw!r} is inside '{folder}/', a folder that Tier 1 whole-tree scans skip "
        f"({_UNSCANNED_FOLDERS_TEXT}), so the client loads files that are never security-scanned",
        reader.display(manifest_rel),
        "Move the component out of dependency, VCS, evaluation-output, and version-snapshot folders.",
        metadata={"plugin_component_ref": declared.raw},
    )


def _unscanned_packaged_finding(
    reader: PluginRootReader, component_type: str, rel: PurePosixPath, folder_rel: PurePosixPath
) -> Finding | None:
    """HIGH for a packaged agent, command, or output style a client loads from a folder the scans skip.

    Claude Code loads ``agents/``, ``commands/``, and ``output-styles/`` at any
    depth, so ``agents/evals/x.md`` is a live subagent even though the
    whole-tree scans prune ``evals/``.
    """
    folder = _unscanned_folder(rel)
    if folder is None:
        return None
    label = component_type.replace("_", " ")
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_unscanned",
        f"{label} '{rel.as_posix()}' is inside '{folder}/', a folder that Tier 1 whole-tree scans skip "
        f"({_UNSCANNED_FOLDERS_TEXT}). The client loads it from '{folder_rel.as_posix()}/', but it is never "
        "security-scanned",
        reader.display(rel),
        "Move the component out of dependency, VCS, evaluation-output, and version-snapshot folders.",
        metadata={"plugin_component_ref": rel.as_posix()},
    )


def _style_finding(
    reader: PluginRootReader,
    field_name: str,
    declared: DeclaredPath,
    manifest_rel: str,
    *,
    profile: FormatProfile = CLAUDE_PROFILE,
) -> Finding:
    """The finding for a declared path a client does not accept as written (no leading ``./``, or ``./`` alone).

    Its severity follows the client: Claude Code rejects the whole manifest
    (HIGH, the plugin does not load); Codex drops the value and loads the
    format's default location instead (MEDIUM).
    """
    if declared.rel is not None and str(declared.rel) == "." and declared.dot_relative:
        problem = "names the plugin root itself"
        fix = "Name the component folder or file, for example './skills/'."
    else:
        problem = "does not start with './'"
        fix = f"Write the path as './{declared.rel.as_posix() if declared.rel else declared.raw}'."
    if profile.rejects_undotted_paths:
        severity = Severity.HIGH
        outcome = f"{profile.label.removesuffix(' plugin')} rejects the whole manifest and does not load the plugin"
    else:
        severity = Severity.MEDIUM
        outcome = f"the {profile.label} loader ignores this value and loads the default location instead"
    return _plugin_finding(
        severity,
        "plugin_component_path_style",
        f"'{field_name}' path {declared.raw!r} {problem}; {outcome}",
        reader.display(manifest_rel),
        fix,
        metadata={"plugin_component_ref": declared.raw},
    )


def _root_variable_finding(
    reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str, profile: FormatProfile
) -> Finding:
    """MEDIUM for a component path that starts with a root variable the client documents only elsewhere (Cursor)."""
    return _plugin_finding(
        Severity.MEDIUM,
        "plugin_component_path_root_variable",
        f"'{field_name}' path {declared.raw!r} starts with {declared.root_variable}. The {profile.label} reference "
        "documents root variables for hook commands and MCP server fields, not for manifest component paths, so "
        "the client may not expand it there and may not load this component. SkillEvaluator still checks the files "
        "it names",
        reader.display(manifest_rel),
        f"Write the path relative to the plugin root, for example './{declared.rel.as_posix() if declared.rel else ''}'.",
        metadata={"plugin_component_ref": declared.raw},
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


# The name and path suffix of a hook component that is a skill's or command's frontmatter ``hooks`` block.
FRONTMATTER_HOOKS_SUFFIX = "#hooks"


def frontmatter_hook_file(component: Component) -> str | None:
    """The Markdown file whose frontmatter holds a frontmatter-hook component's hooks, else ``None``."""
    path = component.path or ""
    if component.type != "hook" or not path.endswith(FRONTMATTER_HOOKS_SUFFIX):
        return None
    return path.removesuffix(FRONTMATTER_HOOKS_SUFFIX) or None


def is_skill_frontmatter_hook(component: Component) -> bool:
    """Whether a hook component is the frontmatter ``hooks`` block of a skill (staged with its skill)."""
    file = frontmatter_hook_file(component)
    return file is not None and PurePosixPath(file).name in SKILL_MANIFEST_VARIANTS


# A Codex project config a plugin may ship, and its settings that turn off approvals or the sandbox.
_CODEX_CONFIG_FILE = ".codex/config.toml"
_CODEX_BYPASS_SETTINGS = (("approval_policy", "never"), ("sandbox_mode", "danger-full-access"))


def _bounded_text(value: str, limit: int) -> str:
    """Plugin text for a report: whitespace collapsed, credentials redacted, length bounded."""
    from skillevaluator.plugin_component_risk import _bounded

    return _bounded(value, limit)


def _agent_cli_flag_issues(value: Any) -> list[OverrideIssue]:
    """Agent-CLI flags in a hook, LSP, monitor, or settings config: permission bypasses, and the permissive
    ``--permission-mode acceptEdits|auto`` and ``--allowedTools Bash`` flags."""
    from skillevaluator.plugin_component_risk import allowed_tools_flag_issues

    return [*permission_bypass_issues(value), *permission_mode_flag_issues(value), *allowed_tools_flag_issues(value)]


def _lsp_command_findings(name: str, server: dict[str, Any], file_path: str) -> list[Finding]:
    """The MCP stdio command-form checks for one LSP server's ``command`` and ``args``, as ``plugin_lsp_*``.

    Claude Code starts an LSP server like a stdio MCP server (argv, no shell),
    so a shell ``-c`` program, inline credentials, and unpinned package runners
    get the same findings, worded for an LSP server. Without a shell, ``;`` or
    ``|`` inside an argument is passed to the server literally and is not
    flagged (the MCP argv rule); an argument that is only an operator or
    carries command substitution is a LOW note, not the MCP CRITICAL.
    """
    if not isinstance(server.get("command"), str):
        return []
    raw: list[Finding] = []
    mcp_validate_command(name, server, file_path, raw)
    mcp_validate_pinning(name, server, file_path, raw)
    prefix = f"mcpServers['{name}']: "
    findings: list[Finding] = []
    for finding in raw:
        check = f"plugin_lsp_{finding.check_name.removeprefix('mcp_')}"
        severity, message, suggestion = finding.severity, finding.message.removeprefix(prefix), finding.suggestion
        if check == "plugin_lsp_command_shell_metacharacters":
            if any(item.check_name == check for item in findings):
                continue
            severity = Severity.LOW
            message = (
                "an argument contains shell metacharacters (; | & ` $() < >); Claude Code starts LSP servers "
                "without a shell, so they reach the server literally"
            )
            suggestion = "Check that the argument is meant literally; the server does not get a shell."
        findings.append(
            _plugin_finding(
                severity,
                check,
                f"lspServers['{name}']: {_lsp_wording(message)}",
                file_path,
                _lsp_wording(suggestion) if suggestion else suggestion,
                metadata={"plugin_component": {"type": "lsp", "name": name}},
            )
        )
    return findings


def _lsp_wording(text: str) -> str:
    """MCP check text reworded for an LSP server."""
    return (
        text.replace("runnable MCP ", "LSP server ")
        .replace("MCP commands run", "LSP servers run")
        .replace("MCP server", "LSP server")
    )


def _lsp_env_findings(name: str, env: Any, file_path: str) -> list[Finding]:
    """The MCP env checks for an LSP server's ``env``: TLS verification off and inline credentials (CRITICAL)."""
    from skillevaluator.validators.mcp_static import env_tls_and_secret_issues

    return _override_findings(
        env_tls_and_secret_issues(env), file_path, where=f"lspServers['{name}']", component=("lsp", name)
    )


# --------------------------------------------------------------------------- #
# Markdown frontmatter                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Markdown:
    name: str | None
    description: str | None
    body: str
    frontmatter: dict[str, Any]


# Claude Code's lenient frontmatter retry: a ``key: value`` line whose plain
# value holds a YAML indicator (or ": ") is re-read as a double-quoted string.
_LENIENT_KEY_RE = re.compile(r"^([a-zA-Z_-]+):\s+(.+)$")
_LENIENT_SPECIAL_RE = re.compile(r"[{}\[\]*&#!|>%@`]|: ")
_LEADING_TABS_RE = re.compile(r"^\t+", re.MULTILINE)


def _quote_lenient_values(raw: str) -> str:
    """Quote the plain frontmatter values strict YAML rejects, the way Claude Code retries a failed parse.

    Claude Code 2.1.x parses frontmatter with a YAML parser and, when that
    fails, quotes each top-level ``key: value`` whose value contains a YAML
    indicator (``{}[]*&#!|>%@`` or a backtick) or ``": "`` (unless it is
    already quoted or is a valid flow list), turns leading tabs into two spaces
    each, and parses again. So
    ``description: Deploy: runs the script`` and ``tools: *`` load, and the
    privileges and context cost must be read the same way.
    """
    lines: list[str] = []
    for line in raw.split("\n"):
        match = _LENIENT_KEY_RE.match(line)
        if match is not None:
            key, value = match.groups()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                lines.append(line)
                continue
            if value.startswith("[") and value.endswith("]"):
                try:
                    if isinstance(load_bounded_yaml(value), list):
                        lines.append(line)
                        continue
                except StructuredDataError:
                    pass
            if _LENIENT_SPECIAL_RE.search(value):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{key}: "{escaped}"')
                continue
        lines.append(line)
    return _LEADING_TABS_RE.sub(lambda tabs: "  " * len(tabs.group(0)), "\n".join(lines))


def _load_frontmatter(raw: str) -> Any:
    """Bounded YAML frontmatter, with Claude Code's lenient retry when strict YAML rejects it."""
    if not raw.strip():
        return {}
    try:
        return load_bounded_yaml(raw)
    except StructuredDataLimitError:
        raise
    except (StructuredDataError, ValueError):
        return load_bounded_yaml(_quote_lenient_values(raw))


def parse_markdown(text: str) -> _Markdown:
    """Split optional ``---`` YAML frontmatter from a Markdown body (bounded YAML).

    Frontmatter that strict YAML rejects is read again the way Claude Code
    reads it (:func:`_quote_lenient_values`), so a value such as ``Deploy:
    runs the script`` does not hide the fields after it. Frontmatter that
    still fails to parse gives no fields, as in Claude Code.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return _Markdown(None, None, text.strip(), {})
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            raw = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :]).strip()
            try:
                data = _load_frontmatter(raw)
            except (StructuredDataError, ValueError, RecursionError):
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

    def _write_capable_mcp(self) -> list[str]:
        """Runnable MCP servers without a declared read-only mode (tool lists are unknown statically)."""
        return [
            declaration.name
            for declaration in self.inventory.mcp.effective
            if declaration.runnable and not mcp_server_is_read_only(declaration.config)
        ]

    def _privileges(
        self,
        component: Component,
        frontmatter: dict[str, Any],
        display: str,
        *,
        source_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        key = (component.type, component.name, component.path)
        if any((r.type, r.name, r.path) == key for r in self.inventory.privilege_records):
            return
        if component.type == "agent":
            plugin_name = self.manifest.get("name") if self.manifest is not None else None
            record, findings = analyze_agent(
                component.name,
                component.path,
                frontmatter,
                display,
                write_capable_mcp=self._write_capable_mcp(),
                plugin_name=plugin_name if isinstance(plugin_name, str) else None,
            )
        elif component.type == "command":
            record, findings = analyze_command(
                component.name, component.path, frontmatter, display, client=self._grant_client, **kwargs
            )
        elif component.type == "skill":
            record, findings = analyze_skill(
                component.name, component.path, frontmatter, display, client=self._grant_client
            )
        else:
            return
        self.inventory.privilege_records.append(record)
        self.inventory.findings.extend(findings)
        if component.type in {"command", "skill"} and "hooks" in frontmatter:
            file = source_file or component.path or self.manifest_rel
            self._frontmatter_hooks(frontmatter.get("hooks"), file, display)

    @property
    def _grant_client(self) -> str | None:
        """The client whose manifest loads this inventory's commands and skills, when it ignores allowed-tools."""
        return "codex" if self.profile.manifest_type == PLUGIN_CODEX_MANIFEST_TYPE else None

    def _frontmatter_hooks(self, config: Any, file: str, display: str) -> None:
        """Hooks in a skill's or command's frontmatter: Claude Code registers them while it is active.

        Each such ``hooks`` block is a ``hook`` component of its own, named and
        located ``<file>#hooks`` (the fragment keeps path-prefix attribution of
        the file's other findings on the skill or command), so Tier 3 can stage,
        census-wrap, refuse, and report it like any other hook source.
        """
        if not isinstance(config, dict):
            return
        source = f"{file}{FRONTMATTER_HOOKS_SUFFIX}"
        self._add(Component("hook", source, "packaged", source, "unsupported"))
        self.inventory.findings.extend(
            _override_findings(
                _agent_cli_flag_issues(config),
                display,
                where=f"hooks ({source})",
                component=("hook", source),
            )
        )
        analysis = self._hook_analyzer.analyze(config, source=source, file=file, display=display, dialect=CLAUDE_HOOKS)
        self.inventory.hook_records.extend(analysis.records)
        self.inventory.findings.extend(analysis.findings)

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

    def _codex_drops(self, raw: Any) -> bool:
        """Whether Codex ignores this declared path value (and loads the default location instead)."""
        return self.contained and self.profile.codex_path_rules and not codex_accepts_path(raw)

    def _resolve_declared(self, field_name: str, raw: Any, *, style: bool = True) -> tuple[DeclaredPath | None, str]:
        """Normalize + classify one declared path; record findings. Returns (path, kind|problem).

        The kind is ``dropped`` when the client ignores the value: Codex drops
        any path it does not accept (no leading ``./``, ``./`` alone, ``..``, a
        root variable) and loads the default location instead, so the value
        names no component. Its findings are still reported.
        """
        dropped = self._codex_drops(raw)
        if not isinstance(raw, str):
            if dropped and field_name in _CODEX_IGNORED_SCALAR_FIELDS:
                # Codex drops a skills, commands, or hooks value of the wrong type, loads the default location,
                # and still installs the plugin: MEDIUM, as check 1 rates it for the Codex client.
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.MEDIUM,
                        "plugin_component_path_invalid",
                        f"'{field_name}' entry {_bounded_text(repr(raw), 80)} is not a path; Codex ignores the value, "
                        "loads the default location, and still installs the plugin",
                        self.reader.display(self.manifest_rel),
                        f"Write '{field_name}' as a './'-relative path (or a list of them), or remove it to use the "
                        "default location.",
                        metadata={"plugin_component_ref": repr(raw)},
                    )
                )
                return None, "dropped"
            finding = _path_problem_finding(
                self.reader, field_name, DeclaredPath(repr(raw), None), self.manifest_rel, "invalid"
            )
            self.inventory.findings.append(finding)
            return None, "dropped" if dropped else "invalid"
        declared = normalize_declared_path(raw, self.profile.manifest_path_prefixes)
        if declared.problem is not None or declared.rel is None:
            problem = declared.problem if declared.problem == "escape" else "invalid"
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader, field_name, declared, self.manifest_rel, problem, reference=self.profile.reference
                )
            )
            return declared, "dropped" if dropped else problem
        if dropped:
            if style:
                self.inventory.findings.append(self._style(field_name, declared))
            return declared, "dropped"
        if style and self.contained and self.profile.require_dot_relative and not declared.dot_relative:
            self.inventory.findings.append(self._style(field_name, declared))
        if self.contained and declared.root_variable is not None:
            self.inventory.findings.append(
                _root_variable_finding(self.reader, field_name, declared, self.manifest_rel, self.profile)
            )
        # Skill folders a client loads are scanned as skill units even in such a folder
        # (client_skill_dirs_outside_tree_scans), so only other components get this.
        if field_name != "skills" and (
            unscanned := _unscanned_path_finding(self.reader, field_name, declared, self.manifest_rel)
        ):
            self.inventory.findings.append(unscanned)
        if self.contained and (wrong_kind := self._wrong_file_kind(field_name, raw)) is not None:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_invalid",
                    f"'{field_name}' entry {raw!r} must name a {wrong_kind} file; "
                    f"{self.profile.label.removesuffix(' plugin')} rejects the whole manifest otherwise and does not "
                    "load the plugin",
                    self.reader.display(self.manifest_rel),
                    f"List each {wrong_kind} file by its own './'-relative path, as the {self.profile.reference} "
                    "requires.",
                    metadata={"plugin_component_ref": raw},
                )
            )
            return declared, "invalid"
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

    def _style(self, field_name: str, declared: DeclaredPath) -> Finding:
        return _style_finding(self.reader, field_name, declared, self.manifest_rel, profile=self.profile)

    def _wrong_file_kind(self, field_name: str, raw: str) -> str | None:
        """The file kind a field's paths must name when ``raw`` does not (Claude Code: ``.md`` agents, ``.json`` configs).

        The client's schema compares the raw text, so a folder (``./agents/``)
        or another suffix fails it.
        """
        if field_name in self.profile.markdown_file_fields and not raw.endswith(".md"):
            return "Markdown (.md)"
        if field_name in self.profile.json_file_fields and not raw.endswith(".json"):
            return "JSON (.json)"
        return None

    def _declared_replaces(self, field_name: str, value: Any) -> bool:
        """Whether a declared field suppresses default-folder discovery for its type.

        Codex drops a path it does not accept and loads the default location
        instead (:func:`~skillevaluator.plugin_formats.declared_value_replaces_default`).
        """
        return declared_value_replaces_default(self.profile, field_name, value)

    def _broken(self, component_type: str, declared: DeclaredPath | None, raw: Any, problem: str) -> None:
        path = declared.rel.as_posix() if declared is not None and declared.rel is not None else None
        name = raw if isinstance(raw, str) else repr(raw)
        self._add(Component(component_type, name, "declared", path, _TYPE_SUPPORT[component_type], problem=problem))

    def _list_dir(
        self, component_type: str, rel_dir: PurePosixPath, suffixes: tuple[str, ...] | None, *, prune: bool = True
    ) -> list[PurePosixPath] | None:
        try:
            return self.reader.list_files(rel_dir, suffixes=suffixes, prune=prune)
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
        declared_value = self.manifest.get("skills") if self.manifest is not None else None
        default_dir = self.profile.default_skills_dir or "skills"
        if self.contained and self.manifest is not None:
            for raw in self._declared_values("skills", declared_value):
                declared, kind = self._resolve_declared("skills", raw)
                if kind == "dropped":
                    continue  # the client ignores the value and loads skills/ instead
                if declared is None or declared.rel is None or kind in {"escape", "invalid", "missing", "unsafe"}:
                    self._broken("skill", declared, raw, kind)
                    continue
                if kind != "dir":
                    self.inventory.findings.append(
                        _path_problem_finding(self.reader, "skills", declared, self.manifest_rel, "invalid")
                    )
                    self._broken("skill", declared, raw, "invalid")
                    continue
                if declared.rel == PurePosixPath(default_dir):
                    declared_default = True
                    continue
                self._declared_skill_dir(declared.rel)
                self._flag_dependency_folder_skills(declared.rel)
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
            start = self.reader.root if str(rel_dir) == "." else self.reader.root / rel_dir.as_posix()
            try:
                with os.scandir(start) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError:
                entries = []
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

    def _flag_dependency_folder_skills(self, rel_dir: PurePosixPath) -> None:
        """HIGH for a skill a client loads from a dependency folder inside declared skills folder ``rel_dir``.

        Skill discovery prunes ``node_modules/``, ``.venv/``, ``.git/``, and
        ``__pycache__/``, so such a skill is never listed or scanned
        (:func:`find_dependency_folder_skills`). Skills under ``skills/`` are
        reported by the bundled-skill validator, so they are not repeated here.
        """
        if rel_dir.parts[:1] == ("skills",):
            return
        found, complete = find_dependency_folder_skills(
            self.reader.root, rel_dir.as_posix(), recursive=self.profile.skills_recursive
        )
        self.inventory.findings.extend(
            dependency_folder_skill_findings(self.reader.display, rel_dir.as_posix(), found, complete)
        )

    def _skill_cost(self, component: Component, manifest_rel: PurePosixPath) -> None:
        text = self._read(manifest_rel)
        if text is None:
            return
        parsed = parse_markdown(text)
        # Claude Code pre-approves a skill's allowed-tools and registers its frontmatter hooks while it is active.
        self._privileges(
            component, parsed.frontmatter, self.reader.display(manifest_rel), source_file=manifest_rel.as_posix()
        )
        component.cost = CostRow(
            "skill",
            component.name,
            cost_chars((parsed.name or "") + (parsed.description or "")),
            cost_chars(parsed.body),
            "always-on: frontmatter name + description; on-demand: SKILL.md body",
            traits=model_hidden_traits(parsed.frontmatter),
        )

    def _read_lenient(self, rel: PurePosixPath) -> str | None:
        """Like :meth:`_read`, but a file that is not UTF-8 is decoded with replacement characters.

        Claude Code still lists such a component (a Latin-1 agent shows up with
        U+FFFD where the bytes were), so the cost estimate must count it.
        """
        try:
            return self.reader.read_text(rel, CONTENT_DEDUP_MAX_FILE_BYTES)
        except SecurePathError as exc:
            if exc.code != "invalid_text_encoding":
                self.inventory.unread_files += 1
                return None
        except OSError:
            self.inventory.unread_files += 1
            return None
        try:
            raw = self.reader._read_bytes(rel, CONTENT_DEDUP_MAX_FILE_BYTES)
        except (SecurePathError, OSError):
            self.inventory.unread_files += 1
            return None
        return raw.decode("utf-8-sig", errors="replace")

    def finish_costs(self) -> None:
        """Cost facts that need the whole inventory: the estimate's own harness, and first-turn hook output."""
        if self.contained:
            self.inventory.cost_harness = _PRIMARY_COST_HARNESS.get(self.profile.manifest_type, "claude-code")
        for component in self.inventory.components:
            if component.type != "hook" or not component.path or component.problem is not None:
                continue
            records = [
                record
                for record in self.inventory.hook_records
                if (record.source, record.file) == (component.name, component.path)
                and record.event in _FIRST_TURN_HOOK_EVENTS
            ]
            if not records:
                continue
            # A SessionStart group whose matcher does not select "startup" (resume, clear, compact) runs later.
            first = {id(record) for record in records if _runs_in_first_request(record)}
            counted = 0
            later = 0
            unknown: list[str] = []
            for record in records:
                chars = self._hook_output_chars(record)
                if id(record) not in first:
                    later += chars or 0
                elif chars is None:
                    unknown.append(record.event)
                else:
                    counted += chars
            events = ", ".join(sorted({record.event for record in records if id(record) in first}))
            if not first:
                basis = (
                    "on-demand: the SessionStart matcher does not select startup (only resume, clear, or compact), "
                    "so the output is not in the first request"
                )
            elif len(first) < len(records):
                basis = (
                    f"always-on: {events} hook output goes into the first request; SessionStart handlers whose "
                    "matcher does not select startup are on-demand"
                )
            else:
                basis = f"always-on: {events} hook output goes into the first request"
            component.cost = CostRow(
                "hook",
                component.name,
                counted,
                later,
                basis,
                not_counted=(
                    f"output of {len(unknown)} {', '.join(sorted(set(unknown)))} hook handler(s) is produced at run "
                    "time and is not known statically"
                    if unknown
                    else None
                ),
            )

    def _hook_output_chars(self, record: HookRecord) -> int | None:
        """Characters a ``cat <plugin file>`` or literal ``echo`` hook prints; ``None`` when not known statically."""
        import shlex

        if record.handler_type != "command" or not record.target:
            return None
        try:
            words = shlex.split(record.target)
        except ValueError:
            return None
        if len(words) >= 2 and words[0] == "echo" and not any("$" in word or "`" in word for word in words[1:]):
            return cost_chars(" ".join(words[1:]) + "\n")
        if len(words) != 2 or words[0] != "cat":
            return None
        target = words[1].replace("\\", "/")
        for prefix in (*self.profile.root_prefixes, "$CLAUDE_PLUGIN_ROOT"):
            if target.startswith(prefix + "/"):
                rel = PurePosixPath(os.path.normpath(target[len(prefix) + 1 :]))
                break
        else:
            return None  # a relative path is read from the session's working directory, not the plugin
        if rel.is_absolute() or ".." in rel.parts:
            return None
        try:
            return cost_chars(self.reader.read_text(rel, CONTENT_DEDUP_MAX_FILE_BYTES))
        except (SecurePathError, OSError):
            return None

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
                if kind == "dropped":
                    continue
                if declared is None or declared.rel is None or kind in {"escape", "invalid", "missing", "unsafe"}:
                    self._broken("rule", declared, raw, kind)
                    continue
                if declared.rel == PurePosixPath(self.profile.default_rules_dir or "rules"):
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
            text = text.strip()  # Tier 3 stages the stripped file, so a leading blank line hides no frontmatter
            parsed = parse_markdown(text)
            component.cost = CostRow(
                "rule",
                component.name,
                0,
                cost_chars(parsed.body),
                "rule body",
                traits=rule_traits(component.name, text, parsed.frontmatter),
            )

    # -- markdown component types (agents, commands, output styles) ------- #
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
            declared, kind = self._resolve_declared(field_name, raw)
            if kind == "dropped":
                continue  # the client ignores the value and loads the default folder instead
            if declared is None or declared.rel is None or kind in {"escape", "invalid", "missing", "unsafe"}:
                self._broken(component_type, declared, raw, kind)
                continue
            origin: Origin = (
                "declared+packaged"
                if default_dir is not None
                and (declared.rel.parts[:1] == (default_dir,) or declared.rel == PurePosixPath(default_dir))
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
        """Every agent, command, or output style a client loads from a folder, at any depth.

        Nothing is pruned: Claude Code also loads ``agents/evals/x.md``, so a
        file in a folder the whole-tree scans skip is inventoried, checked for
        privileges, and reported as unscanned (HIGH).
        """
        for rel in self._list_dir(component_type, rel_dir, self._suffixes(component_type), prune=False) or []:
            if (unscanned := _unscanned_packaged_finding(self.reader, component_type, rel, rel_dir)) is not None:
                self.inventory.findings.append(unscanned)
            self._markdown_file(component_type, rel, origin)

    def _markdown_file(self, component_type: str, rel: PurePosixPath, origin: Origin) -> None:
        text = self._read_lenient(rel)
        parsed = parse_markdown(text) if text is not None else None
        name = (parsed.name if parsed is not None and parsed.name else None) or rel.stem
        component = self._add(Component(component_type, name, origin, rel.as_posix(), _TYPE_SUPPORT[component_type]))
        if parsed is not None:
            component.cost = _markdown_cost(component_type, component.name, parsed)
            self._privileges(component, parsed.frontmatter, self.reader.display(rel))

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
                    cost_chars(description),
                    cost_chars(content),
                    "always-on: description; on-demand: inline content",
                )
                self._privileges(component, {}, self.manifest_display, entry=entry)
                continue
            declared, kind = self._resolve_declared(f"commands[{command_name!r}].source", entry.get("source"))
            if kind == "dropped":
                continue
            if declared is None or declared.rel is None or kind not in {"file", "dir"}:
                self._broken("command", declared, str(command_name), kind)
                continue
            if kind == "dir":
                self._command_map_folder(str(command_name), declared, entry)
                continue
            component = self._add(
                Component("command", str(command_name), "declared", declared.rel.as_posix(), "unsupported")
            )
            self._command_text(component, declared.rel, description, entry)

    def _command_text(self, component: Component, rel: PurePosixPath, description: str, entry: dict[str, Any]) -> None:
        """Read one command file of a ``commands`` map entry: its context cost and privileges."""
        text = self._read(rel)
        if text is None:
            self._privileges(component, {}, self.manifest_display, entry=entry)
            return
        parsed = parse_markdown(text)
        component.cost = CostRow(
            "command",
            component.name,
            cost_chars(description or parsed.description or ""),
            cost_chars(parsed.body),
            "always-on: description; on-demand: command body",
            traits=model_hidden_traits(parsed.frontmatter),
        )
        self._privileges(component, parsed.frontmatter, self.reader.display(rel), entry=entry)

    def _command_map_folder(self, command_name: str, declared: DeclaredPath, entry: dict[str, Any]) -> None:
        """A ``commands`` map entry whose source is a folder: one command per Markdown file directly in it.

        Claude Code accepts "a command file or skill directory" as the source.
        For a folder it loads each ``.md`` file directly inside as a command
        named after the file (``cmds/a.md`` is ``a``, whatever the map key; a
        ``SKILL.md`` is ``SKILL``) and does not search subfolders. Each row
        names its file, so Tier 3 stages and checks the file itself. A folder
        without Markdown files loads no command: the entry is an invalid row.
        """
        rel_dir = declared.rel
        if rel_dir is None:
            return
        files = self._list_dir("command", rel_dir, self._suffixes("command"), prune=False)
        if files is None:
            return  # an unsafe entry: reported, with a broken row
        direct = [rel for rel in files if rel.parent == rel_dir]
        if not direct:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_command_folder_empty",
                    f"commands[{command_name!r}].source {declared.raw!r} is a folder with no Markdown (.md) file "
                    "directly in it, so Claude Code loads no command from it",
                    self.manifest_display,
                    "Point the source at the command's .md file, or put the command files directly in the folder.",
                    metadata={"plugin_component": {"type": "command", "name": command_name}},
                )
            )
            self._broken("command", declared, command_name, "invalid")
            return
        for rel in direct[:PLUGIN_COMPONENT_MAX_ITEMS]:
            component = self._add(Component("command", rel.stem, "declared", rel.as_posix(), "unsupported"))
            self._command_text(component, rel, "", entry)

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
        self, field_name: str, declared_value: Any, default: PurePosixPath | None, *, merge_default: bool = True
    ) -> Iterator[tuple[str, Origin, str, Any]]:
        """Yield (name, origin, root-relative file, parsed config) for a JSON-config field.

        ``hooks`` and ``lspServers`` merge with their default file; monitors replace it
        (``merge_default=False``) when declared. For Codex, a declared value
        replaces the default only when Codex keeps it; otherwise the default
        file is loaded, as Codex does.
        """
        declared_values = self._declared_values(field_name, declared_value)
        explicit: set[PurePosixPath] = set()
        for raw in declared_values:
            if isinstance(raw, str) and not self._codex_drops(raw):
                declared = normalize_declared_path(raw, self.profile.manifest_path_prefixes)
                if declared.rel is not None:
                    explicit.add(declared.rel)
        if merge_default:
            load_default = True
        elif self.profile.codex_path_rules:
            load_default = not self._declared_replaces(field_name, declared_value)
        else:
            load_default = not declared_values
        default_kind = self.reader.kind(default) if load_default and default is not None else "missing"
        if default_kind == "file" and default not in explicit:
            config = self._load_json(default, field_name)
            yield default.as_posix(), "packaged", default.as_posix(), config
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
                yield f"inline[{index}]" if len(declared_values) > 1 else "inline", "declared", self.manifest_rel, raw
                continue
            declared, kind = self._resolve_declared(field_name, raw)
            if kind == "dropped":
                continue  # the client ignores the value and loads the default file instead
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
        default = PurePosixPath(self.profile.default_hooks_file) if self.profile.default_hooks_file else None
        sources = self._json_sources(
            "hooks", declared_value, default, merge_default=not self.profile.declared_replaces_default
        )
        for name, origin, rel, config in sources:
            self._hook(name, origin, rel, config)

    def _hook(self, name: str, origin: Origin, rel: str, config: Any) -> None:
        if not rel:
            self._add(Component("hook", name, origin, None, "unsupported", problem=str(config)))
            return
        self._add(Component("hook", name, origin, rel, "unsupported"))
        if config is not None:
            self.inventory.findings.extend(
                _override_findings(
                    _agent_cli_flag_issues(config),
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
                dialect=hook_dialect(self.profile.manifest_type, rel),
            )
            self.inventory.hook_records.extend(analysis.records)
            self.inventory.findings.extend(analysis.findings)

    def lsp(self) -> None:
        declared_value = self._declared_field("lspServers")
        default = PurePosixPath(self.profile.default_lsp_file) if self.profile.default_lsp_file else None
        if default is None and declared_value is None:
            return
        for name, origin, rel, config in self._json_sources("lspServers", declared_value, default):
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
                    _override_findings(
                        _agent_cli_flag_issues(server),
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
                    self.inventory.findings.extend(_lsp_env_findings(str(server_name), server.get("env"), display))
                    self.inventory.findings.extend(_lsp_command_findings(str(server_name), server, display))

    def monitors(self) -> None:
        if self.profile.default_monitors_file is None:
            return  # only Claude Code plugins declare monitors
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
                monitor_name = entry_name or f"{name}[{index}]"
                self._add(Component("monitor", monitor_name, origin, rel, "unsupported"))
                self._monitor_command(monitor_name, entry, rel)
                # Each monitor's own bypass findings land on its own inventory row.
                self.inventory.findings.extend(
                    _override_findings(
                        _agent_cli_flag_issues(entry),
                        self.reader.display(rel),
                        where=f"monitors ({name}) entry '{monitor_name}'",
                        component=("monitor", monitor_name),
                    )
                )
            rest = entries[PLUGIN_COMPONENT_MAX_ITEMS:]
            if rest:
                # Entries past the inventory cap are not rows, but are still scanned.
                self.inventory.findings.extend(
                    _override_findings(
                        _agent_cli_flag_issues(rest),
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
        for finding in analysis.findings:
            # The analyzer tags hook sources; a monitor's findings belong to its own inventory row.
            finding.metadata["plugin_component"] = {"type": "monitor", "name": monitor_name}
        self.inventory.hook_records.extend(analysis.records)
        self.inventory.findings.extend(analysis.findings)

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
        self._codex_config()

    def _codex_config(self) -> None:
        """A shipped Codex project config (``.codex/config.toml``), the Codex twin of ``.claude/settings.json``.

        Codex reads it when the folder is opened as a trusted project, so it is
        checked in every format: ``approval_policy = "never"`` or ``sandbox_mode =
        "danger-full-access"`` (top level or in a profile) is HIGH, and its strings
        and argv lists get the agent-CLI flag checks.
        """
        rel = PurePosixPath(_CODEX_CONFIG_FILE)
        kind = self.reader.kind(rel)
        if kind in {"missing", "dir"}:
            return
        name = rel.as_posix()
        if kind in {"link", "special"}:
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader, "settings", DeclaredPath(name, rel), self.manifest_rel, "unsafe", rel
                )
            )
            self._add(Component("settings", name, "packaged", name, "unsupported", problem="unsafe"))
            return
        self._add(Component("settings", name, "packaged", name, "unsupported"))
        display = self.reader.display(rel)
        import tomllib

        try:
            config = tomllib.loads(self.reader.read_text(rel, PLUGIN_CONFIG_MAX_BYTES, config=True))
        except (SecurePathError, OSError, tomllib.TOMLDecodeError, RecursionError, ValueError) as exc:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_unreadable",
                    f"settings config '{name}' could not be read or parsed safely: {_bounded_text(str(exc), 160)}",
                    display,
                    f"Keep '{name}' a regular UTF-8 TOML file under {PLUGIN_CONFIG_MAX_BYTES} bytes.",
                    metadata={"plugin_component": {"type": "settings", "name": name}},
                )
            )
            return
        component = ("settings", name)
        tables = [("", config)]
        profiles = config.get("profiles")
        if isinstance(profiles, dict):
            tables.extend((f"profiles.{key}.", value) for key, value in list(profiles.items())[:64])
        for prefix, table in tables:
            if not isinstance(table, dict):
                continue
            for key, value in _CODEX_BYPASS_SETTINGS:
                if isinstance(table.get(key), str) and table[key].strip().lower() == value:
                    self.inventory.findings.append(
                        _plugin_finding(
                            Severity.HIGH,
                            "plugin_settings_bypass_permissions",
                            f"shipped Codex config '{name}' sets {prefix}{key} = \"{value}\", which "
                            + (
                                "turns off every approval prompt"
                                if key == "approval_policy"
                                else "turns off the command sandbox"
                            ),
                            display,
                            f"Remove {prefix}{key}; let the user choose Codex's approval policy and sandbox.",
                            metadata={"plugin_component": {"type": "settings", "name": name}},
                        )
                    )
        self.inventory.findings.extend(
            _override_findings(_agent_cli_flag_issues(config), display, where=name, component=component)
        )

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
            mode = permissions.get("defaultMode")
            if mode == "bypassPermissions":
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
            elif isinstance(mode, str) and mode in PERMISSIVE_PERMISSION_MODES:
                self.inventory.findings.append(
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
                    self.inventory.findings.append(
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
        enabled = config.get("enabledMcpjsonServers")
        if isinstance(enabled, list) and enabled:
            names = ", ".join(_bounded_text(str(name), 40) for name in enabled[:8])
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_settings_auto_approve",
                    f"shipped settings '{rel}' lists enabledMcpjsonServers ({names}), auto-approving those project "
                    "MCP servers",
                    display,
                    "Remove enabledMcpjsonServers; let users approve MCP servers explicitly.",
                )
            )
        allow = permissions.get("allow") if isinstance(permissions, dict) else None
        if isinstance(allow, list) and any(
            isinstance(rule, str) and "".join(rule.split()).lower() in {"mcp__*", "mcp__*__*"} for rule in allow
        ):
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_settings_auto_approve",
                    f"shipped settings '{rel}' pre-approves every MCP tool (mcp__*) in permissions.allow",
                    display,
                    "Remove the mcp__* allow rule; pre-approve only the MCP tools the plugin needs, by name.",
                )
            )
        from skillevaluator.validators.mcp_static import env_tls_and_secret_issues

        self.inventory.findings.extend(_override_findings(env_override_issues(config.get("env")), display, where=rel))
        # Claude Code ignores env from plugin settings, but it applies if the file is copied into a project: HIGH.
        self.inventory.findings.extend(
            _override_findings(env_tls_and_secret_issues(config.get("env"), severity=Severity.HIGH), display, where=rel)
        )
        self.inventory.findings.extend(_override_findings(_agent_cli_flag_issues(config), display, where=rel))

    # -- MCP -------------------------------------------------------------- #
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
                "always-on: the server's tool schemas",
                not_counted="MCP tool schemas are not known statically (Tier 3 measures them)",
            )
        for raw, path, problem in collection.broken_sources:
            self._add(Component("mcp", raw, "declared", path, "static_only", problem=problem))
        for raw, path in collection.bundles:
            self._add(Component("mcp", raw, "declared", path, "unsupported", bundle=True))

    # -- Codex apps (connectors) ------------------------------------------ #
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
        for name, origin, rel, config in self._json_sources("apps", declared_value, default, merge_default=False):
            if not rel:
                self._add(Component("app", name, origin, None, "unsupported", problem=str(config)))
                continue
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
            if namespace not in directories:
                continue
            base = PurePosixPath(namespace)
            for component_type, folder in (("agent", "agents"), ("command", "commands")):
                if self.reader.kind(base / folder) == "dir":
                    self._markdown_dir(component_type, base / folder, "packaged")
            if self.reader.kind(base / "rules") == "dir":
                self._rule_dir(base / "rules", "packaged", support="unsupported")
            hooks_file = base / "hooks" / "hooks.json"
            if self.reader.kind(hooks_file) == "file":
                self._hook(
                    hooks_file.as_posix(), "packaged", hooks_file.as_posix(), self._load_json(hooks_file, "hooks")
                )

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
        profile = self.profile
        self.profile = CODEX_PROFILE  # Codex path rules and messages for these fields
        try:
            if openai.get("hooks") is not None:
                for name, origin, rel, config in self._json_sources("hooks", openai.get("hooks"), None):
                    label = f"extensions.com.openai.hooks:{name}" if rel == self.manifest_rel else name
                    self._hook(label, origin, rel, config)
            if openai.get("apps") is not None:
                for name, origin, rel, config in self._json_sources("apps", openai.get("apps"), None):
                    apps = config.get("apps") if rel and isinstance(config, dict) else None
                    if not isinstance(apps, dict) or not apps:
                        problem = None if rel else str(config)
                        self._add(Component("app", name, origin, rel or None, "unsupported", problem=problem))
                        continue
                    for alias in list(apps)[:PLUGIN_COMPONENT_MAX_ITEMS]:
                        self._add(Component("app", str(alias), origin, rel, "unsupported"))
        finally:
            self.profile = profile

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
        self.finish_costs()
        return self.inventory


def _markdown_cost(component_type: str, name: str, parsed: _Markdown) -> CostRow:
    """The cost row of an agent, command, or output style (a forced style is resolved per view)."""
    frontmatter = parsed.frontmatter
    body = cost_chars(parsed.body)
    if component_type == "output_style":
        if _claude_flag(frontmatter.get("force-for-plugin")):
            traits = {"forced_style"}
            if _claude_flag(frontmatter.get("keep-coding-instructions")):
                traits.add("keeps_coding_instructions")
            return CostRow(
                component_type, name, body, 0, "always-on: forced output style body", traits=frozenset(traits)
            )
        return CostRow(component_type, name, 0, body, "on-demand: output style body when selected")
    label = "agent" if component_type == "agent" else "command"
    basis = (
        "always-on: description; on-demand: subagent prompt (loads in the subagent's own context)"
        if label == "agent"
        else "always-on: description; on-demand: command body"
    )
    return CostRow(
        component_type, name, cost_chars(parsed.description or ""), body, basis, traits=model_hidden_traits(frontmatter)
    )


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
# Folders the whole-tree scans skip                                           #
# --------------------------------------------------------------------------- #
# Dependency folders and VCS metadata a plugin may ship that Tier 1 whole-tree
# scans skip. Their presence is noted (LOW) so a reader knows what was not
# scanned. A .git folder at the plugin root is the checkout's own metadata.
_NOTED_UNSCANNED_DIRS = frozenset({"node_modules", ".venv", ".git"})
_MAX_UNSCANNED_LINK_FINDINGS = 20
# A virtual environment links its interpreter to the system Python by design.
_VENV_INTERPRETER_RE = re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)?)?(?:\.exe)?$")


@dataclass
class _PrunedWalk:
    noted: list[PurePosixPath] = field(default_factory=list)
    escaping_links: list[PurePosixPath] = field(default_factory=list)
    complete: bool = True


def _link_leaves_root(real_root: str, directory: str, name: str) -> bool:
    """Whether a link (or reparse point) leads outside the plugin root; unresolvable links count as outside.

    An absolute target is outside: it names a host path, not a plugin file.
    A relative target is resolved through every link on the way
    (``os.path.realpath``; targets are never read), so a chain such as
    ``node_modules/up -> ..`` and ``node_modules/evil -> up/../outside.md``
    is caught, and compared with the resolved root (``real_root``). A link
    loop or a path that cannot be resolved counts as outside; a dangling link
    whose missing target would be inside the root does not.
    """
    link = os.path.join(directory, name)  # noqa: PTH118 - plain string path
    try:
        target = os.readlink(link)  # noqa: PTH115 - reads the link text only
    except (OSError, ValueError):
        return True
    if not target or PurePosixPath(target).is_absolute() or _WINDOWS_DRIVE_RE.match(target) or target[0] in "/\\":
        return True
    try:
        resolved = os.path.realpath(link, strict=True)
    except (FileNotFoundError, NotADirectoryError):
        resolved = os.path.realpath(link)  # dangling: resolve as far as the existing parts go
    except (OSError, ValueError, RuntimeError):
        return True  # a loop, or a part that cannot be inspected
    return resolved != real_root and not resolved.startswith(real_root.rstrip(os.sep) + os.sep)


def _is_venv_interpreter(directory: Path, name: str) -> bool:
    """A ``python*`` link in the ``bin/`` (``Scripts/``) folder of a virtual environment (next to ``pyvenv.cfg``)."""
    if directory.name not in {"bin", "Scripts"} or not _VENV_INTERPRETER_RE.match(name):
        return False
    try:
        return stat.S_ISREG((directory.parent / "pyvenv.cfg").lstat().st_mode)
    except OSError:
        return False


def _walk_pruned_folders(root: Path) -> _PrunedWalk:
    """Names-only walk that finds the folders the whole-tree walk prunes and the links inside them.

    The whole-tree verification (``verify_plugin_tree``) refuses every link
    outside these folders and never enters them. Inside them, a link that
    leads out of the plugin root is recorded; links that stay inside (for
    example ``node_modules/.bin`` entries) and a virtual environment's
    interpreter links are not. Links are never followed, and the walk is
    bounded like the whole-tree walk.
    """
    walk = _PrunedWalk()
    root_path = Path(os.path.abspath(os.fspath(root)))  # noqa: PTH100 - lexical, never resolved
    real_root = os.path.realpath(root_path)  # only to compare where links lead
    budgets = {False: PLUGIN_TREE_MAX_DISCOVERED_PATHS, True: PLUGIN_TREE_MAX_DISCOVERED_PATHS}
    # (directory, root-relative path, inside a pruned folder)
    pending: list[tuple[Path, PurePosixPath, bool]] = [(root_path, PurePosixPath(), False)]
    while pending:
        directory, rel_dir, pruned = pending.pop()
        try:
            if stat_is_link_or_reparse(directory.lstat()):
                continue
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            walk.complete = False
            continue
        for entry in entries:
            budgets[pruned] -= 1
            if budgets[pruned] < 0:
                walk.complete = False
                break
            rel = rel_dir / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat_is_link_or_reparse(metadata):
                if (
                    pruned
                    and not _is_venv_interpreter(directory, entry.name)
                    and _link_leaves_root(real_root, str(directory), entry.name)
                ):
                    walk.escaping_links.append(rel)
                continue
            if not stat.S_ISDIR(metadata.st_mode) or len(rel.parts) >= MAX_SECURE_DIRECTORY_DEPTH:
                continue
            enters_pruned = not pruned and entry.name in PLUGIN_TREE_PRUNED_DIRS
            if enters_pruned and entry.name == ".git" and not rel_dir.parts:
                continue  # the checkout's own metadata: no client loads it, and it can be large
            if enters_pruned and entry.name in _NOTED_UNSCANNED_DIRS:
                walk.noted.append(rel)
            pending.append((Path(entry.path), rel, pruned or enters_pruned))
    return walk


def _pruned_folder_findings(reader: PluginRootReader) -> list[Finding]:
    """Say which shipped folders Tier 1 does not scan, and refuse links out of the plugin inside them."""
    walk = _walk_pruned_folders(reader.root)
    findings: list[Finding] = []
    for rel in walk.escaping_links[:_MAX_UNSCANNED_LINK_FINDINGS]:
        folder = _unscanned_folder(rel) or rel.parts[0]
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_unscanned_folder_link",
                f"'{rel.as_posix()}' is a symlink or reparse point that leads outside the plugin root, inside "
                f"'{folder}/', a folder Tier 1 whole-tree scans skip. A client that runs or reads it uses content "
                "SkillEvaluator never checked",
                reader.display(rel),
                "Replace the link with the regular file or folder it points to, inside the plugin root, or remove it.",
                metadata={"path": rel.as_posix()},
            )
        )
    if len(walk.escaping_links) > _MAX_UNSCANNED_LINK_FINDINGS:
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_unscanned_folder_link",
                f"{len(walk.escaping_links)} links lead outside the plugin root from folders Tier 1 scans skip; only "
                f"the first {_MAX_UNSCANNED_LINK_FINDINGS} are listed",
                reader.display("."),
                "Replace links with regular files and folders inside the plugin root.",
            )
        )
    if walk.noted:
        shown = ", ".join(f"'{rel.as_posix()}/'" for rel in walk.noted[:10])
        more = f" and {len(walk.noted) - 10} more" if len(walk.noted) > 10 else ""
        findings.append(
            _plugin_finding(
                Severity.LOW,
                "plugin_unscanned_folders",
                f"Tier 1 whole-tree security, secret, and Unicode scans skip {shown}{more} (dependency folders and "
                "VCS metadata). Their files are not scanned; a declared component or MCP server that loads from "
                "them is reported separately",
                reader.display("."),
                "Do not ship dependency folders; let the client install pinned dependencies, or vendor reviewed "
                "code outside these folders.",
                metadata={"paths": [rel.as_posix() for rel in walk.noted[:10]]},
            )
        )
    if not walk.complete:
        findings.append(
            _plugin_finding(
                Severity.LOW,
                "plugin_unscanned_folder_scan_incomplete",
                f"the link check of folders Tier 1 scans skip stopped early (more than "
                f"{PLUGIN_TREE_MAX_DISCOVERED_PATHS} entries, or a folder could not be listed); later entries were "
                "not checked for links",
                reader.display("."),
                "Keep dependency folders and generated output out of the plugin package.",
            )
        )
    return findings


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
    profile: FormatProfile = CLAUDE_PROFILE,
) -> McpCollection:
    """Collect every MCP server a plugin declares, in Claude Code load order.

    Claude Code loads ``.mcp.json`` at the plugin root first, then each declared
    ``mcpServers`` shape in order (inline map, ``.json`` path, or an array mixing
    them); a later same-name server replaces an earlier one. A config file is
    either ``{"mcpServers": {...}}`` or a bare server map (both documented).
    ``.mcpb``/``.dxt`` bundles are recorded but not inspected. For
    ``agent_plugin.yaml`` the ``mcp`` list is used (Pydantic validates it).

    ``profile`` supplies the format's default MCP files (``.mcp.json`` for
    Claude Code and Codex, ``mcp.json`` for Cursor and Agent Plugins), whether a
    declared ``mcpServers`` replaces them, and the config dialect: Codex and
    Agent Plugins entries are normalized (``streamable-http`` is ``http``, Codex
    ``http_headers`` are merged into ``headers``) before the static policy runs.

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
        # Codex drops an mcpServers value of the wrong type, loads the default .mcp.json, and still installs the
        # plugin (MEDIUM, as check 1 rates it); Claude Code refuses the manifest.
        codex_ignores = contained and profile.codex_path_rules
        collection.findings.append(
            _plugin_finding(
                Severity.MEDIUM if codex_ignores else Severity.HIGH,
                "mcp_servers_not_object",
                "'mcpServers' must be an inline server map, a config-file path string, or an array of those "
                f"(got {type(declared).__name__})"
                + (
                    "; Codex ignores the value, loads the default .mcp.json, and still installs the plugin"
                    if codex_ignores
                    else ""
                ),
                manifest_display,
                'Use {"<name>": {"command"|"url"|"provider": ...}}, "./.mcp.json", or an array mixing both.',
                category=MCP_CATEGORY,
            )
        )
        entries = []

    explicit_files: set[PurePosixPath] = set()
    for entry in entries:
        if isinstance(entry, str) and not (contained and profile.codex_path_rules and not codex_accepts_path(entry)):
            normalized = normalize_declared_path(entry, profile.manifest_path_prefixes)
            if normalized.rel is not None:
                explicit_files.add(normalized.rel)

    # 1. The default root MCP file (unless the manifest names it explicitly, or
    #    the format lets a declared mcpServers replace default discovery; Codex
    #    only when it keeps the declared value).
    replace_default = contained and declared_value_replaces_default(profile, "mcpServers", declared)
    for default_name in profile.default_mcp_files if not replace_default else ():
        default_rel = PurePosixPath(default_name)
        if default_rel in explicit_files:
            continue
        kind = reader.kind(default_rel)
        if kind == "file":
            _load_mcp_file(reader, collection, default_rel, "mcp_json", default_rel.as_posix())
            if contained and declared is not None and "mcpServers" in profile.merged_default_fields:
                collection.findings.append(_merged_default_mcp_note(reader, default_rel, profile))
        elif kind in {"link", "special"}:
            collection.findings.append(
                _path_problem_finding(
                    reader, "mcpServers", DeclaredPath(default_name, default_rel), manifest_rel, "unsafe", default_rel
                )
            )
            collection.broken_sources.append((default_name, default_rel.as_posix(), "unsafe"))

    # 2. Declared shapes in order.
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            # A Cursor/Codex inline value may also use the {"mcpServers": {...}} wrapper.
            inline = entry.get("mcpServers") if isinstance(entry.get("mcpServers"), dict) else entry
            if len(inline) > PLUGIN_COMPONENT_MAX_ITEMS:
                collection.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "mcp_config_file_too_large",
                        f"inline 'mcpServers' map declares {len(inline)} servers; only "
                        f"{PLUGIN_COMPONENT_MAX_ITEMS} are inspected",
                        manifest_display,
                        "Reduce the number of MCP servers per map.",
                        category=MCP_CATEGORY,
                    )
                )
            for name, config in list(inline.items())[:PLUGIN_COMPONENT_MAX_ITEMS]:
                collection.declarations.append(McpDeclaration(str(name), config, "inline", manifest_rel))
        elif isinstance(entry, str):
            _collect_path_ref(reader, collection, entry, manifest_rel, profile, contained=contained)
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

    # 3. Dialect normalization (Codex, Agent Plugins) before any static check.
    if profile.mcp_dialect != "claude":
        _normalize_dialect(reader, collection, profile, manifest)

    # 4. Duplicate names across Claude Code sources (later replaces earlier).
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
            server_findings = validate_mcp_server_declaration(
                declaration.name,
                declaration.config,
                reader.display(declaration.file),
                allowed_private_hosts=allowed_private_hosts,
                manifest_type=profile.manifest_type,
            )
            for finding in server_findings:
                # Every finding here is about this one server (an invalid name included), so the
                # inventory row of the server counts it.
                if isinstance(finding.metadata, dict):
                    finding.metadata.setdefault("mcp_server", declaration.name)
            collection.server_findings.extend(server_findings)
            collection.server_findings.extend(_unscanned_server_files(reader, declaration, profile))
    return collection


def _merged_default_mcp_note(reader: PluginRootReader, default_rel: PurePosixPath, profile: FormatProfile) -> Finding:
    """LOW: a format's root MCP file is checked next to a declared ``mcpServers`` (Cursor)."""
    return _plugin_finding(
        Severity.LOW,
        "mcp_root_config_also_checked",
        f"The {profile.label} declares mcpServers and also ships '{default_rel.as_posix()}'. The format's reference "
        "says a declared field replaces the default location, but the client loader may read both files, so "
        "SkillEvaluator checks the servers in both",
        reader.display(default_rel),
        f"Move the servers of '{default_rel.as_posix()}' into the declared mcpServers file, or remove "
        f"'{default_rel.as_posix()}'.",
        category=MCP_CATEGORY,
    )


# A root placeholder followed by a plugin path, anywhere in a command or argument.
_ROOT_VARIABLE_PATH_RE = re.compile(
    r"\$\{?(?:CLAUDE_PLUGIN_ROOT|PLUGIN_ROOT|CURSOR_PLUGIN_ROOT)\}?/([^\s\"'`;|&<>()]+)"
)
# A root placeholder alone: the plugin root itself.
_ROOT_VARIABLE_ONLY_RE = re.compile(r"^\$\{?(?:CLAUDE_PLUGIN_ROOT|PLUGIN_ROOT|CURSOR_PLUGIN_ROOT)\}?/?$")


def _launch_path(value: str) -> str:
    """The path part of a launch value: the value of a ``--flag=path`` argument, else the value itself."""
    text = value.strip().replace("\\", "/")
    return text.split("=", 1)[1] if text.startswith("-") and "=" in text else text


def _lexical_plugin_path(candidate: str) -> PurePosixPath | None:
    """A plugin-root-relative path with ``.`` and ``..`` resolved lexically; ``None`` when it leaves the root.

    ``scripts/../node_modules/srv/index.js`` is ``node_modules/srv/index.js``;
    a path that climbs out of the root is not a plugin file.
    """
    if candidate.startswith(("/", "~", "$")) or _WINDOWS_DRIVE_RE.match(candidate):
        return None
    resolved = posixpath.normpath(candidate or ".")
    if resolved == ".." or resolved.startswith("../"):
        return None
    return PurePosixPath(resolved)


def _rooted_launch_paths(value: str) -> list[PurePosixPath]:
    """Plugin files a launch value names from the plugin root: after a root variable, or with a leading ``./``."""
    candidates = [match.group(1) for match in _ROOT_VARIABLE_PATH_RE.finditer(value)]
    path = _launch_path(value)
    if path.startswith("./"):
        candidates.append(path[2:])
    return [rel for rel in map(_lexical_plugin_path, candidates) if rel is not None]


def _launch_cwd(cwd: Any, *, relative_cwd: bool) -> PurePosixPath | None:
    """The plugin folder an MCP server runs in, from its ``cwd``, or ``None``.

    A root variable or a ``./`` path names a plugin folder in any format.
    Codex (``relative_cwd``) also resolves a bare relative ``cwd``, ``.``
    included, against the plugin folder.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        return None
    text = cwd.strip().replace("\\", "/")
    if _ROOT_VARIABLE_ONLY_RE.match(text):
        return PurePosixPath(".")
    rooted = _rooted_launch_paths(text)
    if rooted:
        return rooted[0]
    if not relative_cwd:
        return None
    return _lexical_plugin_path(text)


def _server_launch_files(config: dict[str, Any], *, relative_cwd: bool) -> list[tuple[PurePosixPath, bool]]:
    """Plugin-root-relative paths an MCP server launch runs or reads, in order, each with whether it is the ``cwd``.

    ``command`` and ``args`` name plugin files through a root variable
    (``${CLAUDE_PLUGIN_ROOT}/node_modules/srv/server.py``), a ``./`` path,
    or a ``--flag=./path`` argument. A ``cwd`` inside the plugin is a plugin
    folder itself, and relative ``command`` and ``args`` values resolve
    against it (``cwd: "node_modules/srv"`` with ``args: ["index.js"]``).
    """
    values: list[str] = []
    if isinstance(config.get("command"), str):
        values.append(config["command"])
    args = config.get("args")
    if isinstance(args, list):
        values.extend(arg for arg in args[:PLUGIN_COMPONENT_MAX_ITEMS] if isinstance(arg, str))
    paths: list[tuple[PurePosixPath, bool]] = []
    for value in values:
        paths.extend((rel, False) for rel in _rooted_launch_paths(value))
    base = _launch_cwd(config.get("cwd"), relative_cwd=relative_cwd)
    if base is None:
        return paths
    if _in_unscanned_folder(base):
        return [*paths, (base, True)]  # everything the server runs from there is in the skipped folder
    for value in values:
        path = _launch_path(value)
        if not path or path.startswith(("-", "/", "~", "$")) or "://" in path or _WINDOWS_DRIVE_RE.match(path):
            continue  # an option, a URL, or an absolute, home, or variable path
        joined = _lexical_plugin_path(path if str(base) == "." else f"{base.as_posix()}/{path}")
        if joined is not None:
            paths.append((joined, False))
    return paths


def _unscanned_server_files(
    reader: PluginRootReader, declaration: McpDeclaration, profile: FormatProfile = CLAUDE_PROFILE
) -> list[Finding]:
    """HIGH when an MCP server runs a plugin file from a folder that Tier 1 whole-tree scans skip.

    The server's launch names plugin files (:func:`_server_launch_files`):
    a root variable, a ``./`` or ``--flag=./`` path, ``..`` resolved
    lexically, and its ``cwd`` (Codex resolves a relative ``cwd`` against the
    plugin folder). Code there (``node_modules/``, ``.venv/``, ...) is never
    security-scanned, so the server is reported like any other component that
    loads from such a folder. A virtual environment's own interpreter
    (``.venv/bin/python3``) is not plugin code and is not reported.
    """
    config = declaration.config
    if not isinstance(config, dict):
        return []
    seen: set[str] = set()
    findings: list[Finding] = []
    for rel, is_cwd in _server_launch_files(config, relative_cwd=profile.mcp_dialect == "codex"):
        folder = _unscanned_folder(rel)
        if folder is None or rel.as_posix() in seen:
            continue
        parts = rel.parts
        if len(parts) >= 2 and parts[-2] in {"bin", "Scripts"} and _VENV_INTERPRETER_RE.match(parts[-1]):
            continue  # a virtual environment's interpreter; the script it runs is checked on its own
        seen.add(rel.as_posix())
        verb = "runs in" if is_cwd else "runs"
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_component_path_unscanned",
                f"MCP server '{declaration.name}' {verb} '{rel.as_posix()}', which is inside '{folder}/', "
                f"a folder that Tier 1 whole-tree scans skip ({_UNSCANNED_FOLDERS_TEXT}); the server code is never "
                "security-scanned",
                reader.display(declaration.file),
                "Ship the server entry point outside dependency, VCS, evaluation-output, and version-snapshot "
                "folders, or run a pinned package with a package runner.",
                category=MCP_CATEGORY,
                metadata={"mcp_server": declaration.name},
            )
        )
    return findings


def _normalize_dialect(
    reader: PluginRootReader, collection: McpCollection, profile: FormatProfile, manifest: dict[str, Any] | None
) -> None:
    """Map Codex and Agent Plugins MCP entries onto the fields the static policy checks.

    Codex (``McpServerConfig``): ``type`` is optional and ``streamable_http`` /
    ``streamable-http`` mean ``http``; ``http_headers`` hold literal headers, so
    they are merged into ``headers`` and checked with them (see
    :func:`_codex_headers`); ``env_vars``, ``env_http_headers``,
    ``bearer_token_env_var``, ``http_headers_helper``, and ``oauth`` configure
    auth or environment that the evaluation runtime does not apply. Codex
    rejects an inline ``bearer_token`` in plugins.

    Agent Plugins 1.0.0 (``mcp.json``): ``$schema`` must name the MCP schema of
    the same version as ``plugin.json`` (otherwise clients disable MCP), and
    each server must declare ``type`` (``stdio``, ``streamable-http``, or
    ``sse``); ``streamable-http`` is checked as ``http``.
    """
    if profile.mcp_dialect == "agent_plugins":
        plugin_version = agent_plugins_schema_version((manifest or {}).get("$schema"))
        for file_rel, schema in collection.file_schemas.items():
            mcp_version = agent_plugins_schema_version(schema, kind="mcp")
            if mcp_version is None or (plugin_version is not None and mcp_version != plugin_version):
                expected = plugin_version or "1.0.0"
                collection.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "mcp_config_schema_mismatch",
                        f"MCP config '{file_rel}' must declare \"$schema\": "
                        f'"https://agent-plugins.org/schemas/{expected}/mcp.schema.json" (the Agent Plugins version '
                        "of plugin.json); clients disable the plugin's MCP servers otherwise",
                        reader.display(file_rel),
                        "Declare the Agent Plugins MCP schema of the same version as plugin.json.",
                        category=MCP_CATEGORY,
                    )
                )
    for declaration in collection.declarations:
        config = declaration.config
        if not isinstance(config, dict):
            continue
        normalized = dict(config)
        raw_type = normalized.get("type")
        if isinstance(raw_type, str) and raw_type.strip() in _HTTP_TYPE_ALIASES:
            normalized["type"] = "http"
        if profile.mcp_dialect == "agent_plugins" and "type" not in config:
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_transport_missing",
                    f"MCP server '{declaration.name}' in '{declaration.file}' must declare 'type' (stdio, "
                    "streamable-http, or sse); Agent Plugins clients skip servers without it",
                    reader.display(declaration.file),
                    'Add "type": "stdio" (or streamable-http / sse) to the server.',
                    category=MCP_CATEGORY,
                    metadata={"mcp_server": declaration.name},
                )
            )
        if profile.mcp_dialect == "codex":
            http_headers = normalized.pop("http_headers", None)
            if http_headers is not None:
                normalized["headers"] = _codex_headers(normalized.get("headers"), http_headers)
            declaration.unapplied = tuple(key for key in _CODEX_UNAPPLIED_MCP_FIELDS if config.get(key))
            if "bearer_token" in config:
                collection.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "mcp_inline_bearer_token",
                        f"MCP server '{declaration.name}' declares an inline 'bearer_token'; Codex rejects it in "
                        "plugins, and a literal token is a shipped credential (value not shown)",
                        reader.display(declaration.file),
                        "Use 'bearer_token_env_var' to name an environment variable instead.",
                        category=MCP_CATEGORY,
                        metadata={"mcp_server": declaration.name},
                    )
                )
                normalized.pop("bearer_token", None)
        declaration.config = normalized


def _codex_headers(headers: Any, http_headers: Any) -> Any:
    """Merge a Codex server's ``http_headers`` into ``headers`` for the static policy and Tier 3.

    Every header either map declares is kept, so a literal credential in either
    one is reported. On a key collision the ``http_headers`` value (the one Codex
    sends) wins, unless only the ``headers`` value is an inline credential. A map
    that is not an object is kept as is, for the policy to report.
    """
    if headers is None:
        return http_headers
    if not isinstance(headers, dict):
        return headers
    if not isinstance(http_headers, dict):
        return http_headers
    merged = dict(headers)
    for key, value in http_headers.items():
        shadowed = merged.get(key)
        if _is_inline_secret(key, shadowed) and not _is_inline_secret(key, value):
            continue
        merged[key] = value
    return merged


def _is_inline_secret(key: Any, value: Any) -> bool:
    return isinstance(value, str) and looks_like_inline_secret(str(key), value)


def _collect_path_ref(
    reader: PluginRootReader,
    collection: McpCollection,
    raw: str,
    manifest_rel: str,
    profile: FormatProfile = CLAUDE_PROFILE,
    *,
    contained: bool = True,
) -> None:
    manifest_display = reader.display(manifest_rel)
    lowered = raw.strip().lower()
    if lowered.startswith(("https://", "http://")):
        shown = redacted_url(raw)  # reports and the inventory never carry userinfo or query credentials
        if lowered.split("?", 1)[0].endswith(_MCP_BUNDLE_SUFFIXES):
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
            if lowered.startswith("http://"):
                # Code fetched over plaintext can be swapped in transit: block it like a plaintext url server,
                # and record a broken source so Tier 3 staging fails closed.
                collection.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "mcp_url_insecure_scheme",
                        f"mcpServers bundle {shown!r} is downloaded over plaintext http; the code it runs can be "
                        "replaced in transit",
                        manifest_display,
                        "Serve the bundle over https:// or vendor it into the plugin.",
                        category=MCP_CATEGORY,
                    )
                )
                collection.broken_sources.append((shown, None, "invalid"))
            else:
                collection.bundles.append((shown, None))
            return
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_path_invalid",
                f"mcpServers URL {shown!r} is not an .mcpb/.dxt bundle; only bundles may be referenced by URL",
                manifest_display,
                "Reference a './'-relative .json config file, an inline server map, or an .mcpb bundle.",
                category=MCP_CATEGORY,
                metadata={"plugin_component_ref": shown},
            )
        )
        collection.broken_sources.append((shown, None, "invalid"))
        return

    # Codex drops a path it does not accept and loads the default .mcp.json instead.
    dropped = contained and profile.codex_path_rules and not codex_accepts_path(raw)
    declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
    if declared.problem is not None or declared.rel is None:
        problem = "escape" if declared.problem == "escape" else "invalid"
        collection.findings.append(
            _path_problem_finding(reader, "mcpServers", declared, manifest_rel, problem, reference=profile.reference)
        )
        if not dropped:
            collection.broken_sources.append((raw, None, problem))
        return
    rel = declared.rel
    if dropped:
        collection.findings.append(_style_finding(reader, "mcpServers", declared, manifest_rel, profile=profile))
        return
    if profile.require_dot_relative and not declared.dot_relative:
        collection.findings.append(_style_finding(reader, "mcpServers", declared, manifest_rel, profile=profile))
    if contained and declared.root_variable is not None:
        collection.findings.append(_root_variable_finding(reader, "mcpServers", declared, manifest_rel, profile))
    if (unscanned := _unscanned_path_finding(reader, "mcpServers", declared, manifest_rel)) is not None:
        collection.findings.append(unscanned)
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
                metadata={"plugin_component_ref": raw},
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
                    metadata={"plugin_component_ref": raw},
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
                    metadata={"plugin_component_ref": raw},
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
                    metadata={"plugin_component_ref": raw},
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
                metadata={"plugin_component_ref": raw},
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
                metadata={"plugin_component_ref": raw},
            )
        )
        collection.broken_sources.append((raw, rel.as_posix(), "invalid"))
        return
    if isinstance(data, dict):
        collection.file_schemas[rel.as_posix()] = data.get("$schema")
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
                    metadata={"plugin_component_ref": raw},
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
                metadata={"plugin_component_ref": raw},
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
    from skillevaluator.plugin_manifest import manifest_relative_path

    relative = manifest_relative_path(manifest_path)
    if relative is not None and len(relative.parts) > 1:
        return relative.as_posix()
    try:
        return manifest_path.relative_to(root).as_posix()
    except ValueError:
        return manifest_path.name


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
    additional: list[tuple[str, str, dict[str, Any] | None]] = []
    if located is None:
        from skillevaluator.cli_core import manifestless_plugin_markers

        if manifestless_plugin_markers(root):
            # Claude Code --plugin-dir loads a folder without a manifest from its default locations.
            return build_plugin_inventory(
                root, None, contained=True, manifest_rel=CLAUDE_PROFILE.manifest_path, manifestless=True
            )
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
        for candidate in located.additional:
            extra = candidate.parse_for_audit()
            if extra is not None:
                additional.append((candidate.manifest_type, candidate.manifest_filename, extra))
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
    manifestless: bool = False,
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

    ``manifestless`` inventories a folder without any manifest the way Claude
    Code ``--plugin-dir`` loads it: its default locations only. No other client
    loads such a folder, so there is no cross-client pass.

    Every inventory also reports the shipped folders the whole-tree scans skip
    (``node_modules/``, ``.venv/``, nested ``.git/``) and refuses links out of
    the plugin root inside any skipped folder.
    """
    allowed_private_hosts = tuple(allowed_private_hosts)
    hook_allowed_urls = tuple(hook_allowed_urls)
    if manifestless:
        inventory = _Builder(
            root,
            None,
            contained=True,
            manifest_rel=CLAUDE_PROFILE.manifest_path,
            allowed_private_hosts=allowed_private_hosts,
            hook_allowed_urls=hook_allowed_urls,
            profile=CLAUDE_PROFILE,
        ).build()
        inventory.findings.extend(_pruned_folder_findings(PluginRootReader(root)))
        return inventory

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
    # Whether every file Claude Code loads from this folder is in the inventory: it reads a .claude-plugin
    # manifest, or (in a folder without one) the --plugin-dir defaults view merged below.
    claude_covered = any(loaded_type == PLUGIN_CONTAINED_MANIFEST_TYPE for loaded_type, _data, _rel in loaded)
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
            claude_covered=claude_covered,
        )
        claude_covered = claude_covered or profile is _CLAUDE_DEFAULTS_VIEW
    inventory.findings.extend(_pruned_folder_findings(PluginRootReader(root)))
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
    claude_covered: bool = False,
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

    ``claude_covered`` says the inventory already holds every file Claude Code
    loads from the folder. Then a grant that only Claude Code would honor (Codex
    ignores ``allowed-tools``) in a file only this view loads has no effect in
    either client (:func:`_inert_grant_finding`).
    """
    covered = _covered_paths(inventory) | _MANIFEST_PATHS | set(manifest_files)
    # A skill is covered by its folder, but its grant findings point at its SKILL.md.
    covered.update(
        f"{component.path}/SKILL.md"
        for component in inventory.components
        if component.type == "skill" and component.path not in {None, "", "."}
    )
    components = [component for component in view.components if component.type not in _CROSS_CLIENT_SKIPPED_TYPES]
    for component in components:
        component.declared_by = declared_by or component.path or loaded_by
        component.loaded_by = loaded_by
    # A finding the inventory already has (the same file seen by both views, such as a shipped .env) is
    # dropped before the loading client is named in the others.
    seen = {_finding_key(finding) for finding in inventory.findings}
    findings = [
        finding
        for finding in view.findings
        if _finding_rel_path(str(finding.file_path or ""), root) not in covered and _finding_key(finding) not in seen
    ]
    privilege_records = [record for record in view.privilege_records if record.path not in covered]
    for finding in findings:
        if claude_covered and finding.metadata.get("claude_only_grant"):
            _inert_grant_finding(finding)
        _cross_client_finding(finding, loaded_by)
    for record in privilege_records:
        if claude_covered and "claude_only_grant" in record.flags:
            record.flags[:] = ["inert_grant" if flag == "claude_only_grant" else flag for flag in record.flags]
        if any(flag not in _BENIGN_PRIVILEGE_RECORD_FLAGS for flag in record.flags):
            record.flags.append("cross_client_only")
    extra = PluginInventory(
        components=components,
        findings=findings,
        hook_records=[record for record in view.hook_records if record.file not in covered],
        privilege_records=privilege_records,
    )
    _merge_additional(inventory, extra, None)


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
    by_exact_path: dict[str, Component] = {}
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
            target = by_type_name.get((str(tagged.get("type")), str(tagged.get("name"))))
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


def _component_for_ref(ref: str, by_exact_path: dict[str, Component]) -> Component | None:
    """The component whose root-relative path a declared ref names, compared after normalization.

    ``./agents/x.md``, ``agents/x.md``, and ``${CLAUDE_PLUGIN_ROOT}/agents/x.md``
    all name the component at ``agents/x.md``, whatever its frontmatter name.
    """
    declared = normalize_declared_path(ref, _ALL_ROOT_VARIABLES)
    if declared.rel is None or declared.problem is not None:
        return None
    return by_exact_path.get(declared.rel.as_posix())


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
                problem=row.get("problem") if isinstance(row.get("problem"), str) else None,
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
        # ``loaded`` (native load census) and ``exercised`` (runtime evidence)
        # rank above ``staged``; they are set after the run.
        "not_evaluated": sum(1 for row in rows if row["state"] not in {"staged", "loaded", "exercised"}),
    }


def problem_reason(component: Component) -> str:
    return {
        "missing": "declared path does not exist",
        "escape": "declared path escapes the plugin root",
        "unsafe": "path is a symlink, hard link, or special file and was not followed",
        "invalid": "declaration is invalid for this component type",
    }.get(component.problem or "", "component could not be loaded")
