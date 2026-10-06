# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-format schemas and component mappings for contained plugin manifests.

SkillEvaluator supports these native plugin manifest formats. Each is a
*contained* plugin: its components ship inside the plugin root.

* **Claude Code** (``.claude-plugin/plugin.json``). Source: the manifest schema
  that Claude Code 2.1.284 applies when it loads a plugin.
* **Agent Plugins v1** (root ``plugin.json`` declaring an
  ``https://agent-plugins.org/schemas/1.x.y/plugin.schema.json`` ``$schema``).
  Source: the Agent Plugins 1.0.0 specification, ``spec/1.0.0.md`` and
  ``schemas/1.0.0/{plugin,mcp}.schema.json`` in
  ``github.com/agentplugins/agent-plugins-spec`` at ``ff8ab5e``. 1.1.0 is a
  working draft that Codex and Hermes do not accept yet, so it is an
  unrecognized version like any other 1.x.
* **Codex** (``.codex-plugin/plugin.json``). Sources: ``plugins/plugin-eval``
  and the ``plugin-creator`` manifest reference in ``github.com/openai/plugins``
  at ``610a632``, the Codex runtime loader, and
  ``developers.openai.com/plugins/build/plugins``.
* **Cursor** (``.cursor-plugin/plugin.json``). Source: the Cursor plugins
  reference (``cursor.com/docs/reference/plugins``) and
  ``schemas/plugin.schema.json`` in ``github.com/cursor/plugins`` at
  ``4b4d98e``.

Each format has a :class:`FormatProfile` that tells the static inventory which
manifest fields declare components and where each component type lives by
default. :func:`validate_manifest_fields` checks the required fields of the
format and returns neutral :class:`ManifestIssue` rows; the plugin schema
validator turns them into findings.

Assumptions are recorded next to each profile, so a reader can see exactly
what was taken from a specification and what SkillEvaluator decided.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from skillevaluator.constants import (
    NAME_MAX_LENGTH,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_DIR,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CURSOR_MANIFEST_DIR,
    PLUGIN_CURSOR_MANIFEST_TYPE,
    PLUGIN_MANIFEST_TYPE,
)
from skillevaluator.utils.structured_data import load_bounded_json, load_bounded_yaml

IssueLevel = Literal["error", "warning", "note"]

# --------------------------------------------------------------------------- #
# Agent Plugins version identifiers                                           #
# --------------------------------------------------------------------------- #
AGENT_PLUGINS_SCHEMA_PREFIX = "https://agent-plugins.org/schemas/"
# Versions whose rules SkillEvaluator implements and that clients load. The
# 1.1.0 working draft has the same rules, but Codex and Hermes accept only the
# exact 1.0.0 schema identifiers, so it is not listed here.
AGENT_PLUGINS_SUPPORTED_VERSIONS = ("1.0.0",)
_AGENT_PLUGINS_SCHEMA_RE = re.compile(
    r"^https://agent-plugins\.org/schemas/(?P<version>\d+\.\d+\.\d+)/(?P<kind>plugin|mcp)\.schema\.json$"
)
_AGENT_PLUGINS_NAME_RE = re.compile(r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_CURSOR_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def agent_plugins_schema_version(value: Any, *, kind: str = "plugin") -> str | None:
    """Return the Agent Plugins version named by a ``$schema`` URL, else ``None``.

    The comparison is exact, as in Codex and Hermes: surrounding whitespace
    makes the value an invalid identifier.
    """
    if not isinstance(value, str):
        return None
    match = _AGENT_PLUGINS_SCHEMA_RE.match(value)
    if match is None or match.group("kind") != kind:
        return None
    return match.group("version")


def declares_agent_plugins_schema(data: Any) -> bool:
    """Whether a parsed root ``plugin.json`` opts into Agent Plugins semantics.

    Clients such as VS Code read a root ``plugin.json`` with Agent Plugins
    semantics only when it declares an Agent Plugins ``$schema``; other root
    ``plugin.json`` files (for example the legacy Copilot format) are not
    Agent Plugins manifests. The opt-in ignores surrounding whitespace on
    purpose, so a padded ``$schema`` is validated (and reported as invalid)
    instead of hiding the manifest.
    """
    return (
        isinstance(data, dict)
        and isinstance(data.get("$schema"), str)
        and (data["$schema"].strip().startswith(AGENT_PLUGINS_SCHEMA_PREFIX))
    )


# --------------------------------------------------------------------------- #
# Profiles                                                                    #
# --------------------------------------------------------------------------- #
# The default skills folder of every format (skills/<name>/SKILL.md). A format
# whose declared skills replace it says so in its profile.
DEFAULT_SKILLS_DIR = "skills"


@dataclass(frozen=True)
class FormatProfile:
    """Where one manifest format declares components and where they live by default.

    Every default path is root-relative POSIX. ``None`` means the format has no
    such default location, so the inventory does not look there. Skills live in
    :data:`DEFAULT_SKILLS_DIR` in every format.
    """

    manifest_type: str
    label: str
    manifest_path: str
    # Placeholders that name the plugin root in hook commands and MCP server fields.
    root_prefixes: tuple[str, ...] = ()
    # Placeholders the client also expands in manifest component paths (Cursor).
    # Claude Code and Codex reject or ignore a placeholder there.
    manifest_path_prefixes: tuple[str, ...] = ()
    # A declared field replaces its default location only when Codex accepts the
    # value: a './'-relative path (not './', no '..'), or an inline form. Codex
    # drops any other value and loads the default location instead.
    codex_path_rules: bool = False
    # Hook commands name bundled scripts with plain relative paths ("./approve.sh"),
    # so the hook risk check reads those scripts too (Cursor).
    relative_hook_scripts: bool = False
    # Report a style finding for a declared path without a leading "./".
    require_dot_relative: bool = False
    # The client rejects the whole manifest, and so loads no part of the plugin,
    # when a component path lacks "./" (Claude Code's schema requires it). The
    # style finding is then HIGH; for Codex, which drops the value and loads the
    # default location, it stays MEDIUM.
    rejects_undotted_paths: bool = False
    # Fields whose declared paths must name Markdown files (Claude Code agents).
    markdown_file_fields: frozenset[str] = frozenset()
    # Fields whose declared paths must name JSON files (Claude Code hooks,
    # lspServers, and monitors; mcpServers has its own check).
    json_file_fields: frozenset[str] = frozenset()
    # A declared component field replaces default-folder discovery for that type
    # (Cursor); otherwise it supplements the default folder (Claude Code).
    declared_replaces_default: bool = False
    # Fields whose default file stays inventoried and checked even when the
    # field is declared (Cursor mcpServers: the loader may merge the root
    # mcp.json with the declared file).
    merged_default_fields: frozenset[str] = frozenset()
    # Manifest fields that declare components (the builder reads only these).
    component_fields: frozenset[str] = frozenset()
    # Agent Plugins discovers only immediate children of skills/.
    skills_immediate_children_only: bool = False
    # Declared skill folders are searched at any depth (Codex), not one level.
    skills_recursive: bool = False
    # A root SKILL.md is a single-skill plugin when no skills are declared or bundled.
    root_skill_fallback: bool = False
    default_rules_dir: str | None = None
    rule_suffixes: tuple[str, ...] | None = None
    default_agents_dir: str | None = None
    agent_suffixes: tuple[str, ...] = (".md",)
    default_commands_dir: str | None = None
    command_suffixes: tuple[str, ...] = (".md",)
    default_hooks_file: str | None = None
    default_mcp_files: tuple[str, ...] = ()
    default_lsp_file: str | None = None
    default_output_styles_dir: str | None = None
    default_monitors_file: str | None = None
    settings_files: tuple[str, ...] = ()
    # Default location of an app (connector) declaration file.
    default_apps_file: str | None = None
    # Agent Plugins client-extension namespaces (``extensions`` + namespace dirs).
    client_extensions: bool = False
    # Tier 3 stages declared skill directories and rule files from the inventory
    # (new formats) rather than only the skills/ and rules/ folders (Claude Code).
    stage_from_inventory: bool = False
    # Reference text used in finding suggestions.
    reference: str = "plugin manifest reference"
    # MCP config dialect: how server entries are normalized before the static policy.
    mcp_dialect: Literal["claude", "codex", "agent_plugins"] = "claude"


_CLAUDE_COMPONENT_FIELDS = frozenset(
    {
        "skills",
        "rules",
        "commands",
        "agents",
        "hooks",
        "mcpServers",
        "lspServers",
        "outputStyles",
        "experimental",
        "monitors",
        "settings",
    }
)

CLAUDE_PROFILE = FormatProfile(
    manifest_type=PLUGIN_CONTAINED_MANIFEST_TYPE,
    label="Claude Code plugin",
    manifest_path=f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}",
    root_prefixes=("${CLAUDE_PLUGIN_ROOT}",),
    require_dot_relative=True,
    # Claude Code's manifest schema: every component path starts with "./",
    # agents are ".md" files, and hooks, LSP, and monitor configs are ".json"
    # files. Any other value fails the whole manifest ("Invalid input").
    rejects_undotted_paths=True,
    markdown_file_fields=frozenset({"agents"}),
    json_file_fields=frozenset({"hooks", "lspServers", "experimental.monitors"}),
    component_fields=_CLAUDE_COMPONENT_FIELDS,
    default_rules_dir="rules",
    default_agents_dir="agents",
    default_commands_dir="commands",
    default_hooks_file="hooks/hooks.json",
    default_mcp_files=(".mcp.json",),
    default_lsp_file=".lsp.json",
    default_output_styles_dir="output-styles",
    default_monitors_file="monitors/monitors.json",
    settings_files=("settings.json", ".claude/settings.json", ".claude/settings.local.json"),
    reference="Claude Code plugin manifest reference",
)

# Agent Plugins v1 core: skills in skills/<name>/SKILL.md (immediate children
# only) and MCP servers in the root mcp.json. plugin.json declares no component
# fields and cannot override these locations (spec sections 5.2, 6.1, 7.2.1).
# Everything else (hooks, agents, commands, rules, LSP servers) belongs to a
# client extension namespace: ``extensions["<reverse.domain>"]`` in the
# manifest and/or a top-level ``<reverse.domain>/`` directory (spec section 8).
AGENT_PLUGINS_V1_PROFILE = FormatProfile(
    manifest_type=PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    label="Agent Plugins v1 plugin",
    manifest_path=PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE,
    root_prefixes=("${PLUGIN_ROOT}",),
    # Not component fields: the builder reads $schema (MCP version match) and
    # extensions (client-extension namespaces).
    component_fields=frozenset({"$schema", "extensions"}),
    skills_immediate_children_only=True,
    default_mcp_files=("mcp.json",),
    client_extensions=True,
    stage_from_inventory=True,
    mcp_dialect="agent_plugins",
    reference="Agent Plugins specification (agent-plugins.org)",
)

# Cursor: every component field replaces default-folder discovery for its type
# ("If a manifest field is specified, it replaces folder discovery for that
# component"). Paths are plugin-root relative; "./" is conventional but not
# required. ${CURSOR_PLUGIN_ROOT} and ${CLAUDE_PLUGIN_ROOT} name the root in hook
# commands and MCP server fields. Cursor does not document them for manifest
# component paths, so such a path is resolved (its files are still checked) but
# gets a MEDIUM finding. The root mcp.json stays checked when mcpServers is
# declared: a reading of the cursor-agent loader suggests it merges both files
# (not confirmed with a Cursor run), so both are checked rather than one hidden.
CURSOR_PROFILE = FormatProfile(
    manifest_type=PLUGIN_CURSOR_MANIFEST_TYPE,
    label="Cursor plugin",
    manifest_path=f"{PLUGIN_CURSOR_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}",
    root_prefixes=("${CURSOR_PLUGIN_ROOT}", "${CLAUDE_PLUGIN_ROOT}"),
    manifest_path_prefixes=("${CURSOR_PLUGIN_ROOT}", "${CLAUDE_PLUGIN_ROOT}"),
    relative_hook_scripts=True,
    declared_replaces_default=True,
    merged_default_fields=frozenset({"mcpServers"}),
    component_fields=frozenset({"skills", "rules", "agents", "commands", "hooks", "mcpServers"}),
    root_skill_fallback=True,
    default_rules_dir="rules",
    rule_suffixes=(".md", ".mdc", ".markdown"),
    default_agents_dir="agents",
    agent_suffixes=(".md", ".mdc", ".markdown"),
    default_commands_dir="commands",
    command_suffixes=(".md", ".mdc", ".markdown", ".txt"),
    default_hooks_file="hooks/hooks.json",
    default_mcp_files=("mcp.json",),
    stage_from_inventory=True,
    reference="Cursor plugins reference",
)

# Codex (``.codex-plugin/plugin.json``, the "Codex compatibility" format).
# Sources: openai/plugins at 610a632 (plugins/plugin-eval validator and
# .agents/skills/plugin-creator/references/plugin-json-spec.md), the Codex
# runtime loader (openai/codex codex-rs/core-plugins), and
# developers.openai.com/plugins/build/plugins. Component paths must start with
# "./", must not be "./", and must not contain "..". Codex never expands a root
# placeholder in them. It drops any other value and falls back to the default
# location. An accepted declared field replaces its default location: skills/
# (searched recursively), .mcp.json, .app.json, hooks/hooks.json, and
# commands/. Apps (connectors) and hooks are inventoried as unsupported;
# commands are migrated into skills by the Codex installer and are inventoried
# as unsupported commands. Codex has no plugin agents.
CODEX_PROFILE = FormatProfile(
    manifest_type=PLUGIN_CODEX_MANIFEST_TYPE,
    label="Codex plugin",
    manifest_path=f"{PLUGIN_CODEX_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}",
    root_prefixes=("${PLUGIN_ROOT}", "${CLAUDE_PLUGIN_ROOT}"),
    codex_path_rules=True,
    require_dot_relative=True,
    declared_replaces_default=True,
    component_fields=frozenset({"skills", "mcpServers", "apps", "hooks", "commands"}),
    skills_recursive=True,
    default_commands_dir="commands",
    default_hooks_file="hooks/hooks.json",
    default_mcp_files=(".mcp.json",),
    default_apps_file=".app.json",
    stage_from_inventory=True,
    mcp_dialect="codex",
    reference="Codex plugin packaging reference (developers.openai.com/plugins)",
)

PROFILES: dict[str, FormatProfile] = {
    profile.manifest_type: profile
    for profile in (CLAUDE_PROFILE, AGENT_PLUGINS_V1_PROFILE, CODEX_PROFILE, CURSOR_PROFILE)
}


def profile_for(manifest_type: str | None) -> FormatProfile:
    """Return the profile of a contained manifest type (Claude Code by default)."""
    return PROFILES.get(manifest_type or "", CLAUDE_PROFILE)


def codex_accepts_path(raw: Any) -> bool:
    """Whether Codex keeps one manifest path value (``resolve_manifest_path`` in the Codex loader).

    The path must start with ``./``, must not be ``./``, must not contain a
    ``..`` segment, and must not be absolute after the ``./``. Codex does not
    strip whitespace or expand root placeholders, so neither does this check.
    """
    if not isinstance(raw, str) or not raw.startswith("./"):
        return False
    relative = raw[2:]
    if not relative or relative.startswith(("/", "\\")):
        return False
    return ".." not in relative.replace("\\", "/").split("/")


# Inline forms Codex accepts per field: hooks take an object or a non-empty list
# of objects, mcpServers an object; skills, commands, and apps take paths only.
_CODEX_INLINE_FIELDS = frozenset({"hooks", "mcpServers"})


def declared_value_replaces_default(profile: FormatProfile, field_name: str, value: Any) -> bool:
    """Whether a declared component field replaces the format's default location.

    Claude Code merges declared fields with its defaults. Cursor replaces the
    default whenever the field is set, except for the fields in
    :attr:`FormatProfile.merged_default_fields` (its root ``mcp.json`` stays
    checked next to a declared ``mcpServers``). Codex replaces it only when it
    keeps the value: a path it accepts (:func:`codex_accepts_path`), a list with
    at least one such path, or an inline form for ``hooks`` and ``mcpServers``.
    Any other value is dropped by Codex, which then loads the default location,
    so the default must still be inventoried and checked.
    """
    if not profile.declared_replaces_default or value is None or field_name in profile.merged_default_fields:
        return False
    if not profile.codex_path_rules:
        return True
    if isinstance(value, dict):
        return field_name in _CODEX_INLINE_FIELDS
    if isinstance(value, str):
        return codex_accepts_path(value)
    if isinstance(value, list) and field_name not in {"mcpServers", "apps"}:
        if field_name == "hooks" and value and all(isinstance(item, dict) for item in value):
            return True
        return all(isinstance(item, str) for item in value) and any(codex_accepts_path(item) for item in value)
    return False


def manifest_syntax(manifest_type: str) -> Literal["json", "yaml"]:
    """Every supported format is JSON except the bundle-reference YAML."""
    return "yaml" if manifest_type == PLUGIN_MANIFEST_TYPE else "json"


def parse_manifest_text(manifest_type: str, text: str) -> Any:
    """Parse manifest text with the bounded parser of its format's syntax (:func:`manifest_syntax`).

    Raises :class:`~skillevaluator.utils.structured_data.StructuredDataError`
    (a ``ValueError``) when the text does not parse or exceeds the
    structured-data bounds; the bounded parsers report parser recursion that
    way too.
    """
    if manifest_syntax(manifest_type) == "json":
        return load_bounded_json(text)
    return load_bounded_yaml(text)


# --------------------------------------------------------------------------- #
# Field validation                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManifestIssue:
    """One manifest field problem, independent of the finding model.

    ``level`` is ``error`` (the client rejects the plugin: HIGH), ``warning``
    (advisory: MEDIUM), or ``note`` (LOW).
    """

    field: str
    error: str
    message: str
    level: IssueLevel = "error"
    suggestion: str = ""


# Unknown top-level fields reported one by one (in name order); a note counts the rest.
_MAX_UNKNOWN_FIELD_ISSUES = 32
# Agent Plugins extension namespaces whose value is checked to be an object.
_MAX_CHECKED_EXTENSIONS = 64
# Characters of a field value quoted in a message. A name is quoted past the
# 64-character name limit, so a value that is too long shows that it is.
_MAX_QUOTED_NAME_CHARS = 80
_MAX_QUOTED_VERSION_CHARS = 64


def _issue(
    field_name: str, error: str, message: str, level: IssueLevel = "error", suggestion: str = ""
) -> ManifestIssue:
    return ManifestIssue(field_name, error, message, level, suggestion)


def _type_name(value: Any) -> str:
    """JSON type name of a parsed value, for messages (``null``, ``array``, ``object``, ...)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string" if isinstance(value, str) else type(value).__name__


def _check_string(
    data: dict[str, Any],
    key: str,
    issues: list[ManifestIssue],
    *,
    required: bool = False,
    non_empty: bool = False,
    label: str,
    level: IssueLevel = "error",
    consequence: str = "",
) -> str | None:
    if key not in data or data[key] is None:
        if required:
            issues.append(_issue(key, "missing", f"{label} must define '{key}'.", suggestion=f"Add a '{key}' string."))
        return None
    value = data[key]
    if not isinstance(value, str):
        issues.append(
            _issue(
                key,
                "type",
                f"{label} field '{key}' must be a string (got {_type_name(value)}){consequence}.",
                level,
            )
        )
        return None
    if non_empty and not value.strip():
        issues.append(_issue(key, "empty", f"{label} field '{key}' must not be empty."))
        return None
    return value


def _check_string_list(
    data: dict[str, Any],
    key: str,
    issues: list[ManifestIssue],
    *,
    label: str,
    level: IssueLevel = "error",
    consequence: str = "",
) -> None:
    value = data.get(key)
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        issues.append(_issue(key, "type", f"{label} field '{key}' must be an array of strings{consequence}.", level))


def _check_string_or_list(
    data: dict[str, Any],
    key: str,
    issues: list[ManifestIssue],
    *,
    label: str,
    level: IssueLevel = "error",
    consequence: str = "",
) -> None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        issues.append(
            _issue(
                key,
                "type",
                f"{label} field '{key}' must be a path string or an array of path strings{consequence}.",
                level,
            )
        )


def _closest_field(key: str, allowed: frozenset[str]) -> str | None:
    """The known field a misspelled key most likely means (``mcpServer`` -> ``mcpServers``), if any."""
    import difflib

    matches = difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=0.75)
    if matches:
        return matches[0]
    folded = {name.casefold(): name for name in allowed}
    return folded.get(key.casefold())


def _check_unknown_fields(
    data: dict[str, Any],
    allowed: frozenset[str],
    issues: list[ManifestIssue],
    *,
    label: str,
    consequence: str,
    ignored: frozenset[str] = frozenset(),
) -> None:
    """MEDIUM for each top-level field the format does not define (``ignored`` fields are skipped silently)."""
    unknown = sorted(str(key) for key in data if key not in allowed and key not in ignored)
    for key in unknown[:_MAX_UNKNOWN_FIELD_ISSUES]:
        closest = _closest_field(key, allowed)
        suggestion = (
            f"Did you mean '{closest}'? Rename '{key}' to '{closest}'."
            if closest is not None
            else f"Remove '{key}' or move it where the format allows client-specific data."
        )
        issues.append(
            _issue(
                key,
                "unknown_field",
                f"{label} does not define the top-level field '{key}'; {consequence}.",
                "warning",
                suggestion,
            )
        )
    if len(unknown) > _MAX_UNKNOWN_FIELD_ISSUES:
        omitted = len(unknown) - _MAX_UNKNOWN_FIELD_ISSUES
        issues.append(
            _issue(
                "<root>",
                "unknown_fields_truncated",
                f"{label} has {len(unknown)} top-level fields it does not define; only the first "
                f"{_MAX_UNKNOWN_FIELD_ISSUES} (by name) are reported, and {omitted} more are not listed.",
                "note",
                "Remove the fields the format does not define.",
            )
        )


def _check_author(
    data: dict[str, Any],
    issues: list[ManifestIssue],
    *,
    label: str,
    allowed_keys: frozenset[str],
    required_keys: frozenset[str] = frozenset(),
) -> None:
    author = data.get("author")
    if author is None:
        return
    if not isinstance(author, dict):
        issues.append(
            _issue("author", "type", f"{label} field 'author' must be an object (got {type(author).__name__}).")
        )
        return
    for key in sorted(required_keys):
        value = author.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(_issue(f"author.{key}", "missing", f"{label} 'author' must define a non-empty '{key}'."))
    for key, value in author.items():
        if key not in allowed_keys:
            issues.append(
                _issue(
                    f"author.{key}",
                    "unknown_field",
                    f"{label} 'author' allows only {', '.join(sorted(allowed_keys))} (found '{key}').",
                )
            )
        elif not isinstance(value, str):
            issues.append(_issue(f"author.{key}", "type", f"{label} 'author.{key}' must be a string."))


def _check_semver(data: dict[str, Any], issues: list[ManifestIssue], *, label: str, level: IssueLevel) -> None:
    version = data.get("version")
    if isinstance(version, str) and version.strip() and not _SEMVER_RE.match(version.strip()):
        issues.append(
            _issue(
                "version",
                "not_semver",
                f"{label} 'version' {version[:_MAX_QUOTED_VERSION_CHARS]!r} is not a semantic version "
                "(MAJOR.MINOR.PATCH).",
                level,
                "Use a semantic version such as 1.0.0.",
            )
        )


# --------------------------------------------------------------------------- #
# Claude Code manifest                                                         #
# --------------------------------------------------------------------------- #
# Source: the plugin manifest schema of Claude Code 2.1.284. Claude Code applies
# it when it loads a plugin and in ``claude plugin validate``. Any schema error
# makes Claude Code refuse the whole plugin ("plugin manifest failed schema
# validation"), so every schema error below is HIGH. Claude Code ignores unknown
# top-level fields at load time (MEDIUM, as for the other formats). Only the
# name has softer rules: a name Claude Code accepts but cannot install from a
# marketplace is MEDIUM, and a name that is only not kebab-case is LOW. Claude
# Code has no name length limit. Path rules for component fields (a leading
# "./", existence) belong to the component path checks, so only the value
# shapes are checked here.
_CLAUDE_FIELDS = frozenset(
    {
        "$schema",
        "name",
        "displayName",
        "version",
        "description",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
        "defaultEnabled",
        "dependencies",
        "metadata",
        "hooks",
        "commands",
        "agents",
        "skills",
        "outputStyles",
        "themes",
        "workflows",
        "channels",
        "mcpServers",
        "lspServers",
        "monitors",
        "settings",
        "types",
        "userConfig",
        "binaries",
        "experimental",
    }
)
# Store-listing fields that Claude Code's validator accepts without a warning.
_CLAUDE_IGNORED_FIELDS = frozenset(
    {
        "icon",
        "screenshots",
        "classification",
        "privacyPolicyUrl",
        "privacy_policy",
        "privacyPolicy",
        "supportUrl",
        "support",
        "bugs",
        "termsOfServiceUrl",
        "terms_of_service",
        "documentationUrl",
        "docs",
    }
)
# Marketplace-entry fields: harmless in plugin.json, but Claude Code does not read them there.
_CLAUDE_MARKETPLACE_ENTRY_FIELDS = frozenset(
    {"category", "source", "tags", "strict", "id", "relevance", "headers", "headersHelper"}
)
_CLAUDE_EXPERIMENTAL_FIELDS = frozenset({"themes", "monitors", "outputStyles", "evals", "syntaxHighlighting"})
# Claude Code refuses a name with a space or a control or bidirectional-formatting character.
_CLAUDE_NAME_CONTROL_RE = re.compile("[\x00-\x1f\x7f-\x9f\u200e\u200f\u202a-\u202e\u2066-\u2069]")
# A plugin id part (``name@marketplace``); a name outside it cannot be installed from a marketplace.
_CLAUDE_ID_PART_RE = re.compile(r"^[A-Za-z0-9][-A-Za-z0-9._]*$")
_CLAUDE_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# ``dependencies`` entries: ``name``, ``name@marketplace``, optionally with an ``@^range`` suffix.
CLAUDE_DEPENDENCY_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][-A-Za-z0-9._]*)(?:@(?P<marketplace>[A-Za-z0-9][-A-Za-z0-9._]*))?(?:@\^[^@]*)?$"
)
_CLAUDE_REFUSED = "Claude Code refuses the whole plugin"
_CLAUDE_LSP_KEYS = frozenset(
    {
        "command",
        "args",
        "extensionToLanguage",
        "transport",
        "env",
        "initializationOptions",
        "settings",
        "workspaceFolder",
        "startupTimeout",
        "shutdownTimeout",
        "restartOnCrash",
        "maxRestarts",
        "diagnostics",
    }
)
_CLAUDE_MONITOR_KEYS = frozenset({"name", "command", "description", "when"})
_CLAUDE_USER_CONFIG_KEYS = frozenset(
    {"type", "title", "description", "required", "default", "multiple", "sensitive", "min", "max", "options"}
)
_CLAUDE_USER_CONFIG_TYPES = ("string", "number", "boolean", "directory", "file")
_CLAUDE_USER_CONFIG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CLAUDE_HLJS_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
# Entries checked per map or list; the rest are still loaded by the client, so the cap only bounds this check.
_CLAUDE_MAX_ENTRIES = 64


def _shown(value: Any, limit: int = 64) -> str:
    """A bounded, quoted rendering of a manifest value for messages."""
    return repr(value[:limit] if isinstance(value, str) else value)[: limit + 16]


def _is_int(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and float(value).is_integer()


def _is_str_list(value: Any, *, non_empty: bool = False) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and (item or not non_empty) for item in value)


def _claude_name_refused_by_path(name: str) -> bool:
    """Whether Claude Code refuses this name when it loads the plugin by path (``--plugin-dir``).

    The rule matches '@', ':', '/', '\\', any whitespace, and control, format,
    surrogate, private-use, and unassigned characters.
    """
    import unicodedata

    return any(
        char in "@:/\\\u2028\u2029" or char.isspace() or unicodedata.category(char) in {"Cc", "Cf", "Cs", "Co", "Cn"}
        for char in name
    )


def _check_claude_name(data: dict[str, Any], issues: list[ManifestIssue], *, label: str) -> None:
    suggestion = "Use a kebab-case name such as 'my-plugin'."
    if "name" not in data:
        issues.append(_issue("name", "missing", f"{label} must define 'name'; {_CLAUDE_REFUSED}.", "error", suggestion))
        return
    name = data["name"]
    if not isinstance(name, str):
        issues.append(
            _issue(
                "name",
                "type",
                f"{label} field 'name' must be a string (got {_type_name(name)}); {_CLAUDE_REFUSED}.",
                "error",
                suggestion,
            )
        )
        return
    if not name:
        issues.append(_issue("name", "empty", f"{label} 'name' is empty; {_CLAUDE_REFUSED}.", "error", suggestion))
        return
    if " " in name:
        issues.append(
            _issue(
                "name",
                "spaces",
                f"{label} 'name' {_shown(name)} contains a space (including a leading or trailing one); "
                f"{_CLAUDE_REFUSED}.",
                "error",
                suggestion,
            )
        )
        return
    if _CLAUDE_NAME_CONTROL_RE.search(name):
        issues.append(
            _issue(
                "name",
                "control_characters",
                f"{label} 'name' {_shown(name)} contains a control or bidirectional-formatting character; "
                f"{_CLAUDE_REFUSED}.",
                "error",
                suggestion,
            )
        )
        return
    if _claude_name_refused_by_path(name):
        issues.append(
            _issue(
                "name",
                "path_unsafe",
                f"{label} 'name' {_shown(name)} contains '@', ':', '/', '\\', whitespace, or an invisible "
                "character. Claude Code refuses a plugin with this name when it loads it by path (such as "
                "--plugin-dir), and cannot install it from a marketplace.",
                "error",
                suggestion,
            )
        )
        return
    if not _CLAUDE_ID_PART_RE.match(name):
        issues.append(
            _issue(
                "name",
                "not_installable",
                f"{label} 'name' {_shown(name)} must start with a letter or digit and use only letters, digits, "
                "'-', '.', and '_' for Claude Code to install it from a marketplace; it still loads by path.",
                "warning",
                suggestion,
            )
        )
        return
    if not _CLAUDE_KEBAB_RE.match(name):
        issues.append(
            _issue(
                "name",
                "not_kebab_case",
                f"{label} 'name' {_shown(name)} is not kebab-case. Claude Code accepts it, but the Claude.ai "
                "marketplace sync requires lowercase letters, digits, and hyphens.",
                "note",
                suggestion,
            )
        )


def _claude_type_issue(key: str, expected: str, value: Any, *, label: str) -> ManifestIssue:
    return _issue(
        key,
        "type",
        f"{label} field '{key}' must be {expected} (got {_type_name(value)}); {_CLAUDE_REFUSED}.",
        "error",
        f"Fix '{key}' to match the Claude Code plugin manifest reference.",
    )


_WHATWG_SPECIAL_SCHEMES = frozenset({"http", "https", "ws", "wss", "ftp"})
# WHATWG forbidden host code points; a domain also forbids C0 controls, '%' and DEL.
_WHATWG_FORBIDDEN_HOST = frozenset("\x00\t\n\r #/:<>?@[\\]^|")
_WHATWG_FORBIDDEN_DOMAIN = _WHATWG_FORBIDDEN_HOST | frozenset("%\x7f") | frozenset(chr(code) for code in range(0x20))
_PERCENT_ESCAPE_RE = re.compile(rb"%([0-9A-Fa-f]{2})")


def _whatwg_ipv4_number(part: str) -> int | None:
    """One dotted part of a WHATWG IPv4 host (decimal, ``0x`` hex, or leading-zero octal), else ``None``."""
    text, base = part, 10
    if text[:2] in {"0x", "0X"}:
        text, base = text[2:], 16
    elif len(text) > 1 and text.startswith("0"):
        text, base = text[1:], 8
    if not text:
        return 0 if base != 10 else None
    try:
        return int(text, base) if text.isascii() and text.isalnum() else None
    except ValueError:
        return None


def _whatwg_ipv4_ok(host: str) -> bool | None:
    """``None`` when ``host`` does not end in a number (a domain); else whether it is a valid WHATWG IPv4 host."""
    parts = host.split(".")
    if parts[-1] == "" and len(parts) > 1:
        parts.pop()
    last = parts[-1]
    if not (last.isascii() and last.isdigit()) and not re.fullmatch(r"0[xX][0-9A-Fa-f]*", last):
        return None
    if len(parts) > 4:
        return False
    numbers: list[int] = []
    for part in parts:
        number = _whatwg_ipv4_number(part)
        if number is None:
            return False
        numbers.append(number)
    *leading, final = numbers
    return all(number <= 255 for number in leading) and final < 256 ** (5 - len(numbers))


def _whatwg_domain_ok(raw: str) -> bool:
    """Whether the WHATWG host parser accepts ``raw`` as a special-scheme domain or IPv4 host."""
    import unicodedata

    data = _PERCENT_ESCAPE_RE.sub(lambda match: bytes([int(match.group(1), 16)]), raw.encode("utf-8"))
    try:
        domain = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not domain.isascii():
        # UTS #46 maps much like NFKC case folding: a full-width '<' or an ideographic space still counts.
        domain = unicodedata.normalize("NFKC", domain).casefold()
        if any(unicodedata.category(char) == "Cc" for char in domain):
            return False
    if not domain or any(char in _WHATWG_FORBIDDEN_DOMAIN for char in domain):
        return False
    for label in domain.split("."):
        if label[:4].lower() == "xn--":
            try:
                decoded = label[4:].encode("ascii").decode("punycode")
            except (UnicodeError, ValueError):
                return False
            if any(unicodedata.category(char) == "Cc" or char in _WHATWG_FORBIDDEN_DOMAIN for char in decoded):
                return False
    ipv4 = _whatwg_ipv4_ok(domain)
    return ipv4 is not False


def _whatwg_ipv6_ok(text: str) -> bool:
    import ipaddress

    if "%" in text:  # WHATWG has no zone identifiers; Python's parser would accept one.
        return False
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        return False
    return True


def _claude_url_ok(value: str) -> bool:
    """Whether a WHATWG URL parser (Claude Code's zod ``url()``) accepts ``value`` as an absolute URL.

    It checks what makes ``new URL()`` fail: a missing scheme; for special
    schemes an empty host, a forbidden host code point (space, '<', '>', '^',
    '|', ...) also after percent-decoding, a percent escape that does not
    decode to UTF-8, an invalid IPv4 or bracketed IPv6 host; and for any
    authority a port that is not a number up to 65535. Unicode domains are
    accepted without the full UTS #46 check.
    """
    # zod trims (JavaScript whitespace, BOM included), then the URL parser drops C0 controls and spaces.
    text = value.strip().strip("\ufeff").strip("".join(chr(code) for code in range(0x21)))
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):(.*)$", re.sub(r"[\t\n\r]", "", text), re.DOTALL)
    if match is None:
        return False
    scheme, rest = match.group(1).lower(), match.group(2)
    special = scheme in _WHATWG_SPECIAL_SCHEMES
    if scheme == "file":
        if rest[:2].replace("\\", "/") != "//":
            return True  # no host
        host = re.split(r"[/?#\\]", rest[2:], maxsplit=1)[0]
        if not host or re.fullmatch(r"[A-Za-z][:|]", host):
            return True  # no host, or a Windows drive letter
        return _whatwg_ipv6_ok(host[1:-1]) if host.startswith("[") and host.endswith("]") else _whatwg_domain_ok(host)
    if special:
        rest = rest.lstrip("/\\")
    elif not rest.startswith("//"):
        return True  # an opaque path, as in mailto:
    else:
        rest = rest[2:]
    authority = re.split(r"[/?#\\]" if special else r"[/?#]", rest, maxsplit=1)[0]
    host_port = authority.rpartition("@")[2]
    if host_port.startswith("["):
        end = host_port.find("]")
        if end == -1 or not _whatwg_ipv6_ok(host_port[1:end]):
            return False
        after = host_port[end + 1 :]
        if after and not after.startswith(":"):
            return False
        port = after[1:] if after else ""
    else:
        host, _, port = host_port.partition(":")
        if special and not _whatwg_domain_ok(host):
            return False
        if not special and any(char in _WHATWG_FORBIDDEN_HOST for char in host):
            return False
    return not port or (port.isascii() and port.isdigit() and int(port) <= 65535)


def _claude_dependency_problem(entry: Any) -> str | None:
    if isinstance(entry, str):
        if CLAUDE_DEPENDENCY_RE.match(entry):
            return None
        return f"{_shown(entry)} must be a plugin name, optionally qualified with @marketplace"
    if isinstance(entry, dict):
        name, marketplace = entry.get("name"), entry.get("marketplace")
        if not isinstance(name, str) or not _CLAUDE_ID_PART_RE.match(name):
            return "an object entry needs a 'name' that starts with a letter or digit"
        if "marketplace" in entry and (not isinstance(marketplace, str) or not _CLAUDE_ID_PART_RE.match(marketplace)):
            return f"entry {_shown(name)} has an invalid 'marketplace'"
        return None
    return f"entries must be strings or {{name, marketplace}} objects (got {_type_name(entry)})"


def _claude_lsp_problems(config: Any) -> list[str]:
    """Why Claude Code rejects one LSP server config (a strict object), as short phrases."""
    if not isinstance(config, dict):
        return [f"must be an object (got {_type_name(config)})"]
    problems: list[str] = []
    command = config.get("command")
    if "command" not in config:
        problems.append("has no 'command'")
    elif not isinstance(command, str) or not command:
        problems.append("'command' must be a non-empty string")
    elif " " in command and not command.startswith("/"):
        problems.append("'command' contains a space; put the arguments in 'args'")
    if "args" in config and not _is_str_list(config["args"], non_empty=True):
        problems.append("'args' must be an array of non-empty strings")
    mapping = config.get("extensionToLanguage")
    if "extensionToLanguage" not in config:
        problems.append("has no 'extensionToLanguage' map")
    elif not isinstance(mapping, dict) or not mapping:
        problems.append("'extensionToLanguage' must map at least one file extension to a language ID")
    else:
        bad = [
            key
            for key, value in list(mapping.items())[:_CLAUDE_MAX_ENTRIES]
            if not (isinstance(key, str) and len(key) >= 2 and key.startswith("."))
            or not (isinstance(value, str) and value)
        ]
        if bad:
            problems.append(
                "'extensionToLanguage' keys must be extensions that start with '.' and values non-empty language "
                f"IDs (bad: {', '.join(_shown(key, 16) for key in bad[:3])})"
            )
    if "transport" in config and config["transport"] not in ("stdio", "socket"):
        problems.append("'transport' must be 'stdio' or 'socket'")
    env = config.get("env")
    if "env" in config and not (
        isinstance(env, dict) and all(isinstance(value, str) for value in list(env.values())[:_CLAUDE_MAX_ENTRIES])
    ):
        problems.append("'env' must map names to string values")
    if "workspaceFolder" in config and not isinstance(config["workspaceFolder"], str):
        problems.append("'workspaceFolder' must be a string")
    for key in ("startupTimeout", "shutdownTimeout"):
        if key in config and not (_is_int(config[key]) and config[key] > 0):
            problems.append(f"'{key}' must be a positive integer (milliseconds)")
    if "maxRestarts" in config and not (_is_int(config["maxRestarts"]) and config["maxRestarts"] >= 0):
        problems.append("'maxRestarts' must be a non-negative integer")
    for key in ("restartOnCrash", "diagnostics"):
        if key in config and not isinstance(config[key], bool):
            problems.append(f"'{key}' must be true or false")
    unknown = sorted(str(key) for key in config if key not in _CLAUDE_LSP_KEYS)
    if unknown:
        problems.append(f"has unknown key(s) {', '.join(_shown(key, 32) for key in unknown[:5])}")
    return problems


def _check_claude_lsp_servers(value: Any, issues: list[ManifestIssue], *, label: str) -> None:
    """``lspServers``: a path, a server map, or an array of those; every inline server config is strict."""

    def check_map(servers: dict[str, Any], where: str) -> None:
        for name, config in list(servers.items())[:_CLAUDE_MAX_ENTRIES]:
            for problem in _claude_lsp_problems(config):
                issues.append(
                    _issue(
                        "lspServers",
                        "invalid",
                        f"{label} {where} server {_shown(name)} {problem}; {_CLAUDE_REFUSED}.",
                        "error",
                        "Give each LSP server a 'command' without spaces, an 'extensionToLanguage' map, and only "
                        "the keys of the Claude Code LSP server reference.",
                    )
                )

    if isinstance(value, str):
        return
    if isinstance(value, dict):
        check_map(value, "lspServers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value[:_CLAUDE_MAX_ENTRIES]):
            if isinstance(item, dict):
                check_map(item, f"lspServers[{index}]")
            elif not isinstance(item, str):
                issues.append(
                    _issue(
                        "lspServers",
                        "type",
                        f"{label} 'lspServers[{index}]' must be a path or a server map (got {_type_name(item)}); "
                        f"{_CLAUDE_REFUSED}.",
                    )
                )
        return
    issues.append(_claude_type_issue("lspServers", "a path, a server map, or an array of those", value, label=label))


def _claude_monitor_problems(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return [f"must be an object (got {_type_name(item)})"]
    problems = [
        f"needs a non-empty '{key}' string"
        for key in ("name", "command", "description")
        if not (isinstance(item.get(key), str) and item[key])
    ]
    when = item.get("when", "always")
    if not (when == "always" or (isinstance(when, str) and when.startswith("on-skill-invoke:") and len(when) > 16)):
        problems.append("'when' must be 'always' or 'on-skill-invoke:<skill>'")
    unknown = sorted(str(key) for key in item if key not in _CLAUDE_MONITOR_KEYS)
    if unknown:
        problems.append(f"has unknown key(s) {', '.join(_shown(key, 32) for key in unknown[:5])}")
    return problems


def _check_claude_monitors(value: Any, field_name: str, issues: list[ManifestIssue], *, label: str) -> None:
    """``monitors``: a path to a monitors file, or an array of monitor objects with unique names."""
    if isinstance(value, str):
        return
    if not isinstance(value, list):
        issues.append(_claude_type_issue(field_name, "a path or an array of monitors", value, label=label))
        return
    names: list[str] = []
    for index, item in enumerate(value[:_CLAUDE_MAX_ENTRIES]):
        for problem in _claude_monitor_problems(item):
            issues.append(
                _issue(
                    field_name,
                    "invalid",
                    f"{label} '{field_name}[{index}]' {problem}; {_CLAUDE_REFUSED}.",
                    "error",
                    "Give each monitor a name, command, and description, and only the keys of the Claude Code "
                    "monitor reference.",
                )
            )
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    if len(set(names)) != len(names):
        issues.append(
            _issue(field_name, "invalid", f"{label} '{field_name}' repeats a monitor name; {_CLAUDE_REFUSED}.")
        )


def _claude_user_config_problems(option: Any) -> list[str]:
    if not isinstance(option, dict):
        return [f"must be an object (got {_type_name(option)})"]
    problems: list[str] = []
    kind = option.get("type")
    if kind not in _CLAUDE_USER_CONFIG_TYPES:
        problems.append(f"'type' must be one of {', '.join(_CLAUDE_USER_CONFIG_TYPES)}")
    for key in ("title", "description"):
        if not isinstance(option.get(key), str):
            problems.append(f"needs a '{key}' string")
    for key in ("required", "multiple", "sensitive"):
        if key in option and not isinstance(option[key], bool):
            problems.append(f"'{key}' must be true or false")
    for key in ("min", "max"):
        if key in option and (not isinstance(option[key], int | float) or isinstance(option[key], bool)):
            problems.append(f"'{key}' must be a number")
    default = option.get("default")
    if "default" in option and not (isinstance(default, str | int | float | bool) or _is_str_list(default)):
        problems.append("'default' must be a string, number, boolean, or array of strings")
    if "options" in option:
        choices = option["options"]
        if not _is_str_list(choices) or not choices:
            problems.append("'options' must be a non-empty array of strings")
        elif any(not choice or len(choice) > 64 or choice.strip() != choice for choice in choices) or len(
            {choice.lower() for choice in choices}
        ) != len(choices):
            problems.append("'options' entries must be 1-64 characters, unpadded, and distinct in any letter case")
        elif kind != "string" or option.get("multiple") is True or option.get("sensitive") is True:
            problems.append("'options' is only for a 'string' field that is neither multiple nor sensitive")
        elif "default" not in option and option.get("required") is not True:
            problems.append("a field with 'options' needs a default among them, or 'required': true")
        elif "default" in option and default not in choices:
            problems.append("'default' must be one of the 'options'")
    unknown = sorted(str(key) for key in option if key not in _CLAUDE_USER_CONFIG_KEYS)
    if unknown:
        problems.append(f"has unknown key(s) {', '.join(_shown(key, 32) for key in unknown[:5])}")
    return problems


def _check_claude_user_config(value: Any, where: str, issues: list[ManifestIssue], *, label: str) -> None:
    field_name = where.split("[", 1)[0].split(".", 1)[0]
    if not isinstance(value, dict):
        issues.append(_claude_type_issue(field_name, "an object of option definitions", value, label=label))
        return
    for key, option in list(value.items())[:_CLAUDE_MAX_ENTRIES]:
        problems = [] if _CLAUDE_USER_CONFIG_KEY_RE.match(str(key)) else ["has a key that is not an identifier"]
        for problem in [*problems, *_claude_user_config_problems(option)]:
            issues.append(
                _issue(
                    field_name,
                    "invalid",
                    f"{label} {where} option {_shown(key, 32)} {problem}; {_CLAUDE_REFUSED}.",
                    "error",
                    "Fix the option to match the Claude Code user configuration reference.",
                )
            )


def _check_claude_channels(value: Any, issues: list[ManifestIssue], *, label: str) -> None:
    if not isinstance(value, list):
        issues.append(_claude_type_issue("channels", "an array of channel objects", value, label=label))
        return
    for index, channel in enumerate(value[:_CLAUDE_MAX_ENTRIES]):
        where = f"'channels[{index}]'"
        if not isinstance(channel, dict):
            problems = [f"must be an object (got {_type_name(channel)})"]
        else:
            problems = [] if isinstance(channel.get("server"), str) and channel["server"] else ["needs a 'server'"]
            if "displayName" in channel and not isinstance(channel["displayName"], str):
                problems.append("'displayName' must be a string")
            unknown = sorted(str(key) for key in channel if key not in {"server", "displayName", "userConfig"})
            if unknown:
                problems.append(f"has unknown key(s) {', '.join(_shown(key, 32) for key in unknown[:5])}")
            if "userConfig" in channel:
                _check_claude_user_config(channel["userConfig"], f"channels[{index}].userConfig", issues, label=label)
        for problem in problems:
            issues.append(_issue("channels", "invalid", f"{label} {where} {problem}; {_CLAUDE_REFUSED}."))


def _check_claude_commands(value: Any, issues: list[ManifestIssue], *, label: str) -> None:
    """``commands``: a path, an array of paths, or a map of command names to command objects."""
    if isinstance(value, str) or _is_str_list(value):
        return
    if not isinstance(value, dict):
        issues.append(_claude_type_issue("commands", "a path, an array of paths, or a command map", value, label=label))
        return
    for name, command in list(value.items())[:_CLAUDE_MAX_ENTRIES]:
        if not isinstance(command, dict):
            problems = [f"must be an object (got {_type_name(command)})"]
        else:
            has_source, has_content = bool(command.get("source")), bool(command.get("content"))
            problems = [] if has_source != has_content else ["needs exactly one of 'source' or 'content'"]
            problems += [
                f"'{key}' must be a string"
                for key in ("source", "content", "description", "argumentHint", "model")
                if key in command and not isinstance(command[key], str)
            ]
            if "allowedTools" in command and not _is_str_list(command["allowedTools"]):
                problems.append("'allowedTools' must be an array of strings")
        for problem in problems:
            issues.append(
                _issue("commands", "invalid", f"{label} command {_shown(name, 32)} {problem}; {_CLAUDE_REFUSED}.")
            )


def _check_claude_experimental(value: Any, issues: list[ManifestIssue], *, label: str) -> None:
    if not isinstance(value, dict):
        issues.append(
            _issue(
                "experimental",
                "type",
                f"{label} field 'experimental' must be an object (got {_type_name(value)}); Claude Code ignores it, "
                "so the components it declares do not load.",
                "warning",
            )
        )
        return
    for key in ("themes", "outputStyles", "evals"):
        # Named 'experimental.<key>', as Claude Code reports it; zod's optional() refuses null here too.
        if key in value and not (isinstance(value[key], str) or _is_str_list(value[key])):
            issues.append(
                _issue(
                    f"experimental.{key}",
                    "type",
                    f"{label} 'experimental.{key}' must be a path string or an array of path strings (got "
                    f"{_type_name(value[key])}); {_CLAUDE_REFUSED}.",
                )
            )
    if "monitors" in value:
        _check_claude_monitors(value["monitors"], "experimental.monitors", issues, label=label)
    highlighting = value.get("syntaxHighlighting")
    if "syntaxHighlighting" in value:
        languages = highlighting.get("hljsLanguages") if isinstance(highlighting, dict) else None
        if (
            not isinstance(highlighting, dict)
            or set(highlighting) != {"hljsLanguages"}
            or not isinstance(languages, list)
            or len(languages) > 16
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and len(item["id"]) <= 64
                and _CLAUDE_HLJS_ID_RE.match(item["id"])
                and set(item) <= {"id", "remote", "integrity"}
                for item in languages
            )
        ):
            issues.append(
                _issue(
                    "experimental.syntaxHighlighting",
                    "invalid",
                    f"{label} 'experimental.syntaxHighlighting' must be {{hljsLanguages: [{{id, remote?, "
                    f"integrity?}}]}} with at most 16 languages; {_CLAUDE_REFUSED}.",
                )
            )
    for key in sorted(str(key) for key in value if key not in _CLAUDE_EXPERIMENTAL_FIELDS)[:32]:
        issues.append(
            _issue(
                "experimental",
                "unknown_key",
                f"{label} 'experimental' does not define {_shown(key, 32)}; Claude Code ignores it.",
                "warning",
            )
        )


def _validate_claude(data: dict[str, Any]) -> list[ManifestIssue]:
    """Field problems of a Claude Code manifest, as Claude Code 2.1.284 decides them (see the comment above)."""
    label = "Claude Code plugin manifest"
    issues: list[ManifestIssue] = []
    _check_claude_name(data, issues, label=label)
    # zod's optional() accepts a missing key but not null, so null is a wrong type here.
    for key in ("$schema", "displayName", "version", "description", "repository", "license"):
        if key in data and not isinstance(data[key], str):
            issues.append(_claude_type_issue(key, "a string", data[key], label=label))
    if "homepage" in data:
        homepage = data["homepage"]
        if not isinstance(homepage, str):
            issues.append(_claude_type_issue("homepage", "a URL string", homepage, label=label))
        elif not _claude_url_ok(homepage):
            issues.append(
                _issue(
                    "homepage",
                    "invalid_url",
                    f"{label} 'homepage' {_shown(homepage)} is not a valid absolute URL; {_CLAUDE_REFUSED}.",
                    "error",
                    "Use a full URL such as https://example.com/docs.",
                )
            )
    if "author" in data:
        author = data["author"]
        if not isinstance(author, dict):
            issues.append(_claude_type_issue("author", "an object with a 'name'", author, label=label))
        else:
            if not isinstance(author.get("name"), str) or not author["name"]:
                issues.append(
                    _issue(
                        "author.name",
                        "missing",
                        f"{label} 'author' needs a non-empty 'name' string; {_CLAUDE_REFUSED}.",
                    )
                )
            for key in ("email", "url"):
                if key in author and not isinstance(author[key], str):
                    issues.append(_issue(f"author.{key}", "type", f"{label} 'author.{key}' must be a string."))
    if "keywords" in data and not _is_str_list(data["keywords"]):
        issues.append(_claude_type_issue("keywords", "an array of strings", data["keywords"], label=label))
    if "defaultEnabled" in data and not isinstance(data["defaultEnabled"], bool):
        issues.append(_claude_type_issue("defaultEnabled", "true or false", data["defaultEnabled"], label=label))
    if "dependencies" in data:
        dependencies = data["dependencies"]
        if not isinstance(dependencies, list):
            issues.append(_claude_type_issue("dependencies", "an array of plugin names", dependencies, label=label))
        else:
            for index, entry in enumerate(dependencies[:_CLAUDE_MAX_ENTRIES]):
                problem = _claude_dependency_problem(entry)
                if problem is not None:
                    issues.append(
                        _issue(
                            "dependencies",
                            "invalid",
                            f"{label} 'dependencies[{index}]': {problem}; {_CLAUDE_REFUSED}.",
                            "error",
                            "List each dependency as 'plugin-name' or 'plugin-name@marketplace'.",
                        )
                    )
    if "metadata" in data and not isinstance(data["metadata"], dict):
        issues.append(
            _issue(
                "metadata",
                "type",
                f"{label} field 'metadata' must be an object (got {_type_name(data['metadata'])}); Claude Code "
                "ignores it.",
                "warning",
            )
        )
    for key in ("agents", "skills", "outputStyles", "themes", "workflows"):
        if key in data:
            _check_string_or_list(data, key, issues, label=label, consequence=f"; {_CLAUDE_REFUSED}")
            if data[key] is None:
                issues.append(_claude_type_issue(key, "a path or an array of paths", None, label=label))
    for key in ("hooks", "mcpServers"):
        value = data.get(key, "")
        if not (
            isinstance(value, str | dict)
            or (isinstance(value, list) and all(isinstance(item, str | dict) for item in value))
        ):
            issues.append(_claude_type_issue(key, "a path, an object, or an array of those", value, label=label))
    if "commands" in data:
        _check_claude_commands(data["commands"], issues, label=label)
    if "lspServers" in data:
        _check_claude_lsp_servers(data["lspServers"], issues, label=label)
    if "monitors" in data:
        _check_claude_monitors(data["monitors"], "monitors", issues, label=label)
    if "settings" in data and not isinstance(data["settings"], dict):
        issues.append(_claude_type_issue("settings", "an object", data["settings"], label=label))
    if "types" in data:
        types = data["types"]
        if not (
            isinstance(types, str)
            and types.startswith("./")
            and types.endswith(".d.ts")
            and ".." not in re.split(r"[\\/]", types)
        ):
            issues.append(
                _claude_type_issue("types", "a './'-relative '.d.ts' path inside the plugin", types, label=label)
            )
    if "userConfig" in data:
        _check_claude_user_config(data["userConfig"], "userConfig", issues, label=label)
    if "channels" in data:
        _check_claude_channels(data["channels"], issues, label=label)
    if "experimental" in data:
        _check_claude_experimental(data["experimental"], issues, label=label)
    _check_unknown_fields(
        data,
        _CLAUDE_FIELDS,
        issues,
        label=label,
        consequence="Claude Code ignores it at load time, so whatever it declares does not load",
        ignored=_CLAUDE_IGNORED_FIELDS | _CLAUDE_MARKETPLACE_ENTRY_FIELDS,
    )
    for key in sorted(str(key) for key in data if key in _CLAUDE_MARKETPLACE_ENTRY_FIELDS):
        issues.append(
            _issue(
                key,
                "marketplace_field",
                f"{label} field '{key}' belongs in the marketplace entry (marketplace.json); Claude Code does not "
                "read it from plugin.json.",
                "note",
                f"Move '{key}' to the plugin's marketplace.json entry.",
            )
        )
    return issues


# Agent Plugins 1.0.0 section 5.2: "the only permitted top-level fields are
# $schema, name, version, description, author, homepage, repository, license,
# keywords, and extensions".
_AGENT_PLUGINS_FIELDS = frozenset(
    {
        "$schema",
        "name",
        "version",
        "description",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
        "extensions",
    }
)


def _validate_agent_plugins(data: dict[str, Any]) -> list[ManifestIssue]:
    label = "Agent Plugins manifest"
    issues: list[ManifestIssue] = []
    schema = data.get("$schema")
    version = agent_plugins_schema_version(schema)
    if schema is None:
        issues.append(
            _issue(
                "$schema",
                "missing",
                f"{label} must declare '$schema' (the Agent Plugins version it targets).",
                suggestion='Add "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json".',
            )
        )
    elif version is None and isinstance(schema, str) and agent_plugins_schema_version(schema.strip()) is not None:
        issues.append(
            _issue(
                "$schema",
                "invalid",
                f"{label} '$schema' has leading or trailing whitespace. Codex and Hermes compare it exactly, so "
                "they do not load the plugin.",
                suggestion="Remove the whitespace around the '$schema' value.",
            )
        )
    elif version is None:
        issues.append(
            _issue(
                "$schema",
                "invalid",
                f"{label} '$schema' must be an Agent Plugins plugin schema identifier "
                "(https://agent-plugins.org/schemas/<version>/plugin.schema.json).",
                suggestion='Use "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json".',
            )
        )
    elif version.split(".", 1)[0] != "1":
        issues.append(
            _issue(
                "$schema",
                "unsupported_version",
                f"{label} targets Agent Plugins {version}; SkillEvaluator implements version 1 only, and clients "
                "reject versions they do not support.",
                suggestion="Target Agent Plugins 1.0.0.",
            )
        )
    elif version not in AGENT_PLUGINS_SUPPORTED_VERSIONS:
        issues.append(
            _issue(
                "$schema",
                "unrecognized_version",
                f"{label} targets Agent Plugins {version}; SkillEvaluator validates it with the 1.0.0 rules. Codex "
                "and Hermes accept only 1.0.0 and do not load the plugin.",
                "warning",
                "Target the published Agent Plugins version (1.0.0).",
            )
        )
    name = _check_string(data, "name", issues, required=True, label=label)
    if name is not None and (not name or len(name) > NAME_MAX_LENGTH or not _AGENT_PLUGINS_NAME_RE.match(name)):
        issues.append(
            _issue(
                "name",
                "pattern",
                f"{label} 'name' {name[:_MAX_QUOTED_NAME_CHARS]!r} must be 1-{NAME_MAX_LENGTH} characters of a-z, "
                "0-9, '-' and '.', start and end with a letter or digit, and contain no '--' or '..'.",
                suggestion="Rename the plugin, for example 'my-plugin'.",
            )
        )
    for key in ("version", "description", "homepage", "repository", "license"):
        _check_string(data, key, issues, label=label)
    _check_string_list(data, "keywords", issues, label=label)
    _check_author(data, issues, label=label, allowed_keys=frozenset({"name", "email", "url"}))
    extensions = data.get("extensions")
    if extensions is not None:
        if not isinstance(extensions, dict):
            issues.append(
                _issue(
                    "extensions",
                    "type",
                    f"{label} 'extensions' must be an object keyed by reverse-domain namespace; clients ignore it.",
                    "warning",
                )
            )
        else:
            for namespace, value in list(extensions.items())[:_MAX_CHECKED_EXTENSIONS]:
                if not isinstance(value, dict):
                    issues.append(
                        _issue(
                            f"extensions.{namespace}",
                            "type",
                            f"{label} extension '{namespace}' must be an object.",
                        )
                    )
    # Section 5.2: clients report and ignore unknown top-level fields.
    _check_unknown_fields(
        data,
        _AGENT_PLUGINS_FIELDS,
        issues,
        label=label,
        consequence="clients report and ignore it, and component fields (skills, mcpServers, hooks, ...) are not "
        "allowed in plugin.json",
    )
    return issues


_CURSOR_FIELDS = frozenset(
    {
        "name",
        "displayName",
        "description",
        "version",
        "minClientVersions",
        "author",
        "publisher",
        "homepage",
        "repository",
        "license",
        "logo",
        "keywords",
        "category",
        "tags",
        "commands",
        "agents",
        "skills",
        "rules",
        "hooks",
        "variables",
        "mcpServers",
    }
)


def _validate_cursor(data: dict[str, Any]) -> list[ManifestIssue]:
    label = "Cursor plugin manifest"
    issues: list[ManifestIssue] = []
    name = _check_string(data, "name", issues, required=True, label=label)
    if name is not None and not _CURSOR_NAME_RE.match(name):
        issues.append(
            _issue(
                "name",
                "pattern",
                f"{label} 'name' {name[:_MAX_QUOTED_NAME_CHARS]!r} must be lowercase kebab-case: a-z, 0-9, '-' "
                "and '.', starting and ending with a letter or digit.",
                suggestion="Rename the plugin, for example 'my-plugin'.",
            )
        )
    for key in (
        "displayName",
        "description",
        "version",
        "publisher",
        "homepage",
        "repository",
        "license",
        "logo",
        "category",
    ):
        _check_string(data, key, issues, label=label)
    for key in ("keywords", "tags"):
        _check_string_list(data, key, issues, label=label)
    for key in ("commands", "agents", "skills", "rules"):
        _check_string_or_list(data, key, issues, label=label)
    hooks = data.get("hooks")
    if hooks is not None and not isinstance(hooks, str | dict):
        issues.append(_issue("hooks", "type", f"{label} 'hooks' must be a path string or an inline hooks object."))
    variables = data.get("variables")
    if variables is not None and (not isinstance(variables, dict) or variables.get("type") != "object"):
        issues.append(
            _issue("variables", "type", f'{label} \'variables\' must be a JSON Schema object with "type": "object".')
        )
    min_versions = data.get("minClientVersions")
    if min_versions is not None and (not isinstance(min_versions, dict) or not min_versions):
        issues.append(_issue("minClientVersions", "type", f"{label} 'minClientVersions' must be a non-empty object."))
    _check_author(
        data,
        issues,
        label=label,
        allowed_keys=frozenset({"name", "email"}),
        required_keys=frozenset({"name"}),
    )
    # The docs describe version as semantic; the published schema does not enforce it.
    _check_semver(data, issues, label=label, level="note")
    # The published schema sets additionalProperties: false.
    _check_unknown_fields(
        data,
        _CURSOR_FIELDS,
        issues,
        label=label,
        consequence="the published Cursor plugin schema rejects unknown fields",
    )
    return issues


# Codex. Two sources decide the severities. The Codex runtime (Codex 0.142.5,
# ``codex plugin add`` from a local marketplace) refuses a manifest with "missing
# or invalid plugin.json" when a field it reads has the wrong type: version,
# description, keywords, apps, interface, the interface text and image fields
# (logo, logoDark, composerIcon), capabilities and screenshots. Those are HIGH.
# keywords, capabilities and screenshots are string arrays it reads without an
# Option, so an explicit null there is refused as well. It ignores homepage, repository, license,
# author, and extensions, and drops a skills, commands, mcpServers, or hooks
# value of the wrong shape (loading the default location instead), so it still
# installs the plugin: those are MEDIUM. Its plugin key allows only ASCII
# letters, digits, '_' and '-' in a name, of any length. The packaging
# validator (plugin-eval in openai/plugins 610a632, and the plugin-creator
# scaffold) adds kebab-case, at most 64 characters, semantic versioning,
# description, author, a complete interface, and no unknown fields: those are
# MEDIUM.
_CODEX_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_CODEX_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# Codex stores the plugin under a folder named after its version.
_CODEX_VERSION_RE = re.compile(r"^[A-Za-z0-9.+_-]+$")
CODEX_PACKAGING_NAME_MAX_LENGTH = 64
_CODEX_FIELDS = frozenset(
    {
        "id",
        "name",
        "version",
        "description",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
        "skills",
        "hooks",
        "mcpServers",
        "apps",
        "commands",
        "interface",
        "extensions",
    }
)
_CODEX_INTERFACE_FIELDS = (
    "displayName",
    "shortDescription",
    "longDescription",
    "developerName",
    "category",
    "capabilities",
    "websiteURL",
    "privacyPolicyURL",
    "termsOfServiceURL",
    "defaultPrompt",
)
# Interface fields Codex reads as optional strings; null is fine, any other non-string makes it refuse the manifest.
_CODEX_INTERFACE_STRING_FIELDS = (
    "displayName",
    "shortDescription",
    "longDescription",
    "developerName",
    "category",
    "websiteURL",
    "privacyPolicyURL",
    "termsOfServiceURL",
    "brandColor",
    "composerIcon",
    "logo",
    "logoDark",
)
# Interface fields Codex reads as arrays of strings. They may be left out, but an explicit null, a non-array,
# or a non-string item makes it refuse the manifest (Codex 0.142.5).
_CODEX_INTERFACE_STRING_LIST_FIELDS = ("capabilities", "screenshots")
_CODEX_INTERFACE_ALLOWED = frozenset(
    {
        *_CODEX_INTERFACE_FIELDS,
        "brandColor",
        "composerIcon",
        "logo",
        "logoDark",
        "screenshots",
        "default_prompt",
    }
)
_CODEX_REFUSES = "; Codex refuses the manifest (missing or invalid plugin.json)"
_CODEX_IGNORES = "; Codex ignores the value and still installs the plugin"
# The types Codex accepts for its component fields, the rule a type error states,
# and its level: a wrong 'apps' makes Codex refuse the manifest; a wrong
# 'mcpServers', 'hooks', or 'extensions' is dropped and the plugin still installs.
_CODEX_FIELD_TYPES: dict[str, tuple[tuple[type, ...], str, IssueLevel]] = {
    "apps": ((str,), f"must be a './'-relative path to an .app.json file{_CODEX_REFUSES}", "error"),
    "mcpServers": (
        (str, dict),
        f"must be a './'-relative path or an inline server map{_CODEX_IGNORES}, without these servers",
        "warning",
    ),
    "hooks": (
        (str, dict, list),
        f"must be a path, an array, or an inline object{_CODEX_IGNORES}, without these hooks",
        "warning",
    ),
    "extensions": ((dict,), f"must be an object{_CODEX_IGNORES}", "warning"),
}


def _check_codex_types(
    data: dict[str, Any], issues: list[ManifestIssue], fields: tuple[str, ...], *, label: str
) -> None:
    """Report each of *fields* that is set to a type Codex does not accept (see ``_CODEX_FIELD_TYPES``)."""
    for key in fields:
        accepted, rule, level = _CODEX_FIELD_TYPES[key]
        value = data.get(key)
        if value is not None and not isinstance(value, accepted):
            issues.append(_issue(key, "type", f"{label} '{key}' {rule}.", level))


def _check_codex_name(data: dict[str, Any], issues: list[ManifestIssue], *, label: str) -> None:
    name = _check_string(data, "name", issues, required=True, non_empty=True, label=label, consequence=_CODEX_REFUSES)
    if name is None:
        return
    if not _CODEX_NAME_RE.match(name):
        issues.append(
            _issue(
                "name",
                "pattern",
                f"{label} 'name' {name[:_MAX_QUOTED_NAME_CHARS]!r} may use only ASCII letters, digits, '_' and "
                f"'-'{_CODEX_REFUSES}.",
                suggestion="Rename the plugin in kebab-case, for example 'my-plugin'.",
            )
        )
        return
    if len(name) > CODEX_PACKAGING_NAME_MAX_LENGTH:
        issues.append(
            _issue(
                "name",
                "too_long",
                f"{label} 'name' is {len(name)} characters. Codex installs it, but Codex plugin packaging allows at "
                f"most {CODEX_PACKAGING_NAME_MAX_LENGTH}.",
                "warning",
                "Shorten the name before packaging the plugin.",
            )
        )
    if not _CODEX_KEBAB_RE.match(name):
        issues.append(
            _issue(
                "name",
                "not_kebab_case",
                f"{label} 'name' {name[:_MAX_QUOTED_NAME_CHARS]!r} is not lowercase kebab-case, which Codex plugin "
                "packaging expects.",
                "warning",
                "Use lowercase letters, digits, and single hyphens.",
            )
        )


def _check_codex_version(data: dict[str, Any], issues: list[ManifestIssue], *, label: str) -> None:
    version = data.get("version")
    if version is None:
        issues.append(
            _issue(
                "version",
                "missing",
                f"{label} has no 'version'. Codex plugin packaging expects one; the Codex runtime loads the plugin "
                "without it.",
                "warning",
                "Add a 'version' string before packaging the plugin.",
            )
        )
        return
    if not isinstance(version, str):
        issues.append(
            _issue(
                "version",
                "type",
                f"{label} field 'version' must be a string (got {_type_name(version)}){_CODEX_REFUSES}.",
            )
        )
        return
    if not version.strip():
        issues.append(_issue("version", "empty", f"{label} 'version' is blank{_CODEX_REFUSES}."))
        return
    if not _CODEX_VERSION_RE.match(version) or version in {".", ".."}:
        issues.append(
            _issue(
                "version",
                "invalid",
                f"{label} 'version' {version[:_MAX_QUOTED_VERSION_CHARS]!r} may use only ASCII letters, digits, "
                "'.', '+', '_' and '-'; Codex stores the plugin in a folder named after it, so it does not install "
                "the plugin.",
                suggestion="Use a semantic version such as 1.0.0.",
            )
        )
        return
    _check_semver(data, issues, label=label, level="warning")


def _check_codex_interface(
    interface: Any, issues: list[ManifestIssue], *, label: str, require_complete: bool = True
) -> None:
    if interface is None:
        if require_complete:
            issues.append(
                _issue(
                    "interface",
                    "missing",
                    f"{label} has no 'interface' block; Codex plugin packaging requires one for the plugin's display "
                    "metadata.",
                    "warning",
                    "Add an interface object with displayName, shortDescription, and the other presentation fields.",
                )
            )
        return
    if not isinstance(interface, dict):
        issues.append(
            _issue(
                "interface",
                "type",
                f"{label} 'interface' must be an object (got {_type_name(interface)}){_CODEX_REFUSES}.",
            )
        )
        return
    for key in _CODEX_INTERFACE_STRING_FIELDS:
        if key in interface and interface[key] is not None and not isinstance(interface[key], str):
            issues.append(
                _issue(
                    f"interface.{key}",
                    "type",
                    f"{label} 'interface.{key}' must be a string (got {_type_name(interface[key])}){_CODEX_REFUSES}.",
                )
            )
    for key in _CODEX_INTERFACE_STRING_LIST_FIELDS:
        if key in interface and not _is_str_list(interface[key]):
            example = '["Interactive"]' if key == "capabilities" else '["./assets/screenshot.png"]'
            issues.append(
                _issue(
                    f"interface.{key}",
                    "type",
                    f"{label} 'interface.{key}' must be an array of strings (got {_type_name(interface[key])})"
                    f"{_CODEX_REFUSES}.",
                    suggestion=f"Write it as a list, for example {example}, or leave it out.",
                )
            )
    if not require_complete:
        return
    missing = [key for key in _CODEX_INTERFACE_FIELDS if key not in interface]
    if missing:
        issues.append(
            _issue(
                "interface",
                "incomplete",
                f"{label} 'interface' is missing {', '.join(missing)}, which the plugin-eval validator requires.",
                "warning",
                "Complete the interface block before packaging the plugin.",
            )
        )
    for key in sorted(str(key) for key in interface if key not in _CODEX_INTERFACE_ALLOWED)[:32]:
        issues.append(
            _issue(
                "interface",
                "unknown_key",
                f"{label} 'interface' does not define {key[:64]!r}; Codex plugin packaging rejects it and Codex "
                "ignores it.",
                "warning",
            )
        )


def _check_codex_keywords(data: dict[str, Any], issues: list[ManifestIssue], *, label: str) -> None:
    """Codex reads 'keywords' as an array of strings: it may be left out, but an explicit null is refused too."""
    if "keywords" in data and data["keywords"] is None:
        issues.append(
            _issue(
                "keywords",
                "type",
                f"{label} field 'keywords' must be an array of strings (got null){_CODEX_REFUSES}.",
                suggestion="Write it as a list of strings, or leave it out.",
            )
        )
        return
    _check_string_list(data, "keywords", issues, label=label, consequence=_CODEX_REFUSES)


def _validate_codex(data: dict[str, Any]) -> list[ManifestIssue]:
    label = "Codex plugin manifest"
    issues: list[ManifestIssue] = []
    _check_codex_name(data, issues, label=label)
    _check_codex_version(data, issues, label=label)
    description = _check_string(data, "description", issues, label=label, consequence=_CODEX_REFUSES)
    if (description is None and data.get("description") is None) or (
        description is not None and not description.strip()
    ):
        issues.append(
            _issue(
                "description",
                "missing",
                f"{label} has no non-empty 'description'. Codex plugin packaging expects one; the Codex runtime "
                "loads the plugin without it.",
                "warning",
                "Add a 'description' string before packaging the plugin.",
            )
        )
    author = data.get("author")
    if author is None:
        issues.append(
            _issue(
                "author",
                "missing",
                f"{label} has no 'author'. Codex plugin packaging expects one; the Codex runtime ignores it.",
                "warning",
                "Add an author object with a name.",
            )
        )
    elif not isinstance(author, dict):
        issues.append(
            _issue(
                "author",
                "type",
                f"{label} field 'author' should be an object; Codex plugin packaging expects one.",
                "warning",
            )
        )
    elif not isinstance(author.get("name"), str) or not author["name"].strip():
        issues.append(
            _issue("author.name", "missing", f"{label} 'author' should define a non-empty 'name'.", "warning")
        )
    for key in ("homepage", "repository", "license"):
        _check_string(data, key, issues, label=label, level="warning", consequence=_CODEX_IGNORES)
    _check_codex_keywords(data, issues, label=label)
    for key in ("skills", "commands"):
        _check_string_or_list(data, key, issues, label=label, level="warning", consequence=_CODEX_IGNORES)
    _check_codex_types(data, issues, ("apps", "mcpServers", "hooks"), label=label)
    _check_codex_interface(data.get("interface"), issues, label=label)
    _check_codex_types(data, issues, ("extensions",), label=label)
    _check_unknown_fields(
        data,
        _CODEX_FIELDS,
        issues,
        label=label,
        consequence="Codex ignores it, so whatever it declares does not load, and Codex plugin packaging rejects it",
    )
    return issues


def _validate_codex_overlay(data: dict[str, Any]) -> list[ManifestIssue]:
    """Field problems of a ``.codex-plugin/plugin.json`` overlay beside a root Agent Plugins manifest.

    OpenAI documents this overlay as the source of the OpenAI-specific settings
    (``apps``, ``hooks``, and ``interface``) when the root manifest has no
    ``extensions["com.openai"]`` object. The root manifest carries the plugin's
    identity, so the overlay needs no name, version, description, or author.
    The value types follow the same Codex rules as a full Codex manifest, and
    Codex reads the overlay's 'keywords' too, so a wrong or null value there
    makes it refuse the overlay just the same.
    """
    label = "Codex overlay manifest"
    issues: list[ManifestIssue] = []
    _check_codex_types(data, issues, ("apps", "hooks"), label=label)
    _check_codex_keywords(data, issues, label=label)
    _check_codex_interface(data.get("interface"), issues, label=label, require_complete=False)
    return issues


_VALIDATORS: dict[str, Callable[[dict[str, Any]], list[ManifestIssue]]] = {
    PLUGIN_CONTAINED_MANIFEST_TYPE: _validate_claude,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE: _validate_agent_plugins,
    PLUGIN_CODEX_MANIFEST_TYPE: _validate_codex,
    PLUGIN_CURSOR_MANIFEST_TYPE: _validate_cursor,
}


def validate_manifest_fields(manifest_type: str, data: dict[str, Any], *, overlay: bool = False) -> list[ManifestIssue]:
    """Return the field problems of one parsed contained manifest (Claude Code, Agent Plugins, Codex, Cursor).

    Bundle-reference manifests keep their pydantic model and return no issues
    here. ``overlay`` validates a Codex manifest as the
    documented overlay of a root Agent Plugins manifest (only ``apps``,
    ``hooks``, and ``interface`` apply).
    """
    if overlay and manifest_type == PLUGIN_CODEX_MANIFEST_TYPE:
        return _validate_codex_overlay(data)
    validator = _VALIDATORS.get(manifest_type)
    return validator(data) if validator is not None else []


def normalized_component_manifest(manifest_type: str, data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the component declarations the inventory reads, in Claude Code field names.

    Only the fields a format defines as component fields are kept, so a field
    another format uses (for example ``lspServers`` in a Cursor manifest) is not
    mistaken for a component declaration.
    """
    if not isinstance(data, dict):
        return None
    if manifest_type in {PLUGIN_CONTAINED_MANIFEST_TYPE, PLUGIN_MANIFEST_TYPE}:
        return data
    profile = profile_for(manifest_type)
    return {key: value for key, value in data.items() if key in profile.component_fields}
