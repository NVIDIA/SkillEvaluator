# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-format schemas and component mappings for contained plugin manifests.

SkillEvaluator supports these native plugin manifest formats. Each is a
*contained* plugin: its components ship inside the plugin root.

* **Claude Code** (``.claude-plugin/plugin.json``): shallow validation only (a
  non-empty ``name`` of at most 64 characters), as before.
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
@dataclass(frozen=True)
class FormatProfile:
    """Where one manifest format declares components and where they live by default.

    Every default path is root-relative POSIX. ``None`` means the format has no
    such default location, so the inventory does not look there.
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
    # Report a MEDIUM style finding for a declared path without a leading "./".
    require_dot_relative: bool = False
    # A declared component field replaces default-folder discovery for that type
    # (Cursor); otherwise it supplements the default folder (Claude Code).
    declared_replaces_default: bool = False
    # Manifest fields that declare components (the builder reads only these).
    component_fields: frozenset[str] = frozenset()
    default_skills_dir: str | None = "skills"
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
# required. ${CURSOR_PLUGIN_ROOT} and ${CLAUDE_PLUGIN_ROOT} name the root.
CURSOR_PROFILE = FormatProfile(
    manifest_type=PLUGIN_CURSOR_MANIFEST_TYPE,
    label="Cursor plugin",
    manifest_path=f"{PLUGIN_CURSOR_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}",
    root_prefixes=("${CURSOR_PLUGIN_ROOT}", "${CLAUDE_PLUGIN_ROOT}"),
    manifest_path_prefixes=("${CURSOR_PLUGIN_ROOT}", "${CLAUDE_PLUGIN_ROOT}"),
    relative_hook_scripts=True,
    declared_replaces_default=True,
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
    default whenever the field is set. Codex replaces it only when it keeps the
    value: a path it accepts (:func:`codex_accepts_path`), a list with at least
    one such path, or an inline form for ``hooks`` and ``mcpServers``. Any other
    value is dropped by Codex, which then loads the default location, so the
    default must still be inventoried and checked.
    """
    if not profile.declared_replaces_default or value is None:
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


def _issue(
    field_name: str, error: str, message: str, level: IssueLevel = "error", suggestion: str = ""
) -> ManifestIssue:
    return ManifestIssue(field_name, error, message, level, suggestion)


def _check_string(
    data: dict[str, Any],
    key: str,
    issues: list[ManifestIssue],
    *,
    required: bool = False,
    non_empty: bool = False,
    label: str,
) -> str | None:
    if key not in data or data[key] is None:
        if required:
            issues.append(_issue(key, "missing", f"{label} must define '{key}'.", suggestion=f"Add a '{key}' string."))
        return None
    value = data[key]
    if not isinstance(value, str):
        issues.append(_issue(key, "type", f"{label} field '{key}' must be a string (got {type(value).__name__})."))
        return None
    if non_empty and not value.strip():
        issues.append(_issue(key, "empty", f"{label} field '{key}' must not be empty."))
        return None
    return value


def _check_string_list(data: dict[str, Any], key: str, issues: list[ManifestIssue], *, label: str) -> None:
    value = data.get(key)
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        issues.append(_issue(key, "type", f"{label} field '{key}' must be an array of strings."))


def _check_string_or_list(data: dict[str, Any], key: str, issues: list[ManifestIssue], *, label: str) -> None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        issues.append(_issue(key, "type", f"{label} field '{key}' must be a path string or an array of path strings."))


def _check_unknown_fields(
    data: dict[str, Any],
    allowed: frozenset[str],
    issues: list[ManifestIssue],
    *,
    label: str,
    consequence: str,
) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    for key in unknown[:32]:
        issues.append(
            _issue(
                key,
                "unknown_field",
                f"{label} does not define the top-level field '{key}'; {consequence}.",
                "warning",
                f"Remove '{key}' or move it where the format allows client-specific data.",
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
                f"{label} 'version' {version[:64]!r} is not a semantic version (MAJOR.MINOR.PATCH).",
                level,
                "Use a semantic version such as 1.0.0.",
            )
        )


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
    if name is not None and (not name or len(name) > 64 or not _AGENT_PLUGINS_NAME_RE.match(name)):
        issues.append(
            _issue(
                "name",
                "pattern",
                f"{label} 'name' {name[:80]!r} must be 1-64 characters of a-z, 0-9, '-' and '.', start and end "
                "with a letter or digit, and contain no '--' or '..'.",
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
            for namespace, value in list(extensions.items())[:64]:
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
                f"{label} 'name' {name[:80]!r} must be lowercase kebab-case: a-z, 0-9, '-' and '.', starting and "
                "ending with a letter or digit.",
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


# Codex: plugin-eval (openai/plugins 610a632) requires name, version,
# description, author, and interface; the package validator adds semantic
# versioning, author.name, and the name rule below. The Codex runtime itself is
# lenient: it needs only a usable name and correctly typed fields. So the name
# rule and type errors are HIGH, and the other packaging requirements (version,
# description, author, interface) are advisory.
_CODEX_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_CODEX_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
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


def _validate_codex(data: dict[str, Any]) -> list[ManifestIssue]:
    label = "Codex plugin manifest"
    issues: list[ManifestIssue] = []
    name = _check_string(data, "name", issues, required=True, non_empty=True, label=label)
    if name is not None:
        if len(name) > NAME_MAX_LENGTH or not _CODEX_NAME_RE.match(name):
            issues.append(
                _issue(
                    "name",
                    "pattern",
                    f"{label} 'name' {name[:80]!r} must be at most {NAME_MAX_LENGTH} characters, start with a letter "
                    "or digit, and use only letters, digits, '_' and '-'.",
                    suggestion="Rename the plugin in kebab-case, for example 'my-plugin'.",
                )
            )
        elif not _CODEX_KEBAB_RE.match(name):
            issues.append(
                _issue(
                    "name",
                    "not_kebab_case",
                    f"{label} 'name' {name!r} is not lowercase kebab-case, which Codex plugin packaging expects.",
                    "warning",
                    "Use lowercase letters, digits, and single hyphens.",
                )
            )
    # The Codex runtime reads version and description as optional strings and
    # ignores author, so only a wrong type is a load failure (HIGH). Missing
    # values are packaging metadata (MEDIUM).
    for key in ("version", "description"):
        value = _check_string(data, key, issues, label=label)
        if value is None and key in data and data[key] is not None:
            continue  # a wrong type was reported above
        if value is None or not value.strip():
            issues.append(
                _issue(
                    key,
                    "missing",
                    f"{label} has no non-empty '{key}'. Codex plugin packaging expects one; the Codex runtime loads "
                    "the plugin without it.",
                    "warning",
                    f"Add a '{key}' string before packaging the plugin.",
                )
            )
    _check_semver(data, issues, label=label, level="warning")
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
        _check_string(data, key, issues, label=label)
    _check_string_list(data, "keywords", issues, label=label)
    _check_string_or_list(data, "skills", issues, label=label)
    _check_string_or_list(data, "commands", issues, label=label)
    apps = data.get("apps")
    if apps is not None and not isinstance(apps, str):
        issues.append(_issue("apps", "type", f"{label} 'apps' must be a './'-relative path to an .app.json file."))
    mcp_servers = data.get("mcpServers")
    if mcp_servers is not None and not isinstance(mcp_servers, str | dict):
        issues.append(
            _issue("mcpServers", "type", f"{label} 'mcpServers' must be a './'-relative path or an inline server map.")
        )
    hooks = data.get("hooks")
    if hooks is not None and not isinstance(hooks, str | dict | list):
        issues.append(_issue("hooks", "type", f"{label} 'hooks' must be a path, an array, or an inline object."))
    interface = data.get("interface")
    if interface is None:
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
    elif not isinstance(interface, dict):
        issues.append(_issue("interface", "type", f"{label} 'interface' must be an object."))
    else:
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
    extensions = data.get("extensions")
    if extensions is not None and not isinstance(extensions, dict):
        issues.append(_issue("extensions", "type", f"{label} 'extensions' must be an object."))
    return issues


def _validate_codex_overlay(data: dict[str, Any]) -> list[ManifestIssue]:
    """Field problems of a ``.codex-plugin/plugin.json`` overlay beside a root Agent Plugins manifest.

    OpenAI documents this overlay as the source of the OpenAI-specific settings
    (``apps``, ``hooks``, and ``interface``) when the root manifest has no
    ``extensions["com.openai"]`` object. The root manifest carries the plugin's
    identity, so the overlay needs no name, version, description, or author.
    """
    label = "Codex overlay manifest"
    issues: list[ManifestIssue] = []
    apps = data.get("apps")
    if apps is not None and not isinstance(apps, str):
        issues.append(_issue("apps", "type", f"{label} 'apps' must be a './'-relative path to an .app.json file."))
    hooks = data.get("hooks")
    if hooks is not None and not isinstance(hooks, str | dict | list):
        issues.append(_issue("hooks", "type", f"{label} 'hooks' must be a path, an array, or an inline object."))
    interface = data.get("interface")
    if interface is not None and not isinstance(interface, dict):
        issues.append(_issue("interface", "type", f"{label} 'interface' must be an object."))
    return issues


_VALIDATORS: dict[str, Callable[[dict[str, Any]], list[ManifestIssue]]] = {
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE: _validate_agent_plugins,
    PLUGIN_CODEX_MANIFEST_TYPE: _validate_codex,
    PLUGIN_CURSOR_MANIFEST_TYPE: _validate_cursor,
}


def validate_manifest_fields(manifest_type: str, data: dict[str, Any], *, overlay: bool = False) -> list[ManifestIssue]:
    """Return the field problems of one parsed manifest of a new contained format.

    Claude Code and bundle-reference manifests keep their existing validators and
    return no issues here. ``overlay`` validates a Codex manifest as the
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
