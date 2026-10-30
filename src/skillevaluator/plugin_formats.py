# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"Private format profiles and parsing; no public discovery or validation registration."

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from skillevaluator.constants import (
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

AGENT_PLUGINS_SCHEMA_PREFIX = "https://agent-plugins.org/schemas/"

_AGENT_PLUGINS_SCHEMA_RE = re.compile(
    r"^https://agent-plugins\.org/schemas/(?P<version>\d+\.\d+\.\d+)/(?P<kind>plugin|mcp)\.schema\.json$"
)


def agent_plugins_schema_version(value: Any, *, kind: str = "plugin") -> str | None:
    """Return the Agent Plugins version named by a ``$schema`` URL, else ``None``.

    The comparison is exact, as in Codex and Hermes: surrounding whitespace
    makes the value an invalid identifier.
    """
    if not isinstance(value, str):
        return None
    match = _AGENT_PLUGINS_SCHEMA_RE.fullmatch(value)
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


def normalized_component_manifest(manifest_type: str, data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only declarations defined by this manifest's component dialect.

    Claude and bundle manifests retain their existing fields. Other clients
    must not accidentally load fields that belong to another format.
    """
    if not isinstance(data, dict):
        return None
    if manifest_type in {PLUGIN_CONTAINED_MANIFEST_TYPE, PLUGIN_MANIFEST_TYPE}:
        return data
    profile = profile_for(manifest_type)
    return {key: value for key, value in data.items() if key in profile.component_fields}
