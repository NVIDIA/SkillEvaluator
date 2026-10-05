# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native plugin loading for Tier 3 (``--plugin-load native|auto``).

The default Tier 3 plugin path (``wrapper``) stages a generated wrapper
``SKILL.md`` with the plugin rules inlined, the member skills beside it, and
the runnable MCP servers through Harbor's task MCP list. Native loading instead
stages the plugin the way each harness loads plugins on its own, for the
with-plugin arm only. The member-skills and without-plugin arms are unchanged,
so lift comparisons stay meaningful.

This module is host-side and imports no Harbor code. It owns:

* the per-harness adapters and their component-support matrices;
* the ``plugin_load`` plan recorded in plugin provenance;
* the plugin snapshot read from the plugin root (bounded, no-follow reads);
* the generated configs, the census-wrapped hooks, and the in-container
  ``setup.sh`` that stages per-run config and writes the load census;
* parsing the load census and mapping it into the C2 coverage states.

The load census has two strengths of evidence. ``listed`` means setup found
the staged files where the harness reads them; nothing proves the harness
loaded them. ``loaded`` means the harness itself reported the component (for
Claude Code, the ``system/init`` event it writes at startup, parsed on the host
by the collector). Only ``loaded`` promotes a coverage row to ``loaded``; a
``listed`` component stays ``staged``.

Filesystem staging into a Harbor task (secure copies of member skills and the
plugin tree) lives in :mod:`skillevaluator.tier3.harbor.native_staging`, and the
Harbor agent wrappers that run ``setup.sh`` right before the agent starts live
in :mod:`skillevaluator.tier3.harbor.native_agents`.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import stat
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES, PLUGIN_CONFIG_MAX_BYTES
from skillevaluator.plugin_components import (
    COMPONENT_TYPES,
    PluginInventory,
    PluginRootReader,
    normalize_declared_path,
    parse_markdown,
    summarize_coverage,
)
from skillevaluator.tier3.toml_utils import toml_quote
from skillevaluator.tier3_environments import PLUGIN_LOAD_CHOICES
from skillevaluator.utils.secure_fs import SecurePathError, stat_is_link_or_reparse
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_json

#: Build-context directory (inside a task's ``environment/``) copied to ``/skilleval``.
BUNDLE_DIRNAME = "skilleval"
CONTAINER_ROOT = "/skilleval"
NATIVE_ROOT = "/skilleval/native"
SETUP_SCRIPT = "/skilleval/native/setup.sh"
HOOK_CENSUS_SCRIPT = "/skilleval/hook_census.sh"
HOOK_CENSUS_TEMPLATE = Path(__file__).resolve().parent / "harbor" / "templates" / "hook_census.sh"
LOAD_CENSUS_FILENAME = "skilleval-load-census.json"
WRAPPER_ADAPTER = "wrapper"
STAGED_EVIDENCE = "staged"

#: What the default wrapper path does with each component type.
WRAPPER_COMPONENTS: dict[str, str] = {
    component_type: ("wrapper" if component_type in {"skill", "rule", "mcp"} else "unsupported")
    for component_type in COMPONENT_TYPES
}

# Bounds for plugin-controlled hook configs and census files.
MAX_HOOK_EVENTS = 64
MAX_HOOK_GROUPS = 64
MAX_HOOK_HANDLERS = 256
MAX_CENSUS_BYTES = 256 * 1024
MAX_CENSUS_ENTRIES = 512
MAX_CENSUS_TEXT = 300
#: Bytes read from the start of a harness log when looking for its startup event.
MAX_HARNESS_LOG_PREFIX_BYTES = 4 * 1024 * 1024
#: Harbor tees Claude Code's stream-json output here (``<trial>/agent/`` on the host).
CLAUDE_CODE_LOG_FILENAME = "claude-code.txt"
LISTED_KEY = "listed"
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PLUGIN_NAME_RE = re.compile(r"[^a-z0-9-]+")
_BYPASS_PERMISSION_MODES = frozenset({"bypasspermissions"})


class PluginLoadError(ValueError):
    """Raised when ``--plugin-load native`` cannot be honored for a selected agent."""


# --------------------------------------------------------------------------- #
# Plugin snapshot                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NativeTextComponent:
    """One Markdown component (agent, command, or output style), read bounded and no-follow."""

    type: str
    name: str
    rel: str | None
    text: str


@dataclass(frozen=True)
class NativeHookSource:
    """One hooks config source. ``name`` is the inventory component name (the hook_id source).

    ``dialect`` is the hook vocabulary of the source format: ``claude`` (Claude
    Code and Codex event names) or ``cursor`` (Cursor event names, which Claude
    Code ignores unless they are translated). ``root_prefixes`` are the format's
    plugin-root placeholders (``${PLUGIN_ROOT}`` for Codex); Claude Code expands
    only ``${CLAUDE_PLUGIN_ROOT}``, so the others are rewritten to it.
    """

    name: str
    rel: str | None
    config: Any
    dialect: str = "claude"
    root_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativePluginSource:
    """Everything a harness adapter needs to stage the plugin natively.

    Built once per run from the plugin root with bounded, no-follow reads.
    Member skills are staged later with the secure tree copy.
    ``excluded_paths`` is evaluator data (the resolved evals source) that a
    whole-plugin-tree copy must never carry when it lies inside the plugin root.

    ``plugin_file_mcp_servers`` launch from files inside the plugin through the
    plugin-root variable (already rewritten to ``${CLAUDE_PLUGIN_ROOT}``). Only an
    adapter that copies the plugin tree (Claude Code) can start them.
    ``mcp_declared`` holds each server's declared ``env``/``headers``, which only
    the Claude Code adapter applies. ``refusals`` lists components that enable a
    permission bypass as ``(type, name, message)``; an adapter that would stage
    that type natively refuses (:func:`native_refusal`).
    """

    plugin_name: str
    description: str
    contained: bool
    plugin_root: Path
    manifest: Mapping[str, Any]
    manifest_rel: str
    member_skills: tuple[Path, ...] = ()
    rules: tuple[tuple[str, str], ...] = ()
    mcp_servers: tuple[Mapping[str, Any], ...] = ()
    texts: tuple[NativeTextComponent, ...] = ()
    hooks: tuple[NativeHookSource, ...] = ()
    other: tuple[tuple[str, str], ...] = ()
    excluded_paths: tuple[Path, ...] = ()
    plugin_file_mcp_servers: tuple[Mapping[str, Any], ...] = ()
    mcp_declared: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    refusals: tuple[tuple[str, str, str], ...] = ()
    #: The plugin ``settings.json`` keys Claude Code applies to plugins.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: LSP server configs by name, from ``.lsp.json`` or ``lspServers``.
    lsp_servers: Mapping[str, Any] = field(default_factory=dict)

    def present_types(self) -> set[str]:
        present: set[str] = set()
        if self.member_skills:
            present.add("skill")
        if self.rules:
            present.add("rule")
        if self.mcp_servers:
            present.add("mcp")
        if self.hooks:
            present.add("hook")
        present.update(text.type for text in self.texts)
        present.update(component_type for component_type, _name in self.other)
        return present


def plugin_slug(name: str) -> str:
    """A Claude Code-safe plugin name: kebab-case, no spaces, ``@``, ``:``, or separators."""
    slug = _PLUGIN_NAME_RE.sub("-", str(name).lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)[:64] or "plugin"


def safe_name(value: str, *, limit: int = 96) -> str:
    """A filesystem- and shell-safe token for generated file names and census rows."""
    cleaned = _SAFE_NAME_RE.sub("-", str(value)).strip("-.") or "item"
    return cleaned[:limit]


def _bypass_refusal(value: Any, *, where: str) -> str | None:
    """Why ``value`` must not be staged natively (a permission-bypass flag), or ``None``."""
    from skillevaluator.validators.mcp_static import permission_bypass_issues

    issues = permission_bypass_issues(value)
    if issues:
        return (
            f"Refusing to stage {where} natively: {issues[0].message}. "
            "Native plugin loading never enables a permission-bypass flag."
        )
    return None


def _bypass_frontmatter_refusal(frontmatter: Mapping[str, Any], *, where: str) -> str | None:
    mode = frontmatter.get("permissionMode", frontmatter.get("permission-mode"))
    if isinstance(mode, str) and mode.strip().replace("_", "").replace("-", "").casefold() in _BYPASS_PERMISSION_MODES:
        return (
            f"Refusing to stage {where} natively: it sets permissionMode to bypassPermissions. "
            "Native plugin loading never enables a permission-bypass mode."
        )
    return _bypass_refusal(dict(frontmatter), where=where)


def native_refusal(adapter: HarnessAdapter, source: NativePluginSource) -> str | None:
    """The first bypass refusal for a component type ``adapter`` stages natively, or ``None``.

    A component that enables a permission bypass blocks only the adapters that
    would stage it: Codex, which never stages hooks, can still load a plugin
    whose hook carries a bypass flag (the hook is reported ``not_loaded``).
    """
    modes = adapter.component_modes()
    for component_type, _name, message in source.refusals:
        if modes.get(component_type) == "native":
            return message
    return None


def native_component_types(adapter: HarnessAdapter, source: NativePluginSource) -> set[str]:
    """Component types present in ``source`` that ``adapter`` loads natively.

    Plugin-file MCP servers count only for an adapter that copies the plugin
    tree; settings and LSP count only when there is something to stage.
    """
    present = source.present_types()
    if adapter.copies_plugin_tree and source.plugin_file_mcp_servers:
        present.add("mcp")
    if not source.settings:
        present.discard("settings")
    if not source.lsp_servers:
        present.discard("lsp")
    modes = adapter.component_modes()
    return {component_type for component_type in present if modes.get(component_type) == "native"}


def _inline_hook_config(manifest: Mapping[str, Any], name: str) -> Any:
    declared = manifest.get("hooks")
    values = declared if isinstance(declared, list) else [declared]
    inline = [value for value in values if isinstance(value, dict | list) and not isinstance(value, str)]
    if name == "inline":
        return inline[0] if inline else None
    match = re.fullmatch(r"inline\[(\d+)\]", name)
    if match is None:
        return None
    index = int(match.group(1))
    return values[index] if 0 <= index < len(values) else None


def _inline_command_content(manifest: Mapping[str, Any], name: str) -> str | None:
    commands = manifest.get("commands")
    if not isinstance(commands, dict):
        return None
    entry = commands.get(name)
    if isinstance(entry, dict) and isinstance(entry.get("content"), str):
        description = entry.get("description") if isinstance(entry.get("description"), str) else ""
        header = f"---\ndescription: {json.dumps(description)}\n---\n\n" if description else ""
        return header + entry["content"]
    return None


def build_native_source(
    *,
    inventory: PluginInventory,
    manifest: Mapping[str, Any],
    plugin_root: Path,
    contained: bool,
    manifest_rel: str,
    plugin_name: str,
    description: str,
    member_skills: Sequence[Path],
    rules: Sequence[tuple[str, str]],
    mcp_servers: Sequence[Mapping[str, Any]],
    nest_flat_hooks: bool = False,
    excluded_paths: Sequence[Path] = (),
    plugin_file_mcp_servers: Sequence[Mapping[str, Any]] = (),
    mcp_declared: Mapping[str, Mapping[str, Any]] | None = None,
    hook_dialect: str | None = None,
    hook_root_prefixes: Sequence[str] = (),
) -> NativePluginSource:
    """Snapshot the components a native adapter can stage.

    *manifest* uses Claude Code component field names (callers normalize the
    Codex, Cursor, and Agent Plugins formats first). *nest_flat_hooks* converts
    Cursor's flat per-event hook handlers into matcher groups before wrapping;
    *hook_dialect* names the hook event vocabulary (``cursor`` when flat hooks
    are nested, ``claude`` otherwise); *hook_root_prefixes* are the format's
    plugin-root placeholders, rewritten to ``${CLAUDE_PLUGIN_ROOT}`` in staged
    hook commands.

    Markdown and config files are read through :class:`PluginRootReader`
    (anchored, bounded, never following links). A hook config, subagent,
    settings file, or LSP server that enables a permission-bypass flag or mode
    is recorded in ``refusals``: every adapter that would stage that type
    natively refuses it (fail closed), and adapters that never stage the type
    are unaffected.
    """
    reader = PluginRootReader(plugin_root)
    dialect = hook_dialect or ("cursor" if nest_flat_hooks else "claude")
    texts: list[NativeTextComponent] = []
    hooks: list[NativeHookSource] = []
    other: list[tuple[str, str]] = []
    refusals: list[tuple[str, str, str]] = []
    settings: dict[str, Any] = {}
    lsp_servers: dict[str, Any] = {}

    def _refuse(component_type: str, name: str, message: str | None) -> None:
        if message is not None:
            refusals.append((component_type, name, message))

    for component in inventory.components:
        # Components only an additional manifest declares are not staged (the
        # selected manifest's view defines the plugin), as in the wrapper path.
        if component.problem is not None or component.declared_by is not None:
            continue
        if component.type in {"agent", "command", "output_style"}:
            text: str | None
            if component.path is None:
                text = _inline_command_content(manifest, component.name) if component.type == "command" else None
            else:
                try:
                    text = reader.read_text(PurePosixPath(component.path), CONTENT_DEDUP_MAX_FILE_BYTES)
                except (SecurePathError, OSError) as exc:
                    raise ValueError(
                        f"Refusing unsafe or unreadable plugin {component.type} '{component.path}': {exc}"
                    ) from exc
            if text is None:
                continue
            _refuse(
                component.type,
                component.name,
                _bypass_frontmatter_refusal(
                    parse_markdown(text).frontmatter, where=f"plugin {component.type} '{component.name}'"
                ),
            )
            texts.append(NativeTextComponent(component.type, component.name, component.path, text))
        elif component.type == "hook":
            if component.path == manifest_rel:
                config = _inline_hook_config(manifest, component.name)
            elif component.path:
                config = _read_plugin_json(reader, component.path, "hooks")
            else:
                config = None
            if config is None:
                continue
            if nest_flat_hooks:
                from skillevaluator.plugin_components import _nested_hook_groups

                config = _nested_hook_groups(config)
            _refuse("hook", component.name, _bypass_refusal(config, where=f"plugin hooks '{component.name}'"))
            hooks.append(NativeHookSource(component.name, component.path, config, dialect, tuple(hook_root_prefixes)))
        elif component.type not in {"skill", "rule", "mcp"}:
            # LSP, monitors, settings, Codex apps, Agent Plugins extensions: every
            # adapter lists them (staged or not_loaded) in its load census.
            other.append((component.type, component.name))
            if component.type == "settings" and component.path == _PLUGIN_SETTINGS_FILE:
                applied = _plugin_settings(_read_plugin_json(reader, component.path, "settings"))
                _refuse("settings", component.name, _bypass_refusal(applied, where="plugin settings.json"))
                settings.update(applied)
            elif component.type == "lsp" and component.path:
                config = (
                    manifest.get("lspServers")
                    if component.path == manifest_rel
                    else _read_plugin_json(reader, component.path, "LSP servers")
                )
                server = _lsp_server_config(config, component.name)
                if server is not None:
                    _refuse(
                        "lsp", component.name, _bypass_refusal(server, where=f"plugin LSP server '{component.name}'")
                    )
                    lsp_servers[component.name] = server
    # bin/ executables join the Bash PATH while a Claude Code plugin is enabled.
    if reader.kind(PurePosixPath("bin")) == "dir":
        other.append(("bin", "bin"))
    return NativePluginSource(
        plugin_name=plugin_name,
        description=description,
        contained=contained,
        plugin_root=plugin_root,
        manifest=dict(manifest),
        manifest_rel=manifest_rel,
        member_skills=tuple(member_skills),
        rules=tuple((str(name), str(content)) for name, content in rules),
        mcp_servers=tuple(dict(server) for server in mcp_servers),
        texts=tuple(texts),
        hooks=tuple(hooks),
        other=tuple(other),
        excluded_paths=tuple(excluded_paths),
        plugin_file_mcp_servers=tuple(dict(server) for server in plugin_file_mcp_servers),
        mcp_declared={str(name): dict(values) for name, values in (mcp_declared or {}).items()},
        refusals=tuple(refusals),
        settings=settings,
        lsp_servers=lsp_servers,
    )


_PLUGIN_SETTINGS_FILE = "settings.json"
#: The only plugin ``settings.json`` keys Claude Code applies.
_PLUGIN_SETTINGS_KEYS = ("agent", "subagentStatusLine")


def _read_plugin_json(reader: PluginRootReader, rel: str, label: str) -> Any:
    try:
        return load_bounded_json(reader.read_text(PurePosixPath(rel), PLUGIN_CONFIG_MAX_BYTES, config=True))
    except (SecurePathError, StructuredDataError, OSError, ValueError) as exc:
        raise ValueError(f"Refusing unsafe or unreadable plugin {label} '{rel}': {exc}") from exc


def _plugin_settings(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    return {key: json.loads(json.dumps(config[key])) for key in _PLUGIN_SETTINGS_KEYS if key in config}


def _lsp_server_config(config: Any, name: str) -> dict[str, Any] | None:
    """One named LSP server from an ``lspServers`` value or ``.lsp.json`` document."""
    documents = config if isinstance(config, list) else [config]
    for document in reversed(documents):
        servers = document.get("lspServers", document) if isinstance(document, dict) else None
        server = servers.get(name) if isinstance(servers, dict) else None
        if isinstance(server, dict):
            return json.loads(json.dumps(server))
    return None


# --------------------------------------------------------------------------- #
# Hooks: census wrapping                                                       #
# --------------------------------------------------------------------------- #
def hook_id(source: str, event: str, group_index: int, handler_index: int) -> str:
    """The shared hook identifier: ``<source>#<event>[<group>].hooks[<handler>]``.

    Matches the Tier 1 ``hook_risk`` rows, so census rows join to them.
    """
    return f"{source}#{event}[{group_index}].hooks[{handler_index}]"


def iter_hook_handlers(config: Any) -> Iterator[tuple[str, int, Any, int, Any]]:
    """Yield ``(event, group_index, group, handler_index, handler)`` for a hooks config.

    Accepts the ``hooks.json`` / settings shape ``{"hooks": {Event: [...]}}`` and
    the inline shape ``{Event: [...]}``. Bounded by the ``MAX_HOOK_*`` limits.
    """
    events = config.get("hooks") if isinstance(config, dict) and isinstance(config.get("hooks"), dict) else config
    if not isinstance(events, dict):
        return
    emitted = 0
    for event, groups in list(events.items())[:MAX_HOOK_EVENTS]:
        if not isinstance(groups, list):
            continue
        for group_index, group in enumerate(groups[:MAX_HOOK_GROUPS]):
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            for handler_index, handler in enumerate(group["hooks"]):
                if emitted >= MAX_HOOK_HANDLERS:
                    return
                emitted += 1
                yield str(event), group_index, group, handler_index, handler


def wrap_hook_handler(handler: Any, *, hook_id_value: str, event: str) -> Any:
    """Wrap one command handler with the hook census logger.

    Shell form becomes ``/bin/sh /skilleval/hook_census.sh <hook_id> <event> --
    '<original command>'``: the original command is one quoted argument, which
    the logger runs with ``/bin/sh -c`` exactly as the harness would. Exec form
    (``args``) stays exec form: ``command`` becomes ``/bin/sh`` and the logger,
    the ids, ``--``, the original program, and its args become the argument
    vector (a program with no args is passed as one quoted shell word).
    Non-command handlers (``prompt``, ``agent``, ``http``, ``mcp_tool``) are
    returned unchanged.
    """
    if not isinstance(handler, dict) or handler.get("type", "command") != "command":
        return handler
    command = handler.get("command")
    if not isinstance(command, str) or not command.strip():
        return handler
    wrapped = dict(handler)
    args = handler.get("args")
    if isinstance(args, list):
        argv = [command, *[str(arg) for arg in args]] if args else [shlex.quote(command)]
        wrapped["command"] = "/bin/sh"
        wrapped["args"] = [HOOK_CENSUS_SCRIPT, hook_id_value, event, "--", *argv]
        return wrapped
    wrapped["command"] = (
        f"/bin/sh {HOOK_CENSUS_SCRIPT} {shlex.quote(hook_id_value)} {shlex.quote(event)} -- "
        f"{shlex.quote(command.strip())}"
    )
    wrapped.pop("shell", None)
    return wrapped


@dataclass(frozen=True)
class WrappedHooks:
    """Merged, census-wrapped hooks plus the hook ids per source.

    ``dropped`` lists ``(source, event)`` pairs that were not staged because
    the source's event has no Claude Code equivalent.
    """

    config: dict[str, Any]
    ids: tuple[tuple[str, str, str], ...]  # (source, event, hook_id) for wrapped command handlers
    dropped: tuple[tuple[str, str], ...] = ()


#: Cursor hook events with a Claude Code equivalent: ``(Claude event, matcher or None)``.
#: Claude Code ignores unknown event names, so the other Cursor events (agent
#: response and thought, Tab, and workspace events, and the generic tool events,
#: whose matchers name Cursor tools) are not translated or staged.
CURSOR_TO_CLAUDE_HOOK_EVENTS: dict[str, tuple[str, str | None]] = {
    "sessionStart": ("SessionStart", None),
    "sessionEnd": ("SessionEnd", None),
    "beforeShellExecution": ("PreToolUse", "Bash"),
    "afterShellExecution": ("PostToolUse", "Bash"),
    "beforeMCPExecution": ("PreToolUse", "mcp__.*"),
    "afterMCPExecution": ("PostToolUse", "mcp__.*"),
    "beforeReadFile": ("PreToolUse", "Read"),
    "afterFileEdit": ("PostToolUse", "Edit|Write"),
    "beforeSubmitPrompt": ("UserPromptSubmit", None),
    "subagentStart": ("SubagentStart", None),
    "subagentStop": ("SubagentStop", None),
    "preCompact": ("PreCompact", None),
    "stop": ("Stop", None),
}
CLAUDE_PLUGIN_ROOT_VAR = "${CLAUDE_PLUGIN_ROOT}"
#: The install-time variables Claude Code expands when it loads a plugin.
CLAUDE_ROOT_VAR_NAMES = frozenset({"CLAUDE_PLUGIN_ROOT", "CLAUDE_PLUGIN_DATA"})
_ROOT_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LEADING_WORD_RE = re.compile(r"(\s*)(\S+)")
_RELATIVE_WORD_RE = re.compile(r"(^|\s)\./")
_PARENT_WORD_RE = re.compile(r"(^|\s)\.\./")
# A bare relative word (``scripts/x.sh``, ``check.py``) between shell word boundaries.
_BARE_WORD_RE = re.compile(r"(^|[\s;&|(])([A-Za-z0-9_][A-Za-z0-9_./-]*)(?=$|[\s;&|)])")


def plugin_root_var_names(root_prefixes: Sequence[str]) -> tuple[str, ...]:
    """The variable names that name the plugin root: a format's own placeholders plus Claude Code's.

    *root_prefixes* are the format's placeholders as written (``${PLUGIN_ROOT}``,
    ``${CURSOR_PLUGIN_ROOT}``).
    """
    names = {str(prefix).strip().removeprefix("$").removeprefix("{").removesuffix("}") for prefix in root_prefixes}
    return tuple(sorted({name for name in names if _ROOT_VAR_NAME_RE.fullmatch(name)} | CLAUDE_ROOT_VAR_NAMES))


@functools.lru_cache(maxsize=32)
def foreign_root_var_re(root_prefixes: tuple[str, ...]) -> re.Pattern[str] | None:
    """Match a format's own plugin-root placeholder that Claude Code does not expand, braced or bare.

    ``None`` when every placeholder in *root_prefixes* is one Claude Code expands.
    """
    foreign = [name for name in plugin_root_var_names(root_prefixes) if name not in CLAUDE_ROOT_VAR_NAMES]
    if not foreign:
        return None
    alternatives = "|".join(re.escape(name) for name in foreign)
    return re.compile(r"\$\{(?:" + alternatives + r")\}|\$(?:" + alternatives + r")\b")


def to_claude_root(value: Any, pattern: re.Pattern[str] | None) -> Any:
    """A string with each *pattern* placeholder rewritten to ``${CLAUDE_PLUGIN_ROOT}``; other values unchanged."""
    if pattern is None or not isinstance(value, str):
        return value
    return pattern.sub(lambda _match: CLAUDE_PLUGIN_ROOT_VAR, value)


#: The Cursor and Agent Plugins root placeholders, rewritten in every translated Cursor hook command.
_CURSOR_ROOT_VAR_RE = foreign_root_var_re(("${CURSOR_PLUGIN_ROOT}", "${PLUGIN_ROOT}"))
#: The plugin-root prefixes Claude Code expands in the paths a ``plugin.json`` declares.
CLAUDE_ROOT_PATH_PREFIXES = (CLAUDE_PLUGIN_ROOT_VAR, "$CLAUDE_PLUGIN_ROOT")


def declared_plugin_paths(value: Any) -> list[PurePosixPath]:
    """The root-relative paths a ``plugin.json`` path field declares (one path or a list of them).

    Each path is normalized like the Tier 1 inventory does
    (:func:`~skillevaluator.plugin_components.normalize_declared_path`): a path
    that escapes the plugin root (absolute, home-relative, or through ``..``),
    one under another placeholder, and the plugin root itself are skipped.
    """
    paths: list[PurePosixPath] = []
    for raw in value if isinstance(value, list) else [value]:
        if not isinstance(raw, str):
            continue
        rel = normalize_declared_path(raw, CLAUDE_ROOT_PATH_PREFIXES).rel
        if rel is not None and str(rel) != ".":
            paths.append(rel)
    return list(dict.fromkeys(paths))


def _plugin_file_checker(plugin_root: Path | None) -> Callable[[str], bool] | None:
    """A predicate for a relative command word that names a regular file inside *plugin_root*."""
    if plugin_root is None:
        return None
    reader = PluginRootReader(plugin_root)

    def _is_plugin_file(word: str) -> bool:
        rel = os.path.normpath(word)
        if rel.startswith(("/", "..")) or rel == ".":
            return False
        return reader.kind(PurePosixPath(rel)) == "file"

    return _is_plugin_file


def claude_root_command(
    command: str,
    *,
    foreign_root: re.Pattern[str] | None = None,
    plugin_file: Callable[[str], bool] | None = None,
) -> str:
    """Point a Cursor hook command at the staged plugin root.

    Claude Code runs hook commands in the session working directory and only
    expands ``${CLAUDE_PLUGIN_ROOT}``, so ``${CURSOR_PLUGIN_ROOT}`` (and the
    *foreign_root* placeholders) becomes ``${CLAUDE_PLUGIN_ROOT}``, and relative
    paths are rooted there: ``./scripts/x.sh``, ``python3 ./hooks/x.py``, a
    program such as ``scripts/x.sh``, and, with *plugin_file*, any bare word
    that names a plugin file (``sh scripts/x.sh``, ``python3 hooks/x.py``).
    """
    root = f'"{CLAUDE_PLUGIN_ROOT_VAR}"/'
    rewritten = to_claude_root(to_claude_root(command, _CURSOR_ROOT_VAR_RE), foreign_root)
    match = _LEADING_WORD_RE.match(rewritten)
    if match is not None:
        program = match.group(2)
        if "/" in program and not program.startswith(("/", "$", "~", '"', "'", ".")):
            rewritten = f"{match.group(1)}{root}{program}{rewritten[match.end() :]}"
    rewritten = _RELATIVE_WORD_RE.sub(lambda word: f"{word.group(1)}{root}", rewritten)
    rewritten = _PARENT_WORD_RE.sub(lambda word: f"{word.group(1)}{root}../", rewritten)
    if plugin_file is None:
        return rewritten

    def _bare(word: re.Match[str]) -> str:
        lead, value = word.groups()
        if {"/", "."} & set(value) and plugin_file(value):
            return f"{lead}{root}{value}"
        return word.group(0)

    return _BARE_WORD_RE.sub(_bare, rewritten)


def _claude_hook_handler(
    handler: Any,
    *,
    foreign_root: re.Pattern[str] | None = None,
    plugin_file: Callable[[str], bool] | None = None,
) -> Any:
    if not isinstance(handler, dict) or not isinstance(handler.get("command"), str):
        return handler
    command = claude_root_command(handler["command"], foreign_root=foreign_root, plugin_file=plugin_file)
    return handler if command == handler["command"] else {**handler, "command": command}


def _claude_root_vars(handler: Any, foreign_root: re.Pattern[str]) -> Any:
    """Rewrite a format's plugin-root placeholder to ``${CLAUDE_PLUGIN_ROOT}`` in a command handler."""
    if not isinstance(handler, dict) or not isinstance(handler.get("command"), str):
        return handler
    rewritten = {**handler, "command": to_claude_root(handler["command"], foreign_root)}
    if isinstance(handler.get("args"), list):
        rewritten["args"] = [to_claude_root(arg, foreign_root) for arg in handler["args"]]
    return handler if rewritten == handler else rewritten


def wrap_hook_sources(sources: Sequence[NativeHookSource], *, plugin_root: Path | None = None) -> WrappedHooks:
    """Merge every hooks source into one Claude Code ``hooks.json`` and wrap each command handler.

    Hook ids always use the source's own event names, so census rows join the
    Tier 1 ``hook_risk`` rows. A ``cursor`` source is translated: events with a
    Claude Code equivalent (:data:`CURSOR_TO_CLAUDE_HOOK_EVENTS`) get the Claude
    event and tool matcher, and relative commands (and, with *plugin_root*, any
    word naming a plugin file) are rooted at ``${CLAUDE_PLUGIN_ROOT}``; the other
    events are listed in ``dropped``. In every other source the format's own
    plugin-root placeholder (``${PLUGIN_ROOT}`` for Codex) becomes
    ``${CLAUDE_PLUGIN_ROOT}``, the only one Claude Code sets.
    """
    merged: dict[str, list[Any]] = {}
    ids: list[tuple[str, str, str]] = []
    dropped: list[tuple[str, str]] = []
    plugin_file = _plugin_file_checker(plugin_root)
    for source in sources:
        foreign_root = foreign_root_var_re(tuple(source.root_prefixes))
        rebuilt: dict[tuple[str, int], dict[str, Any]] = {}
        order: list[tuple[str, int]] = []
        targets: dict[tuple[str, int], str] = {}
        for event, group_index, group, handler_index, handler in iter_hook_handlers(source.config):
            fields = {k: v for k, v in group.items() if k != "hooks"}
            staged_event = event
            if source.dialect == "cursor":
                mapped = CURSOR_TO_CLAUDE_HOOK_EVENTS.get(event)
                if mapped is None:
                    if (source.name, event) not in dropped:
                        dropped.append((source.name, event))
                    continue
                staged_event, matcher = mapped
                fields = {"matcher": matcher} if matcher else {}
                handler = _claude_hook_handler(handler, foreign_root=foreign_root, plugin_file=plugin_file)
            elif foreign_root is not None:
                handler = _claude_root_vars(handler, foreign_root)
            key = (event, group_index)
            if key not in rebuilt:
                rebuilt[key] = fields | {"hooks": []}
                targets[key] = staged_event
                order.append(key)
            identifier = hook_id(source.name, event, group_index, handler_index)
            wrapped = wrap_hook_handler(handler, hook_id_value=identifier, event=staged_event)
            if wrapped is not handler:
                ids.append((source.name, staged_event, identifier))
            rebuilt[key]["hooks"].append(wrapped)
        for key in order:
            merged.setdefault(targets[key], []).append(rebuilt[key])
    return WrappedHooks(
        {
            "description": "Plugin hooks staged by SkillEvaluator; each command handler logs to the hook census.",
            "hooks": merged,
        },
        tuple(ids),
        tuple(dropped),
    )


# --------------------------------------------------------------------------- #
# Census script generation                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CensusCheck:
    """One post-setup listing check that the in-container census script runs.

    ``kind`` is ``file`` (a regular file exists), ``dir`` (a directory exists),
    ``contains`` (a file exists and contains ``needle``), or ``yaml_key`` (a
    YAML file has the exact key ``name`` directly under the top-level key
    ``needle``). ``target`` may use ``$HOME``, ``$CLAUDE_CONFIG_DIR``,
    ``$CODEX_HOME``, or ``$HERMES_HOME``; it is expanded in the agent's launch
    environment. ``requires_env`` lists ``(VAR, value)`` pairs the launch
    environment must have for the check to count (for example OpenCode's
    ``OPENCODE_CONFIG``). ``harness_name`` is the name the harness reports for
    the component when it differs from ``name`` (Claude Code names commands
    by file stem and skills by directory name).

    A passing check only proves the files are where the harness reads them, so
    the census records it as ``listed``, never as ``loaded``.
    """

    type: str
    name: str
    kind: str
    target: str
    needle: str = ""
    label: str = "file-listing"
    requires_env: tuple[tuple[str, str], ...] = ()
    harness_name: str = ""

    def declared(self) -> dict[str, str]:
        entry = {"type": self.type, "name": self.name}
        if self.harness_name and self.harness_name != self.name:
            entry["harness_name"] = self.harness_name
        return entry


# POSIX awk: exit 0 when the YAML read on stdin has the exact key ENVIRON["SKILLEVAL_KEY"]
# directly under the top-level key ENVIRON["SKILLEVAL_PARENT"] (plain, single-, or
# double-quoted keys; block style only). Never a substring match: a longer name, a
# value, or a nested key elsewhere does not count.
_YAML_KEY_AWK = r"""
function keyof(text,   colon, next_char, key) {
    colon = index(text, ":")
    if (colon == 0) return ""
    next_char = substr(text, colon + 1, 1)
    if (next_char != "" && next_char != " " && next_char != "\t") return ""
    key = substr(text, 1, colon - 1)
    if (length(key) >= 2 && ((substr(key, 1, 1) == "\"" && substr(key, length(key), 1) == "\"") ||
        (substr(key, 1, 1) == "'" && substr(key, length(key), 1) == "'"))) key = substr(key, 2, length(key) - 2)
    return key
}
BEGIN { parent = ENVIRON["SKILLEVAL_PARENT"]; wanted = ENVIRON["SKILLEVAL_KEY"]; inside = 0; child = -1; found = 0 }
{
    sub(/\r$/, "")
    if ($0 ~ /^[ ]*(#.*)?$/) next
    match($0, /^[ ]*/)
    depth = RLENGTH
    line = substr($0, depth + 1)
    if (depth == 0) { inside = (keyof(line) == parent); child = -1; next }
    if (!inside) next
    if (child < 0) child = depth
    if (depth == child && keyof(line) == wanted) found = 1
}
END { exit found ? 0 : 1 }
"""


def _json_text(value: str) -> str:
    return json.dumps(str(value)[:MAX_CENSUS_TEXT])


_ENV_PREFIX_RE = re.compile(r"^\$(HOME|CLAUDE_CONFIG_DIR|CODEX_HOME|HERMES_HOME)(/.*)?$")


def shell_path(target: str) -> str:
    """Quote a census target for sh: a leading ``$VAR`` expands, the rest is literal."""
    match = _ENV_PREFIX_RE.match(target)
    if match is None:
        return shlex.quote(target)
    rest = match.group(2) or ""
    return f'"${match.group(1)}"' + (shlex.quote(rest) if rest else "")


def render_setup_script(
    *,
    agent: str,
    setup_lines: Sequence[str],
    checks: Sequence[CensusCheck],
    not_loaded: Sequence[tuple[str, str, str]],
) -> str:
    """Render ``/skilleval/native/setup.sh``: stage per-run config, then write the load census.

    The script is POSIX sh, never exits non-zero (a failed step is reported in
    the census instead), refuses to write through links, and writes the census
    atomically to ``/logs/agent/skilleval-load-census.json``.
    """
    lines = [
        "#!/bin/sh",
        "# Generated by SkillEvaluator for native plugin loading. Do not edit.",
        f"# Agent: {agent}",
        "umask 022",
        "SKILLEVAL_SETUP_ERRORS=''",
        'skilleval_note() { SKILLEVAL_SETUP_ERRORS="$SKILLEVAL_SETUP_ERRORS $1"; }',
        "skilleval_esc() { printf '%s' \"$1\" | tr -d '\\000-\\037' | sed -e 's/\\\\/\\\\\\\\/g' -e 's/\"/\\\\\"/g'; }",
        "skilleval_copy() {",
        "  src=$1; dest=$2",
        '  [ -f "$src" ] || { skilleval_note "missing:$src"; return 1; }',
        '  if [ -L "$dest" ]; then skilleval_note "refused-link:$dest"; return 1; fi',
        '  mkdir -p "$(dirname "$dest")" 2>/dev/null || { skilleval_note "mkdir:$dest"; return 1; }',
        '  cp "$src" "$dest" 2>/dev/null || { skilleval_note "copy:$dest"; return 1; }',
        "}",
        "skilleval_append() {",
        "  src=$1; dest=$2",
        '  [ -f "$src" ] || { skilleval_note "missing:$src"; return 1; }',
        '  if [ -L "$dest" ]; then skilleval_note "refused-link:$dest"; return 1; fi',
        '  mkdir -p "$(dirname "$dest")" 2>/dev/null || { skilleval_note "mkdir:$dest"; return 1; }',
        '  { [ ! -s "$dest" ] || printf \'\\n\'; cat "$src"; } >> "$dest" 2>/dev/null || { skilleval_note "append:$dest"; return 1; }',
        "}",
        "# skilleval_yaml_key PARENT KEY FILE: FILE has the exact key KEY directly under the top-level key PARENT.",
        "skilleval_yaml_key() {",
        f'  SKILLEVAL_PARENT="$1" SKILLEVAL_KEY="$2" awk {shlex.quote(_YAML_KEY_AWK)} "$3"',
        "}",
        "",
        "# --- Per-run staging (runs after Harbor's own agent setup) ---",
        *setup_lines,
        "",
        "# --- Load census ---",
        "SKILLEVAL_LOADED=''",
        "SKILLEVAL_NOT_LOADED=''",
        "skilleval_add() {",
        '  if [ -n "$SKILLEVAL_LOADED" ]; then SKILLEVAL_LOADED="$SKILLEVAL_LOADED,"; fi',
        '  SKILLEVAL_LOADED="$SKILLEVAL_LOADED{\\"type\\":$1,\\"name\\":$2,\\"evidence\\":\\"$(skilleval_esc "$3")\\"}"',
        "}",
        "skilleval_miss() {",
        '  if [ -n "$SKILLEVAL_NOT_LOADED" ]; then SKILLEVAL_NOT_LOADED="$SKILLEVAL_NOT_LOADED,"; fi',
        '  SKILLEVAL_NOT_LOADED="$SKILLEVAL_NOT_LOADED{\\"type\\":$1,\\"name\\":$2,\\"reason\\":\\"$(skilleval_esc "$3")\\"}"',
        "}",
    ]
    for check in checks:
        type_json = shlex.quote(_json_text(check.type))
        name_json = shlex.quote(_json_text(check.name))
        target = shell_path(check.target)
        if check.kind == "dir":
            test = f"[ -d {target} ] && [ ! -L {target} ]"
        elif check.kind == "contains":
            test = f"[ -f {target} ] && [ ! -L {target} ] && grep -F -q -- {shlex.quote(check.needle)} {target}"
        elif check.kind == "yaml_key":
            test = (
                f"[ -f {target} ] && [ ! -L {target} ] && "
                f"skilleval_yaml_key {shlex.quote(check.needle)} {shlex.quote(check.name)} {target}"
            )
        else:
            test = f"[ -f {target} ] && [ ! -L {target} ]"
        label = shlex.quote(f"{check.label}: ")
        guards = ""
        for variable, value in check.requires_env:
            if not _ENV_NAME_RE.match(variable):
                raise ValueError(f"invalid census environment variable name: {variable!r}")
            reason = shlex.quote(f"the launch environment does not set {variable} to {value}"[:MAX_CENSUS_TEXT])
            guards += (
                f'if [ "${{{variable}-}}" != {shlex.quote(value)} ]; then '
                f"skilleval_miss {type_json} {name_json} {reason}; el"
            )
        lines.append(
            f"{guards}if {test} 2>/dev/null; then skilleval_add {type_json} {name_json} {label}{target}; "
            f"else skilleval_miss {type_json} {name_json} 'not found after setup: '{target}; fi"
        )
    for component_type, name, reason in not_loaded:
        lines.append(
            f"skilleval_miss {shlex.quote(_json_text(component_type))} {shlex.quote(_json_text(name))} "
            f"{shlex.quote(reason[:MAX_CENSUS_TEXT])}"
        )
    lines.extend(
        [
            "SKILLEVAL_CENSUS_DIR=/logs/agent",
            'mkdir -p "$SKILLEVAL_CENSUS_DIR" 2>/dev/null',
            f'SKILLEVAL_CENSUS="$SKILLEVAL_CENSUS_DIR/{LOAD_CENSUS_FILENAME}"',
            'if [ ! -L "$SKILLEVAL_CENSUS" ]; then',
            f'  printf \'{{"agent":{json.dumps(agent)},"mode":"native","{LISTED_KEY}":[%s],"not_loaded":[%s],'
            '"setup_errors":"%s"}\\n\' "$SKILLEVAL_LOADED" "$SKILLEVAL_NOT_LOADED" '
            '"$(skilleval_esc "$SKILLEVAL_SETUP_ERRORS")" > "$SKILLEVAL_CENSUS.tmp.$$" 2>/dev/null '
            '&& mv -f "$SKILLEVAL_CENSUS.tmp.$$" "$SKILLEVAL_CENSUS" 2>/dev/null',
            "fi",
            "exit 0",
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Harness adapters                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class NativeBundle:
    """What one adapter stages into a with-plugin task.

    ``generated`` maps a path relative to the bundle root (``/skilleval``) to
    generated text. ``trees`` lists ``(source dir, bundle-relative dest)``
    member-skill copies; ``plugin_tree`` asks the stager to securely copy the
    plugin root to a bundle-relative destination (Claude Code only).
    ``hook_ids`` maps each hooks source to the exact census ids of the command
    handlers that were wrapped; only those ids count as hook runs.
    ``harness`` tells the collector which harness report confirms loading (for
    Claude Code: its ``system/init`` event and the plugin name it reports).
    ``skill_aliases`` are the extra names the harness may report for the staged
    skills (Claude Code's ``<plugin>:<skill>``), which routing grades accept.
    """

    agent: str
    adapter: str
    components: dict[str, str]
    generated: dict[str, str] = field(default_factory=dict)
    trees: list[tuple[Path, str]] = field(default_factory=list)
    plugin_tree: str | None = None
    declared: list[dict[str, str]] = field(default_factory=list)
    hook_ids: dict[str, list[str]] = field(default_factory=dict)
    harness: dict[str, str] = field(default_factory=dict)
    skill_aliases: list[str] = field(default_factory=list)

    def census_plan(self) -> dict[str, Any]:
        """What the collector needs to read and check this arm's load census."""
        plan: dict[str, Any] = {
            "declared": [dict(item) for item in self.declared],
            "components": dict(self.components),
            "hook_ids": {source: list(ids) for source, ids in self.hook_ids.items()},
        }
        if self.harness:
            plan["harness"] = dict(self.harness)
        return plan


class HarnessAdapter:
    """Base class: one harness's native plugin layout."""

    agent = ""
    adapter_id = ""
    description = ""
    components: ClassVar[Mapping[str, str]] = {}
    #: Keep the generated wrapper SKILL.md in the with-plugin arm (rules stay ``wrapper``).
    stage_wrapper_skill = False
    #: Keep member skills in the task ``skills/`` projection (the harness's native skills dir).
    stage_member_skills = True
    #: Pass the plugin's MCP servers through Harbor's task MCP list.
    plugin_mcp_via_task = False
    #: Copy the plugin root into the task, so MCP servers that launch from plugin files can start.
    copies_plugin_tree = False
    #: Why ``--plugin-load auto`` uses the generated wrapper for this harness even
    #: where the adapter works (``None``: ``auto`` uses the adapter).
    auto_wrapper_reason: str | None = None

    def component_modes(self) -> dict[str, str]:
        return {
            component_type: self.components.get(component_type, "unsupported") for component_type in COMPONENT_TYPES
        }

    def unsupported_reason(self, *, env_mode: str, task_source: str) -> str | None:
        if env_mode == "local":
            return "local environment mode runs the agent on the host, and native staging needs a task container"
        if task_source != "evals_json":
            return "native Harbor task sources (evals/harbor/) keep their own environment"
        return None

    #: The harness reports plugin skills and commands as ``<plugin>:<name>``.
    namespaces_plugin_names = False

    def skill_namespace(self, source: NativePluginSource) -> str:
        """The ``<plugin>`` in the harness's ``<plugin>:<name>`` skill and command names; empty for bare names.

        The Harbor verifier strips exactly this prefix before it compares names.
        """
        return plugin_slug(source.plugin_name) if self.namespaces_plugin_names else ""

    def launch_env(self) -> dict[str, str]:
        return {}

    # -- staging ---------------------------------------------------------- #
    def build(self, source: NativePluginSource) -> NativeBundle:
        raise NotImplementedError

    def _base_bundle(self) -> NativeBundle:
        return NativeBundle(agent=self.agent, adapter=self.adapter_id, components=self.component_modes())

    def _unsupported_rows(self, source: NativePluginSource) -> list[tuple[str, str, str]]:
        modes = self.component_modes()
        rows: list[tuple[str, str, str]] = []
        reason = f"unsupported by the {self.agent} native adapter ({self.adapter_id}); not staged"
        if modes["hook"] == "unsupported":
            rows.extend(("hook", hook.name, reason) for hook in source.hooks)
        rows.extend((text.type, text.name, reason) for text in source.texts if modes.get(text.type) == "unsupported")
        rows.extend((component_type, name, reason) for component_type, name in source.other)
        if not self.copies_plugin_tree:
            # Only an adapter that copies the plugin tree can start a server that launches from plugin files.
            rows.extend(
                (
                    "mcp",
                    str(server.get("name") or ""),
                    f"launches from plugin files; not staged by the {self.agent} native adapter ({self.adapter_id})",
                )
                for server in source.plugin_file_mcp_servers
                if server.get("name")
            )
        return rows

    def _finish(
        self,
        bundle: NativeBundle,
        source: NativePluginSource,
        *,
        setup_lines: Sequence[str],
        checks: Sequence[CensusCheck],
        not_loaded: Sequence[tuple[str, str, str]] = (),
    ) -> NativeBundle:
        bundle.declared = [check.declared() for check in checks]
        bundle.generated["native/setup.sh"] = render_setup_script(
            agent=self.agent,
            setup_lines=setup_lines,
            checks=checks,
            not_loaded=[*self._unsupported_rows(source), *not_loaded],
        )
        return bundle


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().casefold() == "true")


def _rule_globs(scope: Any) -> list[str]:
    """Cursor ``globs`` or Claude ``paths``: a list, or a comma-separated string, of glob patterns."""
    items = scope if isinstance(scope, list) else str(scope).split(",")
    return [text for item in items if (text := str(item).strip())][:64]


@dataclass(frozen=True)
class _RuleActivation:
    """When a plugin rule applies, read from its frontmatter (which is never staged).

    ``body`` is the rule without its frontmatter (stripped when there was one).
    ``globs`` are the file patterns a scoped rule applies to. ``on_request``
    names a rule that is neither always on nor scoped: ``agent-requested`` (it
    has a description) or ``manual``.
    """

    body: str
    globs: tuple[str, ...] = ()
    on_request: str | None = None


def _rule_activation(name: str, content: str) -> _RuleActivation:
    """Classify one plugin rule as always on, scoped to file patterns, or applied on request.

    A rule with no frontmatter, or with ``alwaysApply: true``, is always on.
    Claude ``paths`` or Cursor ``globs`` (a list, or a comma-separated string)
    scope it to matching files. Otherwise a rule that sets ``alwaysApply``, or
    a Cursor ``.mdc`` rule, is agent-requested or manual; any other rule is
    always on.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---" or not any(line.strip() == "---" for line in lines[1:]):
        return _RuleActivation(content)
    parsed = parse_markdown(content)
    meta = parsed.frontmatter
    body = parsed.body.strip()
    if _is_true(meta.get("alwaysApply")):
        return _RuleActivation(body)
    globs = _rule_globs(meta.get("paths") or meta.get("globs") or [])
    if globs:
        return _RuleActivation(body, globs=tuple(globs))
    if "alwaysApply" in meta or name.casefold().endswith(".mdc"):
        return _RuleActivation(body, on_request="agent-requested" if parsed.description else "manual")
    return _RuleActivation(body)


def _always_on_rule(name: str, content: str) -> tuple[str | None, str | None]:
    """Return ``(body, None)`` for a rule that applies to every task, or ``(None, reason)``.

    For harnesses whose only rules channel is always on: a scoped,
    agent-requested, or manual rule is not staged.
    """
    rule = _rule_activation(name, content)
    if rule.globs:
        return None, (
            f"scoped rule (applies only to files matching {', '.join(rule.globs)[:80]}); this harness has only an "
            "always-on rules channel, so it is not staged"
        )
    if rule.on_request:
        return None, (
            f"{rule.on_request} rule (alwaysApply is not true); this harness has only an always-on rules channel, "
            "so it is not staged"
        )
    return rule.body.strip(), None


def _claude_user_rule(name: str, content: str) -> tuple[str | None, str | None]:
    """The staged Claude Code user rule for one plugin rule, or ``(None, reason)``.

    Claude Code loads a user rule on every task unless its frontmatter has
    ``paths``, so a scoped rule is staged with its patterns as ``paths``. A
    Cursor agent-requested or manual rule has no Claude Code equivalent and is
    not staged.
    """
    rule = _rule_activation(name, content)
    if rule.globs:
        listed = "".join(f"  - {json.dumps(pattern)}\n" for pattern in rule.globs)
        return f"---\npaths:\n{listed}---\n\n{rule.body}\n", None
    if rule.on_request:
        return None, (
            f"{rule.on_request} rule (alwaysApply is not true); Claude Code user rules are always on or scoped by "
            "paths, so it is not staged"
        )
    return rule.body.rstrip() + "\n", None


def _staged_rules(source: NativePluginSource) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Split the plugin rules into always-on ``(name, body)`` pairs and census ``not_loaded`` rows."""
    staged: list[tuple[str, str]] = []
    skipped: list[tuple[str, str, str]] = []
    for name, content in source.rules:
        body, reason = _always_on_rule(name, content)
        if body is None:
            skipped.append(("rule", name, reason or "not an always-on rule"))
        else:
            staged.append((name, body))
    return staged, skipped


def _rules_markdown(rules: Sequence[tuple[str, str]], *, heading: str) -> str:
    blocks = [f"# {heading}", ""]
    for name, body in rules:
        blocks.extend([f"## {name}", "", body.strip(), ""])
    return "\n".join(blocks).rstrip() + "\n"


def _skill_names(source: NativePluginSource) -> list[str]:
    return [path.name for path in source.member_skills]


def _mcp_entry_claude(server: Mapping[str, Any], declared: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One Claude Code ``.mcp.json`` entry: launch fields redacted like the wrapper TOML.

    Claude Code applies a server's ``env`` (stdio) and ``headers`` (http/sse), so
    the declared values are kept. They are ``${VAR}`` references or plain
    values; the blocking MCP checks already refuse inline credentials, and a
    value that still looks like a literal secret is refused here too.
    """
    name = str(server.get("name") or "")
    if server.get("command"):
        entry: dict[str, Any] = {"type": "stdio", "command": _redacted(server["command"])}
        if server.get("args"):
            entry["args"] = _redacted(list(server["args"]))
        applied = "env"
    else:
        transport = str(server.get("transport") or "http")
        entry = {"type": "sse" if transport == "sse" else "http", "url": _redacted(server["url"])}
        applied = "headers"
    values = (declared or {}).get(applied)
    if isinstance(values, Mapping) and values:
        kept: dict[str, str] = {}
        for key, value in values.items():
            text = str(value)
            if _redacted(text) != text:
                raise ValueError(
                    f"Refusing to stage MCP server '{name}' natively: {applied}.{key} looks like a literal "
                    "secret; reference an environment variable (${VAR}) instead"
                )
            kept[str(key)] = text
        entry[applied] = kept
    return entry


def _toml_string(value: str) -> str:
    """A TOML basic string. JSON escapes are not TOML: an emoji would become a surrogate pair."""
    try:
        return toml_quote(str(value))
    except ValueError as exc:
        raise PluginLoadError(f"plugin MCP value cannot be written as TOML: {exc}") from exc


def _redacted(value: Any) -> Any:
    from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line

    if isinstance(value, str):
        return redact_secrets_in_log_line(value)
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def _staged_handler_needle(config: Any) -> str | None:
    """A one-line ``"key": value`` fragment of the source's first non-``type`` handler field.

    The staged ``hooks.json`` is indent-2 JSON, so each scalar handler field
    renders on its own line exactly as ``json.dumps(key): json.dumps(value)``.
    """
    for _event, _group_index, _group, _handler_index, handler in iter_hook_handlers(config):
        if isinstance(handler, dict):
            for key, value in handler.items():
                if key != "type" and isinstance(value, str) and value.strip():
                    return f"{json.dumps(str(key))}: {json.dumps(value)}"
    return None


def _claude_rule_file_name(name: str) -> str:
    """The staged user-rule file for one plugin rule: the full rule name, ending in ``.md``.

    Keeping the full name keeps ``style.md`` and ``style.mdc`` apart.
    """
    file_name = safe_name(name)
    return file_name if file_name.lower().endswith(".md") else f"{file_name}.md"


def _staged_hook_events(hook: NativeHookSource) -> Any:
    """The part of a hook source Claude Code loads: a Cursor source keeps only translatable events."""
    if hook.dialect != "cursor":
        return hook.config
    config = hook.config
    events = config.get("hooks") if isinstance(config, dict) and isinstance(config.get("hooks"), dict) else config
    if not isinstance(events, dict):
        return None
    return {event: groups for event, groups in events.items() if event in CURSOR_TO_CLAUDE_HOOK_EVENTS}


def _redacted_servers(source: NativePluginSource) -> list[dict[str, Any]]:
    """Runnable servers with command/args/url redacted, exactly as the wrapper TOML does."""
    return [
        {
            key: _redacted(value)
            for key, value in server.items()
            if key in {"name", "command", "args", "url", "transport"}
        }
        for server in source.mcp_servers
    ]


class ClaudeCodeAdapter(HarnessAdapter):
    """Claude Code: a session plugin loaded with the documented ``--plugin-dir`` flag."""

    agent = "claude-code"
    adapter_id = "claude-code-plugin-dir"
    description = (
        "stages the plugin under /skilleval/native/claude-code/plugin and loads it with --plugin-dir; "
        "rules become user rules in $CLAUDE_CONFIG_DIR/rules/"
    )
    #: Plugin skills and commands run as ``<plugin>:<name>`` (``<plugin>`` is the staged manifest name).
    namespaces_plugin_names = True
    components: ClassVar[Mapping[str, str]] = {
        "skill": "native",
        "rule": "native",
        "mcp": "native",
        "hook": "native",
        "agent": "native",
        "command": "native",
        "output_style": "native",
        "lsp": "native",
        "settings": "native",
    }
    stage_member_skills = False
    #: The plugin root is copied, so MCP servers that launch from plugin files start here.
    copies_plugin_tree = True
    plugin_dir = f"{NATIVE_ROOT}/claude-code/plugin"

    def plugin_name(self, source: NativePluginSource) -> str:
        return plugin_slug(source.plugin_name)

    def manifest(self, source: NativePluginSource) -> dict[str, Any]:
        """The staged ``plugin.json``: identity plus the component keys that stay native.

        MCP servers move to the staged ``.mcp.json``, hooks to the census-wrapped
        ``hooks/hooks.json``, LSP servers to ``.lsp.json``, and the applied
        settings to ``settings.json``. ``userConfig`` stays, so Claude Code applies
        its defaults. Monitors, channels, and plugin dependencies are dropped.
        """
        manifest: dict[str, Any] = {"name": self.plugin_name(source), "description": source.description}
        if source.contained:
            for key in ("version", "skills", "commands", "agents", "outputStyles", "userConfig"):
                if key in source.manifest:
                    value = source.manifest[key]
                    manifest[key] = json.loads(json.dumps(value)) if isinstance(value, dict | list) else value
        return manifest

    def _declared_skill_dirs(self, source: NativePluginSource) -> list[PurePosixPath]:
        """Root-relative skill directories the staged ``plugin.json`` declares (it keeps a contained ``skills``)."""
        return declared_plugin_paths(source.manifest.get("skills")) if source.contained else []

    def staged_skills(self, source: NativePluginSource) -> list[tuple[str, str, Path | None]]:
        """Every skill the staged plugin loads: ``(census name, plugin-relative dir, copy source or None)``.

        Claude Code reads ``skills/<name>/SKILL.md`` plus the skill directories
        ``plugin.json`` declares. A member skill already in one of those places
        stays where the plugin copy puts it; any other member skill (outside the
        plugin root, or inside it but somewhere Claude Code does not read) is
        copied to ``skills/<name>``. A name that is already taken fails closed,
        because Claude Code would load the other skill under that name.
        """
        root = source.plugin_root.resolve()
        reader = PluginRootReader(root)
        loaded: list[str] = []
        for read_dir in (PurePosixPath("skills"), *self._declared_skill_dirs(source)):
            if reader.kind(read_dir) != "dir":
                continue
            if reader.kind(read_dir / "SKILL.md") == "file":
                loaded.append(read_dir.as_posix())
                continue
            try:
                with os.scandir(root / read_dir.as_posix()) as iterator:
                    names = sorted(entry.name for entry in iterator if not entry.name.startswith("."))
            except OSError:
                names = []
            for name in names[:MAX_CENSUS_ENTRIES]:
                child = read_dir / name
                if reader.kind(child) == "dir" and reader.kind(child / "SKILL.md") == "file":
                    loaded.append(child.as_posix())
        loaded = list(dict.fromkeys(loaded))
        # Claude Code names a plugin skill by its directory name, in any skill dir.
        loaded_names = {PurePosixPath(rel).name: rel for rel in loaded}
        staged: list[tuple[str, str, Path | None]] = []
        placed: set[str] = set()
        for skill in source.member_skills:
            resolved = skill.resolve()
            rel = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else None
            if rel is not None and rel in loaded:
                staged.append((skill.name, rel, None))
                placed.add(rel)
                continue
            dest = PurePosixPath("skills") / safe_name(skill.name)
            taken = loaded_names.get(dest.name)
            if taken is None and (dest.as_posix() in placed or reader.kind(dest) != "missing"):
                taken = dest.as_posix()
            if taken is not None:
                raise ValueError(
                    f"Refusing to stage member skill '{skill.name}' natively: the staged plugin already has "
                    f"{taken}, so Claude Code would load another skill under that name"
                )
            staged.append((skill.name, dest.as_posix(), skill))
            placed.add(dest.as_posix())
        staged.extend((PurePosixPath(rel).name, rel, None) for rel in loaded if rel not in placed)
        return staged

    def _unsupported_rows(self, source: NativePluginSource) -> list[tuple[str, str, str]]:
        staged = {("bin", "bin"), *(("lsp", name) for name in source.lsp_servers)}
        if source.settings:
            staged.add(("settings", _PLUGIN_SETTINGS_FILE))
        rows: list[tuple[str, str, str]] = []
        for component_type, name, reason in super()._unsupported_rows(source):
            if (component_type, name) in staged:
                continue
            if component_type == "settings":
                reason = (
                    "Claude Code applies only the agent and subagentStatusLine keys of the plugin's root "
                    "settings.json; nothing from this settings source is staged"
                )
            elif component_type == "lsp":
                reason = "no LSP server config with this name was found in its source; not staged"
            rows.append((component_type, name, reason))
        return rows

    def build(self, source: NativePluginSource) -> NativeBundle:
        bundle = self._base_bundle()
        base = "native/claude-code/plugin"
        slug = self.plugin_name(source)
        checks: list[CensusCheck] = []
        # The plugin root is copied (secure copy, evals/ and unsupported files
        # excluded) so hook scripts and other plugin-relative files resolve.
        bundle.plugin_tree = base
        manifest = self.manifest(source)
        # Census targets come from what the staged plugin actually loads. Claude Code
        # names each one ``<plugin>:<skill dir>``, member skill or not.
        for name, rel, copy_from in self.staged_skills(source):
            bundle.skill_aliases.append(f"{slug}:{PurePosixPath(rel).name}")
            if copy_from is not None:
                bundle.trees.append((copy_from, f"{base}/{rel}"))
            checks.append(
                CensusCheck(
                    "skill",
                    name,
                    "file",
                    f"{self.plugin_dir}/{rel}/SKILL.md",
                    label="plugin-dir listing",
                    # Claude Code names a plugin skill by its directory.
                    harness_name=PurePosixPath(rel).name,
                )
            )
        # Runnable servers plus the ones that launch from the copied plugin files
        # (Claude Code expands ${CLAUDE_PLUGIN_ROOT} for them), with their env/headers.
        servers = {
            _redacted(str(server["name"])): _mcp_entry_claude(server, source.mcp_declared.get(str(server["name"])))
            for server in (*source.mcp_servers, *source.plugin_file_mcp_servers)
        }
        bundle.generated[f"{base}/.mcp.json"] = json.dumps({"mcpServers": servers}, indent=2) + "\n"
        checks.extend(
            CensusCheck(
                "mcp",
                name,
                "contains",
                f"{self.plugin_dir}/.mcp.json",
                needle=json.dumps(name),
                label="plugin-dir config listing",
            )
            for name in servers
        )
        wrapped = wrap_hook_sources(source.hooks, plugin_root=source.plugin_root)
        bundle.generated[f"{base}/hooks/hooks.json"] = json.dumps(wrapped.config, indent=2) + "\n"
        for owner, _event, identifier in wrapped.ids:
            bundle.hook_ids.setdefault(owner, []).append(identifier)
        bundle.harness = {"kind": "claude-code-init", "plugin": slug}
        not_loaded: list[tuple[str, str, str]] = []
        for hook in source.hooks:
            identifiers = [identifier for owner, _event, identifier in wrapped.ids if owner == hook.name]
            dropped = [event for owner, event in wrapped.dropped if owner == hook.name]
            not_loaded.extend(
                (
                    "hook",
                    f"{hook.name}#{event}",
                    f"the Cursor hook event '{event}' is not translated to a Claude Code event; not staged",
                )
                for event in dropped
            )
            # Command handlers carry their hook id; a source with only other
            # handler types is found by its first staged handler field.
            if identifiers:
                needle: str | None = json.dumps(identifiers[0])[1:-1]
            else:
                needle = _staged_handler_needle(_staged_hook_events(hook))
            if needle is None:
                reason = (
                    "no hook handlers staged: none of its Cursor hook events is translated to a Claude Code event"
                    if dropped
                    else "no hook handlers staged: the config has no event matcher groups to load"
                )
                not_loaded.append(("hook", hook.name, reason))
                continue
            checks.append(
                CensusCheck(
                    "hook",
                    hook.name,
                    "contains",
                    f"{self.plugin_dir}/hooks/hooks.json",
                    needle=needle,
                    label="plugin-dir hooks listing",
                )
            )
        commands = manifest.get("commands")
        for text in source.texts:
            rel = text.rel
            if rel is None:
                # An inline command-map entry becomes a staged file the map points to.
                rel = f"commands/{safe_name(text.name)}.md"
                bundle.generated[f"{base}/{rel}"] = text.text
                if isinstance(commands, dict) and isinstance(commands.get(text.name), dict):
                    entry = {key: value for key, value in commands[text.name].items() if key != "content"}
                    entry["source"] = f"./{rel}"
                    commands[text.name] = entry
            # Claude Code names a file command by its stem and a ``commands`` map entry by its key.
            command_name = text.name if text.rel is None or isinstance(commands, dict) else PurePosixPath(rel).stem
            checks.append(
                CensusCheck(
                    text.type,
                    text.name,
                    "file",
                    f"{self.plugin_dir}/{rel}",
                    label="plugin-dir listing",
                    harness_name=command_name if text.type == "command" else "",
                )
            )
        bundle.generated[f"{base}/.claude-plugin/plugin.json"] = json.dumps(manifest, indent=2) + "\n"
        if source.settings:
            bundle.generated[f"{base}/{_PLUGIN_SETTINGS_FILE}"] = json.dumps(dict(source.settings), indent=2) + "\n"
            checks.append(
                CensusCheck(
                    "settings",
                    _PLUGIN_SETTINGS_FILE,
                    "file",
                    f"{self.plugin_dir}/{_PLUGIN_SETTINGS_FILE}",
                    label="plugin-dir listing",
                )
            )
        if source.lsp_servers:
            bundle.generated[f"{base}/.lsp.json"] = json.dumps(dict(source.lsp_servers), indent=2) + "\n"
            checks.extend(
                CensusCheck(
                    "lsp",
                    name,
                    "contains",
                    f"{self.plugin_dir}/.lsp.json",
                    needle=json.dumps(name),
                    label="plugin-dir config listing",
                )
                for name in source.lsp_servers
            )
        if ("bin", "bin") in source.other:
            checks.append(CensusCheck("bin", "bin", "dir", f"{self.plugin_dir}/bin", label="plugin-dir listing"))
        setup_lines: list[str] = []
        if source.rules:
            rules_dir = f"skilleval-{slug}"
            setup_lines.append(': "${CLAUDE_CONFIG_DIR:=$HOME/.claude}"')
            destinations: dict[str, str] = {}
            for name, content in source.rules:
                staged, reason = _claude_user_rule(name, content)
                if staged is None:
                    not_loaded.append(("rule", name, reason or "not a Claude Code user rule"))
                    continue
                file_name = _claude_rule_file_name(name)
                if file_name in destinations:
                    raise ValueError(
                        f"Refusing to stage plugin rules natively: '{destinations[file_name]}' and '{name}' would "
                        f"both be staged as rules/{file_name}"
                    )
                destinations[file_name] = name
                bundle.generated[f"native/claude-code/rules/{file_name}"] = staged
                dest = f"$CLAUDE_CONFIG_DIR/rules/{rules_dir}/{file_name}"
                setup_lines.append(f'skilleval_copy "{NATIVE_ROOT}/claude-code/rules/{file_name}" "{dest}"')
                checks.append(CensusCheck("rule", name, "file", dest, label="user rules listing"))
        return self._finish(bundle, source, setup_lines=setup_lines, checks=checks, not_loaded=not_loaded)


def _check_codex_mcp_toml(text: str, servers: Sequence[Mapping[str, Any]]) -> None:
    """Parse the generated Codex MCP tables now, so a bad config fails staging, not every Codex trial."""
    try:
        parsed = tomllib.loads(text).get("mcp_servers", {})
    except tomllib.TOMLDecodeError as exc:
        raise PluginLoadError(f"generated Codex MCP config is not valid TOML: {exc}") from exc
    for server in servers:
        entry = parsed.get(str(server["name"])) if isinstance(parsed, dict) else None
        if server.get("command"):
            expected: dict[str, Any] = {"command": str(server["command"])}
            if server.get("args"):
                expected["args"] = [str(arg) for arg in server["args"]]
        else:
            expected = {"url": str(server["url"])}
        if entry != expected:
            raise PluginLoadError(f"generated Codex MCP config does not round-trip for server {server['name']!r}")


class CodexAdapter(HarnessAdapter):
    """Codex: skills in ``~/.agents/skills``, MCP in ``$CODEX_HOME/config.toml``, rules in ``$CODEX_HOME/AGENTS.md``."""

    agent = "codex"
    adapter_id = "codex-home"
    description = (
        "member skills in the Codex skills directory, rules in $CODEX_HOME/AGENTS.md, and MCP servers as "
        "[mcp_servers.<name>] tables in $CODEX_HOME/config.toml"
    )
    components: ClassVar[Mapping[str, str]] = {"skill": "native", "rule": "native", "mcp": "native"}

    def build(self, source: NativePluginSource) -> NativeBundle:
        bundle = self._base_bundle()
        checks = [
            CensusCheck("skill", name, "file", f"$HOME/.agents/skills/{name}/SKILL.md", label="skills listing")
            for name in _skill_names(source)
        ]
        setup_lines = [': "${CODEX_HOME:=$HOME/.codex}"']
        rules, skipped_rules = _staged_rules(source)
        if rules:
            bundle.generated["native/codex/AGENTS.md"] = _rules_markdown(
                rules, heading=f"Plugin rules: {source.plugin_name}"
            )
            setup_lines.append(f'skilleval_append "{NATIVE_ROOT}/codex/AGENTS.md" "$CODEX_HOME/AGENTS.md"')
            checks.extend(
                CensusCheck(
                    "rule", name, "contains", "$CODEX_HOME/AGENTS.md", needle=f"## {name}", label="AGENTS.md listing"
                )
                for name, _content in rules
            )
        servers = _redacted_servers(source)
        if servers:
            lines: list[str] = ["# Plugin MCP servers staged by SkillEvaluator (native loading)."]
            for server in servers:
                lines.append(f"[mcp_servers.{_toml_string(server['name'])}]")
                if server.get("command"):
                    lines.append(f"command = {_toml_string(server['command'])}")
                    if server.get("args"):
                        lines.append("args = [" + ", ".join(_toml_string(arg) for arg in server["args"]) + "]")
                else:
                    lines.append(f"url = {_toml_string(server['url'])}")
                lines.append("")
            toml_text = "\n".join(lines)
            _check_codex_mcp_toml(toml_text, servers)
            bundle.generated["native/codex/mcp_servers.toml"] = toml_text
            setup_lines.append(f'skilleval_append "{NATIVE_ROOT}/codex/mcp_servers.toml" "$CODEX_HOME/config.toml"')
            checks.extend(
                CensusCheck(
                    "mcp",
                    server["name"],
                    "contains",
                    "$CODEX_HOME/config.toml",
                    needle=f"[mcp_servers.{_toml_string(server['name'])}]",
                    label="config.toml listing",
                )
                for server in servers
            )
        return self._finish(bundle, source, setup_lines=setup_lines, checks=checks, not_loaded=skipped_rules)


#: OpenCode's own agents (1.2.x). OpenCode has no plugin namespace: a staged agent
#: file with one of these names replaces the built-in, and forcing ``build`` to
#: ``mode: subagent`` makes ``opencode run`` fall back to the read-only ``plan`` agent.
OPENCODE_BUILTIN_AGENTS = frozenset({"build", "plan", "general", "explore", "compaction", "summary", "title"})


def opencode_agent_name(plugin_name: str, agent_name: str) -> str:
    """The name OpenCode knows a plugin agent by once it is staged natively.

    OpenCode has no plugin namespace, so an agent named like a built-in
    (``build``, ``plan``, ...) is staged as ``<plugin-slug>-<name>`` instead of
    replacing the built-in. Report-only activation coverage maps it back.
    """
    stem = safe_name(agent_name)
    if stem.casefold() in OPENCODE_BUILTIN_AGENTS:
        return safe_name(f"{plugin_slug(plugin_name)}-{stem}")
    return stem


#: Claude Code tool name (casefolded) -> OpenCode permission key. OpenCode's
#: ``edit`` permission covers its edit, write, and patch tools.
_OPENCODE_TOOL_PERMISSIONS = {
    "read": "read",
    "grep": "grep",
    "glob": "glob",
    "ls": "list",
    "bash": "bash",
    "edit": "edit",
    "multiedit": "edit",
    "write": "edit",
    "webfetch": "webfetch",
    "websearch": "websearch",
    "task": "task",
    "agent": "task",
    "todowrite": "todowrite",
    "todoread": "todoread",
    "skill": "skill",
}


def _tool_entries(value: Any) -> list[str]:
    """Claude ``tools``-style values: a YAML list or a comma list (commas inside ``(...)`` kept)."""
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    entries: list[str] = []
    depth = 0
    current = ""
    for char in value:
        if char == "," and depth == 0:
            entries.append(current)
            current = ""
            continue
        depth += {"(": 1, ")": -1}.get(char, 0)
        current += char
    entries.append(current)
    return [entry.strip() for entry in entries if entry.strip()]


def _bash_patterns(spec: str) -> list[str]:
    """OpenCode bash patterns for a Claude ``Bash(<spec>)`` rule (``prefix:*`` is a command prefix)."""
    spec = spec.strip()
    if spec.endswith(":*"):
        prefix = spec[:-2].rstrip()
        return [prefix, f"{prefix} *"] if prefix else ["*"]
    return [spec] if spec else []


def _opencode_permission(frontmatter: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Translate a Claude subagent's ``tools``/``disallowedTools`` into an OpenCode ``permission`` block.

    Returns ``(permission, narrowed, unenforceable)``. ``permission`` is ``None``
    when the subagent declares no limits (it inherits every tool, as in Claude
    Code). ``narrowed`` lists allowed tools with no OpenCode permission: they stay
    denied, so access only shrinks. ``unenforceable`` lists denied tools OpenCode
    cannot deny; the caller must not stage that agent, since it would widen access.
    """
    allowed_raw = next(
        (frontmatter[key] for key in ("tools", "allowed-tools", "allowedTools") if frontmatter.get(key) is not None),
        None,
    )
    denied_raw = next(
        (frontmatter[key] for key in ("disallowedTools", "disallowed-tools") if frontmatter.get(key) is not None),
        None,
    )
    allowed = _tool_entries(allowed_raw)
    denied = _tool_entries(denied_raw)
    allow_list = allowed_raw is not None and "*" not in allowed
    if not allow_list and not denied:
        return None, [], []
    tool_rules: dict[str, str] = {}
    denied_tools: set[str] = set()
    bash_patterns: dict[str, str] = {}
    narrowed: list[str] = []
    unenforceable: list[str] = []
    for entries, action, misses in (
        (allowed if allow_list else [], "allow", narrowed),
        (denied, "deny", unenforceable),
    ):
        for entry in entries:
            match = re.fullmatch(r"([A-Za-z]+)(?:\((.*)\))?", entry, flags=re.DOTALL)
            key = _OPENCODE_TOOL_PERMISSIONS.get(match.group(1).casefold()) if match else None
            scope = match.group(2) if match else None
            if key is None or (scope is not None and (key != "bash" or not _bash_patterns(scope))):
                misses.append(entry)
            elif scope is not None:
                # Denies come after allows, and OpenCode applies the last matching rule.
                bash_patterns.update(dict.fromkeys(_bash_patterns(scope), action))
            elif action == "deny":
                tool_rules[key] = "deny"
                denied_tools.add(key)
            else:
                tool_rules.setdefault(key, "allow")
    permission: dict[str, Any] = {"*": "deny"} if allow_list else {}
    permission.update(tool_rules)
    if bash_patterns and "bash" not in denied_tools:
        default = tool_rules.get("bash") or ("deny" if allow_list else "allow")
        permission["bash"] = {"*": default, **bash_patterns}
    return permission, narrowed, unenforceable


def _opencode_markdown(text: NativeTextComponent, permission: Mapping[str, Any] | None = None) -> str:
    parsed = parse_markdown(text.text)
    frontmatter: dict[str, Any] = {}
    if parsed.description:
        frontmatter["description"] = parsed.description
    if text.type == "agent":
        frontmatter["mode"] = "subagent"
        frontmatter.setdefault("description", f"Plugin subagent {text.name}")
        if permission is not None:
            frontmatter["permission"] = dict(permission)
    header = "\n".join(f"{key}: {json.dumps(value)}" for key, value in frontmatter.items())
    return f"---\n{header}\n---\n\n{parsed.body.strip()}\n" if header else parsed.body.strip() + "\n"


class OpenCodeAdapter(HarnessAdapter):
    """OpenCode: ``OPENCODE_CONFIG`` (mcp, instructions) and ``OPENCODE_CONFIG_DIR`` (agents, commands)."""

    agent = "opencode"
    adapter_id = "opencode-config"
    description = (
        "member skills in the OpenCode skills directory; OPENCODE_CONFIG carries the plugin mcp servers and "
        "an instructions file with the rules; OPENCODE_CONFIG_DIR carries agents/ and commands/"
    )
    components: ClassVar[Mapping[str, str]] = {
        "skill": "native",
        "rule": "native",
        "mcp": "native",
        "agent": "native",
        "command": "native",
    }
    config_file = f"{NATIVE_ROOT}/opencode/opencode.json"
    config_dir = f"{NATIVE_ROOT}/opencode/config"

    def launch_env(self) -> dict[str, str]:
        return {"OPENCODE_CONFIG": self.config_file, "OPENCODE_CONFIG_DIR": self.config_dir}

    def build(self, source: NativePluginSource) -> NativeBundle:
        bundle = self._base_bundle()
        checks = [
            CensusCheck("skill", name, "file", f"$HOME/.config/opencode/skills/{name}/SKILL.md", label="skills listing")
            for name in _skill_names(source)
        ]
        config: dict[str, Any] = {}
        rules, not_loaded = _staged_rules(source)
        if rules:
            bundle.generated["native/opencode/AGENTS.md"] = _rules_markdown(
                rules, heading=f"Plugin rules: {source.plugin_name}"
            )
            config["instructions"] = [f"{NATIVE_ROOT}/opencode/AGENTS.md"]
            checks.extend(
                CensusCheck(
                    "rule",
                    name,
                    "contains",
                    f"{NATIVE_ROOT}/opencode/AGENTS.md",
                    needle=f"## {name}",
                    label="OPENCODE_CONFIG instructions listing",
                    requires_env=(("OPENCODE_CONFIG", self.config_file),),
                )
                for name, _content in rules
            )
        servers = _redacted_servers(source)
        if servers:
            mcp: dict[str, Any] = {}
            for server in servers:
                if server.get("command"):
                    mcp[server["name"]] = {
                        "type": "local",
                        "command": [server["command"], *server.get("args", [])],
                        "enabled": True,
                    }
                else:
                    mcp[server["name"]] = {"type": "remote", "url": server["url"], "enabled": True}
            config["mcp"] = mcp
            checks.extend(
                CensusCheck(
                    "mcp",
                    server["name"],
                    "contains",
                    self.config_file,
                    needle=json.dumps(server["name"]),
                    label="OPENCODE_CONFIG listing",
                    requires_env=(("OPENCODE_CONFIG", self.config_file),),
                )
                for server in servers
            )
        bundle.generated["native/opencode/opencode.json"] = json.dumps(config, indent=2) + "\n"
        used: set[tuple[str, str]] = set()
        for text in source.texts:
            if text.type not in {"agent", "command"}:
                continue
            folder = "agents" if text.type == "agent" else "commands"
            stem = self.staged_name(source, text)
            notes: list[str] = []
            permission: dict[str, Any] | None = None
            if text.type == "agent":
                permission, narrowed, unenforceable = _opencode_permission(parse_markdown(text.text).frontmatter)
                if unenforceable:
                    not_loaded.append(
                        (
                            "agent",
                            text.name,
                            f"OpenCode cannot deny {', '.join(unenforceable)[:100]} (disallowedTools); not staged, "
                            "so the subagent never gets more tools than it declares",
                        )
                    )
                    continue
                if stem != safe_name(text.name):
                    notes.append(f"staged as {stem}: {safe_name(text.name)} is an OpenCode built-in agent")
                if narrowed:
                    notes.append(f"no OpenCode permission for {', '.join(narrowed)[:80]}, so it stays denied")
            if (folder, stem.casefold()) in used:
                not_loaded.append((text.type, text.name, f"another plugin {text.type} is already staged as {stem}"))
                continue
            used.add((folder, stem.casefold()))
            file_name = f"{stem}.md"
            bundle.generated[f"native/opencode/config/{folder}/{file_name}"] = _opencode_markdown(text, permission)
            label = "OPENCODE_CONFIG_DIR listing" + (f" ({'; '.join(notes)})" if notes else "")
            checks.append(
                CensusCheck(
                    text.type,
                    text.name,
                    "file",
                    f"{self.config_dir}/{folder}/{file_name}",
                    label=label,
                    requires_env=(("OPENCODE_CONFIG_DIR", self.config_dir),),
                )
            )
        return self._finish(bundle, source, setup_lines=[], checks=checks, not_loaded=not_loaded)

    def staged_name(self, source: NativePluginSource, text: NativeTextComponent) -> str:
        """The OpenCode file stem (and so the agent or command name) for a plugin component.

        OpenCode has no plugin namespace, so an agent named like a built-in
        (``build``, ``plan``, ...) is staged as ``<plugin-slug>-<name>`` instead
        of replacing the built-in.
        """
        if text.type == "agent":
            return opencode_agent_name(source.plugin_name, text.name)
        return safe_name(text.name)


class HermesAdapter(HarnessAdapter):
    """Hermes: the wrapper task plus a load census (Hermes has no native plugin path yet).

    The with-plugin task is the same as in wrapper mode: the generated wrapper
    skill with the rules, the member skills (Harbor copies them to
    ``$HERMES_HOME/skills``), and the plugin MCP servers through Harbor's task
    list (into ``$HERMES_HOME/config.yaml``). So every component is labeled
    ``wrapper``, and ``auto`` picks the wrapper.
    """

    agent = "hermes"
    adapter_id = "hermes-home"
    description = (
        "the wrapper task (wrapper skill with the rules, member skills in $HERMES_HOME/skills, MCP servers in "
        "$HERMES_HOME/config.yaml through Harbor's Hermes agent) plus the load census; Hermes has no native "
        "plugin path yet"
    )
    components: ClassVar[Mapping[str, str]] = {"skill": "wrapper", "rule": "wrapper", "mcp": "wrapper"}
    stage_wrapper_skill = True
    plugin_mcp_via_task = True
    auto_wrapper_reason = (
        "hermes has no native plugin path yet (the hermes-home adapter stages the wrapper task and adds only "
        "the load census)"
    )

    def build(self, source: NativePluginSource) -> NativeBundle:
        bundle = self._base_bundle()
        checks = [
            CensusCheck(
                "skill", name, "file", f"$HERMES_HOME/skills/{name}/SKILL.md", label="HERMES_HOME skills listing"
            )
            for name in _skill_names(source)
        ]
        checks.extend(
            CensusCheck(
                "mcp",
                server["name"],
                "yaml_key",
                "$HERMES_HOME/config.yaml",
                needle="mcp_servers",
                label="config.yaml mcp_servers key",
            )
            for server in _redacted_servers(source)
        )
        setup_lines = [': "${HERMES_HOME:=$HOME/.hermes}"']
        return self._finish(bundle, source, setup_lines=setup_lines, checks=checks)


HARNESS_ADAPTERS: dict[str, HarnessAdapter] = {
    adapter.agent: adapter for adapter in (ClaudeCodeAdapter(), CodexAdapter(), OpenCodeAdapter(), HermesAdapter())
}


def adapter_for(agent: str) -> HarnessAdapter | None:
    return HARNESS_ADAPTERS.get(agent)


# --------------------------------------------------------------------------- #
# Plan resolution and provenance                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentLoadDecision:
    """How one agent's with-plugin arm loads the plugin."""

    agent: str
    mode: str
    reason: str
    adapter: str
    components: Mapping[str, str]

    @property
    def native(self) -> bool:
        return self.mode == "native"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "adapter": self.adapter,
            "components": dict(self.components),
        }


def wrapper_decision(agent: str, reason: str) -> AgentLoadDecision:
    """A wrapper-mode decision with ``reason`` (the default, or an ``auto`` fallback)."""
    return AgentLoadDecision(agent, "wrapper", reason, WRAPPER_ADAPTER, dict(WRAPPER_COMPONENTS))


def refuse_or_fall_back(requested: str, agent: str, reason: str) -> AgentLoadDecision:
    """The decision for an agent that cannot load the plugin natively, for ``reason``.

    ``native`` refuses: it raises :class:`PluginLoadError`. ``auto`` falls back
    to the generated wrapper and records why.
    """
    if requested == "native":
        raise PluginLoadError(f"--plugin-load native is not supported for {agent}: {reason}")
    return wrapper_decision(agent, f"auto: {reason}; using the generated wrapper")


def apply_native_refusals(
    requested: str,
    decisions: Mapping[str, AgentLoadDecision],
    source: NativePluginSource | None,
) -> dict[str, AgentLoadDecision]:
    """Check each native decision against the prepared plugin snapshot.

    A native decision needs the snapshot, and an adapter refuses to stage a
    component type natively when that component enables a permission bypass
    (:func:`native_refusal`). Either way the agent is refused or falls back
    (:func:`refuse_or_fall_back`); wrapper decisions are kept as they are.
    """
    checked = dict(decisions)
    for agent, decision in decisions.items():
        if not decision.native:
            continue
        adapter = adapter_for(agent)
        if source is None:
            reason: str | None = "no native plugin snapshot was prepared for this run"
        else:
            reason = native_refusal(adapter, source) if adapter is not None else None
        if reason is not None:
            checked[agent] = refuse_or_fall_back(requested, agent, reason)
    return checked


def resolve_plugin_load(
    requested: str,
    agents: Sequence[str],
    *,
    env_mode: str,
    task_source: str = "evals_json",
) -> dict[str, AgentLoadDecision]:
    """Decide native or wrapper loading per agent.

    ``wrapper`` keeps today's behavior. ``auto`` uses the native adapter when
    it supports the agent and environment and loads the plugin natively (not
    Hermes, see :attr:`HarnessAdapter.auto_wrapper_reason`), and the wrapper
    otherwise, with the reason recorded. ``native`` requires native loading and raises
    :class:`PluginLoadError` for any agent it cannot honor.
    """
    if requested not in PLUGIN_LOAD_CHOICES:
        raise PluginLoadError(f"--plugin-load must be one of: {', '.join(PLUGIN_LOAD_CHOICES)}")
    decisions: dict[str, AgentLoadDecision] = {}
    for agent in agents:
        if requested == "wrapper":
            decisions[agent] = wrapper_decision(
                agent, "wrapper requested (default): generated wrapper skill with the plugin rules embedded"
            )
            continue
        adapter = adapter_for(agent)
        reason = (
            f"no native plugin adapter exists for {agent}"
            if adapter is None
            else adapter.unsupported_reason(env_mode=env_mode, task_source=task_source)
        )
        if reason is not None:
            decisions[agent] = refuse_or_fall_back(requested, agent, reason)
            continue
        assert adapter is not None
        if requested == "auto" and adapter.auto_wrapper_reason:
            decisions[agent] = wrapper_decision(
                agent, f"auto: {adapter.auto_wrapper_reason}; using the generated wrapper"
            )
            continue
        decisions[agent] = AgentLoadDecision(
            agent,
            "native",
            f"{requested}: {adapter.description}",
            adapter.adapter_id,
            adapter.component_modes(),
        )
    return decisions


_NATIVE_AGENTS = "skillevaluator.tier3.harbor.native_agents"
_LOCAL_AGENTS = "skillevaluator.tier3.harbor.local_agents"
#: (agent, base Harbor import path or ``None`` for the stock agent) -> native wrapper import path.
NATIVE_AGENT_IMPORT_PATHS: dict[tuple[str, str | None], str] = {
    ("claude-code", None): f"{_NATIVE_AGENTS}:NativeClaudeCode",
    (
        "claude-code",
        f"{_LOCAL_AGENTS}:SkillEvaluatorNvidiaBuildClaudeCode",
    ): f"{_NATIVE_AGENTS}:NativeNvidiaBuildClaudeCode",
    ("codex", None): f"{_NATIVE_AGENTS}:NativeCodex",
    ("codex", f"{_LOCAL_AGENTS}:SkillEvaluatorGatewayCodex"): f"{_NATIVE_AGENTS}:NativeGatewayCodex",
    ("codex", f"{_LOCAL_AGENTS}:SkillEvaluatorNvidiaBuildCodex"): f"{_NATIVE_AGENTS}:NativeNvidiaBuildCodex",
    ("opencode", None): f"{_NATIVE_AGENTS}:NativeOpenCode",
    ("opencode", f"{_LOCAL_AGENTS}:SkillEvaluatorGatewayOpenCode"): f"{_NATIVE_AGENTS}:NativeGatewayOpenCode",
    ("hermes", None): f"{_NATIVE_AGENTS}:NativeHermes",
}


def native_agent_import_path(agent: str, base_import_path: str | None) -> str:
    """Return the native Harbor wrapper for ``agent`` over its routing wrapper, or raise ``PluginLoadError``."""
    try:
        return NATIVE_AGENT_IMPORT_PATHS[(agent, base_import_path)]
    except KeyError:
        raise PluginLoadError(
            f"native plugin loading has no Harbor wrapper for {agent} over {base_import_path or 'the stock agent'}"
        ) from None


def plugin_load_provenance(requested: str, decisions: Mapping[str, AgentLoadDecision]) -> dict[str, Any]:
    """The contract's ``plugin_load`` provenance block."""
    return {
        "requested": requested,
        "by_agent": {agent: decision.to_dict() for agent, decision in decisions.items()},
    }


# --------------------------------------------------------------------------- #
# Load census: parse, fall back, summarize, and map to coverage               #
# --------------------------------------------------------------------------- #
def _bounded_text(value: Any) -> str:
    return str(value)[:MAX_CENSUS_TEXT] if value is not None else ""


def _bounded_detail(value: Any) -> str:
    """Census evidence or reason text of at most ``MAX_CENSUS_TEXT`` characters.

    Unlike names and ids (cut at the end, so the same id always reads the
    same), a long detail is cut in the middle, so a listed path keeps its file
    name (``.../skills/<name>/SKILL.md``).
    """
    if value is None:
        return ""
    text = str(value)
    if len(text) <= MAX_CENSUS_TEXT:
        return text
    head = (MAX_CENSUS_TEXT - 1) // 3
    return f"{text[:head]}\u2026{text[-(MAX_CENSUS_TEXT - 1 - head) :]}"


#: Census lists from strongest to weakest evidence, with each entry's detail field.
_CENSUS_LISTS: tuple[tuple[str, str], ...] = (
    ("loaded", "evidence"),
    (LISTED_KEY, "evidence"),
    ("staged", "evidence"),
    ("not_loaded", "reason"),
)
_EVIDENCE_STRENGTH = {"loaded": 3, LISTED_KEY: 2, "staged": 1}
#: Component types Claude Code's ``system/init`` event can confirm, and the
#: types that live inside the plugin (rules are user rules, outside it).
_CLAUDE_INIT_TYPES = frozenset({"skill", "agent", "command", "mcp"})
_CLAUDE_PLUGIN_TYPES = frozenset({"skill", "agent", "command", "mcp", "hook", "output_style"})
CLAUDE_INIT_SOURCE = "claude-code system/init event"


def normalize_census(raw: Any) -> dict[str, Any] | None:
    """Validate a census object's shape; ``None`` when it is not a census.

    Keeps the ``loaded``, ``listed``, ``staged``, and ``not_loaded`` lists
    (bounded, strings only). ``loaded``, ``listed``, and ``not_loaded`` are
    always present in the result.
    """
    if not isinstance(raw, dict):
        return None
    mode = raw.get("mode")
    if mode not in {"native", "wrapper"} or not isinstance(raw.get("agent"), str):
        return None
    census: dict[str, Any] = {"agent": _bounded_text(raw["agent"]), "mode": mode}
    for key, detail in _CENSUS_LISTS:
        entries = raw.get(key, [])
        if not isinstance(entries, list):
            return None
        parsed: list[dict[str, str]] = []
        for entry in entries[:MAX_CENSUS_ENTRIES]:
            if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
                continue
            parsed.append(
                {
                    "type": _bounded_text(entry.get("type")),
                    "name": _bounded_text(entry.get("name")),
                    detail: _bounded_detail(entry.get(detail)),
                }
            )
        if parsed or key != "staged":
            census[key] = parsed
    if isinstance(raw.get("setup_errors"), str) and raw["setup_errors"].strip():
        census["setup_errors"] = _bounded_detail(raw["setup_errors"].strip())
    return census


def read_census_file(path: Path) -> dict[str, Any] | None:
    """Read one trial's census with a bounded, no-follow, regular-file read.

    The file is written inside the sandbox, so it can only *list* components:
    any ``loaded`` entry in it is read as ``listed``. Harness evidence is added
    on the host (see :func:`apply_claude_init_evidence`).
    """
    from skillevaluator.utils.secure_fs import secure_read_path_text

    try:
        text = secure_read_path_text(path, MAX_CENSUS_BYTES)
    except (SecurePathError, OSError, ValueError):
        return None
    try:
        raw = load_bounded_json(text)
    except (StructuredDataError, ValueError):
        return None
    census = normalize_census(raw)
    if census is None:
        return None
    census[LISTED_KEY] = [*census.pop("loaded"), *census[LISTED_KEY]][:MAX_CENSUS_ENTRIES]
    census["loaded"] = []
    census.pop("staged", None)
    return census


def _declared_keys(declared: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """``(type, name) -> harness name`` for every declared component."""
    keys: dict[tuple[str, str], str] = {}
    for item in declared:
        if not isinstance(item, Mapping):
            continue
        key = (str(item.get("type") or ""), str(item.get("name") or ""))
        if all(key):
            keys[key] = str(item.get("harness_name") or key[1])
    return keys


def restrict_census(census: Mapping[str, Any], declared: Sequence[Mapping[str, Any]], *, agent: str) -> dict[str, Any]:
    """Keep only census entries for components SkillEvaluator staged for ``agent``.

    The census file is writable from inside the sandbox. An entry for a
    component that was never staged (another type, another name) is moved to
    ``not_loaded`` as ignored, and counted in ``ignored_entries``.
    """
    allowed = _declared_keys(declared)
    result: dict[str, Any] = {
        key: value for key, value in census.items() if key not in {name for name, _detail in _CENSUS_LISTS}
    }
    result["agent"] = agent
    seen: set[tuple[str, str]] = set()
    ignored: list[dict[str, str]] = []
    for key, _detail in _CENSUS_LISTS[:-1]:
        kept: list[dict[str, str]] = []
        for entry in census.get(key) or ():
            item = (str(entry.get("type") or ""), str(entry.get("name") or ""))
            if item not in allowed:
                ignored.append(
                    {
                        "type": item[0],
                        "name": item[1],
                        "reason": f"ignored census entry: SkillEvaluator did not stage this component for {agent}",
                    }
                )
                continue
            if item in seen:
                continue
            seen.add(item)
            kept.append(dict(entry))
        if kept or key != "staged":
            result[key] = kept
    not_loaded = [dict(entry) for entry in census.get("not_loaded") or () if isinstance(entry, Mapping)]
    result["not_loaded"] = [*not_loaded, *ignored][:MAX_CENSUS_ENTRIES]
    if ignored:
        result["ignored_entries"] = len(ignored)
    return result


def read_harness_log_prefix(path: Path, max_bytes: int = MAX_HARNESS_LOG_PREFIX_BYTES) -> str | None:
    """Read at most ``max_bytes`` from the start of a harness log without following links.

    Harness logs can be large; the startup event this module needs is near the
    top, so a prefix is enough. ``None`` for a missing, linked, or special file,
    including one whose parent is not a directory; ``SecurePathError`` when the
    path cannot be inspected for another reason, such as a permission error.
    """
    try:
        metadata = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise SecurePathError("path_access_error", f"Cannot inspect path safely: {path.name}: {exc}") from exc
    if stat_is_link_or_reparse(metadata):
        return None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        chunks: list[bytes] = []
        remaining = max_bytes
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    return b"".join(chunks).decode("utf-8", errors="replace")


def claude_init_event(text: str) -> dict[str, Any] | None:
    """The first ``{"type": "system", "subtype": "init"}`` event of Claude Code's stream-json output."""
    for index, line in enumerate(text.splitlines()):
        if index >= 5_000:
            break
        stripped = line.strip()
        if not stripped.startswith("{") or '"init"' not in stripped:
            continue
        try:
            event = json.loads(stripped)
        except (ValueError, RecursionError):
            continue
        if isinstance(event, dict) and event.get("type") == "system" and event.get("subtype") == "init":
            return event
    return None


def _string_set(value: Any) -> set[str]:
    return {item for item in value if isinstance(item, str)} if isinstance(value, list) else set()


def apply_claude_init_evidence(
    census: Mapping[str, Any],
    declared: Sequence[Mapping[str, Any]],
    init: Mapping[str, Any],
    *,
    plugin: str,
) -> dict[str, Any]:
    """Confirm or refute the census with Claude Code's own ``system/init`` report.

    A plugin skill, subagent, or command is ``loaded`` only when init lists
    ``<plugin>:<name>``; a plugin MCP server only when init reports
    ``plugin:<plugin>:<server>`` with status ``connected`` (``failed`` and
    ``pending`` become ``not_loaded`` with that status). When init does not
    list the plugin at all, every plugin component is ``not_loaded``. Hooks,
    output styles, and rules are not in the init event and keep their listing.
    """
    result = {
        key: [dict(entry) for entry in value] if isinstance(value, list) else value for key, value in census.items()
    }
    plugins = {
        str(entry.get("name"))
        for entry in init.get("plugins") or ()
        if isinstance(entry, Mapping) and entry.get("name")
    }
    plugin_loaded = plugin in plugins
    names = {
        "skill": _string_set(init.get("skills")) | _string_set(init.get("slash_commands")),
        "agent": _string_set(init.get("agents")),
        "command": _string_set(init.get("slash_commands")),
    }
    mcp_status: dict[str, str] = {}
    for entry in init.get("mcp_servers") or ():
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            mcp_status[entry["name"]] = _bounded_text(entry.get("status") or "unknown")
    decided: dict[tuple[str, str], tuple[str, str]] = {}
    for (component_type, name), harness_name in _declared_keys(declared).items():
        if component_type not in _CLAUDE_PLUGIN_TYPES:
            continue
        if not plugin_loaded:
            decided[(component_type, name)] = (
                "not_loaded",
                f"claude-code did not load plugin {plugin} (not listed in its init event)",
            )
            continue
        if component_type not in _CLAUDE_INIT_TYPES:
            continue
        if component_type == "mcp":
            server = f"plugin:{plugin}:{harness_name}"
            status = mcp_status.get(server)
            if status == "connected":
                decided[(component_type, name)] = ("loaded", f"{CLAUDE_INIT_SOURCE}: {server} connected")
            elif status is None:
                decided[(component_type, name)] = ("not_loaded", f"not reported by claude-code init (no {server})")
            else:
                decided[(component_type, name)] = (
                    "not_loaded",
                    f"claude-code init reported MCP server {server} with status {status}",
                )
            continue
        full_name = f"{plugin}:{harness_name}"
        if full_name in names[component_type]:
            decided[(component_type, name)] = ("loaded", f"{CLAUDE_INIT_SOURCE}: {full_name}")
        else:
            decided[(component_type, name)] = ("not_loaded", f"not reported by claude-code init (no {full_name})")
    for key, _detail in _CENSUS_LISTS:
        entries = result.get(key)
        if isinstance(entries, list):
            result[key] = [
                entry for entry in entries if (str(entry.get("type")), str(entry.get("name"))) not in decided
            ]
    result.setdefault("loaded", [])
    result.setdefault("not_loaded", [])
    for (component_type, name), (state, detail) in decided.items():
        field_name = "evidence" if state == "loaded" else "reason"
        result[state].append({"type": component_type, "name": name, field_name: _bounded_detail(detail)})
    result["harness"] = CLAUDE_INIT_SOURCE
    if not plugin_loaded:
        result["plugin_missing"] = True
    return result


def fallback_census(agent: str, mode: str, declared: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """The declared-staged census used when no in-container census exists."""
    return {
        "agent": agent,
        "mode": mode,
        "loaded": [],
        LISTED_KEY: [],
        "staged": [
            {"type": str(item.get("type", "")), "name": str(item.get("name", "")), "evidence": STAGED_EVIDENCE}
            for item in declared
        ],
        "not_loaded": [],
        "fallback": True,
    }


def summarize_censuses(
    agent: str,
    mode: str,
    censuses: Sequence[Mapping[str, Any]],
    *,
    staged_hook_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Per-agent summary over every trial's census.

    A component is ``loaded`` only when the harness confirmed it in every
    trial, ``listed`` when every trial at least listed it, ``staged`` when some
    trial had no census at all (only the staging plan), and ``not_loaded``
    otherwise. ``staged_hook_ids`` (the exact wrapped hook ids per source) is
    kept so runtime evidence can count only hooks that were really staged.
    """
    trials = len(censuses)
    strengths: dict[tuple[str, str], list[int]] = {}
    details: dict[tuple[str, str], dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    for index, census in enumerate(censuses):
        for key, detail in _CENSUS_LISTS:
            for entry in census.get(key) or ():
                if not isinstance(entry, Mapping):
                    continue
                item = (str(entry.get("type")), str(entry.get("name")))
                if item not in strengths:
                    strengths[item] = [0] * trials
                    details[item] = {}
                    order.append(item)
                strength = _EVIDENCE_STRENGTH.get(key, 0)
                strengths[item][index] = max(strengths[item][index], strength)
                text_value = str(entry.get(detail) or "")
                if text_value and key not in details[item]:
                    details[item][key] = text_value
    lists: dict[str, list[dict[str, str]]] = {"loaded": [], LISTED_KEY: [], "staged": [], "not_loaded": []}
    for item in order:
        weakest = min(strengths[item]) if trials else 0
        found = sum(1 for value in strengths[item] if value >= _EVIDENCE_STRENGTH[LISTED_KEY])
        entry = {"type": item[0], "name": item[1]}
        if weakest >= _EVIDENCE_STRENGTH["loaded"]:
            lists["loaded"].append({**entry, "evidence": details[item].get("loaded", "")})
        elif weakest >= _EVIDENCE_STRENGTH[LISTED_KEY]:
            lists[LISTED_KEY].append({**entry, "evidence": details[item].get(LISTED_KEY, "")})
        elif weakest >= _EVIDENCE_STRENGTH["staged"]:
            lists["staged"].append({**entry, "evidence": STAGED_EVIDENCE})
        else:
            reason = details[item].get("not_loaded") or f"listed in {found} of {trials} trial census(es)"
            lists["not_loaded"].append({**entry, "reason": reason})
    summary: dict[str, Any] = {
        "agent": agent,
        "mode": mode,
        "trials": trials,
        "fallback_trials": sum(1 for census in censuses if census.get("fallback")),
        "harness_trials": sum(1 for census in censuses if census.get("harness")),
        **{key: value[:MAX_CENSUS_ENTRIES] for key, value in lists.items()},
    }
    harness = next((str(census["harness"]) for census in censuses if census.get("harness")), "")
    if harness:
        summary["harness"] = harness
        summary["plugin_missing_trials"] = sum(1 for census in censuses if census.get("plugin_missing"))
    ignored = sum(int(census.get("ignored_entries") or 0) for census in censuses)
    if ignored:
        summary["ignored_entries"] = ignored
    if staged_hook_ids:
        summary["staged_hook_ids"] = {
            _bounded_text(source): [_bounded_text(identifier) for identifier in list(ids)[:MAX_HOOK_HANDLERS]]
            for source, ids in list(staged_hook_ids.items())[:MAX_CENSUS_ENTRIES]
            if isinstance(ids, list | tuple)
        }
    return summary


def native_load_unverified(summary: Mapping[str, Any]) -> str | None:
    """Why a native arm's plugin load was never confirmed in any trial, or ``None``.

    Every trial falling back to the staging plan means ``setup.sh`` never ran,
    so the launch was not rewritten and the plugin may not have been loaded.
    """
    if summary.get("mode") != "native":
        return None
    trials = summary.get("trials")
    if not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0:
        return None
    agent = _bounded_text(summary.get("agent") or "the agent")
    if summary.get("fallback_trials") == trials:
        return (
            f"{agent}: no load census in any of {trials} with-plugin trial(s); the native setup never ran, "
            "so the plugin may not have been loaded"
        )
    if summary.get("plugin_missing_trials") == trials:
        return f"{agent}: the harness did not load the plugin in any of {trials} with-plugin trial(s)"
    return None


#: Coverage states in which the with-plugin arm had the component, from the weakest evidence to the
#: strongest: staged for it, loaded by the harness, exercised at runtime. A row is never downgraded.
COVERAGE_STATE_RANK = {"staged": 1, "loaded": 2, "exercised": 3}
EVALUATED_COVERAGE_STATES = frozenset(COVERAGE_STATE_RANK)


def _coverage_names(row: Mapping[str, Any]) -> set[str]:
    names = {str(row.get("name") or "")}
    path = row.get("path")
    if isinstance(path, str) and path:
        names.update({path, PurePosixPath(path).name, PurePosixPath(path).stem, path.removeprefix("rules/")})
        if row.get("type") == "skill":
            names.add(PurePosixPath(path).parent.name)
    return {name for name in names if name}


def native_types_by_agent(plugin_load: Any) -> dict[str, set[str]]:
    """The component types each native with-plugin arm loads natively, from the ``plugin_load`` provenance."""
    by_agent = plugin_load.get("by_agent") if isinstance(plugin_load, Mapping) else None
    result: dict[str, set[str]] = {}
    for agent, entry in (by_agent or {}).items() if isinstance(by_agent, Mapping) else ():
        components = entry.get("components") if isinstance(entry, Mapping) else None
        if isinstance(entry, Mapping) and entry.get("mode") == "native" and isinstance(components, Mapping):
            result[str(agent)] = {str(kind) for kind, mode in components.items() if mode == "native"}
    return result


def apply_load_census(
    coverage: Mapping[str, Any] | None,
    summaries: Mapping[str, Mapping[str, Any]],
    plugin_load: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fold native load-census evidence into the coverage rows.

    Only component types ``plugin_load`` reports as ``native`` for that agent
    count; anything else in a census was not staged natively and is ignored.
    Harness evidence (``loaded``) promotes a row to ``loaded``. A listing alone
    (``listed``) promotes a row to ``staged`` at most. A ``not_loaded`` entry
    adds its reason. Precedence is ``exercised`` > ``loaded`` > ``staged``: a
    row is never downgraded. Notes are appended to the existing reason, so an
    earlier note (for example an INCOMPLETE MCP note) survives, also on a row
    that stays unsupported; a row promoted from an unevaluated state gets the
    census note in place of the old "not staged" reason.
    """
    if not isinstance(coverage, Mapping):
        return None if coverage is None else dict(coverage)
    native_types = native_types_by_agent(plugin_load)
    hits: dict[str, dict[tuple[str, str], list[tuple[str, str]]]] = {"loaded": {}, LISTED_KEY: {}, "not_loaded": {}}
    for agent, summary in summaries.items():
        types = native_types.get(str(agent), set())
        if summary.get("mode") != "native" or not types:
            continue
        for key, detail in (("loaded", "evidence"), (LISTED_KEY, "evidence"), ("not_loaded", "reason")):
            for entry in summary.get(key) or ():
                if not isinstance(entry, Mapping) or str(entry.get("type")) not in types:
                    continue
                item = (str(entry.get("type")), str(entry.get("name")))
                hits[key].setdefault(item, []).append((str(agent), str(entry.get(detail) or "")))
    rows: list[dict[str, Any]] = []
    for raw in coverage.get("components", []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        state = str(row.get("state") or "")
        if state in {"invalid", "unavailable", "exercised"}:
            rows.append(row)
            continue

        def lookup(key: str, row: dict[str, Any] = row) -> list[tuple[str, str]]:
            return next(
                (
                    hits[key][(str(row.get("type")), name)]
                    for name in _coverage_names(row)
                    if (str(row.get("type")), name) in hits[key]
                ),
                [],
            )

        new_state = state
        note = ""
        if found := lookup("loaded"):
            agents = ", ".join(sorted({agent for agent, _detail in found}))
            new_state = "loaded" if COVERAGE_STATE_RANK.get(state, 0) < COVERAGE_STATE_RANK["loaded"] else state
            note = f"loaded natively by {agents} (load census, harness evidence: {found[0][1]})"
        elif found := lookup(LISTED_KEY):
            agents = ", ".join(sorted({agent for agent, _detail in found}))
            new_state = "staged" if COVERAGE_STATE_RANK.get(state, 0) < COVERAGE_STATE_RANK["staged"] else state
            staged = f"staged natively for {agents}"
            # A row the resolved plan already marked staged for these agents keeps one copy of that phrase.
            old_reason = str(row.get("reason") or "")
            planned = (staged, f"staged as a native rule for {agents}")
            prefix = "" if any(phrase in old_reason for phrase in planned) else f"{staged}; "
            note = f"{prefix}the load census listed it ({found[0][1]}) but the harness did not confirm it was loaded"
        elif found := lookup("not_loaded"):
            agents = ", ".join(sorted({agent for agent, _detail in found}))
            note = f"not loaded natively by {agents}: {found[0][1]}"
        if note:
            note = note[: MAX_CENSUS_TEXT * 2]
            old = str(row.get("reason") or "")
            # A promotion replaces an old "not staged" reason; a not-loaded note keeps the reason it adds to.
            promoted = new_state != state
            row["reason"] = f"{old}; {note}" if old and (state in EVALUATED_COVERAGE_STATES or not promoted) else note
            row["state"] = new_state
        rows.append(row)
    return summarize_coverage(rows)


def finalize_native_provenance(provenance: dict[str, Any], engine_result: Any) -> None:
    """Add ``plugin_load`` and ``load_census`` to plugin provenance and fold the census into coverage.

    Reads what the runner recorded (``run_config.plugin_load``) and what the
    collector summarized per agent (``plugin_load_census``). Tolerates absent
    keys. A native arm whose plugin load was never confirmed in any trial
    (see :func:`native_load_unverified`) marks the run INCOMPLETE.
    """
    if not isinstance(engine_result, Mapping):
        return
    run_config = engine_result.get("run_config")
    plugin_load = run_config.get("plugin_load") if isinstance(run_config, Mapping) else None
    if isinstance(plugin_load, Mapping):
        provenance["plugin_load"] = dict(plugin_load)
    agents = engine_result.get("agents")
    summaries: dict[str, Mapping[str, Any]] = {}
    if isinstance(agents, Mapping):
        for agent, payload in agents.items():
            census = payload.get("plugin_load_census") if isinstance(payload, Mapping) else None
            if isinstance(census, Mapping):
                summaries[str(agent)] = census
    if not summaries:
        return
    provenance["load_census"] = {agent: dict(summary) for agent, summary in summaries.items()}
    promoted = apply_load_census(
        provenance.get("component_coverage"),
        summaries,
        plugin_load if isinstance(plugin_load, Mapping) else None,
    )
    if promoted is not None:
        provenance["component_coverage"] = promoted
    unverified = {agent: reason for agent, summary in summaries.items() if (reason := native_load_unverified(summary))}
    if unverified:
        provenance["native_load_unverified"] = unverified
        provenance["partial"] = True
