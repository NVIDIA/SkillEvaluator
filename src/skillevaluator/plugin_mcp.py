# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server declarations of a plugin, collected in Claude Code load order.

:func:`collect_mcp_declarations` reads every documented ``mcpServers`` form (an
inline server map, a ``.json`` config path, an ``.mcpb``/``.dxt`` bundle, or an
array mixing them) and the format's default MCP files through the plugin-root
reader (:mod:`skillevaluator.plugin_paths`). It normalizes Codex and Agent
Plugins entries and records the config-source findings and the per-server
static checks, including server code launched from a folder Tier 1 scans skip.
The component inventory
(:mod:`skillevaluator.plugin_components`) turns the collection into ``mcp``
components.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from skillevaluator.constants import PLUGIN_COMPONENT_MAX_ITEMS, PLUGIN_CONFIG_MAX_BYTES
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_formats import (
    CLAUDE_PROFILE,
    FormatProfile,
    agent_plugins_schema_version,
    codex_accepts_path,
    declared_value_replaces_default,
)
from skillevaluator.plugin_paths import (
    _UNSCANNED_FOLDERS_TEXT,
    _VENV_INTERPRETER_RE,
    _WINDOWS_DRIVE_RE,
    PLUGIN_CATEGORY,
    DeclaredPath,
    PluginRootReader,
    _in_unscanned_folder,
    _path_problem_finding,
    _plugin_finding,
    _root_variable_finding,
    _style_finding,
    _unscanned_folder,
    _unscanned_path_finding,
    normalize_declared_path,
)
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.structured_data import StructuredDataError, StructuredDataLimitError, load_bounded_json
from skillevaluator.validators.mcp_static import (
    CATEGORY as MCP_CATEGORY,
)
from skillevaluator.validators.mcp_static import (
    CODEX_UNAPPLIED_MCP_FIELDS,
    McpPinning,
    classify_mcp_pinning,
    validate_mcp_server_declaration,
)
from skillevaluator.validators.url_policy import looks_like_inline_secret, safe_url

McpSource = Literal["inline", "mcp_json", "path_ref", "agent_plugin_yaml"]
_MCP_BUNDLE_SUFFIXES = (".mcpb", ".dxt")
# Codex MCP transport aliases and Agent Plugins' streamable-http type both mean http.
_HTTP_TYPE_ALIASES = frozenset({"streamable-http", "streamable_http"})


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

    def add_broken_source(self, finding: Finding, ref: str, path: str | None, problem: str = "invalid") -> None:
        """Record a declared config source that cannot be loaded, with the finding that says why."""
        self.findings.append(finding)
        self.broken_sources.append((ref, path, problem))


def mcp_pinning_summary(declarations: Iterable[McpDeclaration]) -> dict[str, Any]:
    """C1 ``mcp.pinning`` / C2 ``mcp_pinning`` of ``declarations`` (see :func:`summarize_pinning`)."""
    return summarize_pinning(declaration.pinning() for declaration in declarations)


def summarize_pinning(pins: Iterable[McpPinning]) -> dict[str, Any]:
    """Counts of classified MCP pins plus ``pinned / (pinned + unpinned)``.

    ``ratio`` is ``None`` when no declaration runs a package (nothing to pin).
    """
    counts = {"pinned": 0, "unpinned": 0, "not_applicable": 0}
    total = 0
    for pin in pins:
        total += 1
        counts[pin.status] += 1
    applicable = counts["pinned"] + counts["unpinned"]
    return {
        "total": total,
        **counts,
        "ratio": round(counts["pinned"] / applicable, 4) if applicable else None,
    }


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
    if manifest is not None and not contained:
        _collect_bundle_manifest_servers(collection, manifest, manifest_rel)
    declared = manifest.get("mcpServers") if manifest is not None else None
    entries = _declared_entries(reader, collection, declared, manifest_rel, profile, contained=contained)

    # 1. The default root MCP file (unless the manifest names it explicitly, or
    #    the format lets a declared mcpServers replace default discovery; Codex
    #    only when it keeps the declared value).
    if not (contained and declared_value_replaces_default(profile, "mcpServers", declared)):
        _collect_default_files(
            reader, collection, entries, manifest_rel, profile, declared=declared, contained=contained
        )

    # 2. Declared shapes in order.
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            _collect_inline_map(reader, collection, entry, manifest_rel)
        elif isinstance(entry, str):
            _collect_path_ref(reader, collection, entry, manifest_rel, profile, contained=contained)
        else:
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_servers_entry_invalid",
                    f"mcpServers[{index}] must be a config-file path string or an inline server map "
                    f"(got {type(entry).__name__})",
                    reader.display(manifest_rel),
                    'Use "./path/to/servers.json" or {"<name>": {"command"|"url": ...}} for each array entry.',
                    category=MCP_CATEGORY,
                )
            )

    # 3. Dialect normalization (Codex, Agent Plugins) before any static check.
    if profile.mcp_dialect != "claude":
        _normalize_dialect(reader, collection, profile, manifest)

    # 4. Duplicate names across Claude Code sources (later replaces earlier).
    _flag_duplicate_names(reader, collection)

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


def _collect_bundle_manifest_servers(collection: McpCollection, manifest: dict[str, Any], manifest_rel: str) -> None:
    """The ``mcp`` list of a bundle-reference ``agent_plugin.yaml``, kept as declared (Pydantic validates it)."""
    entries = manifest.get("mcp")
    if not isinstance(entries, list):
        return
    for entry in entries[:PLUGIN_COMPONENT_MAX_ITEMS]:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            config = {key: value for key, value in entry.items() if key != "name"}
            collection.declarations.append(McpDeclaration(entry["name"], config, "agent_plugin_yaml", manifest_rel))


def _declared_entries(
    reader: PluginRootReader,
    collection: McpCollection,
    declared: Any,
    manifest_rel: str,
    profile: FormatProfile,
    *,
    contained: bool,
) -> list[Any]:
    """The declared ``mcpServers`` shapes in order: an inline map, a path, or the first entries of an array.

    An array past the item cap gets a HIGH finding. A value of any other type
    is HIGH too, except that Codex drops it, loads the default ``.mcp.json``,
    and still installs the plugin (MEDIUM).
    """
    if declared is None:
        return []
    if isinstance(declared, dict | str):
        return [declared]
    if isinstance(declared, list):
        if len(declared) > PLUGIN_COMPONENT_MAX_ITEMS:
            collection.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "mcp_config_file_too_large",
                    f"'mcpServers' has {len(declared)} entries; only {PLUGIN_COMPONENT_MAX_ITEMS} are inspected",
                    reader.display(manifest_rel),
                    "Reduce the number of mcpServers entries.",
                    category=MCP_CATEGORY,
                )
            )
        return declared[:PLUGIN_COMPONENT_MAX_ITEMS]
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
            reader.display(manifest_rel),
            'Use {"<name>": {"command"|"url"|"provider": ...}}, "./.mcp.json", or an array mixing both.',
            category=MCP_CATEGORY,
        )
    )
    return []


def _collect_default_files(
    reader: PluginRootReader,
    collection: McpCollection,
    entries: list[Any],
    manifest_rel: str,
    profile: FormatProfile,
    *,
    declared: Any,
    contained: bool,
) -> None:
    """The format's default MCP files at the root, unless a declared path names one (then it loads as declared).

    A path Codex drops does not count as naming one. When the format's client
    may read the default file beside a declared ``mcpServers`` (Cursor), a LOW
    note says both are checked.
    """
    explicit_files: set[PurePosixPath] = set()
    for entry in entries:
        if isinstance(entry, str) and not (contained and profile.codex_path_rules and not codex_accepts_path(entry)):
            normalized = normalize_declared_path(entry, profile.manifest_path_prefixes)
            if normalized.rel is not None:
                explicit_files.add(normalized.rel)
    for default_name in profile.default_mcp_files:
        default_rel = PurePosixPath(default_name)
        if default_rel in explicit_files:
            continue
        kind = reader.kind(default_rel)
        if kind == "file":
            _load_mcp_file(reader, collection, default_rel, "mcp_json", default_rel.as_posix())
            if contained and declared is not None and "mcpServers" in profile.merged_default_fields:
                collection.findings.append(_merged_default_mcp_note(reader, default_rel, profile))
        elif kind in {"link", "special"}:
            finding = _path_problem_finding(
                reader, "mcpServers", DeclaredPath(default_name, default_rel), manifest_rel, "unsafe", default_rel
            )
            collection.add_broken_source(finding, default_name, default_rel.as_posix(), "unsafe")


def _collect_inline_map(
    reader: PluginRootReader, collection: McpCollection, entry: dict[str, Any], manifest_rel: str
) -> None:
    """An inline ``mcpServers`` server map (Cursor and Codex may also wrap it in ``{"mcpServers": {...}}``)."""
    inline = entry.get("mcpServers") if isinstance(entry.get("mcpServers"), dict) else entry
    if len(inline) > PLUGIN_COMPONENT_MAX_ITEMS:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_file_too_large",
                f"inline 'mcpServers' map declares {len(inline)} servers; only "
                f"{PLUGIN_COMPONENT_MAX_ITEMS} are inspected",
                reader.display(manifest_rel),
                "Reduce the number of MCP servers per map.",
                category=MCP_CATEGORY,
            )
        )
    for name, config in list(inline.items())[:PLUGIN_COMPONENT_MAX_ITEMS]:
        collection.declarations.append(McpDeclaration(str(name), config, "inline", manifest_rel))


def _flag_duplicate_names(reader: PluginRootReader, collection: McpCollection) -> None:
    """MEDIUM for each server name declared again by a later Claude Code source, which replaces the earlier one."""
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
            declaration.unapplied = tuple(key for key in CODEX_UNAPPLIED_MCP_FIELDS if config.get(key))
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
    """Collect one ``mcpServers`` string: a remote bundle URL, a ``.json`` config file, or a bundle file.

    A reference that cannot be loaded is a broken source with a blocking finding.
    A path Codex drops (it loads the default ``.mcp.json`` instead) gets its
    finding but is not a broken source.
    """
    if raw.strip().lower().startswith(("https://", "http://")):
        _collect_url_ref(reader, collection, raw, manifest_rel)
        return
    # Codex drops a path it does not accept and loads the default .mcp.json instead.
    dropped = contained and profile.codex_path_rules and not codex_accepts_path(raw)
    declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
    if declared.problem is not None or declared.rel is None:
        problem = "escape" if declared.problem == "escape" else "invalid"
        finding = _path_problem_finding(
            reader, "mcpServers", declared, manifest_rel, problem, reference=profile.reference
        )
        if dropped:
            collection.findings.append(finding)
        else:
            collection.add_broken_source(finding, raw, None, problem)
        return
    rel = declared.rel
    path = rel.as_posix()

    def fail(finding: Finding, problem: str = "invalid") -> None:
        collection.add_broken_source(finding, raw, path, problem)

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
        fail(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_path_invalid",
                f"mcpServers path {raw!r} must name a .json config file or an .mcpb/.dxt bundle",
                reader.display(manifest_rel),
                "Point mcpServers at a JSON file such as './.mcp.json'.",
                category=MCP_CATEGORY,
                metadata={"plugin_component_ref": raw},
            )
        )
        return
    kind = reader.kind(rel)
    if kind == "missing":
        fail(_path_problem_finding(reader, "mcpServers", declared, manifest_rel, "missing"), "missing")
        return
    if kind in {"link", "special"}:
        fail(_path_problem_finding(reader, "mcpServers", declared, manifest_rel, "unsafe", rel), "unsafe")
        return
    if kind != "file":
        fail(_path_problem_finding(reader, "mcpServers", declared, manifest_rel, "invalid"))
        return
    if suffix in _MCP_BUNDLE_SUFFIXES:
        collection.bundles.append((raw, path))
        collection.findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "mcp_bundle_not_inspected",
                f"mcpServers references the MCP bundle '{path}'; bundle contents are not inspected "
                "statically and are not staged for evaluation",
                reader.display(rel),
                "Declare the server as a .json config (command/url) so it can be validated and evaluated.",
                category=MCP_CATEGORY,
            )
        )
        return
    _load_mcp_file(reader, collection, rel, "path_ref", raw)


def _collect_url_ref(reader: PluginRootReader, collection: McpCollection, raw: str, manifest_rel: str) -> None:
    """Collect one remote ``mcpServers`` reference; only an ``.mcpb``/``.dxt`` bundle may be named by URL.

    The bundle is recorded but not inspected. One fetched over plaintext http
    is a broken source with a blocking finding. Reports and the inventory
    never carry the URL's userinfo or query credentials.
    """
    manifest_display = reader.display(manifest_rel)
    shown = safe_url(raw)
    lowered = raw.strip().lower()

    def fail(check: str, message: str, suggestion: str, *, metadata: dict[str, Any] | None = None) -> None:
        finding = _plugin_finding(
            Severity.HIGH, check, message, manifest_display, suggestion, category=MCP_CATEGORY, metadata=metadata
        )
        collection.add_broken_source(finding, shown, None)

    if not lowered.split("?", 1)[0].endswith(_MCP_BUNDLE_SUFFIXES):
        fail(
            "mcp_config_path_invalid",
            f"mcpServers URL {shown!r} is not an .mcpb/.dxt bundle; only bundles may be referenced by URL",
            "Reference a './'-relative .json config file, an inline server map, or an .mcpb bundle.",
            metadata={"plugin_component_ref": shown},
        )
        return
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
        fail(
            "mcp_url_insecure_scheme",
            f"mcpServers bundle {shown!r} is downloaded over plaintext http; the code it runs can be replaced in "
            "transit",
            "Serve the bundle over https:// or vendor it into the plugin.",
        )
    else:
        collection.bundles.append((shown, None))


def _load_mcp_file(
    reader: PluginRootReader, collection: McpCollection, rel: PurePosixPath, source: McpSource, raw: str
) -> None:
    """Collect the servers of one MCP config file; a file that cannot be used is a broken source.

    Its finding names the declared ref (``plugin_component_ref``), so the broken source's inventory row counts it.
    """
    display = reader.display(rel)
    path = rel.as_posix()

    def fail(
        check: str, message: str, suggestion: str, *, problem: str = "invalid", category: str = MCP_CATEGORY
    ) -> None:
        finding = _plugin_finding(
            Severity.HIGH,
            check,
            message,
            display,
            suggestion,
            category=category,
            metadata={"plugin_component_ref": raw},
        )
        collection.add_broken_source(finding, raw, path, problem)

    try:
        text = reader.read_text(rel, PLUGIN_CONFIG_MAX_BYTES, config=True)
    except SecurePathError as exc:
        if exc.code in {"file_size_limit", "total_size_limit"}:
            detail = (
                f"exceeds the {PLUGIN_CONFIG_MAX_BYTES}-byte limit"
                if exc.code == "file_size_limit"
                else f"could not be read: {exc}"
            )
            fail(
                "mcp_config_file_too_large",
                f"MCP config '{path}' {detail}",
                "Keep MCP config files small; declare only server entries.",
            )
        elif exc.code == "invalid_text_encoding":
            fail(
                "mcp_config_file_invalid",
                f"MCP config '{path}' is not valid UTF-8",
                "Save the MCP config as UTF-8 JSON.",
            )
        else:
            fail(
                "plugin_component_path_unsafe",
                f"MCP config '{path}' could not be read safely ({exc}); it was not followed",
                "Replace links and special files with one regular contained JSON file.",
                problem="unsafe",
                category=PLUGIN_CATEGORY,
            )
        return
    try:
        data = load_bounded_json(text)
    except StructuredDataLimitError as exc:
        fail(
            "mcp_config_file_too_large",
            f"MCP config '{path}' exceeds structured-data complexity limits: {exc}",
            "Reduce nesting and collection sizes in the MCP config.",
        )
        return
    except StructuredDataError as exc:
        fail(
            "mcp_config_file_invalid",
            f"MCP config '{path}' is not valid JSON: {exc}",
            "Fix the JSON syntax of the MCP config.",
        )
        return
    if isinstance(data, dict):
        collection.file_schemas[path] = data.get("$schema")
    if isinstance(data, dict) and "mcpServers" in data:
        servers = data["mcpServers"]
        if not isinstance(servers, dict):
            fail(
                "mcp_servers_not_object",
                f"'mcpServers' in '{path}' must be an object mapping server names to their config",
                'Use {"mcpServers": {"<name>": {"command"|"url": ...}}}.',
            )
            return
    elif isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
        # Documented bare form: servers at the top level without the wrapper.
        servers = data
    else:
        fail(
            "mcp_config_file_invalid",
            f"MCP config '{path}' is neither {{\"mcpServers\": {{...}}}} nor a map of server objects",
            'Use {"mcpServers": {"<name>": {"command"|"url": ...}}} (or the same map without the wrapper).',
        )
        return
    if len(servers) > PLUGIN_COMPONENT_MAX_ITEMS:
        collection.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "mcp_config_file_too_large",
                f"MCP config '{path}' declares {len(servers)} servers; only {PLUGIN_COMPONENT_MAX_ITEMS} are inspected",
                display,
                "Reduce the number of MCP servers per config file.",
                category=MCP_CATEGORY,
            )
        )
    for name, config in list(servers.items())[:PLUGIN_COMPONENT_MAX_ITEMS]:
        collection.declarations.append(McpDeclaration(str(name), config, source, path))
