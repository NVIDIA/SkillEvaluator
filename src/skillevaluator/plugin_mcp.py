# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server declarations of a plugin, collected in Claude Code load order.

:func:`collect_mcp_declarations` reads every documented ``mcpServers`` form (an
inline server map, a ``.json`` config path, an ``.mcpb``/``.dxt`` bundle, or an
array mixing them) and the format's default MCP files through the plugin-root
reader (:mod:`skillevaluator.plugin_paths`). It normalizes Codex and Agent
Plugins entries and records the config-source findings and the per-server
static checks. The component inventory
(:mod:`skillevaluator.plugin_components`) turns the collection into ``mcp``
components.
"""

from __future__ import annotations

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
    declared_value_replaces_default,
)
from skillevaluator.plugin_paths import (
    PLUGIN_CATEGORY,
    DeclaredPath,
    PluginRootReader,
    _path_problem_finding,
    _plugin_finding,
    _style_finding,
    _unscanned_path_finding,
    normalize_declared_path,
)
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.structured_data import StructuredDataError, StructuredDataLimitError, load_bounded_json
from skillevaluator.validators.mcp_static import (
    CATEGORY as MCP_CATEGORY,
)
from skillevaluator.validators.mcp_static import (
    McpPinning,
    classify_mcp_pinning,
    looks_like_inline_secret,
    redacted_url,
    validate_mcp_server_declaration,
)

McpSource = Literal["inline", "mcp_json", "path_ref", "agent_plugin_yaml"]
_MCP_BUNDLE_SUFFIXES = (".mcpb", ".dxt")
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

    def add_broken_source(self, finding: Finding, ref: str, path: str | None, problem: str = "invalid") -> None:
        """Record a declared config source that cannot be loaded, with the finding that says why."""
        self.findings.append(finding)
        self.broken_sources.append((ref, path, problem))


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
        elif kind in {"link", "special"}:
            finding = _path_problem_finding(
                reader, "mcpServers", DeclaredPath(default_name, default_rel), manifest_rel, "unsafe", default_rel
            )
            collection.add_broken_source(finding, default_name, default_rel.as_posix(), "unsafe")

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
            _collect_path_ref(reader, collection, entry, manifest_rel, profile)
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
            collection.server_findings.extend(
                validate_mcp_server_declaration(
                    declaration.name,
                    declaration.config,
                    reader.display(declaration.file),
                    allowed_private_hosts=allowed_private_hosts,
                )
            )
    return collection


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
) -> None:
    """Collect one ``mcpServers`` string: a remote bundle URL, a ``.json`` config file, or a bundle file.

    A reference that cannot be loaded is a broken source with a blocking finding.
    """
    if raw.strip().lower().startswith(("https://", "http://")):
        _collect_url_ref(reader, collection, raw, manifest_rel)
        return
    declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
    if declared.problem is not None or declared.rel is None:
        problem = "escape" if declared.problem == "escape" else "invalid"
        finding = _path_problem_finding(
            reader, "mcpServers", declared, manifest_rel, problem, reference=profile.reference
        )
        collection.add_broken_source(finding, raw, None, problem)
        return
    rel = declared.rel
    path = rel.as_posix()

    def fail(finding: Finding, problem: str = "invalid") -> None:
        collection.add_broken_source(finding, raw, path, problem)

    if profile.require_dot_relative and not declared.dot_relative:
        collection.findings.append(_style_finding(reader, "mcpServers", declared, manifest_rel, profile))
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
    shown = redacted_url(raw)
    lowered = raw.strip().lower()

    def fail(check: str, message: str, suggestion: str) -> None:
        finding = _plugin_finding(Severity.HIGH, check, message, manifest_display, suggestion, category=MCP_CATEGORY)
        collection.add_broken_source(finding, shown, None)

    if not lowered.split("?", 1)[0].endswith(_MCP_BUNDLE_SUFFIXES):
        fail(
            "mcp_config_path_invalid",
            f"mcpServers URL {shown!r} is not an .mcpb/.dxt bundle; only bundles may be referenced by URL",
            "Reference a './'-relative .json config file, an inline server map, or an .mcpb bundle.",
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
    """Collect the servers of one MCP config file; a file that cannot be used is a broken source."""
    display = reader.display(rel)
    path = rel.as_posix()

    def fail(
        check: str, message: str, suggestion: str, *, problem: str = "invalid", category: str = MCP_CATEGORY
    ) -> None:
        finding = _plugin_finding(Severity.HIGH, check, message, display, suggestion, category=category)
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
