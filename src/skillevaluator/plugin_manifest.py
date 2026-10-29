# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, race-resistant discovery for supported plugin manifests.

A plugin root may carry more than one supported manifest, for example both a
``.claude-plugin/plugin.json`` and a ``.codex-plugin/plugin.json``. Discovery
is deterministic: every supported manifest path is discovered in one
no-follow, root-bounded walk, the first one in
:data:`~skillevaluator.constants.PLUGIN_MANIFEST_PRECEDENCE` is selected, and
the others are returned as additional manifest declarations.

Client manifests (the JSON ones) are matched without regard to case. On a
case-insensitive filesystem, such as the macOS default, a client that opens
``.codex-plugin/plugin.json`` reads ``.Codex-Plugin/plugin.json``, so that
spelling is the client's manifest too. A case variant is used only when the
exact spelling is absent, and it is flagged (see
:attr:`PluginManifestFile.case_variant`).
"""

from __future__ import annotations

import codecs
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_CONTAINED_MANIFEST_TYPES,
    PLUGIN_MANIFEST_FILES,
    PLUGIN_MANIFEST_PRECEDENCE,
    PLUGIN_NATIVE_MANIFEST_DIRS,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.plugin_formats import (
    AGENT_PLUGINS_SCHEMA_PREFIX,
    declares_agent_plugins_schema,
    parse_manifest_text,
)
from skillevaluator.utils.secure_fs import (
    SecureFile,
    SecurePathError,
    SecureRoot,
    discover_secure_files,
    stat_is_link_or_reparse,
)
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_json

_MANIFEST_TYPES_BY_PATH: dict[Path, str] = {
    Path(path): manifest_type for path, manifest_type in PLUGIN_MANIFEST_PRECEDENCE
}
_MANIFEST_PATHS: tuple[Path, ...] = tuple(Path(path) for path, _manifest_type in PLUGIN_MANIFEST_PRECEDENCE)
# Client-loaded (JSON) manifest paths by case-folded spelling. Only
# SkillEvaluator reads agent_plugin.yaml/.yml, so those match exactly.
_CLIENT_MANIFEST_PATHS_FOLDED: dict[str, Path] = {
    Path(path).as_posix().casefold(): Path(path)
    for path, manifest_type in PLUGIN_MANIFEST_PRECEDENCE
    if manifest_type in PLUGIN_CONTAINED_MANIFEST_TYPES
}
# Vendor manifest directories by case-folded name: clients on a case-insensitive
# filesystem open .Claude-Plugin/plugin.json as .claude-plugin/plugin.json.
NATIVE_MANIFEST_DIRS_FOLDED = frozenset(name.casefold() for name in PLUGIN_NATIVE_MANIFEST_DIRS)
# Bytes of a root plugin.json read to decide the Agent Plugins opt-in. Clients
# read the whole file, so this is the lenient read bound, not the 1 MiB manifest
# bound. A larger file opts in, so it fails as an unreadable manifest.
AGENT_PLUGINS_OPT_IN_MAX_BYTES = CONTENT_DEDUP_MAX_TOTAL_BYTES


def canonical_manifest_relative(relative: Path | str) -> Path | None:
    """Return the supported manifest path that a root-relative path names, ignoring case for client manifests."""
    relative = Path(relative)
    if relative in _MANIFEST_TYPES_BY_PATH:
        return relative
    return _CLIENT_MANIFEST_PATHS_FOLDED.get(relative.as_posix().casefold())


ManifestReadProblem = Literal["unsafe", "encoding", "size_limit"]


class PluginManifestPathError(SecurePathError):
    """Raised when a plugin root/manifest cannot be trusted or read safely.

    ``reason`` is ``"unsafe"`` for a link, special, hard-linked, or changed
    file and for root or containment problems (fail closed). ``"encoding"``
    (the bytes are not valid in the requested encoding) and ``"size_limit"``
    (the file is over the read bound) describe the content of a safely
    discovered regular file; see :attr:`content_error`.
    """

    def __init__(self, message: str, *, relative_path: str = ".", reason: ManifestReadProblem = "unsafe") -> None:
        super().__init__("unsafe_plugin_manifest", message, relative_path=relative_path)
        self.reason: ManifestReadProblem = reason

    @property
    def content_error(self) -> bool:
        """Whether the file is safe to read but its content is unusable (not decodable, or over the size bound)."""
        return self.reason != "unsafe"


def _read_secure_manifest_bytes(secure_file: SecureFile, declared_path: Path, *, max_bytes: int) -> bytes:
    try:
        # ``SecureFile.root`` is the absolute lexical root captured during
        # discovery. The caller's spelling of the root remains for diagnostics
        # and may be relative to a cwd that later changes.
        with SecureRoot(secure_file.root) as secure_root:
            raw, _metadata = secure_root.read_bytes(
                secure_file.relative_path,
                max_bytes,
                expected=secure_file.metadata,
            )
    except SecurePathError as exc:
        if exc.code == "file_size_limit":
            raise PluginManifestPathError(
                f"Plugin manifest exceeds the {max_bytes}-byte read limit: {declared_path}",
                relative_path=secure_file.rel_path,
                reason="size_limit",
            ) from exc
        raise PluginManifestPathError(
            f"Plugin manifest changed, is unsafe, or cannot be decoded: {declared_path}: {exc}",
            relative_path=secure_file.rel_path,
        ) from exc
    return raw


def _read_secure_manifest(
    secure_file: SecureFile,
    declared_path: Path,
    *,
    encoding: str,
    max_bytes: int,
) -> str:
    raw = _read_secure_manifest_bytes(secure_file, declared_path, max_bytes=max_bytes)
    try:
        return raw.decode(encoding)
    except UnicodeError as exc:
        raise PluginManifestPathError(
            f"Plugin manifest cannot be decoded as {encoding}: {declared_path}: {exc}",
            relative_path=secure_file.rel_path,
            reason="encoding",
        ) from exc
    except LookupError as exc:
        raise PluginManifestPathError(
            f"Unknown plugin manifest encoding {encoding!r}: {declared_path}",
            relative_path=secure_file.rel_path,
        ) from exc


def decode_manifest_leniently(raw: bytes) -> str:
    """Decode manifest bytes the way a lenient client does, never failing.

    A UTF-32 or UTF-16 byte-order mark, or the NUL pattern of BOM-less UTF-16
    or UTF-32 JSON, selects that codec; anything else is UTF-8. Bytes that are
    not valid in the codec become replacement characters, so one bad byte does
    not hide the rest of the manifest.
    """
    for bom, codec in (
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if raw.startswith(bom):
            return raw[len(bom) :].decode(codec, errors="replace")
    return raw.decode(json.detect_encoding(raw), errors="replace")


def _read_lenient_manifest(secure_file: SecureFile, declared_path: Path, *, max_bytes: int) -> str:
    """Read a discovered manifest inode up to ``max_bytes`` and decode it leniently."""
    raw = _read_secure_manifest_bytes(secure_file, declared_path, max_bytes=max_bytes)
    return decode_manifest_leniently(raw)


@dataclass(frozen=True)
class PluginManifestFile:
    """A supported manifest, kept as the inode that no-follow discovery found.

    The base of :class:`PluginManifestLocation` (the selected manifest) and
    :class:`PluginManifestCandidate` (an additional one); every read goes
    through the anchored plugin root descriptor and checks that inode.
    """

    declared_path: Path
    manifest_type: str
    secure_file: SecureFile

    @property
    def manifest_filename(self) -> str:
        """Root-relative POSIX path of the manifest, e.g. ``.codex-plugin/plugin.json``."""
        return self.secure_file.relative_path.as_posix()

    @property
    def case_variant(self) -> bool:
        """Whether the file is spelled differently from the supported path (only case-insensitive clients load it)."""
        return self.secure_file.relative_path not in _MANIFEST_TYPES_BY_PATH

    def read_text(self, *, encoding: str = "utf-8", max_bytes: int = CONTENT_DEDUP_MAX_FILE_BYTES) -> str:
        """Read the discovered inode through the anchored plugin root descriptor."""
        return _read_secure_manifest(self.secure_file, self.declared_path, encoding=encoding, max_bytes=max_bytes)

    def read_lenient_text(self, *, max_bytes: int = CONTENT_DEDUP_MAX_TOTAL_BYTES) -> str:
        """Read the discovered inode with a larger bound and lenient decoding.

        Clients do not share SkillEvaluator's 1 MiB manifest bound or its strict
        UTF-8 decoding, so a manifest that fails :meth:`read_text` for its size
        or encoding is still read this way to check what it declares. Link,
        special-file, and identity-change problems still raise.
        """
        return _read_lenient_manifest(self.secure_file, self.declared_path, max_bytes=max_bytes)

    def parse_for_audit(self) -> dict[str, Any] | None:
        """What a client reads from this manifest, for checks that must see every component (``None`` if unusable).

        The bounded strict read comes first. A client JSON manifest that is over
        the 1 MiB read bound or not UTF-8 is then read leniently, as Tier 1 does
        (``manifest_unreadable`` for the selected manifest,
        ``plugin_manifest_additional_unreadable`` for an additional one),
        because the client that loads it shares neither limit: its hooks and
        MCP servers must not disappear from Tier 3 coverage or the dependency
        audit. A link, special file, or changed inode is never read (it raises,
        and Tier 1 fails it closed); anything that still does not parse gives
        ``None``.
        """
        try:
            text = self.read_text(encoding="utf-8-sig")
        except PluginManifestPathError as exc:
            # Only client (JSON) manifests are read leniently; agent_plugin.yaml/.yml is SkillEvaluator's own.
            if not exc.content_error or self.manifest_type not in PLUGIN_CONTAINED_MANIFEST_TYPES:
                return None
            try:
                text = self.read_lenient_text().removeprefix("\ufeff")
            except PluginManifestPathError:
                return None
        try:
            data = parse_manifest_text(self.manifest_type, text)
        except (StructuredDataError, ValueError):
            return None
        return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class PluginManifestCandidate(PluginManifestFile):
    """An additional supported manifest found beside the selected one."""


@dataclass(frozen=True)
class PluginManifestLocation(PluginManifestFile):
    """The selected manifest, retained from no-follow discovery through reads."""

    root: Path
    # Other supported manifests in the same root, in precedence order.
    additional: tuple[PluginManifestCandidate, ...] = ()

    @property
    def path(self) -> Path:
        """Return the declared lexical path for diagnostics and compatibility."""
        return self.declared_path

    @property
    def contained(self) -> bool:
        """Whether the selected manifest describes a contained plugin."""
        return self.manifest_type in PLUGIN_CONTAINED_MANIFEST_TYPES

    def parsed_additional(self) -> list[tuple[str, str, dict[str, Any]]]:
        """``(manifest_type, manifest_filename, data)`` of each additional manifest a client can read.

        Each one is parsed with :meth:`~PluginManifestFile.parse_for_audit`, so
        an oversize or non-UTF-8 client manifest is read leniently, and an
        unsafe or unparseable one is left out (Tier 1 reports it). The rows
        are the ``additional`` argument of
        :func:`~skillevaluator.plugin_components.build_plugin_inventory`.
        """
        parsed: list[tuple[str, str, dict[str, Any]]] = []
        for candidate in self.additional:
            data = candidate.parse_for_audit()
            if data is not None:
                parsed.append((candidate.manifest_type, candidate.manifest_filename, data))
        return parsed


_AGENT_PLUGINS_RELATIVE = Path(PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE)
# The schema host as it appears in the raw bytes of UTF-8 (and ASCII-compatible
# single-byte), UTF-16, and UTF-32 text. The host has no slash, so it also
# matches the "https:\/\/agent-plugins.org\/..." spelling that JSON allows.
_AGENT_PLUGINS_HOST_MARKERS: tuple[bytes, ...] = tuple(
    urlsplit(AGENT_PLUGINS_SCHEMA_PREFIX).netloc.encode(codec)
    for codec in ("utf-8", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be")
)


def agent_plugins_opt_in(raw: bytes) -> bool:
    """Whether the bytes of a root ``plugin.json`` opt into Agent Plugins semantics.

    ``raw`` is the whole file. Callers do not read a file over
    :data:`AGENT_PLUGINS_OPT_IN_MAX_BYTES`: it opts in unread, because clients
    read any size, so the file is treated as the Agent Plugins manifest and
    then fails as unreadable. The bytes are decoded leniently
    (:func:`decode_manifest_leniently`) and parsed; a JSON object opts in when
    it declares an Agent Plugins ``$schema``, however much whitespace comes
    first and however its slashes are escaped. JSON that does not parse still
    counts when its bytes name the schema host in UTF-8, UTF-16, or UTF-32, so
    its syntax or encoding error is reported rather than hidden.
    """
    try:
        return declares_agent_plugins_schema(load_bounded_json(decode_manifest_leniently(raw)))
    except (StructuredDataError, ValueError):
        pass
    return any(marker in raw for marker in _AGENT_PLUGINS_HOST_MARKERS)


def agent_plugins_path_opt_in(path: Path) -> bool:
    """:func:`agent_plugins_opt_in` for the root ``plugin.json`` at ``path`` (bounded, anchored, no-follow read).

    The file is read like every other manifest, through the descriptor of its
    parent directory, and a file over :data:`AGENT_PLUGINS_OPT_IN_MAX_BYTES`
    opts in unread. Raises :class:`SecurePathError` or :class:`OSError` when
    the file cannot be read safely (a link, special or hard-linked file, a file
    that changes while it is read, or a missing file).
    """
    absolute = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100 - lexical, never resolved
    try:
        with SecureRoot(absolute.parent) as secure_root:
            raw, _metadata = secure_root.read_bytes(Path(absolute.name), AGENT_PLUGINS_OPT_IN_MAX_BYTES)
    except SecurePathError as exc:
        if exc.code != "file_size_limit":
            raise
        return True
    return agent_plugins_opt_in(raw)


def _is_agent_plugins_manifest(secure_file: SecureFile, declared_path: Path) -> bool:
    """Whether a discovered root ``plugin.json`` opts into Agent Plugins semantics.

    A root ``plugin.json`` is an Agent Plugins manifest only when it declares an
    ``https://agent-plugins.org/schemas/...`` ``$schema`` (the rule VS Code and
    Codex apply). Other root ``plugin.json`` files, such as the legacy Copilot
    format, are not plugin manifests SkillEvaluator supports and are ignored.

    Only link, special-file, identity-change, and containment problems fail
    closed. Content problems of a regular file decide the opt-in instead (see
    :func:`agent_plugins_opt_in`). Clients read the whole file whatever its
    size, so the file is read up to the lenient bound
    (:data:`AGENT_PLUGINS_OPT_IN_MAX_BYTES`), not the 1 MiB manifest bound; a
    file over even that bound opts in. When an oversize file opts in, it is
    then reported like any other unreadable manifest.
    """
    try:
        raw = _read_secure_manifest_bytes(secure_file, declared_path, max_bytes=AGENT_PLUGINS_OPT_IN_MAX_BYTES)
    except PluginManifestPathError as exc:
        if exc.reason != "size_limit":
            raise
        return True
    return agent_plugins_opt_in(raw)


def _wrap_security_error(exc: SecurePathError) -> PluginManifestPathError:
    return PluginManifestPathError(str(exc), relative_path=exc.relative_path)


def manifest_relative_path(path: Path) -> Path | None:
    """Return the root-relative manifest path that *path* names, if it names one.

    ``agent_plugin.yaml``/``.yml`` and a root ``plugin.json`` (Agent Plugins v1)
    sit at the plugin root; ``plugin.json`` inside ``.claude-plugin/``,
    ``.codex-plugin/``, or ``.cursor-plugin/`` roots the plugin at the vendor
    directory's parent. This is a lexical check only. Client manifest names
    match without regard to case; the returned path keeps the given spelling.
    """
    if path.name in PLUGIN_MANIFEST_FILES:
        return Path(path.name)
    name = path.name.casefold()
    if name == PLUGIN_CONTAINED_MANIFEST_FILE and path.parent.name.casefold() in NATIVE_MANIFEST_DIRS_FOLDED:
        return Path(path.parent.name) / path.name
    if name == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE:
        return Path(path.name)
    return None


def manifest_root_for(path: Path) -> Path | None:
    """Return the plugin root that a manifest path names, or ``None`` if *path* names no manifest.

    A lexical check (see :func:`manifest_relative_path`): the root is the
    manifest's directory, or the parent of its vendor directory. Nothing is
    read, so a root ``plugin.json`` names a root whether or not it opts into
    Agent Plugins.
    """
    relative = manifest_relative_path(path)
    if relative is None:
        return None
    return path.parent if len(relative.parts) == 1 else path.parent.parent


def manifest_type_for_relative_path(relative: Path | str) -> str | None:
    """Return the manifest type of one root-relative manifest path (case variants of client manifests included)."""
    canonical = canonical_manifest_relative(relative)
    return _MANIFEST_TYPES_BY_PATH.get(canonical) if canonical is not None else None


def locate_plugin_manifest(path: Path) -> PluginManifestLocation | None:
    """Locate one regular single-link manifest beneath a regular plugin root.

    All supported manifest variants are selected during discovery, so a linked,
    hard-linked, special, or reparse manifest fails even when another regular
    variant would otherwise win precedence. The returned object carries the
    discovered inode metadata and must perform the eventual bounded read. The
    other supported manifests in the same root are returned in ``additional``.
    """
    target = path.expanduser()
    direct_relative: Path | None = None
    try:
        target_metadata = target.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PluginManifestPathError(f"Cannot inspect plugin path safely: {target}: {exc}") from exc

    # A regular directory remains a plugin root even when its basename happens
    # to equal a supported manifest filename. Non-directories with a manifest
    # lexical shape are treated as direct manifests so links/specials reach the
    # secure selected-file checks and fail explicitly.
    if not stat_is_link_or_reparse(target_metadata) and stat.S_ISDIR(target_metadata.st_mode):
        root = target
    elif (manifest_root := manifest_root_for(target)) is not None:
        root = manifest_root
        direct_relative = target.relative_to(root)
    else:
        if stat_is_link_or_reparse(target_metadata):
            raise PluginManifestPathError(f"Plugin root is a symlink, junction, or reparse point: {target}")
        if not stat.S_ISREG(target_metadata.st_mode):
            raise PluginManifestPathError(f"Plugin root is not a regular directory: {target}")
        return None

    try:
        files = discover_secure_files(
            root,
            selected=lambda relative: canonical_manifest_relative(relative) is not None,
            excluded_dirs=SCAN_EXCLUDED_DIRS,
            max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
            max_depth=2,
        )
    except SecurePathError as exc:
        raise _wrap_security_error(exc) from exc

    # Keyed by the supported path. The exact spelling wins; otherwise the first
    # case variant (in name order) stands for it, as on a case-insensitive
    # filesystem, where only one spelling can exist.
    by_relative: dict[Path, SecureFile] = {}
    for file in sorted(files, key=lambda item: (item.relative_path not in _MANIFEST_TYPES_BY_PATH, item.rel_path)):
        canonical = canonical_manifest_relative(file.relative_path)
        if canonical is not None and canonical not in by_relative:
            by_relative[canonical] = file
    direct_canonical = canonical_manifest_relative(direct_relative) if direct_relative is not None else None
    agent_plugins_file = by_relative.get(_AGENT_PLUGINS_RELATIVE)
    if (
        agent_plugins_file is not None
        and direct_canonical != _AGENT_PLUGINS_RELATIVE
        and not _is_agent_plugins_manifest(agent_plugins_file, root / agent_plugins_file.relative_path)
    ):
        del by_relative[_AGENT_PLUGINS_RELATIVE]
    present = [relative for relative in _MANIFEST_PATHS if relative in by_relative]
    if direct_relative is not None:
        if direct_canonical is None or direct_canonical not in by_relative:
            raise PluginManifestPathError(
                f"Declared plugin manifest is missing or unsafe: {target}",
                relative_path=direct_relative.as_posix(),
            )
        selected_relative = direct_canonical
    elif present:
        selected_relative = present[0]
    else:
        return None

    selected_file = by_relative[selected_relative]
    additional = tuple(
        PluginManifestCandidate(
            declared_path=root / by_relative[relative].relative_path,
            manifest_type=_MANIFEST_TYPES_BY_PATH[relative],
            secure_file=by_relative[relative],
        )
        for relative in present
        if relative != selected_relative
    )
    return PluginManifestLocation(
        declared_path=root / selected_file.relative_path,
        root=root,
        manifest_type=_MANIFEST_TYPES_BY_PATH[selected_relative],
        secure_file=selected_file,
        additional=additional,
    )


# --------------------------------------------------------------------------- #
# Claude Code marketplace lookup for plugin dependencies                      #
# --------------------------------------------------------------------------- #
CLAUDE_MARKETPLACE_RELATIVE = Path(".claude-plugin") / "marketplace.json"
# How far above the plugin root a marketplace root may be (``plugins/<name>`` is two levels).
CLAUDE_MARKETPLACE_SEARCH_LEVELS = 3
# Bounds on what is read from a marketplace manifest and from a dependencies list.
CLAUDE_MARKETPLACE_MAX_PLUGINS = 1024
CLAUDE_DEPENDENCY_MAX_ENTRIES = 256


@dataclass(frozen=True)
class ClaudeMarketplace:
    """A Claude Code marketplace manifest that lists the evaluated plugin."""

    name: str
    root: Path
    # Plugin name -> its relative source folder ("" when the source is not a local path).
    plugins: dict[str, str]
    allowed_marketplaces: frozenset[str]

    @property
    def manifest_filename(self) -> str:
        return CLAUDE_MARKETPLACE_RELATIVE.as_posix()


def _marketplace_source_dir(root: Path, source: Any, plugin_root_setting: Any) -> Path | None:
    """The folder a relative marketplace ``source`` names (Claude Code resolves it from the marketplace root)."""
    if not isinstance(source, str) or not source or "://" in source or Path(source).is_absolute():
        return None
    if not source.startswith(("./", "../")) and source not in {".", "./"}:
        # A bare name is resolved under metadata.pluginRoot.
        if not isinstance(plugin_root_setting, str) or not plugin_root_setting.startswith("./"):
            return None
        source = f"{plugin_root_setting.rstrip('/')}/{source}"
    return Path(os.path.normpath(root / source))


def _parse_claude_marketplace(
    data: Any, root: Path, plugin_root: Path, plugin_name: str | None
) -> ClaudeMarketplace | None:
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not isinstance(data.get("plugins"), list):
        return None
    metadata = data.get("metadata")
    plugin_root_setting = metadata.get("pluginRoot") if isinstance(metadata, dict) else None
    plugins: dict[str, str] = {}
    lists_this_plugin = False
    for entry in data["plugins"][:CLAUDE_MARKETPLACE_MAX_PLUGINS]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        folder = _marketplace_source_dir(root, entry.get("source"), plugin_root_setting)
        plugins.setdefault(entry["name"], os.path.relpath(folder, root) if folder is not None else "")
        if folder is not None:
            lists_this_plugin = lists_this_plugin or folder == plugin_root
        elif plugin_name is not None and entry["name"] == plugin_name:
            lists_this_plugin = True
    if not lists_this_plugin:
        return None
    allowed = data.get("allowCrossMarketplaceDependenciesOn")
    return ClaudeMarketplace(
        name=data["name"],
        root=root,
        plugins=plugins,
        allowed_marketplaces=frozenset(item for item in allowed if isinstance(item, str))
        if isinstance(allowed, list)
        else frozenset(),
    )


def find_claude_marketplace(
    plugin_root: Path, *, plugin_name: str | None, stop: Path | None
) -> ClaudeMarketplace | None:
    """The Claude Code marketplace that lists the plugin at ``plugin_root``, or ``None``.

    Claude Code resolves a bare dependency name against the marketplace a
    plugin is installed from. Its manifest is ``.claude-plugin/marketplace.json``
    in the marketplace root: the plugin root itself, or a parent up to
    :data:`CLAUDE_MARKETPLACE_SEARCH_LEVELS` levels above it, never above
    ``stop`` (the repository root). Each file is read bounded through an
    anchored, no-follow read; a linked, unreadable, or malformed one is
    skipped. A marketplace counts only when an entry is this plugin: a relative
    ``source`` that names the plugin root, or else an entry with its name.
    """
    root = Path(os.path.abspath(plugin_root))  # noqa: PTH100 - callers pass a resolved root
    directory = root
    for _level in range(CLAUDE_MARKETPLACE_SEARCH_LEVELS + 1):
        try:
            metadata = (directory / CLAUDE_MARKETPLACE_RELATIVE).lstat()
        except OSError:
            metadata = None
        if metadata is not None and stat.S_ISREG(metadata.st_mode):
            try:
                with SecureRoot(directory) as secure_root:
                    raw, _metadata = secure_root.read_bytes(CLAUDE_MARKETPLACE_RELATIVE, CONTENT_DEDUP_MAX_FILE_BYTES)
                data = load_bounded_json(raw.decode("utf-8-sig"))
            except (SecurePathError, OSError, UnicodeError, StructuredDataError, ValueError):
                data = None
            marketplace = _parse_claude_marketplace(data, directory, root, plugin_name)
            if marketplace is not None:
                return marketplace
        if (stop is not None and directory == stop) or directory.parent == directory:
            break
        directory = directory.parent
    return None


def _source_folder_problem(marketplace: ClaudeMarketplace, folder: str) -> str | None:
    """Why a listed plugin's local source folder is unusable (no-follow), or ``None``."""
    current = marketplace.root
    for part in Path(folder).parts:
        if part in {".", ""}:
            continue
        if part == "..":
            return f"its source '{folder}' leaves the marketplace root"
        current = current / part
        try:
            metadata = current.lstat()
        except OSError:
            return f"its source folder '{folder}' does not exist"
        if stat_is_link_or_reparse(metadata):
            return f"its source '{folder}' passes through a link"
    try:
        if not stat.S_ISDIR(current.lstat().st_mode):
            return f"its source '{folder}' is not a folder"
    except OSError:
        return f"its source folder '{folder}' does not exist"
    return None


def classify_claude_dependency(name: str, marketplace_name: str | None, marketplace: ClaudeMarketplace | None) -> Any:
    """Classify one Claude Code dependency as a :class:`~skillevaluator.plugin_dependencies.DependencyRow`.

    ``referenced`` when the plugin's marketplace lists it, ``missing`` when that
    marketplace does not (or lists it with a local source folder that is not
    there), ``external`` when it names another marketplace, and ``unresolved``
    when no marketplace that lists this plugin was found.
    """
    from skillevaluator.plugin_dependencies import DependencyRow

    ref = f"{name}@{marketplace_name}" if marketplace_name else name
    if marketplace is None:
        if marketplace_name:
            return DependencyRow(ref, "external", None, f"it is in marketplace '{marketplace_name}'")
        return DependencyRow(
            ref,
            "unresolved",
            None,
            "no .claude-plugin/marketplace.json that lists this plugin was found above it, so the marketplace "
            "Claude Code resolves a bare name against is unknown",
        )
    where = f"{marketplace.name}' ({marketplace.manifest_filename}"
    if marketplace_name and marketplace_name != marketplace.name:
        if marketplace_name in marketplace.allowed_marketplaces:
            reason = (
                f"it is in marketplace '{marketplace_name}', which marketplace '{marketplace.name}' allows; Claude "
                "Code installs it only when that marketplace is added"
            )
        else:
            reason = (
                f"it is in marketplace '{marketplace_name}', and marketplace '{marketplace.name}' does not list it in "
                "allowCrossMarketplaceDependenciesOn, so Claude Code blocks installing it and users must install it "
                "first"
            )
        return DependencyRow(ref, "external", None, reason)
    if name not in marketplace.plugins:
        return DependencyRow(ref, "missing", None, f"marketplace '{where}) lists no plugin named '{name}'")
    folder = marketplace.plugins[name]
    if folder:
        problem = _source_folder_problem(marketplace, folder)
        if problem is not None:
            return DependencyRow(ref, "missing", folder, f"marketplace '{where}) lists it, but {problem}")
    return DependencyRow(ref, "referenced", folder or None, f"listed in marketplace '{where})")
