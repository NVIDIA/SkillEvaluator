# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-bounded reads and declared component paths of a plugin.

Every path a plugin declares is plugin-controlled input.
:func:`normalize_declared_path` turns a manifest path into a contained
root-relative POSIX path, or names its problem. :class:`PluginRootReader`
classifies paths with ``lstat`` on every component (links are never
followed) and reads files through
:class:`~skillevaluator.utils.secure_fs.SecureRoot` with a byte bound. The
finding helpers report a declared path that is missing, escapes the root, is
unsafe, lacks the ``./`` prefix, or lies in a folder Tier 1 scans skip. The
component inventory (:mod:`skillevaluator.plugin_components`) and the MCP
collection (:mod:`skillevaluator.plugin_mcp`) share them.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    SCAN_ARTIFACT_DIRS,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_formats import CLAUDE_PROFILE
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, discover_secure_files, stat_is_link_or_reparse

PLUGIN_CATEGORY = "PLUGIN_SCHEMA"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


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


def normalize_declared_path(raw: str, root_prefixes: Iterable[str] = ("${CLAUDE_PLUGIN_ROOT}",)) -> DeclaredPath:
    """Normalize one manifest path to a contained root-relative POSIX path.

    Claude Code requires ``./``-relative paths (``"."``/``"./"`` names the root).
    A root placeholder in ``root_prefixes`` names the root only when a separator
    (``/`` or ``\\``) or nothing follows it: a client expands it as text, so
    ``${CURSOR_PLUGIN_ROOT}foo/x.sh`` loads ``<root>foo/x.sh`` beside the root,
    an escape. The inventory passes only the placeholders a format's client
    expands in manifest paths (:attr:`FormatProfile.manifest_path_prefixes`,
    Cursor's), so any other leading ``${...}`` is the ``placeholder`` problem.
    The ``${CLAUDE_PLUGIN_ROOT}`` default is kept for existing callers; Claude
    Code itself expands no placeholder there. Absolute paths, home-relative
    paths, drive letters, and ``..`` segments are escapes.
    """
    text = raw.strip()
    if not text:
        return DeclaredPath(raw, None, "empty")
    if "\x00" in text or len(text) > 4096:
        return DeclaredPath(raw, None, "invalid")
    normalized = text.replace("\\", "/")
    dot_relative = normalized in {".", "./"} or normalized.startswith("./")
    matched = [prefix for prefix in root_prefixes if prefix and normalized.startswith(prefix)]
    if matched:
        below_root = normalized[len(max(matched, key=len)) :]
        if below_root and not below_root.startswith("/"):
            return DeclaredPath(raw, None, "escape")  # "<root>foo/x.sh" is beside the root, not in it
        normalized = "./" + below_root.lstrip("/")
        dot_relative = True
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
# Path findings                                                               #
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


def _in_unscanned_folder(rel: PurePosixPath) -> bool:
    """Whether a plugin-root-relative path is inside a folder that Tier 1 whole-tree scans skip.

    The scans prune ``evals/``, ``results/``, and ``versions/`` (and their
    dotted forms) at any depth. Only ``skills/<name>`` at the first level of
    ``skills/`` is scanned anyway, because bundled-skill discovery scans it as
    a skill.
    """
    parts = rel.parts
    return any(
        part in SCAN_ARTIFACT_DIRS and not (index == 1 and parts[0] == "skills") for index, part in enumerate(parts)
    )


def _unscanned_path_finding(
    reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str
) -> Finding | None:
    """HIGH when a declared component lives in a folder that Tier 1 whole-tree scans skip.

    ``evals/``, ``results/``, and ``versions/`` (and their dotted forms) hold
    evaluation output and snapshots, so the security, secret, and Unicode
    scans prune them. A component the manifest loads from there would never be
    scanned. Only ``skills/<name>`` at the first level of ``skills/`` is
    searched, because bundled-skill discovery scans it as a skill.
    """
    rel = declared.rel
    if rel is None or not _in_unscanned_folder(rel):
        return None
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_unscanned",
        f"'{field_name}' path {declared.raw!r} is inside a folder that Tier 1 whole-tree scans skip (evals/, "
        "results/, versions/), so the client loads files that are never security-scanned",
        reader.display(manifest_rel),
        "Move the component out of evaluation-output and version-snapshot folders.",
        metadata={"plugin_component_ref": declared.raw},
    )


def _style_finding(
    reader: PluginRootReader,
    field_name: str,
    declared: DeclaredPath,
    manifest_rel: str,
    *,
    client: str = "Claude Code rejects",
) -> Finding:
    return _plugin_finding(
        Severity.MEDIUM,
        "plugin_component_path_style",
        f"'{field_name}' path {declared.raw!r} does not start with './'; {client} such manifest paths",
        reader.display(manifest_rel),
        f"Write the path as './{declared.rel.as_posix() if declared.rel else declared.raw}'.",
    )
