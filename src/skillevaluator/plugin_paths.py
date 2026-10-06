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
from skillevaluator.plugin_formats import CLAUDE_PROFILE, DEFAULT_SKILLS_DIR, FormatProfile
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, discover_secure_files, lstat_walk

# The category of every plugin schema and component finding; a policy overlay changes a severity with
# PLUGIN_SCHEMA.<check>.
PLUGIN_CATEGORY = "PLUGIN_SCHEMA"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
# A longer declared path is "invalid": no component path comes close, and it bounds the work per path.
_MAX_DECLARED_PATH_CHARS = 4096


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


def normalize_declared_path(raw: str, root_prefixes: Iterable[str]) -> DeclaredPath:
    """Normalize one manifest path to a contained root-relative POSIX path.

    Claude Code requires ``./``-relative paths (``"."``/``"./"`` names the root).
    ``root_prefixes`` are the root placeholders the format's client expands in
    manifest paths: callers pass :attr:`FormatProfile.manifest_path_prefixes`,
    which only Cursor's profile fills, so any other leading ``${...}`` is the
    ``placeholder`` problem. A root placeholder names the root only when a
    separator (``/`` or ``\\``) or nothing follows it: a client expands it as
    text, so ``${CURSOR_PLUGIN_ROOT}foo/x.sh`` loads ``<root>foo/x.sh`` beside
    the root, an escape. Absolute paths, home-relative paths, a drive letter in
    any part (``./C:/Users``), and ``..`` segments are escapes; any other colon
    (an NTFS data stream, ``x.md:hidden``) is ``invalid``.
    """
    text = raw.strip()
    if not text:
        return DeclaredPath(raw, None, "empty")
    if "\x00" in text or len(text) > _MAX_DECLARED_PATH_CHARS:
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
    # Windows reads a character and a colon as a drive in any part, so joining
    # "./C:/Users/x" to the root leaves it; any other colon names a data stream.
    if any(part == ".." or part[1:2] == ":" for part in parts):
        return DeclaredPath(raw, None, "escape")
    if any(":" in part for part in parts):
        return DeclaredPath(raw, None, "invalid")
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
        walk = lstat_walk(self.root, rel)
        if walk.outcome in ("missing", "not_dir"):
            return "missing"
        if walk.outcome == "link":
            return "link"
        metadata = walk.metadata
        if walk.outcome == "error" or metadata is None:
            return "special"
        if stat.S_ISDIR(metadata.st_mode):
            return "dir"
        if stat.S_ISREG(metadata.st_mode):
            return "file" if allow_hard_links or getattr(metadata, "st_nlink", 1) == 1 else "special"
        return "special"

    def _read_bytes(
        self, rel: PurePosixPath, max_bytes: int, *, config: bool = False, allow_hardlinks: bool = False
    ) -> bytes:
        """Bounded, anchored, no-follow read counted against a read budget; raises :class:`SecurePathError`.

        ``config`` charges the read to the separate config budget.
        ``allow_hardlinks`` also reads a regular file with more than one link.
        """
        used = self.config_bytes_read if config else self.bytes_read
        budget = "config" if config else "inventory"
        remaining = CONTENT_DEDUP_MAX_TOTAL_BYTES - used
        if remaining <= 0:
            raise SecurePathError("total_size_limit", f"Plugin {budget} read budget exhausted.")
        try:
            with SecureRoot(self.root) as secure_root:
                raw, _metadata = secure_root.read_bytes(
                    Path(*rel.parts), min(max_bytes, remaining), allow_hardlinks=allow_hardlinks
                )
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
        anywhere in the path, special files, a file that changes while it is
        read, a file over ``max_bytes``, and an exhausted read budget raise
        :class:`SecurePathError`.
        """
        return self._read_bytes(rel, max_bytes, allow_hardlinks=True)

    def list_files(self, rel_dir: PurePosixPath, *, suffixes: tuple[str, ...] | None = None) -> list[PurePosixPath]:
        """Securely list regular files below a contained directory (raises on links).

        Nothing is pruned: a client loads a nested component folder whatever its
        name, so ``commands/evals/`` and ``agents/node_modules/`` are listed too,
        although the Tier 1 whole-tree scans skip them (see
        :func:`_in_unscanned_folder`).
        """
        start = self.root if str(rel_dir) == "." else self.root / rel_dir.as_posix()

        def _selected(relative: Path) -> bool:
            return suffixes is None or relative.name.lower().endswith(suffixes)

        files = discover_secure_files(
            start,
            selected=_selected,
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


# The folders named in plugin_component_path_unscanned messages.
_UNSCANNED_FOLDERS = "evals/, results/, versions/, .git/, .venv/, node_modules/, __pycache__/"
_UNSCANNED_SUGGESTION = (
    "Move the component out of evaluation-output, version-snapshot, VCS, virtualenv, package, and bytecode-cache "
    "folders."
)


def _in_unscanned_folder(rel: PurePosixPath) -> bool:
    """Whether a plugin-root-relative path is inside a folder that Tier 1 whole-tree scans skip.

    The security, secret, and Unicode scans prune every folder in
    :data:`~skillevaluator.constants.SCAN_EXCLUDED_DIRS` at any depth:
    ``evals/``, ``results/``, and ``versions/`` (and their dotted forms),
    ``.git/``, ``.venv/``, ``node_modules/``, and ``__pycache__/``. Only an
    evaluation-output or snapshot name at the first level of ``skills/``
    (``skills/evals/``) is scanned anyway, because bundled-skill discovery
    scans it as a skill.
    """
    parts = rel.parts
    return any(
        part in SCAN_EXCLUDED_DIRS
        and not (index == 1 and parts[0] == DEFAULT_SKILLS_DIR and part in SCAN_ARTIFACT_DIRS)
        for index, part in enumerate(parts)
    )


def _unscanned_path_finding(
    reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str
) -> Finding | None:
    """HIGH when a declared component lives in a folder that Tier 1 whole-tree scans skip.

    The scans prune evaluation output, snapshots, VCS metadata, virtualenvs,
    packages, and bytecode caches (:func:`_in_unscanned_folder`), so a
    component the manifest loads from there would never be scanned.
    """
    rel = declared.rel
    if rel is None or not _in_unscanned_folder(rel):
        return None
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_unscanned",
        f"'{field_name}' path {declared.raw!r} is inside a folder that Tier 1 whole-tree scans skip "
        f"({_UNSCANNED_FOLDERS}), so the client loads files that are never security-scanned",
        reader.display(manifest_rel),
        _UNSCANNED_SUGGESTION,
        metadata={"plugin_component_ref": declared.raw},
    )


def _unscanned_file_finding(reader: PluginRootReader, component_type: str, rel: PurePosixPath) -> Finding:
    """HIGH for a component file a client loads from a nested folder that Tier 1 whole-tree scans skip.

    For example ``commands/evals/deploy.md``: clients load nested component
    folders whatever their name (:meth:`PluginRootReader.list_files`), and the
    scans prune that one (:func:`_in_unscanned_folder`).
    """
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_unscanned",
        f"{component_type} file '{rel.as_posix()}' is inside a folder that Tier 1 whole-tree scans skip "
        f"({_UNSCANNED_FOLDERS}), so the client loads a file that is never security-scanned",
        reader.display(rel),
        _UNSCANNED_SUGGESTION,
        metadata={"path": rel.as_posix()},
    )


def _style_finding(
    reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str, profile: FormatProfile
) -> Finding:
    """MEDIUM for a manifest path without a leading ``./``, which the format's client rejects or ignores."""
    client = "Claude Code rejects" if profile is CLAUDE_PROFILE else f"the {profile.label} loader ignores"
    return _plugin_finding(
        Severity.MEDIUM,
        "plugin_component_path_style",
        f"'{field_name}' path {declared.raw!r} does not start with './'; {client} such manifest paths",
        reader.display(manifest_rel),
        f"Write the path as './{declared.rel.as_posix() if declared.rel else declared.raw}'.",
    )
