# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed filesystem primitives for untrusted Tier 2 inputs.

Discovery is lexical and no-descent: redirects are counted and rejected from
no-follow metadata, except for the exact validated ``CLAUDE.md -> AGENTS.md``
compatibility alias. Selected files are read through
directory-file descriptors where the platform supports them, with identity,
type, link-count, size, and containment checks around the open.
"""

from __future__ import annotations

import errno
import functools
import os
import secrets
import stat
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Literal, NamedTuple

_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_READLINK_SUPPORTS_DIR_FD = os.readlink in os.supports_dir_fd
_SCANDIR_SUPPORTS_FD = os.scandir in os.supports_fd
MAX_SECURE_DIRECTORY_DEPTH = 64

# Native Windows access/share/create values used by both the selected-file
# reader and the atomic cache writer. Reader handles intentionally omit
# FILE_SHARE_DELETE (0x4), pinning every opened directory/file identity while
# it participates in an anchored traversal.
_WINDOWS_FILE_READ_DATA = 0x1
_WINDOWS_FILE_TRAVERSE = 0x20
_WINDOWS_FILE_READ_ATTRIBUTES = 0x80
_WINDOWS_SYNCHRONIZE = 0x100000
_WINDOWS_DELETE = 0x10000
_WINDOWS_GENERIC_WRITE = 0x40000000
_WINDOWS_SHARE_READ = 0x1
_WINDOWS_SHARE_WRITE = 0x2
_WINDOWS_SHARE_READ_WRITE = _WINDOWS_SHARE_READ | _WINDOWS_SHARE_WRITE
_WINDOWS_FILE_OPEN = 1
_WINDOWS_FILE_CREATE = 2
_WINDOWS_FILE_ATTRIBUTE_NORMAL = 0x80
_WINDOWS_FILE_DIRECTORY_FILE = 0x1
_WINDOWS_FILE_WRITE_THROUGH = 0x2
_WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT = 0x20
_WINDOWS_FILE_NON_DIRECTORY_FILE = 0x40
_WINDOWS_FILE_OPEN_FOR_BACKUP_INTENT = 0x00004000
_WINDOWS_FILE_OPEN_REPARSE_POINT = 0x00200000
# CreateFileW opens the volume anchor and the atomic writer's parent: an
# existing directory, without following a reparse point.
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_FLAG_BACKUP_SEMANTICS = 0x02000000
_WINDOWS_DIRECTORY_HANDLE_FLAGS = _WINDOWS_FLAG_BACKUP_SEMANTICS | _WINDOWS_FILE_OPEN_REPARSE_POINT
# Information classes for NtSetInformationFile and SetFileInformationByHandle.
_WINDOWS_FILE_RENAME_INFORMATION = 10
_WINDOWS_FILE_DISPOSITION_INFO = 4
# ERROR_FILE_EXISTS and ERROR_ALREADY_EXISTS.
_WINDOWS_FILE_EXISTS_ERRORS = frozenset({80, 183})
_WINDOWS_OBJ_CASE_INSENSITIVE = 0x40
_WINDOWS_OBJ_DONT_REPARSE = 0x1000
_WINDOWS_OBJECT_ATTRIBUTES_FLAGS = _WINDOWS_OBJ_CASE_INSENSITIVE | _WINDOWS_OBJ_DONT_REPARSE
_WINDOWS_DIRECTORY_READ_ACCESS = _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_FILE_TRAVERSE | _WINDOWS_SYNCHRONIZE
_WINDOWS_FILE_READ_ACCESS = _WINDOWS_FILE_READ_DATA | _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_SYNCHRONIZE
_WINDOWS_DISCOVERY_ENTRY_ACCESS = _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_SYNCHRONIZE
_WINDOWS_DIRECTORY_OPEN_OPTIONS = (
    _WINDOWS_FILE_DIRECTORY_FILE
    | _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT
    | _WINDOWS_FILE_OPEN_FOR_BACKUP_INTENT
    | _WINDOWS_FILE_OPEN_REPARSE_POINT
)
_WINDOWS_FILE_OPEN_OPTIONS = (
    _WINDOWS_FILE_NON_DIRECTORY_FILE | _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT | _WINDOWS_FILE_OPEN_REPARSE_POINT
)
_WINDOWS_DISCOVERY_ENTRY_OPTIONS = (
    _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT | _WINDOWS_FILE_OPEN_FOR_BACKUP_INTENT | _WINDOWS_FILE_OPEN_REPARSE_POINT
)


class SecurePathError(ValueError):
    """An unsafe, racy, inaccessible, or unbounded filesystem input."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        relative_path: str = ".",
        metadata: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.relative_path = relative_path
        self.metadata = metadata or {}


@dataclass(frozen=True)
class SecureFile:
    """A lexically contained regular single-link file discovered without follow."""

    root: Path
    path: Path
    relative_path: Path
    metadata: os.stat_result

    @property
    def rel_path(self) -> str:
        return self.relative_path.as_posix()


@dataclass
class _DirectoryFrame:
    """One live directory in the iterative descriptor-anchored DFS."""

    descriptor: int
    relative_path: Path
    expected: os.stat_result
    parent_name: str | None = None
    children: list[tuple[str, os.stat_result]] | None = None
    next_child: int = 0


@dataclass
class _WindowsDirectoryFrame:
    """One pinned directory in the iterative native Windows discovery DFS."""

    handle: int
    path: Path
    relative_path: Path
    expected: _WindowsHandleMetadata
    owns_handle: bool
    children: list[tuple[str, _WindowsHandleMetadata]] | None = None
    next_child: int = 0


@dataclass(frozen=True)
class _WindowsHandleMetadata:
    """Stable metadata queried from one open native Windows handle."""

    attributes: int
    volume_serial: int
    file_id: int
    size: int
    link_count: int
    last_write_time: int = 0

    @property
    def is_reparse(self) -> bool:
        return bool(self.attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)

    @property
    def is_directory(self) -> bool:
        return bool(self.attributes & stat.FILE_ATTRIBUTE_DIRECTORY)

    @property
    def is_plain_directory(self) -> bool:
        """A directory that is not itself a reparse point (a junction or directory symlink is one)."""
        return self.is_directory and not self.is_reparse

    def same_identity(self, other: _WindowsHandleMetadata) -> bool:
        """Whether both snapshots describe the same file: one volume and one file index."""
        return self.volume_serial == other.volume_serial and self.file_id == other.file_id


def stat_is_link_or_reparse(metadata: os.stat_result) -> bool:
    """Return whether metadata identifies a symlink or Windows reparse point."""
    file_attributes = getattr(metadata, "st_file_attributes", 0)  # Windows only
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def read_bounded(descriptor: int, max_bytes: int, *, truncate: bool = False, relative_path: str = ".") -> bytes:
    """Read at most ``max_bytes`` bytes from an open descriptor, 64 KiB at a time.

    With ``truncate`` the first ``max_bytes`` bytes are returned and the rest
    of the file is never read. Otherwise one more byte is requested to detect
    a larger file, which raises ``SecurePathError`` (``file_size_limit``,
    naming ``relative_path``) after reading at most ``max_bytes + 1`` bytes.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    limit = max_bytes if truncate else max_bytes + 1
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        chunk = os.read(descriptor, min(65_536, limit - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > max_bytes:
        raise SecurePathError(
            "file_size_limit",
            f"Selected file exceeds the {max_bytes}-byte limit: {relative_path}",
            relative_path=relative_path,
            metadata={"actual_bytes": total, "limit_bytes": max_bytes},
        )
    return b"".join(chunks)


LstatOutcome = Literal["ok", "missing", "not_dir", "link", "error"]


class LstatWalk(NamedTuple):
    """Where :func:`lstat_walk` stopped and what it saw there.

    ``ok``: every component was inspected; ``metadata`` is the last one's
    (``None`` for an empty path) and its type is left to the caller.
    ``missing`` (``FileNotFoundError`` or ``NotADirectoryError``) and
    ``error`` (any other ``OSError``) carry the exception in ``error``;
    ``link`` (a symlink or reparse point) and ``not_dir`` (a component before
    the last is not a directory) carry that component's ``metadata``.
    ``failing_index`` is the index into ``relative.parts`` of the component
    that stopped the walk.
    """

    outcome: LstatOutcome
    metadata: os.stat_result | None
    failing_index: int | None
    error: OSError | None = None


def lstat_walk(root: Path, relative: PurePath) -> LstatWalk:
    """Inspect ``root / relative`` one component at a time with ``lstat``, never following a link.

    The walk stops at the first component that is missing, cannot be
    inspected, is a symlink or reparse point, or (before the last) is not a
    directory, so nothing below a link is ever inspected.
    """
    current = root
    metadata: os.stat_result | None = None
    last_index = len(relative.parts) - 1
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except (FileNotFoundError, NotADirectoryError) as exc:
            return LstatWalk("missing", None, index, exc)
        except OSError as exc:
            return LstatWalk("error", None, index, exc)
        if stat_is_link_or_reparse(metadata):
            return LstatWalk("link", metadata, index)
        if index < last_index and not stat.S_ISDIR(metadata.st_mode):
            return LstatWalk("not_dir", metadata, index)
    return LstatWalk("ok", metadata, None)


def _absolute_no_resolve(path: Path) -> Path:
    """Return an absolute lexical path without resolving links."""
    return Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100


def _relative_path(path: Path) -> Path:
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SecurePathError("unsafe_path", f"Path must be relative and normalized: {path.as_posix()}")
    return path


def _raise_unsafe_file(relative: Path, *, hardlink: bool = False) -> None:
    if hardlink:
        raise SecurePathError(
            "unsafe_hardlink",
            f"Refusing hard-linked selected file with link count greater than one: {relative.as_posix()}",
            relative_path=relative.as_posix(),
        )
    raise SecurePathError(
        "unsafe_path",
        f"Refusing selected path that is not a regular file: {relative.as_posix()}",
        relative_path=relative.as_posix(),
    )


def _compatibility_alias_target(target_text: str, relative: Path) -> Path | None:
    """Return the recognized contained CLAUDE.md -> AGENTS.md alias target."""
    if relative.name != "CLAUDE.md" or target_text != "AGENTS.md":
        return None
    return relative.parent / "AGENTS.md"


def _validate_discovery_depth(max_depth: int | None) -> int | None:
    """Validate the optional shallow-discovery cutoff against the hard cap."""
    if max_depth is None:
        return None
    if type(max_depth) is not int or not 1 <= max_depth <= MAX_SECURE_DIRECTORY_DEPTH:
        raise ValueError(f"max_depth must be an integer from 1 to {MAX_SECURE_DIRECTORY_DEPTH}")
    return max_depth


def _raise_directory_depth_limit(relative: Path) -> None:
    actual = len(relative.parts)
    raise SecurePathError(
        "directory_depth_limit",
        (f"Tier 2 tree exceeds the directory depth limit of {MAX_SECURE_DIRECTORY_DEPTH}: {relative.as_posix()}"),
        relative_path=relative.as_posix(),
        metadata={"actual": actual, "limit": MAX_SECURE_DIRECTORY_DEPTH},
    )


def _inspect_root(root: Path) -> os.stat_result:
    """Return the no-follow metadata of a declared root, which must be a real directory."""
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise SecurePathError("invalid_root", f"Cannot inspect Tier 2 root: {exc}") from exc
    if stat_is_link_or_reparse(metadata):
        raise SecurePathError("unsafe_root", f"Tier 2 root is a symlink or reparse point: {root.name}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SecurePathError("invalid_root", f"Tier 2 root is not a regular directory: {root}")
    return metadata


class _DiscoveryAdmission:
    """Entry admission rules and the path budget, shared by the POSIX and Windows walkers.

    The walkers own every open, snapshot re-validation, and identity check.
    They hand each listed entry, in name order, to :meth:`admit_directory` or
    :meth:`admit_file`, so selection, exclusion, link, depth, and budget rules
    are applied identically on every platform and the path reported by a
    budget error does not depend on directory listing order.
    """

    def __init__(
        self,
        root: Path,
        *,
        selected: Callable[[Path], bool],
        excluded_dirs: Iterable[str],
        max_paths: int,
        max_depth: int | None,
        allow_context_alias: bool,
        refuse_selected_dirs: bool = True,
    ) -> None:
        self.root = root
        self.selected = selected
        self.refuse_selected_dirs = refuse_selected_dirs
        self.excluded = frozenset(excluded_dirs)
        self.max_paths = max_paths
        self.max_depth = max_depth
        self.allow_context_alias = allow_context_alias
        self.discovered_paths = 0
        self.files: list[SecureFile] = []
        # Keep exact authored spelling. ``WindowsPath`` keys compare
        # case-insensitively, which would otherwise let ``agents.md`` satisfy the
        # required exact ``CLAUDE.md -> AGENTS.md`` compatibility target.
        self.regular_by_relative: dict[str, os.stat_result] = {}
        self.pending_aliases: list[tuple[Path, Path]] = []

    def check_listing(self, directory: Path, listed: int) -> None:
        """Stop listing ``directory`` once its names must exceed the path budget.

        An excluded directory is the only entry that does not consume the
        budget, and each excluded name appears at most once per directory, so
        a listing longer than the remaining budget plus the excluded names
        cannot fit. This bounds memory before the listing is sorted.
        """
        if listed > self.max_paths - self.discovered_paths + len(self.excluded):
            self._raise_path_count_limit(directory, self.discovered_paths + listed)

    def admit_directory(self, relative: Path, *, linked: bool) -> bool:
        """Apply the directory rules; return whether the walker should descend."""
        if linked:
            raise SecurePathError(
                "unsafe_path",
                f"Refusing linked directory or reparse point before descent: {relative.as_posix()}",
                relative_path=relative.as_posix(),
            )
        if relative.name in self.excluded:
            return False
        if self.refuse_selected_dirs and self.selected(relative):
            _raise_unsafe_file(relative)
        self._consume_path(relative)
        directory_depth = len(relative.parts)
        if directory_depth > MAX_SECURE_DIRECTORY_DEPTH:
            _raise_directory_depth_limit(relative)
        return self.max_depth is None or directory_depth < self.max_depth

    def admit_file(
        self,
        relative: Path,
        metadata: os.stat_result,
        read_alias_target: Callable[[], str],
    ) -> None:
        """Apply the rules for a non-directory entry and keep it when selected.

        Redirects fail closed except the exact ``CLAUDE.md -> AGENTS.md`` alias,
        whose target is only recorded here and checked by :meth:`selected_files`.
        """
        self._consume_path(relative)
        is_selected = self.selected(relative)
        if stat_is_link_or_reparse(metadata):
            self._admit_compatibility_alias(relative, metadata, read_alias_target)
            return
        if stat.S_ISREG(metadata.st_mode):
            self.regular_by_relative[relative.as_posix()] = metadata
        if not is_selected:
            return
        if not stat.S_ISREG(metadata.st_mode):
            _raise_unsafe_file(relative)
        if getattr(metadata, "st_nlink", 1) != 1:
            _raise_unsafe_file(relative, hardlink=True)
        self.files.append(SecureFile(self.root, self.root / relative, relative, metadata))

    def selected_files(self) -> list[SecureFile]:
        """Require each alias target to be an independently discovered regular file; return the selection."""
        for alias, target in self.pending_aliases:
            target_metadata = self.regular_by_relative.get(target.as_posix())
            if target_metadata is None or getattr(target_metadata, "st_nlink", 1) != 1:
                raise SecurePathError(
                    "unsafe_path",
                    f"Compatibility alias target is not an independently enumerated regular file: {alias.as_posix()}",
                    relative_path=alias.as_posix(),
                )
        return sorted(self.files, key=lambda item: item.rel_path)

    def _admit_compatibility_alias(
        self,
        relative: Path,
        metadata: os.stat_result,
        read_alias_target: Callable[[], str],
    ) -> None:
        target: Path | None = None
        if self.allow_context_alias and relative.name == "CLAUDE.md":
            try:
                target = _compatibility_alias_target(read_alias_target(), relative)
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot inspect selected compatibility alias: {relative.as_posix()}: {exc}",
                    relative_path=relative.as_posix(),
                ) from exc
        if target is None:
            raise SecurePathError(
                "unsafe_path",
                f"Refusing symlink or reparse point: {relative.as_posix()}",
                relative_path=relative.as_posix(),
            )
        if getattr(metadata, "st_nlink", 1) != 1:
            _raise_unsafe_file(relative, hardlink=True)
        self.pending_aliases.append((relative, target))

    def _consume_path(self, relative: Path) -> None:
        self.discovered_paths += 1
        if self.discovered_paths > self.max_paths:
            self._raise_path_count_limit(relative, self.discovered_paths)

    def _raise_path_count_limit(self, relative: Path, actual: int) -> None:
        raise SecurePathError(
            "path_count_limit",
            f"Tier 2 tree exceeds the path limit of {self.max_paths} entries.",
            relative_path=relative.as_posix(),
            metadata={"actual": actual, "limit": self.max_paths},
        )


def discover_secure_files(
    root: Path,
    *,
    selected: Callable[[Path], bool],
    excluded_dirs: Iterable[str] = (),
    max_paths: int,
    max_depth: int | None = None,
    allow_context_alias: bool = True,
    refuse_selected_dirs: bool = True,
) -> list[SecureFile]:
    """Discover selected files below ``root`` without following redirects.

    Excluded directories are pruned before they consume the path budget.
    Every other authored entry consumes the budget, in name order within each
    directory. File and directory redirects fail closed without target content
    reads except for the exact contained ``CLAUDE.md -> AGENTS.md``
    compatibility alias, whose regular target must be independently
    discovered; only that target is returned and read.

    A directory that ``selected`` matches is refused: a caller that selects a
    file by name (``SKILL.md``) expects no folder there. Pass
    ``refuse_selected_dirs=False`` when ``selected`` filters every file below
    ``root``, so an ordinary subfolder (``rules/team/``, ``agents/team.md/``)
    is walked instead.
    """
    if max_paths < 1:
        raise ValueError("max_paths must be positive")
    max_depth = _validate_discovery_depth(max_depth)
    root = _absolute_no_resolve(root)
    root_metadata = _inspect_root(root)
    admission = _DiscoveryAdmission(
        root,
        selected=selected,
        excluded_dirs=excluded_dirs,
        max_paths=max_paths,
        max_depth=max_depth,
        allow_context_alias=allow_context_alias,
        refuse_selected_dirs=refuse_selected_dirs,
    )
    if os.name == "posix":
        _walk_posix(root, root_metadata, admission)
    elif os.name == "nt":
        _walk_windows(root, root_metadata, admission)
    else:
        raise SecurePathError(
            "secure_open_unavailable",
            "This platform cannot guarantee no-follow Tier 2 discovery.",
        )
    return admission.selected_files()


def _posix_directory_flags() -> int:
    """No-follow directory open flags (``AttributeError`` where the platform lacks them)."""
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _walk_posix(root: Path, root_metadata: os.stat_result, admission: _DiscoveryAdmission) -> None:
    """Admit every entry below ``root`` through held directory descriptors, never following links."""
    if not (_OPEN_SUPPORTS_DIR_FD and _READLINK_SUPPORTS_DIR_FD and _SCANDIR_SUPPORTS_FD):
        raise SecurePathError(
            "secure_open_unavailable",
            "This platform cannot guarantee descriptor-anchored no-follow Tier 2 discovery.",
        )
    frames = [_DirectoryFrame(_open_absolute_directory_posix(root), Path(), root_metadata)]
    try:
        # A descriptor stack makes the walk linear: root is opened once,
        # each descended child is opened once relative to its held parent,
        # and only the descriptors on the active DFS path remain live.
        while frames:
            frame = frames[-1]
            if frame.children is None:
                _admit_posix_directory(frame, admission)
            elif frame.next_child < len(frame.children):
                frames.append(_open_posix_child_directory(frame))
            else:
                _revalidate_finished_posix_directory(frames, root)
                os.close(frames.pop().descriptor)
    finally:
        while frames:
            os.close(frames.pop().descriptor)


def _admit_posix_directory(frame: _DirectoryFrame, admission: _DiscoveryAdmission) -> None:
    """List one held directory between stable snapshots and admit its entries in name order."""
    current = os.fstat(frame.descriptor)
    _validate_directory_snapshot(current, frame.relative_path, frame.expected)
    entries = _list_posix_directory(frame, admission)
    after_scan = os.fstat(frame.descriptor)
    _validate_directory_snapshot(after_scan, frame.relative_path, current)
    children: list[tuple[str, os.stat_result]] = []
    for name, metadata in entries:
        relative = frame.relative_path / name
        if stat.S_ISDIR(metadata.st_mode):
            if admission.admit_directory(relative, linked=stat_is_link_or_reparse(metadata)):
                children.append((name, metadata))
            continue
        admission.admit_file(
            relative,
            metadata,
            lambda name=name, directory_fd=frame.descriptor: os.readlink(name, dir_fd=directory_fd),
        )
    stable = os.fstat(frame.descriptor)
    _validate_directory_snapshot(stable, frame.relative_path, after_scan)
    frame.expected = stable
    frame.children = children


def _list_posix_directory(frame: _DirectoryFrame, admission: _DiscoveryAdmission) -> list[tuple[str, os.stat_result]]:
    """Return one held directory's entries with their no-follow metadata, sorted by name."""
    entries: list[tuple[str, os.stat_result]] = []
    try:
        with os.scandir(frame.descriptor) as iterator:
            for listed, entry in enumerate(iterator, start=1):
                admission.check_listing(frame.relative_path, listed)
                relative = frame.relative_path / entry.name
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise SecurePathError(
                        "path_access_error",
                        f"Cannot inspect Tier 2 path {relative.as_posix()}: {exc}",
                        relative_path=relative.as_posix(),
                    ) from exc
                entries.append((entry.name, metadata))
    except SecurePathError:
        raise
    except OSError as exc:
        raise SecurePathError(
            "path_access_error",
            f"Cannot enumerate Tier 2 directory {frame.relative_path.as_posix()}: {exc}",
            relative_path=frame.relative_path.as_posix(),
        ) from exc
    return sorted(entries, key=lambda item: item[0])


def _open_posix_child_directory(frame: _DirectoryFrame) -> _DirectoryFrame:
    """Open the frame's next admitted child relative to its held descriptor, still unchanged."""
    name, discovered = frame.children[frame.next_child]
    frame.next_child += 1
    relative = frame.relative_path / name
    try:
        before_open = os.stat(name, dir_fd=frame.descriptor, follow_symlinks=False)
    except OSError as exc:
        raise SecurePathError(
            "unsafe_path",
            f"Cannot revalidate Tier 2 directory before descent: {relative.as_posix()}: {exc}",
            relative_path=relative.as_posix(),
        ) from exc
    _validate_directory_snapshot(before_open, relative, discovered)
    try:
        child_fd = os.open(name, _posix_directory_flags(), dir_fd=frame.descriptor)
    except OSError as exc:
        raise SecurePathError(
            "unsafe_path",
            f"Cannot securely open Tier 2 directory {relative.as_posix()}: {exc}",
            relative_path=relative.as_posix(),
        ) from exc
    try:
        opened = os.fstat(child_fd)
        _validate_directory_snapshot(opened, relative, before_open)
    except BaseException:
        os.close(child_fd)
        raise
    return _DirectoryFrame(child_fd, relative, opened, parent_name=name)


def _revalidate_finished_posix_directory(frames: list[_DirectoryFrame], root: Path) -> None:
    """Require a finished directory to be unchanged and still bound to its declared name."""
    frame = frames[-1]
    stable = os.fstat(frame.descriptor)
    _validate_directory_snapshot(stable, frame.relative_path, frame.expected)
    if frame.parent_name is None:
        try:
            declared_root = root.lstat()
        except OSError as exc:
            raise SecurePathError(
                "unsafe_root",
                f"Cannot revalidate declared Tier 2 root after discovery: {exc}",
            ) from exc
        _validate_directory_snapshot(declared_root, Path(), stable)
        return
    parent = frames[-2]
    try:
        parent_entry = os.stat(frame.parent_name, dir_fd=parent.descriptor, follow_symlinks=False)
    except OSError as exc:
        raise SecurePathError(
            "unsafe_path",
            f"Cannot revalidate Tier 2 directory after its subtree: {frame.relative_path.as_posix()}: {exc}",
            relative_path=frame.relative_path.as_posix(),
        ) from exc
    _validate_directory_snapshot(parent_entry, frame.relative_path, stable)


def _walk_windows(root: Path, root_metadata: os.stat_result, admission: _DiscoveryAdmission) -> None:
    """Admit every entry below ``root`` through pinned native handles, never following reparse points."""
    root_handles: list[int] = []
    frames: list[_WindowsDirectoryFrame] = []
    try:
        root_handles = _windows_open_anchored_directory_chain(root, expected=root_metadata)
        root_handle = root_handles[-1]
        root_snapshot = _validate_windows_read_directory_handle(root_handle, Path())
        frames.append(_WindowsDirectoryFrame(root_handle, root, Path(), root_snapshot, owns_handle=False))
        while frames:
            frame = frames[-1]
            if frame.children is None:
                _admit_windows_directory(frame, admission)
            elif frame.next_child < len(frame.children):
                frames.append(_open_windows_child_directory(frame))
            else:
                current = _windows_handle_metadata(frame.handle)
                _validate_windows_discovery_directory_snapshot(current, frame.relative_path, frame.expected)
                finished = frames.pop()
                if finished.owns_handle:
                    _windows_close_handle(finished.handle)
    finally:
        while frames:
            frame = frames.pop()
            if frame.owns_handle:
                _windows_close_handle(frame.handle)
        while root_handles:
            _windows_close_handle(root_handles.pop())


def _admit_windows_directory(frame: _WindowsDirectoryFrame, admission: _DiscoveryAdmission) -> None:
    """List one pinned directory and admit its entries in name order, then require it unchanged."""
    names, stable = _windows_enumerate_pinned_directory_names(
        frame.path,
        frame.handle,
        frame.relative_path,
        frame.expected,
        admission=admission,
    )
    children: list[tuple[str, _WindowsHandleMetadata]] = []
    for name in names:
        child = _admit_windows_entry(frame, name, admission)
        if child is not None:
            children.append(child)
    current = _windows_handle_metadata(frame.handle)
    _validate_windows_discovery_directory_snapshot(current, frame.relative_path, stable)
    frame.expected = current
    frame.children = children


def _admit_windows_entry(
    frame: _WindowsDirectoryFrame,
    name: str,
    admission: _DiscoveryAdmission,
) -> tuple[str, _WindowsHandleMetadata] | None:
    """Admit one entry while a no-follow handle pins it; return it when it is a directory to descend."""
    path = frame.path / name
    relative = frame.relative_path / name
    try:
        entry_handle, handle_metadata = _windows_open_discovery_handle(frame.handle, name)
    except OSError as exc:
        raise SecurePathError(
            "path_access_error",
            f"Cannot securely inspect Tier 2 Windows path {relative.as_posix()}: {exc}",
            relative_path=relative.as_posix(),
        ) from exc
    try:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SecurePathError(
                "path_access_error",
                f"Cannot inspect pinned Tier 2 Windows path {relative.as_posix()}: {exc}",
                relative_path=relative.as_posix(),
            ) from exc
        _validate_windows_entry_snapshot(metadata, handle_metadata, relative)

        if handle_metadata.is_directory:
            if admission.admit_directory(relative, linked=handle_metadata.is_reparse):
                return name, handle_metadata
            return None

        # Read an alias target only while the entry handle pins the reparse point.
        alias_target = ""
        if handle_metadata.is_reparse and admission.allow_context_alias and relative.name == "CLAUDE.md":
            try:
                alias_target = os.readlink(path)  # noqa: PTH115
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot inspect selected compatibility alias: {relative.as_posix()}: {exc}",
                    relative_path=relative.as_posix(),
                ) from exc
        admission.admit_file(relative, metadata, lambda: alias_target)
        return None
    finally:
        _windows_close_handle(entry_handle)


def _open_windows_child_directory(frame: _WindowsDirectoryFrame) -> _WindowsDirectoryFrame:
    """Open the frame's next admitted child relative to its pinned handle, still unchanged."""
    name, discovered = frame.children[frame.next_child]
    frame.next_child += 1
    relative = frame.relative_path / name
    try:
        child_handle = _windows_open_relative_handle(
            frame.handle,
            name,
            access=_WINDOWS_DIRECTORY_READ_ACCESS,
            share=_WINDOWS_SHARE_READ_WRITE,
            disposition=_WINDOWS_FILE_OPEN,
            file_attributes=0,
            create_options=_WINDOWS_DIRECTORY_OPEN_OPTIONS,
        )
    except OSError as exc:
        raise SecurePathError(
            "unsafe_path",
            f"Cannot securely open Tier 2 Windows directory {relative.as_posix()}: {exc}",
            relative_path=relative.as_posix(),
        ) from exc
    try:
        opened = _windows_handle_metadata(child_handle)
        _validate_windows_discovery_directory_snapshot(opened, relative, discovered)
    except BaseException:
        _windows_close_handle(child_handle)
        raise
    return _WindowsDirectoryFrame(child_handle, frame.path / name, relative, opened, owns_handle=True)


class SecureRoot:
    """Descriptor-anchored reads beneath one verified regular root."""

    def __init__(self, root: Path, *, expected: os.stat_result | None = None) -> None:
        self.root = _absolute_no_resolve(root)
        self._expected = expected
        self._root_fd: int | None = None
        self._windows_root_handles: list[int] = []
        self._entered = False

    def __enter__(self) -> SecureRoot:
        if self._entered:
            raise SecurePathError("unsafe_root", "Secure Tier 2 root context is already active.")
        metadata = _inspect_root(self.root)
        if self._expected is not None:
            _validate_directory_snapshot(metadata, Path(), self._expected)

        if os.name == "posix":
            if not (hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW") and _OPEN_SUPPORTS_DIR_FD):
                raise SecurePathError(
                    "secure_open_unavailable",
                    "This platform cannot guarantee descriptor-anchored no-follow Tier 2 reads.",
                )
            root_fd = _open_absolute_directory_posix(self.root)
            try:
                opened = os.fstat(root_fd)
                if not stat.S_ISDIR(opened.st_mode) or not os.path.samestat(metadata, opened):
                    raise SecurePathError("unsafe_root", "Tier 2 root changed while being opened.")
            except BaseException:
                os.close(root_fd)
                raise
            self._root_fd = root_fd
            self._entered = True
            return self

        if os.name == "nt":
            self._windows_root_handles = _windows_open_anchored_directory_chain(self.root, expected=metadata)
            self._entered = True
            return self

        raise SecurePathError(
            "secure_open_unavailable",
            "This platform cannot guarantee no-follow Tier 2 reads.",
        )

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None
        while self._windows_root_handles:
            _windows_close_handle(self._windows_root_handles.pop())
        self._entered = False

    def duplicate_posix_root_descriptor(self) -> int:
        """Return a caller-owned descriptor pinned to the active POSIX root.

        Descriptor-relative output operations need the same no-follow root
        pinning as reads, while retaining ownership of their own descriptor.
        The caller must close the returned descriptor.
        """
        if not self._entered or os.name != "posix" or self._root_fd is None:
            raise SecurePathError(
                "secure_open_unavailable",
                "A pinned POSIX root descriptor is unavailable.",
            )
        return os.dup(self._root_fd)

    def read_bytes(
        self,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
        allow_hardlinks: bool = False,
    ) -> tuple[bytes, os.stat_result]:
        """Read one bounded regular file without following redirects.

        The file must have a single link unless ``allow_hardlinks`` is set, for
        content that is only scanned, such as a file a package manager links
        to its store. ``expected`` is the file's discovery snapshot.
        """
        descriptor, opened = self._open_file(relative_path, max_bytes, expected, allow_hardlinks=allow_hardlinks)
        try:
            if opened.st_size > max_bytes:
                raise SecurePathError(
                    "file_size_limit",
                    f"Selected file exceeds the {max_bytes}-byte limit: {relative_path.as_posix()}",
                    relative_path=relative_path.as_posix(),
                    metadata={"actual_bytes": opened.st_size, "limit_bytes": max_bytes},
                )
            content = read_bounded(descriptor, max_bytes, relative_path=relative_path.as_posix())
            _validate_opened_file(os.fstat(descriptor), relative_path, opened, allow_hardlinks=allow_hardlinks)
            return content, opened
        finally:
            os.close(descriptor)

    def read_text(
        self,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
    ) -> str:
        raw, _metadata = self.read_bytes(relative_path, max_bytes, expected=expected)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SecurePathError(
                "invalid_text_encoding",
                f"Selected Tier 2 file is not valid UTF-8: {relative_path.as_posix()}",
                relative_path=relative_path.as_posix(),
            ) from exc

    def read_file_text(self, file: SecureFile, max_bytes: int) -> str:
        # Compare lexical absolute forms: a caller may record the same root
        # relative to the working directory or with ``..`` components.
        if _absolute_no_resolve(file.root) != self.root:
            raise SecurePathError("unsafe_path", "Secure file belongs to a different Tier 2 root.")
        return self.read_text(file.relative_path, max_bytes, expected=file.metadata)

    def _open_file(
        self,
        relative_path: Path,
        max_bytes: int,
        expected: os.stat_result | None,
        *,
        allow_hardlinks: bool = False,
    ) -> tuple[int, os.stat_result]:
        """Open one selected file for reading; return its descriptor and verified open-time metadata."""
        if not self._entered:
            raise SecurePathError("secure_open_unavailable", "Secure Tier 2 root context is not active.")
        relative_path = _relative_path(relative_path)
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if os.name == "posix":
            return self._open_posix_file(relative_path, expected, allow_hardlinks=allow_hardlinks)
        if os.name == "nt":
            return self._open_windows_file(relative_path, expected, allow_hardlinks=allow_hardlinks)
        raise SecurePathError("secure_open_unavailable", "Secure no-follow reads are unavailable.")

    def _open_posix_file(
        self,
        relative_path: Path,
        expected: os.stat_result | None,
        *,
        allow_hardlinks: bool = False,
    ) -> tuple[int, os.stat_result]:
        if self._root_fd is None:
            raise SecurePathError("secure_open_unavailable", "Tier 2 root descriptor is unavailable.")
        directory_fd = os.dup(self._root_fd)
        file_flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0)
        )
        try:
            for component in relative_path.parts[:-1]:
                try:
                    child_fd = os.open(component, _posix_directory_flags(), dir_fd=directory_fd)
                except OSError as exc:
                    raise SecurePathError(
                        "unsafe_path",
                        f"Cannot securely traverse Tier 2 path component {component!r}: {exc}",
                        relative_path=relative_path.as_posix(),
                    ) from exc
                try:
                    child_metadata = os.fstat(child_fd)
                except BaseException:
                    os.close(child_fd)
                    raise
                if not stat.S_ISDIR(child_metadata.st_mode) or stat_is_link_or_reparse(child_metadata):
                    os.close(child_fd)
                    raise SecurePathError(
                        "unsafe_path",
                        f"Tier 2 path component is not a regular directory: {component}",
                        relative_path=relative_path.as_posix(),
                    )
                os.close(directory_fd)
                directory_fd = child_fd

            try:
                before = os.stat(relative_path.name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot inspect selected Tier 2 file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(before, relative_path, expected, allow_hardlinks=allow_hardlinks)
            try:
                descriptor = os.open(relative_path.name, file_flags, dir_fd=directory_fd)
            except OSError as exc:
                message = "Selected Tier 2 path is a symlink or unsafe file" if exc.errno == errno.ELOOP else str(exc)
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot securely open {relative_path.as_posix()}: {message}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            try:
                # Matching the pre-open snapshot, which matched ``expected``,
                # makes the descriptor's metadata the verified read baseline.
                opened = os.fstat(descriptor)
                _validate_opened_file(opened, relative_path, before, allow_hardlinks=allow_hardlinks)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor, opened
        finally:
            os.close(directory_fd)

    def _open_windows_file(
        self,
        relative_path: Path,
        expected: os.stat_result | None,
        *,
        allow_hardlinks: bool = False,
    ) -> tuple[int, os.stat_result]:
        if not self._windows_root_handles:
            raise SecurePathError("secure_open_unavailable", "Tier 2 root handle is unavailable.")

        import msvcrt

        directory_handles: list[int] = []
        parent_handle = self._windows_root_handles[-1]
        declared_path = self.root / relative_path
        descriptor = -1
        native_file_handle = -1
        try:
            for component in relative_path.parts[:-1]:
                native_directory_handle = _windows_open_relative_handle(
                    parent_handle,
                    component,
                    access=_WINDOWS_DIRECTORY_READ_ACCESS,
                    share=_WINDOWS_SHARE_READ_WRITE,
                    disposition=_WINDOWS_FILE_OPEN,
                    file_attributes=0,
                    create_options=_WINDOWS_DIRECTORY_OPEN_OPTIONS,
                )
                try:
                    _validate_windows_read_directory_handle(native_directory_handle, relative_path)
                except BaseException:
                    _windows_close_handle(native_directory_handle)
                    raise
                directory_handles.append(native_directory_handle)
                parent_handle = native_directory_handle

            # Python's Windows path stat and CRT descriptor stat do not expose
            # a reliably comparable ``st_dev``/``st_ino`` pair. Revalidate the
            # declared name with the same no-follow stat family used during
            # discovery, then pin it with a native handle that denies delete
            # sharing and require the declared name to remain unchanged.
            try:
                before_open = declared_path.lstat()
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot inspect selected Tier 2 file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(before_open, relative_path, expected, allow_hardlinks=allow_hardlinks)

            native_file_handle = _windows_open_relative_handle(
                parent_handle,
                relative_path.name,
                access=_WINDOWS_FILE_READ_ACCESS,
                share=_WINDOWS_SHARE_READ,
                disposition=_WINDOWS_FILE_OPEN,
                file_attributes=0,
                create_options=_WINDOWS_FILE_OPEN_OPTIONS,
            )
            _validate_windows_read_file_handle(native_file_handle, relative_path, allow_hardlinks=allow_hardlinks)
            try:
                after_open = declared_path.lstat()
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot revalidate selected Tier 2 file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(after_open, relative_path, before_open, allow_hardlinks=allow_hardlinks)
            descriptor = msvcrt.open_osfhandle(
                native_file_handle,
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0),
            )
            native_file_handle = -1  # ownership transferred to the CRT descriptor
            # CRT descriptor identity fields are not comparable to the path
            # snapshots above, so this is the baseline for the post-read check.
            opened = os.fstat(descriptor)
            _validate_opened_file(opened, relative_path, None, allow_hardlinks=allow_hardlinks)
            return descriptor, opened
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            raise SecurePathError(
                "unsafe_path",
                f"Cannot securely open Tier 2 file {relative_path.as_posix()}: {exc}",
                relative_path=relative_path.as_posix(),
            ) from exc
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        finally:
            if native_file_handle >= 0:
                _windows_close_handle(native_file_handle)
            while directory_handles:
                _windows_close_handle(directory_handles.pop())


def _validate_opened_file(
    metadata: os.stat_result,
    relative_path: Path,
    expected: os.stat_result | None,
    *,
    allow_hardlinks: bool = False,
) -> None:
    if stat_is_link_or_reparse(metadata):
        raise SecurePathError(
            "unsafe_path",
            f"Refusing selected symlink or reparse point: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )
    if not stat.S_ISREG(metadata.st_mode):
        _raise_unsafe_file(relative_path)
    if not allow_hardlinks and getattr(metadata, "st_nlink", 1) != 1:
        _raise_unsafe_file(relative_path, hardlink=True)
    if expected is not None and _snapshot_changed(metadata, expected):
        raise SecurePathError(
            "unsafe_path",
            f"Selected Tier 2 file changed identity or contents while being opened: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )


def _snapshot_changed(metadata: os.stat_result, expected: os.stat_result) -> bool:
    """Return whether two no-follow snapshots differ in identity, size, or modification or change time."""
    return not os.path.samestat(metadata, expected) or any(
        getattr(metadata, field, None) != getattr(expected, field, None)
        for field in ("st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _validate_directory_snapshot(
    metadata: os.stat_result,
    relative_path: Path,
    expected: os.stat_result,
) -> None:
    """Require one regular directory identity and entry snapshot to stay stable."""
    if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode) or _snapshot_changed(metadata, expected):
        label = relative_path.as_posix()
        raise SecurePathError(
            "unsafe_path",
            f"Tier 2 directory snapshot changed during discovery: {label}",
            relative_path=label,
        )


def _open_absolute_directory_posix(path: Path) -> int:
    flags = _posix_directory_flags()
    try:
        expected = path.lstat()
    except OSError as exc:
        raise SecurePathError("unsafe_root", f"Cannot inspect declared root safely: {exc}") from exc
    if stat_is_link_or_reparse(expected) or not stat.S_ISDIR(expected.st_mode):
        raise SecurePathError("unsafe_root", f"Declared Tier 2 root is a symlink or non-directory: {path}")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SecurePathError("unsafe_root", f"Cannot securely open declared Tier 2 root: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not os.path.samestat(expected, opened):
            raise SecurePathError("unsafe_root", "Declared Tier 2 root changed while being opened.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def secure_read_path_text(path: Path, max_bytes: int) -> str:
    """Read an arbitrary path through its filesystem anchor without follow."""
    absolute = _absolute_no_resolve(path)
    with SecureRoot(absolute.parent) as secure_root:
        return secure_root.read_text(Path(absolute.name), max_bytes)


def secure_atomic_write_text(path: Path, text: str, max_bytes: int) -> None:
    """Atomically replace one regular single-link file through a safe parent."""
    try:
        payload = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SecurePathError("invalid_text_encoding", "Output text is not valid UTF-8.") from exc
    if len(payload) > max_bytes:
        raise SecurePathError(
            "file_size_limit",
            f"Output exceeds the {max_bytes}-byte limit.",
            metadata={"actual_bytes": len(payload), "limit_bytes": max_bytes},
        )
    if os.name == "posix":
        _atomic_write_posix(path, payload)
        return
    if os.name == "nt":
        _atomic_write_windows(path, payload)
        return
    raise SecurePathError("secure_open_unavailable", "Secure atomic writes are unavailable on this platform.")


def _inspect_destination_posix(parent_fd: int, name: str, *, missing_ok: bool) -> os.stat_result | None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if stat_is_link_or_reparse(metadata):
        raise SecurePathError("unsafe_path", f"Destination is a symlink or reparse point: {name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise SecurePathError("unsafe_path", f"Destination is not a regular file: {name}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise SecurePathError("unsafe_hardlink", f"Destination is hard-linked (link count > 1): {name}")
    return metadata


def _validate_declared_parent_posix(parent: Path, parent_fd: int) -> None:
    """Require the held parent descriptor to remain at the declared path."""
    try:
        declared = parent.lstat()
        opened = os.fstat(parent_fd)
    except OSError as exc:
        raise SecurePathError("unsafe_path", f"Cannot revalidate declared output parent: {exc}") from exc
    if (
        stat_is_link_or_reparse(declared)
        or not stat.S_ISDIR(declared.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or not os.path.samestat(declared, opened)
    ):
        raise SecurePathError("unsafe_path", "Declared output parent changed identity during the atomic write.")


def _atomic_write_posix(path: Path, payload: bytes) -> None:
    absolute = _absolute_no_resolve(path)
    if not absolute.name or absolute.name in {".", ".."}:
        raise SecurePathError("unsafe_path", "Destination must name a file.")
    parent_fd = _open_absolute_directory_posix(absolute.parent)
    temporary_name: str | None = None
    descriptor = -1
    try:
        before = _inspect_destination_posix(parent_fd, absolute.name, missing_ok=True)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0)
        )
        for _attempt in range(128):
            candidate = f".{absolute.name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(candidate, flags, 0o600, dir_fd=parent_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor < 0 or temporary_name is None:
            raise SecurePathError("path_access_error", "Cannot allocate a secure temporary output file.")
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or getattr(opened, "st_nlink", 1) != 1:
            raise SecurePathError("unsafe_path", "Temporary output is not a regular single-link file.")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        written_metadata = os.fstat(descriptor)
        _validate_opened_file(written_metadata, Path(temporary_name), None)
        if written_metadata.st_size != len(payload):
            raise SecurePathError("unsafe_path", "Temporary output size changed while being written.")
        current = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
        _validate_opened_file(current, Path(temporary_name), written_metadata)
        destination = _inspect_destination_posix(parent_fd, absolute.name, missing_ok=True)
        if (before is None) != (destination is None) or (
            before is not None and destination is not None and not os.path.samestat(before, destination)
        ):
            raise SecurePathError("unsafe_path", "Destination changed identity while output was prepared.")
        _validate_declared_parent_posix(absolute.parent, parent_fd)
        os.replace(temporary_name, absolute.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temporary_name = None
        # Rename legitimately changes ctime. Capture a fresh descriptor phase,
        # then require the published name to match that new snapshot exactly.
        published_descriptor = os.fstat(descriptor)
        _validate_opened_file(published_descriptor, Path(absolute.name), None)
        if published_descriptor.st_size != len(payload):
            raise SecurePathError("unsafe_path", "Published output size changed during atomic replacement.")
        published_metadata = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        _validate_opened_file(published_metadata, Path(absolute.name), published_descriptor)
        _validate_declared_parent_posix(absolute.parent, parent_fd)
        stable_descriptor = os.fstat(descriptor)
        _validate_opened_file(stable_descriptor, Path(absolute.name), published_descriptor)
        stable_path = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        _validate_opened_file(stable_path, Path(absolute.name), stable_descriptor)
        _validate_declared_parent_posix(absolute.parent, parent_fd)
    except OSError as exc:
        raise SecurePathError("path_access_error", f"Cannot securely write output: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # Do not unlink by name on failure. Without a conditional unlink-by-
        # inode primitive, an attacker could swap either name between a stat
        # and unlink and make cleanup delete unrelated data. Successful replace
        # consumes the temporary name; failures may leave a mode-0600 orphan.
        os.close(parent_fd)


def _validate_windows_parent_components(path: Path) -> None:
    """Require every directory from the volume anchor to ``path``'s parent to be a real directory."""
    absolute = _absolute_no_resolve(path)
    parents = PurePath(*absolute.parent.parts[1:])
    walk = lstat_walk(Path(absolute.anchor), parents)
    if walk.error is not None:
        raise SecurePathError("path_access_error", f"Cannot inspect parent directory: {walk.error}") from walk.error
    if walk.outcome == "ok" and (walk.metadata is None or stat.S_ISDIR(walk.metadata.st_mode)):
        return
    component = parents.parts[-1 if walk.failing_index is None else walk.failing_index]
    raise SecurePathError(
        "unsafe_path",
        f"Path contains a symlink, junction, reparse point, or non-directory component: {component}",
    )


# Held while the native Windows structures and API are first built. On its own,
# functools.cache runs a builder twice when two threads miss at once, so the
# API prototypes could be bound to structure classes other than the cached
# ones, and every later native call would fail with ctypes.ArgumentError.
# Reentrant, because binding the API builds the structures.
_WINDOWS_CTYPES_LOCK = threading.RLock()


def _windows_types() -> SimpleNamespace:
    """ctypes structures for the native Windows calls, defined once per process.

    ``ctypes.POINTER`` keeps every structure class it is given for the life of
    the process, so structures defined inside each call would accumulate.
    """
    with _WINDOWS_CTYPES_LOCK:
        return _define_windows_types()


@functools.cache
def _define_windows_types() -> SimpleNamespace:
    """Define the structures; only :func:`_windows_types` calls this, under the lock."""
    import ctypes
    from ctypes import wintypes

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class IoStatusValue(ctypes.Union):
        _fields_ = [("Status", wintypes.LONG), ("Pointer", wintypes.LPVOID)]  # noqa: RUF012

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Value", IoStatusValue), ("Information", ctypes.c_size_t)]

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    class FileRenameInfo(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOLEAN),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    return SimpleNamespace(
        UnicodeString=UnicodeString,
        ObjectAttributes=ObjectAttributes,
        IoStatusBlock=IoStatusBlock,
        ByHandleFileInformation=ByHandleFileInformation,
        FileRenameInfo=FileRenameInfo,
        FileDispositionInfo=FileDispositionInfo,
    )


def _windows_api() -> SimpleNamespace:
    """Native Windows functions with their prototypes, bound once per process.

    The private kernel32 handle uses ``use_last_error`` so that
    ``ctypes.get_last_error()`` reports the failed call's own error, and its
    prototypes never touch the process-wide ``ctypes.windll`` functions. The
    ntdll calls return an NTSTATUS instead. The prototypes take the structures
    :func:`_windows_types` returns.
    """
    with _WINDOWS_CTYPES_LOCK:
        return _bind_windows_api()


@functools.cache
def _bind_windows_api() -> SimpleNamespace:
    """Bind the functions; only :func:`_windows_api` calls this, under the lock."""
    if os.name != "nt":
        raise OSError("Windows handle operations are unavailable on this platform")
    import ctypes
    from ctypes import wintypes

    types = _windows_types()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")

    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE

    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    get_file_information = kernel32.GetFileInformationByHandle
    get_file_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(types.ByHandleFileInformation)]
    get_file_information.restype = wintypes.BOOL

    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD

    set_file_information = kernel32.SetFileInformationByHandle
    set_file_information.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    set_file_information.restype = wintypes.BOOL

    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(types.ObjectAttributes),
        ctypes.POINTER(types.IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
    ]
    nt_create_file.restype = wintypes.LONG

    nt_set_information_file = ntdll.NtSetInformationFile
    nt_set_information_file.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(types.IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.c_int,
    ]
    nt_set_information_file.restype = wintypes.LONG

    rtl_nt_status_to_dos_error = ntdll.RtlNtStatusToDosError
    rtl_nt_status_to_dos_error.argtypes = [wintypes.LONG]
    rtl_nt_status_to_dos_error.restype = wintypes.ULONG

    return SimpleNamespace(
        invalid_handle=ctypes.c_void_p(-1).value,
        create_file=create_file,
        close_handle=close_handle,
        get_file_information=get_file_information,
        get_final_path=get_final_path,
        set_file_information=set_file_information,
        nt_create_file=nt_create_file,
        nt_set_information_file=nt_set_information_file,
        rtl_nt_status_to_dos_error=rtl_nt_status_to_dos_error,
    )


def _windows_last_error(message: str) -> OSError:
    """Return an ``OSError`` for the kernel32 call that just failed on this thread."""
    import ctypes

    return OSError(ctypes.get_last_error(), message)


def _windows_nt_status_error(status: int, message: str) -> OSError:
    """Return an ``OSError`` with the Win32 error code for a failed NTSTATUS."""
    return OSError(int(_windows_api().rtl_nt_status_to_dos_error(status)), message)


def _windows_open_handle(
    path: Path,
    *,
    access: int,
    share: int,
    disposition: int,
    flags: int,
) -> int:
    api = _windows_api()
    handle = api.create_file(os.fspath(path), access, share, None, disposition, flags, None)
    if handle == api.invalid_handle:
        raise _windows_last_error(f"Cannot open Windows filesystem handle: {path}")
    return int(handle)


def _windows_open_relative_handle(
    parent_handle: int,
    name: str,
    *,
    access: int,
    share: int,
    disposition: int,
    file_attributes: int,
    create_options: int,
    object_attributes_flags: int = _WINDOWS_OBJECT_ATTRIBUTES_FLAGS,
) -> int:
    """Open one path component relative to a held native directory handle."""
    import ctypes
    from ctypes import wintypes

    _validate_windows_path_component(name, label="Anchored path component")
    types = _windows_types()
    encoded_name = name.encode("utf-16-le")
    name_buffer = ctypes.create_unicode_buffer(name)
    unicode_name = types.UnicodeString(
        Length=len(encoded_name),
        MaximumLength=len(encoded_name) + ctypes.sizeof(wintypes.WCHAR),
        Buffer=ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    object_attributes = types.ObjectAttributes(
        Length=ctypes.sizeof(types.ObjectAttributes),
        RootDirectory=parent_handle,
        ObjectName=ctypes.pointer(unicode_name),
        Attributes=object_attributes_flags,
        SecurityDescriptor=None,
        SecurityQualityOfService=None,
    )
    io_status = types.IoStatusBlock()
    handle = wintypes.HANDLE()
    status = int(
        _windows_api().nt_create_file(
            ctypes.byref(handle),
            access,
            ctypes.byref(object_attributes),
            ctypes.byref(io_status),
            None,
            file_attributes,
            share,
            disposition,
            create_options,
            None,
            0,
        )
    )
    if status < 0:
        raise _windows_nt_status_error(status, f"Cannot open anchored Windows path component: {name}")
    if not handle.value:
        raise OSError("NtCreateFile succeeded without returning a file handle")
    return int(handle.value)


def _windows_open_discovery_handle(
    parent_handle: int,
    name: str,
) -> tuple[int, _WindowsHandleMetadata]:
    """Open one authored entry without following it, including exact alias reparses."""
    try:
        handle = _windows_open_relative_handle(
            parent_handle,
            name,
            access=_WINDOWS_DISCOVERY_ENTRY_ACCESS,
            share=_WINDOWS_SHARE_READ_WRITE,
            disposition=_WINDOWS_FILE_OPEN,
            file_attributes=0,
            create_options=_WINDOWS_DISCOVERY_ENTRY_OPTIONS,
            object_attributes_flags=_WINDOWS_OBJECT_ATTRIBUTES_FLAGS,
        )
    except OSError as no_reparse_error:
        # OBJ_DONT_REPARSE deliberately reports a reparse encounter instead of
        # returning a handle. Re-open the same single component with
        # FILE_OPEN_REPARSE_POINT while its parent remains pinned so we can
        # inspect (but never follow) the compatibility alias itself.
        try:
            handle = _windows_open_relative_handle(
                parent_handle,
                name,
                access=_WINDOWS_DISCOVERY_ENTRY_ACCESS,
                share=_WINDOWS_SHARE_READ_WRITE,
                disposition=_WINDOWS_FILE_OPEN,
                file_attributes=0,
                create_options=_WINDOWS_DISCOVERY_ENTRY_OPTIONS,
                object_attributes_flags=_WINDOWS_OBJ_CASE_INSENSITIVE,
            )
        except OSError:
            raise no_reparse_error from None
        try:
            metadata = _windows_handle_metadata(handle)
            if not metadata.is_reparse:
                raise no_reparse_error from None
            return handle, metadata
        except BaseException:
            _windows_close_handle(handle)
            raise

    try:
        return handle, _windows_handle_metadata(handle)
    except BaseException:
        _windows_close_handle(handle)
        raise


def _validate_windows_discovery_directory_snapshot(
    metadata: _WindowsHandleMetadata,
    relative_path: Path,
    expected: _WindowsHandleMetadata,
) -> None:
    if (
        not metadata.is_plain_directory
        or not metadata.same_identity(expected)
        or metadata.size != expected.size
        or metadata.last_write_time != expected.last_write_time
    ):
        label = relative_path.as_posix()
        raise SecurePathError(
            "unsafe_path",
            f"Tier 2 Windows directory changed during discovery: {label}",
            relative_path=label,
        )


def _validate_windows_entry_snapshot(
    metadata: os.stat_result,
    handle_metadata: _WindowsHandleMetadata,
    relative_path: Path,
) -> None:
    # The native no-follow handle is authoritative for reparses. Python's
    # Windows ``lstat`` can report a junction or symlink with a different mode
    # and link count; the pinned reparse is rejected or exact-alias validated
    # immediately by the caller, without descent or target reads.
    if handle_metadata.is_reparse:
        return
    if (
        stat_is_link_or_reparse(metadata)
        or stat.S_ISDIR(metadata.st_mode) != handle_metadata.is_directory
        or getattr(metadata, "st_nlink", 1) != handle_metadata.link_count
        or (not handle_metadata.is_directory and metadata.st_size != handle_metadata.size)
    ):
        raise SecurePathError(
            "unsafe_path",
            f"Unsafe Tier 2 Windows entry changed while being inspected: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )


def _windows_enumerate_pinned_directory_names(
    path: Path,
    handle: int,
    relative_path: Path,
    expected: _WindowsHandleMetadata,
    *,
    admission: _DiscoveryAdmission,
) -> tuple[list[str], _WindowsHandleMetadata]:
    """Enumerate names by path only while native handles pin every path component."""
    before = _windows_handle_metadata(handle)
    _validate_windows_discovery_directory_snapshot(before, relative_path, expected)
    names: list[str] = []
    try:
        with os.scandir(path) as iterator:
            for listed, entry in enumerate(iterator, start=1):
                admission.check_listing(relative_path, listed)
                names.append(entry.name)
    except SecurePathError:
        raise
    except OSError as exc:
        raise SecurePathError(
            "path_access_error",
            f"Cannot enumerate pinned Tier 2 Windows directory {relative_path.as_posix()}: {exc}",
            relative_path=relative_path.as_posix(),
        ) from exc
    names.sort()
    after = _windows_handle_metadata(handle)
    _validate_windows_discovery_directory_snapshot(after, relative_path, before)
    return names, after


def _windows_create_relative_file(parent_handle: int, name: str, *, access: int) -> int:
    """Create one exclusive regular file relative to a held Windows directory."""
    return _windows_open_relative_handle(
        parent_handle,
        name,
        access=access,
        share=0,  # no sharing while the stage handle is live
        disposition=_WINDOWS_FILE_CREATE,
        file_attributes=_WINDOWS_FILE_ATTRIBUTE_NORMAL,
        create_options=(
            _WINDOWS_FILE_NON_DIRECTORY_FILE
            | _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT
            | _WINDOWS_FILE_OPEN_REPARSE_POINT
            | _WINDOWS_FILE_WRITE_THROUGH
        ),
    )


def _windows_close_handle(handle: int) -> None:
    if not _windows_api().close_handle(handle):
        raise _windows_last_error("Cannot close Windows filesystem handle")


def _windows_handle_metadata(handle: int) -> _WindowsHandleMetadata:
    import ctypes

    information = _windows_types().ByHandleFileInformation()
    if not _windows_api().get_file_information(handle, ctypes.byref(information)):
        raise _windows_last_error("Cannot inspect open Windows filesystem handle")
    return _WindowsHandleMetadata(
        attributes=int(information.dwFileAttributes),
        volume_serial=int(information.dwVolumeSerialNumber),
        file_id=(int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow),
        size=(int(information.nFileSizeHigh) << 32) | int(information.nFileSizeLow),
        link_count=int(information.nNumberOfLinks),
        last_write_time=(int(information.ftLastWriteTime.dwHighDateTime) << 32)
        | int(information.ftLastWriteTime.dwLowDateTime),
    )


def _windows_final_path_from_handle(handle: int) -> Path:
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    length = _windows_api().get_final_path(handle, buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise _windows_last_error("Cannot resolve opened Windows filesystem handle")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def windows_final_path(descriptor: int) -> Path:
    """Return the final path of an open CRT descriptor on Windows.

    The extended-length prefix is removed, so the result compares with an
    ordinary absolute path. Raises ``OSError`` on other platforms or when the
    handle cannot be resolved.
    """
    if os.name != "nt":
        raise OSError("Windows handle verification is unavailable on this platform")
    import msvcrt

    return _windows_final_path_from_handle(msvcrt.get_osfhandle(descriptor))


def _verify_windows_handle_path(handle: int, expected: Path) -> None:
    expected_text = os.path.normcase(os.path.abspath(os.fspath(expected)))  # noqa: PTH100
    actual_text = os.path.normcase(os.path.abspath(os.fspath(_windows_final_path_from_handle(handle))))  # noqa: PTH100
    if actual_text != expected_text:
        raise SecurePathError(
            "unsafe_path",
            "Opened Windows handle resolves through a reparse point or unexpected path.",
        )


def _validate_windows_read_directory_handle(handle: int, relative_path: Path) -> _WindowsHandleMetadata:
    """Require one opened Windows traversal component to be a plain directory."""
    metadata = _windows_handle_metadata(handle)
    if not metadata.is_plain_directory:
        raise SecurePathError(
            "unsafe_path",
            f"Tier 2 path contains a non-directory or reparse component: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )
    return metadata


def _validate_windows_read_file_handle(
    handle: int,
    relative_path: Path,
    *,
    allow_hardlinks: bool = False,
) -> _WindowsHandleMetadata:
    """Require one selected Windows handle to be regular, no-follow, and single-link unless allowed."""
    metadata = _windows_handle_metadata(handle)
    if metadata.is_directory or metadata.is_reparse:
        raise SecurePathError(
            "unsafe_path",
            f"Refusing selected directory or reparse point: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )
    if not allow_hardlinks and metadata.link_count != 1:
        _raise_unsafe_file(relative_path, hardlink=True)
    return metadata


def _windows_open_anchored_directory_chain(
    path: Path,
    *,
    expected: os.stat_result,
) -> list[int]:
    """Pin an absolute directory from its volume/share anchor without following reparses."""
    absolute = _absolute_no_resolve(path)
    if not absolute.anchor:
        raise SecurePathError("unsafe_root", "Tier 2 Windows root has no filesystem anchor.")

    anchor = Path(absolute.anchor)
    handles: list[int] = []
    try:
        anchor_handle = _windows_open_handle(
            anchor,
            access=_WINDOWS_DIRECTORY_READ_ACCESS,
            share=_WINDOWS_SHARE_READ_WRITE,
            disposition=_WINDOWS_OPEN_EXISTING,
            flags=_WINDOWS_DIRECTORY_HANDLE_FLAGS,
        )
        handles.append(anchor_handle)
        _validate_windows_read_directory_handle(anchor_handle, anchor)

        current_path = anchor
        parent_handle = anchor_handle
        for component in absolute.parts[1:]:
            current_path /= component
            child_handle = _windows_open_relative_handle(
                parent_handle,
                component,
                access=_WINDOWS_DIRECTORY_READ_ACCESS,
                share=_WINDOWS_SHARE_READ_WRITE,
                disposition=_WINDOWS_FILE_OPEN,
                file_attributes=0,
                create_options=_WINDOWS_DIRECTORY_OPEN_OPTIONS,
            )
            handles.append(child_handle)
            _validate_windows_read_directory_handle(child_handle, current_path)
            parent_handle = child_handle

        try:
            declared = absolute.lstat()
        except OSError as exc:
            raise SecurePathError("unsafe_root", f"Cannot revalidate declared Tier 2 root: {exc}") from exc
        if stat_is_link_or_reparse(declared) or not stat.S_ISDIR(declared.st_mode):
            raise SecurePathError("unsafe_root", "Declared Tier 2 root became a reparse point or non-directory.")
        if not os.path.samestat(expected, declared):
            raise SecurePathError("unsafe_root", "Tier 2 root changed identity while native handles were opened.")
        return handles
    except BaseException:
        while handles:
            _windows_close_handle(handles.pop())
        raise


def _validate_windows_parent_handle(
    handle: int,
    expected_path: Path,
    original: _WindowsHandleMetadata | None = None,
) -> _WindowsHandleMetadata:
    """Require the held output parent to be a plain directory at its declared path, still ``original``."""
    metadata = _windows_handle_metadata(handle)
    if not metadata.is_plain_directory:
        raise SecurePathError("unsafe_path", "Output parent handle is a reparse point or non-directory.")
    if original is not None and not metadata.same_identity(original):
        raise SecurePathError("unsafe_path", "Output parent changed identity during the atomic write.")
    _verify_windows_handle_path(handle, expected_path)
    return metadata


def _validate_windows_regular_handle(
    handle: int,
    *,
    expected: _WindowsHandleMetadata | None,
    expected_size: int,
) -> _WindowsHandleMetadata:
    metadata = _windows_handle_metadata(handle)
    if metadata.is_directory or metadata.is_reparse:
        raise SecurePathError("unsafe_path", "Windows output handle is a directory or reparse point.")
    if metadata.link_count != 1:
        raise SecurePathError("unsafe_hardlink", "Windows output handle is hard-linked (link count > 1).")
    if metadata.size != expected_size:
        raise SecurePathError(
            "unsafe_path",
            f"Windows output size changed unexpectedly (expected {expected_size}, got {metadata.size}).",
        )
    if expected is not None and not metadata.same_identity(expected):
        raise SecurePathError("unsafe_path", "Windows output changed identity during the atomic write.")
    return metadata


def _inspect_destination_windows(path: Path) -> os.stat_result | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SecurePathError("path_access_error", f"Cannot inspect output destination: {exc}") from exc
    if stat_is_link_or_reparse(metadata):
        raise SecurePathError("unsafe_path", f"Destination is a symlink or reparse point: {path.name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise SecurePathError("unsafe_path", f"Destination is not a regular file: {path.name}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise SecurePathError("unsafe_hardlink", f"Destination is hard-linked (link count > 1): {path.name}")
    return metadata


def _validate_windows_destination_unchanged(
    before: os.stat_result | None,
    current: os.stat_result | None,
) -> None:
    if (before is None) != (current is None):
        raise SecurePathError("unsafe_path", "Windows output destination appeared or disappeared during the write.")
    if before is not None and current is not None and _snapshot_changed(current, before):
        raise SecurePathError("unsafe_path", "Windows output destination changed while output was prepared.")


def _validate_windows_path_component(name: str, *, label: str) -> None:
    """Reject Win32 normalization aliases, device names, ADS, and invalid UTF-16."""
    invalid_characters = '<>:"/\\|?*'
    stem = name.split(".", 1)[0].rstrip(" .").casefold()
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
        *(f"com{index}" for index in "¹²³"),
        *(f"lpt{index}" for index in "¹²³"),
    }
    try:
        utf16_units = len(name.encode("utf-16-le")) // 2
    except UnicodeEncodeError as exc:
        raise SecurePathError("unsafe_path", f"{label} has an unsafe Windows file name.") from exc
    if (
        not name
        or name in {".", ".."}
        or utf16_units > 255
        or any(ord(character) < 32 or character in invalid_characters for character in name)
        or name.endswith((" ", "."))
        or stem in reserved
    ):
        raise SecurePathError("unsafe_path", f"{label} has an unsafe Windows file name.")


def _rename_windows_handle(
    descriptor: int,
    parent_handle: int,
    destination_name: str,
    *,
    replace: bool,
) -> None:
    import ctypes
    import msvcrt

    types = _windows_types()
    encoded_name = destination_name.encode("utf-16-le")
    filename_offset = types.FileRenameInfo.FileName.offset
    buffer_size = max(ctypes.sizeof(types.FileRenameInfo), filename_offset + len(encoded_name))
    buffer = ctypes.create_string_buffer(buffer_size)
    information = ctypes.cast(buffer, ctypes.POINTER(types.FileRenameInfo)).contents
    information.ReplaceIfExists = int(replace)
    information.RootDirectory = parent_handle
    information.FileNameLength = len(encoded_name)
    ctypes.memmove(ctypes.addressof(buffer) + filename_offset, encoded_name, len(encoded_name))

    io_status = types.IoStatusBlock()
    status = int(
        _windows_api().nt_set_information_file(
            msvcrt.get_osfhandle(descriptor),
            ctypes.byref(io_status),
            buffer,
            buffer_size,
            _WINDOWS_FILE_RENAME_INFORMATION,
        )
    )
    if status < 0:
        raise _windows_nt_status_error(status, "Cannot rename Windows output through its parent handle")


def _mark_windows_handle_for_deletion(descriptor: int) -> None:
    """Best-effort handle-only cleanup for an unpublished Windows stage."""
    import ctypes
    import msvcrt

    disposition = _windows_types().FileDispositionInfo(DeleteFile=1)
    # Failure is deliberately non-fatal: leaving the held orphan is safer than
    # falling back to path cleanup that could delete an attacker-swapped name.
    _windows_api().set_file_information(
        msvcrt.get_osfhandle(descriptor),
        _WINDOWS_FILE_DISPOSITION_INFO,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    )


def _atomic_write_windows(path: Path, payload: bytes) -> None:
    import msvcrt

    absolute = _absolute_no_resolve(path)
    _validate_windows_path_component(absolute.name, label="Destination")
    _validate_windows_parent_components(absolute)
    before = _inspect_destination_windows(absolute)

    parent_handle = _windows_open_handle(
        absolute.parent,
        access=_WINDOWS_DIRECTORY_READ_ACCESS,
        # Deliberately omit FILE_SHARE_DELETE so the held parent cannot be
        # renamed or removed between validation and handle-relative publish.
        share=_WINDOWS_SHARE_READ_WRITE,
        disposition=_WINDOWS_OPEN_EXISTING,
        flags=_WINDOWS_DIRECTORY_HANDLE_FLAGS,
    )
    descriptor = -1
    temporary_path: Path | None = None
    publication_attempted = False
    try:
        parent_metadata = _validate_windows_parent_handle(parent_handle, absolute.parent)
        native_handle = -1
        for _attempt in range(128):
            temporary_path = absolute.parent / f".skillevaluator-{secrets.token_hex(8)}.tmp"
            try:
                native_handle = _windows_create_relative_file(
                    parent_handle,
                    temporary_path.name,
                    access=(
                        _WINDOWS_GENERIC_WRITE | _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_DELETE | _WINDOWS_SYNCHRONIZE
                    ),
                )
            except OSError as exc:
                if exc.errno in _WINDOWS_FILE_EXISTS_ERRORS:
                    continue
                raise
            break
        if native_handle < 0 or temporary_path is None:
            raise SecurePathError("path_access_error", "Cannot allocate a secure Windows temporary output file.")
        try:
            descriptor = msvcrt.open_osfhandle(native_handle, os.O_WRONLY | getattr(os, "O_BINARY", 0))
        except BaseException:
            _windows_close_handle(native_handle)
            raise

        raw_descriptor = msvcrt.get_osfhandle(descriptor)
        opened = os.fstat(descriptor)
        _validate_opened_file(opened, Path(temporary_path.name), None)
        opened_handle = _validate_windows_regular_handle(raw_descriptor, expected=None, expected_size=0)
        _verify_windows_handle_path(raw_descriptor, temporary_path)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        prepared = os.fstat(descriptor)
        _validate_opened_file(prepared, Path(temporary_path.name), None)
        if prepared.st_size != len(payload):
            raise SecurePathError("unsafe_path", "Temporary Windows output size changed while being written.")
        _validate_windows_regular_handle(raw_descriptor, expected=opened_handle, expected_size=len(payload))
        _validate_windows_parent_components(absolute)
        _validate_windows_parent_handle(parent_handle, absolute.parent, parent_metadata)
        destination = _inspect_destination_windows(absolute)
        _validate_windows_destination_unchanged(before, destination)
        # From this point an asynchronous exception cannot tell whether the
        # kernel completed publication. Never disposition-delete the handle
        # after the replacement attempt begins.
        publication_attempted = True
        try:
            _rename_windows_handle(descriptor, parent_handle, absolute.name, replace=True)
        except OSError:
            # A synchronous FALSE return proves the rename did not publish;
            # handle-only cleanup is safe. BaseException remains ambiguous.
            publication_attempted = False
            raise
        temporary_path = None
        _verify_windows_handle_path(raw_descriptor, absolute)
        published = os.fstat(descriptor)
        _validate_opened_file(published, Path(absolute.name), None)
        if published.st_size != len(payload):
            raise SecurePathError("unsafe_path", "Published Windows output size changed during replacement.")
        _validate_windows_regular_handle(raw_descriptor, expected=opened_handle, expected_size=len(payload))
        _validate_windows_parent_handle(parent_handle, absolute.parent, parent_metadata)
    except OSError as exc:
        raise SecurePathError("path_access_error", f"Cannot securely write Windows output: {exc}") from exc
    finally:
        try:
            if descriptor >= 0:
                if temporary_path is not None and not publication_attempted:
                    _mark_windows_handle_for_deletion(descriptor)
                os.close(descriptor)
        finally:
            _windows_close_handle(parent_handle)
