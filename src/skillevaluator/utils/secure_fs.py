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
import os
import secrets
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

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
_WINDOWS_SHARE_READ = 0x1
_WINDOWS_SHARE_READ_WRITE = _WINDOWS_SHARE_READ | 0x2
_WINDOWS_FILE_OPEN = 1
_WINDOWS_FILE_CREATE = 2
_WINDOWS_FILE_ATTRIBUTE_NORMAL = 0x80
_WINDOWS_FILE_DIRECTORY_FILE = 0x1
_WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT = 0x20
_WINDOWS_FILE_NON_DIRECTORY_FILE = 0x40
_WINDOWS_FILE_OPEN_FOR_BACKUP_INTENT = 0x00004000
_WINDOWS_FILE_OPEN_REPARSE_POINT = 0x00200000
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


def stat_is_link_or_reparse(metadata: os.stat_result) -> bool:
    """Return whether metadata identifies a symlink or Windows reparse point."""
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def is_link_or_reparse(path: Path) -> bool:
    """Inspect one path without following it."""
    try:
        return stat_is_link_or_reparse(path.lstat())
    except OSError as exc:
        raise SecurePathError("path_access_error", f"Cannot inspect path safely: {path.name}: {exc}") from exc


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


def discover_secure_files(
    root: Path,
    *,
    selected: Callable[[Path], bool],
    excluded_dirs: Iterable[str] = (),
    max_paths: int,
    max_depth: int | None = None,
    allow_context_alias: bool = True,
) -> list[SecureFile]:
    """Discover selected files below ``root`` without following redirects.

    Excluded directories are pruned before they consume the path budget.
    Every other authored entry consumes the budget. File and directory redirects
    fail closed without target content reads except for the exact contained
    ``CLAUDE.md -> AGENTS.md`` compatibility alias, whose regular target must be
    independently discovered; only that target is returned and read.
    """
    if max_paths < 1:
        raise ValueError("max_paths must be positive")
    max_depth = _validate_discovery_depth(max_depth)
    root = _absolute_no_resolve(root)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise SecurePathError("invalid_root", f"Cannot inspect Tier 2 root: {exc}") from exc
    if stat_is_link_or_reparse(root_metadata):
        raise SecurePathError("unsafe_root", f"Tier 2 root is a symlink or reparse point: {root.name}")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise SecurePathError("invalid_root", f"Tier 2 root is not a regular directory: {root}")

    excluded = frozenset(excluded_dirs)
    files: list[SecureFile] = []
    # Keep exact authored spelling. ``WindowsPath`` keys compare
    # case-insensitively, which would otherwise let ``agents.md`` satisfy the
    # required exact ``CLAUDE.md -> AGENTS.md`` compatibility target.
    regular_by_relative: dict[str, os.stat_result] = {}
    pending_aliases: list[tuple[Path, Path]] = []
    discovered_paths = 0

    def consume_path(relative: Path) -> None:
        nonlocal discovered_paths
        discovered_paths += 1
        if discovered_paths > max_paths:
            raise SecurePathError(
                "path_count_limit",
                f"Tier 2 tree exceeds the path limit of {max_paths} entries.",
                relative_path=relative.as_posix(),
                metadata={"actual": discovered_paths, "limit": max_paths},
            )

    def record_file(
        relative: Path,
        metadata: os.stat_result,
        read_alias_target: Callable[[], str],
        *,
        selected_result: bool | None = None,
    ) -> None:
        is_selected = selected(relative) if selected_result is None else selected_result
        if stat_is_link_or_reparse(metadata):
            target: Path | None = None
            if allow_context_alias and relative.name == "CLAUDE.md":
                try:
                    target = _compatibility_alias_target(read_alias_target(), relative)
                except OSError as exc:
                    raise SecurePathError(
                        "unsafe_path",
                        f"Cannot inspect selected compatibility alias: {relative.as_posix()}: {exc}",
                        relative_path=relative.as_posix(),
                    ) from exc
            if target is not None:
                if getattr(metadata, "st_nlink", 1) != 1:
                    _raise_unsafe_file(relative, hardlink=True)
                pending_aliases.append((relative, target))
                return
            raise SecurePathError(
                "unsafe_path",
                f"Refusing symlink or reparse point: {relative.as_posix()}",
                relative_path=relative.as_posix(),
            )
        if stat.S_ISREG(metadata.st_mode):
            regular_by_relative[relative.as_posix()] = metadata
        if not is_selected:
            return
        if not stat.S_ISREG(metadata.st_mode):
            _raise_unsafe_file(relative)
        if getattr(metadata, "st_nlink", 1) != 1:
            _raise_unsafe_file(relative, hardlink=True)
        secure_file = SecureFile(root, root / relative, relative, metadata)
        files.append(secure_file)

    if os.name == "posix":
        if not (_OPEN_SUPPORTS_DIR_FD and _READLINK_SUPPORTS_DIR_FD and _SCANDIR_SUPPORTS_FD):
            raise SecurePathError(
                "secure_open_unavailable",
                "This platform cannot guarantee descriptor-anchored no-follow Tier 2 discovery.",
            )
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        root_fd = _open_absolute_directory_posix(root)
        frames = [_DirectoryFrame(root_fd, Path(), root_metadata)]
        try:
            # A descriptor stack makes the walk linear: root is opened once,
            # each descended child is opened once relative to its held parent,
            # and only the descriptors on the active DFS path remain live.
            while frames:
                frame = frames[-1]
                if frame.children is None:
                    current = os.fstat(frame.descriptor)
                    _validate_directory_snapshot(current, frame.relative_path, frame.expected)
                    directory_entries: list[tuple[str, os.stat_result]] = []
                    file_entries: list[tuple[str, os.stat_result, bool | None]] = []
                    try:
                        with os.scandir(frame.descriptor) as iterator:
                            for entry in iterator:
                                relative = frame.relative_path / entry.name
                                try:
                                    metadata = entry.stat(follow_symlinks=False)
                                except OSError as exc:
                                    raise SecurePathError(
                                        "path_access_error",
                                        f"Cannot inspect Tier 2 path {relative.as_posix()}: {exc}",
                                        relative_path=relative.as_posix(),
                                    ) from exc
                                entry_selected = selected(relative)
                                linked_or_reparse = stat_is_link_or_reparse(metadata)
                                if stat.S_ISDIR(metadata.st_mode):
                                    if linked_or_reparse:
                                        raise SecurePathError(
                                            "unsafe_path",
                                            "Refusing linked directory or reparse point before descent: "
                                            f"{relative.as_posix()}",
                                            relative_path=relative.as_posix(),
                                        )
                                    if entry.name in excluded:
                                        continue
                                    if entry_selected:
                                        _raise_unsafe_file(relative)
                                    consume_path(relative)
                                    directory_depth = len(relative.parts)
                                    if directory_depth > MAX_SECURE_DIRECTORY_DEPTH:
                                        _raise_directory_depth_limit(relative)
                                    if max_depth is None or directory_depth < max_depth:
                                        directory_entries.append((entry.name, metadata))
                                    continue
                                consume_path(relative)
                                file_entries.append((entry.name, metadata, entry_selected))
                    except SecurePathError:
                        raise
                    except OSError as exc:
                        raise SecurePathError(
                            "path_access_error",
                            f"Cannot enumerate Tier 2 directory {frame.relative_path.as_posix()}: {exc}",
                            relative_path=frame.relative_path.as_posix(),
                        ) from exc

                    after_scan = os.fstat(frame.descriptor)
                    _validate_directory_snapshot(after_scan, frame.relative_path, current)
                    for name, metadata, selected_result in sorted(file_entries, key=lambda item: item[0]):
                        relative = frame.relative_path / name
                        record_file(
                            relative,
                            metadata,
                            lambda name=name, directory_fd=frame.descriptor: os.readlink(
                                name,
                                dir_fd=directory_fd,
                            ),
                            selected_result=selected_result,
                        )
                    stable = os.fstat(frame.descriptor)
                    _validate_directory_snapshot(stable, frame.relative_path, after_scan)
                    frame.expected = stable
                    frame.children = sorted(directory_entries, key=lambda item: item[0])
                    continue

                if frame.next_child < len(frame.children):
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
                        child_fd = os.open(name, directory_flags, dir_fd=frame.descriptor)
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
                    frames.append(
                        _DirectoryFrame(
                            child_fd,
                            relative,
                            opened,
                            parent_name=name,
                        )
                    )
                    continue

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
                else:
                    parent = frames[-2]
                    try:
                        parent_entry = os.stat(
                            frame.parent_name,
                            dir_fd=parent.descriptor,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise SecurePathError(
                            "unsafe_path",
                            (
                                "Cannot revalidate Tier 2 directory after its subtree: "
                                f"{frame.relative_path.as_posix()}: {exc}"
                            ),
                            relative_path=frame.relative_path.as_posix(),
                        ) from exc
                    _validate_directory_snapshot(parent_entry, frame.relative_path, stable)

                finished = frames.pop()
                os.close(finished.descriptor)
        finally:
            while frames:
                os.close(frames.pop().descriptor)
    elif os.name == "nt":
        root_handles: list[int] = []
        frames: list[_WindowsDirectoryFrame] = []
        try:
            root_handles = _windows_open_anchored_directory_chain(root, expected=root_metadata)
            root_handle = root_handles[-1]
            root_snapshot = _windows_handle_metadata(root_handle)
            _validate_windows_read_directory_handle(root_handle, Path())
            frames.append(
                _WindowsDirectoryFrame(
                    root_handle,
                    root,
                    Path(),
                    root_snapshot,
                    owns_handle=False,
                )
            )

            while frames:
                frame = frames[-1]
                if frame.children is None:
                    names, stable = _windows_enumerate_pinned_directory_names(
                        frame.path,
                        frame.handle,
                        frame.relative_path,
                        frame.expected,
                        max_names=max_paths + len(excluded),
                        path_limit=max_paths,
                    )
                    directory_entries: list[tuple[str, _WindowsHandleMetadata]] = []
                    file_entries: list[tuple[str, os.stat_result, str | None]] = []

                    for name in names:
                        path = frame.path / name
                        relative = frame.relative_path / name
                        entry_handle = -1
                        try:
                            try:
                                entry_handle, handle_metadata = _windows_open_discovery_handle(frame.handle, name)
                            except OSError as exc:
                                raise SecurePathError(
                                    "path_access_error",
                                    f"Cannot securely inspect Tier 2 Windows path {relative.as_posix()}: {exc}",
                                    relative_path=relative.as_posix(),
                                ) from exc
                            try:
                                metadata = path.lstat()
                            except OSError as exc:
                                raise SecurePathError(
                                    "path_access_error",
                                    f"Cannot inspect pinned Tier 2 Windows path {relative.as_posix()}: {exc}",
                                    relative_path=relative.as_posix(),
                                ) from exc
                            _validate_windows_entry_snapshot(metadata, handle_metadata, relative)

                            is_reparse = bool(handle_metadata.attributes & 0x400)
                            is_directory = bool(handle_metadata.attributes & 0x10)
                            if is_reparse and is_directory:
                                raise SecurePathError(
                                    "unsafe_path",
                                    f"Refusing linked directory or reparse point before descent: {relative.as_posix()}",
                                    relative_path=relative.as_posix(),
                                )
                            if is_directory:
                                if name in excluded:
                                    continue
                                directory_entries.append((name, handle_metadata))
                                continue

                            alias_target: str | None = None
                            if is_reparse and allow_context_alias and relative.name == "CLAUDE.md":
                                try:
                                    alias_target = os.readlink(path)  # noqa: PTH115
                                except OSError as exc:
                                    raise SecurePathError(
                                        "unsafe_path",
                                        f"Cannot inspect selected compatibility alias: {relative.as_posix()}: {exc}",
                                        relative_path=relative.as_posix(),
                                    ) from exc
                            file_entries.append((name, metadata, alias_target))
                        finally:
                            if entry_handle >= 0:
                                _windows_close_handle(entry_handle)

                    kept_directories: list[tuple[str, _WindowsHandleMetadata]] = []
                    for name, handle_metadata in directory_entries:
                        relative = frame.relative_path / name
                        if selected(relative):
                            _raise_unsafe_file(relative)
                        consume_path(relative)
                        directory_depth = len(relative.parts)
                        if directory_depth > MAX_SECURE_DIRECTORY_DEPTH:
                            _raise_directory_depth_limit(relative)
                        if max_depth is None or directory_depth < max_depth:
                            kept_directories.append((name, handle_metadata))

                    for name, metadata, alias_target in file_entries:
                        relative = frame.relative_path / name
                        consume_path(relative)
                        record_file(
                            relative,
                            metadata,
                            lambda alias_target=alias_target: alias_target or "",
                            selected_result=selected(relative),
                        )

                    current = _windows_handle_metadata(frame.handle)
                    _validate_windows_discovery_directory_snapshot(current, frame.relative_path, stable)
                    frame.expected = current
                    frame.children = kept_directories
                    continue

                if frame.next_child < len(frame.children):
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
                    frames.append(
                        _WindowsDirectoryFrame(
                            child_handle,
                            frame.path / name,
                            relative,
                            opened,
                            owns_handle=True,
                        )
                    )
                    continue

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
    else:
        raise SecurePathError(
            "secure_open_unavailable",
            "This platform cannot guarantee no-follow Tier 2 discovery.",
        )

    for alias, target in pending_aliases:
        target_metadata = regular_by_relative.get(target.as_posix())
        if target_metadata is None or getattr(target_metadata, "st_nlink", 1) != 1:
            raise SecurePathError(
                "unsafe_path",
                f"Compatibility alias target is not an independently enumerated regular file: {alias.as_posix()}",
                relative_path=alias.as_posix(),
            )

    return sorted(files, key=lambda item: item.rel_path)


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
        try:
            metadata = self.root.lstat()
        except OSError as exc:
            raise SecurePathError("invalid_root", f"Cannot inspect Tier 2 root: {exc}") from exc
        if stat_is_link_or_reparse(metadata):
            raise SecurePathError("unsafe_root", f"Tier 2 root is a symlink or reparse point: {self.root.name}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise SecurePathError("invalid_root", f"Tier 2 root is not a regular directory: {self.root}")
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

    def read_bytes(
        self,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
    ) -> tuple[bytes, os.stat_result]:
        """Read one bounded regular single-link file without following redirects."""
        if not self._entered:
            raise SecurePathError("secure_open_unavailable", "Secure Tier 2 root context is not active.")
        relative_path = _relative_path(relative_path)
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if os.name == "posix":
            descriptor = self._open_posix(relative_path, expected)
        elif os.name == "nt":
            descriptor = self._open_windows(relative_path, expected)
        else:
            raise SecurePathError("secure_open_unavailable", "Secure no-follow reads are unavailable.")

        try:
            opened = os.fstat(descriptor)
            # Windows discovery identity is revalidated with path ``lstat``
            # inside ``_open_windows``; CRT descriptor identity fields are not
            # comparable to that path-stat snapshot. POSIX uses one stat
            # family for both phases and can compare directly here.
            _validate_opened_file(opened, relative_path, expected if os.name == "posix" else None)
            if opened.st_size > max_bytes:
                raise SecurePathError(
                    "file_size_limit",
                    f"Selected file exceeds the {max_bytes}-byte limit: {relative_path.as_posix()}",
                    relative_path=relative_path.as_posix(),
                    metadata={"actual_bytes": opened.st_size, "limit_bytes": max_bytes},
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(65_536, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise SecurePathError(
                        "file_size_limit",
                        f"Selected file exceeds the {max_bytes}-byte limit: {relative_path.as_posix()}",
                        relative_path=relative_path.as_posix(),
                        metadata={"actual_bytes": total, "limit_bytes": max_bytes},
                    )
            after = os.fstat(descriptor)
            _validate_opened_file(after, relative_path, opened)
            return b"".join(chunks), opened
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
        if file.root != self.root:
            raise SecurePathError("unsafe_path", "Secure file belongs to a different Tier 2 root.")
        return self.read_text(file.relative_path, max_bytes, expected=file.metadata)

    def _open_posix(self, relative_path: Path, expected: os.stat_result | None) -> int:
        if self._root_fd is None:
            raise SecurePathError("secure_open_unavailable", "Tier 2 root descriptor is unavailable.")
        directory_fd = os.dup(self._root_fd)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
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
                    child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
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
            _validate_opened_file(before, relative_path, expected)
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
                _validate_opened_file(os.fstat(descriptor), relative_path, before)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
        finally:
            os.close(directory_fd)

    def _open_windows(self, relative_path: Path, expected: os.stat_result | None) -> int:
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
            _validate_opened_file(before_open, relative_path, expected)

            native_file_handle = _windows_open_relative_handle(
                parent_handle,
                relative_path.name,
                access=_WINDOWS_FILE_READ_ACCESS,
                share=_WINDOWS_SHARE_READ,
                disposition=_WINDOWS_FILE_OPEN,
                file_attributes=0,
                create_options=_WINDOWS_FILE_OPEN_OPTIONS,
            )
            _validate_windows_read_file_handle(native_file_handle, relative_path)
            try:
                after_open = declared_path.lstat()
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot revalidate selected Tier 2 file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(after_open, relative_path, before_open)
            descriptor = msvcrt.open_osfhandle(
                native_file_handle,
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0),
            )
            native_file_handle = -1  # ownership transferred to the CRT descriptor
            opened = os.fstat(descriptor)
            _validate_opened_file(opened, relative_path, None)
            return descriptor
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
) -> None:
    if stat_is_link_or_reparse(metadata):
        raise SecurePathError(
            "unsafe_path",
            f"Refusing selected symlink or reparse point: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )
    if not stat.S_ISREG(metadata.st_mode):
        _raise_unsafe_file(relative_path)
    if getattr(metadata, "st_nlink", 1) != 1:
        _raise_unsafe_file(relative_path, hardlink=True)
    if expected is not None:
        changed = not os.path.samestat(metadata, expected)
        for attribute in ("st_size", "st_mtime_ns", "st_ctime_ns"):
            if getattr(metadata, attribute, None) != getattr(expected, attribute, None):
                changed = True
        if changed:
            raise SecurePathError(
                "unsafe_path",
                f"Selected Tier 2 file changed identity or contents while being opened: {relative_path.as_posix()}",
                relative_path=relative_path.as_posix(),
            )


def _validate_directory_snapshot(
    metadata: os.stat_result,
    relative_path: Path,
    expected: os.stat_result,
) -> None:
    """Require one regular directory identity and entry snapshot to stay stable."""
    changed = (
        stat_is_link_or_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or not os.path.samestat(metadata, expected)
    )
    for attribute in ("st_size", "st_mtime_ns", "st_ctime_ns"):
        if getattr(metadata, attribute, None) != getattr(expected, attribute, None):
            changed = True
    if changed:
        label = relative_path.as_posix()
        raise SecurePathError(
            "unsafe_path",
            f"Tier 2 directory snapshot changed during discovery: {label}",
            relative_path=label,
        )


def _open_absolute_directory_posix(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
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
    before: os.stat_result | None = None
    opened: os.stat_result | None = None
    written_metadata: os.stat_result | None = None
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
    absolute = _absolute_no_resolve(path)
    current = Path(absolute.anchor)
    for component in absolute.parent.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SecurePathError("path_access_error", f"Cannot inspect parent directory: {exc}") from exc
        if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise SecurePathError(
                "unsafe_path",
                f"Path contains a symlink, junction, reparse point, or non-directory component: {current.name}",
            )


def _windows_kernel32():
    if os.name != "nt":
        raise OSError("Windows handle operations are unavailable on this platform")
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)


def _windows_raise_last_error(message: str) -> OSError:
    import ctypes

    error = ctypes.get_last_error()
    return OSError(error, message)


def _windows_open_handle(
    path: Path,
    *,
    access: int,
    share: int,
    disposition: int,
    flags: int,
) -> int:
    import ctypes
    from ctypes import wintypes

    kernel32 = _windows_kernel32()
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
    handle = create_file(os.fspath(path), access, share, None, disposition, flags, None)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise _windows_raise_last_error(f"Cannot open Windows filesystem handle: {path}")
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

    class _UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class _ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class _IoStatusValue(ctypes.Union):
        _fields_ = [("Status", wintypes.LONG), ("Pointer", wintypes.LPVOID)]  # noqa: RUF012

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [("Value", _IoStatusValue), ("Information", ctypes.c_size_t)]

    encoded_name = name.encode("utf-16-le")
    name_buffer = ctypes.create_unicode_buffer(name)
    unicode_name = _UnicodeString(
        Length=len(encoded_name),
        MaximumLength=len(encoded_name) + ctypes.sizeof(wintypes.WCHAR),
        Buffer=ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    object_attributes = _ObjectAttributes(
        Length=ctypes.sizeof(_ObjectAttributes),
        RootDirectory=parent_handle,
        ObjectName=ctypes.pointer(unicode_name),
        Attributes=object_attributes_flags,
        SecurityDescriptor=None,
        SecurityQualityOfService=None,
    )
    io_status = _IoStatusBlock()
    handle = wintypes.HANDLE()

    ntdll = ctypes.WinDLL("ntdll")
    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_ObjectAttributes),
        ctypes.POINTER(_IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
    ]
    nt_create_file.restype = wintypes.LONG
    status = int(
        nt_create_file(
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
        rtl_status_to_error = ntdll.RtlNtStatusToDosError
        rtl_status_to_error.argtypes = [wintypes.LONG]
        rtl_status_to_error.restype = wintypes.ULONG
        error = int(rtl_status_to_error(status))
        raise OSError(error, f"Cannot open anchored Windows path component: {name}")
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
            if not metadata.attributes & 0x400:
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
    directory_attribute = 0x10
    reparse_attribute = 0x400
    changed = (
        metadata.attributes & reparse_attribute
        or not metadata.attributes & directory_attribute
        or metadata.volume_serial != expected.volume_serial
        or metadata.file_id != expected.file_id
        or metadata.size != expected.size
        or metadata.last_write_time != expected.last_write_time
    )
    if changed:
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
    handle_is_reparse = bool(handle_metadata.attributes & 0x400)
    handle_is_directory = bool(handle_metadata.attributes & 0x10)
    # The native no-follow handle is authoritative for reparses. Python's
    # Windows ``lstat`` can report a junction or symlink with a different mode
    # and link count; the pinned reparse is rejected or exact-alias validated
    # immediately by the caller, without descent or target reads.
    if handle_is_reparse:
        return
    changed = stat_is_link_or_reparse(metadata) != handle_is_reparse or (
        stat.S_ISDIR(metadata.st_mode) != handle_is_directory
    )
    if getattr(metadata, "st_nlink", 1) != handle_metadata.link_count:
        changed = True
    if not handle_is_directory and metadata.st_size != handle_metadata.size:
        changed = True
    if changed:
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
    max_names: int | None = None,
    path_limit: int | None = None,
) -> tuple[list[str], _WindowsHandleMetadata]:
    """Enumerate names by path only while native handles pin every path component."""
    before = _windows_handle_metadata(handle)
    _validate_windows_discovery_directory_snapshot(before, relative_path, expected)
    try:
        with os.scandir(path) as iterator:
            names: list[str] = []
            for entry in iterator:
                names.append(entry.name)
                if max_names is not None and len(names) > max_names:
                    limit = path_limit if path_limit is not None else max_names
                    raise SecurePathError(
                        "path_count_limit",
                        f"Tier 2 tree exceeds the path limit of {limit} entries.",
                        relative_path=relative_path.as_posix(),
                        metadata={"actual": len(names), "limit": limit},
                    )
            names.sort()
    except SecurePathError:
        raise
    except OSError as exc:
        raise SecurePathError(
            "path_access_error",
            f"Cannot enumerate pinned Tier 2 Windows directory {relative_path.as_posix()}: {exc}",
            relative_path=relative_path.as_posix(),
        ) from exc
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
            | 0x2  # FILE_WRITE_THROUGH
        ),
    )


def _windows_close_handle(handle: int) -> None:
    from ctypes import wintypes

    kernel32 = _windows_kernel32()
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        raise _windows_raise_last_error("Cannot close Windows filesystem handle")


def _windows_handle_metadata(handle: int) -> _WindowsHandleMetadata:
    import ctypes
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
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

    kernel32 = _windows_kernel32()
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ByHandleFileInformation)]
    get_information.restype = wintypes.BOOL
    information = _ByHandleFileInformation()
    if not get_information(handle, ctypes.byref(information)):
        raise _windows_raise_last_error("Cannot inspect open Windows filesystem handle")
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
    from ctypes import wintypes

    kernel32 = _windows_kernel32()
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_final_path(handle, buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise _windows_raise_last_error("Cannot resolve opened Windows filesystem handle")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


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
    directory_attribute = 0x10
    reparse_attribute = 0x400
    if metadata.attributes & reparse_attribute or not metadata.attributes & directory_attribute:
        raise SecurePathError(
            "unsafe_path",
            f"Tier 2 path contains a non-directory or reparse component: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )
    return metadata


def _validate_windows_read_file_handle(handle: int, relative_path: Path) -> _WindowsHandleMetadata:
    """Require one selected Windows handle to be regular, single-link, and no-follow."""
    metadata = _windows_handle_metadata(handle)
    directory_attribute = 0x10
    reparse_attribute = 0x400
    if metadata.attributes & (directory_attribute | reparse_attribute):
        raise SecurePathError(
            "unsafe_path",
            f"Refusing selected directory or reparse point: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )
    if metadata.link_count != 1:
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
            disposition=3,  # OPEN_EXISTING for CreateFileW
            flags=0x02000000 | _WINDOWS_FILE_OPEN_REPARSE_POINT,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
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


def _validate_windows_parent_handle(handle: int, expected: Path, original: _WindowsHandleMetadata | None) -> None:
    metadata = _windows_handle_metadata(handle)
    directory_attribute = 0x10
    reparse_attribute = 0x400
    if metadata.attributes & reparse_attribute or not metadata.attributes & directory_attribute:
        raise SecurePathError("unsafe_path", "Output parent handle is a reparse point or non-directory.")
    if original is not None and (
        metadata.volume_serial != original.volume_serial or metadata.file_id != original.file_id
    ):
        raise SecurePathError("unsafe_path", "Output parent changed identity during the atomic write.")
    _verify_windows_handle_path(handle, expected)


def _validate_windows_regular_handle(
    handle: int,
    *,
    expected: _WindowsHandleMetadata | None,
    expected_size: int,
) -> _WindowsHandleMetadata:
    metadata = _windows_handle_metadata(handle)
    directory_attribute = 0x10
    reparse_attribute = 0x400
    if metadata.attributes & (directory_attribute | reparse_attribute):
        raise SecurePathError("unsafe_path", "Windows output handle is a directory or reparse point.")
    if metadata.link_count != 1:
        raise SecurePathError("unsafe_hardlink", "Windows output handle is hard-linked (link count > 1).")
    if metadata.size != expected_size:
        raise SecurePathError(
            "unsafe_path",
            f"Windows output size changed unexpectedly (expected {expected_size}, got {metadata.size}).",
        )
    if expected is not None and (
        metadata.volume_serial != expected.volume_serial or metadata.file_id != expected.file_id
    ):
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
    if before is None or current is None:
        return
    changed = not os.path.samestat(before, current)
    for attribute in ("st_size", "st_mtime_ns", "st_ctime_ns"):
        if getattr(before, attribute, None) != getattr(current, attribute, None):
            changed = True
    if changed:
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


def _validate_windows_output_name(name: str) -> None:
    _validate_windows_path_component(name, label="Destination")


def _rename_windows_handle(
    descriptor: int,
    parent_handle: int,
    destination_name: str,
    *,
    replace: bool,
) -> None:
    import ctypes
    from ctypes import wintypes

    class _FileRenameInfo(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOLEAN),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    class _IoStatusValue(ctypes.Union):
        _fields_ = [("Status", wintypes.LONG), ("Pointer", wintypes.LPVOID)]  # noqa: RUF012

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [("Value", _IoStatusValue), ("Information", ctypes.c_size_t)]

    encoded_name = destination_name.encode("utf-16-le")
    filename_offset = _FileRenameInfo.FileName.offset
    buffer_size = max(ctypes.sizeof(_FileRenameInfo), filename_offset + len(encoded_name))
    buffer = ctypes.create_string_buffer(buffer_size)
    information = ctypes.cast(buffer, ctypes.POINTER(_FileRenameInfo)).contents
    information.ReplaceIfExists = int(replace)
    information.RootDirectory = parent_handle
    information.FileNameLength = len(encoded_name)
    ctypes.memmove(ctypes.addressof(buffer) + filename_offset, encoded_name, len(encoded_name))

    import msvcrt

    io_status = _IoStatusBlock()
    ntdll = ctypes.WinDLL("ntdll")
    set_information = ntdll.NtSetInformationFile
    set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.c_int,
    ]
    set_information.restype = wintypes.LONG
    status = int(
        set_information(
            msvcrt.get_osfhandle(descriptor),
            ctypes.byref(io_status),
            buffer,
            buffer_size,
            10,  # FileRenameInformation
        )
    )
    if status < 0:
        rtl_status_to_error = ntdll.RtlNtStatusToDosError
        rtl_status_to_error.argtypes = [wintypes.LONG]
        rtl_status_to_error.restype = wintypes.ULONG
        error = int(rtl_status_to_error(status))
        raise OSError(error, "Cannot rename Windows output through its parent handle")


def _mark_windows_handle_for_deletion(descriptor: int) -> None:
    """Best-effort handle-only cleanup for an unpublished Windows stage."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    kernel32 = _windows_kernel32()
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    set_information.restype = wintypes.BOOL
    disposition = _FileDispositionInfo(DeleteFile=1)
    # Failure is deliberately non-fatal: leaving the held orphan is safer than
    # falling back to path cleanup that could delete an attacker-swapped name.
    set_information(
        msvcrt.get_osfhandle(descriptor),
        4,  # FileDispositionInfo
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    )


def _atomic_write_windows(path: Path, payload: bytes) -> None:
    import msvcrt

    absolute = _absolute_no_resolve(path)
    _validate_windows_output_name(absolute.name)
    _validate_windows_parent_components(absolute)
    before = _inspect_destination_windows(absolute)

    file_read_attributes = 0x80
    file_traverse = 0x20
    synchronize = 0x100000
    delete = 0x10000
    generic_write = 0x40000000
    share_read_write = 0x1 | 0x2
    open_existing = 3
    file_flag_open_reparse_point = 0x00200000
    file_flag_backup_semantics = 0x02000000

    parent_handle = _windows_open_handle(
        absolute.parent,
        access=file_read_attributes | file_traverse | synchronize,
        # Deliberately omit FILE_SHARE_DELETE so the held parent cannot be
        # renamed or removed between validation and handle-relative publish.
        share=share_read_write,
        disposition=open_existing,
        flags=file_flag_backup_semantics | file_flag_open_reparse_point,
    )
    descriptor = -1
    temporary_path: Path | None = None
    publication_attempted = False
    try:
        parent_metadata = _windows_handle_metadata(parent_handle)
        _validate_windows_parent_handle(parent_handle, absolute.parent, parent_metadata)
        native_handle = -1
        for _attempt in range(128):
            temporary_path = absolute.parent / f".skillevaluator-{secrets.token_hex(8)}.tmp"
            try:
                native_handle = _windows_create_relative_file(
                    parent_handle,
                    temporary_path.name,
                    access=generic_write | file_read_attributes | delete | synchronize,
                )
            except OSError as exc:
                if exc.errno in {80, 183}:  # file already exists
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


def _windows_final_path(descriptor: int) -> Path:
    if os.name != "nt":
        raise OSError("Windows handle verification is unavailable on this platform")
    import msvcrt

    return _windows_final_path_from_handle(msvcrt.get_osfhandle(descriptor))
