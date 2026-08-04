# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed filesystem primitives for untrusted Tier 2 inputs.

Discovery is lexical and no-descent: irrelevant links are counted but never
content-opened, read, or descended. One metadata-only target-type query follows
an irrelevant redirect just enough to distinguish an existing linked directory
from an irrelevant file link; the directory redirect is rejected immediately.
Selected files are read through
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
    allow_context_alias: bool = False,
) -> list[SecureFile]:
    """Discover selected files below ``root`` without following redirects.

    Excluded directories are pruned before they consume the path budget.
    Every other authored entry consumes the budget, including irrelevant links.
    Irrelevant links are never content-opened, read, or descended. A metadata-
    only target-type query follows them just enough to identify existing
    directory redirects, which fail before pruning/descent. A selected link
    never gets that target query and fails closed except for
    the exact contained ``CLAUDE.md -> AGENTS.md``
    compatibility alias, whose regular target must be independently discovered;
    only that target is returned and read.
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
    by_relative: dict[Path, SecureFile] = {}
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
            if not is_selected:
                return
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
                f"Refusing selected symlink or reparse point: {relative.as_posix()}",
                relative_path=relative.as_posix(),
            )
        if not is_selected:
            return
        if not stat.S_ISREG(metadata.st_mode):
            _raise_unsafe_file(relative)
        if getattr(metadata, "st_nlink", 1) != 1:
            _raise_unsafe_file(relative, hardlink=True)
        secure_file = SecureFile(root, root / relative, relative, metadata)
        files.append(secure_file)
        by_relative[relative] = secure_file

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
                                linked_directory = False
                                linked_or_reparse = stat_is_link_or_reparse(metadata)
                                selected_link = linked_or_reparse and entry_selected
                                if linked_or_reparse and not selected_link:
                                    # This follows target metadata only to distinguish
                                    # a directory redirect from an irrelevant file
                                    # link. The redirect is never opened or descended.
                                    try:
                                        linked_directory = entry.is_dir(follow_symlinks=True)
                                    except OSError:
                                        linked_directory = False
                                if stat.S_ISDIR(metadata.st_mode) or linked_directory:
                                    if linked_directory or linked_or_reparse:
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
        # Windows fallback: lstat/reparse checks are applied immediately before
        # each descent. Python does not expose a portable descriptor-relative
        # scandir/openat traversal on Windows, so concurrent directory swaps
        # remain a narrower platform limitation and selected reads are checked
        # again through final file handles.
        def raise_walk_error(exc: OSError) -> None:
            raise SecurePathError("path_access_error", f"Cannot safely traverse Tier 2 directory: {exc}") from exc

        for directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            onerror=raise_walk_error,
            followlinks=False,
        ):
            directory_path = Path(directory)
            relative_directory = directory_path.relative_to(root)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                path = directory_path / name
                relative = relative_directory / name
                try:
                    metadata = path.lstat()
                except OSError as exc:
                    raise SecurePathError(
                        "path_access_error",
                        f"Cannot inspect Tier 2 path {relative.as_posix()}: {exc}",
                        relative_path=relative.as_posix(),
                    ) from exc
                if stat_is_link_or_reparse(metadata):
                    raise SecurePathError(
                        "unsafe_path",
                        f"Refusing linked directory or reparse point before descent: {relative.as_posix()}",
                        relative_path=relative.as_posix(),
                    )
                if name in excluded:
                    continue
                if selected(relative):
                    _raise_unsafe_file(relative)
                consume_path(relative)
                directory_depth = len(relative.parts)
                if directory_depth > MAX_SECURE_DIRECTORY_DEPTH:
                    _raise_directory_depth_limit(relative)
                if max_depth is None or directory_depth < max_depth:
                    kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                path = directory_path / name
                relative = relative_directory / name
                try:
                    metadata = path.lstat()
                except OSError as exc:
                    raise SecurePathError(
                        "path_access_error",
                        f"Cannot inspect Tier 2 path {relative.as_posix()}: {exc}",
                        relative_path=relative.as_posix(),
                    ) from exc
                consume_path(relative)
                record_file(
                    relative,
                    metadata,
                    # Exact raw link text is security-significant for the one
                    # compatibility alias; Path.readlink() would normalize it.
                    lambda path=path: os.readlink(path),  # noqa: PTH115
                    selected_result=selected(relative),
                )
    else:
        raise SecurePathError(
            "secure_open_unavailable",
            "This platform cannot guarantee no-follow Tier 2 discovery.",
        )

    for alias, target in pending_aliases:
        if target not in by_relative:
            raise SecurePathError(
                "unsafe_path",
                f"Compatibility alias target is not an independently enumerated regular file: {alias.as_posix()}",
                relative_path=alias.as_posix(),
            )

    return sorted(files, key=lambda item: item.rel_path)


class SecureRoot:
    """Descriptor-anchored reads beneath one verified regular root."""

    def __init__(self, root: Path) -> None:
        self.root = _absolute_no_resolve(root)
        self._root_fd: int | None = None
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
            _validate_windows_parent_components(self.root / "placeholder")
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
            _validate_opened_file(opened, relative_path, expected)
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
        candidate = self.root / relative_path
        current = self.root
        for component in relative_path.parts:
            current /= component
            try:
                metadata = current.lstat()
            except OSError as exc:
                raise SecurePathError("unsafe_path", f"Cannot inspect Tier 2 path: {exc}") from exc
            if stat_is_link_or_reparse(metadata):
                raise SecurePathError(
                    "unsafe_path",
                    f"Tier 2 path contains a symlink or Windows reparse point: {component}",
                    relative_path=relative_path.as_posix(),
                )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError as exc:
            raise SecurePathError("unsafe_path", f"Cannot securely open Tier 2 file: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            _validate_opened_file(opened, relative_path, expected)
            final_path = _windows_final_path(descriptor)
            try:
                final_path.relative_to(self.root)
            except ValueError as exc:
                raise SecurePathError("unsafe_path", "Opened Tier 2 file escapes its verified root.") from exc
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise


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


def _inspect_destination_windows(path: Path, *, missing_ok: bool) -> os.stat_result | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if stat_is_link_or_reparse(metadata):
        raise SecurePathError("unsafe_path", f"Destination is a symlink or reparse point: {path.name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise SecurePathError("unsafe_path", f"Destination is not a regular file: {path.name}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise SecurePathError("unsafe_hardlink", f"Destination is hard-linked (link count > 1): {path.name}")
    return metadata


def _validate_windows_parent_identity(parent: Path, expected: os.stat_result) -> None:
    _validate_windows_parent_components(parent / "placeholder")
    try:
        current = parent.lstat()
    except OSError as exc:
        raise SecurePathError("unsafe_path", f"Cannot revalidate declared output parent: {exc}") from exc
    if stat_is_link_or_reparse(current) or not stat.S_ISDIR(current.st_mode) or not os.path.samestat(current, expected):
        raise SecurePathError("unsafe_path", "Declared output parent changed identity during the atomic write.")


def _same_windows_path(first: Path, second: Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(first))) == os.path.normcase(  # noqa: PTH100
        os.path.abspath(os.fspath(second))  # noqa: PTH100
    )


def _atomic_write_windows(path: Path, payload: bytes) -> None:
    """Best-effort no-follow atomic replacement for native Windows.

    Python does not expose directory-handle-relative replacement on Windows, so
    this path cannot match the POSIX branch's anchored-parent race guarantee.
    It still rejects reparse/link parents and destinations, pins the temporary
    file by handle while writing, verifies its final handle path, and rechecks
    identities immediately before and after the atomic replacement.
    """
    absolute = _absolute_no_resolve(path)
    if not absolute.name or absolute.name in {".", ".."}:
        raise SecurePathError("unsafe_path", "Destination must name a file.")

    _validate_windows_parent_components(absolute)
    try:
        parent_metadata = absolute.parent.lstat()
    except OSError as exc:
        raise SecurePathError("path_access_error", f"Cannot inspect output parent: {exc}") from exc
    if stat_is_link_or_reparse(parent_metadata) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise SecurePathError("unsafe_path", "Declared output parent is linked, reparsed, or not a directory.")

    before = _inspect_destination_windows(absolute, missing_ok=True)
    temporary_path: Path | None = None
    descriptor = -1
    written_metadata: os.stat_result | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        for _attempt in range(128):
            candidate = absolute.parent / f".{absolute.name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(candidate, flags, 0o600)
            except FileExistsError:
                continue
            temporary_path = candidate
            break
        if descriptor < 0 or temporary_path is None:
            raise SecurePathError("path_access_error", "Cannot allocate a secure temporary output file.")

        opened = os.fstat(descriptor)
        _validate_opened_file(opened, Path(temporary_path.name), None)
        if not _same_windows_path(_windows_final_path(descriptor), temporary_path):
            raise SecurePathError("unsafe_path", "Temporary output handle escaped its declared parent.")

        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        written_metadata = os.fstat(descriptor)
        _validate_opened_file(written_metadata, Path(temporary_path.name), None)
        if written_metadata.st_size != len(payload):
            raise SecurePathError("unsafe_path", "Temporary output size changed while being written.")
        temporary_metadata = temporary_path.lstat()
        _validate_opened_file(temporary_metadata, Path(temporary_path.name), written_metadata)
        if not _same_windows_path(_windows_final_path(descriptor), temporary_path):
            raise SecurePathError("unsafe_path", "Temporary output handle changed final path while being written.")

        _validate_windows_parent_identity(absolute.parent, parent_metadata)
        destination = _inspect_destination_windows(absolute, missing_ok=True)
        if (before is None) != (destination is None) or (
            before is not None and destination is not None and not os.path.samestat(before, destination)
        ):
            raise SecurePathError("unsafe_path", "Destination changed identity while output was prepared.")

        # Native Windows normally prevents replacement while this process still
        # holds the temporary file open. Close only after the handle-path and
        # inode checks above, then revalidate the name once more before publish.
        os.close(descriptor)
        descriptor = -1
        temporary_metadata = temporary_path.lstat()
        _validate_opened_file(temporary_metadata, Path(temporary_path.name), written_metadata)
        _validate_windows_parent_identity(absolute.parent, parent_metadata)
        temporary_path.replace(absolute)
        temporary_path = None

        published = _inspect_destination_windows(absolute, missing_ok=False)
        if (
            published is None
            or written_metadata is None
            or not os.path.samestat(published, written_metadata)
            or published.st_size != len(payload)
        ):
            raise SecurePathError("unsafe_path", "Published output changed identity or size during replacement.")

        read_flags = (
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(absolute, read_flags)
        opened_published = os.fstat(descriptor)
        _validate_opened_file(opened_published, Path(absolute.name), published)
        if not _same_windows_path(_windows_final_path(descriptor), absolute):
            raise SecurePathError("unsafe_path", "Published output handle escaped its declared parent.")
        _validate_windows_parent_identity(absolute.parent, parent_metadata)
    except SecurePathError:
        raise
    except OSError as exc:
        raise SecurePathError("path_access_error", f"Cannot securely write output on Windows: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # As in the POSIX branch, do not unlink a failed temporary name: without
        # conditional unlink-by-inode, a swap between validation and cleanup
        # could delete an unrelated file. Successful replacement consumes it.


def _windows_final_path(descriptor: int) -> Path:
    if os.name != "nt":
        raise OSError("Windows handle verification is unavailable on this platform")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    get_final_path = ctypes.windll.kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_final_path(msvcrt.get_osfhandle(descriptor), buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), "Cannot resolve opened Windows file handle")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)

