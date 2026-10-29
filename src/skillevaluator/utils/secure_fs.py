# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-anchored, bounded reads for evaluator-owned artifacts.

Selected files are opened relative to a pinned root without following links.
The reader verifies identity, type, link count, size, and timestamps before and
after the read. Native Windows file handles allow reads but deny concurrent
writes and deletes until the selected-file descriptor closes.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Literal, NamedTuple

from skillevaluator.utils.path_security import matches_filesystem_name

_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd

_WINDOWS_FILE_READ_DATA = 0x1
_WINDOWS_FILE_TRAVERSE = 0x20
_WINDOWS_FILE_READ_ATTRIBUTES = 0x80
_WINDOWS_SYNCHRONIZE = 0x100000
_WINDOWS_SHARE_READ = 0x1
_WINDOWS_SHARE_READ_WRITE = _WINDOWS_SHARE_READ | 0x2
_WINDOWS_FILE_OPEN = 1
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
_WINDOWS_DIRECTORY_OPEN_OPTIONS = (
    _WINDOWS_FILE_DIRECTORY_FILE
    | _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT
    | _WINDOWS_FILE_OPEN_FOR_BACKUP_INTENT
    | _WINDOWS_FILE_OPEN_REPARSE_POINT
)
_WINDOWS_FILE_OPEN_OPTIONS = (
    _WINDOWS_FILE_NON_DIRECTORY_FILE | _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT | _WINDOWS_FILE_OPEN_REPARSE_POINT
)


_READLINK_SUPPORTS_DIR_FD = os.readlink in os.supports_dir_fd

_SCANDIR_SUPPORTS_FD = os.scandir in os.supports_fd

MAX_SECURE_DIRECTORY_DEPTH = 64

_WINDOWS_DISCOVERY_ENTRY_ACCESS = _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_SYNCHRONIZE

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
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


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

    The caller must verify the root and supply a normalized relative path;
    this helper does not reject absolute paths or parent traversal. Its
    observations are best-effort metadata probes, not descriptor-anchored
    reads or a coherent filesystem snapshot. Use SecureRoot for reads.
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
    if expected is not None:
        changed = not os.path.samestat(metadata, expected)
        for attribute in ("st_size", "st_mtime_ns", "st_ctime_ns"):
            if getattr(metadata, attribute, None) != getattr(expected, attribute, None):
                changed = True
        if changed:
            raise SecurePathError(
                "unsafe_path",
                f"Selected file changed identity or contents while being opened: {relative_path.as_posix()}",
                relative_path=relative_path.as_posix(),
            )


def _validate_directory_snapshot(
    metadata: os.stat_result,
    relative_path: Path,
    expected: os.stat_result,
) -> None:
    """Require one directory identity and entry snapshot to stay stable."""
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
            f"Secure root directory changed identity or snapshot: {label}",
            relative_path=label,
        )


def _open_absolute_directory_posix(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        expected = path.lstat()
    except OSError as exc:
        raise SecurePathError("unsafe_root", f"Cannot inspect declared root safely: {exc}") from exc
    if stat_is_link_or_reparse(expected) or not stat.S_ISDIR(expected.st_mode):
        raise SecurePathError("unsafe_root", f"Declared root is a symlink or non-directory: {path}")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SecurePathError("unsafe_root", f"Cannot securely open declared root: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not os.path.samestat(expected, opened):
            raise SecurePathError("unsafe_root", "Declared root changed while being opened.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


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
            raise SecurePathError("unsafe_root", "Secure root context is already active.")
        try:
            metadata = self.root.lstat()
        except OSError as exc:
            raise SecurePathError("invalid_root", f"Cannot inspect secure root: {exc}") from exc
        if stat_is_link_or_reparse(metadata):
            raise SecurePathError("unsafe_root", f"Secure root is a symlink or reparse point: {self.root.name}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise SecurePathError("invalid_root", f"Secure root is not a regular directory: {self.root}")
        if self._expected is not None:
            _validate_directory_snapshot(metadata, Path(), self._expected)

        if os.name == "posix":
            if not (hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW") and _OPEN_SUPPORTS_DIR_FD):
                raise SecurePathError(
                    "secure_open_unavailable",
                    "This platform cannot guarantee descriptor-anchored no-follow reads.",
                )
            root_fd = _open_absolute_directory_posix(self.root)
            try:
                opened = os.fstat(root_fd)
                if not stat.S_ISDIR(opened.st_mode) or not os.path.samestat(metadata, opened):
                    raise SecurePathError("unsafe_root", "Secure root changed while being opened.")
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

        raise SecurePathError("secure_open_unavailable", "This platform cannot guarantee no-follow reads.")

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
        """Read a bounded regular file without redirects; hard links require an explicit opt-in."""
        if not self._entered:
            raise SecurePathError("secure_open_unavailable", "Secure root context is not active.")
        relative_path = _relative_path(relative_path)
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if os.name == "posix":
            descriptor = self._open_posix(relative_path, expected, allow_hardlinks=allow_hardlinks)
        elif os.name == "nt":
            descriptor = self._open_windows(relative_path, expected, allow_hardlinks=allow_hardlinks)
        else:
            raise SecurePathError("secure_open_unavailable", "Secure no-follow reads are unavailable.")

        try:
            opened = os.fstat(descriptor)
            # Windows path stat and CRT descriptor stat do not expose a
            # reliably comparable st_dev/st_ino pair. _open_windows rechecks
            # the declared name with path lstat around its native no-follow
            # handle open; only POSIX compares the descriptor to discovery
            # metadata here.
            _validate_opened_file(
                opened, relative_path, expected if os.name == "posix" else None, allow_hardlinks=allow_hardlinks
            )
            if opened.st_size > max_bytes:
                raise SecurePathError(
                    "file_size_limit",
                    f"Selected file exceeds the {max_bytes}-byte limit: {relative_path.as_posix()}",
                    relative_path=relative_path.as_posix(),
                    metadata={"actual_bytes": opened.st_size, "limit_bytes": max_bytes},
                )
            raw = read_bounded(descriptor, max_bytes, relative_path=relative_path.as_posix())
            after = os.fstat(descriptor)
            _validate_opened_file(after, relative_path, opened, allow_hardlinks=allow_hardlinks)
            return raw, opened
        finally:
            os.close(descriptor)

    def _open_posix(
        self, relative_path: Path, expected: os.stat_result | None, *, allow_hardlinks: bool = False
    ) -> int:
        if self._root_fd is None:
            raise SecurePathError("secure_open_unavailable", "Secure root descriptor is unavailable.")
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
                        f"Cannot securely traverse path component {component!r}: {exc}",
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
                        f"Path component is not a regular directory: {component}",
                        relative_path=relative_path.as_posix(),
                    )
                os.close(directory_fd)
                directory_fd = child_fd

            try:
                before = os.stat(relative_path.name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot inspect selected file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(before, relative_path, expected, allow_hardlinks=allow_hardlinks)
            try:
                descriptor = os.open(relative_path.name, file_flags, dir_fd=directory_fd)
            except OSError as exc:
                message = "Selected path is a symlink or unsafe file" if exc.errno == errno.ELOOP else str(exc)
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot securely open {relative_path.as_posix()}: {message}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            try:
                _validate_opened_file(os.fstat(descriptor), relative_path, before, allow_hardlinks=allow_hardlinks)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
        finally:
            os.close(directory_fd)

    def _open_windows(
        self, relative_path: Path, expected: os.stat_result | None, *, allow_hardlinks: bool = False
    ) -> int:
        if not self._windows_root_handles:
            raise SecurePathError("secure_open_unavailable", "Secure root handle is unavailable.")

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

            # Revalidate with the same path-stat family used during discovery,
            # then pin that name through a native handle which denies write and
            # delete sharing. CRT fstat identity is not comparable to lstat on
            # Windows, so name stability is established before conversion.
            try:
                before_open = declared_path.lstat()
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot inspect selected file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(before_open, relative_path, expected, allow_hardlinks=allow_hardlinks)

            native_file_handle = _windows_open_relative_handle(
                parent_handle,
                relative_path.name,
                access=_WINDOWS_FILE_READ_ACCESS,
                # Deny write/delete sharing for the selected file while the
                # CRT descriptor is live. This closes same-size rewrite races
                # that Windows creation-time metadata cannot detect.
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
                    f"Cannot revalidate selected file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(after_open, relative_path, before_open, allow_hardlinks=allow_hardlinks)
            descriptor = msvcrt.open_osfhandle(
                native_file_handle,
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0),
            )
            native_file_handle = -1
            opened = os.fstat(descriptor)
            _validate_opened_file(opened, relative_path, None, allow_hardlinks=allow_hardlinks)
            return descriptor
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            raise SecurePathError(
                "unsafe_path",
                f"Cannot securely open selected file {relative_path.as_posix()}: {exc}",
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


def _validate_windows_read_directory_handle(handle: int, relative_path: Path) -> _WindowsHandleMetadata:
    """Require one opened Windows traversal component to be a plain directory."""
    metadata = _windows_handle_metadata(handle)
    directory_attribute = 0x10
    reparse_attribute = 0x400
    if metadata.attributes & reparse_attribute or not metadata.attributes & directory_attribute:
        raise SecurePathError(
            "unsafe_path",
            f"Path contains a non-directory or reparse component: {relative_path.as_posix()}",
            relative_path=relative_path.as_posix(),
        )
    return metadata


def _validate_windows_read_file_handle(
    handle: int, relative_path: Path, *, allow_hardlinks: bool = False
) -> _WindowsHandleMetadata:
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
    if not allow_hardlinks and metadata.link_count != 1:
        _raise_unsafe_file(relative_path, hardlink=True)
    return metadata


def _windows_open_anchored_directory_chain(
    path: Path,
    *,
    expected: os.stat_result,
) -> list[int]:
    """Pin an absolute directory from its volume anchor without following reparses."""
    absolute = _absolute_no_resolve(path)
    if not absolute.anchor:
        raise SecurePathError("unsafe_root", "Windows root has no filesystem anchor.")

    anchor = Path(absolute.anchor)
    handles: list[int] = []
    try:
        anchor_handle = _windows_open_handle(
            anchor,
            access=_WINDOWS_DIRECTORY_READ_ACCESS,
            share=_WINDOWS_SHARE_READ_WRITE,
            disposition=3,
            flags=0x02000000 | _WINDOWS_FILE_OPEN_REPARSE_POINT,
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
            raise SecurePathError("unsafe_root", f"Cannot revalidate declared root: {exc}") from exc
        if stat_is_link_or_reparse(declared) or not stat.S_ISDIR(declared.st_mode):
            raise SecurePathError("unsafe_root", "Declared root became a reparse point or non-directory.")
        if not os.path.samestat(expected, declared):
            raise SecurePathError("unsafe_root", "Root changed identity while native handles were opened.")
        return handles
    except BaseException:
        while handles:
            _windows_close_handle(handles.pop())
        raise


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
        if relative.name in self.excluded or (
            # A case alias of an excluded name is excluded only where the filesystem makes it the same directory.
            any(name.casefold() == relative.name.casefold() for name in self.excluded)
            and matches_filesystem_name(self.root / relative, self.excluded)
        ):
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
