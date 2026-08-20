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
from dataclasses import dataclass
from pathlib import Path

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


def stat_is_link_or_reparse(metadata: os.stat_result) -> bool:
    """Return whether metadata identifies a symlink or Windows reparse point."""
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


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
    ) -> tuple[bytes, os.stat_result]:
        """Read one bounded regular single-link file without following redirects."""
        if not self._entered:
            raise SecurePathError("secure_open_unavailable", "Secure root context is not active.")
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
            # Windows path stat and CRT descriptor stat do not expose a
            # reliably comparable st_dev/st_ino pair. _open_windows rechecks
            # the declared name with path lstat around its native no-follow
            # handle open; only POSIX compares the descriptor to discovery
            # metadata here.
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

    def _open_posix(self, relative_path: Path, expected: os.stat_result | None) -> int:
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
            _validate_opened_file(before, relative_path, expected)
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
                _validate_opened_file(os.fstat(descriptor), relative_path, before)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
        finally:
            os.close(directory_fd)

    def _open_windows(self, relative_path: Path, expected: os.stat_result | None) -> int:
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
            _validate_opened_file(before_open, relative_path, expected)

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
            _validate_windows_read_file_handle(native_file_handle, relative_path)
            try:
                after_open = declared_path.lstat()
            except OSError as exc:
                raise SecurePathError(
                    "unsafe_path",
                    f"Cannot revalidate selected file securely: {relative_path.as_posix()}: {exc}",
                    relative_path=relative_path.as_posix(),
                ) from exc
            _validate_opened_file(after_open, relative_path, before_open)
            descriptor = msvcrt.open_osfhandle(
                native_file_handle,
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0),
            )
            native_file_handle = -1
            opened = os.fstat(descriptor)
            _validate_opened_file(opened, relative_path, None)
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
