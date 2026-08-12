# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional, no-follow staging for untrusted Harbor inputs.

On POSIX, authored source traversal and copying are descriptor-relative and use
``O_NOFOLLOW``.  Other platforms use a checked lstat/fstat implementation with
pre/post identity validation; that fallback rejects the same input types but
has a narrower race-resistance boundary.  Neither backend provides a security
guarantee against concurrent mutation by another process with the same UID.

Opposite-order stability passes provide best-effort detection of incidental
changes, and staged roots remain private until their final exposure step.  They
are not a coherent filesystem snapshot.  All concurrent mutation by another
process running as the same UID--including one-shot mutation of source, stage,
destination, rollback, or reserve paths--is outside this module's security
guarantee.  Callers needing that threat model must provide filesystem isolation,
exclusive ownership, or snapshot support.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import secrets
import stat
import sys
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

IgnoreCallback = Callable[[str, list[str]], Iterable[str]]

_CHUNK_SIZE = 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_BINARY_FLAG = getattr(os, "O_BINARY", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = (
    os.O_RDONLY
    | _BINARY_FLAG
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOCTTY", 0)
)
_WINDOWS_CHMOD_SEMANTICS = os.name == "nt"
_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE = os.name == "posix"
_DESCRIPTOR_BACKEND = (
    os.name == "posix"
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.scandir in os.supports_fd
)
_RENAME_NOREPLACE = 0x1
_RENAME_EXCL = 0x4


class UnsafeStagingError(ValueError):
    """Raised when authored or destination content is unsafe to stage."""


class _AtomicRenameUnsupported(UnsafeStagingError):
    """Raised when the first no-replace rename can safely use the fallback."""


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    parts: tuple[str, ...]
    kind: str
    mode: int
    device: int
    inode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int
    digest: str | None = None


@dataclass(frozen=True, slots=True)
class _TreeManifest:
    source: Path
    allowed_root: Path
    role: str
    root: _ManifestEntry
    entries: tuple[_ManifestEntry, ...]
    ignore: IgnoreCallback | None = None

    @property
    def by_parts(self) -> dict[tuple[str, ...], _ManifestEntry]:
        return {entry.parts: entry for entry in (self.root, *self.entries)}


@dataclass(frozen=True, slots=True)
class _FileManifest:
    source: Path
    allowed_root: Path
    entry: _ManifestEntry


@dataclass(slots=True)
class _StagedNode:
    name: str
    descriptor: int
    device: int
    inode: int
    kind: str

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


def _load_atomic_rename_backend():
    if sys.platform == "darwin":
        symbol = "renameatx_np"
        flag = _RENAME_EXCL
    elif sys.platform.startswith("linux"):
        symbol = "renameat2"
        flag = _RENAME_NOREPLACE
    else:
        return None, 0
    try:
        library = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None, flag
    function = getattr(library, symbol, None)
    if function is None:
        return None, flag
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    return function, flag


_ATOMIC_RENAME, _ATOMIC_RENAME_FLAG = _load_atomic_rename_backend()


def _canonicalize_platform_root_alias(path: Path) -> Path:
    """Normalize only an immutable, administrator-owned first-level alias.

    macOS exposes normal temporary paths below the root-owned ``/var`` link.
    User-controlled symlinks at every later component remain forbidden.
    """

    if os.name != "posix" or len(path.parts) < 2:
        return path
    root = Path(path.anchor)
    alias = root / path.parts[1]
    try:
        root_metadata = root.stat()
        alias_metadata = alias.lstat()
    except OSError:
        return path
    if (
        not stat.S_ISLNK(alias_metadata.st_mode)
        or root_metadata.st_uid != 0
        or alias_metadata.st_uid != 0
        or stat.S_IMODE(root_metadata.st_mode) & 0o022
    ):
        return path
    target = alias.readlink()
    if not target.is_absolute():
        target = root / target
    normalized = Path(os.path.abspath(os.fspath(target)))  # noqa: PTH100
    return normalized.joinpath(*path.parts[2:])


def _absolute_lexical(path: Path | str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    return _canonicalize_platform_root_alias(absolute)


def _relative_to_allowed(source: Path, allowed_root: Path) -> tuple[str, ...]:
    try:
        return source.relative_to(allowed_root).parts
    except ValueError as exc:
        raise UnsafeStagingError(f"source is outside the allowed root: {source}") from exc


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _is_link(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata)


def _source_device(metadata: os.stat_result, *, path: Path) -> int:
    """Return source device identity (a narrow test hook for mount crossings)."""

    del path
    return metadata.st_dev


def _fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _node_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.S_IFMT(metadata.st_mode),
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
    )


def _fallback_opened_matches_named(opened: os.stat_result, named: os.stat_result) -> bool:
    """Compare fallback file metadata without assuming Windows stat identities."""
    if _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE:
        return _fingerprint(opened) == _fingerprint(named)
    return (
        stat.S_IFMT(opened.st_mode) == stat.S_IFMT(named.st_mode)
        and opened.st_nlink == named.st_nlink
        and opened.st_size == named.st_size
    )


def _portable_fingerprint_mode(mode: int) -> int:
    """Normalize mode bits to the semantics the active copier preserves."""
    normalized = stat.S_IMODE(mode) & 0o777
    if _WINDOWS_CHMOD_SEMANTICS:
        return int(bool(normalized & stat.S_IWRITE))
    return normalized


def _entry_from_stat(
    parts: tuple[str, ...],
    kind: str,
    metadata: os.stat_result,
    *,
    digest: str | None = None,
) -> _ManifestEntry:
    return _ManifestEntry(
        parts=parts,
        kind=kind,
        mode=stat.S_IMODE(metadata.st_mode) & 0o777,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        links=metadata.st_nlink,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
        digest=digest,
    )


def _display(root: Path, parts: tuple[str, ...]) -> Path:
    return root.joinpath(*parts) if parts else root


def _validate_type(
    metadata: os.stat_result,
    *,
    path: Path,
    role: str,
    root_device: int | None,
) -> str:
    if _is_link(metadata):
        raise UnsafeStagingError(f"{role} path contains a symlink or reparse point: {path}")
    if stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
    elif stat.S_ISREG(metadata.st_mode):
        kind = "file"
        if metadata.st_nlink != 1:
            raise UnsafeStagingError(f"{role} contains a hard-linked file with multiple links: {path}")
    else:
        raise UnsafeStagingError(f"{role} entry is not a regular file or directory: {path}")
    if root_device is not None and _source_device(metadata, path=path) != root_device:
        raise UnsafeStagingError(f"{role} crosses a filesystem device or mount: {path}")
    return kind


def _changed(path: Path, detail: str = "") -> UnsafeStagingError:
    suffix = f" ({detail})" if detail else ""
    return UnsafeStagingError(f"source changed after validation: {path}{suffix}")


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while data := os.read(descriptor, _CHUNK_SIZE):
        digest.update(data)
    return digest.hexdigest()


def _open_absolute_directory(path: Path, *, purpose: str, create: bool = False) -> int:
    """Open one absolute path component-by-component without following links."""

    absolute = _absolute_lexical(path)
    if not absolute.anchor:
        raise UnsafeStagingError(f"{purpose} path has no filesystem anchor: {path}")
    descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for component in absolute.parts[1:]:
            created = False
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                try:
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError as exc:
                    raise UnsafeStagingError(
                        f"{purpose} path contains a symlink, reparse point, or non-directory: {absolute}"
                    ) from exc
            except OSError as exc:
                raise UnsafeStagingError(
                    f"{purpose} path contains a symlink, reparse point, or non-directory: {absolute}"
                ) from exc
            opened = os.fstat(child)
            if _is_link(opened) or not stat.S_ISDIR(opened.st_mode):
                os.close(child)
                raise UnsafeStagingError(
                    f"{purpose} path contains a symlink, reparse point, or non-directory: {absolute}"
                )
            if created:
                os.fchmod(child, 0o700)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_tree_root(source: Path, allowed_root: Path, *, role: str) -> tuple[int, int]:
    try:
        descriptor = _open_absolute_directory(allowed_root, purpose=f"{role} allowed root")
    except FileNotFoundError as exc:
        raise UnsafeStagingError(f"{role} allowed root does not exist: {allowed_root}") from exc
    try:
        allowed_metadata = os.fstat(descriptor)
        allowed_device = _source_device(allowed_metadata, path=allowed_root)
        relative = _relative_to_allowed(source, allowed_root)
        for depth, component in enumerate(relative, start=1):
            path = allowed_root.joinpath(*relative[:depth])
            try:
                before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise UnsafeStagingError(f"{role} path does not exist: {source}") from exc
            _validate_type(before, path=path, role=role, root_device=allowed_device)
            if not stat.S_ISDIR(before.st_mode):
                raise UnsafeStagingError(f"{role} tree source is not a directory: {source}")
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise UnsafeStagingError(f"{role} path changed to a symlink or non-directory: {path}") from exc
            opened = os.fstat(child)
            if _fingerprint(opened) != _fingerprint(before):
                os.close(child)
                raise _changed(path, "directory was replaced")
            os.close(descriptor)
            descriptor = child
        return descriptor, allowed_device
    except BaseException:
        os.close(descriptor)
        raise


def _open_file_parent(source: Path, allowed_root: Path, *, role: str) -> tuple[int, int]:
    relative = _relative_to_allowed(source, allowed_root)
    if not relative:
        raise UnsafeStagingError(f"{role} file source must be below its allowed root: {source}")
    parent = source.parent
    descriptor, allowed_device = _open_tree_root(parent, allowed_root, role=role)
    return descriptor, allowed_device


def _scan_descriptor_tree(
    *,
    source: Path,
    descriptor: int,
    parts: tuple[str, ...],
    root_device: int,
    role: str,
    ignore: IgnoreCallback | None,
    entries: list[_ManifestEntry],
) -> None:
    with os.scandir(descriptor) as iterator:
        names = sorted(item.name for item in iterator)
    ignored = set(ignore(os.fspath(_display(source, parts)), names)) if ignore else set()
    for name in names:
        if name in ignored:
            continue
        child_parts = (*parts, name)
        child_path = _display(source, child_parts)
        try:
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise _changed(child_path, "entry disappeared") from exc
        kind = _validate_type(before, path=child_path, role=role, root_device=root_device)
        if kind == "directory":
            try:
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise _changed(child_path, "directory became a symlink or was replaced") from exc
            try:
                opened = os.fstat(child)
                if _fingerprint(opened) != _fingerprint(before):
                    raise _changed(child_path, "directory was replaced")
                entries.append(_entry_from_stat(child_parts, kind, opened))
                _scan_descriptor_tree(
                    source=source,
                    descriptor=child,
                    parts=child_parts,
                    root_device=root_device,
                    role=role,
                    ignore=ignore,
                    entries=entries,
                )
                if _fingerprint(os.fstat(child)) != _fingerprint(opened):
                    raise _changed(child_path, "directory changed while it was validated")
            finally:
                os.close(child)
            continue
        try:
            child = os.open(name, _FILE_FLAGS, dir_fd=descriptor)
        except OSError as exc:
            raise _changed(child_path, "file became a symlink or was replaced") from exc
        try:
            opened = os.fstat(child)
            _validate_type(opened, path=child_path, role=role, root_device=root_device)
            if _fingerprint(opened) != _fingerprint(before):
                raise _changed(child_path, "file was replaced")
            digest = _hash_descriptor(child)
            after = os.fstat(child)
            if _fingerprint(after) != _fingerprint(opened):
                raise _changed(child_path, "file changed while it was validated")
            entries.append(_entry_from_stat(child_parts, kind, after, digest=digest))
        finally:
            os.close(child)


def _build_tree_manifest_posix(
    source: Path,
    allowed_root: Path,
    *,
    role: str,
    ignore: IgnoreCallback | None,
) -> _TreeManifest:
    descriptor, allowed_device = _open_tree_root(source, allowed_root, role=role)
    try:
        before = os.fstat(descriptor)
        kind = _validate_type(before, path=source, role=role, root_device=allowed_device)
        if kind != "directory":
            raise UnsafeStagingError(f"{role} tree source is not a directory: {source}")
        entries: list[_ManifestEntry] = []
        _scan_descriptor_tree(
            source=source,
            descriptor=descriptor,
            parts=(),
            root_device=allowed_device,
            role=role,
            ignore=ignore,
            entries=entries,
        )
        if _fingerprint(os.fstat(descriptor)) != _fingerprint(before):
            raise _changed(source, "root directory changed while it was validated")
        return _TreeManifest(
            source=source,
            allowed_root=allowed_root,
            role=role,
            root=_entry_from_stat((), "directory", before),
            entries=tuple(entries),
            ignore=ignore,
        )
    finally:
        os.close(descriptor)


def _validate_fallback_components(path: Path, *, purpose: str, create: bool = False) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                raise
            current.mkdir(mode=0o700)
            metadata = current.lstat()
        if _is_link(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise UnsafeStagingError(f"{purpose} path contains a symlink, reparse point, or non-directory: {current}")


def _open_fallback_regular(
    path: Path,
    before: os.stat_result,
    *,
    role: str,
    root_device: int,
) -> int:
    """Open without blocking, then verify identity before any fallback read.

    ``O_NOFOLLOW`` is used where the platform provides it.  Elsewhere a link
    can be followed by the kernel, but fstat/reparse/type/identity checks occur
    before the first byte is read; this is the documented checked fallback
    boundary rather than POSIX-strength race closure.
    """

    try:
        descriptor = os.open(path, _FILE_FLAGS)
    except OSError as exc:
        with suppress(OSError):
            if _is_link(path.lstat()):
                raise UnsafeStagingError(f"{role} changed to a symlink or reparse point: {path}") from exc
        raise UnsafeStagingError(f"cannot safely open {role} file {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        _validate_type(
            opened,
            path=path,
            role=role,
            root_device=root_device if _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE else None,
        )
        if not _fallback_opened_matches_named(opened, before):
            raise _changed(path, "file was replaced")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _hash_path_checked(path: Path, before: os.stat_result, *, role: str, root_device: int) -> str:
    descriptor = _open_fallback_regular(path, before, role=role, root_device=root_device)
    try:
        opened = os.fstat(descriptor)
        digest = _hash_descriptor(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
    except OSError as exc:
        raise _changed(path, "file disappeared after reading") from exc
    if (
        _fingerprint(after) != _fingerprint(opened)
        or _fingerprint(named) != _fingerprint(before)
        or not _fallback_opened_matches_named(after, named)
    ):
        raise _changed(path, "file changed while it was validated")
    return digest


def _scan_fallback_tree(
    source: Path,
    *,
    parts: tuple[str, ...],
    root_device: int,
    role: str,
    ignore: IgnoreCallback | None,
    entries: list[_ManifestEntry],
) -> None:
    directory = _display(source, parts)
    names = sorted(item.name for item in os.scandir(directory))
    ignored = set(ignore(os.fspath(directory), names)) if ignore else set()
    for name in names:
        if name in ignored:
            continue
        child_parts = (*parts, name)
        child = _display(source, child_parts)
        before = child.lstat()
        kind = _validate_type(before, path=child, role=role, root_device=root_device)
        if kind == "directory":
            entries.append(_entry_from_stat(child_parts, kind, before))
            _scan_fallback_tree(
                source,
                parts=child_parts,
                root_device=root_device,
                role=role,
                ignore=ignore,
                entries=entries,
            )
            if _fingerprint(child.lstat()) != _fingerprint(before):
                raise _changed(child, "directory changed while it was validated")
        else:
            digest = _hash_path_checked(child, before, role=role, root_device=root_device)
            entries.append(_entry_from_stat(child_parts, kind, before, digest=digest))


def _build_tree_manifest_fallback(
    source: Path,
    allowed_root: Path,
    *,
    role: str,
    ignore: IgnoreCallback | None,
) -> _TreeManifest:
    _validate_fallback_components(allowed_root, purpose=f"{role} allowed root")
    relative = _relative_to_allowed(source, allowed_root)
    _validate_fallback_components(source, purpose=role)
    allowed_metadata = allowed_root.lstat()
    root_device = _source_device(allowed_metadata, path=allowed_root)
    before = source.lstat()
    kind = _validate_type(before, path=source, role=role, root_device=root_device)
    if kind != "directory" or relative is None:
        raise UnsafeStagingError(f"{role} tree source is not a directory: {source}")
    entries: list[_ManifestEntry] = []
    _scan_fallback_tree(
        source,
        parts=(),
        root_device=root_device,
        role=role,
        ignore=ignore,
        entries=entries,
    )
    if _fingerprint(source.lstat()) != _fingerprint(before):
        raise _changed(source, "root directory changed while it was validated")
    return _TreeManifest(source, allowed_root, role, _entry_from_stat((), kind, before), tuple(entries), ignore)


def _build_tree_manifest(
    source: Path | str,
    *,
    allowed_root: Path | str | None = None,
    ignore: IgnoreCallback | None = None,
    role: str = "source",
) -> _TreeManifest:
    """Validate a tree before copying (separate for change-detection tests)."""

    source_path = _absolute_lexical(source)
    root_path = _absolute_lexical(allowed_root if allowed_root is not None else source_path)
    if _DESCRIPTOR_BACKEND:
        return _build_tree_manifest_posix(source_path, root_path, role=role, ignore=ignore)
    return _build_tree_manifest_fallback(source_path, root_path, role=role, ignore=ignore)


def tree_content_fingerprint_secure(
    source: Path | str,
    *,
    allowed_root: Path | str | None = None,
    ignore: IgnoreCallback | None = None,
) -> str:
    """Return a deterministic content fingerprint for one securely validated tree.

    Filesystem identity and timestamps intentionally do not participate: a
    securely copied tree has different inodes and may have different directory
    metadata.  Relative paths, node kinds, and regular-file bytes do, so the
    result can bind source selection to the exact tree later staged elsewhere.
    """
    manifest = _build_tree_manifest(source, allowed_root=allowed_root, ignore=ignore)
    digest = hashlib.sha256()
    for entry in sorted((manifest.root, *manifest.entries), key=lambda item: item.parts):
        path = os.fsencode("/".join(entry.parts))
        kind = entry.kind.encode("ascii")
        mode = _portable_fingerprint_mode(entry.mode).to_bytes(2, "big")
        content = (entry.digest or "").encode("ascii")
        for field in (path, kind, mode, content):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _verify_entry(entry: _ManifestEntry, metadata: os.stat_result, path: Path) -> None:
    if _is_link(metadata):
        raise UnsafeStagingError(f"source changed to a symlink or reparse point: {path}")
    actual_kind = (
        "directory" if stat.S_ISDIR(metadata.st_mode) else "file" if stat.S_ISREG(metadata.st_mode) else "other"
    )
    if actual_kind == "file" and metadata.st_nlink != 1:
        raise UnsafeStagingError(f"source changed to a hard-linked file with multiple links: {path}")
    if (
        actual_kind != entry.kind
        or metadata.st_dev != entry.device
        or metadata.st_ino != entry.inode
        or metadata.st_nlink != entry.links
        or metadata.st_size != entry.size
        or metadata.st_mtime_ns != entry.modified_ns
        or metadata.st_ctime_ns != entry.changed_ns
        or stat.S_IMODE(metadata.st_mode) & 0o777 != entry.mode
    ):
        raise _changed(path)


def _open_manifest_root(manifest: _TreeManifest) -> int:
    descriptor, _ = _open_tree_root(manifest.source, manifest.allowed_root, role=manifest.role)
    _verify_entry(manifest.root, os.fstat(descriptor), manifest.source)
    return descriptor


def _open_relative_directory(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    manifest: _TreeManifest | None = None,
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for depth, component in enumerate(parts, start=1):
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                path = manifest.source.joinpath(*parts[:depth]) if manifest else Path(*parts[:depth])
                raise _changed(path, "directory became a symlink or was replaced") from exc
            if manifest is not None:
                _verify_entry(
                    manifest.by_parts[parts[:depth]], os.fstat(child), manifest.source.joinpath(*parts[:depth])
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_staged_node(parent_descriptor: int, *, prefix: str, kind: str) -> _StagedNode:
    for _ in range(128):
        name = f".{prefix}.staging-{secrets.token_hex(8)}"
        descriptor = -1
        try:
            if kind == "directory":
                os.mkdir(name, 0o700, dir_fd=parent_descriptor)
                descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
            else:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                os.fchmod(descriptor, 0o600)
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            expected = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
            if (
                _is_link(named)
                or not expected(named.st_mode)
                or not expected(opened.st_mode)
                or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise UnsafeStagingError(f"private staging {kind} was replaced during creation")
            if kind == "directory":
                os.fchmod(descriptor, 0o700)
            return _StagedNode(name, descriptor, opened.st_dev, opened.st_ino, kind)
        except FileExistsError:
            continue
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(OSError):
                if kind == "directory":
                    os.rmdir(name, dir_fd=parent_descriptor)
                else:
                    os.unlink(name, dir_fd=parent_descriptor)
            raise
    raise FileExistsError("could not allocate a private staging name")


def _verify_staged_node_name(parent_descriptor: int, node: _StagedNode) -> None:
    try:
        named = os.stat(node.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise UnsafeStagingError(f"private staging {node.kind} disappeared") from exc
    opened = os.fstat(node.descriptor)
    expected = stat.S_ISDIR if node.kind == "directory" else stat.S_ISREG
    if (
        _is_link(named)
        or not expected(named.st_mode)
        or not expected(opened.st_mode)
        or (named.st_dev, named.st_ino) != node.identity
        or (opened.st_dev, opened.st_ino) != node.identity
    ):
        raise UnsafeStagingError(f"private staging {node.kind} was replaced")


def _verify_published_node(parent_descriptor: int, name: str, node: _StagedNode) -> None:
    published = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    opened = os.fstat(node.descriptor)
    expected = stat.S_ISDIR if node.kind == "directory" else stat.S_ISREG
    if (
        _is_link(published)
        or not expected(published.st_mode)
        or not expected(opened.st_mode)
        or (published.st_dev, published.st_ino) != node.identity
        or (opened.st_dev, opened.st_ino) != node.identity
    ):
        raise UnsafeStagingError(f"published staging {node.kind} has an unexpected identity or type")


def _open_named_node(parent_descriptor: int, name: str, *, kind: str) -> _StagedNode:
    flags = _DIRECTORY_FLAGS if kind == "directory" else _FILE_FLAGS
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        expected = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
        if (
            _is_link(named)
            or not expected(named.st_mode)
            or not expected(opened.st_mode)
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise UnsafeStagingError(f"named staging {kind} has an unexpected identity or type")
        return _StagedNode(name, descriptor, opened.st_dev, opened.st_ino, kind)
    except BaseException:
        os.close(descriptor)
        raise


def _destination_metadata(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _remove_tree_at(
    parent_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    metadata = _destination_metadata(parent_descriptor, name)
    if metadata is None:
        return
    if expected_identity is not None and (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise UnsafeStagingError("refusing to remove a replaced staging entry")
    if _is_link(metadata):
        raise UnsafeStagingError("refusing to remove a staging symlink or reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise UnsafeStagingError("refusing to remove an unsafe staging entry")
        os.unlink(name, dir_fd=parent_descriptor)
        return
    os.chmod(name, 0o700, dir_fd=parent_descriptor, follow_symlinks=False)
    writable = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if _is_link(writable) or (writable.st_dev, writable.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise UnsafeStagingError("staging directory changed while making it removable")
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise UnsafeStagingError("staging directory changed during cleanup")
        os.fchmod(descriptor, 0o700)
        with os.scandir(descriptor) as iterator:
            children = [entry.name for entry in iterator]
        for child in children:
            _remove_tree_at(descriptor, child)
    finally:
        os.close(descriptor)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if _is_link(current) or (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise UnsafeStagingError("staging directory changed before cleanup")
    os.rmdir(name, dir_fd=parent_descriptor)


def _cleanup_staged_node(parent_descriptor: int, node: _StagedNode, *, suppress_errors: bool) -> None:
    try:
        _remove_tree_at(parent_descriptor, node.name, expected_identity=node.identity)
    except (OSError, UnsafeStagingError):
        if not suppress_errors:
            raise
    finally:
        with suppress(OSError):
            os.close(node.descriptor)


def _ensure_private_directories(root_descriptor: int, manifest: _TreeManifest) -> None:
    for entry in manifest.entries:
        if entry.kind != "directory":
            continue
        parent = _open_relative_directory(root_descriptor, entry.parts[:-1])
        try:
            current = _destination_metadata(parent, entry.parts[-1])
            if current is None:
                os.mkdir(entry.parts[-1], 0o700, dir_fd=parent)
                child = os.open(entry.parts[-1], _DIRECTORY_FLAGS, dir_fd=parent)
                try:
                    os.fchmod(child, 0o700)
                finally:
                    os.close(child)
                continue
            if _is_link(current) or not stat.S_ISDIR(current.st_mode):
                raise UnsafeStagingError(
                    f"cannot merge source directory over destination non-directory: {manifest.source.joinpath(*entry.parts)}"
                )
        finally:
            os.close(parent)


def _copy_manifest_file(
    *,
    source_root_descriptor: int,
    destination_root_descriptor: int,
    entry: _ManifestEntry,
    manifest: _TreeManifest,
    overwrite: bool,
) -> None:
    source_parent = _open_relative_directory(
        source_root_descriptor,
        entry.parts[:-1],
        manifest=manifest,
    )
    destination_parent = _open_relative_directory(destination_root_descriptor, entry.parts[:-1])
    source_descriptor = -1
    destination_descriptor = -1
    path = manifest.source.joinpath(*entry.parts)
    try:
        try:
            source_descriptor = os.open(entry.parts[-1], _FILE_FLAGS, dir_fd=source_parent)
        except OSError as exc:
            raise _changed(path, "file became a symlink or was replaced") from exc
        _verify_entry(entry, os.fstat(source_descriptor), path)
        existing = _destination_metadata(destination_parent, entry.parts[-1])
        if existing is not None:
            if not overwrite:
                raise FileExistsError(path)
            if _is_link(existing) or not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                raise UnsafeStagingError(f"private destination collision is unsafe: {path}")
            os.unlink(entry.parts[-1], dir_fd=destination_parent)
        destination_descriptor = os.open(
            entry.parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_parent,
        )
        os.fchmod(destination_descriptor, 0o600)
        digest = hashlib.sha256()
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while data := os.read(source_descriptor, _CHUNK_SIZE):
            digest.update(data)
            view = memoryview(data)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        _verify_entry(entry, os.fstat(source_descriptor), path)
        if digest.hexdigest() != entry.digest:
            raise _changed(path, "file contents changed")
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(destination_parent)
        os.close(source_parent)


def _apply_manifest(
    staged_descriptor: int,
    manifest: _TreeManifest,
    *,
    overwrite: bool,
) -> None:
    source_descriptor = _open_manifest_root(manifest)
    try:
        _ensure_private_directories(staged_descriptor, manifest)
        for entry in manifest.entries:
            if entry.kind == "file":
                _copy_manifest_file(
                    source_root_descriptor=source_descriptor,
                    destination_root_descriptor=staged_descriptor,
                    entry=entry,
                    manifest=manifest,
                    overwrite=overwrite,
                )
        _verify_entry(manifest.root, os.fstat(source_descriptor), manifest.source)
        current = _build_tree_manifest(
            manifest.source,
            allowed_root=manifest.allowed_root,
            ignore=manifest.ignore,
            role=manifest.role,
        )
        if current != manifest:
            raise _changed(manifest.source, "tree changed while it was copied")
    finally:
        os.close(source_descriptor)


def _merged_entries(manifests: tuple[_TreeManifest, ...]) -> dict[tuple[str, ...], _ManifestEntry]:
    merged: dict[tuple[str, ...], _ManifestEntry] = {}
    for manifest in manifests:
        for entry in manifest.entries:
            previous = merged.get(entry.parts)
            if previous is not None and previous.kind != entry.kind:
                raise UnsafeStagingError(
                    f"cannot transactionally merge unlike filesystem entries: {manifest.source.joinpath(*entry.parts)}"
                )
            merged[entry.parts] = entry
    return merged


def _hash_exact_descriptor(descriptor: int, *, parts: tuple[str, ...]) -> str:
    """Hash one staged file (a narrow hook for change-detection tests)."""

    del parts
    return _hash_descriptor(descriptor)


def _validate_tree_exact_pass(
    descriptor: int,
    manifests: tuple[_TreeManifest, ...],
    *,
    root_device: int,
    child_final_modes: bool,
    root_mode: int,
    reverse: bool,
) -> None:
    """Verify one ordered, post-order-stable exact-tree pass."""

    expected_entries = _merged_entries(manifests)
    children: dict[tuple[str, ...], dict[str, _ManifestEntry]] = {}
    for entry in expected_entries.values():
        children.setdefault(entry.parts[:-1], {})[entry.parts[-1]] = entry

    def verify_directory(
        directory: int,
        parts: tuple[str, ...],
        named_metadata: os.stat_result | None = None,
    ) -> None:
        opened_directory = os.fstat(directory)
        directory_entry = expected_entries.get(parts)
        expected_mode = (
            root_mode
            if not parts
            else directory_entry.mode
            if child_final_modes and directory_entry is not None
            else 0o700
        )
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or opened_directory.st_dev != root_device
            or stat.S_IMODE(opened_directory.st_mode) & 0o777 != expected_mode
        ):
            raise UnsafeStagingError("staging directory has an unsafe identity, device, or mode")
        if named_metadata is not None and _node_identity(opened_directory) != _node_identity(named_metadata):
            raise UnsafeStagingError("staging directory identity changed during exact verification")
        expected = children.get(parts, {})
        before_fingerprint = _fingerprint(opened_directory)
        with os.scandir(directory) as iterator:
            before_names = sorted(item.name for item in iterator)
        if set(before_names) != set(expected):
            raise UnsafeStagingError("staging tree has unexpected or missing entries")
        ordered_names = sorted(expected, reverse=reverse)
        for name in ordered_names:
            entry = expected[name]
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if _is_link(metadata) or metadata.st_dev != root_device:
                raise UnsafeStagingError("staging tree contains a link or mount crossing")
            if entry.kind == "directory":
                if not stat.S_ISDIR(metadata.st_mode):
                    raise UnsafeStagingError("staging directory changed type")
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory)
                try:
                    verify_directory(child, entry.parts, metadata)
                finally:
                    os.close(child)
                continue
            expected_file_mode = entry.mode if child_final_modes else 0o600
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != entry.size
                or stat.S_IMODE(metadata.st_mode) & 0o777 != expected_file_mode
            ):
                raise UnsafeStagingError("staging file has an unsafe type, link count, size, or mode")
            child = os.open(name, _FILE_FLAGS, dir_fd=directory)
            try:
                opened = os.fstat(child)
                before = _fingerprint(opened)
                if (
                    _node_identity(opened) != _node_identity(metadata)
                    or stat.S_IMODE(opened.st_mode) & 0o777 != expected_file_mode
                    or _hash_exact_descriptor(child, parts=entry.parts) != entry.digest
                    or _fingerprint(os.fstat(child)) != before
                ):
                    raise UnsafeStagingError("staging file contents or digest changed")
            finally:
                os.close(child)
            named_after = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if _fingerprint(named_after) != _fingerprint(metadata):
                raise UnsafeStagingError("staging file changed after exact verification")

        with os.scandir(directory) as iterator:
            after_names = sorted(item.name for item in iterator)
        if after_names != before_names or _fingerprint(os.fstat(directory)) != before_fingerprint:
            raise UnsafeStagingError("staging directory names or identity changed during exact verification")

    verify_directory(descriptor, ())


def _validate_tree_exact_stable(
    descriptor: int,
    manifests: tuple[_TreeManifest, ...],
    *,
    root_device: int,
    child_final_modes: bool,
    root_mode: int,
) -> None:
    """Sample opposite sibling orders to detect more incidental changes."""

    _validate_tree_exact_pass(
        descriptor,
        manifests,
        root_device=root_device,
        child_final_modes=child_final_modes,
        root_mode=root_mode,
        reverse=False,
    )
    _validate_tree_exact_pass(
        descriptor,
        manifests,
        root_device=root_device,
        child_final_modes=child_final_modes,
        root_mode=root_mode,
        reverse=True,
    )


def _validate_private_tree(
    descriptor: int,
    manifests: tuple[_TreeManifest, ...],
    *,
    root_device: int,
) -> None:
    _validate_tree_exact_stable(
        descriptor,
        manifests,
        root_device=root_device,
        child_final_modes=False,
        root_mode=0o700,
    )


def _apply_final_tree_modes(descriptor: int, manifests: tuple[_TreeManifest, ...]) -> None:
    entries = _merged_entries(manifests)
    files = [entry for entry in entries.values() if entry.kind == "file"]
    directories = [entry for entry in entries.values() if entry.kind == "directory"]
    for entry in files:
        parent = _open_relative_directory(descriptor, entry.parts[:-1])
        child = -1
        try:
            child = os.open(entry.parts[-1], _FILE_FLAGS, dir_fd=parent)
            os.fchmod(child, entry.mode)
            if stat.S_IMODE(os.fstat(child).st_mode) & 0o777 != entry.mode:
                raise UnsafeStagingError("published staging file mode could not be applied")
        finally:
            if child >= 0:
                os.close(child)
            os.close(parent)
    for entry in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        child = _open_relative_directory(descriptor, entry.parts)
        try:
            os.fchmod(child, entry.mode)
            if stat.S_IMODE(os.fstat(child).st_mode) & 0o777 != entry.mode:
                raise UnsafeStagingError("published staging directory mode could not be applied")
        finally:
            os.close(child)


def _validate_hidden_final_tree(node: _StagedNode, manifests: tuple[_TreeManifest, ...]) -> None:
    _validate_tree_exact_stable(
        node.descriptor,
        manifests,
        root_device=node.device,
        child_final_modes=True,
        root_mode=0o700,
    )


def _prepare_hidden_final_tree(node: _StagedNode, manifests: tuple[_TreeManifest, ...]) -> None:
    _apply_final_tree_modes(node.descriptor, manifests)
    _validate_hidden_final_tree(node, manifests)


def _expose_tree_root_exact(node: _StagedNode, manifests: tuple[_TreeManifest, ...]) -> None:
    """Validate while root-private, then expose only by changing root mode."""

    _validate_hidden_final_tree(node, manifests)
    before = os.fstat(node.descriptor)
    root_mode = manifests[-1].root.mode
    os.fchmod(node.descriptor, root_mode)
    after = os.fstat(node.descriptor)
    if _node_identity(after) != _node_identity(before) or stat.S_IMODE(after.st_mode) & 0o777 != root_mode:
        raise UnsafeStagingError("published staging root identity or mode changed during exposure")


def _validate_named_hidden_tree(
    parent_descriptor: int,
    name: str,
    manifest: _TreeManifest,
) -> None:
    node = _open_named_node(parent_descriptor, name, kind="directory")
    try:
        _validate_hidden_final_tree(node, (manifest,))
    finally:
        os.close(node.descriptor)


def _expose_named_tree_root(
    parent_descriptor: int,
    name: str,
    manifest: _TreeManifest,
) -> None:
    node = _open_named_node(parent_descriptor, name, kind="directory")
    try:
        _expose_tree_root_exact(node, (manifest,))
    finally:
        os.close(node.descriptor)


def _stage_manifests(
    parent_descriptor: int,
    destination_name: str,
    manifests: tuple[_TreeManifest, ...],
) -> _StagedNode:
    staged = _create_staged_node(parent_descriptor, prefix=destination_name or "tree", kind="directory")
    complete = False
    primary_error = False
    try:
        for index, manifest in enumerate(manifests):
            _apply_manifest(staged.descriptor, manifest, overwrite=index > 0)
        _validate_private_tree(staged.descriptor, manifests, root_device=staged.device)
        _verify_staged_node_name(parent_descriptor, staged)
        complete = True
        return staged
    except BaseException:
        primary_error = True
        raise
    finally:
        if not complete:
            _cleanup_staged_node(parent_descriptor, staged, suppress_errors=primary_error)


def _rename_no_replace(
    source_name: str,
    destination_name: str,
    *,
    source_parent: int,
    destination_parent: int,
) -> None:
    for name in (source_name, destination_name):
        if not name or name in {".", ".."} or "\0" in name or os.sep in name or (os.altsep and os.altsep in name):
            raise UnsafeStagingError("atomic rename names must be single non-special path components")
    if _ATOMIC_RENAME is None:
        raise UnsafeStagingError("destination filesystem lacks atomic no-replace rename support")
    ctypes.set_errno(0)
    result = _ATOMIC_RENAME(
        source_parent,
        os.fsencode(source_name),
        destination_parent,
        os.fsencode(destination_name),
        _ATOMIC_RENAME_FLAG,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination_name)
    unsupported = {errno.EINVAL, errno.ENOSYS}
    unsupported.update(value for name in ("ENOTSUP", "EOPNOTSUPP") if (value := getattr(errno, name, None)) is not None)
    if error in unsupported:
        raise _AtomicRenameUnsupported(
            f"destination filesystem does not support atomic no-replace publication: {destination_name}"
        )
    raise OSError(error, os.strerror(error), destination_name)


def _allocate_unused_name(parent_descriptor: int, prefix: str) -> str:
    for _ in range(128):
        name = f".{prefix}.backup-{secrets.token_hex(8)}"
        if _destination_metadata(parent_descriptor, name) is None:
            return name
    raise FileExistsError("could not allocate a private backup name")


def _prepare_moved_backup(
    parent_descriptor: int,
    backup_name: str,
    *,
    backup_path: Path,
    expected_existing: os.stat_result,
    kind: str,
) -> os.stat_result:
    """Verify a moved original and immediately make any residue private."""

    moved = os.stat(backup_name, dir_fd=parent_descriptor, follow_symlinks=False)
    expected_type = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
    if (
        _is_link(moved)
        or not expected_type(moved.st_mode)
        or _node_identity(moved) != _node_identity(expected_existing)
    ):
        raise UnsafeStagingError(f"destination changed while it was moved to backup: {backup_path}")
    private_mode = 0o700 if kind == "directory" else 0o600
    os.chmod(backup_name, private_mode, dir_fd=parent_descriptor, follow_symlinks=False)
    private = os.stat(backup_name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        _is_link(private)
        or _node_identity(private) != _node_identity(expected_existing)
        or stat.S_IMODE(private.st_mode) & 0o777 != private_mode
    ):
        raise UnsafeStagingError(f"moved destination backup could not be made private: {backup_path}")
    return private


def _cleanup_backup_best_effort(parent_descriptor: int, backup_name: str) -> None:
    # Once a new or restored destination is exact, cleanup must not turn that
    # success into a destructive pseudo-rollback.  A failed cleanup leaves the
    # already-private original under its hidden sibling name.
    with suppress(OSError, UnsafeStagingError):
        backup = os.stat(backup_name, dir_fd=parent_descriptor, follow_symlinks=False)
        _remove_tree_at(parent_descriptor, backup_name, expected_identity=(backup.st_dev, backup.st_ino))


def _rename_after_public_mutation(
    source_name: str,
    destination_name: str,
    *,
    parent_descriptor: int,
) -> None:
    """Rename during commit/rollback without ever enabling fallback."""

    try:
        _rename_no_replace(
            source_name,
            destination_name,
            source_parent=parent_descriptor,
            destination_parent=parent_descriptor,
        )
    except _AtomicRenameUnsupported as exc:
        raise UnsafeStagingError("atomic rename became unavailable after public destination mutation") from exc


def _publish_transactional(
    parent_descriptor: int,
    destination_name: str,
    staged: _StagedNode,
    *,
    expected_existing: os.stat_result | None,
    rollback: _StagedNode | None,
    backup_path: Path,
    prepublish: Callable[[], None],
    finalize: Callable[[], None],
    validate_rollback: Callable[[], None] | None,
    finalize_rollback: Callable[[], None] | None,
    validate_reserve: Callable[[str], None] | None,
    finalize_reserve: Callable[[str], None] | None,
) -> None:
    _verify_staged_node_name(parent_descriptor, staged)
    prepublish()
    current = _destination_metadata(parent_descriptor, destination_name)
    if expected_existing is None:
        if current is not None:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination_name)
        private_name = staged.name
        _rename_no_replace(
            private_name,
            destination_name,
            source_parent=parent_descriptor,
            destination_parent=parent_descriptor,
        )
        staged.name = destination_name
        try:
            _verify_published_node(parent_descriptor, destination_name, staged)
            finalize()
            _verify_published_node(parent_descriptor, destination_name, staged)
        except BaseException:
            _rename_after_public_mutation(
                destination_name,
                private_name,
                parent_descriptor=parent_descriptor,
            )
            staged.name = private_name
            raise
        return

    if (
        rollback is None
        or validate_rollback is None
        or finalize_rollback is None
        or validate_reserve is None
        or finalize_reserve is None
    ):
        raise UnsafeStagingError("existing destination publication requires a private rollback snapshot")
    _verify_staged_node_name(parent_descriptor, rollback)
    validate_rollback()
    if current is None or _fingerprint(current) != _fingerprint(expected_existing):
        raise UnsafeStagingError("destination changed before transactional publication")
    backup_name = _allocate_unused_name(parent_descriptor, destination_name or "destination")
    private_name = staged.name
    old_moved = False
    new_moved = False
    try:
        _rename_no_replace(
            destination_name,
            backup_name,
            source_parent=parent_descriptor,
            destination_parent=parent_descriptor,
        )
        old_moved = True
        _prepare_moved_backup(
            parent_descriptor,
            backup_name,
            backup_path=backup_path.parent / backup_name,
            expected_existing=expected_existing,
            kind=staged.kind,
        )
        prepublish()
        validate_rollback()
        try:
            _rename_no_replace(
                private_name,
                destination_name,
                source_parent=parent_descriptor,
                destination_parent=parent_descriptor,
            )
        except _AtomicRenameUnsupported as exc:
            raise UnsafeStagingError("atomic publication became unavailable after the old destination moved") from exc
        staged.name = destination_name
        new_moved = True
        _verify_published_node(parent_descriptor, destination_name, staged)
        finalize()
        _verify_published_node(parent_descriptor, destination_name, staged)
    except BaseException:
        if new_moved:
            _rename_after_public_mutation(
                destination_name,
                private_name,
                parent_descriptor=parent_descriptor,
            )
            staged.name = private_name
        if old_moved:
            rollback_private_name = rollback.name
            rollback_published = False
            try:
                validate_rollback()
                _rename_after_public_mutation(
                    rollback.name,
                    destination_name,
                    parent_descriptor=parent_descriptor,
                )
                rollback.name = destination_name
                rollback_published = True
                _verify_published_node(parent_descriptor, destination_name, rollback)
                finalize_rollback()
                _verify_published_node(parent_descriptor, destination_name, rollback)
            except BaseException:
                if rollback_published:
                    private_mode = 0o700 if rollback.kind == "directory" else 0o600
                    os.fchmod(rollback.descriptor, private_mode)
                    _rename_after_public_mutation(
                        destination_name,
                        rollback_private_name,
                        parent_descriptor=parent_descriptor,
                    )
                    rollback.name = rollback_private_name
                validate_reserve(backup_name)
                _rename_after_public_mutation(
                    backup_name,
                    destination_name,
                    parent_descriptor=parent_descriptor,
                )
                finalize_reserve(destination_name)
            else:
                _cleanup_backup_best_effort(parent_descriptor, backup_name)
        raise
    _cleanup_backup_best_effort(parent_descriptor, backup_name)


def _build_file_manifest_posix(source: Path, allowed_root: Path) -> _FileManifest:
    parent, root_device = _open_file_parent(source, allowed_root, role="source")
    descriptor = -1
    try:
        try:
            before = os.stat(source.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise UnsafeStagingError(f"source path does not exist: {source}") from exc
        kind = _validate_type(before, path=source, role="source", root_device=root_device)
        if kind != "file":
            raise UnsafeStagingError(f"source is not a regular file: {source}")
        descriptor = os.open(source.name, _FILE_FLAGS, dir_fd=parent)
        opened = os.fstat(descriptor)
        if _fingerprint(opened) != _fingerprint(before):
            raise _changed(source, "file was replaced")
        digest = _hash_descriptor(descriptor)
        after = os.fstat(descriptor)
        if _fingerprint(after) != _fingerprint(opened):
            raise _changed(source, "file changed while it was validated")
        return _FileManifest(source, allowed_root, _entry_from_stat((), kind, after, digest=digest))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _build_file_manifest_fallback(source: Path, allowed_root: Path) -> _FileManifest:
    _relative_to_allowed(source, allowed_root)
    _validate_fallback_components(allowed_root, purpose="source allowed root")
    _validate_fallback_components(source.parent, purpose="source")
    root_device = _source_device(allowed_root.lstat(), path=allowed_root)
    before = source.lstat()
    kind = _validate_type(before, path=source, role="source", root_device=root_device)
    if kind != "file":
        raise UnsafeStagingError(f"source is not a regular file: {source}")
    digest = _hash_path_checked(source, before, role="source", root_device=root_device)
    return _FileManifest(source, allowed_root, _entry_from_stat((), kind, before, digest=digest))


def _build_file_manifest(source: Path | str, allowed_root: Path | str) -> _FileManifest:
    source_path = _absolute_lexical(source)
    root_path = _absolute_lexical(allowed_root)
    if _DESCRIPTOR_BACKEND:
        return _build_file_manifest_posix(source_path, root_path)
    return _build_file_manifest_fallback(source_path, root_path)


def _copy_file_into_node(manifest: _FileManifest, staged: _StagedNode) -> None:
    parent, _ = _open_file_parent(manifest.source, manifest.allowed_root, role="source")
    source_descriptor = -1
    try:
        source_descriptor = os.open(manifest.source.name, _FILE_FLAGS, dir_fd=parent)
        _verify_entry(manifest.entry, os.fstat(source_descriptor), manifest.source)
        digest = hashlib.sha256()
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while data := os.read(source_descriptor, _CHUNK_SIZE):
            digest.update(data)
            view = memoryview(data)
            while view:
                written = os.write(staged.descriptor, view)
                view = view[written:]
        _verify_entry(manifest.entry, os.fstat(source_descriptor), manifest.source)
        if digest.hexdigest() != manifest.entry.digest:
            raise _changed(manifest.source, "file contents changed")
        named = os.stat(manifest.source.name, dir_fd=parent, follow_symlinks=False)
        _verify_entry(manifest.entry, named, manifest.source)
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(parent)


def _validate_file_node_exact(
    parent: int,
    staged: _StagedNode,
    manifest: _FileManifest,
    *,
    final_mode: bool,
) -> None:
    _verify_staged_node_name(parent, staged)
    metadata = os.fstat(staged.descriptor)
    expected_mode = manifest.entry.mode if final_mode else 0o600
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != manifest.entry.size
        or stat.S_IMODE(metadata.st_mode) & 0o777 != expected_mode
    ):
        raise UnsafeStagingError("staging file has an unsafe type, link count, size, or mode")
    readable = os.open(staged.name, _FILE_FLAGS, dir_fd=parent)
    try:
        opened = os.fstat(readable)
        before = _fingerprint(opened)
        if (
            (opened.st_dev, opened.st_ino) != staged.identity
            or _hash_descriptor(readable) != manifest.entry.digest
            or _fingerprint(os.fstat(readable)) != before
        ):
            raise UnsafeStagingError("staging file contents, digest, or identity changed")
    finally:
        os.close(readable)


def _validate_private_file_node(parent: int, staged: _StagedNode, manifest: _FileManifest) -> None:
    _validate_file_node_exact(parent, staged, manifest, final_mode=False)


def _validate_existing_destination_file(metadata: os.stat_result, destination: Path) -> None:
    if _is_link(metadata):
        raise UnsafeStagingError(f"destination is a symlink or reparse point: {destination}")
    if not stat.S_ISREG(metadata.st_mode):
        raise UnsafeStagingError(f"destination is not a regular file: {destination}")
    if metadata.st_nlink != 1:
        raise UnsafeStagingError(f"destination is a hard-linked file with multiple links: {destination}")


def _safe_remove_fallback(path: Path) -> None:
    if not os.path.lexists(path):
        return
    metadata = path.lstat()
    if _is_link(metadata):
        raise UnsafeStagingError(f"refusing to remove fallback staging symlink: {path}")
    if stat.S_ISDIR(metadata.st_mode):
        path.chmod(0o700)
        for child in path.iterdir():
            _safe_remove_fallback(child)
        path.rmdir()
    elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
        path.chmod(0o600)
        path.unlink()
    else:
        raise UnsafeStagingError(f"refusing to remove unsafe fallback staging entry: {path}")


def _fallback_mode_matches(actual: int, expected: int) -> bool:
    """Check the mode bits that the platform's path-based chmod can set."""

    if _WINDOWS_CHMOD_SEMANTICS:
        return bool(actual & stat.S_IWRITE) == bool(expected & stat.S_IWRITE)
    return stat.S_IMODE(actual) & 0o777 == expected


def _apply_fallback_open_file_mode(path: Path, descriptor: int, mode: int) -> None:
    """Apply a mode through the strongest API available and verify identity."""

    opened_before = os.fstat(descriptor)
    named_before = path.lstat()
    if (
        _is_link(named_before)
        or not stat.S_ISREG(opened_before.st_mode)
        or not stat.S_ISREG(named_before.st_mode)
        or opened_before.st_nlink != 1
        or named_before.st_nlink != 1
        or not _fallback_opened_matches_named(opened_before, named_before)
    ):
        raise UnsafeStagingError("fallback staging file changed before its mode was applied")
    fchmod = getattr(os, "fchmod", None)
    if fchmod is not None:
        fchmod(descriptor, mode)
    else:
        path.chmod(mode)
    opened_after = os.fstat(descriptor)
    named_after = path.lstat()
    if (
        _is_link(named_after)
        or _node_identity(opened_after) != _node_identity(opened_before)
        or _node_identity(named_after) != _node_identity(named_before)
        or not _fallback_opened_matches_named(opened_after, named_after)
        or not _fallback_mode_matches(opened_after.st_mode, mode)
        or not _fallback_mode_matches(named_after.st_mode, mode)
    ):
        raise UnsafeStagingError("fallback staging file identity or mode changed while applying its mode")


def _copy_manifest_file_fallback(manifest: _TreeManifest, entry: _ManifestEntry, destination: Path) -> None:
    source = manifest.source.joinpath(*entry.parts)
    before = source.lstat()
    _verify_entry(entry, before, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        metadata = destination.lstat()
        if _is_link(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise UnsafeStagingError(f"private destination collision is unsafe: {destination}")
        destination.unlink()
    source_descriptor = _open_fallback_regular(
        source,
        before,
        role=manifest.role,
        root_device=manifest.root.device,
    )
    destination_descriptor = -1
    try:
        source_opened = os.fstat(source_descriptor)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_FLAG,
            0o600,
        )
        _apply_fallback_open_file_mode(destination, destination_descriptor, 0o600)
        digest = hashlib.sha256()
        while data := os.read(source_descriptor, _CHUNK_SIZE):
            digest.update(data)
            view = memoryview(data)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        source_after = os.fstat(source_descriptor)
        named_after = source.lstat()
        if _fingerprint(source_after) != _fingerprint(source_opened) or not _fallback_opened_matches_named(
            source_after, named_after
        ):
            raise _changed(source, "file changed while it was copied")
        _verify_entry(entry, named_after, source)
        if digest.hexdigest() != entry.digest:
            raise _changed(source, "file contents changed")
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _apply_manifest_fallback(stage: Path, manifest: _TreeManifest) -> None:
    for entry in manifest.entries:
        destination = stage.joinpath(*entry.parts)
        if entry.kind == "directory":
            if os.path.lexists(destination):
                metadata = destination.lstat()
                if _is_link(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    raise UnsafeStagingError(f"private destination collision is unsafe: {destination}")
            else:
                destination.mkdir(mode=0o700)
                destination.chmod(0o700)
        else:
            _copy_manifest_file_fallback(manifest, entry, destination)
    current = _build_tree_manifest(
        manifest.source,
        allowed_root=manifest.allowed_root,
        ignore=manifest.ignore,
        role=manifest.role,
    )
    if current != manifest:
        raise _changed(manifest.source, "tree changed while it was copied")


def _apply_modes_fallback(stage: Path, manifests: tuple[_TreeManifest, ...]) -> None:
    entries = _merged_entries(manifests)
    for entry in (item for item in entries.values() if item.kind == "file"):
        path = stage.joinpath(*entry.parts)
        path.chmod(entry.mode)
        if not _fallback_mode_matches(path.lstat().st_mode, entry.mode):
            raise UnsafeStagingError("published fallback file mode could not be applied")
    directories = [item for item in entries.values() if item.kind == "directory"]
    for entry in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path = stage.joinpath(*entry.parts)
        path.chmod(entry.mode)
        if not _fallback_mode_matches(path.lstat().st_mode, entry.mode):
            raise UnsafeStagingError("published fallback directory mode could not be applied")


def _validate_fallback_tree_exact_pass(
    stage: Path,
    manifests: tuple[_TreeManifest, ...],
    *,
    child_final_modes: bool,
    root_mode: int,
    reverse: bool,
) -> None:
    expected = _merged_entries(manifests)
    children: dict[tuple[str, ...], dict[str, _ManifestEntry]] = {}
    for entry in expected.values():
        children.setdefault(entry.parts[:-1], {})[entry.parts[-1]] = entry
    root_device = stage.lstat().st_dev

    def verify_directory(directory: Path, parts: tuple[str, ...]) -> None:
        before = directory.lstat()
        entry = expected.get(parts)
        expected_mode = root_mode if not parts else entry.mode if child_final_modes and entry is not None else 0o700
        if (
            _is_link(before)
            or not stat.S_ISDIR(before.st_mode)
            or before.st_dev != root_device
            or not _fallback_mode_matches(before.st_mode, expected_mode)
        ):
            raise UnsafeStagingError("fallback staging directory is unsafe")
        before_fingerprint = _fingerprint(before)
        before_names = sorted(item.name for item in os.scandir(directory))
        expected_children = children.get(parts, {})
        if set(before_names) != set(expected_children):
            raise UnsafeStagingError("fallback staging tree has unexpected or missing entries")
        for name in sorted(expected_children, reverse=reverse):
            child_entry = expected_children[name]
            child = directory / name
            metadata = child.lstat()
            if _is_link(metadata) or metadata.st_dev != root_device:
                raise UnsafeStagingError("fallback staging tree contains a link or mount crossing")
            if child_entry.kind == "directory":
                verify_directory(child, child_entry.parts)
            else:
                expected_file_mode = child_entry.mode if child_final_modes else 0o600
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_size != child_entry.size
                    or not _fallback_mode_matches(metadata.st_mode, expected_file_mode)
                ):
                    raise UnsafeStagingError("fallback staging file is unsafe")
                if (
                    _hash_path_checked(child, metadata, role="fallback staging", root_device=root_device)
                    != child_entry.digest
                ):
                    raise UnsafeStagingError("fallback staging file contents or digest changed")
                if _fingerprint(child.lstat()) != _fingerprint(metadata):
                    raise UnsafeStagingError("fallback staging file changed after exact verification")
        after_names = sorted(item.name for item in os.scandir(directory))
        if after_names != before_names or _fingerprint(directory.lstat()) != before_fingerprint:
            raise UnsafeStagingError("fallback staging directory changed during exact verification")

    verify_directory(stage, ())


def _validate_fallback_tree_exact_stable(
    stage: Path,
    manifests: tuple[_TreeManifest, ...],
    *,
    child_final_modes: bool,
    root_mode: int,
) -> None:
    _validate_fallback_tree_exact_pass(
        stage,
        manifests,
        child_final_modes=child_final_modes,
        root_mode=root_mode,
        reverse=False,
    )
    _validate_fallback_tree_exact_pass(
        stage,
        manifests,
        child_final_modes=child_final_modes,
        root_mode=root_mode,
        reverse=True,
    )


def _validate_fallback_stage_exact(stage: Path, manifests: tuple[_TreeManifest, ...]) -> None:
    _validate_fallback_tree_exact_stable(
        stage,
        manifests,
        child_final_modes=False,
        root_mode=0o700,
    )


def _prepare_hidden_fallback_tree(stage: Path, manifests: tuple[_TreeManifest, ...]) -> None:
    _apply_modes_fallback(stage, manifests)
    _validate_fallback_tree_exact_stable(
        stage,
        manifests,
        child_final_modes=True,
        root_mode=0o700,
    )


def _validate_hidden_fallback_tree(stage: Path, manifests: tuple[_TreeManifest, ...]) -> None:
    _validate_fallback_tree_exact_stable(
        stage,
        manifests,
        child_final_modes=True,
        root_mode=0o700,
    )


def _expose_fallback_tree_root_exact(stage: Path, manifests: tuple[_TreeManifest, ...]) -> None:
    _validate_hidden_fallback_tree(stage, manifests)
    before = stage.lstat()
    root_mode = manifests[-1].root.mode
    stage.chmod(root_mode)
    after = stage.lstat()
    if _node_identity(after) != _node_identity(before) or not _fallback_mode_matches(after.st_mode, root_mode):
        raise UnsafeStagingError("published fallback root identity or mode changed during exposure")


def _verify_fallback_publication(path: Path, expected: os.stat_result, *, kind: str) -> None:
    metadata = path.lstat()
    expected_type = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
    if (
        _is_link(metadata)
        or not expected_type(metadata.st_mode)
        or _node_identity(metadata) != _node_identity(expected)
    ):
        raise UnsafeStagingError(f"published fallback {kind} has an unexpected identity or type")


def _prepare_fallback_moved_backup(
    backup: Path,
    expected: os.stat_result,
    *,
    kind: str,
) -> None:
    metadata = backup.lstat()
    expected_type = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
    if (
        _is_link(metadata)
        or not expected_type(metadata.st_mode)
        or _node_identity(metadata) != _node_identity(expected)
    ):
        raise UnsafeStagingError("destination changed while it was moved to the fallback backup")
    private_mode = 0o700 if kind == "directory" else 0o600
    backup.chmod(private_mode)
    private = backup.lstat()
    if _node_identity(private) != _node_identity(expected) or not _fallback_mode_matches(private.st_mode, private_mode):
        raise UnsafeStagingError("fallback destination backup could not be made private")


def _cleanup_fallback_backup(backup: Path) -> None:
    with suppress(OSError, UnsafeStagingError):
        _safe_remove_fallback(backup)


def _copytree_fallback(
    source: Path,
    destination: Path,
    *,
    dirs_exist_ok: bool,
    replace_existing: bool,
    ignore: IgnoreCallback | None,
    allowed_root: Path,
) -> None:
    source_manifest = _build_tree_manifest(source, allowed_root=allowed_root, ignore=ignore)
    _validate_fallback_components(destination.parent, purpose="destination", create=True)
    existing_manifest: _TreeManifest | None = None
    expected_existing: os.stat_result | None = None
    if os.path.lexists(destination):
        expected_existing = destination.lstat()
        if _is_link(expected_existing):
            raise UnsafeStagingError(f"destination is a symlink or reparse point: {destination}")
        if not dirs_exist_ok and not replace_existing:
            raise FileExistsError(destination)
        if not stat.S_ISDIR(expected_existing.st_mode):
            raise UnsafeStagingError(f"destination is not a directory: {destination}")
        existing_manifest = _build_tree_manifest(destination, allowed_root=destination, role="destination")
    stage = destination.parent / f".{destination.name}.staging-{secrets.token_hex(8)}"
    stage.mkdir(mode=0o700)
    stage.chmod(0o700)
    backup: Path | None = None
    rollback_stage: Path | None = None
    try:
        manifests = (
            (source_manifest,)
            if replace_existing
            else tuple(item for item in (existing_manifest, source_manifest) if item is not None)
        )
        for manifest in manifests:
            _apply_manifest_fallback(stage, manifest)
        _validate_fallback_stage_exact(stage, manifests)
        _prepare_hidden_fallback_tree(stage, manifests)
        if existing_manifest is not None:
            rollback_stage = destination.parent / f".{destination.name}.rollback-{secrets.token_hex(8)}"
            rollback_stage.mkdir(mode=0o700)
            rollback_stage.chmod(0o700)
            _apply_manifest_fallback(rollback_stage, existing_manifest)
            _validate_fallback_stage_exact(rollback_stage, (existing_manifest,))
            _prepare_hidden_fallback_tree(rollback_stage, (existing_manifest,))
            current_destination = _build_tree_manifest(
                destination,
                allowed_root=destination,
                role="destination",
            )
            if current_destination != existing_manifest:
                raise UnsafeStagingError("destination changed while its fallback snapshot was built")
        _validate_fallback_components(destination.parent, purpose="destination")
        staged_identity = stage.lstat()
        if expected_existing is None:
            if os.path.lexists(destination):
                raise FileExistsError(destination)
            private_stage = stage
            stage.rename(destination)
            stage = destination
            try:
                _verify_fallback_publication(destination, staged_identity, kind="directory")
                _expose_fallback_tree_root_exact(destination, manifests)
                _verify_fallback_publication(destination, staged_identity, kind="directory")
            except BaseException:
                destination.rename(private_stage)
                stage = private_stage
                raise
        else:
            current = destination.lstat()
            if _fingerprint(current) != _fingerprint(expected_existing):
                raise UnsafeStagingError("destination changed before transactional publication")
            backup = destination.parent / f".{destination.name}.backup-{secrets.token_hex(8)}"
            destination.rename(backup)
            try:
                _prepare_fallback_moved_backup(backup, expected_existing, kind="directory")
                if rollback_stage is None or existing_manifest is None:
                    raise UnsafeStagingError("fallback publication is missing its private rollback snapshot")
                _validate_hidden_fallback_tree(rollback_stage, (existing_manifest,))
                private_stage = stage
                stage.rename(destination)
                stage = destination
                _verify_fallback_publication(destination, staged_identity, kind="directory")
                _expose_fallback_tree_root_exact(destination, manifests)
                _verify_fallback_publication(destination, staged_identity, kind="directory")
            except BaseException:
                if stage == destination:
                    destination.rename(private_stage)
                    stage = private_stage
                if rollback_stage is None or existing_manifest is None:
                    raise
                rollback_private = rollback_stage
                rollback_identity = rollback_stage.lstat()
                rollback_published = False
                try:
                    _validate_hidden_fallback_tree(rollback_stage, (existing_manifest,))
                    rollback_stage.rename(destination)
                    rollback_stage = destination
                    rollback_published = True
                    _verify_fallback_publication(destination, rollback_identity, kind="directory")
                    _expose_fallback_tree_root_exact(destination, (existing_manifest,))
                    _verify_fallback_publication(destination, rollback_identity, kind="directory")
                except BaseException:
                    if rollback_published:
                        destination.chmod(0o700)
                        destination.rename(rollback_private)
                        rollback_stage = rollback_private
                    _validate_hidden_fallback_tree(backup, (existing_manifest,))
                    reserve_identity = backup.lstat()
                    backup.rename(destination)
                    backup = None
                    _verify_fallback_publication(destination, reserve_identity, kind="directory")
                    _expose_fallback_tree_root_exact(destination, (existing_manifest,))
                    _verify_fallback_publication(destination, reserve_identity, kind="directory")
                else:
                    _cleanup_fallback_backup(backup)
                    backup = None
                raise
            _cleanup_fallback_backup(backup)
            backup = None
    finally:
        if os.path.lexists(stage) and stage != destination:
            _safe_remove_fallback(stage)
        if rollback_stage is not None and os.path.lexists(rollback_stage) and rollback_stage != destination:
            _safe_remove_fallback(rollback_stage)


def copytree_secure(
    source: Path | str,
    destination: Path | str,
    *,
    dirs_exist_ok: bool = False,
    replace_existing: bool = False,
    ignore: IgnoreCallback | None = None,
    allowed_root: Path | str | None = None,
) -> None:
    """Copy a link-free authored tree and publish it transactionally.

    Every non-ignored source entry must be a single-link regular file or a
    directory on the source root's device.  Existing destination merges are
    built as a complete private sibling and swapped only after all validation
    and copying succeeds; with quiescent scoped paths, the old tree is restored
    if publication fails.

    ``replace_existing`` publishes the source as an exact replacement instead
    of retaining destination-only entries. It is mutually exclusive with
    ``dirs_exist_ok``.

    Security boundary: *all* concurrent mutation by another process running as
    the same UID is outside this function's security guarantee, including a
    one-shot change after the final validation sample.  Callers must keep
    source, stage, destination, rollback, and reserve paths exclusive and
    quiescent, or use filesystem isolation/snapshots when coherent
    hostile-concurrency semantics are required.
    """

    if dirs_exist_ok and replace_existing:
        raise ValueError("dirs_exist_ok and replace_existing are mutually exclusive")

    source_path = _absolute_lexical(source)
    destination_path = _absolute_lexical(destination)
    root_path = _absolute_lexical(allowed_root if allowed_root is not None else source_path)
    if not _DESCRIPTOR_BACKEND or _ATOMIC_RENAME is None:
        _copytree_fallback(
            source_path,
            destination_path,
            dirs_exist_ok=dirs_exist_ok,
            replace_existing=replace_existing,
            ignore=ignore,
            allowed_root=root_path,
        )
        return

    source_manifest = _build_tree_manifest(source_path, allowed_root=root_path, ignore=ignore)
    parent = _open_absolute_directory(destination_path.parent, purpose="destination", create=True)
    staged: _StagedNode | None = None
    rollback: _StagedNode | None = None
    published = False
    fallback_required = False
    primary_error = False
    try:
        existing = _destination_metadata(parent, destination_path.name)
        destination_manifest: _TreeManifest | None = None
        if existing is not None:
            if _is_link(existing):
                raise UnsafeStagingError(f"destination is a symlink or reparse point: {destination_path}")
            if not dirs_exist_ok and not replace_existing:
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fspath(destination_path))
            if not stat.S_ISDIR(existing.st_mode):
                raise UnsafeStagingError(f"destination is not a directory: {destination_path}")
            try:
                destination_manifest = _build_tree_manifest(
                    destination_path,
                    allowed_root=destination_path,
                    role="destination",
                )
            except UnsafeStagingError as exc:
                raise UnsafeStagingError(f"unsafe destination: {exc}") from exc
        manifests = (
            (source_manifest,)
            if replace_existing
            else tuple(item for item in (destination_manifest, source_manifest) if item is not None)
        )
        staged = _stage_manifests(parent, destination_path.name, manifests)
        if destination_manifest is not None:
            rollback = _stage_manifests(
                parent,
                f"{destination_path.name}.rollback",
                (destination_manifest,),
            )
            current_destination = _build_tree_manifest(
                destination_path,
                allowed_root=destination_path,
                role="destination",
            )
            if current_destination != destination_manifest:
                raise UnsafeStagingError("destination changed while its transactional snapshot was built")
        try:
            _publish_transactional(
                parent,
                destination_path.name,
                staged,
                expected_existing=existing,
                rollback=rollback,
                backup_path=destination_path,
                prepublish=lambda: _prepare_hidden_final_tree(staged, manifests),
                finalize=lambda: _expose_tree_root_exact(staged, manifests),
                validate_rollback=(
                    None
                    if rollback is None or destination_manifest is None
                    else lambda: _prepare_hidden_final_tree(rollback, (destination_manifest,))
                ),
                finalize_rollback=(
                    None
                    if rollback is None or destination_manifest is None
                    else lambda: _expose_tree_root_exact(rollback, (destination_manifest,))
                ),
                validate_reserve=(
                    None
                    if destination_manifest is None
                    else lambda name: _validate_named_hidden_tree(parent, name, destination_manifest)
                ),
                finalize_reserve=(
                    None
                    if destination_manifest is None
                    else lambda name: _expose_named_tree_root(parent, name, destination_manifest)
                ),
            )
        except _AtomicRenameUnsupported:
            fallback_required = True
        else:
            published = True
    except BaseException:
        primary_error = True
        raise
    finally:
        if staged is not None:
            if published:
                os.close(staged.descriptor)
            else:
                _cleanup_staged_node(parent, staged, suppress_errors=primary_error)
        if rollback is not None:
            if rollback.name == destination_path.name:
                os.close(rollback.descriptor)
            else:
                _cleanup_staged_node(
                    parent,
                    rollback,
                    suppress_errors=primary_error or published or fallback_required,
                )
        os.close(parent)
    if fallback_required:
        _copytree_fallback(
            source_path,
            destination_path,
            dirs_exist_ok=dirs_exist_ok,
            replace_existing=replace_existing,
            ignore=ignore,
            allowed_root=root_path,
        )


def _validate_fallback_file_exact(
    path: Path,
    manifest: _FileManifest,
    *,
    final_mode: bool,
) -> None:
    metadata = path.lstat()
    expected_mode = manifest.entry.mode if final_mode else 0o600
    if (
        _is_link(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != manifest.entry.size
        or not _fallback_mode_matches(metadata.st_mode, expected_mode)
    ):
        raise UnsafeStagingError("fallback staging file has an unsafe type, link count, size, or mode")
    if (
        _hash_path_checked(path, metadata, role="fallback staging", root_device=metadata.st_dev)
        != manifest.entry.digest
    ):
        raise UnsafeStagingError("fallback staging file contents or digest changed")


def _stage_fallback_file(manifest: _FileManifest, stage: Path) -> None:
    before = manifest.source.lstat()
    _verify_entry(manifest.entry, before, manifest.source)
    source_descriptor = _open_fallback_regular(
        manifest.source,
        before,
        role="source",
        root_device=manifest.entry.device,
    )
    destination_descriptor = -1
    try:
        source_opened = os.fstat(source_descriptor)
        destination_descriptor = os.open(
            stage,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_FLAG,
            0o600,
        )
        _apply_fallback_open_file_mode(stage, destination_descriptor, 0o600)
        digest = hashlib.sha256()
        while data := os.read(source_descriptor, _CHUNK_SIZE):
            digest.update(data)
            view = memoryview(data)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        source_after = os.fstat(source_descriptor)
        named_after = manifest.source.lstat()
        if _fingerprint(source_after) != _fingerprint(source_opened) or not _fallback_opened_matches_named(
            source_after, named_after
        ):
            raise _changed(manifest.source, "file changed while it was copied")
        _verify_entry(manifest.entry, named_after, manifest.source)
        if digest.hexdigest() != manifest.entry.digest:
            raise _changed(manifest.source, "file contents changed")
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    _validate_fallback_file_exact(stage, manifest, final_mode=False)


def _finalize_fallback_file(path: Path, manifest: _FileManifest) -> None:
    path.chmod(manifest.entry.mode)
    _validate_fallback_file_exact(path, manifest, final_mode=True)


def _copy_file_fallback(source: Path, destination: Path, *, allowed_root: Path) -> None:
    manifest = _build_file_manifest(source, allowed_root)
    _validate_fallback_components(destination.parent, purpose="destination", create=True)
    existing = destination.lstat() if os.path.lexists(destination) else None
    rollback_manifest: _FileManifest | None = None
    if existing is not None:
        _validate_existing_destination_file(existing, destination)
        rollback_manifest = _build_file_manifest(destination, destination.parent)
        _verify_entry(rollback_manifest.entry, existing, destination)
    stage = destination.parent / f".{destination.name}.staging-{secrets.token_hex(8)}"
    rollback_stage: Path | None = None
    backup: Path | None = None
    try:
        _stage_fallback_file(manifest, stage)
        if rollback_manifest is not None:
            rollback_stage = destination.parent / f".{destination.name}.rollback-{secrets.token_hex(8)}"
            _stage_fallback_file(rollback_manifest, rollback_stage)
            _verify_entry(rollback_manifest.entry, destination.lstat(), destination)
        staged_identity = stage.lstat()
        _validate_fallback_components(destination.parent, purpose="destination")
        if existing is None:
            if os.path.lexists(destination):
                raise FileExistsError(destination)
            private_stage = stage
            stage.rename(destination)
            stage = destination
            try:
                _verify_fallback_publication(destination, staged_identity, kind="file")
                _finalize_fallback_file(destination, manifest)
                _verify_fallback_publication(destination, staged_identity, kind="file")
            except BaseException:
                destination.rename(private_stage)
                stage = private_stage
                raise
        else:
            if _fingerprint(destination.lstat()) != _fingerprint(existing):
                raise UnsafeStagingError("destination changed before transactional publication")
            backup = destination.parent / f".{destination.name}.backup-{secrets.token_hex(8)}"
            destination.rename(backup)
            try:
                _prepare_fallback_moved_backup(backup, existing, kind="file")
                if rollback_stage is None or rollback_manifest is None:
                    raise UnsafeStagingError("fallback file publication is missing its private rollback snapshot")
                _validate_fallback_file_exact(rollback_stage, rollback_manifest, final_mode=False)
                private_stage = stage
                stage.rename(destination)
                stage = destination
                _verify_fallback_publication(destination, staged_identity, kind="file")
                _finalize_fallback_file(destination, manifest)
                _verify_fallback_publication(destination, staged_identity, kind="file")
            except BaseException:
                if stage == destination:
                    destination.rename(private_stage)
                    stage = private_stage
                if rollback_stage is None or rollback_manifest is None:
                    raise
                rollback_private = rollback_stage
                rollback_identity = rollback_stage.lstat()
                rollback_published = False
                try:
                    _validate_fallback_file_exact(rollback_stage, rollback_manifest, final_mode=False)
                    rollback_stage.rename(destination)
                    rollback_stage = destination
                    rollback_published = True
                    _verify_fallback_publication(destination, rollback_identity, kind="file")
                    _finalize_fallback_file(destination, rollback_manifest)
                    _verify_fallback_publication(destination, rollback_identity, kind="file")
                except BaseException:
                    if rollback_published:
                        destination.chmod(0o600)
                        destination.rename(rollback_private)
                        rollback_stage = rollback_private
                    _validate_fallback_file_exact(backup, rollback_manifest, final_mode=False)
                    reserve_identity = backup.lstat()
                    backup.rename(destination)
                    backup = None
                    _verify_fallback_publication(destination, reserve_identity, kind="file")
                    _finalize_fallback_file(destination, rollback_manifest)
                    _verify_fallback_publication(destination, reserve_identity, kind="file")
                else:
                    _cleanup_fallback_backup(backup)
                    backup = None
                raise
            _cleanup_fallback_backup(backup)
            backup = None
    finally:
        if os.path.lexists(stage) and stage != destination:
            _safe_remove_fallback(stage)
        if rollback_stage is not None and os.path.lexists(rollback_stage) and rollback_stage != destination:
            _safe_remove_fallback(rollback_stage)


def _apply_final_file_mode(descriptor: int, mode: int) -> None:
    os.fchmod(descriptor, mode)
    if stat.S_IMODE(os.fstat(descriptor).st_mode) & 0o777 != mode:
        raise UnsafeStagingError("published staging file mode could not be applied")


def _finalize_file_exact(
    parent: int,
    node: _StagedNode,
    manifest: _FileManifest,
) -> None:
    _apply_final_file_mode(node.descriptor, manifest.entry.mode)
    _validate_file_node_exact(parent, node, manifest, final_mode=True)


def _validate_named_hidden_file(
    parent: int,
    name: str,
    manifest: _FileManifest,
) -> None:
    node = _open_named_node(parent, name, kind="file")
    try:
        _validate_file_node_exact(parent, node, manifest, final_mode=False)
    finally:
        os.close(node.descriptor)


def _expose_named_file(
    parent: int,
    name: str,
    manifest: _FileManifest,
) -> None:
    node = _open_named_node(parent, name, kind="file")
    try:
        _finalize_file_exact(parent, node, manifest)
    finally:
        os.close(node.descriptor)


def copy_file_secure(
    source: Path | str,
    destination: Path | str,
    *,
    allowed_root: Path | str,
) -> None:
    """Copy one authored regular file with atomic overwrite and rollback.

    The rollback guarantee assumes the source, destination, private stage, and
    reserve remain quiescent.  Concurrent mutation of any scoped path by a
    process running as the same UID--even a one-shot change after validation--is
    outside the security guarantee.  Use exclusive filesystem ownership,
    isolation, or snapshot support for that threat model.
    """

    source_path = _absolute_lexical(source)
    destination_path = _absolute_lexical(destination)
    root_path = _absolute_lexical(allowed_root)
    if not _DESCRIPTOR_BACKEND or _ATOMIC_RENAME is None:
        _copy_file_fallback(source_path, destination_path, allowed_root=root_path)
        return

    manifest = _build_file_manifest(source_path, root_path)
    parent = _open_absolute_directory(destination_path.parent, purpose="destination", create=True)
    staged: _StagedNode | None = None
    rollback: _StagedNode | None = None
    rollback_manifest: _FileManifest | None = None
    published = False
    fallback_required = False
    primary_error = False
    try:
        existing = _destination_metadata(parent, destination_path.name)
        if existing is not None:
            _validate_existing_destination_file(existing, destination_path)
            rollback_manifest = _build_file_manifest(destination_path, destination_path.parent)
            _verify_entry(rollback_manifest.entry, existing, destination_path)
        staged = _create_staged_node(parent, prefix=destination_path.name or "file", kind="file")
        _copy_file_into_node(manifest, staged)
        _verify_staged_node_name(parent, staged)
        if rollback_manifest is not None:
            rollback = _create_staged_node(
                parent,
                prefix=f"{destination_path.name}.rollback",
                kind="file",
            )
            _copy_file_into_node(rollback_manifest, rollback)
            _validate_private_file_node(parent, rollback, rollback_manifest)
            current = os.stat(destination_path.name, dir_fd=parent, follow_symlinks=False)
            _verify_entry(rollback_manifest.entry, current, destination_path)
        try:
            _publish_transactional(
                parent,
                destination_path.name,
                staged,
                expected_existing=existing,
                rollback=rollback,
                backup_path=destination_path,
                prepublish=lambda: _validate_private_file_node(parent, staged, manifest),
                finalize=lambda: _finalize_file_exact(parent, staged, manifest),
                validate_rollback=(
                    None
                    if rollback is None or rollback_manifest is None
                    else lambda: _validate_private_file_node(parent, rollback, rollback_manifest)
                ),
                finalize_rollback=(
                    None
                    if rollback is None or rollback_manifest is None
                    else lambda: _finalize_file_exact(parent, rollback, rollback_manifest)
                ),
                validate_reserve=(
                    None
                    if rollback_manifest is None
                    else lambda name: _validate_named_hidden_file(parent, name, rollback_manifest)
                ),
                finalize_reserve=(
                    None
                    if rollback_manifest is None
                    else lambda name: _expose_named_file(parent, name, rollback_manifest)
                ),
            )
        except _AtomicRenameUnsupported:
            fallback_required = True
        else:
            published = True
    except BaseException:
        primary_error = True
        raise
    finally:
        if staged is not None:
            if published:
                os.close(staged.descriptor)
            else:
                _cleanup_staged_node(parent, staged, suppress_errors=primary_error)
        if rollback is not None:
            if rollback.name == destination_path.name:
                os.close(rollback.descriptor)
            else:
                _cleanup_staged_node(
                    parent,
                    rollback,
                    suppress_errors=primary_error or published or fallback_required,
                )
        os.close(parent)
    if fallback_required:
        _copy_file_fallback(source_path, destination_path, allowed_root=root_path)
