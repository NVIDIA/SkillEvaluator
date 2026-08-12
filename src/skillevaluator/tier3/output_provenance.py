# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-bound provenance for evaluator-owned generated output trees."""

from __future__ import annotations

import base64
import contextlib
import hmac
import os
import secrets
import stat
import sys
import time
from pathlib import Path

from skillevaluator.tier3.case_ids import validate_output_directory_path
from skillevaluator.tier3.harbor.secure_copy import _DESCRIPTOR_BACKEND, _DIRECTORY_FLAGS, _remove_tree_at
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

GENERATED_OUTPUT_MARKER = ".skillevaluator-generated-output"
OUTPUT_PROVENANCE_KEY_ENV = "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"

_KEY_BYTES = 32
_MAX_STORED_KEY_BYTES = 4096
_KEY_TEMP_PREFIX = ".output-provenance.key.tmp-"
_MARKER_PREFIX = b"SkillEvaluator generated output v2\n"
_MARKER_CONTEXT = b"skillevaluator.generated-output.v2\0"
_MARKER_SIZE = len(_MARKER_PREFIX) + len(base64.urlsafe_b64encode(bytes(_KEY_BYTES)).rstrip(b"=")) + 1
_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE = os.name == "posix"


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _node_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink, metadata.st_size


def _artifact_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return identity and content-change fields from one stat API family."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_read_metadata(
    before: os.stat_result,
    opened: os.stat_result,
    after: os.stat_result,
) -> os.stat_result | None:
    """Select comparable metadata after a descriptor-pinned read."""
    if _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE:
        if _artifact_fingerprint(opened) != _artifact_fingerprint(before) or _artifact_fingerprint(
            after
        ) != _artifact_fingerprint(opened):
            return None
        return opened
    if _artifact_fingerprint(after) != _artifact_fingerprint(before) or opened.st_size != after.st_size:
        return None
    return after


def _fsync_directory(path: Path) -> None:
    """Durably publish a directory entry where the platform supports it."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_same_file(path: Path, expected: os.stat_result) -> bool:
    """Unlink only the exact regular file created by the current operation."""
    try:
        observed = path.lstat()
    except OSError:
        return False
    if _is_link_or_reparse(observed) or not stat.S_ISREG(observed.st_mode):
        return False
    if not os.path.samestat(expected, observed):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _inspect_atomic_destination(path: Path) -> os.stat_result | None:
    """Return destination metadata after rejecting unsafe file types."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Generated output artifact must be a single-link regular file: {path}")
    return metadata


def write_output_file_atomically(path: Path, payload: bytes) -> None:
    """Durably replace one evaluator-owned output file with complete bytes."""
    validate_output_directory_path(path.parent)
    existing = _inspect_atomic_destination(path)
    existing_mode = stat.S_IMODE(existing.st_mode) if existing is not None else None
    temporary: Path | None = None
    temporary_named_metadata: os.stat_result | None = None
    descriptor = -1
    try:
        for _ in range(32):
            candidate = path.parent / f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                    0o666,
                )
            except FileExistsError:
                continue
            temporary = candidate
            break
        if temporary is None or descriptor < 0:
            raise OSError(f"Cannot allocate a temporary output artifact for: {path}")

        temporary_opened = os.fstat(descriptor)
        temporary_named_metadata = temporary.lstat()
        if (
            _is_link_or_reparse(temporary_opened)
            or _is_link_or_reparse(temporary_named_metadata)
            or not stat.S_ISREG(temporary_opened.st_mode)
            or not stat.S_ISREG(temporary_named_metadata.st_mode)
            or temporary_opened.st_nlink != 1
            or temporary_named_metadata.st_nlink != 1
            or (
                _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE
                and not os.path.samestat(temporary_opened, temporary_named_metadata)
            )
        ):
            raise ValueError(f"Generated output temporary artifact is unsafe: {temporary}")
        if existing_mode is not None and hasattr(os, "fchmod"):
            os.fchmod(descriptor, existing_mode)

        handle = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = -1
        with handle:
            if handle.write(payload) != len(payload):
                raise OSError(f"Short write while publishing generated output: {path}")
            handle.flush()
            os.fsync(handle.fileno())
            written = os.fstat(handle.fileno())

        observed = temporary.lstat()
        temporary_identity_matches = os.path.samestat(temporary_named_metadata, observed) and (
            not _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE or os.path.samestat(written, observed)
        )
        if (
            _is_link_or_reparse(observed)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or written.st_size != len(payload)
            or observed.st_size != len(payload)
            or not temporary_identity_matches
        ):
            raise ValueError(f"Generated output temporary artifact changed while writing: {temporary}")
        validate_output_directory_path(path.parent)
        destination = _inspect_atomic_destination(path)
        if (existing is None) != (destination is None) or (
            existing is not None
            and destination is not None
            and _artifact_fingerprint(existing) != _artifact_fingerprint(destination)
        ):
            raise ValueError(f"Generated output destination changed while writing: {path}")
        os.replace(temporary, path)  # noqa: PTH105 -- same-directory atomic replacement is the contract
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None and temporary_named_metadata is not None:
            _unlink_if_same_file(temporary, temporary_named_metadata)


def _protect_key_for_storage(key: bytes) -> bytes:
    """Bind key bytes to the current Windows user with DPAPI."""
    if os.name != "nt":
        return key
    import ctypes
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    source = ctypes.create_string_buffer(key)
    source_blob = _DataBlob(len(key), ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)))
    protected_blob = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    if not crypt32.CryptProtectData(
        ctypes.byref(source_blob),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(protected_blob),
    ):
        raise ValueError("Cannot protect output provenance key for the current Windows user") from ctypes.WinError(
            ctypes.get_last_error()
        )
    try:
        return ctypes.string_at(protected_blob.pbData, protected_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(protected_blob.pbData, wintypes.HLOCAL))


def _unprotect_stored_key(payload: bytes) -> bytes:
    """Decode DPAPI-protected Windows bytes or return POSIX bytes unchanged."""
    if os.name != "nt":
        return payload
    import ctypes
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    source = ctypes.create_string_buffer(payload)
    source_blob = _DataBlob(len(payload), ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)))
    key_blob = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source_blob),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(key_blob),
    ):
        raise ValueError("Output provenance key is not protected for the current Windows user") from ctypes.WinError(
            ctypes.get_last_error()
        )
    try:
        key = ctypes.string_at(key_blob.pbData, key_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(key_blob.pbData, wintypes.HLOCAL))
    if len(key) != _KEY_BYTES:
        raise ValueError("Output provenance key has an invalid decrypted size")
    return key


def _stored_key_size_is_valid(size: int) -> bool:
    return 0 < size <= _MAX_STORED_KEY_BYTES if os.name == "nt" else size == _KEY_BYTES


def _default_key_path() -> Path:
    override = os.environ.get(OUTPUT_PROVENANCE_KEY_ENV)
    if override:
        return Path(override).expanduser().absolute()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "skillevaluator" / "output-provenance.key"


def output_provenance_key_path() -> Path:
    """Return the configured private-key path without creating the key."""
    return _default_key_path()


def _validate_key_metadata(path: Path, metadata: os.stat_result) -> None:
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Output provenance key must be a single-link regular file: {path}")
    if not _stored_key_size_is_valid(metadata.st_size):
        raise ValueError(f"Output provenance key has an invalid size: {path}")
    if os.name == "posix":
        if metadata.st_uid != os.getuid():
            raise ValueError(f"Output provenance key is not owned by the current user: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"Output provenance key must not be accessible by group or other users: {path}")


def _read_key(path: Path) -> bytes:
    before = path.lstat()
    _validate_key_metadata(path, before)
    try:
        with SecureRoot(path.parent) as secure_root:
            payload, opened = secure_root.read_bytes(Path(path.name), before.st_size, expected=before)
            after = path.lstat()
    except SecurePathError as exc:
        raise ValueError(f"Output provenance key changed while it was read: {path}") from exc
    _validate_key_metadata(path, opened)
    _validate_key_metadata(path, after)
    stable = _stable_read_metadata(before, opened, after)
    if stable is None or len(payload) != before.st_size or len(payload) != stable.st_size:
        raise ValueError(f"Output provenance key has an invalid size: {path}")
    return _unprotect_stored_key(payload)


def _validate_key_directory(path: Path, metadata: os.stat_result) -> None:
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Output provenance key directory must be a real directory: {path}")
    if os.name == "posix":
        if metadata.st_uid != os.getuid():
            raise ValueError(f"Output provenance key directory is not owned by the current user: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"Output provenance key directory must not be accessible by group or other users: {path}")


def _load_existing_key() -> bytes | None:
    path = _default_key_path()
    try:
        parent_metadata = path.parent.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Cannot inspect output provenance key directory: {path.parent}") from exc
    _validate_key_directory(path.parent, parent_metadata)
    try:
        return _read_key(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Cannot safely read output provenance key: {path}") from exc


def _recover_interrupted_key_publish(path: Path) -> bool:
    """Remove only same-inode private temp links left by an interrupted publish."""
    try:
        target = path.lstat()
    except OSError:
        return False
    if _is_link_or_reparse(target) or not stat.S_ISREG(target.st_mode) or not _stored_key_size_is_valid(target.st_size):
        return False
    if os.name == "posix" and target.st_uid != os.getuid():
        return False
    recovered = False
    for candidate in path.parent.iterdir():
        if not candidate.name.startswith(_KEY_TEMP_PREFIX):
            continue
        try:
            metadata = candidate.lstat()
            if (
                not _is_link_or_reparse(metadata)
                and stat.S_ISREG(metadata.st_mode)
                and metadata.st_dev == target.st_dev
                and metadata.st_ino == target.st_ino
            ):
                candidate.unlink()
                recovered = True
        except OSError:
            continue
    return recovered


def _read_key_after_concurrent_publish(path: Path) -> bytes:
    last_error: OSError | ValueError | None = None
    for _ in range(50):
        try:
            return _read_key(path)
        except (OSError, ValueError) as exc:
            last_error = exc
            _recover_interrupted_key_publish(path)
            time.sleep(0.002)
    if last_error is not None:
        raise last_error
    raise ValueError(f"Cannot safely read concurrently published output provenance key: {path}")


def _load_or_create_key() -> bytes:
    path = _default_key_path()
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        parent_metadata = path.parent.lstat()
    except OSError as exc:
        raise ValueError(f"Cannot inspect output provenance key directory: {path.parent}") from exc
    _validate_key_directory(path.parent, parent_metadata)
    try:
        return _read_key(path)
    except FileNotFoundError:
        pass
    except ValueError:
        return _read_key_after_concurrent_publish(path)
    except OSError as exc:
        raise ValueError(f"Cannot safely read output provenance key: {path}") from exc

    key = secrets.token_bytes(_KEY_BYTES)
    stored_key = _protect_key_for_storage(key)
    temporary = path.parent / f"{_KEY_TEMP_PREFIX}{os.getpid()}-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(stored_key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _read_key_after_concurrent_publish(path)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return _read_key(path)


def _canonical_output_binding(path: Path) -> bytes:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve generated output path for provenance: {path}") from exc
    normalized = os.path.normcase(os.path.normpath(os.fspath(resolved)))
    return os.fsencode(normalized)


def _path_is_contained(path: Path, root: Path) -> bool:
    for candidate, candidate_root in (
        (path.expanduser().absolute(), root.expanduser().absolute()),
        (path.expanduser().resolve(strict=False), root.expanduser().resolve(strict=False)),
    ):
        normalized = Path(os.path.normcase(os.path.normpath(os.fspath(candidate))))
        normalized_root = Path(os.path.normcase(os.path.normpath(os.fspath(candidate_root))))
        try:
            normalized.relative_to(normalized_root)
        except ValueError:
            continue
        return True
    return False


def validate_provenance_key_outside(workspace_root: Path) -> None:
    """Reject a private-key location that could enter an evaluated or generated tree."""
    try:
        inside_workspace = _path_is_contained(_default_key_path(), workspace_root)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot validate output provenance key location against: {workspace_root}") from exc
    if inside_workspace:
        raise ValueError(f"Output provenance key must be outside evaluated and generated trees: {workspace_root}")


def _marker_payload(destination: Path, key: bytes) -> bytes:
    digest = hmac.digest(key, _MARKER_CONTEXT + _canonical_output_binding(destination), "sha256")
    signature = base64.urlsafe_b64encode(digest).rstrip(b"=")
    return _MARKER_PREFIX + signature + b"\n"


def generated_output_marker_payload(destination: Path) -> bytes:
    """Return marker bytes signed for one canonical public destination."""
    validate_provenance_key_outside(destination)
    return _marker_payload(destination, _load_or_create_key())


def _read_marker(path: Path, expected_size: int) -> bytes | None:
    try:
        before = path.lstat()
    except OSError:
        return None
    if (
        _is_link_or_reparse(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected_size
    ):
        return None
    try:
        with SecureRoot(path.parent) as secure_root:
            payload, opened = secure_root.read_bytes(Path(path.name), expected_size, expected=before)
            after = path.lstat()
        if (
            _is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or _is_link_or_reparse(after)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_size != expected_size
            or _stable_read_metadata(before, opened, after) is None
            or len(payload) != expected_size
        ):
            return None
        return payload
    except (OSError, SecurePathError, ValueError):
        return None


def is_generated_output_root(path: Path) -> bool:
    """Return whether *path* has a valid marker bound to its canonical location."""
    observed = _read_marker(path / GENERATED_OUTPUT_MARKER, _MARKER_SIZE)
    if observed is None or not observed.startswith(_MARKER_PREFIX):
        return False
    validate_provenance_key_outside(path)
    key = _load_existing_key()
    if key is None:
        return False
    return hmac.compare_digest(observed, _marker_payload(path, key))


def validate_generated_output_replacement(path: Path) -> None:
    """Allow replacement only for absent, empty, or authentically owned roots."""
    validate_output_directory_path(path)
    if not os.path.lexists(path):
        return
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"Cannot inspect generated output root before replacement: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Generated output root must be a real directory: {path}")
    try:
        has_contents = next(path.iterdir(), None) is not None
    except OSError as exc:
        raise ValueError(f"Cannot inspect generated output root before replacement: {path}") from exc
    if has_contents and not is_generated_output_root(path):
        raise ValueError(f"Generated output marker is missing, invalid, or bound to another path: {path}")


def write_generated_output_marker(root: Path, *, destination: Path | None = None) -> None:
    """Write provenance into a caller-owned tree, optionally for a later public path."""
    validate_provenance_key_outside(root)
    validate_output_directory_path(root)
    root.mkdir(parents=True, exist_ok=True)
    validate_output_directory_path(root)
    expected = generated_output_marker_payload(destination or root)
    marker = root / GENERATED_OUTPUT_MARKER
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        observed = _read_marker(marker, len(expected))
        if observed is None or not hmac.compare_digest(observed, expected):
            raise ValueError(f"Generated output marker is invalid or unsafe: {marker}") from None
        return
    created_named: os.stat_result | None = None
    try:
        created_opened = os.fstat(descriptor)
        created_named = marker.lstat()
        if (
            _is_link_or_reparse(created_opened)
            or _is_link_or_reparse(created_named)
            or not stat.S_ISREG(created_opened.st_mode)
            or not stat.S_ISREG(created_named.st_mode)
            or created_opened.st_nlink != 1
            or created_named.st_nlink != 1
            or (_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE and not os.path.samestat(created_opened, created_named))
        ):
            raise ValueError(f"Generated output marker is unsafe: {marker}")
        handle = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = -1
        with handle:
            if handle.write(expected) != len(expected):
                raise OSError(f"Short write while publishing generated output marker: {marker}")
            handle.flush()
            os.fsync(handle.fileno())
            written = os.fstat(handle.fileno())
        observed = marker.lstat()
        if (
            _is_link_or_reparse(observed)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or written.st_size != len(expected)
            or observed.st_size != len(expected)
            or not os.path.samestat(created_named, observed)
            or (_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE and not os.path.samestat(written, observed))
        ):
            raise ValueError(f"Generated output marker changed while it was written: {marker}")
        _fsync_directory(root)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created_named is not None:
            _unlink_if_same_file(marker, created_named)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def mark_generated_output_root(path: Path) -> None:
    """Claim or validate an output root before any generated child is replaced."""
    validate_generated_output_replacement(path)
    path.mkdir(parents=True, exist_ok=True)
    validate_output_directory_path(path)
    write_generated_output_marker(path)


def _remove_output_tree_at_identity(path: Path, expected_identity: tuple[int, int]) -> bool:
    """Remove only one no-follow tree matching a previously observed identity."""
    if not _DESCRIPTOR_BACKEND:
        return False
    parent_descriptor = -1
    try:
        validate_output_directory_path(path.parent)
        parent_descriptor = os.open(path.parent, _DIRECTORY_FLAGS)
        _remove_tree_at(parent_descriptor, path.name, expected_identity=expected_identity)
        os.fsync(parent_descriptor)
    except (OSError, ValueError):
        return False
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    return not os.path.lexists(path)


def _remove_empty_output_tree_fallback(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    remove_authenticated_marker: bool,
) -> bool:
    """Conservatively remove an empty exact reservation without recursive deletion."""
    marker = path / GENERATED_OUTPUT_MARKER
    marker_metadata: os.stat_result | None = None
    try:
        validate_output_directory_path(path)
        before = path.lstat()
        if (
            _is_link_or_reparse(before)
            or not stat.S_ISDIR(before.st_mode)
            or (before.st_dev, before.st_ino) != expected_identity
        ):
            return False
        entries = list(path.iterdir())
        if remove_authenticated_marker:
            if len(entries) != 1 or entries[0].name != GENERATED_OUTPUT_MARKER or not is_generated_output_root(path):
                return False
            marker_metadata = marker.lstat()
            if (
                _is_link_or_reparse(marker_metadata)
                or not stat.S_ISREG(marker_metadata.st_mode)
                or marker_metadata.st_nlink != 1
            ):
                return False
        elif entries:
            return False

        current = path.lstat()
        if (current.st_dev, current.st_ino) != expected_identity or _is_link_or_reparse(current):
            return False
        if marker_metadata is not None and not _unlink_if_same_file(marker, marker_metadata):
            return False
        current = path.lstat()
        if (current.st_dev, current.st_ino) != expected_identity or _is_link_or_reparse(current):
            return False
        if next(path.iterdir(), None) is not None:
            return False
        path.rmdir()
        _fsync_directory(path.parent)
    except (OSError, ValueError):
        return False
    return not os.path.lexists(path)


def remove_output_reservation_if_identity_matches(path: Path, expected_identity: tuple[int, int]) -> bool:
    """Clean an exact new reservation; fallback platforms require it to be empty."""
    if not _DESCRIPTOR_BACKEND:
        return _remove_empty_output_tree_fallback(
            path,
            expected_identity,
            remove_authenticated_marker=False,
        )
    return _remove_output_tree_at_identity(path, expected_identity)


def remove_generated_output_root_if_owned(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> bool:
    """Remove an authenticated failed reservation without following replacements.

    Descriptor-capable platforms use identity-bound recursive removal. Other
    platforms remove only a marker-only directory and never recursively delete.
    """
    try:
        validate_output_directory_path(path)
        before = path.lstat()
        if not stat.S_ISDIR(before.st_mode) or _is_link_or_reparse(before):
            return False
        identity = before.st_dev, before.st_ino
        if expected_identity is not None and identity != expected_identity:
            return False
        if not is_generated_output_root(path):
            return False
        after = path.lstat()
        if _node_fingerprint(after) != _node_fingerprint(before):
            return False
    except (OSError, ValueError):
        return False
    if not _DESCRIPTOR_BACKEND:
        return _remove_empty_output_tree_fallback(
            path,
            identity,
            remove_authenticated_marker=True,
        )
    return _remove_output_tree_at_identity(path, identity)
