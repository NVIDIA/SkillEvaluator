# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base reporter interface for SkillEvaluator.

This module defines the abstract base class that all reporters must implement.
Reporters are responsible for rendering ValidationResult objects in their
specific format (CLI, JSON, HTML, Markdown, etc.).

The separation between validators (data producers) and reporters (data consumers)
follows the Single Responsibility Principle and allows easy addition of new
output formats without modifying validators.
"""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import click

from skillevaluator.utils.path_security import canonicalize_trusted_root_alias

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_TEMPORARY_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_BINARY", 0)
)
_USE_POSIX_DESCRIPTOR_WRITES = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and all(function in os.supports_dir_fd for function in (os.open, os.mkdir, os.stat, os.rename, os.unlink))
    and os.stat in os.supports_follow_symlinks
)


class UnsafeReportPathError(click.ClickException, ValueError):
    """Raised when a report output path cannot be written without following links."""


def _unsafe_report_path(path: Path, reason: str) -> UnsafeReportPathError:
    return UnsafeReportPathError(f"Unsafe report output path '{path.name or 'report'}': {reason}")


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute path without resolving links or parent traversal."""
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise _unsafe_report_path(path, "parent traversal is not allowed")
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    if not expanded.name:
        raise _unsafe_report_path(path, "a report filename is required")
    return canonicalize_trusted_root_alias(expanded)


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _REPARSE_POINT)


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except OSError as exc:
        raise _unsafe_report_path(path, "cannot inspect the path for a Windows junction") from exc


def _validate_directory_metadata(metadata: os.stat_result, path: Path) -> None:
    if _is_link_or_reparse(metadata) or _is_junction(path):
        raise _unsafe_report_path(path, "a parent is a symlink, reparse point, or junction")
    if not stat.S_ISDIR(metadata.st_mode):
        raise _unsafe_report_path(path, "a parent component is not a directory")


def _validate_file_metadata(metadata: os.stat_result, path: Path) -> None:
    if _is_link_or_reparse(metadata) or _is_junction(path):
        raise _unsafe_report_path(path, "the destination is a symlink, reparse point, or junction")
    if not stat.S_ISREG(metadata.st_mode):
        raise _unsafe_report_path(path, "the destination is not a regular file")


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise OSError("Short write while saving report")
        written += count


def _open_posix_parent(output_path: Path, *, create: bool) -> int:
    """Open the report parent component-by-component without following links."""
    parent = output_path.parent
    try:
        descriptor = os.open(parent.anchor, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise _unsafe_report_path(output_path, "cannot securely open its filesystem anchor") from exc

    try:
        for component in parent.parts[1:]:
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise _unsafe_report_path(output_path, "its parent directory changed while writing") from None
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise _unsafe_report_path(output_path, "cannot securely create its parent directory") from exc
                try:
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError as exc:
                    raise _unsafe_report_path(
                        output_path,
                        "a parent is a symlink, reparse point, or non-directory",
                    ) from exc
            except OSError as exc:
                raise _unsafe_report_path(
                    output_path,
                    "a parent is a symlink, reparse point, or non-directory",
                ) from exc

            metadata = os.fstat(child)
            if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise _unsafe_report_path(
                    output_path,
                    "a parent is a symlink, reparse point, or non-directory",
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_posix_destination(parent_descriptor: int, output_path: Path) -> os.stat_result | None:
    try:
        metadata = os.stat(output_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _unsafe_report_path(output_path, "cannot inspect the destination") from exc
    _validate_file_metadata(metadata, output_path)
    return metadata


def _create_posix_temporary(parent_descriptor: int, output_path: Path) -> tuple[int, str]:
    for _ in range(16):
        name = f".{output_path.name}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(name, _TEMPORARY_FLAGS, 0o600, dir_fd=parent_descriptor), name
        except FileExistsError:
            continue
        except OSError as exc:
            raise _unsafe_report_path(output_path, "cannot create a secure temporary report") from exc
    raise _unsafe_report_path(output_path, "cannot allocate a unique temporary report")


def _write_report_posix(output_path: Path, payload: bytes) -> None:
    parent_descriptor = _open_posix_parent(output_path, create=True)
    temporary_descriptor = -1
    temporary_name: str | None = None
    try:
        destination_metadata = _validate_posix_destination(parent_descriptor, output_path)
        temporary_descriptor, temporary_name = _create_posix_temporary(parent_descriptor, output_path)
        opened_metadata = os.fstat(temporary_descriptor)
        if _is_link_or_reparse(opened_metadata) or not stat.S_ISREG(opened_metadata.st_mode):
            raise _unsafe_report_path(output_path, "the temporary report is not a regular file")
        if destination_metadata is not None:
            os.fchmod(temporary_descriptor, stat.S_IMODE(destination_metadata.st_mode))

        _write_all(temporary_descriptor, payload)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1

        named_temporary = os.stat(temporary_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _is_link_or_reparse(named_temporary) or not os.path.samestat(opened_metadata, named_temporary):
            raise _unsafe_report_path(output_path, "the temporary report changed while writing")

        verification_descriptor = _open_posix_parent(output_path, create=False)
        try:
            if not os.path.samestat(os.fstat(parent_descriptor), os.fstat(verification_descriptor)):
                raise _unsafe_report_path(output_path, "its parent directory changed while writing")
        finally:
            os.close(verification_descriptor)

        _validate_posix_destination(parent_descriptor, output_path)
        os.rename(
            temporary_name,
            output_path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None

        published = os.stat(output_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _is_link_or_reparse(published) or not os.path.samestat(opened_metadata, published):
            raise _unsafe_report_path(output_path, "the published report changed unexpectedly")
        os.fsync(parent_descriptor)
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _prepare_checked_parent(output_path: Path) -> list[tuple[Path, os.stat_result]]:
    """Create and snapshot parent components for platforms without dir-fd writes."""
    snapshots: list[tuple[Path, os.stat_result]] = []
    parent = output_path.parent
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise _unsafe_report_path(output_path, "cannot securely create its parent directory") from exc
            try:
                metadata = current.lstat()
            except OSError as exc:
                raise _unsafe_report_path(output_path, "cannot inspect its newly created parent") from exc
        except OSError as exc:
            raise _unsafe_report_path(output_path, "cannot inspect a parent component") from exc
        _validate_directory_metadata(metadata, current)
        snapshots.append((current, metadata))
    return snapshots


def _validate_checked_destination(output_path: Path) -> os.stat_result | None:
    try:
        metadata = output_path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _unsafe_report_path(output_path, "cannot inspect the destination") from exc
    _validate_file_metadata(metadata, output_path)
    return metadata


def _revalidate_components(
    output_path: Path,
    snapshots: list[tuple[Path, os.stat_result]],
) -> None:
    for component, previous in snapshots:
        try:
            current = component.lstat()
        except OSError as exc:
            raise _unsafe_report_path(output_path, "a parent directory changed while writing") from exc
        _validate_directory_metadata(current, component)
        if not os.path.samestat(previous, current):
            raise _unsafe_report_path(output_path, "a parent directory changed while writing")


def _write_report_checked(output_path: Path, payload: bytes) -> None:
    """Atomic checked fallback for Windows and platforms without dir-fd support."""
    snapshots = _prepare_checked_parent(output_path)
    destination_metadata = _validate_checked_destination(output_path)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        opened_metadata = os.fstat(descriptor)
        named_temporary = temporary_path.lstat()
        if (
            _is_link_or_reparse(opened_metadata)
            or not stat.S_ISREG(opened_metadata.st_mode)
            or _is_link_or_reparse(named_temporary)
            or not os.path.samestat(opened_metadata, named_temporary)
        ):
            raise _unsafe_report_path(output_path, "the temporary report is not a stable regular file")
        if destination_metadata is not None and hasattr(os, "fchmod"):
            os.fchmod(descriptor, stat.S_IMODE(destination_metadata.st_mode))

        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        _revalidate_components(output_path, snapshots)
        _validate_checked_destination(output_path)
        named_temporary = temporary_path.lstat()
        if _is_link_or_reparse(named_temporary) or not os.path.samestat(opened_metadata, named_temporary):
            raise _unsafe_report_path(output_path, "the temporary report changed while writing")

        temporary_path.replace(output_path)
        temporary_path = None
        published = output_path.lstat()
        if _is_link_or_reparse(published) or not os.path.samestat(opened_metadata, published):
            raise _unsafe_report_path(output_path, "the published report changed unexpectedly")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_report_atomically(output_path: Path, payload: bytes) -> None:
    absolute = _absolute_lexical(output_path)
    if _USE_POSIX_DESCRIPTOR_WRITES:
        _write_report_posix(absolute, payload)
    else:
        _write_report_checked(absolute, payload)


def is_advisory_agent_eval_skip(result: ValidationResult) -> bool:
    """Return whether a Tier 3 result records a non-blocking skipped run."""
    if result.validator_name != "AGENT_EVAL":
        return False
    gating = result.metadata.get("gating") if isinstance(result.metadata, dict) else None
    if isinstance(gating, dict) and gating.get("blocking", False):
        return False
    payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    return bool(
        isinstance(provenance, dict) and provenance.get("advisory") and provenance.get("reason") == "skipped"
    )


def passes_required_gate(result: ValidationResult) -> bool:
    """Return whether *result* permits the required validation gate to pass."""
    gating = result.metadata.get("gating") if isinstance(result.metadata, dict) else None
    if isinstance(gating, dict):
        return bool(result.passed or not gating.get("blocking", True))
    return bool(result.passed or is_advisory_agent_eval_skip(result))


class ReporterBase(ABC):
    """Abstract base class for validation result reporters.

    All reporters must implement:
    - name: Unique identifier for the reporter
    - render(): Render a single ValidationResult to string
    - render_all(): Render multiple ValidationResults to string

    Optional override:
    - save(): Save rendered output to file (default implementation provided)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return unique identifier for this reporter (e.g., 'cli', 'json')."""
        ...

    @property
    def description(self) -> str:
        """Return human-readable description of the reporter."""
        return f"{self.name} reporter"

    @abstractmethod
    def render(self, result: ValidationResult) -> str:
        """Render a single validation result to string.

        Args:
            result: ValidationResult to render

        Returns:
            String representation in the reporter's format
        """
        ...

    @abstractmethod
    def render_all(self, results: list[ValidationResult]) -> str:
        """Render multiple validation results to string.

        Args:
            results: List of ValidationResults to render

        Returns:
            String representation of all results in the reporter's format
        """
        ...

    def save(self, results: list[ValidationResult], output_path: Path) -> None:
        """Save rendered output to file.

        Args:
            results: List of ValidationResults to render
            output_path: Path to save the output file

        The default implementation renders all results and atomically writes a
        regular file without following symlink or reparse-point output paths.
        Subclasses may override for format-specific behavior (e.g., binary output).

        Raises:
            UnsafeReportPathError: If the destination cannot be written safely.
        """
        payload = self.render_all(results).encode("utf-8")
        _write_report_atomically(output_path, payload)

    def get_file_extension(self) -> str:
        """Return the default file extension for this reporter's output.

        Returns:
            File extension including the dot (e.g., '.json', '.html')
        """
        extensions = {
            "cli": ".txt",
            "json": ".json",
            "html": ".html",
            "markdown": ".md",
        }
        return extensions.get(self.name, ".txt")
