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

import hashlib
import json
import math
import os
import re
import secrets
import stat
import tempfile
import unicodedata
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from skillevaluator.constants import (
    DIMENSION_MAPPING,
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
)
from skillevaluator.publication_evidence import result_publication_evidence
from skillevaluator.publication_identity import PUBLICATION_TARGET_DIGEST_ALGORITHM
from skillevaluator.publication_text import publication_identity_present, publication_semantic_text
from skillevaluator.utils.path_security import canonicalize_trusted_root_alias

_AGENT_EVAL_MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=5)

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

# Reporters share these limits so a payload that would be truncated in a
# self-contained artifact cannot receive a different publication verdict in a
# text-only artifact.
AGENT_EVAL_REPORT_MAX_DEPTH = 64
AGENT_EVAL_REPORT_MAX_NODES = 100_000
AGENT_EVAL_REPORT_MAX_TEXT_BYTES = 2 * 1024 * 1024
_AGENT_EVAL_FINGERPRINT_MAX_DEPTH = 8
_AGENT_EVAL_FINGERPRINT_MAX_NODES = 256
_AGENT_EVAL_FINGERPRINT_MAX_MAPPING_ITEMS = 64
_AGENT_EVAL_FINGERPRINT_MAX_COLLECTION_ITEMS = 64
_AGENT_EVAL_FINGERPRINT_MAX_TEXT_CHARS = 4096
_AGENT_EVAL_FINGERPRINT_PRIORITY_KEYS = (
    "schema_version",
    "verdict",
    "execution_status",
    "summary",
    "skill_name",
    "publication_target",
    "run_id",
    "evaluated_at",
    "evaluator_version",
    "dataset_digest",
    "dataset_digest_algorithm",
    "benchmark_policy",
    "attempt_policy",
    "dataset_summary",
    "agents",
)
_AGENT_EVAL_DATASET_DIGEST = re.compile(r"sha256:[0-9a-f]{64}", flags=re.IGNORECASE)
_AGENT_EVAL_DATASET_DIGEST_ALGORITHM = "skill-evaluator-dataset-snapshot/1"
_PUBLICATION_TARGET_DIGEST = re.compile(r"sha256:[0-9a-f]{64}", flags=re.IGNORECASE)
_AGENT_EVAL_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")
_PUBLICATION_TARGET_SKILL_NAME_MAX_BYTES = 1024
_PUBLICATION_TARGET_CONFLICT_MAX_BYTES = 256
_PUBLICATION_TARGET_CONFLICT_FALLBACK = "publication target identity conflict"


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
    raw_payload = result.metadata.get("agent_eval") if isinstance(result.metadata, dict) else None
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    provenance = payload.get("provenance", {})
    verdict, execution_status, truth_consistent = _agent_eval_truth_state(payload)
    return bool(
        isinstance(provenance, dict)
        and provenance.get("advisory")
        and provenance.get("reason") == "skipped"
        and truth_consistent
        and execution_status == "skipped"
        and verdict in {"neutral", "skipped"}
    )


def is_cleanly_skipped(result: ValidationResult) -> bool:
    """Return whether a result records a non-failing skipped execution."""
    metadata_skip = isinstance(result.metadata, dict) and bool(result.metadata.get("skipped"))
    if result.passed and metadata_skip:
        raw_payload = result.metadata.get("agent_eval")
        if isinstance(raw_payload, dict):
            verdict, execution_status, truth_consistent = _agent_eval_truth_state(raw_payload)
            return bool(truth_consistent and execution_status == "skipped" and verdict in {"neutral", "skipped"})
        return True
    return is_advisory_agent_eval_skip(result)


def get_skip_reason(result: ValidationResult) -> str:
    """Return a stable human-readable reason for a skipped result."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    reason = metadata.get("skip_reason")
    if reason:
        return str(reason)

    payload = metadata.get("agent_eval")
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    advisory_message = provenance.get("message") if isinstance(provenance, dict) else None
    if advisory_message:
        return str(advisory_message)

    if result.warnings:
        return str(result.warnings[0])
    return "Prerequisite unavailable"


def is_tier2_validator_name(validator_name: str | None) -> bool:
    """Return whether a normalized validator name belongs to Tier 2."""
    normalized = " ".join((validator_name or "").casefold().replace("_", " ").replace("-", " ").split())
    return any(marker in normalized for marker in ("similarity", "dedup", "context optimization"))


def agent_eval_report_serialization_limits() -> tuple[int, int]:
    """Return the shared depth and node limits for embedded Tier 3 payloads."""
    return AGENT_EVAL_REPORT_MAX_DEPTH, AGENT_EVAL_REPORT_MAX_NODES


def agent_eval_report_text_limit() -> int:
    """Return the aggregate UTF-8 text budget for embedded Tier 3 payloads."""
    return AGENT_EVAL_REPORT_MAX_TEXT_BYTES


def agent_eval_report_serialization_issue(value: object) -> str | None:
    """Return why Tier 3 cannot be emitted losslessly within report limits."""
    if not isinstance(value, dict):
        return "The Tier 3 payload is not a mapping."
    if isinstance(value, dict) and value.get("_serialization_truncated") is True:
        return "The emitted Tier 3 payload was truncated."

    max_depth, max_nodes = agent_eval_report_serialization_limits()
    max_text_bytes = agent_eval_report_text_limit()
    active_containers: set[int] = set()
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    nodes = 0
    text_bytes = 0
    while stack:
        current, depth, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(current))
            continue

        nodes += 1
        if nodes > max_nodes:
            return f"The Tier 3 payload exceeds the {max_nodes:,}-node report limit."
        if depth > max_depth:
            return f"The Tier 3 payload exceeds the {max_depth}-level report depth limit."
        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, str):
            if len(current) > max_text_bytes - text_bytes:
                return f"The Tier 3 payload exceeds the {max_text_bytes:,}-byte report text limit."
            utf8_safe = current.encode("utf-8", errors="replace").decode("utf-8")
            normalized = publication_semantic_text(current)
            if utf8_safe != current or normalized != unicodedata.normalize("NFKC", utf8_safe):
                return "The Tier 3 payload contains text that cannot be emitted losslessly."
            text_bytes += len(normalized.encode("utf-8"))
            if text_bytes > max_text_bytes:
                return f"The Tier 3 payload exceeds the {max_text_bytes:,}-byte report text limit."
            continue
        if isinstance(current, int):
            if current.bit_length() > 256:
                return "The Tier 3 payload contains an oversized integer."
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                return "The Tier 3 payload contains a non-finite number."
            continue
        if isinstance(current, tuple):
            return "The Tier 3 payload contains a tuple that would be normalized to a list."
        if not isinstance(current, (dict, list)):
            return "The Tier 3 payload contains a value that is not JSON-compatible."
        if len(current) > max_nodes - nodes:
            return f"The Tier 3 payload exceeds the {max_nodes:,}-node report limit."

        container_id = id(current)
        if container_id in active_containers:
            return "The Tier 3 payload contains a recursive container."
        active_containers.add(container_id)
        stack.append((current, depth, True))
        if isinstance(current, dict):
            safe_keys: set[str] = set()
            for key in current:
                if not isinstance(key, str):
                    return "The Tier 3 payload contains a non-string mapping key."
                if len(key) > max_text_bytes - text_bytes:
                    return f"The Tier 3 payload exceeds the {max_text_bytes:,}-byte report text limit."
                utf8_safe_key = key.encode("utf-8", errors="replace").decode("utf-8")
                safe_key = publication_semantic_text(key)
                if utf8_safe_key != key or safe_key != unicodedata.normalize("NFKC", utf8_safe_key):
                    return "The Tier 3 payload contains a mapping key that cannot be emitted losslessly."
                if not safe_key or safe_key in safe_keys:
                    return "The Tier 3 payload contains colliding normalized mapping keys."
                text_bytes += len(safe_key.encode("utf-8"))
                if text_bytes > max_text_bytes:
                    return f"The Tier 3 payload exceeds the {max_text_bytes:,}-byte report text limit."
                safe_keys.add(safe_key)
            children = current.values()
        else:
            children = current
        for child in reversed(children):
            stack.append((child, depth + 1, False))
    return None


def _agent_eval_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _agent_eval_finite_score(value: object) -> float | None:
    number = _agent_eval_finite_number(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _agent_eval_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if 0 <= value <= 2**63 - 1 else 0


def _agent_eval_consistent_text_field(payload: dict[str, Any] | None, key: str) -> str | None:
    """Return a duplicated text field only when every persisted value agrees."""
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    values: list[str] = []
    for container in (payload, summary):
        if key not in container:
            continue
        value = container[key]
        if (
            not isinstance(value, str)
            or not value
            or len(value) > AGENT_EVAL_REPORT_MAX_TEXT_BYTES
            or value != value.strip()
        ):
            return None
        values.append(value)
    if not values or len(set(values)) != 1:
        return None
    return values[0]


def _agent_eval_consistent_count_field(payload: dict[str, Any] | None, key: str) -> int | None:
    """Return a duplicated non-negative count only when persisted values agree."""
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    values: list[int] = []
    for container in (payload, summary):
        if key not in container:
            continue
        value = container[key]
        if isinstance(value, bool) or not isinstance(value, int) or _agent_eval_count(value) != value:
            return None
        values.append(value)
    if not values or len(set(values)) != 1:
        return None
    return values[0]


def _agent_eval_truth_state(payload: dict[str, Any] | None) -> tuple[str, str, bool]:
    """Return conservative Tier 3 truth plus whether duplicated claims agree."""
    if not isinstance(payload, dict):
        return "", "", False
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}

    def collect(key: str, allowed: frozenset[str], rank: dict[str, int]) -> tuple[str, bool]:
        values: list[str] = []
        valid = True
        for container in (payload, summary):
            if key not in container:
                continue
            raw_value = container[key]
            if not isinstance(raw_value, str) or not 1 <= len(raw_value) <= 16 or raw_value != raw_value.strip():
                valid = False
                continue
            value = raw_value.casefold()
            if value not in allowed:
                valid = False
                continue
            values.append(value)
        if not values:
            return "", False
        conservative = max(values, key=lambda value: rank[value])
        return conservative, valid and len(set(values)) == 1

    verdict, verdict_consistent = collect(
        "verdict",
        frozenset({"pass", "neutral", "fail", "skipped", "incomplete"}),
        {"pass": 0, "skipped": 1, "neutral": 2, "incomplete": 3, "fail": 4},
    )
    execution_status, execution_consistent = collect(
        "execution_status",
        frozenset({"succeeded", "skipped", "incomplete", "failed"}),
        {"succeeded": 0, "skipped": 1, "incomplete": 2, "failed": 3},
    )
    return verdict, execution_status, verdict_consistent and execution_consistent


def _agent_eval_rejected_truth_state(payload: dict[str, Any]) -> tuple[str, str, bool, int]:
    """Read only bounded root truth fields from a rejected Tier 3 payload."""
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}

    def collect(key: str, allowed: frozenset[str], rank: dict[str, int]) -> str:
        values: list[str] = []
        for container in (payload, summary):
            raw_value = container.get(key)
            if not isinstance(raw_value, str) or not 1 <= len(raw_value) <= 16:
                continue
            if raw_value != raw_value.strip():
                continue
            value = raw_value.casefold()
            if value in allowed:
                values.append(value)
        return max(values, key=lambda value: rank[value]) if values else ""

    verdict = collect(
        "verdict",
        frozenset({"pass", "neutral", "fail", "skipped", "incomplete"}),
        {"pass": 0, "skipped": 1, "neutral": 2, "incomplete": 3, "fail": 4},
    )
    execution_status = collect(
        "execution_status",
        frozenset({"succeeded", "skipped", "incomplete", "failed"}),
        {"succeeded": 0, "skipped": 1, "incomplete": 2, "failed": 3},
    )
    explicit_failure = verdict == "fail" or execution_status == "failed"
    conservative_outcome_rank = {"pass": 0, "neutral": 1, "incomplete": 2, "fail": 3}.get(verdict, -1)
    if execution_status == "failed":
        conservative_outcome_rank = max(conservative_outcome_rank, 3)
    elif execution_status == "incomplete":
        conservative_outcome_rank = max(conservative_outcome_rank, 2)
    return verdict, execution_status, explicit_failure, conservative_outcome_rank


def _agent_eval_evaluated_at_datetime(value: object) -> datetime | None:
    """Parse one timezone-aware evaluation instant within bounded clock skew."""
    timestamp = _agent_eval_safe_text(value)
    if not timestamp or timestamp != timestamp.strip():
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        normalized = parsed.astimezone(UTC)
        if normalized > datetime.now(UTC) + _AGENT_EVAL_MAX_FUTURE_CLOCK_SKEW:
            return None
    except (OverflowError, OSError, ValueError):
        return None
    return normalized


def agent_eval_publication_evaluated_at(payload: dict[str, Any] | None) -> str | None:
    """Return a canonical ISO evaluation timestamp, rejecting shaped text."""
    value = _agent_eval_consistent_text_field(payload, "evaluated_at")
    if value is None or _agent_eval_evaluated_at_datetime(value) is None:
        return None
    return value


def agent_eval_publication_dataset_provenance(
    payload: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """Return canonical digest provenance required by public benchmark cards."""
    digest = _agent_eval_consistent_text_field(payload, "dataset_digest") or ""
    algorithm = _agent_eval_consistent_text_field(payload, "dataset_digest_algorithm") or ""
    if not _AGENT_EVAL_DATASET_DIGEST.fullmatch(digest):
        return None
    if algorithm != _AGENT_EVAL_DATASET_DIGEST_ALGORITHM:
        return None
    return digest, algorithm


def _agent_eval_safe_text(value: object) -> str:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value) if value.bit_length() <= 256 else ""
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return ""


@dataclass(frozen=True)
class PublicationTargetIdentity:
    """Canonical source identity shared by every publication-required tier."""

    skill_name: str
    skill_key: str
    skill_digest: str
    skill_digest_algorithm: str


def _publication_target_identity(value: object) -> PublicationTargetIdentity | None:
    if not isinstance(value, dict):
        return None
    raw_skill_name = value.get("skill_name")
    raw_skill_digest = value.get("skill_digest")
    raw_algorithm = value.get("skill_digest_algorithm")
    if (
        not isinstance(raw_skill_name, str)
        or not isinstance(raw_skill_digest, str)
        or not isinstance(raw_algorithm, str)
    ):
        return None
    if (
        len(raw_skill_name) > _PUBLICATION_TARGET_SKILL_NAME_MAX_BYTES
        or len(raw_skill_digest) != 71
        or raw_algorithm != PUBLICATION_TARGET_DIGEST_ALGORITHM
    ):
        return None
    skill_name = _agent_eval_safe_text(raw_skill_name)
    if unicodedata.normalize("NFC", skill_name) != skill_name or not publication_identity_present(skill_name):
        return None
    if len(skill_name.encode("utf-8")) > _PUBLICATION_TARGET_SKILL_NAME_MAX_BYTES:
        return None
    if not _PUBLICATION_TARGET_DIGEST.fullmatch(raw_skill_digest):
        return None
    return PublicationTargetIdentity(
        skill_name=skill_name,
        skill_key=skill_name,
        skill_digest=raw_skill_digest.casefold(),
        skill_digest_algorithm=raw_algorithm,
    )


def _result_publication_target(result: ValidationResult) -> PublicationTargetIdentity | None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if _result_has_publication_target_conflict(result):
        return None
    return _publication_target_identity(metadata.get("publication_target"))


def publication_target_dict(value: object) -> dict[str, str] | None:
    """Project an untrusted target claim to its canonical three fields."""
    identity = _publication_target_identity(value)
    if identity is None:
        return None
    return {
        "skill_name": identity.skill_name,
        "skill_digest": identity.skill_digest,
        "skill_digest_algorithm": identity.skill_digest_algorithm,
    }


def result_publication_target_dict(result: ValidationResult) -> dict[str, str] | None:
    """Return a fresh, canonical three-field result target safe for output."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if _result_has_publication_target_conflict(result):
        return None
    return publication_target_dict(metadata.get("publication_target"))


def publication_target_conflict_marker(value: object) -> str:
    """Flatten an untrusted conflict claim to one bounded, printable line."""
    if not isinstance(value, str):
        return _PUBLICATION_TARGET_CONFLICT_FALLBACK
    if len(value) > _PUBLICATION_TARGET_CONFLICT_MAX_BYTES:
        return _PUBLICATION_TARGET_CONFLICT_FALLBACK
    single_line = "".join(" " if character.isspace() else character for character in value)
    normalized = " ".join(publication_semantic_text(single_line).split())
    if not normalized or len(normalized.encode("utf-8")) > _PUBLICATION_TARGET_CONFLICT_MAX_BYTES:
        return _PUBLICATION_TARGET_CONFLICT_FALLBACK
    return normalized


def result_publication_target_conflict_marker(result: ValidationResult) -> str | None:
    """Return a safe conflict marker when a producer persisted that field."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if "publication_target_conflict" not in metadata:
        return None
    return publication_target_conflict_marker(metadata.get("publication_target_conflict"))


def _result_has_publication_target_conflict(result: ValidationResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    payload = metadata.get("agent_eval")
    return "publication_target_conflict" in metadata or _agent_eval_has_publication_target_conflict(
        payload if isinstance(payload, dict) else None
    )


def _agent_eval_has_publication_target_conflict(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return "publication_target_conflict" in payload or "publication_target_conflict" in summary


def agent_eval_publication_target(payload: dict[str, Any] | None) -> PublicationTargetIdentity | None:
    """Return the duplicated Tier 3 target claim only when both copies agree."""
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    top_level = _publication_target_identity(payload.get("publication_target"))
    summarized = _publication_target_identity(summary.get("publication_target"))
    if top_level is None or summarized is None or top_level != summarized:
        return None
    return top_level


def agent_eval_publication_run_id(payload: dict[str, Any] | None) -> str | None:
    """Return the exact bounded Tier 3 run identity duplicated in its summary."""
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    run_id = payload.get("run_id")
    summarized_run_id = summary.get("run_id")
    if (
        not isinstance(run_id, str)
        or not isinstance(summarized_run_id, str)
        or not 1 <= len(run_id) <= 160
        or run_id != summarized_run_id
        or _AGENT_EVAL_RUN_ID.fullmatch(run_id) is None
    ):
        return None
    return run_id


def _selected_publication_target(
    results: list[ValidationResult],
    payload: dict[str, Any] | None,
    expected_skill_name: str | None,
) -> PublicationTargetIdentity | None:
    """Select one canonical target without accepting anonymous policy claims."""
    expected_key = _publication_identity_key(expected_skill_name)
    payload_target = agent_eval_publication_target(payload)
    if payload_target is not None and (
        expected_key is None or _publication_identity_key(payload_target.skill_name) == expected_key
    ):
        return payload_target
    claims = {
        claim
        for result in results
        if (claim := _result_publication_target(result)) is not None
        and (is_tier3_result(result) or result_publication_evidence(result) is not None)
        and (expected_key is None or _publication_identity_key(claim.skill_name) == expected_key)
    }
    return next(iter(claims)) if len(claims) == 1 else None


def publication_target_for_results(
    results: list[ValidationResult],
    payload: dict[str, Any] | None = None,
    *,
    expected_skill_name: str | None = None,
) -> PublicationTargetIdentity | None:
    """Return the one canonical target shared by publication evidence."""
    selected_payload = payload if isinstance(payload, dict) else select_agent_eval_payload(results)
    return _selected_publication_target(results, selected_payload, expected_skill_name)


def result_matches_publication_target(
    result: ValidationResult,
    target: PublicationTargetIdentity | None,
) -> bool:
    """Return whether one result is bound to the selected exact source."""
    return target is not None and _result_publication_target(result) == target


def _agent_eval_agents(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    raw_agents = (payload or {}).get("agents")
    if not isinstance(raw_agents, dict):
        return {}
    agents: dict[str, dict[str, Any]] = {}
    normalized_names: set[str] = set()
    for raw_name, raw_agent in raw_agents.items():
        if not isinstance(raw_name, str) or not publication_identity_present(raw_name):
            return {}
        name = publication_semantic_text(raw_name).strip()
        normalized_name = name.casefold()
        if not publication_identity_present(name) or not isinstance(raw_agent, dict):
            # A malformed peer must not disappear while another agent certifies
            # the same run. Treat the whole agent set as unusable evidence.
            return {}
        if normalized_name in normalized_names:
            return {}
        normalized_names.add(normalized_name)
        agents[name] = raw_agent
    return agents


def _agent_eval_attempt_coverage_complete(payload: dict[str, Any] | None) -> bool:
    """Return whether succeeded Tier 3 evidence proves positive attempt coverage."""
    expected_attempts = _agent_eval_consistent_count_field(payload, "expected_attempts")
    scored_attempts = _agent_eval_consistent_count_field(payload, "scored_attempts")
    if (
        expected_attempts is None
        or scored_attempts is None
        or expected_attempts <= 0
        or scored_attempts <= 0
        or scored_attempts > expected_attempts
    ):
        return False

    scored_agents = 0
    summed_expected = 0
    summed_scored = 0
    for agent in _agent_eval_agents(payload).values():
        raw_expected = agent.get("expected_attempts")
        raw_scored = agent.get("scored_attempts")
        if (
            isinstance(raw_expected, bool)
            or not isinstance(raw_expected, int)
            or isinstance(raw_scored, bool)
            or not isinstance(raw_scored, int)
        ):
            return False
        agent_expected = _agent_eval_count(agent.get("expected_attempts"))
        agent_scored = _agent_eval_count(agent.get("scored_attempts"))
        if agent_scored > agent_expected:
            return False
        summed_expected += agent_expected
        summed_scored += agent_scored
        if _agent_eval_safe_text(agent.get("execution_status")).casefold() != "succeeded":
            continue
        if agent_expected <= 0 or agent_scored <= 0:
            return False
        scored_agents += 1
    return bool(scored_agents > 0 and summed_expected == expected_attempts and summed_scored == scored_attempts)


def _agent_eval_dimension_scores(agent: dict[str, Any]) -> list[float] | None:
    raw_dimensions = agent.get("dimensions")
    if not isinstance(raw_dimensions, list):
        return None
    dimensions: dict[str, dict[str, Any]] = {}
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            continue
        dimension_id = _agent_eval_safe_text(raw_dimension.get("id"))
        if not dimension_id or dimension_id in dimensions:
            return None
        dimensions[dimension_id] = raw_dimension

    scores: list[float] = []
    for dimension_id in DIMENSION_MAPPING:
        dimension = dimensions.get(dimension_id)
        if dimension is None:
            return None
        value = dimension.get("with_skill") if "with_skill" in dimension else dimension.get("score")
        score = _agent_eval_finite_score(value)
        if score is None:
            return None
        scores.append(score)
    return scores


def agent_eval_dimension_verdict(payload: dict[str, Any] | None) -> str | None:
    """Recompute a Tier 3 verdict from complete supported-agent dimensions."""
    supported_agents = [
        agent
        for agent in _agent_eval_agents(payload).values()
        if _agent_eval_safe_text(agent.get("execution_status")).lower() == "succeeded"
    ]
    if not supported_agents:
        return None

    verdicts: list[str] = []
    has_partial_evidence = False
    for agent in supported_agents:
        scores = _agent_eval_dimension_scores(agent)
        if scores is None:
            has_partial_evidence = True
            continue
        if any(score < DIMENSION_VERDICT_NEUTRAL_THRESHOLD for score in scores):
            verdicts.append("fail")
        elif any(score < DIMENSION_VERDICT_PASS_THRESHOLD for score in scores):
            verdicts.append("neutral")
        else:
            verdicts.append("pass")

    if "pass" in verdicts:
        return "pass"
    if has_partial_evidence or not verdicts:
        return None
    if "neutral" in verdicts:
        return "neutral"
    return "fail"


def _agent_eval_dataset_count(payload: dict[str, Any]) -> int:
    summary = payload.get("dataset_summary")
    if isinstance(summary, dict):
        return _agent_eval_count(summary.get("total_tasks"))
    dataset = payload.get("dataset")
    if isinstance(dataset, list):
        count = sum(isinstance(item, dict) for item in dataset)
        if count:
            return count
    trials = payload.get("trials")
    task_ids: set[str] = set()
    if isinstance(trials, list):
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            for key in ("entry_id", "case_id", "task_id", "id"):
                task_id = _agent_eval_safe_text(trial.get(key)).strip()
                if task_id:
                    task_ids.add(task_id)
                    break
    return len(task_ids)


def agent_eval_publication_evidence_complete(payload: dict[str, Any] | None) -> bool:
    """Return whether a payload carries the minimum publication Tier 3 evidence."""
    if not isinstance(payload, dict):
        return False
    if agent_eval_report_serialization_issue(payload) is not None:
        return False
    agents = _agent_eval_agents(payload)
    if not agents or any(
        not any(publication_identity_present(agent.get(key)) for key in ("model", "model_name", "llm_model"))
        for agent in agents.values()
    ):
        return False
    verdict, execution_status, truth_consistent = _agent_eval_truth_state(payload)
    evaluator_version = _agent_eval_consistent_text_field(payload, "evaluator_version")
    environment = _agent_eval_consistent_text_field(payload, "environment")
    skill_name = _agent_eval_consistent_text_field(payload, "skill_name")
    publication_target = agent_eval_publication_target(payload)
    attempt_policy = payload.get("attempt_policy") if isinstance(payload.get("attempt_policy"), dict) else {}
    return bool(
        truth_consistent
        and verdict in {"pass", "neutral", "fail"}
        and execution_status == "succeeded"
        and agent_eval_dimension_verdict(payload) is not None
        and agent_eval_publication_evaluated_at(payload) is not None
        and publication_identity_present(evaluator_version)
        and agent_eval_publication_dataset_provenance(payload) is not None
        and _agent_eval_dataset_count(payload) > 0
        and _agent_eval_count(attempt_policy.get("max_attempts")) > 0
        and _agent_eval_attempt_coverage_complete(payload)
        and publication_identity_present(environment)
        and publication_identity_present(skill_name)
        and publication_target is not None
        and publication_target.skill_name == skill_name
        and agent_eval_publication_run_id(payload) is not None
    )


def _agent_eval_fingerprint_text(value: str) -> str | dict[str, object]:
    """Return a bounded canonical text projection without scanning a huge tail."""
    if len(value) > _AGENT_EVAL_FINGERPRINT_MAX_TEXT_CHARS:
        prefix = value[:_AGENT_EVAL_FINGERPRINT_MAX_TEXT_CHARS]
        return {
            "type": "text",
            "length": len(value),
            "prefix": publication_semantic_text(prefix),
            "truncated": True,
        }
    return publication_semantic_text(value)


def _agent_eval_fingerprint_key_rank(value: object) -> tuple[object, ...]:
    """Return a bounded total ordering for the sampled keys of a malformed map."""
    if isinstance(value, str):
        prefix = value[:_AGENT_EVAL_FINGERPRINT_MAX_TEXT_CHARS]
        return (0, len(value), publication_semantic_text(prefix))
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int):
        return (2, value.bit_length(), value if value.bit_length() <= 256 else 0)
    if isinstance(value, float):
        if math.isnan(value):
            return (3, 1, "nan")
        if math.isinf(value):
            return (3, 1, "positive-infinity" if value > 0 else "negative-infinity")
        return (3, 0, value)
    value_type = type(value)
    return (4, value_type.__module__, value_type.__qualname__)


def _agent_eval_fingerprint_key(value: object) -> object:
    """Return a bounded JSON-safe description of one sampled mapping key."""
    if isinstance(value, str):
        return ["text", _agent_eval_fingerprint_text(value)]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value] if value.bit_length() <= 256 else ["oversized-int", value.bit_length()]
    if isinstance(value, float):
        if math.isfinite(value):
            return ["float", value]
        if math.isnan(value):
            return ["non-finite-float", "nan"]
        return ["non-finite-float", "positive-infinity" if value > 0 else "negative-infinity"]
    value_type = type(value)
    return ["unsupported", value_type.__module__, value_type.__qualname__]


def _agent_eval_bounded_fingerprint_value(
    value: object,
    *,
    depth: int = 0,
    remaining_nodes: list[int] | None = None,
    active_containers: set[int] | None = None,
) -> object:
    """Project malformed evidence into a small deterministic fingerprint shape."""
    remaining = remaining_nodes if remaining_nodes is not None else [_AGENT_EVAL_FINGERPRINT_MAX_NODES]
    active = active_containers if active_containers is not None else set()
    if remaining[0] <= 0:
        return {"truncated": "node-limit"}
    remaining[0] -= 1
    if depth > _AGENT_EVAL_FINGERPRINT_MAX_DEPTH:
        return {"truncated": "depth-limit"}
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _agent_eval_fingerprint_text(value)
    if isinstance(value, int):
        return value if value.bit_length() <= 256 else {"oversized_integer_bits": value.bit_length()}
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"non_finite_number": "nan"}
        return {"non_finite_number": "positive-infinity" if value > 0 else "negative-infinity"}
    if not isinstance(value, (dict, list, tuple)):
        value_type = type(value)
        return {"unsupported_type": [value_type.__module__, value_type.__qualname__]}

    container_id = id(value)
    if container_id in active:
        return {"recursive_container": True}
    active.add(container_id)
    try:
        if isinstance(value, dict):
            priority = frozenset(_AGENT_EVAL_FINGERPRINT_PRIORITY_KEYS)
            priority_keys = [key for key in _AGENT_EVAL_FINGERPRINT_PRIORITY_KEYS if key in value]
            sampled_keys: list[object] = []
            max_nonpriority = max(0, _AGENT_EVAL_FINGERPRINT_MAX_MAPPING_ITEMS - len(priority_keys))
            iterator = iter(value)
            attempts_remaining = max_nonpriority + len(priority_keys)
            while len(sampled_keys) < max_nonpriority and attempts_remaining > 0:
                attempts_remaining -= 1
                try:
                    key = next(iterator)
                except StopIteration:
                    break
                if key not in priority:
                    sampled_keys.append(key)
            sampled_keys.sort(key=_agent_eval_fingerprint_key_rank)
            selected_keys = [*priority_keys, *sampled_keys]
            entries: list[list[object]] = []
            for key in selected_keys:
                if remaining[0] <= 0:
                    break
                try:
                    item = value[key]
                except (KeyError, RuntimeError):
                    entries.append([_agent_eval_fingerprint_key(key), {"unavailable": True}])
                    continue
                entries.append(
                    [
                        _agent_eval_fingerprint_key(key),
                        _agent_eval_bounded_fingerprint_value(
                            item,
                            depth=depth + 1,
                            remaining_nodes=remaining,
                            active_containers=active,
                        ),
                    ]
                )
            return {
                "type": "mapping",
                "length": len(value),
                "entries": entries,
                "truncated": len(value) > len(selected_keys),
            }

        item_limit = min(len(value), _AGENT_EVAL_FINGERPRINT_MAX_COLLECTION_ITEMS)
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "length": len(value),
            "items": [
                _agent_eval_bounded_fingerprint_value(
                    value[index],
                    depth=depth + 1,
                    remaining_nodes=remaining,
                    active_containers=active,
                )
                for index in range(item_limit)
                if remaining[0] > 0
            ],
            "truncated": len(value) > item_limit,
        }
    finally:
        active.remove(container_id)


def _agent_eval_payload_fingerprint(
    payload: dict[str, Any],
    *,
    serialization_issue: str | None = None,
) -> str:
    """Return a stable tie-breaker without fully serializing rejected evidence."""
    if serialization_issue is None:
        serialization_issue = agent_eval_report_serialization_issue(payload)
    serialized: str | None = None
    if serialization_issue is None:
        try:
            serialized = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (OverflowError, RecursionError, TypeError, ValueError):
            serialization_issue = "Canonical JSON serialization failed."
    if serialized is None:
        serialized = json.dumps(
            {
                "serialization_issue": serialization_issue,
                "payload": _agent_eval_bounded_fingerprint_value(payload),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    return hashlib.sha256(serialized.encode("utf-8", errors="surrogatepass")).hexdigest()


def _agent_eval_evaluated_at_rank(value: object) -> float:
    parsed = _agent_eval_evaluated_at_datetime(value)
    if parsed is None:
        return float("-inf")
    try:
        return parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return float("-inf")


def _agent_eval_candidate_rank(
    result: ValidationResult,
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    serialization_issue = agent_eval_report_serialization_issue(payload)
    if serialization_issue is not None:
        verdict, execution_status, explicit_failure, conservative_outcome_rank = _agent_eval_rejected_truth_state(
            payload
        )
        result_evidence = len(result.success_details) + len(result.findings) + result.summary.checks_performed
        priority = 3 if execution_status in {"failed", "incomplete", "skipped"} else 1
        return (
            int(explicit_failure),
            priority,
            conservative_outcome_rank,
            float("-inf"),
            result_evidence,
            0,
            0,
            -1.0,
            -1.0,
            len(payload),
            "",
            "",
            verdict,
            _agent_eval_payload_fingerprint(payload, serialization_issue=serialization_issue),
        )

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    verdict, execution_status, truth_consistent = _agent_eval_truth_state(payload)
    payload_attempts = max(
        _agent_eval_count(payload.get("scored_attempts")),
        _agent_eval_count(summary.get("scored_attempts")),
    )
    agents = payload.get("agents") if isinstance(payload.get("agents"), dict) else {}
    agent_evidence = False
    strong_agent_evidence = False
    for agent in agents.values():
        if not isinstance(agent, dict) or agent.get("execution_status") != "succeeded":
            continue
        score = _agent_eval_finite_score(agent.get("with_skill"))
        if score is None:
            score = _agent_eval_finite_score(agent.get("overall_score"))
        dimensions = agent.get("dimensions")
        dimension_evidence = isinstance(dimensions, list) and any(
            isinstance(dimension, dict)
            and _agent_eval_finite_score(dimension.get("with_skill", dimension.get("score"))) is not None
            for dimension in dimensions
        )
        if score is None and not dimension_evidence:
            continue
        agent_evidence = True
        if max(payload_attempts, _agent_eval_count(agent.get("scored_attempts"))) > 0:
            strong_agent_evidence = True

    valid_verdict = truth_consistent and verdict in {"pass", "neutral", "fail"}
    dimension_verdict = agent_eval_dimension_verdict(payload)
    conservative_outcome_rank = max(
        ({"pass": 0, "neutral": 1, "fail": 3}.get(candidate, -1) for candidate in (verdict, dimension_verdict)),
        default=-1,
    )
    if execution_status == "failed":
        conservative_outcome_rank = max(conservative_outcome_rank, 3)
    elif execution_status == "incomplete":
        conservative_outcome_rank = max(conservative_outcome_rank, 2)
    explicit_failure_rank = int(verdict == "fail" or dimension_verdict == "fail" or execution_status == "failed")
    result_evidence = len(result.success_details) + len(result.findings) + result.summary.checks_performed
    if agent_eval_publication_evidence_complete(payload):
        priority = 8
    elif execution_status == "succeeded" and valid_verdict and strong_agent_evidence:
        priority = 6
    elif execution_status == "succeeded" and valid_verdict and agent_evidence:
        priority = 5
    elif not execution_status and valid_verdict and result_evidence:
        priority = 4
    elif execution_status in {"failed", "incomplete", "skipped"}:
        priority = 3
    elif payload:
        priority = 1
    else:
        priority = 0

    dataset_count = _agent_eval_dataset_count(payload)
    evaluated_at = _agent_eval_safe_text(payload.get("evaluated_at") or summary.get("evaluated_at"))
    evaluated_at_rank = _agent_eval_evaluated_at_rank(evaluated_at)
    digest = _agent_eval_safe_text(payload.get("dataset_digest") or summary.get("dataset_digest"))
    best_score = max(
        (
            score
            for agent in agents.values()
            if isinstance(agent, dict)
            and (score := _agent_eval_finite_score(agent.get("with_skill", agent.get("overall_score")))) is not None
        ),
        default=-1.0,
    )
    runtime = _agent_eval_finite_number(payload.get("runtime_seconds"))
    return (
        explicit_failure_rank,
        priority,
        conservative_outcome_rank,
        evaluated_at_rank,
        result_evidence,
        dataset_count,
        payload_attempts,
        best_score,
        runtime if runtime is not None else -1.0,
        len(payload),
        evaluated_at,
        digest,
        verdict,
        _agent_eval_payload_fingerprint(payload),
    )


def select_agent_eval_candidate(
    results: list[ValidationResult],
) -> tuple[ValidationResult, dict[str, Any]] | None:
    """Select the strongest Tier 3 result and payload independent of input order."""
    candidates = [
        (result, payload)
        for result in results
        if isinstance(result.metadata, dict) and isinstance((payload := result.metadata.get("agent_eval")), dict)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: _agent_eval_candidate_rank(*candidate),
    )


def select_agent_eval_payload(results: list[ValidationResult]) -> dict[str, Any] | None:
    """Select the strongest Tier 3 payload without depending on result order."""
    selected = select_agent_eval_candidate(results)
    return selected[1] if selected is not None else None


def resolve_benchmark_policy(
    results: list[ValidationResult],
    agent_eval: dict[str, Any] | None,
    *,
    expected_skill_name: str | None = None,
) -> dict[str, bool]:
    """Resolve required publication tiers from persisted policy metadata.

    Each policy key is resolved independently. The selected canonical payload
    may waive evidence only when persisted peer payloads agree; conflicts and
    missing values fail closed to required evidence.
    """
    selected_payload = agent_eval if isinstance(agent_eval, dict) else select_agent_eval_payload(results)
    expected_target = _selected_publication_target(results, selected_payload, expected_skill_name)
    has_publication_target_conflict = _agent_eval_has_publication_target_conflict(selected_payload) or any(
        _result_has_publication_target_conflict(result) for result in results
    )

    def payload_matches_target(payload: dict[str, Any]) -> bool:
        return bool(
            expected_target is not None
            and agent_eval_publication_target(payload) == expected_target
            and _agent_eval_target_skill_issue(payload, expected_skill_name) is None
        )

    selected_policy_payload = (
        selected_payload if isinstance(selected_payload, dict) and payload_matches_target(selected_payload) else None
    )
    payloads: list[dict[str, Any]] = [selected_policy_payload] if isinstance(selected_policy_payload, dict) else []
    foreign_payloads: list[dict[str, Any]] = (
        [selected_payload] if isinstance(selected_payload, dict) and selected_policy_payload is None else []
    )
    for result in results:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        payload = metadata.get("agent_eval")
        if not isinstance(payload, dict):
            continue
        destination = payloads if payload_matches_target(payload) else foreign_payloads
        if all(payload is not candidate for candidate in destination):
            destination.append(payload)
    result_policies: list[object] = []
    foreign_result_policies: list[object] = []
    for result in results:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        policy = metadata.get("benchmark_policy")
        producer = result_publication_evidence(result)
        destination = (
            result_policies
            if producer is not None
            and expected_target is not None
            and _result_publication_target(result) == expected_target
            else foreign_result_policies
        )
        destination.append(policy)

    def peer_value(candidates: list[object], key: str) -> bool | None:
        values = [
            value
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance((value := candidate.get(key)), bool)
        ]
        if not values:
            return None
        return values[0] if len(set(values)) == 1 else True

    def payload_value(payload: dict[str, Any], key: str) -> bool | None:
        summary = payload.get("summary")
        summary_policy = summary.get("benchmark_policy") if isinstance(summary, dict) else None
        values: list[bool] = []
        for policy in (payload.get("benchmark_policy"), summary_policy):
            if not isinstance(policy, dict) or key not in policy:
                continue
            value = policy[key]
            if isinstance(value, bool):
                values.append(value)
        if not values:
            return None
        return values[0] if len(set(values)) == 1 else True

    resolved: dict[str, bool] = {}
    for key in ("tier2_required", "tier3_required"):
        if has_publication_target_conflict:
            resolved[key] = True
            continue
        selected_value = (
            payload_value(selected_policy_payload, key) if isinstance(selected_policy_payload, dict) else None
        )
        peer_payload_values = [
            payload_value(payload, key) for payload in payloads if payload is not selected_policy_payload
        ]
        peer_payload_values = [value for value in peer_payload_values if value is not None]
        # Foreign payloads or result producers can require evidence but can
        # never waive it for this report, even across policy precedence levels.
        if any(payload_value(payload, key) is True for payload in foreign_payloads):
            peer_payload_values.append(True)
        if peer_value(foreign_result_policies, key) is True:
            peer_payload_values.append(True)
        if selected_value is not None:
            # The canonical payload may waive a tier only when every peer
            # payload that persists the same key agrees. A weak or stale peer
            # can force required evidence, but cannot create a waiver.
            resolved[key] = selected_value if all(value == selected_value for value in peer_payload_values) else True
            continue
        if any(peer_payload_values):
            resolved[key] = True
            continue

        # Result objects are peers, not an ordered precedence chain. A conflict
        # therefore fails closed regardless of aggregator input order.
        result_value = peer_value(result_policies, key)
        if peer_value(foreign_result_policies, key) is True:
            result_value = True
        resolved[key] = result_value if result_value is not None else True
    return resolved


def _result_agent_eval_run_id_issue(result: ValidationResult) -> str | None:
    """Reject a persisted outer Tier 3 run claim that contradicts its payload."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if "run_id" not in metadata:
        return None
    outer_run_id = metadata.get("run_id")
    if (
        not isinstance(outer_run_id, str)
        or not 1 <= len(outer_run_id) <= 160
        or _AGENT_EVAL_RUN_ID.fullmatch(outer_run_id) is None
    ):
        return "Tier 3 result contains an invalid outer run identity."
    payload = metadata.get("agent_eval")
    payload_run_id = agent_eval_publication_run_id(payload if isinstance(payload, dict) else None)
    if payload_run_id != outer_run_id:
        return "Tier 3 result contains contradictory run identities."
    return None


@dataclass(frozen=True)
class Tier3EvidenceAssessment:
    """Publication-facing interpretation of the canonical Tier 3 payload."""

    status: str
    evidence_complete: bool
    execution_status: str
    verdict: str
    payload: dict[str, Any] | None
    reason: str | None = None


@dataclass(frozen=True)
class PublicationAssessment:
    """Publication status kept separate from a command's process exit gate."""

    status: str
    benchmark_policy: dict[str, bool]
    tier3: Tier3EvidenceAssessment
    reasons: tuple[str, ...] = ()


def result_has_execution_evidence(result: ValidationResult) -> bool:
    """Return whether a non-skipped validator proves that it executed work."""
    return bool(result.success_details or result.findings or result.summary.checks_performed > 0)


def is_tier3_result(result: ValidationResult) -> bool:
    """Return whether a result belongs to live Tier 3 evaluation."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    return result.validator_name == "AGENT_EVAL" or isinstance(metadata.get("agent_eval"), dict)


def is_tier2_result(result: ValidationResult) -> bool:
    """Return whether a result belongs to semantic Tier 2 validation."""
    return bool(
        not is_tier3_result(result)
        and (
            is_tier2_validator_name(result.validator_name)
            or any(finding.category == "CONTENT_DEDUP" for finding in result.findings)
        )
    )


def _publication_identity_key(value: object) -> str | None:
    """Normalize canonical aliases without folding filesystem distinctions."""
    if not publication_identity_present(value):
        return None
    return unicodedata.normalize("NFC", _agent_eval_safe_text(value))


def _agent_eval_target_skill_issue(
    payload: dict[str, Any] | None,
    expected_skill_name: str | None,
    *,
    require_publication_target: bool = True,
) -> str | None:
    """Return a fail-closed reason when Tier 3 targets a different skill."""
    expected_key = _publication_identity_key(expected_skill_name)
    if expected_skill_name is not None and expected_key is None:
        return "The expected target skill identity is invalid."

    persisted_skill_name = _agent_eval_consistent_text_field(payload, "skill_name")
    persisted_key = _publication_identity_key(persisted_skill_name)
    if persisted_key is None:
        return "Tier 3 evidence lacks a consistent target skill identity."
    if expected_key is not None and persisted_key != expected_key:
        return "Tier 3 evidence belongs to a different target skill."
    if not require_publication_target:
        return None
    publication_target = agent_eval_publication_target(payload)
    if publication_target is None:
        return "Tier 3 evidence lacks a canonical target source identity."
    if publication_target.skill_name != persisted_skill_name:
        return "Tier 3 evidence contains contradictory target skill identities."
    return None


def assess_tier3_evidence(
    results: list[ValidationResult],
    payload: dict[str, Any] | None = None,
    *,
    expected_skill_name: str | None = None,
) -> Tier3EvidenceAssessment:
    """Classify Tier 3 evidence once for every reporter."""
    tier3_results = [result for result in results if is_tier3_result(result)]
    selected_payload = payload if isinstance(payload, dict) else select_agent_eval_payload(tier3_results)
    summary = (
        selected_payload.get("summary")
        if isinstance(selected_payload, dict) and isinstance(selected_payload.get("summary"), dict)
        else {}
    )
    serialization_issue = (
        agent_eval_report_serialization_issue(selected_payload) if isinstance(selected_payload, dict) else None
    )
    if serialization_issue is not None:
        raw_verdict, execution_status, explicit_failure, _ = _agent_eval_rejected_truth_state(selected_payload)
        truth_consistent = False
        dimension_verdict = None
    else:
        raw_verdict, execution_status, truth_consistent = _agent_eval_truth_state(selected_payload)
        dimension_verdict = agent_eval_dimension_verdict(selected_payload)
        explicit_failure = False

    if any(result.is_incomplete for result in tier3_results):
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status or "incomplete",
            raw_verdict or "incomplete",
            selected_payload,
            "Tier 3 scanner evidence is incomplete.",
        )
    if any(
        not result.passed and (serialization_issue is not None or not is_cleanly_skipped(result))
        for result in tier3_results
    ):
        return Tier3EvidenceAssessment(
            "fail",
            False,
            execution_status or "failed",
            raw_verdict or "fail",
            selected_payload,
            "A Tier 3 validator failed.",
        )
    if _agent_eval_has_publication_target_conflict(selected_payload) or any(
        _result_has_publication_target_conflict(result) for result in tier3_results
    ):
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status or "incomplete",
            raw_verdict or "incomplete",
            selected_payload,
            "Tier 3 source identity changed during evaluation.",
        )
    if run_id_issue := next(
        (issue for result in tier3_results if (issue := _result_agent_eval_run_id_issue(result)) is not None),
        None,
    ):
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status or "incomplete",
            raw_verdict or "incomplete",
            selected_payload,
            run_id_issue,
        )
    if serialization_issue is not None:
        if explicit_failure:
            return Tier3EvidenceAssessment(
                "fail",
                False,
                execution_status or "failed",
                raw_verdict or "fail",
                selected_payload,
                f"{serialization_issue} The rejected Tier 3 payload also records an explicit failure.",
            )
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status or "incomplete",
            raw_verdict or "incomplete",
            selected_payload,
            f"{serialization_issue} Publication completeness cannot be proven.",
        )
    if tier3_results and all(is_cleanly_skipped(result) for result in tier3_results):
        if selected_payload is not None and (
            target_skill_issue := _agent_eval_target_skill_issue(
                selected_payload,
                expected_skill_name,
                require_publication_target=False,
            )
        ):
            return Tier3EvidenceAssessment(
                "incomplete",
                False,
                execution_status or "incomplete",
                raw_verdict or "incomplete",
                selected_payload,
                target_skill_issue,
            )
        return Tier3EvidenceAssessment(
            "skipped",
            False,
            execution_status or "skipped",
            raw_verdict or "skipped",
            selected_payload,
            "Tier 3 was skipped.",
        )
    if (selected_payload is not None or tier3_results) and not truth_consistent:
        if raw_verdict == "fail" or dimension_verdict == "fail" or execution_status == "failed":
            return Tier3EvidenceAssessment(
                "fail",
                False,
                execution_status or "failed",
                raw_verdict or "fail",
                selected_payload,
                "Tier 3 contains contradictory truth fields including an explicit failure.",
            )
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status or "incomplete",
            raw_verdict or "incomplete",
            selected_payload,
            "Tier 3 truth fields are missing, invalid, or contradictory.",
        )
    if raw_verdict == "fail" or dimension_verdict == "fail" or execution_status == "failed":
        return Tier3EvidenceAssessment(
            "fail",
            agent_eval_publication_evidence_complete(selected_payload),
            execution_status or "failed",
            raw_verdict or "fail",
            selected_payload,
            "Tier 3 evidence records a failing verdict.",
        )
    if execution_status == "incomplete":
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status,
            raw_verdict or "incomplete",
            selected_payload,
            "Tier 3 execution evidence is incomplete.",
        )
    if any(
        isinstance(result.metadata, dict) and bool(result.metadata.get("skipped")) and not is_cleanly_skipped(result)
        for result in tier3_results
    ):
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status or "incomplete",
            raw_verdict or "incomplete",
            selected_payload,
            "Tier 3 skip metadata contradicts the persisted execution evidence.",
        )
    if selected_payload is not None and (
        target_skill_issue := _agent_eval_target_skill_issue(selected_payload, expected_skill_name)
    ):
        return Tier3EvidenceAssessment(
            "incomplete",
            False,
            execution_status or "incomplete",
            raw_verdict or "incomplete",
            selected_payload,
            target_skill_issue,
        )
    evidence_complete = agent_eval_publication_evidence_complete(selected_payload)
    if not evidence_complete:
        payload_attempts = max(
            _agent_eval_count((selected_payload or {}).get("scored_attempts")),
            _agent_eval_count(summary.get("scored_attempts")),
        )
        has_scored_agent = any(
            _agent_eval_safe_text(agent.get("execution_status")).lower() == "succeeded"
            and (
                _agent_eval_finite_score(agent.get("with_skill", agent.get("overall_score"))) is not None
                or bool(_agent_eval_dimension_scores(agent))
            )
            and max(payload_attempts, _agent_eval_count(agent.get("scored_attempts"))) > 0
            for agent in _agent_eval_agents(selected_payload).values()
        )
        if execution_status == "succeeded" and raw_verdict in {"pass", "neutral"} and has_scored_agent:
            effective_verdict = "neutral" if "neutral" in {raw_verdict, dimension_verdict} else "pass"
            return Tier3EvidenceAssessment(
                effective_verdict,
                False,
                execution_status,
                raw_verdict,
                selected_payload,
                "Tier 3 ran but lacks publication-complete provenance or dimension evidence.",
            )
        return Tier3EvidenceAssessment(
            "incomplete" if tier3_results or selected_payload else "not_run",
            False,
            execution_status or ("incomplete" if tier3_results else "not_run"),
            raw_verdict or ("incomplete" if tier3_results else "not_run"),
            selected_payload,
            "Tier 3 lacks publication-complete execution evidence.",
        )

    effective_verdict = "neutral" if "neutral" in {raw_verdict, dimension_verdict} else "pass"
    return Tier3EvidenceAssessment(
        effective_verdict,
        True,
        execution_status,
        raw_verdict,
        selected_payload,
    )


def assess_publication(
    results: list[ValidationResult],
    agent_eval: dict[str, Any] | None = None,
    *,
    expected_skill_name: str | None = None,
) -> PublicationAssessment:
    """Return a shared, fail-closed publication assessment for all reporters."""
    payload = agent_eval if isinstance(agent_eval, dict) else select_agent_eval_payload(results)
    tier3_results = [result for result in results if is_tier3_result(result)]
    producer_results = [
        (result, producer) for result in results if (producer := result_publication_evidence(result)) is not None
    ]
    tier1_results = [result for result, producer in producer_results if producer.tier == 1]
    tier2_results = [result for result, producer in producer_results if producer.tier == 2]
    publication_target = _selected_publication_target(results, payload, expected_skill_name)

    peer_skill_names: dict[str, str] = {}
    for result in results:
        if not is_tier3_result(result) and result_publication_evidence(result) is None:
            continue
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        candidates: list[object] = [metadata.get("skill_name"), metadata.get("target_skill_name")]
        quality = metadata.get("quality_scores")
        if isinstance(quality, dict):
            candidates.append(quality.get("skill_name"))
        quality_all = metadata.get("quality_scores_all")
        if isinstance(quality_all, list):
            candidates.extend(item.get("skill_name") for item in quality_all if isinstance(item, dict))
        agent_eval_payload = metadata.get("agent_eval")
        if isinstance(agent_eval_payload, dict):
            candidates.append(_agent_eval_consistent_text_field(agent_eval_payload, "skill_name"))
        if (result_target := _result_publication_target(result)) is not None:
            candidates.append(result_target.skill_name)
        for candidate in candidates:
            if (key := _publication_identity_key(candidate)) is not None:
                peer_skill_names.setdefault(key, str(candidate))

    explicit_expected_key = _publication_identity_key(expected_skill_name)
    peer_expected_key = next(iter(peer_skill_names), None) if len(peer_skill_names) == 1 else None
    tier3_expected_skill_name = (
        expected_skill_name
        if expected_skill_name is not None
        else publication_target.skill_name
        if publication_target is not None
        else peer_skill_names.get(peer_expected_key)
        if peer_expected_key is not None
        else None
    )
    policy = resolve_benchmark_policy(
        results,
        payload,
        expected_skill_name=tier3_expected_skill_name,
    )
    tier3 = assess_tier3_evidence(
        results,
        payload,
        expected_skill_name=tier3_expected_skill_name,
    )

    reasons: list[str] = []
    if any(result.is_incomplete for result in results):
        reasons.append("One or more validators reported incomplete scanner evidence.")

    def blocking_skip(result: ValidationResult) -> bool:
        producer = result_publication_evidence(result)
        tier2 = producer.tier == 2 if producer is not None else is_tier2_result(result)
        return bool(
            is_cleanly_skipped(result)
            and not is_advisory_agent_eval_skip(result)
            and not (tier2 and not policy["tier2_required"])
            and not (is_tier3_result(result) and not policy["tier3_required"])
        )

    if any(blocking_skip(result) for result in results):
        reasons.append("A publication-required validator was skipped.")
    if reasons:
        return PublicationAssessment("incomplete", policy, tier3, tuple(reasons))

    if any(not result.passed and not is_cleanly_skipped(result) for result in results):
        return PublicationAssessment("fail", policy, tier3, ("A validator failed.",))
    if tier3.status == "fail":
        return PublicationAssessment("fail", policy, tier3, (tier3.reason or "Tier 3 failed.",))

    if expected_skill_name is not None and explicit_expected_key is None:
        reasons.append("The expected target skill identity is invalid.")
    if len(peer_skill_names) > 1:
        reasons.append("Validation results contain conflicting target skill identities.")
    if (
        explicit_expected_key is not None
        and peer_expected_key is not None
        and explicit_expected_key != peer_expected_key
    ):
        reasons.append("Validation results belong to a different target skill.")

    if publication_target is None:
        reasons.append("Validation results lack one canonical publication target identity.")
    elif (
        explicit_expected_key is not None
        and _publication_identity_key(publication_target.skill_name) != explicit_expected_key
    ):
        reasons.append("Validation results belong to a different publication target.")

    def require_bound_evidence(
        tier_results: list[ValidationResult],
        tier_label: str,
    ) -> None:
        for result in tier_results:
            result_target = _result_publication_target(result)
            if result_target is None:
                reasons.append(f"{tier_label} evidence lacks a canonical publication target identity.")
            elif publication_target is None or result_target != publication_target:
                reasons.append(f"{tier_label} evidence belongs to a different publication target.")

    executed_tier1 = [result for result in tier1_results if not is_cleanly_skipped(result)]
    if not executed_tier1:
        reasons.append("Recognized built-in Tier 1 evidence is missing.")
    elif any(not result_has_execution_evidence(result) for result in executed_tier1):
        reasons.append("Tier 1 lacks trustworthy execution evidence.")
    require_bound_evidence(executed_tier1, "Tier 1")

    executed_tier2 = [result for result in tier2_results if not is_cleanly_skipped(result)]
    if any(not result_has_execution_evidence(result) for result in executed_tier2):
        reasons.append("Tier 2 lacks trustworthy execution evidence.")
    elif policy["tier2_required"] and not executed_tier2:
        reasons.append("Required recognized built-in Tier 2 evidence is missing.")
    require_bound_evidence(executed_tier2, "Tier 2")

    executed_tier3 = [result for result in tier3_results if not is_cleanly_skipped(result)]
    require_bound_evidence(executed_tier3, "Tier 3")

    has_present_tier3 = bool(tier3_results) and tier3.status != "skipped"
    if (policy["tier3_required"] or has_present_tier3) and not tier3.evidence_complete:
        reasons.append(tier3.reason or "Required Tier 3 evidence is missing.")

    if reasons:
        return PublicationAssessment("incomplete", policy, tier3, tuple(dict.fromkeys(reasons)))
    if tier3.status == "neutral":
        return PublicationAssessment("neutral", policy, tier3)
    return PublicationAssessment("pass", policy, tier3)


def passes_required_gate(result: ValidationResult) -> bool:
    """Return whether *result* permits the required validation gate to pass."""
    gating = result.metadata.get("gating") if isinstance(result.metadata, dict) else None
    if isinstance(gating, dict):
        if not gating.get("blocking", True):
            return True
        if is_tier3_result(result):
            selected = select_agent_eval_candidate([result])
            payload = selected[1] if selected is not None else None
            verdict, execution_status, truth_consistent = _agent_eval_truth_state(payload)
            return bool(
                result.passed
                and not result.is_incomplete
                and truth_consistent
                and verdict == "pass"
                and execution_status == "succeeded"
                and agent_eval_dimension_verdict(payload) == "pass"
                and isinstance(payload, dict)
                and _agent_eval_dataset_count(payload) > 0
                and _agent_eval_attempt_coverage_complete(payload)
            )
        return bool(result.passed)
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
