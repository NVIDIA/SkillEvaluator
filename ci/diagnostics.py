# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone CI artifact helper.
"""Bounded, redacted native-worker startup diagnostics; never agent trajectories."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path

from skillevaluator.utils.secret_redaction import REDACTED, Redactor
from skillevaluator.utils.secure_fs import SecureRoot, _validate_opened_file, read_bounded

MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FILES = 100
MAX_ENTRIES = 10_000
MAX_RECORDS = 200
_NAMES = frozenset({"failure.json", "exception.txt", "traceback.txt", "trial.log", "stderr.log", "stdout.log"})
_EXCLUDED = frozenset(
    {
        "agent",
        "agents",
        "user-agent",
        "verifier",
        "artifacts",
        "trajectories",
        "payloads",
        "credentials",
        "config",
        "configs",
        "workspace",
        "consumer",
        "task",
        "tasks",
        ".git",
        ".env",
        ".ssh",
        ".aws",
        ".config",
        "node_modules",
        "__pycache__",
    }
)
_KEY = re.compile(
    r"(?i)(?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|password|passwd|pwd|authorization)"
)
_NVIDIA = re.compile(r"\bnvapi-[A-Za-z0-9_-]{8,}")
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s\"'`<>;,]+")
_AUTHORIZATION = re.compile(r"(?i)(\b(?:proxy[-_])?authorization[\"']?[ \t]*[:=][ \t]*)[^\r\n]+")
_ASSIGNMENT = re.compile(
    r"(?i)((?<![a-z0-9_-])(?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|password|passwd|pwd)[\"']?[ \t]*[:=][ \t]*)"
    r"""(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^\s"'`;,}{]+)"""
)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_MARKER = b"\n[diagnostic truncated]\n"


def _mask_text(value: str, redactor: Redactor) -> str:
    value = redactor.text(value)
    value = _NVIDIA.sub(REDACTED, value)
    value = _AUTHORIZATION.sub(lambda match: match.group(1) + REDACTED, value)
    value = _BEARER.sub("Bearer " + REDACTED, value)
    return _ASSIGNMENT.sub(lambda match: match.group(1) + REDACTED, value)


def _mask_object(value, redactor: Redactor):
    if isinstance(value, str):
        return _mask_text(value, redactor)
    if isinstance(value, list):
        return [_mask_object(item, redactor) for item in value]
    if isinstance(value, dict):
        return {key: REDACTED if _KEY.fullmatch(key) else _mask_object(item, redactor) for key, item in value.items()}
    return value


def _regular_root(path: Path) -> Path:
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Diagnostic roots must not contain symlinks")
    return path


def _write(root: Path, relative: Path, payload: bytes) -> None:
    target = root / relative
    for parent in (target.parent, *target.parent.parents):
        if parent.is_symlink():
            raise ValueError("Diagnostic output must not contain symlinks")
        if parent == root:
            break
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Diagnostic output must be regular")
    descriptor, temporary = tempfile.mkstemp(prefix=".diagnostic-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read(root: SecureRoot, relative: Path, metadata: os.stat_result) -> tuple[bytes, bool]:
    if metadata.st_size <= MAX_FILE_BYTES:
        return root.read_bytes(relative, MAX_FILE_BYTES, expected=metadata)[0], False
    if relative.suffix == ".json" or os.name != "posix":
        raise ValueError("oversized_json_or_platform")
    # The secure reader pins every directory and rejects redirects/hard links.
    descriptor = root._open_posix(relative, metadata, allow_hardlinks=False)
    try:
        budget = max(0, (MAX_FILE_BYTES - len(_MARKER)) // 2)
        head = read_bounded(descriptor, budget, truncate=True)
        os.lseek(descriptor, max(0, metadata.st_size - budget), os.SEEK_SET)
        tail = read_bounded(descriptor, budget, truncate=True)
        _validate_opened_file(os.fstat(descriptor), relative, metadata)
    finally:
        os.close(descriptor)
    # Drop split edge lines so truncation cannot expose an unknown credential's
    # suffix after its NVIDIA/Bearer/assignment prefix has been cut away.
    head = head[: head.rfind(b"\n") + 1] if b"\n" in head else b""
    tail = tail[tail.find(b"\n") + 1 :] if b"\n" in tail else b""
    return head + _MARKER + tail, True


def retain_diagnostics(raw: Path, destination: Path, tokens: set[str], *, command_log: Path | None = None) -> dict:
    """Retain whitelisted worker diagnostics below ``destination/diagnostics``.

    The caller supplies an evaluator-owned output root. The optional command log
    is an explicit separate input, retained as ``command.log``. Each unsafe or
    unsupported child is omitted with a fixed reason; input content and absolute
    paths never appear in index errors. No HTML or agent/state/config files are
    copied, and no diagnostic changes the evaluation verdict.
    """
    raw, destination = _regular_root(raw), _regular_root(destination)
    if not raw.is_dir() or destination.is_relative_to(raw) or raw.is_relative_to(destination):
        raise ValueError("Separate regular worker and artifact roots are required")
    output = destination / "diagnostics"
    if output.exists() or output.is_symlink():
        raise ValueError("Diagnostics require a fresh output directory")
    output.mkdir(parents=True)
    redactor = Redactor(tokens)
    result = {
        "schema_version": 1,
        "index": "diagnostics/index.json",
        "retained": [],
        "omitted": [],
        "truncated": [],
        "errors": [],
        "omitted_count": 0,
        "total_bytes": 0,
    }
    reads = visited = 0

    def label(relative: Path) -> str:
        text = relative.as_posix()
        if len(text) > 1024 or any(ord(c) < 32 or ord(c) == 127 for c in text) or _mask_text(text, redactor) != text:
            return "[REDACTED PATH]"
        return text

    def record(bucket: str, relative: Path, reason: str) -> None:
        if bucket == "omitted":
            result["omitted_count"] += 1
        if len(result[bucket]) < MAX_RECORDS:
            result[bucket].append({"path": label(relative), "reason": reason})

    def collect(source: SecureRoot, relative: Path, target: Path, metadata: os.stat_result) -> None:
        nonlocal reads
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            record("errors", target, "special_file")
            return
        if reads >= MAX_FILES or len(result["retained"]) >= MAX_FILES:
            record("omitted", target, "file_limit")
            return
        if result["total_bytes"] >= MAX_TOTAL_BYTES:
            record("omitted", target, "total_limit")
            return
        if label(target) == "[REDACTED PATH]":
            record("omitted", target, "unsafe_or_sensitive_path")
            return
        reads += 1
        try:
            content, truncated = _read(source, relative, metadata)
            if b"\x00" in content:
                record("omitted", target, "binary_or_encoding")
                return
            text = _ANSI.sub("", content.decode("utf-8-sig"))
            if target.suffix == ".json":
                value = json.loads(text, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")))
                value = _mask_object(redactor.object(value), redactor)
                payload = (json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode()
                if len(payload) > MAX_FILE_BYTES:
                    record("omitted", target, "redacted_json_limit")
                    result["truncated"].append(target.as_posix())
                    return
            else:
                payload = _mask_text(text, redactor).encode("utf-8")
                if len(payload) > MAX_FILE_BYTES:
                    # Redaction may grow short literals. Trim complete lines of
                    # already-redacted text, preserving the terminal error.
                    payload = payload[-MAX_FILE_BYTES:]
                    payload = payload[payload.find(b"\n") + 1 :] if b"\n" in payload else b""
                    truncated = True
            if result["total_bytes"] + len(payload) > MAX_TOTAL_BYTES:
                record("omitted", target, "total_limit")
                return
            _write(output, target, payload)
            result["retained"].append(
                {"path": target.as_posix(), "bytes": len(payload), "source_bytes": metadata.st_size}
            )
            result["total_bytes"] += len(payload)
            if truncated:
                result["truncated"].append(target.as_posix())
        except UnicodeError:
            record("omitted", target, "binary_or_encoding")
        except (ValueError, RecursionError):
            record("omitted", target, "invalid_or_oversized_diagnostic")
            if metadata.st_size > MAX_FILE_BYTES:
                result["truncated"].append(target.as_posix())
        except OSError:
            record("errors", target, "read_or_write_error")

    def collect_setup(source: SecureRoot, agent: Path) -> None:
        nonlocal visited
        setup = agent / "setup"
        if visited >= MAX_ENTRIES:
            record("omitted", setup.relative_to(raw), "discovery_limit")
            return
        visited += 1
        if setup.is_symlink():
            record("errors", setup.relative_to(raw), "unsafe_link")
            return
        if not setup.is_dir():
            return
        try:
            with os.scandir(setup) as entries:
                for entry in entries:
                    relative = Path(entry.path).relative_to(raw)
                    if visited >= MAX_ENTRIES:
                        record("omitted", relative, "discovery_limit")
                        break
                    visited += 1
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(metadata.st_mode):
                        record("errors", relative, "unsafe_link")
                    elif re.fullmatch(
                        r"install-(?:root-prereqs|agent-runtime)(?:\.attempt-[0-9]+)?(?:\.(?:failed|timeout))?\.log",
                        entry.name,
                    ):
                        collect(source, relative, relative, metadata)
                    else:
                        record("omitted", relative, "not_allowed")
        except OSError:
            record("errors", setup.relative_to(raw), "read_or_write_error")

    with SecureRoot(raw) as source:
        for directory, directories, names in os.walk(raw, followlinks=False):
            directories.sort()
            names.sort()
            for name in list(directories):
                visited += 1
                if visited > MAX_ENTRIES:
                    record("omitted", Path(directory).relative_to(raw), "discovery_limit")
                    directories.clear()
                    break
                path = Path(directory) / name
                relative = path.relative_to(raw)
                if path.is_symlink():
                    directories.remove(name)
                    record("errors", relative, "unsafe_link")
                elif name.casefold() in _EXCLUDED or len(relative.parts) > 64:
                    directories.remove(name)
                    if name == "agent":
                        collect_setup(source, path)
                    record("omitted", relative, "not_allowed")
            for name in names:
                visited += 1
                relative = (Path(directory) / name).relative_to(raw)
                if visited > MAX_ENTRIES:
                    record("omitted", relative, "discovery_limit")
                    directories.clear()
                    break
                try:
                    metadata = (raw / relative).lstat()
                except OSError:
                    record("errors", relative, "read_or_write_error")
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    record("errors", relative, "unsafe_link")
                elif not stat.S_ISREG(metadata.st_mode):
                    record("errors", relative, "special_file")
                elif name not in _NAMES and re.fullmatch(r"(?:docker|environment)[A-Za-z0-9_.-]*\.log", name) is None:
                    record("omitted", relative, "not_allowed")
                else:
                    collect(source, relative, relative, metadata)
            if visited > MAX_ENTRIES:
                break
    if command_log is not None:
        command_log = _regular_root(command_log)
        try:
            with SecureRoot(command_log.parent) as source:
                collect(source, Path(command_log.name), Path("command.log"), command_log.lstat())
        except OSError:
            record("errors", Path("command.log"), "read_or_write_error")
    _write(output, Path("index.json"), (json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode())
    return result
