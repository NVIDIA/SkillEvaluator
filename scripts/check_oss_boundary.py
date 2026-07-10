# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed when OSS source or distributions contain private integrations."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import string
import subprocess
import sys
import tarfile
import warnings
import zipfile
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import NamedTuple

_MAX_MEMBER_BYTES = 10 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_STATIC_STRING_CHARS = 1024 * 1024
_MAX_STATIC_STRING_DEPTH = 64
_MAX_SHELL_DECODE_CHARS = 1024 * 1024
_BINARY_SUFFIXES = frozenset(
    {
        ".7z",
        ".a",
        ".class",
        ".dll",
        ".dylib",
        ".eot",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".o",
        ".pdf",
        ".png",
        ".pyc",
        ".so",
        ".tar",
        ".tgz",
        ".ttf",
        ".webp",
        ".whl",
        ".woff",
        ".woff2",
        ".xz",
        ".zip",
    }
)
_BINARY_ROOT_FILES = frozenset({".coverage", ".DS_Store", "Thumbs.db"})
_PYTHON_SUFFIXES = frozenset({".py", ".pyi", ".pyw"})
_BASH_ANSI_C_PREFIX = chr(36) + chr(39)
_BASH_LOCALE_QUOTE_RE = re.compile(r'(?<!\\)\$(?=")')
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_OCTAL_DIGITS = frozenset("01234567")
_BASH_COMMON_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}


class BoundaryFinding(NamedTuple):
    """One denied release-boundary match."""

    path: str
    rule: str
    line: int
    message: str
    excerpt: str


class BoundaryAllowance(NamedTuple):
    """One line-specific, justified exception."""

    path: str
    rule: str
    line: int
    reason: str
    expires: date
    review_owner: str


class BoundaryRule(NamedTuple):
    """A named deny expression and user-facing explanation."""

    rule_id: str
    pattern: re.Pattern[str]
    message: str


def _pattern(*parts: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile("".join(parts), flags)


DENY_RULES = (
    BoundaryRule(
        "private-repository-host",
        _pattern(r"\bgitlab", r"(?:-master)?", r"\.nvidia\.(?:com|net)\b", flags=re.IGNORECASE),
        "private repository host",
    ),
    BoundaryRule(
        "private-package-host",
        _pattern(
            r"(?:\b(?:urm|artifactory|pypi(?:\.[a-z0-9-]+)?)",
            r"\.nvidia\.(?:com|net)\b|\bnv[-_ ]shared[-_ ]pypi\b)",
            flags=re.IGNORECASE,
        ),
        "private package host",
    ),
    BoundaryRule(
        "internal-nvidia-credential",
        _pattern(r"\bnvidia", r"(?:[\s\"'()+]{0,24})?", r"_inference_key\b", flags=re.IGNORECASE),
        "retired NVIDIA credential name",
    ),
    BoundaryRule(
        "inference-service-integration",
        _pattern(
            r"(?:\binference",
            r"[\s_.-]+hub\b|\binference-api\.nvidia\.(?:com|net)\b|\bnv_",
            r"inference\b)",
            flags=re.IGNORECASE,
        ),
        "private inference product integration",
    ),
    BoundaryRule(
        "execution-service-integration",
        _pattern(
            r"(?:\bASTRA_[A-Z0-9_]+\b|\bastra",
            r"[-_. ](?:api|client|eval|execution|sandbox|skill[-_. ]eval)\b|\bastra",
            r"[\s_.-]+harbor[\s_.-]+hub\b)",
            flags=re.IGNORECASE,
        ),
        "private execution integration",
    ),
    BoundaryRule(
        "deployment-integration",
        _pattern(
            r"(?:\bopenshift_(?:api|cluster|namespace|project|server|url)\b|\bopenshift",
            r"[-_. ](?:api|cluster|deployment|namespace|project)\b|\b(?:corporate|corp)",
            r"[-_. ]deploy(?:ment)?(?:[-_. ]hook)?\b)",
            flags=re.IGNORECASE,
        ),
        "private deployment integration",
    ),
    BoundaryRule(
        "active-internal-token",
        _pattern(r"\b(?:open", r"shift_", r"token|harbor_viewer_", r"token)\b", flags=re.IGNORECASE),
        "private service token",
    ),
    BoundaryRule(
        "internal-observability",
        _pattern(r"\binternal[-_ ]telemetry(?:[-_ ](?:endpoint|url|token))?\b", flags=re.IGNORECASE),
        "private observability integration",
    ),
    BoundaryRule(
        "internal-runtime-dependency",
        _pattern(r"\b(?:(?:py)?mil", r"vus|sandbox[-_]k8s)\b", flags=re.IGNORECASE),
        "private runtime dependency",
    ),
    BoundaryRule(
        "internal-product-name",
        _pattern(r"\b(?:nv[-_ ]?(?:carps|aces|base)|ipp", r"bot)\b", flags=re.IGNORECASE),
        "internal product name",
    ),
    BoundaryRule(
        "harbor-upload-integration",
        _pattern(
            r"(?:\bharbor[-_. ]viewer[-_. ]upload\b|https?://[^\s/]*harbor",
            r"[-.]viewer[^\s/]*)",
            flags=re.IGNORECASE,
        ),
        "private Harbor upload integration",
    ),
)

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<name>[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION|AUTH_HEADER|COOKIE|SESSION)"
    r"[A-Z0-9_]*)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://)[^/@\s]+@")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|token|secret|password|authorization|auth[_-]?header|cookie|session)=)[^&\s]+"
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)((?:\"?authorization\"?|auth[_-]?header)\s*[:=]\s*[\"']?(?:bearer|basic)\s+)[^\"'\s,;}]+"
)
_MAX_EXCERPT_CHARS = 180


class _ReconstructedCandidate(NamedTuple):
    value: str
    active_token_exempt: bool = False


class _StaticReconstructionLimit(ValueError):
    """A bounded static reconstruction exceeded its safety budget."""


class _ShellDecodeError(ValueError):
    """A Bash ANSI-C literal is malformed or uses an unsupported escape."""


class _ShellDecodeLimit(_ShellDecodeError):
    """Decoded Bash ANSI-C literals exceeded the safety budget."""


def _redacted_match_excerpt(match: re.Match[str]) -> str:
    excerpt = _SECRET_ASSIGNMENT_RE.sub(r"\g<name>\g<separator><redacted>", match.group(0).strip())
    excerpt = _URL_CREDENTIAL_RE.sub(r"\1<redacted>@", excerpt)
    excerpt = _QUERY_SECRET_RE.sub(r"\1<redacted>", excerpt)
    excerpt = _AUTHORIZATION_RE.sub(r"\1<redacted>", excerpt)
    excerpt = "".join(character if character.isprintable() else "?" for character in excerpt)
    if len(excerpt) > _MAX_EXCERPT_CHARS:
        excerpt = f"{excerpt[: _MAX_EXCERPT_CHARS - 3]}..."
    return excerpt


def _bounded_join(parts: Iterable[str], separator: str = "") -> str | None:
    values = tuple(parts)
    size = sum(len(value) for value in values) + max(0, len(values) - 1) * len(separator)
    if size > _MAX_STATIC_STRING_CHARS:
        raise _StaticReconstructionLimit
    return separator.join(values)


def _formatted_constant_string(node: ast.Call, *, depth: int) -> str | None:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "format" or node.keywords:
        return None
    template = _static_string(node.func.value, depth=depth + 1)
    arguments = [_static_string(argument, depth=depth + 1) for argument in node.args]
    if template is None or any(argument is None for argument in arguments):
        return None
    values = [argument for argument in arguments if argument is not None]
    pieces: list[str] = []
    auto_index = 0
    try:
        parsed = string.Formatter().parse(template)
        for literal, field_name, format_spec, conversion in parsed:
            pieces.append(literal)
            if field_name is None:
                continue
            if format_spec not in {"", "s"} or conversion not in {None, "s"}:
                return None
            if field_name == "":
                index = auto_index
                auto_index += 1
            elif field_name.isdecimal():
                index = int(field_name)
            else:
                return None
            if index >= len(values):
                return None
            pieces.append(values[index])
    except ValueError:
        return None
    return _bounded_join(pieces)


def _static_string(node: ast.AST, *, depth: int = 0) -> str | None:
    if depth > _MAX_STATIC_STRING_DEPTH:
        raise _StaticReconstructionLimit
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if len(node.value) > _MAX_STATIC_STRING_CHARS:
            raise _StaticReconstructionLimit
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, depth=depth + 1)
        right = _static_string(node.right, depth=depth + 1)
        return _bounded_join((left, right)) if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        values: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                value = part.value
            elif isinstance(part, ast.FormattedValue) and part.conversion in {-1, ord("s")}:
                value = _static_string(part.value, depth=depth + 1)
                format_spec = "" if part.format_spec is None else _static_string(part.format_spec, depth=depth + 1)
                if format_spec not in {"", "s"}:
                    return None
            else:
                return None
            if value is None:
                return None
            values.append(value)
        return _bounded_join(values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "join" and len(node.args) == 1 and not node.keywords:
            separator = _static_string(node.func.value, depth=depth + 1)
            sequence = node.args[0]
            if separator is None or not isinstance(sequence, (ast.List, ast.Tuple)):
                return None
            values = [_static_string(item, depth=depth + 1) for item in sequence.elts]
            if any(value is None for value in values):
                return None
            return _bounded_join((value for value in values if value is not None), separator)
        return _formatted_constant_string(node, depth=depth)
    return None


def _is_re_compile_call(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        and node.func.attr == "compile"
        and node.args
        and _static_string(node.args[0]) is not None
    )


def _character_column(line: str, byte_column: int) -> int:
    return len(line.encode("utf-8")[:byte_column].decode("utf-8"))


def _source_ranges_by_line(node: ast.AST, lines: list[str]) -> dict[int, list[tuple[int, int]]]:
    if not hasattr(node, "lineno") or node.end_lineno is None or node.end_col_offset is None:
        return {}
    ranges: dict[int, list[tuple[int, int]]] = {}
    for line_number in range(node.lineno, node.end_lineno + 1):
        line = lines[line_number - 1]
        start = _character_column(line, node.col_offset) if line_number == node.lineno else 0
        end = _character_column(line, node.end_col_offset) if line_number == node.end_lineno else len(line)
        ranges.setdefault(line_number, []).append((start, end))
    return ranges


def _python_reconstructions(
    text: str,
) -> tuple[dict[int, set[_ReconstructedCandidate]], dict[int, list[tuple[int, int]]], set[int]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}, {}, set()
    except RecursionError:
        return {}, {}, {1}
    lines = text.splitlines()
    defensive_node_ids: set[int] = set()
    defensive_ranges: dict[int, list[tuple[int, int]]] = {}
    limit_lines: set[int] = set()
    for node in ast.walk(tree):
        try:
            is_re_compile = _is_re_compile_call(node)
        except _StaticReconstructionLimit:
            if hasattr(node, "lineno"):
                limit_lines.add(node.lineno)
            continue
        if not is_re_compile:
            continue
        pattern = node.args[0]
        defensive_node_ids.update(id(descendant) for descendant in ast.walk(pattern))
        for line_number, ranges in _source_ranges_by_line(pattern, lines).items():
            defensive_ranges.setdefault(line_number, []).extend(ranges)

    reconstructed: dict[int, set[_ReconstructedCandidate]] = {}
    for node in ast.walk(tree):
        try:
            value = _static_string(node)
        except _StaticReconstructionLimit:
            if hasattr(node, "lineno"):
                limit_lines.add(node.lineno)
            continue
        if value is not None and hasattr(node, "lineno"):
            reconstructed.setdefault(node.lineno, set()).add(
                _ReconstructedCandidate(value, active_token_exempt=id(node) in defensive_node_ids)
            )
    return reconstructed, defensive_ranges, limit_lines


def _decoded_shell_codepoint(value: int) -> str:
    if value == 0 or value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
        raise _ShellDecodeError
    return chr(value)


def _decode_bash_escape(value: str, index: int) -> tuple[str, int]:
    if index >= len(value):
        raise _ShellDecodeError
    escape = value[index]
    if escape in _BASH_COMMON_ESCAPES:
        return _BASH_COMMON_ESCAPES[escape], index + 1
    if escape == "\n":
        return "", index + 1
    if escape == "x":
        end = index + 1
        while end < len(value) and end < index + 3 and value[end] in _HEX_DIGITS:
            end += 1
        if end == index + 1:
            raise _ShellDecodeError
        return _decoded_shell_codepoint(int(value[index + 1 : end], 16)), end
    if escape in {"u", "U"}:
        width = 4 if escape == "u" else 8
        end = index + 1 + width
        digits = value[index + 1 : end]
        if len(digits) != width or any(digit not in _HEX_DIGITS for digit in digits):
            raise _ShellDecodeError
        return _decoded_shell_codepoint(int(digits, 16)), end
    if escape in _OCTAL_DIGITS:
        end = index + 1
        while end < len(value) and end < index + 3 and value[end] in _OCTAL_DIGITS:
            end += 1
        return _decoded_shell_codepoint(int(value[index:end], 8) & 0xFF), end
    if escape == "c":
        if index + 1 >= len(value):
            raise _ShellDecodeError
        control = value[index + 1]
        codepoint = 0x7F if control == "?" else ord(control.upper()) & 0x1F
        return _decoded_shell_codepoint(codepoint), index + 2
    raise _ShellDecodeError


def _is_backslash_escaped(value: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and value[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _decode_bash_ansi_c_literals(value: str) -> str:
    pieces: list[str] = []
    cursor = 0
    index = 0
    decoded_chars = 0
    while index < len(value):
        if not value.startswith(_BASH_ANSI_C_PREFIX, index) or _is_backslash_escaped(value, index):
            index += 1
            continue
        pieces.append(value[cursor:index])
        index += 2
        decoded: list[str] = []
        while index < len(value) and value[index] != "'":
            if value[index] == "\\":
                fragment, index = _decode_bash_escape(value, index + 1)
            else:
                fragment = value[index]
                index += 1
            decoded_chars += len(fragment)
            if decoded_chars > _MAX_SHELL_DECODE_CHARS:
                raise _ShellDecodeLimit
            decoded.append(fragment)
        if index >= len(value):
            raise _ShellDecodeError
        pieces.append(shlex.quote("".join(decoded)))
        index += 1
        cursor = index
    pieces.append(value[cursor:])
    return "".join(pieces)


def _shell_normalized_value(value: str) -> str | None:
    bash_normalized = _decode_bash_ansi_c_literals(value)
    bash_normalized = _BASH_LOCALE_QUOTE_RE.sub("", bash_normalized)
    try:
        tokens = shlex.split(bash_normalized, comments=True, posix=True)
    except ValueError:
        return None
    return " ".join(tokens) if tokens else None


def _shell_reconstruction(path: str, line: str) -> str | None:
    archive_member = path.split("!", 1)[1] if "!" in path else path
    member = PurePosixPath(archive_member)
    if member.suffix.lower() in _PYTHON_SUFFIXES:
        return None
    return _shell_normalized_value(line)


def _match_is_inside_ranges(match: re.Match[str], ranges: Iterable[tuple[int, int]]) -> bool:
    start, end = match.span()
    return any(range_start <= start and end <= range_end for range_start, range_end in ranges)


def _shell_failure_finding(path: str, line_number: int, error: _ShellDecodeError) -> BoundaryFinding:
    if isinstance(error, _ShellDecodeLimit):
        return BoundaryFinding(
            path,
            "scanner-resource-limit",
            line_number,
            "shell string reconstruction exceeded the safety limit",
            "<shell reconstruction limit>",
        )
    return BoundaryFinding(
        path,
        "scanner-shell-decode-error",
        line_number,
        "Bash ANSI-C string reconstruction failed closed",
        "<shell reconstruction error>",
    )


def scan_text(path: str, text: str) -> list[BoundaryFinding]:
    """Return every denied rule match in decoded text."""
    reconstructed, defensive_ranges, limit_lines = _python_reconstructions(text)
    findings = [
        BoundaryFinding(
            path,
            "scanner-resource-limit",
            line_number,
            "static string reconstruction exceeded the safety limit",
            "<static reconstruction limit>",
        )
        for line_number in sorted(limit_lines)
    ]
    for line_number, line in enumerate(text.splitlines(), start=1):
        candidates = [_ReconstructedCandidate(line)]
        reconstructed_candidates = tuple(reconstructed.get(line_number, ()))
        candidates.extend(reconstructed_candidates)
        shell_failures: dict[str, BoundaryFinding] = {}
        for candidate in reconstructed_candidates:
            try:
                shell_value = _shell_normalized_value(candidate.value)
            except _ShellDecodeError as exc:
                failure = _shell_failure_finding(path, line_number, exc)
                shell_failures.setdefault(failure.rule, failure)
                continue
            if shell_value is not None and shell_value != candidate.value:
                candidates.append(_ReconstructedCandidate(shell_value, candidate.active_token_exempt))
        try:
            shell_value = _shell_reconstruction(path, line)
        except _ShellDecodeError as exc:
            failure = _shell_failure_finding(path, line_number, exc)
            shell_failures.setdefault(failure.rule, failure)
            shell_value = None
        findings.extend(shell_failures.values())
        if shell_value is not None:
            candidates.append(_ReconstructedCandidate(shell_value))
        for rule in DENY_RULES:
            selected_match: re.Match[str] | None = None
            for index, candidate in enumerate(candidates):
                if rule.rule_id == "active-internal-token" and candidate.active_token_exempt:
                    continue
                for match in rule.pattern.finditer(candidate.value):
                    if (
                        rule.rule_id == "active-internal-token"
                        and index == 0
                        and _match_is_inside_ranges(match, defensive_ranges.get(line_number, ()))
                    ):
                        continue
                    selected_match = match
                    break
                if selected_match is not None:
                    break
            if selected_match is not None:
                findings.append(
                    BoundaryFinding(
                        path,
                        rule.rule_id,
                        line_number,
                        rule.message,
                        _redacted_match_excerpt(selected_match),
                    )
                )
    return findings


def load_allowlist(path: Path | None) -> tuple[BoundaryAllowance, ...]:
    """Load a strict line-specific allowlist; unknown metadata fails closed."""
    if path is None:
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load OSS boundary allowlist {path}: {exc}") from exc

    if not isinstance(payload, dict) or set(payload) != {"version", "allowlist"} or payload.get("version") != 1:
        raise ValueError("OSS boundary allowlist must contain only version=1 and allowlist")
    entries = payload.get("allowlist")
    if not isinstance(entries, list):
        raise ValueError("OSS boundary allowlist must be a list")

    allowances: list[BoundaryAllowance] = []
    required = {"path", "rule", "line", "reason", "expires", "review_owner"}
    known_rules = {rule.rule_id for rule in DENY_RULES}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(
                f"allowlist entry {index} must contain exactly path, rule, line, reason, expires, and review_owner"
            )
        if (
            not isinstance(entry["path"], str)
            or not entry["path"].strip()
            or not isinstance(entry["rule"], str)
            or entry["rule"] not in known_rules
            or not isinstance(entry["line"], int)
            or isinstance(entry["line"], bool)
            or entry["line"] < 1
            or not isinstance(entry["reason"], str)
            or not entry["reason"].strip()
            or not isinstance(entry["expires"], str)
            or not entry["expires"].strip()
            or not isinstance(entry["review_owner"], str)
            or not entry["review_owner"].strip()
        ):
            raise ValueError(
                f"allowlist entry {index} must contain valid path, rule, line, reason, expires, and review_owner values"
            )
        allow_path = PurePosixPath(entry["path"])
        if (
            allow_path.is_absolute()
            or ".." in allow_path.parts
            or not allow_path.parts
            or allow_path.parts[0] != "tests"
        ):
            raise ValueError(f"allowlist entry {index} is restricted to exact negative test paths")
        try:
            expires = date.fromisoformat(entry["expires"])
        except ValueError as exc:
            raise ValueError(f"allowlist entry {index} has an invalid expiry date") from exc
        if expires < datetime.now(UTC).date():
            raise ValueError(f"allowlist entry {index} expired on {expires.isoformat()}")
        allowances.append(
            BoundaryAllowance(
                path=allow_path.as_posix(),
                rule=entry["rule"],
                line=entry["line"],
                reason=entry["reason"].strip(),
                expires=expires,
                review_owner=entry["review_owner"].strip(),
            )
        )
    return tuple(allowances)


def _is_scanned_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if not path.parts or path.parts[0] == ".git":
        return False
    return path.name not in _BINARY_ROOT_FILES and path.suffix.lower() not in _BINARY_SUFFIXES


def _tracked_paths(root: Path) -> list[str] | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return [path for path in result.stdout.decode("utf-8", errors="surrogateescape").split("\0") if path]


def _repository_paths(root: Path) -> Iterator[tuple[str, Path]]:
    tracked = _tracked_paths(root)
    if tracked is None:
        candidates = (path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    else:
        candidates = iter(tracked)
    for relative in sorted(candidates):
        if _is_scanned_path(relative):
            yield relative, root / relative


def _decode_text(data: bytes, *, path: str) -> str:
    if b"\0" in data:
        raise ValueError(f"release file {path} is not UTF-8 text")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"release file {path} is not UTF-8 text") from exc


def _is_archive_text(path: str) -> bool:
    member = PurePosixPath(path)
    return member.name not in _BINARY_ROOT_FILES and member.suffix.lower() not in _BINARY_SUFFIXES


def _without_allowances(
    findings: Iterable[BoundaryFinding],
    allowances: Iterable[BoundaryAllowance],
    *,
    archive_source_paths: dict[str, str] | None = None,
) -> list[BoundaryFinding]:
    entries = tuple(allowances)
    source_paths = archive_source_paths or {}

    def is_allowed(finding: BoundaryFinding) -> bool:
        finding_path = finding.path
        source_path = source_paths.get(finding_path)
        for entry in entries:
            if finding.rule != entry.rule or finding.line != entry.line:
                continue
            if entry.path in (finding_path, source_path):
                return True
        return False

    return [finding for finding in findings if not is_allowed(finding)]


def scan_repository(root: Path, *, allowlist_path: Path | None = None) -> list[BoundaryFinding]:
    """Scan the tracked public release surface, or the equivalent tree without Git metadata."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    findings: list[BoundaryFinding] = []
    for relative, path in _repository_paths(root):
        try:
            text = _decode_text(path.read_bytes(), path=relative)
        except OSError as exc:
            raise ValueError(f"could not read release file {relative}: {exc}") from exc
        findings.extend(scan_text(relative, text))
    return _without_allowances(findings, load_allowlist(allowlist_path))


def _validated_archive_member_path(member_name: str, *, expected_root: str | None = None) -> str:
    normalized = member_name.rstrip("/")
    raw_parts = normalized.split("/")
    member = PurePosixPath(normalized)
    if (
        not normalized
        or member.is_absolute()
        or "\\" in normalized
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(f"unsafe archive member path {member_name}")
    if expected_root is None:
        return member.as_posix()
    if raw_parts[0] != expected_root:
        raise ValueError(f"archive member {member_name} is outside expected source distribution root {expected_root}")
    return PurePosixPath(*raw_parts[1:]).as_posix() if len(raw_parts) > 1 else ""


def _source_distribution_root(path: Path) -> str:
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar"):
        if path.name.endswith(suffix):
            root = path.name[: -len(suffix)]
            if root:
                return root
    raise ValueError(f"unsupported source distribution filename {path.name}")


def _zip_members(path: Path) -> Iterator[tuple[str, str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > _MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"archive exceeds the member count limit of {_MAX_ARCHIVE_MEMBERS}")
        total_size = sum(info.file_size for info in infos if not info.is_dir())
        if total_size > _MAX_ARCHIVE_BYTES:
            raise ValueError(f"archive exceeds the aggregate decompressed size limit of {_MAX_ARCHIVE_BYTES} bytes")
        supported_compression = {
            zipfile.ZIP_STORED,
            zipfile.ZIP_DEFLATED,
            zipfile.ZIP_BZIP2,
            zipfile.ZIP_LZMA,
        }
        optional_zstandard = getattr(zipfile, "ZIP_ZSTANDARD", None)
        if optional_zstandard is not None:
            supported_compression.add(optional_zstandard)
        for info in infos:
            source_path = _validated_archive_member_path(info.filename)
            if info.flag_bits & 0x1:
                raise ValueError(f"archive contains encrypted ZIP member {info.filename}")
            if not info.is_dir() and info.compress_type not in supported_compression:
                raise ValueError(f"archive contains unsupported ZIP member {info.filename}")
            if info.is_dir() or not _is_archive_text(info.filename):
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(f"archive member {info.filename} exceeds the scan limit")
            try:
                data = archive.read(info)
            except (NotImplementedError, RuntimeError) as exc:
                raise ValueError(f"archive contains unsupported ZIP member {info.filename}") from exc
            yield info.filename, source_path, data


def _tar_members(path: Path) -> Iterator[tuple[str, str, bytes]]:
    expected_root = _source_distribution_root(path)
    with tarfile.open(path, "r:*") as archive:
        member_count = 0
        total_size = 0
        for info in archive:
            source_path = _validated_archive_member_path(info.name, expected_root=expected_root)
            member_count += 1
            if member_count > _MAX_ARCHIVE_MEMBERS:
                raise ValueError(f"archive exceeds the member count limit of {_MAX_ARCHIVE_MEMBERS}")
            if info.isfile():
                total_size += info.size
                if total_size > _MAX_ARCHIVE_BYTES:
                    raise ValueError(
                        f"archive exceeds the aggregate decompressed size limit of {_MAX_ARCHIVE_BYTES} bytes"
                    )
            if not info.isfile() or not _is_archive_text(info.name):
                continue
            if info.size > _MAX_MEMBER_BYTES:
                raise ValueError(f"archive member {info.name} exceeds the scan limit")
            member = archive.extractfile(info)
            if member is not None:
                yield info.name, source_path, member.read()


def scan_archive(path: Path, *, allowlist_path: Path | None = None) -> list[BoundaryFinding]:
    """Scan a wheel or source distribution in memory without extracting it."""
    path = path.resolve()
    try:
        members = _zip_members(path) if zipfile.is_zipfile(path) else _tar_members(path)
        findings: list[BoundaryFinding] = []
        archive_source_paths: dict[str, str] = {}
        for member_name, source_path, data in members:
            finding_path = f"{path}!{member_name}"
            archive_source_paths[finding_path] = source_path
            text = _decode_text(data, path=finding_path)
            findings.extend(scan_text(finding_path, text))
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not scan distribution {path}: {exc}") from exc
    return _without_allowances(
        findings,
        load_allowlist(allowlist_path),
        archive_source_paths=archive_source_paths,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root to scan")
    parser.add_argument("--allowlist", type=Path, help="strict JSON allowlist")
    parser.add_argument("--archive", type=Path, action="append", default=[], help="wheel or sdist to scan")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        findings = scan_repository(args.root, allowlist_path=args.allowlist)
        for archive in args.archive:
            findings.extend(scan_archive(archive, allowlist_path=args.allowlist))
    except ValueError as exc:
        print(f"OSS boundary scan failed: {exc}", file=sys.stderr)
        return 2

    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.rule}: {finding.message}: {finding.excerpt}")
    if findings:
        print(f"OSS boundary scan found {len(findings)} denied release integration(s).", file=sys.stderr)
        return 1
    print("OSS boundary scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
