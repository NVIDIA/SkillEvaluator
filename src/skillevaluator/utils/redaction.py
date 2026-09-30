# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential redaction helpers for logs and generated artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_SECRET_KEY_PARTS = {
    "auth",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "key",
    "password",
    "private",
    "secret",
    "token",
}
_TOKEN_COUNT_KEYS = {
    "completion_tokens",
    "input_tokens",
    "n_input_tokens",
    "n_output_tokens",
    "output_tokens",
    "prompt_tokens",
    "last_token_usage",
    "token_count",
    "tokens",
    "total_tokens",
}
# Assignment keys are runs of [a-z0-9_.-] (the patterns below are case-insensitive).
# A key used to be ``\b[a-z0-9_.-]*(?:word)[a-z0-9_.-]*``, which the regex engine
# retried at every word boundary inside a run and for every way of splitting the
# run around a sensitive word, so one substitution took cubic time on text such as
# ``"token-" * n``. The pattern below captures the same ``key``: the part of a
# maximal key-character run that starts at the run's first word boundary and
# contains a sensitive word. It only starts matching at the beginning of a run,
# captures the characters before that first boundary (the ``--`` of ``--api-key``)
# as ``lead`` so they are written back unchanged, and never backtracks into the
# key, so each substitution is linear in the input length.
_SENSITIVE_KEY_CHAR = r"[a-z0-9_.-]"
_SENSITIVE_KEY_WORD = (
    r"(?:api[_-]?key|secret|password|credential|authorization|bearer|token|"
    r"access[_-]?key|session[_-]?token|private[_-]?key)"
)
_SENSITIVE_KEY_PATTERN = (
    rf"(?<!{_SENSITIVE_KEY_CHAR})"
    # ``lead`` is empty when the run starts at a word boundary. Otherwise the first
    # boundary follows the run's leading [.-] characters ("--api-key"), or its
    # leading [a-z0-9_] characters when the run is glued to a word character that
    # is not a key character ("é" in "étoken.secret").
    r"(?P<lead>\b|(?<=\w)[a-z0-9_]++(?=[.-])|(?<!\w)[.-]++(?=[a-z0-9_]))"
    rf"(?P<key>(?>{_SENSITIVE_KEY_CHAR}*?{_SENSITIVE_KEY_WORD}){_SENSITIVE_KEY_CHAR}*+)"
)
_AUTH_HEADER_RE = re.compile(r"(?im)\b(?P<key>(?:proxy-)?authorization)\s*:\s*(?P<scheme>[A-Za-z]+)\s+[^\r\n]+")
_SENSITIVE_QUOTED_ASSIGNMENT_RE = re.compile(
    rf"(?i){_SENSITIVE_KEY_PATTERN}\s*+(?P<sep>[:=])\s*+(?:\"[^\"\r\n]*+\"|'[^'\r\n]*+')"
)
# The whitespace after ":" must stay backtrackable: in "token:  ," the value is the last blank.
_SENSITIVE_COLON_ASSIGNMENT_RE = re.compile(rf"(?im){_SENSITIVE_KEY_PATTERN}\s*+(?P<sep>:)\s*[^\r\n,;]+")
_SENSITIVE_EQUALS_ASSIGNMENT_RE = re.compile(rf"(?i){_SENSITIVE_KEY_PATTERN}\s*+(?P<sep>=)\s*+[^\s\"',;]+")
_PRIVATE_KEY_LABEL = r"(?:[A-Z0-9][A-Z0-9-]* )*PRIVATE KEY(?: [A-Z0-9][A-Z0-9-]*)*"
_PEM_REDACTIONS = (
    (
        re.compile(
            rf"-----BEGIN (?P<private_key_label>{_PRIVATE_KEY_LABEL})-----"
            r"(?:(?!-----BEGIN |-----END )[\s\S])*?"
            r"-----END (?P=private_key_label)-----"
        ),
        "private-key-<redacted>",
    ),
    (
        re.compile(
            rf"-----BEGIN {_PRIVATE_KEY_LABEL}-----"
            r"(?:(?!-----BEGIN |-----END )[\s\S])*?"
            r"-----END (?:[A-Z0-9][A-Z0-9-]* )*[A-Z0-9][A-Z0-9-]*-----"
        ),
        "private-key-<redacted>",
    ),
    (
        re.compile(
            rf"-----BEGIN {_PRIVATE_KEY_LABEL}-----"
            r"(?![\s\S]*-----END )[\s\S]*\Z"
        ),
        "private-key-<redacted>",
    ),
)
_REDACTIONS = (
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "jwt-<redacted>",
    ),
    (re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"), "aws-access-key-<redacted>"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])sk-[a-zA-Z0-9_-]{8,}"), "sk-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])nvapi-[a-zA-Z0-9_-]{8,}"), "nvapi-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])crsr_[a-f0-9]{16,}"), "crsr_<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])sha256~[A-Za-z0-9._~-]+"), "sha256~<redacted>"),
)


def _normalized_key_parts(key: str) -> tuple[str, set[str]]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key or ""))
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", camel_split).strip("_").lower()
    parts = {part for part in normalized.split("_") if part}
    return normalized, parts


def is_sensitive_key(key: str) -> bool:
    """Return whether a structured-data key conventionally carries a secret."""
    normalized, parts = _normalized_key_parts(key)
    if normalized in _TOKEN_COUNT_KEYS:
        return False
    compact = normalized.replace("_", "")
    if "api_key" in normalized or "apikey" in compact:
        return True
    if "accesskey" in compact or "privatekey" in compact or "sessiontoken" in compact:
        return True
    if "token" in parts or compact.endswith("token"):
        return True
    return bool(parts & _SECRET_KEY_PARTS)


def _redact_sensitive_assignment(match: re.Match[str]) -> str:
    key = match.group("key")
    if not is_sensitive_key(key):
        return match.group(0)
    return f"{match.group('lead')}{key}{match.group('sep')}<redacted>"


def _redact_auth_header(match: re.Match[str]) -> str:
    return f"{match.group('key')}: {match.group('scheme')} <redacted>"


def redact_sensitive_text(value: str, *, max_len: int | None = None) -> str:
    """Best-effort masking for credentials before writing logs or artifacts."""
    out = value
    # Remove multiline private-key material before any single-line assignment
    # or header rule can consume only its BEGIN delimiter and orphan the body.
    for pattern, replacement in _PEM_REDACTIONS:
        out = pattern.sub(replacement, out)
    out = _AUTH_HEADER_RE.sub(_redact_auth_header, out)
    out = _SENSITIVE_QUOTED_ASSIGNMENT_RE.sub(_redact_sensitive_assignment, out)
    out = _SENSITIVE_COLON_ASSIGNMENT_RE.sub(_redact_sensitive_assignment, out)
    out = _SENSITIVE_EQUALS_ASSIGNMENT_RE.sub(_redact_sensitive_assignment, out)
    for pattern, replacement in _REDACTIONS:
        out = pattern.sub(replacement, out)
    if max_len is not None and len(out) > max_len:
        if max_len <= 14:
            return out[:max_len]
        return out[: max_len - 14] + "...<truncated>"
    return out


def redact_sensitive_data(value: Any, *, parent_key: str = "", max_str_len: int | None = None) -> Any:
    """Recursively redact structured data using secret-looking key names."""
    if is_sensitive_key(parent_key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(key): redact_sensitive_data(item, parent_key=str(key), max_str_len=max_str_len)
            for key, item in value.items()
        }
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return [redact_sensitive_data(item, parent_key=parent_key, max_str_len=max_str_len) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value, max_len=max_str_len)
    return value
