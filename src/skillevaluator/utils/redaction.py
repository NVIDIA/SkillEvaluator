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
_SENSITIVE_KEY_PATTERN = (
    r"[a-z0-9_.-]*(?:api[_-]?key|secret|password|credential|authorization|bearer|token|"
    r"access[_-]?key|session[_-]?token|private[_-]?key)[a-z0-9_.-]*"
)
_AUTH_HEADER_RE = re.compile(r"(?im)\b(?P<key>(?:proxy-)?authorization)\s*:\s*(?P<scheme>[A-Za-z]+)\s+[^\r\n]+")
_SENSITIVE_QUOTED_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>[:=])\s*(?P<quote>[\"'])(?P<value>[^\r\n]*?)(?P=quote)"
)
_SENSITIVE_COLON_ASSIGNMENT_RE = re.compile(rf"(?im)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>:)\s*[^\r\n,;]+")
_SENSITIVE_EQUALS_ASSIGNMENT_RE = re.compile(rf"(?i)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>=)\s*[^\s\"',;]+")
_REDACTIONS = (
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


def _is_sensitive_key(key: str) -> bool:
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
    if not _is_sensitive_key(key):
        return match.group(0)
    return f"{key}{match.group('sep')}<redacted>"


def _redact_auth_header(match: re.Match[str]) -> str:
    return f"{match.group('key')}: {match.group('scheme')} <redacted>"


def redact_sensitive_text(value: str, *, max_len: int | None = None) -> str:
    """Best-effort masking for credentials before writing logs or artifacts."""
    out = value
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
    if _is_sensitive_key(parent_key):
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
