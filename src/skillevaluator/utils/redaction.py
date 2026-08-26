# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential redaction helpers for logs and generated artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import unquote

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
_PLURAL_SECRET_KEY_PARTS = {
    "auths",
    "authorizations",
    "bearers",
    "credentials",
    "passwords",
    "secrets",
    "tokens",
}
_TOKEN_COUNT_KEYS = {
    "cached_tokens",
    "completion_tokens",
    "expected_max_tokens",
    "frontmatter_tokens",
    "input_tokens",
    "instructions_tokens",
    "n_input_tokens",
    "n_cache_tokens",
    "n_output_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_output_tokens",
    "last_token_usage",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
    "recommended_max_tokens",
    "token_count",
    "tokens",
    "total_cached_tokens",
    "total_completion_tokens",
    "total_prompt_tokens",
    "total_tokens",
}
_SENSITIVE_KEY_PATTERN = (
    r"[a-z0-9_.-]*(?:api[_-]?key|secret|password|credential|authorization|bearer|token|"
    r"access[_-]?key|session[_-]?token|private[_-]?key)[a-z0-9_.-]*"
)
_AUTH_HEADER_RE = re.compile(r"(?im)\b(?P<key>(?:proxy-)?authorization)\s*:\s*(?P<scheme>[A-Za-z]+)\s+[^\r\n]+")
_CREDENTIAL_URI_USERINFO_RE = re.compile(r"(?i)(?P<scheme>[a-z][a-z0-9+.-]{0,31}://)(?P<userinfo>[^\s/?#]+@)")
_SENSITIVE_QUOTED_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>[:=])\s*(?P<quote>[\"'])(?P<value>[^\r\n]*?)(?P=quote)"
)
_SENSITIVE_COLON_ASSIGNMENT_RE = re.compile(rf"(?im)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>:)\s*[^\r\n,;]+")
_SENSITIVE_EQUALS_ASSIGNMENT_RE = re.compile(rf"(?i)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>=)\s*[^\s\"',;]+")
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
    (re.compile(r"(?<![A-Za-z0-9])sk-[a-zA-Z0-9_-]{8,}"), "sk-<redacted>"),
    (re.compile(r"nvapi-[a-zA-Z0-9_-]{8,}"), "nvapi-<redacted>"),
    (re.compile(r"crsr_[a-f0-9]{16,}"), "crsr_<redacted>"),
    (re.compile(r"sha256~[A-Za-z0-9._~-]+"), "sha256~<redacted>"),
    # GitHub's p/o/u/r families retain the 36-character opaque body. The s
    # family also has a variable-length ``ghs_APPID_JWT`` stateless format.
    (
        re.compile(r"(?i)gh[pour]_[A-Za-z0-9]{36}"),
        "github-token-<redacted>",
    ),
    (
        re.compile(r"(?i)ghs_[A-Za-z0-9.\-_]{36,}"),
        "github-token-<redacted>",
    ),
    (re.compile(r"(?i)github_pat_[A-Za-z0-9_]{20,}"), "github-token-<redacted>"),
    (re.compile(r"(?i)xox[baprs]-[A-Za-z0-9-]{10,}"), "slack-token-<redacted>"),
    (re.compile(r"AIza[A-Za-z0-9_-]{20,}"), "google-api-key-<redacted>"),
    (re.compile(r"(?i)glpat-[A-Za-z0-9_-]{20,}"), "gitlab-token-<redacted>"),
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
    return bool(parts & (_SECRET_KEY_PARTS | _PLURAL_SECRET_KEY_PARTS))


def _redact_sensitive_assignment(match: re.Match[str]) -> str:
    key = match.group("key")
    if not is_sensitive_key(key):
        return match.group(0)
    return f"{key}{match.group('sep')}<redacted>"


def _redact_auth_header(match: re.Match[str]) -> str:
    return f"{match.group('key')}: {match.group('scheme')} <redacted>"


def credential_uri_secret_values(value: str, *, allow_schemeless: bool = False) -> set[str]:
    """Return raw and decoded credential components from one URI authority."""
    raw = str(value or "")
    if not raw or "@" not in raw:
        return set()
    if "://" in raw:
        _scheme, _separator, remainder = raw.partition("://")
    elif allow_schemeless:
        remainder = raw
    else:
        return set()
    authority_end = min(
        (index for delimiter in "/?#" if (index := remainder.find(delimiter)) >= 0),
        default=len(remainder),
    )
    authority = remainder[:authority_end]
    if "@" not in authority:
        return set()
    userinfo = authority.rsplit("@", 1)[0]
    if not userinfo:
        return set()

    protected = {raw, userinfo}
    decoded_userinfo = unquote(userinfo)
    protected.add(decoded_userinfo)
    for candidate in (userinfo, decoded_userinfo):
        username, separator, password = candidate.partition(":")
        if username:
            protected.add(username)
        if separator and password:
            protected.add(password)
    return {item for item in protected if item}


def redact_sensitive_text(value: str, *, max_len: int | None = None) -> str:
    """Best-effort masking for credentials before writing logs or artifacts."""
    out = value
    # Remove multiline private-key material before any single-line assignment
    # or header rule can consume only its BEGIN delimiter and orphan the body.
    for pattern, replacement in _PEM_REDACTIONS:
        out = pattern.sub(replacement, out)
    if "://" in out:
        out = _CREDENTIAL_URI_USERINFO_RE.sub(r"\g<scheme><redacted>@", out)
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


def contains_credential_value(value: object) -> bool:
    """Return whether text contains credential material, including embedded tokens."""
    if value is None:
        return False
    text = str(value)
    return bool(text) and redact_sensitive_text(text) != text


def _is_finite_token_count(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _is_token_count_value(value: Any) -> bool:
    if _is_finite_token_count(value):
        return True
    if not isinstance(value, Mapping) or not value:
        return False
    return all(
        _normalized_key_parts(str(key))[0] in _TOKEN_COUNT_KEYS and _is_finite_token_count(item)
        for key, item in value.items()
    )


def redact_sensitive_data(value: Any, *, parent_key: str = "", max_str_len: int | None = None) -> Any:
    """Recursively redact structured data using secret-looking key names."""
    normalized_parent, _parts = _normalized_key_parts(parent_key)
    if normalized_parent in _TOKEN_COUNT_KEYS and not _is_token_count_value(value):
        return "<redacted>"
    if is_sensitive_key(parent_key):
        return "<redacted>"
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            # A credential can itself be used as a mapping key.  Redacting the
            # corresponding value is insufficient, and replacing the key can
            # collapse distinct entries.  Drop such entries collision-safely.
            if redact_sensitive_text(raw_key, max_len=max_str_len) != raw_key:
                continue
            redacted[raw_key] = redact_sensitive_data(item, parent_key=raw_key, max_str_len=max_str_len)
        return redacted
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return [redact_sensitive_data(item, parent_key=parent_key, max_str_len=max_str_len) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value, max_len=max_str_len)
    return value
