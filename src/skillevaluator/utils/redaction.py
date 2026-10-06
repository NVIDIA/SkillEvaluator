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
_CREDENTIAL_URI_USERINFO_RE = re.compile(r"(?i)(?P<scheme>[a-z][a-z0-9+.-]{0,31}://)(?P<userinfo>[^\s/?#]+@)")
_SENSITIVE_QUOTED_ASSIGNMENT_RE = re.compile(
    rf"(?i){_SENSITIVE_KEY_PATTERN}\s*+(?P<sep>[:=])\s*+(?:\"[^\"\r\n]*+\"|'[^'\r\n]*+')"
)
# The whitespace after ":" must stay backtrackable: in "token:  ," the value is the last blank.
_SENSITIVE_COLON_ASSIGNMENT_RE = re.compile(rf"(?im){_SENSITIVE_KEY_PATTERN}\s*+(?P<sep>:)\s*[^\r\n,;]+")
_SENSITIVE_EQUALS_ASSIGNMENT_RE = re.compile(rf"(?i){_SENSITIVE_KEY_PATTERN}\s*+(?P<sep>=)\s*+[^\s\"',;]+")
# A PEM label word is letters and digits joined by single "-" (the RFC 7468 label
# shape). It used to be ``[A-Z0-9][A-Z0-9-]*``, which can contain "-----", so the
# label of one "-----BEGIN " header ran on across every later header on the line and
# each header rescanned the rest of the text ("-----BEGIN A" * n was quadratic).
# Trade-off: a header is no longer read as a private-key header when a label word has
# "--" or a trailing "-" ("X- PRIVATE KEY"), or when the label only reaches "PRIVATE
# KEY" by running on into the next delimiter on the same line ("-----BEGIN
# CERTIFICATE----------END RSA PRIVATE KEY-----"). RFC 7468 labels do neither.
_PEM_LABEL_WORD = r"[A-Z0-9]++(?:-[A-Z0-9]++)*+"
# The label is committed at its first "PRIVATE KEY" pair that is followed by a space
# or "-----". The former ``(?:word )*PRIVATE KEY`` retried every later pair and
# rescanned the words after it ("PRIVATE KEY " * n was quadratic), but a later pair
# can only reach the same label end, because the words after the first pair lead to
# the same end.
_PRIVATE_KEY_LABEL = rf"(?>(?:{_PEM_LABEL_WORD} )*?PRIVATE KEY(?= |-----))(?: {_PEM_LABEL_WORD})*+"
_PRIVATE_KEY_HEADER_RE = re.compile(rf"-----BEGIN {_PRIVATE_KEY_LABEL}-----")
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
)
# A JWT used to start at any ``\beyJ``, so in a run of JWT characters such as
# "eyJ-" * n every "-eyJ" was a start, and each start scanned to the end of the run
# looking for ".". Now a match starts only at the beginning of a run. The part of
# the run before its first ``\beyJ`` is captured as ``lead`` and written back
# unchanged, which keeps JWTs glued to a "-" (x-eyJ...) redacted. Later starts in the
# same run are never tried: their first segment reaches the same "." with fewer
# characters, so they could only fail where the first start failed.
_JWT_CHAR = r"[A-Za-z0-9_-]"
_JWT_RE = re.compile(
    rf"(?<!{_JWT_CHAR})(?P<lead>(?>(?:\b|{_JWT_CHAR}*?-)(?=eyJ)))"
    rf"eyJ{_JWT_CHAR}{{10,}}+\.eyJ{_JWT_CHAR}{{10,}}+\.{_JWT_CHAR}{{10,}}\b"
)
_REDACTIONS = (
    (_JWT_RE, r"\g<lead>jwt-<redacted>"),
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
    return f"{match.group('lead')}{key}{match.group('sep')}<redacted>"


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


def _redact_unterminated_private_key(text: str) -> str:
    """Redact from the first private-key header with no later ``-----END `` to the end of the text."""
    # Same result as substituting ``HEADER(?![\s\S]*-----END )[\s\S]*\Z``, whose
    # lookahead rescanned the rest of the text from every header. A header qualifies
    # when the last "-----END " in the text starts before the header ends.
    last_end = text.rfind("-----END ")
    match = _PRIVATE_KEY_HEADER_RE.search(text)
    while match is not None:
        if match.end() > last_end:
            return f"{text[: match.start()]}private-key-<redacted>"
        # Headers can overlap ("...PRIVATE KEY-----BEGIN ..."), so resume inside this one.
        match = _PRIVATE_KEY_HEADER_RE.search(text, match.start() + 1)
    return text


def redact_sensitive_text(value: str, *, max_len: int | None = None) -> str:
    """Best-effort masking for credentials before writing logs or artifacts."""
    out = value
    # Remove multiline private-key material before any single-line assignment
    # or header rule can consume only its BEGIN delimiter and orphan the body.
    for pattern, replacement in _PEM_REDACTIONS:
        out = pattern.sub(replacement, out)
    out = _redact_unterminated_private_key(out)
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
