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
# Tokens whose prefix names their service: GitHub classic (ghp_/gho_/ghu_/ghs_/ghr_) and
# fine-grained (github_pat_) tokens, GitLab personal access tokens (glpat-), and Slack
# tokens (xoxa-/xoxb-/xoxp-/xoxr-/xoxs-). Redaction keeps the ``prefix`` group. Each
# pattern is the prefix and one character class of at most 255 characters, so a match
# attempt reads a bounded number of characters and a scan stays linear in the text.
# tier3/eval_core/secret_redaction.py redacts log lines with these patterns, and the
# standalone Harbor verifier keeps a copy that a drift test pins.
GITHUB_TOKEN_RE = re.compile(r"\b(?P<prefix>gh[pousr]_)[A-Za-z0-9]{36,255}\b")
GITHUB_PAT_RE = re.compile(r"\b(?P<prefix>github_pat_)[A-Za-z0-9_]{22,255}\b")
GITLAB_PAT_RE = re.compile(r"\b(?P<prefix>glpat-)[A-Za-z0-9_-]{20,255}")
SLACK_TOKEN_RE = re.compile(r"\b(?P<prefix>xox[abprs]-)[A-Za-z0-9-]{10,255}")
PREFIXED_TOKEN_PATTERNS = (GITHUB_TOKEN_RE, GITHUB_PAT_RE, GITLAB_PAT_RE, SLACK_TOKEN_RE)
_REDACTIONS = (
    (_JWT_RE, r"\g<lead>jwt-<redacted>"),
    (re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"), "aws-access-key-<redacted>"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])sk-[a-zA-Z0-9_-]{8,}"), "sk-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])nvapi-[a-zA-Z0-9_-]{8,}"), "nvapi-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])crsr_[a-f0-9]{16,}"), "crsr_<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])sha256~[A-Za-z0-9._~-]+"), "sha256~<redacted>"),
    *((pattern, r"\g<prefix><redacted>") for pattern in PREFIXED_TOKEN_PATTERNS),
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
