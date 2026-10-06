# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared best-effort secret redaction for Layer 2 output surfaces."""

from __future__ import annotations

import os
import re

# Prefix-style key detectors match either (a) a prefix at a token boundary
# (negative lookbehind), with any body, or (b) a prefix glued directly onto a
# word char, but only when followed by a strong real-key signature: a
# contiguous run of >=20 alphanumerics containing lower, upper AND a digit.
# The boundary form stops "sk-" matching inside ordinary hyphenated words
# ("task-granularity" -> "sk-granularity") and mangling log text; the glued
# form still catches a key jammed onto a word ("xsk-Ab1Cd2...") without
# matching dictionary words, lowercase hex IDs/hashes, or short tokens.
# Mirrors skillevaluator.utils.redaction and skillevaluator.tier3.eval_core.checks._SECRET_PATTERNS.
_GLUED_KEY_BODY = r"(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{20,}"
LOG_SK_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[a-zA-Z0-9_-]{8,}|sk-" + _GLUED_KEY_BODY)
LOG_NVAPI_RE = re.compile(r"(?<![A-Za-z0-9_-])nvapi-[a-zA-Z0-9_-]{8,}|nvapi-" + _GLUED_KEY_BODY)
LOG_CRSR_RE = re.compile(r"(?<![A-Za-z0-9_-])crsr_[a-f0-9]{16,}")
OPENSHIFT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])sha256~[A-Za-z0-9._~-]+")
# A JWT used to start at any ``\beyJ``, so in a run of JWT characters such as
# "eyJ-" * n every "-eyJ" was a start, and each start scanned to the end of the
# run looking for ".". Now a match starts only at the beginning of a run. The part
# of the run before its first ``\beyJ`` is captured as ``lead`` and written back
# unchanged, which keeps JWTs glued to a "-" (x-eyJ...) redacted. Later starts in
# the same run are never tried: their first segment reaches the same "." with
# fewer characters, so they could only fail where the first start failed. ``lead``
# is found inside a lookahead with a lazy one-character repeat, then consumed by a
# backreference. Python never backtracks into a lookahead, and a one-character
# repeat keeps no state per character, so the scan is linear in time and flat in
# memory. (A repeated group such as ``(?:(?!\beyJ)[A-Za-z0-9_-])*`` keeps
# backtracking state for every character, about 75 bytes each.) No atomic groups
# or possessive quantifiers: the Harbor verifier copy runs on the task image's
# python3, which may predate 3.11.
LOG_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?=(?P<lead>(?:\b|[A-Za-z0-9_-]*?-)(?=eyJ)))(?P=lead)"
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"
)
# Tokens whose prefix names their service: GitHub classic (ghp_/gho_/ghu_/ghs_/ghr_) and
# fine-grained (github_pat_) tokens, GitLab personal access tokens (glpat-), Slack tokens
# (xoxa-/xoxb-/xoxp-/xoxr-/xoxs-, and xoxe- refresh tokens), Hugging Face tokens (hf_), and
# npm tokens (npm_). Redaction keeps the ``prefix`` group. Each pattern is the prefix and one
# character class of at most 255 characters, so a match attempt reads a bounded number of
# characters and a scan stays linear in the text. AWS access key IDs (AKIA long-term, ASIA
# temporary) are redacted whole. They are defined here, in a module that imports nothing from
# the package, so that eval_core loads on its own; skillevaluator.utils.redaction redacts
# artifacts with the same patterns, and the standalone Harbor verifier keeps a copy that a
# drift test pins.
LOG_GITHUB_TOKEN_RE = re.compile(r"\b(?P<prefix>gh[pousr]_)[A-Za-z0-9]{36,255}\b")
LOG_GITHUB_PAT_RE = re.compile(r"\b(?P<prefix>github_pat_)[A-Za-z0-9_]{22,255}\b")
LOG_GITLAB_PAT_RE = re.compile(r"\b(?P<prefix>glpat-)[A-Za-z0-9_-]{20,255}")
LOG_SLACK_TOKEN_RE = re.compile(r"\b(?P<prefix>xox[abeprs]-)[A-Za-z0-9-]{10,255}")
LOG_HUGGING_FACE_TOKEN_RE = re.compile(r"\b(?P<prefix>hf_)[A-Za-z0-9]{30,255}")
LOG_NPM_TOKEN_RE = re.compile(r"\b(?P<prefix>npm_)[A-Za-z0-9]{36}\b")
LOG_PREFIXED_TOKEN_PATTERNS = (
    LOG_GITHUB_TOKEN_RE,
    LOG_GITHUB_PAT_RE,
    LOG_GITLAB_PAT_RE,
    LOG_SLACK_TOKEN_RE,
    LOG_HUGGING_FACE_TOKEN_RE,
    LOG_NPM_TOKEN_RE,
)
LOG_AWS_ACCESS_KEY_RE = re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")
# All of the prefixed patterns in one pass over the text. Each pattern's ``prefix`` group is
# renamed for its alternative, so the alternative that matched is the match's last group.
LOG_PREFIXED_TOKEN_RE = re.compile(
    "|".join(
        pattern.pattern.replace("(?P<prefix>", f"(?P<prefix{index}>", 1)
        for index, pattern in enumerate(LOG_PREFIXED_TOKEN_PATTERNS)
    )
)


def keep_token_prefix(match: re.Match[str]) -> str:
    """The replacement for a ``LOG_PREFIXED_TOKEN_RE`` match: its prefix, then ``<redacted>``."""
    return f"{match.group(match.lastgroup)}<redacted>"


# Match verifier log redaction; shorter placeholders can corrupt ordinary diagnostic text.
_MIN_EXACT_SECRET_LENGTH = 8
_CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "ANTHROPIC_API_KEY",
    "SKILL_EVAL_LLM_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SECURITY_TOKEN",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)


def _configured_secret_values(extra_secret_values: tuple[str | None, ...] = ()) -> list[str]:
    """The credential values this process holds, plus *extra_secret_values*, longest first."""
    values = {
        value
        for name in _CREDENTIAL_ENV_VARS
        if (value := os.environ.get(name, "")) and len(value) >= _MIN_EXACT_SECRET_LENGTH
    }
    for value in extra_secret_values:
        text = str(value) if value else ""
        if len(text) >= _MIN_EXACT_SECRET_LENGTH:
            values.add(text)
    return sorted(values, key=len, reverse=True)


def redact_secrets_in_log_line(
    line: str,
    *,
    extra_secret_values: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    """Best-effort mask common key shapes in Layer 2 output text."""
    for secret in sorted(set(extra_secret_values or ()), key=len, reverse=True):
        if secret and len(secret) >= _MIN_EXACT_SECRET_LENGTH:
            line = line.replace(secret, "<redacted>")
    line = LOG_SK_RE.sub("sk-<redacted>", line)
    line = LOG_NVAPI_RE.sub("nvapi-<redacted>", line)
    line = LOG_CRSR_RE.sub("crsr_<redacted>", line)
    line = LOG_PREFIXED_TOKEN_RE.sub(keep_token_prefix, line)
    line = LOG_AWS_ACCESS_KEY_RE.sub("aws-access-key-<redacted>", line)
    line = OPENSHIFT_TOKEN_RE.sub("sha256~<redacted>", line)
    if "eyJ" not in line:  # every JWT match contains "eyJ"; skip the scan on ordinary lines
        return line
    return LOG_JWT_RE.sub(r"\g<lead>jwt-<redacted>", line)
