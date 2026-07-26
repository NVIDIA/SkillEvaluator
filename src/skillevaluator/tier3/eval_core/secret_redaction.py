# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared best-effort secret redaction for Layer 2 output surfaces."""

from __future__ import annotations

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
LOG_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b")


def redact_secrets_in_log_line(
    line: str,
    *,
    extra_secret_values: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    """Best-effort mask common key shapes in Layer 2 output text."""
    for secret in sorted(set(extra_secret_values or ()), key=len, reverse=True):
        if secret and len(secret) >= 8:
            line = line.replace(secret, "<redacted>")
    line = LOG_SK_RE.sub("sk-<redacted>", line)
    line = LOG_NVAPI_RE.sub("nvapi-<redacted>", line)
    line = LOG_CRSR_RE.sub("crsr_<redacted>", line)
    line = OPENSHIFT_TOKEN_RE.sub("sha256~<redacted>", line)
    return LOG_JWT_RE.sub("jwt-<redacted>", line)
