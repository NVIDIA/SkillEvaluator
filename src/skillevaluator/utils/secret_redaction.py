# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded redaction helpers for credential findings and CI report evidence."""

from __future__ import annotations

import html
import json
import re
from urllib.parse import quote, unquote, urlsplit

SENSITIVE_PII_CATEGORIES = frozenset(
    {
        "database_credentials",
        "jwt_tokens",
        "hardcoded_secrets",
        "webhook_urls",
        "aws_identifiers",
        "github_tokens",
        "private_keys",
    }
)
REDACTED = "[REDACTED]"


def sensitive_values(category: str, matched: str, line: str = "") -> set[str]:
    """Find literal values to redact without ever emitting them in diagnostics."""
    if category not in SENSITIVE_PII_CATEGORIES or not matched:
        return set()
    values = {matched}
    if category == "hardcoded_secrets":
        values.update(re.findall(r"[\"']([^\"']+)[\"']", matched))
    if category == "database_credentials":
        try:
            parsed = urlsplit(matched)
            if parsed.password:
                values.add(parsed.password)
                values.add(unquote(parsed.password))
        except ValueError:
            pass
        values.update(re.findall(r"(?i)password\s*=\s*[\"']?([^\s\"';&]+)", line))
    if category == "webhook_urls":
        values.add(matched.rsplit("/", 1)[-1])
    return {value for value in values if value}


def variants(tokens: set[str]) -> tuple[str, ...]:
    """Include encoded values and truncated prefixes of credential literals."""
    encoded: set[str] = set()
    pattern_size = 0
    for token in tokens:
        forms = {
            token,
            html.escape(token),
            html.escape(token, quote=False),
            json.dumps(token)[1:-1],
            quote(token, safe=""),
        }
        context = (
            "://" in token
            or token.startswith("-----BEGIN ")
            or re.match(r"(?i)^(?:secret|password|passwd|pwd|api[-_]?key|apikey|auth[-_]?token|bearer)\s*[=:]", token)
        )
        for value in forms:
            candidates = [value]
            if not context and len(value) >= 16:
                if len(value) > 512:
                    raise ValueError("A credential is too long for safe partial-value redaction.")
                candidates.extend(value[:length] for length in range(5, len(value)))
            for candidate in candidates:
                if candidate not in encoded:
                    pattern_size += len(candidate)
                    if pattern_size > 1024 * 1024:
                        raise ValueError("Credential redaction patterns exceed their limit.")
                    encoded.add(candidate)
    return tuple(sorted((value for value in encoded if value), key=len, reverse=True))


class Redactor:
    """Compile the token set once for efficient, longest-first replacement."""

    def __init__(self, tokens: set[str]) -> None:
        if len(tokens) > 4096 or sum(len(token) for token in tokens) > 256 * 1024:
            raise ValueError("Credential redaction values exceed their limit.")
        values = variants(tokens)
        if sum(len(value) for value in values) > 1024 * 1024:
            raise ValueError("Credential redaction patterns exceed their limit.")
        self.pattern = re.compile("|".join(re.escape(value) for value in values)) if values else None

    def text(self, text: str) -> str:
        return self.pattern.sub(REDACTED, text) if self.pattern else text

    def object(self, value: object) -> object:
        if isinstance(value, str):
            return self.text(value)
        if type(value) in {int, float}:
            raw = json.dumps(value, allow_nan=False)
            safe = self.text(raw)
            return safe if safe != raw else value
        if isinstance(value, list):
            return [self.object(item) for item in value]
        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                safe_key = self.text(key) if isinstance(key, str) else key
                if safe_key != key or safe_key in output:
                    raise ValueError("Redaction would alter report keys.")
                output[safe_key] = self.object(item)
            return output
        return value
