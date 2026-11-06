# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded redaction helpers for credential findings and CI report evidence."""

from __future__ import annotations

import html
import json
import re
from itertools import groupby
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
REPEATED_REDACTED = re.compile(f"(?:{re.escape(REDACTED)}){{2,}}")
# Snippets can cut a database URL before the "@" that ends its credentials,
# and a short password only matches as a whole value. Mask user:password after
# a database scheme even when the text ends inside it. A bare port is kept.
DATABASE_USERINFO = re.compile(
    r"(?i)\b((?:postgres(?:ql)?|mysql|mariadb|mongodb|rediss?)(?:\+\w+)?://)"
    r"[^\s/?#@:\"'`<>]*:(?:[^\s/?#@\"'`<>]+(?=@)|(?!\d*$)[^\s/?#@\"'`<>]+$)"
)


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


def _invisible(character: str) -> bool:
    # The Unicode check echoes these characters as U+2423 in line content.
    return not character.isprintable() and character not in " \t\r\n"


def _visible_forms(token: str) -> set[str]:
    """Add the forms another check can echo for a value with invisible characters."""
    if not any(_invisible(character) for character in token):
        return {token}
    visible = "".join(character for character in token if not _invisible(character))
    if len(visible) < 8:
        return {token}
    forms = {token, visible, "".join("\u2423" if _invisible(character) else character for character in token)}
    pieces = ("".join(group) for hidden, group in groupby(token, key=_invisible) if not hidden)
    forms.update(piece for piece in pieces if len(piece) >= 8)
    return forms


def _password_tails(token: str) -> set[str]:
    """Return the ends of a short URL password, each with the "@" after it."""
    try:
        password = urlsplit(token).password if "://" in token else None
    except ValueError:
        return set()
    # Upstream text can start inside the password and cut off the scheme, so
    # DATABASE_USERINFO does not match. A password shorter than 16 characters
    # has no partial values, but the "@" after it still marks its end.
    if not password or len(password) >= 16:
        return set()
    return {password[start:] + "@" for start in range(1, len(password) - 7)}


def variants(tokens: set[str]) -> tuple[str, ...]:
    """Include encoded values and truncated parts of credential literals."""
    encoded: set[str] = set()
    pattern_size = 0
    tails = {tail for value in tokens for tail in _password_tails(value)}
    for token in {form for value in tokens for form in _visible_forms(value)} | tails:
        forms = {
            token,
            html.escape(token),
            html.escape(token, quote=False),
            json.dumps(token)[1:-1],
            quote(token, safe=""),
            html.escape(token, quote=False).replace("|", "&#124;").replace("`", "&#96;"),
        }
        context = (
            # A password tail is matched only as a whole value.
            token in tails
            or "://" in token
            or token.startswith("-----BEGIN ")
            or re.match(r"(?i)^(?:secret|password|passwd|pwd|api[-_]?key|apikey|auth[-_]?token|bearer)\s*[=:]", token)
            # A connection string match holds ordinary words; its password is collected separately.
            or token.lower().endswith("password=")
        )
        for value in forms:
            candidates = [value]
            if not context and len(value) >= 16:
                if len(value) > 512:
                    raise ValueError("A credential is too long for safe partial-value redaction.")
                candidates.extend(value[:length] for length in range(5, len(value)))
                # Upstream text can also start inside a value, or start and end
                # inside it. Match every window of 8 characters and suffixes of 8
                # to 15, so no fragment of 8 or more characters is left.
                candidates.extend(value[-length:] for length in range(8, 16))
                candidates.extend(value[start : start + 8] for start in range(len(value) - 7))
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
        # Group values by first character so that windows starting with any
        # character stay fast to scan. The longest value in a group wins.
        groups: dict[str, list[str]] = {}
        for value in values:
            groups.setdefault(value[0], []).append(re.escape(value[1:]))
        alternatives = (re.escape(first) + "(?:" + "|".join(rest) + ")" for first, rest in groups.items())
        self.pattern = re.compile("|".join(alternatives)) if values else None

    def text(self, text: str) -> str:
        if self.pattern:
            # Windows of one value are replaced one at a time; show one marker.
            text = REPEATED_REDACTED.sub(REDACTED, self.pattern.sub(REDACTED, text))
        return DATABASE_USERINFO.sub(lambda match: match.group(1) + REDACTED, text)

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
