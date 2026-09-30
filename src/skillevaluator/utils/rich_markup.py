# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Escape untrusted text for Rich console markup."""

from __future__ import annotations

import re

# Every "[" with the whole backslash run before it (the lookbehind and possessive
# quantifier keep long runs linear). The optional group is set when the bracket
# opens something Rich parses as a tag (same shape as ``rich.markup.RE_TAGS``).
_OPEN_BRACKET = re.compile(r"(?<!\\)(\\*+)\[(?=([a-z#/@][^\[\]]*\])?)")


def _escape_bracket(match: re.Match[str]) -> str:
    backslashes, tag = match.group(1), match.group(2)
    if tag is not None:
        # Tag-shaped: literal backslashes are doubled, then the tag is escaped.
        return f"{backslashes}{backslashes}\\["
    # Rich renders a plain-text "\[" as "[", so one extra backslash keeps the rest.
    return f"{backslashes}\\["


def escape_markup(text: str) -> str:
    """Return ``text`` escaped so Rich renders it literally inside markup.

    ``rich.markup.escape`` only escapes tag-shaped brackets, but Rich also turns
    every plain-text ``\\[`` into ``[``, so Windows paths such as
    ``C:\\skills\\[x]`` lose a separator. This escapes every bracket.
    """
    escaped = _OPEN_BRACKET.sub(_escape_bracket, text)
    body = escaped.rstrip("\\")
    trailing = len(escaped) - len(body)
    # Double trailing backslashes so they stay literal and cannot escape the tag that follows.
    return body + "\\" * (2 * trailing) if trailing else escaped
