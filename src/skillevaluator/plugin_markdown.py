# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, client-compatible frontmatter parsing for private plugin analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from skillevaluator.utils.structured_data import StructuredDataError, StructuredDataLimitError, load_bounded_yaml


@dataclass(frozen=True)
class _Markdown:
    name: str | None
    description: str | None
    body: str
    frontmatter: dict[str, Any]
    # Why frontmatter over a parser limit was not read; its fields are unknown, not absent.
    frontmatter_error: str | None = None


_LENIENT_KEY_RE = re.compile(r"^([a-zA-Z_-]+):\s+(.+)$")


_LENIENT_SPECIAL_RE = re.compile(r"[{}\[\]*&#!|>%@`]|: ")


_LEADING_TABS_RE = re.compile(r"^\t+", re.MULTILINE)


# Claude Code's frontmatter split: the block ends at the first ``---`` after the opening line, even mid-line.
_FRONTMATTER_RE = re.compile(r"---\s*\n([\s\S]*?)---\s*\n?")


def _quote_lenient_values(raw: str) -> str:
    """Quote the plain frontmatter values strict YAML rejects, the way Claude Code retries a failed parse.

    Claude Code 2.1.x parses frontmatter with a YAML parser and, when that
    fails, quotes each top-level ``key: value`` whose value contains a YAML
    indicator (``{}[]*&#!|>%@`` or a backtick) or ``": "`` (unless it is
    already quoted or is a valid flow list), turns leading tabs into two spaces
    each, and parses again. So
    ``description: Deploy: runs the script`` and ``tools: *`` load, and the
    privileges and context cost must be read the same way.
    """
    lines: list[str] = []
    for line in raw.split("\n"):
        match = _LENIENT_KEY_RE.match(line)
        if match is not None:
            key, value = match.groups()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                lines.append(line)
                continue
            if value.startswith("[") and value.endswith("]"):
                try:
                    if isinstance(load_bounded_yaml(value), list):
                        lines.append(line)
                        continue
                except StructuredDataError:
                    pass
            if _LENIENT_SPECIAL_RE.search(value):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{key}: "{escaped}"')
                continue
        lines.append(line)
    return _LEADING_TABS_RE.sub(lambda tabs: "  " * len(tabs.group(0)), "\n".join(lines))


def _load_frontmatter(raw: str) -> Any:
    """Bounded YAML frontmatter, with Claude Code's lenient retry when strict YAML rejects it.

    A key written twice keeps its last value, as in Claude Code.
    """
    if not raw.strip():
        return {}
    try:
        return load_bounded_yaml(raw, last_key_wins=True)
    except StructuredDataLimitError:
        raise
    except (StructuredDataError, ValueError):
        return load_bounded_yaml(_quote_lenient_values(raw), last_key_wins=True)


def parse_markdown(text: str) -> _Markdown:
    """Split optional ``---`` YAML frontmatter from a Markdown body (bounded YAML).

    Frontmatter that strict YAML rejects is read again the way Claude Code
    reads it (:func:`_quote_lenient_values`), so a value such as ``Deploy:
    runs the script`` does not hide the fields after it. Frontmatter that
    still fails to parse gives no fields, as in Claude Code. Frontmatter over a
    parser limit (a 65,536-character value, 1,024 list or mapping items) gives
    no fields and a ``frontmatter_error``: Claude Code still reads it, so its
    grants are unknown and the caller must fail closed.

    The block is split the way Claude Code splits it: after a leading BOM, an
    opening ``---`` line, up to the first ``---`` that follows, even when it
    begins a longer line such as ``----`` or ``--- # end``.
    """
    match = _FRONTMATTER_RE.match(text.removeprefix("\ufeff"))
    if match is None:
        return _Markdown(None, None, text.strip(), {})
    raw = match.group(1)
    body = match.string[match.end() :].strip()
    try:
        data = _load_frontmatter(raw)
    except StructuredDataLimitError as exc:
        return _Markdown(None, None, body, {}, frontmatter_error=str(exc))
    except (StructuredDataError, ValueError, RecursionError):
        return _Markdown(None, None, body, {})
    if not isinstance(data, dict):
        return _Markdown(None, None, body, {})
    name = data.get("name") if isinstance(data.get("name"), str) else None
    description = data.get("description") if isinstance(data.get("description"), str) else None
    return _Markdown(name, description, body, data)
