"""Shared JSON span scanning and strict decode hooks for LLM replies.

Both the public inference client and the Tier 3 judge recover JSON that a model
wrapped in prose. They share one span scanner and one set of decode hooks so
the two paths accept and refuse the same replies.
"""

from __future__ import annotations

import math
from typing import Any

# Bracket depth beyond which a span is not decoded. Verdicts are shallow, and
# the cap keeps the decoder well away from the recursion limit.
MAX_JSON_NESTING = 128


def balanced_json_container_end(text: str, start: int) -> int | None:
    """Return the exclusive end of the string-aware ``{...}``/``[...]`` span at *start*.

    ``None`` means *start* is not a container opener, or the span never closes,
    closes with a mismatched bracket, or nests deeper than ``MAX_JSON_NESTING``.
    """
    if start >= len(text) or text[start] not in "{[":
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
            if len(stack) > MAX_JSON_NESTING:
                return None
        elif ch in "}]":
            expected = "{" if ch == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


def json_nesting_within_limit(text: str) -> bool:
    """Bound structural nesting of *text* without recursively parsing it.

    Checked before decoding a whole reply, because the decoder's own depth
    limit depends on the Python version and the thread's C stack size.
    """
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > MAX_JSON_NESTING:
                return False
        elif character in "}]" and depth:
            depth -= 1
    return True


def reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting ambiguous duplicate members."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object member")
        result[key] = value
    return result


def reject_nonstandard_json_constant(_value: str) -> None:
    """``parse_constant`` hook that refuses ``NaN`` and ``Infinity``."""
    raise ValueError("Non-standard JSON constant")


def parse_finite_json_float(value: str) -> float:
    """``parse_float`` hook that refuses numbers overflowing to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number overflowed to a non-finite value")
    return parsed


STRICT_JSON_DECODE_HOOKS: dict[str, Any] = {
    "object_pairs_hook": reject_duplicate_object_pairs,
    "parse_constant": reject_nonstandard_json_constant,
    "parse_float": parse_finite_json_float,
}
