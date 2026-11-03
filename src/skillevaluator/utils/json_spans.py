"""Bounded JSON span scanning and strict decode hooks for LLM replies.

The Tier 3 judge and the inference client both recover JSON that a model
wrapped in prose. They share these helpers so the two extractors agree on what
a container span is and on which decoded values are ambiguous: duplicate
object members, ``NaN``/``Infinity`` constants and numbers that overflow to a
non-finite float are all rejected.
"""

from __future__ import annotations

import json
import math
from typing import Any

# Structural depth beyond which a span is not decoded. LLM verdicts are
# shallow, and the cap keeps the decoder away from recursion and C stack
# limits, which differ between Python versions and threads.
MAX_JSON_NESTING = 128


def balanced_json_container_end(text: str, start: int) -> int | None:
    """Return the exclusive end of the string-aware ``{...}``/``[...]`` span at *start*.

    ``None`` means *start* is not a container opener, or the span never
    closes, closes with a mismatched bracket, or nests deeper than
    ``MAX_JSON_NESTING``.
    """
    if start >= len(text) or text[start] not in "{[":
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
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
                return i + 1
    return None


def json_nesting_within_limit(text: str) -> bool:
    """Bound structural nesting without recursively parsing partial JSON."""
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
    raise ValueError("Non-standard JSON constant")


def parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number overflowed to a non-finite value")
    return parsed


def loads_strict_json(text: str) -> Any:
    """Decode *text* with the shared hooks.

    Raises ``ValueError`` (including ``json.JSONDecodeError``) for invalid or
    ambiguous JSON, and may raise ``RecursionError`` for deep input; callers
    that need a depth bound check ``json_nesting_within_limit`` first.
    """
    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_object_pairs,
        parse_constant=reject_nonstandard_json_constant,
        parse_float=parse_finite_json_float,
    )
