# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free normalization for native Codex ``tools.exec`` wrappers."""

from __future__ import annotations

import json
import re
from typing import Any

UNSUPPORTED_NATIVE_CODEX_EXEC = "unsupported_native_codex_exec_wrapper"
AMBIGUOUS_OUTER_EXEC_OBSERVATION = "ambiguous_outer_exec_result"
MAPPED_OUTER_EXEC_OBSERVATION = "mapped_outer_exec_result"
UNOBSERVED_INNER_CALL = "unobserved_inner_call"

_MAX_SOURCE_CHARS = 64 * 1024
_MAX_OBJECT_NESTING = 64
_MAX_STATEMENTS = 256
_MAX_TOOL_CALLS = 128


def iter_normalized_tool_calls(traj: dict[str, Any]):
    """Yield each trajectory tool call after safe native-Codex normalization."""
    for step in traj.get("steps", []):
        for raw_index, tool_call in enumerate(step.get("tool_calls") or []):
            for normalized in normalize_tool_call(tool_call):
                yield step, {**normalized, "_atif_raw_tool_index": raw_index}


def normalized_tool_call_observation(step: dict[str, Any], tool_call: dict[str, Any]) -> str:
    """Return an outer observation only when its normalized owner is known."""
    if tool_call.get("_atif_observation_status") not in (None, MAPPED_OUTER_EXEC_OBSERVATION):
        return ""
    return "".join(
        str(result.get("content", ""))
        for result in (step.get("observation") or {}).get("results") or []
        if result.get("source_call_id") == tool_call.get("tool_call_id") or not result.get("source_call_id")
    )


def _skip_js_quoted(source: str, start: int, quote: str) -> int:
    index = start + 1
    while index < len(source):
        if source[index] == "\\":
            index += 2
        elif source[index] == quote:
            return index + 1
        else:
            index += 1
    return -1


def _decode_static_js_object(source: str, start: int) -> tuple[dict[str, Any], int] | None:
    """Decode a JSON-compatible object literal, including unquoted property names."""
    rendered: list[str] = []
    stack: list[str] = []
    index = start
    previous_significant = ""
    while index < len(source):
        char = source[index]
        if char == '"':
            end = _skip_js_quoted(source, index, char)
            if end < 0:
                return None
            rendered.append(source[index:end])
            previous_significant = '"'
            index = end
            continue
        if char in "'`" or source.startswith(("//", "/*"), index):
            return None
        if char in "{[":
            if len(stack) >= _MAX_OBJECT_NESTING:
                return None
            stack.append(char)
        elif char in "}]":
            if not stack or stack.pop() != ("{" if char == "}" else "["):
                return None
        if (char.isalpha() or char in "_$") and previous_significant in {"{", ","}:
            end = index + 1
            while end < len(source) and (source[end].isalnum() or source[end] in "_$"):
                end += 1
            cursor = end
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
            if cursor < len(source) and source[cursor] == ":":
                rendered.append(json.dumps(source[index:end]))
                previous_significant = '"'
                index = end
                continue
        rendered.append(char)
        if not char.isspace():
            previous_significant = char
        index += 1
        if not stack:
            try:
                arguments = json.loads("".join(rendered))
            except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
                return None
            return (arguments, index) if isinstance(arguments, dict) else None
    return None


_JS_IDENTIFIER = r"[A-Za-z_$][\w$]*"
_CODEX_CALL_RE = re.compile(rf"const\s+({_JS_IDENTIFIER})\s*=\s*await\s+tools\.({_JS_IDENTIFIER})\s*\(\s*")
_CODEX_RENDER_RE = re.compile(
    rf"text\s*\(\s*(?:JSON\.stringify\(\s*({_JS_IDENTIFIER})\s*\)|"
    rf"({_JS_IDENTIFIER})(?:\.({_JS_IDENTIFIER}))?)\s*\)\s*;"
)
_CODEX_PRAGMA_RE = re.compile(r"[ \t]*// @exec:[^\r\n]*\r?\n")
_CODEX_TOOL_REF_RE = re.compile(
    rf"\btools\s*(?:\.\s*{_JS_IDENTIFIER}|"
    rf"\[\s*(?P<quote>['\"]){_JS_IDENTIFIER}(?P=quote)\s*\])\s*(?:\\)?\("
)


def _static_codex_tool_calls(source: str) -> tuple[list[tuple[str, dict[str, Any]]], int | None] | None:
    """Decode the complete, bounded statement grammar emitted by native Codex."""
    if len(source) > _MAX_SOURCE_CHARS:
        return None
    calls: list[tuple[str, dict[str, Any]]] = []
    variables: list[str] = []
    rendered_variables: list[str] = []
    statements = 0
    pragma = _CODEX_PRAGMA_RE.match(source)
    index = pragma.end() if pragma else 0
    while index < len(source):
        while index < len(source) and source[index].isspace():
            index += 1
        if index == len(source):
            break

        call = _CODEX_CALL_RE.match(source, index)
        if call:
            statements += 1
            if statements > _MAX_STATEMENTS or len(calls) >= _MAX_TOOL_CALLS:
                return None
            variable, function_name = call.groups()
            if variable in variables:
                return None
            decoded = _decode_static_js_object(source, call.end())
            if decoded is None:
                return None
            arguments, end = decoded
            close = re.match(r"\s*\)\s*;", source[end:])
            if close is None:
                return None
            variables.append(variable)
            calls.append((function_name, arguments))
            index = end + close.end()
            continue

        render = _CODEX_RENDER_RE.match(source, index)
        rendered_variable = next((name for name in render.groups() if name), None) if render else None
        if rendered_variable in variables:
            statements += 1
            if statements > _MAX_STATEMENTS:
                return None
            rendered_variables.append(rendered_variable)
            index = render.end()
            continue
        return None

    if not calls:
        return None
    rendered_indices = {variables.index(variable) for variable in rendered_variables}
    if not rendered_indices:
        return calls, -1
    if len(rendered_indices) == 1:
        return calls, rendered_indices.pop()
    return calls, None


def normalize_tool_call(tool_call: dict[str, Any]) -> list[dict[str, Any]]:
    """Unwrap proven native Codex calls without interpreting arbitrary JavaScript."""
    tool_call = {key: value for key, value in tool_call.items() if not key.startswith("_atif_")}
    if tool_call.get("function_name") != "exec":
        return [tool_call]
    arguments = tool_call.get("arguments") or {}
    if not isinstance(arguments, dict) or not isinstance(arguments.get("input"), str):
        return [tool_call]
    source = arguments["input"]
    if len(source) > _MAX_SOURCE_CHARS:
        return [{**tool_call, "_atif_normalization_status": UNSUPPORTED_NATIVE_CODEX_EXEC}]
    parsed = _static_codex_tool_calls(source)
    if parsed is None:
        if not (_CODEX_PRAGMA_RE.match(source) or _CODEX_TOOL_REF_RE.search(source)):
            return [tool_call]
        return [{**tool_call, "_atif_normalization_status": UNSUPPORTED_NATIVE_CODEX_EXEC}]

    calls, observation_owner = parsed
    normalized: list[dict[str, Any]] = []
    for inner_index, (function_name, inner_arguments) in enumerate(calls):
        if observation_owner is None:
            observation_status = AMBIGUOUS_OUTER_EXEC_OBSERVATION
        elif observation_owner < 0:
            observation_status = UNOBSERVED_INNER_CALL
        elif inner_index == observation_owner:
            observation_status = MAPPED_OUTER_EXEC_OBSERVATION
        else:
            observation_status = UNOBSERVED_INNER_CALL
        normalized.append(
            {
                **tool_call,
                "function_name": function_name,
                "arguments": inner_arguments,
                "_atif_inner_tool_index": inner_index,
                "_atif_observation_status": observation_status,
            }
        )
    return normalized
