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
    depth = 0
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
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
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
        if depth == 0:
            try:
                arguments = json.loads("".join(rendered))
            except (json.JSONDecodeError, TypeError):
                return None
            return (arguments, index) if isinstance(arguments, dict) else None
    return None


_JS_IDENTIFIER = r"[A-Za-z_$][\w$]*"
_CODEX_CALL_RE = re.compile(rf"const\s+({_JS_IDENTIFIER})\s*=\s*await\s+tools\.({_JS_IDENTIFIER})\s*\(\s*")
_CODEX_RENDER_RE = re.compile(
    rf"text\s*\(\s*(?:JSON\.stringify\(\s*({_JS_IDENTIFIER})\s*\)|"
    rf"({_JS_IDENTIFIER})(?:\.({_JS_IDENTIFIER}))?)\s*\)\s*;"
)


def _static_codex_tool_calls(source: str) -> tuple[list[tuple[str, dict[str, Any]]], int | None] | None:
    """Decode the complete, bounded statement grammar emitted by native Codex."""
    calls: list[tuple[str, dict[str, Any]]] = []
    variables: list[str] = []
    rendered_variables: list[str] = []
    pragma = re.match(r"[ \t]*// @exec:[^\r\n]*\r?\n", source)
    index = pragma.end() if pragma else 0
    while index < len(source):
        while index < len(source) and source[index].isspace():
            index += 1
        if index == len(source):
            break

        call = _CODEX_CALL_RE.match(source, index)
        if call:
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
    if tool_call.get("function_name") != "exec":
        return [tool_call]
    arguments = tool_call.get("arguments") or {}
    if not isinstance(arguments, dict) or not isinstance(arguments.get("input"), str):
        return [tool_call]
    parsed = _static_codex_tool_calls(arguments["input"])
    if parsed is None:
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
