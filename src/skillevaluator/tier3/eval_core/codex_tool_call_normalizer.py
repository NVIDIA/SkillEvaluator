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


def _outer_tool_call_observation(
    step: dict[str, Any], tool_call: dict[str, Any], *, include_unscoped: bool = True
) -> str:
    tool_call_id = tool_call.get("tool_call_id")
    return "".join(
        str(result.get("content", ""))
        for result in (step.get("observation") or {}).get("results") or []
        if (
            tool_call_id
            and result.get("source_call_id")
            and result.get("source_call_id") == tool_call_id
        )
        or (include_unscoped and not result.get("source_call_id"))
    )


def normalized_tool_call_observation(step: dict[str, Any], tool_call: dict[str, Any]) -> str:
    """Return an outer observation only when its normalized owner is known."""
    if tool_call.get("_atif_observation_status") not in (None, MAPPED_OUTER_EXEC_OBSERVATION):
        return ""
    return _outer_tool_call_observation(
        step,
        tool_call,
        include_unscoped=tool_call.get("_atif_raw_tool_index", 0) == 0,
    )


def normalized_tool_call_wrapper_observation(step: dict[str, Any], tool_call: dict[str, Any]) -> str:
    """Return an unattributed wrapper observation once for security scanning."""
    if tool_call.get("_atif_observation_status") not in (
        AMBIGUOUS_OUTER_EXEC_OBSERVATION,
        UNOBSERVED_INNER_CALL,
    ) or tool_call.get("_atif_inner_tool_index") != 0:
        return ""
    observation_owner = tool_call.get("_atif_outer_observation_owner")
    if isinstance(observation_owner, int) and observation_owner >= 0:
        return ""
    return _outer_tool_call_observation(
        step,
        tool_call,
        include_unscoped=tool_call.get("_atif_raw_tool_index", 0) == 0,
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
_CODEX_PRAGMA_PREFIX_RE = re.compile(r"[ \t]*// @exec:")
_CODEX_TOOL_REF_RE = re.compile(r"\b(?:tools|globalThis)\b|\\u(?:\{[0-9A-Fa-f]{1,6}\}|[0-9A-Fa-f]{4})")
_JS_IDENTIFIER_ESCAPE_RE = re.compile(r"\\u(?:\{([0-9A-Fa-f]{1,6})\}|([0-9A-Fa-f]{4}))")
_JS_LINE_TERMINATORS = "\r\n\u2028\u2029"
_JS_EXPRESSION_OPERATORS = {
    "...",
    "!",
    "%",
    "&",
    "&&",
    "*",
    "**",
    "+",
    "-",
    "/",
    "<",
    "<<",
    "<=",
    "=",
    "==",
    "===",
    "!=",
    "!==",
    ">",
    ">=",
    ">>",
    ">>>",
    "?",
    "??",
    "^",
    "|",
    "||",
    "~",
}
_JS_EXPRESSION_KEYWORDS = {
    "await",
    "case",
    "delete",
    "do",
    "else",
    "extends",
    "in",
    "instanceof",
    "new",
    "of",
    "return",
    "throw",
    "typeof",
    "void",
    "yield",
}


def _decode_js_identifier_escapes(source: str) -> str:
    """Decode JavaScript Unicode escapes for lexical tool-reference matching."""

    def replace(match: re.Match[str]) -> str:
        try:
            value = chr(int(match.group(1) or match.group(2), 16))
        except (ValueError, OverflowError):
            return match.group(0)
        return value if value.isidentifier() or value.isdigit() else match.group(0)

    return _JS_IDENTIFIER_ESCAPE_RE.sub(replace, source)


def _read_js_escape(source: str, start: int) -> tuple[str, int]:
    if start + 1 >= len(source):
        return "", start + 1
    escaped = source[start + 1]
    if escaped == "\r":
        end = start + 2
        return "", end + 1 if end < len(source) and source[end] == "\n" else end
    if escaped in "\n\u2028\u2029":
        return "", start + 2
    if escaped == "x" and start + 3 < len(source):
        try:
            return chr(int(source[start + 2 : start + 4], 16)), start + 4
        except ValueError:
            pass
    return escaped, start + 2


def _read_js_string(source: str, start: int) -> tuple[str, int]:
    quote = source[start]
    value: list[str] = []
    index = start + 1
    while index < len(source):
        if source[index] == "\\":
            escaped, index = _read_js_escape(source, index)
            value.append(escaped)
            continue
        if source[index] == quote:
            return "".join(value), index + 1
        value.append(source[index])
        index += 1
    return "".join(value), len(source)


def _read_static_js_template(source: str, start: int) -> tuple[str, int] | None:
    """Read a template with no substitutions as a string-like property key."""
    value: list[str] = []
    index = start + 1
    while index < len(source):
        if source[index] == "\\":
            escaped, index = _read_js_escape(source, index)
            value.append(escaped)
            continue
        if source.startswith("${", index):
            return None
        if source[index] == "`":
            return "".join(value), index + 1
        value.append(source[index])
        index += 1
    return None


def _skip_js_regex(source: str, start: int) -> int:
    """Return the end of one plausible regex literal or its invalid line."""
    index = start + 1
    in_character_class = False
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char in _JS_LINE_TERMINATORS:
            return index
        if char == "[":
            in_character_class = True
        elif char == "]":
            in_character_class = False
        elif char == "/" and not in_character_class:
            index += 1
            while index < len(source) and source[index].isalpha():
                index += 1
            return index
        index += 1
    return len(source)


def _js_code_tokens(source: str) -> list[tuple[str, str]]:
    """Tokenize enough JavaScript to distinguish code references from prose."""
    tokens: list[tuple[str, str]] = []
    contexts: list[tuple[str, int]] = [("code", 0)]
    control_parentheses: list[bool] = []
    control_condition_closes: set[int] = set()
    block_braces: list[bool] = []
    block_closes: set[int] = set()
    index = 0
    while index < len(source):
        mode, depth = contexts[-1]
        char = source[index]

        if mode == "template":
            if char == "\\":
                index += 2
            elif char == "`":
                contexts.pop()
                index += 1
            elif source.startswith("${", index):
                contexts.append(("expression", 1))
                index += 2
            else:
                index += 1
            continue

        if char.isspace():
            if char in _JS_LINE_TERMINATORS and (not tokens or tokens[-1][1] != "newline"):
                if char == "\r" and index + 1 < len(source) and source[index + 1] == "\n":
                    index += 1
                tokens.append(("newline", "newline"))
            index += 1
            continue
        if source.startswith("//", index):
            index += 2
            while index < len(source) and source[index] not in _JS_LINE_TERMINATORS:
                index += 1
            if not tokens or tokens[-1][1] != "newline":
                tokens.append(("newline", "newline"))
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            comment_end = len(source) if end < 0 else end + 2
            if any(char in _JS_LINE_TERMINATORS for char in source[index:comment_end]) and (
                not tokens or tokens[-1][1] != "newline"
            ):
                tokens.append(("newline", "newline"))
            index = comment_end
            continue
        if char in "'\"":
            value, index = _read_js_string(source, index)
            tokens.append(("string", value))
            continue
        if char == "`":
            static_template = _read_static_js_template(source, index)
            if static_template is None:
                contexts.append(("template", 0))
                index += 1
            else:
                value, index = static_template
                tokens.append(("string", value))
            continue
        if char == "/":
            previous = tokens[-1][1] if tokens else ""
            previous_significant = previous
            previous_significant_index = len(tokens) - 1
            if previous == "newline":
                previous_significant_index -= 1
                previous_significant = tokens[previous_significant_index][1] if previous_significant_index >= 0 else ""
            regex_context = previous_significant in (
                _JS_EXPRESSION_OPERATORS
                | {
                    "",
                    "(",
                    "[",
                    "{",
                    "=",
                    ":",
                    ",",
                    ";",
                    "=>",
                }
            ) or previous_significant in _JS_EXPRESSION_KEYWORDS
            if previous_significant == ")":
                regex_context = previous_significant_index in control_condition_closes
            elif previous_significant == "}":
                regex_context = previous_significant_index in block_closes
            if regex_context:
                end = _skip_js_regex(source, index)
                tokens.append(("literal", "regex"))
                index = end
                continue
        if char.isalpha() or char in "_$":
            end = index + 1
            while end < len(source) and (source[end].isalnum() or source[end] in "_$"):
                end += 1
            tokens.append(("identifier", source[index:end]))
            index = end
            continue
        if char.isdigit():
            end = index + 1
            while end < len(source) and (source[end].isalnum() or source[end] in "._"):
                end += 1
            tokens.append(("number", source[index:end]))
            index = end
            continue

        operator = next(
            (
                candidate
                for candidate in (
                    ">>>",
                    "===",
                    "!==",
                    "...",
                    "**",
                    "=>",
                    "?.",
                    "&&",
                    "||",
                    "??",
                    "==",
                    "!=",
                    "<=",
                    ">=",
                    "<<",
                    ">>",
                )
                if source.startswith(candidate, index)
            ),
            char,
        )
        if mode == "expression" and operator == "}":
            if depth == 1:
                contexts.pop()
                index += 1
                continue
            contexts[-1] = (mode, depth - 1)
        elif mode == "expression" and operator == "{":
            contexts[-1] = (mode, depth + 1)
        previous_significant_index = len(tokens) - 1
        if previous_significant_index >= 0 and tokens[previous_significant_index][1] == "newline":
            previous_significant_index -= 1
        previous_significant = (
            tokens[previous_significant_index][1] if previous_significant_index >= 0 else ""
        )
        if operator == "(":
            control_parentheses.append(previous_significant in {"if", "for", "while", "switch", "with"})
        elif operator == ")" and control_parentheses and control_parentheses.pop():
            control_condition_closes.add(len(tokens))
        if operator == "{":
            block_braces.append(
                previous_significant in {"", ";", "newline", "{", "}", "do", "else", "finally", "try"}
                or previous_significant == "=>"
                or (
                    previous_significant == ")"
                    and previous_significant_index in control_condition_closes
                )
            )
        elif operator == "}" and block_braces and block_braces.pop():
            block_closes.add(len(tokens))
        tokens.append(("punctuation", operator))
        index += len(operator)
    return tokens


def _skip_newlines(tokens: list[tuple[str, str]], cursor: int) -> int:
    while cursor < len(tokens) and tokens[cursor][1] == "newline":
        cursor += 1
    return cursor


def _global_tools_references(tokens: list[tuple[str, str]]) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans for possible Codex ``tools`` expressions."""
    references: list[tuple[int, int]] = []
    for index, (kind, value) in enumerate(tokens):
        if kind == "identifier" and value == "tools":
            before = index - 1
            while before >= 0 and tokens[before][1] == "newline":
                before -= 1
            after = _skip_newlines(tokens, index + 1)
            before_value = tokens[before][1] if before >= 0 else ""
            after_value = tokens[after][1] if after < len(tokens) else ""
            if after_value == ":" and before_value not in {"?", "case"}:
                continue
            if after_value == "(" or before_value in {"#", ".", "?."}:
                continue
            references.append((index, index + 1))
            continue
        if kind != "identifier" or value != "globalThis":
            continue
        cursor = _skip_newlines(tokens, index + 1)
        while cursor < len(tokens) and tokens[cursor][1] == ")":
            cursor = _skip_newlines(tokens, cursor + 1)
        if cursor < len(tokens) and tokens[cursor][1] in {".", "?."}:
            member = _skip_newlines(tokens, cursor + 1)
            if member < len(tokens) and tokens[member] == ("identifier", "tools"):
                references.append((index, member + 1))
                continue
            if cursor + 1 < len(tokens) and tokens[cursor][1] == "?.":
                cursor = member
        if cursor + 2 < len(tokens) and tokens[cursor][1] == "[":
            key = _skip_newlines(tokens, cursor + 1)
            parentheses = 0
            while key < len(tokens) and tokens[key][1] == "(":
                parentheses += 1
                key = _skip_newlines(tokens, key + 1)
            if key >= len(tokens) or tokens[key] != ("string", "tools"):
                continue
            close = _skip_newlines(tokens, key + 1)
            while parentheses and close < len(tokens) and tokens[close][1] == ")":
                parentheses -= 1
                close = _skip_newlines(tokens, close + 1)
            if not parentheses and close < len(tokens) and tokens[close][1] == "]":
                references.append((index, close + 1))
    return references


def _skip_balanced_member(tokens: list[tuple[str, str]], start: int, ends: dict[int, int]) -> int:
    end = ends.get(start)
    return len(tokens) if end is None else end + 1


def _tools_reference_is_called(
    tokens: list[tuple[str, str]], end: int, bracket_ends: dict[int, int]
) -> bool:
    cursor = end
    saw_member = False
    while cursor < len(tokens):
        cursor = _skip_newlines(tokens, cursor)
        while cursor < len(tokens) and tokens[cursor][1] == ")":
            cursor = _skip_newlines(tokens, cursor + 1)
        if cursor < len(tokens) and tokens[cursor][1] == "(" and saw_member:
            return True
        if cursor < len(tokens) and tokens[cursor][1] in {".", "?."}:
            member = _skip_newlines(tokens, cursor + 1)
            if member < len(tokens) and tokens[member][1] == "(":
                return saw_member and tokens[cursor][1] == "?."
            if member < len(tokens) and tokens[member][1] == "[" and tokens[cursor][1] == "?.":
                cursor = member
                continue
            if member >= len(tokens) or tokens[member][0] != "identifier":
                return False
            saw_member = True
            cursor = member + 1
            continue
        if cursor < len(tokens) and tokens[cursor][1] == "[":
            saw_member = True
            cursor = _skip_balanced_member(tokens, cursor, bracket_ends)
            continue
        return False
    return False


def _matching_delimiters(tokens: list[tuple[str, str]], opening: str, closing: str) -> dict[int, int]:
    stack: list[int] = []
    matches: dict[int, int] = {}
    for index, (_, value) in enumerate(tokens):
        if value == opening:
            stack.append(index)
        elif value == closing and stack:
            matches[stack.pop()] = index
    return matches


def _reference_contexts(
    tokens: list[tuple[str, str]], references: list[tuple[int, int]]
) -> list[bool]:
    results: list[bool] = []
    reference_index = 0
    strong_context = False
    last_identifier = ""
    previous = ""
    bracket_ends = _matching_delimiters(tokens, "[", "]")
    call_prefixes = _JS_EXPRESSION_OPERATORS | {"", "(", ",", "[", "{", ":", ";", "=>", "newline", "}"}
    line_continuations = _JS_EXPRESSION_OPERATORS | {"(", "[", "{", ",", ".", "?.", ":", "=>"}
    for index in range(len(tokens) + 1):
        while reference_index < len(references) and references[reference_index][0] == index:
            _, end = references[reference_index]
            called = _tools_reference_is_called(tokens, end, bracket_ends)
            immediate_expression_context = called and previous in call_prefixes
            results.append(
                strong_context
                or immediate_expression_context
                or last_identifier in _JS_EXPRESSION_KEYWORDS
            )
            reference_index += 1
        if index == len(tokens):
            break
        kind, value = tokens[index]
        if value in {";", "newline"}:
            if value == "newline" and previous in line_continuations:
                continue
            strong_context = False
            last_identifier = ""
            previous = value
            continue
        if value in {"=", "await", "=>"}:
            strong_context = True
        if previous in {"if", "for", "while", "switch", "with"} and value == "(":
            strong_context = True
        if kind == "identifier":
            last_identifier = value
        previous = value
    return results


def _has_codex_tool_reference(source: str) -> bool:
    """Detect executable global Codex-tool references without evaluating JavaScript."""
    if _CODEX_TOOL_REF_RE.search(source) is None:
        return False
    decoded = _decode_js_identifier_escapes(source)
    tokens = _js_code_tokens(decoded)
    references = _global_tools_references(tokens)
    return any(_reference_contexts(tokens, references))


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
        if not (_CODEX_PRAGMA_PREFIX_RE.match(source) or _has_codex_tool_reference(source)):
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
                "_atif_outer_observation_owner": observation_owner,
            }
        )
    return normalized
