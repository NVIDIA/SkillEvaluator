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


def atif_content_text(content: object) -> str:
    """Return the text of an ATIF message or observation content value.

    ATIF content is a string or a list of content parts. Text parts are joined;
    image and audio parts become ``[image]``/``[audio]`` markers so their
    agent-controlled source paths never reach judges or reports.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                kind = part.get("type")
                if kind == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif kind in ("image", "audio"):
                    parts.append(f"[{kind}]")
        return "\n".join(parts)
    return str(content)


def atif_content_has_text(content: object) -> bool:
    """Check for meaningful text rather than media-only placeholders.

    Keep atif_content_text unchanged: other evidence sections still need
    safe [image]/[audio] markers for non-text media.
    """
    if isinstance(content, list):
        return any(
            (isinstance(part, str) and bool(part.strip()))
            or (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
                and bool(part["text"].strip())
            )
            for part in content
        )
    return bool(atif_content_text(content).strip())


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
        atif_content_text(result.get("content"))
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


def normalized_tool_call_error_flag(step: dict[str, Any], tool_call: dict[str, Any]) -> bool | None:
    """Return an attributable tool result's explicit ``tool_result_is_error`` flag.

    ``True`` if any attributable result reports an error, ``False`` if they
    explicitly report success, and ``None`` when none carries the flag.
    """
    if tool_call.get("_atif_observation_status") not in (None, MAPPED_OUTER_EXEC_OBSERVATION):
        return None
    tool_call_id = tool_call.get("tool_call_id")
    flag = None
    for result in (step.get("observation") or {}).get("results") or []:
        source_call_id = result.get("source_call_id")
        # Unscoped observations cannot prove that this particular call failed.
        if not tool_call_id or source_call_id != tool_call_id:
            continue
        extra = result.get("extra") or {}
        if not isinstance(extra, dict):
            continue
        if extra.get("tool_result_is_error") is True:
            return True
        if extra.get("tool_result_is_error") is False:
            flag = False
    return flag


def normalized_tool_call_is_error(step: dict[str, Any], tool_call: dict[str, Any]) -> bool:
    """Preserve explicit failure metadata only for an attributable tool result."""
    return normalized_tool_call_error_flag(step, tool_call) is True


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


# ── Extended static grammar for inline and Promise.allSettled wrappers (#235) ──
# Codex 0.161.0 also emits `text(await tools.NAME(...))`, single-quoted strings and literal
# `Promise.allSettled` batches. This grammar decodes only those static forms, without evaluating
# JavaScript, and runs only where the grammar above returned unsupported. A wrapper with more
# than one call is decoded only when each call renders one JSON record of the outer output: the
# call owns that record, and a missing, extra, reordered, changed or truncated record leaves the
# whole wrapper unsupported.

_RECORD_FORMAT = "_atif_record_format"
_RECORD_INDEX = "_atif_record_index"
_RECORD_OBSERVATION = "_atif_record_observation"
_RECORD_IS_ERROR = "_atif_record_is_error"
_EXEC_OUTPUT_RE = re.compile(r"Script completed\r?\nWall time [^\r\n]*\r?\nOutput:\r?\n\r?\n")
_JS_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w$.])")
_JS_NAME_RE = re.compile(rf"{_JS_IDENTIFIER}(?![\w$\\])")
# Names a binding may not take: the globals the grammar relies on, and reserved words.
_RESERVED_WORDS = (
    "Infinity JSON NaN Promise arguments async await break case catch class const continue debugger "
    "default delete do else enum eval export extends false finally for function globalThis if implements "
    "import in instanceof interface let new null of package private protected public return static super "
    "switch text this throw tools true try typeof undefined var void while with yield"
)
_RESERVED_BINDINGS = frozenset(_RESERVED_WORDS.split())


class _NotStatic(Exception):
    """The wrapper source is outside the extended static grammar."""


class _StaticExecSource:
    """A cursor over one native Codex wrapper that accepts literal tokens only."""

    def __init__(self, source: str) -> None:
        self.source = source
        pragma = _CODEX_PRAGMA_RE.match(source)
        self.index = pragma.end() if pragma else 0

    def at_end(self) -> bool:
        while self.index < len(self.source) and self.source[self.index].isspace():
            self.index += 1
        return self.index == len(self.source)

    def peek(self) -> str:
        self.at_end()
        return self.source[self.index : self.index + 1]

    def take(self, token: str) -> bool:
        if self.at_end() or not self.source.startswith(token, self.index):
            return False
        end = self.index + len(token)
        if (
            token[-1].isalnum()
            and end < len(self.source)
            and (self.source[end].isalnum() or self.source[end] in "_$\\")
        ):
            return False
        self.index = end
        return True

    def expect(self, *tokens: str) -> None:
        for token in tokens:
            if not self.take(token):
                raise _NotStatic(token)

    def name(self) -> str:
        self.at_end()
        match = _JS_NAME_RE.match(self.source, self.index)
        if match is None:
            raise _NotStatic("identifier")
        self.index = match.end()
        return match.group(0)

    def binding(self, taken: set[str]) -> str:
        """Read a new binding; reusing or shadowing a name is rejected."""
        name = self.name()
        if name in _RESERVED_BINDINGS or name in taken:
            raise _NotStatic(name)
        taken.add(name)
        return name

    def string(self) -> str:
        """Decode one plain single- or double-quoted string that uses JSON escapes only."""
        quote = self.source[self.index]
        end = self.index + 1
        while end < len(self.source) and self.source[end] != quote:
            end += 2 if self.source[end] == "\\" else 1
        if end >= len(self.source):
            raise _NotStatic("string")
        body = self.source[self.index + 1 : end]
        if quote == "'":
            body = re.sub(r"\\'|\\.|\"", lambda m: {"\\'": "'", '"': '\\"'}.get(m.group(0), m.group(0)), body)
        try:
            value = json.loads(f'"{body}"')
        except (json.JSONDecodeError, ValueError) as exc:
            raise _NotStatic("string") from exc
        self.index = end + 1
        return value

    def literal(self, depth: int = 0) -> Any:
        """Decode a JSON-compatible literal: no templates, spreads, computed keys or comments."""
        char = self.peek()
        if depth > _MAX_OBJECT_NESTING or not char:
            raise _NotStatic("literal")
        if char in "\"'":
            return self.string()
        if char in "{[":
            self.index += 1
            items: list[Any] = []
            value: dict[str, Any] = {}
            while not self.take("}" if char == "{" else "]"):
                if items or value:
                    self.expect(",")
                if char == "[":
                    items.append(self.literal(depth + 1))
                    continue
                key = self.string() if self.peek() in ("'", '"') else self.name()
                if key in value or key == "__proto__":
                    raise _NotStatic(key)
                self.expect(":")
                value[key] = self.literal(depth + 1)
            return value if char == "{" else items
        for word, word_value in (("true", True), ("false", False), ("null", None)):
            if self.take(word):
                return word_value
        number = _JS_NUMBER_RE.match(self.source, self.index)
        if number is None:
            raise _NotStatic("literal")
        try:
            number_value = json.loads(number.group(0))
        except ValueError as exc:  # an integer beyond Python's digit limit
            raise _NotStatic("number") from exc
        self.index = number.end()
        return number_value

    def tool_call(self) -> tuple[str, dict[str, Any]]:
        """Read `tools.NAME(<object literal>)` or `tools.apply_patch(<string literal>)`."""
        self.expect("tools", ".")
        function_name = self.name()
        self.expect("(")
        if function_name == "apply_patch" and self.peek() in ("'", '"'):
            # Codex records its freeform apply_patch tool with the patch as `input`.
            arguments: Any = {"input": self.string()}
        else:
            arguments = self.literal()
            if not isinstance(arguments, dict):
                raise _NotStatic("arguments")
        self.expect(")")
        return function_name, arguments

    def end_statement(self) -> None:
        if not self.take(";") and not self.at_end():
            raise _NotStatic(";")


def _settled_renderer(cursor: _StaticExecSource, results: str, taken: set[str]) -> tuple[str, ...]:
    """Read the one renderer of a settled batch and return the format of the records it prints.

    Accepted: `for (let i = 0; i < results.length; i++) text({i, ...results[i]})`,
    `results.forEach((r, i) => text({i, ...r}))` (both also with `{i, result: ...}` and a braced
    body), and `results.forEach(text)` or `results.forEach(r => text(r))`.
    """

    def payload(index: str, element) -> tuple[str, ...]:
        if index in ("status", "value", "reason"):
            raise _NotStatic(index)
        cursor.expect("text", "(", "{", index, ",")
        if cursor.take("..."):
            element()
            record_format: tuple[str, ...] = ("settled_spread", index)
        else:
            key = cursor.name()
            if key == index:
                raise _NotStatic(key)
            cursor.expect(":")
            element()
            record_format = ("settled_field", index, key)
        cursor.expect("}", ")")
        return record_format

    def body(render) -> tuple[str, ...]:
        braced = cursor.take("{")
        record_format = render()
        if braced:
            cursor.take(";")
            cursor.expect("}")
        return record_format

    def render_element(element: str) -> tuple[str, ...]:
        cursor.expect("text", "(", element, ")")
        return ("settled",)

    if cursor.take("for"):
        cursor.expect("(", "let")
        index = cursor.binding(taken)
        cursor.expect("=", "0", ";", index, "<", results, ".", "length", ";", index, "++", ")")
        return body(lambda: payload(index, lambda: cursor.expect(results, "[", index, "]")))
    cursor.expect(results, ".", "forEach", "(")
    if cursor.take("text"):
        cursor.expect(")")
        return ("settled",)
    parenthesized = cursor.take("(")
    element = cursor.binding(taken)
    index = cursor.binding(taken) if parenthesized and cursor.take(",") else None
    if parenthesized:
        cursor.expect(")")
    cursor.expect("=>")
    if index is None:
        record_format = body(lambda: render_element(element))
    else:
        record_format = body(lambda: payload(index, lambda: cursor.expect(element)))
    cursor.expect(")")
    return record_format


def _extended_codex_program(source: str) -> tuple[list[tuple[str, dict[str, Any]]], int | None, tuple] | None:
    """Decode the extended grammar into ``(calls, owner, record_format)``, or None outside it.

    A single call keeps the upstream ownership: ``owner`` 0 owns the whole outer observation,
    -1 renders nothing. With records, ``owner`` is None and ``record_format`` says how to split
    the outer observation; for inline renders it lists the call of each record in output order.
    """
    cursor = _StaticExecSource(source)
    calls: list[tuple[str, dict[str, Any]]] = []
    bindings: dict[str, int] = {}
    renders: list[tuple[int, bool]] = []
    taken: set[str] = set()
    statements = 0
    try:
        while not cursor.at_end():
            statements += 1
            if statements > _MAX_STATEMENTS or len(calls) >= _MAX_TOOL_CALLS:
                return None
            if cursor.take("const"):
                variable = cursor.binding(taken)
                cursor.expect("=", "await")
                if cursor.take("Promise"):
                    if calls:
                        return None
                    cursor.expect(".", "allSettled", "(", "[")
                    while not cursor.take("]"):
                        if calls:
                            cursor.expect(",")
                            if cursor.take("]"):
                                break
                        if len(calls) >= _MAX_TOOL_CALLS:
                            return None
                        calls.append(cursor.tool_call())
                    cursor.expect(")")
                    cursor.end_statement()
                    record_format = _settled_renderer(cursor, variable, taken)
                    cursor.end_statement()
                    return (calls, None, record_format) if calls and cursor.at_end() else None
                bindings[variable] = len(calls)
                calls.append(cursor.tool_call())
            else:
                cursor.expect("text", "(")
                if cursor.take("("):
                    cursor.expect("await")
                    renders.append((len(calls), False))
                    calls.append(cursor.tool_call())
                    cursor.expect(")", ".")
                    cursor.name()
                elif cursor.take("await"):
                    renders.append((len(calls), True))
                    calls.append(cursor.tool_call())
                elif cursor.take("JSON"):
                    cursor.expect(".", "stringify", "(")
                    renders.append((bindings[cursor.name()], True))
                    cursor.expect(")")
                else:
                    call_index = bindings[cursor.name()]
                    whole = not cursor.take(".")
                    if not whole:
                        cursor.name()
                    renders.append((call_index, whole))
                cursor.expect(")")
            cursor.end_statement()
    except (_NotStatic, KeyError, RecursionError):
        return None
    if len(calls) == 1:
        return calls, 0 if renders else -1, ()
    owners = [call_index for call_index, _ in renders]
    if not calls or not all(whole for _, whole in renders) or sorted(owners) != list(range(len(calls))):
        return None
    return calls, None, ("result", *owners)


def _settled_result(value: object) -> dict[str, Any] | None:
    """Return a `Promise.allSettled` result with exactly its own keys, else None."""
    if not isinstance(value, dict):
        return None
    if value.get("status") == "fulfilled" and set(value) == {"status", "value"}:
        return value
    if value.get("status") == "rejected" and set(value) == {"status", "reason"}:
        return value
    return None


def _exec_record(value: object, record_format: tuple, record_index: int) -> bool | None:
    """Check one record against its format; return whether it is a rejected promise."""
    kind = record_format[0]
    if kind == "result":
        return False if isinstance(value, dict) else None
    if kind == "settled":
        settled = _settled_result(value)
    elif (
        not isinstance(value, dict)
        or type(value.get(record_format[1])) is not int
        or value[record_format[1]] != record_index
    ):
        return None
    elif kind == "settled_spread":
        settled = _settled_result({key: item for key, item in value.items() if key != record_format[1]})
    else:
        settled = _settled_result(value.get(record_format[2])) if len(value) == 2 else None
    return None if settled is None else settled["status"] == "rejected"


def _codex_exec_records(observation: str, record_format: tuple, count: int) -> list[tuple[str, bool]] | None:
    """Split a completed outer observation into exactly ``count`` checked records, in output order."""
    header = _EXEC_OUTPUT_RE.match(observation)
    if header is None:
        return None
    body = observation[header.end() :]
    decoder = json.JSONDecoder()
    position = 0
    records: list[tuple[str, bool]] = []
    for record_index in range(count):
        if record_index:
            if body.startswith("\r\n", position):
                position += 1
            if not body.startswith("\n", position):
                return None
            position += 1
        try:
            value, end = decoder.raw_decode(body, position)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return None
        is_error = _exec_record(value, record_format, record_index)
        if is_error is None:
            return None
        records.append((body[position:end], is_error))
        position = end
    return records if not body[position:].strip() else None


_upstream_normalize_tool_call = normalize_tool_call
_upstream_normalized_tool_call_observation = normalized_tool_call_observation
_upstream_normalized_tool_call_is_error = normalized_tool_call_is_error
_upstream_normalized_tool_call_error_flag = normalized_tool_call_error_flag


def _normalize_tool_call_with_records(tool_call: dict[str, Any]) -> list[dict[str, Any]]:
    """Unwrap proven native Codex calls, adding the inline and allSettled forms of #235."""
    normalized = _upstream_normalize_tool_call(tool_call)
    if len(normalized) != 1 or normalized[0].get("_atif_normalization_status") != UNSUPPORTED_NATIVE_CODEX_EXEC:
        return normalized
    source = normalized[0]["arguments"]["input"]
    parsed = _extended_codex_program(source) if len(source) <= _MAX_SOURCE_CHARS else None
    if parsed is None:
        return normalized
    calls, observation_owner, record_format = parsed
    outer = {key: value for key, value in normalized[0].items() if key != "_atif_normalization_status"}
    result: list[dict[str, Any]] = []
    for inner_index, (function_name, inner_arguments) in enumerate(calls):
        call = {
            **outer,
            "function_name": function_name,
            "arguments": inner_arguments,
            "_atif_inner_tool_index": inner_index,
            "_atif_outer_observation_owner": observation_owner,
        }
        if observation_owner is not None:
            call["_atif_observation_status"] = (
                MAPPED_OUTER_EXEC_OBSERVATION if observation_owner == inner_index else UNOBSERVED_INNER_CALL
            )
        else:
            # Ambiguous until the outer observation proves which record belongs to this call.
            call["_atif_observation_status"] = AMBIGUOUS_OUTER_EXEC_OBSERVATION
            call[_RECORD_FORMAT] = record_format
            call[_RECORD_INDEX] = record_format[1:].index(inner_index) if record_format[0] == "result" else inner_index
        result.append(call)
    return result


def _resolve_exec_records(
    step: dict[str, Any], tool_call: dict[str, Any], raw_index: int, normalized: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Give each call of a record wrapper its own record, or leave the wrapper unsupported."""
    observation = _outer_tool_call_observation(step, tool_call, include_unscoped=raw_index == 0)
    records = _codex_exec_records(observation, normalized[0][_RECORD_FORMAT], len(normalized))
    if records is None:
        return _upstream_normalize_tool_call(tool_call)
    return [
        {
            **call,
            "_atif_observation_status": MAPPED_OUTER_EXEC_OBSERVATION,
            "_atif_outer_observation_owner": call["_atif_inner_tool_index"],
            _RECORD_OBSERVATION: records[call[_RECORD_INDEX]][0],
            _RECORD_IS_ERROR: records[call[_RECORD_INDEX]][1],
        }
        for call in normalized
    ]


def _iter_normalized_tool_calls_with_records(traj: dict[str, Any]):
    """Yield each trajectory tool call after safe native-Codex normalization."""
    for step in traj.get("steps", []):
        for raw_index, tool_call in enumerate(step.get("tool_calls") or []):
            normalized = normalize_tool_call(tool_call)
            if normalized and _RECORD_FORMAT in normalized[0]:
                normalized = _resolve_exec_records(step, tool_call, raw_index, normalized)
            for call in normalized:
                yield step, {**call, "_atif_raw_tool_index": raw_index}


def _normalized_tool_call_observation_with_records(step: dict[str, Any], tool_call: dict[str, Any]) -> str:
    """Return a call's own record, else the outer observation when its owner is known."""
    if tool_call.get("_atif_observation_status") == MAPPED_OUTER_EXEC_OBSERVATION and _RECORD_OBSERVATION in tool_call:
        return tool_call[_RECORD_OBSERVATION]
    return _upstream_normalized_tool_call_observation(step, tool_call)


def _normalized_tool_call_error_flag_with_records(
    step: dict[str, Any], tool_call: dict[str, Any]
) -> bool | None:
    """Report a call's own record outcome before falling back to the upstream flag."""
    if (
        tool_call.get("_atif_observation_status") == MAPPED_OUTER_EXEC_OBSERVATION
        and _RECORD_IS_ERROR in tool_call
    ):
        return bool(tool_call[_RECORD_IS_ERROR])
    return _upstream_normalized_tool_call_error_flag(step, tool_call)


def _normalized_tool_call_is_error_with_records(step: dict[str, Any], tool_call: dict[str, Any]) -> bool:
    """Also report the explicit failure of a call's own record: a rejected promise."""
    if (
        tool_call.get("_atif_observation_status") == MAPPED_OUTER_EXEC_OBSERVATION
        and _RECORD_IS_ERROR in tool_call
    ):
        return bool(tool_call[_RECORD_IS_ERROR])
    return _upstream_normalized_tool_call_is_error(step, tool_call)


normalize_tool_call = _normalize_tool_call_with_records
iter_normalized_tool_calls = _iter_normalized_tool_calls_with_records  # noqa: F811
normalized_tool_call_observation = _normalized_tool_call_observation_with_records
normalized_tool_call_error_flag = _normalized_tool_call_error_flag_with_records
normalized_tool_call_is_error = _normalized_tool_call_is_error_with_records
