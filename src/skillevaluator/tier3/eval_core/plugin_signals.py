# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only plugin component signals computed on the host from ATIF trajectories.

The Harbor collector calls :func:`compute_plugin_signals` for each scored trial
of a *plugin* evaluation and :func:`summarize_plugin_signals` per arm. Nothing
here changes a trial score, pass/fail result, or verdict: every value is
advisory and is persisted next to the existing reward/summary data.

Component activation conventions
--------------------------------
Each normalized tool call is mapped to zero or more component activations:

* ``skill``    -- the ``Skill`` tool (name from ``skill``/``name``/``command``),
  or a read of a *declared* member's ``<member>/SKILL.md`` through a file-read
  tool (``Read``/``read_file``/``view``...), an MCP filesystem read, or a shell
  reader command (``cat``/``sed``/``head``...). Shell reads are recognized only
  when the manifest is an operand of a known reader verb, so ``rm``/``ls``/
  ``grep``/``printf`` of the same path never count.
* ``mcp``      -- ``mcp__<server>__<tool>``; for *declared* runnable servers the
  ``<server>__<tool>``/``<server>.<tool>``/``<server>/<tool>`` spellings used by
  other agents, Hermes ``mcp_<server>_<tool>`` and OpenCode ``<server>_<tool>``
  are also recognized and canonicalized to ``mcp__<server>__<tool>``. Claude
  Code's plugin servers (``mcp__plugin_<plugin>_<server>__<tool>``) and
  harness-sanitized names map back to the declared server; Codex bare tool
  names map through the server its log recorded (``mcp_call_servers``).
* ``subagent`` -- ``Task``/``Agent`` (name from ``subagent_type``...).
* ``command``  -- ``SlashCommand`` (name from the first token of ``command``),
  or the ``Skill`` tool naming a declared command (``<plugin>:<command>``),
  which is how Claude Code runs plugin commands.
* ``rule_read`` is reserved by the output contract but never emitted: plugin
  rules are inlined into the generated wrapper skill, so there is no separate
  rule file for an agent to read.

Native Codex ``exec`` wrappers are unwrapped with
:mod:`~skillevaluator.tier3.eval_core.codex_tool_call_normalizer` first. An
inner call only inherits the outer observation when the normalizer proves it
owns that observation; otherwise its outcome is unknown.

``succeeded`` is tri-state. ``True``: a correlated result came back without a
structured error flag or a failure/unavailable marker in the head of any of its
content blocks or in its tail. ``False``: the correlated result is flagged or
carries such a marker ("no such file" only counts for non-shell calls, whose
output is not a mix of several commands). A file read or skill load returns
file text, so only its first line counts (past a harness's shell status lines),
plus, for a shell read, any shell error line that names a ``SKILL.md``. ``None``:
no result could be attributed to this call (no id match, or an ambiguous
multi-call step), or the correlated body is empty. A sibling call's result never
stands in for this call's outcome.

Case references
---------------
Dataset fields refer to components or tools with *refs*. A ref is a string or a
list of alternative strings. Matching is case-insensitive and supports
``fnmatch`` globs:

* ``Skill:<name>``, ``Agent:<name>`` (aliases ``Subagent:``/``Task:``),
  ``Command:<name>`` (alias ``SlashCommand:``) and ``MCP:<server>`` (or
  ``MCP:<server>/<tool>``) match a component activation of that type;
* any other ref matches the canonical tool label (``mcp__server__tool``,
  ``Skill:<name>``...), the raw tool function name (``Bash``, ``Read``...), or a
  component name.

Handoff heuristics (conservative)
---------------------------------
Calls are *attributed* to a ref when they match it directly, or when they run
inside the window opened by a matching skill/command activation (a window lasts
until the next skill/command activation). Subagent work happens outside the
parent trajectory, so a ``Task`` call only contributes its own arguments and
result.

* ``value``: the literal must appear in the result of a producer-attributed
  call, then in the arguments of a consumer-attributed call in a strictly later
  step (a result is only visible to the model after its step). The value must
  not appear in any user/system message, since then the consumer could have
  taken it from the prompt rather than from the producer.
* ``artifact``: a producer-attributed call must write the path (write/edit
  tools, the file headers of an ``apply_patch`` body, shell redirects/``tee``/
  ``cp``/``mv``/``touch``, or an MCP tool whose name or argument key says it
  writes), and a later consumer-attributed call must read it (read tools,
  shell readers/interpreters, or an MCP/subagent call whose arguments name the
  path). Relative artifact paths match an observed path equal to it or ending
  with ``/<artifact>``; absolute ones must match exactly.
* When both are given, both must hold.

Trajectory content is untrusted: step, call, text and regex-subject sizes are
bounded, dataset patterns run under a deadline, and every persisted string is
redacted (from a bounded window) and truncated. Argument values are never
persisted: failure details describe them by type and size, and a component
name taken from tool arguments is kept only when it is identifier-shaped or a
declared member.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
import shlex
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import regex

from skillevaluator.tier3.eval_core.atif_helpers import _patch_file_paths
from skillevaluator.tier3.eval_core.checks import _APPLY_PATCH_COMMAND_RE
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import (
    MAPPED_OUTER_EXEC_OBSERVATION,
    normalize_tool_call,
)
from skillevaluator.utils.redaction import redact_sensitive_text

COMPONENT_SKILL = "skill"
COMPONENT_MCP = "mcp"
COMPONENT_SUBAGENT = "subagent"
COMPONENT_COMMAND = "command"
COMPONENT_RULE_READ = "rule_read"

STATUS_SCORED = "scored"
STATUS_NOT_APPLICABLE = "not_applicable"

ARM_WITH_SKILL = "with_skill"
ARM_WITHOUT_SKILL = "without_skill"
ARM_SUM_OF_PARTS = "sum_of_parts"

PLUGIN_CASE_FIELDS = (
    "expected_tools",
    "acceptable_tools",
    "decoy_tools",
    "tool_arguments",
    "expected_order",
    "handoffs",
    "conflict_probes",
)

# -- dataset field bounds ----------------------------------------------------
MAX_TOOL_PATTERNS = 64
MAX_REF_CHARS = 256
MAX_REF_ALTERNATIVES = 16
MAX_ARGUMENT_RULES = 32
MAX_ARGUMENT_NAMES = 64
MAX_ARGUMENT_NAME_CHARS = 128
MAX_SCHEMA_DEPTH = 6
MAX_SCHEMA_NODES = 256
MAX_ENUM_ITEMS = 64
MAX_REGEX_CHARS = 512
MAX_LITERAL_JSON_CHARS = 4096
MAX_CONTAINS_CHARS = 1024
MAX_ORDER_EDGES = 64
MAX_HANDOFFS = 32
MAX_HANDOFF_VALUE_CHARS = 1024
MAX_ARTIFACT_CHARS = 512
MAX_CONFLICT_PROBES = 32
MAX_PROBE_ID_CHARS = 128
MAX_DESCRIPTION_CHARS = 1024
MAX_CONTEXT_CASES = 4096
MAX_DECLARED_COMPONENTS = 256
MAX_COMPONENT_NAME_CHARS = 256

# -- trajectory bounds ---------------------------------------------------------
_MAX_STEPS = 10_000
_MAX_CALLS = 4_000
_MAX_OBSERVATION_CHARS = 64 * 1024
_MAX_ARGS_TEXT_CHARS = 64 * 1024
_MAX_ARGS_JSON_CHARS = 256 * 1024
_MAX_SHELL_TEXT_CHARS = 32 * 1024
_MAX_SHELL_TOKENS = 4_096
_MAX_SHELL_DEPTH = 3
_MAX_PATTERN_SUBJECT_CHARS = 4_096
# Dataset patterns run under a deadline: per check, and shared by all checks of one trial.
_PATTERN_TIMEOUT_SECONDS = 0.1
_PATTERN_BUDGET_SECONDS = 5.0
_FAILURE_SCAN_CHARS = 2_048

# -- output bounds -------------------------------------------------------------
_MAX_ACTIVATIONS = 512
_MAX_FAILURES = 50
_MAX_CALLED = 128
_MAX_SERVER_TOOLS = 64
_MAX_LABEL_CHARS = 256
_MAX_DETAIL_CHARS = 240
_MAX_PREVIEW_CHARS = 80
_MAX_SCHEMA_ERRORS_PER_CALL = 5

_SKILL_TOOLS = frozenset({"skill"})
_SUBAGENT_TOOLS = frozenset({"task", "agent"})
_COMMAND_TOOLS = frozenset({"slashcommand", "slash_command"})
_SHELL_TOOLS = frozenset(
    {
        "bash",
        "shell",
        "exec_command",
        "exec",
        "local_shell",
        "local_shell_call",
        "run",
        "run_command",
        "run_shell_command",
        "execute",
        "execute_command",
        "run_code",
        "command",
        "terminal",
    }
)
_READ_TOOLS = frozenset(
    {"read", "read_file", "read_text_file", "view", "open", "open_file", "cat", "get_file_contents", "view_file"}
)
# Edit tools that take an apply_patch body: Codex ``apply_patch``/``applypatch``
# and the Hermes ``patch`` tool (whose ``replace`` mode names a ``path`` instead).
_PATCH_TOOLS = frozenset({"apply_patch", "applypatch", "patch"})
# Where harnesses put that body: Codex ``input``, OpenCode ``patchText``, Hermes
# ``patch``, and ``raw`` for a tool input that was not a JSON object.
_PATCH_BODY_KEYS = ("input", "patch", "patchText", "content", "raw")
_WRITE_TOOLS = _PATCH_TOOLS | frozenset(
    {
        "write",
        "write_file",
        "create_file",
        "edit",
        "edit_file",
        "multiedit",
        "multi_edit",
        "notebookedit",
        "notebook_edit",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "save_file",
    }
)
_PATH_ARG_KEYS = (
    "file_path",
    "path",
    "filename",
    "target_file",
    "filePath",
    "absolute_path",
    "notebook_path",
    "raw",
)
_SHELL_COMMAND_KEYS = ("cmd", "command", "code", "script", "input")
_TOOL_NAME_SEPARATORS = ("__", ".", ":", "/")
# MCP tool-name spellings (see ``_mcp_identity``). Hermes names MCP tools
# ``mcp_<server>_<tool>`` with every character outside ``[A-Za-z0-9_]`` as ``_``;
# Claude Code names plugin servers ``plugin_<plugin>_<server>``.
_HERMES_NAME_RE = re.compile(r"[^a-z0-9_]")
_CLAUDE_PLUGIN_SERVER_PREFIX = "plugin_"
# Harnesses that never name an MCP tool ``<server>_<tool>`` (only OpenCode does).
_NO_BARE_MCP_PREFIX_AGENTS = frozenset({"claude-code", "codex", "hermes"})
# Built-in tool names that a declared server called ``web``/``read``/... must not claim as ``<server>_<tool>``.
_BUILTIN_TOOL_NAMES = (
    _SKILL_TOOLS
    | _SUBAGENT_TOOLS
    | _COMMAND_TOOLS
    | _SHELL_TOOLS
    | _READ_TOOLS
    | _WRITE_TOOLS
    | frozenset(
        {
            "web_search",
            "web_fetch",
            "web_extract",
            "update_plan",
            "write_stdin",
            "view_image",
            "execute_code",
            "search_files",
            "todo_write",
            "todo_read",
        }
    )
)
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
_SHELL_PREFIX_WORDS = frozenset({"sudo", "env", "time", "nohup", "command", "exec", "builtin", "nice"})
_FILE_READER_VERBS = frozenset(
    {"cat", "bat", "batcat", "less", "more", "head", "tail", "view", "nl", "sed", "awk", "xxd", "od", "tac"}
)
_ARTIFACT_CONSUMER_VERBS = _FILE_READER_VERBS | frozenset(
    {"jq", "yq", "python", "python3", "node", "grep", "rg", "wc", "sort", "uniq", "cut", "diff", "cmp", "base64"}
)
_SHELL_SEPARATORS = frozenset({"&&", "||", ";", "|", "&", "(", ")", ";;", "|&", ";&"})
_OUTPUT_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", "1>", "2>"})
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# ``<<EOF``/``<<-'EOF'``/``<<"EOF"`` (not ``<<<`` here-strings); the groups hold the delimiter.
_HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?[ \t]*(?:'([^'\n]+)'|\"([^\"\n]+)\"|\\?([A-Za-z_][A-Za-z0-9_]*))")
_WRITE_VERB_RE = re.compile(
    r"(?:^|_|-)(?:write|create|save|export|put|upload|dump|store|generate|render|append)(?:$|_|-)"
)
_OUTPUT_ARG_KEYS = frozenset(
    {"output", "out", "output_path", "out_path", "output_file", "dest", "destination", "target", "save_path"}
)
_READ_VERB_RE = re.compile(r"(?:^|_|-)(?:read|open|get|load|view|fetch|cat)(?:$|_|-)")

_REF_PREFIXES = {
    "skill": COMPONENT_SKILL,
    "agent": COMPONENT_SUBAGENT,
    "subagent": COMPONENT_SUBAGENT,
    "task": COMPONENT_SUBAGENT,
    "command": COMPONENT_COMMAND,
    "slashcommand": COMPONENT_COMMAND,
    "mcp": COMPONENT_MCP,
}
_IDENTITY_PREFIX = {
    COMPONENT_SKILL: "Skill",
    COMPONENT_SUBAGENT: "Agent",
    COMPONENT_COMMAND: "Command",
}
# Skill/subagent/command names come from agent-written tool arguments. Only an
# identifier-shaped name (or a declared member) is persisted; anything else, such
# as a URL with credentials or key material, is reported as ``<non-name>``.
_PERSISTABLE_NAME_RE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}")
_NON_NAME = "<non-name>"

# Conservative markers that a component could not be served at runtime.
_UNAVAILABLE_MARKERS = (
    "no such tool",
    "tool not found",
    "unknown tool",
    "unknown skill",
    "skill not found",
    "no such command",
    "command not found",
    "unknown command",
    "not available",
    "is unavailable",
    "failed to connect",
    "connection refused",
    "mcp server not",
    "no mcp server",
)
# A missing file means a failed call only for MCP and file-read tools. A shell
# call's output mixes several commands, so ``cat SKILL.md missing.md`` read the
# manifest even though its output says "No such file or directory".
_FILE_MISSING_MARKERS = ("no such file or directory", "file does not exist")
# Status lines a harness puts in front of shell output (Codex ``exec_command``).
# They are not file text, so a shell read looks past them for its first line.
_SHELL_STATUS_LINE_RE = re.compile(
    r"chunk id:.*|wall time:.*|process exited with code -?\d+|exit code:\s*-?\d+"
    r"|original token count:.*|total output lines:.*|output:",
    re.IGNORECASE,
)
# A shell tool's error line about a file (``cat: <dir>/SKILL.md: Permission denied``).
_SHELL_ERROR_LINE_RE = re.compile(r"[\w./+-]+:\s")
# A result that came back as a runtime/transport failure rather than an answer.
# A 4xx/5xx status counts at the start of a line ("403: forbidden", "Error: 500
# - boom", "HTTP/1.1 404 - x"), after a request line ("GET /repos/x: 404 - Not
# Found"), or after a space later on a line that names an error ("Request failed
# with status 404: Not Found", "Error calling tool: 404: not found"). Issue
# numbers, file:line references and totals inside a successful answer are not
# failures. Each gap is bounded, so a long line is scanned in linear time.
_FAILED_CALL_RE = re.compile(
    r"(?:^[ \t]*(?:(?:error|http/\d(?:\.\d)?|status(?:[ _]code)?)[ \t:=]*)?[45]\d{2}\b[ \t]*[:\-])"
    r"|(?:^[ \t]*(?:get|head|post|put|patch|delete|options)[ \t]+\S{1,200}[ \t]+[45]\d{2}\b[ \t]*[:\-])"
    r"|(?:\b(?:error|failed|status(?:[ _]code)?)\b[^\n]{0,80}?[ \t][45]\d{2}\b[ \t]*[:\-])"
    r"|<tool_use_error>"
    r"|\bmcp error\b"
    r"|\bpermission denied\b"
    r"|\bunauthorized\b"
    r"|\bforbidden\b"
    r"|\binternal server error\b"
    r"|\bfailed to decrypt\b"
    r"|\binvalid[_ ]token\b"
    r"|\btoken (?:expired|is expired)\b",
    re.IGNORECASE | re.MULTILINE,
)
_SCHEMA_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
_SCHEMA_KEYWORDS = frozenset({"type", "properties", "required", "enum", "pattern", "minimum", "maximum"})
_SCHEMA_ANNOTATIONS = frozenset({"description", "title", "$comment", "examples", "default"})
_ARGUMENT_RULE_KEYS = frozenset({"tool", "required", "schema", "equals", "contains", "pattern", "description"})
_HANDOFF_KEYS = frozenset({"producer", "consumer", "value", "artifact", "description"})
_PROBE_KEYS = frozenset({"id", "must_use", "must_not_use", "description"})
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MISSING = object()


# =============================================================================
# Dataset case fields
# =============================================================================


class _FieldErrors:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, where: str, problem: str) -> None:
        self.messages.append(f"{where}: {problem}")


def _check_description(raw: Mapping[str, Any], where: str, errors: _FieldErrors) -> bool:
    description = raw.get("description")
    if description is not None and (not isinstance(description, str) or len(description) > MAX_DESCRIPTION_CHARS):
        errors.add(f"{where}.description", f"must be a string of at most {MAX_DESCRIPTION_CHARS} characters")
        return False
    return True


def _check_ref_string(value: Any, where: str, errors: _FieldErrors) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.add(where, "must be a non-empty string")
        return None
    text = value.strip()
    if len(text) > MAX_REF_CHARS:
        errors.add(where, f"must be at most {MAX_REF_CHARS} characters")
        return None
    if _CONTROL_CHARS_RE.search(text):
        errors.add(where, "must not contain control characters")
        return None
    prefix, sep, rest = text.partition(":")
    if sep and prefix.strip().casefold() in _REF_PREFIXES and not rest.strip():
        errors.add(where, f"'{prefix}:' must be followed by a name")
        return None
    return text


def _check_ref_alternatives(value: Any, where: str, errors: _FieldErrors) -> list[str] | None:
    if isinstance(value, str):
        ref = _check_ref_string(value, where, errors)
        return [ref] if ref is not None else None
    if not isinstance(value, list) or not value:
        errors.add(where, "must be a ref string or a non-empty list of alternative refs")
        return None
    if len(value) > MAX_REF_ALTERNATIVES:
        errors.add(where, f"must list at most {MAX_REF_ALTERNATIVES} alternatives")
        return None
    refs: list[str] = []
    for index, item in enumerate(value):
        ref = _check_ref_string(item, f"{where}[{index}]", errors)
        if ref is None:
            return None
        refs.append(ref)
    return refs


def _check_tool_patterns(value: Any, where: str, errors: _FieldErrors) -> list[str] | None:
    if not isinstance(value, list):
        errors.add(where, "must be a list of tool refs")
        return None
    if len(value) > MAX_TOOL_PATTERNS:
        errors.add(where, f"must list at most {MAX_TOOL_PATTERNS} tool refs")
        return None
    patterns: list[str] = []
    for index, item in enumerate(value):
        ref = _check_ref_string(item, f"{where}[{index}]", errors)
        if ref is None:
            return None
        if ref not in patterns:
            patterns.append(ref)
    return patterns


def _check_regex(value: Any, where: str, errors: _FieldErrors) -> str | None:
    if not isinstance(value, str) or not value:
        errors.add(where, "must be a non-empty regular expression string")
        return None
    if len(value) > MAX_REGEX_CHARS:
        errors.add(where, f"must be at most {MAX_REGEX_CHARS} characters")
        return None
    # Checked with the engine that runs it (see _regex_search): ``re`` rejects
    # ``\p{L}``, which ``regex`` runs, and accepts ``[[:alpha:]``, which it does not.
    try:
        regex.compile(value)
    except (regex.error, ValueError, RecursionError, OverflowError) as exc:
        errors.add(where, f"is not a valid regular expression ({exc})")
        return None
    return value


def _is_number(value: Any) -> bool:
    # ``float(10**400)`` overflows, so an int is finite by type; floats must be finite.
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def _bounded_json_text(value: Any, limit: int) -> str | None:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return None
    return text if len(text) <= limit else None


def _check_schema(schema: Any, where: str, errors: _FieldErrors, *, depth: int, budget: list[int]) -> bool:
    budget[0] += 1
    if budget[0] > MAX_SCHEMA_NODES:
        errors.add(where, f"schema must have at most {MAX_SCHEMA_NODES} nodes")
        return False
    if depth > MAX_SCHEMA_DEPTH:
        errors.add(where, f"schema nesting must be at most {MAX_SCHEMA_DEPTH} levels")
        return False
    if not isinstance(schema, dict):
        errors.add(where, "schema must be an object")
        return False
    ok = True
    for key in schema:
        if key not in _SCHEMA_KEYWORDS and key not in _SCHEMA_ANNOTATIONS:
            errors.add(
                where,
                f"unsupported JSON-Schema keyword {str(key)[:64]!r}; supported: {', '.join(sorted(_SCHEMA_KEYWORDS))}",
            )
            ok = False
    if "type" in schema:
        types = schema["type"]
        type_list = [types] if isinstance(types, str) else types
        if (
            not isinstance(type_list, list)
            or not type_list
            or not all(isinstance(item, str) and item in _SCHEMA_TYPES for item in type_list)
        ):
            errors.add(f"{where}.type", f"must be one of {', '.join(sorted(_SCHEMA_TYPES))} or a list of them")
            ok = False
    if "required" in schema:
        required = schema["required"]
        if (
            not isinstance(required, list)
            or len(required) > MAX_ARGUMENT_NAMES
            or not all(isinstance(item, str) and 0 < len(item) <= MAX_ARGUMENT_NAME_CHARS for item in required)
        ):
            errors.add(f"{where}.required", "must be a list of property names")
            ok = False
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or len(enum) > MAX_ENUM_ITEMS:
            errors.add(f"{where}.enum", f"must be a non-empty list of at most {MAX_ENUM_ITEMS} values")
            ok = False
        elif _bounded_json_text(enum, MAX_LITERAL_JSON_CHARS) is None:
            errors.add(f"{where}.enum", f"must be JSON values of at most {MAX_LITERAL_JSON_CHARS} characters")
            ok = False
    if "pattern" in schema and _check_regex(schema["pattern"], f"{where}.pattern", errors) is None:
        ok = False
    for bound in ("minimum", "maximum"):
        if bound in schema and not _is_number(schema[bound]):
            errors.add(f"{where}.{bound}", "must be a finite number")
            ok = False
    if (
        _is_number(schema.get("minimum"))
        and _is_number(schema.get("maximum"))
        and schema["minimum"] > schema["maximum"]
    ):
        errors.add(where, "minimum must not exceed maximum")
        ok = False
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict) or len(properties) > MAX_ARGUMENT_NAMES:
            errors.add(f"{where}.properties", f"must be an object with at most {MAX_ARGUMENT_NAMES} properties")
            ok = False
        else:
            for name, sub in properties.items():
                if not isinstance(name, str) or not name or len(name) > MAX_ARGUMENT_NAME_CHARS:
                    errors.add(f"{where}.properties", "property names must be non-empty short strings")
                    ok = False
                    continue
                if not _check_schema(sub, f"{where}.properties.{name}", errors, depth=depth + 1, budget=budget):
                    ok = False
    return ok


def _check_arg_mapping(value: Any, where: str, errors: _FieldErrors, *, kind: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        errors.add(where, "must be a non-empty object mapping argument names to values")
        return None
    if len(value) > MAX_ARGUMENT_NAMES:
        errors.add(where, f"must name at most {MAX_ARGUMENT_NAMES} arguments")
        return None
    parsed: dict[str, Any] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name or len(name) > MAX_ARGUMENT_NAME_CHARS:
            errors.add(where, "argument names must be non-empty strings")
            return None
        item_where = f"{where}.{name}"
        if kind == "equals":
            if _bounded_json_text(item, MAX_LITERAL_JSON_CHARS) is None:
                errors.add(item_where, f"must be a JSON value of at most {MAX_LITERAL_JSON_CHARS} characters")
                return None
        elif kind == "contains":
            if not isinstance(item, str) or not item or len(item) > MAX_CONTAINS_CHARS:
                errors.add(item_where, f"must be a non-empty string of at most {MAX_CONTAINS_CHARS} characters")
                return None
        elif _check_regex(item, item_where, errors) is None:
            return None
        parsed[name] = item
    return parsed


def _parse_tool_arguments(value: Any, errors: _FieldErrors) -> list[dict[str, Any]] | None:
    where = "tool_arguments"
    if not isinstance(value, list):
        errors.add(where, "must be a list of argument rules")
        return None
    if len(value) > MAX_ARGUMENT_RULES:
        errors.add(where, f"must list at most {MAX_ARGUMENT_RULES} rules")
        return None
    rules: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        rule_where = f"{where}[{index}]"
        if not isinstance(raw, dict):
            errors.add(rule_where, "must be an object")
            return None
        unknown = sorted(str(key)[:64] for key in raw if key not in _ARGUMENT_RULE_KEYS)
        if unknown:
            errors.add(rule_where, f"unsupported keys: {', '.join(unknown)}")
            return None
        tool = _check_ref_string(raw.get("tool"), f"{rule_where}.tool", errors)
        if tool is None or not _check_description(raw, rule_where, errors):
            return None
        rule: dict[str, Any] = {"tool": tool}
        if "required" in raw:
            required = raw["required"]
            if (
                not isinstance(required, list)
                or len(required) > MAX_ARGUMENT_NAMES
                or not all(isinstance(item, str) and 0 < len(item) <= MAX_ARGUMENT_NAME_CHARS for item in required)
            ):
                errors.add(f"{rule_where}.required", "must be a list of argument names")
                return None
            if required:
                rule["required"] = list(dict.fromkeys(required))
        if "schema" in raw:
            if not _check_schema(raw["schema"], f"{rule_where}.schema", errors, depth=1, budget=[0]):
                return None
            rule["schema"] = raw["schema"]
        for kind in ("equals", "contains", "pattern"):
            if kind in raw:
                parsed = _check_arg_mapping(raw[kind], f"{rule_where}.{kind}", errors, kind=kind)
                if parsed is None:
                    return None
                rule[kind] = parsed
        if rule.keys() == {"tool"}:
            errors.add(rule_where, "must declare at least one of required, schema, equals, contains, or pattern")
            return None
        rules.append(rule)
    return rules


def _parse_expected_order(value: Any, errors: _FieldErrors) -> list[list[list[str]]] | None:
    where = "expected_order"
    if not isinstance(value, list):
        errors.add(where, "must be a list of [before, after] edges")
        return None
    if len(value) > MAX_ORDER_EDGES:
        errors.add(where, f"must list at most {MAX_ORDER_EDGES} edges")
        return None
    edges: list[list[list[str]]] = []
    for index, edge in enumerate(value):
        edge_where = f"{where}[{index}]"
        if not isinstance(edge, list) or len(edge) != 2:
            errors.add(edge_where, "must be a two-item [before, after] list")
            return None
        before = _check_ref_alternatives(edge[0], f"{edge_where}[0]", errors)
        after = _check_ref_alternatives(edge[1], f"{edge_where}[1]", errors)
        if before is None or after is None:
            return None
        edges.append([before, after])
    return edges


def _parse_handoffs(value: Any, errors: _FieldErrors) -> list[dict[str, Any]] | None:
    where = "handoffs"
    if not isinstance(value, list):
        errors.add(where, "must be a list of handoff objects")
        return None
    if len(value) > MAX_HANDOFFS:
        errors.add(where, f"must list at most {MAX_HANDOFFS} handoffs")
        return None
    handoffs: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(raw, dict):
            errors.add(item_where, "must be an object")
            return None
        unknown = sorted(str(key)[:64] for key in raw if key not in _HANDOFF_KEYS)
        if unknown:
            errors.add(item_where, f"unsupported keys: {', '.join(unknown)}")
            return None
        producer = _check_ref_alternatives(raw.get("producer"), f"{item_where}.producer", errors)
        consumer = _check_ref_alternatives(raw.get("consumer"), f"{item_where}.consumer", errors)
        if producer is None or consumer is None:
            return None
        literal = raw.get("value")
        artifact = raw.get("artifact")
        if literal is not None and (
            not isinstance(literal, str) or not literal.strip() or len(literal) > MAX_HANDOFF_VALUE_CHARS
        ):
            errors.add(
                f"{item_where}.value", f"must be a non-empty string of at most {MAX_HANDOFF_VALUE_CHARS} characters"
            )
            return None
        if artifact is not None and (
            not isinstance(artifact, str)
            or not artifact.strip()
            or len(artifact) > MAX_ARTIFACT_CHARS
            or _CONTROL_CHARS_RE.search(artifact)
        ):
            errors.add(f"{item_where}.artifact", f"must be a workspace path of at most {MAX_ARTIFACT_CHARS} characters")
            return None
        if literal is None and artifact is None:
            errors.add(item_where, "must declare a value or an artifact as handoff evidence")
            return None
        if not _check_description(raw, item_where, errors):
            return None
        handoff: dict[str, Any] = {"producer": producer, "consumer": consumer}
        if literal is not None:
            handoff["value"] = literal
        if artifact is not None:
            handoff["artifact"] = artifact
        handoffs.append(handoff)
    return handoffs


def _parse_conflict_probes(value: Any, errors: _FieldErrors) -> list[dict[str, Any]] | None:
    where = "conflict_probes"
    if not isinstance(value, list):
        errors.add(where, "must be a list of probe objects")
        return None
    if len(value) > MAX_CONFLICT_PROBES:
        errors.add(where, f"must list at most {MAX_CONFLICT_PROBES} probes")
        return None
    probes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(raw, dict):
            errors.add(item_where, "must be an object")
            return None
        unknown = sorted(str(key)[:64] for key in raw if key not in _PROBE_KEYS)
        if unknown:
            errors.add(item_where, f"unsupported keys: {', '.join(unknown)}")
            return None
        probe_id = raw.get("id")
        if not isinstance(probe_id, str) or not probe_id.strip() or len(probe_id) > MAX_PROBE_ID_CHARS:
            errors.add(f"{item_where}.id", f"must be a non-empty string of at most {MAX_PROBE_ID_CHARS} characters")
            return None
        if probe_id in seen_ids:
            errors.add(f"{item_where}.id", "must be unique")
            return None
        seen_ids.add(probe_id)
        must_use = _check_ref_alternatives(raw.get("must_use"), f"{item_where}.must_use", errors)
        must_not_use = _check_ref_alternatives(raw.get("must_not_use"), f"{item_where}.must_not_use", errors)
        if must_use is None or must_not_use is None:
            return None
        if not _check_description(raw, item_where, errors):
            return None
        description = raw.get("description")
        probes.append(
            {
                "id": probe_id.strip(),
                "must_use": must_use,
                "must_not_use": must_not_use,
                "description": description or "",
            }
        )
    return probes


def _parse_case_field(name: str, value: Any, errors: _FieldErrors) -> Any:
    if name in {"expected_tools", "acceptable_tools", "decoy_tools"}:
        return _check_tool_patterns(value, name, errors)
    if name == "tool_arguments":
        return _parse_tool_arguments(value, errors)
    if name == "expected_order":
        return _parse_expected_order(value, errors)
    if name == "handoffs":
        return _parse_handoffs(value, errors)
    return _parse_conflict_probes(value, errors)


def _spec_field(spec: Mapping[str, Any] | None, name: str) -> list[Any]:
    """One normalized case field (raw entries and :func:`plugin_case_spec` output both work)."""
    if not isinstance(spec, Mapping) or spec.get(name) is None:
        return []
    errors = _FieldErrors()
    parsed = _parse_case_field(name, spec[name], errors)
    return list(parsed) if parsed is not None and not errors.messages else []


def validate_plugin_case_fields(entry: Any) -> list[str]:
    """Return problems with the optional advisory plugin-signal fields of one case.

    Absent fields are fine. ``null`` is treated as absent. Every present field is
    type-checked and size-bounded; see the ``MAX_*`` constants.
    """
    if not isinstance(entry, Mapping):
        return []
    errors = _FieldErrors()
    for name in PLUGIN_CASE_FIELDS:
        value = entry.get(name)
        if value is None:
            continue
        _parse_case_field(name, value, errors)
    return errors.messages


def plugin_case_spec(entry: Any) -> dict[str, Any]:
    """Return the valid, normalized plugin-signal fields of one case entry.

    Invalid fields are dropped (the dataset validators report them), so a
    malformed advisory field can never break result collection.
    """
    if not isinstance(entry, Mapping):
        return {}
    spec: dict[str, Any] = {}
    for name in PLUGIN_CASE_FIELDS:
        value = entry.get(name)
        if value is None:
            continue
        errors = _FieldErrors()
        parsed = _parse_case_field(name, value, errors)
        if parsed is not None and not errors.messages:
            spec[name] = parsed
    return spec


# =============================================================================
# Collector context
# =============================================================================


def _clean_names(values: Iterable[Any]) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        name = value.strip()
        if not name or len(name) > MAX_COMPONENT_NAME_CHARS or _CONTROL_CHARS_RE.search(name):
            continue
        if name not in names:
            names.append(name)
        if len(names) >= MAX_DECLARED_COMPONENTS:
            break
    return tuple(names)


@dataclass(frozen=True)
class PluginSignalsContext:
    """What the collector needs to compute plugin signals for one run.

    ``member_skills`` and ``mcp_servers`` are the plugin's declared, locally
    evaluated components (the member skills staged in the with-plugin arm and
    the runnable MCP servers wired into it). ``wrapper_skills`` names the
    generated wrapper skill so it is not scored as a tool selection. ``cases``
    maps case id to :func:`plugin_case_spec` output.
    """

    member_skills: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    wrapper_skills: tuple[str, ...] = ()
    cases: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    baseline_has_members: bool = False
    # Declared plugin subagents and commands. They count toward activation
    # coverage in the with-plugin arm only, the one arm that carries the plugin.
    subagents: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    # Staged subagent name (casefolded) -> declared name, for harnesses that
    # rename a plugin agent (OpenCode stages ``build`` as ``<plugin>-build``).
    subagent_aliases: Mapping[str, str] = field(default_factory=dict)

    def arm_enabled(self, arm: str) -> bool:
        if arm in {ARM_WITH_SKILL, ARM_SUM_OF_PARTS}:
            return True
        return arm == ARM_WITHOUT_SKILL and self.baseline_has_members

    def declared_for(self, arm: str) -> dict[str, list[str]]:
        """Declared components staged in ``arm`` (MCP is wired into the with-plugin arm only)."""
        declared: dict[str, list[str]] = {COMPONENT_SKILL: list(self.member_skills)}
        with_plugin = arm == ARM_WITH_SKILL
        declared[COMPONENT_MCP] = list(self.mcp_servers) if with_plugin else []
        if with_plugin and self.subagents:
            declared[COMPONENT_SUBAGENT] = list(self.subagents)
        if with_plugin and self.commands:
            declared[COMPONENT_COMMAND] = list(self.commands)
        return declared

    def aliases_for(self, arm: str) -> Mapping[str, str]:
        """Subagent name aliases for ``arm`` (declared subagents count in the with-plugin arm only)."""
        return self.subagent_aliases if arm == ARM_WITH_SKILL and self.subagents else {}

    def case_spec(self, case_id: str) -> Mapping[str, Any]:
        return self.cases.get(case_id) or {}


def build_plugin_signals_context(
    *,
    member_skills: Iterable[Any] = (),
    mcp_servers: Iterable[Any] = (),
    wrapper_skills: Iterable[Any] = (),
    entries: Iterable[Any] = (),
    baseline_has_members: bool = False,
    subagents: Iterable[Any] = (),
    commands: Iterable[Any] = (),
    subagent_aliases: Mapping[str, Any] | None = None,
) -> PluginSignalsContext:
    """Build a bounded :class:`PluginSignalsContext` from dataset case entries."""
    cases: dict[str, Mapping[str, Any]] = {}
    for count, entry in enumerate(entries):
        if count >= MAX_CONTEXT_CASES:
            break
        if not isinstance(entry, Mapping):
            continue
        case_id = entry.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in cases:
            continue
        spec = plugin_case_spec(entry)
        if spec:
            cases[case_id] = spec
    declared_subagents = _clean_names(subagents)
    return PluginSignalsContext(
        member_skills=_clean_names(member_skills),
        mcp_servers=_clean_names(mcp_servers),
        wrapper_skills=_clean_names(wrapper_skills),
        cases=cases,
        baseline_has_members=bool(baseline_has_members),
        subagents=declared_subagents,
        commands=_clean_names(str(name).lstrip("/") for name in commands if isinstance(name, str)),
        subagent_aliases=_clean_aliases(subagent_aliases, declared_subagents),
    )


def _clean_aliases(aliases: Mapping[str, Any] | None, declared: Sequence[str]) -> dict[str, str]:
    """Bounded ``staged name (casefolded) -> declared name`` pairs whose target is a declared subagent."""
    by_folded = {name.casefold(): name for name in declared}
    cleaned: dict[str, str] = {}
    for alias, name in (aliases or {}).items():
        if len(cleaned) >= MAX_DECLARED_COMPONENTS:
            break
        if not isinstance(alias, str) or not isinstance(name, str):
            continue
        target = by_folded.get(name.strip().casefold())
        if target is not None and alias.strip():
            cleaned[alias.strip().casefold()[:_MAX_LABEL_CHARS]] = target
    return cleaned


# =============================================================================
# Trajectory -> calls
# =============================================================================


@dataclass(frozen=True)
class _Ident:
    """One identity of a tool call: a component activation or a plain tool."""

    label: str
    kind: str | None
    name: str
    fn: str
    server: str | None = None
    tool: str | None = None
    tool_label: str = ""
    persist_name: bool = True

    @property
    def persisted_name(self) -> str:
        return self.name if self.persist_name else _NON_NAME

    @property
    def persisted_label(self) -> str:
        return self.label if self.persist_name else f"{_IDENTITY_PREFIX[self.kind or '']}:{_NON_NAME}"


@dataclass
class _Call:
    """One normalized tool call.

    ``mcp`` is the call's MCP identity, however the harness spelled the tool
    name, and ``fn_base`` is the bare tool name: the MCP tool's for an MCP
    call, so every spelling of one MCP call is treated alike.
    """

    seq: int
    step_index: int
    fn: str
    fn_base: str
    args: dict[str, Any]
    idents: list[_Ident]
    mcp: _Ident | None = None
    observation: str | None = None
    succeeded: bool | None = None
    owner: int | None = None
    _args_text: str | None = None
    _shell_paths: tuple[list[str], list[str]] | None = None

    @property
    def is_shell(self) -> bool:
        """A local shell call (an MCP tool named like a shell is still an MCP call)."""
        return self.fn_base in _SHELL_TOOLS and self.mcp is None

    @property
    def is_content_read(self) -> bool:
        """A skill load or a plain file read: its result is file or skill text, not a status message."""
        if self.mcp is not None:
            return False
        if self.fn.casefold() in _SKILL_TOOLS or any(ident.kind == COMPONENT_SKILL for ident in self.idents):
            return True
        return self.fn_base in _READ_TOOLS

    @property
    def args_text(self) -> str:
        if self._args_text is None:
            try:
                text = json.dumps(self.args, ensure_ascii=False, sort_keys=True, default=str)
            except (TypeError, ValueError, RecursionError):
                text = ""
            self._args_text = text[:_MAX_ARGS_TEXT_CHARS]
        return self._args_text

    @property
    def component_idents(self) -> list[_Ident]:
        return [ident for ident in self.idents if ident.kind is not None]

    @property
    def shell_paths(self) -> tuple[list[str], list[str]]:
        """Distinct normalized ``(read, written)`` shell path operands, parsed once per call."""
        if self._shell_paths is None:
            reads: dict[str, None] = {}
            writes: dict[str, None] = {}
            if self.is_shell:
                for text in _shell_texts(self.fn_base, self.args):
                    text_reads, text_writes = _shell_io(text, reader_verbs=_ARTIFACT_CONSUMER_VERBS)
                    if _APPLY_PATCH_COMMAND_RE.search(text):
                        # A shell ``apply_patch <<'EOF'`` (Codex) writes the files its patch headers name.
                        text_writes.extend(_patch_file_paths(text))
                    reads.update(dict.fromkeys(_normalize_path(path) for path in text_reads))
                    writes.update(dict.fromkeys(_normalize_path(path) for path in text_writes))
            self._shell_paths = (list(reads), list(writes))
        return self._shell_paths


def _safe_text(value: Any, limit: int = _MAX_LABEL_CHARS) -> str:
    # The redactor is superlinear on some inputs, so it only ever sees a bounded
    # window; text past the window would be truncated away anyway. The slack
    # leaves room for redaction to shorten the text and still fill ``limit``.
    text = str(value if value is not None else "")[: limit * 4 + 256]
    text = _CONTROL_CHARS_RE.sub(" ", text)
    return redact_sensitive_text(text, max_len=limit)


def _tool_name(tool_call: Mapping[str, Any]) -> str:
    for key in ("function_name", "name", "tool_name"):
        value = tool_call.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_MAX_LABEL_CHARS]
    return ""


def _arguments(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("arguments")
    if raw is None:
        raw = tool_call.get("input", tool_call.get("args"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if len(raw) <= _MAX_ARGS_JSON_CHARS:
            try:
                decoded = json.loads(raw)
            except (ValueError, RecursionError):
                decoded = None
            if isinstance(decoded, dict):
                return decoded
        return {"input": raw[:_MAX_ARGS_JSON_CHARS]}
    return {}


def _content_parts(content: Any) -> list[str]:
    """Bounded text of each content block (one part for a plain string/object)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [content[:_MAX_OBSERVATION_CHARS]]
    if isinstance(content, list):
        parts: list[str] = []
        size = 0
        for block in content[:256]:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                part = block["text"]
            elif isinstance(block, str):
                part = block
            else:
                part = _bounded_json_text(block, _MAX_OBSERVATION_CHARS) or ""
            parts.append(part)
            size += len(part)
            if size >= _MAX_OBSERVATION_CHARS:
                break
        return parts
    return [(_bounded_json_text(content, _MAX_OBSERVATION_CHARS) or "")[:_MAX_OBSERVATION_CHARS]]


def _content_text(content: Any) -> str:
    return "\n".join(_content_parts(content))[:_MAX_OBSERVATION_CHARS]


def _failure_scan_text(parts: Sequence[str], text: str) -> str:
    """What failure markers are checked against: each block's head plus the result's tail.

    Bounded (at most one window per block, within the observation cap) but
    fail-closed for the usual shapes: an error block after a long first block,
    or an error line appended to a long body.
    """
    windows = [part[:_FAILURE_SCAN_CHARS] for part in parts]
    if len(text) > _FAILURE_SCAN_CHARS:
        windows.append(text[-_FAILURE_SCAN_CHARS:])
    return "\n".join(windows)


def _result_flagged(result: Mapping[str, Any]) -> bool:
    for source in (result, result.get("extra")):
        if not isinstance(source, Mapping):
            continue
        for key in ("is_error", "isError"):
            if source.get(key) is True:
                return True
        error = source.get("error")
        if error not in (None, False, "", {}, []):
            return True
    content = result.get("content")
    return isinstance(content, Mapping) and content.get("isError") is True


def _observations(step: Mapping[str, Any]) -> list[tuple[str, str, str, bool]]:
    """``(source_call_id, text, failure_scan_text, flagged)`` for each result of ``step``."""
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        return []
    results = observation.get("results")
    if not isinstance(results, list):
        return []
    entries: list[tuple[str, str, str, bool]] = []
    for result in results[:256]:
        if not isinstance(result, Mapping):
            continue
        parts = _content_parts(result.get("content"))
        text = "\n".join(parts)[:_MAX_OBSERVATION_CHARS]
        entries.append(
            (
                str(result.get("source_call_id") or ""),
                text,
                _failure_scan_text(parts, text),
                _result_flagged(result),
            )
        )
    return entries


def _results_for_call(
    results: list[tuple[str, str, str, bool]], call_id: str, *, call_count: int
) -> list[tuple[str, str, bool]]:
    """Results attributable to ``call_id``; id matches win, ambiguity stays unknown."""
    if call_id:
        matched = [(text, scan, flagged) for rid, text, scan, flagged in results if rid == call_id]
        if matched:
            return matched
    if call_count == 1 and len(results) == 1 and not results[0][0]:
        return [results[0][1:]]
    return []


def _first_line(text: str, *, shell: bool = False) -> str:
    for line in text[:_FAILURE_SCAN_CHARS].splitlines():
        stripped = line.strip()
        if stripped and not (shell and _SHELL_STATUS_LINE_RE.fullmatch(stripped)):
            return line
    return ""


def _content_read_scan(text: str, *, shell: bool) -> str:
    """The lines of a file read or skill load that can say the read failed.

    The first line, past any shell status lines. A shell read can also fail
    after other output, so every shell error line that names a skill manifest
    (``cat: <dir>/SKILL.md: Permission denied``) counts too.
    """
    lines = [_first_line(text, shell=shell)]
    if shell:
        lines.extend(
            line
            for line in text.splitlines()
            if "skill.md" in line.casefold() and _SHELL_ERROR_LINE_RE.match(line.strip())
        )
    return "\n".join(lines)


def _outcome(
    correlated: list[tuple[str, str, bool]], *, shell: bool, content_read: bool = False
) -> tuple[str | None, bool | None]:
    """``(observation text, succeeded)`` for one call (see the module docs).

    For a ``content_read`` (a file read or a skill load) the body is file or
    skill text, so words like "not available" in it say nothing about the call:
    only a structured error flag, a marker on the first line of a result (past
    a harness's shell status lines), or a shell error line that names a skill
    manifest counts.
    """
    if not correlated:
        return None, None
    text = "".join(item for item, _, _ in correlated)[:_MAX_OBSERVATION_CHARS]
    flagged = any(is_error for _, _, is_error in correlated)
    if not flagged and not text.strip():
        return text, None
    if content_read:
        scan = "\n".join(_content_read_scan(item, shell=shell) for item, _, _ in correlated)
    else:
        scan = "\n".join(window for _, window, _ in correlated)
    lowered = scan.casefold()
    markers = _UNAVAILABLE_MARKERS if shell else (*_UNAVAILABLE_MARKERS, *_FILE_MISSING_MARKERS)
    failed = flagged or any(marker in lowered for marker in markers) or bool(_FAILED_CALL_RE.search(scan))
    return text, not failed


def _base_tool_name(fn: str) -> str:
    low = fn.casefold()
    for separator in _TOOL_NAME_SEPARATORS:
        if separator in low:
            low = low.rsplit(separator, 1)[-1]
    return low


def _norm_server(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", value.casefold())


def _server_spellings(server: str) -> tuple[str, ...]:
    """Casefolded spellings of a declared server name inside harness tool names, exact spelling first.

    Claude Code and OpenCode replace characters outside ``[A-Za-z0-9_-]`` with
    ``_``; Hermes also replaces ``-``.
    """
    low = server.casefold()
    return tuple(dict.fromkeys((low, _norm_server(server), _HERMES_NAME_RE.sub("_", low))))


def match_declared_mcp_server(observed: str, declared: Iterable[str]) -> str | None:
    """The one declared server that ``observed`` (a server name taken from a tool name) refers to.

    An exact (case-insensitive) match wins. Otherwise a harness spelling of a
    declared name counts only when exactly one declared server has it, so
    ``my.docs`` and ``my-docs`` never credit each other. Claude Code names
    plugin servers ``plugin_<plugin>_<server>``; the plugin slug never holds
    ``_``, so the server is what follows the first ``_`` (a longest declared
    suffix is the fallback).
    """
    names = [name for name in declared if isinstance(name, str) and name]
    low = observed.casefold()
    if not low or not names:
        return None
    exact = [name for name in names if name.casefold() == low]
    if exact:
        return exact[0]
    spelled = list(dict.fromkeys(name for name in names if low in _server_spellings(name)))
    if len(spelled) == 1:
        return spelled[0]
    if spelled or not low.startswith(_CLAUDE_PLUGIN_SERVER_PREFIX):
        return None
    rest = low[len(_CLAUDE_PLUGIN_SERVER_PREFIX) :]
    slug, sep, server = rest.partition("_")
    if slug and sep and server:
        found = match_declared_mcp_server(server, names)
        if found is not None:
            return found
    suffixes: dict[int, list[str]] = {}
    for name in names:
        for spelling in _server_spellings(name):
            if rest.endswith("_" + spelling) and len(rest) > len(spelling) + 1:
                suffixes.setdefault(len(spelling), []).append(name)
    if not suffixes:
        return None
    longest = list(dict.fromkeys(suffixes[max(suffixes)]))
    return longest[0] if len(longest) == 1 else None


def _mcp_identity(fn: str, declared_mcp: Sequence[str], *, agent: str = "") -> tuple[str, str] | None:
    """``(server, tool)`` of an MCP tool call, with the server mapped to its declared name when known.

    Recognized spellings: ``mcp__<server>__<tool>`` (Claude Code, Codex; Claude
    Code plugin servers appear as ``plugin_<plugin>_<server>``), and for
    declared servers ``<server>__<tool>``/``.``/``/``/``:``, Hermes
    ``mcp_<server>_<tool>``, and OpenCode ``<server>_<tool>`` (not for
    harnesses that never use it). The longest matching prefix wins; on a tie an
    exact spelling beats a normalized one, and a remaining tie between servers
    is left unattributed.
    """
    low = fn.casefold()
    if low[:5] == "mcp__":
        server, _, tool = fn[5:].partition("__")
        if not server:
            return None
        return match_declared_mcp_server(server, declared_mcp) or server, tool
    bare_prefix = agent.casefold() not in _NO_BARE_MCP_PREFIX_AGENTS and low not in _BUILTIN_TOOL_NAMES
    best: tuple[int, int] | None = None
    best_spelling = ""
    winners: list[str] = []
    for server in declared_mcp:
        for rank, spelling in enumerate(_server_spellings(server)):
            prefixes = [spelling + separator for separator in _TOOL_NAME_SEPARATORS] + [f"mcp_{spelling}_"]
            if bare_prefix:
                prefixes.append(spelling + "_")
            for prefix in prefixes:
                if not (low.startswith(prefix) and len(low) > len(prefix)):
                    continue
                key = (len(prefix), -min(rank, 1))
                if best is None or key > best:
                    best, best_spelling, winners = key, spelling, [server]
                elif key == best and server not in winners:
                    winners.append(server)
    if best is None:
        return None
    # Two declared servers share this spelling: count the call under the spelling itself.
    return (winners[0] if len(winners) == 1 else best_spelling), fn[best[0] :]


def _first_string(args: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_path(value: str) -> str:
    text = value.strip().strip("'\"").replace("\\", "/")
    text = re.sub(r"/{2,}", "/", text)
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/") if len(text) > 1 else text


def _path_args(args: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in _PATH_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value)
    return paths


def _string_values(value: Any, *, limit: int = 256, depth: int = 0) -> list[str]:
    """Bounded list of string leaves in a JSON-ish value."""
    found: list[str] = []
    if depth > 8:
        return found
    if isinstance(value, str):
        return [value[:_MAX_ARGS_TEXT_CHARS]]
    if isinstance(value, Mapping):
        items: Iterable[Any] = list(value.values())[:limit]
    elif isinstance(value, list):
        items = value[:limit]
    else:
        return found
    for item in items:
        found.extend(_string_values(item, limit=limit, depth=depth + 1))
        if len(found) >= limit:
            break
    return found[:limit]


def _shell_texts(fn_base: str, args: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in _SHELL_COMMAND_KEYS:
        if key == "input" and fn_base == "exec":
            # Native Codex ``exec`` input is JavaScript, not a shell command.
            continue
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value[:_MAX_SHELL_TEXT_CHARS])
        elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            words = [str(item) for item in value[:_MAX_SHELL_TOKENS]]
            texts.append(shlex.join(words)[:_MAX_SHELL_TEXT_CHARS])
    return texts


def _strip_heredoc_bodies(text: str) -> str:
    """Drop here-document bodies: they are data, not commands, and may hold stray quotes.

    A body is dropped only when its terminator line is found, so a ``<<`` that
    is not really a here-document never swallows the rest of the script.
    """
    if "<<" not in text:
        return text
    lines = text.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        index += 1
        for match in _HEREDOC_RE.finditer(line):
            delimiter = next(group for group in match.groups() if group)
            end = next((at for at in range(index, len(lines)) if lines[at].strip() == delimiter), None)
            if end is None:
                return text
            index = end + 1
    return "\n".join(kept)


def _shell_tokens(text: str) -> list[str]:
    normalized = _strip_heredoc_bodies(text.replace("\r\n", "\n").replace("\r", "\n")).replace("\n", " ; ")
    lexer = shlex.shlex(normalized, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: list[str] = []
    try:
        for token in lexer:
            tokens.append(token)
            if len(tokens) >= _MAX_SHELL_TOKENS:
                break
    except ValueError:
        # Unbalanced quote: fall back on the same newline-normalized text so a
        # multi-line script still splits into its commands.
        try:
            return shlex.split(normalized, posix=True)[:_MAX_SHELL_TOKENS]
        except ValueError:
            return normalized.split()[:_MAX_SHELL_TOKENS]
    return tokens


def _shell_payload(command: list[str]) -> str | None:
    """Return the ``-c`` payload of ``bash -lc '...'``-style invocations."""
    for index, token in enumerate(command[1:], start=1):
        if token.startswith("-") and not token.startswith("--") and "c" in token[1:]:
            return command[index + 1] if index + 1 < len(command) else None
        if not token.startswith("-"):
            return None
    return None


def _shell_commands(text: str, depth: int = 0) -> list[list[str]]:
    if depth > _MAX_SHELL_DEPTH or len(text) > _MAX_SHELL_TEXT_CHARS:
        return []
    commands: list[list[str]] = []
    current: list[str] = []
    for token in _shell_tokens(text):
        if token in _SHELL_SEPARATORS:
            if current:
                commands.append(current)
            current = []
        else:
            current.append(token)
    if current:
        commands.append(current)
    expanded: list[list[str]] = []
    for command in commands:
        index = 0
        while index < len(command) and (
            _ENV_ASSIGNMENT_RE.match(command[index]) or command[index] in _SHELL_PREFIX_WORDS
        ):
            index += 1
        command = command[index:]
        if not command:
            continue
        if command[0].rsplit("/", 1)[-1] in _SHELLS:
            payload = _shell_payload(command)
            if payload is not None:
                expanded.extend(_shell_commands(payload, depth + 1))
                continue
        expanded.append(command)
    return expanded


def _shell_io(text: str, *, reader_verbs: frozenset[str]) -> tuple[list[str], list[str]]:
    """Return ``(read_paths, written_paths)`` operands found in shell ``text``."""
    reads: list[str] = []
    writes: list[str] = []
    for command in _shell_commands(text):
        verb = ""
        operands: list[str] = []
        index = 0
        while index < len(command):
            token = command[index]
            if token == ">&":
                # ``2>&1``-style descriptor duplication, not a file target.
                index += 2
                continue
            if token in _OUTPUT_REDIRECTS:
                if index + 1 < len(command):
                    writes.append(command[index + 1])
                index += 2
                continue
            if token == "<":
                if index + 1 < len(command):
                    reads.append(command[index + 1])
                index += 2
                continue
            if token in {"<<", "<<<", "<<-"}:
                break
            if not verb:
                verb = token.rsplit("/", 1)[-1].casefold()
            elif not token.startswith("-"):
                operands.append(token)
            index += 1
        if verb in reader_verbs:
            reads.extend(operands)
        elif verb == "tee":
            writes.extend(operands)
        elif verb in {"cp", "mv", "install"} and len(operands) >= 2:
            writes.append(operands[-1])
            reads.extend(operands[:-1])
        elif verb == "touch":
            writes.extend(operands)
    return reads, writes


def _member_manifest_match(path: str, members: Sequence[str]) -> str | None:
    normalized = _normalize_path(path).casefold()
    for member in members:
        suffix = f"{member.casefold()}/skill.md"
        if normalized == suffix or normalized.endswith("/" + suffix):
            return member
    return None


def _declared_skill_reads(fn_base: str, args: Mapping[str, Any], members: Sequence[str], *, is_mcp: bool) -> list[str]:
    """The declared members whose ``SKILL.md`` this call reads, in order (every one in a chained shell command)."""
    if not members:
        return []
    reads_file = fn_base in _READ_TOOLS or (is_mcp and bool(_READ_VERB_RE.search(fn_base)))
    if fn_base in {"str_replace_editor", "str_replace_based_edit_tool"}:
        reads_file = str(args.get("command") or "").casefold() == "view"
    paths: list[str] = []
    if reads_file:
        paths = _path_args(args)
    elif fn_base in _SHELL_TOOLS and not is_mcp:
        for text in _shell_texts(fn_base, args):
            if "skill.md" in text.casefold():
                paths.extend(_shell_io(text, reader_verbs=_FILE_READER_VERBS)[0])
    found: dict[str, None] = {}
    for path in paths:
        member = _member_manifest_match(path, members)
        if member is not None:
            found[member] = None
    return list(found)


def _declared_command(name: str, declared: Mapping[str, Sequence[str]]) -> bool:
    """Whether a ``Skill`` tool name is a declared plugin command (``<plugin>:<command>`` or bare).

    Claude Code runs plugin commands through its ``Skill`` tool. A name that is
    also a declared member skill stays a skill.
    """
    commands = {command.casefold() for command in declared.get(COMPONENT_COMMAND) or ()}
    if not name or not commands:
        return False
    candidates = _name_candidates(name)
    skills = {skill.casefold() for skill in declared.get(COMPONENT_SKILL) or ()}
    return candidates[-1] in commands and not any(candidate in skills for candidate in candidates)


def _component_ident(kind: str, name: str, fn: str, tool_label: str, *, persist: bool = True) -> _Ident:
    label = f"{_IDENTITY_PREFIX[kind]}:{name}" if name else _IDENTITY_PREFIX[kind]
    return _Ident(label=label, kind=kind, name=name, fn=fn, tool_label=tool_label, persist_name=persist)


def _persistable_name(name: str, members: Sequence[str]) -> bool:
    """Whether an argument-derived component name may be persisted (identifier-shaped or declared)."""
    if not name or (_PERSISTABLE_NAME_RE.fullmatch(name) and "://" not in name):
        return True
    folded = name.casefold()
    return any(member.casefold() == folded for member in members)


def _mcp_ident(fn: str, declared_mcp: Sequence[str], *, agent: str) -> _Ident | None:
    """The MCP identity of a call named ``fn``, labeled ``mcp__<server>__<tool>`` (see :func:`_mcp_identity`)."""
    identity = _mcp_identity(fn, declared_mcp, agent=agent)
    if identity is None:
        return None
    server, tool = identity
    canonical = f"mcp__{server}__{tool}" if tool else f"mcp__{server}"
    return _Ident(
        label=canonical,
        kind=COMPONENT_MCP,
        name=server,
        fn=fn,
        server=server,
        tool=tool,
        tool_label=canonical,
    )


def _identities(
    fn: str,
    fn_base: str,
    mcp: _Ident | None,
    args: Mapping[str, Any],
    declared: Mapping[str, Sequence[str]],
    *,
    subagent_aliases: Mapping[str, str] | None = None,
) -> list[_Ident]:
    low = fn.casefold()
    idents: list[_Ident] = []
    if low in _SKILL_TOOLS:
        name = _first_string(args, ("skill", "name", "command"))
        if _declared_command(name.lstrip("/"), declared):
            name = name.lstrip("/")
            persist = _persistable_name(name, declared.get(COMPONENT_COMMAND) or ())
            idents.append(_component_ident(COMPONENT_COMMAND, name, fn, fn, persist=persist))
        else:
            persist = _persistable_name(name, declared.get(COMPONENT_SKILL) or ())
            idents.append(_component_ident(COMPONENT_SKILL, name, fn, fn, persist=persist))
    elif low in _SUBAGENT_TOOLS:
        name = _first_string(args, ("subagent_type", "subagent", "agent", "agent_name", "agent_type"))
        # A harness that renamed a plugin agent when staging it calls it by the staged name.
        name = (subagent_aliases or {}).get(name.casefold(), name)
        idents.append(_component_ident(COMPONENT_SUBAGENT, name, fn, fn, persist=_persistable_name(name, ())))
    elif low in _COMMAND_TOOLS:
        command = _first_string(args, ("command", "name"))
        name = command.split()[0].lstrip("/") if command.split() else ""
        idents.append(_component_ident(COMPONENT_COMMAND, name, fn, fn, persist=_persistable_name(name, ())))
    if mcp is not None:
        idents.append(mcp)
    if not any(ident.kind in {COMPONENT_SKILL, COMPONENT_COMMAND} for ident in idents):
        members = declared.get(COMPONENT_SKILL) or ()
        for member in _declared_skill_reads(fn_base, args, members, is_mcp=mcp is not None):
            idents.append(_component_ident(COMPONENT_SKILL, member, fn, f"{fn}:skill-md-read"))
    if not idents:
        idents.append(_Ident(label=fn, kind=None, name=fn, fn=fn, tool_label=fn))
    return idents


def _trajectory_agent(trajectory: Mapping[str, Any]) -> str:
    agent = trajectory.get("agent")
    name = agent.get("name") if isinstance(agent, Mapping) else None
    return name.strip()[:_MAX_LABEL_CHARS] if isinstance(name, str) else ""


def _extract_calls(
    trajectory: Mapping[str, Any],
    declared: Mapping[str, Sequence[str]],
    mcp_call_servers: Mapping[str, str] | None = None,
    subagent_aliases: Mapping[str, str] | None = None,
) -> list[_Call] | None:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
    agent = _trajectory_agent(trajectory)
    declared_mcp = declared.get(COMPONENT_MCP) or ()
    calls: list[_Call] = []
    owner: int | None = None
    for step_index, step in enumerate(steps[:_MAX_STEPS]):
        if not isinstance(step, Mapping):
            continue
        raw_calls = step.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            continue
        raw_calls = [item for item in raw_calls if isinstance(item, Mapping)]
        results = _observations(step)
        for raw in raw_calls:
            outer_id = str(raw.get("tool_call_id") or raw.get("id") or "")
            name = _tool_name(raw)
            server = (mcp_call_servers or {}).get(outer_id) if outer_id else None
            if server and name and name[:5].casefold() != "mcp__":
                # The harness log names the server this bare MCP tool name came from (Codex).
                name = f"mcp__{server}__{name}"[:_MAX_LABEL_CHARS]
            prepared = {**raw, "function_name": name, "arguments": _arguments(raw)}
            try:
                normalized = normalize_tool_call(prepared)
            except (TypeError, ValueError, RecursionError):
                normalized = [prepared]
            for tool_call in normalized:
                if len(calls) >= _MAX_CALLS:
                    return calls
                status = tool_call.get("_atif_observation_status")
                if status is None or status == MAPPED_OUTER_EXEC_OBSERVATION:
                    correlated = _results_for_call(results, outer_id, call_count=len(raw_calls))
                else:
                    correlated = []
                fn = str(tool_call.get("function_name") or "")[:_MAX_LABEL_CHARS]
                args = tool_call.get("arguments")
                args = args if isinstance(args, dict) else {}
                mcp = _mcp_ident(fn, declared_mcp, agent=agent)
                fn_base = _base_tool_name(mcp.tool if mcp is not None and mcp.tool else fn)
                idents = _identities(fn, fn_base, mcp, args, declared, subagent_aliases=subagent_aliases)
                seq = len(calls)
                if any(ident.kind in {COMPONENT_SKILL, COMPONENT_COMMAND} for ident in idents):
                    owner = seq
                call = _Call(
                    seq=seq,
                    step_index=step_index,
                    fn=fn,
                    fn_base=fn_base,
                    args=args,
                    idents=idents,
                    mcp=mcp,
                    owner=owner,
                )
                call.observation, call.succeeded = _outcome(
                    correlated, shell=call.is_shell, content_read=call.is_content_read
                )
                calls.append(call)
    return calls


def _prompt_texts(trajectory: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return texts
    for step in steps[:_MAX_STEPS]:
        source = step.get("source") if isinstance(step, Mapping) else None
        if not isinstance(source, str) or source not in {"user", "system"}:
            continue
        message = step.get("message")
        text = message if isinstance(message, str) else _content_text(message)
        if text:
            texts.append(text[:_MAX_OBSERVATION_CHARS])
    return texts


# =============================================================================
# Ref matching
# =============================================================================


@dataclass(frozen=True)
class _Ref:
    raw: str
    kind: str | None
    pattern: str


def _parse_ref(raw: str) -> _Ref:
    text = raw.strip()
    prefix, sep, rest = text.partition(":")
    kind = _REF_PREFIXES.get(prefix.strip().casefold()) if sep else None
    if kind is not None and rest.strip():
        pattern = rest.strip().casefold()
        return _Ref(raw=text, kind=kind, pattern=pattern.lstrip("/") if kind == COMPONENT_COMMAND else pattern)
    return _Ref(raw=text, kind=None, pattern=text.casefold())


def _glob(value: str, pattern: str) -> bool:
    return bool(value) and fnmatch.fnmatchcase(value.casefold(), pattern)


def _name_candidates(name: str) -> list[str]:
    low = name.casefold()
    return [low, low.rsplit(":", 1)[-1]] if ":" in low else [low]


def _ref_matches(ref: _Ref, ident: _Ident) -> bool:
    if ref.kind is not None:
        if ident.kind != ref.kind:
            return False
        if ident.kind == COMPONENT_MCP:
            server = ident.server or ""
            candidates = [server.casefold(), _norm_server(server)]
            if ident.tool:
                candidates += [f"{server}/{ident.tool}".casefold(), f"{server}__{ident.tool}".casefold()]
            return any(_glob(candidate, ref.pattern) for candidate in candidates)
        return any(_glob(candidate, ref.pattern) for candidate in _name_candidates(ident.name))
    candidates = [ident.label, ident.fn]
    if ident.kind is not None:
        candidates.extend(_name_candidates(ident.name))
    if ident.kind == COMPONENT_COMMAND and ident.name:
        candidates.append(f"/{ident.name}")
    return any(_glob(candidate, ref.pattern) for candidate in candidates)


def _refs(values: Sequence[str]) -> list[_Ref]:
    return [_parse_ref(value) for value in values]


def _call_matches(refs: Sequence[_Ref], call: _Call) -> bool:
    return any(_ref_matches(ref, ident) for ref in refs for ident in call.idents)


def _ref_label(values: Sequence[str]) -> str:
    return _safe_text(" | ".join(values))


def _attributed(refs: Sequence[_Ref], calls: Sequence[_Call]) -> list[_Call]:
    """Calls matching ``refs`` directly or running in a matching skill/command window."""
    openers: set[int] = set()
    for call in calls:
        if any(
            ident.kind in {COMPONENT_SKILL, COMPONENT_COMMAND} and _ref_matches(ref, ident)
            for ref in refs
            for ident in call.idents
        ):
            openers.add(call.seq)
    return [call for call in calls if _call_matches(refs, call) or (call.owner is not None and call.owner in openers)]


def _is_wrapper(ident: _Ident, wrapper_skills: Sequence[str]) -> bool:
    if ident.kind != COMPONENT_SKILL:
        return False
    names = set(_name_candidates(ident.name))
    return any(wrapper.casefold() in names for wrapper in wrapper_skills)


# =============================================================================
# Graders
# =============================================================================


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def grade_tool_selection(
    calls: Sequence[_Call], spec: Mapping[str, Any] | None, *, wrapper_skills: Sequence[str] = ()
) -> dict[str, Any]:
    """Precision/recall/F1 of called tools against ``expected_tools`` + ``acceptable_tools``.

    ``called`` holds the distinct identities of every component activation
    (except the generated wrapper skill) plus any other call matched by an
    expected/acceptable/decoy ref, so ordinary unlisted tools (``Read``,
    ``Bash``...) do not dilute precision. ``precision`` is ``None`` only when
    nothing is expected and nothing in scope was called; ``recall``/``f1`` are
    ``None`` when nothing is expected.
    """
    expected = _spec_field(spec, "expected_tools")
    acceptable = _spec_field(spec, "acceptable_tools")
    decoys = _spec_field(spec, "decoy_tools")
    expected_refs, acceptable_refs, decoy_refs = _refs(expected), _refs(acceptable), _refs(decoys)
    all_refs = [*expected_refs, *acceptable_refs, *decoy_refs]

    called: dict[str, _Ident] = {}
    decoy_calls = 0
    for call in calls:
        if decoy_refs and _call_matches(decoy_refs, call):
            decoy_calls += 1
        for ident in call.idents:
            matched = any(_ref_matches(ref, ident) for ref in all_refs)
            in_scope = matched or (ident.kind is not None and not _is_wrapper(ident, wrapper_skills))
            if in_scope and ident.label not in called and len(called) < _MAX_CALLED:
                called[ident.label] = ident

    scored = bool(expected or acceptable or decoys)
    allowed = [*expected_refs, *acceptable_refs]
    precise = sum(1 for ident in called.values() if any(_ref_matches(ref, ident) for ref in allowed))
    if called:
        precision: float | None = _ratio(precise, len(called))
    else:
        precision = 0.0 if expected else None
    recall: float | None = None
    f1: float | None = None
    if expected:
        hits = sum(1 for ref in expected_refs if any(_ref_matches(ref, ident) for ident in called.values()))
        recall = _ratio(hits, len(expected))
        if precision is not None and recall is not None:
            f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return {
        "expected": [_safe_text(item) for item in expected],
        "acceptable": [_safe_text(item) for item in acceptable],
        "decoys": [_safe_text(item) for item in decoys],
        "called": [_safe_text(ident.persisted_label) for ident in called.values()],
        "precision": precision if scored else None,
        "recall": recall,
        "f1": f1,
        "decoy_calls": decoy_calls,
        "status": STATUS_SCORED if scored else STATUS_NOT_APPLICABLE,
    }


def _lookup(args: Any, name: str) -> Any:
    if isinstance(args, Mapping) and name in args:
        return args[name]
    current = args
    for part in name.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isascii() and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if _is_number(left) and _is_number(right):
        # int/float comparison is exact in Python and never overflows.
        return bool(left == right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _type_ok(value: Any, expected: str) -> bool:
    actual = _json_type(value)
    if expected == "number":
        return actual in {"integer", "number"}
    if expected == "integer":
        return actual == "integer" or (actual == "number" and float(value).is_integer())
    return actual == expected


def _preview(value: Any) -> str:
    """Short redacted rendering of a dataset-authored value (never of agent arguments)."""
    text = value if isinstance(value, str) else (_bounded_json_text(value, _MAX_ARGS_TEXT_CHARS) or _json_type(value))
    return _safe_text(text, _MAX_PREVIEW_CHARS)


def _describe(value: Any) -> str:
    """Type and size of an agent argument value: failure details never echo the value itself.

    Redaction cannot recognize every credential shape (URL userinfo, PEM
    bodies, ``mysql -p...``), so argument values stay out of persisted signals.
    """
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, str | list):
        return f"{_json_type(value)}(len={len(value)})"
    if isinstance(value, Mapping):
        return f"object(keys={len(value)})"
    return _json_type(value)


def _regex_search(pattern: str, value: Any, budget: list[float]) -> bool | str:
    """Search a bounded subject under a deadline; returns whether it matched, or why it was not checked.

    Patterns are dataset-authored and subjects agent-written, so a pattern that
    backtracks catastrophically (``^(\\w+\\s?)+$``, ``[a-z]*[a-z0-9]*!``...) must
    fail the check rather than hang collection. ``re`` cannot be interrupted, so
    matching uses the ``regex`` engine (``re``-compatible syntax) with a
    per-check timeout drawn from the trial's shared ``budget`` of seconds.
    """
    text = value if isinstance(value, str) else _bounded_json_text(value, _MAX_PATTERN_SUBJECT_CHARS)
    if text is None or len(text) > _MAX_PATTERN_SUBJECT_CHARS:
        return "value exceeds the pattern-check size limit"
    timeout = min(_PATTERN_TIMEOUT_SECONDS, budget[0])
    if timeout <= 0:
        return "pattern check timed out"
    started = time.monotonic()
    try:
        return regex.search(pattern, text, timeout=timeout) is not None
    except TimeoutError:
        return "pattern check timed out"
    except (regex.error, RecursionError, OverflowError, ValueError):
        return False
    finally:
        budget[0] -= time.monotonic() - started


def _schema_errors(
    value: Any, schema: Mapping[str, Any], path: str, errors: list[tuple[str, str]], depth: int, budget: list[float]
) -> None:
    if len(errors) >= _MAX_SCHEMA_ERRORS_PER_CALL or depth > MAX_SCHEMA_DEPTH + 1:
        return
    if "type" in schema:
        types = schema["type"]
        type_list = [types] if isinstance(types, str) else list(types)
        if not any(_type_ok(value, item) for item in type_list):
            errors.append((path, f"expected type {'|'.join(type_list)}, got {_json_type(value)}"))
            return
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        errors.append((path, f"value {_describe(value)} is not one of the allowed values"))
    if "pattern" in schema and isinstance(value, str):
        found = _regex_search(str(schema["pattern"]), value, budget)
        if isinstance(found, str):
            errors.append((path, found))
        elif not found:
            errors.append((path, f"value {_describe(value)} does not match the schema pattern"))
    if _is_number(value):
        if _is_number(schema.get("minimum")) and value < schema["minimum"]:
            errors.append((path, f"value is below minimum {_preview(schema['minimum'])}"))
        if _is_number(schema.get("maximum")) and value > schema["maximum"]:
            errors.append((path, f"value is above maximum {_preview(schema['maximum'])}"))
    if isinstance(value, Mapping):
        for name in schema.get("required") or ():
            if name not in value:
                errors.append((f"{path}.{name}", "required property is missing"))
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for name, sub in properties.items():
                if name in value and isinstance(sub, Mapping):
                    _schema_errors(value[name], sub, f"{path}.{name}", errors, depth + 1, budget)


def _argument_failures(
    args: Mapping[str, Any], rule: Mapping[str, Any], budget: list[float]
) -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    for name in rule.get("required") or ():
        if _lookup(args, name) is _MISSING:
            failures.append((name, "required", "argument is missing"))
    schema = rule.get("schema")
    if isinstance(schema, Mapping):
        schema_errors: list[tuple[str, str]] = []
        _schema_errors(args, schema, "$", schema_errors, 1, budget)
        for path, message in schema_errors:
            failures.append((path.removeprefix("$.") if path != "$" else "$", "schema", message))
    for name, expected in (rule.get("equals") or {}).items():
        actual = _lookup(args, name)
        if actual is _MISSING:
            failures.append((name, "equals", "argument is missing"))
        elif not _json_equal(actual, expected):
            failures.append((name, "equals", f"expected {_preview(expected)}, got {_describe(actual)}"))
    for name, needle in (rule.get("contains") or {}).items():
        actual = _lookup(args, name)
        if actual is _MISSING:
            failures.append((name, "contains", "argument is missing"))
            continue
        if isinstance(actual, str):
            found = needle in actual
        elif isinstance(actual, list):
            found = any(item == needle or (isinstance(item, str) and needle in item) for item in actual[:1024])
        else:
            found = needle in (_bounded_json_text(actual, _MAX_ARGS_TEXT_CHARS) or "")
        if not found:
            failures.append((name, "contains", f"value {_describe(actual)} does not contain {_preview(needle)}"))
    for name, pattern in (rule.get("pattern") or {}).items():
        actual = _lookup(args, name)
        if actual is _MISSING:
            failures.append((name, "pattern", "argument is missing"))
            continue
        found = _regex_search(pattern, actual, budget)
        if isinstance(found, str):
            failures.append((name, "pattern", found))
        elif not found:
            failures.append((name, "pattern", f"value {_describe(actual)} does not match {_preview(pattern)}"))
    return failures


def grade_arguments(calls: Sequence[_Call], spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Check ``tool_arguments`` rules against every call that matches each rule's ``tool``.

    ``checked``/``passed`` count (call, rule) pairs. A rule that matched no call
    is listed as a ``not_called`` failure but is not counted as checked, since
    whether the tool was called is ``tool_selection``'s job. Pattern checks
    share one time budget; a check that runs out of time is a failure, never a
    pass.
    """
    rules = _spec_field(spec, "tool_arguments")
    failures: list[dict[str, str]] = []
    checked = 0
    passed = 0
    budget = [_PATTERN_BUDGET_SECONDS]

    def _record(tool: str, arg: str, rule_name: str, detail: str) -> None:
        if len(failures) < _MAX_FAILURES:
            failures.append(
                {
                    "tool": _safe_text(tool),
                    "arg": _safe_text(arg, MAX_ARGUMENT_NAME_CHARS),
                    "rule": rule_name,
                    "detail": _safe_text(detail, _MAX_DETAIL_CHARS),
                }
            )

    for rule in rules:
        refs = _refs([rule["tool"]])
        matched = [call for call in calls if _call_matches(refs, call)]
        if not matched:
            _record(rule["tool"], "", "not_called", "no call matched this rule")
            continue
        for call in matched:
            checked += 1
            call_failures = _argument_failures(call.args, rule, budget)
            if not call_failures:
                passed += 1
            label = next(
                (ident.persisted_label for ident in call.idents if any(_ref_matches(r, ident) for r in refs)),
                call.fn,
            )
            for arg, rule_name, detail in call_failures:
                _record(label, arg, rule_name, detail)
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "status": STATUS_SCORED if rules else STATUS_NOT_APPLICABLE,
    }


def _outcome_counts() -> dict[str, Any]:
    return {"total": 0, "succeeded": 0, "failed": 0, "unknown": 0}


def _count_outcome(bucket: dict[str, Any], succeeded: bool | None) -> None:
    bucket["total"] += 1
    if succeeded is True:
        bucket["succeeded"] += 1
    elif succeeded is False:
        bucket["failed"] += 1
    else:
        bucket["unknown"] += 1


def grade_mcp_calls(calls: Sequence[_Call]) -> dict[str, Any]:
    """Outcome counts across every MCP call (any server), with a per-server breakdown.

    ``success_rate`` is ``succeeded / (succeeded + failed)``: calls whose outcome
    is unknown are reported separately rather than counted as either.
    """
    totals = _outcome_counts()
    by_server: dict[str, dict[str, Any]] = {}
    for call in calls:
        ident = call.mcp
        if ident is None:
            continue
        _count_outcome(totals, call.succeeded)
        server = _safe_text(ident.server or "")
        bucket = by_server.setdefault(server, {**_outcome_counts(), "tools": []})
        _count_outcome(bucket, call.succeeded)
        label = _safe_text(ident.label)
        if label not in bucket["tools"] and len(bucket["tools"]) < _MAX_SERVER_TOOLS:
            bucket["tools"].append(label)
    return {
        **totals,
        "success_rate": _ratio(totals["succeeded"], totals["succeeded"] + totals["failed"]),
        "by_server": by_server,
    }


def _first_seq(refs: Sequence[_Ref], calls: Sequence[_Call]) -> int | None:
    return next((call.seq for call in calls if _call_matches(refs, call)), None)


def grade_order(calls: Sequence[_Call], spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Check ``expected_order`` precedence edges by first occurrence (emission order)."""
    edges = _spec_field(spec, "expected_order")
    satisfied = 0
    violated: list[dict[str, str]] = []
    for before, after in edges:
        first_before = _first_seq(_refs(before), calls)
        first_after = _first_seq(_refs(after), calls)
        if first_before is not None and first_after is not None and first_before < first_after:
            satisfied += 1
        elif len(violated) < _MAX_FAILURES:
            violated.append({"before": _ref_label(before), "after": _ref_label(after)})
    return {
        "edges": len(edges),
        "satisfied": satisfied,
        "violated": violated,
        "status": STATUS_SCORED if edges else STATUS_NOT_APPLICABLE,
    }


def _normalized_path_matches(observed: str, artifact: str) -> bool:
    """Both paths already passed through :func:`_normalize_path`."""
    if not observed or not artifact:
        return False
    if artifact.startswith("/"):
        return observed == artifact
    return observed == artifact or observed.endswith("/" + artifact)


def _path_matches(observed: str, artifact: str) -> bool:
    """``artifact`` is normalized; ``observed`` is raw."""
    return _normalized_path_matches(_normalize_path(observed), artifact)


def _patch_targets(args: Mapping[str, Any]) -> list[str]:
    """Paths named by the file headers of an apply_patch body (``Add``/``Update``/``Delete File``, ``Move to``)."""
    paths: list[str] = []
    for key in _PATCH_BODY_KEYS:
        body = args.get(key)
        if isinstance(body, str):
            paths.extend(_patch_file_paths(body[:_MAX_ARGS_TEXT_CHARS]))
    return paths


def _call_writes(call: _Call, artifact: str) -> bool:
    fn_base = call.fn_base
    if fn_base in _WRITE_TOOLS:
        if fn_base in {"str_replace_editor", "str_replace_based_edit_tool"} and (
            str(call.args.get("command") or "").casefold() == "view"
        ):
            return False
        if any(_path_matches(path, artifact) for path in _path_args(call.args)):
            return True
        return fn_base in _PATCH_TOOLS and any(_path_matches(path, artifact) for path in _patch_targets(call.args))
    if call.is_shell:
        return any(_normalized_path_matches(path, artifact) for path in call.shell_paths[1])
    if call.mcp is None:
        return False
    tool = (call.mcp.tool or "").casefold()
    for key, value in call.args.items():
        if not isinstance(value, str) or not _path_matches(value, artifact):
            continue
        if _WRITE_VERB_RE.search(tool) or str(key).casefold() in _OUTPUT_ARG_KEYS:
            return True
    return False


def _call_reads(call: _Call, artifact: str) -> bool:
    fn_base = call.fn_base
    if (
        fn_base in _READ_TOOLS
        or (
            fn_base in {"str_replace_editor", "str_replace_based_edit_tool"}
            and str(call.args.get("command") or "").casefold() == "view"
        )
    ) and any(_path_matches(path, artifact) for path in _path_args(call.args)):
        return True
    if call.is_shell:
        return any(_normalized_path_matches(path, artifact) for path in call.shell_paths[0])
    if call.mcp is not None or any(ident.kind == COMPONENT_SUBAGENT for ident in call.idents):
        return any(
            _path_matches(token, artifact)
            for value in _string_values(call.args)
            for token in ([value, *value.split()] if len(value) <= _MAX_SHELL_TEXT_CHARS else [value])
        )
    return False


def _handoff_value_failure(
    value: str, producer_calls: Sequence[_Call], consumer_calls: Sequence[_Call], prompts: Sequence[str]
) -> str:
    if any(value in text for text in prompts):
        return "value also appears in the task prompt, so the handoff cannot be attributed"
    produced = next((call for call in producer_calls if call.observation and value in call.observation), None)
    if produced is None:
        return "value was not observed in producer output"
    received = any(
        call.seq != produced.seq and call.step_index > produced.step_index and value in call.args_text
        for call in consumer_calls
    )
    if not received:
        return "value did not reach consumer input after the producer produced it"
    return ""


def _handoff_artifact_failure(artifact: str, producer_calls: Sequence[_Call], consumer_calls: Sequence[_Call]) -> str:
    artifact = _normalize_path(artifact)
    written = next((call for call in producer_calls if _call_writes(call, artifact)), None)
    if written is None:
        return "artifact was not written by the producer"
    if not any(call.seq > written.seq and _call_reads(call, artifact) for call in consumer_calls):
        return "artifact was not read by the consumer after the producer wrote it"
    return ""


def grade_handoff(
    calls: Sequence[_Call], spec: Mapping[str, Any] | None, *, prompts: Sequence[str] = ()
) -> dict[str, Any]:
    """Verify each ``handoffs[i]`` carried producer output into consumer input (see module docs)."""
    handoffs = _spec_field(spec, "handoffs")
    failures: list[dict[str, str]] = []
    passed = 0
    for handoff in handoffs:
        producer_refs = _refs(handoff["producer"])
        consumer_refs = _refs(handoff["consumer"])
        producer_calls = _attributed(producer_refs, calls)
        consumer_calls = _attributed(consumer_refs, calls)
        problems: list[str] = []
        if not any(_call_matches(producer_refs, call) for call in calls):
            problems.append("producer was not activated")
        if not any(_call_matches(consumer_refs, call) for call in calls):
            problems.append("consumer was not activated")
        if not problems:
            if handoff.get("value") is not None:
                problem = _handoff_value_failure(handoff["value"], producer_calls, consumer_calls, prompts)
                if problem:
                    problems.append(problem)
            if handoff.get("artifact") is not None:
                problem = _handoff_artifact_failure(handoff["artifact"], producer_calls, consumer_calls)
                if problem:
                    problems.append(problem)
        if not problems:
            passed += 1
        elif len(failures) < _MAX_FAILURES:
            failures.append(
                {
                    "producer": _ref_label(handoff["producer"]),
                    "consumer": _ref_label(handoff["consumer"]),
                    "detail": _safe_text("; ".join(problems), _MAX_DETAIL_CHARS),
                }
            )
    return {
        "checked": len(handoffs),
        "passed": passed,
        "failures": failures,
        "status": STATUS_SCORED if handoffs else STATUS_NOT_APPLICABLE,
    }


def grade_conflict(calls: Sequence[_Call], spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Each probe passes when ``must_use`` was activated and ``must_not_use`` was not."""
    probes = _spec_field(spec, "conflict_probes")
    failures: list[dict[str, str]] = []
    passed = 0
    for probe in probes:
        used = any(_call_matches(_refs(probe["must_use"]), call) for call in calls)
        forbidden = any(_call_matches(_refs(probe["must_not_use"]), call) for call in calls)
        if used and not forbidden:
            passed += 1
            continue
        problems = []
        if not used:
            problems.append(f"must_use {_ref_label(probe['must_use'])} was not activated")
        if forbidden:
            problems.append(f"must_not_use {_ref_label(probe['must_not_use'])} was activated")
        if len(failures) < _MAX_FAILURES:
            failures.append(
                {"probe": _safe_text(probe["id"]), "detail": _safe_text("; ".join(problems), _MAX_DETAIL_CHARS)}
            )
    return {
        "checked": len(probes),
        "passed": passed,
        "failures": failures,
        "status": STATUS_SCORED if probes else STATUS_NOT_APPLICABLE,
    }


def _declared_keys(declared: Mapping[str, Sequence[str]] | None) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for kind in (COMPONENT_SKILL, COMPONENT_MCP, COMPONENT_SUBAGENT, COMPONENT_COMMAND):
        for name in (declared or {}).get(kind) or ():
            if isinstance(name, str) and name and (kind, name) not in keys:
                keys.append((kind, name))
    return keys


def _ident_is_component(ident: _Ident, kind: str, name: str, declared_mcp: Sequence[str] = ()) -> bool:
    if ident.kind != kind:
        return False
    if kind == COMPONENT_MCP:
        # ``_mcp_identity`` already maps a recognizable server to its declared name.
        return match_declared_mcp_server(ident.server or "", declared_mcp or (name,)) == name
    return name.casefold() in _name_candidates(ident.name)


def grade_activation_coverage(calls: Sequence[_Call], declared: Mapping[str, Sequence[str]] | None) -> dict[str, Any]:
    """Declared components exercised, never activated, or whose every activation failed.

    Entries are ``"<type>:<name>"``. ``unavailable`` is a subset of ``exercised``.
    """
    declared_labels: list[str] = []
    exercised: list[str] = []
    unverified: list[str] = []
    unavailable: list[str] = []
    declared_mcp = [name for kind, name in _declared_keys(declared) if kind == COMPONENT_MCP]
    for kind, name in _declared_keys(declared):
        label = _safe_text(f"{kind}:{name}")
        declared_labels.append(label)
        outcomes = [
            call.succeeded
            for call in calls
            if any(_ident_is_component(ident, kind, name, declared_mcp) for ident in call.idents)
        ]
        if not outcomes:
            unverified.append(label)
            continue
        exercised.append(label)
        if all(outcome is False for outcome in outcomes):
            unavailable.append(label)
    return {"declared": declared_labels, "exercised": exercised, "unverified": unverified, "unavailable": unavailable}


def _activations(calls: Sequence[_Call]) -> list[dict[str, Any]]:
    activations: list[dict[str, Any]] = []
    for call in calls:
        for ident in call.component_idents:
            if len(activations) >= _MAX_ACTIVATIONS:
                return activations
            activations.append(
                {
                    "type": ident.kind,
                    "name": _safe_text(ident.persisted_name),
                    "tool": _safe_text(ident.tool_label or ident.fn),
                    "server": _safe_text(ident.server) if ident.server is not None else None,
                    "step_index": call.step_index,
                    "succeeded": call.succeeded,
                }
            )
    return activations


def detect_component_activations(
    trajectory: Mapping[str, Any],
    declared: Mapping[str, Sequence[str]] | None = None,
    *,
    mcp_call_servers: Mapping[str, str] | None = None,
    subagent_aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Ordered component activations (C3 ``activations``) in one ATIF trajectory."""
    calls = (
        _extract_calls(trajectory, declared or {}, mcp_call_servers, subagent_aliases)
        if isinstance(trajectory, Mapping)
        else None
    )
    return _activations(calls or [])


def compute_plugin_signals(
    trajectory: Any,
    case: Mapping[str, Any] | None = None,
    *,
    declared: Mapping[str, Sequence[str]] | None = None,
    wrapper_skills: Sequence[str] = (),
    mcp_call_servers: Mapping[str, str] | None = None,
    subagent_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Per-trial C3 ``plugin_signals`` for one ATIF trajectory, or ``None`` if unreadable.

    ``case`` is a :func:`plugin_case_spec` result (raw entries are normalized
    defensively). ``declared`` maps ``skill``/``mcp`` (and, in the with-plugin
    arm, ``subagent``/``command``) to the declared component names in this arm.
    ``mcp_call_servers`` maps a tool call id to the MCP server the harness log
    says it went to, for harnesses whose trajectory keeps only the bare tool
    name (Codex). ``subagent_aliases`` maps a staged subagent name (casefolded)
    to the declared name, for harnesses that rename a plugin agent (OpenCode).
    """
    if not isinstance(trajectory, Mapping):
        return None
    declared_map: Mapping[str, Sequence[str]] = declared or {}
    calls = _extract_calls(trajectory, declared_map, mcp_call_servers, subagent_aliases)
    if calls is None:
        return None
    spec: Mapping[str, Any] = case if isinstance(case, Mapping) else {}
    return {
        "activations": _activations(calls),
        "tool_selection": grade_tool_selection(calls, spec, wrapper_skills=wrapper_skills),
        "arguments": grade_arguments(calls, spec),
        "mcp_calls": grade_mcp_calls(calls),
        "order": grade_order(calls, spec),
        "handoff": grade_handoff(calls, spec, prompts=_prompt_texts(trajectory)),
        "conflict": grade_conflict(calls, spec),
        "activation_coverage": grade_activation_coverage(calls, declared_map),
    }


# =============================================================================
# Per-arm summary
# =============================================================================


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _scored(signals: Sequence[Mapping[str, Any]], key: str) -> list[Mapping[str, Any]]:
    return [
        block
        for item in signals
        if isinstance(block := item.get(key), Mapping) and block.get("status") == STATUS_SCORED
    ]


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _summarize_checked(signals: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    blocks = _scored(signals, key)
    checked = sum(_int(block.get("checked")) for block in blocks)
    passed = sum(_int(block.get("passed")) for block in blocks)
    return {
        "n_scored": len(blocks),
        "checked": checked,
        "passed": passed,
        "pass_rate": _ratio(passed, checked),
        "status": STATUS_SCORED if blocks else STATUS_NOT_APPLICABLE,
    }


def _summarize_mcp(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = _outcome_counts()
    by_server: dict[str, dict[str, Any]] = {}
    for item in signals:
        block = item.get("mcp_calls")
        if not isinstance(block, Mapping):
            continue
        for key in ("total", "succeeded", "failed", "unknown"):
            totals[key] += _int(block.get(key))
        servers = block.get("by_server")
        if not isinstance(servers, Mapping):
            continue
        for server, counts in servers.items():
            if not isinstance(counts, Mapping):
                continue
            bucket = by_server.setdefault(str(server), {**_outcome_counts(), "tools": []})
            for key in ("total", "succeeded", "failed", "unknown"):
                bucket[key] += _int(counts.get(key))
            for tool in counts.get("tools") or ():
                if isinstance(tool, str) and tool not in bucket["tools"] and len(bucket["tools"]) < _MAX_SERVER_TOOLS:
                    bucket["tools"].append(tool)
    for bucket in by_server.values():
        bucket["success_rate"] = _ratio(bucket["succeeded"], bucket["succeeded"] + bucket["failed"])
    return {
        **totals,
        "success_rate": _ratio(totals["succeeded"], totals["succeeded"] + totals["failed"]),
        "by_server": by_server,
    }


def _summarize_coverage(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    declared: list[str] = []
    exercised_counts: dict[str, int] = {}
    unavailable_counts: dict[str, int] = {}
    for item in signals:
        block = item.get("activation_coverage")
        if not isinstance(block, Mapping):
            continue
        for label in block.get("declared") or ():
            if isinstance(label, str) and label not in declared:
                declared.append(label)
        for label in block.get("exercised") or ():
            if isinstance(label, str):
                exercised_counts[label] = exercised_counts.get(label, 0) + 1
        for label in block.get("unavailable") or ():
            if isinstance(label, str):
                unavailable_counts[label] = unavailable_counts.get(label, 0) + 1
    n = len(signals)
    exercised = [label for label in declared if exercised_counts.get(label)]
    return {
        "declared": declared,
        "exercised": exercised,
        "unverified": [label for label in declared if not exercised_counts.get(label)],
        # Unavailable across the arm: exercised in some trial and failed in every trial that exercised it.
        "unavailable": [
            label
            for label in exercised
            if unavailable_counts.get(label, 0) and unavailable_counts[label] == exercised_counts[label]
        ],
        "exercise_rate": {label: _ratio(exercised_counts.get(label, 0), n) for label in declared},
    }


def summarize_plugin_signals(signals: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Aggregate per-trial ``plugin_signals`` into one per-arm ``plugin_signals_summary``.

    ``None`` entries are trials whose trajectory could not be read; they are
    counted in ``n_missing_trajectory`` and otherwise ignored.
    """
    present = [item for item in signals if isinstance(item, Mapping)]
    n_trials = len(present)
    activations: dict[str, int] = {}
    total_activations = 0
    for item in present:
        for activation in item.get("activations") or ():
            if isinstance(activation, Mapping):
                kind = str(activation.get("type") or "")
                activations[kind] = activations.get(kind, 0) + 1
                total_activations += 1

    selection_blocks = _scored(present, "tool_selection")

    def _values(name: str) -> list[float]:
        return [float(block[name]) for block in selection_blocks if _is_number(block.get(name))]

    order_blocks = _scored(present, "order")
    edges = sum(_int(block.get("edges")) for block in order_blocks)
    satisfied = sum(_int(block.get("satisfied")) for block in order_blocks)
    return {
        "n_trials": n_trials,
        "n_missing_trajectory": len(signals) - n_trials,
        "activations": {
            "total": total_activations,
            "mean_per_trial": _ratio(total_activations, n_trials),
            "by_type": activations,
        },
        "tool_selection": {
            "n_scored": len(selection_blocks),
            "precision": _mean(_values("precision")),
            "recall": _mean(_values("recall")),
            "f1": _mean(_values("f1")),
            "decoy_calls": sum(_int(block.get("decoy_calls")) for block in selection_blocks),
            "decoy_call_rate": _ratio(
                sum(1 for block in selection_blocks if _int(block.get("decoy_calls")) > 0), len(selection_blocks)
            ),
            "status": STATUS_SCORED if selection_blocks else STATUS_NOT_APPLICABLE,
        },
        "arguments": _summarize_checked(present, "arguments"),
        "mcp_calls": _summarize_mcp(present),
        "order": {
            "n_scored": len(order_blocks),
            "edges": edges,
            "satisfied": satisfied,
            "satisfaction_rate": _ratio(satisfied, edges),
            "status": STATUS_SCORED if order_blocks else STATUS_NOT_APPLICABLE,
        },
        "handoff": _summarize_checked(present, "handoff"),
        "conflict": _summarize_checked(present, "conflict"),
        "activation_coverage": _summarize_coverage(present),
    }


__all__ = [
    "ARM_SUM_OF_PARTS",
    "ARM_WITHOUT_SKILL",
    "ARM_WITH_SKILL",
    "COMPONENT_COMMAND",
    "COMPONENT_MCP",
    "COMPONENT_RULE_READ",
    "COMPONENT_SKILL",
    "COMPONENT_SUBAGENT",
    "PLUGIN_CASE_FIELDS",
    "STATUS_NOT_APPLICABLE",
    "STATUS_SCORED",
    "PluginSignalsContext",
    "build_plugin_signals_context",
    "compute_plugin_signals",
    "detect_component_activations",
    "match_declared_mcp_server",
    "plugin_case_spec",
    "summarize_plugin_signals",
    "validate_plugin_case_fields",
]
