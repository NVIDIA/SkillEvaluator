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
  other agents are also recognized and canonicalized to ``mcp__<server>__<tool>``.
* ``subagent`` -- ``Task``/``Agent`` (name from ``subagent_type``...).
* ``command``  -- ``SlashCommand`` (name from the first token of ``command``).
* ``rule_read`` is reserved by the output contract but never emitted: plugin
  rules are inlined into the generated wrapper skill, so there is no separate
  rule file for an agent to read.

Native Codex ``exec`` wrappers are unwrapped with
:mod:`~skillevaluator.tier3.eval_core.codex_tool_call_normalizer` first. An
inner call only inherits the outer observation when the normalizer proves it
owns that observation; otherwise its outcome is unknown.

``succeeded`` is tri-state. ``True``: a correlated result came back without a
structured error flag or a failure/unavailable marker in its leading text.
``False``: the correlated result is flagged or carries such a marker. ``None``:
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
  tools, shell redirects/``tee``/``cp``/``mv``/``touch``, or an MCP tool whose
  name or argument key says it writes), and a later consumer-attributed call
  must read it (read tools, shell readers/interpreters, or an MCP/subagent call
  whose arguments name the path). Relative artifact paths match an observed
  path equal to it or ending with ``/<artifact>``; absolute ones must match
  exactly.
* When both are given, both must hold.

Trajectory content is untrusted: step, call, text and regex-subject sizes are
bounded, and every persisted string is redacted (from a bounded window) and
truncated. Argument values are never persisted: failure details describe them
by type and size, and a component name taken from tool arguments is kept only
when it is identifier-shaped or a declared member.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
_WRITE_TOOLS = frozenset(
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
        "apply_patch",
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
    "no such file or directory",
    "file does not exist",
)
# A result that came back as a runtime/transport failure rather than an answer.
_FAILED_CALL_RE = re.compile(
    r"(?:\b[45]\d{2}\b\s*[:\-])"
    r"|<tool_use_error>"
    r"|\bmcp error\b"
    r"|\bpermission denied\b"
    r"|\bunauthorized\b"
    r"|\bforbidden\b"
    r"|\binternal server error\b"
    r"|\bfailed to decrypt\b"
    r"|\binvalid[_ ]token\b"
    r"|\btoken (?:expired|is expired)\b",
    re.IGNORECASE,
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
    try:
        re.compile(value)
    except (re.error, RecursionError, OverflowError) as exc:
        errors.add(where, f"is not a valid regular expression ({exc})")
        return None
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


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
        and float(schema["minimum"]) > float(schema["maximum"])
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

    def arm_enabled(self, arm: str) -> bool:
        if arm in {ARM_WITH_SKILL, ARM_SUM_OF_PARTS}:
            return True
        return arm == ARM_WITHOUT_SKILL and self.baseline_has_members

    def declared_for(self, arm: str) -> dict[str, list[str]]:
        """Declared components staged in ``arm`` (MCP is wired into the with-plugin arm only)."""
        declared: dict[str, list[str]] = {COMPONENT_SKILL: list(self.member_skills)}
        declared[COMPONENT_MCP] = list(self.mcp_servers) if arm == ARM_WITH_SKILL else []
        return declared

    def case_spec(self, case_id: str) -> Mapping[str, Any]:
        return self.cases.get(case_id) or {}


def build_plugin_signals_context(
    *,
    member_skills: Iterable[Any] = (),
    mcp_servers: Iterable[Any] = (),
    wrapper_skills: Iterable[Any] = (),
    entries: Iterable[Any] = (),
    baseline_has_members: bool = False,
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
    return PluginSignalsContext(
        member_skills=_clean_names(member_skills),
        mcp_servers=_clean_names(mcp_servers),
        wrapper_skills=_clean_names(wrapper_skills),
        cases=cases,
        baseline_has_members=bool(baseline_has_members),
    )


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
    seq: int
    step_index: int
    fn: str
    args: dict[str, Any]
    observation: str | None
    succeeded: bool | None
    idents: list[_Ident]
    owner: int | None = None
    _args_text: str | None = None
    _shell_paths: tuple[list[str], list[str]] | None = None

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
            fn_base = _base_tool_name(self.fn)
            if fn_base in _SHELL_TOOLS and not any(ident.kind == COMPONENT_MCP for ident in self.idents):
                for text in _shell_texts(fn_base, self.args):
                    text_reads, text_writes = _shell_io(text, reader_verbs=_ARTIFACT_CONSUMER_VERBS)
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


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:_MAX_OBSERVATION_CHARS]
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
        return "\n".join(parts)[:_MAX_OBSERVATION_CHARS]
    return (_bounded_json_text(content, _MAX_OBSERVATION_CHARS) or "")[:_MAX_OBSERVATION_CHARS]


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


def _observations(step: Mapping[str, Any]) -> list[tuple[str, str, bool]]:
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        return []
    results = observation.get("results")
    if not isinstance(results, list):
        return []
    entries: list[tuple[str, str, bool]] = []
    for result in results[:256]:
        if not isinstance(result, Mapping):
            continue
        entries.append(
            (
                str(result.get("source_call_id") or ""),
                _content_text(result.get("content")),
                _result_flagged(result),
            )
        )
    return entries


def _results_for_call(results: list[tuple[str, str, bool]], call_id: str, *, call_count: int) -> list[tuple[str, bool]]:
    """Results attributable to ``call_id``; id matches win, ambiguity stays unknown."""
    if call_id:
        matched = [(text, flagged) for rid, text, flagged in results if rid == call_id]
        if matched:
            return matched
    if call_count == 1 and len(results) == 1 and not results[0][0]:
        return [(results[0][1], results[0][2])]
    return []


def _outcome(correlated: list[tuple[str, bool]]) -> tuple[str | None, bool | None]:
    if not correlated:
        return None, None
    text = "".join(item for item, _ in correlated)[:_MAX_OBSERVATION_CHARS]
    flagged = any(is_error for _, is_error in correlated)
    if not flagged and not text.strip():
        return text, None
    head = text[:_FAILURE_SCAN_CHARS]
    lowered = head.casefold()
    failed = flagged or any(marker in lowered for marker in _UNAVAILABLE_MARKERS) or bool(_FAILED_CALL_RE.search(head))
    return text, not failed


def _base_tool_name(fn: str) -> str:
    low = fn.casefold()
    for separator in _TOOL_NAME_SEPARATORS:
        if separator in low:
            low = low.rsplit(separator, 1)[-1]
    return low


def _norm_server(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", value.casefold())


def _mcp_identity(fn: str, declared_mcp: Sequence[str]) -> tuple[str, str] | None:
    if fn[:5].casefold() == "mcp__":
        rest = fn[5:]
        server, _, tool = rest.partition("__")
        return (server, tool) if server else None
    low = fn.casefold()
    for server in declared_mcp:
        for candidate in dict.fromkeys((server.casefold(), _norm_server(server))):
            for separator in ("__", ".", "/", ":"):
                prefix = candidate + separator
                if low.startswith(prefix) and len(low) > len(prefix):
                    return server, fn[len(prefix) :]
    return None


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


def _shell_tokens(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ; ")
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
        try:
            return shlex.split(text, posix=True)[:_MAX_SHELL_TOKENS]
        except ValueError:
            return text.split()[:_MAX_SHELL_TOKENS]
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


def _declared_skill_read(fn: str, fn_base: str, args: Mapping[str, Any], members: Sequence[str]) -> str | None:
    """Return the declared member whose ``SKILL.md`` this call reads, if any."""
    if not members:
        return None
    is_mcp = fn[:5].casefold() == "mcp__"
    reads_file = fn_base in _READ_TOOLS or (is_mcp and bool(_READ_VERB_RE.search(fn_base)))
    if fn_base in {"str_replace_editor", "str_replace_based_edit_tool"}:
        reads_file = str(args.get("command") or "").casefold() == "view"
    if reads_file:
        for path in _path_args(args):
            member = _member_manifest_match(path, members)
            if member is not None:
                return member
        return None
    if fn_base in _SHELL_TOOLS and not is_mcp:
        for text in _shell_texts(fn_base, args):
            if "skill.md" not in text.casefold():
                continue
            reads, _writes = _shell_io(text, reader_verbs=_FILE_READER_VERBS)
            for path in reads:
                member = _member_manifest_match(path, members)
                if member is not None:
                    return member
    return None


def _component_ident(kind: str, name: str, fn: str, tool_label: str, *, persist: bool = True) -> _Ident:
    label = f"{_IDENTITY_PREFIX[kind]}:{name}" if name else _IDENTITY_PREFIX[kind]
    return _Ident(label=label, kind=kind, name=name, fn=fn, tool_label=tool_label, persist_name=persist)


def _persistable_name(name: str, members: Sequence[str]) -> bool:
    """Whether an argument-derived component name may be persisted (identifier-shaped or declared)."""
    if not name or (_PERSISTABLE_NAME_RE.fullmatch(name) and "://" not in name):
        return True
    folded = name.casefold()
    return any(member.casefold() == folded for member in members)


def _identities(fn: str, args: Mapping[str, Any], declared: Mapping[str, Sequence[str]]) -> list[_Ident]:
    low = fn.casefold()
    fn_base = _base_tool_name(fn)
    idents: list[_Ident] = []
    if low in _SKILL_TOOLS:
        name = _first_string(args, ("skill", "name", "command"))
        persist = _persistable_name(name, declared.get(COMPONENT_SKILL) or ())
        idents.append(_component_ident(COMPONENT_SKILL, name, fn, fn, persist=persist))
    elif low in _SUBAGENT_TOOLS:
        name = _first_string(args, ("subagent_type", "subagent", "agent", "agent_name", "agent_type"))
        idents.append(_component_ident(COMPONENT_SUBAGENT, name, fn, fn, persist=_persistable_name(name, ())))
    elif low in _COMMAND_TOOLS:
        command = _first_string(args, ("command", "name"))
        name = command.split()[0].lstrip("/") if command.split() else ""
        idents.append(_component_ident(COMPONENT_COMMAND, name, fn, fn, persist=_persistable_name(name, ())))
    mcp = _mcp_identity(fn, declared.get(COMPONENT_MCP) or ())
    if mcp is not None:
        server, tool = mcp
        canonical = f"mcp__{server}__{tool}" if tool else f"mcp__{server}"
        idents.append(
            _Ident(
                label=canonical,
                kind=COMPONENT_MCP,
                name=server,
                fn=fn,
                server=server,
                tool=tool,
                tool_label=canonical,
            )
        )
    if not any(ident.kind == COMPONENT_SKILL for ident in idents):
        member = _declared_skill_read(fn, fn_base, args, declared.get(COMPONENT_SKILL) or ())
        if member is not None:
            idents.append(_component_ident(COMPONENT_SKILL, member, fn, f"{fn}:skill-md-read"))
    if not idents:
        idents.append(_Ident(label=fn, kind=None, name=fn, fn=fn, tool_label=fn))
    return idents


def _extract_calls(trajectory: Mapping[str, Any], declared: Mapping[str, Sequence[str]]) -> list[_Call] | None:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
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
            prepared = {**raw, "function_name": _tool_name(raw), "arguments": _arguments(raw)}
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
                observation, succeeded = _outcome(correlated)
                fn = str(tool_call.get("function_name") or "")[:_MAX_LABEL_CHARS]
                args = tool_call.get("arguments")
                args = args if isinstance(args, dict) else {}
                idents = _identities(fn, args, declared)
                seq = len(calls)
                if any(ident.kind in {COMPONENT_SKILL, COMPONENT_COMMAND} for ident in idents):
                    owner = seq
                calls.append(
                    _Call(
                        seq=seq,
                        step_index=step_index,
                        fn=fn,
                        args=args,
                        observation=observation,
                        succeeded=succeeded,
                        idents=idents,
                        owner=owner,
                    )
                )
    return calls


def _prompt_texts(trajectory: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return texts
    for step in steps[:_MAX_STEPS]:
        if not isinstance(step, Mapping) or step.get("source") not in {"user", "system"}:
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
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if _is_number(left) and _is_number(right):
        return float(left) == float(right)
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


def _regex_search(pattern: str, value: Any) -> bool | None:
    """``re.search`` on a bounded subject; ``None`` when the subject is too large."""
    text = value if isinstance(value, str) else _bounded_json_text(value, _MAX_PATTERN_SUBJECT_CHARS)
    if text is None or len(text) > _MAX_PATTERN_SUBJECT_CHARS:
        return None
    try:
        return re.search(pattern, text) is not None
    except (re.error, RecursionError, OverflowError):
        return False


def _schema_errors(value: Any, schema: Mapping[str, Any], path: str, errors: list[tuple[str, str]], depth: int) -> None:
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
        found = _regex_search(str(schema["pattern"]), value)
        if found is None:
            errors.append((path, "value exceeds the pattern-check size limit"))
        elif not found:
            errors.append((path, f"value {_describe(value)} does not match the schema pattern"))
    if _is_number(value):
        if _is_number(schema.get("minimum")) and float(value) < float(schema["minimum"]):
            errors.append((path, f"value is below minimum {_preview(schema['minimum'])}"))
        if _is_number(schema.get("maximum")) and float(value) > float(schema["maximum"]):
            errors.append((path, f"value is above maximum {_preview(schema['maximum'])}"))
    if isinstance(value, Mapping):
        for name in schema.get("required") or ():
            if name not in value:
                errors.append((f"{path}.{name}", "required property is missing"))
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for name, sub in properties.items():
                if name in value and isinstance(sub, Mapping):
                    _schema_errors(value[name], sub, f"{path}.{name}", errors, depth + 1)


def _argument_failures(args: Mapping[str, Any], rule: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    for name in rule.get("required") or ():
        if _lookup(args, name) is _MISSING:
            failures.append((name, "required", "argument is missing"))
    schema = rule.get("schema")
    if isinstance(schema, Mapping):
        schema_errors: list[tuple[str, str]] = []
        _schema_errors(args, schema, "$", schema_errors, 1)
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
        found = _regex_search(pattern, actual)
        if found is None:
            failures.append((name, "pattern", "value exceeds the pattern-check size limit"))
        elif not found:
            failures.append((name, "pattern", f"value {_describe(actual)} does not match {_preview(pattern)}"))
    return failures


def grade_arguments(calls: Sequence[_Call], spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Check ``tool_arguments`` rules against every call that matches each rule's ``tool``.

    ``checked``/``passed`` count (call, rule) pairs. A rule that matched no call
    is listed as a ``not_called`` failure but is not counted as checked, since
    whether the tool was called is ``tool_selection``'s job.
    """
    rules = _spec_field(spec, "tool_arguments")
    failures: list[dict[str, str]] = []
    checked = 0
    passed = 0

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
            call_failures = _argument_failures(call.args, rule)
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
        ident = next((item for item in call.idents if item.kind == COMPONENT_MCP), None)
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


def _is_mcp_call(call: _Call) -> bool:
    return any(ident.kind == COMPONENT_MCP for ident in call.idents)


def _call_writes(call: _Call, artifact: str) -> bool:
    fn_base = _base_tool_name(call.fn)
    if fn_base in _WRITE_TOOLS:
        if fn_base in {"str_replace_editor", "str_replace_based_edit_tool"} and (
            str(call.args.get("command") or "").casefold() == "view"
        ):
            return False
        if any(_path_matches(path, artifact) for path in _path_args(call.args)):
            return True
        if fn_base == "apply_patch":
            patch = _first_string(call.args, ("input", "patch", "content"))
            for match in re.finditer(r"^\*\*\* (?:Add|Update) File: (.+)$", patch[:_MAX_ARGS_TEXT_CHARS], re.MULTILINE):
                if _path_matches(match.group(1), artifact):
                    return True
        return False
    if fn_base in _SHELL_TOOLS and not _is_mcp_call(call):
        return any(_normalized_path_matches(path, artifact) for path in call.shell_paths[1])
    ident = next((item for item in call.idents if item.kind == COMPONENT_MCP), None)
    if ident is None:
        return False
    tool = (ident.tool or "").casefold()
    for key, value in call.args.items():
        if not isinstance(value, str) or not _path_matches(value, artifact):
            continue
        if _WRITE_VERB_RE.search(tool) or str(key).casefold() in _OUTPUT_ARG_KEYS:
            return True
    return False


def _call_reads(call: _Call, artifact: str) -> bool:
    fn_base = _base_tool_name(call.fn)
    if (
        fn_base in _READ_TOOLS
        or (
            fn_base in {"str_replace_editor", "str_replace_based_edit_tool"}
            and str(call.args.get("command") or "").casefold() == "view"
        )
    ) and any(_path_matches(path, artifact) for path in _path_args(call.args)):
        return True
    if fn_base in _SHELL_TOOLS and not _is_mcp_call(call):
        return any(_normalized_path_matches(path, artifact) for path in call.shell_paths[0])
    if any(ident.kind in {COMPONENT_MCP, COMPONENT_SUBAGENT} for ident in call.idents):
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
    for kind in (COMPONENT_SKILL, COMPONENT_MCP):
        for name in (declared or {}).get(kind) or ():
            if isinstance(name, str) and name and (kind, name) not in keys:
                keys.append((kind, name))
    return keys


def _ident_is_component(ident: _Ident, kind: str, name: str) -> bool:
    if ident.kind != kind:
        return False
    if kind == COMPONENT_MCP:
        return _norm_server(ident.server or "") == _norm_server(name)
    return name.casefold() in _name_candidates(ident.name)


def grade_activation_coverage(calls: Sequence[_Call], declared: Mapping[str, Sequence[str]] | None) -> dict[str, Any]:
    """Declared components exercised, never activated, or whose every activation failed.

    Entries are ``"<type>:<name>"``. ``unavailable`` is a subset of ``exercised``.
    """
    declared_labels: list[str] = []
    exercised: list[str] = []
    unverified: list[str] = []
    unavailable: list[str] = []
    for kind, name in _declared_keys(declared):
        label = _safe_text(f"{kind}:{name}")
        declared_labels.append(label)
        outcomes = [
            call.succeeded for call in calls if any(_ident_is_component(ident, kind, name) for ident in call.idents)
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
    trajectory: Mapping[str, Any], declared: Mapping[str, Sequence[str]] | None = None
) -> list[dict[str, Any]]:
    """Ordered component activations (C3 ``activations``) in one ATIF trajectory."""
    calls = _extract_calls(trajectory, declared or {}) if isinstance(trajectory, Mapping) else None
    return _activations(calls or [])


def compute_plugin_signals(
    trajectory: Any,
    case: Mapping[str, Any] | None = None,
    *,
    declared: Mapping[str, Sequence[str]] | None = None,
    wrapper_skills: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Per-trial C3 ``plugin_signals`` for one ATIF trajectory, or ``None`` if unreadable.

    ``case`` is a :func:`plugin_case_spec` result (raw entries are normalized
    defensively). ``declared`` maps ``skill``/``mcp`` to the declared component
    names staged in this arm.
    """
    if not isinstance(trajectory, Mapping):
        return None
    declared_map: Mapping[str, Sequence[str]] = declared or {}
    calls = _extract_calls(trajectory, declared_map)
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
    "plugin_case_spec",
    "summarize_plugin_signals",
    "validate_plugin_case_fields",
]
