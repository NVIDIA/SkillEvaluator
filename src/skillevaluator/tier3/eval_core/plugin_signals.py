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
  ``grep``/``printf``/``cp``/``sed -i`` of the same path, or ``<`` input to a
  non-reader, never count. The scored skill-read check is stricter (see
  ``_FILE_READER_VERBS``), so a read credited here may not count toward a
  score. Each read of a shell list has its own outcome: a
  read after a failed ``&&`` part never ran, and a read whose own error line
  names it failed. Skill, subagent and command names are this plugin's
  declared names with or without its ``<plugin>:`` namespace; another plugin's
  namespaced name stays foreign and never matches.
* ``mcp``      -- ``mcp__<server>__<tool>``; for *declared* runnable servers the
  ``<server>__<tool>``/``<server>.<tool>``/``<server>/<tool>`` spellings used by
  other agents, Hermes ``mcp_<server>_<tool>`` and OpenCode ``<server>_<tool>``
  are also recognized and canonicalized to ``mcp__<server>__<tool>``. Claude
  Code's plugin servers (``mcp__plugin_<plugin>_<server>__<tool>``) and
  harness-sanitized names map back to the declared server; Codex bare tool
  names map through the server its log recorded (``mcp_call_servers``).
* ``subagent`` -- ``Task``/``Agent`` (name from ``subagent_type``..., else ``type``).
* ``command``  -- ``SlashCommand`` (name from the first token of ``command``),
  or the ``Skill`` tool naming a declared command (``<plugin>:<command>``),
  which is how Claude Code runs plugin commands.
* ``lsp``      -- Claude Code's ``LSP`` tool, credited to the one plugin LSP
  server staged in this arm whose ``extensionToLanguage`` covers the extension
  of the call's ``filePath`` (``lsp_servers``). An extension that two staged
  servers claim names neither.
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
  ``Command:<name>`` (alias ``SlashCommand:``), ``MCP:<server>`` (or
  ``MCP:<server>/<tool>``) and ``LSP:<server>`` match a component activation
  of that type (a bare ``LSP`` ref is the tool itself);
* any other ref matches the canonical tool label (``mcp__server__tool``,
  ``Skill:<name>``...), the raw tool function name (``Bash``, ``Read``...) or
  its harness-neutral alias (``Bash`` is also Codex ``exec_command``; ``Write``
  and ``Edit`` are also any shell call that writes a file), an MCP tool's bare
  name, or a component name -- never a component through the tool that
  carried it, and never the generated wrapper skill (the ``<plugin>-plugin-eval``
  package, or the plugin's own name when no member skill or command has it).

Check 15 (``routing``) scores the refs about skills, subagents, and commands;
check 22 (``tool_selection``) scores the refs about MCP, LSP, and plain tools.

Handoff heuristics (conservative)
---------------------------------
Calls are *attributed* to a ref when they match it directly, or when they run
inside the window opened by a matching skill/command activation (see
``_handoff_calls``: only a successful switch to one declared skill or command
opens or closes a window, and the consumer's own activation does not close the
producer's). A subagent's own calls (Claude Code sidechain steps) keep their
own window and also belong to a ref naming that subagent: the parent call whose
result names the sidechain's ``agentId``, else the parent's latest subagent call.

* ``value``: the literal must appear in the result of a producer-attributed
  call, then in the arguments of a consumer-attributed call in a strictly later
  step (a result is only visible to the model after its step). The value must
  not appear in the task prompt (the user/system messages before the agent's
  first step), since then the consumer could have taken it from the prompt
  rather than from the producer. Later user steps (Claude Code's skill-load
  step, a subagent's prompt) are written by the harness or the agent.
* ``artifact``: a producer-attributed call must write the path (write/edit
  tools, the ``Add File``/``Update File``/``Move to`` headers of an
  ``apply_patch`` body, shell redirects/``tee``/``cp``/``mv``/``touch``/``sed -i``,
  interpreter code given inline or as a here-document, or an MCP tool whose
  name or argument key says it writes; a patch that only deletes the path, or
  moves it away, does not write it),
  and a consumer-attributed call in a later step must read it (read tools,
  shell readers/interpreters/globs, or an MCP tool taking the path as input).
  Paths resolve against the call's working directory and the task's; when a
  directory is unknown a path ending with ``/<artifact>`` matches.

Values are compared as parsed values (argument string leaves, and JSON text
inside results), as whole tokens.
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
import posixpath
import re
import shlex
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, NamedTuple

import regex

from skillevaluator.tier3.eval_core.checks import _APPLY_PATCH_COMMAND_RE, _APPLY_PATCH_HEADER_RE, _FILE_READ_VERBS
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import (
    MAPPED_OUTER_EXEC_OBSERVATION,
    normalize_tool_call,
)
from skillevaluator.tier3.harbor.stats import ARM_SUM_OF_PARTS, ARM_WITH, ARM_WITHOUT
from skillevaluator.tier3.plugin_native import OPENCODE_BUILTIN_AGENTS
from skillevaluator.utils.redaction import redact_sensitive_text

COMPONENT_SKILL = "skill"
COMPONENT_MCP = "mcp"
COMPONENT_SUBAGENT = "subagent"
COMPONENT_COMMAND = "command"
COMPONENT_LSP = "lsp"
COMPONENT_RULE_READ = "rule_read"

STATUS_SCORED = "scored"
STATUS_NOT_APPLICABLE = "not_applicable"

# The arm names are harbor.stats'; these aliases keep the names the collector and tests import.
ARM_WITH_SKILL = ARM_WITH
ARM_WITHOUT_SKILL = ARM_WITHOUT

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
_MAX_STEP_RESULTS = 256
_MAX_CONTENT_BLOCKS = 256
# String values (never key names) read from a JSON argument, result or ``contains`` object.
_MAX_STRING_LEAVES = 1_024
# List elements a ``contains`` argument check looks at.
_MAX_CONTAINS_ITEMS = 1_024
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
# How much of an unsupported dataset key a problem message quotes.
_MAX_KEY_PREVIEW_CHARS = 64
_MAX_PREVIEW_CHARS = 80
_MAX_SCHEMA_ERRORS_PER_CALL = 5

_SKILL_TOOLS = frozenset({"skill"})
_SUBAGENT_TOOLS = frozenset({"task", "agent"})
_COMMAND_TOOLS = frozenset({"slashcommand", "slash_command"})
# Claude Code's LSP tool: Claude Code sends it to the server whose ``extensionToLanguage`` maps the file's extension.
_LSP_TOOLS = frozenset({"lsp"})
_LSP_PATH_KEYS = ("filePath", "file_path")
# An ``extensionToLanguage`` key: a dot and one suffix (``.lab``; a file's extension is never ``.d.ts``).
_LSP_EXTENSION_RE = re.compile(r"\.[^./\\\s]{1,32}")
_MAX_LSP_EXTENSIONS = 64
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
# ``patch``, and ``raw`` or ``value`` for a tool input that was not a JSON object.
_PATCH_BODY_KEYS = ("input", "patch", "patchText", "content", "raw", "value")
# Text editor tools: they write, except for their ``view`` command, which reads.
_STR_REPLACE_EDITORS = frozenset({"str_replace_editor", "str_replace_based_edit_tool"})
_WRITE_TOOLS = (
    _PATCH_TOOLS
    | _STR_REPLACE_EDITORS
    | frozenset(
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
            "save_file",
        }
    )
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
# MCP tool-name spellings (see ``_McpNames.identity``). Hermes names MCP tools
# ``mcp_<server>_<tool>`` with every character outside ``[A-Za-z0-9_]`` as ``_``;
# Claude Code names plugin servers ``plugin_<plugin>_<server>``.
_HERMES_NAME_RE = re.compile(r"[^a-z0-9_]")
# Claude Code and OpenCode write a server name with every character outside ``[A-Za-z0-9_-]`` as ``_``.
_SANITIZED_NAME_RE = re.compile(r"[^a-z0-9_-]")
_MCP_PREFIX = "mcp__"
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
# Shell commands that read a SKILL.md: the scored check's reader verbs plus a few
# more viewers. This report-only detector is deliberately more lenient than the
# scored checks._cmd_reads_skill_md: it also looks past prefix words (``sudo``,
# ``env``...) and into ``bash -c`` payloads. Unlike the scored check it does not
# expand shell variables, so ``D=skills/x; cat $D/SKILL.md`` counts only there.
_FILE_READER_VERBS = frozenset(_FILE_READ_VERBS) | frozenset({"batcat", "view", "xxd", "od", "tac"})
_ARTIFACT_CONSUMER_VERBS = _FILE_READER_VERBS | frozenset(
    {"jq", "yq", "python", "python3", "node", "grep", "rg", "wc", "sort", "uniq", "cut", "diff", "cmp", "base64"}
)
_SHELL_SEPARATORS = frozenset({"&&", "||", ";", "|", "&", "(", ")", ";;", "|&", ";&"})
_OUTPUT_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", "1>", "2>"})
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_REPEATED_SLASHES_RE = re.compile(r"/{2,}")
# ``<<EOF``/``<<-'EOF'``/``<<"EOF"`` (not ``<<<`` here-strings); the groups hold the delimiter.
_HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?[ \t]*(?:'([^'\n]+)'|\"([^\"\n]+)\"|\\?([A-Za-z_][A-Za-z0-9_]*))")
_WRITE_VERB_RE = re.compile(
    r"(?:^|_|-)(?:write|create|save|export|put|upload|dump|store|generate|render|append)(?:$|_|-)"
)
_OUTPUT_ARG_KEYS = frozenset(
    {"output", "out", "output_path", "out_path", "output_file", "dest", "destination", "target", "save_path"}
)
_READ_VERB_RE = re.compile(r"(?:^|_|-)(?:read|open|get|load|view|fetch|cat)(?:$|_|-)")
# A ``declared`` mapping may name the plugin itself under this key. Claude Code
# namespaces plugin skills, agents, and commands as ``<plugin>:<name>``; a name
# under another plugin's namespace is not this plugin's component.
DECLARED_PLUGIN = "plugin"
# ... and may list the component types the plugin declares but this arm does not stage.
DECLARED_UNSTAGED = "unstaged"
# ... and, as ``<type>:<name>``, the subagents and commands an arm leaves out of its declared names
# entirely, so an untyped ref that names one is still read as a ref to that component.
DECLARED_UNSTAGED_NAMES = "unstaged_names"
_WRAPPER_PACKAGE_SUFFIX = "-plugin-eval"
# A shell tool's error line that says one file operand could not be read.
_READ_FAILURE_PHRASES = (
    "no such file",
    "permission denied",
    "is a directory",
    "not a directory",
    "cannot open",
    "can't open",
    "cannot read",
    "can't read",
    "cannot access",
)
# Harness-neutral names for plain tools, so one unprefixed ref means the same
# thing on every harness: Codex writes files with ``apply_patch``, runs shell
# commands with ``exec_command``, has one hosted web tool (``web_search_call``,
# both search and page open), and plans with ``update_plan``.
_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "bash": ("exec_command", "shell", "local_shell"),
    "exec_command": ("bash", "shell"),
    "shell": ("bash", "exec_command"),
    "local_shell": ("bash", "exec_command", "shell"),
    "local_shell_call": ("bash", "exec_command", "shell"),
    "shell_command": ("bash", "exec_command", "shell"),
    "apply_patch": ("write", "edit", "multiedit"),
    "write": ("apply_patch",),
    "edit": ("apply_patch",),
    "multiedit": ("apply_patch",),
    "web_search_call": ("websearch", "webfetch", "web_search", "web_fetch"),
    "web_search": ("websearch", "web_search_call"),
    "websearch": ("web_search_call", "web_search"),
    "webfetch": ("web_search_call", "web_fetch"),
    "update_plan": ("todowrite",),
    "todowrite": ("update_plan",),
}
# A shell call that writes a file also answers to the file-writing tool names. Codex
# writes most files through ``exec_command`` (``cat > f <<'EOF'``, ``python3 - <<'PY'``),
# so an unprefixed ``Write`` or ``Edit`` ref means "wrote a file" on every harness.
_SHELL_WRITE_ALIASES = ("write", "edit", "multiedit")
# A failed result that says the called tool, skill, command, or agent does not exist.
_MISSING_TOOL_MARKERS = (
    "no such tool",
    "tool not found",
    "unknown tool",
    "unknown skill",
    "skill not found",
    "no such skill",
    "unknown command",
    "no such command",
    "not found. available",
    "unsupported call",
    "unknown agent type",
)
# Check 15 routes work to components; check 22 selects tools (MCP and plain).
_ROUTING_KINDS = frozenset({COMPONENT_SKILL, COMPONENT_SUBAGENT, COMPONENT_COMMAND})
# Harnesses that cannot stage plugin subagents or commands, so refs to those can never match there.
_NO_PLUGIN_AGENT_HARNESSES = frozenset({"codex"})
# Each harness's own agents (casefolded). A bare one of these names reaches the built-in, never
# the plugin's agent: Claude Code names a plugin agent ``<plugin>:<name>``, and OpenCode stages a
# plugin agent named like its own as ``<plugin>-<name>``.
_BUILTIN_AGENTS = {
    "claude-code": frozenset(
        {"claude", "explore", "general-purpose", "output-style-setup", "plan", "statusline-setup"}
    ),
    "opencode": OPENCODE_BUILTIN_AGENTS,
}

_REF_PREFIXES = {
    "skill": COMPONENT_SKILL,
    "agent": COMPONENT_SUBAGENT,
    "subagent": COMPONENT_SUBAGENT,
    "task": COMPONENT_SUBAGENT,
    "command": COMPONENT_COMMAND,
    "slashcommand": COMPONENT_COMMAND,
    "mcp": COMPONENT_MCP,
    "lsp": COMPONENT_LSP,
}
# Rules are inlined into the wrapper skill and hooks run outside the agent's tool calls.
_UNSUPPORTED_REF_PREFIXES = frozenset({"rule", "rules", "hook", "hooks"})
_REF_PREFIX_SPELLING = {
    "skill": "Skill",
    "agent": "Agent",
    "subagent": "Subagent",
    "task": "Task",
    "command": "Command",
    "slashcommand": "SlashCommand",
    "mcp": "MCP",
    "lsp": "LSP",
}
_IDENTITY_PREFIX = {
    COMPONENT_SKILL: "Skill",
    COMPONENT_SUBAGENT: "Agent",
    COMPONENT_COMMAND: "Command",
    COMPONENT_LSP: "LSP",
}
# Skill/subagent/command names come from agent-written tool arguments. Only an
# identifier-shaped name (or a declared member) is persisted; anything else, such
# as a URL with credentials or key material, is reported as ``<non-name>``.
_PERSISTABLE_NAME_RE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}")
_NON_NAME = "<non-name>"

# Failure text is read only when a result has no structured outcome, and only
# on its lead lines (the first line of each block, past harness status lines
# and MCP content wrappers, and the last line): a good answer that talks about
# errors ("Permission denied errors happen when ...", "AT-412: Fix race",
# "400-500 rps", "Error: ENOENT means ...") is not a failed call. Only harness,
# transport and HTTP status text counts; a server's own "Error: ..." does not.
# Harness and transport phrases: a failure anywhere on a lead line.
_UNAVAILABLE_MARKERS = (
    "no such tool",
    "tool not found",
    "unknown tool",
    "unknown skill",
    "skill not found",
    "no such command",
    "command not found",
    "unknown command",
    "failed to connect",
    "connection refused",
    "mcp server not",
    "no mcp server",
    "no lsp server",
)
_FAILURE_PHRASES = (
    *_UNAVAILABLE_MARKERS,
    "<tool_use_error>",
    "mcp error",
    "internal server error",
    "failed to decrypt",
    # Harbor appends this line to a Claude Code result the harness flagged.
    "[error] tool reported failure",
)
# Words that say a call failed only when they open a lead line (after an
# optional ``name:`` chain such as ``cat: <path>:``) and end a clause there:
# "Permission denied: notes_path" fails, "Permission denied errors happen" does not.
_OPENING_FAILURE_RE = re.compile(
    r"(?:permission denied|access denied|unauthorized|forbidden|invalid[_ ]token|token (?:is )?expired)"
    r"(?=\s*(?:$|[:.,;!)\-]))",
    re.IGNORECASE,
)
# A missing file means a failed call only for MCP and file-read tools. A shell
# call's output mixes several commands, so ``cat SKILL.md missing.md`` read the
# manifest even though its output says "No such file or directory".
_FILE_MISSING_RE = re.compile(
    r"(?:no such file or directory|file does not exist)(?=\s*(?:$|[:.,;!)\-]))", re.IGNORECASE
)
# A line about a tool, server, skill, or command that ends with "is not available":
# "Tool 'x' is not available." fails; "Streaming is not available." and
# "Streaming is not available on the free plan" do not.
_UNAVAILABLE_TAIL_RE = re.compile(
    r"\b(?:tool|server|skill|command|mcp)\b.*\b(?:not available|is unavailable)\s*[.!]?\s*$", re.IGNORECASE
)
# A harness's own error prefix opening a lead line: "tool call error: ...". A
# bare "Error: ...", "TypeError: ..." or "ERROR: ..." is not one: a server's
# own error text with no flag is an answer the agent saw ("Error: ENOENT means
# the path does not exist", a log search hit), and Claude Code and Codex both
# report a failed call with a structured flag that wins over any text.
_HARNESS_ERROR_PREFIX_RE = re.compile(r"tool\s+call\s+error\s*:", re.IGNORECASE)
# An HTTP status 400-599 counts only as a status line: it opens the line
# (optionally after "HTTP", "HTTP/1.1", "HTTP Error", "Error", "status" or
# "status code"), a standard reason phrase follows, and the clause ends there.
# "404 Not Found", "HTTP Error 404: Not Found", "401 - Unauthorized: missing
# token" and "403: Forbidden - token lacks scope" fail; "403: Forbidden is
# returned when ...", "404: page moved" and "429 - Too Many Requests means ..."
# do not.
_HTTP_REASONS = (
    "bad request|unauthorized|payment required|forbidden|not found|method not allowed|not acceptable"
    "|request timeout|conflict|gone|precondition failed|payload too large|unprocessable entity|too many requests"
    "|internal server error|not implemented|bad gateway|service unavailable|gateway timeout"
)
_HTTP_STATUS_OPENING_RE = re.compile(
    r"(?:(?:https?(?:/\d(?:\.\d)?)?\s+)?(?:error\s+)?|status(?:\s+code)?\s*[:=]?\s*)"
    rf"[45]\d{{2}}(?:\s*[:\-]\s*|\s+)(?:{_HTTP_REASONS})(?=\s*(?:$|[:.,;!)\-]))",
    re.IGNORECASE,
)
# An HTTP response status line ("HTTP/1.1 404 - x", "HTTP/2 500") is a status
# line whatever its reason text says, so it fails without a standard reason
# phrase when the code ends the line or a ":" or "-" follows it. Prose such
# as "HTTP/1.1 404 responses are cached" does not.
_HTTP_VERSION_STATUS_RE = re.compile(r"https?/\d(?:\.\d)?\s+[45]\d{2}(?=\s*(?:$|[:\-]))", re.IGNORECASE)
# A client's failed-request line opening a lead line: "Request failed with
# status code 404" (also after a ``name:`` prefix such as ``AxiosError:``).
# "Fixed: retries after status code 503" is not one.
_REQUEST_FAILED_RE = re.compile(
    r"(?:request\s+)?failed\s+with\s+status(?:\s+code)?\s*[:=]?\s*[45]\d{2}\b(?=\s*(?:$|[:.,;!)\-]))",
    re.IGNORECASE,
)
# A leading ``name:`` segment (a program, a path, a server, an issue key) that
# can precede the failure words: ``cat: <dir>/SKILL.md: Permission denied``.
_LEAD_SEGMENT_RE = re.compile(r"[^\s:]{1,256}:\s+")
# What can open a lead line before an HTTP status line: a request line ("GET
# /repos/x: 404 - Not Found") or a clause that starts with an error word
# ("Error calling tool: 404: not found"). Each gap is bounded, so a long line is
# scanned in linear time. The status line itself still needs a reason phrase.
_STATUS_LINE_PREFIX_RES = (
    re.compile(r"(?:get|head|post|put|patch|delete|options)[ \t]+\S{1,200}?:?[ \t]+", re.IGNORECASE),
    re.compile(r"(?:error|failed)\b[^:\n]{0,80}:[ \t]+", re.IGNORECASE),
)
# Status lines a harness puts in front of shell output (Codex ``exec_command``).
# They are not file text, so a shell read looks past them for its first line.
_SHELL_STATUS_LINE_RE = re.compile(
    r"chunk id:.*|wall time:.*|process exited with code -?\d+|exit code:\s*-?\d+"
    r"|original token count:.*|total output lines:.*|output:",
    re.IGNORECASE,
)
# A shell tool's error line about a file (``cat: <dir>/SKILL.md: Permission denied``).
_SHELL_ERROR_LINE_RE = re.compile(r"[\w./+-]+:\s")
# The MCP content wrapper a harness keeps around a tool's text: Codex prints
# ``[{"type":"text","text":"..."}]`` and Harbor's Claude converter may keep one
# ``{"type": "text", "text": "..."}`` block as a string.
_MCP_TEXT_WRAPPER_RE = re.compile(r'\[?\s*\{\s*"type"\s*:\s*"text"\s*,\s*"text"\s*:\s*"')
# Wrappers up to this size are parsed whole; a larger one yields its first text only.
_MAX_WRAPPER_JSON_CHARS = 16 * 1024
# Structured per-call flags. Harbor keeps Claude Code's own flag as
# ``extra.tool_result_is_error`` and ``extra.tool_result_metadata.is_error``
# (the raw block under ``raw_tool_result``); other producers use a top-level
# ``is_error``/``isError``. A string or number spelling counts the same way.
_ERROR_FLAG_KEYS = ("is_error", "isError")
_TRUE_FLAGS = frozenset({"true", "1", "yes"})
_FALSE_FLAGS = frozenset({"false", "0", "no"})
#: Result ``extra`` key the collector fills from a harness's own per-call status
#: (Codex ``codex.txt``: ``completed`` or ``failed``).
HARNESS_STATUS_KEY = "harness_status"
_HARNESS_FAILED = frozenset({"failed", "error", "errored"})
_HARNESS_SUCCEEDED = frozenset({"completed", "succeeded", "success"})
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


def _check_list(value: Any, where: str, errors: _FieldErrors, *, items: str, limit: int, unit: str) -> list[Any] | None:
    """``value`` if it is a list of at most ``limit`` entries; otherwise the problem is recorded."""
    if not isinstance(value, list):
        errors.add(where, f"must be a list of {items}")
        return None
    if len(value) > limit:
        errors.add(where, f"must list at most {limit} {unit}")
        return None
    return value


def _check_object(raw: Any, where: str, errors: _FieldErrors, allowed_keys: frozenset[str]) -> bool:
    """Whether ``raw`` is an object with no keys outside ``allowed_keys``; otherwise the problem is recorded."""
    if not isinstance(raw, dict):
        errors.add(where, "must be an object")
        return False
    unknown = sorted(str(key)[:_MAX_KEY_PREVIEW_CHARS] for key in raw if key not in allowed_keys)
    if unknown:
        errors.add(where, f"unsupported keys: {', '.join(unknown)}")
        return False
    return True


def _is_name(value: Any) -> bool:
    """An argument or property name: a non-empty string of at most ``MAX_ARGUMENT_NAME_CHARS``."""
    return isinstance(value, str) and 0 < len(value) <= MAX_ARGUMENT_NAME_CHARS


def _is_name_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) <= MAX_ARGUMENT_NAMES and all(_is_name(item) for item in value)


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
    if sep and prefix.strip().casefold() in _UNSUPPORTED_REF_PREFIXES:
        errors.add(
            where,
            f"'{_safe_text(prefix, 32)}:' refs are not supported: rules and hooks are not tool calls, so the ref "
            "could never match (the hook census reports hook runs)",
        )
        return None
    near = _near_ref_prefix(prefix.strip().casefold()) if sep else None
    if near is not None:
        errors.add(
            where,
            f"unknown ref prefix '{_safe_text(prefix, 32)}:' (did you mean '{_REF_PREFIX_SPELLING[near]}:'?); "
            "write a namespaced name as an explicit type, such as 'Skill:<plugin>:<name>'",
        )
        return None
    return text


def _near_ref_prefix(word: str) -> str | None:
    """A known ref type one typo away from ``word`` (``Skil`` -> ``skill``), so the ref could never match."""
    if len(word) < 3 or word in _REF_PREFIXES:
        return None
    return next((prefix for prefix in _REF_PREFIX_SPELLING if _within_one_edit(word, prefix)), None)


def _within_one_edit(left: str, right: str) -> bool:
    """One insertion, deletion, substitution, or swap of neighbors apart."""
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        diff = [index for index, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
        if len(diff) == 1:
            return True
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and left[diff[0]] == right[diff[1]]
            and left[diff[1]] == right[diff[0]]
        )
    short, long = (left, right) if len(left) < len(right) else (right, left)
    return any(long[:index] + long[index + 1 :] == short for index in range(len(long)))


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
    items = _check_list(value, where, errors, items="tool refs", limit=MAX_TOOL_PATTERNS, unit="tool refs")
    if items is None:
        return None
    patterns: list[str] = []
    for index, item in enumerate(items):
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
    except Exception as exc:
        # Some malformed patterns raise other errors: KeyError for ``(?V0)(?V1)``,
        # RuntimeError for a fuzzy count the engine cannot compile.
        errors.add(where, f"is not a valid regular expression (the regex engine raised {type(exc).__name__})")
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
                f"unsupported JSON-Schema keyword {str(key)[:_MAX_KEY_PREVIEW_CHARS]!r}; "
                f"supported: {', '.join(sorted(_SCHEMA_KEYWORDS))}",
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
    if "required" in schema and not _is_name_list(schema["required"]):
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
                if not _is_name(name):
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
        if not _is_name(name):
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


def _parse_argument_rule(raw: Any, rule_where: str, errors: _FieldErrors) -> dict[str, Any] | None:
    if not _check_object(raw, rule_where, errors, _ARGUMENT_RULE_KEYS):
        return None
    tool = _check_ref_string(raw.get("tool"), f"{rule_where}.tool", errors)
    if tool is None or not _check_description(raw, rule_where, errors):
        return None
    rule: dict[str, Any] = {"tool": tool}
    if "required" in raw:
        required = raw["required"]
        if not _is_name_list(required):
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
    return rule


def _parse_tool_arguments(value: Any, where: str, errors: _FieldErrors) -> list[dict[str, Any]] | None:
    """Every valid rule, or ``None`` when any rule is bad; each bad rule is reported, not only the first."""
    items = _check_list(value, where, errors, items="argument rules", limit=MAX_ARGUMENT_RULES, unit="rules")
    if items is None:
        return None
    rules: list[dict[str, Any]] = []
    valid = True
    for index, raw in enumerate(items):
        rule = _parse_argument_rule(raw, f"{where}[{index}]", errors)
        if rule is None:
            valid = False
        else:
            rules.append(rule)
    return rules if valid else None


def sanitize_input_schema(schema: Any, *, depth: int = 1, budget: list[int] | None = None) -> dict[str, Any] | None:
    """An MCP server's tool ``inputSchema`` cut to the subset argument checks run, or ``None``.

    Keeps ``type``, ``properties``, ``required``, ``enum``, ``pattern``,
    ``minimum`` and ``maximum`` (the ``tool_arguments`` schema subset) and drops
    everything else (``$ref``, ``anyOf``, ``additionalProperties``, annotations),
    so the check never fails on a keyword it cannot evaluate. Values that would
    fail the dataset schema rules (an invalid regex, a non-finite bound, an
    oversize enum) are dropped too. Bounded by the dataset limits: nesting
    depth, node count, and 64 properties per object.
    """
    budget = budget if budget is not None else [0]
    budget[0] += 1
    if not isinstance(schema, Mapping) or depth > MAX_SCHEMA_DEPTH or budget[0] > MAX_SCHEMA_NODES:
        return None
    errors = _FieldErrors()
    out: dict[str, Any] = {}
    types = schema.get("type")
    type_list = [types] if isinstance(types, str) else types
    if isinstance(type_list, list) and type_list and all(item in _SCHEMA_TYPES for item in type_list):
        out["type"] = types
    required = schema.get("required")
    if isinstance(required, list):
        names = [item for item in required if _is_name(item)]
        if names:
            out["required"] = list(dict.fromkeys(names))[:MAX_ARGUMENT_NAMES]
    enum = schema.get("enum")
    if isinstance(enum, list) and 0 < len(enum) <= MAX_ENUM_ITEMS and _bounded_json_text(enum, MAX_LITERAL_JSON_CHARS):
        out["enum"] = list(enum)
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and _check_regex(pattern, "pattern", errors) is not None:
        out["pattern"] = pattern
    for bound in ("minimum", "maximum"):
        if _is_number(schema.get(bound)):
            out[bound] = schema[bound]
    if _is_number(out.get("minimum")) and _is_number(out.get("maximum")) and out["minimum"] > out["maximum"]:
        out.pop("minimum")
        out.pop("maximum")
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        kept: dict[str, Any] = {}
        for name, sub in list(properties.items())[:MAX_ARGUMENT_NAMES]:
            if _is_name(name):
                cleaned = sanitize_input_schema(sub, depth=depth + 1, budget=budget)
                if cleaned:
                    kept[name] = cleaned
        if kept:
            out["properties"] = kept
    return out or None


def _parse_expected_order(value: Any, where: str, errors: _FieldErrors) -> list[list[list[str]]] | None:
    items = _check_list(value, where, errors, items="[before, after] edges", limit=MAX_ORDER_EDGES, unit="edges")
    if items is None:
        return None
    edges: list[list[list[str]]] = []
    for index, edge in enumerate(items):
        edge_where = f"{where}[{index}]"
        if not isinstance(edge, list) or len(edge) != 2:
            errors.add(edge_where, "must be a two-item [before, after] list")
            return None
        before = _check_ref_alternatives(edge[0], f"{edge_where}[0]", errors)
        after = _check_ref_alternatives(edge[1], f"{edge_where}[1]", errors)
        if before is None or after is None:
            return None
        before_refs, after_refs = _refs(before), _refs(after)
        if all(_ref_covers(later, earlier) for later in after_refs for earlier in before_refs):
            # Whatever uses ``before`` also uses ``after``, so ``after`` is never strictly later.
            errors.add(edge_where, "can never pass: every 'after' ref also matches every 'before' ref")
            return None
        edges.append([before, after])
    return edges


def _parse_handoffs(value: Any, where: str, errors: _FieldErrors) -> list[dict[str, Any]] | None:
    items = _check_list(value, where, errors, items="handoff objects", limit=MAX_HANDOFFS, unit="handoffs")
    if items is None:
        return None
    handoffs: list[dict[str, Any]] = []
    for index, raw in enumerate(items):
        item_where = f"{where}[{index}]"
        if not _check_object(raw, item_where, errors, _HANDOFF_KEYS):
            return None
        producer = _check_ref_alternatives(raw.get("producer"), f"{item_where}.producer", errors)
        consumer = _check_ref_alternatives(raw.get("consumer"), f"{item_where}.consumer", errors)
        if producer is None or consumer is None:
            return None
        shared = {ref.casefold() for ref in producer} & {ref.casefold() for ref in consumer}
        if shared:
            errors.add(
                item_where, "producer and consumer must name different components (a self-handoff always passes)"
            )
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
            not isinstance(artifact, str) or not artifact.strip() or len(artifact) > MAX_ARTIFACT_CHARS
        ):
            errors.add(f"{item_where}.artifact", f"must be a workspace path of at most {MAX_ARTIFACT_CHARS} characters")
            return None
        if artifact is not None and _CONTROL_CHARS_RE.search(artifact):
            errors.add(f"{item_where}.artifact", "must not contain control characters")
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


def _parse_conflict_probes(value: Any, where: str, errors: _FieldErrors) -> list[dict[str, Any]] | None:
    items = _check_list(value, where, errors, items="probe objects", limit=MAX_CONFLICT_PROBES, unit="probes")
    if items is None:
        return None
    probes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(items):
        item_where = f"{where}[{index}]"
        if not _check_object(raw, item_where, errors, _PROBE_KEYS):
            return None
        probe_id = raw.get("id")
        if not isinstance(probe_id, str) or not probe_id.strip() or len(probe_id) > MAX_PROBE_ID_CHARS:
            errors.add(f"{item_where}.id", f"must be a non-empty string of at most {MAX_PROBE_ID_CHARS} characters")
            return None
        # Ids are stored trimmed, so ``dup`` and `` dup`` are the same id.
        if probe_id.strip() in seen_ids:
            errors.add(f"{item_where}.id", "must be unique (ids are compared without surrounding spaces)")
            return None
        seen_ids.add(probe_id.strip())
        must_use = _check_ref_alternatives(raw.get("must_use"), f"{item_where}.must_use", errors)
        must_not_use = _check_ref_alternatives(raw.get("must_not_use"), f"{item_where}.must_not_use", errors)
        if must_use is None or must_not_use is None:
            return None
        forbidden = _refs(must_not_use)
        if all(any(_ref_covers(ref, wanted) for ref in forbidden) for wanted in _refs(must_use)):
            errors.add(item_where, "can never pass: every must_use ref is also matched by must_not_use")
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


# Each advisory case field and its parser: ``parser(value, field_name, errors)``
# returns the normalized value, or ``None`` after recording the problem.
_FIELD_PARSERS: dict[str, Callable[[Any, str, _FieldErrors], list[Any] | None]] = {
    "expected_tools": _check_tool_patterns,
    "acceptable_tools": _check_tool_patterns,
    "decoy_tools": _check_tool_patterns,
    "tool_arguments": _parse_tool_arguments,
    "expected_order": _parse_expected_order,
    "handoffs": _parse_handoffs,
    "conflict_probes": _parse_conflict_probes,
}
PLUGIN_CASE_FIELDS = tuple(_FIELD_PARSERS)


def validate_plugin_case_fields(entry: Any) -> list[str]:
    """Return problems with the optional advisory plugin-signal fields of one case.

    Absent fields are fine. ``null`` is treated as absent. Every present field is
    type-checked and size-bounded; see the ``MAX_*`` constants.
    """
    if not isinstance(entry, Mapping):
        return []
    errors = _FieldErrors()
    parsed: dict[str, Any] = {}
    for name, parse in _FIELD_PARSERS.items():
        value = entry.get(name)
        if value is None:
            continue
        before = len(errors.messages)
        result = parse(value, name, errors)
        if result is not None and len(errors.messages) == before:
            parsed[name] = result
    _check_selection_overlap(parsed, errors)
    return errors.messages


def _ref_covers(general: _Ref, specific: _Ref) -> bool:
    """Whether every call ``specific`` matches is also matched by ``general`` (same type, glob over the text)."""
    if general.kind != specific.kind:
        return False
    if general.kind == COMPONENT_MCP and "/" not in general.pattern and "/" in specific.pattern:
        # ``MCP:<server>`` matches every tool of that server.
        return fnmatch.fnmatchcase(specific.pattern.split("/", 1)[0], general.pattern)
    return fnmatch.fnmatchcase(specific.pattern, general.pattern)


def _check_selection_overlap(parsed: Mapping[str, Any], errors: _FieldErrors) -> None:
    """A decoy ref must not also name an expected or acceptable call: it would be scored both ways."""
    decoys = parsed.get("decoy_tools") or []
    for field_name in ("expected_tools", "acceptable_tools"):
        for index, raw in enumerate(decoys):
            decoy = _parse_ref(raw)
            for other in parsed.get(field_name) or []:
                wanted = _parse_ref(other)
                if _ref_covers(wanted, decoy) or _ref_covers(decoy, wanted):
                    errors.add(
                        f"decoy_tools[{index}]",
                        f"overlaps {field_name} ref {_safe_text(other, 64)!r}: a call cannot be both wanted and a decoy",
                    )
                    break


def plugin_case_spec(entry: Any) -> dict[str, Any]:
    """Return the valid, normalized plugin-signal fields of one case entry.

    Invalid fields are dropped (the dataset validators report them), so a
    malformed advisory field can never break result collection.
    """
    if not isinstance(entry, Mapping):
        return {}
    spec: dict[str, Any] = {}
    for name, parse in _FIELD_PARSERS.items():
        value = entry.get(name)
        if value is None:
            continue
        errors = _FieldErrors()
        parsed = parse(value, name, errors)
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
    maps case id to :func:`plugin_case_spec` output. The context is shared by
    every agent of a run, so what only one agent's with-plugin arm stages, or
    does not stage, is kept per agent (``agent_mcp_servers``,
    ``agent_unstaged``).
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
    # Probed URL servers' tool input schemas ({server: {tool: schema}}, --probe-mcp), checked as arguments.
    mcp_input_schemas: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # The plugin's own name: the namespace Claude Code gives its skills, agents,
    # and commands (``<plugin>:<name>``).
    plugin_names: tuple[str, ...] = ()
    # Agent -> the MCP servers its with-plugin arm stages beyond ``mcp_servers``: Claude Code's
    # native plugin copy also starts the servers that launch from plugin files.
    agent_mcp_servers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Agent -> the declared component types its with-plugin arm does not stage: plugin subagents
    # and commands unless that agent loads them natively.
    agent_unstaged: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Agent -> the plugin LSP servers its with-plugin arm stages natively (Claude Code's ``.lsp.json``),
    # each with the file extensions its ``extensionToLanguage`` maps.
    agent_lsp_servers: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)

    def arm_enabled(self, arm: str) -> bool:
        if arm in {ARM_WITH, ARM_SUM_OF_PARTS}:
            return True
        return arm == ARM_WITHOUT and self.baseline_has_members

    def declared_for(self, arm: str, agent: str = "") -> dict[str, list[str]]:
        """Declared components staged in ``agent``'s ``arm`` (MCP is wired into the with-plugin arm only)."""
        declared: dict[str, list[str]] = {COMPONENT_SKILL: list(self.member_skills)}
        with_plugin = arm == ARM_WITH
        # Every agent's other arms stage the same members only, so a server any with-plugin arm runs is unstaged there.
        agent_servers = [self.agent_mcp_servers.get(agent, ())] if with_plugin else self.agent_mcp_servers.values()
        mcp_servers = list(dict.fromkeys((*self.mcp_servers, *(name for names in agent_servers for name in names))))
        declared[COMPONENT_MCP] = mcp_servers if with_plugin else []
        if with_plugin and self.subagents:
            declared[COMPONENT_SUBAGENT] = list(self.subagents)
        if with_plugin and self.commands:
            declared[COMPONENT_COMMAND] = list(self.commands)
        lsp_servers = list(self.lsp_servers_for(arm, agent))
        if lsp_servers:
            declared[COMPONENT_LSP] = lsp_servers
        # An LSP server only the native Claude Code arm stages is unstaged in every other arm.
        lsp_elsewhere = not lsp_servers and any(self.agent_lsp_servers.values())
        if self.plugin_names:
            declared[DECLARED_PLUGIN] = list(self.plugin_names)
        # Refs to these can never match in this arm, so graders skip them instead of failing them.
        if not with_plugin:
            unstaged = [
                kind
                for kind, names in (
                    (COMPONENT_MCP, mcp_servers),
                    (COMPONENT_SUBAGENT, self.subagents),
                    (COMPONENT_COMMAND, self.commands),
                    (COMPONENT_LSP, lsp_elsewhere),
                )
                if names
            ]
            hidden = [
                f"{kind}:{name}"
                for kind, names in ((COMPONENT_SUBAGENT, self.subagents), (COMPONENT_COMMAND, self.commands))
                for name in names
            ]
            if hidden:
                declared[DECLARED_UNSTAGED_NAMES] = hidden
        else:
            # Still declared, so a call the agent makes to one anyway is recorded.
            unstaged = [kind for kind in self.agent_unstaged.get(agent, ()) if declared.get(kind)]
            if lsp_elsewhere:
                unstaged.append(COMPONENT_LSP)
        if unstaged:
            declared[DECLARED_UNSTAGED] = unstaged
        return declared

    def lsp_servers_for(self, arm: str, agent: str = "") -> Mapping[str, tuple[str, ...]]:
        """The LSP servers ``agent``'s ``arm`` stages, with their file extensions (a native with-plugin arm only)."""
        return self.agent_lsp_servers.get(agent, {}) if arm == ARM_WITH else {}

    def aliases_for(self, arm: str) -> Mapping[str, str]:
        """Subagent name aliases for ``arm`` (declared subagents count in the with-plugin arm only)."""
        return self.subagent_aliases if arm == ARM_WITH and self.subagents else {}

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
    plugin_name: str | None = None,
    agent_mcp_servers: Mapping[str, Iterable[Any]] | None = None,
    agent_unstaged: Mapping[str, Iterable[Any]] | None = None,
    agent_lsp_servers: Mapping[str, Mapping[str, Any]] | None = None,
) -> PluginSignalsContext:
    """Build a bounded :class:`PluginSignalsContext` from dataset case entries.

    ``plugin_name`` defaults to the name in the generated wrapper package
    (``<plugin>-plugin-eval``) when ``wrapper_skills`` holds one.
    ``agent_mcp_servers`` maps an agent to the MCP servers only its
    with-plugin arm stages, ``agent_unstaged`` to the declared component
    types its with-plugin arm does not stage, and ``agent_lsp_servers`` to the
    LSP servers its with-plugin arm stages (``{server: extensions}``).
    """
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
    wrappers = _clean_names(wrapper_skills)
    return PluginSignalsContext(
        member_skills=_clean_names(member_skills),
        mcp_servers=_clean_names(mcp_servers),
        wrapper_skills=wrappers,
        cases=cases,
        baseline_has_members=bool(baseline_has_members),
        subagents=declared_subagents,
        commands=_clean_names(name.lstrip("/") for name in commands if isinstance(name, str)),
        subagent_aliases=_clean_aliases(subagent_aliases, declared_subagents),
        plugin_names=_clean_names([plugin_name] if plugin_name else _plugin_names_from_wrappers(wrappers)),
        agent_mcp_servers={
            agent: names
            for agent, servers in (agent_mcp_servers or {}).items()
            if isinstance(agent, str) and (names := _clean_names(servers))
        },
        agent_unstaged={
            agent: kinds
            for agent, values in (agent_unstaged or {}).items()
            if isinstance(agent, str) and (kinds := _clean_names(values))
        },
        agent_lsp_servers={
            agent: servers
            for agent, value in (agent_lsp_servers or {}).items()
            if isinstance(agent, str) and (servers := _clean_lsp_servers(value))
        },
    )


def _plugin_names_from_wrappers(wrapper_skills: Iterable[str]) -> list[str]:
    """The plugin name behind a generated ``<plugin>-plugin-eval`` wrapper package name."""
    return [
        name[: -len(_WRAPPER_PACKAGE_SUFFIX)]
        for name in wrapper_skills
        if isinstance(name, str) and name.casefold().endswith(_WRAPPER_PACKAGE_SUFFIX) and len(name) > 12
    ]


def _clean_lsp_servers(servers: Any) -> dict[str, tuple[str, ...]]:
    """Bounded ``server -> file extensions`` (casefolded, each with its leading dot) of staged LSP servers."""
    cleaned: dict[str, tuple[str, ...]] = {}
    if not isinstance(servers, Mapping):
        return cleaned
    for raw_name, raw_extensions in servers.items():
        if len(cleaned) >= MAX_DECLARED_COMPONENTS:
            break
        names = _clean_names([raw_name])
        if not names or isinstance(raw_extensions, str | bytes) or not isinstance(raw_extensions, Iterable):
            continue
        extensions = [
            extension.strip().casefold()
            for extension in raw_extensions
            if isinstance(extension, str) and _LSP_EXTENSION_RE.fullmatch(extension.strip())
        ]
        cleaned[names[0]] = tuple(dict.fromkeys(extensions))[:_MAX_LSP_EXTENSIONS]
    return cleaned


def _lsp_extension_index(lsp_servers: Any, declared: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """File extension -> the one LSP server declared in this arm that serves it.

    An extension two declared servers claim is left out: the call cannot say which one answered.
    """
    staged = {name for name in declared.get(COMPONENT_LSP) or () if isinstance(name, str)}
    owners: dict[str, set[str]] = {}
    for server, extensions in _clean_lsp_servers(lsp_servers).items():
        if server in staged:
            for extension in extensions:
                owners.setdefault(extension, set()).add(server)
    return {extension: next(iter(servers)) for extension, servers in owners.items() if len(servers) == 1}


def _lsp_ident(fn: str, args: Mapping[str, Any], lsp_extensions: Mapping[str, str]) -> _Ident | None:
    """The plugin LSP server a Claude Code ``LSP`` call reached, from its file's extension, or ``None``."""
    path = _normalize_path(_first_string(args, _LSP_PATH_KEYS))
    suffix = posixpath.splitext(posixpath.basename(path))[1].casefold()
    server = lsp_extensions.get(suffix) if suffix else None
    if server is None:
        return None
    return _Ident(label=f"LSP:{server}", kind=COMPONENT_LSP, name=server, fn=fn, tool_label=fn)


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
    """One identity of a tool call: a component activation or a plain tool.

    ``foreign`` marks a namespaced component name from another plugin
    (``acme-tools:release-notes``): it never matches this plugin's refs.
    ``via_read`` marks a ``SKILL.md`` read (not an explicit activation tool);
    ``position`` orders the reads of one chained shell command; ``failed``
    marks a read whose own error line says it failed. ``wrapper`` marks the
    generated wrapper skill, which is the harness's way in, not a component.
    ``builtin`` marks a harness's own agent called by the bare name of a
    declared plugin agent (Claude Code ``Explore`` for a plugin's ``explore``):
    it is neither that component nor what a ref to it names.
    """

    label: str
    kind: str | None
    name: str
    fn: str
    server: str | None = None
    tool: str | None = None
    tool_label: str = ""
    persist_name: bool = True
    foreign: bool = False
    namespaces: tuple[str, ...] = ()
    via_read: bool = False
    position: int = 0
    failed: bool = False
    wrapper: bool = False
    aliases: tuple[str, ...] = ()
    builtin: bool = False

    @property
    def persisted_name(self) -> str:
        return self.name if self.persist_name else _NON_NAME

    @property
    def persisted_label(self) -> str:
        return self.label if self.persist_name else f"{_IDENTITY_PREFIX[self.kind or '']}:{_NON_NAME}"

    @property
    def name_candidates(self) -> list[str]:
        """Casefolded names this component answers to (another plugin's namespaced name only to itself)."""
        low = self.name.casefold()
        if self.foreign:
            return [low]
        candidates = [low, low.rsplit(":", 1)[-1]] if ":" in low else [low]
        short = candidates[-1]
        candidates.extend(f"{namespace}:{short}" for namespace in self.namespaces)
        return list(dict.fromkeys(candidates))

    @cached_property
    def typed_names(self) -> tuple[str, ...]:
        """What a ``<Type>:<pattern>`` ref of this identity's type is matched against."""
        if self.kind == COMPONENT_MCP:
            return _match_names(_mcp_ref_candidates(self))
        return _match_names(self.name_candidates)

    @cached_property
    def untyped_names(self) -> tuple[str, ...]:
        """What any other ref is matched against (see :func:`_ref_matches`)."""
        if self.kind is None:
            names = [self.label, self.fn, *self.aliases]
        elif self.kind == COMPONENT_MCP:
            # The bare tool name is what Codex and the MCP server call the tool (``list_changes``).
            names = [self.label, self.fn, *_mcp_ref_candidates(self), self.tool or ""]
        else:
            names = [self.label, *self.name_candidates]
            if self.kind == COMPONENT_COMMAND and self.name:
                names.append(f"/{self.name}")
        return _match_names(names)


class _ShellPaths(NamedTuple):
    reads: list[str]
    writes: list[str]


class _Result(NamedTuple):
    """One tool result of a step.

    ``scan`` is the part of ``text`` that failure text is checked against, and
    ``flagged`` the structured outcome of :func:`_result_flagged`.
    """

    call_id: str
    text: str
    scan: str
    flagged: bool | None


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
    # The result says the called tool, skill, command, or agent does not exist.
    missing_tool: bool = False
    # A subagent's own call (Claude Code sidechain step), and the parent's subagent call that started it.
    sidechain: bool = False
    # Which subagent run made the call ("" for the main agent): Harbor's ``agent_id``, else one id per run.
    chain: str = ""
    spawner: list[_Ident] = field(default_factory=list)
    # The one declared skill or command this call switched to, when it really did (see ``_window_opener``).
    opener: _Ident | None = None
    # The working directory of the call, when the trajectory says it.
    cwd: str | None = None
    # Components a shell call used without its tool: a JSON-RPC ``tools/call`` piped to an MCP server,
    # or a member skill's script run directly (see ``_side_channel_idents``).
    side_idents: list[_Ident] = field(default_factory=list)

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

    @cached_property
    def result_texts(self) -> list[str]:
        """The result text plus the string values of JSON inside it, decoded once per call."""
        return _decoded_texts(self.observation) if self.observation else []

    @cached_property
    def argument_texts(self) -> list[str]:
        """The argument string values (and JSON inside them), decoded once per call."""
        return [text for leaf in _string_values(self.args) for text in _decoded_texts(leaf)]

    @property
    def component_idents(self) -> list[_Ident]:
        return [ident for ident in self.idents if ident.kind is not None]

    def ident_ok(self, ident: _Ident) -> bool:
        """Whether this identity's activation did not fail (``None`` outcomes count as not failed)."""
        return self.succeeded is not False and not ident.failed

    @cached_property
    def shell_paths(self) -> _ShellPaths:
        """Distinct shell paths this call reads and writes, resolved against the call's directory.

        A path may be a glob (``out/*.json``); see :func:`_observed_path_matches`.
        """
        reads: dict[str, None] = {}
        writes: dict[str, None] = {}
        if self.is_shell:
            for text in _shell_texts(self.fn_base, self.args):
                text_reads, text_writes = _shell_file_io(text, self.cwd)
                reads.update(dict.fromkeys(text_reads))
                writes.update(dict.fromkeys(text_writes))
        return _ShellPaths(list(reads), list(writes))


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
        for block in content[:_MAX_CONTENT_BLOCKS]:
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


def _meaningful_lines(text: str, *, last: bool = False) -> Iterable[str]:
    """Non-blank lines of a bounded window, past a harness's status lines (``Wall time:``, ``Output:``)."""
    window = text[-_FAILURE_SCAN_CHARS:] if last else text[:_FAILURE_SCAN_CHARS]
    lines = window.splitlines()
    for line in reversed(lines) if last else lines:
        stripped = line.strip()
        if stripped and not _SHELL_STATUS_LINE_RE.fullmatch(stripped):
            yield stripped


def _unwrapped_texts(text: str) -> list[str]:
    """The tool texts inside an MCP content wrapper that opens ``text``, else ``[]``.

    Codex prints an MCP result as ``[{"type":"text","text":"..."}]`` after its
    status lines; Harbor's Claude converter may keep one such block as a
    string. A cut-off wrapper still yields the start of its first text.
    """
    head = _MCP_TEXT_WRAPPER_RE.match(text)
    if head is None:
        return []
    decoded: Any = None
    if len(text) <= _MAX_WRAPPER_JSON_CHARS:
        try:
            decoded = json.loads(text, strict=False)
        except (ValueError, RecursionError):
            decoded = None
    blocks = decoded if isinstance(decoded, list) else [decoded]
    texts = [block["text"] for block in blocks[:8] if isinstance(block, dict) and isinstance(block.get("text"), str)]
    if texts:
        return texts
    literal = re.match(r'(?:[^"\\]|\\.){0,2048}', text[head.end() :])
    try:
        return [json.loads(f'"{literal.group()}"', strict=False)] if literal else []
    except (ValueError, RecursionError):
        return []


def _lead_lines(part: str) -> list[str]:
    """The lines of one result block that can say the call failed: its first and last lines.

    Harness status lines are skipped, and an MCP content wrapper is opened so
    the tool's own first and last lines are read. The middle of a long answer
    is never read: a good result that talks about errors is not a failure.
    """
    first = next(iter(_meaningful_lines(part)), "")
    if not first:
        return []
    body = part[part.find(first) :] if first in part else first
    texts = _unwrapped_texts(body[:_MAX_OBSERVATION_CHARS])
    if not texts:
        return [first, next(iter(_meaningful_lines(part, last=True)), "")]
    lines: list[str] = []
    for text in texts:
        lines.append(next(iter(_meaningful_lines(text)), ""))
        lines.append(next(iter(_meaningful_lines(text, last=True)), ""))
    return lines


def _failure_scan_text(parts: Sequence[str], text: str) -> str:
    """The lead lines failure text is checked against: each block's first and last lines, and the result's last."""
    lines = [line for part in parts[:64] for line in _lead_lines(part)]
    if len(parts) > 1:
        lines.append(next(iter(_meaningful_lines(text, last=True)), ""))
    return "\n".join(line[:_FAILURE_SCAN_CHARS] for line in lines if line)


def _line_failed(line: str, *, shell: bool) -> bool:
    """Whether one lead line says the call failed (see the marker notes above)."""
    lowered = line.casefold()
    if any(phrase in lowered for phrase in _FAILURE_PHRASES):
        return True
    if _UNAVAILABLE_TAIL_RE.search(line[-64:]):
        return True
    for prefix in _STATUS_LINE_PREFIX_RES:
        opened = prefix.match(line)
        if opened is not None and _HTTP_STATUS_OPENING_RE.match(line, opened.end()):
            return True
    # The line, then the line past each leading ``name:`` segment (at most three).
    head = line
    for _ in range(4):
        if any(
            pattern.match(head)
            for pattern in (
                _HARNESS_ERROR_PREFIX_RE,
                _HTTP_STATUS_OPENING_RE,
                _HTTP_VERSION_STATUS_RE,
                _REQUEST_FAILED_RE,
                _OPENING_FAILURE_RE,
            )
        ):
            return True
        if not shell and _FILE_MISSING_RE.match(head):
            return True
        segment = _LEAD_SEGMENT_RE.match(head)
        if segment is None:
            return False
        head = head[segment.end() :]
    return False


def _flag_value(value: Any) -> bool | None:
    """``True``/``False`` for a boolean flag in any common spelling, else ``None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return value == 1
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in _TRUE_FLAGS:
            return True
        if lowered in _FALSE_FLAGS:
            return False
    return None


def _result_flagged(result: Mapping[str, Any]) -> bool | None:
    """A result's structured outcome: ``True`` failed, ``False`` succeeded, ``None`` no signal.

    A failure flag is any ``is_error``/``isError`` (top level or under
    ``extra``), Harbor's Claude Code keys ``extra.tool_result_is_error`` and
    ``extra.tool_result_metadata.is_error`` (and its ``raw_tool_result``), a
    non-empty ``error``, an MCP ``content.isError``, or a failed
    ``extra.harness_status`` (Codex ``codex.txt``). Only the harness's own
    report says a call succeeded: an explicit ``tool_result_is_error: false``
    or a ``completed`` harness status.
    """
    extra = result.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    flags: list[Any] = []
    for source in (result, extra):
        flags.extend(source.get(key) for key in _ERROR_FLAG_KEYS)
        error = source.get("error")
        if error not in (None, False, "", {}, []):
            return True
    for key in ("tool_result_metadata", "metadata"):
        metadata = extra.get(key)
        if isinstance(metadata, Mapping):
            flags.extend(metadata.get(name) for name in _ERROR_FLAG_KEYS)
            raw = metadata.get("raw_tool_result")
            if isinstance(raw, Mapping):
                flags.append(raw.get("is_error"))
    content = result.get("content")
    if isinstance(content, Mapping):
        flags.append(content.get("isError"))
    if any(_flag_value(flag) is True for flag in flags):
        return True
    harness = _flag_value(extra.get("tool_result_is_error"))
    if harness is True:
        return True
    status = extra.get(HARNESS_STATUS_KEY)
    status = status.strip().casefold() if isinstance(status, str) else ""
    if status in _HARNESS_FAILED:
        return True
    if harness is False or status in _HARNESS_SUCCEEDED:
        return False
    return None


def _observations(step: Mapping[str, Any]) -> list[_Result]:
    """Each result of ``step``.

    ``flagged`` is the structured outcome of :func:`_result_flagged`. Harbor
    puts the flag of an orphan Claude Code result on its single-call step's
    ``extra``, so a single-call step's own flags count for its results too.
    """
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        return []
    results = observation.get("results")
    if not isinstance(results, list):
        return []
    calls = step.get("tool_calls")
    step_extra = step.get("extra")
    step_flag = (
        _result_flagged({"extra": step_extra})
        if isinstance(calls, list) and len(calls) == 1 and isinstance(step_extra, Mapping)
        else None
    )
    entries: list[_Result] = []
    for result in results[:_MAX_STEP_RESULTS]:
        if not isinstance(result, Mapping):
            continue
        parts = _content_parts(result.get("content"))
        text = "\n".join(parts)[:_MAX_OBSERVATION_CHARS]
        flagged = _result_flagged(result)
        if step_flag is True or (flagged is None and step_flag is False):
            flagged = step_flag
        entries.append(
            _Result(
                call_id=str(result.get("source_call_id") or ""),
                text=text,
                scan=_failure_scan_text(parts, text),
                flagged=flagged,
            )
        )
    return entries


def _results_for_call(results: Sequence[_Result], call_id: str, *, call_count: int) -> list[_Result]:
    """Results attributable to ``call_id``; id matches win, ambiguity stays unknown."""
    if call_id:
        matched = [result for result in results if result.call_id == call_id]
        if matched:
            return matched
    if call_count == 1 and len(results) == 1 and not results[0].call_id:
        return [results[0]]
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
    correlated: Sequence[_Result], *, shell: bool, content_read: bool = False
) -> tuple[str | None, bool | None]:
    """``(observation text, succeeded)`` for one call (see the module docs).

    A structured outcome wins: any failure flag fails the call, and the
    harness's own success report (Claude Code ``tool_result_is_error: false``,
    a Codex ``completed`` status) passes it whatever the text says. Without
    one, failure text is read on the result's lead lines only. For a
    ``content_read`` (a file read or a skill load) the body is file or skill
    text, so only the first line of a result (past a harness's shell status
    lines) or a shell error line that names a skill manifest counts.
    """
    if not correlated:
        return None, None
    text = "".join(result.text for result in correlated)[:_MAX_OBSERVATION_CHARS]
    flags = [result.flagged for result in correlated]
    if any(flag is True for flag in flags):
        return text, False
    if any(flag is False for flag in flags):
        return text, True
    if not text.strip():
        return text, None
    if content_read:
        scan = "\n".join(_content_read_scan(result.text, shell=shell) for result in correlated)
    else:
        scan = "\n".join(result.scan for result in correlated)
    failed = any(_line_failed(line.strip(), shell=shell) for line in scan.splitlines() if line.strip())
    return text, not failed


def _base_tool_name(fn: str) -> str:
    low = fn.casefold()
    for separator in _TOOL_NAME_SEPARATORS:
        if separator in low:
            low = low.rsplit(separator, 1)[-1]
    return low


def _norm_server(value: str) -> str:
    return _SANITIZED_NAME_RE.sub("_", value.casefold())


def _server_spellings(server: str) -> tuple[str, ...]:
    """Casefolded spellings of a declared server name inside harness tool names, exact spelling first.

    Claude Code and OpenCode replace characters outside ``[A-Za-z0-9_-]`` with
    ``_``; Hermes also replaces ``-``.
    """
    low = server.casefold()
    return tuple(dict.fromkeys((low, _norm_server(server), _HERMES_NAME_RE.sub("_", low))))


class _McpPrefix(NamedTuple):
    """A tool-name prefix that names a declared MCP server (see :meth:`_McpNames.identity`)."""

    order: int  # scan order: declared server, then its spelling, then the prefix form
    length: int
    rank: int  # 0 for the server's exact spelling, 1 or 2 for a harness-normalized one
    server: str
    spelling: str
    bare: bool  # OpenCode's ``<server>_<tool>``, a form other harnesses never use


class _McpNames:
    """The declared MCP servers of one arm, indexed by every spelling and tool-name prefix harnesses give them.

    Built once per trajectory (or per batch of lookups), so naming the server
    of a call takes a few dictionary lookups however many servers are
    declared. The tool-name prefix table is built on the first
    :meth:`identity`, since :meth:`match` does not use it.
    """

    def __init__(self, declared: Iterable[Any]) -> None:
        self.names = list(dict.fromkeys(name for name in declared if isinstance(name, str) and name))
        self._exact: dict[str, str] = {}  # casefolded name -> first declared name
        self._spelled: dict[str, list[str]] = {}  # spelling -> the declared names that have it
        for name in self.names:
            self._exact.setdefault(name.casefold(), name)
            for spelling in _server_spellings(name):
                holders = self._spelled.setdefault(spelling, [])
                if name not in holders:
                    holders.append(name)

    @cached_property
    def _prefixes(self) -> dict[str, list[_McpPrefix]]:
        """Tool-name prefix -> the declared-server entries it names (see :meth:`identity`)."""
        prefixes: dict[str, list[_McpPrefix]] = {}
        order = 0
        for name in self.names:
            for rank, spelling in enumerate(_server_spellings(name)):
                forms = [(spelling + separator, False) for separator in _TOOL_NAME_SEPARATORS]
                forms += [(f"mcp_{spelling}_", False), (spelling + "_", True)]
                for prefix, bare in forms:
                    entry = _McpPrefix(order, len(prefix), rank, name, spelling, bare)
                    prefixes.setdefault(prefix, []).append(entry)
                    order += 1
        return prefixes

    @cached_property
    def _prefix_ends(self) -> frozenset[str]:
        return frozenset(prefix[-1] for prefix in self._prefixes)

    def match(self, observed: str) -> str | None:
        """See :func:`match_declared_mcp_server`."""
        low = observed.casefold()
        if not low or not self.names:
            return None
        exact = self._exact.get(low)
        if exact is not None:
            return exact
        spelled = self._spelled.get(low, [])
        if len(spelled) == 1:
            return spelled[0]
        if spelled or not low.startswith(_CLAUDE_PLUGIN_SERVER_PREFIX):
            return None
        rest = low[len(_CLAUDE_PLUGIN_SERVER_PREFIX) :]
        slug, sep, server = rest.partition("_")
        if slug and sep and server:
            found = self.match(server)
            if found is not None:
                return found
        # The longest declared spelling ``rest`` ends with, after a ``_`` that is not its first character.
        for index in range(1, len(rest)):
            if rest[index] == "_" and (longest := self._spelled.get(rest[index + 1 :])):
                return longest[0] if len(longest) == 1 else None
        return None

    def identity(self, fn: str, *, agent: str) -> tuple[str, str] | None:
        """``(server, tool)`` of an MCP tool call, with the server mapped to its declared name when known.

        Recognized spellings: ``mcp__<server>__<tool>`` (Claude Code, Codex;
        Claude Code plugin servers appear as ``plugin_<plugin>_<server>``), and
        for declared servers ``<server>__<tool>``/``.``/``/``/``:``, Hermes
        ``mcp_<server>_<tool>``, and OpenCode ``<server>_<tool>`` (not for
        harnesses that never use it). The longest matching prefix wins; on a tie
        an exact spelling beats a normalized one, and a remaining tie between
        servers is left unattributed.
        """
        low = fn.casefold()
        if low.startswith(_MCP_PREFIX):
            server, _, tool = fn[len(_MCP_PREFIX) :].partition("__")
            if not server:
                return None
            return self.match(server) or server, tool
        bare_allowed = agent.casefold() not in _NO_BARE_MCP_PREFIX_AGENTS and low not in _BUILTIN_TOOL_NAMES
        matches: list[_McpPrefix] = []
        for end in range(1, len(low)):  # a prefix leaves at least one character of tool name
            if low[end - 1] in self._prefix_ends:
                matches.extend(self._prefixes.get(low[:end], ()))
        best: tuple[int, int] | None = None
        best_spelling = ""
        winners: list[str] = []
        for entry in sorted(matches):
            if entry.bare and not bare_allowed:
                continue
            key = (entry.length, -min(entry.rank, 1))
            if best is None or key > best:
                best, best_spelling, winners = key, entry.spelling, [entry.server]
            elif key == best and entry.server not in winners:
                winners.append(entry.server)
        if best is None:
            return None
        # Two declared servers share this spelling: count the call under the spelling itself.
        return (winners[0] if len(winners) == 1 else best_spelling), fn[best[0] :]


def match_declared_mcp_server(observed: str, declared: Iterable[str]) -> str | None:
    """The one declared server that ``observed`` (a server name taken from a tool name) refers to.

    An exact (case-insensitive) match wins. Otherwise a harness spelling of a
    declared name counts only when exactly one declared server has it, so
    ``my.docs`` and ``my-docs`` never credit each other. Claude Code names
    plugin servers ``plugin_<plugin>_<server>``; the plugin slug never holds
    ``_``, so the server is what follows the first ``_`` (a longest declared
    suffix is the fallback).
    """
    return _McpNames(declared).match(observed)


def declared_mcp_server_matcher(declared: Iterable[str]) -> Callable[[str], str | None]:
    """:func:`match_declared_mcp_server` for one set of declared servers, indexed once for many lookups."""
    return _McpNames(declared).match


def _first_string(args: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_path(value: str) -> str:
    text = value.strip().strip("'\"").replace("\\", "/")
    text = _REPEATED_SLASHES_RE.sub("/", text)
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


def _split_heredocs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """``(text without here-document bodies, [(delimiter, body), ...] in order)``.

    Bodies are data, not commands, and may hold stray quotes, so the shell
    parser never sees them. A body is split off only when its terminator line
    is found, so a ``<<`` that is not really a here-document never swallows
    the rest of the script.
    """
    if "<<" not in text:
        return text, []
    lines = text.split("\n")
    kept: list[str] = []
    bodies: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        index += 1
        for match in _HEREDOC_RE.finditer(line):
            delimiter = next(group for group in match.groups() if group)
            end = next((at for at in range(index, len(lines)) if lines[at].strip() == delimiter), None)
            if end is None:
                return text, []
            bodies.append((delimiter, "\n".join(lines[index:end])))
            index = end + 1
    return "\n".join(kept), bodies


def _strip_heredoc_bodies(text: str) -> str:
    """``text`` without its here-document bodies (see :func:`_split_heredocs`)."""
    return _split_heredocs(text)[0]


def _take_heredoc_bodies(delimiters: Sequence[str], pending: list[tuple[str, str]]) -> list[str]:
    """Pop the bodies of one command's here-documents (by delimiter, in order) off ``pending``.

    A ``<<`` the shell parser saw with no body (``$((1<<2))``) takes nothing.
    A body passed over belonged to a ``<<`` that was not a real operator.
    """
    bodies: list[str] = []
    for delimiter in delimiters:
        at = next((index for index, (name, _) in enumerate(pending) if name == delimiter), None)
        if at is None:
            continue
        bodies.append(pending[at][1])
        del pending[: at + 1]
    return bodies


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


def _shell_chain(text: str, depth: int = 0) -> list[tuple[list[str], str]]:
    """``(command words, joiner before it)`` for each simple command, in order.

    The joiner is the operator that links a command to the one before it
    (``&&``, ``||``, ``;``, ``|``...; ``""`` for the first). ``bash -lc '...'``
    payloads are expanded in place; their first command takes the outer joiner.
    """
    if depth > _MAX_SHELL_DEPTH or len(text) > _MAX_SHELL_TEXT_CHARS:
        return []
    chain: list[tuple[list[str], str]] = []
    current: list[str] = []
    joiner = ""
    for token in _shell_tokens(text):
        if token in _SHELL_SEPARATORS:
            if current:
                chain.append((current, joiner))
                current = []
            # A subshell keeps the operator that leads into it; after it, the next operator links.
            if token == ")":
                joiner = ";"
            elif token != "(":
                joiner = token
        else:
            current.append(token)
    if current:
        chain.append((current, joiner))
    expanded: list[tuple[list[str], str]] = []
    for command, link in chain:
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
                inner = _shell_chain(payload, depth + 1)
                if inner:
                    inner[0] = (inner[0][0], link)
                expanded.extend(inner)
                continue
        expanded.append((command, link))
    return expanded


def _shell_commands(text: str, depth: int = 0) -> list[list[str]]:
    return [command for command, _ in _shell_chain(text, depth)]


@dataclass
class _CommandIO:
    """The file operands of one simple shell command."""

    verb: str = ""
    operands: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    # Here-document delimiters, in order (``cat <<'EOF' > out.json`` has ``EOF``).
    heredocs: list[str] = field(default_factory=list)


def _command_io(command: Sequence[str]) -> _CommandIO:
    io = _CommandIO()
    index = 0
    while index < len(command):
        token = command[index]
        if token == ">&":
            # ``2>&1``-style descriptor duplication, not a file target.
            index += 2
            continue
        if token in _OUTPUT_REDIRECTS:
            if index + 1 < len(command):
                io.outputs.append(command[index + 1])
            index += 2
            continue
        if token == "<":
            if index + 1 < len(command):
                io.inputs.append(command[index + 1])
            index += 2
            continue
        if token in {"<<", "<<-"}:
            # A here-document: its delimiter is not an operand, and redirects may follow it.
            if index + 1 < len(command):
                io.heredocs.append(command[index + 1].lstrip("-"))
            index += 2
            continue
        if token == "<<<":
            # A here-string is stdin data, not a file operand.
            index += 2
            continue
        if not io.verb:
            io.verb = token.rsplit("/", 1)[-1].casefold()
        elif token.startswith("-"):
            io.flags.append(token)
        else:
            io.operands.append(token)
        index += 1
    return io


def _sed_in_place(io: _CommandIO) -> bool:
    return io.verb == "sed" and any(flag.startswith("-i") or flag.startswith("--in-place") for flag in io.flags)


_INTERPRETERS = frozenset({"python", "python3", "node", "ruby", "perl", "deno", "bun"})
_INLINE_CODE_FLAGS = frozenset({"-c", "-e", "--eval", "-p", "--print"})
# Interpreter code that opens a file, given as a path literal (``open('out/a.json', 'w')``,
# ``Path('out/a.json').write_text(...)``) or as a name assigned one path literal
# (``out = Path('out/a.json')`` ... ``out.write_text(...)``). ``{T}`` is the file.
_CODE_LITERAL = r"""(?P<q>['"])(?P<p>[^'"\n]{1,512})(?P=q)"""
_CODE_PATH_LITERAL = r"\b(?:pathlib\.)?Path\(\s*" + _CODE_LITERAL + r"\s*\)"
_CODE_NAME = r"(?<![\w.])(?P<v>[A-Za-z_]\w{0,63})\b"
_CODE_WRITE_MODE = r"""\s*(?:mode\s*=\s*)?['"][^'"\n]{0,8}[wax]"""
_CODE_NOT_WRITE_MODE = r"""(?!\s*(?:mode\s*=\s*)?['"][^'"\n]{0,8}[wax])"""
# ``name = 'out/a.json'``, ``name = Path('out/a.json')``, or ``const name = 'out/a.json';`` on a line of its own.
_CODE_ASSIGN_RE = re.compile(
    r"^[ \t]*(?:(?:const|let|var)\s+)?(?P<name>[A-Za-z_]\w{0,63})\s*=\s*"
    r"(?P<path_call>\b(?:pathlib\.)?Path\(\s*)?" + _CODE_LITERAL + r"(?(path_call)\s*\))\s*;?[ \t]*(?:#[^\n]*)?$",
    re.MULTILINE,
)


def _code_forms(*forms: tuple[str, str]) -> tuple[tuple[re.Pattern[str], bool], ...]:
    """``(pattern, by_name)`` for each form, once with its literal file and once with a name."""
    compiled: list[tuple[re.Pattern[str], bool]] = []
    for template, literal in forms:
        compiled.append((re.compile(template.replace("{T}", literal)), False))
        compiled.append((re.compile(template.replace("{T}", _CODE_NAME)), True))
    return tuple(compiled)


_CODE_WRITE_RES = _code_forms(
    (r"\bopen\(\s*{T}\s*," + _CODE_WRITE_MODE, _CODE_LITERAL),
    (r"{T}\s*\.\s*write_(?:text|bytes)\(", _CODE_PATH_LITERAL),
    (r"{T}\s*\.\s*open\(" + _CODE_WRITE_MODE, _CODE_PATH_LITERAL),
    (r"\b(?:writeFileSync|appendFileSync|writeFile|appendFile)\(\s*{T}", _CODE_LITERAL),
    (r"\.to_(?:csv|json|parquet|excel)\(\s*{T}", _CODE_LITERAL),
)
_CODE_READ_RES = _code_forms(
    (r"\bopen\(\s*{T}\s*(?:\)|," + _CODE_NOT_WRITE_MODE + ")", _CODE_LITERAL),
    (r"{T}\s*\.\s*read_(?:text|bytes)\(", _CODE_PATH_LITERAL),
    (r"{T}\s*\.\s*open\(" + _CODE_NOT_WRITE_MODE, _CODE_PATH_LITERAL),
    (
        r"\b(?:readFileSync|readFile|read_csv|read_json|read_table|read_parquet|read_excel|load_workbook)\(\s*{T}",
        _CODE_LITERAL,
    ),
)


def _resolve_path(path: str, cwd: str | None) -> str:
    """``path`` normalized and, when relative and the directory is known, made absolute."""
    text = _normalize_path(path)
    if not text or text.startswith("~"):
        return text
    if not text.startswith("/") and cwd:
        text = f"{cwd.rstrip('/')}/{text}"
    return posixpath.normpath(text) if text.startswith("/") else text


def _code_paths(code: str, forms: Sequence[tuple[re.Pattern[str], bool]], names: Mapping[str, str]) -> list[str]:
    paths: list[str] = []
    for pattern, by_name in forms:
        for match in pattern.finditer(code):
            path = names.get(match.group("v")) if by_name else match.group("p")
            if path:
                paths.append(path)
    return paths


def _inline_code_io(code: str) -> tuple[list[str], list[str]]:
    """``(read, written)`` paths that interpreter code opens.

    The code is a ``-c``/``-e`` program (``python3 -c "open('f','w')..."``) or
    a here-document fed to the interpreter (``python3 - <<'PY' ... PY``). A file
    is a path literal or a name assigned exactly one path literal.
    """
    code = code[:_MAX_SHELL_TEXT_CHARS]
    assigned: dict[str, set[str]] = {}
    for match in _CODE_ASSIGN_RE.finditer(code):
        assigned.setdefault(match.group("name"), set()).add(match.group("p"))
    names = {name: next(iter(paths)) for name, paths in assigned.items() if len(paths) == 1}
    writes = list(dict.fromkeys(_code_paths(code, _CODE_WRITE_RES, names)))
    reads = [path for path in dict.fromkeys(_code_paths(code, _CODE_READ_RES, names)) if path not in writes]
    return reads, writes


def _shell_file_io(text: str, cwd: str | None) -> tuple[list[str], list[str]]:
    """``(read_paths, written_paths)`` of shell ``text``, resolved against ``cwd`` (``cd`` moves it).

    Reads are operands of reader and consumer verbs (``cat``, ``jq``, ``python3
    script.py <path>``...) and ``<`` input; writes are output redirects,
    ``tee``, ``cp``/``mv``/``install`` targets, ``touch``, and ``sed -i`` (also a
    read). Interpreter code, inline (``-c``/``-e``) or fed as a here-document,
    is scanned for the paths it opens. A glob operand stays a glob. A shell
    ``apply_patch`` (Codex) writes the files its patch headers name (see
    :func:`_patch_written_paths`).
    """
    reads: list[str] = []
    writes: list[str] = []
    pending = _split_heredocs(text.replace("\r\n", "\n").replace("\r", "\n"))[1]
    patched = False
    for command, _ in _shell_chain(text):
        io = _command_io(command)
        verb, operands = io.verb, io.operands
        bodies = _take_heredoc_bodies(io.heredocs, pending) if io.heredocs and pending else []
        if verb == "cd":
            target = operands[0] if operands else ""
            cwd = _resolve_path(target, cwd) if target and not target.startswith(("-", "~", "$")) else None
            continue
        command_reads = list(io.inputs)
        command_writes = list(io.outputs)
        if not patched and any(_APPLY_PATCH_COMMAND_RE.fullmatch(word.rsplit("/", 1)[-1]) for word in command):
            # The patch may be a here-document, an argument or piped text, so its headers are read from
            # the whole script, against the directory this command runs in.
            patched = True
            command_writes += _patch_written_paths(text)
        if verb in _INTERPRETERS and any(flag in _INLINE_CODE_FLAGS for flag in io.flags):
            code_reads, code_writes = _inline_code_io(operands[0] if operands else "")
            command_reads += code_reads
            command_writes += code_writes
        elif verb in _INTERPRETERS and bodies and not operands:
            # ``python3 - <<'PY' ... PY``: the here-document is the program.
            code_reads, code_writes = _inline_code_io("\n".join(bodies))
            command_reads += code_reads
            command_writes += code_writes
        elif _sed_in_place(io):
            edited = operands[1:] if len(operands) > 1 else operands
            command_reads += edited
            command_writes += edited
        elif verb in _ARTIFACT_CONSUMER_VERBS:
            command_reads += operands
        elif verb == "tee":
            command_writes += operands
        elif verb in {"cp", "mv", "install"} and len(operands) >= 2:
            command_writes.append(operands[-1])
            command_reads += operands[:-1]
        elif verb == "touch":
            command_writes += operands
        reads.extend(_resolve_path(path, cwd) for path in command_reads)
        writes.extend(_resolve_path(path, cwd) for path in command_writes)
    return reads, writes


def _manifest_operands(command: Sequence[str]) -> list[str]:
    """Operands this command really reads as file text: reader verbs only.

    ``cp``/``mv`` sources, ``<`` input to a non-reader (``grep x < SKILL.md``),
    and ``sed -i`` edits do not load a skill.
    """
    io = _command_io(command)
    if io.verb not in _FILE_READER_VERBS or _sed_in_place(io):
        return []
    return [*io.operands, *io.inputs]


def _is_editor_view(fn_base: str, args: Mapping[str, Any]) -> bool:
    """A text editor tool call running its ``view`` command (a read)."""
    return fn_base in _STR_REPLACE_EDITORS and str(args.get("command") or "").casefold() == "view"


def _member_manifest_match(path: str, members: Sequence[str]) -> str | None:
    normalized = _normalize_path(path).casefold()
    for member in members:
        suffix = f"{member.casefold()}/skill.md"
        if normalized == suffix or normalized.endswith("/" + suffix):
            return member
    return None


@dataclass(frozen=True)
class _ManifestRead:
    """One read of a declared member's ``SKILL.md``: the operand as written and its shell command."""

    member: str
    operand: str
    command: int = 0


@dataclass(frozen=True)
class _ManifestReads:
    reads: tuple[_ManifestRead, ...] = ()
    # Joiner before each shell command of the call (empty for a file-read tool).
    joiners: tuple[str, ...] = ()


def _manifest_reads(fn_base: str, args: Mapping[str, Any], members: Sequence[str], *, is_mcp: bool) -> _ManifestReads:
    """The declared members whose ``SKILL.md`` this call reads, in order (every one in a chained shell command)."""
    if not members:
        return _ManifestReads()
    if fn_base in _STR_REPLACE_EDITORS:
        reads_file = _is_editor_view(fn_base, args)
    else:
        reads_file = fn_base in _READ_TOOLS or (is_mcp and bool(_READ_VERB_RE.search(fn_base)))
    found: list[_ManifestRead] = []
    joiners: list[str] = []
    if reads_file:
        for path in _path_args(args):
            member = _member_manifest_match(path, members)
            if member is not None:
                found.append(_ManifestRead(member, path))
    elif fn_base in _SHELL_TOOLS and not is_mcp:
        for text in _shell_texts(fn_base, args):
            if "skill.md" not in text.casefold():
                continue
            for command, joiner in _shell_chain(text):
                for operand in _manifest_operands(command):
                    member = _member_manifest_match(operand, members)
                    if member is not None:
                        found.append(_ManifestRead(member, operand, len(joiners)))
                joiners.append(joiner)
    return _ManifestReads(tuple(found), tuple(joiners))


def _failed_operands(observation: str, operands: Iterable[str]) -> set[str]:
    """Operands that a shell error line in ``observation`` names as unreadable."""
    lines = [
        line.casefold()
        for line in observation[:_MAX_OBSERVATION_CHARS].splitlines()
        if any(phrase in line.casefold() for phrase in _READ_FAILURE_PHRASES)
    ]
    if not lines:
        return set()
    return {operand for operand in operands if operand and any(operand.casefold() in line for line in lines)}


def _commands_that_ran(joiners: Sequence[str], failed: set[int]) -> list[bool]:
    """Which commands of a shell list ran: ``a && b`` skips ``b`` when ``a`` failed, ``a || b`` when it did not."""
    ran: list[bool] = []
    status_ok = True
    for index, joiner in enumerate(joiners):
        run = not ((joiner == "&&" and not status_ok) or (joiner == "||" and status_ok))
        ran.append(run)
        if run:
            status_ok = index not in failed
    return ran


def _resolve_manifest_reads(manifest: _ManifestReads, observation: str | None) -> list[tuple[_ManifestRead, bool]]:
    """``(read, failed)`` for each read that ran, in order; a read skipped by a failed ``&&`` part is dropped."""
    if not manifest.reads:
        return []
    if not manifest.joiners:
        return [(read, False) for read in manifest.reads]
    failed_operands = _failed_operands(observation or "", {read.operand for read in manifest.reads})
    failed_commands = {read.command for read in manifest.reads if read.operand in failed_operands}
    ran = _commands_that_ran(manifest.joiners, failed_commands)
    return [
        (read, read.operand in failed_operands)
        for read in manifest.reads
        if read.command >= len(ran) or ran[read.command]
    ]


def _norm_namespace(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _plugin_namespaces(declared: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Casefolded names of the plugin under evaluation (its Claude Code namespace)."""
    names = declared.get(DECLARED_PLUGIN) or ()
    return tuple(dict.fromkeys(name.strip().casefold() for name in names if isinstance(name, str) and name.strip()))


def _foreign_namespace(name: str, namespaces: Sequence[str]) -> bool:
    """Whether ``<namespace>:<name>`` names another plugin's component (unknown plugin name: never)."""
    namespace, sep, _ = name.rpartition(":")
    if not sep or not namespaces:
        return False
    return _norm_namespace(namespace) not in {_norm_namespace(item) for item in namespaces}


def _component_name(name: str, kind: str, declared: Mapping[str, Sequence[str]]) -> tuple[str, bool]:
    """``(name, foreign)``: a declared component's own spelling when the name is this plugin's.

    ``release-kit:version-bump`` and ``version-bump`` both become the declared
    ``version-bump``, so one component is one identity. A name under another
    plugin's namespace stays as written and is marked foreign.
    """
    if _foreign_namespace(name, _plugin_namespaces(declared)):
        return name, True
    short = name.rpartition(":")[2] if ":" in name else name
    folded = short.casefold()
    for member in declared.get(kind) or ():
        if isinstance(member, str) and member.casefold() == folded:
            return member, False
    return name, False


def _declared_command(name: str, declared: Mapping[str, Sequence[str]]) -> bool:
    """Whether a ``Skill`` tool name is a declared plugin command (``<plugin>:<command>`` or bare).

    Claude Code runs plugin commands through its ``Skill`` tool. A name that is
    also a declared member skill stays a skill; another plugin's name is never
    this plugin's command.
    """
    commands = {command.casefold() for command in declared.get(COMPONENT_COMMAND) or ()}
    if not name or not commands or _foreign_namespace(name, _plugin_namespaces(declared)):
        return False
    candidates = _name_candidates(name)
    skills = {skill.casefold() for skill in declared.get(COMPONENT_SKILL) or ()}
    return candidates[-1] in commands and not any(candidate in skills for candidate in candidates)


def _component_ident(
    kind: str,
    name: str,
    fn: str,
    tool_label: str,
    *,
    persist: bool = True,
    declared: Mapping[str, Sequence[str]] | None = None,
    wrapper_skills: Sequence[str] = (),
    **extra: Any,
) -> _Ident:
    foreign = False
    if declared is not None and name:
        name, foreign = _component_name(name, kind, declared)
    label = f"{_IDENTITY_PREFIX[kind]}:{name}" if name else _IDENTITY_PREFIX[kind]
    ident = _Ident(
        label=label,
        kind=kind,
        name=name,
        fn=fn,
        tool_label=tool_label,
        persist_name=persist,
        foreign=foreign,
        namespaces=_plugin_namespaces(declared or {}),
        **extra,
    )
    if kind == COMPONENT_SKILL and wrapper_skills and _is_wrapper(ident, wrapper_skills, declared or {}):
        ident = replace(ident, wrapper=True)
    return ident


def _persistable_name(name: str, members: Sequence[str]) -> bool:
    """Whether an argument-derived component name may be persisted (identifier-shaped or declared)."""
    if not name or (_PERSISTABLE_NAME_RE.fullmatch(name) and "://" not in name):
        return True
    folded = name.casefold()
    return any(member.casefold() == folded for member in members)


def _mcp_ident(fn: str, mcp_names: _McpNames, *, agent: str) -> _Ident | None:
    """The MCP identity of a call named ``fn``, labeled ``mcp__<server>__<tool>`` (see :meth:`_McpNames.identity`)."""
    identity = mcp_names.identity(fn, agent=agent)
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


def _builtin_agent(name: str, agent: str, declared: Mapping[str, Sequence[str]]) -> bool:
    """Whether a bare subagent name reaches ``agent``'s own built-in agent, though a declared plugin agent has it."""
    folded = name.casefold()
    if ":" in folded or folded not in _BUILTIN_AGENTS.get(agent.casefold(), ()):
        return False
    return any(isinstance(item, str) and item.casefold() == folded for item in declared.get(COMPONENT_SUBAGENT) or ())


def _identities(
    fn: str,
    fn_base: str,
    mcp: _Ident | None,
    args: Mapping[str, Any],
    declared: Mapping[str, Sequence[str]],
    *,
    subagent_aliases: Mapping[str, str] | None = None,
    wrapper_skills: Sequence[str] = (),
    manifest: Sequence[tuple[_ManifestRead, bool]] | None = None,
    agent: str = "",
    lsp_extensions: Mapping[str, str] | None = None,
) -> list[_Ident]:
    """The identities of one call. ``manifest`` is the call's resolved ``SKILL.md`` reads, if known.

    ``agent`` names the harness, whose own agents a bare subagent name may reach.
    ``lsp_extensions`` maps a file extension to the staged LSP server that serves it.
    """
    low = fn.casefold()
    idents: list[_Ident] = []
    named = {"declared": declared, "wrapper_skills": wrapper_skills}
    if low in _SKILL_TOOLS:
        name = _first_string(args, ("skill", "name", "command"))
        if _declared_command(name.lstrip("/"), declared):
            name = name.lstrip("/")
            persist = _persistable_name(name, declared.get(COMPONENT_COMMAND) or ())
            idents.append(_component_ident(COMPONENT_COMMAND, name, fn, fn, persist=persist, **named))
        else:
            persist = _persistable_name(name, declared.get(COMPONENT_SKILL) or ())
            idents.append(_component_ident(COMPONENT_SKILL, name, fn, fn, persist=persist, **named))
    elif low in _SUBAGENT_TOOLS:
        # Claude Code 2.1.29x's Agent tool names the subagent in ``type`` (read for these tools only).
        name = _first_string(args, ("subagent_type", "subagent", "agent", "agent_name", "agent_type", "type"))
        # A harness that renamed a plugin agent when staging it calls it by the staged name.
        alias = (subagent_aliases or {}).get(name.casefold())
        persist = _persistable_name(name, ())
        if alias is None and _builtin_agent(name, agent, declared):
            idents.append(_component_ident(COMPONENT_SUBAGENT, name, fn, fn, persist=persist, builtin=True))
        else:
            name = alias or name
            idents.append(_component_ident(COMPONENT_SUBAGENT, name, fn, fn, persist=persist, **named))
    elif low in _COMMAND_TOOLS:
        words = _first_string(args, ("command", "name")).split()
        name = words[0].lstrip("/") if words else ""
        idents.append(_component_ident(COMPONENT_COMMAND, name, fn, fn, persist=_persistable_name(name, ()), **named))
    elif low in _LSP_TOOLS and lsp_extensions and (lsp := _lsp_ident(fn, args, lsp_extensions)) is not None:
        idents.append(lsp)
    if mcp is not None:
        idents.append(mcp)
    if not any(ident.kind in {COMPONENT_SKILL, COMPONENT_COMMAND} for ident in idents):
        if manifest is None:
            reads = _manifest_reads(fn_base, args, declared.get(COMPONENT_SKILL) or (), is_mcp=mcp is not None)
            manifest = [(read, False) for read in reads.reads]
        # One identity per member, in first-read order; it failed only when every read of it failed.
        members: dict[str, bool] = {}
        for read, failed in manifest:
            members[read.member] = members.get(read.member, True) and failed
        for position, (member, failed) in enumerate(members.items()):
            idents.append(
                _component_ident(
                    COMPONENT_SKILL,
                    member,
                    fn,
                    f"{fn}:skill-md-read",
                    via_read=True,
                    position=position,
                    failed=failed,
                    **named,
                )
            )
    if mcp is None:
        # The tool itself is a plain identity too, so ``Bash`` or ``Read`` refs see a call that also
        # loaded a skill, without that skill answering to the tool's name.
        aliases = _TOOL_ALIASES.get(fn_base, ())
        idents.append(_Ident(label=fn, kind=None, name=fn, fn=fn, tool_label=fn, aliases=aliases))
    return idents


def _trajectory_agent(trajectory: Mapping[str, Any]) -> str:
    agent = trajectory.get("agent")
    name = agent.get("name") if isinstance(agent, Mapping) else None
    return name.strip()[:_MAX_LABEL_CHARS] if isinstance(name, str) else ""


def _step_tool_calls(step: Any) -> list[Mapping[str, Any]]:
    if not isinstance(step, Mapping) or not isinstance(step.get("tool_calls"), list):
        return []
    return [item for item in step["tool_calls"] if isinstance(item, Mapping)]


def _expanded_tool_calls(raw: Mapping[str, Any], outer_id: str, mcp_call_servers: Mapping[str, str]) -> list[Any]:
    """One raw tool call as normalized calls: a Codex ``exec`` wrapper becomes the calls it made."""
    name = _tool_name(raw)
    server = mcp_call_servers.get(outer_id) if outer_id else None
    if server and name and not name.casefold().startswith(_MCP_PREFIX):
        # The harness log names the server this bare MCP tool name came from (Codex).
        name = f"mcp__{server}__{name}"[:_MAX_LABEL_CHARS]
    prepared = {**raw, "function_name": name, "arguments": _arguments(raw)}
    try:
        return normalize_tool_call(prepared)
    except (TypeError, ValueError, RecursionError):
        return [prepared]


def _claimable_results(
    tool_call: Mapping[str, Any], results: Sequence[_Result], outer_id: str, *, call_count: int
) -> list[_Result]:
    """The step results this call may claim: an unwrapped inner call only when the normalizer proved it owns them."""
    status = tool_call.get("_atif_observation_status")
    if status is None or status == MAPPED_OUTER_EXEC_OBSERVATION:
        return _results_for_call(results, outer_id, call_count=call_count)
    return []


def _identify_call(
    tool_call: Mapping[str, Any],
    correlated: Sequence[_Result],
    declared: Mapping[str, Sequence[str]],
    mcp_names: _McpNames,
    *,
    seq: int,
    step_index: int,
    agent: str,
    step_cwd: str | None,
    root_cwd: str | None,
    subagent_aliases: Mapping[str, str] | None,
    wrapper_skills: Sequence[str],
    lsp_extensions: Mapping[str, str] | None = None,
) -> _Call:
    """A normalized tool call with its identities and outcome (its subagent fields are filled in later)."""
    fn = str(tool_call.get("function_name") or "")[:_MAX_LABEL_CHARS]
    args = tool_call.get("arguments")
    args = args if isinstance(args, dict) else {}
    mcp = _mcp_ident(fn, mcp_names, agent=agent)
    fn_base = _base_tool_name(mcp.tool if mcp is not None and mcp.tool else fn)
    manifest = _manifest_reads(fn_base, args, declared.get(COMPONENT_SKILL) or (), is_mcp=mcp is not None)
    named = {
        "subagent_aliases": subagent_aliases,
        "wrapper_skills": wrapper_skills,
        "agent": agent,
        "lsp_extensions": lsp_extensions,
    }
    unresolved = [(read, False) for read in manifest.reads]
    call = _Call(
        seq=seq,
        step_index=step_index,
        fn=fn,
        fn_base=fn_base,
        args=args,
        idents=_identities(fn, fn_base, mcp, args, declared, manifest=unresolved, **named),
        mcp=mcp,
        cwd=_call_cwd(args, step_cwd, root_cwd),
    )
    call.observation, call.succeeded = _outcome(correlated, shell=call.is_shell, content_read=call.is_content_read)
    if call.is_shell and manifest.reads:
        # Each SKILL.md read of a shell list has its own outcome: its own error line, or a
        # failed ``&&`` part before it (then it never ran).
        resolved = _resolve_manifest_reads(manifest, call.observation)
        if resolved != unresolved:
            call.idents = _identities(fn, fn_base, mcp, args, declared, manifest=resolved, **named)
    call.missing_tool = _says_missing_tool(call.observation, call.succeeded)
    call.opener = _window_opener(call, declared)
    if call.is_shell:
        call.side_idents = _side_channel_idents(fn_base, args, declared)
        if _writes_a_file(call):
            # Codex runs its file tool as a shell ``apply_patch``, so that call is ``apply_patch`` too.
            aliases = (*_SHELL_WRITE_ALIASES, "apply_patch") if _runs_apply_patch(call) else _SHELL_WRITE_ALIASES
            call.idents = [
                replace(ident, aliases=(*ident.aliases, *aliases)) if ident.kind is None else ident
                for ident in call.idents
            ]
    return call


def _extract_calls(
    trajectory: Mapping[str, Any],
    declared: Mapping[str, Sequence[str]],
    mcp_names: _McpNames,
    mcp_call_servers: Mapping[str, str] | None = None,
    subagent_aliases: Mapping[str, str] | None = None,
    wrapper_skills: Sequence[str] = (),
    lsp_extensions: Mapping[str, str] | None = None,
) -> list[_Call] | None:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
    agent = _trajectory_agent(trajectory)
    root_cwd = _trajectory_cwd(trajectory)
    named = {"subagent_aliases": subagent_aliases, "wrapper_skills": wrapper_skills, "lsp_extensions": lsp_extensions}
    calls: list[_Call] = []
    # The parent's latest subagent call: a sidechain run that follows it is that subagent's work,
    # unless the parent's call result names the subagent's ``agentId`` (see ``_spawned_agent_id``).
    last_spawn: list[_Ident] = []
    spawned_by: dict[str, list[_Ident]] = {}
    spawn_by_agent: dict[str, list[_Ident]] = {}
    runs = 0
    previous_sidechain = False
    for step_index, step in enumerate(steps[:_MAX_STEPS]):
        if not isinstance(step, Mapping):
            continue
        extra = step.get("extra")
        sidechain = isinstance(extra, Mapping) and extra.get("is_sidechain") is True
        chain = ""
        if sidechain:
            # Concurrent subagents are told apart by Harbor's ``agent_id``; without it, each run of
            # consecutive sidechain steps is one subagent.
            agent_id = extra.get("agent_id") if isinstance(extra, Mapping) else None
            if not previous_sidechain:
                runs += 1
            known = isinstance(agent_id, str) and bool(agent_id)
            chain = f"agent:{agent_id}"[:_MAX_LABEL_CHARS] if known else f"run:{runs}"
            spawner = spawn_by_agent.get(str(agent_id)[:_MAX_LABEL_CHARS]) if known else None
            spawned_by.setdefault(chain, list(last_spawn if spawner is None else spawner))
        previous_sidechain = sidechain
        raw_calls = _step_tool_calls(step)
        if not raw_calls:
            continue
        results = _observations(step)
        step_cwd = extra.get("cwd") if isinstance(extra, Mapping) and isinstance(extra.get("cwd"), str) else None
        for raw in raw_calls:
            outer_id = str(raw.get("tool_call_id") or raw.get("id") or "")
            for tool_call in _expanded_tool_calls(raw, outer_id, mcp_call_servers or {}):
                if len(calls) >= _MAX_CALLS:
                    return calls
                call = _identify_call(
                    tool_call,
                    _claimable_results(tool_call, results, outer_id, call_count=len(raw_calls)),
                    declared,
                    mcp_names,
                    seq=len(calls),
                    step_index=step_index,
                    agent=agent,
                    step_cwd=step_cwd,
                    root_cwd=root_cwd,
                    **named,
                )
                call.sidechain = sidechain
                call.chain = chain
                call.spawner = list(spawned_by.get(chain, [])) if sidechain else []
                if not sidechain and any(ident.kind == COMPONENT_SUBAGENT for ident in call.idents):
                    last_spawn = [ident for ident in call.idents if ident.kind == COMPONENT_SUBAGENT]
                    spawned_id = _spawned_agent_id(step, outer_id)
                    if spawned_id:
                        spawn_by_agent[spawned_id] = last_spawn
                calls.append(call)
    return calls


def _writes_a_file(call: _Call) -> bool:
    """Whether a shell call writes a file (a redirect, ``tee``, ``cp``, ``sed -i``, interpreter code...).

    Writes to devices such as ``/dev/null`` do not count.
    """
    return any(path and not path.startswith("/dev/") for path in call.shell_paths.writes)


def _runs_apply_patch(call: _Call) -> bool:
    """Whether a shell call runs ``apply_patch`` on a patch (``apply_patch <<'PATCH'``, as Codex does)."""
    return any(
        _APPLY_PATCH_HEADER_RE.search(text)
        and any(
            _APPLY_PATCH_COMMAND_RE.fullmatch(word.rsplit("/", 1)[-1])
            for command, _ in _shell_chain(text)
            for word in command
        )
        for text in _shell_texts(call.fn_base, call.args)
    )


_AGENT_ID_RE = re.compile(r'"agentId"\s*:\s*"([A-Za-z0-9_.:-]{1,128})"')
_METADATA_MARKER = "[metadata] "


def _spawned_agent_id(step: Mapping[str, Any], call_id: str) -> str | None:
    """The ``agentId`` of the subagent a parent ``Agent``/``Task`` call started, when its result says it.

    Claude Code returns the subagent's ``agentId`` in the call's result, and
    Harbor keeps it in ``tool_result_metadata.tool_use_result`` and in the
    ``[metadata]`` line of the result text. Sidechain steps carry the same id
    as ``agent_id``, so each subagent's calls are credited to the call that
    started it, even when the parent starts several subagents at once.
    """
    observation = step.get("observation")
    results = observation.get("results") if isinstance(observation, Mapping) else None
    if not call_id or not isinstance(results, list):
        return None
    for result in results[:_MAX_STEP_RESULTS]:
        if not isinstance(result, Mapping) or str(result.get("source_call_id") or "") != call_id:
            continue
        extra = result.get("extra")
        metadata = extra.get("tool_result_metadata") if isinstance(extra, Mapping) else None
        use = metadata.get("tool_use_result") if isinstance(metadata, Mapping) else None
        agent_id = use.get("agentId") if isinstance(use, Mapping) else None
        if isinstance(agent_id, str) and agent_id.strip():
            return agent_id.strip()[:_MAX_LABEL_CHARS]
        content = result.get("content")
        texts = [content] if isinstance(content, str) else _content_parts(content)
        for text in texts:
            at = text.rfind(_METADATA_MARKER) if isinstance(text, str) else -1
            if at < 0 or (at and text[at - 1] != "\n"):
                continue
            line = text[at : at + _MAX_OBSERVATION_CHARS].split("\n", 1)[0]
            match = _AGENT_ID_RE.search(line)
            if match:
                return match.group(1)[:_MAX_LABEL_CHARS]
    return None


def _trajectory_cwd(trajectory: Mapping[str, Any]) -> str | None:
    """The task's working directory: the agent's recorded cwd, else the first step's."""
    agent = trajectory.get("agent")
    extra = agent.get("extra") if isinstance(agent, Mapping) else None
    if isinstance(extra, Mapping):
        if isinstance(extra.get("cwd"), str) and extra["cwd"].startswith("/"):
            return _normalize_path(extra["cwd"])
        cwds = extra.get("cwds")
        if isinstance(cwds, list) and cwds and isinstance(cwds[0], str) and cwds[0].startswith("/"):
            return _normalize_path(cwds[0])
    for step in (trajectory.get("steps") or [])[:_MAX_STEPS]:
        extra = step.get("extra") if isinstance(step, Mapping) else None
        if isinstance(extra, Mapping) and isinstance(extra.get("cwd"), str) and extra["cwd"].startswith("/"):
            return _normalize_path(extra["cwd"])
    return None


def _call_cwd(args: Mapping[str, Any], step_cwd: str | None, root_cwd: str | None) -> str | None:
    for value in (args.get("workdir"), args.get("cwd"), step_cwd):
        if isinstance(value, str) and value.startswith("/"):
            return _normalize_path(value)
    return root_cwd


def _declared_component(ident: _Ident, declared: Mapping[str, Sequence[str]]) -> bool:
    folded = ident.name.casefold()
    return any(isinstance(name, str) and name.casefold() == folded for name in declared.get(ident.kind or "") or ())


def _window_opener(call: _Call, declared: Mapping[str, Sequence[str]]) -> _Ident | None:
    """The declared skill or command a call switched the agent to, or ``None``.

    Only a call that activated exactly one declared plugin skill or command,
    and did not fail, opens a window. A load of a skill this plugin does not
    declare (a built-in, a made-up ``mcp_tool``, another plugin's skill), the
    wrapper, a failed load, and one read of several ``SKILL.md`` files (which
    of them is the agent following?) neither open nor close a window.
    """
    if call.succeeded is False:
        return None
    openers = {
        ident.label: ident
        for ident in call.idents
        if ident.kind in {COMPONENT_SKILL, COMPONENT_COMMAND}
        and not ident.failed
        and not ident.wrapper
        and not ident.foreign
        and _declared_component(ident, declared)
    }
    return next(iter(openers.values())) if len(openers) == 1 else None


# A JSON-RPC MCP ``tools/call`` written into a shell command (quotes may be backslash-escaped).
_JSONRPC_TOOL_CALL_RE = re.compile(
    r'\\?"method\\?"\s*:\s*\\?"tools/call\\?"[^\n]{0,512}?\\?"name\\?"\s*:\s*\\?"([A-Za-z0-9_.:/-]{1,128})\\?"'
)
_SCRIPT_SUFFIXES = (".py", ".sh", ".bash", ".js", ".mjs", ".cjs", ".ts", ".rb", ".pl")


def _side_channel_idents(fn_base: str, args: Mapping[str, Any], declared: Mapping[str, Sequence[str]]) -> list[_Ident]:
    """Plugin components a shell call used without their own tool.

    * An MCP ``tools/call`` sent as JSON-RPC through the shell to a declared
      server's program (``printf '{..."method":"tools/call"...}' | /x/reltools``)
      is that server's tool; with one declared server, it is that server.
    * Running a file inside a member skill's folder (``python3
      <skills>/version-bump/scripts/bump.py``) uses that skill.
    """
    idents: list[_Ident] = []
    servers = [name for name in declared.get(COMPONENT_MCP) or () if isinstance(name, str) and name]
    members = [name for name in declared.get(COMPONENT_SKILL) or () if isinstance(name, str) and name]
    for text in _shell_texts(fn_base, args):
        chain = _shell_chain(text)
        tools = _JSONRPC_TOOL_CALL_RE.findall(text) if "tools/call" in text and servers else []
        if tools:
            words = {word.rsplit("/", 1)[-1].casefold() for command, _ in chain for word in command}
            named = [name for name in servers if words & set(_server_spellings(name))]
            server = named[0] if len(named) == 1 else (servers[0] if len(servers) == 1 else None)
            for tool in dict.fromkeys(tools):
                if server is not None:
                    label = f"mcp__{server}__{tool}"
                    idents.append(
                        _Ident(
                            label=label,
                            kind=COMPONENT_MCP,
                            name=server,
                            fn=label,
                            server=server,
                            tool=tool,
                            tool_label=f"shell:{label}",
                        )
                    )
        for command, _ in chain:
            programs = [command[0]]
            if command[0].rsplit("/", 1)[-1].casefold() in _INTERPRETERS | _SHELLS:
                # The script is the interpreter's first operand, unless inline code or a module comes first.
                for token in command[1:]:
                    if token in _INLINE_CODE_FLAGS or token == "-m":
                        break
                    if not token.startswith("-"):
                        programs.append(token)
                        break
            for program in programs:
                path = _normalize_path(program)
                if not path.casefold().endswith(_SCRIPT_SUFFIXES):
                    continue
                for member in members:
                    if f"/{member.casefold()}/" in f"/{path.casefold()}" and not path.casefold().endswith("skill.md"):
                        idents.append(
                            _component_ident(
                                COMPONENT_SKILL, member, fn_base, f"{fn_base}:skill-script", declared=declared
                            )
                        )
    return list({ident.label: ident for ident in idents}.values())


def _says_missing_tool(observation: str | None, succeeded: bool | None) -> bool:
    """Whether a failed result says the called tool (or skill, command, agent) does not exist."""
    if succeeded is not False or not observation:
        return False
    head = observation[:_FAILURE_SCAN_CHARS].casefold()
    return any(marker in head for marker in _MISSING_TOOL_MARKERS)


def _prompt_texts(trajectory: Mapping[str, Any]) -> list[str]:
    """The task prompt: the user and system messages before the agent's first step.

    Every harness puts the task (and context such as ``AGENTS.md``) in front of
    the agent's first step. A user or system step after that is not the task:
    Claude Code injects one after each ``Skill`` call that repeats the agent's
    own Skill arguments, and a subagent's prompt (a sidechain step) is written
    by the parent agent. Counting those as prompt would reject a value the
    agent really carried from the producer.
    """
    texts: list[str] = []
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return texts
    for step in steps[:_MAX_STEPS]:
        source = step.get("source") if isinstance(step, Mapping) else None
        if not isinstance(source, str):
            continue
        if source not in {"user", "system"}:
            break
        extra = step.get("extra")
        if isinstance(extra, Mapping) and extra.get("is_sidechain") is True:
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
    pattern: str  # a casefolded ``fnmatch`` glob

    @cached_property
    def glob(self) -> re.Pattern[str]:
        """The compiled glob, as :func:`fnmatch.fnmatchcase` compiles it."""
        return re.compile(fnmatch.translate(self.pattern))


def _parse_ref(raw: str) -> _Ref:
    text = raw.strip()
    prefix, sep, rest = text.partition(":")
    kind = _REF_PREFIXES.get(prefix.strip().casefold()) if sep else None
    if kind is not None and rest.strip():
        pattern = rest.strip().casefold()
        return _Ref(raw=text, kind=kind, pattern=pattern.lstrip("/") if kind == COMPONENT_COMMAND else pattern)
    return _Ref(raw=text, kind=None, pattern=text.casefold())


def _match_names(values: Iterable[str]) -> tuple[str, ...]:
    """The non-empty ``values``, casefolded: refs match case-insensitively."""
    return tuple(value.casefold() for value in values if value)


def _name_candidates(name: str) -> list[str]:
    low = name.casefold()
    return [low, low.rsplit(":", 1)[-1]] if ":" in low else [low]


def _ref_matches(ref: _Ref, ident: _Ident) -> bool:
    """Whether ``ref`` names ``ident`` (see the module docs).

    The generated wrapper skill is the harness's way in, not a component, so
    no ref matches it (not even a ``Skill:<plugin>*`` glob). An untyped ref
    matches a component by its label or name, never through the tool that
    carried it: the underlying tool (``Read``, ``Bash``...) is a separate
    plain identity of the same call. A harness's own agent called by a
    declared agent's bare name is not that agent, so no ref matches it either.
    """
    if ident.wrapper or ident.builtin:
        return False
    if ref.kind is None:
        names = ident.untyped_names
    elif ref.kind == ident.kind:
        names = ident.typed_names
    else:
        return False
    return any(ref.glob.match(name) for name in names)


def _mcp_ref_candidates(ident: _Ident) -> list[str]:
    """``<server>`` and ``<server>/<tool>``/``<server>__<tool>``, in the declared and the harness spelling.

    Claude Code writes the declared ``docs.v2`` as ``docs_v2``; both spellings
    match, alone or with a tool.
    """
    server = ident.server or ""
    spellings = list(dict.fromkeys((server.casefold(), _norm_server(server))))
    candidates = list(spellings)
    if ident.tool:
        tool = ident.tool.casefold()
        for spelling in spellings:
            candidates += [f"{spelling}/{tool}", f"{spelling}__{tool}"]
    return candidates


def _refs(values: Sequence[str]) -> list[_Ref]:
    return [_parse_ref(value) for value in values]


def _call_matches(refs: Sequence[_Ref], call: _Call) -> bool:
    return any(_ref_matches(ref, ident) for ref in refs for ident in call.idents)


def _ref_label(values: Sequence[str]) -> str:
    return _safe_text(" | ".join(values))


def _handoff_calls(
    producer: Sequence[_Ref], consumer: Sequence[_Ref], calls: Sequence[_Call]
) -> tuple[list[_Call], list[_Call]]:
    """``(producer calls, consumer calls)``: calls matching a side directly or running in its window.

    A window opens at a call that switched to a declared skill or command (see
    :func:`_window_opener`) and lasts until a call switches to another one.
    The consumer's own activation does not close the producer's window: the
    agent may look up the consumer's instructions before it saves the
    producer's output. Any other switch closes it; any switch, including to the
    producer, closes the consumer's. A subagent's own calls (sidechain steps)
    keep their own windows, start from the parent's, never change the parent's,
    and also belong to a side that names the subagent that started them.
    """
    states: dict[str, list[bool]] = {"": [False, False]}
    producer_calls: list[_Call] = []
    consumer_calls: list[_Call] = []
    for call in calls:
        # A subagent run starts from the parent's window at the time it first acts.
        state = states.setdefault(call.chain, list(states[""]))
        if call.opener is not None:
            is_producer = any(_ref_matches(ref, call.opener) for ref in producer)
            is_consumer = any(_ref_matches(ref, call.opener) for ref in consumer)
            state[0] = is_producer or (state[0] and is_consumer)
            state[1] = is_consumer
        spawned = [ident for ident in call.spawner if call.sidechain]
        if state[0] or _call_matches(producer, call) or any(_ref_matches(ref, i) for ref in producer for i in spawned):
            producer_calls.append(call)
        if state[1] or _call_matches(consumer, call) or any(_ref_matches(ref, i) for ref in consumer for i in spawned):
            consumer_calls.append(call)
    return producer_calls, consumer_calls


def _is_wrapper(ident: _Ident, wrapper_skills: Sequence[str], declared: Mapping[str, Sequence[str]]) -> bool:
    """Whether a skill identity is the generated wrapper, not a member skill.

    The wrapper package is ``<plugin>-plugin-eval``, and that name always
    means the wrapper. Its ``SKILL.md`` is named after the plugin, so the
    plugin name means the wrapper too, but only when no declared member skill
    or command has that name: a ``frontend-design`` plugin may ship a
    ``frontend-design`` skill, and that skill is a component. A ``SKILL.md``
    read names its folder, so only the package name marks a read as the wrapper.
    """
    if ident.kind != COMPONENT_SKILL:
        return False
    names = set(_name_candidates(ident.name))
    members = {
        name.casefold()
        for kind in (COMPONENT_SKILL, COMPONENT_COMMAND)
        for name in declared.get(kind) or ()
        if isinstance(name, str)
    }
    for wrapper in wrapper_skills:
        folded = wrapper.casefold()
        if folded not in names:
            continue
        if folded.endswith(_WRAPPER_PACKAGE_SUFFIX) or (folded not in members and not ident.via_read):
            return True
    return False


# =============================================================================
# Graders
# =============================================================================


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _unavailable_kinds(declared: Mapping[str, Sequence[str]], agent: str = "") -> frozenset[str]:
    """Component types this arm cannot carry: declared by the plugin but not staged here, or not on this harness."""
    kinds = {kind for kind in declared.get(DECLARED_UNSTAGED) or () if isinstance(kind, str)}
    if agent.casefold() in _NO_PLUGIN_AGENT_HARNESSES:
        kinds |= {COMPONENT_SUBAGENT, COMPONENT_COMMAND}
    return frozenset(kinds)


def _component_names(
    declared: Mapping[str, Sequence[str]],
    kinds: Iterable[str] = (COMPONENT_SKILL, COMPONENT_SUBAGENT, COMPONENT_COMMAND),
) -> list[str]:
    """Casefolded declared skill, subagent, and command names, bare and under the plugin namespace.

    The subagents and commands an arm does not declare (``DECLARED_UNSTAGED_NAMES``) count too, so an
    untyped ref that names one is a ref to that component in every arm.
    """
    namespaces = _plugin_namespaces(declared)
    hidden = [entry.partition(":") for entry in declared.get(DECLARED_UNSTAGED_NAMES) or () if isinstance(entry, str)]
    names: list[str] = []
    for kind in kinds:
        for name in (*(declared.get(kind) or ()), *(name for hidden_kind, _, name in hidden if hidden_kind == kind)):
            if isinstance(name, str) and name:
                names.append(name.casefold())
                names.extend(f"{namespace}:{name.casefold()}" for namespace in namespaces)
    return names


def _names_by_kind(declared: Mapping[str, Sequence[str]], unavailable: frozenset[str]) -> dict[str, list[str]]:
    """Declared skill, subagent, and command names per type (see :func:`_ref_unavailable`).

    Empty when the arm carries every type, since then no ref is skipped.
    """
    return {kind: _component_names(declared, [kind]) for kind in _ROUTING_KINDS} if unavailable else {}


def _ref_unavailable(ref: _Ref, unavailable: frozenset[str], names_by_kind: Mapping[str, Sequence[str]]) -> bool:
    """Whether ``ref`` names only component types this arm cannot carry.

    A typed ref names its own type. An untyped ref names the type of each
    declared skill, subagent, or command it matches (``names_by_kind``), so a
    bare ``reviewer`` that matches only a declared subagent is a subagent ref;
    one that matches none names a tool.
    """
    if ref.kind is not None:
        return ref.kind in unavailable
    kinds = {kind for kind, names in names_by_kind.items() if any(ref.glob.match(name) for name in names)}
    return bool(kinds) and kinds <= unavailable


def _routing_ref(ref: _Ref, component_names: Sequence[str]) -> bool:
    """Whether a ref is about routing (check 15) rather than tool selection (check 22).

    Typed ``Skill:``/``Agent:``/``Command:`` refs are routing and ``MCP:`` refs
    are tool selection. An untyped ref is routing when it names a declared
    skill, subagent, or command; otherwise it names a tool.
    """
    if ref.kind is not None:
        return ref.kind in _ROUTING_KINDS
    return any(ref.glob.match(name) for name in component_names)


def _grade_selection(
    calls: Sequence[_Call],
    spec: Mapping[str, Any],
    *,
    routing: bool,
    component_names: Sequence[str],
    unavailable: frozenset[str],
    declared: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    def _family(values: Sequence[str]) -> list[str]:
        return [value for value in values if _routing_ref(_parse_ref(value), component_names) == routing]

    expected = _family(spec.get("expected_tools", []))
    acceptable = _family(spec.get("acceptable_tools", []))
    decoys = _family(spec.get("decoy_tools", []))
    expected_refs, acceptable_refs, decoy_refs = _refs(expected), _refs(acceptable), _refs(decoys)
    # Expected refs to a component type this arm cannot carry are reported, not counted against recall.
    names_by_kind = _names_by_kind(declared, unavailable)
    missing = [_ref_unavailable(ref, unavailable, names_by_kind) for ref in expected_refs]
    applicable = [ref for ref, absent in zip(expected_refs, missing, strict=True) if not absent]
    skipped = [value for value, absent in zip(expected, missing, strict=True) if absent]
    allowed = [*expected_refs, *acceptable_refs]
    all_refs = [*allowed, *decoy_refs]

    def _in_family(ident: _Ident) -> bool:
        return (ident.kind in _ROUTING_KINDS) == routing and not ident.failed and not ident.wrapper

    # label -> (identity, credited): a call to a tool that does not exist is a wrong choice, never a hit.
    called: dict[str, list[Any]] = {}
    # Labels a decoy ref matched: a wrong choice even when an allowed ref matches them too, as a shell call
    # that wrote a file matches both an acceptable ``Bash`` and a decoy ``Write``.
    decoyed: set[str] = set()
    decoy_calls = 0
    for call in calls:
        idents = [ident for ident in call.idents if _in_family(ident)]
        matched = {ident.label for ident in idents if any(_ref_matches(ref, ident) for ref in decoy_refs)}
        if matched:
            decoy_calls += 1
            decoyed |= matched
        credited = not call.missing_tool
        for ident in idents:
            if ident.kind is None and not any(_ref_matches(ref, ident) for ref in all_refs):
                continue
            entry = called.get(ident.label)
            if entry is not None:
                entry[1] = entry[1] or credited
                if not set(ident.aliases) <= set(entry[0].aliases):
                    # One label, several calls: a shell call that wrote a file also answers to ``Write``.
                    entry[0] = replace(entry[0], aliases=tuple(dict.fromkeys((*entry[0].aliases, *ident.aliases))))
            elif len(called) < _MAX_CALLED:
                called[ident.label] = [ident, credited]

    precise = sum(
        1
        for label, (ident, credited) in called.items()
        if credited and label not in decoyed and any(_ref_matches(ref, ident) for ref in allowed)
    )
    # Precision is undefined when nothing in scope was called (a miss, not a wrong choice).
    precision = _ratio(precise, len(called)) if called else None
    recall: float | None = None
    f1: float | None = None
    if applicable:
        hits = sum(
            1 for ref in applicable if any(credited and _ref_matches(ref, ident) for ident, credited in called.values())
        )
        recall = _ratio(hits, len(applicable))
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = round(2 * precision * recall / (precision + recall), 4)
    scored = bool(applicable or acceptable or decoys)
    return {
        "expected": [_safe_text(item) for item in expected],
        "acceptable": [_safe_text(item) for item in acceptable],
        "decoys": [_safe_text(item) for item in decoys],
        "skipped": [_safe_text(item) for item in skipped],
        "called": [_safe_text(ident.persisted_label) for ident, _ in called.values()],
        "precision": precision if scored else None,
        "recall": recall,
        "f1": f1,
        "decoy_calls": decoy_calls,
        "status": STATUS_SCORED if scored else STATUS_NOT_APPLICABLE,
    }


def _grade_routing(
    calls: Sequence[_Call],
    spec: Mapping[str, Any],
    *,
    declared: Mapping[str, Sequence[str]] | None = None,
    agent: str = "",
) -> dict[str, Any]:
    """Check 15: precision/recall/F1 of skill, subagent, and command routing.

    Only routing refs count (typed ``Skill:``/``Agent:``/``Command:`` refs and
    untyped refs naming a declared component), and only skill, subagent, and
    command activations are routing choices, so MCP calls never dilute it. The
    wrapper skill, a ``SKILL.md`` read that failed by itself, and a call the
    harness says does not exist never earn credit; the last still counts as a
    wrong choice. ``precision`` is ``None`` when nothing was routed.
    """
    declared_map = declared or {}
    return _grade_selection(
        calls,
        spec,
        routing=True,
        component_names=_component_names(declared_map),
        unavailable=_unavailable_kinds(declared_map, agent),
        declared=declared_map,
    )


def _grade_tool_selection(
    calls: Sequence[_Call],
    spec: Mapping[str, Any],
    *,
    declared: Mapping[str, Sequence[str]] | None = None,
    agent: str = "",
) -> dict[str, Any]:
    """Check 22: precision/recall/F1 of MCP and plain tool selection.

    Only tool refs count (``MCP:`` refs and untyped refs that do not name a
    declared component); ``called`` holds every distinct MCP identity plus any
    plain tool a ref names, so ordinary unlisted tools (``Read``, ``Bash``...)
    do not dilute precision, and skills never do. A call to a tool that does
    not exist is a wrong choice. ``precision`` is ``None`` when nothing in
    scope was called; ``recall``/``f1`` are ``None`` when nothing applicable is
    expected. The wrapper skill is marked when calls are extracted.
    """
    declared_map = declared or {}
    return _grade_selection(
        calls,
        spec,
        routing=False,
        component_names=_component_names(declared_map),
        unavailable=_unavailable_kinds(declared_map, agent),
        declared=declared_map,
    )


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
    """Short redacted rendering of a dataset-authored value (never of agent arguments).

    Token shapes the general redactor does not know (GitHub ``ghp_``/``gho_``
    and ``github_pat_`` tokens, among others) are masked first.
    """
    from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line

    text = value if isinstance(value, str) else (_bounded_json_text(value, _MAX_ARGS_TEXT_CHARS) or _json_type(value))
    return _safe_text(redact_secrets_in_log_line(text[: _MAX_PREVIEW_CHARS * 4 + 256]), _MAX_PREVIEW_CHARS)


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
    except Exception:
        # A pattern the engine cannot run fails the check: besides regex.error and
        # ValueError it can raise KeyError, RuntimeError..., and grading never raises.
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
            if len(errors) >= _MAX_SCHEMA_ERRORS_PER_CALL:
                return
            if name not in value:
                errors.append((f"{path}.{name}", "required property is missing"))
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for name, sub in properties.items():
                if name in value and isinstance(sub, Mapping):
                    _schema_errors(value[name], sub, f"{path}.{name}", errors, depth + 1, budget)


def _string_values(value: Any, *, limit: int = _MAX_STRING_LEAVES) -> list[str]:
    """String values inside a JSON object or list (bounded, depth first), never key names."""
    found: list[str] = []
    stack = [value]
    while stack and len(found) < limit:
        current = stack.pop()
        if isinstance(current, str):
            found.append(current)
        elif isinstance(current, Mapping):
            stack.extend(reversed(list(current.values())[:limit]))
        elif isinstance(current, list):
            stack.extend(reversed(current[:limit]))
    return found


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
            found = any(
                item == needle or (isinstance(item, str) and needle in item) for item in actual[:_MAX_CONTAINS_ITEMS]
            )
        elif isinstance(actual, Mapping):
            # The object's string values, never its key names.
            found = any(needle in text for text in _string_values(actual))
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


#: Case-spec key the collector fills with the probed servers' tool input schemas
#: (``{server: {tool: schema}}``, from ``--probe-mcp``).
MCP_INPUT_SCHEMAS_KEY = "mcp_input_schemas"
_CALL_FAILED_DETAIL = "the call failed (the harness refused it or the server rejected it), so its arguments do not pass"


def _server_input_schemas(spec: Mapping[str, Any] | None) -> dict[str, dict[str, Mapping[str, Any]]]:
    raw = spec.get(MCP_INPUT_SCHEMAS_KEY) if isinstance(spec, Mapping) else None
    schemas: dict[str, dict[str, Mapping[str, Any]]] = {}
    for server, tools in list(raw.items())[:64] if isinstance(raw, Mapping) else ():
        if isinstance(server, str) and isinstance(tools, Mapping):
            kept = {
                tool: schema for tool, schema in tools.items() if isinstance(tool, str) and isinstance(schema, Mapping)
            }
            if kept:
                schemas[server] = kept
    return schemas


def _call_input_schema(
    call: _Call, schemas: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[str, Mapping[str, Any]] | None:
    """``(label, schema)`` when the server this MCP call went to published an input schema for its tool."""
    if call.mcp is None or not call.mcp.server:
        return None
    schema = schemas.get(call.mcp.server, {}).get(call.mcp.tool or "")
    return (call.mcp.persisted_label, schema) if schema is not None else None


def _grade_arguments(
    calls: Sequence[_Call], spec: Mapping[str, Any], schemas: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Check ``tool_arguments`` rules, and the servers' own input schemas, against every matching call.

    ``checked``/``passed`` count (call, rule) pairs. A rule that matched no
    call counts as one checked pair that did not pass, with a ``not_called``
    failure, so a rate can never be 100% while a rule's tool was never called.
    A call that failed (the harness refused it, or the server rejected it)
    never passes, with a ``call_failed`` failure. When the probe recorded a
    server's ``inputSchema`` for a tool (``schemas``, from ``MCP_INPUT_SCHEMAS_KEY``),
    every call to that tool is also checked against it, as an ``input_schema``
    pair. Pattern checks share one time budget; a check that runs out of time is
    a failure, never a pass.
    """
    rules = spec.get("tool_arguments", [])
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

    def _grade(label: str, call: _Call, call_failures: list[tuple[str, str, str]]) -> None:
        nonlocal checked, passed
        checked += 1
        if call.succeeded is False:
            call_failures = [*call_failures, ("", "call_failed", _CALL_FAILED_DETAIL)]
        if not call_failures:
            passed += 1
        for arg, rule_name, detail in call_failures:
            _record(label, arg, rule_name, detail)

    for rule in rules:
        refs = _refs([rule["tool"]])
        matched = [call for call in calls if _call_matches(refs, call)]
        if not matched:
            checked += 1
            _record(rule["tool"], "", "not_called", "no call matched this rule")
            continue
        for call in matched:
            label = next(
                (ident.persisted_label for ident in call.idents if any(_ref_matches(r, ident) for r in refs)),
                call.fn,
            )
            _grade(label, call, _argument_failures(call.args, rule, budget))
    schema_checked = False
    for call in calls if schemas else ():
        found = _call_input_schema(call, schemas)
        if found is None:
            continue
        label, schema = found
        schema_errors: list[tuple[str, str]] = []
        _schema_errors(call.args, schema, "$", schema_errors, 1, budget)
        _grade(
            label,
            call,
            [(path.removeprefix("$.") if path != "$" else "$", "input_schema", msg) for path, msg in schema_errors],
        )
        schema_checked = True
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "status": STATUS_SCORED if rules or schema_checked else STATUS_NOT_APPLICABLE,
    }


def top_argument_failures(signals: Iterable[Any], limit: int = 5) -> list[dict[str, Any]]:
    """The most frequent ``arguments.failures`` rows over per-trial signals, with exact counts.

    Every recorded row counts (a trial keeps at most 50), keyed by tool,
    argument, and rule; ties sort by tool and argument.
    """
    counts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in signals:
        arguments = item.get("arguments") if isinstance(item, Mapping) else None
        rows = arguments.get("failures") if isinstance(arguments, Mapping) else None
        for failure in rows[:_MAX_FAILURES] if isinstance(rows, list) else ():
            if not isinstance(failure, Mapping):
                continue
            key = (
                str(failure.get("tool") or "")[:200],
                str(failure.get("arg") or "")[:200],
                str(failure.get("rule") or "")[:200],
            )
            entry = counts.setdefault(
                key, {"tool": key[0], "arg": key[1], "rule": key[2], "detail": str(failure.get("detail") or "")[:300]}
            )
            entry["count"] = entry.get("count", 0) + 1
    return sorted(counts.values(), key=lambda row: (-row["count"], row["tool"], row["arg"]))[:limit]


_OUTCOME_KEYS = ("total", "succeeded", "failed", "unknown")


def _outcome_counts() -> dict[str, Any]:
    return dict.fromkeys(_OUTCOME_KEYS, 0)


def _count_outcome(bucket: dict[str, Any], succeeded: bool | None) -> None:
    bucket["total"] += 1
    if succeeded is True:
        bucket["succeeded"] += 1
    elif succeeded is False:
        bucket["failed"] += 1
    else:
        bucket["unknown"] += 1


def _with_success_rate(bucket: dict[str, Any]) -> dict[str, Any]:
    """``bucket`` with ``success_rate``, ``succeeded / (succeeded + failed)``: an unknown outcome counts as neither."""
    bucket["success_rate"] = _ratio(bucket["succeeded"], bucket["succeeded"] + bucket["failed"])
    return bucket


def _server_bucket(by_server: dict[str, dict[str, Any]], server: str) -> dict[str, Any]:
    """The outcome counts, tool labels and per-tool counts of ``server``, added empty the first time."""
    return by_server.setdefault(server, {**_outcome_counts(), "tools": [], "by_tool": {}})


def _add_server_tool(bucket: dict[str, Any], label: str) -> None:
    if label not in bucket["tools"] and len(bucket["tools"]) < _MAX_SERVER_TOOLS:
        bucket["tools"].append(label)


def _grade_mcp_calls(calls: Sequence[_Call]) -> dict[str, Any]:
    """Outcome counts across every MCP call (any server), with per-server and per-tool breakdowns.

    ``success_rate`` is ``succeeded / (succeeded + failed)``: calls whose outcome
    is unknown are reported separately rather than counted as either. Each
    server keeps its tool labels (``tools``) and the same counts per tool
    (``by_tool``, at most 64 tools).
    """
    totals = _outcome_counts()
    by_server: dict[str, dict[str, Any]] = {}
    for call in calls:
        if call.mcp is None:
            continue
        _count_outcome(totals, call.succeeded)
        bucket = _server_bucket(by_server, _safe_text(call.mcp.server or ""))
        _count_outcome(bucket, call.succeeded)
        label = _safe_text(call.mcp.label)
        _add_server_tool(bucket, label)
        if label in bucket["tools"]:
            _count_outcome(bucket["by_tool"].setdefault(label, _outcome_counts()), call.succeeded)
    for bucket in by_server.values():
        _with_success_rate(bucket)
        for tool_counts in bucket["by_tool"].values():
            _with_success_rate(tool_counts)
    return _with_success_rate({**totals, "by_server": by_server})


# Why an ``expected_order`` edge did not pass.
ORDER_NEVER_CALLED = "never_called"
ORDER_BEFORE_NEVER_CALLED = "before_never_called"
ORDER_AFTER_NEVER_CALLED = "after_never_called"
ORDER_SAME_CALL = "same_call"
ORDER_SAME_STEP = "same_step"
ORDER_REVERSED = "reversed"
_UNORDERED_REASONS = frozenset({ORDER_SAME_CALL, ORDER_SAME_STEP})


def _uses(refs: Sequence[_Ref], call: _Call) -> bool:
    """Whether ``call`` really used something ``refs`` names: a matching identity that did not fail.

    A shell side channel (an MCP tool called through the shell, a skill's
    script run directly) is a use too.
    """
    return any(
        call.ident_ok(ident) and _ref_matches(ref, ident) for ref in refs for ident in (*call.idents, *call.side_idents)
    )


def _attempts(refs: Sequence[_Ref], call: _Call) -> bool:
    """Whether ``call`` tried to use something ``refs`` names, whatever the outcome.

    A ``SKILL.md`` read that failed by itself is not an attempt (the file was
    not the component), and neither is one read of several ``SKILL.md`` files:
    looking through instructions together is not choosing one of them.
    """
    batched = sum(ident.via_read and not ident.failed for ident in call.idents) > 1
    return any(
        _ref_matches(ref, ident)
        for ref in refs
        for ident in (*call.idents, *call.side_idents)
        if not ident.failed and not (batched and ident.via_read)
    )


def _first_use(refs: Sequence[_Ref], calls: Sequence[_Call]) -> _Call | None:
    return next((call for call in calls if _uses(refs, call)), None)


def _side_unavailable(
    refs: Sequence[_Ref], unavailable: frozenset[str], names_by_kind: Mapping[str, Sequence[str]]
) -> bool:
    """Every alternative names a component type this arm cannot carry (see :func:`_ref_unavailable`)."""
    return bool(refs) and all(_ref_unavailable(ref, unavailable, names_by_kind) for ref in refs)


def _edge_reason(before: Sequence[_Ref], after: Sequence[_Ref], calls: Sequence[_Call]) -> str:
    """``""`` when the edge holds, else why not.

    The first use of ``before`` (any alternative) must come in an earlier step
    than the first use of some ``after`` alternative. Each ``after``
    alternative is checked on its own, so an early look at one alternative
    (reading its ``SKILL.md`` while planning) does not hide the real work done
    in order through another. Calls in one step were issued together (parallel
    calls), and reads in one shell command were loaded together, so neither
    has an order.
    """
    first_before = _first_use(before, calls)
    firsts = [_first_use([ref], calls) for ref in after]
    found = [call for call in firsts if call is not None]
    if first_before is None:
        return ORDER_NEVER_CALLED if not found else ORDER_BEFORE_NEVER_CALLED
    if not found:
        return ORDER_AFTER_NEVER_CALLED
    if any(call.step_index > first_before.step_index for call in found):
        return ""
    if any(call.seq == first_before.seq for call in found):
        return ORDER_SAME_CALL
    if any(call.step_index == first_before.step_index for call in found):
        return ORDER_SAME_STEP
    return ORDER_REVERSED


def _grade_order(
    calls: Sequence[_Call],
    spec: Mapping[str, Any],
    *,
    declared: Mapping[str, Sequence[str]] | None = None,
    agent: str = "",
) -> dict[str, Any]:
    """Check ``expected_order`` precedence edges by first use (see :func:`_edge_reason`).

    Failed calls and failed ``SKILL.md`` reads are not uses. Every edge that
    does not hold is listed in ``violated`` with its ``reason``
    (``never_called``, ``before_never_called``, ``after_never_called``,
    ``same_call``, ``same_step``, or ``reversed``); ``unordered`` counts the
    ``same_call``/``same_step`` ones. An edge one side of which names only a
    component type this arm cannot carry is listed in ``skipped`` and not
    counted in ``edges``.
    """
    unavailable = _unavailable_kinds(declared or {}, agent)
    names_by_kind = _names_by_kind(declared or {}, unavailable)
    satisfied = 0
    unordered = 0
    counted = 0
    violated: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for before, after in spec.get("expected_order", []):
        before_refs, after_refs = _refs(before), _refs(after)
        labels = {"before": _ref_label(before), "after": _ref_label(after)}
        if any(_side_unavailable(side, unavailable, names_by_kind) for side in (before_refs, after_refs)):
            skipped.append({**labels, "reason": "this arm cannot carry that component type"})
            continue
        counted += 1
        reason = _edge_reason(before_refs, after_refs, calls)
        if not reason:
            satisfied += 1
            continue
        unordered += reason in _UNORDERED_REASONS
        if len(violated) < MAX_ORDER_EDGES:
            violated.append({**labels, "reason": reason})
    return {
        "edges": counted,
        "satisfied": satisfied,
        "unordered": unordered,
        "violated": violated,
        "skipped": skipped,
        "status": STATUS_SCORED if counted else STATUS_NOT_APPLICABLE,
    }


_GLOB_CHARS = frozenset("*?[")
# MCP tools whose path argument is a destination they write or remove, so naming a path is not reading it.
# (``upload``/``export``/``put`` send a local file somewhere else: they do read it.)
_LOCAL_WRITE_VERB_RE = re.compile(
    r"(?:^|_|-)(?:write|create|save|dump|store|generate|render|append|overwrite"
    r"|delete|remove|rm|unlink|drop|erase|move|rename|trash)(?:$|_|-)"
)
_INPUT_ARG_KEYS = frozenset(
    {"path", "file", "file_path", "filepath", "filename", "input", "input_path", "input_file", "source", "src", "from"}
)


def _observed_path_matches(observed: str, artifact: str) -> bool:
    """``observed`` and ``artifact`` are resolved (:func:`_resolve_path`); ``observed`` may be a glob.

    Two absolute paths must be equal. A relative path is only left when its
    directory is unknown; then a path ending with ``/<relative>`` matches.
    """
    if not observed or not artifact:
        return False
    if _GLOB_CHARS & set(observed):
        return fnmatch.fnmatchcase(artifact, observed) or (
            not observed.startswith("/") and fnmatch.fnmatchcase(artifact, "*/" + observed)
        )
    if observed.startswith("/") and artifact.startswith("/"):
        return observed == artifact
    if artifact.startswith("/"):
        return artifact.endswith("/" + observed)
    return observed == artifact or observed.endswith("/" + artifact)


def _path_matches(observed: str, artifact: str, cwd: str | None = None) -> bool:
    """``artifact`` is resolved; ``observed`` is raw and resolved against ``cwd``."""
    return _observed_path_matches(_resolve_path(observed, cwd), artifact)


def _patch_written_paths(text: str) -> list[str]:
    """Files an apply_patch body writes: the ``Add File``, ``Update File`` and ``Move to`` headers.

    A file the patch only deletes is not written, and neither is a file it
    moves away: ``*** Update File: a`` followed by ``*** Move to: b`` writes
    only ``b``. A file the patch deletes and adds again is still written.
    """
    written: dict[str, None] = {}
    pending_update = ""  # an ``Update File`` path, written unless the next header moves it away
    for match in _APPLY_PATCH_HEADER_RE.finditer(text):
        operation = text[match.start() : match.start(1)]
        path = match.group(1).strip()
        if pending_update and "Move to" not in operation:
            written[pending_update] = None
        pending_update = ""
        if "Update File" in operation:
            pending_update = path
        elif "Delete File" not in operation and path:
            written[path] = None
    if pending_update:
        written[pending_update] = None
    return list(written)


def _patch_targets(args: Mapping[str, Any]) -> list[str]:
    """Paths an apply_patch tool call writes (see :func:`_patch_written_paths`)."""
    paths: list[str] = []
    for key in _PATCH_BODY_KEYS:
        body = args.get(key)
        if isinstance(body, str):
            paths.extend(_patch_written_paths(body[:_MAX_ARGS_TEXT_CHARS]))
    return paths


def _call_writes(call: _Call, artifact: str) -> bool:
    """Whether a call that did not fail wrote ``artifact`` (resolved)."""
    if call.succeeded is False:
        return False
    fn_base = call.fn_base
    if fn_base in _WRITE_TOOLS:
        if _is_editor_view(fn_base, call.args):
            return False
        if any(_path_matches(path, artifact, call.cwd) for path in _path_args(call.args)):
            return True
        if fn_base in _PATCH_TOOLS and any(
            _path_matches(path, artifact, call.cwd) for path in _patch_targets(call.args)
        ):
            return True
        # An MCP write tool (``write_file``...) may name its file under another key, such as
        # ``destination``: the MCP rules below still apply to it.
    elif call.is_shell:
        return any(_observed_path_matches(path, artifact) for path in call.shell_paths.writes)
    if call.mcp is None:
        return False
    tool = (call.mcp.tool or "").casefold()
    for key, value in call.args.items():
        if not isinstance(value, str) or not _path_matches(value, artifact, call.cwd):
            continue
        if _WRITE_VERB_RE.search(tool) or str(key).casefold() in _OUTPUT_ARG_KEYS:
            return True
    return False


def _call_reads(call: _Call, artifact: str) -> bool:
    """Whether a call that did not fail read ``artifact`` (resolved).

    An MCP tool reads it when the tool is not a write or delete tool and the
    path is an input argument (or the tool name says it reads). A subagent
    call that only names the path is not a read: the subagent's own reads are
    its sidechain calls.
    """
    if call.succeeded is False:
        return False
    fn_base = call.fn_base
    if (fn_base in _READ_TOOLS or _is_editor_view(fn_base, call.args)) and any(
        _path_matches(path, artifact, call.cwd) for path in _path_args(call.args)
    ):
        return True
    if call.is_shell:
        # A shell read whose own error line names the file did not read it (``cat: x: No such file``).
        name = artifact.rsplit("/", 1)[-1]
        return any(_observed_path_matches(path, artifact) for path in call.shell_paths.reads) and not _failed_operands(
            call.observation or "", {name}
        )
    if call.mcp is None:
        return False
    tool = (call.mcp.tool or "").casefold()
    if _LOCAL_WRITE_VERB_RE.search(tool):
        return False
    reads_by_name = bool(_READ_VERB_RE.search(tool))
    return any(
        isinstance(value, str)
        and (reads_by_name or str(key).casefold() in _INPUT_ARG_KEYS)
        and _path_matches(value, artifact, call.cwd)
        for key, value in call.args.items()
    )


def _value_in_text(value: str, text: str) -> bool:
    """``value`` occurs in ``text`` as a whole token: ``AT-48`` is not in ``AT-4821``.

    An occurrence counts unless a letter continues a letter, or a digit a
    digit, across its edge (``1.5.0`` is in ``v1.5.0``; ``abc`` is not in ``abcd``).
    """
    start = text.find(value)
    while start >= 0:
        end = start + len(value)
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if not (_continues(before, value[0]) or _continues(after, value[-1])):
            return True
        start = text.find(value, start + 1)
    return False


def _continues(neighbor: str, edge: str) -> bool:
    return bool(neighbor) and ((neighbor.isalpha() and edge.isalpha()) or (neighbor.isdigit() and edge.isdigit()))


def _decoded_texts(text: str, *, depth: int = 0) -> list[str]:
    """``text`` plus the string values of any JSON in it, recursively (MCP results are JSON in text)."""
    texts = [text]
    if depth >= 3 or len(text) > _MAX_OBSERVATION_CHARS:
        return texts
    start = min((index for index in (text.find("{"), text.find("[")) if index >= 0), default=-1)
    if start < 0:
        return texts
    try:
        decoded, _ = json.JSONDecoder().raw_decode(text[start:])
    except (ValueError, RecursionError):
        return texts
    for leaf in _string_values(decoded):
        texts.extend(_decoded_texts(leaf, depth=depth + 1))
    return texts


def _value_produced(value: str, call: _Call) -> bool:
    return call.succeeded is not False and any(_value_in_text(value, text) for text in call.result_texts)


def _value_received(value: str, call: _Call) -> bool:
    """The value is in the consumer's arguments, compared as parsed values (quotes and backslashes are data)."""
    return any(_value_in_text(value, text) for text in call.argument_texts)


def _only_batched(refs: Sequence[_Ref], calls: Sequence[_Call]) -> bool:
    """Whether ``refs`` were used, but never by a call that opened a window (one read of several SKILL.md files)."""
    used = [call for call in calls if _uses(refs, call)]
    return bool(used) and all(
        call.opener is None and sum(ident.via_read and not ident.failed for ident in call.idents) > 1 for call in used
    )


def _handoff_value_failure(
    value: str, producer_calls: Sequence[_Call], consumer_calls: Sequence[_Call], prompts: Sequence[str]
) -> str:
    if any(_value_in_text(value, text) for text in prompts):
        return "value also appears in the task prompt, so the handoff cannot be attributed"
    produced = next((call for call in producer_calls if _value_produced(value, call)), None)
    if produced is None:
        return "value was not observed in producer output"
    # A call in a later step is never the producing call itself.
    received = any(call.step_index > produced.step_index and _value_received(value, call) for call in consumer_calls)
    if not received:
        return "value did not reach consumer input after the producer produced it"
    return ""


def _handoff_artifact_failure(
    artifact: str, producer_calls: Sequence[_Call], consumer_calls: Sequence[_Call], root_cwd: str | None
) -> str:
    target = _resolve_path(artifact, root_cwd)
    written = next((call for call in producer_calls if _call_writes(call, target)), None)
    if written is None:
        return "artifact was not written by the producer"
    # Like a value, a file is only there for the consumer after the step that wrote it.
    if not any(call.step_index > written.step_index and _call_reads(call, target) for call in consumer_calls):
        return "artifact was not read by the consumer after the producer wrote it"
    return ""


def _grade_handoff(
    calls: Sequence[_Call],
    spec: Mapping[str, Any],
    *,
    prompts: Sequence[str] = (),
    root_cwd: str | None = None,
    declared: Mapping[str, Sequence[str]] | None = None,
    agent: str = "",
) -> dict[str, Any]:
    """Verify each ``handoffs[i]`` carried producer output into consumer input (see module docs).

    A handoff whose producer or consumer names only component types this arm
    cannot carry is listed in ``skipped`` and not checked.
    """
    unavailable = _unavailable_kinds(declared or {}, agent)
    names_by_kind = _names_by_kind(declared or {}, unavailable)
    failures: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    checked = 0
    passed = 0
    for handoff in spec.get("handoffs", []):
        producer_refs = _refs(handoff["producer"])
        consumer_refs = _refs(handoff["consumer"])
        if any(_side_unavailable(side, unavailable, names_by_kind) for side in (producer_refs, consumer_refs)):
            skipped.append(
                {
                    "producer": _ref_label(handoff["producer"]),
                    "consumer": _ref_label(handoff["consumer"]),
                    "reason": "this arm cannot carry that component type",
                }
            )
            continue
        checked += 1
        producer_calls, consumer_calls = _handoff_calls(producer_refs, consumer_refs, calls)
        problems: list[str] = []
        if not any(_uses(producer_refs, call) for call in calls):
            problems.append("producer was not activated")
        if not any(_uses(consumer_refs, call) for call in calls):
            problems.append("consumer was not activated")
        if not problems:
            if handoff.get("value") is not None:
                problem = _handoff_value_failure(handoff["value"], producer_calls, consumer_calls, prompts)
                if problem:
                    problems.append(problem)
            if handoff.get("artifact") is not None:
                problem = _handoff_artifact_failure(handoff["artifact"], producer_calls, consumer_calls, root_cwd)
                if problem:
                    problems.append(problem)
            if problems and _only_batched(producer_refs, calls):
                problems.append(
                    "the producer was only read together with other skills in one call, so no later call is its"
                )
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
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "skipped": skipped,
        "status": STATUS_SCORED if checked else STATUS_NOT_APPLICABLE,
    }


def _grade_conflict(
    calls: Sequence[_Call],
    spec: Mapping[str, Any],
    *,
    declared: Mapping[str, Sequence[str]] | None = None,
    agent: str = "",
) -> dict[str, Any]:
    """Each probe passes when ``must_use`` was used and ``must_not_use`` was not tried.

    ``must_use`` needs a use that did not fail (a failed call or a missing
    tool is not a use); ``must_not_use`` counts any attempt, except a failed or
    batched ``SKILL.md`` read (see :func:`_attempts`). Shell side channels
    count on both sides. A probe whose ``must_use`` names only component types
    this arm cannot carry is listed in ``skipped`` and not checked. The check
    is about presence over the whole trial, not which one ran first.
    """
    unavailable = _unavailable_kinds(declared or {}, agent)
    names_by_kind = _names_by_kind(declared or {}, unavailable)
    failures: list[dict[str, str]] = []
    skipped: list[str] = []
    checked = 0
    passed = 0
    for probe in spec.get("conflict_probes", []):
        must_use, must_not_use = _refs(probe["must_use"]), _refs(probe["must_not_use"])
        if _side_unavailable(must_use, unavailable, names_by_kind):
            skipped.append(_safe_text(probe["id"]))
            continue
        checked += 1
        used = any(_uses(must_use, call) and not call.missing_tool for call in calls)
        forbidden = any(_attempts(must_not_use, call) for call in calls)
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
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "skipped": skipped,
        "status": STATUS_SCORED if checked else STATUS_NOT_APPLICABLE,
    }


def _declared_keys(declared: Mapping[str, Sequence[str]] | None) -> list[tuple[str, str]]:
    keys: dict[tuple[str, str], None] = {}
    for kind in (COMPONENT_SKILL, COMPONENT_MCP, COMPONENT_SUBAGENT, COMPONENT_COMMAND, COMPONENT_LSP):
        for name in (declared or {}).get(kind) or ():
            if isinstance(name, str) and name:
                keys[kind, name] = None
    return list(keys)


def _ident_components(
    ident: _Ident, mcp_names: _McpNames, by_folded_name: Mapping[tuple[str, str], Sequence[str]]
) -> set[tuple[str, str]]:
    """The declared ``(type, name)`` components that ``ident`` activates (the wrapper skill activates none)."""
    if ident.wrapper or ident.builtin:
        return set()
    if ident.kind == COMPONENT_MCP:
        # ``_McpNames.identity`` already maps a recognizable server to its declared name.
        server = mcp_names.match(ident.server or "")
        return {(COMPONENT_MCP, server)} if server is not None else set()
    # Another plugin's ``<plugin>:<name>`` only answers to its full name, never to this plugin's.
    kind = ident.kind or ""
    return {(kind, name) for candidate in ident.name_candidates for name in by_folded_name.get((kind, candidate), ())}


def _grade_activation_coverage(
    calls: Sequence[_Call], declared: Mapping[str, Sequence[str]], mcp_names: _McpNames
) -> dict[str, Any]:
    """Declared components exercised, never activated, or whose every activation failed.

    Entries are ``"<type>:<name>"``. The three outcome lists do not overlap: a
    component whose every activation failed (a failed call, or a ``SKILL.md``
    read whose own error line says it failed) is ``unavailable``, not
    ``exercised``. ``mcp_names`` indexes the declared MCP servers.
    """
    keys = _declared_keys(declared)
    # A skill, subagent or command activation names its component case-insensitively.
    by_folded_name: dict[tuple[str, str], list[str]] = {}
    for kind, name in keys:
        if kind != COMPONENT_MCP:
            by_folded_name.setdefault((kind, name.casefold()), []).append(name)
    activated: set[tuple[str, str]] = set()
    worked: set[tuple[str, str]] = set()  # activated by at least one identity that did not fail
    for call in calls:
        for ident in call.component_idents:
            components = _ident_components(ident, mcp_names, by_folded_name)
            activated |= components
            if call.ident_ok(ident):
                worked |= components
    declared_labels: list[str] = []
    exercised: list[str] = []
    unverified: list[str] = []
    unavailable: list[str] = []
    for kind, name in keys:
        label = _safe_text(f"{kind}:{name}")
        declared_labels.append(label)
        if (kind, name) not in activated:
            unverified.append(label)
        elif (kind, name) in worked:
            exercised.append(label)
        else:
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
                    "succeeded": False if ident.failed else call.succeeded,
                }
            )
    return activations


def _with_plugin_names(
    declared: Mapping[str, Sequence[str]], wrapper_skills: Sequence[str]
) -> Mapping[str, Sequence[str]]:
    """``declared`` with the plugin name, taken from the wrapper package name when not given."""
    if declared.get(DECLARED_PLUGIN):
        return declared
    names = _plugin_names_from_wrappers(wrapper_skills)
    return {**declared, DECLARED_PLUGIN: names} if names else declared


def compute_plugin_signals(
    trajectory: Any,
    case: Mapping[str, Any] | None = None,
    *,
    declared: Mapping[str, Sequence[str]] | None = None,
    wrapper_skills: Sequence[str] = (),
    mcp_call_servers: Mapping[str, str] | None = None,
    subagent_aliases: Mapping[str, str] | None = None,
    lsp_servers: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any] | None:
    """Per-trial C3 ``plugin_signals`` for one ATIF trajectory, or ``None`` if unreadable.

    ``case`` is a dataset case entry or a :func:`plugin_case_spec` result; it is
    normalized once here, and invalid fields are ignored. Its
    ``MCP_INPUT_SCHEMAS_KEY`` holds the probed servers' tool input schemas.
    ``declared`` maps ``skill``/``mcp`` (and, in the with-plugin
    arm, ``subagent``/``command``) to the declared component names in this arm.
    ``mcp_call_servers`` maps a tool call id to the MCP server the harness log
    says it went to, for harnesses whose trajectory keeps only the bare tool
    name (Codex). ``subagent_aliases`` maps a staged subagent name (casefolded)
    to the declared name, for harnesses that rename a plugin agent (OpenCode).
    ``lsp_servers`` maps each LSP server staged in this arm (also listed under
    ``declared['lsp']``) to the file extensions its ``extensionToLanguage`` maps.
    """
    if not isinstance(trajectory, Mapping):
        return None
    declared_map = _with_plugin_names(declared or {}, wrapper_skills)
    mcp_names = _McpNames(declared_map.get(COMPONENT_MCP) or ())
    calls = _extract_calls(
        trajectory,
        declared_map,
        mcp_names,
        mcp_call_servers,
        subagent_aliases,
        wrapper_skills,
        lsp_extensions=_lsp_extension_index(lsp_servers, declared_map),
    )
    if calls is None:
        return None
    spec = plugin_case_spec(case)
    # The prompts only matter to rule out a handoff value the consumer could have copied from them.
    has_value = any(handoff.get("value") is not None for handoff in spec.get("handoffs", []))
    agent = _trajectory_agent(trajectory)
    return {
        "activations": _activations(calls),
        "routing": _grade_routing(calls, spec, declared=declared_map, agent=agent),
        "tool_selection": _grade_tool_selection(calls, spec, declared=declared_map, agent=agent),
        "arguments": _grade_arguments(calls, spec, _server_input_schemas(case)),
        "mcp_calls": _grade_mcp_calls(calls),
        "order": _grade_order(calls, spec, declared=declared_map, agent=agent),
        "handoff": _grade_handoff(
            calls,
            spec,
            prompts=_prompt_texts(trajectory) if has_value else (),
            root_cwd=_trajectory_cwd(trajectory),
            declared=declared_map,
            agent=agent,
        ),
        "conflict": _grade_conflict(calls, spec, declared=declared_map, agent=agent),
        "activation_coverage": _grade_activation_coverage(calls, declared_map, mcp_names),
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


def _summarize_selection(signals: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    """Mean precision/recall/F1 over the trials that have a number; ``decoy_call_rate`` is a share of trials."""
    blocks = _scored(signals, key)

    def _values(name: str) -> list[float]:
        return [float(block[name]) for block in blocks if _is_number(block.get(name))]

    return {
        "n_scored": len(blocks),
        "precision": _mean(_values("precision")),
        "recall": _mean(_values("recall")),
        "f1": _mean(_values("f1")),
        "decoy_calls": sum(_int(block.get("decoy_calls")) for block in blocks),
        # The share of scored trials with at least one decoy call (not a share of calls).
        "decoy_call_rate": _ratio(sum(1 for block in blocks if _int(block.get("decoy_calls")) > 0), len(blocks)),
        "status": STATUS_SCORED if blocks else STATUS_NOT_APPLICABLE,
    }


_MAX_SUMMARY_EDGES = 20


def _skipped(signals: Sequence[Mapping[str, Any]], key: str) -> int:
    """How many items the trials skipped because this arm cannot carry their component type (scored or not)."""
    return sum(
        len(skipped)
        for item in signals
        if isinstance(block := item.get(key), Mapping) and isinstance(skipped := block.get("skipped"), list)
    )


def _summarize_order(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Edge totals, trials with every edge in order, and the most common edges that did not hold, with why."""
    blocks = _scored(signals, "order")
    edges = sum(_int(block.get("edges")) for block in blocks)
    satisfied = sum(_int(block.get("satisfied")) for block in blocks)
    counts: dict[tuple[str, str, str], int] = {}
    for block in blocks:
        for item in block.get("violated") or ():
            if isinstance(item, Mapping):
                key = (str(item.get("before") or ""), str(item.get("after") or ""), str(item.get("reason") or ""))
                counts[key] = counts.get(key, 0) + 1
    top = sorted(counts.items(), key=lambda pair: -pair[1])[:_MAX_SUMMARY_EDGES]
    return {
        "n_scored": len(blocks),
        "edges": edges,
        "satisfied": satisfied,
        "unordered": sum(_int(block.get("unordered")) for block in blocks),
        "satisfaction_rate": _ratio(satisfied, edges),
        # The unit above is edges; this is the trial-level picture.
        "trials_in_order": sum(
            1 for block in blocks if _int(block.get("edges")) and block.get("satisfied") == block.get("edges")
        ),
        "violated_edges": [
            {"before": before, "after": after, "reason": reason, "trials": count}
            for (before, after, reason), count in top
        ],
        "skipped": _skipped(signals, "order"),
        "status": STATUS_SCORED if blocks else STATUS_NOT_APPLICABLE,
    }


def _summarize_conflict(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pooled probe pass rate, plus which probes failed in how many trials (the per-probe view)."""
    summary = _summarize_checked(signals, "conflict")
    counts: dict[str, int] = {}
    for block in _scored(signals, "conflict"):
        for item in block.get("failures") or ():
            if isinstance(item, Mapping) and isinstance(item.get("probe"), str):
                counts[item["probe"]] = counts.get(item["probe"], 0) + 1
    summary["failed_probes"] = [
        {"probe": probe, "trials": count}
        for probe, count in sorted(counts.items(), key=lambda pair: -pair[1])[:_MAX_SUMMARY_EDGES]
    ]
    summary["skipped"] = _skipped(signals, "conflict")
    return summary


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
        for key in _OUTCOME_KEYS:
            totals[key] += _int(block.get(key))
        servers = block.get("by_server")
        if not isinstance(servers, Mapping):
            continue
        for server, counts in servers.items():
            if not isinstance(counts, Mapping):
                continue
            bucket = _server_bucket(by_server, str(server))
            for key in _OUTCOME_KEYS:
                bucket[key] += _int(counts.get(key))
            for tool in counts.get("tools") or ():
                if isinstance(tool, str):
                    _add_server_tool(bucket, tool)
            tools = counts.get("by_tool")
            for tool, tool_counts in tools.items() if isinstance(tools, Mapping) else ():
                if not isinstance(tool_counts, Mapping) or tool not in bucket["tools"]:
                    continue
                tool_bucket = bucket["by_tool"].setdefault(tool, _outcome_counts())
                for key in _OUTCOME_KEYS:
                    tool_bucket[key] += _int(tool_counts.get(key))
    for bucket in by_server.values():
        _with_success_rate(bucket)
        for tool_counts in bucket["by_tool"].values():
            _with_success_rate(tool_counts)
    return _with_success_rate({**totals, "by_server": by_server})


def _summarize_coverage(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Union of per-trial activation coverage.

    A component is ``exercised`` when it was exercised in at least one trial,
    ``unavailable`` when it was tried but every try failed in every trial, and
    ``unverified`` otherwise. ``exercise_rate`` is the share of trials that
    exercised it. A trial block that lists a component as both exercised and
    unavailable (the older shape) counts as unavailable for that trial.
    """
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
        unavailable = {label for label in block.get("unavailable") or () if isinstance(label, str)}
        for label in unavailable:
            unavailable_counts[label] = unavailable_counts.get(label, 0) + 1
        for label in dict.fromkeys(block.get("exercised") or ()):
            if isinstance(label, str) and label not in unavailable:
                exercised_counts[label] = exercised_counts.get(label, 0) + 1
    n = len(signals)
    return {
        "declared": declared,
        "exercised": [label for label in declared if exercised_counts.get(label)],
        "unverified": [
            label for label in declared if not exercised_counts.get(label) and not unavailable_counts.get(label)
        ],
        "unavailable": [
            label for label in declared if unavailable_counts.get(label) and not exercised_counts.get(label)
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

    return {
        "n_trials": n_trials,
        "n_missing_trajectory": len(signals) - n_trials,
        "activations": {
            "total": total_activations,
            "mean_per_trial": _ratio(total_activations, n_trials),
            "by_type": activations,
        },
        "routing": _summarize_selection(present, "routing"),
        "tool_selection": _summarize_selection(present, "tool_selection"),
        "arguments": _summarize_checked(present, "arguments"),
        "mcp_calls": _summarize_mcp(present),
        "order": _summarize_order(present),
        "handoff": {**_summarize_checked(present, "handoff"), "skipped": _skipped(present, "handoff")},
        "conflict": _summarize_conflict(present),
        "activation_coverage": _summarize_coverage(present),
    }


__all__ = [
    "ARM_SUM_OF_PARTS",
    "ARM_WITHOUT_SKILL",
    "ARM_WITH_SKILL",
    "COMPONENT_COMMAND",
    "COMPONENT_LSP",
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
    "declared_mcp_server_matcher",
    "match_declared_mcp_server",
    "plugin_case_spec",
    "summarize_plugin_signals",
    "validate_plugin_case_fields",
]
