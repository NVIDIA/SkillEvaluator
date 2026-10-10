# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static privilege and hook risk analysis for plugin agents, commands, and hooks.

Everything here inspects plugin-controlled data (already read through the
bounded, no-follow plugin-root reader) and never runs a hook, an agent, or a
command. It answers two questions:

* **What does each subagent, command, and skill grant?** Agent frontmatter
  (``tools``, ``disallowedTools``, ``model``, ``permissionMode``) and command and
  skill frontmatter (``allowed-tools``, ``model``, ``disable-model-invocation``)
  are recorded per component. Pre-approved unrestricted or interpreter ``Bash``
  grants, wildcard ``*`` tools, permissive permission modes, and agents that
  inherit every tool next to write-capable MCP servers become findings.
* **What can each hook do?** ``hooks.json``, inline hook configs, and skill or
  command frontmatter hooks are parsed per the Claude Code hooks reference (event
  -> matcher groups -> handlers of type ``command``, ``http``, ``mcp_tool``,
  ``prompt``, or ``agent``), with each client's own event names, approval output,
  and command keys (:class:`HookDialect`: Codex, Cursor, GitHub Copilot). Each handler is
  recorded with its event, matcher, handler type, and risk flags; the risky ones
  (auto-approval, context injection, fetch-and-execute of remote code, unpinned
  package runners, HTTP endpoints, and files outside the plugin root) become
  findings. Plugin monitor commands get the same command checks. Every scan is
  bounded, and a bound that leaves a handler or a hook script unanalyzed is itself
  a finding, so padding cannot hide a hook.

Findings use the ``PLUGIN_SCHEMA`` category, so a policy overlay can change any
severity with ``severity_overrides`` (``PLUGIN_SCHEMA.<check>``).
"""

from __future__ import annotations

import base64
import binascii
import bisect
import functools
import posixpath
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, NamedTuple
from urllib.parse import unquote, urlparse, urlsplit

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CURSOR_MANIFEST_TYPE,
)
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_paths import PLUGIN_CATEGORY
from skillevaluator.validators.mcp_static import (
    TRUTHY_VALUES,
    EndpointClass,
    HostAllowlist,
    OverrideIssue,
    _split_words,
    classify_endpoint_host,
    classify_mcp_pinning,
    iter_config_strings,
    shell_program,
    unwrap_launch_command,
)
from skillevaluator.validators.url_policy import (
    DEFAULT_PORTS,
    is_env_reference,
    looks_like_inline_secret,
    report_text,
    safe_url,
    url_ambiguities,
    url_credentials,
    whatwg_url,
)

# Report text: whitespace collapsed, URL userinfo removed, credentials redacted, length bounded.
_bounded = report_text

# Documented Claude Code hook events (hooks reference). Unknown events are still
# recorded, with the ``unknown_event`` flag, because newer releases add events.
HOOK_EVENTS: frozenset[str] = frozenset(
    {
        "SessionStart",
        "SessionEnd",
        "Setup",
        "UserPromptSubmit",
        "UserPromptExpansion",
        "Stop",
        "StopFailure",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "PermissionDenied",
        "PostToolBatch",
        "PreModelSwitch",
        "PostModelSwitch",
        "SubagentStart",
        "SubagentStop",
        "TaskCreated",
        "TaskCompleted",
        "TeammateIdle",
        "WorktreeCreate",
        "WorktreeRemove",
        "CwdChanged",
        "DirectoryAdded",
        "FileChanged",
        "ConfigChange",
        "InstructionsLoaded",
        "PreCompact",
        "PostCompact",
        "Notification",
        "MessageDisplay",
        "Elicitation",
        "ElicitationResult",
    }
)
HANDLER_TYPES: tuple[str, ...] = ("command", "http", "mcp_tool", "prompt", "agent")
# Events whose hook output can approve a tool call or a permission prompt.
APPROVAL_EVENTS: frozenset[str] = frozenset({"PreToolUse", "PermissionRequest"})
# Events whose command stdout (or ``additionalContext``) is injected into the conversation.
CONTEXT_EVENTS: frozenset[str] = frozenset(
    {"UserPromptSubmit", "UserPromptExpansion", "SessionStart", "PostModelSwitch"}
)
# Handler types Claude Code runs on each context event; SessionStart runs only command and mcp_tool handlers.
_CONTEXT_HANDLER_TYPES: frozenset[str] = frozenset({"command", "http", "mcp_tool"})
_SESSION_START_HANDLER_TYPES: frozenset[str] = frozenset({"command", "mcp_tool"})
# Events whose 'decision: block' reason reaches the model as text: Stop and SubagentStop hooks tell it how to
# continue, and Claude Code 2.1.x adds a PostToolUse or PostToolUseFailure block reason to the context as a
# "hook blocking error" message, the same way it adds additionalContext. A PreToolUse block is a guard's denial.
_BLOCK_REASON_EVENTS: frozenset[str] = frozenset(
    {"Stop", "SubagentStop", "PostToolUse", "PostToolUseFailure", "stop", "subagentStop", "agentStop"}
)

MAX_HOOK_EVENTS = 64
MAX_HOOK_GROUPS = 256
MAX_HOOK_HANDLERS = 512
# A hook script is read and scanned whole up to the plugin reader's per-file limit; a larger
# script cannot be read and is reported as unanalyzed (never scanned in part).
MAX_SCRIPT_BYTES = CONTENT_DEDUP_MAX_FILE_BYTES
MAX_SCRIPT_READS = 32
# File names a script mentions that are checked as possible second-level scripts; more is a finding.
MAX_SCRIPT_REFERENCES = 256
# Distinct files one shell text runs that are matched against downloads; past it the hook is unanalyzed.
MAX_RUN_SITES = 2048
MAX_TOOL_ENTRIES = 256
MAX_MATCHER_CHARS = 256
MAX_OUTSIDE_REFS = 5
# Tool grants, allow rules, or server names one finding message quotes (the finding covers them all).
MAX_QUOTED_ENTRIES = 8

_WILDCARD_TOOLS = frozenset({"*", "*(*)", "mcp__*", "mcp__*__*"})
# Claude Code reads a matcher of only these characters as an exact tool name or '|' list, not a regex.
_EXACT_MATCHER_RE = re.compile(r"[A-Za-z0-9_|]+")
_MATCHER_NAME_SEPARATOR_RE = re.compile(r"[|,]")
# A looser name list ('Edit, Bash'): one that names Bash is treated as matching Bash.
_NAME_LIST_RE = re.compile(r"[A-Za-z0-9_\- ,|]+")
# A regex matcher that matches a shell tool and all of these other tool names is treated as "every tool".
_NON_SHELL_SAMPLE_TOOLS: tuple[str, ...] = ("Read", "Write", "Edit", "WebFetch", "mcp__server__tool")
# Tools whose auto-approval skips a prompt that guards writes, network fetches, or MCP side effects. Codex's
# file-edit tool is apply_patch (its hooks also accept Edit and Write as aliases); Codex reads a Claude Code
# plugin's hooks/hooks.json too, so every dialect counts it.
_SENSITIVE_TOOLS: tuple[str, ...] = (
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "apply_patch",
    "mcp__server__tool",
)

# Bash permission rules, classified the way Claude Code 2.1.x classifies its own allow rules:
# empty or only-'*' content is a bare grant, and an interpreter or wrapper prefix
# ("dangerous_prefix") runs any command.
_TOOL_RULE_RE = re.compile(r"\s*([A-Za-z_][\w-]*)\s*(?:\((.*)\))?\s*", re.DOTALL)
_DANGEROUS_BASH_PREFIXES: tuple[str, ...] = (
    "python",
    "python3",
    "python2",
    "node",
    "deno",
    "tsx",
    "ruby",
    "perl",
    "php",
    "lua",
    "npx",
    "bunx",
    "npm run",
    "yarn run",
    "pnpm run",
    "bun run",
    "bash",
    "sh",
    "ssh",
    "zsh",
    "fish",
    "eval",
    "exec",
    "env",
    "xargs",
    "sudo",
)
# 'python -m pkg.module *' is the one option-prefixed interpreter rule Claude Code treats as scoped.
_PYTHON_DOTTED_MODULE_RE = re.compile(r"-m\s+\w+\.[\w.]+(?:\s*:|\s+)")
_ENV_REF_ANYWHERE_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*")
# The auth scheme in front of a header value's literal part ('Bearer ', 'Basic ', 'token ').
_AUTH_SCHEME_PREFIX_RE = re.compile(r"(?i)^\s*(bearer|basic|token)\s*")
_URL_IN_TEXT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]{0,31}://[^\s'\"`<>]{1,2048}")

# Hook output that approves a tool call or a permission request: Claude Code's and Codex's shapes
# ("permissionDecision": "allow", "decision": "approve", "behavior": "allow").
_ALLOW_DECISION_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"permissionDecision\W{0,8}allow\b"),
    re.compile(r"\bdecision\W{0,8}approve\b"),
    re.compile(r"\bbehavior\W{0,8}allow\b"),
)
# Cursor's approval output: {"permission": "allow"} (beforeShellExecution, beforeMCPExecution,
# beforeReadFile) and {"decision": "allow"} (preToolUse, subagentStart).
_CURSOR_ALLOW_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpermission\W{0,8}allow\b"),
    re.compile(r"\bdecision\W{0,8}allow\b"),
)
_ALLOW_SHAPES: dict[str, tuple[re.Pattern[str], ...]] = {"claude": _ALLOW_DECISION_RES, "cursor": _CURSOR_ALLOW_RES}
# JSON hook output that puts text in front of the model on any event: additionalContext, a replaced tool
# output, or (on Stop and SubagentStop) a block decision whose reason tells the model how to continue.
_CONTEXT_OUTPUT_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("additionalContext", re.compile(r"\badditionalContext\b")),
    ("an updated tool output", re.compile(r"\bupdated(?:MCP)?ToolOutput\b")),
    ("a block reason", re.compile(r"\bdecision\W{0,8}block\b")),
)
# A base64 run long enough to hide a JSON decision ('{"permissionDecision":"allow"}' is 40 characters).
_BASE64_RUN_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,4096}={0,2}(?![A-Za-z0-9+/=])")
# A hook that emits a base64-hidden decision must decode it when it runs (base64, b64decode, atob), so only
# such text has its base64 runs decoded.
_MAX_BASE64_RUNS = 64
# JSON ("\u0061") and shell ("\x61") character escapes, decoded before the allow patterns run. A match only
# starts at the first backslash of a run, so a long run of backslashes with no escape after it stays linear.
_CHAR_ESCAPE_RE = re.compile(r"(?<!\\)\\++(?:u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2}))")
# Remote-code detection works on shell text (hook commands and the scripts they
# run). Every pattern is linear: commands are split first, and each regex has
# bounded or possessive spans, so a long or padded script cannot make a scan
# quadratic. Downloads and run sites are matched by set lookups (a run path's
# leading directories against the unpack directories), never pair by pair.
_FETCHERS = r"(?:curl|wget|fetch|aria2c|iwr|irm|invoke-webrequest|invoke-restmethod|downloadstring)"
_EXECUTORS = r"(?:(?:ba|z|da|k|fi|a)?sh|python[0-9.]{0,8}|node|perl|ruby|php|pwsh|powershell|iex|invoke-expression)"
_PATH_CHARS = r"\w.~$+{}/-"
_OPTION = r"-{1,2}[^\s|;&]{0,64}"
# '$SHELL' and '${BASH}' name a shell too.
_SHELL_VARIABLE = r"\$\{?(?:SHELL|BASH)\}?"
# An interpreter, optionally by path (/bin/sh), quoted ("sh"), and behind wrappers (sudo -E, /usr/bin/env,
# env A=1, xargs, busybox).
_INTERPRETER = (
    rf"(?<![{_PATH_CHARS}])[\"']?(?:[{_PATH_CHARS}]{{0,128}}/)?"
    rf"(?:(?:sudo|doas|env|xargs|command|exec|nohup|nice|time|busybox)\s+"
    rf"(?:(?:{_OPTION}|\w{{1,64}}=[^\s|;&]{{0,256}})\s+){{0,8}}+[\"']?(?:[{_PATH_CHARS}]{{0,128}}/)?){{0,3}}+"
    rf"(?:{_SHELL_VARIABLE}|{_EXECUTORS})[\"']?(?![\w.-])"
)
# 'source' or '.' as a command (at the start of a statement), never '.' as an argument ('jq . file').
_SOURCE_COMMAND = r"(?:(?:^|[;&|(\n`{\"'])[ \t]{0,64}|\b(?:then|do|else)[ \t]{1,64})(?:source|\.)"
# One shell command: quoted strings, escapes, and single pipes stay inside it; ';', '&', '&&', '||', and
# newlines end it.
_SHELL_COMMAND_RE = re.compile(r"""(?:'[^']*'|"(?:[^"\\]|\\.)*"|\\.|\|(?!\|)&?|[^'"\\|;&\n])+""", re.DOTALL)
_FETCHER_RE = re.compile(rf"\b{_FETCHERS}\b", re.IGNORECASE)
# A pipeline stage that starts with an interpreter: '| sh', '| sudo bash', '|& /usr/bin/env python3'.
_STAGE_INTERPRETER_RE = re.compile(rf"&?\s*(?P<interpreter>{_INTERPRETER})", re.IGNORECASE)
_XARGS_RE = re.compile(r"\bxargs\b", re.IGNORECASE)
# A pipeline stage that sources its standard input: '| source /dev/stdin', '| . /dev/fd/0'.
_STAGE_SOURCE_STDIN_RE = re.compile(
    r"&?\s*(?:source|\.)\s+[\"']?(?:/dev/stdin|/dev/fd/0|/proc/self/fd/0|-)[\"']?(?:\s|$)"
)
# bash -c "$(curl ...)", sh <(wget ...), eval "$(curl ...)", source <(curl ...), . <(curl ...)
_SUBSTITUTION_RE = re.compile(
    rf"(?:{_INTERPRETER}|(?<![\w.-])eval|{_SOURCE_COMMAND})\s+(?:{_OPTION}\s+){{0,4}}+[\"']?(?:<\(|\$\(|`)\s*"
    rf"{_FETCHERS}\b",
    re.IGNORECASE | re.MULTILINE,
)
# A here-string fed to an interpreter: 'source /dev/stdin <<< "$(curl ...)"', 'bash <<< "$code"'.
_HERE_STRING_RE = re.compile(
    rf"(?:{_INTERPRETER}|{_SOURCE_COMMAND})[^;&|\n<>]{{0,128}}<<<\s*[\"']?"
    rf"(?:(?:\$\(|`)\s*(?:[{_PATH_CHARS}]{{0,128}}/)?(?P<fetch>{_FETCHERS})\b|\$\{{?(?P<var>[A-Za-z_]\w{{0,63}})\b)",
    re.IGNORECASE | re.MULTILINE,
)
# A variable that holds downloaded content ('code=$(curl -fsSL ...)'), and a command that runs one
# ('eval "$code"', 'bash -c "$code"', 'source "$code"').
_FETCH_ASSIGNMENT_RE = re.compile(
    rf"(?<![\w$])(?P<var>[A-Za-z_]\w{{0,63}})=[\"']?(?:\$\(|`)\s*(?:[{_PATH_CHARS}]{{0,128}}/)?{_FETCHERS}\b",
    re.IGNORECASE,
)
_VARIABLE_RUN_RE = re.compile(
    rf"(?:(?<![\w.-])eval|{_SOURCE_COMMAND}|{_INTERPRETER})\s+(?:{_OPTION}\s+){{0,4}}+[\"']?"
    r"\$\{?(?P<var>[A-Za-z_]\w{0,63})\b",
    re.IGNORECASE | re.MULTILINE,
)
_VARIABLE_REF_RE = re.compile(r"\$\{?([A-Za-z_]\w{0,63})")
# Python or JavaScript that evaluates a download: exec(urlopen(url).read()), eval(await (await fetch(u)).text()).
_EXEC_FETCHED_RE = re.compile(
    r"\b(?:exec|eval)\s*\([^\n]{0,256}?\b(?:urlopen|urlretrieve|requests\.get|httpx\.get|fetch)\s*\("
)
# iex (iwr ...), Invoke-Expression (New-Object Net.WebClient).DownloadString(...)
_INVOKE_EXPRESSION_RE = re.compile(r"\b(?:iex|invoke-expression)\b", re.IGNORECASE)
_POWERSHELL_FETCH_RE = re.compile(r"\b(?:iwr|irm|invoke-webrequest|invoke-restmethod|downloadstring)\b", re.IGNORECASE)
# A fetch that writes a file: '-o FILE' (also '-fsSLo FILE'), wget's '-O FILE', '--output[-document] FILE',
# '-OutFile FILE', 'tee FILE', or a '> FILE' redirect. curl -O (a URL follows) and wget save under the
# URL's last path segment.
_DOWNLOADER_RE = re.compile(r"\b(?:curl|wget|aria2c|iwr|invoke-webrequest)\b", re.IGNORECASE)
_WGET_RE = re.compile(r"\bwget\b", re.IGNORECASE)
_OUTPUT_FILE_RE = re.compile(
    r"(?:\s-outfile|\s-[A-Za-z]{0,8}o|\s--output(?:-document)?|\btee(?:\s+-a)?\s|>{1,2})"
    r"(?:\s*=)?\s*[\"']?(?P<file>[^\s;&|'\"<>()`]{1,256})",
    re.IGNORECASE,
)
# An archive tool that unpacks a download: 'tar xzf x.tgz -C dir', 'curl ... | tar xz', 'unzip x.zip -d dir'.
_EXTRACTOR_RE = re.compile(r"\b(?:tar|bsdtar|unzip)\b")
# A file the text runs: with an interpreter or 'source' / '.', or as a command. A shell runs any command
# word with a '/' as a file ('./x', '/tmp/x', '&& $DIR/x', 'd/run'), and a variable there runs what it holds
# ('"$TOOL" check').
_RUN_FILE_RE = re.compile(
    rf"(?:{_INTERPRETER}|{_SOURCE_COMMAND})[ \t]+(?:{_OPTION}[ \t]+){{0,4}}+[\"']?"
    r"(?P<script>[^\s;&|'\"<>()`]{1,256})"
    r"|(?:^|[;&|(\n`]|\b(?:then|do|else|exec)\b)[ \t]{0,64}(?:(?:sudo|doas|nohup|command|env)\s+"
    rf"(?:{_OPTION}\s+){{0,4}}+)?[\"']?(?P<command>(?:\.{{1,2}}/|/|~/|\$\{{?\w{{1,64}}\}}?/?|[\w.-]{{1,128}}/)"
    r"[^\s;&|'\"<>()`]{0,256})",
    re.IGNORECASE | re.MULTILINE,
)
_BRACED_VARIABLE_RE = re.compile(r"\$\{(\w{1,64})\}")
# A simple variable assignment whose value is one word, without command substitution: 'D="$CLAUDE_PLUGIN_DATA/bin"',
# 'export BIN=${HOME}/.tool/bin'. Download and run paths are read with these expanded.
_SIMPLE_ASSIGNMENT_RE = re.compile(
    r"(?:^|[;&(\n{]|\b(?:export|local|readonly)[ \t])[ \t]{0,64}(?P<name>[A-Za-z_]\w{0,63})="
    r"(?P<quote>[\"']?)(?P<value>[^\s;&|'\"<>()`]{1,256})(?P=quote)(?=[\s;&|)]|$)",
    re.MULTILINE,
)
_VARIABLE_USE_RE = re.compile(r"\$(?:\{([A-Za-z_]\w{0,63})\}|([A-Za-z_]\w{0,63}))")
# A parameter-default chain that may name the plugin root ('${CLAUDE_PLUGIN_ROOT:-${PLUGIN_ROOT}}',
# '${CLAUDE_PLUGIN_ROOT:-}'): each '${NAME:-' (or '${NAME-') opens a level, and the innermost alternative is
# empty or one plain variable.
_DEFAULT_OPEN_RE = re.compile(r"\$\{([A-Za-z_]\w{0,63}):?-")
_PLAIN_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_]\w{0,63})\}|\$([A-Za-z_]\w{0,63})(?!\w)")
_MAX_DEFAULT_DEPTH = 8
# A hook command assignment that may set a variable to the plugin root: the whole command, one word, not
# single-quoted (a single-quoted value is literal text).
_ROOT_ALIAS_ASSIGNMENT_RE = re.compile(
    r"[ \t]*(?:export[ \t]+)?(?P<name>[A-Za-z_]\w{0,63})=(?P<quote>\"?)"
    r"(?P<value>[^\s;&|'\"<>()`\\]{1,256})(?P=quote)[ \t]*"
)
# What can set a variable without the text naming it, or run an assignment in another shell: a subshell,
# group, or command substitution ('(', ')', '`', a '{' that does not open '${'), a heredoc, 'eval', and
# 'source' / '.' of a file.
_OPAQUE_VARIABLES_RE = re.compile(
    rf"[()`]|(?<!\$)\{{|(?<!<)<<(?!<)|(?<![\w.-])eval(?![\w.-])|{_SOURCE_COMMAND}[ \t]", re.MULTILINE
)
# Shell words that open, continue, or close a compound command: an assignment inside 'if ...; then' or
# 'while ...; do' may never run, so a command with one resolves no variable either.
_COMPOUND_COMMAND_WORDS = frozenset(
    {"if", "then", "elif", "else", "fi", "while", "until", "do", "done", "for", "select", "case", "esac"}
)
# Shell text as quoting segments: a single- or double-quoted string, an escape, other text, or a lone quote
# (unbalanced quoting).
_QUOTING_SEGMENT_RE = re.compile(r"""'[^']*'|"(?:[^"\\]|\\.)*"|\\.|[^'"\\]+|.""", re.DOTALL)
_NAME_WORD_RE = re.compile(r"(?<!\w)[A-Za-z_]\w*")
# Variables the shell sets by itself ('cd' sets PWD, 'read' sets REPLY), so one assignment does not fix them.
_SHELL_SET_VARIABLES = frozenset(
    {
        "PWD",
        "OLDPWD",
        "REPLY",
        "OPTARG",
        "OPTIND",
        "MAPFILE",
        "DIRSTACK",
        "RANDOM",
        "SRANDOM",
        "SECONDS",
        "LINENO",
        "HISTCMD",
        "EPOCHSECONDS",
        "EPOCHREALTIME",
        "_",
    }
)
_MAX_ASSIGNMENTS = 256
_MAX_EXPANDED_CHARS = 1024
# Leading directories of a run path matched against unpack directories; a deeper unpack directory is cut to
# this depth, which can only widen a match.
_MAX_DIR_DEPTH = 8
# Where Claude Code (CLAUDE_PLUGIN_DATA) and Codex (PLUGIN_DATA) keep a plugin's persistent data: code run
# from there is not code the plugin ships.
_PLUGIN_DATA_REFS: tuple[str, ...] = ("$CLAUDE_PLUGIN_DATA", "$PLUGIN_DATA")
_ASSIGNMENT_WORD_RE = re.compile(r"[A-Za-z_]\w*=")
_SYSTEM_PATH_PREFIXES: tuple[str, ...] = (
    "/bin/",
    "/sbin/",
    "/usr/bin/",
    "/usr/sbin/",
    "/usr/local/bin/",
    "/usr/local/sbin/",
    "/opt/homebrew/bin/",
    "/dev/",
)
_PLUGIN_ROOT_REFS: tuple[str, ...] = ("${CLAUDE_PLUGIN_ROOT}", "$CLAUDE_PLUGIN_ROOT")
# A bare relative token that may name a script (Cursor hook commands such as "./approve.sh").
_RELATIVE_SCRIPT_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_./\\-]*$")
_PROJECT_DIR_REFS: tuple[str, ...] = ("${CLAUDE_PROJECT_DIR}", "$CLAUDE_PROJECT_DIR")
_HOME_REFS: tuple[str, ...] = ("~", "$HOME", "${HOME}")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


# --------------------------------------------------------------------------- #
# Hook dialects                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HookDialect:
    """How one client names hook events, approves a tool call, and spells a handler's command.

    ``allow_shapes`` name the approval outputs the client acts on (``claude``:
    ``permissionDecision``/``decision``/``behavior``; ``cursor``: ``permission``/
    ``decision``). ``command_keys`` are the handler keys that hold shell text.
    ``shell_tools`` are the tool names a matcher uses for the shell tool, and
    ``event_scopes`` fix the scope of approval events that only ever decide one
    kind of call (Cursor's ``beforeShellExecution`` decides shell commands).
    """

    name: str
    events: frozenset[str]
    approval_events: frozenset[str]
    context_events: frozenset[str]
    allow_shapes: tuple[str, ...] = ("claude",)
    command_keys: tuple[str, ...] = ("command",)
    shell_tools: tuple[str, ...] = ("Bash",)
    event_scopes: tuple[tuple[str, str], ...] = ()

    def event_scope(self, event: str) -> str | None:
        return dict(self.event_scopes).get(event)


# Claude Code. Its shell tool is Bash, and PowerShell on Windows.
CLAUDE_HOOKS = HookDialect("claude", HOOK_EVENTS, APPROVAL_EVENTS, CONTEXT_EVENTS, shell_tools=("Bash", "PowerShell"))
# Codex hooks (the Codex hooks reference): Claude Code's event names and outputs plus Interrupt;
# SubagentStart output reaches the model; the shell tool is reported as Bash (exec_command, shell, and
# local_shell are the tool's own names) and the file-edit tool is apply_patch. Claude Code --plugin-dir
# loads a Codex plugin's default hooks/hooks.json too, so Claude Code's names keep their meaning.
CODEX_HOOKS = HookDialect(
    "codex",
    HOOK_EVENTS | frozenset({"Interrupt"}),
    APPROVAL_EVENTS,
    CONTEXT_EVENTS | frozenset({"SubagentStart"}),
    shell_tools=("Bash", "PowerShell", "exec_command", "shell", "local_shell"),
)
# Cursor hooks reference (cursor.com/docs/agent/hooks): camelCase events, flat handler lists, and
# {"permission": "allow"} / {"decision": "allow"} outputs. sessionStart returns additional_context.
# Cursor also runs Claude Code-shaped hooks, so Claude Code's event names keep their meaning.
CURSOR_HOOKS = HookDialect(
    "cursor",
    HOOK_EVENTS
    | frozenset(
        {
            "sessionStart",
            "sessionEnd",
            "preToolUse",
            "postToolUse",
            "postToolUseFailure",
            "subagentStart",
            "subagentStop",
            "beforeShellExecution",
            "afterShellExecution",
            "beforeMCPExecution",
            "afterMCPExecution",
            "beforeReadFile",
            "afterFileEdit",
            "beforeSubmitPrompt",
            "preCompact",
            "stop",
            "afterAgentResponse",
            "afterAgentThought",
            "beforeTabFileRead",
            "afterTabFileEdit",
        }
    ),
    APPROVAL_EVENTS
    | frozenset({"preToolUse", "beforeShellExecution", "beforeMCPExecution", "beforeReadFile", "subagentStart"}),
    CONTEXT_EVENTS | frozenset({"sessionStart"}),
    allow_shapes=("claude", "cursor"),
    shell_tools=("Shell", "Bash"),
    event_scopes=(
        ("beforeShellExecution", "bash"),
        ("beforeMCPExecution", "mcp"),
        ("beforeReadFile", "narrow"),
        ("subagentStart", "narrow"),
    ),
)
# GitHub Copilot hooks (Agent Plugins com.github.copilot/): camelCase events, and handlers that hold the
# command under 'bash' or 'powershell' (or 'command'). VS Code also reads Claude Code's event names there.
COPILOT_HOOKS = HookDialect(
    "copilot",
    HOOK_EVENTS
    | frozenset(
        {
            "sessionStart",
            "sessionEnd",
            "userPromptSubmitted",
            "preToolUse",
            "postToolUse",
            "errorOccurred",
            "agentStop",
            "subagentStop",
            "permissionRequest",
        }
    ),
    APPROVAL_EVENTS | frozenset({"preToolUse", "permissionRequest"}),
    CONTEXT_EVENTS | frozenset({"sessionStart", "userPromptSubmitted"}),
    command_keys=("command", "bash", "powershell", "exec"),
    shell_tools=("Bash", "bash", "powershell"),
)
# An Agent Plugins namespace no dialect covers: every known event, output, and command key (fail closed).
GENERIC_HOOKS = HookDialect(
    "generic",
    CLAUDE_HOOKS.events | CODEX_HOOKS.events | CURSOR_HOOKS.events | COPILOT_HOOKS.events,
    CLAUDE_HOOKS.approval_events | CURSOR_HOOKS.approval_events | COPILOT_HOOKS.approval_events,
    CODEX_HOOKS.context_events | CURSOR_HOOKS.context_events | COPILOT_HOOKS.context_events,
    allow_shapes=("claude", "cursor"),
    command_keys=COPILOT_HOOKS.command_keys,
    shell_tools=tuple(dict.fromkeys((*CODEX_HOOKS.shell_tools, "Shell", "bash", "powershell"))),
    event_scopes=CURSOR_HOOKS.event_scopes,
)
# Plugin monitors: one long-running command each, whose every stdout line reaches the model.
MONITOR_EVENT = "Monitor"
MONITOR_HOOKS = HookDialect("monitor", frozenset({MONITOR_EVENT}), frozenset(), frozenset({MONITOR_EVENT}))
_NAMESPACE_DIALECTS: dict[str, HookDialect] = {
    "com.cursor": CURSOR_HOOKS,
    "com.github.copilot": COPILOT_HOOKS,
    "com.openai": CODEX_HOOKS,
}


def hook_dialect(manifest_type: str | None, hook_file: str = "") -> HookDialect:
    """The hook dialect of a hooks file: by manifest format, and by namespace for Agent Plugins."""
    if manifest_type == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE:
        namespace = PurePosixPath(hook_file).parts[0] if hook_file else ""
        return _NAMESPACE_DIALECTS.get(namespace, GENERIC_HOOKS)
    if manifest_type == PLUGIN_CURSOR_MANIFEST_TYPE:
        return CURSOR_HOOKS
    if manifest_type == PLUGIN_CODEX_MANIFEST_TYPE:
        return CODEX_HOOKS
    return CLAUDE_HOOKS


class HookScriptUnreadable(Exception):
    """A script a hook runs exists under the plugin root but cannot be read safely.

    Raised by a :class:`HookAnalyzer` ``read_script`` callback for a link, a
    special or hard-linked file, or a failed bounded read, so the analyzer can
    record that the script's evidence is missing instead of passing silently.
    """


def component_finding(
    severity: Severity,
    check_name: str,
    message: str,
    file_path: str,
    suggestion: str,
    *,
    component: tuple[str, str] | None,
    extra: dict[str, Any] | None = None,
) -> Finding:
    """A ``PLUGIN_SCHEMA`` finding attributed to a plugin ``component`` (``(type, name)``) in its metadata."""
    metadata: dict[str, Any] = {}
    if component is not None:
        metadata["plugin_component"] = {"type": component[0], "name": component[1]}
    if extra:
        metadata.update(extra)
    return Finding(
        category=PLUGIN_CATEGORY,
        severity=severity,
        check_name=check_name,
        message=message,
        file_path=file_path,
        suggestion=suggestion,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Tool lists                                                                  #
# --------------------------------------------------------------------------- #
def parse_tool_list(value: Any) -> list[str] | None:
    """Parse a ``tools`` / ``allowed-tools`` value; ``None`` when the field is absent.

    Accepts a YAML list or a comma- or whitespace-separated string. Separators
    inside parentheses (``Bash(git add *)``) do not split an entry. Every entry
    is returned (the frontmatter is already byte-bounded), so risk checks see
    all of them; privilege records store at most ``MAX_TOOL_ENTRIES``.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [text for text in (str(item).strip() for item in value) if text]
    if not isinstance(value, str):
        return [str(value)]
    entries: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if depth == 0 and (char == "," or char.isspace()):
            if current:
                entries.append("".join(current).strip())
                current = []
            continue
        current.append(char)
    if current:
        entries.append("".join(current).strip())
    return [entry for entry in entries if entry]


def _stored_tools(entries: list[str] | None) -> list[str] | None:
    """The bounded copy of a tool list kept in a privilege record; risk checks use the full list."""
    return None if entries is None else entries[:MAX_TOOL_ENTRIES]


def bash_rule_risk(entry: str) -> str | None:
    """Classify one ``Bash`` tool or permission rule the way Claude Code classifies its allow rules.

    ``"unrestricted"``: ``Bash`` with no content, empty content, or only ``*``
    and whitespace (``Bash()``, ``Bash(*)``, ``Bash(**)``), plus the legacy
    ``Bash(:*)`` / ``Bash(*:*)``. ``"interpreter"``: an interpreter or wrapper
    prefix that runs any command (``Bash(python3:*)``, ``Bash(sh -c *)``,
    ``Bash(env *)``, ``Bash(npm run:*)``). ``None`` for a scoped rule or another
    tool.
    """
    match = _TOOL_RULE_RE.fullmatch(entry)
    if match is None or match.group(1).lower() != "bash":
        return None
    content = match.group(2)
    if content is None or not content.strip(" \t\n\r*"):
        return "unrestricted"
    if "".join(content.split()) in {":*", "*:*"}:
        return "unrestricted"
    return "interpreter" if _dangerous_bash_prefix(content) else None


def _dangerous_bash_prefix(content: str) -> bool:
    """Claude Code's ``dangerous_prefix`` test for Bash rule content."""
    rule = content.strip().lower()
    if rule == "*":
        return True
    for prefix in _DANGEROUS_BASH_PREFIXES:
        if rule in {prefix, f"{prefix}:*", f"{prefix} *", f"{prefix}*"}:
            return True
        if rule.startswith(f"{prefix} ") and rule.endswith("*"):
            rest = rule[len(prefix) + 1 :]
            module_rule = prefix.startswith("python") and _PYTHON_DOTTED_MODULE_RE.fullmatch(rest[:-1])
            if rest.startswith("-") and not module_rule:
                return True
    return False


def is_unrestricted_bash(entry: str) -> bool:
    """A ``Bash`` grant that runs any command: unrestricted, or an interpreter or wrapper prefix."""
    return bash_rule_risk(entry) is not None


def is_broad_allow_rule(rule: str) -> bool:
    """A permission ``allow`` rule that pre-approves any shell command, or every tool (``*``)."""
    return is_unrestricted_bash(rule) or "".join(rule.split()) == "*"


# Claude Code permission modes that approve some tool calls without a prompt ('bypassPermissions', which
# approves every call, is a HIGH permission-bypass flag of its own).
PERMISSIVE_PERMISSION_MODES: frozenset[str] = frozenset({"acceptEdits", "auto"})


# One value of the flag: a quoted list ("Bash(python3:*) Edit") or one unquoted word.
_ALLOWED_TOOLS_VALUE = r"\"[^\"\n]{0,1024}\"|'[^'\n]{0,1024}'|[^\s;&|]{1,1024}"
_ALLOWED_TOOLS_VALUE_RE = re.compile(_ALLOWED_TOOLS_VALUE)
# Claude Code's CLI pre-approval flag: '--allowedTools Bash', '--allowed-tools "Bash(python3:*) Edit"'. It takes a
# list of values, so after a space every word up to the next option is one ('--allowedTools Read Bash -p x');
# after '=' only the one value is.
_ALLOWED_TOOLS_FLAG_RE = re.compile(
    rf"(?<![\w-])--allowed-?tools(?:=(?P<value>{_ALLOWED_TOOLS_VALUE})"
    rf"|\s+(?P<values>(?:{_ALLOWED_TOOLS_VALUE})(?:[ \t]+(?!-)(?:{_ALLOWED_TOOLS_VALUE}))*))",
    re.IGNORECASE,
)
_ALLOWED_TOOLS_FLAGS = frozenset({"--allowedtools", "--allowed-tools"})
# The JSON path of an argv token ('hooks.PreToolUse[0].args[2]'): the flag and its value are adjacent items.
_LIST_ITEM_PATH_RE = re.compile(r"^(?P<parent>.*)\[(?P<index>\d+)\]$")


def allowed_tools_flag_issues(value: Any) -> list[OverrideIssue]:
    """``--allowedTools`` passed to an agent CLI with a grant that runs any command (or ``*``), HIGH each.

    The launched agent then runs any shell command without a prompt, like an
    unrestricted ``allowed-tools`` grant. Scoped grants (``Bash(git status *)``)
    are not flagged. Read in any config string or argv list.
    """
    hits: dict[tuple[str, str], None] = {}
    items: dict[str, dict[int, str]] = {}

    def _broad(values: Iterable[str]) -> str | None:
        return next(
            (rule for raw in values for rule in parse_tool_list(raw.strip("\"'")) or [] if is_broad_allow_rule(rule)),
            None,
        )

    for path, text in iter_config_strings(value):
        for match in _ALLOWED_TOOLS_FLAG_RE.finditer(text):
            spaced = match.group("values")
            rule = _broad([match.group("value")] if spaced is None else _ALLOWED_TOOLS_VALUE_RE.findall(spaced))
            if rule is not None:
                hits.setdefault((path, rule))
        item = _LIST_ITEM_PATH_RE.match(path)
        if item is not None:
            items.setdefault(item.group("parent"), {})[int(item.group("index"))] = text
    for parent, tokens in items.items():
        for index, token in tokens.items():
            if token.strip().lower() in _ALLOWED_TOOLS_FLAGS:
                # Every item up to the next option is a value: ['--allowedTools', 'Read', 'Bash', '-p', 'x'].
                end = index + 1
                while end in tokens and not tokens[end].lstrip().startswith("-"):
                    end += 1
                rule = _broad(tokens[position] for position in range(index + 1, end))
                if rule is not None:
                    hits.setdefault((f"{parent}[{index}]", rule))
    return [
        OverrideIssue(
            "permission_allow_flag",
            Severity.HIGH,
            f"agent-CLI flag '--allowedTools {_bounded(rule, 60)}'{f' in {path!r}' if path else ''} pre-approves "
            "any shell command (or every tool) in the launched agent, so it runs without a prompt",
            "Remove the flag, or scope it to exact commands such as Bash(git status *).",
        )
        for path, rule in hits
    ]


def _bash_grant_label(entry: str) -> str:
    if bash_rule_risk(entry) == "interpreter":
        return (
            f"{entry!r} (an interpreter or wrapper prefix, which Claude Code itself classes as able to run any command)"
        )
    return f"unrestricted {entry!r}"


def is_wildcard_tool(entry: str) -> bool:
    """A wildcard grant of every tool (``*``) or every MCP tool (``mcp__*``)."""
    return "".join(entry.split()).lower() in _WILDCARD_TOOLS


def _is_star_grant(entry: str) -> bool:
    """``*`` (or ``*(*)``): no tool has that name, so as an allowed-tools entry it pre-approves nothing."""
    return "".join(entry.split()) in {"*", "*(*)"}


def denies_every_bash_command(entries: Iterable[str] | None) -> bool:
    """Whether a ``disallowedTools`` list removes the Bash tool, the way Claude Code matches it.

    Tool names are case-sensitive (``bash`` denies nothing), and a rule whose
    content matches every command (``Bash(*)``, ``Bash()``) denies the whole tool.
    A scoped rule (``Bash(rm *)``) leaves Bash available.
    """
    for entry in entries or []:
        match = _TOOL_RULE_RE.fullmatch(entry)
        if match is None or match.group(1) != "Bash":
            continue
        content = match.group(2)
        if content is None or not content.strip(" \t\n\r*"):
            return True
    return False


def normalized_permission_mode(value: str | None) -> str | None:
    """A ``permissionMode`` spelled any way Tier 3 accepts it (case, ``-``, ``_``), as Claude Code names it."""
    if value is None:
        return None
    key = value.strip().replace("_", "").replace("-", "").casefold()
    return _PERMISSION_MODE_NAMES.get(key, value.strip())


_PERMISSION_MODE_NAMES = {
    "bypasspermissions": "bypassPermissions",
    "acceptedits": "acceptEdits",
    "auto": "auto",
    "default": "default",
    "plan": "plan",
    "dontask": "dontAsk",
}


def _is_true_flag(value: Any) -> bool:
    """YAML ``true`` or the string ``"true"``: Claude Code reads both as set (a probe hid both commands)."""
    return value is True or (isinstance(value, str) and value.strip().casefold() == "true")


# --------------------------------------------------------------------------- #
# Subagents and commands                                                      #
# --------------------------------------------------------------------------- #
_READ_ONLY_FLAGS = frozenset({"--read-only", "--readonly", "--read_only"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off"})


def _declares_read_only_flag(tokens: list[str]) -> bool:
    """A bare ``--read-only`` (not followed by a false value) or ``--read-only=<true value>``."""
    for index, token in enumerate(tokens):
        name, separator, value = token.partition("=")
        if name not in _READ_ONLY_FLAGS:
            continue
        if separator:
            if value.strip().strip("'\"") in TRUTHY_VALUES:
                return True
            continue
        following = tokens[index + 1].strip("'\"") if index + 1 < len(tokens) else ""
        if following not in _FALSY_VALUES:
            return True
    return False


def mcp_server_is_read_only(config: Any) -> bool:
    """Whether a runnable MCP server declares a read-only mode (flag or env).

    The flag counts when it is bare or set to a true value; ``--read-only=false``
    (or ``--read-only false``) does not make a server read-only.
    """
    if not isinstance(config, dict):
        return False
    args = config.get("args")
    tokens = [str(arg).lower() for arg in args] if isinstance(args, list) else []
    command = config.get("command")
    if isinstance(command, str):
        tokens.extend(command.lower().split())
    if _declares_read_only_flag(tokens):
        return True
    env = config.get("env")
    if isinstance(env, dict):
        for key, value in env.items():
            normalized = str(key).upper().replace("-", "_")
            read_only_key = normalized in {"READ_ONLY", "READONLY"} or normalized.endswith("_READ_ONLY")
            if read_only_key and str(value).strip().lower() in TRUTHY_VALUES:
                return True
    return False


@dataclass
class PrivilegeRecord:
    """Tool and permission grants of one subagent, command, or skill."""

    type: str
    name: str
    path: str | None
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    allowed_tools: list[str] | None = None
    model: str | None = None
    permission_mode: str | None = None
    model_invocable: bool | None = None
    inherits_all_tools: bool = False
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"type": self.type, "name": self.name, "path": self.path, "flags": list(self.flags)}
        if self.type == "agent":
            row.update(
                {
                    "tools": self.tools,
                    "disallowed_tools": self.disallowed_tools,
                    "model": self.model,
                    "permission_mode": self.permission_mode,
                    "inherits_all_tools": self.inherits_all_tools,
                }
            )
        else:
            row.update(
                {
                    "allowed_tools": self.allowed_tools,
                    "model": self.model,
                    "model_invocable": self.model_invocable,
                }
            )
        return row


def _string_field(frontmatter: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = frontmatter.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:128]
    return None


def plugin_mcp_tool_prefix(plugin_name: str, server: str) -> str:
    """Claude Code's permission name for one plugin MCP server's tools: ``mcp__plugin_<plugin>_<server>``.

    A ``query`` tool on server ``db`` of plugin ``my-plugin`` is
    ``mcp__plugin_my-plugin_db__query``; that is the name permission rules and
    ``disallowedTools`` must use.
    """
    return f"mcp__plugin_{''.join(plugin_name.split())}_{''.join(server.split())}".lower()


def _disallows_all_mcp(disallowed: list[str] | None, write_servers: Iterable[str], plugin_name: str | None) -> bool:
    """Whether ``disallowedTools`` entries (whitespace removed, lower-cased) deny every one of the servers' tools."""
    entries = {"".join(entry.split()).lower() for entry in disallowed or []}
    if "mcp__*" in entries or "mcp__*__*" in entries:
        return True
    servers = list(write_servers)
    if not servers or not plugin_name or not plugin_name.strip():
        return False
    prefixes = [plugin_mcp_tool_prefix(plugin_name, server) for server in servers]
    return all(prefix in entries or f"{prefix}__*" in entries for prefix in prefixes)


class _ModeRisk(NamedTuple):
    """What a subagent's ``permissionMode`` value does: its record flag and finding."""

    flag: str
    severity: Severity
    check: str
    effect: str
    suggestion: str


# Subagent permission modes that skip permission prompts. Claude Code ignores permissionMode for plugin
# subagents, but it applies if the file is copied into a project.
_PERMISSION_MODE_RISKS: dict[str, _ModeRisk] = {
    "bypassPermissions": _ModeRisk(
        "bypass_permissions",
        Severity.HIGH,
        "plugin_agent_bypass_permissions",
        "skips every permission prompt",
        "Remove permissionMode from the subagent.",
    ),
    "acceptEdits": _ModeRisk(
        "accept_edits",
        Severity.MEDIUM,
        "plugin_agent_accept_edits",
        "auto-accepts file edits and filesystem commands",
        "Remove permissionMode from the subagent, or let the user choose the mode.",
    ),
    "auto": _ModeRisk(
        "auto_mode",
        Severity.MEDIUM,
        "plugin_agent_auto_mode",
        "lets a classifier approve tool calls without a prompt",
        "Remove permissionMode from the subagent, or let the user choose the mode.",
    ),
}


def analyze_agent(
    name: str,
    path: str | None,
    frontmatter: dict[str, Any],
    file_path: str,
    *,
    write_capable_mcp: list[str],
    plugin_name: str | None = None,
) -> tuple[PrivilegeRecord, list[Finding]]:
    """Record one subagent's grants and flag risky ones (the agent is never loaded).

    ``plugin_name`` builds the ``mcp__plugin_<plugin>_<server>`` names that a
    ``disallowedTools`` entry must use to deny a plugin MCP server's tools;
    without it, only ``mcp__*`` denies them.
    """
    record = PrivilegeRecord("agent", name, path)
    findings: list[Finding] = []
    component = ("agent", name)
    # Claude Code loads an agent without frontmatter under its file name, with every field at its default, so it
    # inherits every tool like an agent that omits 'tools'.
    omits_tools = "omits 'tools'"
    if not frontmatter:
        record.flags.append("no_frontmatter")
        omits_tools = "has no frontmatter"
    # Risk checks read every entry; the record keeps a bounded copy for reports.
    tools = parse_tool_list(frontmatter.get("tools"))
    disallowed_tools = parse_tool_list(frontmatter.get("disallowedTools", frontmatter.get("disallowed-tools")))
    record.tools = _stored_tools(tools)
    record.disallowed_tools = _stored_tools(disallowed_tools)
    record.model = _string_field(frontmatter, "model")
    record.permission_mode = _string_field(frontmatter, "permissionMode", "permission-mode")
    record.inherits_all_tools = tools is None
    # Claude Code matches tool names case-sensitively: 'bash' denies nothing, and 'Bash(*)' denies all of Bash.
    denies_bash = denies_every_bash_command(disallowed_tools)
    bash = [entry for entry in tools if is_unrestricted_bash(entry)] if tools is not None else []
    if (bash or tools is None) and not denies_bash:
        # Advisory only: a subagent's tools list limits what it may call, it never pre-approves a tool,
        # so the session's permission prompts still apply. An agent that omits 'tools' inherits Bash too,
        # so both get the same severity.
        record.flags.append("unrestricted_bash")
        grant = (
            f"is granted {_bash_grant_label(bash[0])}"
            if bash
            else f"{omits_tools}, so it inherits every tool, including unrestricted Bash"
        )
        findings.append(
            component_finding(
                Severity.LOW,
                "plugin_agent_unrestricted_bash",
                f"subagent '{name}' {grant}; it can ask to run any shell command (the session's permission "
                "prompts still apply)",
                file_path,
                "List the tools the subagent needs and scope the shell grant to exact commands (for example "
                "Bash(git status *)), or deny Bash with disallowedTools.",
                component=component,
            )
        )
    if tools is not None:
        wildcards = [entry for entry in tools if is_wildcard_tool(entry)]
        if wildcards:
            record.flags.append("wildcard_tools")
            findings.append(
                component_finding(
                    Severity.MEDIUM,
                    "plugin_agent_wildcard_tools",
                    f"subagent '{name}' lists a wildcard tool grant ({', '.join(wildcards[:MAX_QUOTED_ENTRIES])})",
                    file_path,
                    "List the specific tools the subagent needs instead of a wildcard.",
                    component=component,
                )
            )
    elif write_capable_mcp and not _disallows_all_mcp(disallowed_tools, write_capable_mcp, plugin_name):
        record.flags.append("inherits_all_tools_with_write_mcp")
        servers = ", ".join(write_capable_mcp[:MAX_QUOTED_ENTRIES])
        example = (
            plugin_mcp_tool_prefix(plugin_name, write_capable_mcp[0])
            if plugin_name and plugin_name.strip()
            else "mcp__plugin_<plugin>_<server>"
        )
        findings.append(
            component_finding(
                Severity.MEDIUM,
                "plugin_agent_inherits_all_tools",
                f"subagent '{name}' {omits_tools}, so it inherits every tool, including the tools of the plugin's "
                f"MCP servers ({servers}); their tool lists are not known statically and may write",
                file_path,
                "Declare an explicit 'tools' list, or deny the servers' tools with 'disallowedTools: mcp__*' or one "
                f"entry per server such as '{example}' (Claude Code names plugin MCP tools "
                "mcp__plugin_<plugin>_<server>__<tool>).",
                component=component,
                extra={"mcp_servers": write_capable_mcp[:32]},
            )
        )
    else:
        record.flags.append("inherits_all_tools")

    # Read the mode the way Tier 3 native staging does (case, '-' and '_' ignored), so both tiers agree.
    mode = normalized_permission_mode(record.permission_mode) or ""
    risk = _PERMISSION_MODE_RISKS.get(mode)
    if risk is not None:
        record.flags.append(risk.flag)
        findings.append(
            component_finding(
                risk.severity,
                risk.check,
                f"subagent '{name}' sets permissionMode: {mode}, which {risk.effect} (Claude Code ignores "
                "permissionMode for plugin subagents, but it applies if the file is copied into a project)",
                file_path,
                risk.suggestion,
                component=component,
            )
        )
    for ignored in ("hooks", "mcpServers"):
        if ignored in frontmatter:
            record.flags.append(f"ignored_{ignored}")
    return record, findings


def analyze_command(
    name: str,
    path: str | None,
    frontmatter: dict[str, Any],
    file_path: str,
    *,
    entry: dict[str, Any] | None = None,
    client: str | None = None,
) -> tuple[PrivilegeRecord, list[Finding]]:
    """Record one command's pre-approved tools and flag unrestricted Bash or wildcard grants.

    ``entry`` is a ``plugin.json`` ``commands`` map entry, whose ``allowedTools``
    and ``model`` override the file's frontmatter. ``client`` is the client whose
    manifest loads the command (:func:`_analyze_allowed_tools`).
    """
    allowed = parse_tool_list(frontmatter.get("allowed-tools", frontmatter.get("allowedTools")))
    if entry is not None and "allowedTools" in entry:
        allowed = parse_tool_list(entry.get("allowedTools"))
    model = _string_field(entry or {}, "model") or _string_field(frontmatter, "model")
    return _analyze_allowed_tools("command", name, path, frontmatter, file_path, allowed, model, client=client)


def analyze_skill(
    name: str, path: str | None, frontmatter: dict[str, Any], file_path: str, *, client: str | None = None
) -> tuple[PrivilegeRecord, list[Finding]]:
    """Record one skill's pre-approved tools (``allowed-tools``) and flag unrestricted Bash or wildcard grants.

    Claude Code pre-approves a plugin skill's ``allowed-tools`` while the skill
    is active, exactly like a command's, so the same grants get the same findings.
    """
    allowed = parse_tool_list(frontmatter.get("allowed-tools", frontmatter.get("allowedTools")))
    return _analyze_allowed_tools(
        "skill", name, path, frontmatter, file_path, allowed, _string_field(frontmatter, "model"), client=client
    )


# Clients whose manifest loads a skill or command but that ignore 'allowed-tools': Codex has no Bash tool and
# does not read the field, so the grant matters only when Claude Code loads the same file.
CLIENTS_IGNORING_ALLOWED_TOOLS: frozenset[str] = frozenset({"codex"})
_CLIENT_LABELS = {"codex": "Codex"}


def _analyze_allowed_tools(
    kind: str,
    name: str,
    path: str | None,
    frontmatter: dict[str, Any],
    file_path: str,
    allowed: list[str] | None,
    model: str | None,
    *,
    client: str | None = None,
) -> tuple[PrivilegeRecord, list[Finding]]:
    """Grant findings for a command's or skill's ``allowed-tools``, which Claude Code pre-approves.

    For a client in :data:`CLIENTS_IGNORING_ALLOWED_TOOLS` (Codex) the grant is
    inert in that client and applies only if Claude Code loads the same file, so
    it is capped at MEDIUM (it does not fail the plugin), the message names both
    clients, and the record gets the ``claude_only_grant`` flag.
    """
    record = PrivilegeRecord(kind, name, path)
    findings: list[Finding] = []
    component = (kind, name)
    claude_only = client in CLIENTS_IGNORING_ALLOWED_TOOLS
    loader = _CLIENT_LABELS.get(client or "", client or "")
    # A cross-client merge reads this mark: when Claude Code does not load the file either, the grant is inert.
    claude_only_extra = {"claude_only_grant": True} if claude_only else None
    # Risk checks read every entry; the record keeps a bounded copy for reports.
    record.allowed_tools = _stored_tools(allowed)
    record.model = model
    record.model_invocable = not _is_true_flag(frontmatter.get("disable-model-invocation"))
    invocation = (
        f"Claude can also invoke this {kind} on its own (disable-model-invocation is not set)"
        if record.model_invocable
        else "only the user can invoke it"
    )
    for tool in allowed or []:
        if is_unrestricted_bash(tool):
            record.flags.append("unrestricted_bash")
            if claude_only:
                message = (
                    f"{kind} '{name}' pre-approves {_bash_grant_label(tool)} in allowed-tools; {loader} ignores "
                    "allowed-tools (it has no Bash tool), but if Claude Code loads this "
                    f"{kind} (claude --plugin-dir reads skills/ and commands/ from this folder), any shell command "
                    "runs without a prompt while it is active"
                )
            else:
                message = (
                    f"{kind} '{name}' pre-approves {_bash_grant_label(tool)} in allowed-tools, so any shell "
                    f"command runs without a prompt while it is active; {invocation}"
                )
            findings.append(
                component_finding(
                    Severity.MEDIUM if claude_only else Severity.HIGH,
                    f"plugin_{kind}_unrestricted_bash",
                    message,
                    file_path,
                    "Scope allowed-tools to the exact commands it needs, such as Bash(git status *).",
                    component=component,
                    extra=claude_only_extra,
                )
            )
            break
    wildcards = [tool for tool in allowed or [] if is_wildcard_tool(tool)]
    mcp_wildcards = [tool for tool in wildcards if not _is_star_grant(tool)]
    if mcp_wildcards:
        record.flags.append("wildcard_tools")
        findings.append(
            component_finding(
                Severity.MEDIUM,
                f"plugin_{kind}_wildcard_tools",
                f"{kind} '{name}' pre-approves a wildcard tool grant "
                f"({', '.join(mcp_wildcards[:MAX_QUOTED_ENTRIES])}); "
                + (
                    f"{loader} ignores allowed-tools, so this applies only if Claude Code loads this {kind}"
                    if claude_only
                    else invocation
                ),
                file_path,
                f"List the specific tools the {kind} needs in allowed-tools instead of a wildcard.",
                component=component,
                extra=claude_only_extra,
            )
        )
    elif wildcards:
        # Claude Code pre-approves allowed-tools entries by tool name, and no tool is named '*'.
        record.flags.append("wildcard_ignored")
        findings.append(
            component_finding(
                Severity.LOW,
                f"plugin_{kind}_wildcard_tools",
                f"{kind} '{name}' lists '*' in allowed-tools, which pre-approves nothing in Claude Code (no tool "
                "is named '*'); the tools still prompt as usual",
                file_path,
                f"List the specific tools the {kind} needs in allowed-tools, or remove the '*' entry.",
                component=component,
            )
        )
    if claude_only and any(flag in record.flags for flag in ("unrestricted_bash", "wildcard_tools")):
        record.flags.append("claude_only_grant")
    return record, findings


# --------------------------------------------------------------------------- #
# Hooks                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class HookRecord:
    """One hook handler: where it runs, what it runs, and its static risk flags."""

    id: str
    source: str
    file: str
    event: str
    matcher: str | None
    handler_type: str
    target: str
    risk_flags: list[str] = field(default_factory=list)
    # Raw http handler URL for the opt-in endpoint resolution; never serialized.
    url: str | None = field(default=None, repr=False)

    def add_flag(self, flag: str) -> None:
        """Record a risk flag; each flag is listed (and counted in the summary) once per handler."""
        if flag not in self.risk_flags:
            self.risk_flags.append(flag)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "file": self.file,
            "event": self.event,
            "matcher": self.matcher,
            "handler_type": self.handler_type,
            "target": self.target,
            "risk_flags": list(self.risk_flags),
        }


def iter_hook_handlers(
    config: Any, truncated: list[str] | None = None
) -> Iterator[tuple[str, int, str | None, int, Any]]:
    """Yield ``(event, group_index, matcher, handler_index, handler)`` for a hooks config.

    Accepts the ``hooks.json`` / settings shape ``{"hooks": {Event: [...]}}`` and
    the inline shape ``{Event: [...]}``. Bounded by the ``MAX_HOOK_*`` limits;
    each limit that cuts the scan short is described in ``truncated`` (when a
    list is given), so a caller can report the handlers that were never yielded.
    """
    events = config.get("hooks") if isinstance(config, dict) and isinstance(config.get("hooks"), dict) else config
    if not isinstance(events, dict):
        return
    limits = truncated if truncated is not None else []
    items = list(events.items())
    if len(items) > MAX_HOOK_EVENTS:
        limits.append(f"more than {MAX_HOOK_EVENTS} events")
    emitted = 0
    for event, groups in items[:MAX_HOOK_EVENTS]:
        if not isinstance(groups, list):
            continue
        if len(groups) > MAX_HOOK_GROUPS:
            limits.append(f"more than {MAX_HOOK_GROUPS} matcher groups for {_bounded(str(event), 64)!r}")
        for group_index, group in enumerate(groups[:MAX_HOOK_GROUPS]):
            if not isinstance(group, dict):
                continue
            raw_matcher = group.get("matcher")
            matcher = raw_matcher if isinstance(raw_matcher, str) else None
            handlers = group.get("hooks")
            if not isinstance(handlers, list):
                continue
            for handler_index, handler in enumerate(handlers):
                if emitted >= MAX_HOOK_HANDLERS:
                    limits.append(f"more than {MAX_HOOK_HANDLERS} handlers")
                    return
                emitted += 1
                yield str(event), group_index, matcher, handler_index, handler


# --------------------------------------------------------------------------- #
# Hook matchers                                                               #
# --------------------------------------------------------------------------- #
# A parsed regex matcher is a tree of tuples: ("class", items, negated) is one
# character, each item a (code-point ranges, negated) pair; ("seq", nodes),
# ("alt", nodes), ("rep", node, low, high or None), and the ("bol",) / ("eol",)
# anchors. Lookarounds and word boundaries become the empty sequence.
_Node = tuple[Any, ...]
_ClassItem = tuple[tuple[tuple[int, int], ...], bool]

_MAX_MATCHER_DEPTH = 16
# Repetition counts above this match the same tool names (every sample is far shorter).
_MAX_MATCHER_REPEAT = 64
# Evaluation steps per matcher (a few milliseconds); realistic matchers need a few hundred.
_MAX_MATCHER_WORK = 10_000
_QUANTIFIERS: dict[str, tuple[int, int | None]] = {"*": (0, None), "+": (1, None), "?": (0, 1)}
_BRACED_QUANTIFIER_RE = re.compile(r"\{(\d+)(,(\d*))?\}")
_GROUP_NAME_RE = re.compile(r"\?<[A-Za-z_$][\w$]*>")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_DIGIT_RANGES = ((0x30, 0x39),)
_WORD_RANGES = ((0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A))
_SPACE_RANGES = (
    (0x09, 0x0D),
    (0x20, 0x20),
    (0xA0, 0xA0),
    (0x1680, 0x1680),
    (0x2000, 0x200A),
    (0x2028, 0x2029),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
    (0xFEFF, 0xFEFF),
)
_CLASS_ESCAPES: dict[str, _ClassItem] = {
    "d": (_DIGIT_RANGES, False),
    "D": (_DIGIT_RANGES, True),
    "w": (_WORD_RANGES, False),
    "W": (_WORD_RANGES, True),
    "s": (_SPACE_RANGES, False),
    "S": (_SPACE_RANGES, True),
}
_CONTROL_ESCAPES = {"t": 0x09, "n": 0x0A, "v": 0x0B, "f": 0x0C, "r": 0x0D}
_DOT: _Node = ("class", ((((0x0A, 0x0A), (0x0D, 0x0D), (0x2028, 0x2029)), True),), False)
_EMPTY: _Node = ("seq", ())


class _UnsupportedMatcher(Exception):
    """Matcher syntax the classifier does not model; the matcher fails closed as ``all``."""


class _InvalidMatcher(Exception):
    """A matcher JavaScript rejects as a regular expression, so Claude Code never matches it."""


@functools.lru_cache(maxsize=1024)
def _char_node(code: int) -> _Node:
    """One literal character (shared, so its positions in a name are computed once)."""
    return ("class", ((((code, code),), False),), False)


def _class_item(atom: int | _ClassItem) -> _ClassItem:
    return (((atom, atom),), False) if isinstance(atom, int) else atom


class _MatcherParser:
    """Bounded recursive-descent parser for the JavaScript regex syntax hook matchers use (no flags)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.depth = 0

    def parse(self) -> _Node:
        node = self._alternation()
        if self.pos < len(self.text):
            raise _InvalidMatcher  # an unmatched ')'
        return node

    def _peek(self, offset: int = 0) -> str:
        index = self.pos + offset
        return self.text[index] if index < len(self.text) else ""

    def _alternation(self) -> _Node:
        branches = [self._sequence()]
        while self._peek() == "|":
            self.pos += 1
            branches.append(self._sequence())
        return branches[0] if len(branches) == 1 else ("alt", tuple(branches))

    def _sequence(self) -> _Node:
        items: list[_Node] = []
        while self._peek() not in {"", "|", ")"}:
            node, quantifiable = self._atom()
            bounds = self._quantifier()
            if bounds is not None:
                if not quantifiable:
                    raise _UnsupportedMatcher  # a quantified assertion
                node = ("rep", node, *bounds)
            items.append(node)
        return items[0] if len(items) == 1 else ("seq", tuple(items))

    def _braced(self) -> re.Match[str] | None:
        return _BRACED_QUANTIFIER_RE.match(self.text, self.pos) if self._peek() == "{" else None

    def _quantifier(self) -> tuple[int, int | None] | None:
        char = self._peek()
        braced = self._braced()
        if char in _QUANTIFIERS:
            self.pos += 1
            low, high = _QUANTIFIERS[char]
        elif braced is not None:
            self.pos = braced.end()
            low = int(braced.group(1))
            high = low if braced.group(2) is None else (int(braced.group(3)) if braced.group(3) else None)
            if high is not None and high < low:
                raise _UnsupportedMatcher  # numbers out of order
        else:
            return None
        if self._peek() == "?":  # lazy: the same matches
            self.pos += 1
        if self._peek() in _QUANTIFIERS or self._braced() is not None:
            raise _UnsupportedMatcher  # a quantifier of a quantifier
        return min(low, _MAX_MATCHER_REPEAT), (None if high is None or high > _MAX_MATCHER_REPEAT else high)

    def _atom(self) -> tuple[_Node, bool]:
        """One atom, and whether a quantifier may follow it."""
        char = self.text[self.pos]
        self.pos += 1
        if char in _QUANTIFIERS:
            raise _InvalidMatcher  # nothing to repeat
        if char == "(":
            return self._group()
        if char == "[":
            return self._class(), True
        if char == "\\":
            return self._escape()
        if char == ".":
            return _DOT, True
        if char in {"^", "$"}:
            return ("bol",) if char == "^" else ("eol",), False
        if char == "{" and _BRACED_QUANTIFIER_RE.match(self.text, self.pos - 1):
            raise _UnsupportedMatcher  # a braced quantifier with nothing to repeat
        return _char_node(ord(char)), True

    def _group(self) -> tuple[_Node, bool]:
        self.depth += 1
        if self.depth > _MAX_MATCHER_DEPTH:
            raise _UnsupportedMatcher
        assertion = False
        if self._peek() == "?":
            named = _GROUP_NAME_RE.match(self.text, self.pos)
            if self.text.startswith(("?=", "?!"), self.pos):
                self.pos += 2
                assertion = True
            elif self.text.startswith(("?<=", "?<!"), self.pos):
                self.pos += 3
                assertion = True
            elif self.text.startswith("?:", self.pos):
                self.pos += 2
            elif named is not None:
                self.pos = named.end()
            else:
                raise _UnsupportedMatcher  # inline flags and other group syntax
        node = self._alternation()
        if self._peek() != ")":
            raise _InvalidMatcher  # an unterminated group
        self.pos += 1
        self.depth -= 1
        # A lookaround only narrows a match, so reading it as always true can only widen the scope.
        return (_EMPTY, False) if assertion else (node, True)

    def _escape(self) -> tuple[_Node, bool]:
        if not self._peek():
            raise _InvalidMatcher  # '\' at the end of the pattern
        char = self.text[self.pos]
        self.pos += 1
        if char in {"b", "B"}:
            return _EMPTY, False  # a word boundary only narrows a match
        if char in _CLASS_ESCAPES:
            return ("class", (_CLASS_ESCAPES[char],), False), True
        return _char_node(self._escaped_code(char)), True

    def _escaped_code(self, char: str) -> int:
        """Code point of a character escape; backreferences and flag-dependent escapes fail closed."""
        if char in _CONTROL_ESCAPES:
            return _CONTROL_ESCAPES[char]
        if char == "0" and not self._peek().isdigit():
            return 0
        width = {"x": 2, "u": 4}.get(char, 0)
        digits = self.text[self.pos : self.pos + width]
        if width and len(digits) == width and set(digits) <= _HEX_DIGITS:
            self.pos += width
            return int(digits, 16)
        if char.isalnum():
            raise _UnsupportedMatcher  # \1, \k<name>, \c, \p, and letters whose meaning depends on flags
        return ord(char)

    def _class(self) -> _Node:
        negated = self._peek() == "^"
        if negated:
            self.pos += 1
        items: list[_ClassItem] = []
        while self._peek() != "]":
            if not self._peek():
                raise _InvalidMatcher  # an unterminated character class
            low = self._class_atom()
            if self._peek() == "-" and self._peek(1) not in {"", "]"}:
                self.pos += 1
                high = self._class_atom()
                if isinstance(low, int) and isinstance(high, int):
                    if low > high:
                        raise _UnsupportedMatcher  # a range out of order
                    items.append((((low, high),), False))
                else:  # a class escape at either end makes '-' a literal
                    items.extend(_class_item(atom) for atom in (low, ord("-"), high))
                continue
            items.append(_class_item(low))
        self.pos += 1
        return ("class", tuple(items), negated)

    def _class_atom(self) -> int | _ClassItem:
        char = self.text[self.pos]
        self.pos += 1
        if char != "\\":
            return ord(char)
        if not self._peek():
            raise _InvalidMatcher  # '\' at the end of the pattern
        char = self.text[self.pos]
        self.pos += 1
        if char in _CLASS_ESCAPES:
            return _CLASS_ESCAPES[char]
        return 0x08 if char == "b" else self._escaped_code(char)


@functools.lru_cache(maxsize=512)
def _compiled_matcher(text: str) -> _Node:
    """The parsed regex of a matcher (cached; ``_UnsupportedMatcher`` and ``_InvalidMatcher`` propagate uncached).

    Parse trees are immutable, so the scope and sensitive-tool checks share one;
    each check evaluates it with its own :class:`_MatcherEvaluator` and work budget.
    """
    return _MatcherParser(text).parse()


def _class_matches(node: _Node, code: int) -> bool:
    hit = any(any(low <= code <= high for low, high in ranges) != negated for ranges, negated in node[1])
    return hit != node[2]


class _MatcherEvaluator:
    """Tests one parsed matcher against tool names, without backtracking and within a work budget.

    Positions in a name are bits: each node maps the set of positions it may
    start at to the set it may end at, for all starts in one step. A repetition
    of a repetition is expanded once per start and memoized, so nesting cannot
    multiply the work, which stays polynomial in the pattern and name sizes.
    Realistic matchers need a few hundred steps; one that exhausts
    ``_MAX_MATCHER_WORK`` raises :class:`_UnsupportedMatcher` and fails closed.
    """

    def __init__(self, root: _Node) -> None:
        self.root = root
        self.budget = _MAX_MATCHER_WORK
        self._nested: dict[int, bool] = {}

    def _has_repetition(self, node: _Node) -> bool:
        key = id(node)
        if key not in self._nested:
            kind = node[0]
            self._nested[key] = kind == "rep" or (
                kind in {"seq", "alt"} and any(self._has_repetition(child) for child in node[1])
            )
        return self._nested[key]

    def matches(self, subject: str) -> bool:
        """Whether the matcher matches anywhere in ``subject``, like JavaScript's ``RegExp.test``."""
        size = len(subject)
        char_masks: dict[int, int] = {}
        repeated: dict[tuple[int, int], int] = {}
        budget = self.budget

        def advance(node: _Node, starts: int) -> int:
            """The positions ``node`` can end at, from every start in the ``starts`` bitmask."""
            nonlocal budget
            budget -= 1
            if budget < 0:
                raise _UnsupportedMatcher
            kind = node[0]
            if kind == "class":
                mask = char_masks.get(id(node))
                if mask is None:
                    mask = sum(1 << at for at, char in enumerate(subject) if _class_matches(node, ord(char)))
                    char_masks[id(node)] = mask
                return (starts & mask) << 1
            if kind in {"bol", "eol"}:
                return starts & (1 if kind == "bol" else 1 << size)
            if kind == "alt":
                reached = 0
                for child in node[1]:
                    reached |= advance(child, starts)
                return reached
            if kind == "seq":
                for child in node[1]:
                    starts = advance(child, starts)
                    if not starts:
                        break
                return starts
            if not self._has_repetition(node[1]):
                return repeat(node, starts)
            reached = 0
            position = 0
            while starts:
                budget -= 1
                if starts & 1:
                    key = (id(node), position)
                    if key not in repeated:
                        repeated[key] = repeat(node, 1 << position)
                    reached |= repeated[key]
                starts >>= 1
                position += 1
            return reached

        def repeat(node: _Node, starts: int) -> int:
            """``low`` rounds of the repeated node, then further rounds while they reach new positions."""
            child, low, high = node[1], node[2], node[3]
            reached = starts
            for _ in range(low):
                following = advance(child, reached)
                if following in {0, reached}:
                    reached = following
                    break
                reached = following
            frontier, rounds = reached, low
            while frontier and (high is None or rounds < high):
                frontier = advance(child, frontier) & ~reached
                reached |= frontier
                rounds += 1
            return reached

        try:
            return advance(self.root, (1 << (size + 1)) - 1) != 0
        finally:
            self.budget = budget


def matcher_scope(matcher: str | None, shell_tools: tuple[str, ...] = ("Bash",)) -> str:
    """Classify a tool matcher: ``all`` (every tool), ``bash`` (matches Bash), or ``narrow``.

    Follows Claude Code's matcher rules without running the plugin's pattern on a
    backtracking regex engine:

    * no matcher, ``""``, or ``*`` matches every tool: ``all``;
    * only letters, digits, ``_``, and ``|``: an exact tool name or ``|`` list,
      ``bash`` when it names ``Bash``, else ``narrow`` (a list separated by
      commas that names ``Bash`` also counts as ``bash``);
    * anything else is a JavaScript regular expression searched in the tool
      name. A bounded parser reads it, and a position-set evaluator (no
      backtracking) tests it against sample tool names (``Bash``, ``Read``,
      ``Write``, ``Edit``, ``WebFetch``, and an MCP tool): ``all`` when it
      matches every sample, ``bash`` when it matches ``Bash``, else ``narrow``.
      Lookarounds and ``\\b``/``\\B`` count as always true, which can only widen
      the scope;
    * it fails closed: a regex longer than ``MAX_MATCHER_CHARS``, nested more
      than 16 groups deep, needing more evaluation steps than the work budget,
      or using syntax the parser does not model (backreferences, inline flags,
      quantified assertions, ...) is ``all``. A pattern JavaScript rejects (an
      unterminated ``[`` or group, an unmatched ``)``, a leading quantifier)
      never matches a tool, so it is ``narrow``.

    ``shell_tools`` are the names a client gives its shell tool (Cursor's
    ``Shell``); a matcher that matches one of them is ``bash``. Results are cached
    per matcher.
    """
    if matcher is None:
        return "all"
    text = matcher.strip()
    if text in {"", "*"}:
        return "all"
    if len(text) > MAX_MATCHER_CHARS:
        # A long exact name list is still decided; any other long matcher fails closed.
        return _name_list_scope(text, shell_tools) if _EXACT_MATCHER_RE.fullmatch(text) else "all"
    return _matcher_scope(text, shell_tools)


def _matcher_names(text: str) -> set[str]:
    return {part.strip() for part in _MATCHER_NAME_SEPARATOR_RE.split(text)}


def _name_list_scope(text: str, shell_tools: tuple[str, ...] = ("Bash",)) -> str:
    return "bash" if _matcher_names(text) & set(shell_tools) else "narrow"


@functools.lru_cache(maxsize=512)
def _matcher_scope(text: str, shell_tools: tuple[str, ...] = ("Bash",)) -> str:
    """:func:`matcher_scope` of a stripped matcher of at most ``MAX_MATCHER_CHARS`` characters."""
    scope = _name_list_scope(text, shell_tools) if _NAME_LIST_RE.fullmatch(text) else "narrow"
    if _EXACT_MATCHER_RE.fullmatch(text):
        return scope
    try:
        evaluator = _MatcherEvaluator(_compiled_matcher(text))
        if not any(evaluator.matches(tool) for tool in shell_tools):
            return scope
        return "all" if all(evaluator.matches(tool) for tool in _NON_SHELL_SAMPLE_TOOLS) else "bash"
    except _UnsupportedMatcher:
        return "all"
    except _InvalidMatcher:
        return scope


@functools.lru_cache(maxsize=512)
def matcher_sensitive_tools(matcher: str | None) -> tuple[str, ...]:
    """The write, fetch, and MCP tools (``Write``, ``Edit``, ``MultiEdit``, ``NotebookEdit``, ``WebFetch``,
    ``mcp__*``) a matcher matches, read the same way as :func:`matcher_scope` (failing closed to all of them).

    MCP tool names are not known statically, so a regex that mentions ``mcp``
    (``mcp__github__.*``, ``mcp__plugin_x_.*``) is taken to cover MCP tools.
    """
    if matcher is None or matcher.strip() in {"", "*"}:
        return _SENSITIVE_TOOLS
    text = matcher.strip()
    names = _matcher_names(text) if _NAME_LIST_RE.fullmatch(text) else set()
    listed = tuple(
        tool
        for tool in _SENSITIVE_TOOLS
        if tool in names or (tool.startswith("mcp__") and any(name.startswith("mcp__") for name in names))
    )
    if _EXACT_MATCHER_RE.fullmatch(text):
        return listed
    if len(text) > MAX_MATCHER_CHARS:
        return _SENSITIVE_TOOLS
    names_mcp = "mcp" in text.lower()
    try:
        evaluator = _MatcherEvaluator(_compiled_matcher(text))
        return tuple(
            tool
            for tool in _SENSITIVE_TOOLS
            if tool in listed or (names_mcp and tool.startswith("mcp__")) or evaluator.matches(tool)
        )
    except _UnsupportedMatcher:
        return _SENSITIVE_TOOLS
    except _InvalidMatcher:
        return listed


def _condition_scope(condition: str, shell_tools: tuple[str, ...] = ("Bash",)) -> tuple[str, tuple[str, ...]] | None:
    """The scope a handler's ``if`` condition allows, read as one permission rule (``Bash(git status)``).

    A shell rule that runs any command (``Bash``, ``Bash(*)``, an interpreter
    prefix) is ``bash``; any other shell rule is ``narrow``. A write, fetch, or
    MCP tool is ``scoped``; another tool is ``narrow``. ``None`` when the text is
    not one rule, so the caller keeps the matcher's scope (fail closed).
    """
    match = _TOOL_RULE_RE.fullmatch(condition)
    if match is None:
        return None
    tool, content = match.group(1), match.group(2)
    if tool in shell_tools:
        rule = "Bash" if content is None else f"Bash({content})"
        return ("bash", ()) if bash_rule_risk(rule) is not None else ("narrow", ())
    if tool in _SENSITIVE_TOOLS or tool.startswith("mcp__"):
        label = "mcp__server__tool" if tool.startswith("mcp__") else tool
        return "scoped", (_SENSITIVE_TOOL_LABELS.get(label, tool),)
    return "narrow", ()


def _condition_tool(condition: str) -> str:
    """The tool an ``if`` permission rule names (``Write`` for ``Write(*.md)``); ``""`` when it is not one rule."""
    match = _TOOL_RULE_RE.fullmatch(condition)
    return match.group(1) if match else ""


def _matcher_selects(matcher: str | None, tool: str) -> bool:
    """Whether a tool matcher selects ``tool``, read the same way as :func:`matcher_scope`.

    It fails closed: a matcher the evaluator cannot decide selects the tool.
    """
    if matcher is None or matcher.strip() in {"", "*"}:
        return True
    text = matcher.strip()
    if _NAME_LIST_RE.fullmatch(text) and tool in _matcher_names(text):
        return True
    if _EXACT_MATCHER_RE.fullmatch(text):
        return False
    if len(text) > MAX_MATCHER_CHARS:
        return True
    try:
        return _MatcherEvaluator(_compiled_matcher(text)).matches(tool)
    except _UnsupportedMatcher:
        return True
    except _InvalidMatcher:
        return False


def _command_tokens(handler: dict[str, Any], keys: tuple[str, ...] = ("command",)) -> list[str]:
    tokens: list[str] = []
    for key in keys:
        command = handler.get(key)
        if isinstance(command, str):
            tokens.extend(_split_words(command))
    args = handler.get("args")
    if isinstance(args, list):  # every arg: the config is byte-bounded, and a padded tail must not hide one
        tokens.extend(str(arg) for arg in args)
    return tokens


def _command_text(handler: dict[str, Any], keys: tuple[str, ...] = ("command",)) -> str:
    parts: list[str] = []
    for key in keys:
        command = handler.get(key)
        if isinstance(command, str):
            parts.append(command)
    args = handler.get("args")
    if isinstance(args, list):
        parts.extend(str(arg) for arg in args)
    return " ".join(parts)


def _plugin_root_refs(root_prefixes: Iterable[str] = ()) -> tuple[str, ...]:
    """Root placeholders a hook command may use, each also in its unbraced ``$VAR`` form.

    ``${CLAUDE_PLUGIN_ROOT}`` is always included, plus the manifest format's own
    (``${PLUGIN_ROOT}`` for Codex and Agent Plugins, ``${CURSOR_PLUGIN_ROOT}`` for Cursor).
    """
    refs = list(_PLUGIN_ROOT_REFS)
    for prefix in root_prefixes:
        refs.append(prefix)
        if prefix.startswith("${") and prefix.endswith("}"):
            refs.append(f"${prefix[2:-1]}")
    return tuple(dict.fromkeys(refs))


def _root_ref(token: str, refs: tuple[str, ...]) -> str | None:
    """The root placeholder *token* starts with (followed by a separator or nothing), or ``None``."""
    for prefix in refs:
        if token.startswith(prefix) and token[len(prefix) : len(prefix) + 1] in ("", "/", "\\", "="):
            return prefix
    return None


def _plugin_root_path(token: str, refs: tuple[str, ...] = _PLUGIN_ROOT_REFS) -> tuple[PurePosixPath | None, bool]:
    """Return ``(root-relative path, escapes)`` for a ``${CLAUDE_PLUGIN_ROOT}/...`` (or *refs*) token."""
    prefix = _root_ref(token, refs)
    if prefix is None:
        return None, False
    return _contained_path(token[len(prefix) :].split("=", 1)[0])


def _contained_path(rest: str) -> tuple[PurePosixPath | None, bool]:
    """Return ``(normalized relative path, escapes)`` for a path below some base directory."""
    normalized = posixpath.normpath("/" + rest.lstrip("/").replace("\\", "/"))
    raw_parts = [part for part in rest.replace("\\", "/").split("/") if part]
    depth = 0
    for part in raw_parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                return None, True
        elif part != ".":
            depth += 1
    relative = normalized.lstrip("/")
    return (PurePosixPath(relative) if relative else None), False


def _outside_root_reference(token: str, refs: tuple[str, ...] = _PLUGIN_ROOT_REFS) -> str | None:
    """Return why a command token references a file outside the plugin root, or ``None``."""
    value = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
    value = value.strip().strip("\"'")
    if not value or "://" in value:
        return None
    prefix = _root_ref(value, refs)
    if prefix is not None:
        _rel, escapes = _plugin_root_path(value, refs)
        return f"escapes {prefix} with '..'" if escapes else None
    if value.startswith(_PROJECT_DIR_REFS):
        return "points into the user's project (${CLAUDE_PROJECT_DIR})"
    if value == "~" or value.startswith(("~/", "$HOME/", "${HOME}/")) or value in _HOME_REFS:
        return "points into the user's home directory"
    if _WINDOWS_ABSOLUTE_RE.match(value):
        return "is an absolute path outside the plugin"
    if value.startswith("/"):
        if value.startswith(_SYSTEM_PATH_PREFIXES):
            return None
        return "is an absolute path outside the plugin"
    # Only a '..' that climbs above the start counts: './a/../b' stays inside.
    _rel, escapes = _contained_path(value)
    if escapes:
        return "climbs out of the working directory with '..'"
    return None


# A command word that only wraps the command after it, and an interpreter whose first operand is a script.
_RELATIVE_WRAPPER_WORDS = frozenset(
    {"sudo", "doas", "env", "exec", "command", "nohup", "nice", "time", "then", "do", "else", "if", "!", "(", "{"}
)
_SCRIPT_INTERPRETER_RE = re.compile(
    r"(?:(?:ba|z|da|k|fi|a)?sh|python[0-9.]{0,8}|node|perl|ruby|php|pwsh|powershell|deno|bun|tsx|ts-node)"
)
_SCRIPT_FILE_RE = re.compile(r"[\w.-]{1,128}\.(?:sh|bash|zsh|py|js|mjs|cjs|ts|rb|pl|php|ps1)")


def _relative_word(word: str, *, program: bool) -> bool:
    """Whether ``word`` names a file by a path relative to the working directory.

    ``./x`` always does. A program with a ``/`` (``scripts/check.sh``) does; a
    bare program name is found on ``PATH`` instead. A script operand of an
    interpreter does when it has a ``/`` or is a script file name (``check.py``).
    ``../`` paths are reported by :func:`_outside_root_reference`.
    """
    value = word.strip().strip("\"'")
    if not value or value.startswith(("/", "$", "~", "-", "..")) or "://" in value or "=" in value:
        return False
    if value.startswith("./"):
        return True
    if not _RELATIVE_SCRIPT_RE.match(value):
        return False
    return "/" in value or (not program and _SCRIPT_FILE_RE.fullmatch(value) is not None)


def _relative_run_paths(text: str, refs: tuple[str, ...] = _PLUGIN_ROOT_REFS) -> list[str]:
    """Relative paths a hook command runs, other than after ``cd <plugin root>``.

    Claude Code and Codex run a hook command in the session's working directory
    (the user's project), so ``./scripts/x.sh`` or ``python3 hooks/check.py``
    runs the project's file, not the one the plugin ships. Each shell command is
    read for its program and, for an interpreter, its script operand; any
    ``./`` word counts too.
    """
    found: dict[str, None] = {}
    plugin_cwd = False
    for command in _SHELL_COMMAND_RE.finditer(text):
        for stage in command.group(0).split("|"):
            words = _split_words(stage)
            index = 0
            while index < len(words) and (
                words[index] in _RELATIVE_WRAPPER_WORDS or _ASSIGNMENT_WORD_RE.match(words[index]) is not None
            ):
                index += 1
            if index >= len(words):
                continue
            if words[index] in {"cd", "pushd"}:
                target = words[index + 1].strip("\"'") if index + 1 < len(words) else ""
                plugin_cwd = _root_ref(target, refs) is not None
                continue
            if plugin_cwd:
                continue
            program = words[index]
            if _relative_word(program, program=True):
                found.setdefault(program)
            if _SCRIPT_INTERPRETER_RE.fullmatch(posixpath.basename(program.strip("\"'"))):
                operand = next((word for word in words[index + 1 :] if not word.startswith("-")), None)
                if operand is not None and _relative_word(operand, program=False):
                    found.setdefault(operand)
            for word in words[index + 1 :]:
                if word.strip("\"'").startswith("./"):
                    found.setdefault(word)
    return list(found)


def _decoded_escapes(text: str) -> str:
    """``text`` with JSON ``\\uXXXX`` and shell ``\\xXX`` escapes decoded, so an escaped 'allow' still reads."""
    if "\\" not in text:
        return text
    return _CHAR_ESCAPE_RE.sub(lambda match: chr(int(match.group(1) or match.group(2), 16)), text)


def _code_text(text: str) -> str:
    """``text`` without ``#`` comments: each line up to a ``#`` that starts a word outside quotes.

    A line whose quoting does not balance is kept whole, so a quote that spans
    lines never hides code. Linear in the text.
    """
    if "#" not in text:
        return text
    lines: list[str] = []
    for line in text.split("\n"):
        code_end = _CODE_PART_RE.match(line).end()  # type: ignore[union-attr]  # '*+' always matches
        comment = line.startswith("#", code_end) or line[code_end : code_end + 1].isspace()
        lines.append(line[:code_end] if comment and code_end < len(line) else line)
    return "\n".join(lines)


def _base64_texts(text: str) -> Iterator[str]:
    """Text decoded from the first base64 runs in ``text`` (an allow piped through ``base64 -d``)."""
    for index, match in enumerate(_BASE64_RUN_RE.finditer(text)):
        if index >= _MAX_BASE64_RUNS:
            return
        run = match.group(0)
        try:
            decoded = base64.b64decode(run + "=" * (-len(run) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        yield decoded.decode("utf-8", errors="replace")


def _allow_shapes(text: str) -> frozenset[str]:
    """The approval output shapes (``claude``, ``cursor``) ``text`` emits with an allow value.

    ``#`` comments are not read (a comment that mentions the allow string is not a
    decision), JSON and shell character escapes are decoded, and base64 runs are
    decoded too when the text decodes base64 at run time.
    """
    code = _code_text(text)
    return _allow_shapes_of(code, _decoded_escapes(code))


def _allow_shapes_of(code: str, decoded: str) -> frozenset[str]:
    """:func:`_allow_shapes` of comment-free ``code`` and its escape-decoded form ``decoded``."""
    decodes_base64 = ("64" in code or "atob" in code) and any(
        word in code.lower() for word in ("base64", "b64", "atob")
    )
    texts = [decoded, *(_base64_texts(code) if decodes_base64 else ())]
    return frozenset(
        shape
        for shape, patterns in _ALLOW_SHAPES.items()
        if any(pattern.search(candidate) for candidate in texts for pattern in patterns)
    )


def _context_outputs(decoded: str) -> frozenset[str]:
    """The JSON context outputs (``additionalContext``, a replaced tool output, a block reason) in comment-free,
    escape-decoded hook text."""
    if not ("additionalContext" in decoded or "ToolOutput" in decoded or "block" in decoded):
        return frozenset()
    return frozenset(label for label, pattern in _CONTEXT_OUTPUT_RES if pattern.search(decoded))


@dataclass(frozen=True)
class _Downloads:
    """Files a text saves downloads to, and directories it unpacks downloads into."""

    paths: frozenset[str] = frozenset()
    names: frozenset[str] = frozenset()
    bare: frozenset[str] = frozenset()
    dirs: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.paths or self.dirs)

    def runs_any(self, facts: Iterable[_ShellFacts]) -> bool:
        """Whether a file one of ``facts`` runs is a downloaded file.

        Files match by normalized path, or by base name when either side names
        the file without a directory (``curl -O https://h/i.sh && bash ./i.sh``);
        a run below a directory a download was unpacked into matches too. Each
        test is a set intersection, so the cost is linear in the sites.
        """
        if not self:
            return False
        for fact in facts:
            sites = fact.run_sites
            if not (
                self.paths.isdisjoint(sites.paths)
                and self.bare.isdisjoint(sites.names)
                and self.names.isdisjoint(sites.bare_names)
                and self.dirs.isdisjoint(sites.directories)
            ):
                return True
        return False


@dataclass(frozen=True)
class _RunSites:
    """The files one shell text runs, indexed for set lookups against downloads."""

    paths: frozenset[str] = frozenset()
    # Base names of every run, and of the runs named without a directory.
    names: frozenset[str] = frozenset()
    bare_names: frozenset[str] = frozenset()
    # Every directory a run is below (``_directory_prefixes``), as unpack-directory keys.
    directories: frozenset[str] = frozenset()
    # The text runs more than MAX_RUN_SITES distinct files; the rest were not indexed.
    truncated: bool = False


def _directory_prefixes(path: str) -> Iterator[str]:
    """The unpack-directory keys a normalized run ``path`` is below.

    ``.`` for a relative path, ``/`` for an absolute one, then each leading
    directory up to ``_MAX_DIR_DEPTH`` deep: ``/a/b/x`` gives ``/``, ``/a``,
    and ``/a/b``.
    """
    if path.startswith("/"):
        yield "/"
    elif not path.startswith(("$", "~")):
        yield "."
    end = path.find("/", 1)
    for _depth in range(_MAX_DIR_DEPTH):
        if end == -1:
            return
        yield path[:end]
        end = path.find("/", end + 1)


def _directory_key(directory: str) -> str:
    """A normalized unpack directory as a ``_directory_prefixes`` key (cut to ``_MAX_DIR_DEPTH`` deep)."""
    key = directory.rstrip("/") or "/"
    if key in {".", "/"}:
        return key
    end = key.find("/", 1)
    for _depth in range(_MAX_DIR_DEPTH - 1):
        if end == -1:
            return key
        end = key.find("/", end + 1)
    return key if end == -1 else key[:end]


@dataclass(frozen=True)
class _PackageRun:
    """A package runner (``npx``, ``uvx``, ``deno run``, ...) that runs an unpinned package."""

    detail: str
    # The package comes from a git or URL spec (github:user/repo, git+https://, https://...), not a registry.
    remote: bool
    # The version or tag names a moving channel (@latest, :main): blocking, like MCP and LSP servers.
    floating: bool = False


class _ShellFacts:
    """What one shell text (a hook command or a script) does, for the hook risk findings.

    Run sites are found only when something needs them (a download in this or
    another hook, or a run from the plugin's data directory), so a plugin that
    downloads nothing never pays for that scan.
    """

    def __init__(
        self,
        text: str,
        *,
        remote_code: bool = False,
        allow_shapes: frozenset[str] = frozenset(),
        downloads: _Downloads | None = None,
        packages: tuple[_PackageRun, ...] = (),
        context_outputs: frozenset[str] = frozenset(),
    ) -> None:
        self._text = text
        self.remote_code = remote_code
        self.allow_shapes = allow_shapes
        self.context_outputs = context_outputs
        self.downloads = downloads if downloads is not None else _Downloads()
        self.packages = packages
        # HookAnalyzer state: whether these run sites are indexed for cross-hook matching, and (for a script,
        # which many hooks can run) how far the plugin-wide indexes were matched against this text.
        self.indexed = False
        self.cross: _CrossState | None = None

    @functools.cached_property
    def variables(self) -> dict[str, str]:
        """The text's simple variable assignments, expanded in order (``D="$CLAUDE_PLUGIN_DATA/bin"``)."""
        return _simple_assignments(self._text)

    @functools.cached_property
    def run_sites(self) -> _RunSites:
        return _run_sites(self._text, self.variables)

    @property
    def runs_truncated(self) -> bool:
        """Whether the run sites were needed and the text runs more than ``MAX_RUN_SITES`` files."""
        sites = self.__dict__.get("run_sites")
        return sites is not None and sites.truncated

    @property
    def unshipped_runs(self) -> list[str]:
        """Run sites in the plugin's data directory: code the plugin does not ship."""
        if not any(ref[1:] in self._text for ref in _PLUGIN_DATA_REFS):
            return []
        prefixes = tuple(f"{ref}/" for ref in _PLUGIN_DATA_REFS)
        return sorted(path for path in self.run_sites.paths if path.startswith(prefixes))


def _shell_facts(text: str) -> _ShellFacts:
    """What shell ``text`` does; ``remote_code`` says whether it runs downloaded content.

    Covers a fetch piped into an interpreter that reads its program from
    standard input, possibly through other stages (``curl … | sh``,
    ``wget -qO- … | tee x | /usr/bin/env bash``, ``| $SHELL``, ``| busybox sh``,
    ``| source /dev/stdin``), a fetch in a command substitution or here-string
    (``bash -c "$(curl …)"``, ``sh <(wget …)``, ``source /dev/stdin <<< "$(curl …)"``),
    a variable that holds a download and is evaluated (``c=$(curl …); eval "$c"``),
    Python or JavaScript that evaluates a download (``exec(urlopen(…).read())``),
    PowerShell's ``iex (iwr …)``, a download to a file (or unpacked into a
    directory) that the text also runs (``curl -o /tmp/x … && sh /tmp/x``), and a
    package runner of a git or URL spec (``npx github:user/repo``,
    ``deno run https://…``). An interpreter that gets its program from ``-m`` or
    a script path, or from ``-c`` / ``-e`` with a program that does not
    evaluate its standard input, only reads the download as data
    (``curl … | python3 -m json.tool``), so it does not count; a ``-c`` / ``-e``
    program that does (``| python3 -c 'exec(sys.stdin.read())'``,
    ``| bash -c "$(cat)"``) does. Linear in the text.
    """
    commands = [match.group(0) for match in _SHELL_COMMAND_RE.finditer(text)]
    packages = _package_runs(commands)
    remote_code = _EXEC_FETCHED_RE.search(text) is not None or any(package.remote for package in packages)
    downloads = _Downloads()
    code = _code_text(text)
    decoded = _decoded_escapes(code)
    facts = _ShellFacts(
        text, allow_shapes=_allow_shapes_of(code, decoded), packages=packages, context_outputs=_context_outputs(decoded)
    )
    if _FETCHER_RE.search(text) is not None:  # every other remote-code shape needs a fetch
        fetched = _fetched_variables(text)
        downloads = _download_sites(commands, facts.variables)
        remote_code = (
            remote_code
            or _pipes_fetch_into_interpreter(commands, fetched)
            or (_has_substitution(text) and _SUBSTITUTION_RE.search(text) is not None)
            or _runs_here_string(text, fetched)
            or _runs_fetched_variable(text, fetched)
            or any(_invokes_fetched_string(command) for command in commands)
            or downloads.runs_any([facts])
        )
    facts.remote_code = remote_code
    facts.downloads = downloads
    return facts


def _has_substitution(text: str) -> bool:
    return "$(" in text or "<(" in text or "`" in text


def _fetched_variables(text: str) -> frozenset[str]:
    """Names of shell variables assigned downloaded content (``code=$(curl …)``)."""
    if "=" not in text or not _has_substitution(text):
        return frozenset()
    return frozenset(match.group("var") for match in _FETCH_ASSIGNMENT_RE.finditer(text))


def _uses_variable(text: str, names: frozenset[str]) -> bool:
    return bool(names) and any(match.group(1) in names for match in _VARIABLE_REF_RE.finditer(text))


def _runs_here_string(text: str, fetched: frozenset[str]) -> bool:
    if "<<<" not in text:
        return False
    return any(match.group("fetch") or match.group("var") in fetched for match in _HERE_STRING_RE.finditer(text))


def _runs_fetched_variable(text: str, fetched: frozenset[str]) -> bool:
    return bool(fetched) and any(match.group("var") in fetched for match in _VARIABLE_RUN_RE.finditer(text))


def _pipes_fetch_into_interpreter(commands: list[str], fetched_variables: frozenset[str] = frozenset()) -> bool:
    """A pipeline stage that fetches, followed by a later stage that runs its standard input as a program."""
    for command in commands:
        if "|" not in command:
            continue
        fetched = False
        for stage in command.split("|"):
            if fetched and _stage_runs_stdin(stage):
                return True
            fetched = fetched or _FETCHER_RE.search(stage) is not None or _uses_variable(stage, fetched_variables)
    return False


# Per interpreter family: option letters that take the program from an argument (not standard input),
# options whose next word is their value, and option prefixes that carry an attached value.
_PROGRAM_LETTERS = {"shell": "c", "python": "cm", "perl": "eE", "ruby": "e", "php": "rRBEFf"}
_VALUE_OPTIONS: dict[str, frozenset[str]] = {
    "shell": frozenset({"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}),
    "python": frozenset({"-W", "-X"}),
    "node": frozenset({"-r", "--require", "--import", "--loader", "--experimental-loader", "--input-type"}),
    "php": frozenset({"-c", "-d", "-z"}),
}
_ATTACHED_VALUE_PREFIXES: dict[str, tuple[str, ...]] = {
    "python": ("-W", "-X"),
    "perl": ("-M", "-m", "-I"),
    "ruby": ("-r", "-I", "-E"),
    "php": ("-d", "-c", "-z"),
}
_NODE_PROGRAM_OPTIONS = frozenset({"-e", "-p", "--eval", "--print"})
_POWERSHELL_PROGRAM_OPTIONS = frozenset({"-c", "-command", "-f", "-file", "-e", "-ec", "-encodedcommand"})
_POWERSHELL_COMMAND_OPTIONS = frozenset({"-c", "-command"})
_STDIN_WORDS = frozenset({"-", "/dev/stdin", "/dev/fd/0", "/proc/self/fd/0"})
# Option letters whose program is a file, not program text (python -m module, php -f file).
_FILE_PROGRAM_LETTERS = {"python": "m", "php": "Ff"}
# Option letters that loop the program over standard input lines (perl -n / -p, ruby -n / -p, php -R).
_LINE_LOOP_LETTERS = {"perl": "np", "ruby": "np"}
_LINE_LOOP_PROGRAM_LETTERS = {"php": "R"}
_MAX_PROGRAM_CHARS = 4096
# A -c / -e program that evaluates its standard input runs the download: code that evaluates text ...
_EVALUATES_RES: dict[str, re.Pattern[str]] = {
    "shell": re.compile(r"(?<![\w.$-])(?:eval|source)(?![\w.-])|(?:^|[;&|({\s])\.[ \t]"),
    "python": re.compile(r"(?<![\w.])(?:exec|eval|execfile)\s*\("),
    "node": re.compile(r"(?<![\w.$])(?:eval|Function)\s*\(|\bnew\s+Function\b|\bvm\s*\.\s*run\w*|\brunIn\w*Context\b"),
    "perl": re.compile(r"(?<![\w$@%&])eval\b"),
    "ruby": re.compile(r"(?<![\w.$@:])(?:eval|instance_eval|class_eval|module_eval|instance_exec)\b"),
    "php": re.compile(r"(?<![\w$>:])(?:eval\s*\(|(?:include|require)(?:_once)?\b)", re.IGNORECASE),
    "powershell": re.compile(
        r"(?<![\w-])(?:iex|invoke-expression)(?![\w-])|\[scriptblock\]\s*::\s*create", re.IGNORECASE
    ),
}
# ... together with code that reads standard input: sys.stdin, process.stdin, <STDIN>, readFileSync(0),
# /dev/stdin, php://stdin, $input, ARGF, gets, input(), and a shell's $(cat) or 'read'.
_READS_STDIN_RE = re.compile(
    r"\bstdin\b|\bSTDIN\b|<>|/dev/stdin|/dev/fd/0|/proc/self/fd/0|readFileSync\s*\(\s*0\b"
    r"|(?<![\w.])open\s*\(\s*0\b|(?<![\w.])input\s*\(|\$input\b|(?i:\[console\]\s*::\s*in\b)|\bARGF\b"
    r"|(?<![\w.$])gets\b|\$argn\b|\$\(\s*cat(?:\s+-)?\s*\)|`\s*cat(?:\s+-)?\s*`"
    r"|(?:^|[;&|({\s])read(?:\s|$)"
)
# A shell program that is standard input itself: bash -c "$(cat)".
_STDIN_PROGRAM_RE = re.compile(
    r"\s*(?:\$\(\s*(?:cat(?:\s+(?:-|/dev/stdin|/dev/fd/0))?|<\s*/dev/(?:stdin|fd/0))\s*\)|`\s*cat(?:\s+-)?\s*`)"
)


def _interpreter_family(name: str) -> str:
    if name.startswith("python"):
        return "python"
    if name in {"node", "nodejs"}:
        return "node"
    if name in {"perl", "ruby", "php"}:
        return name
    if name in {"pwsh", "powershell"}:
        return "powershell"
    if name in {"iex", "invoke-expression"}:
        return "evaluator"
    return "shell"


def _stage_runs_stdin(stage: str, depth: int = 0) -> bool:
    """Whether a pipeline stage runs its standard input as a program ('| sh', '| python3 -', '| iex')."""
    if _STAGE_SOURCE_STDIN_RE.match(stage):
        return True
    match = _STAGE_INTERPRETER_RE.match(stage)
    if match is None:
        return False
    head = match.group("interpreter")
    if _XARGS_RE.search(head):
        return True  # xargs hands the downloaded text to the interpreter as arguments
    name = head.split()[-1].strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower().strip("${}")
    return _program_from_stdin(_interpreter_family(name), _stage_arguments(stage[match.end() :][:2048]), depth)


# A redirection operator at the start of an unquoted word: '>', '>>', '2>', '&>', '>&2', '<&0', '{fd}>', '<<<'.
_REDIRECTION_RE = re.compile(r"(?:\d+|\{[A-Za-z_]\w*\})?(?:&>>|&>|>>|>&|>\||<>|<<<|<<-|<<|<&|>|<)")


def _stage_arguments(text: str) -> list[str]:
    """The arguments a pipeline stage passes its interpreter, as the shell reads them.

    Quotes are removed. An unquoted ``#`` at the start of a word ends the
    command (``| sh # install``), and redirections (``> /dev/null``,
    ``2>&1``, ``<&0``) are not arguments, so neither reads as a script path.
    A quoted ``'#x'`` or ``'>'`` stays an argument. Linear in the text.
    """
    words: list[tuple[str, int]] = []  # (word without quotes, length of its leading unquoted part)
    chars: list[str] = []
    plain = 0  # leading characters of the word that were neither quoted nor escaped
    in_word = False
    unquoted = True
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        index += 1
        if quote:
            if char == quote:
                quote = ""
            elif char == "\\" and quote == '"' and index < len(text) and text[index] in '"\\$`':
                chars.append(text[index])
                index += 1
            else:
                chars.append(char)
        elif char in " \t\r\n":
            if in_word:
                words.append(("".join(chars), plain))
            chars, plain, in_word, unquoted = [], 0, False, True
        elif char == "#" and not in_word:
            break  # a comment: the rest of the line is not part of the command
        elif char in "'\"":
            quote, in_word, unquoted = char, True, False
        elif char == "\\":
            in_word, unquoted = True, False
            if index < len(text):
                chars.append(text[index])
                index += 1
        else:
            chars.append(char)
            in_word = True
            plain += unquoted
    if in_word:
        words.append(("".join(chars), plain))
    arguments: list[str] = []
    target = False
    for word, plain in words:
        if target:  # the file a bare '>' or '2>' redirects to
            target = False
            continue
        operator = _REDIRECTION_RE.match(word)
        if operator is not None and operator.end() <= plain:
            target = operator.end() == len(word)
            continue
        arguments.append(word)
    return arguments


def _program_from_stdin(family: str, words: list[str], depth: int = 0) -> bool:
    """Whether an interpreter with these arguments runs its standard input as a program.

    It does when it reads its program from standard input (no program
    argument, ``-``, ``bash -s``), or when the program it gets from ``-c``,
    ``-e``, ``--eval``, or ``-Command`` evaluates standard input
    (:func:`_program_runs_stdin`).
    """
    if family == "evaluator":
        return True
    if family == "powershell":
        lowered = [word.lower() for word in words]
        for index, word in enumerate(lowered):
            if word in _POWERSHELL_PROGRAM_OPTIONS:
                if index + 1 >= len(words):
                    return False
                if lowered[index + 1] == "-":
                    return True
                program = " ".join(words[index + 1 :])
                return word in _POWERSHELL_COMMAND_OPTIONS and _program_runs_stdin(family, program, depth)
            if not word.startswith("-"):
                return False
        return True
    value_options = _VALUE_OPTIONS.get(family, frozenset())
    attached = _ATTACHED_VALUE_PREFIXES.get(family, ())
    letters = _PROGRAM_LETTERS.get(family, "")
    line_loop = False
    skip = False
    options_done = False
    for index, word in enumerate(words):
        if skip:
            skip = False
            continue
        if word in _STDIN_WORDS:
            return True
        if not options_done and word == "--":
            options_done = True
            continue
        if options_done or not word.startswith(("-", "+")) or len(word) < 2:
            return False  # a script path or module: the program is a file, and the download only data
        if family == "shell" and word.startswith("-") and not word.startswith("--") and "s" in word[1:]:
            return True  # 'bash -s': read the program from standard input
        if word in value_options:
            skip = True
            continue
        if attached and word.startswith(attached) and len(word) > 2:
            continue
        following = words[index + 1] if index + 1 < len(words) else ""
        if family == "node" and word.split("=", 1)[0] in _NODE_PROGRAM_OPTIONS:
            program = word.split("=", 1)[1] if "=" in word else following
            return _program_runs_stdin(family, program, depth)
        if word.startswith("--"):
            continue
        flags = word[1:]
        line_loop = line_loop or any(letter in _LINE_LOOP_LETTERS.get(family, "") for letter in flags)
        position = next((at for at, letter in enumerate(flags) if letter in letters), None)
        if position is None:
            continue
        letter = flags[position]
        if letter in _FILE_PROGRAM_LETTERS.get(family, ""):
            return False  # python -m module, php -f file: the program is a file
        line_loop = line_loop or letter in _LINE_LOOP_PROGRAM_LETTERS.get(family, "")
        attached_program = flags[position + 1 :] if family != "shell" else ""
        return _program_runs_stdin(family, attached_program or following, depth, line_loop=line_loop)
    return True


def _program_runs_stdin(family: str, program: str, depth: int, *, line_loop: bool = False) -> bool:
    """Whether a program passed as an argument evaluates its standard input (the download).

    True for code that evaluates text it reads from standard input
    (``exec(sys.stdin.read())``, ``eval(readFileSync(0))``, ``eval <STDIN>``,
    ``eval "$(cat)"``; for ``perl -n``, ``ruby -n``, and ``php -R`` every
    input line is read, so an ``eval`` is enough), for a shell program that is
    standard input itself (``bash -c "$(cat)"``), and for a shell program
    whose commands start an interpreter that reads the inherited standard input
    (``sh -c 'exec bash'``; followed one level).
    """
    program = program[:_MAX_PROGRAM_CHARS]
    evaluates = _EVALUATES_RES.get(family)
    if evaluates is not None and evaluates.search(program) and (line_loop or _READS_STDIN_RE.search(program)):
        return True
    if family != "shell":
        return False
    if _STDIN_PROGRAM_RE.match(program):
        return True
    if depth > 0:
        return False
    return any(
        _stage_runs_stdin(command.group(0).split("|", 1)[0], depth + 1)
        for command in _SHELL_COMMAND_RE.finditer(program)
    )


def _invokes_fetched_string(command: str) -> bool:
    invoke = _INVOKE_EXPRESSION_RE.search(command)
    return invoke is not None and _POWERSHELL_FETCH_RE.search(command, invoke.end()) is not None


def _expanded(value: str, variables: dict[str, str]) -> str:
    """``value`` with the ``$NAME`` / ``${NAME}`` references to known ``variables`` replaced (bounded)."""
    if not variables or "$" not in value:
        return value
    result = _VARIABLE_USE_RE.sub(lambda match: variables.get(match.group(1) or match.group(2), match.group(0)), value)
    return result if len(result) <= _MAX_EXPANDED_CHARS else value


def _simple_assignments(text: str) -> dict[str, str]:
    """Simple ``NAME=value`` assignments in ``text``, each value expanded with the earlier ones (bounded)."""
    variables: dict[str, str] = {}
    if "=" not in text:
        return variables
    for match in _SIMPLE_ASSIGNMENT_RE.finditer(text):
        name = match.group("name")
        if name in variables or len(variables) < _MAX_ASSIGNMENTS:
            variables[name] = _expanded(match.group("value"), variables)
    return variables


def _root_default_chain(value: str, names: frozenset[str]) -> tuple[str, int] | None:
    """``(first variable, length)`` of the parameter-default chain that starts ``value`` when every alternative
    in it is one of ``names`` (``${CLAUDE_PLUGIN_ROOT:-${PLUGIN_ROOT}}``, ``${CLAUDE_PLUGIN_ROOT:-}``), else ``None``.

    The chain expands to the first variable that is set, so it names the plugin root only when each alternative
    does: a literal or another variable (``${CLAUDE_PLUGIN_ROOT:-/tmp/x}``, ``${CLAUDE_PLUGIN_ROOT:-$HOME}``) can
    make it any directory. An empty default only keeps an unset variable empty, like the plain placeholder.
    """
    first: str | None = None
    position = depth = 0
    while (opened := _DEFAULT_OPEN_RE.match(value, position)) is not None:
        if opened.group(1) not in names or depth == _MAX_DEFAULT_DEPTH:
            return None
        first = first or opened.group(1)
        depth += 1
        position = opened.end()
    if first is None:
        return None
    plain = _PLAIN_VARIABLE_RE.match(value, position)
    if plain is not None:
        if (plain.group(1) or plain.group(2)) not in names:
            return None
        position = plain.end()
    end = position + depth
    return (first, end) if value[position:end] == "}" * depth else None


def _file_key(raw: str, variables: dict[str, str] | None = None) -> tuple[str, str] | None:
    """``(normalized path, base name)`` of a file argument; ``None`` for stdout, devices, and empty names.

    Simple variables the same text assigns are expanded first, so ``-o "$D/tool"``
    after ``D="$CLAUDE_PLUGIN_DATA/bin"`` is ``$CLAUDE_PLUGIN_DATA/bin/tool``.
    """
    value = raw
    if "://" in value:  # a URL stands for the file curl -O and wget name after its last path segment
        value = value.split("#", 1)[0].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    if variables:
        value = _expanded(value, variables)
    if not value or value == "-" or value.startswith("/dev/"):
        return None
    path = posixpath.normpath(_BRACED_VARIABLE_RE.sub(r"$\1", value))
    return path, posixpath.basename(path)


def _download_sites(commands: list[str], variables: dict[str, str] | None = None) -> _Downloads:
    """Files the commands save downloads to, and directories they unpack a download into (in order)."""
    paths: set[str] = set()
    names: set[str] = set()
    bare: set[str] = set()
    dirs: set[str] = set()
    for command in commands:
        downloader = _DOWNLOADER_RE.search(command)
        if downloader is not None:
            outputs = [match.group("file") for match in _OUTPUT_FILE_RE.finditer(command, downloader.start())]
            if _WGET_RE.search(command):
                outputs.extend(match.group(0) for match in _URL_IN_TEXT_RE.finditer(command))
            for output in outputs:
                key = _file_key(output, variables)
                if key is not None:
                    paths.add(key[0])
                    names.add(key[1])
                    if "/" not in key[0]:
                        bare.add(key[1])
        if _EXTRACTOR_RE.search(command):
            directory = _extracted_directory(command, paths, names, variables)
            if directory is not None:
                dirs.add(_directory_key(directory))
    return _Downloads(frozenset(paths), frozenset(names), frozenset(bare), frozenset(dirs))


def _extracted_directory(
    command: str, paths: set[str], names: set[str], variables: dict[str, str] | None = None
) -> str | None:
    """Where an archive tool unpacks a download (``tar xzf x.tgz -C dir``, ``curl … | tar xz``), or ``None``."""
    piped = False  # an earlier stage of the pipeline fetches
    for stage in command.split("|"):
        fetches = _FETCHER_RE.search(stage) is not None
        words = _split_words(stage[:2048]) if _EXTRACTOR_RE.search(stage) else []
        start = next((i for i, word in enumerate(words) if word.rsplit("/", 1)[-1] in {"tar", "bsdtar", "unzip"}), None)
        if start is not None:
            args = words[start + 1 :]
            archive = any(
                (key := _file_key(arg, variables)) is not None
                and (key[0] in paths or key[0] in names or key[1] in paths or key[1] in names)
                for arg in args
            )
            if piped or archive:
                directory = "."
                for position, arg in enumerate(args):
                    if arg in {"-C", "--directory", "-d"} and position + 1 < len(args):
                        directory = args[position + 1]
                    elif arg.startswith("--directory="):
                        directory = arg.split("=", 1)[1]
                key = _file_key(directory, variables)
                return key[0] if key is not None else "."
        piped = piped or fetches
    return None


def _run_sites(text: str, variables: dict[str, str] | None = None) -> _RunSites:
    """Every file the text runs (by interpreter, 'source', or as a command), at most ``MAX_RUN_SITES``."""
    keys: dict[tuple[str, str], None] = {}
    truncated = False
    for match in _RUN_FILE_RE.finditer(text):
        key = _file_key(match.group("script") or match.group("command"), variables)
        if key is None or key in keys:
            continue
        if len(keys) >= MAX_RUN_SITES:
            truncated = True
            break
        keys[key] = None
    paths = frozenset(path for path, _name in keys)
    return _RunSites(
        paths=paths,
        names=frozenset(name for _path, name in keys),
        bare_names=frozenset(name for path, name in keys if "/" not in path),
        directories=frozenset(prefix for path in paths for prefix in _directory_prefixes(path)),
        truncated=truncated,
    )


# Launchers the pinning classifier knows (package runners, container runtimes, and the wrappers around them).
_PACKAGE_RUN_HINT_RE = re.compile(
    r"\b(?:npx|bunx|pnpx|bun|pnpm|yarn|npm|uvx|uv|pipx|deno|go|dnx|docker|podman|nerdctl)\b", re.IGNORECASE
)
# Shell words that start a command list without being the command ('then npx ...', '! npx ...').
_SHELL_LIST_WORDS = frozenset({"then", "do", "else", "elif", "if", "while", "until", "!", "(", "{"})
_MAX_SHELL_NESTING = 2


def _package_runs(commands: list[str], depth: int = 0) -> tuple[_PackageRun, ...]:
    """Package runners and container runs (``npx``, ``pnpm dlx``, ``uvx``, ``docker run``, ...) of unpinned code.

    Each command is classified like an MCP server command, so hooks get the
    same rules: wrappers (``env X=1``, ``timeout 30``, ``sudo``, ``cmd /c``) are
    looked through, a ``bash -c '<program>'`` is read command by command, exact
    versions and digests are pinned, and a moving tag (``@latest``, ``:main``)
    is ``floating``. A git or URL spec without a commit is ``remote``.
    """
    runs: list[_PackageRun] = []
    for command in commands:
        if not _PACKAGE_RUN_HINT_RE.search(command):
            continue
        for stage in command.split("|"):
            words = _split_words(stage[:4096])
            index = 0
            while index < len(words) and (
                words[index] in _SHELL_LIST_WORDS or _ASSIGNMENT_WORD_RE.match(words[index]) is not None
            ):
                index += 1
            argv = unwrap_launch_command(words[index:], expand_shell=False)
            if not argv:
                continue
            program = shell_program(argv)
            if program is not None:
                if depth < _MAX_SHELL_NESTING:
                    inner = [match.group(0) for match in _SHELL_COMMAND_RE.finditer(program[:4096])]
                    runs.extend(_package_runs(inner, depth + 1))
            else:
                pin = classify_mcp_pinning({"command": argv[0], "args": argv[1:]})
                if pin.status == "unpinned":
                    runs.append(_PackageRun(_bounded(pin.detail, 160), pin.remote, pin.floating))
            if len(runs) >= MAX_OUTSIDE_REFS:
                return tuple(runs[:MAX_OUTSIDE_REFS])
    return tuple(runs)


def _url_parts(url: str) -> tuple[str, str, int | None, str] | None:
    """``(scheme, host, effective port, normalized path)`` of an absolute URL, or ``None`` when unparseable.

    The URL is read the way Claude Code's HTTP client (WHATWG) reads it, so a
    backslash in the authority or path cannot move the host or escape a prefix.
    """
    try:
        parsed = urlsplit(whatwg_url(url))
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if not scheme or not host:
        return None
    path = posixpath.normpath("/" + unquote(parsed.path).lstrip("/"))
    return scheme, host.rstrip(".").lower(), port or DEFAULT_PORTS.get(scheme), path


def hook_url_entry_problem(entry: str) -> str | None:
    """Why a ``hooks.allowed_urls`` URL entry can never match, or ``None`` (also for a host pattern).

    The matcher compares only the origin and path, so an entry with userinfo, a
    query, or a fragment would admit more than it names. Such an entry matches
    nothing (the policy loader warns about it) and implies no allowed host.
    """
    if "://" not in entry:
        return None
    try:
        parsed = urlsplit(entry.strip())
        _port = parsed.port
    except ValueError as exc:
        return f"is not a valid URL ({exc})"
    if not parsed.scheme or not parsed.hostname:
        return "has no scheme or host"
    if "@" in parsed.netloc or parsed.query or parsed.fragment:
        return "has userinfo, a query, or a fragment"
    return None


_UrlParts = tuple[str, str, int | None, str]


@dataclass(frozen=True)
class _HookUrlAllowlist:
    """``hooks.allowed_urls``, parsed once: URL prefixes, and host patterns read like ``mcp.allowed_private_hosts``."""

    # Usable URL prefixes as _url_parts; an entry with userinfo, a query, or a fragment admits nothing.
    prefixes: tuple[_UrlParts, ...] = ()
    hosts: HostAllowlist = field(default_factory=HostAllowlist)

    @classmethod
    def from_entries(cls, entries: Iterable[str]) -> _HookUrlAllowlist:
        prefixes: list[_UrlParts] = []
        hosts: list[str] = []
        for raw in entries:
            entry = raw.strip()
            if "://" not in entry:
                hosts.append(entry)
            elif hook_url_entry_problem(entry) is None and (parts := _url_parts(entry)) is not None:
                prefixes.append(parts)
        return cls(tuple(prefixes), HostAllowlist.from_entries(hosts))

    def matches(self, url: str, host: str | None, endpoint: EndpointClass | None) -> bool:
        """Whether a hook may post to ``url``, whose client host is ``host`` and static class ``endpoint``.

        A URL prefix needs the same scheme, host, and effective port (so
        ``https://hooks.example.com`` does not admit ``https://hooks.example.com.evil.net``
        or ``https://hooks.example.com@evil.net``), and the path must be the
        prefix's path or below it on a ``/`` segment boundary (``/hooks`` admits
        ``/hooks/x`` but not ``/hooksx``); a prefix without a path admits every
        path. A host pattern never admits a cloud metadata host.
        """
        target = _url_parts(url) if self.prefixes else None
        if target is not None and any(_url_is_under(target, prefix) for prefix in self.prefixes):
            return True
        return host is not None and self.hosts.allows_host(host, endpoint)


def _url_is_under(target: _UrlParts, prefix: _UrlParts) -> bool:
    if target[:3] != prefix[:3]:
        return False
    base = prefix[3].rstrip("/")
    return not base or target[3] == base or target[3].startswith(base + "/")


def hook_allowlist_hosts(allowed_urls: Iterable[str]) -> list[str]:
    """Host patterns implied by ``hooks.allowed_urls`` (usable URL prefixes contribute their host)."""
    hosts: list[str] = []
    for raw in allowed_urls:
        entry = raw.strip()
        if "://" in entry:
            if hook_url_entry_problem(entry) is not None:
                continue
            try:
                parsed_host = urlparse(whatwg_url(entry)).hostname
            except ValueError:
                parsed_host = None
            if parsed_host:
                hosts.append(parsed_host)
        elif entry:
            hosts.append(entry)
    return hosts


def _command_url_credentials(text: str) -> bool:
    """Whether a command line embeds credentials in a URL (``https://user:password@host``, ``?token=...``)."""
    return any(url_credentials(match.group(0), userinfo_rule="secret") for match in _URL_IN_TEXT_RE.finditer(text))


def _header_secret(key: str, value: str) -> bool:
    """Inline credential in an HTTP hook header (``$VAR`` interpolation is allowed)."""
    stripped = value.strip()
    if not stripped or is_env_reference(stripped):
        return False
    if _ENV_REF_ANYWHERE_RE.search(stripped):
        literal = _ENV_REF_ANYWHERE_RE.sub("", stripped)
        literal = _AUTH_SCHEME_PREFIX_RE.sub("", literal).strip()
        return bool(literal) and looks_like_inline_secret("value", literal)
    return looks_like_inline_secret(key, stripped)


@dataclass
class HookAnalysis:
    records: list[HookRecord] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


@dataclass
class _HookSite:
    """One hook handler under analysis: its record, where its findings go, and what it may approve."""

    record: HookRecord
    analysis: HookAnalysis
    # The message prefix that locates the handler: "hook PreToolUse (matcher 'Bash') in hooks/hooks.json".
    where: str
    # The file path findings show.
    display: str
    dialect: HookDialect
    # What an approval hook can approve (HookAnalyzer._scope): all, bash, scoped, or narrow; None for other events.
    scope: str | None
    # The write, fetch, or MCP tools a "scoped" approval hook covers.
    scoped_tools: tuple[str, ...] = ()

    def report(self, flag: str, severity: Severity, check: str, message: str, suggestion: str) -> None:
        """Flag the handler (each flag once) and add a finding attributed to it."""
        self.record.add_flag(flag)
        self.analysis.findings.append(
            component_finding(
                severity,
                check,
                f"{self.where}: {message}",
                self.display,
                suggestion,
                component=("hook", self.record.source),
                extra={"hook_id": self.record.id, "hook_event": self.record.event},
            )
        )


@dataclass(frozen=True)
class _ScriptEvidence:
    """What one plugin script a hook runs does; each script is read and scanned once."""

    facts: _ShellFacts
    # Root-relative paths of other plugin scripts this one names; followed one more level.
    references: tuple[PurePosixPath, ...] = ()
    # Why not every script this one names is followed (it names too many files).
    note: str | None = None


class _SiteIndex:
    """Plugin-wide sites (normalized paths or directories), each with the first hook that had it, in order."""

    def __init__(self) -> None:
        self.records: dict[str, HookRecord] = {}
        self.order: list[str] = []

    def __bool__(self) -> bool:
        return bool(self.records)

    def __len__(self) -> int:
        return len(self.order)

    def add(self, keys: Iterable[str], record: HookRecord) -> None:
        for key in keys:
            if key not in self.records:
                self.records[key] = record
                self.order.append(key)

    def first(self, keys: frozenset[str], start: int = 0) -> HookRecord | None:
        """The hook of one of ``keys`` among the sites added from position ``start`` on (by set lookups)."""
        if not keys or start >= len(self.order):
            return None
        if start == 0 and len(keys) <= len(self.order):
            hits = self.records.keys() & keys
        else:
            hits = keys.intersection(self.order[start:])
        return self.records[min(hits)] if hits else None


@dataclass
class _CrossState:
    """Cross-hook matching of one script's facts, which every hook that runs the script shares."""

    # How many entries of each plugin-wide index were already matched against the script.
    downloaded: int = 0
    unpacked: int = 0
    ran: int = 0
    ran_under: int = 0
    # The hook the script was found to share a download with, once found.
    hit: HookRecord | None = None
    # Whether the script's downloads are in the plugin-wide index.
    registered: bool = False


# A path relative to the running script's own directory: "$(dirname "$0")/x", "${0%/*}/x", "$SCRIPT_DIR/x".
_SCRIPT_DIR_REF_RE = re.compile(
    r"(?:\$\(\s*dirname\s+[\"']?\$\{?(?:0|BASH_SOURCE(?:\[0\])?)\}?[\"']?\s*\)|\$\{0%/\*\}"
    r"|\$\{?[A-Za-z_]{0,32}(?:DIR|HERE)[A-Za-z_]{0,32}\}?)[\"']?/(?P<path>[^\s\"'`;&|<>(){}$]{1,256})"
)
# A script file a shell or Python script names ("approve.py", 'hooks/check.sh').
_SCRIPT_NAME_RE = re.compile(
    r"(?<![\w./$-])(?P<path>(?:\.{1,2}/)?[\w.-]{1,128}(?:/[\w.-]{1,128}){0,8}\.(?:sh|bash|zsh|py|js|mjs|cjs|ts|rb|pl|php|ps1))"
    r"(?![\w.-])"
)
# A command that runs something: an interpreter, exec/source, a subprocess call, or a ./path command. A script
# reference counts only in such a command (not in an echo of usage text), or as the command itself.
_EXEC_CONTEXT_RE = re.compile(
    r"(?<![\w.$-])(?:(?:ba|z|da|k)?sh|python[0-9.]{0,8}|node|perl|ruby|exec|source|subprocess|system|spawn|"
    r"execFile|Popen)\b|__file__|(?:^|[;&|(])[ \t]*\.{1,2}/|(?:^|[;&|(])[ \t]*\.[ \t]",
    re.MULTILINE,
)
# One pipeline stage of a line: quoted strings and escapes stay inside it; '|', ';', '&', and newlines end it.
_SHELL_STAGE_RE = re.compile(r"""(?:'[^']*'|"(?:[^"\\]|\\.)*"|\\.|[^'"\\|;&\n])+""")
# The code part of a line, up to a '#' comment (a '#' that starts a word outside quotes).
_CODE_PART_RE = re.compile(r"""(?:'[^']*'|"(?:[^"\\]|\\.)*"|\\.|[^'"\\\s#]|(?<=\S)#|\s(?!#))*+""")
# A command that only prints or tests the paths it names ('echo', 'printf', 'print(', 'test', '[', '[[', ':').
_MENTION_COMMAND_RE = re.compile(
    r"[ \t]*(?:(?:then|do|else|if|elif|while|until|!|\{|\()[ \t]+)*+(?:echo|printf|print|test|\[\[?|:)(?![\w.-])"
)
# What may come before a path a command runs: wrappers, options, assignments, and an opening quote or command
# substitution ('then "$R/x.sh"', 'exec sudo -E "$R/x"', 'out=$("$R/x")', 'TOOL="$R/x"').
_COMMAND_POSITION_RE = re.compile(
    r"[ \t]*(?:(?:exec|sudo|doas|env|nohup|command|time|nice|then|do|else|if|elif|while|until|!|\{|\()[ \t]+"
    r"|-\S{0,64}[ \t]+|[A-Za-z_]\w{0,63}=|\$\(|`)*+[\"']?"
)
_MAX_COMMAND_PREFIX = 256
_SHELL_OR_PYTHON_SHEBANG_RE = re.compile(r"#![^\n]{0,128}\b(?:(?:ba|z|da|k)?sh|python[0-9.]{0,8})\b")
_PYTHON_SHEBANG_RE = re.compile(r"#![^\n]{0,128}\bpython[0-9.]{0,8}\b")
# A Python import statement at the start of a line: 'import core.rules as r', 'from core.rule_engine import X'.
_PYTHON_IMPORT_RE = re.compile(
    r"^[ \t]*(?:from[ \t]+(?P<package>[A-Za-z_][\w.]{0,255})[ \t]+import[ \t]+(?P<names>[\w \t,()]{1,512})"
    r"|import[ \t]+(?P<modules>[\w \t,.]{1,512}))",
    re.MULTILINE,
)
_NAMING_SUFFIXES = frozenset({"", ".sh", ".bash", ".zsh", ".py"})
_SENSITIVE_TOOL_LABELS = {"mcp__server__tool": "MCP tools (mcp__*)", "apply_patch": "apply_patch (Codex file edits)"}


def _python_import_paths(text: str) -> list[str]:
    """Files of the modules a Python script imports, other than the standard library's.

    ``from core.rule_engine import RuleEngine`` names ``core/rule_engine.py`` or
    its package's ``core/rule_engine/__init__.py``, and ``core/rule_engine/RuleEngine.py``
    in case the name is a submodule; ``import a.b as c`` names ``a/b.py`` or ``a/b/__init__.py``.
    """
    paths: dict[str, None] = {}
    for match in _PYTHON_IMPORT_RE.finditer(text):
        package = match.group("package")
        listed = match.group("modules") if package is None else match.group("names").strip("() \t")
        names = [part.split()[0] for part in listed.split(",") if part.split()]
        for module in names if package is None else [package]:
            if module.split(".")[0] in sys.stdlib_module_names:
                continue
            path = module.replace(".", "/")
            paths.update(dict.fromkeys((f"{path}.py", f"{path}/__init__.py")))
            if package is not None:
                paths.update(dict.fromkeys(f"{path}/{name}.py" for name in names))
    return list(paths)


class _RunReferences:
    """Decides whether a script reference in a script's text is one the script runs.

    A reference runs when it is not in a ``#`` comment, the pipeline stage
    around it does not only print or test it (``echo``, ``printf``, ``test``,
    ``[``), and that stage runs something (an interpreter, exec, source, a
    subprocess call) or has the reference as its command. Each line is split
    and each stage classified once, so the cost is linear in the text.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self._start = self._end = -1
        self._comment = -1
        self._stages: list[tuple[int, int]] = []
        self._verdicts: dict[int, tuple[bool, bool]] = {}

    def _load(self, position: int) -> None:
        if self._start <= position < self._end:
            return
        text = self.text
        self._start = text.rfind("\n", 0, position) + 1
        end = text.find("\n", position)
        self._end = len(text) if end == -1 else end
        line = text[self._start : self._end]
        code_end = _CODE_PART_RE.match(line).end()  # type: ignore[union-attr]  # '*+' always matches
        comment = line.startswith("#", code_end) or (line[code_end : code_end + 1].isspace())
        self._comment = code_end if comment and code_end < len(line) else len(line)
        self._stages = [(match.start(), match.end()) for match in _SHELL_STAGE_RE.finditer(line)]
        self._verdicts = {}

    def runs(self, position: int) -> bool:
        self._load(position)
        offset = position - self._start
        if offset >= self._comment:
            return False
        index = bisect.bisect_right(self._stages, (offset, len(self.text))) - 1
        if index < 0 or not self._stages[index][0] <= offset < self._stages[index][1]:
            return False
        start, end = self._stages[index]
        verdict = self._verdicts.get(start)
        if verdict is None:
            stage = self.text[self._start + start : self._start + end]
            verdict = (_MENTION_COMMAND_RE.match(stage) is not None, _EXEC_CONTEXT_RE.search(stage) is not None)
            self._verdicts[start] = verdict
        mention, executes = verdict
        if mention:
            return False
        if executes:
            return True
        prefix = self.text[self._start + start : position]
        return len(prefix) <= _MAX_COMMAND_PREFIX and _COMMAND_POSITION_RE.fullmatch(prefix) is not None


class HookAnalyzer:
    """Analyze every handler of a plugin's hooks config sources.

    ``read_script`` returns the whole text of a script under the plugin root
    (at most ``MAX_SCRIPT_BYTES``), ``None`` when nothing is there, and raises
    :class:`HookScriptUnreadable` when something is there that cannot be read
    safely (a link, a special file, or a script over the size or read budget).
    At most ``MAX_SCRIPT_READS`` scripts are read (missing paths do not count).
    A script's text is scanned whole, and the plugin scripts it runs (named by
    ``${CLAUDE_PLUGIN_ROOT}/...``, ``$(dirname "$0")/...``, or a script file
    name, but not in a comment or an ``echo`` / ``test``) are followed one more
    level. A hook whose script is unreadable or past that cap, whose texts run
    more than ``MAX_RUN_SITES`` files while the plugin downloads something, or an
    approval hook that names the plugin root but runs no script that could be
    read, is flagged ``script_unanalyzed``; a config past the ``MAX_HOOK_*``
    limits ``scan_truncated``: evidence is never dropped silently.

    Downloads and run sites are matched across every handler and script the
    analyzer sees, so a file one hook downloads and another runs is remote code.

    ``root_prefixes`` are the manifest format's root placeholders
    (``${CLAUDE_PLUGIN_ROOT}`` is always recognized), braced or not. With
    ``relative_scripts`` (Cursor), relative command tokens such as
    ``./approve.sh`` are also read, against the plugin root and against the
    hooks file's directory; after ``cd <root placeholder>``, relative tokens are
    read against that directory in every format.
    """

    def __init__(
        self,
        *,
        read_script: Callable[[PurePosixPath], str | None],
        hook_allowed_urls: Iterable[str] = (),
        allowed_private_hosts: Iterable[str] = (),
        root_prefixes: Iterable[str] = (),
        relative_scripts: bool = False,
    ) -> None:
        self.read_script = read_script
        self.hook_allowed_urls = tuple(hook_allowed_urls)
        self._allowed_urls = _HookUrlAllowlist.from_entries(self.hook_allowed_urls)
        # Private hosts a hook may post to: mcp.allowed_private_hosts and the hosts hooks.allowed_urls names.
        self._private_hosts = HostAllowlist.from_entries(
            (*allowed_private_hosts, *hook_allowlist_hosts(self.hook_allowed_urls))
        )
        self.root_refs = _plugin_root_refs(root_prefixes)
        # The root placeholders' variable names, for parameter-default chains ('${CLAUDE_PLUGIN_ROOT:-$PLUGIN_ROOT}').
        self._root_names = frozenset(ref[2:-1] for ref in self.root_refs if ref.startswith("${") and ref.endswith("}"))
        self._root_ref_re = re.compile(
            "(?:"
            + "|".join(re.escape(ref) for ref in sorted(self.root_refs, key=len, reverse=True))
            + r")(?P<path>/[^\s\"'`;&|<>(){}$]{1,256})"
        )
        # A root placeholder's variable, also in a parameter expansion ('${CLAUDE_PLUGIN_ROOT:-}'), but not a longer
        # variable name that starts with it ('${CLAUDE_PLUGIN_ROOT_BACKUP}').
        self._root_name_re = re.compile(
            "(?:" + "|".join(re.escape(ref.removesuffix("}")) for ref in self.root_refs) + ")(?![A-Za-z0-9_])"
        )
        self.relative_scripts = relative_scripts
        self._hook_dir = PurePosixPath()
        # Per script: its evidence, None when nothing is there, or why it could not be read.
        self._scripts: dict[PurePosixPath, _ScriptEvidence | str | None] = {}
        self._script_reads = 0
        # Plugin-wide download and run sites (normalized paths), for cross-handler remote code: files
        # downloaded, directories unpacked into, files run, and every directory a run is below.
        self._downloaded = _SiteIndex()
        self._unpacked = _SiteIndex()
        self._ran = _SiteIndex()
        self._ran_under = _SiteIndex()
        self._pending: list[tuple[HookRecord, list[_ShellFacts]]] = []
        # Whether one script runs a file another (or the same) script downloads, per pair of script facts.
        self._pair_hits: dict[tuple[int, int], bool] = {}

    # -- scripts ----------------------------------------------------------- #
    def _script_paths(self, token: str, cwd: PurePosixPath | None = None) -> list[PurePosixPath]:
        """Root-relative paths of the plugin script a command token may name (none when it escapes the root or
        still names a variable)."""
        value = token.strip("\"'")
        if _root_ref(value, self.root_refs) is not None:
            rel, escapes = _plugin_root_path(value, self.root_refs)
            # A path that still names a variable ('scripts/$X.sh') is not the file with that literal name.
            return [rel] if rel is not None and not escapes and "$" not in rel.as_posix() else []
        if not (self.relative_scripts or cwd is not None):
            return []
        if not _RELATIVE_SCRIPT_RE.match(value) or not {"/", "."} & set(value):
            return []
        bases = [cwd] if cwd is not None else []
        if self.relative_scripts:
            bases.extend((PurePosixPath(), self._hook_dir))
        paths: list[PurePosixPath] = []
        for base in dict.fromkeys(bases):
            rel, escapes = _contained_path((base / value.replace("\\", "/")).as_posix())
            if rel is not None and not escapes:
                paths.append(rel)
        return paths

    def _root_directory(self, token: str) -> PurePosixPath | None:
        """The root-relative directory a ``cd`` target names, or ``None`` when it is not under the plugin root or
        still names a variable."""
        value = token.strip("\"'")
        if _root_ref(value, self.root_refs) is None:
            return None
        rel, escapes = _plugin_root_path(value, self.root_refs)
        if escapes or (rel is not None and "$" in rel.as_posix()):
            return None
        return rel if rel is not None else PurePosixPath()

    def _plain_root(self, value: str) -> str:
        """``value`` with a leading root default chain (``${CLAUDE_PLUGIN_ROOT:-${PLUGIN_ROOT}}/x``) written as
        its first placeholder (``${CLAUDE_PLUGIN_ROOT}/x``); any other value unchanged."""
        chain = _root_default_chain(value, self._root_names)
        return value if chain is None else f"${{{chain[0]}}}{value[chain[1] :]}"

    def _rooted(self, word: str, aliases: dict[str, str]) -> str:
        """A command word (quotes removed) with the command's root variables expanded and a root default chain
        made plain, so ``"$R/x.sh"`` after ``R="${CLAUDE_PLUGIN_ROOT:-}"`` reads ``${CLAUDE_PLUGIN_ROOT}/x.sh``."""
        value = word.strip("\"'")
        if aliases and "$" in value:
            value = _expanded(value, aliases)
        return self._plain_root(value)

    def _root_aliases(self, text: str) -> dict[str, str]:
        """Variables a hook command sets to the plugin root or a path under it, each with its plain placeholder.

        Reading ``$R/x.sh`` as a plugin script clears ``script_unanalyzed``, so only an assignment that fixes
        ``R`` for the rest of the command counts: one statement of its own (``R="${CLAUDE_PLUGIN_ROOT}";``, not an
        env prefix, a pipeline stage, a background job, or after ``&&`` / ``||``), before every use, of a variable
        the shell does not set itself, that the text names nowhere else but as ``$R`` / ``${R}`` outside single
        quotes. A command with a subshell, group, command substitution, heredoc, ``eval``, or ``source`` resolves
        no variable: those can set one without naming it. Nor does one with a compound command (``if``, ``while``,
        ``for``, ``case``): an assignment inside it may never run.
        """
        if "=" not in text or "$" not in text or _OPAQUE_VARIABLES_RE.search(text):
            return {}
        single_quoted: set[str] = set()
        for segment in _QUOTING_SEGMENT_RE.finditer(text):
            part = segment.group(0)
            if part in {"'", '"'}:
                return {}  # unbalanced quoting
            if part.startswith("'"):
                single_quoted.update(use.group(1) or use.group(2) for use in _VARIABLE_USE_RE.finditer(part))
        mentions = Counter(match.group(0) for match in _NAME_WORD_RE.finditer(text))
        uses: Counter[str] = Counter()
        first_use: dict[str, int] = {}
        for use in _VARIABLE_USE_RE.finditer(text):
            name = use.group(1) or use.group(2)
            uses[name] += 1
            first_use.setdefault(name, use.start())
        # Per name: (value, end of the assignment) for its one standalone assignment, None for any other.
        candidates: dict[str, tuple[str, int] | None] = {}
        previous = 0
        for command in _SHELL_COMMAND_RE.finditer(text):
            words = command.group(0).split(maxsplit=1)
            if words and words[0] in _COMPOUND_COMMAND_WORDS:
                return {}
            gap, previous = text[previous : command.start()], command.end()
            assignment = _ROOT_ALIAS_ASSIGNMENT_RE.fullmatch(command.group(0))
            if assignment is None:
                continue
            name, follows = assignment.group("name"), text[command.end() : command.end() + 2]
            standalone = set(gap) <= {";", "\n"} and not (follows.startswith("&") and follows != "&&")
            candidates[name] = (
                (assignment.group("value"), command.end()) if standalone and name not in candidates else None
            )
        aliases: dict[str, str] = {}
        for name, candidate in candidates.items():
            if (
                candidate is None
                or name in _SHELL_SET_VARIABLES
                or name.startswith("BASH")
                or name in self._root_names
                or name in single_quoted
                or mentions[name] != uses[name] + 1
                or first_use.get(name, len(text)) < candidate[1]
            ):
                continue
            value = self._plain_root(_expanded(candidate[0], aliases))
            if _root_ref(value, self.root_refs) is not None:
                aliases[name] = value
        return aliases

    def _first_level_paths(self, tokens: list[str], aliases: dict[str, str] | None = None) -> list[PurePosixPath]:
        """Script paths a command names: quoted command strings are split once more, relative paths after
        ``cd <root placeholder>`` resolve against that directory, and *aliases* (the command's root variables,
        :meth:`_root_aliases`) and root default chains read as the plain placeholder."""
        words: list[str] = []
        for token in tokens:
            words.append(token)
            if any(char.isspace() for char in token.strip()):
                words.extend(_split_words(token))  # 'bash -c "cd ${CLAUDE_PLUGIN_ROOT} && ./approve.sh"'
        aliases = aliases or {}
        paths: list[PurePosixPath] = []
        cwd: PurePosixPath | None = None
        for index, word in enumerate(words):
            if word in {"cd", "pushd"} and index + 1 < len(words):
                cwd = self._root_directory(self._rooted(words[index + 1], aliases))
                continue
            paths.extend(self._script_paths(self._rooted(word, aliases), cwd))
        return list(dict.fromkeys(paths))

    def _script_evidence(
        self, tokens: list[str], aliases: dict[str, str] | None = None
    ) -> tuple[list[_ShellFacts], list[str], bool]:
        """Facts of the plugin scripts a command runs (and of the scripts they name, one level), why any of
        them was not analyzed, and whether any script was found at all."""
        facts: list[_ShellFacts] = []
        unanalyzed: list[str] = []
        first = self._first_level_paths(tokens, aliases)
        seen = set(first)
        second: list[PurePosixPath] = []
        found = False
        for rel in first:
            evidence = self._evidence(rel, unanalyzed)
            if evidence is None:
                continue
            found = True
            facts.append(evidence.facts)
            if evidence.note:
                unanalyzed.append(evidence.note)
            for reference in evidence.references:
                if reference not in seen:
                    seen.add(reference)
                    second.append(reference)
        for rel in second:
            evidence = self._evidence(rel, unanalyzed)
            if evidence is not None:
                facts.append(evidence.facts)
        return facts, unanalyzed, found or bool(unanalyzed)

    def _evidence(self, rel: PurePosixPath, unanalyzed: list[str]) -> _ScriptEvidence | None:
        if rel not in self._scripts:
            if self._script_reads >= MAX_SCRIPT_READS:
                unanalyzed.append(
                    f"'{_bounded(rel.as_posix(), 80)}' was not read: the plugin's hooks reference more than "
                    f"{MAX_SCRIPT_READS} scripts"
                )
                return None
            self._scripts[rel] = self._load_script(rel)
        cached = self._scripts[rel]
        if isinstance(cached, str):
            unanalyzed.append(cached)
            return None
        return cached

    def _load_script(self, rel: PurePosixPath) -> _ScriptEvidence | str | None:
        try:
            text = self.read_script(rel)
        except HookScriptUnreadable as exc:
            return _bounded(str(exc), 160) or f"'{_bounded(rel.as_posix(), 80)}' could not be read safely"
        if text is None:
            return None
        self._script_reads += 1
        if len(text) > MAX_SCRIPT_BYTES:
            return f"'{_bounded(rel.as_posix(), 80)}' is larger than the {MAX_SCRIPT_BYTES}-byte scan limit"
        references, note = self._script_references(text, rel)
        facts = _shell_facts(text)
        facts.cross = _CrossState()  # a script's facts are shared by every hook that runs it
        return _ScriptEvidence(facts, references, note)

    def _script_references(self, text: str, rel: PurePosixPath) -> tuple[tuple[PurePosixPath, ...], str | None]:
        """Plugin scripts a script runs: named by root placeholder, relative to its own directory, or (in shell
        and Python scripts) by a script file name, against its directory and the plugin root; a Python script's
        imported modules count too. A name in a comment, or one a command only prints or tests (``echo``,
        ``[ -f … ]``), is not followed."""
        base = rel.parent
        found: dict[PurePosixPath, None] = {}
        runs = _RunReferences(text)
        candidates: list[str] = [
            match.group("path") for match in self._root_ref_re.finditer(text) if runs.runs(match.start())
        ]
        relative: list[str] = [
            match.group("path") for match in _SCRIPT_DIR_REF_RE.finditer(text) if runs.runs(match.start())
        ]
        if rel.suffix.lower() in _NAMING_SUFFIXES or _SHELL_OR_PYTHON_SHEBANG_RE.match(text[:256]):
            relative.extend(match.group("path") for match in _SCRIPT_NAME_RE.finditer(text) if runs.runs(match.start()))
        if rel.suffix.lower() == ".py" or _PYTHON_SHEBANG_RE.match(text[:256]):
            relative.extend(_python_import_paths(text))
        for raw in dict.fromkeys(candidates):
            path, escapes = _contained_path(raw)
            if path is not None and not escapes:
                found.setdefault(path)
        for raw in dict.fromkeys(relative):
            for directory in dict.fromkeys((base, PurePosixPath())):
                path, escapes = _contained_path((directory / raw).as_posix())
                if path is not None and not escapes:
                    found.setdefault(path)
        found.pop(rel, None)
        paths = tuple(found)
        if len(paths) > MAX_SCRIPT_REFERENCES:
            note = (
                f"'{_bounded(rel.as_posix(), 80)}' names more than {MAX_SCRIPT_REFERENCES} other files, so the "
                "scripts it runs were not all followed"
            )
            return paths[:MAX_SCRIPT_REFERENCES], note
        return paths, None

    def _names_root(self, text: str) -> bool:
        """Whether ``text`` names a root placeholder, also in a parameter expansion (``${CLAUDE_PLUGIN_ROOT:-}``)."""
        return self._root_name_re.search(text) is not None

    # -- cross-handler download and run sites ------------------------------ #
    def _runs_download(self, facts: list[_ShellFacts]) -> bool:
        """Whether one of ``facts`` runs a file one of them downloads; each pair of scripts is decided once."""
        downloaders = [fact for fact in facts if fact.downloads]
        for downloader in downloaders:
            for runner in facts:
                shared = downloader.cross is not None and runner.cross is not None
                key = (id(runner), id(downloader))
                hit = self._pair_hits.get(key) if shared else None
                if hit is None:
                    hit = downloader.downloads.runs_any([runner])
                    if shared:
                        self._pair_hits[key] = hit
                if hit:
                    return True
        return False

    def _cross_handler_download(self, record: HookRecord, facts: list[_ShellFacts]) -> tuple[str | None, bool]:
        """The id of another hook that downloads a file this one runs (or runs a file this one downloads),
        and whether another hook's run sites indexed now were cut at ``MAX_RUN_SITES``.

        Only whole normalized paths (and directories a download was unpacked
        into, other than the working directory) match across hooks. Run sites
        are indexed only once the plugin downloads something, each text once.
        Every lookup is a hash lookup, and a script many hooks run is matched
        only against the sites added since it was last matched, so many hooks
        and large scripts stay linear.
        """
        if not (any(fact.downloads for fact in facts) or self._downloaded or self._unpacked):
            self._pending.append((record, facts))
            return None, False
        truncated = False
        for pending_record, pending_facts in self._pending:
            truncated = self._index_runs(pending_record, pending_facts) or truncated
        self._pending.clear()
        other = next((hit for fact in facts if (hit := self._cross_hit(fact)) is not None), None)
        for fact in facts:
            if fact.cross is None or not fact.cross.registered:
                self._downloaded.add(fact.downloads.paths, record)
                self._unpacked.add(fact.downloads.dirs - {"."}, record)
                if fact.cross is not None:
                    fact.cross.registered = True
        self._index_runs(record, facts)
        if other is None or other is record:
            return None, truncated
        other.add_flag("remote_code")  # the other half of the pair runs remote code too
        return other.id, truncated

    def _cross_hit(self, fact: _ShellFacts) -> HookRecord | None:
        """An earlier hook that downloads a file ``fact`` runs (or unpacks into a directory above it), or that
        runs a file ``fact`` downloads (or one below a directory it unpacks into)."""
        state = fact.cross
        if state is not None and state.hit is not None:
            return state.hit
        sites = fact.run_sites
        hit = (
            self._downloaded.first(sites.paths, state.downloaded if state else 0)
            or self._unpacked.first(sites.directories, state.unpacked if state else 0)
            or self._ran.first(fact.downloads.paths, state.ran if state else 0)
            or self._ran_under.first(fact.downloads.dirs - {"."}, state.ran_under if state else 0)
        )
        if state is not None:  # a shared script: next time, match only the sites added after this
            state.downloaded, state.unpacked = len(self._downloaded), len(self._unpacked)
            state.ran, state.ran_under = len(self._ran), len(self._ran_under)
            state.hit = hit
        return hit

    def _index_runs(self, record: HookRecord, facts: list[_ShellFacts]) -> bool:
        """Index the run sites of ``facts`` not indexed yet; whether any of those was cut at ``MAX_RUN_SITES``."""
        truncated = False
        for fact in facts:
            if fact.indexed:
                continue
            fact.indexed = True
            sites = fact.run_sites
            truncated = truncated or sites.truncated
            self._ran.add(sites.paths, record)
            self._ran_under.add(sites.directories, record)
        return truncated

    # -- handlers ---------------------------------------------------------- #
    def analyze(
        self, config: Any, *, source: str, file: str, display: str, dialect: HookDialect = CLAUDE_HOOKS
    ) -> HookAnalysis:
        analysis = HookAnalysis()
        self._hook_dir = PurePosixPath(file).parent
        truncated: list[str] = []
        for event, group_index, matcher, handler_index, handler in iter_hook_handlers(config, truncated):
            hook_id = f"{source}#{event}[{group_index}].hooks[{handler_index}]"
            if not isinstance(handler, dict):
                analysis.records.append(HookRecord(hook_id, source, file, event, matcher, "invalid", "", ["invalid"]))
                continue
            raw_type = handler.get("type")
            handler_type = raw_type if isinstance(raw_type, str) and raw_type in HANDLER_TYPES else "unknown"
            record = HookRecord(hook_id, source, file, event, matcher, handler_type, "")
            if event not in dialect.events:
                record.add_flag("unknown_event")
            monitor = dialect is MONITOR_HOOKS
            where = "monitor command" if monitor else f"hook {event}"
            if matcher:
                where += f" (matcher {matcher[:60]!r})"
            condition = handler.get("if") if isinstance(handler.get("if"), str) else None
            if condition is not None:
                where += f" (if {_bounded(condition, 60)!r})"
            where += f" in {source}"
            scope, scoped_tools = self._scope(dialect, event, matcher, condition)
            site = _HookSite(record, analysis, where, display, dialect, scope, scoped_tools)

            if handler_type == "command":
                self._command_hook(handler, site)
            elif handler_type == "http":
                self._http_hook(handler, site)
            elif handler_type == "mcp_tool":
                server = handler.get("server") if isinstance(handler.get("server"), str) else ""
                tool = handler.get("tool") if isinstance(handler.get("tool"), str) else ""
                record.target = _bounded(f"{server}/{tool}")
                if scope in {"all", "bash"}:
                    self._remote_approval(site, "an MCP tool")
            elif handler_type in {"prompt", "agent"}:
                prompt = handler.get("prompt") if isinstance(handler.get("prompt"), str) else ""
                record.target = _bounded(prompt, 80)

            runs_here = _SESSION_START_HANDLER_TYPES if event == "SessionStart" else _CONTEXT_HANDLER_TYPES
            if (
                event in dialect.context_events
                and handler_type in runs_here
                and "context_injection" not in record.risk_flags
            ):
                injected = (
                    "every line the monitor prints is sent to the model while it runs"
                    if monitor
                    else f"the {handler_type} handler's output is injected into the agent's context on every {event}"
                )
                site.report(
                    "context_injection",
                    Severity.LOW,
                    "plugin_hook_context_injection",
                    injected,
                    "Review what the hook emits; keep injected context minimal and never derived from untrusted "
                    "remote content.",
                )
            analysis.records.append(record)
        if truncated:
            self._scan_truncated(analysis, list(dict.fromkeys(truncated)), source=source, file=file, display=display)
        return analysis

    @staticmethod
    def _scope(
        dialect: HookDialect, event: str, matcher: str | None, condition: str | None = None
    ) -> tuple[str | None, tuple[str, ...]]:
        """What an approval hook can approve: ``all``, ``bash``, ``scoped`` (write, fetch, or MCP tools,
        listed), or ``narrow``; ``None`` for an event that approves nothing.

        A handler's ``if`` condition (a permission rule such as ``Bash(git status)``)
        narrows what the matcher selects (:func:`_condition_scope`).
        """
        if event not in dialect.approval_events:
            return None, ()
        fixed = dialect.event_scope(event)
        if fixed == "mcp":
            scope, tools = "scoped", ("MCP tool calls",)
        elif fixed is not None:
            scope, tools = fixed, ()
        else:
            scope, tools = matcher_scope(matcher, dialect.shell_tools), ()
            if scope == "narrow":
                sensitive = matcher_sensitive_tools(matcher)
                if sensitive:
                    scope, tools = "scoped", tuple(_SENSITIVE_TOOL_LABELS.get(tool, tool) for tool in sensitive)
        if condition is None or scope == "narrow":
            return scope, tools
        narrowed = _condition_scope(condition, dialect.shell_tools)
        if narrowed is None:
            return scope, tools  # a condition the analyzer does not model keeps the matcher's scope
        if scope == "all":
            return narrowed
        if narrowed[0] == "bash":
            return narrowed if scope == "bash" else ("narrow", ())
        if narrowed[0] == "scoped":
            # The handler runs for the condition's tool only when the matcher (or the event) selects it too.
            # A matcher that names a shell tool and a write tool ('Edit|Bash' with if 'Edit') still approves Edit.
            tool = _condition_tool(condition)
            if fixed == "mcp":
                selected = tool.startswith("mcp__")
            elif fixed is not None:
                selected = False
            else:
                selected = _matcher_selects(matcher, tool)
            if selected:
                return narrowed
        return "narrow", ()

    def _scan_truncated(
        self, analysis: HookAnalysis, limits: list[str], *, source: str, file: str, display: str
    ) -> None:
        """Record that ``source`` has handlers past the scan limits: a ``truncated`` row and a blocking finding."""
        reason = "; ".join(limits)
        analysis.records.append(
            HookRecord(
                f"{source}#truncated",
                source,
                file,
                "*",
                None,
                "truncated",
                _bounded(f"handlers not analyzed: {reason}"),
                ["scan_truncated"],
            )
        )
        analysis.findings.append(
            component_finding(
                Severity.HIGH,
                "plugin_hook_scan_truncated",
                f"hooks in {source}: {reason}; the handlers past the static scan limits were not analyzed, so padding "
                "can hide a risky hook",
                display,
                f"Keep a hooks config within {MAX_HOOK_EVENTS} events, {MAX_HOOK_GROUPS} matcher groups per event, and "
                f"{MAX_HOOK_HANDLERS} handlers; review the rest by hand, then override with severity_overrides "
                "PLUGIN_SCHEMA.plugin_hook_scan_truncated if intended.",
                component=("hook", source),
                extra={
                    "hook_scan_limits": {
                        "events": MAX_HOOK_EVENTS,
                        "groups_per_event": MAX_HOOK_GROUPS,
                        "handlers": MAX_HOOK_HANDLERS,
                    }
                },
            )
        )

    def _command_hook(self, handler: dict[str, Any], site: _HookSite) -> None:
        record = site.record
        text = _command_text(handler, site.dialect.command_keys)
        record.target = _bounded(text)
        tokens = _command_tokens(handler, site.dialect.command_keys)
        scripts, unanalyzed, found_script = self._script_evidence(tokens, self._root_aliases(text))
        facts = [_shell_facts(text), *scripts]
        local = any(fact.remote_code for fact in facts) or self._runs_download(facts)
        other, others_truncated = self._cross_handler_download(record, facts)
        if local or other is not None:
            how = (
                "fetches remote content and executes it (for example 'curl ... | sh')"
                if local
                else f"runs a file that another hook ({_bounded(str(other), 120)}) downloads, or downloads a file "
                "that hook runs"
            )
            site.report(
                "remote_code",
                Severity.CRITICAL,
                "plugin_hook_remote_code",
                f"the command {how}, so the code that runs is not part of the reviewed plugin",
                "Ship the script inside the plugin and run it from ${CLAUDE_PLUGIN_ROOT}; never pipe downloads "
                "into an interpreter or run downloaded files.",
            )
        else:
            self._unshipped_code(facts, site)
            if others_truncated or any(fact.runs_truncated for fact in facts):
                unanalyzed.append(
                    f"it or another hook runs more than {MAX_RUN_SITES} distinct files, so not every run was "
                    "matched against the plugin's downloads"
                )
        self._auto_approve(facts, site)
        self._context_output(facts, site)
        if site.scope in {"all", "bash", "scoped"} and not found_script and self._names_root(text):
            unanalyzed.append("it names the plugin root, but no plugin script it runs could be found and read")
        if unanalyzed:
            # HIGH, not lower: an unread script can hide an auto-approval (HIGH) or remote code (CRITICAL).
            site.report(
                "script_unanalyzed",
                Severity.HIGH,
                "plugin_hook_script_unanalyzed",
                f"the command runs plugin scripts that were not analyzed ({'; '.join(unanalyzed[:5])}), so an "
                "auto-approval or remote code in them would go unreported",
                "Ship hook scripts as regular files (no symlinks) under ${CLAUDE_PLUGIN_ROOT}, run them by "
                "their ${CLAUDE_PLUGIN_ROOT} path, and keep their number and size small; review the script, "
                "then override with severity_overrides PLUGIN_SCHEMA.plugin_hook_script_unanalyzed if intended.",
            )
        if _command_url_credentials(text):
            site.report(
                "inline_secret",
                Severity.CRITICAL,
                "plugin_hook_inline_secret",
                "the command embeds a credential in a URL (user:password@, or a credential query or fragment "
                "parameter)",
                "Remove the credential from the hook command; read it from an environment variable or a "
                "credential helper when the hook runs.",
            )
        outside = []
        for token in tokens:
            reason = _outside_root_reference(token, self.root_refs)
            if reason and len(outside) < MAX_OUTSIDE_REFS:
                outside.append(f"{_bounded(token, 80)} ({reason})")
        if not self.relative_scripts:
            for path in _relative_run_paths(text, self.root_refs):
                if len(outside) < MAX_OUTSIDE_REFS:
                    outside.append(
                        f"{_bounded(path, 80)} (is a relative path, which the client resolves against the session's "
                        "working directory, the user's project, not the plugin root)"
                    )
        if outside:
            site.report(
                "outside_root",
                Severity.MEDIUM,
                "plugin_hook_outside_root",
                f"the command references files outside the plugin root: {'; '.join(outside)}",
                "Reference bundled files through ${CLAUDE_PLUGIN_ROOT}; do not read or execute files the "
                "plugin does not ship.",
            )

    @staticmethod
    def _unshipped_code(facts: list[_ShellFacts], site: _HookSite) -> None:
        """Findings for code a hook runs that the plugin does not ship (and that is not remote code).

        A package or image with a floating version or tag is HIGH, like an MCP or
        LSP server's, and is not reported again as unpinned; other unpinned
        packages and code run from the plugin's data directory are MEDIUM.
        """
        packages = [package for fact in facts for package in fact.packages]
        floating = [package for package in packages if package.floating]
        if floating:
            # Same rule as MCP and LSP servers: a moving tag blocks, and is not reported again as unpinned.
            site.report(
                "unpinned_package",
                Severity.HIGH,
                "plugin_hook_command_floating_version",
                "the command runs a package or image with a floating version or tag "
                f"({'; '.join(dict.fromkeys(package.detail for package in floating[:3]))}); each run may fetch "
                "different code",
                "Pin the package or image to an exact version or digest, not a moving tag such as latest.",
            )
            packages = [package for package in packages if not package.floating]
        if packages:
            details = "; ".join(dict.fromkeys(package.detail for package in packages[:3]))
            site.report(
                "unpinned_package",
                Severity.MEDIUM,
                "plugin_hook_unpinned_package",
                f"the command runs a package that is not pinned to an exact version ({details}); each run may "
                "fetch different code",
                "Pin the package to an exact version (pkg@1.2.3, pkg==1.2.3), or ship the code in the plugin.",
            )
        unshipped = sorted({path for fact in facts for path in fact.unshipped_runs})
        if unshipped:
            shown = ", ".join(_bounded(path, 80) for path in unshipped[:MAX_OUTSIDE_REFS])
            site.report(
                "unshipped_code",
                Severity.MEDIUM,
                "plugin_hook_runs_unshipped_code",
                f"the command runs code from the plugin's data directory ({shown}); the plugin does not ship that "
                "code, so it was not reviewed",
                "Run code the plugin ships under ${CLAUDE_PLUGIN_ROOT}; keep ${CLAUDE_PLUGIN_DATA} for data.",
            )

    @staticmethod
    def _auto_approve(facts: list[_ShellFacts], site: _HookSite) -> None:
        shapes = frozenset().union(*(fact.allow_shapes for fact in facts))
        if site.scope is None or not shapes & set(site.dialect.allow_shapes):
            return
        site.record.add_flag("auto_approve")  # also for a narrow matcher, which gets no finding
        if site.scope in {"all", "bash"}:
            target = "every tool call" if site.scope == "all" else "shell commands"
            site.report(
                "auto_approve",
                Severity.HIGH,
                "plugin_hook_auto_approve",
                f"the command emits an allow decision for {target}, which skips the user's permission prompt",
                "Do not auto-approve tool calls from a plugin hook; narrow the matcher and return 'ask' or "
                "no decision.",
            )
        elif site.scope == "scoped":
            site.report(
                "auto_approve",
                Severity.MEDIUM,
                "plugin_hook_auto_approve_scoped",
                f"the command emits an allow decision for {', '.join(site.scoped_tools)}, which skips the user's "
                "permission prompt for file writes, web fetches, or MCP tool calls (and, for edits, the "
                "working-directory limit that acceptEdits keeps)",
                "Do not auto-approve write, fetch, or MCP tool calls from a plugin hook; return 'ask' or no decision.",
            )

    @staticmethod
    def _context_output(facts: list[_ShellFacts], site: _HookSite) -> None:
        """LOW when the command emits JSON that puts text in front of the model on an event whose plain output
        does not: ``additionalContext``, a replaced tool output, or a Stop or PostToolUse block reason."""
        event = site.record.event
        if event in site.dialect.context_events:
            return  # every output of a context event is already reported
        outputs = set().union(*(fact.context_outputs for fact in facts))
        if event not in _BLOCK_REASON_EVENTS:
            outputs.discard("a block reason")  # a PreToolUse block reason is a guard's denial message
        if not outputs:
            return
        shown = ", ".join(label for label, _pattern in _CONTEXT_OUTPUT_RES if label in outputs)
        site.report(
            "context_injection",
            Severity.LOW,
            "plugin_hook_context_injection",
            f"the command emits {shown}, which is added to the agent's context",
            "Review what the hook emits; keep injected context minimal and never derived from untrusted "
            "remote content.",
        )

    @staticmethod
    def _remote_approval(site: _HookSite, decider: str) -> None:
        site.report(
            "remote_approval",
            Severity.MEDIUM,
            "plugin_hook_remote_approval",
            f"{decider} decides this broad-matcher approval hook, so it can return permissionDecision: allow for "
            "tool calls without a user prompt",
            "Narrow the matcher, or keep approval decisions in reviewed plugin code.",
        )

    def _http_hook(self, handler: dict[str, Any], site: _HookSite) -> None:
        record = site.record
        url = handler.get("url")
        if not isinstance(url, str) or not url.strip():
            site.report(
                "invalid_url",
                Severity.HIGH,
                "plugin_hook_http_url_invalid",
                "the http handler has no 'url'",
                "Set 'url' to the https:// endpoint that receives the hook input.",
            )
            return
        record.target = safe_url(url)
        record.url = url.strip()
        # Claude Code posts the hook with a WHATWG URL parser (Node), which reads a
        # backslash as '/' and drops tabs and line breaks. Every check below reads
        # the URL that way; text that urllib reads differently is itself invalid.
        client_url = whatwg_url(url)
        problems = url_ambiguities(url, percent_in_host=True)
        if problems:
            site.report(
                "invalid_url",
                Severity.HIGH,
                "plugin_hook_http_url_invalid",
                f"the http handler url contains {', and '.join(problems)}, so URL parsers disagree on where it "
                f"points; Claude Code posts to {record.target!r}",
                "Write the URL as a plain https://host/path without backslashes, whitespace, control "
                "characters, or percent-encoding in the host.",
            )
        # Read from the URL text, so credentials are flagged even when the authority is malformed. Both
        # readings count: any userinfo Claude Code would send, and a literal password or token in the
        # raw text (committed with the plugin even when a backslash moves it out of the client's userinfo).
        if url_credentials(client_url, userinfo_rule="any") or url_credentials(record.url, userinfo_rule="secret"):
            site.report(
                "inline_secret",
                Severity.CRITICAL,
                "plugin_hook_inline_secret",
                "the http handler url embeds credentials (user:password, or a credential query or fragment parameter)",
                "Remove credentials from the URL; pass them through headers with $VAR interpolation and "
                "allowedEnvVars.",
            )
        try:
            parsed = urlparse(client_url)
            host = parsed.hostname
            _ = parsed.port
        except ValueError:
            site.report(
                "invalid_url",
                Severity.HIGH,
                "plugin_hook_http_url_invalid",
                f"the http handler url has a malformed authority: {record.target!r}",
                "Use a valid https://host[:port]/path URL.",
            )
            return
        scheme = (parsed.scheme or "").lower()
        endpoint = classify_endpoint_host(host) if host else None
        if scheme not in {"http", "https"} or not host:
            site.report(
                "invalid_url",
                Severity.HIGH,
                "plugin_hook_http_url_invalid",
                f"the http handler url {record.target!r} must be an http(s) URL with a host",
                "Use an https:// URL with a host.",
            )
            return
        if scheme == "http" and not (endpoint is not None and endpoint.is_loopback):
            site.report(
                "insecure_scheme",
                Severity.HIGH,
                "plugin_hook_http_insecure_scheme",
                f"hook input (tool arguments, prompts) is posted over plaintext http to {record.target!r}",
                "Use https:// for any non-loopback hook endpoint.",
            )
        headers = handler.get("headers")
        if isinstance(headers, dict):
            # Every header is read: the hook config is already size-bounded, and a cap would hide a later secret.
            for key, value in headers.items():
                if isinstance(value, str) and _header_secret(str(key), value):
                    site.report(
                        "inline_secret",
                        Severity.CRITICAL,
                        "plugin_hook_inline_secret",
                        f"header '{key}' carries an inline credential",
                        "Reference the secret as $VAR and list it in allowedEnvVars; never inline a credential.",
                    )
                    break
        allowlisted = bool(self.hook_allowed_urls) and self._allowed_urls.matches(client_url, host, endpoint)
        if endpoint is not None and endpoint.kind == "metadata":
            site.report(
                "metadata_endpoint",
                Severity.HIGH,
                "plugin_hook_http_endpoint_metadata",
                f"the http handler targets a cloud instance-metadata endpoint {record.target!r}",
                "Remove the instance-metadata endpoint; it can never be allowlisted.",
            )
        elif endpoint is not None and not allowlisted and not self._private_hosts.allows(endpoint):
            site.report(
                "private_endpoint",
                Severity.MEDIUM,
                "plugin_hook_http_endpoint_private",
                f"the http handler targets a {endpoint.reason} address {record.target!r} (a static check of the "
                "URL's host; --resolve-endpoints adds DNS and redirect checks of public hosts)",
                "Allow the intended host through hooks.allowed_urls or mcp.allowed_private_hosts in the policy.",
            )
        if self.hook_allowed_urls and not allowlisted and not (endpoint is not None and endpoint.kind == "metadata"):
            site.report(
                "not_allowlisted",
                Severity.HIGH,
                "plugin_hook_http_url_not_allowed",
                f"the http handler url {record.target!r} is not in the policy's hooks.allowed_urls",
                "Point the hook at an allowed endpoint, or add the endpoint to hooks.allowed_urls.",
            )
        elif not self.hook_allowed_urls:
            site.report(
                "remote_endpoint",
                Severity.MEDIUM,
                "plugin_hook_http_endpoint",
                f"hook input (tool arguments, prompts, file paths) is posted to {record.target!r}",
                "Review the endpoint; allow it explicitly with hooks.allowed_urls in the validation policy.",
            )
        if site.scope in {"all", "bash"}:
            self._remote_approval(site, "a remote HTTP endpoint")


def hook_risk_summary(records: Iterable[HookRecord]) -> dict[str, Any]:
    """The ``plugin.hook_risk`` metadata block.

    A ``truncated`` row stands for the handlers of a source past the scan
    limits: it is listed (flag ``scan_truncated``) and sets ``counts.truncated``,
    but it is not counted as a handler.
    """
    rows = [record.to_dict() for record in records]
    handlers = [row for row in rows if row["handler_type"] != "truncated"]
    by_flag: dict[str, int] = {}
    by_event: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        for flag in row["risk_flags"]:
            by_flag[flag] = by_flag.get(flag, 0) + 1
    for row in handlers:
        by_event[row["event"]] = by_event.get(row["event"], 0) + 1
        by_type[row["handler_type"]] = by_type.get(row["handler_type"], 0) + 1
    return {
        "hooks": rows,
        "counts": {
            "total": len(handlers),
            "flagged": sum(1 for row in handlers if row["risk_flags"]),
            "truncated": len(handlers) < len(rows),
            "by_flag": dict(sorted(by_flag.items())),
            "by_event": dict(sorted(by_event.items())),
            "by_handler_type": dict(sorted(by_type.items())),
        },
    }


# Privilege flags that describe a component without raising a risk; they do not count as flagged.
_BENIGN_FLAGS = frozenset(
    {"no_frontmatter", "inherits_all_tools", "ignored_hooks", "ignored_mcpServers", "wildcard_ignored"}
)


def privilege_summary(records: Iterable[PrivilegeRecord]) -> dict[str, Any]:
    """The ``plugin.privileges`` metadata block."""
    rows = [record.to_dict() for record in records]
    by_flag: dict[str, int] = {}
    for row in rows:
        for flag in row["flags"]:
            by_flag[flag] = by_flag.get(flag, 0) + 1
    return {
        "components": rows,
        "counts": {
            "agents": sum(1 for row in rows if row["type"] == "agent"),
            "commands": sum(1 for row in rows if row["type"] == "command"),
            "skills": sum(1 for row in rows if row["type"] == "skill"),
            "flagged": sum(
                1
                for row in rows
                if "inert_grant" not in row["flags"] and any(flag not in _BENIGN_FLAGS for flag in row["flags"])
            ),
            "by_flag": dict(sorted(by_flag.items())),
        },
    }
