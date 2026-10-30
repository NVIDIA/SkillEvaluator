# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared bounded permission rules for private MCP and hook policy analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from skillevaluator.models.result import Severity
from skillevaluator.utils.structured_data import MAX_STRUCTURED_NODES
from skillevaluator.validators.url_policy import report_text

_DOC_KEYS = frozenset({"description", "title", "summary", "notes", "note", "comment", "comments", "help"})


_MAX_SCAN_NODES = MAX_STRUCTURED_NODES


@dataclass(frozen=True)
class OverrideIssue:
    """A dangerous flag/env/config override, independent of where it was declared."""

    concept: Literal[
        "permission_bypass_flag",
        "permission_bypass_scan_truncated",
        "permission_mode_flag",
        "permission_allow_flag",
        "env_code_injection",
        "env_traffic_redirect",
        "env_insecure_tls",
        "env_inline_secret",
        "auto_approve",
    ]
    severity: Severity
    message: str
    suggestion: str


class _ConfigWalk:
    """Bounded, iterative walk yielding ``(json_path, node)`` for a config value.

    ``complete`` turns False when the walk stops at ``_MAX_SCAN_NODES`` with nodes
    still unvisited.
    """

    def __init__(self, value: Any, *, skip_doc_keys: bool = True) -> None:
        self.value = value
        self.skip_doc_keys = skip_doc_keys
        self.complete = True

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        stack: list[tuple[str, Any]] = [("", self.value)]
        seen = 0
        while stack:
            if seen >= _MAX_SCAN_NODES:
                self.complete = False
                return
            path, node = stack.pop()
            seen += 1
            yield path, node
            if isinstance(node, dict):
                for key, child in node.items():
                    if self.skip_doc_keys and str(key).lower() in _DOC_KEYS:
                        continue
                    stack.append((f"{path}.{key}" if path else str(key), child))
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    stack.append((f"{path}[{index}]", child))


def iter_config_strings(value: Any, *, skip_doc_keys: bool = True) -> Iterator[tuple[str, str]]:
    """Yield ``(json_path, string)`` for string leaves of a config value (bounded, iterative)."""
    for path, node in _ConfigWalk(value, skip_doc_keys=skip_doc_keys):
        if isinstance(node, str):
            yield path, node


_bounded = report_text


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


_PYTHON_DOTTED_MODULE_RE = re.compile(r"-m\s+\w+\.[\w.]+(?:\s*:|\s+)")


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


_ALLOWED_TOOLS_VALUE = r"\"[^\"\n]{0,1024}\"|'[^'\n]{0,1024}'|[^\s;&|]{1,1024}"


_ALLOWED_TOOLS_VALUE_RE = re.compile(_ALLOWED_TOOLS_VALUE)


_ALLOWED_TOOLS_FLAG_RE = re.compile(
    rf"(?<![\w-])--allowed-?tools(?:=(?P<value>{_ALLOWED_TOOLS_VALUE})"
    rf"|\s+(?P<values>(?:{_ALLOWED_TOOLS_VALUE})(?:[ \t]+(?!-)(?:{_ALLOWED_TOOLS_VALUE}))*))",
    re.IGNORECASE,
)


_ALLOWED_TOOLS_FLAGS = frozenset({"--allowedtools", "--allowed-tools"})


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
