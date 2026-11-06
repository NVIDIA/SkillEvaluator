# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private static context estimates from passive members; no runtime dependencies."""

from __future__ import annotations

import dataclasses
import math
import os
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CURSOR_MANIFEST_TYPE,
)
from skillevaluator.plugin_formats import profile_for
from skillevaluator.plugin_markdown import _Markdown, parse_markdown
from skillevaluator.utils.secure_fs import SecurePathError

if TYPE_CHECKING:
    from skillevaluator.plugin_component_risk import HookRecord
    from skillevaluator.plugin_discovery import PluginMember
    from skillevaluator.plugin_paths import PluginRootReader


@dataclass
class CostRow:
    """One component's static context-cost row.

    ``always_on_chars`` and ``on_demand_chars`` are token-weighted characters
    (:func:`cost_chars`) as the component's own harness loads it natively.
    :func:`_view_rows` turns them into the numbers of one harness and load
    mode. ``not_counted`` names always-on text whose size is not known
    statically (MCP tool schemas, hook output). ``traits`` are the frontmatter
    facts the views read: ``model_hidden`` (``disable-model-invocation``),
    ``forced_style`` and ``keeps_coding_instructions`` (output styles), and
    the rule traits of :func:`rule_traits` (``always_apply``, ``paths_rule``,
    ``requested``).
    """

    type: str
    name: str
    always_on_chars: int
    on_demand_chars: int
    basis: str
    not_counted: str | None = None
    traits: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "always_on_tokens": estimate_tokens(self.always_on_chars),
            "on_demand_tokens": estimate_tokens(self.on_demand_chars),
            "basis": self.basis,
        }
        if self.not_counted:
            row["not_counted"] = self.not_counted
        return row


_CONTEXT_COST_NOTES: tuple[str, ...] = (
    "Static estimate: tokens are approximated as characters / 4, with each CJK or other full-width character "
    "counted as 1 token (chars_div_4_cjk); no tokenizer is run.",
    "Always-on, for the harness and load mode shown: skill names and descriptions (Claude Code leaves out "
    "disable-model-invocation items), agent and command descriptions, the one forced output style Claude Code "
    "applies (it replaces the default coding instructions unless keep-coding-instructions is set, so it can "
    "shrink the prompt), rules the native adapter stages as always on (not agent-requested or manual rules, and "
    "for Codex not paths- or globs-scoped rules, which it does not stage), and UserPromptSubmit or startup "
    "SessionStart hook output that reads a plugin file.",
    "On-demand: SKILL.md bodies, agent and command bodies, output styles the user must select, and rules in "
    "wrapper mode, where they are embedded in the generated wrapper SKILL.md.",
    "MCP tool schemas are not known statically, so MCP servers are marked not counted, like hook output produced "
    "at run time. Claude Code may shorten long skill descriptions in its listing, depending on its version. "
    "by_harness gives the estimate for each harness and load mode.",
)


def estimate_tokens(chars: int) -> int:
    """Tokens for token-weighted characters (:func:`cost_chars`): divided by 4, rounded away from zero."""
    return math.ceil(chars / 4) if chars >= 0 else -math.ceil(-chars / 4)


COST_ESTIMATOR = "chars_div_4_cjk"
COST_NATIVE = "native"
COST_WRAPPER = "wrapper"
#: The harness and load-mode views every static estimate reports in ``by_harness``.
COST_VIEWS: tuple[tuple[str, str], ...] = (
    ("claude-code", COST_NATIVE),
    ("claude-code", COST_WRAPPER),
    ("codex", COST_NATIVE),
    ("codex", COST_WRAPPER),
)
COST_HARNESS_LABELS = {"claude-code": "Claude Code", "codex": "Codex", "cursor": "Cursor"}
#: The component types each view puts into the model's context. Native Claude Code loads every
#: type; SkillEvaluator's native Codex adapter stages skills, rules (AGENTS.md), and MCP servers;
#: the generated wrapper stages member skills, embeds rules, and starts runnable MCP servers.
_VIEW_TYPES: dict[tuple[str, str], frozenset[str]] = {
    ("claude-code", COST_NATIVE): frozenset({"skill", "rule", "agent", "command", "output_style", "mcp", "hook"}),
    ("claude-code", COST_WRAPPER): frozenset({"skill", "rule", "mcp"}),
    ("codex", COST_NATIVE): frozenset({"skill", "rule", "mcp"}),
    ("codex", COST_WRAPPER): frozenset({"skill", "rule", "mcp"}),
    ("cursor", COST_NATIVE): frozenset({"skill", "rule", "agent", "command", "mcp"}),
}
#: Harnesses that read ``disable-model-invocation`` and keep such skills and commands out of the listing.
#: Codex rejects the field in its own skill validator and lists the skill anyway.
_HIDES_MODEL_DISABLED = frozenset({"claude-code", "cursor"})
#: Characters of Claude Code's default system text ("Doing tasks" coding instructions) that a forced
#: output style replaces unless it sets ``keep-coding-instructions`` (2.1.284 captures: 3,233-3,262).
CLAUDE_CODE_CODING_INSTRUCTIONS_CHARS = 3_250
#: Hook events whose output Claude Code adds to the first model request.
_FIRST_TURN_HOOK_EVENTS = frozenset({"SessionStart", "UserPromptSubmit"})
#: The SessionStart source of a new session; resume, clear, and compact start later.
_SESSION_START_SOURCE = "startup"


def _runs_in_first_request(record: HookRecord) -> bool:
    """Whether a first-turn hook handler runs before the first request of a new session.

    UserPromptSubmit has no matcher. A SessionStart matcher is matched against
    the start source (``startup``, ``resume``, ``clear``, ``compact``) the way
    Claude Code matches tool names, read like ``matcher_scope`` in
    ``plugin_component_risk``: no matcher, ``""`` or ``*`` selects every
    source, a name or ``|`` list must name ``startup``, and anything else is a
    regular expression searched in ``startup``. A matcher the evaluator cannot
    decide counts (fail closed).
    """
    from skillevaluator.plugin_component_risk import (
        _EXACT_MATCHER_RE,
        _NAME_LIST_RE,
        MAX_MATCHER_CHARS,
        _InvalidMatcher,
        _matcher_names,
        _MatcherEvaluator,
        _MatcherParser,
        _UnsupportedMatcher,
    )

    if record.event != "SessionStart" or record.matcher is None or record.matcher.strip() in {"", "*"}:
        return True
    text = record.matcher.strip()
    if _NAME_LIST_RE.fullmatch(text) and _SESSION_START_SOURCE in _matcher_names(text):
        return True
    if _EXACT_MATCHER_RE.fullmatch(text):
        return False
    if len(text) > MAX_MATCHER_CHARS:
        return True
    try:
        return _MatcherEvaluator(_MatcherParser(text).parse()).matches(_SESSION_START_SOURCE)
    except _UnsupportedMatcher:
        return True
    except _InvalidMatcher:
        return False


_PRIMARY_COST_HARNESS = {PLUGIN_CODEX_MANIFEST_TYPE: "codex", PLUGIN_CURSOR_MANIFEST_TYPE: "cursor"}
_TYPE_WORDS = {
    "skill": "skills",
    "rule": "rules",
    "agent": "subagents",
    "command": "commands",
    "output_style": "output styles",
    "mcp": "MCP servers",
    "hook": "plugin hooks",
}


def cost_chars(text: str) -> int:
    """Token-weighted length: 1 per character, 4 per CJK or other full-width character (about 1 token each).

    Characters divided by 4 fits English text. CJK text runs near one token
    per character (a 225-character Chinese description is 172 o200k tokens),
    so wide characters weigh 4.
    """
    if text.isascii():
        return len(text)
    import unicodedata

    wide = sum(1 for char in text if unicodedata.east_asian_width(char) in {"W", "F"})
    return len(text) + 3 * wide


def _claude_flag(value: Any) -> bool:
    """Whether Claude Code reads a frontmatter flag as on.

    A Claude Code 2.1.284 probe forced an output style for ``true``,
    ``"true"``, ``yes`` and ``1``, and not for ``false`` or ``"false"``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def model_hidden_traits(frontmatter: dict[str, Any]) -> frozenset[str]:
    """``{"model_hidden"}`` when ``disable-model-invocation`` is on (as Claude Code reads it), else empty."""
    return frozenset({"model_hidden"}) if _claude_flag(frontmatter.get("disable-model-invocation")) else frozenset()


def _rule_flag(value: Any) -> bool:
    """``alwaysApply`` as the native adapters read it: YAML ``true`` or the string ``"true"`` (any case)."""
    return value is True or (isinstance(value, str) and value.strip().casefold() == "true")


def _rule_has_frontmatter(text: str) -> bool:
    """Whether a rule file opens with a closed ``---`` frontmatter block, as the native adapters test it."""
    lines = text.splitlines()
    return bool(lines) and lines[0].strip() == "---" and any(line.strip() == "---" for line in lines[1:])


def _rule_patterns(scope: Any) -> list[str]:
    """Cursor ``globs`` or Claude ``paths``: a list, or a comma-separated string, of glob patterns."""
    items = scope if isinstance(scope, list) else str(scope).split(",")
    return [text for item in items if (text := str(item).strip())]


def rule_traits(name: str, text: str, frontmatter: dict[str, Any]) -> frozenset[str]:
    """How SkillEvaluator's native adapters stage one rule (``tier3.plugin_native``).

    A rule without frontmatter is always on. With frontmatter:
    ``always_apply`` when ``alwaysApply`` is true; else ``paths_rule`` when
    ``paths`` (or ``globs``) lists a pattern, which Claude Code native loading
    stages as a ``paths`` rule and Codex native loading does not stage (it has
    only an always-on rules channel); and ``requested`` for a Cursor
    agent-requested or manual rule (``alwaysApply`` present but not true, or a
    ``.mdc`` rule), which neither adapter stages. Like the adapters
    (``_rule_activation``), every channel reads the same patterns: a ``paths``
    or ``globs`` value without one scopes nothing.
    """
    if not _rule_has_frontmatter(text):
        return frozenset()
    if _rule_flag(frontmatter.get("alwaysApply")):
        return frozenset({"always_apply"})
    traits: set[str] = set()
    if _rule_patterns(frontmatter.get("paths") or frontmatter.get("globs") or []):
        traits.add("paths_rule")
    if "alwaysApply" in frontmatter or name.casefold().endswith(".mdc"):
        traits.add("requested")
    return frozenset(traits)


def native_rule_staging(harness: str, traits: frozenset[str]) -> str:
    """``always_on``, ``on_demand`` (a ``paths`` rule), or ``not_staged`` under one harness's native loading."""
    if "always_apply" in traits:
        return "always_on"
    if harness == "codex" and "paths_rule" in traits:
        return "not_staged"
    if harness != "codex" and "paths_rule" in traits:
        return "on_demand"
    return "not_staged" if "requested" in traits else "always_on"


def _rule_view(row: CostRow, harness: str, load_mode: str) -> CostRow:
    body = row.on_demand_chars + row.always_on_chars
    if load_mode == COST_WRAPPER:
        return CostRow("rule", row.name, 0, body, "on-demand: embedded in the generated wrapper SKILL.md")
    if harness == "cursor":
        if "always_apply" in row.traits:
            return CostRow("rule", row.name, body, 0, "always-on: alwaysApply rule (Cursor adds it to every request)")
        return CostRow("rule", row.name, 0, body, "on-demand: Cursor attaches it by globs, description, or mention")
    label = COST_HARNESS_LABELS.get(harness, harness)
    staging = native_rule_staging(harness, row.traits)
    if staging == "on_demand":
        return CostRow("rule", row.name, 0, body, "on-demand: a paths- or globs-scoped rule loads with matching files")
    if staging == "not_staged" and harness == "codex" and "paths_rule" in row.traits:
        return CostRow(
            "rule",
            row.name,
            0,
            0,
            f"not loaded: {label} native loading has only an always-on rules channel, so a paths- or "
            "globs-scoped rule is not staged",
        )
    if staging == "not_staged":
        return CostRow(
            "rule",
            row.name,
            0,
            0,
            f"not loaded: {label} native loading does not stage an agent-requested or manual rule "
            "(alwaysApply is not true)",
        )
    where = "appends it to $CODEX_HOME/AGENTS.md" if harness == "codex" else "stages it as a Claude Code user rule"
    return CostRow("rule", row.name, body, 0, f"always-on: native loading {where}")


def _view_rows(rows: Iterable[CostRow], harness: str, load_mode: str) -> list[CostRow]:
    """The rows as one harness and load mode loads them (see :data:`_VIEW_TYPES`).

    Claude Code applies one forced output style (the first in file order) and
    warns about the rest; the applied style replaces the default coding
    instructions unless it sets ``keep-coding-instructions``.
    """
    loads = _VIEW_TYPES[(harness, load_mode)]
    label = f"{COST_HARNESS_LABELS.get(harness, harness)} ({load_mode})"
    applied_style: str | None = None
    out: list[CostRow] = []
    for row in rows:
        if row.type not in loads:
            words = _TYPE_WORDS.get(row.type, f"{row.type} components")
            out.append(CostRow(row.type, row.name, 0, 0, f"not loaded: {label} does not load {words}"))
        elif row.type == "rule":
            out.append(_rule_view(row, harness, load_mode))
        elif "model_hidden" in row.traits and harness in _HIDES_MODEL_DISABLED:
            out.append(
                CostRow(
                    row.type,
                    row.name,
                    0,
                    row.always_on_chars + row.on_demand_chars,
                    "on-demand only: disable-model-invocation keeps it out of the model's listing",
                )
            )
        elif "forced_style" in row.traits and applied_style is not None:
            out.append(
                CostRow(
                    row.type,
                    row.name,
                    0,
                    row.always_on_chars,
                    f"on-demand: forced too, but Claude Code applies only one forced output style ({applied_style})",
                )
            )
        elif "forced_style" in row.traits:
            applied_style = row.name
            header = cost_chars(f"# Output Style: {row.name}\n\n")
            if "keeps_coding_instructions" in row.traits:
                out.append(dataclasses.replace(row, always_on_chars=header + row.always_on_chars))
            else:
                net = header + row.always_on_chars - CLAUDE_CODE_CODING_INSTRUCTIONS_CHARS
                basis = (
                    "always-on: forced output style body; it replaces Claude Code's default coding instructions "
                    f"(about {CLAUDE_CODE_CODING_INSTRUCTIONS_CHARS:,} characters), so the net cost can be negative"
                )
                out.append(dataclasses.replace(row, always_on_chars=net, basis=basis))
        else:
            out.append(row)
    return out


def _view_summary(rows: list[CostRow], harness: str, load_mode: str) -> dict[str, Any]:
    by_component = [row.to_dict() for row in _view_rows(rows, harness, load_mode)]
    not_counted = sum(1 for row in by_component if row.get("not_counted"))
    return {
        "harness": harness,
        "load_mode": load_mode,
        "always_on_tokens": sum(row["always_on_tokens"] for row in by_component),
        "on_demand_tokens": sum(row["on_demand_tokens"] for row in by_component),
        "lower_bound": bool(not_counted),
        "not_counted": not_counted,
    }


def _markdown_cost(component_type: str, name: str, parsed: _Markdown) -> CostRow:
    """The cost row of an agent, command, or output style (a forced style is resolved per view)."""
    frontmatter = parsed.frontmatter
    body = cost_chars(parsed.body)
    if component_type == "output_style":
        if _claude_flag(frontmatter.get("force-for-plugin")):
            traits = {"forced_style"}
            if _claude_flag(frontmatter.get("keep-coding-instructions")):
                traits.add("keeps_coding_instructions")
            return CostRow(
                component_type, name, body, 0, "always-on: forced output style body", traits=frozenset(traits)
            )
        return CostRow(component_type, name, 0, body, "on-demand: output style body when selected")
    label = "agent" if component_type == "agent" else "command"
    basis = (
        "always-on: description; on-demand: subagent prompt (loads in the subagent's own context)"
        if label == "agent"
        else "always-on: description; on-demand: command body"
    )
    return CostRow(
        component_type, name, cost_chars(parsed.description or ""), body, basis, traits=model_hidden_traits(frontmatter)
    )


def _hook_output_chars(record: HookRecord, reader: PluginRootReader, manifest_type: str) -> int | None:
    """Size literal echo output or a safely read plugin-root file; dynamic output is unknown."""
    if record.handler_type != "command" or not record.target:
        return None
    try:
        words = shlex.split(record.target)
    except ValueError:
        return None
    if len(words) >= 2 and words[0] == "echo" and not any("$" in word or "`" in word for word in words[1:]):
        return cost_chars(" ".join(words[1:]) + "\n")
    if len(words) != 2 or words[0] != "cat":
        return None
    target = words[1].replace("\\", "/")
    for prefix in (*profile_for(manifest_type).root_prefixes, "$CLAUDE_PLUGIN_ROOT"):
        if target.startswith(prefix + "/"):
            rel = PurePosixPath(os.path.normpath(target[len(prefix) + 1 :]))
            break
    else:
        return None
    if rel.is_absolute() or ".." in rel.parts:
        return None
    try:
        return cost_chars(reader.read_text(rel, CONTENT_DEDUP_MAX_FILE_BYTES))
    except (SecurePathError, OSError):
        return None


def member_cost_rows(
    members: Iterable[PluginMember], hook_records: Iterable[HookRecord], reader: PluginRootReader
) -> list[CostRow]:
    """Selected effective members' text and static first-turn hook output, without reading runtime state."""
    rows: list[CostRow] = []
    records = tuple(hook_records)
    for member in members:
        if member.problem:
            continue
        parsed = parse_markdown(member.text) if member.text is not None else None
        if member.type == "command" and member.path is None and isinstance(member.config, dict):
            rows.append(
                CostRow(
                    "command",
                    member.name,
                    cost_chars(str(member.config.get("description") or "")),
                    cost_chars(member.text or ""),
                    "always-on: description; on-demand: literal inline command content",
                )
            )
        elif member.type == "skill" and parsed is not None:
            rows.append(
                CostRow(
                    "skill",
                    member.name,
                    cost_chars((parsed.name or "") + (parsed.description or "")),
                    cost_chars(parsed.body),
                    "always-on: frontmatter name + description; on-demand: SKILL.md body",
                    traits=model_hidden_traits(member.frontmatter),
                )
            )
        elif member.type == "rule" and parsed is not None:
            rows.append(
                CostRow(
                    "rule",
                    member.name,
                    0,
                    cost_chars(parsed.body),
                    "rule body",
                    traits=rule_traits(member.name, member.text or "", member.frontmatter),
                )
            )
        elif member.type in {"agent", "command", "output_style"} and parsed is not None:
            row = _markdown_cost(member.type, member.name, parsed)
            if member.type == "command" and isinstance(member.config, dict) and member.config.get("description"):
                row.always_on_chars = cost_chars(str(member.config["description"]))
            rows.append(row)
        elif member.type == "mcp" and member.mcp is not None:
            rows.append(
                CostRow(
                    "mcp",
                    member.name,
                    0,
                    0,
                    "always-on: the server's tool schemas",
                    not_counted="MCP tool schemas are not known statically",
                )
            )
        elif member.type == "hook":
            matching = [
                record
                for record in records
                if (record.source, record.file) == (member.name, (member.path or "").removesuffix("#hooks"))
                and record.event in _FIRST_TURN_HOOK_EVENTS
            ]
            if not matching:
                continue
            always_on = on_demand = unknown = 0
            for record in matching:
                chars = _hook_output_chars(record, reader, member.manifest_type)
                if _runs_in_first_request(record):
                    unknown += chars is None
                    always_on += chars or 0
                else:
                    on_demand += chars or 0
            rows.append(
                CostRow(
                    "hook",
                    member.name,
                    always_on,
                    on_demand,
                    "always-on: startup SessionStart and UserPromptSubmit output; on-demand: later SessionStart output",
                    not_counted=f"Output of {unknown} hook handler(s) is produced at run time" if unknown else None,
                )
            )
    return rows


def context_cost(
    members: Iterable[PluginMember],
    hook_records: Iterable[HookRecord],
    reader: PluginRootReader,
    *,
    manifest_type: str,
    unread_files: int = 0,
    rows: Iterable[CostRow] | None = None,
    extra_rows: Iterable[CostRow] = (),
    extra_notes: Iterable[str] = (),
    harness: str | None = None,
    load_mode: str | None = None,
) -> dict[str, Any]:
    """A bounded static lower-bound estimate with every supported harness/load-mode view."""
    primary = _PRIMARY_COST_HARNESS.get(manifest_type, "claude-code")
    selected = (harness or primary, load_mode or COST_NATIVE)
    if selected not in _VIEW_TYPES:
        selected = (primary, COST_NATIVE)
    rows = list(rows) if rows is not None else member_cost_rows(members, hook_records, reader)
    rows.extend(extra_rows)
    views = list(COST_VIEWS)
    if (primary, COST_NATIVE) not in views:
        views.insert(0, (primary, COST_NATIVE))
    by_component = [row.to_dict() for row in _view_rows(rows, *selected)]
    not_counted = [f"{row['type']} {row['name']}" for row in by_component if row.get("not_counted")]
    notes = [*_CONTEXT_COST_NOTES, *extra_notes]
    if not_counted:
        notes.append("Lower bound: always-on text with unknown static size: " + ", ".join(not_counted[:5]) + ".")
    if unread_files:
        notes.append(f"{unread_files} component file(s) could not be read safely and are not counted.")
    return {
        "method": "static_estimate",
        "estimator": COST_ESTIMATOR,
        "harness": selected[0],
        "load_mode": selected[1],
        "always_on_tokens": sum(row["always_on_tokens"] for row in by_component),
        "on_demand_tokens": sum(row["on_demand_tokens"] for row in by_component),
        "lower_bound": bool(not_counted or unread_files),
        "not_counted": not_counted,
        "by_component": by_component,
        "by_harness": [_view_summary(rows, *view) for view in views],
        "notes": notes,
    }
