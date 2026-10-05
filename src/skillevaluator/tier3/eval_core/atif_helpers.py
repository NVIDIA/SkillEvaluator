# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ATIF trajectory helpers -- pure functions for extracting data from ATIF JSON.

Works on plain dicts (json.loads output) with no Pydantic or harness-specific
deps. Supports trajectories from Harbor agents such as Claude Code, Codex,
OpenHands, and Opencode.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from typing import Any

from skillevaluator.evidence import evidence_ref_identity
from skillevaluator.tier3.eval_core.checks import (
    _APPLY_PATCH_COMMAND_RE,
    _APPLY_PATCH_HEADER_RE,
    _FD_REDIRECT_TARGET_RE,
    _REDIRECT_TARGET_RE,
    _SED_IN_PLACE_FLAG_RE,
    _SED_OPERANDS_RE,
    _SHELL_WORD,
    _TEE_OPERANDS_RE,
)
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import (
    iter_normalized_tool_calls as iter_tool_calls,
)
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import (
    normalized_tool_call_observation as _tool_call_observation,
)
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import (
    normalized_tool_call_wrapper_observation as _tool_call_wrapper_observation,
)
from skillevaluator.tier3.eval_core.secret_redaction import _configured_secret_values, redact_secrets_in_log_line


def get_all_tool_calls(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all tool calls with function name, arguments, and observation text.

    Returns ``[{"fn": str, "args": dict, "args_text": str, "obs": str}]``.
    """
    calls: list[dict[str, Any]] = []
    for step, tc in iter_tool_calls(traj):
        fn = tc.get("function_name") or ""
        args = tc.get("arguments") or {}
        calls.append(
            {
                "fn": fn,
                "args": args,
                "args_text": json.dumps(args).lower(),
                "obs": _tool_call_observation(step, tc).lower(),
            }
        )
    return calls


def get_skill_tool_calls(traj: dict[str, Any]) -> list[str]:
    """Get skill names from Claude Code's native ``Skill`` tool invocations."""
    skills: list[str] = []
    for tc in get_all_tool_calls(traj):
        if tc["fn"].lower() == "skill":
            name = tc["args"].get("skill", tc["args"].get("name", ""))
            if name:
                skills.append(str(name))
    return skills


def get_read_calls(traj: dict[str, Any]) -> list[str]:
    """Get file paths from read calls and shell commands that inspect ``SKILL.md``."""
    paths: list[str] = []
    for tc in get_all_tool_calls(traj):
        fn = tc["fn"].lower()
        if fn in ("read", "read_file"):
            path = tc["args"].get("path", tc["args"].get("file_path", ""))
            if path:
                paths.append(str(path))
        elif fn in ("bash", "execute", "exec_command", "run", "run_code", "shell", "command"):
            cmd = tc["args"].get("command", "") or tc["args"].get("cmd", "") or tc["args"].get("code", "")
            if "skill.md" in str(cmd).lower():
                paths.append(str(cmd))
    return paths


def get_bash_commands(traj: dict[str, Any]) -> list[str]:
    """Extract command strings from bash/execute/run_code tool calls."""
    cmds: list[str] = []
    for _, tc in iter_tool_calls(traj):
        fn = (tc.get("function_name") or "").lower()
        if fn in ("bash", "execute", "exec_command", "run_code", "run", "shell", "command"):
            cmd = (tc.get("arguments") or {}).get("command", "")
            if not cmd:
                cmd = (tc.get("arguments") or {}).get("cmd", "")
            if not cmd:
                cmd = (tc.get("arguments") or {}).get("code", "")
            if cmd:
                cmds.append(str(cmd))
    return cmds


def get_agent_text(traj: dict[str, Any]) -> str:
    """Concatenate all agent message text from the trajectory."""
    parts: list[str] = []
    for step in traj.get("steps", []):
        if step.get("source") == "agent":
            msg = step.get("message") or ""
            if isinstance(msg, str) and msg.strip():
                parts.append(msg)
    return "\n".join(parts)


def _agent_tool_calls(traj: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
    """``(step index, step, tool call)`` for each normalized tool call of the agent's steps, in order."""
    for step_index, step in enumerate(traj.get("steps", [])):
        if step.get("source") == "agent":
            for _, tool_call in iter_tool_calls({"steps": [step]}):
                yield step_index, step, tool_call


def get_final_response(traj: dict[str, Any]) -> str:
    """Get the last non-empty agent message from the trajectory."""
    for step in reversed(traj.get("steps", [])):
        if step.get("source") == "agent":
            msg = step.get("message") or ""
            if isinstance(msg, str) and msg.strip():
                return msg
    return ""


def get_output_tokens(traj: dict[str, Any]) -> int:
    """Extract total completion tokens from trajectory metrics."""
    final = traj.get("final_metrics") or {}
    if final.get("total_completion_tokens"):
        return int(final["total_completion_tokens"])
    last = 0
    for step in traj.get("steps", []):
        m = step.get("metrics") or {}
        if m.get("completion_tokens"):
            last = int(m["completion_tokens"])
    return last


def build_conversation_summary(
    traj: dict[str, Any],
    question: str,
    max_chars: int | None = None,
) -> str:
    """Build the compact tool history the behavior judge reads.

    Every entry is secret-redacted, and cut text keeps its head and tail around
    a marker that says how much was cut. A write call (write and edit tools,
    ``apply_patch``, shell writes) shows its body with the room of one FILE
    CHANGES entry, so the judge can see what the agent wrote. With
    ``max_chars``, an over-budget history shrinks, then drops, tool calls, tool
    results, and messages from its middle first, so the start (the skill call)
    and the end (test runs) stay longest; the final answer and the latest write
    to each path go last.
    """
    return _fit_history(_history_entries(traj, question), max_chars)


def _fit_history(entries: list[list[Any]], max_chars: int | None) -> str:
    return _fit_entries(
        entries,
        max_chars,
        sep="\n",
        stub_chars=_HISTORY_STUB_CHARS,
        noun="history entries",
        middle_first=True,
        exact_shrink=True,
    )


_BEHAVIOR_EVIDENCE_MAX_CHARS = 4000
# USER REQUEST room in behavior evidence. FINAL RESPONSE room is configurable
# (SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT) and defaults to the same size.
_BEHAVIOR_SECTION_CHARS = 800
_DEFAULT_BEHAVIOR_FINAL_RESPONSE_LIMIT = _BEHAVIOR_SECTION_CHARS
_DEFAULT_BEHAVIOR_CHECK_BUDGET = 8000
_DEFAULT_TOOL_HISTORY_HEADROOM = 4000
# The final response leaves at least this much (or half the budget) for the
# user request and the tool history.
_MIN_BEHAVIOR_HISTORY_HEADROOM = 1600
# File changes leave up to 1/4 of the behavior evidence for the user request
# and the tool history, so skill calls and test runs stay in view.
_BEHAVIOR_HISTORY_SHARE = 4
_SECTION_COMPACT_TOOL_HISTORY = "COMPACT TOOL HISTORY"
_SECTION_FILE_CHANGES = "FILE CHANGES"
_SECTION_FINAL_RESPONSE = "FINAL RESPONSE"
_SECTION_USER_REQUEST = "USER REQUEST"


def _env_positive_int(name: str, default: int) -> int:
    """Parse a positive integer from environment variable *name* or return *default*."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return default


def _behavior_final_response_limit() -> int:
    """Return the configured behavior final response section limit or default."""
    return _env_positive_int("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", _DEFAULT_BEHAVIOR_FINAL_RESPONSE_LIMIT)


def _behavior_check_budget() -> int:
    """Return the configured behavior check evidence budget or reconciled default."""
    final_limit = _behavior_final_response_limit()
    base_budget = _env_positive_int("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", _DEFAULT_BEHAVIOR_CHECK_BUDGET)
    return max(base_budget, final_limit + _DEFAULT_TOOL_HISTORY_HEADROOM)


# Hermes (and older OpenCode) name the edit tool ``patch``.
_BEHAVIOR_WRITE_TOOLS = {
    "write",
    "write_file",
    "edit",
    "edit_file",
    "multiedit",
    "notebookedit",
    "apply_patch",
    "applypatch",
    "patch",
}
_APPLY_PATCH_TOOLS = {"apply_patch", "applypatch"}
# Hermes runs shell commands with ``terminal`` and Python with ``execute_code``.
_BEHAVIOR_EXEC_TOOLS = {
    "bash",
    "execute",
    "exec_command",
    "run_code",
    "run",
    "shell",
    "command",
    "terminal",
    "execute_code",
}
_BEHAVIOR_EXEC_COMMAND_KEYS = ("command", "cmd", "code")
_BEHAVIOR_PYTHON_WRITE_RE = re.compile(
    r"\b(?:write_text|write_bytes)\s*\(|\bopen\s*\([^)]*,\s*['\"][wa]",
    re.IGNORECASE,
)
# sed options that give the script, so every operand of a ``sed -i`` is a file it edits.
_SED_SCRIPT_OPTIONS = ("-e", "--expression", "-f", "--file")
_TOOL_NAME_SEPARATORS = (".", ":", "/", "__")
# Harnesses name the written file and text differently: Claude Code uses
# file_path, content, new_string, notebook_path, and new_source; OpenCode uses
# filePath, newString, and patchText; the Codex apply_patch tool uses input.
_WRITE_PATH_KEYS = ("file_path", "filePath", "path", "filename", "target_file", "notebook_path")
_WRITE_BODY_KEYS = ("content", "contents", "new_string", "newString", "new_source", "patch", "patchText", "code")
_APPLY_PATCH_BODY_KEYS = ("input", "raw", "value")
_WRITE_MAX_EDITS = 5
# The security extractor's drift-pinned apply_patch file headers.
_PATCH_FILE_HEADER_RE = _APPLY_PATCH_HEADER_RE

# What the judges see of each tool-history entry before any budget applies.
_HISTORY_ARGS_CHARS = 200
_HISTORY_REASONING_CHARS = 200
_HISTORY_RESULT_CHARS = 400
_HISTORY_MESSAGE_CHARS = 1500
# Size an older, lower-priority entry shrinks to when the history is over budget.
_HISTORY_STUB_CHARS = 160
# 1800 is the room FILE CHANGES already gave one write body. The tool history
# gives a write this much, and FILE CHANGES never gives a body less before older
# writes shrink, so one write still fits the smallest behavior budget (4000).
_WRITE_BODY_CHARS = 1800
_WRITE_RESULT_CHARS = 500
_FILE_CHANGE_STUB_CHARS = 300
# Very long text is redacted only at the two ends that can be shown, plus this
# slack, so a secret that crosses a cut is still matched whole.
_REDACTION_SLACK_CHARS = 4096
_TRUNCATION_MARKER = "\n...[{} chars truncated]...\n"
# At most 13 digits, the widest count _TRUNCATION_MARKER_ROOM allows. Tool output
# can hold marker-like text, and int() refuses very long digit strings.
_TRUNCATION_MARKER_RE = re.compile(r"\n\.\.\.\[(\d{1,13}) chars truncated\]\.\.\.\n")
# Room kept for the marker, wide enough for any count.
_TRUNCATION_MARKER_ROOM = len(_TRUNCATION_MARKER.format(10**12))
_OMITTED_MARKER = "...[{} {} omitted to fit the budget]..."
# Ranks for _fit_entries: lower ranks are shrunk and dropped first.
_RANK_LOW = 0  # tool results, reasoning, and non-write tool calls
_RANK_MESSAGE = 1  # the user request, intermediate agent messages, and a final answer shown elsewhere
_RANK_OLD_WRITE = 2  # a write to a path the agent wrote again later
_RANK_KEEP = 3  # the final answer and the latest write to each path


def _cut_sizes(limit: int) -> tuple[int, int] | None:
    """Head and tail sizes for text cut to *limit* chars, or ``None`` when no marker fits."""
    room = limit - _TRUNCATION_MARKER_ROOM
    if room < 2:
        return None
    head = room * 2 // 3
    return head, room - head


def _truncate_for_behavior(text: str, limit: int, *, recount: bool = True) -> str:
    """Keep the head and tail of *text* within *limit* chars, around a marker giving the cut size.

    With *recount*, markers inside the cut part add their counts, so text cut
    before (a shrunk history entry) reports every cut character. Raw trajectory
    text is cut with ``recount=False``: marker-like text in it is not ours.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    sizes = _cut_sizes(limit)
    if sizes is None:
        return text[:limit]
    head, tail = sizes
    cut = len(text) - head - tail
    if recount:
        for match in _TRUNCATION_MARKER_RE.finditer(text, head, len(text) - tail):
            cut += int(match.group(1)) - len(match.group(0))
    return f"{text[:head]}{_TRUNCATION_MARKER.format(cut)}{text[-tail:]}"


def _judge_excerpt(text: Any, limit: int) -> str:
    """Return the secret-redacted head and tail of raw *text*, at most *limit* chars."""
    text = str(text or "")
    if limit <= 0:
        return ""
    window = limit + _REDACTION_SLACK_CHARS
    if len(text) <= 2 * window:
        return _truncate_for_behavior(_redact_evidence_text(text), limit, recount=False)
    # Only the two ends can be shown, so only they are redacted.
    head_text = _redact_evidence_text(text[:window])
    sizes = _cut_sizes(limit)
    if sizes is None:
        return head_text[:limit]
    head, tail = sizes
    tail_text = _redact_evidence_text(text[-window:])
    return f"{head_text[:head]}{_TRUNCATION_MARKER.format(len(text) - head - tail)}{tail_text[-tail:]}"


def _append_section_with_budget(
    parts: list[str],
    title: str,
    body: str,
    max_chars: int,
    *,
    section_limit: int | None = None,
) -> int:
    budget = max_chars if section_limit is None else min(max_chars, section_limit)
    if budget <= len(title) + 2 or not body.strip():
        return max_chars
    excerpt = _judge_excerpt(body, budget - len(title) - 2)
    if not excerpt:
        return max_chars
    section = f"{title}\n{excerpt}"
    parts.append(section)
    return max(0, max_chars - len(section) - 2)


def _section_room(title: str, body: str, section_limit: int) -> int:
    """Upper bound on what ``_append_section_with_budget`` takes for this section."""
    body = body.strip()
    return min(section_limit, len(title) + 1 + len(body)) + 2 if body else 0


def _tool_file_path(args: dict[str, Any]) -> str:
    for key in _WRITE_PATH_KEYS:
        value = args.get(key)
        if value:
            return str(value)
    return ""


def _tool_write_body(args: dict[str, Any]) -> str:
    snippets: list[str] = []
    for key in _WRITE_BODY_KEYS:
        value = args.get(key)
        if value:
            snippets.append(f"{key}:\n{value}")
    edits = args.get("edits")
    if isinstance(edits, list):
        for idx, edit in enumerate(edits[:_WRITE_MAX_EDITS], start=1):
            if isinstance(edit, dict):
                new_string = edit.get("new_string") or edit.get("newString") or edit.get("replacement")
                if new_string:
                    snippets.append(f"edit {idx} new_string:\n{new_string}")
        if len(edits) > _WRITE_MAX_EDITS:
            snippets.append(f"...[{len(edits) - _WRITE_MAX_EDITS} more edits not shown]...")
    return "\n\n".join(str(s) for s in snippets if str(s).strip())


def _command_arg_text(value: Any) -> str:
    """A command argument as text. Codex can pass an argv list such as ``["bash", "-lc", script]``."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    return str(value or "")


def _sed_in_place_files(words: list[str]) -> list[str]:
    """The files a ``sed`` command line (its *words* after ``sed``) edits in place; none without ``-i``."""
    if not any(_SED_IN_PLACE_FLAG_RE.fullmatch(word.lower()) for word in words):
        return []
    operands: list[str] = []
    script_given = False  # by -e/-f, so the first operand is a file too
    option_argument = False
    for word in words:
        if option_argument:
            option_argument = False
        elif word in _SED_SCRIPT_OPTIONS:
            script_given = option_argument = True
        elif word.startswith(("--expression=", "--file=")):
            script_given = True
        elif not word.startswith("-"):
            operands.append(word)
    return operands if script_given else operands[1:]


def _shell_write_words(text: str) -> list[str]:
    """Files *text* writes as a shell command: redirect targets, ``tee`` operands, and ``sed -i`` files.

    Reads the command with the security extractor's patterns (``_shell_write_targets``)
    but keeps each path as written, without its quotes. A device such as ``/dev/null``
    is not a file change.
    """
    words = [
        match.group(1)
        for match in _REDIRECT_TARGET_RE.finditer(text)
        if not _FD_REDIRECT_TARGET_RE.fullmatch(match.group(1))
    ]
    for match in _TEE_OPERANDS_RE.finditer(text):
        words.extend(word for word in re.findall(_SHELL_WORD, match.group(1)) if not word.startswith("-"))
    for match in _SED_OPERANDS_RE.finditer(text):
        words.extend(_sed_in_place_files(re.findall(_SHELL_WORD, match.group(1))))
    paths = (word.strip("'\"") for word in words)
    return [path for path in paths if path and not path.startswith("/dev/")]


def _command_looks_like_write(command: str) -> bool:
    return bool(
        _shell_write_words(command)
        or _APPLY_PATCH_COMMAND_RE.search(command)
        or _BEHAVIOR_PYTHON_WRITE_RE.search(command)
    )


def _tool_name_candidates(fn_lower: str) -> set[str]:
    """The tool name and its last segment after each namespace separator."""
    candidates = {fn_lower}
    for separator in _TOOL_NAME_SEPARATORS:
        if separator in fn_lower:
            candidates.add(fn_lower.rsplit(separator, 1)[-1])
    return candidates


def _tool_name_looks_like_write(fn_lower: str) -> bool:
    return not _tool_name_candidates(fn_lower).isdisjoint(_BEHAVIOR_WRITE_TOOLS)


def _tool_name_looks_like_exec(fn_lower: str) -> bool:
    """A shell or code tool, also under a namespace (``functions.exec_command``, ``mcp__shell__bash``)."""
    return not _tool_name_candidates(fn_lower).isdisjoint(_BEHAVIOR_EXEC_TOOLS)


def _patch_file_paths(text: str) -> list[str]:
    """Files an apply_patch body adds, updates, deletes, or moves to."""
    paths = (match.group(1).strip() for match in _PATCH_FILE_HEADER_RE.finditer(text))
    return list(dict.fromkeys(path for path in paths if path))


def _shell_write_paths(command: str) -> list[str]:
    """Files a shell command writes through a redirect, ``tee``, ``sed -i``, or an apply_patch heredoc."""
    # Read redirects only up to the end of the line that opens a heredoc, so a
    # quoted "> line" inside the heredoc body is not taken for a target.
    heredoc = command.find("<<")
    line_end = command.find("\n", heredoc) if heredoc >= 0 else -1
    head = command if line_end < 0 else command[:line_end]
    return list(dict.fromkeys((*_shell_write_words(head), *_patch_file_paths(command))))


def _write_call_parts(fn: str, args: Any) -> tuple[list[str], str, dict[str, Any]] | None:
    """Return ``(paths, body, other_args)`` when a tool call writes files, else ``None``."""
    if not isinstance(args, dict):
        args = {}
    fn_lower = fn.lower()
    path = _tool_file_path(args)
    if _tool_name_looks_like_write(fn_lower):
        body = _tool_write_body(args)
        used = {*_WRITE_BODY_KEYS, "edits"}
        if not body and not _tool_name_candidates(fn_lower).isdisjoint(_APPLY_PATCH_TOOLS):
            for key in _APPLY_PATCH_BODY_KEYS:
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    body = f"{key}:\n{value}"
                    used.add(key)
                    break
        paths = [path] if path else _patch_file_paths(body)
    elif _tool_name_looks_like_exec(fn_lower):
        key = next((key for key in _BEHAVIOR_EXEC_COMMAND_KEYS if args.get(key)), "")
        command = _command_arg_text(args.get(key)) if key else ""
        if not _command_looks_like_write(command):
            return None
        body = f"command:\n{command}"
        used = {key}
        paths = list(dict.fromkeys(p for p in (path, *_shell_write_paths(command)) if p))
    else:
        return None
    return paths, body, {key: value for key, value in args.items() if key not in used}


def _demote_superseded_writes(entries: list[list[Any]]) -> None:
    """Rank a write below the latest write to each of its paths."""
    last_writer: dict[str, int] = {}
    for index, entry in enumerate(entries):
        for path in entry[2]:
            last_writer[path] = index
    for index, entry in enumerate(entries):
        if entry[2] and all(last_writer[path] != index for path in entry[2]):
            entry[1] = _RANK_OLD_WRITE


def _final_response_index(steps: list[dict[str, Any]]) -> int | None:
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        if step.get("source") == "agent":
            msg = step.get("message") or ""
            if isinstance(msg, str) and msg.strip():
                return index
    return None


def _history_call_entry(tc: dict[str, Any], write_bodies: bool) -> list[Any]:
    fn = str(tc.get("function_name") or "")
    name = _judge_excerpt(fn, _HISTORY_ARGS_CHARS)
    args = tc.get("arguments") or {}
    write = _write_call_parts(fn, args) if isinstance(args, dict) else None
    if write is None or not write_bodies:
        # Without write bodies (FILE CHANGES shows them), a write is a short line like any other call.
        shown = _judge_excerpt(json.dumps(args, default=str), _HISTORY_ARGS_CHARS)
        note = " [written content is under FILE CHANGES]" if write is not None else ""
        return [f"Agent called: {name}({shown}){note}", _RANK_LOW, ()]
    paths, body, other_args = write
    shown = _judge_excerpt(json.dumps(other_args, default=str), _HISTORY_ARGS_CHARS) if other_args else ""
    text = f"Agent called: {name}({shown})"
    if body:
        text = f"{text}\n{_judge_excerpt(body, _WRITE_BODY_CHARS)}"
    return [text, _RANK_KEEP, tuple(paths)]


def _history_entries(
    traj: dict[str, Any],
    question: str,
    *,
    write_bodies: bool = True,
    final_chars: int = _HISTORY_MESSAGE_CHARS,
    final_rank: int = _RANK_KEEP,
) -> list[list[Any]]:
    """``[text, rank, written paths]`` per history line, in trajectory order.

    Without *write_bodies*, write calls are short lines ranked like other tool
    calls, for evidence that shows the bodies under FILE CHANGES. The final
    answer keeps up to *final_chars* at *final_rank*; evidence that shows it
    under FINAL RESPONSE passes less, and a lower rank so the history's copy
    shrinks before skill calls and test runs are dropped.
    """
    entries: list[list[Any]] = [[f"User: {_judge_excerpt(question, _HISTORY_MESSAGE_CHARS)}", _RANK_MESSAGE, ()]]
    steps = traj.get("steps", [])
    final_index = _final_response_index(steps)
    final_listed = False
    for index, step in enumerate(steps):
        if step.get("source") != "agent":
            continue

        reasoning = _judge_excerpt(step.get("reasoning_content"), _HISTORY_REASONING_CHARS)
        if reasoning:
            entries.append([f"Agent reasoning: {reasoning}", _RANK_LOW, ()])

        for _, tc in iter_tool_calls({"steps": [step]}):
            entries.append(_history_call_entry(tc, write_bodies))

        for result in (step.get("observation") or {}).get("results") or []:
            content = _judge_excerpt(result.get("content") if isinstance(result, dict) else "", _HISTORY_RESULT_CHARS)
            if content:
                entries.append([f"Tool returned: {content}", _RANK_LOW, ()])

        msg = step.get("message") or ""
        if isinstance(msg, str) and msg.strip() and not step.get("tool_calls"):
            is_final = index == final_index
            final_listed = final_listed or is_final
            rank = final_rank if is_final else _RANK_MESSAGE
            limit = final_chars if is_final else _HISTORY_MESSAGE_CHARS
            entries.append([f"Agent: {_judge_excerpt(msg, limit)}", rank, ()])

    if final_index is not None and not final_listed:
        final = _judge_excerpt(steps[final_index].get("message"), final_chars)
        entries.append([f"Agent final answer: {final}", final_rank, ()])
    _demote_superseded_writes(entries)
    return entries


def _render_entries(texts: list[str], dropped: list[bool], sep: str, noun: str) -> str:
    out: list[str] = []
    run = 0
    for index, text in enumerate(texts):
        if dropped[index]:
            run += 1
            continue
        if run:
            out.append(_OMITTED_MARKER.format(run, noun))
            run = 0
        out.append(text)
    if run:
        out.append(_OMITTED_MARKER.format(run, noun))
    return sep.join(out)


def _fit_entries(
    entries: list[list[Any]],
    max_chars: int | None,
    *,
    sep: str,
    stub_chars: int,
    noun: str,
    middle_first: bool = False,
    exact_shrink: bool = False,
) -> str:
    """Join ``[text, rank, ...]`` entries, fitting them into *max_chars* by rank.

    Over budget, entries below ``_RANK_KEEP`` shrink to *stub_chars*, lowest rank
    and oldest first (with *middle_first*, the middle of the list first, so both
    ends stay longest), then drop the same way; a run of dropped entries becomes
    one marker. With *exact_shrink*, the entry whose shrink brings the text
    within budget is cut only as far as needed, so no room is left unused.
    Then older top-rank entries shrink and drop. The newest top-rank entry
    (the final answer, or the latest write) is cut only by the last-resort
    head-and-tail cut of the whole text.
    """
    texts = [str(entry[0]) for entry in entries]
    if max_chars is None:
        return sep.join(texts)
    if max_chars <= 0 or not texts:
        return ""
    ranks = [entry[1] for entry in entries]
    dropped = [False] * len(texts)
    marker_cost = len(_OMITTED_MARKER.format(len(texts), noun)) + len(sep)
    total = sum(len(text) for text in texts) + len(sep) * (len(texts) - 1)

    def shrink(index: int, limit: int) -> None:
        nonlocal total
        short = _truncate_for_behavior(texts[index], limit)
        total -= len(texts[index]) - len(short)
        texts[index] = short

    def drop(index: int) -> None:
        nonlocal total
        left = index > 0 and dropped[index - 1]
        right = index + 1 < len(texts) and dropped[index + 1]
        total -= len(texts[index]) + len(sep)
        if left and right:
            total -= marker_cost
        elif not left and not right:
            total += marker_cost
        dropped[index] = True

    order = list(range(len(texts)))
    if middle_first:
        order.sort(key=lambda index: abs(2 * index - len(texts) + 1))
    for action in (shrink, drop):
        for rank in range(_RANK_KEEP):
            for index in order:
                if total <= max_chars:
                    break
                if ranks[index] == rank:
                    if action is shrink:
                        floor = stub_chars
                        if exact_shrink:
                            floor = max(stub_chars, len(texts[index]) - (total - max_chars))
                        shrink(index, floor)
                    else:
                        drop(index)
    keepers = [index for index in range(len(texts)) if ranks[index] >= _RANK_KEEP][:-1]
    for index in keepers:
        if total <= max_chars:
            break
        shrink(index, max(stub_chars, len(texts[index]) - (total - max_chars)))
    for index in keepers:
        if total <= max_chars:
            break
        drop(index)
    return _truncate_for_behavior(_render_entries(texts, dropped, sep, noun), max_chars)


def _write_body_max_chars() -> int:
    """With free room, FILE CHANGES bodies share it evenly, up to this much each: the largest bundle budget."""
    return max(_WRITE_BODY_CHARS, *_bundle_budgets().values())


def _file_change_entries(traj: dict[str, Any]) -> list[list[Any]]:
    """``[text, rank, written paths, head, body, tail, body cut]`` per write call, in trajectory order.

    ``body`` keeps up to ``_write_body_max_chars()``; ``_fit_file_changes`` sizes it to the room.
    """
    body_max = _write_body_max_chars()
    entries: list[list[Any]] = []
    for _, step, tc in _agent_tool_calls(traj):
        fn = str(tc.get("function_name") or "")
        write = _write_call_parts(fn, tc.get("arguments") or {})
        if write is None:
            continue
        paths, body, _ = write
        if not body and not paths:
            continue

        head = f"Agent called: {_judge_excerpt(fn, _HISTORY_ARGS_CHARS)}"
        if paths:
            head = f"{head}\nPath: {_judge_excerpt(', '.join(paths), _HISTORY_ARGS_CHARS)}"
        shown = _judge_excerpt(body, body_max)
        obs = _judge_excerpt(_tool_call_observation(step, tc), _WRITE_RESULT_CHARS)
        tail = f"Tool returned: {obs}" if obs else ""
        text = _file_change_text(head, shown, tail)
        entries.append([text, _RANK_KEEP, tuple(paths), head, shown, tail, len(body) > body_max])
    _demote_superseded_writes(entries)
    return entries


def _file_change_text(head: str, body: str, tail: str) -> str:
    return "\n".join(part for part in (head, body, tail) if part)


def _even_share(lengths: list[int], room: int) -> int:
    """The largest cap that keeps the capped *lengths* within *room* in total."""
    for done, length in enumerate(sorted(lengths)):
        share = room // (len(lengths) - done)
        if length > share:
            return share
        room -= length
    return max(lengths, default=0)


def _fit_file_changes(entries: list[list[Any]], max_chars: int | None) -> tuple[str, bool]:
    """Fit file changes into *max_chars*; also say whether any was cut or dropped.

    Write bodies share the room evenly, and the latest write to each path gets
    what earlier writes to it leave. Each keeps at least ``_WRITE_BODY_CHARS``
    before older writes shrink and drop.
    """
    sep = "\n\n"
    full = sep.join(str(entry[0]) for entry in entries)
    if max_chars is None:
        return full, any(entry[6] for entry in entries)
    fixed = sum(len(entry[0]) - len(entry[4]) for entry in entries) + len(sep) * (len(entries) - 1)
    old_bodies = sum(min(len(entry[4]), _WRITE_BODY_CHARS) for entry in entries if entry[1] < _RANK_KEEP)
    latest_bodies = [len(entry[4]) for entry in entries if entry[1] >= _RANK_KEEP]
    cap = max(_WRITE_BODY_CHARS, _even_share(latest_bodies, max_chars - fixed - old_bodies))
    fitted = []
    for entry in entries:
        body = _truncate_for_behavior(entry[4], cap if entry[1] >= _RANK_KEEP else _WRITE_BODY_CHARS)
        fitted.append([_file_change_text(entry[3], body, entry[5]), entry[1], entry[2]])
    text = _fit_entries(fitted, max_chars, sep=sep, stub_chars=_FILE_CHANGE_STUB_CHARS, noun="file changes")
    return text, text != full or any(entry[6] for entry in entries)


def build_behavior_evidence(
    traj: dict[str, Any],
    question: str,
    max_chars: int | None = None,
    final_response_limit: int | None = None,
) -> str:
    """Build compact, behavior-check-specific evidence from an ATIF trajectory.

    Behavior checks often ask whether the agent produced or changed artifacts.
    Put final output and write/edit evidence before exploratory reads so a
    fixed-size judge prompt does not miss late file creation. File changes
    leave room for the final response and for a share of the tool history,
    and the history is fitted to what is left, so no section is cut blind.
    Once FILE CHANGES shows the write bodies, the history lists each write as
    a short call line, so skill calls and test runs keep their room.
    """
    final_limit = (
        _behavior_final_response_limit() if final_response_limit is None else max(1, int(final_response_limit))
    )
    if max_chars is None:
        if os.environ.get("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "").strip():
            max_chars = _behavior_check_budget()
        else:
            max_chars = max(_BEHAVIOR_EVIDENCE_MAX_CHARS, final_limit + _MIN_BEHAVIOR_HISTORY_HEADROOM)
    return _behavior_evidence(traj, question, max_chars, final_limit, _file_change_entries(traj), {})


def _behavior_evidence(
    traj: dict[str, Any],
    question: str,
    max_chars: int,
    final_limit: int,
    file_entries: list[list[Any]],
    histories: dict[tuple[bool, int, int], list[list[Any]]],
) -> str:
    """``build_behavior_evidence`` with built file changes; *histories* caches the tool history.

    The final response gets up to *final_limit* chars, but leaves the user
    request and the tool history up to ``_MIN_BEHAVIOR_HISTORY_HEADROOM``
    chars (at most half of what is left, and no more than they need).
    """

    # A final response limit of at least the history's message room shows the
    # answer under FINAL RESPONSE, so the history lists it as a short line.
    final_chars = _HISTORY_STUB_CHARS if final_limit >= _HISTORY_MESSAGE_CHARS else _HISTORY_MESSAGE_CHARS
    # When FINAL RESPONSE shows the answer, the history's copy ranks like a
    # message: it shrinks to a short line before any tool call is dropped.
    final = get_final_response(traj)
    final_shown = bool(final.strip()) and final_limit > len(_SECTION_FINAL_RESPONSE) + 2
    final_rank = _RANK_MESSAGE if final_shown else _RANK_KEEP

    def history_entries(write_bodies: bool) -> list[list[Any]]:
        key = (write_bodies, final_chars, final_rank)
        if key not in histories:
            histories[key] = _history_entries(
                traj, question, write_bodies=write_bodies, final_chars=final_chars, final_rank=final_rank
            )
        return histories[key]

    def tail_room(write_bodies: bool) -> int:
        """Room the user request and the whole tool history would take."""
        return (
            _section_room(_SECTION_USER_REQUEST, question, _BEHAVIOR_SECTION_CHARS)
            + len(_SECTION_COMPACT_TOOL_HISTORY)
            + 3
            + len(_fit_history(history_entries(write_bodies), None))
        )

    parts: list[str] = []
    remaining = max_chars
    write_bodies = True

    if file_entries:
        final_cap = min(final_limit, max(1, max_chars - min(_MIN_BEHAVIOR_HISTORY_HEADROOM, max_chars // 2)))
        room = (
            remaining
            - _section_room(_SECTION_FINAL_RESPONSE, final, final_cap)
            - min(tail_room(False), max_chars // _BEHAVIOR_HISTORY_SHARE)
            - len(_SECTION_FILE_CHANGES)
            - 3
        )
        # A long configured final response never pushes file changes out entirely.
        room = max(room, min(_BEHAVIOR_SECTION_CHARS, remaining // 3) - len(_SECTION_FILE_CHANGES) - 3)
        file_changes, _ = _fit_file_changes(file_entries, room)
        remaining = _append_section_with_budget(parts, _SECTION_FILE_CHANGES, file_changes, remaining)
        write_bodies = not parts  # the history shows write bodies only when FILE CHANGES does not

    if final:
        headroom = min(_MIN_BEHAVIOR_HISTORY_HEADROOM, remaining // 2, tail_room(write_bodies))
        if max_chars > final_limit:
            headroom = min(headroom, max(_BEHAVIOR_SECTION_CHARS, remaining - final_limit))
        remaining = _append_section_with_budget(
            parts,
            _SECTION_FINAL_RESPONSE,
            final,
            remaining,
            section_limit=min(final_limit, max(1, remaining - headroom)),
        )

    remaining = _append_section_with_budget(
        parts,
        _SECTION_USER_REQUEST,
        question,
        remaining,
        section_limit=_BEHAVIOR_SECTION_CHARS,
    )

    history_text = _fit_history(history_entries(write_bodies), remaining - len(_SECTION_COMPACT_TOOL_HISTORY) - 3)
    remaining = _append_section_with_budget(parts, _SECTION_COMPACT_TOOL_HISTORY, history_text, remaining)

    return "\n\n".join(parts)[:max_chars]


_METRIC_EVIDENCE_REF_METRICS = ("accuracy", "goal_accuracy", "behavior_check")
_METRIC_EVIDENCE_EXCERPT_CHARS = 300
_METRIC_EVIDENCE_MAX_TOOL_REFS = 20
_METRIC_EVIDENCE_MAX_FILE_REFS = 12
_EXPECTED_ARTIFACT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:logs/agent|workspace/output|output)"
    r"[A-Za-z0-9._/+=:@-]*[A-Za-z0-9_./+=:@-]"
)


# A placeholder key such as sk-your-key-here or nvapi-REPLACE_ME: letters of
# one case in words joined by - or _. No real key looks like this, and a task
# can ask for one in a config file, so the judges see it as written.
_KEY_PLACEHOLDER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])((?:sk|nvapi)-(?:[a-z]+(?:[-_][a-z]+)*|[A-Z]+(?:[-_][A-Z]+)*))(?![A-Za-z0-9_-])"
)


def _redact_evidence_text(text: str) -> str:
    # Mirror harbor/templates/eval.py: also redact the exact values of the
    # configured credentials, which need not match the sk-/nvapi- shapes.
    text = str(text or "")
    secrets = _configured_secret_values()
    # Keep placeholder keys, unless the text holds a secret value: then redact everything.
    parts = [text] if any(secret in text for secret in secrets) else _KEY_PLACEHOLDER_RE.split(text)
    parts[::2] = [redact_secrets_in_log_line(part, extra_secret_values=secrets) for part in parts[::2]]
    return "".join(parts).replace("\x00", "").strip()


def _evidence_excerpt(text: str, limit: int = _METRIC_EVIDENCE_EXCERPT_CHARS) -> str:
    return _truncate_for_behavior(_redact_evidence_text(text), limit, recount=False)


def _evidence_ref(
    *,
    source: str,
    kind: str,
    label: str,
    json_pointer: str | None = None,
    path: str | None = None,
    excerpt: str = "",
    status: str | None = None,
    evidence_id: str | None = None,
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "source": source,
        "kind": kind,
        "label": _evidence_excerpt(label, 160),
    }
    if json_pointer:
        ref["json_pointer"] = json_pointer
    if path:
        ref["path"] = _evidence_excerpt(path)
    if excerpt:
        ref["excerpt"] = _evidence_excerpt(excerpt)
    if status:
        ref["status"] = status
    if evidence_id:
        ref["evidence_id"] = evidence_id
    return ref


def _dedupe_evidence_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ref in refs:
        key = (
            str(ref.get("source") or ""),
            evidence_ref_identity(ref),
            str(ref.get("kind") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _final_response_ref(traj: dict[str, Any]) -> list[dict[str, Any]]:
    steps = traj.get("steps", [])
    for step_idx in range(len(steps) - 1, -1, -1):
        step = steps[step_idx]
        if step.get("source") != "agent":
            continue
        msg = step.get("message") or ""
        if isinstance(msg, str) and msg.strip():
            return [
                _evidence_ref(
                    source="trajectory.json",
                    json_pointer=f"/steps/{step_idx}",
                    kind="final_response",
                    label="Final response",
                    excerpt=msg,
                )
            ]
    return []


def _tool_call_ref(step_idx: int, tc: dict[str, Any], *, kind: str) -> dict[str, Any]:
    fn = str(tc.get("function_name") or "")
    args = tc.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    command = ""
    if _tool_name_looks_like_exec(fn.lower()):
        command = _command_arg_text(args.get("command") or args.get("cmd") or args.get("code"))
    path = _tool_file_path(args)
    if not path and command:
        path = _first_expected_artifact_path(command)
    excerpt = command or path or json.dumps(args, sort_keys=True)
    label_detail = command or path or fn
    json_pointer = f"/steps/{step_idx}/tool_calls/{tc['_atif_raw_tool_index']}"
    inner_index = tc.get("_atif_inner_tool_index")
    return _evidence_ref(
        source="trajectory.json",
        json_pointer=json_pointer,
        kind=kind,
        label=f"{fn}: {label_detail}" if label_detail else fn,
        path=path or None,
        excerpt=excerpt,
        evidence_id=f"{json_pointer}/normalized/{inner_index}" if inner_index is not None else None,
    )


def _tool_call_refs(traj: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for step_idx, _, tc in _agent_tool_calls(traj):
        if len(refs) >= _METRIC_EVIDENCE_MAX_TOOL_REFS:
            return refs
        refs.append(_tool_call_ref(step_idx, tc, kind="tool_call"))
    return refs


def _tool_observation_refs(traj: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for step_idx, step in enumerate(traj.get("steps", [])):
        if step.get("source") != "agent":
            continue
        for result_idx, result in enumerate((step.get("observation") or {}).get("results") or []):
            if len(refs) >= _METRIC_EVIDENCE_MAX_TOOL_REFS:
                return refs
            content = str(result.get("content") or "")
            if not content.strip():
                continue
            call_id = str(result.get("source_call_id") or f"result-{result_idx}")
            refs.append(
                _evidence_ref(
                    source="trajectory.json",
                    json_pointer=f"/steps/{step_idx}/observation/results/{result_idx}",
                    kind="tool_observation",
                    label=f"Tool observation: {call_id}",
                    excerpt=content,
                )
            )
    return refs


def _file_change_refs(traj: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for step_idx, _, tc in _agent_tool_calls(traj):
        if len(refs) >= _METRIC_EVIDENCE_MAX_FILE_REFS:
            return refs
        fn = str(tc.get("function_name") or "")
        fn_lower = fn.lower()
        args = tc.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        command = _command_arg_text(args.get("command") or args.get("cmd") or args.get("code"))
        is_write = _tool_name_looks_like_write(fn_lower) or (
            _tool_name_looks_like_exec(fn_lower) and _command_looks_like_write(command)
        )
        if not is_write:
            continue
        refs.append(_tool_call_ref(step_idx, tc, kind="file_change"))
    return refs


def _first_expected_artifact_path(text: str) -> str:
    match = _EXPECTED_ARTIFACT_PATH_RE.search(str(text or ""))
    if not match:
        return ""
    return match.group(0).rstrip(".,;:)]}'\"")


def _expected_artifact_refs(
    ground_truth: str,
    expected_behavior: list[str],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    sources: list[tuple[str, str, str]] = []
    if ground_truth:
        sources.append(("/ground_truth", "ground_truth", str(ground_truth)))
    for idx, behavior in enumerate(expected_behavior):
        if str(behavior or "").strip():
            sources.append((f"/expected_behavior/{idx}", "expected_behavior", str(behavior)))

    for pointer, source_kind, text in sources:
        for match in _EXPECTED_ARTIFACT_PATH_RE.finditer(text):
            path = match.group(0).rstrip(".,;:)]}'\"")
            refs.append(
                _evidence_ref(
                    source="evals.json",
                    json_pointer=pointer,
                    kind="expected_artifact",
                    label=f"Expected artifact: {path}",
                    path=path,
                    excerpt=text,
                    status="not_checked",
                )
            )
            if source_kind == "expected_behavior":
                break
    return _dedupe_evidence_refs(refs)


def _expected_behavior_refs(expected_behavior: list[str]) -> list[dict[str, Any]]:
    return [
        _evidence_ref(
            source="evals.json",
            json_pointer=f"/expected_behavior/{idx}",
            kind="expected_behavior",
            label=f"Expected behavior {idx + 1}",
            excerpt=str(behavior),
        )
        for idx, behavior in enumerate(expected_behavior)
        if str(behavior or "").strip()
    ]


def _ground_truth_ref(ground_truth: str) -> list[dict[str, Any]]:
    if not str(ground_truth or "").strip():
        return []
    return [
        _evidence_ref(
            source="evals.json",
            json_pointer="/ground_truth",
            kind="ground_truth",
            label="Expected answer",
            excerpt=ground_truth,
        )
    ]


def build_metric_evidence_refs(
    traj: dict[str, Any],
    question: str,
    *,
    ground_truth: str = "",
    expected_behavior: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build compact source refs for LLM-judged metrics.

    The refs are intentionally small: downstream reports can cite the exact
    trajectory or eval-entry location without embedding raw trajectory text in
    ``reward.json``.
    """
    _ = question
    if not isinstance(expected_behavior, list):
        expected_behavior = []

    final_refs = _final_response_ref(traj)
    tool_refs = _tool_call_refs(traj)
    observation_refs = _tool_observation_refs(traj)
    file_refs = _file_change_refs(traj)
    ground_truth_refs = _ground_truth_ref(ground_truth)
    behavior_refs = _expected_behavior_refs(expected_behavior)
    artifact_refs = _expected_artifact_refs(ground_truth, expected_behavior)

    return {
        "accuracy": _dedupe_evidence_refs([*ground_truth_refs, *final_refs]),
        "goal_accuracy": _dedupe_evidence_refs(
            [
                *ground_truth_refs,
                *tool_refs,
                *observation_refs,
                *final_refs,
                *artifact_refs,
            ]
        ),
        "behavior_check": _dedupe_evidence_refs(
            [
                *behavior_refs,
                *file_refs,
                *final_refs,
                *artifact_refs,
            ]
        ),
    }


def attach_metric_evidence_refs(
    details: dict[str, Any],
    evidence_refs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Attach evidence refs to existing metric detail dictionaries in place."""
    for metric in _METRIC_EVIDENCE_REF_METRICS:
        refs = evidence_refs.get(metric) or []
        if not refs:
            continue
        existing = details.get(metric)
        if isinstance(existing, dict):
            existing["evidence_refs"] = refs
        else:
            details[metric] = {"value": existing, "evidence_refs": refs}
    return details


# ---------------------------------------------------------------------------
# Metric Evidence Compiler  (judge-facing evidence, distinct from compact refs)
# ---------------------------------------------------------------------------

_BUNDLE_ITEM_CHARS = 1500  # per-item excerpt for the judge prompt (refs use 300)
_DEFAULT_ACCURACY_BUDGET = 8000
_DEFAULT_GOAL_ACCURACY_BUDGET = 12000
_BUNDLE_ACCURACY_MAX_OBS = 6  # newest observations fed to accuracy (<= ~6x1500 <= budget)
_BUNDLE_GOAL_MAX_OBS = 12  # newest observations fed to goal_accuracy (end-state)
_BUNDLE_RESERVED_OBS = 2  # newest observations file changes always leave room for
# Below this budget, a final response that does not fit takes all of it.
_BUNDLE_FINAL_SHARE_MIN_BUDGET = 160


def _accuracy_budget() -> int:
    """Return the configured accuracy evidence budget or default."""
    return _env_positive_int("SKILL_EVAL_ACCURACY_BUDGET", _DEFAULT_ACCURACY_BUDGET)


def _goal_accuracy_budget() -> int:
    """Return the configured goal accuracy evidence budget or default."""
    return _env_positive_int("SKILL_EVAL_GOAL_ACCURACY_BUDGET", _DEFAULT_GOAL_ACCURACY_BUDGET)


def _bundle_budgets() -> dict[str, int]:
    """Return effective bundle budgets taking into account runtime overrides."""
    return {
        "accuracy": _accuracy_budget(),
        "goal_accuracy": _goal_accuracy_budget(),
        "behavior_check": _behavior_check_budget(),
    }


def _late_observation_excerpts(traj: dict[str, Any], limit: int, max_items: int) -> list[str]:
    """Most-recent tool observations first (end-state evidence), each redacted and clipped."""
    out: list[str] = []
    for step in reversed(traj.get("steps", [])):
        if step.get("source") != "agent":
            continue
        for result in reversed((step.get("observation") or {}).get("results") or []):
            if len(out) >= max_items:
                return out
            content = _judge_excerpt(result.get("content"), limit)
            if content:
                out.append(content)
    return out


def _assemble_bundle(
    final: str,
    entries: list[list[Any]],
    files_title: str,
    observations: list[str],
    obs_title: str,
    budget: int,
) -> tuple[str, int, bool]:
    """FINAL RESPONSE, then file changes, then the newest tool results that fit.

    File changes leave room for the newest ``_BUNDLE_RESERVED_OBS`` results, and
    for more while they fit in that many items' room, so big writes never push
    out a late test run. A final response longer than the budget keeps its head
    and tail in half of it (all of it below ``_BUNDLE_FINAL_SHARE_MIN_BUDGET``
    or with nothing else to show). Returns (text, omitted, truncated).
    """
    final = final.strip()
    final_cut = False
    if final and len(_SECTION_FINAL_RESPONSE) + 1 + len(final) > budget:
        share = budget // 2 if budget >= _BUNDLE_FINAL_SHARE_MIN_BUDGET and (entries or observations) else budget
        final = _truncate_for_behavior(final, share - len(_SECTION_FINAL_RESPONSE) - 1, recount=False)
        final_cut = True
    final_block = len(_SECTION_FINAL_RESPONSE) + 1 + len(final) if final else 0
    used = final_block + 2 if final_block else 0
    reserved = observations[:_BUNDLE_RESERVED_OBS]
    for obs in observations[_BUNDLE_RESERVED_OBS:]:
        if len("\n---\n".join([*reserved, obs])) > _BUNDLE_RESERVED_OBS * _BUNDLE_ITEM_CHARS:
            break
        reserved.append(obs)
    obs_room = len(obs_title) + 1 + len("\n---\n".join(reserved)) + 2 if reserved else 0
    files, files_cut = _fit_file_changes(entries, budget - used - obs_room - len(files_title) - 1)
    if files:
        used += len(files_title) + 1 + len(files) + 2
    kept: list[str] = []
    for obs in observations:
        if used + len(obs_title) + 1 + len("\n---\n".join([*kept, obs])) > budget:
            break
        kept.append(obs)
    text, dropped, truncated = _assemble(
        [(_SECTION_FINAL_RESPONSE, final), (files_title, files), (obs_title, "\n---\n".join(kept))], budget
    )
    omitted = dropped + len(observations) - len(kept) + (1 if entries and not files else 0) + (1 if final_cut else 0)
    return text, omitted, truncated or files_cut or omitted > 0


def _assemble(sections: list[tuple[str, str]], budget: int) -> tuple[str, int, bool]:
    """Join (title, body) sections under a char budget. Returns (text, dropped, truncated).

    A FINAL RESPONSE that does not fit is cut to its head and tail rather than
    dropped (and counted as dropped), leaving room for the smallest later
    section when the budget allows.
    """
    parts: list[str] = []
    used = 0
    dropped = 0
    truncated = False
    non_empty = [(title, str(body or "").strip()) for title, body in sections if str(body or "").strip()]
    for idx, (title, body) in enumerate(non_empty):
        block = f"{title}\n{body}"
        if used + len(block) <= budget:
            parts.append(block)
            used += len(block) + 2
        elif title == _SECTION_FINAL_RESPONSE and budget - used > 0:
            avail = budget - used
            later_blocks = [len(f"{t}\n{b}") + 2 for t, b in non_empty[idx + 1 :]]
            if later_blocks and avail >= _BUNDLE_FINAL_SHARE_MIN_BUDGET:
                reserve_later = min(avail // 2, *later_blocks)
                if avail - reserve_later > len(title) + 16:
                    avail -= reserve_later
            header = f"{title}\n"
            if avail > len(header):
                clipped_block = f"{header}{_truncate_for_behavior(body, avail - len(header), recount=False)}"
            else:
                clipped_block = block[:avail]
            parts.append(clipped_block)
            used += len(clipped_block) + 2
            dropped += 1
            truncated = True
        else:
            dropped += 1
            truncated = True
    return "\n\n".join(parts), dropped, truncated


_BACKTICK_TOKEN_RE = re.compile(r"`([^`]{4,})`")
_VERIFIED_FACTS_MAX = 12
_VERIFIED_FACT_LINE_MAX = 200


def build_verified_facts(
    traj: dict[str, Any],
    expected_behavior: list[str] | None,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Derive deterministic facts from the trajectory vs expected tokens.

    Each fact: {"claim": str, "observed": bool, "step_id": int|None, "evidence": str}.
    Only emits facts for tokens extractable from *expected_behavior* and *ground_truth*
    via artifact-path regex or backtick-quoted snippets. No fuzzy matching, no prose.
    """
    if not isinstance(expected_behavior, list):
        expected_behavior = []

    # 1. Collect checkable tokens from expected_behavior and ground_truth
    tokens: list[tuple[str, str]] = []  # (claim, match_mode) where mode is "path" or "ci"
    seen_claims: set[str] = set()

    sources = list(expected_behavior) + ([ground_truth] if ground_truth else [])
    for source in sources:
        text = str(source or "")
        # a. artifact paths
        for match in _EXPECTED_ARTIFACT_PATH_RE.finditer(text):
            claim = match.group(0).rstrip(".,;:)]}'\"")
            if claim and claim not in seen_claims:
                seen_claims.add(claim)
                tokens.append((claim, "path"))
        # b. backtick-quoted snippets >= 4 chars (strip backticks)
        for match in _BACKTICK_TOKEN_RE.finditer(text):
            claim = match.group(1)  # strip the backticks
            if claim and claim not in seen_claims:
                seen_claims.add(claim)
                tokens.append((claim, "ci"))

    if not tokens:
        return []

    # 2. Read each tool call's checkable text once: command/cmd/code, the
    # file-path argument, and the written body.
    calls: list[tuple[int, str, str, str]] = []
    for idx, _, tc in _agent_tool_calls(traj):
        args = tc.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        command = _command_arg_text(args.get("command") or args.get("cmd") or args.get("code"))
        write = _write_call_parts(str(tc.get("function_name") or ""), args)
        write_body = write[1] if write else _tool_write_body(args)
        calls.append((idx, command, _tool_file_path(args), write_body))

    # 3. Scan the calls for each token
    facts: list[dict[str, Any]] = []
    for claim, mode in tokens:
        if len(facts) >= _VERIFIED_FACTS_MAX:
            break
        observed = False
        step_id = None
        evidence = ""

        for idx, command, file_arg, write_body in calls:
            for candidate in (command, file_arg, write_body):
                if not candidate:
                    continue
                needle = claim
                haystack = candidate
                if mode == "ci":
                    needle = claim.lower()
                    haystack = candidate.lower()
                if needle in haystack:
                    observed = True
                    step_id = idx
                    # Use the actual (non-lowercased) command as evidence
                    evidence = _judge_excerpt(command or file_arg or write_body, 160)
                    break
            if observed:
                break

        facts.append(
            {
                "claim": claim,
                "observed": observed,
                "step_id": step_id,
                "evidence": evidence,
            }
        )

    return facts


def _build_verified_facts_section(facts: list[dict[str, Any]]) -> str:
    """Build the VERIFIED FACTS header string to prepend to prompt_evidence."""
    if not facts:
        return ""
    lines = ["VERIFIED FACTS (deterministic):"]
    for fact in facts:
        claim = fact["claim"]
        if fact["observed"]:
            sid = fact["step_id"]
            ev = fact["evidence"]
            line = f"- [OBSERVED step {sid}] {claim}"
            if ev:
                line = f"{line} :: {ev}"
            if len(line) > _VERIFIED_FACT_LINE_MAX:
                line = line[:_VERIFIED_FACT_LINE_MAX]
        else:
            line = f"- [NOT OBSERVED] {claim}"
            if len(line) > _VERIFIED_FACT_LINE_MAX:
                line = line[:_VERIFIED_FACT_LINE_MAX]
        lines.append(line)
    return "\n".join(lines)


def build_metric_evidence_bundles(
    traj: dict[str, Any],
    question: str,
    *,
    ground_truth: str = "",
    expected_behavior: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compile per-metric judge-facing evidence.

    Each bundle = {prompt_evidence, evidence_refs, omitted, verified}. ``prompt_evidence``
    replaces the old ``agent_text[:3000]`` / ``tool_summary[:2000]`` / 4000-char
    blob: it is relevance-/recency-selected, always includes the final response,
    and records what it dropped or shortened (never silent). Deterministic
    verified facts are prepended at the top when any checkable tokens are
    found. Trajectory text is secret-redacted, and file changes are fitted
    into the room the final response and the newest tool results leave,
    latest write to each path first.
    """
    if not isinstance(expected_behavior, list):
        expected_behavior = []
    refs = build_metric_evidence_refs(traj, question, ground_truth=ground_truth, expected_behavior=expected_behavior)

    # Compute verified facts once; prepend the same section to all metrics
    facts = build_verified_facts(traj, expected_behavior, ground_truth)
    facts_section = _build_verified_facts_section(facts)

    final = _redact_evidence_text(get_final_response(traj))
    file_entries = _file_change_entries(traj)
    late_obs = _late_observation_excerpts(traj, _BUNDLE_ITEM_CHARS, max(_BUNDLE_ACCURACY_MAX_OBS, _BUNDLE_GOAL_MAX_OBS))

    def _prepend_facts(text: str) -> str:
        if not facts_section:
            return text
        if text:
            return f"{facts_section}\n\n{text}"
        return facts_section

    bundles: dict[str, dict[str, Any]] = {}
    budgets = _bundle_budgets()

    acc_text, acc_drop, acc_trunc = _assemble_bundle(
        final,
        file_entries,
        "PRODUCED FILES / WRITES",
        late_obs[:_BUNDLE_ACCURACY_MAX_OBS],
        "KEY OBSERVATIONS",
        budgets["accuracy"],
    )
    bundles["accuracy"] = {
        "prompt_evidence": _prepend_facts(acc_text or _judge_excerpt(get_agent_text(traj), budgets["accuracy"])),
        "evidence_refs": refs["accuracy"],
        "omitted": {
            "count": acc_drop,
            "truncated": acc_trunc,
            "reason": "low-relevance sections dropped or shortened to fit budget" if acc_trunc else "",
        },
        "verified": facts,
    }

    goal_text, goal_drop, goal_trunc = _assemble_bundle(
        final,
        file_entries,
        "END-STATE FILE CHANGES",
        late_obs[:_BUNDLE_GOAL_MAX_OBS],
        "RECENT TOOL RESULTS (newest first)",
        budgets["goal_accuracy"],
    )
    bundles["goal_accuracy"] = {
        "prompt_evidence": _prepend_facts(goal_text or _judge_excerpt(get_agent_text(traj), budgets["goal_accuracy"])),
        "evidence_refs": refs["goal_accuracy"],
        "omitted": {
            "count": goal_drop,
            "truncated": goal_trunc,
            "reason": "older/low-relevance evidence dropped or shortened to fit budget" if goal_trunc else "",
        },
        "verified": facts,
    }

    # Leave room for the facts header so the judge never has to re-cut this.
    bc_budget = max(1, budgets["behavior_check"] - (len(facts_section) + 2 if facts_section else 0))
    final_limit = _behavior_final_response_limit()
    histories: dict[tuple[bool, int, int], list[list[Any]]] = {}
    bc_text = _behavior_evidence(traj, question, bc_budget, final_limit, file_entries, histories)
    bc_full = _behavior_evidence(traj, question, 10**9, final_limit, file_entries, histories)
    bc_trunc = len(bc_full) > len(bc_text) or any(entry[6] for entry in file_entries)
    bundles["behavior_check"] = {
        "prompt_evidence": _prepend_facts(bc_text),
        "evidence_refs": refs["behavior_check"],
        "omitted": {
            "count": 1 if bc_trunc else 0,
            "truncated": bc_trunc,
            "reason": "lower-priority behavior history truncated to fit budget" if bc_trunc else "",
        },
        "verified": facts,
    }
    return bundles


def extract_tool_calls_as_dicts(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert ATIF trajectory to the ``{"action", "action_input", "observation"}``
    format consumed by ``eval_core.checks``.
    """
    result: list[dict[str, Any]] = []
    for _, step, tc in _agent_tool_calls(traj):
        call = {
            "action": tc.get("function_name", ""),
            "action_input": tc.get("arguments") or {},
            "observation": _tool_call_observation(step, tc),
        }
        if status := tc.get("_atif_normalization_status"):
            call["normalization_status"] = status
        if status := tc.get("_atif_observation_status"):
            call["observation_status"] = status
        if wrapper_observation := _tool_call_wrapper_observation(step, tc):
            call["wrapper_observation"] = wrapper_observation
        result.append(call)
    return result
