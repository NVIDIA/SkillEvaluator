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
import logging
import os
import re
from typing import Any

if __package__:
    from skillevaluator.evidence import evidence_ref_identity
    from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import (
        atif_content_has_text,
        atif_content_text,
        normalized_tool_call_error_flag,
        normalized_tool_call_is_error,
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
    from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line
else:  # Standalone helper staged beside the verifier.
    from codex_tool_call_normalizer import (
        atif_content_has_text,
        atif_content_text,
        normalized_tool_call_error_flag,
        normalized_tool_call_is_error,
    )
    from codex_tool_call_normalizer import (
        iter_normalized_tool_calls as iter_tool_calls,
    )
    from codex_tool_call_normalizer import (
        normalized_tool_call_observation as _tool_call_observation,
    )
    from codex_tool_call_normalizer import (
        normalized_tool_call_wrapper_observation as _tool_call_wrapper_observation,
    )
    from evidence import evidence_ref_identity
    from secret_redaction import redact_secrets_in_log_line

logger = logging.getLogger(__name__)


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
    for step, tc in iter_tool_calls(traj):
        if (tc.get("function_name") or "").lower() != "skill":
            continue
        if normalized_tool_call_is_error(step, tc):
            continue
        args = tc.get("arguments") or {}
        name = args.get("skill", args.get("name", ""))
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
            msg = atif_content_text(step.get("message"))
            if msg.strip():
                parts.append(msg)
    return "\n".join(parts)


def get_final_response(traj: dict[str, Any]) -> str:
    """Get the last text-bearing agent message from the trajectory."""
    for step in reversed(traj.get("steps", [])):
        if step.get("source") == "agent":
            content = step.get("message")
            if atif_content_has_text(content):
                return atif_content_text(content)
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


def build_conversation_summary(traj: dict[str, Any], question: str, *, full: bool = False) -> str:
    """Build a human-readable conversation summary for LLM judges.

    The summary is optimized for behavior-check and goal-accuracy prompts.

    Each entry is clipped to a per-entry allowance so one noisy tool result
    cannot crowd out the rest of the run. Pass ``full=True`` for the same
    rendering with those per-entry clips lifted: the completeness accounting
    compares against that, so content this function drops is not mistaken for
    content that reached the judge.
    """

    def clip(text: str, limit: int) -> str:
        return text if full else text[:limit]

    parts = [f"User: {question}"]

    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue

        reasoning = step.get("reasoning_content") or ""
        if reasoning:
            parts.append(f"Agent reasoning: {clip(str(reasoning), 200)}")

        for _, tc in iter_tool_calls({"steps": [step]}):
            fn = tc.get("function_name", "")
            args = tc.get("arguments") or {}
            parts.append(f"Agent called: {fn}({clip(json.dumps(args), 200)})")

        obs = step.get("observation") or {}
        for r in obs.get("results") or []:
            content = atif_content_text(r.get("content"))
            if content:
                parts.append(f"Tool returned: {clip(content, 400)}")

        msg = atif_content_text(step.get("message"))
        if msg.strip() and not step.get("tool_calls"):
            parts.append(f"Agent: {clip(msg, 1500)}")

    final = get_final_response(traj)
    if final and not any(final[:50] in p for p in parts):
        parts.append(f"Agent final answer: {clip(final, 1500)}")

    return "\n".join(parts)


_BEHAVIOR_EVIDENCE_MAX_CHARS = 4000
# A realistic final answer runs to a few thousand characters, so the old 800
# cap cut grading-relevant text out of the middle of ordinary answers (issue
# #234). Allow a whole answer through, and keep the total budget finite by
# reconciling it with the tool-history headroom in _behavior_check_budget.
_DEFAULT_BEHAVIOR_FINAL_RESPONSE_LIMIT = 8000
_DEFAULT_BEHAVIOR_CHECK_BUDGET = _DEFAULT_BEHAVIOR_FINAL_RESPONSE_LIMIT + 4000
_DEFAULT_TOOL_HISTORY_HEADROOM = 4000
_MIN_BEHAVIOR_HISTORY_HEADROOM = 1600
# The user request is also repeated, in full, at the head of the compact tool
# history, so its own section is a convenience copy rather than the only one.
# Giving it the whole final-response allowance therefore buys a duplicate at
# the tool history's expense, which is the one place a non-file behavior
# ("ran the tests before finishing") is observable. Cap it well below the
# answer's allowance instead.
_DEFAULT_BEHAVIOR_USER_REQUEST_LIMIT = 2000
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


def _env_int_or_none(name: str) -> int | None:
    """Return environment variable *name* as an int, or ``None`` when unusable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


_warned_behavior_budgets: set[tuple[int, int]] = set()


def _behavior_check_budget() -> int:
    """Return the behavior check evidence budget.

    A usable ``SKILL_EVAL_BEHAVIOR_CHECK_BUDGET`` is honoured exactly as set.
    Operators pin it to cap the judge prompt for a small-context model or a
    token budget, so quietly raising it past what they asked for defeats the
    only reason to set it. A value too small to hold the final response beside
    some tool history is warned about rather than dropped, and the behavior
    judge's evidence gate then reports the loss instead of scoring it. Only
    the *default* is reconciled upwards with the final-response limit, so
    raising that limit alone still leaves room for the tool history.
    Unparseable and non-positive values are not budgets at all and fall back
    to that reconciled default.
    """
    final_limit = _behavior_final_response_limit()
    configured = _env_int_or_none("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET")
    if configured is None or configured < 1:
        base_budget = _env_positive_int("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", _DEFAULT_BEHAVIOR_CHECK_BUDGET)
        return max(base_budget, final_limit + _DEFAULT_TOOL_HISTORY_HEADROOM)
    if configured < final_limit + _MIN_BEHAVIOR_HISTORY_HEADROOM:
        key = (configured, final_limit)
        if key not in _warned_behavior_budgets:
            _warned_behavior_budgets.add(key)
            logger.warning(
                "SKILL_EVAL_BEHAVIOR_CHECK_BUDGET=%d leaves under %d characters beside a "
                "SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT of %d. The budget is honoured as set, so behavior "
                "evidence will be cut and behavior_check may report incomplete evidence instead of a score. "
                "Lower SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT as well, or raise the budget to at least %d.",
                configured,
                _MIN_BEHAVIOR_HISTORY_HEADROOM,
                final_limit,
                final_limit + _MIN_BEHAVIOR_HISTORY_HEADROOM,
            )
    return configured


def _behavior_user_request_limit() -> int:
    """Return the behavior evidence allowance for the user request section.

    The old fixed 800 characters cut ordinary Tier 3 task prompts, but the
    whole final-response allowance is too much: ``build_conversation_summary``
    already opens the compact tool history with ``User: {question}``, so every
    character spent here is a second copy, and the history — appended after
    this section — absorbs the entire loss. Cap it at
    ``_DEFAULT_BEHAVIOR_USER_REQUEST_LIMIT``, never above the final-response
    limit so that lowering that limit still shrinks the whole bundle.
    """
    return min(_DEFAULT_BEHAVIOR_USER_REQUEST_LIMIT, _behavior_final_response_limit())


_BEHAVIOR_WRITE_TOOLS = {
    "write",
    "write_file",
    "edit",
    "edit_file",
    "multiedit",
    "notebookedit",
    "apply_patch",
}
_BEHAVIOR_EXEC_TOOLS = {"bash", "execute", "exec_command", "run_code", "run", "shell", "command"}
_BEHAVIOR_WRITE_COMMAND_MARKERS = ("tee ", "apply_patch")
_BEHAVIOR_WRITE_REDIRECT_RE = re.compile(r"(?:^|[\s;])(?:>|>>)\s*(?![&0-9])[^&\s;|]+")
_BEHAVIOR_PYTHON_WRITE_RE = re.compile(
    r"\b(?:write_text|write_bytes)\s*\(|\bopen\s*\([^)]*,\s*['\"][wa]",
    re.IGNORECASE,
)
_TOOL_NAME_SEPARATORS = (".", ":", "/", "__")


def _truncate_for_behavior(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker) + 1:
        return text[:limit]
    head = max(1, (limit - len(marker)) * 2 // 3)
    tail = max(1, limit - len(marker) - head)
    return f"{text[:head]}{marker}{text[-tail:]}"


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
    section = f"{title}\n{_truncate_for_behavior(body.strip(), budget - len(title) - 2)}"
    if not section.strip():
        return max_chars
    parts.append(section)
    return max(0, max_chars - len(section) - 2)


def _tool_file_path(args: dict[str, Any]) -> str:
    for key in ("file_path", "path", "filename", "target_file"):
        value = args.get(key)
        if value:
            return str(value)
    return ""


def _tool_write_body(args: dict[str, Any]) -> str:
    snippets: list[str] = []
    for key in ("content", "new_string", "patch", "code"):
        value = args.get(key)
        if value:
            snippets.append(f"{key}:\n{value}")
    edits = args.get("edits")
    if isinstance(edits, list):
        for idx, edit in enumerate(edits[:5], start=1):
            if isinstance(edit, dict):
                new_string = edit.get("new_string") or edit.get("replacement")
                if new_string:
                    snippets.append(f"edit {idx} new_string:\n{new_string}")
    return "\n\n".join(str(s) for s in snippets if str(s).strip())


def _command_looks_like_write(command: str) -> bool:
    lower = command.lower()
    return any(marker in lower for marker in _BEHAVIOR_WRITE_COMMAND_MARKERS) or bool(
        _BEHAVIOR_WRITE_REDIRECT_RE.search(command) or _BEHAVIOR_PYTHON_WRITE_RE.search(command)
    )


def _tool_name_looks_like_write(fn_lower: str) -> bool:
    candidates = {fn_lower}
    for separator in _TOOL_NAME_SEPARATORS:
        if separator in fn_lower:
            candidates.add(fn_lower.rsplit(separator, 1)[-1])
    return any(candidate in _BEHAVIOR_WRITE_TOOLS for candidate in candidates)


def _collect_file_change_evidence(traj: dict[str, Any], *, full: bool = False) -> list[str]:
    """Render one entry per write the agent made.

    Each write body and tool result is clipped to a per-entry allowance. Pass
    ``full=True`` for the same entries with those clips lifted, which is what
    the completeness accounting compares against so a body this function
    shortened is not reported as having reached the judge.
    """
    body_limit = 10**9 if full else 1800
    observation_limit = 10**9 if full else 500
    changes: list[str] = []
    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue
        for _, tc in iter_tool_calls({"steps": [step]}):
            fn = str(tc.get("function_name") or "")
            fn_lower = fn.lower()
            args = tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}

            body = ""
            is_write_call = False
            file_path = _tool_file_path(args)
            if _tool_name_looks_like_write(fn_lower):
                is_write_call = True
                body = _tool_write_body(args)
            elif fn_lower in _BEHAVIOR_EXEC_TOOLS:
                command = str(args.get("command") or args.get("cmd") or args.get("code") or "")
                if _command_looks_like_write(command):
                    is_write_call = True
                    body = f"command:\n{command}"

            if not is_write_call or (not body and not file_path):
                continue

            obs = _tool_call_observation(step, tc)
            entry_parts = [f"Agent called: {fn}"]
            if file_path:
                entry_parts.append(f"Path: {file_path}")
            if body:
                entry_parts.append(_truncate_for_behavior(body, body_limit))
            if obs:
                entry_parts.append(f"Tool returned: {_truncate_for_behavior(obs, observation_limit)}")
            changes.append("\n".join(entry_parts))
    return changes


def build_behavior_evidence(
    traj: dict[str, Any],
    question: str,
    max_chars: int | None = None,
    final_response_limit: int | None = None,
) -> str:
    """Build compact, behavior-check-specific evidence from an ATIF trajectory.

    Behavior checks often ask whether the agent produced or changed artifacts.
    Put final output and write/edit evidence before exploratory reads so a
    fixed-size judge prompt does not miss late file creation.
    """
    effective_final_limit = (
        _behavior_final_response_limit() if final_response_limit is None else max(1, int(final_response_limit))
    )

    if max_chars is None:
        raw_budget = os.environ.get("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "").strip()
        if raw_budget:
            max_chars = _behavior_check_budget()
        else:
            max_chars = max(
                _BEHAVIOR_EVIDENCE_MAX_CHARS,
                effective_final_limit + _MIN_BEHAVIOR_HISTORY_HEADROOM,
            )

    parts: list[str] = []
    remaining = max_chars

    file_changes = "\n\n".join(_collect_file_change_evidence(traj))
    final = get_final_response(traj)
    history = build_conversation_summary(traj, question)
    user_request_limit = _behavior_user_request_limit()
    user_needed = (
        min(user_request_limit, len(_SECTION_USER_REQUEST) + len(question.strip()) + 2)
        if question and question.strip()
        else 0
    )
    history_needed = len(_SECTION_COMPACT_TOOL_HISTORY) + len(history.strip()) + 2 if history and history.strip() else 0
    tail_needed = user_needed + (2 if user_needed and history_needed else 0) + history_needed

    if file_changes:
        file_section_limit: int | None = None
        if final and final.strip():
            reserved_tail = min(_MIN_BEHAVIOR_HISTORY_HEADROOM, max_chars // 4, tail_needed)
            reserved_final = min(
                effective_final_limit,
                len(_SECTION_FINAL_RESPONSE) + len(final.strip()) + 2,
                max(1, max_chars - min(_MIN_BEHAVIOR_HISTORY_HEADROOM, max_chars // 2)),
            )
            if (
                len(_SECTION_FILE_CHANGES) + len(file_changes.strip()) + 2 + reserved_final + reserved_tail + 4
                > remaining
            ):
                file_section_limit = max(
                    min(800, remaining // 3),
                    remaining - reserved_final - reserved_tail - 4,
                )
        remaining = _append_section_with_budget(
            parts,
            _SECTION_FILE_CHANGES,
            file_changes,
            remaining,
            section_limit=file_section_limit,
        )

    if final:
        if max_chars > effective_final_limit:
            reserved_headroom = min(
                _MIN_BEHAVIOR_HISTORY_HEADROOM,
                remaining // 2,
                tail_needed,
                max(800, remaining - effective_final_limit),
            )
        else:
            reserved_headroom = min(_MIN_BEHAVIOR_HISTORY_HEADROOM, remaining // 2, tail_needed)
        bounded_final_limit = min(effective_final_limit, max(1, remaining - reserved_headroom))
        remaining = _append_section_with_budget(
            parts,
            _SECTION_FINAL_RESPONSE,
            final,
            remaining,
            section_limit=bounded_final_limit,
        )

    remaining = _append_section_with_budget(
        parts,
        _SECTION_USER_REQUEST,
        question,
        remaining,
        section_limit=user_request_limit,
    )

    remaining = _append_section_with_budget(parts, _SECTION_COMPACT_TOOL_HISTORY, history, remaining)

    return "\n\n".join(parts)[:max_chars]


_METRIC_EVIDENCE_REF_METRICS = ("accuracy", "goal_accuracy", "behavior_check")
_METRIC_EVIDENCE_EXCERPT_CHARS = 300
_METRIC_EVIDENCE_MAX_TOOL_REFS = 20
_METRIC_EVIDENCE_MAX_FILE_REFS = 12
_EXPECTED_ARTIFACT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:logs/agent|workspace/output|output)"
    r"[A-Za-z0-9._/+=:@-]*[A-Za-z0-9_./+=:@-]"
)


def _redact_evidence_text(text: str) -> str:
    # Mirror harbor/templates/eval.py: also redact the RUNTIME values of the
    # API keys this process holds, which need not match sk-/nvapi- patterns.
    redacted = redact_secrets_in_log_line(
        str(text or ""),
        extra_secret_values=[
            os.environ.get("NVIDIA_API_KEY", ""),
        ],
    )
    return redacted.replace("\x00", "").strip()


def _evidence_excerpt(text: str, limit: int = _METRIC_EVIDENCE_EXCERPT_CHARS) -> str:
    return _truncate_for_behavior(_redact_evidence_text(text), limit)


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
        content = step.get("message")
        if atif_content_has_text(content):
            msg = atif_content_text(content)
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
    if fn.lower() in _BEHAVIOR_EXEC_TOOLS:
        command = str(args.get("command") or args.get("cmd") or args.get("code") or "")
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
    for step_idx, step in enumerate(traj.get("steps", [])):
        if step.get("source") != "agent":
            continue
        for _, tc in iter_tool_calls({"steps": [step]}):
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
            content = atif_content_text(result.get("content"))
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
    for step_idx, step in enumerate(traj.get("steps", [])):
        if step.get("source") != "agent":
            continue
        for _, tc in iter_tool_calls({"steps": [step]}):
            if len(refs) >= _METRIC_EVIDENCE_MAX_FILE_REFS:
                return refs
            fn = str(tc.get("function_name") or "")
            fn_lower = fn.lower()
            args = tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            command = str(args.get("command") or args.get("cmd") or args.get("code") or "")
            is_write = _tool_name_looks_like_write(fn_lower) or (
                fn_lower in _BEHAVIOR_EXEC_TOOLS and _command_looks_like_write(command)
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
_BUNDLE_BUDGETS = {
    "accuracy": _DEFAULT_ACCURACY_BUDGET,
    "goal_accuracy": _DEFAULT_GOAL_ACCURACY_BUDGET,
    "behavior_check": _DEFAULT_BEHAVIOR_CHECK_BUDGET,
}
_BUNDLE_ACCURACY_MAX_OBS = 6  # newest observations fed to accuracy (<= ~6x1500 <= budget)
_BUNDLE_GOAL_MAX_OBS = 12  # newest observations fed to goal_accuracy (end-state)


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


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + " …[clipped]"


def _late_observation_excerpts(traj: dict[str, Any], limit: int) -> list[str]:
    """Most-recent tool observations first (end-state evidence), each clipped."""
    out: list[str] = []
    for step in reversed(traj.get("steps", [])):
        if step.get("source") != "agent":
            continue
        for result in reversed((step.get("observation") or {}).get("results") or []):
            content = atif_content_text(result.get("content")).strip()
            if content:
                out.append(_clip(content, limit))
    return out


def _assemble(sections: list[tuple[str, str]], budget: int) -> tuple[str, int, bool]:
    """Join (title, body) sections under a char budget. Returns (text, dropped, truncated)."""
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
            if later_blocks and avail >= 160:
                reserve_later = min(avail // 2, *later_blocks)
                if avail - reserve_later > len(title) + 16:
                    avail -= reserve_later
            header = f"{title}\n"
            clip_suffix = " …[clipped]"
            if avail > len(header) + len(clip_suffix):
                clipped_body = _clip(body, avail - len(header) - len(clip_suffix))
                clipped_block = f"{header}{clipped_body}"
            elif avail > len(header):
                clipped_block = f"{header}{body[: avail - len(header)]}"
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


def _checkable_tokens(text: str, seen_claims: set[str] | None = None) -> list[tuple[str, str]]:
    """Extract ``(claim, match_mode)`` pairs from *text*.

    Modes are ``"path"`` for an artifact path and ``"ci"`` for a
    backtick-quoted snippet of four or more characters. Claims already in
    *seen_claims* are skipped, and new ones are added to it, so a caller can
    deduplicate across several sources.
    """
    if seen_claims is None:
        seen_claims = set()
    tokens: list[tuple[str, str]] = []
    body = str(text or "")
    # a. artifact paths
    for match in _EXPECTED_ARTIFACT_PATH_RE.finditer(body):
        claim = match.group(0).rstrip(".,;:)]}'\"")
        if claim and claim not in seen_claims:
            seen_claims.add(claim)
            tokens.append((claim, "path"))
    # b. backtick-quoted snippets >= 4 chars (strip backticks)
    for match in _BACKTICK_TOKEN_RE.finditer(body):
        claim = match.group(1)  # strip the backticks
        if claim and claim not in seen_claims:
            seen_claims.add(claim)
            tokens.append((claim, "ci"))
    return tokens


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
        tokens.extend(_checkable_tokens(source, seen_claims))

    if not tokens:
        return []

    # 2. Scan trajectory steps for each token
    steps = traj.get("steps", [])
    facts: list[dict[str, Any]] = []

    for claim, mode in tokens:
        if len(facts) >= _VERIFIED_FACTS_MAX:
            break
        observed = False
        step_id = None
        evidence = ""

        for idx, step in enumerate(steps):
            if step.get("source") != "agent":
                continue
            # Check tool call arguments: command/cmd/code and file-path args
            for _, tc in iter_tool_calls({"steps": [step]}):
                args = tc.get("arguments") or {}
                if not isinstance(args, dict):
                    continue
                # command-like args
                command = str(args.get("command") or args.get("cmd") or args.get("code") or "")
                # file-path args
                file_arg = _tool_file_path(args)
                # write-body content
                write_body = _tool_write_body(args)

                candidate_texts = [command, file_arg, write_body]
                for candidate in candidate_texts:
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
                        evidence = command or file_arg or write_body
                        evidence = evidence[:160]
                        break
                if observed:
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


def _source_reaches_judge(source: str, evidence: str) -> bool:
    """Return True when every character of *source* appears in *evidence*.

    A long source can arrive in two pieces: the compact history repeats the
    first 1,500 characters of a text-only final step, and the final-response
    section keeps the end. Find the longest prefix of the source present in
    *evidence*; the source is complete when the rest is present too.
    Whitespace at the split does not count, because the history section
    drops trailing whitespace.
    """
    answer = source.strip()
    lo, hi = 0, len(answer)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if answer[:mid] in evidence:
            lo = mid
        else:
            hi = mid - 1
    return answer[lo:].lstrip() in evidence


def _final_answer_reaches_judge(final: str, evidence: str) -> bool:
    """Return True when every character of the final answer appears in *evidence*."""
    return _source_reaches_judge(final, evidence)


_BEHAVIOR_SECTION_LABELS = (
    (_SECTION_FINAL_RESPONSE, "final response"),
    (_SECTION_USER_REQUEST, "user request"),
    (_SECTION_FILE_CHANGES, "file changes"),
    (_SECTION_COMPACT_TOOL_HISTORY, "compact tool history"),
)
# Only the final answer gates the behavior judge. It is the one piece of
# required evidence with a small, bounded source and a dedicated cap of its
# own, so a gap in it means a limit cut text the judge needed, never that the
# run was simply large (issue #234). File changes and the tool history have
# unbounded sources that any finite budget shortens on a long run, so their
# completeness is reported for diagnosis but never withholds a score. The user
# request reaches the judge twice (its own reserved section and the head of
# the compact history), so it is reported the same way.
_BEHAVIOR_REQUIRED_EVIDENCE = (_SECTION_FINAL_RESPONSE,)
_BEHAVIOR_EVIDENCE_GAP_HINT = (
    "Raise SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT and SKILL_EVAL_BEHAVIOR_CHECK_BUDGET, "
    "or shorten the case, then re-run"
)


def _behavior_evidence_completeness(
    traj: dict[str, Any],
    question: str,
    evidence: str,
) -> tuple[bool, list[str], list[str]]:
    """Report which behavior evidence sections reached the judge whole.

    Every section is compared against its complete source rather than against
    another budgeted rendering, so a cut the budget and the final-response cap
    both applied cannot hide behind either one. File changes and the tool
    history are rendered with ``full=True`` for the same reason: their own
    per-entry allowances clip write bodies and tool results before any budget
    applies, and comparing against those clipped renderings reported a section
    complete while thousands of characters of a written file had been dropped.
    Returns ``(required_complete, required_missing, incomplete_sections)``.
    """
    sources = {
        _SECTION_FINAL_RESPONSE: get_final_response(traj),
        _SECTION_USER_REQUEST: question,
        _SECTION_FILE_CHANGES: "\n\n".join(_collect_file_change_evidence(traj, full=True)),
        _SECTION_COMPACT_TOOL_HISTORY: build_conversation_summary(traj, question, full=True),
    }
    required_missing: list[str] = []
    incomplete: list[str] = []
    for section, label in _BEHAVIOR_SECTION_LABELS:
        source = str(sources.get(section) or "")
        if not source.strip() or _source_reaches_judge(source, evidence):
            continue
        incomplete.append(label)
        if section in _BEHAVIOR_REQUIRED_EVIDENCE:
            required_missing.append(label)
    return not required_missing, required_missing, incomplete


# Clause separators for an expectation. A dot only separates when whitespace
# follows it, so an artifact path such as ``report.py`` is never split apart.
_BEHAVIOR_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\band\b|\bthen\b|\balso\b|\bplus\b|\bas well as\b|\bwhile\b|\bbefore\b|\bafter\b|[;]|\.\s)",
    re.IGNORECASE,
)


def _expected_behaviors_verified(expected_behavior: Any, facts: list[dict[str, Any]]) -> bool:
    """Return True when deterministic facts already settle every expected behavior.

    An expectation is settled when every checkable token it names — an
    artifact path or a backtick-quoted snippet — appears in *facts* marked
    observed. Those facts are computed from the trajectory, not from the
    budgeted evidence, and they head the judge's prompt, so for such a case
    the judge's answer does not depend on the final response surviving the
    limits: withholding the score would discard a result the evidence
    supports.

    Naming an observed artifact is not enough on its own, because an
    expectation routinely asks for more than the artifact: "writes
    ``report.py`` **and explains why each column was dropped**" is settled
    only in its first half, and the half that needs the final response is
    exactly the half the deterministic facts cannot speak to. Every clause
    must therefore name a checkable token of its own. A clause that does not
    — plain prose such as "explains the trade-offs", whether it stands alone
    or is joined to an artifact clause — keeps the gate.

    The split is deliberately generous: an unrecognized clause shape leaves
    the expectation looking like prose and keeps the gate, which withholds a
    score. Lifting the gate wrongly keeps a score the evidence does not
    support, so the two errors are not symmetric.
    """
    if not isinstance(expected_behavior, list) or not expected_behavior:
        return False
    observed = {str(fact.get("claim")) for fact in facts if fact.get("observed")}
    for behavior in expected_behavior:
        text = str(behavior or "")
        claims = [claim for claim, _mode in _checkable_tokens(text)]
        if not claims or any(claim not in observed for claim in claims):
            return False
        for clause in _BEHAVIOR_CLAUSE_SPLIT_RE.split(text):
            # A clause with no letters or digits is punctuation or filler left
            # behind by the split, not something the agent was asked to do.
            if not any(char.isalnum() for char in clause):
                continue
            if not _checkable_tokens(clause):
                return False
    return True


def behavior_evidence_gap(omitted: Any) -> str:
    """Describe the required behavior evidence missing from a bundle, or ``""``.

    A behavior score read off evidence whose final answer was silently
    shortened grades the evidence loss, not the agent, so callers use this to
    withhold the score and raise an explicit evaluation error instead.

    Nothing is withheld when the bundle reports
    ``expected_behaviors_verified``: the deterministic facts at the head of
    the evidence already settle every expectation, so the missing text cannot
    change the outcome. The key is absent from records built before it
    existed and from records assembled by hand, and absence means "not
    known", which keeps the gate.
    """
    if not isinstance(omitted, dict) or omitted.get("required_complete", True):
        return ""
    if omitted.get("expected_behaviors_verified", False):
        return ""
    missing = [str(item).strip() for item in omitted.get("required_missing") or [] if str(item).strip()]
    names = ", ".join(missing) or "required evidence"
    reason = str(omitted.get("reason") or "").strip()
    detail = f" ({reason})" if reason else ""
    return (
        f"behavior_check evidence is incomplete: {names} did not reach the judge{detail}. {_BEHAVIOR_EVIDENCE_GAP_HINT}"
    )


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
    and records what it dropped (never silent). Deterministic verified facts are
    prepended at the top when any checkable tokens are found.
    """
    if not isinstance(expected_behavior, list):
        expected_behavior = []
    refs = build_metric_evidence_refs(traj, question, ground_truth=ground_truth, expected_behavior=expected_behavior)

    # Compute verified facts once; prepend the same section to all metrics
    facts = build_verified_facts(traj, expected_behavior, ground_truth)
    facts_section = _build_verified_facts_section(facts)

    final = get_final_response(traj)
    file_changes = "\n\n".join(_collect_file_change_evidence(traj))
    late_obs = _late_observation_excerpts(traj, _BUNDLE_ITEM_CHARS)

    def _prepend_facts(text: str) -> str:
        if not facts_section:
            return text
        if text:
            return f"{facts_section}\n\n{text}"
        return facts_section

    bundles: dict[str, dict[str, Any]] = {}
    budgets = _bundle_budgets()

    acc_text, acc_drop, acc_trunc = _assemble(
        [
            (_SECTION_FINAL_RESPONSE, final),
            ("PRODUCED FILES / WRITES", file_changes),
            ("KEY OBSERVATIONS", "\n---\n".join(late_obs[:_BUNDLE_ACCURACY_MAX_OBS])),
        ],
        budgets["accuracy"],
    )
    bundles["accuracy"] = {
        "prompt_evidence": _prepend_facts(acc_text or _clip(get_agent_text(traj), budgets["accuracy"])),
        "evidence_refs": refs["accuracy"],
        "omitted": {
            "count": acc_drop,
            "truncated": acc_trunc,
            "reason": "low-relevance sections dropped to fit budget" if acc_trunc else "",
        },
        "verified": facts,
    }

    goal_text, goal_drop, goal_trunc = _assemble(
        [
            (_SECTION_FINAL_RESPONSE, final),
            ("END-STATE FILE CHANGES", file_changes),
            ("RECENT TOOL RESULTS (newest first)", "\n---\n".join(late_obs[:_BUNDLE_GOAL_MAX_OBS])),
        ],
        budgets["goal_accuracy"],
    )
    bundles["goal_accuracy"] = {
        "prompt_evidence": _prepend_facts(goal_text or _clip(get_agent_text(traj), budgets["goal_accuracy"])),
        "evidence_refs": refs["goal_accuracy"],
        "omitted": {
            "count": goal_drop,
            "truncated": goal_trunc,
            "reason": "older/low-relevance tool results dropped to fit budget" if goal_trunc else "",
        },
        "verified": facts,
    }

    facts_overhead = len(facts_section) + 2 if facts_section else 0
    bc_budget = max(1, budgets["behavior_check"] - facts_overhead)
    bc_text = build_behavior_evidence(traj, question, max_chars=bc_budget)
    # Compare against a rendering with neither limit applied. The budgeted
    # rendering also caps the final-response section at its own limit
    # (independent of max_chars), so comparing two budget renderings shares
    # that cap and misses content the cap dropped from the middle of a long
    # final answer: the bundle then reported truncated=False for evidence
    # that was not complete.
    bc_full = build_behavior_evidence(traj, question, max_chars=10**9, final_response_limit=10**9)
    bc_causes: list[str] = []
    if bc_full != bc_text:
        # An unlimited-budget rendering with the usual final-response cap
        # differs from the reference rendering only in that section, so it
        # separates the two causes. The budget cut evidence when the budgeted
        # rendering differs from it. The cap cut evidence when it changed that
        # section and the complete final answer does not still reach the
        # judge, whole in the compact history or split between the history
        # and the end of that section (see _final_answer_reaches_judge). Name
        # every cause that applied so users know which settings to raise.
        bc_capped = build_behavior_evidence(traj, question, max_chars=10**9) if final else bc_full
        if bc_text != bc_capped:
            bc_causes.append("behavior evidence shortened to fit SKILL_EVAL_BEHAVIOR_CHECK_BUDGET")
        if bc_capped != bc_full and not _final_answer_reaches_judge(final, bc_capped):
            bc_causes.append("final response shortened to fit SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT")
    bc_trunc = bool(bc_causes)
    bc_reason = "; ".join(bc_causes)
    bc_evidence = _prepend_facts(bc_text)
    # ``truncated`` says only that something was dropped, which on a long run
    # is ordinary. Completeness says whether the evidence the judge must read
    # survived, so the judge can refuse to score an evidence-loss artifact.
    bc_required_ok, bc_required_missing, bc_incomplete = _behavior_evidence_completeness(traj, question, bc_evidence)
    # Whether the gap matters is a separate question from whether there is
    # one. When the deterministic facts above already settle every expected
    # behavior, the judge's answer does not rest on the text the limits cut.
    bc_behaviors_verified = _expected_behaviors_verified(expected_behavior, facts)
    bundles["behavior_check"] = {
        "prompt_evidence": bc_evidence,
        "evidence_refs": refs["behavior_check"],
        "omitted": {
            "count": 1 if bc_trunc else 0,
            "truncated": bc_trunc,
            "reason": bc_reason,
            "required_complete": bc_required_ok,
            "required_missing": bc_required_missing,
            "incomplete_sections": bc_incomplete,
            "expected_behaviors_verified": bc_behaviors_verified,
        },
        "verified": facts,
    }
    return bundles


def extract_tool_calls_as_dicts(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert ATIF trajectory to the ``{"action", "action_input", "observation"}``
    format consumed by ``eval_core.checks``.
    """
    result: list[dict[str, Any]] = []
    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue
        for _, tc in iter_tool_calls({"steps": [step]}):
            call = {
                "action": tc.get("function_name", ""),
                "action_input": tc.get("arguments") or {},
                "observation": _tool_call_observation(step, tc),
            }
            # Keep Harbor's explicit success flag too: Claude Code results that
            # exited 0 carry no exit status line.
            if (is_error := normalized_tool_call_error_flag(step, tc)) is not None:
                call["is_error"] = is_error
            if status := tc.get("_atif_normalization_status"):
                call["normalization_status"] = status
            if status := tc.get("_atif_observation_status"):
                call["observation_status"] = status
            if wrapper_observation := _tool_call_wrapper_observation(step, tc):
                call["wrapper_observation"] = wrapper_observation
            result.append(call)
    return result
