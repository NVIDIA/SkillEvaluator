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
from typing import Any

from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line


def iter_tool_calls(traj: dict[str, Any]):
    """Yield ``(step_dict, tool_call_dict)`` for every tool call in the trajectory."""
    for step in traj.get("steps", []):
        for tc in step.get("tool_calls") or []:
            yield step, tc


def get_all_tool_calls(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all tool calls with function name, arguments, and observation text.

    Returns ``[{"fn": str, "args": dict, "args_text": str, "obs": str}]``.
    """
    calls: list[dict[str, Any]] = []
    for step, tc in iter_tool_calls(traj):
        fn = tc.get("function_name") or ""
        args = tc.get("arguments") or {}
        obs_text = ""
        obs = step.get("observation") or {}
        for r in obs.get("results") or []:
            if r.get("source_call_id") == tc.get("tool_call_id") or not r.get("source_call_id"):
                obs_text += str(r.get("content", ""))
        calls.append(
            {
                "fn": fn,
                "args": args,
                "args_text": json.dumps(args).lower(),
                "obs": obs_text.lower(),
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


def build_conversation_summary(traj: dict[str, Any], question: str) -> str:
    """Build a human-readable conversation summary for LLM judges.

    The summary is optimized for behavior-check and goal-accuracy prompts.
    """
    parts = [f"User: {question}"]

    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue

        reasoning = step.get("reasoning_content") or ""
        if reasoning:
            parts.append(f"Agent reasoning: {str(reasoning)[:200]}")

        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name", "")
            args = tc.get("arguments") or {}
            parts.append(f"Agent called: {fn}({json.dumps(args)[:200]})")

        obs = step.get("observation") or {}
        for r in obs.get("results") or []:
            content = str(r.get("content", ""))
            if content:
                parts.append(f"Tool returned: {content[:400]}")

        msg = step.get("message") or ""
        if msg and isinstance(msg, str) and msg.strip() and not step.get("tool_calls"):
            parts.append(f"Agent: {msg[:1500]}")

    final = get_final_response(traj)
    if final and not any(final[:50] in p for p in parts):
        parts.append(f"Agent final answer: {final[:1500]}")

    return "\n".join(parts)


_BEHAVIOR_EVIDENCE_MAX_CHARS = 4000
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
    if limit <= len(marker):
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


def _collect_file_change_evidence(traj: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue
        observations_by_id: dict[str, str] = {}
        for result in (step.get("observation") or {}).get("results") or []:
            call_id = str(result.get("source_call_id") or "")
            content = str(result.get("content") or "")
            if call_id and content:
                observations_by_id[call_id] = content

        for tc in step.get("tool_calls") or []:
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

            obs = observations_by_id.get(str(tc.get("tool_call_id") or ""), "")
            entry_parts = [f"Agent called: {fn}"]
            if file_path:
                entry_parts.append(f"Path: {file_path}")
            if body:
                entry_parts.append(_truncate_for_behavior(body, 1800))
            if obs:
                entry_parts.append(f"Tool returned: {_truncate_for_behavior(obs, 500)}")
            changes.append("\n".join(entry_parts))
    return changes


def build_behavior_evidence(
    traj: dict[str, Any],
    question: str,
    max_chars: int = _BEHAVIOR_EVIDENCE_MAX_CHARS,
) -> str:
    """Build compact, behavior-check-specific evidence from an ATIF trajectory.

    Behavior checks often ask whether the agent produced or changed artifacts.
    Put final output and write/edit evidence before exploratory reads so a
    fixed-size judge prompt does not miss late file creation.
    """
    parts: list[str] = []
    remaining = max_chars

    file_changes = "\n\n".join(_collect_file_change_evidence(traj))
    if file_changes:
        remaining = _append_section_with_budget(parts, "FILE CHANGES", file_changes, remaining)

    final = get_final_response(traj)
    if final:
        remaining = _append_section_with_budget(
            parts,
            "FINAL RESPONSE",
            final,
            remaining,
            section_limit=800,
        )

    remaining = _append_section_with_budget(
        parts,
        "USER REQUEST",
        question,
        remaining,
        section_limit=800,
    )

    history = build_conversation_summary(traj, question)
    remaining = _append_section_with_budget(parts, "COMPACT TOOL HISTORY", history, remaining)

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
    return ref


def _dedupe_evidence_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ref in refs:
        key = (
            str(ref.get("source") or ""),
            str(ref.get("json_pointer") or ""),
            str(ref.get("kind") or ""),
            str(ref.get("path") or ""),
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


def _tool_call_ref(step_idx: int, tool_idx: int, tc: dict[str, Any], *, kind: str) -> dict[str, Any]:
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
    return _evidence_ref(
        source="trajectory.json",
        json_pointer=f"/steps/{step_idx}/tool_calls/{tool_idx}",
        kind=kind,
        label=f"{fn}: {label_detail}" if label_detail else fn,
        path=path or None,
        excerpt=excerpt,
    )


def _tool_call_refs(traj: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for step_idx, step in enumerate(traj.get("steps", [])):
        if step.get("source") != "agent":
            continue
        for tool_idx, tc in enumerate(step.get("tool_calls") or []):
            if len(refs) >= _METRIC_EVIDENCE_MAX_TOOL_REFS:
                return refs
            refs.append(_tool_call_ref(step_idx, tool_idx, tc, kind="tool_call"))
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
    for step_idx, step in enumerate(traj.get("steps", [])):
        if step.get("source") != "agent":
            continue
        for tool_idx, tc in enumerate(step.get("tool_calls") or []):
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
            refs.append(_tool_call_ref(step_idx, tool_idx, tc, kind="file_change"))
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
_BUNDLE_BUDGETS = {"accuracy": 8000, "goal_accuracy": 12000, "behavior_check": 8000}
_BUNDLE_ACCURACY_MAX_OBS = 6  # newest observations fed to accuracy (<= ~6x1500 <= budget)
_BUNDLE_GOAL_MAX_OBS = 12  # newest observations fed to goal_accuracy (end-state)


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
            content = str(result.get("content") or "").strip()
            if content:
                out.append(_clip(content, limit))
    return out


def _assemble(sections: list[tuple[str, str]], budget: int) -> tuple[str, int, bool]:
    """Join (title, body) sections under a char budget. Returns (text, dropped, truncated)."""
    parts: list[str] = []
    used = 0
    dropped = 0
    truncated = False
    for title, body in sections:
        body = str(body or "").strip()
        if not body:
            continue
        block = f"{title}\n{body}"
        if used + len(block) <= budget:
            parts.append(block)
            used += len(block) + 2
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
            for tc in step.get("tool_calls") or []:
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

    acc_text, acc_drop, acc_trunc = _assemble(
        [
            ("FINAL RESPONSE", final),
            ("PRODUCED FILES / WRITES", file_changes),
            ("KEY OBSERVATIONS", "\n---\n".join(late_obs[:_BUNDLE_ACCURACY_MAX_OBS])),
        ],
        _BUNDLE_BUDGETS["accuracy"],
    )
    bundles["accuracy"] = {
        "prompt_evidence": _prepend_facts(acc_text or _clip(get_agent_text(traj), _BUNDLE_BUDGETS["accuracy"])),
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
            ("FINAL RESPONSE", final),
            ("END-STATE FILE CHANGES", file_changes),
            ("RECENT TOOL RESULTS (newest first)", "\n---\n".join(late_obs[:_BUNDLE_GOAL_MAX_OBS])),
        ],
        _BUNDLE_BUDGETS["goal_accuracy"],
    )
    bundles["goal_accuracy"] = {
        "prompt_evidence": _prepend_facts(goal_text or _clip(get_agent_text(traj), _BUNDLE_BUDGETS["goal_accuracy"])),
        "evidence_refs": refs["goal_accuracy"],
        "omitted": {
            "count": goal_drop,
            "truncated": goal_trunc,
            "reason": "older/low-relevance tool results dropped to fit budget" if goal_trunc else "",
        },
        "verified": facts,
    }

    bc_text = build_behavior_evidence(traj, question, max_chars=_BUNDLE_BUDGETS["behavior_check"])
    bc_full = build_behavior_evidence(traj, question, max_chars=10**9)
    bc_trunc = len(bc_full) > len(bc_text)
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
    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue
        for tc in step.get("tool_calls") or []:
            obs_text = ""
            obs = step.get("observation") or {}
            for r in obs.get("results") or []:
                if r.get("source_call_id") == tc.get("tool_call_id") or not r.get("source_call_id"):
                    obs_text += str(r.get("content", ""))
            result.append(
                {
                    "action": tc.get("function_name", ""),
                    "action_input": tc.get("arguments") or {},
                    "observation": obs_text,
                }
            )
    return result
