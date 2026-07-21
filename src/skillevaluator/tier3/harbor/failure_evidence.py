# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured failure evidence from harness-native terminal agent events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from skillevaluator.telemetry import redact_sensitive_text
from skillevaluator.tier3.harbor.secret_redaction import redact_secrets_in_log_line

_AGENT_BY_LOG_NAME = {
    "claude-code.txt": "claude-code",
    "codex.txt": "codex",
    "opencode.txt": "opencode",
}
_AGENT_LOG_NAMES = tuple(_AGENT_BY_LOG_NAME)
# Streaming per-line bound for JSONL agent logs. Lines above this are skipped
# in bounded chunks without ever materializing the full line (a 200MB line was
# measured to skip with +4.3MB RSS), so parsing up to 10MiB stays memory-safe.
# Residual gap: a genuine terminal event on a single line larger than this
# bound is invisible to this parser; Harbor's result.json typed-exception
# backstop remains the authoritative catch for agent process failure.
_MAX_EVENT_LINE_BYTES = 10 * 1024 * 1024
_MAX_EVENT_LOG_BYTES = 64 * 1024 * 1024
_MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024
_MAX_PRESEMANTIC_TRANSCRIPT_BYTES = 256 * 1024
_MAX_PRESEMANTIC_LINES = 64
_MAX_PRESEMANTIC_EVENTS = 16
_MAX_MESSAGE_LENGTH = 2048
_MESSAGE_FIELDS = ("message", "detail", "reason", "name", "code", "status", "statusCode")
_NESTED_MESSAGE_FIELDS = ("data", "cause")
_ZERO_TOKEN_TERMINAL_API_ERROR_RE = re.compile(r"^API Error:\s*\d{3}\b")
_CODEX_DIAGNOSTIC_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\S+Z\s+(?:TRACE|DEBUG|INFO|WARN|ERROR)\s+\S+")


@dataclass(frozen=True)
class FailureEvidence:
    """A bounded, typed reason that an evaluation component failed."""

    component: Literal["agent", "verifier", "environment"]
    phase: Literal["preflight", "execute", "grade", "fallback"]
    kind: str
    source: str
    event_type: str
    message: str
    retryable: bool | None = None


def _message_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int | float):
        return str(value)
    return ""


def _message_parts(value: Any, *, depth: int = 0) -> list[str]:
    scalar = _message_scalar(value)
    if scalar:
        return [scalar]
    if not isinstance(value, dict) or depth > 1:
        return []

    parts: list[str] = []
    for field in _MESSAGE_FIELDS:
        part = _message_scalar(value.get(field))
        if part and part not in parts:
            parts.append(part)
    for field in _NESTED_MESSAGE_FIELDS:
        for part in _message_parts(value.get(field), depth=depth + 1):
            if part not in parts:
                parts.append(part)
    return parts[:4]


def _terminal_message(agent: str, event: dict[str, Any], event_type: str) -> str:
    if agent == "claude-code":
        payloads = (event.get("result"), event.get("error"), event.get("message"))
    else:
        payloads = (event.get("error"), event.get("message"))

    for payload in payloads:
        parts = _message_parts(payload)
        if parts:
            return redact_sensitive_text(
                redact_secrets_in_log_line("; ".join(parts)),
                max_len=_MAX_MESSAGE_LENGTH,
            )
    return redact_sensitive_text(
        redact_secrets_in_log_line(f"{agent} reported terminal event {event_type}"),
        max_len=_MAX_MESSAGE_LENGTH,
    )


def failure_from_agent_event(
    agent: str,
    event: dict[str, Any],
    source: str,
) -> FailureEvidence | None:
    """Return evidence only for an agent harness's accepted terminal failure schema."""
    normalized_agent = agent.strip().lower()
    event_type = event.get("type")

    if normalized_agent == "claude-code":
        if event_type != "result" or event.get("is_error") is not True:
            return None
    elif normalized_agent == "codex":
        if event_type != "turn.failed":
            return None
    elif normalized_agent == "opencode":
        if event_type != "error" or not isinstance(event.get("error"), dict):
            return None
    else:
        return None

    return FailureEvidence(
        component="agent",
        phase="execute",
        kind="agent_runtime",
        source=source,
        event_type=event_type,
        message=_terminal_message(normalized_agent, event, event_type),
    )


def _bounded_jsonl_lines(path: Path):
    """Yield decoded lines within per-line and aggregate byte limits."""
    with path.open("rb") as stream:
        total_bytes = 0
        while raw_line := stream.readline(_MAX_EVENT_LINE_BYTES + 1):
            total_bytes += len(raw_line)
            if total_bytes > _MAX_EVENT_LOG_BYTES:
                return
            if len(raw_line) > _MAX_EVENT_LINE_BYTES:
                while raw_line and not raw_line.endswith(b"\n"):
                    raw_line = stream.readline(_MAX_EVENT_LINE_BYTES + 1)
                    total_bytes += len(raw_line)
                    if total_bytes > _MAX_EVENT_LOG_BYTES:
                        return
                continue
            # A UTF-8 BOM (U+FEFF) survives str.strip(); drop it so a terminal
            # event on the first line of a BOM-prefixed log still parses.
            yield raw_line.decode("utf-8", errors="replace").strip().lstrip("\ufeff")


def codex_presemantic_failure_marker_from_text(text: str | None) -> dict[str, str] | None:
    """Classify only Codex's closed pre-semantic terminal lifecycle.

    This parser is deliberately independent of error prose. It accepts the
    observed machine stream only when a thread and turn start and the turn
    fails before any semantic item or unknown event. Optional top-level errors
    and typed error items are lifecycle diagnostics. Bounded pre-thread text
    and timestamped Codex tracing lines are ignored; schema drift and ambiguity
    otherwise fail closed.
    """

    if not isinstance(text, str):
        return None
    try:
        encoded = text.encode("utf-8")
    except UnicodeError:
        return None
    if len(encoded) > _MAX_PRESEMANTIC_TRANSCRIPT_BYTES:
        return None
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("\ufeff").strip()
        if line:
            lines.append(line)
    if not lines or len(lines) > _MAX_PRESEMANTIC_LINES:
        return None

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value

    state = "thread"
    event_count = 0
    for line in lines:
        if not line:
            return None
        try:
            event = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        except (ValueError, RecursionError):
            # Codex emits bounded launcher diagnostics before its first JSON
            # event even with --json. Nothing before thread.started can be
            # semantic agent activity; after structured output starts, any
            # non-JSON line must have Codex's machine tracing shape.
            if state == "thread" and line[0] not in "[{":
                continue
            if line[0] not in "[{" and _CODEX_DIAGNOSTIC_LINE_RE.match(line):
                continue
            return None
        event_count += 1
        if event_count > _MAX_PRESEMANTIC_EVENTS:
            return None
        if not isinstance(event, dict):
            return None
        event_type = event.get("type")
        if state == "thread":
            if event_type != "thread.started" or not isinstance(event.get("thread_id"), str) or not event["thread_id"]:
                return None
            state = "turn"
        elif state == "turn":
            if event_type == "turn.started":
                state = "terminal"
            elif event_type == "error":
                continue
            elif event_type in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item")
                if not isinstance(item, dict) or item.get("type") != "error":
                    return None
            else:
                return None
        elif state == "terminal":
            if event_type == "error":
                continue
            if event_type in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "error":
                    continue
                return None
            if event_type != "turn.failed" or not isinstance(event.get("error"), dict):
                return None
            state = "done"
        else:
            return None

    if state != "done":
        return None
    return {
        "stage": "agent_adapter_bootstrap",
        "reason_code": "adapter_model_protocol_negotiation_failed",
    }


def first_agent_log_failure(path: Path, agent: str | None = None) -> FailureEvidence | None:
    """Return the first accepted terminal failure event in a JSONL agent log."""
    resolved_agent = agent or _AGENT_BY_LOG_NAME.get(path.name.lower())
    if not resolved_agent:
        return None

    try:
        lines = _bounded_jsonl_lines(path)
        for line in lines:
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, RecursionError):
                # JSONDecodeError subclasses ValueError, but json.loads also
                # raises plain ValueError (e.g. integer literals over the
                # 4300-digit int conversion limit); an agent-writable line
                # must never crash the scan.
                continue
            if not isinstance(event, dict):
                continue
            evidence = failure_from_agent_event(resolved_agent, event, str(path))
            if evidence is not None:
                return evidence
    except OSError:
        return None
    return None


def _terminal_trajectory_failure(path: Path) -> FailureEvidence | None:
    """Return a narrow ATIF backstop for adapters without native event schemas.

    This intentionally inspects only the terminal agent step and exact zero-token
    metrics. It never scans nested tool output or earlier prose, preserving the
    false-positive boundary that motivated structured failure evidence.
    """
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_TRAJECTORY_BYTES + 1)
        if len(raw) > _MAX_TRAJECTORY_BYTES:
            return None
        trajectory = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(trajectory, dict):
        return None
    metrics = trajectory.get("final_metrics")
    if not isinstance(metrics, dict):
        return None
    for key in ("total_prompt_tokens", "total_completion_tokens"):
        value = metrics.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or value != 0:
            return None
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict):
        return None
    terminal = steps[-1]
    message = terminal.get("message")
    if terminal.get("source") != "agent" or not isinstance(message, str):
        return None
    message = message.strip()
    if not _ZERO_TOKEN_TERMINAL_API_ERROR_RE.match(message):
        return None
    return FailureEvidence(
        component="agent",
        phase="execute",
        kind="agent_runtime",
        source=str(path),
        event_type="trajectory.zero_token_api_error",
        message=redact_sensitive_text(
            redact_secrets_in_log_line(message),
            max_len=_MAX_MESSAGE_LENGTH,
        ),
    )


def first_trial_agent_failure(trial_dir: Path) -> FailureEvidence | None:
    """Search root, nested, and multi-step agent logs in deterministic harness order.

    Harbor multi-step tasks keep per-step agent logs under
    ``steps/<step>/(agent/)``; those transcripts must drive classification the
    same way single-step root logs do.
    """
    parents = [trial_dir, trial_dir / "agent"]
    steps_dir = trial_dir / "steps"
    if steps_dir.is_dir():
        try:
            step_dirs = sorted(path for path in steps_dir.iterdir() if path.is_dir())
        except OSError:
            step_dirs = []
        for step_dir in step_dirs:
            parents.extend((step_dir, step_dir / "agent"))
    for log_name in _AGENT_LOG_NAMES:
        for parent in parents:
            evidence = first_agent_log_failure(parent / log_name)
            if evidence is not None:
                return evidence
    for parent in parents:
        evidence = _terminal_trajectory_failure(parent / "trajectory.json")
        if evidence is not None:
            return evidence
    return None
