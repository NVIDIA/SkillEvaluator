# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split one Harbor ATIF trajectory's time into LLM seconds and tool seconds.

Two sources are used, never mixed within one trajectory:

* ``atif``: native timing extras. ``step.extra.llm_duration_ms`` gives LLM
  time only when every agent step carries it. Tool time is the union of each
  call's ``tool_calls[].extra`` ``started_at``/``ended_at`` interval, with
  ``duration_ms`` for calls without an interval, and only when every call is
  timed. When any native extra is present, nothing is inferred.
* ``step_timestamps``: inferred from the ISO ``timestamp`` on every step plus
  the ``Wall time: N seconds`` line that Codex writes in the header of the
  output of its shell tools (``exec_command``, ``write_stdin``, ``wait``,
  ``shell_command`` ...) and ``apply_patch``, or the ``duration_seconds`` that
  Harbor keeps in the structured tool metadata of those calls. Each gap
  between consecutive steps is counted once. Tool time is the wall time of the
  step's calls (parallel calls count their longest); the rest of the gap,
  which also includes harness and queue time, is LLM time. A trajectory with
  any other tool call (``update_plan``, web search, MCP tools ...), or a call
  whose output has no such header, has no measured duration for that call, so
  nothing is inferred.

Each step of a merged multi-step trajectory is timed on its own, so the gap
between Harbor steps (verifier and setup time) is never counted. Missing,
malformed, unordered or inconsistent evidence makes a component ``None``; it
is never reported as zero. A trajectory with no tool calls measures zero tool
seconds.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

TIMING_SOURCES = ("atif", "step_timestamps")

# Codex built-in tools whose output starts with Codex's own status header. The
# name may carry a namespace prefix such as ``functions.exec_command``.
_CODEX_TIMED_TOOL_NAMES = frozenset(
    {"exec_command", "write_stdin", "wait", "wait_command", "shell", "shell_command", "apply_patch"}
)
_WALL_TIME_RE = re.compile(
    r"^Wall time:[ \t]*([0-9]+(?:\.[0-9]+)?)[ \t]*seconds[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)
# Everything after the first ``Output:`` line is the command's own output.
_OUTPUT_LINE_RE = re.compile(r"^Output:", re.MULTILINE)
_MAX_TOOL_CALL_SECONDS = 86400.0
_NATIVE_TOOL_KEYS = ("started_at", "ended_at", "duration_ms")


def trajectory_timing(trajectory: Any) -> dict[str, Any] | None:
    """Return ``{"llm_seconds", "tool_seconds", "source"}`` or ``None``."""
    if not isinstance(trajectory, dict):
        return None
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps or not all(isinstance(step, dict) for step in steps):
        return None
    if _has_native_timing(steps):
        measured = _native_timing_ms(steps)
        source = "atif"
    else:
        measured = _inferred_timing_ms(steps)
        source = "step_timestamps"
    if measured is None:
        return None
    llm_ms, tool_ms = measured
    llm_seconds = _seconds(llm_ms)
    tool_seconds = _seconds(tool_ms)
    if llm_seconds is None and tool_seconds is None:
        return None
    return {"llm_seconds": llm_seconds, "tool_seconds": tool_seconds, "source": source}


def _seconds(milliseconds: float | None) -> float | None:
    if milliseconds is None or not math.isfinite(milliseconds) or milliseconds < 0:
        return None
    return round(milliseconds / 1000.0, 3)


def _tool_calls(step: dict[str, Any]) -> list[Any] | None:
    """Return the step's tool calls (``[]`` when it has none), or ``None`` if malformed."""
    calls = step.get("tool_calls")
    if calls is None:
        return []
    return calls if isinstance(calls, list) else None


def _has_native_timing(steps: list[dict[str, Any]]) -> bool:
    for step in steps:
        extra = step.get("extra")
        if isinstance(extra, dict) and "llm_duration_ms" in extra:
            return True
        for call in _tool_calls(step) or []:
            call_extra = call.get("extra") if isinstance(call, dict) else None
            if isinstance(call_extra, dict) and any(key in call_extra for key in _NATIVE_TOOL_KEYS):
                return True
    return False


def _non_negative_ms(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _timestamp_ms(value: Any) -> float | None:
    """Parse an ISO timestamp with a UTC offset; naive local times are rejected."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            return None
        milliseconds = parsed.timestamp() * 1000.0
    except (ValueError, OverflowError, OSError):
        return None
    return milliseconds if math.isfinite(milliseconds) else None


def _interval_ms(extra: Any) -> tuple[float, float] | None:
    if not isinstance(extra, dict):
        return None
    start = _timestamp_ms(extra.get("started_at"))
    end = _timestamp_ms(extra.get("ended_at"))
    if start is None or end is None or end < start:
        return None
    return start, end


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock length of the union of ``[start, end]`` intervals."""
    total = 0.0
    current: tuple[float, float] | None = None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = (start, end)
        else:
            current = (current[0], max(current[1], end))
    if current is not None:
        total += current[1] - current[0]
    return total


def _native_timing_ms(steps: list[dict[str, Any]]) -> tuple[float | None, float | None] | None:
    agent_steps = [step for step in steps if step.get("source") == "agent"]
    llm_values = [
        _non_negative_ms(extra.get("llm_duration_ms")) if isinstance((extra := step.get("extra")), dict) else None
        for step in agent_steps
    ]
    llm_ms = (
        math.fsum(value for value in llm_values if value is not None)
        if agent_steps and all(value is not None for value in llm_values)
        else None
    )

    calls: list[Any] = []
    for step in steps:
        step_calls = _tool_calls(step)
        if step_calls is None:
            return None
        calls.extend(step_calls)
    intervals: list[tuple[float, float]] = []
    durations: list[float] = []
    tool_ms: float | None = 0.0 if llm_ms is not None else None
    for call in calls:
        extra = call.get("extra") if isinstance(call, dict) else None
        interval = _interval_ms(extra)
        if interval is not None:
            intervals.append(interval)
            continue
        duration = _non_negative_ms(extra.get("duration_ms")) if isinstance(extra, dict) else None
        if duration is None:
            tool_ms = None
            break
        durations.append(duration)
    else:
        if calls:
            tool_ms = _union_ms(intervals) + math.fsum(durations)
    if llm_ms is None and tool_ms is None:
        return None
    return llm_ms, tool_ms


def _wall_time_seconds(value: float) -> float | None:
    return value if math.isfinite(value) and 0 <= value <= _MAX_TOOL_CALL_SECONDS else None


def _header_wall_time_seconds(content: Any) -> float | None:
    """Read ``Wall time`` from the status header, never from the command output."""
    if not isinstance(content, str):
        return None
    output_line = _OUTPUT_LINE_RE.search(content)
    if output_line is None:
        return None
    matches = _WALL_TIME_RE.findall(content[: output_line.start()])
    if len(matches) != 1:
        return None
    return _wall_time_seconds(float(matches[0]))


def _metadata_duration_seconds(step: dict[str, Any], call_id: Any, *, single_call: bool) -> float | None:
    """Read ``duration_seconds`` from the structured tool metadata Harbor keeps for Codex."""
    extra = step.get("extra")
    if not isinstance(extra, dict):
        return None
    metadata: Any = None
    details = extra.get("tool_call_details")
    if isinstance(call_id, str) and call_id and isinstance(details, dict):
        detail = details.get(call_id)
        metadata = detail.get("metadata") if isinstance(detail, dict) else None
    if metadata is None and single_call:
        metadata = extra.get("tool_metadata")
    if not isinstance(metadata, dict):
        return None
    value = _non_negative_ms(metadata.get("duration_seconds"))
    return _wall_time_seconds(value) if value is not None else None


def _tool_call_seconds(step: dict[str, Any]) -> list[float] | None:
    """Measured wall time for each tool call on the step, or ``None`` if any call is unmeasured."""
    calls = _tool_calls(step)
    if calls is None:
        return None
    if not calls:
        return []
    observation = step.get("observation")
    results = observation.get("results") if isinstance(observation, dict) else None
    if not isinstance(results, list):
        results = []

    by_call_id: dict[str, float] = {}
    unidentified: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        seconds = _header_wall_time_seconds(result.get("content"))
        if seconds is None:
            continue
        call_id = result.get("source_call_id")
        if isinstance(call_id, str) and call_id.strip():
            if call_id in by_call_id:
                return None
            by_call_id[call_id] = seconds
        else:
            unidentified.append(seconds)

    durations: list[float] = []
    seen_ids: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            return None
        name = call.get("function_name")
        if not isinstance(name, str) or name.rsplit(".", 1)[-1] not in _CODEX_TIMED_TOOL_NAMES:
            return None
        call_id = call.get("tool_call_id") or call.get("id")
        if isinstance(call_id, str) and call_id:
            if call_id in seen_ids:
                return None
            seen_ids.add(call_id)
            if call_id in by_call_id:
                durations.append(by_call_id[call_id])
                continue
        seconds = _metadata_duration_seconds(step, call_id, single_call=len(calls) == 1)
        if seconds is None and len(calls) == 1 and len(results) == 1 and len(unidentified) == 1:
            seconds = unidentified[0]
        if seconds is None:
            return None
        durations.append(seconds)
    return durations


def _segment_timing_ms(steps: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Attribute each timestamp gap of one Harbor step's trajectory once."""
    stamped: list[tuple[dict[str, Any], float]] = []
    for step in steps:
        timestamp = _timestamp_ms(step.get("timestamp"))
        if timestamp is None or (stamped and timestamp < stamped[-1][1]):
            return None
        stamped.append((step, timestamp))
    if len(stamped) < 2:
        return None

    llm_ms = 0.0
    tool_ms = 0.0
    saw_agent = False
    previous_had_tools = False
    for index, (step, timestamp) in enumerate(stamped):
        if step.get("source") != "agent":
            previous_had_tools = False
            continue
        saw_agent = True
        durations = _tool_call_seconds(step)
        if durations is None:
            return None
        # A preceding tool step already accounted for the gap that ends here.
        if index and not previous_had_tools:
            llm_ms += timestamp - stamped[index - 1][1]
        has_tools = bool(durations)
        if has_tools:
            if index + 1 == len(stamped):
                return None
            gap_ms = stamped[index + 1][1] - timestamp
            step_tool_ms = max(durations) * 1000.0
            if step_tool_ms > gap_ms:
                return None
            llm_ms += gap_ms - step_tool_ms
            tool_ms += step_tool_ms
        previous_had_tools = has_tools
    if not saw_agent:
        return None
    return llm_ms, tool_ms


def _harbor_step_segments(steps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a merged multi-step trajectory at each change of ``extra.harbor_step_name``."""
    segments: list[list[dict[str, Any]]] = []
    current: Any = None
    for step in steps:
        extra = step.get("extra")
        name = extra.get("harbor_step_name") if isinstance(extra, dict) else None
        if not segments or name != current:
            segments.append([])
            current = name
        segments[-1].append(step)
    return segments


def _inferred_timing_ms(steps: list[dict[str, Any]]) -> tuple[float, float] | None:
    llm_ms = 0.0
    tool_ms = 0.0
    for segment in _harbor_step_segments(steps):
        measured = _segment_timing_ms(segment)
        if measured is None:
            return None
        llm_ms += measured[0]
        tool_ms += measured[1]
    return llm_ms, tool_ms
