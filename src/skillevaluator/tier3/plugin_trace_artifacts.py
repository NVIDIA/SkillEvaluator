# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded harness evidence readers for the private runtime observer."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.collector import (
    _materialized_trial_trajectory,
    _read_bounded_text,
    _safe_trial_path_component,
)
from skillevaluator.utils.secure_fs import stat_is_link_or_reparse


def _plugin_signal_trajectory(trial_root: Path) -> dict[str, Any] | None:
    """Reuse the collector's bounded, topology-authorized continuation/child graph."""
    trajectory, _reason = _materialized_trial_trajectory(trial_root, None)
    return trajectory


_CODEX_LOG_MAX_BYTES = 16 * 1024 * 1024
_CODEX_SESSION_FILES = 8
_CODEX_SESSION_WALK_ENTRIES = 4_096
_CODEX_MCP_CALLS = 4_000
_CODEX_SERVER_NAME_RE = re.compile(r"[^\x00-\x1f\x7f-\x9f]{1,128}")


def _codex_server(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    server = value.removeprefix("mcp__").removesuffix("__")
    return server if _CODEX_SERVER_NAME_RE.fullmatch(server) else None


def _codex_json_lines(path: Path) -> list[Any]:
    """Parsed JSON lines that mention MCP (bounded, no-follow read; unparsable lines are skipped)."""
    text = _read_bounded_text(path, max_bytes=_CODEX_LOG_MAX_BYTES)
    if not text or "mcp" not in text.casefold():
        return []
    events: list[Any] = []
    for line in text.splitlines():
        if "mcp" not in line.casefold():
            continue
        try:
            events.append(json.loads(line))
        except (ValueError, RecursionError):
            continue
    return events


def _is_real_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return not stat_is_link_or_reparse(metadata) and stat.S_ISDIR(metadata.st_mode)


def _codex_session_files(sessions: Path) -> list[Path]:
    """Return the first Codex session logs (``*.jsonl``) under ``<trial>/agent/sessions``.

    The tree is agent-writable, so the walk never follows a link or reparse
    point and lists at most ``_CODEX_SESSION_WALK_ENTRIES`` entries. It visits
    names in sorted order, a directory's files before its subdirectories. A
    directory with more entries than the budget has left ends the walk, so the
    logs found never depend on the order the filesystem lists names in.
    """
    if not (_is_real_directory(sessions.parent) and _is_real_directory(sessions)):
        return []
    found: list[Path] = []
    budget = _CODEX_SESSION_WALK_ENTRIES
    pending = [sessions]
    while pending and budget > 0 and len(found) < _CODEX_SESSION_FILES:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                # One entry past the budget shows that the directory does not fit in it.
                entries = list(islice(iterator, budget + 1))
        except OSError:
            continue
        if len(entries) > budget:
            break
        budget -= len(entries)
        subdirectories: list[Path] = []
        for entry in sorted(entries, key=lambda entry: entry.name):
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat_is_link_or_reparse(metadata):
                continue
            if stat.S_ISDIR(metadata.st_mode):
                subdirectories.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode) and entry.name.endswith(".jsonl"):
                found.append(Path(entry.path))
        pending.extend(reversed(subdirectories))
    return found[:_CODEX_SESSION_FILES]


_CODEX_ARGS_KEY_CHARS = 4_096


def _codex_args_key(value: Any) -> str | None:
    """Canonical JSON text of a Codex call's arguments (a JSON string or an object), bounded."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return value if len(value) <= _CODEX_ARGS_KEY_CHARS else None
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return None
    return text if len(text) <= _CODEX_ARGS_KEY_CHARS else None


def _codex_item_status(item: Mapping[str, Any]) -> str | None:
    """``failed`` or ``completed`` for a finished ``mcp_tool_call`` item, else ``None``."""
    error = item.get("error")
    result = item.get("result")
    flagged = isinstance(result, Mapping) and any(result.get(key) is True for key in ("isError", "is_error"))
    if error not in (None, "", {}, []) or flagged:
        return "failed"
    status = item.get("status")
    if isinstance(status, str) and status.strip().casefold() in {"failed", "completed"}:
        return status.strip().casefold()
    return None


def _codex_txt_items(agent_dir: Path) -> list[dict[str, Any]]:
    """``mcp_tool_call`` items of ``codex.txt`` in start order: server, tool, arguments key, status.

    ``codex.txt`` item ids (``item_N``) are not tool call ids, so callers pair
    an item with a call by server, tool, and arguments.
    """
    items: dict[str, dict[str, Any]] = {}
    for event in _codex_json_lines(agent_dir / "codex.txt"):
        event_type = event.get("type") if isinstance(event, Mapping) else None
        if not isinstance(event_type, str) or event_type not in {"item.started", "item.completed"}:
            continue
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "mcp_tool_call":
            continue
        server, tool, item_id = _codex_server(item.get("server")), item.get("tool"), item.get("id")
        if not server or not isinstance(tool, str) or not tool or not isinstance(item_id, str):
            continue
        entry = items.get(item_id)
        if entry is None:
            if len(items) >= _CODEX_MCP_CALLS:
                continue
            entry = items[item_id] = {
                "server": server,
                "tool": tool,
                "args": _codex_args_key(item.get("arguments")),
                "status": None,
            }
        if event.get("type") == "item.completed":
            entry["status"] = _codex_item_status(item)
            entry["text"] = _codex_item_text(item)
    return list(items.values())


_CODEX_TEXT_CHARS = 512


def _codex_item_text(item: Mapping[str, Any]) -> str:
    """The start of a finished item's first text block (the tool's own answer), or ``""``."""
    result = item.get("result")
    content = result.get("content") if isinstance(result, Mapping) else None
    for block in content[:8] if isinstance(content, list) else ():
        if isinstance(block, Mapping) and isinstance(block.get("text"), str) and block["text"].strip():
            return block["text"][:_CODEX_TEXT_CHARS]
    return ""


def _codex_call_outputs(trajectory: Mapping[str, Any] | None) -> dict[str, str]:
    """Tool call id -> the start of its result text in the trajectory (bounded)."""
    outputs: dict[str, str] = {}
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    for step in steps[:_CODEX_MCP_CALLS] if isinstance(steps, list) else ():
        observation = step.get("observation") if isinstance(step, Mapping) else None
        results = observation.get("results") if isinstance(observation, Mapping) else None
        for result in results[:64] if isinstance(results, list) else ():
            if not isinstance(result, Mapping) or not isinstance(result.get("source_call_id"), str):
                continue
            content = result.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    block.get("text", "") if isinstance(block, Mapping) else str(block)
                    for block in content[:8]
                    if isinstance(block, Mapping | str)
                )
            if isinstance(content, str):
                outputs.setdefault(result["source_call_id"], content[: _CODEX_TEXT_CHARS * 16])
    return outputs


def _codex_server_key(server: str) -> str:
    # The session log spells a server the way Codex sanitizes it (``deepwiki_bad``);
    # codex.txt keeps the configured name (``deepwiki-bad``).
    return re.sub(r"[^a-z0-9]", "_", server.casefold())


def _text_in_output(text: str, output: str) -> bool:
    # The trajectory keeps the tool text raw, or JSON-escaped inside Codex's content wrapper.
    return bool(text) and (text in output or json.dumps(text, ensure_ascii=False)[1:-1] in output)


def _take_codex_item(
    items: list[dict[str, Any]], *, tool: str, args: str | None, server: str | None, output: str = ""
) -> Any:
    """Pop the unpaired item with this tool and arguments (and server, when known).

    An answer can disambiguate candidates. Conflicting identities/outcomes
    remain unknown rather than inheriting the first log item's success.
    """
    key = _codex_server_key(server) if server is not None else None
    fits = [
        index
        for index, item in enumerate(items)
        if item["tool"] == tool and item["args"] == args and (key is None or _codex_server_key(item["server"]) == key)
    ]
    if not fits or args is None:
        return None
    matching_output = [index for index in fits if _text_in_output(items[index].get("text") or "", output)]
    candidates = matching_output or fits
    outcomes = {(_codex_server_key(items[index]["server"]), items[index].get("status")) for index in candidates}
    if len(outcomes) != 1:
        return None
    return items.pop(candidates[0])


def _codex_mcp_calls(trial_root: Path, trajectory: Mapping[str, Any] | None) -> tuple[dict[str, str], dict[str, str]]:
    """Codex MCP calls read from the raw Codex logs: ``(call id -> server, call id -> status)``.

    Harbor's Codex trajectory keeps only the bare MCP tool name (for example
    ``lookup``) and no error flag. Codex's own session log records the server
    (``namespace: "mcp__<server>"`` on the call, and an ``McpToolCall`` item
    with the call id); the ``codex.txt`` JSON stream records each call's
    outcome (``status: "failed"`` or ``"completed"``) but under its own item
    id. So each session-log call is paired with the first unpaired
    ``codex.txt`` item of the same server, tool, and arguments. Without a
    readable session log, the trajectory's calls are paired with ``codex.txt``
    items the same way (tool and arguments), never by name and order alone; a
    tool name served by one server only still maps without arguments. When
    several items fit, the one whose answer is in the call's trajectory result
    wins. Empty for other harnesses.
    """
    servers: dict[str, str] = {}
    session_calls: list[tuple[str, str, str, str | None]] = []
    agent_dir = trial_root / "agent"
    for path in _codex_session_files(agent_dir / "sessions"):
        for event in _codex_json_lines(path):
            payload = event.get("payload") if isinstance(event, Mapping) else None
            if not isinstance(payload, Mapping):
                continue
            item = payload.get("item")
            tool: Any = None
            args: str | None = None
            item_type = item.get("type") if isinstance(item, Mapping) else None
            payload_type = payload.get("type")
            if isinstance(item_type, str) and item_type in {"McpToolCall", "mcp_tool_call"}:
                call_id, server = item.get("id"), _codex_server(item.get("server"))
                tool, args = item.get("tool"), _codex_args_key(item.get("arguments"))
            elif isinstance(payload_type, str) and payload_type in {"function_call", "custom_tool_call"}:
                namespace = payload.get("namespace")
                call_id = payload.get("call_id")
                server = _codex_server(namespace) if isinstance(namespace, str) and namespace[:5] == "mcp__" else None
                tool = payload.get("name")
                args = _codex_args_key(payload.get("arguments", payload.get("input")))
            else:
                continue
            if isinstance(call_id, str) and call_id and server and len(servers) < _CODEX_MCP_CALLS:
                if call_id not in servers and isinstance(tool, str) and tool:
                    session_calls.append((call_id, server, tool, args))
                servers.setdefault(call_id, server)
    items = _codex_txt_items(agent_dir)
    outputs = _codex_call_outputs(trajectory) if items else {}
    statuses: dict[str, str] = {}
    if servers:
        for call_id, server, tool, args in session_calls:
            item = _take_codex_item(items, tool=tool, args=args, server=server, output=outputs.get(call_id, ""))
            if item is not None and item["status"]:
                statuses[call_id] = item["status"]
            elif item is None and any(
                entry["tool"] == tool and _codex_server_key(entry["server"]) == _codex_server_key(server)
                for entry in items
            ):
                statuses[call_id] = "unknown"
        return servers, statuses
    if not items or not isinstance(trajectory, Mapping):
        return servers, statuses
    servers_by_tool: dict[str, set[str]] = {}
    for item in items:
        servers_by_tool.setdefault(item["tool"], set()).add(item["server"])
    for step in (trajectory.get("steps") or ())[:_CODEX_MCP_CALLS]:
        for call in (step.get("tool_calls") if isinstance(step, Mapping) else None) or ():
            if not isinstance(call, Mapping):
                continue
            name, call_id = call.get("function_name"), call.get("tool_call_id")
            if not isinstance(name, str) or not isinstance(call_id, str) or not call_id or call_id in servers:
                continue
            item = _take_codex_item(
                items,
                tool=name,
                args=_codex_args_key(call.get("arguments")),
                server=None,
                output=outputs.get(call_id, ""),
            )
            if item is not None:
                servers[call_id] = item["server"]
                if item["status"]:
                    statuses[call_id] = item["status"]
            elif len(servers_by_tool.get(name, ())) == 1:
                servers[call_id] = next(iter(servers_by_tool[name]))
                statuses[call_id] = "unknown"
            if len(servers) >= _CODEX_MCP_CALLS:
                return servers, statuses
    return servers, statuses


_OPENCODE_LOG_MAX_BYTES = 16 * 1024 * 1024
_OPENCODE_FAILED_CALLS = 4_000
_OPENCODE_ERROR_CHARS = 2_048


def _opencode_call_errors(trial_root: Path) -> dict[str, str]:
    """OpenCode tool calls that failed, read from the raw ``opencode.txt``: ``call id -> error text``.

    OpenCode's JSON stream ends each tool part with ``state.status``:
    ``completed`` with an ``output``, or ``error`` with an ``error``. Harbor's
    OpenCode trajectory keeps a call's result only when it has an output, so a
    failed call (a ``task`` to an agent that does not exist, an MCP call whose
    connection closed) is left with no result and no outcome. The part's
    ``callID`` is the trajectory's tool call id. Empty for other harnesses.
    """
    from skillevaluator.tier3.eval_core.log_converters import _opencode_error_text

    text = _read_bounded_text(trial_root / "agent" / "opencode.txt", max_bytes=_OPENCODE_LOG_MAX_BYTES)
    errors: dict[str, str] = {}
    for line in text.splitlines() if text and "tool_use" in text else ():
        if "tool_use" not in line or "error" not in line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        part = event.get("part") if isinstance(event, Mapping) and event.get("type") == "tool_use" else None
        state = part.get("state") if isinstance(part, Mapping) else None
        if not isinstance(state, Mapping) or state.get("status") != "error":
            continue
        call_id = part.get("callID", part.get("id"))
        if isinstance(call_id, str) and call_id and call_id not in errors:
            errors[call_id] = _opencode_error_text(dict(state))[:_OPENCODE_ERROR_CHARS]
            if len(errors) >= _OPENCODE_FAILED_CALLS:
                break
    return errors


def _call_status(result: Any, statuses: Mapping[str, str]) -> str | None:
    call_id = result.get("source_call_id") if isinstance(result, Mapping) else None
    return statuses.get(call_id) if isinstance(call_id, str) else None


def _step_results(step: Any) -> list[Any]:
    observation = step.get("observation") if isinstance(step, Mapping) else None
    results = observation.get("results") if isinstance(observation, Mapping) else None
    return results if isinstance(results, list) else []


def _with_harness_statuses(
    trajectory: dict[str, Any] | None, statuses: Mapping[str, str], errors: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """A copy of ``trajectory`` whose results carry the harness's own per-call status.

    The status goes to ``extra.harness_status`` of every result whose
    ``source_call_id`` names the call, where plugin signals read it as the
    call's structured outcome. A call in ``errors`` (call id -> error text)
    that no result names gets one in its step: the error text, with an
    ``error`` status. The trajectory read from disk is not changed.
    """
    from skillevaluator.tier3.eval_core.plugin_signals import HARNESS_STATUS_KEY

    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if not (statuses or errors) or not isinstance(steps, list):
        return trajectory
    errors = errors or {}
    # Only a call that no result names had its result dropped.
    answered = {
        result["source_call_id"]
        for step in steps
        for result in _step_results(step)
        if errors and isinstance(result, Mapping) and isinstance(result.get("source_call_id"), str)
    }
    updated: list[Any] = []
    for step in steps:
        results = _step_results(step)
        calls = step.get("tool_calls") if errors and isinstance(step, Mapping) else None
        dropped = {
            call_id: None
            for call in (calls if isinstance(calls, list) else ())
            if isinstance(call, Mapping)
            and isinstance(call_id := call.get("tool_call_id"), str)
            and call_id in errors
            and call_id not in answered
        }
        if not dropped and not any(_call_status(result, statuses) for result in results):
            updated.append(step)
            continue
        marked = []
        for result in results:
            status = _call_status(result, statuses)
            if status is None:
                marked.append(result)
                continue
            extra = result.get("extra") if isinstance(result.get("extra"), Mapping) else {}
            marked.append({**result, "extra": {**extra, HARNESS_STATUS_KEY: status}})
        # The harness dropped these results (OpenCode keeps none for a failed call).
        marked.extend(
            {"source_call_id": call_id, "content": errors[call_id], "extra": {HARNESS_STATUS_KEY: "error"}}
            for call_id in dropped
        )
        answered.update(dropped)
        observation = step.get("observation") if isinstance(step.get("observation"), Mapping) else {}
        updated.append({**step, "observation": {**observation, "results": marked}})
    return {**trajectory, "steps": updated}


@dataclass(frozen=True)
class TrialTrace:
    trajectory: dict[str, Any] | None
    call_servers: Mapping[str, str]


def _harness_sources(trial_root: Path, trajectory: dict[str, Any] | None) -> list[tuple[Path, dict[str, Any] | None]]:
    """Partition a materialized native multi-step trajectory by its authorized sources.

    A physical steps layout makes the materializer require authoritative
    ``step_results`` and reject a root trajectory. Only after that succeeds
    are its generated step names trusted as sibling harness-log locations.
    Single-step trajectories cannot select log paths through authored extras.
    """
    if trajectory is None or not _is_real_directory(trial_root / "steps"):
        return [(trial_root, trajectory)]
    extra = trajectory.get("extra")
    multi = extra.get("harbor_multi_step") if isinstance(extra, Mapping) else None
    names = multi.get("step_names") if isinstance(multi, Mapping) else None
    if not isinstance(names, list) or any(_safe_trial_path_component(name) != name for name in names):
        return [(trial_root, None)]
    groups: dict[str, list[Any]] = {name: [] for name in names}
    for step in trajectory.get("steps", []):
        extra = step.get("extra") if isinstance(step, Mapping) else None
        name = extra.get("harbor_step_name") if isinstance(extra, Mapping) else None
        if name not in groups:
            return [(trial_root, None)]
        groups[name].append(step)
    return [(trial_root / "steps" / name, {**trajectory, "steps": groups[name]}) for name in names]


def _scope_harness_call_ids(trajectory: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Keep independent step sessions' reused call IDs distinct in the merged document."""
    updated = []
    for step in trajectory["steps"]:
        if not isinstance(step, Mapping):
            updated.append(step)
            continue
        scoped = dict(step)
        calls = step.get("tool_calls")
        if isinstance(calls, list):
            scoped["tool_calls"] = [
                {**call, "tool_call_id": prefix + call["tool_call_id"]}
                if isinstance(call, Mapping) and isinstance(call.get("tool_call_id"), str)
                else call
                for call in calls
            ]
        observation = step.get("observation")
        results = observation.get("results") if isinstance(observation, Mapping) else None
        if isinstance(results, list):
            scoped["observation"] = {
                **observation,
                "results": [
                    {**result, "source_call_id": prefix + result["source_call_id"]}
                    if isinstance(result, Mapping) and isinstance(result.get("source_call_id"), str)
                    else result
                    for result in results
                ],
            }
        updated.append(scoped)
    return {**trajectory, "steps": updated}


class TrialTraceArtifacts:
    """Read one whole-trial trajectory and harness logs once per logical root."""

    def __init__(self) -> None:
        self._traces: dict[Path, TrialTrace] = {}

    def trace(self, trial_root: Path) -> TrialTrace:
        if trial_root not in self._traces:
            trajectory = _plugin_signal_trajectory(trial_root)
            sources = _harness_sources(trial_root, trajectory)
            servers, steps = {}, []
            for index, (source_root, source) in enumerate(sources):
                if source is None:
                    trajectory = None
                    break
                try:
                    local_servers, statuses = _codex_mcp_calls(source_root, source)
                    errors = _opencode_call_errors(source_root)
                    statuses.update(dict.fromkeys(errors, "error"))
                    marked = _with_harness_statuses(source, statuses, errors)
                except Exception:
                    # Advisory harness logs cannot discard an authorized trajectory.
                    local_servers, marked = {}, source
                if source_root != trial_root:
                    prefix = f"harbor-step:{index}:"
                    marked = _scope_harness_call_ids(marked, prefix)
                    local_servers = {prefix + call_id: server for call_id, server in local_servers.items()}
                servers.update(local_servers)
                steps.extend(marked["steps"])
            if trajectory is not None:
                trajectory = {**trajectory, "steps": steps}
            self._traces[trial_root] = TrialTrace(trajectory, servers)
        return self._traces[trial_root]
