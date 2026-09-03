# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reconstruct minimal ATIF-shaped dicts from agent logs when trajectory.json is missing.

Harbor's ``cursor-cli`` integration currently tees stdout to ``cursor-cli.txt`` but does not
write ``trajectory.json``.  Claude Code stream output is also tee'd to ``claude-code.txt`` as
JSONL; when trajectory conversion fails upstream, we can parse that stream as a fallback.

Output uses the same ``{"steps": [...]}`` shape consumed by ``extract_tool_calls_as_dicts`` /
``get_agent_text`` in the Harbor verifier.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _stringify_tool_result_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict):
                parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _attach_tool_result(steps: list[dict[str, Any]], tool_use_id: str, content_text: str) -> None:
    for step in reversed(steps):
        if step.get("source") != "agent":
            continue
        for tc in step.get("tool_calls") or []:
            if tc.get("tool_call_id") == tool_use_id:
                step.setdefault("observation", {"results": []})
                step["observation"].setdefault("results", [])
                step["observation"]["results"].append(
                    {
                        "source_call_id": tool_use_id,
                        "content": content_text,
                    }
                )
                return


def synthetic_trajectory_from_claude_stream_jsonl(text: str) -> dict[str, Any] | None:
    """Parse Claude Code ``--output-format stream-json`` lines (tee'd to claude-code.txt)."""
    steps: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        et = evt.get("type")
        if et == "assistant":
            msg = evt.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                steps.append(
                    {
                        "source": "agent",
                        "message": content.strip(),
                        "tool_calls": [],
                        "observation": {"results": []},
                    }
                )
                continue
            if not isinstance(content, list):
                continue

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif bt == "thinking":
                    t = block.get("thinking") or block.get("text")
                    if t:
                        text_parts.append(str(t))
                elif bt == "tool_use":
                    tc_id = str(block.get("id") or block.get("tool_use_id") or "")
                    name = str(block.get("name") or "")
                    inp = block.get("input")
                    if isinstance(inp, dict):
                        arguments = dict(inp)
                    elif inp is None:
                        arguments = {}
                    else:
                        arguments = {"raw": inp}
                    if name.lower() in ("read", "read_file") and "file_path" in arguments and "path" not in arguments:
                        arguments["path"] = arguments["file_path"]
                    tool_calls.append(
                        {
                            "tool_call_id": tc_id,
                            "function_name": name,
                            "arguments": arguments,
                        }
                    )

            message = "\n".join(t for t in text_parts if t).strip()
            if message or tool_calls:
                steps.append(
                    {
                        "source": "agent",
                        "message": message,
                        "tool_calls": tool_calls,
                        "observation": {"results": []},
                    }
                )

        elif et == "user":
            msg = evt.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tid = str(block.get("tool_use_id") or "")
                res_text = _stringify_tool_result_content(block.get("content"))
                if tid:
                    _attach_tool_result(steps, tid, res_text)

        elif et == "result":
            res = evt.get("result")
            if not res or not str(res).strip():
                continue
            res_s = str(res).strip()
            if steps and steps[-1].get("source") == "agent":
                last_msg = (steps[-1].get("message") or "").strip()
                if res_s in last_msg or (last_msg and last_msg in res_s):
                    continue
            steps.append(
                {
                    "source": "agent",
                    "message": res_s,
                    "tool_calls": [],
                    "observation": {"results": []},
                }
            )

    if not steps:
        return None
    return {
        "steps": steps,
        "schema_version": "ATIF-v1.2-synthetic-claude-log",
        "final_metrics": {},
    }


def synthetic_trajectory_from_cursor_cli(text: str) -> dict[str, Any] | None:
    """Build a minimal trajectory from ``cursor-agent`` plain-text output (heuristic).

    Cursor CLI does not emit ATIF.  We preserve the full transcript for LLM-based metrics and
    infer likely shell/read actions with regex so deterministic skill_execution checks can
    still fire when the log format matches.
    """
    if not text or not text.strip():
        return None

    tool_calls: list[dict[str, Any]] = []
    n = 0

    shell_patterns = (
        re.compile(r"(?:^|\n)\s*(?:\$|›|>)\s*(cat\s+[^\n]+)", re.IGNORECASE | re.MULTILINE),  # noqa: RUF001 -- matches shell-prompt glyph in agent logs
        re.compile(r"\b(cat\s+[^\n;`]*SKILL\.md[^\n;`]*)", re.IGNORECASE),
        re.compile(r"\b(python3?\s+[^\n;`]{3,})", re.IGNORECASE),
        re.compile(r"\b((?:sudo\s+)?(?:bash|sh)\s+[^\n;`]{3,})", re.IGNORECASE),
    )
    for pat in shell_patterns:
        for m in pat.finditer(text):
            cmd = m.group(1).strip()
            if len(cmd) < 3:
                continue
            n += 1
            tool_calls.append(
                {
                    "tool_call_id": f"cursor-fallback-{n}",
                    "function_name": "bash",
                    "arguments": {"command": cmd},
                }
            )

    for pat in (
        r"\bread_file\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\breadFile\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\*\*Path:\*\*\s*`([^`]+)`",
        r"`([^`]*/skills/[^`]*/SKILL\.md)`",
    ):
        for m in re.finditer(pat, text, re.IGNORECASE):
            path = (m.group(1) if m.lastindex else m.group(0)).strip()
            if len(path) < 2:
                continue
            n += 1
            tool_calls.append(
                {
                    "tool_call_id": f"cursor-fallback-{n}",
                    "function_name": "read",
                    "arguments": {"path": path},
                }
            )

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for tc in tool_calls:
        key = (tc["function_name"], json.dumps(tc["arguments"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tc)

    msg = text.strip()
    if len(msg) > 12000:
        msg = msg[-12000:]

    return {
        "steps": [
            {
                "source": "agent",
                "message": msg,
                "tool_calls": deduped,
                "observation": {"results": []},
            }
        ],
        "schema_version": "ATIF-v1.2-synthetic-cursor-log",
        "final_metrics": {},
    }


def synthetic_trajectory_from_cline_cli(text: str) -> dict[str, Any] | None:
    """Build a trajectory from Cline CLI JSONL output (``cline.txt``).

    Cline CLI emits structured JSONL events with ``type``/``say``/``ask``
    fields.  We extract tool uses (``useSkill``, ``command``, ``read_file``,
    ``write_to_file``), command outputs, text responses, and the final
    ``completion_result`` into ATIF steps.
    """
    if not text or not text.strip():
        return None

    steps: list[dict[str, Any]] = []
    n = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        say = evt.get("say", "")
        ask = evt.get("ask", "")
        evt_text = evt.get("text", "")

        if say == "tool" and evt_text:
            try:
                tool_data = json.loads(evt_text)
            except (json.JSONDecodeError, TypeError):
                tool_data = {}

            tool_name = tool_data.get("tool", "")
            if tool_name == "useSkill":
                path = tool_data.get("path", "")
                n += 1
                steps.append(
                    {
                        "source": "agent",
                        "message": "",
                        "tool_calls": [
                            {
                                "tool_call_id": f"cline-{n}",
                                "function_name": "read",
                                "arguments": {"path": f"/workspace/skills/{path}/SKILL.md"},
                            }
                        ],
                        "observation": {"results": []},
                    }
                )
            elif tool_name in ("read_file", "readFile"):
                path = tool_data.get("path", tool_data.get("content", ""))
                n += 1
                steps.append(
                    {
                        "source": "agent",
                        "message": "",
                        "tool_calls": [
                            {
                                "tool_call_id": f"cline-{n}",
                                "function_name": "read",
                                "arguments": {"path": path},
                            }
                        ],
                        "observation": {"results": []},
                    }
                )

        elif say == "command" and evt_text:
            cmd = evt_text.strip()
            if len(cmd) >= 3:
                n += 1
                steps.append(
                    {
                        "source": "agent",
                        "message": "",
                        "tool_calls": [
                            {
                                "tool_call_id": f"cline-{n}",
                                "function_name": "bash",
                                "arguments": {"command": cmd},
                            }
                        ],
                        "observation": {"results": []},
                    }
                )

        elif ask == "command_output" and evt_text:
            if steps and steps[-1].get("tool_calls"):
                last_tc_id = steps[-1]["tool_calls"][-1].get("tool_call_id", "")
                steps[-1].setdefault("observation", {"results": []})
                steps[-1]["observation"].setdefault("results", [])
                steps[-1]["observation"]["results"].append(
                    {
                        "source_call_id": last_tc_id,
                        "content": evt_text[:8000],
                    }
                )

        elif say == "text" and evt_text:
            msg = evt_text.strip()
            if msg and (not steps or steps[-1].get("tool_calls")):
                steps.append(
                    {
                        "source": "agent",
                        "message": msg,
                        "tool_calls": [],
                        "observation": {"results": []},
                    }
                )
            elif msg and steps:
                prev = steps[-1].get("message", "")
                if msg not in prev:
                    steps[-1]["message"] = (prev + "\n" + msg).strip()

        elif say == "completion_result" and evt_text:
            steps.append(
                {
                    "source": "agent",
                    "message": evt_text.strip(),
                    "tool_calls": [],
                    "observation": {"results": []},
                }
            )

    if not steps:
        return None
    return {
        "steps": steps,
        "schema_version": "ATIF-v1.2-synthetic-cline-log",
        "final_metrics": {},
    }


def _normalize_opencode_tool(tool_name: str, raw_input: Any) -> tuple[str, dict[str, Any]]:
    """Map OpenCode tool names and inputs to ATIF tool-call fields."""
    if not isinstance(raw_input, dict):
        return tool_name, {}
    name = str(tool_name or "").strip()
    lowered = name.lower()
    arguments = dict(raw_input)
    if lowered in {"read", "read_file"}:
        function_name = "read"
        if "filePath" in arguments and "path" not in arguments:
            arguments["path"] = arguments["filePath"]
        if "file_path" in arguments and "path" not in arguments:
            arguments["path"] = arguments["file_path"]
    elif lowered in {"bash", "shell"}:
        function_name = "bash"
    elif lowered in {"write", "edit"}:
        function_name = lowered
        if "filePath" in arguments and "path" not in arguments:
            arguments["path"] = arguments["filePath"]
    else:
        function_name = name or lowered or "tool"
    return function_name, arguments


def _opencode_output_text(state: dict[str, Any]) -> str:
    output = state.get("output")
    if output is not None:
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            message = output.get("message") or output.get("text")
            if message is not None:
                return str(message)
            return json.dumps(output, ensure_ascii=False)
        return str(output)

    error = state.get("error")
    if error is None:
        return ""
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        data = error.get("data")
        if isinstance(data, dict):
            message = data.get("message")
            if message is not None:
                return str(message)
        message = error.get("message")
        if message is not None:
            return str(message)
        return json.dumps(error, ensure_ascii=False)
    return str(error)


def synthetic_trajectory_from_opencode_json(text: str) -> dict[str, Any] | None:
    """Parse OpenCode ``run --format=json`` JSONL (tee'd to ``opencode.txt``)."""
    if not text or not text.strip():
        return None

    steps: list[dict[str, Any]] = []
    saw_content = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(evt, dict):
            continue

        et = str(evt.get("type") or "")
        if et == "text":
            part = evt.get("part")
            if not isinstance(part, dict):
                continue
            msg = str(part.get("text") or "").strip()
            if not msg:
                continue
            saw_content = True
            if steps and not steps[-1].get("tool_calls"):
                prev = (steps[-1].get("message") or "").strip()
                steps[-1]["message"] = (prev + "\n" + msg).strip() if prev else msg
            else:
                steps.append(
                    {
                        "source": "agent",
                        "message": msg,
                        "tool_calls": [],
                        "observation": {"results": []},
                    }
                )
            continue

        if et != "tool_use":
            continue

        part = evt.get("part")
        if not isinstance(part, dict):
            continue
        state = part.get("state")
        if not isinstance(state, dict):
            continue
        status = str(state.get("status") or "").lower()
        if status in {"pending", "running"}:
            continue

        tool_name = str(part.get("tool") or part.get("name") or "")
        call_id = str(part.get("callID") or part.get("call_id") or part.get("id") or "")
        function_name, arguments = _normalize_opencode_tool(tool_name, state.get("input"))
        if not call_id:
            call_id = f"opencode-{len(steps) + 1}"

        saw_content = True
        step = {
            "source": "agent",
            "message": "",
            "tool_calls": [
                {
                    "tool_call_id": call_id,
                    "function_name": function_name,
                    "arguments": arguments,
                }
            ],
            "observation": {"results": []},
        }
        output_text = _opencode_output_text(state)
        if output_text:
            step["observation"]["results"].append(
                {
                    "source_call_id": call_id,
                    "content": output_text[:8000],
                }
            )
        steps.append(step)

    if not saw_content or not steps:
        return None
    return {
        "steps": steps,
        "schema_version": "ATIF-v1.2-synthetic-opencode-log",
        "final_metrics": {},
    }


def _iter_jsonl_dicts(text: str) -> list[dict[str, Any]]:
    """Parse JSONL lines, skipping non-JSON noise (e.g. stderr mixed into tee logs)."""
    events: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(evt, dict):
            events.append(evt)
    return events


def _codex_item_text(item: dict[str, Any]) -> str | None:
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    item_type = str(item.get("type") or "")
    if item_type not in {"message", "agent_message"}:
        return None
    content = item.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    joined = "\n".join(parts).strip()
    return joined or None


def _codex_function_arguments(function_call: dict[str, Any]) -> dict[str, Any]:
    args = function_call.get("arguments")
    if isinstance(args, dict):
        return dict(args)
    return {}


def _codex_thread_item(evt: dict[str, Any]) -> dict[str, Any] | None:
    """Return the Codex thread item payload from a completed event."""
    if str(evt.get("type") or "") == "item.completed":
        item = evt.get("item")
        return item if isinstance(item, dict) else None
    if str(evt.get("type") or "") == "item" and evt.get("item.completed", True):
        item = evt.get("item")
        return item if isinstance(item, dict) else None
    return None


def _codex_mcp_observation(item: dict[str, Any]) -> str:
    result = item.get("result")
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    parts.append(str(text) if text is not None else json.dumps(block, ensure_ascii=False))
                else:
                    parts.append(str(block))
            joined = "\n".join(part for part in parts if part).strip()
            if joined:
                return joined[:8000]
        structured = result.get("structured_content")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False)[:8000]
        return json.dumps(result, ensure_ascii=False)[:8000]

    error = item.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message is not None:
            return str(message)[:8000]
        return json.dumps(error, ensure_ascii=False)[:8000]
    return ""


def _codex_file_change_step(item: dict[str, Any], evt: dict[str, Any], tool_index: int) -> dict[str, Any] | None:
    changes = item.get("changes")
    if not isinstance(changes, list):
        return None
    tool_calls: list[dict[str, Any]] = []
    base_id = str(item.get("id") or evt.get("item_id") or f"codex-{tool_index + 1}")
    for idx, change in enumerate(changes):
        if not isinstance(change, dict):
            continue
        path = str(change.get("path") or "").strip()
        if not path:
            continue
        tool_calls.append(
            {
                "tool_call_id": f"{base_id}-{idx}",
                "function_name": "write",
                "arguments": {"path": path, "kind": str(change.get("kind") or "update")},
            }
        )
    if not tool_calls:
        return None
    return {
        "source": "agent",
        "message": "",
        "tool_calls": tool_calls,
        "observation": {"results": []},
    }


def synthetic_trajectory_from_codex_json(text: str) -> dict[str, Any] | None:
    """Parse Codex ``exec --json`` ThreadEvent JSONL."""
    if not text or not text.strip():
        return None

    steps: list[dict[str, Any]] = []
    saw_content = False
    tool_index = 0

    for evt in _iter_jsonl_dicts(text):
        item = _codex_thread_item(evt)
        if item is None:
            continue

        item_type = str(item.get("type") or "")

        if item_type in {"agent_message", "message"}:
            message = _codex_item_text(item)
            function_call = item.get("function_call")
            if not message and not isinstance(function_call, dict):
                continue
            saw_content = True
            step: dict[str, Any] = {
                "source": "agent",
                "message": message or "",
                "tool_calls": [],
                "observation": {"results": []},
            }
            if isinstance(function_call, dict):
                call_id = str(item.get("id") or evt.get("item_id") or f"codex-{tool_index + 1}")
                function_name = str(function_call.get("name") or "tool")
                if function_name.lower() in {"bash", "shell"}:
                    function_name = "bash"
                arguments = _codex_function_arguments(function_call)
                step["tool_calls"] = [
                    {
                        "tool_call_id": call_id,
                        "function_name": function_name,
                        "arguments": arguments,
                    }
                ]
                output = item.get("output")
                if output is not None:
                    step["observation"]["results"].append(
                        {
                            "source_call_id": call_id,
                            "content": str(output)[:8000],
                        }
                    )
                tool_index += 1
            steps.append(step)
            continue

        if item_type == "command_execution":
            command = str(item.get("command") or "").strip()
            if not command:
                continue
            saw_content = True
            call_id = str(item.get("id") or evt.get("item_id") or f"codex-{tool_index + 1}")
            tool_index += 1
            step = {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": call_id,
                        "function_name": "bash",
                        "arguments": {"command": command},
                    }
                ],
                "observation": {"results": []},
            }
            output = item.get("aggregated_output")
            if output is not None:
                step["observation"]["results"].append(
                    {
                        "source_call_id": call_id,
                        "content": str(output)[:8000],
                    }
                )
            steps.append(step)
            continue

        if item_type == "file_change":
            step = _codex_file_change_step(item, evt, tool_index)
            if step is None:
                continue
            saw_content = True
            tool_index += 1
            steps.append(step)
            continue

        if item_type == "mcp_tool_call":
            tool = str(item.get("tool") or "mcp_tool")
            arguments = item.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            call_id = str(item.get("id") or evt.get("item_id") or f"codex-{tool_index + 1}")
            saw_content = True
            tool_index += 1
            step = {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": call_id,
                        "function_name": tool,
                        "arguments": dict(arguments),
                    }
                ],
                "observation": {"results": []},
            }
            observation = _codex_mcp_observation(item)
            if observation:
                step["observation"]["results"].append(
                    {
                        "source_call_id": call_id,
                        "content": observation,
                    }
                )
            steps.append(step)

    if not saw_content or not steps:
        return None
    return {
        "steps": steps,
        "schema_version": "ATIF-v1.2-synthetic-codex-log",
        "final_metrics": {},
    }


def synthetic_trajectory_from_codex_txt(text: str) -> dict[str, Any] | None:
    """Reconstruct ATIF from Codex tee logs (``exec --json`` JSONL or OpenCode-shaped JSONL)."""
    if not text or not text.strip():
        return None

    synth = synthetic_trajectory_from_codex_json(text)
    if synth and synth.get("steps"):
        return synth

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if all(line.startswith("{") and line.endswith("}") for line in lines):
        synth = synthetic_trajectory_from_opencode_json(text)
        if synth and synth.get("steps"):
            synth["schema_version"] = "ATIF-v1.2-synthetic-codex-log"
            return synth
    return None


def load_trajectory_with_fallback(
    trajectory_path: Path,
    logs_dir: Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load ``trajectory.json`` if present and non-empty; else reconstruct from agent logs.

    Prefers ``claude-code.txt`` (structured JSONL) over ``cursor-cli.txt`` (heuristic text)
    when both exist, because the former yields higher-fidelity tool pairing.
    """
    logs = logs_dir or trajectory_path.parent
    meta: dict[str, Any] = {"source": None, "warning": None, "note": None}

    if trajectory_path.exists():
        try:
            data = json.loads(trajectory_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                steps = data.get("steps")
                if isinstance(steps, list) and len(steps) > 0:
                    meta["source"] = "trajectory.json"
                    return data, meta
            meta["warning"] = "trajectory.json has no steps[]"
        except (json.JSONDecodeError, OSError) as e:
            meta["warning"] = f"trajectory.json unreadable: {e}"

    claude_path = logs / "claude-code.txt"
    if claude_path.exists():
        raw = claude_path.read_text(encoding="utf-8", errors="replace")
        synth = synthetic_trajectory_from_claude_stream_jsonl(raw)
        if synth and synth.get("steps"):
            meta["source"] = "claude-code.txt"
            meta["note"] = "Synthetic ATIF from Claude Code stream JSONL"
            return synth, meta

    cursor_path = logs / "cursor-cli.txt"
    if cursor_path.exists():
        raw = cursor_path.read_text(encoding="utf-8", errors="replace")
        synth = synthetic_trajectory_from_cursor_cli(raw)
        if synth and synth.get("steps"):
            meta["source"] = "cursor-cli.txt"
            meta["note"] = "Synthetic ATIF from Cursor CLI log (heuristic)"
            return synth, meta

    cline_path = logs / "cline.txt"
    if cline_path.exists():
        raw = cline_path.read_text(encoding="utf-8", errors="replace")
        synth = synthetic_trajectory_from_cline_cli(raw)
        if synth and synth.get("steps"):
            meta["source"] = "cline.txt"
            meta["note"] = "Synthetic ATIF from Cline CLI JSONL log"
            return synth, meta

    opencode_path = logs / "opencode.txt"
    if opencode_path.exists():
        raw = opencode_path.read_text(encoding="utf-8", errors="replace")
        synth = synthetic_trajectory_from_opencode_json(raw)
        if synth and synth.get("steps"):
            meta["source"] = "opencode.txt"
            meta["note"] = "Synthetic ATIF from OpenCode JSON stream"
            return synth, meta

    codex_path = logs / "codex.txt"
    if codex_path.exists():
        raw = codex_path.read_text(encoding="utf-8", errors="replace")
        synth = synthetic_trajectory_from_codex_txt(raw)
        if synth and synth.get("steps"):
            meta["source"] = "codex.txt"
            meta["note"] = "Synthetic ATIF from Codex structured log"
            return synth, meta

    return None, meta
