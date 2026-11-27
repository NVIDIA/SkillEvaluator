"""Translate observed Hermes calls into the names used by Tier 3 grading."""

from __future__ import annotations

import json
from typing import Any, NamedTuple

HERMES_HOME = "/tmp/hermes"
HERMES_EXIT_UNKNOWN = "hermes_process_exit_unknown"
HERMES_PROCESS_STATE = "hermes_process_state"
HERMES_TOOLS = ("terminal", "skill_view", "read_file")


class HermesCall(NamedTuple):
    name: str
    arguments: dict[str, Any]
    error: bool | None
    observation: str
    process_state: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self.process_state is None:
            return {}
        metadata = {HERMES_PROCESS_STATE: self.process_state}
        if self.process_state in ("running", "unknown"):
            return {**metadata, HERMES_EXIT_UNKNOWN: True}
        return metadata


def _skill_directory(directory: str) -> bool:
    return not any(char in directory for char in ("\\", ":", "\x00")) and all(
        part not in ("", ".", "..", "SKILL.md") for part in directory.split("/")
    )


def _manifest_path(skill: str, payload: dict[str, Any]) -> str | None:
    """Keep the returned relative manifest path; only synthesize one for older results."""
    directory_requested = skill.removeprefix("./")
    if not _skill_directory(directory_requested) or payload.get("file", "SKILL.md") not in ("SKILL.md", "./SKILL.md"):
        return None
    resolved = payload.get("name", skill)
    if "path" not in payload:
        return f"{directory_requested}/SKILL.md" if resolved == skill else None
    path = payload["path"]
    if not isinstance(path, str) or not path.endswith("/SKILL.md"):
        return None
    directory = path.removesuffix("/SKILL.md")
    if not _skill_directory(directory):
        return None
    leaf = directory.rsplit("/", 1)[-1]
    if "/" in skill:
        return path if directory == directory_requested and resolved in (skill, directory_requested, leaf) else None
    if leaf == skill and resolved == skill:
        return path
    if (
        skill in (leaf, resolved)
        and isinstance(resolved, str)
        and leaf.isascii()
        and resolved.isascii()
        and leaf.lower() == resolved.lower()
    ):
        return path
    return None


def normalize_hermes_call(
    name: str, arguments: dict[str, Any], content: str, *, native_session: bool = False
) -> HermesCall:
    """Return the grading name, arguments, error flag and observation."""
    try:
        payload = json.loads(content)
    except (ValueError, RecursionError):
        payload = None
    if not isinstance(payload, dict):
        payload = {}
    error = bool(payload.get("error"))
    if name == "terminal":
        error = payload.get("error") is not None
        if content == "[Tool execution cancelled — terminal was skipped due to user interrupt]":
            return HermesCall(name, arguments, True, content, "not_started")
        has_exit_code = type(payload.get("exit_code")) is int
        has_output = isinstance(payload.get("output"), str)
        if (
            payload.get("status") == "yielded_to_background"
            and payload.get("exit_code", -1) is None
            and payload.get("session_id")
            and not error
            and has_output
            and isinstance(arguments.get("command"), str)
        ):
            return HermesCall("bash", arguments, None, content, "running")
        shell_result = (has_exit_code and (native_session or has_output)) or (error and has_output)
        if shell_result and isinstance(arguments.get("command"), str):
            refused = payload.get("status") in ("blocked", "pending_approval", "disabled", "error", "degraded")
            if refused or (has_exit_code and payload["exit_code"] == -1 and error):
                return HermesCall(name, arguments, True, content, "not_started")
            if (
                has_exit_code
                and payload["exit_code"] == 0
                and not error
                and payload.get("session_id")
                and payload.get("output") == "Background process started"
            ):
                return HermesCall("bash", arguments, None, content, "running")
            if has_exit_code and (payload["exit_code"] != 0 or not error):
                content = f"{content.lstrip()}\nExit code: {payload['exit_code']}"
            return HermesCall(
                "bash",
                arguments,
                error or payload["exit_code"] != 0,
                content,
                "completed" if has_exit_code else "unknown",
            )
        if error:
            return HermesCall(name, arguments, True, content, "not_started")
        if native_session and payload.get("success") is True:
            return HermesCall("bash", arguments, None, content, "unknown")
        if isinstance(arguments.get("command"), str) and arguments.get("workdir"):
            return HermesCall(name, arguments, None, content, "unknown")
    elif name == "skill_view":
        skill = arguments.get("name")
        file_path = arguments.get("file_path") or "SKILL.md"
        if payload.get("success") is False or error:
            return HermesCall(name, arguments, True, content)
        if payload.get("success") is True and isinstance(payload.get("content"), str) and payload["content"].strip():
            if isinstance(skill, str) and skill and isinstance(file_path, str):
                if file_path in ("SKILL.md", "./SKILL.md"):
                    if manifest := _manifest_path(skill, payload):
                        return HermesCall("read", {"path": f"{HERMES_HOME}/skills/{manifest}"}, False, content)
                elif payload.get("name", skill) == skill:
                    return HermesCall(name, arguments, False, content)
            return HermesCall(name, arguments, None, content)
        if native_session:
            return HermesCall(name, arguments, True, content)
    elif name == "read_file" and native_session:
        return HermesCall(name, arguments, error or payload.get("success") is False, content)
    elif name == "read_file":
        path = arguments.get("path")
        read_error = payload.get("error")
        if (
            isinstance(path, str)
            and read_error is None
            and payload.get("success") is not False
            and isinstance(payload.get("content"), str)
        ):
            return HermesCall(name, arguments, False, content)
        if isinstance(path, str) and (
            (isinstance(read_error, str) and read_error.startswith("File not found: "))
            or (path.endswith("/SKILL.md") and (error or payload.get("success") is False))
        ):
            return HermesCall(name, arguments, True, content)
    return HermesCall(name, arguments, None, content)
