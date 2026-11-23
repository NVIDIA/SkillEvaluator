"""Translate observed Hermes calls into the names used by Tier 3 grading."""

from __future__ import annotations

import json
from typing import Any

HERMES_HOME = "/tmp/hermes"
HERMES_TOOLS = ("terminal", "skill_view", "read_file")


def _matches_categorized_manifest(skill: str, payload: dict[str, Any]) -> bool:
    """A categorized lookup returns the leaf name and the relative path to SKILL.md."""
    parts = skill.split("/")
    return (
        len(parts) > 1
        and payload.get("name") == parts[-1]
        and payload.get("path") == f"{skill}/SKILL.md"
        and "\\" not in skill
        and all(part not in ("", ".", "..", "SKILL.md") for part in parts)
    )


def normalize_hermes_call(
    name: str, arguments: dict[str, Any], content: str, *, native_session: bool = False
) -> tuple[str, dict[str, Any], bool | None]:
    """Return the grading name, arguments and any outcome the result establishes."""
    try:
        payload = json.loads(content)
    except (ValueError, RecursionError):
        payload = None
    if not isinstance(payload, dict):
        payload = {}
    error = bool(payload.get("error"))
    if name == "terminal":
        has_exit_code = type(payload.get("exit_code")) is int
        shell_result = (has_exit_code or error) and (native_session or isinstance(payload.get("output"), str))
        if shell_result and isinstance(arguments.get("command"), str):
            return "bash", arguments, error or payload["exit_code"] != 0
        if error:
            return name, arguments, True
        if native_session:
            return "bash", arguments, None
    elif name == "skill_view":
        skill = arguments.get("name")
        file_path = arguments.get("file_path") or "SKILL.md"
        if payload.get("success") is False or error:
            return name, arguments, True
        if payload.get("success") is True and isinstance(payload.get("content"), str) and payload["content"].strip():
            if isinstance(skill, str) and skill and isinstance(file_path, str):
                same_name = payload.get("name", skill) == skill
                if file_path in ("SKILL.md", "./SKILL.md"):
                    if same_name or _matches_categorized_manifest(skill, payload):
                        return "read", {"path": f"{HERMES_HOME}/skills/{skill}/SKILL.md"}, None
                elif same_name:
                    return name, arguments, False
            return name, arguments, None
        if native_session:
            return name, arguments, True
    elif name == "read_file" and native_session:
        return name, arguments, error or payload.get("success") is False
    elif name == "read_file":
        path = arguments.get("path")
        if (
            isinstance(path, str)
            and path.startswith(f"{HERMES_HOME}/skills/")
            and (payload.get("success") is False or payload.get("error") == f"File not found: {path}")
        ):
            return name, arguments, True
    return name, arguments, None
