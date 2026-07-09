# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for bundled Tier 3 reference scripts."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

REFERENCE_SKILLS = Path(__file__).parents[1] / "src/skillevaluator/tier3/reference_skills"
SCRIPTS = {
    "call_api": REFERENCE_SKILLS / "api-caller/scripts/call_api.py",
    "parse_openapi": REFERENCE_SKILLS / "api-caller/scripts/parse_openapi.py",
    "tasks": REFERENCE_SKILLS / "task-list/scripts/tasks.py",
}


class _SuccessfulResponse:
    """Minimal context-managed response used by request-body tests."""

    status = 200
    headers: ClassVar[dict[str, str]] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @staticmethod
    def read() -> bytes:
        return b"{}"


def _run_script(name: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPTS[name]), *args],
        check=False,
        capture_output=True,
        text=True,
        env=process_env,
    )


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"reference_script_{name}", SCRIPTS[name])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_content_containing_error_label_remains_successful(tmp_path: Path) -> None:
    result = _run_script(
        "tasks",
        "add",
        "--content",
        "Investigate Error: logs",
        env={"TASK_LIST_STATE_PATH": str(tmp_path / "todo.md")},
    )

    assert result.returncode == 0
    assert result.stdout == "Added [task_1]: Investigate Error: logs\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("name", "args", "expected_error"),
    [
        ("call_api", ("--url", "https://example.invalid", "--headers", "not-json"), "Invalid input"),
        ("call_api", (), "No URL provided"),
        ("parse_openapi", (), "Usage:"),
        ("tasks", ("add",), "Error: Content is required"),
    ],
)
def test_handled_input_failures_write_only_to_stderr(
    name: str, args: tuple[str, ...], expected_error: str, tmp_path: Path
) -> None:
    env = {"TASK_LIST_STATE_PATH": str(tmp_path / "todo.md")} if name == "tasks" else None

    result = _run_script(name, *args, env=env)

    assert result.returncode == 1
    assert result.stdout == ""
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("body", "encoded"),
    [
        ({}, b"{}"),
        ([], b"[]"),
        (0, b"0"),
        (False, b"false"),
        ("", b'""'),
        (None, b"null"),
    ],
)
def test_call_api_preserves_explicit_falsey_json_bodies(
    body: object,
    encoded: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("call_api")
    requests: list[object] = []

    def fake_urlopen(request: object, *, timeout: int):
        assert timeout == 30
        requests.append(request)
        return _SuccessfulResponse()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    result = module.make_request("https://example.invalid", method="POST", data=body)

    assert result["success"] is True
    assert len(requests) == 1
    assert requests[0].data == encoded


def test_call_api_omits_body_only_when_data_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("call_api")
    requests: list[object] = []

    def fake_urlopen(request: object, *, timeout: int):
        requests.append(request)
        return _SuccessfulResponse()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    result = module.make_request("https://example.invalid")

    assert result["success"] is True
    assert len(requests) == 1
    assert requests[0].data is None
