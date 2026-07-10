# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from harbor.environments.base import ExecResult

from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
from skillevaluator.tier3.harbor.runner import build_harbor_run_command
from skillevaluator.tier3.harbor.secure_docker_environment import (
    SECURE_DOCKER_ENV_IMPORT_PATH,
    SkillEvaluatorDockerEnvironment,
    _secure_exec_arguments,
)

_SENTINEL = "sentinel-never-visible-in-argv-or-files"


def _write_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Do it", "expected_skill": "skill"}]),
        encoding="utf-8",
    )
    return skill


def test_generated_tasks_stage_only_names_and_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", _SENTINEL)
    task = generate_harbor_tasks(
        _write_skill(tmp_path),
        tmp_path / "tasks",
        runtime_env={"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"},
        verifier_env={
            "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        },
    )[0]

    staged_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in task.rglob("*") if path.is_file()
    )
    assert _SENTINEL not in staged_text
    assert 'NVIDIA_API_KEY = "${NVIDIA_API_KEY}"' in staged_text
    assert 'OPENAI_API_KEY = "${OPENAI_API_KEY}"' in staged_text


def test_docker_command_uses_secure_environment_import_path() -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="secure-docker",
        env_mode="docker",
    )

    assert "--env" not in command
    assert command[command.index("--environment-import-path") + 1] == SECURE_DOCKER_ENV_IMPORT_PATH


def test_exec_uses_name_only_argv_and_subprocess_override(tmp_path: Path) -> None:
    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.environment_dir = tmp_path
    environment.task_env_config = SimpleNamespace(workdir=None, env={"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"})
    environment._persistent_env = {"DATABASE_URL": "old-value"}
    environment.default_user = None
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])
    captured: dict[str, object] = {}

    async def _capture(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        env_overrides=None,
    ) -> ExecResult:
        del self, check, timeout_sec
        captured["command"] = command
        captured["env"] = env_overrides
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    environment._run_docker_compose_command = MethodType(_capture, environment)
    asyncio.run(
        environment.exec(
            "true",
            env={
                "NVIDIA_API_KEY": _SENTINEL,
                "PLAIN_SETTING": "visible",
                "DATABASE_URL": "new-value",
            },
        )
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert _SENTINEL not in " ".join(command)
    assert command[command.index("NVIDIA_API_KEY") - 1] == "-e"
    assert "PLAIN_SETTING=visible" not in command
    assert "DATABASE_URL=new-value" not in command
    assert captured["env"] == {
        "DATABASE_URL": "new-value",
        "NVIDIA_API_KEY": _SENTINEL,
        "PLAIN_SETTING": "visible",
    }


def test_all_values_use_subprocess_env_including_empty_and_special_values() -> None:
    special = "spaces = quotes ' \" and $shell"
    arguments, child_environment = _secure_exec_arguments({"DATABASE_URL": _SENTINEL, "EMPTY": "", "SPECIAL": special})

    assert arguments == ["-e", "DATABASE_URL", "-e", "EMPTY", "-e", "SPECIAL"]
    assert _SENTINEL not in " ".join(arguments)
    assert child_environment == {"DATABASE_URL": _SENTINEL, "EMPTY": "", "SPECIAL": special}


def test_compose_process_receives_value_only_in_env_and_redacts_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-test"
    environment.environment_name = "secure-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._compose_env_vars = MethodType(
        lambda _self, **_kwargs: {"PATH": "/usr/bin"},
        environment,
    )
    captured: dict[str, object] = {}

    class _Process:
        returncode = 7

        async def communicate(self):
            return f"failure included {_SENTINEL}".encode(), None

    async def _create_subprocess(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return _Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create_subprocess)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "-e", "DATABASE_URL", "main", "true"],
                env_overrides={"DATABASE_URL": _SENTINEL},
            )
        )

    assert _SENTINEL not in " ".join(captured["args"])
    assert captured["env"]["DATABASE_URL"] == _SENTINEL
    assert _SENTINEL not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"BAD-NAME": _SENTINEL}, "Invalid environment variable name"),
        ({"VALID_NAME": "bad\x00value"}, "contains a NUL byte"),
    ],
)
def test_invalid_exec_environment_fails_without_serializing_value(
    environment: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError) as caught:
        _secure_exec_arguments(environment)

    assert message in str(caught.value)
    assert _SENTINEL not in str(caught.value)
