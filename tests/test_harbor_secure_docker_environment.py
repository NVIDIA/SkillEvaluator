# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import uuid
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from harbor.environments.base import ExecResult

from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
from skillevaluator.tier3.harbor.runner import build_harbor_run_command
from skillevaluator.tier3.harbor.secure_docker_environment import (
    SECURE_DOCKER_ENV_IMPORT_PATH,
    SkillEvaluatorDockerEnvironment,
    SkillEvaluatorSecureDockerEnvironment,
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
        stop_main_on_interrupt: bool = False,
    ) -> ExecResult:
        del self, check, timeout_sec
        captured["command"] = command
        captured["env"] = env_overrides
        captured["stop_main_on_interrupt"] = stop_main_on_interrupt
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
    assert captured["stop_main_on_interrupt"] is True


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


def test_compose_check_false_redacts_success_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-output-test"
    environment.environment_name = "secure-output-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return f"stdout {_SENTINEL}".encode(), f"stderr {_SENTINEL}".encode()

    async def create_subprocess(*_args: object, **_kwargs: object) -> Process:
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(
        environment._run_docker_compose_command(
            ["exec", "-e", "DATABASE_URL", "main", "true"],
            check=False,
            env_overrides={"DATABASE_URL": _SENTINEL},
        )
    )

    assert result.stdout == "stdout [REDACTED]"
    assert result.stderr == "stderr [REDACTED]"


def test_compose_stdin_handoff_redacts_secret_without_argv_or_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-stdin-test"
    environment.environment_name = "secure-stdin-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    captured: dict[str, object] = {}

    class Process:
        returncode = 9

        async def communicate(self, input_bytes: bytes | None = None) -> tuple[bytes, None]:
            captured["input"] = input_bytes
            return f"failure included {_SENTINEL}".encode(), None

    async def create_subprocess(*args: object, **kwargs: object) -> Process:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["stdin"] = kwargs["stdin"]
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "-T", "main", "true"],
                stdin_bytes=b"private-stream-payload",
                redact_values={_SENTINEL},
            )
        )

    assert captured["input"] == b"private-stream-payload"
    assert captured["stdin"] is asyncio.subprocess.PIPE
    assert _SENTINEL not in " ".join(str(arg) for arg in captured["args"])
    assert _SENTINEL not in captured["env"].values()
    assert _SENTINEL not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_redact_ignores_short_env_values_that_would_corrupt_loopback_origins() -> None:
    from skillevaluator.tier3.harbor.secure_docker_environment import _redact

    origin = "http://127.0.0.1:41927\n"
    assert _redact(origin, {"1"}) == origin
    assert _redact(origin, {"1", "41927"}) == origin
    assert _redact("token=abcdefgh", {"abcdefgh"}) == "token=[REDACTED]"


def test_compose_redacts_long_secrets_without_rewriting_short_env_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-short-secret-test"
    environment.environment_name = "secure-short-secret-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    long_secret = "abcdefgh"

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"http://127.0.0.1:41927\n", f"stderr {long_secret}".encode()

    async def create_subprocess(*_args: object, **_kwargs: object) -> Process:
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(
        environment._run_docker_compose_command(
            ["exec", "-e", "CLAUDE_CODE_DISABLE_POLICY_SKILLS", "-e", "DATABASE_URL", "main", "true"],
            check=False,
            env_overrides={
                "CLAUDE_CODE_DISABLE_POLICY_SKILLS": "1",
                "DATABASE_URL": long_secret,
            },
        )
    )

    assert result.stdout == "http://127.0.0.1:41927\n"
    assert result.stderr == "stderr [REDACTED]"


def test_compose_cancellation_reaps_process_tree_even_when_repeated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-cancellation-test"
    environment.environment_name = "secure-cancellation-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01, raising=False)

    async def run_cancelled() -> list[str]:
        actions: list[str] = []
        communicating = asyncio.Event()
        completed: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()

        class FakeProcess:
            pid = 4343
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                communicating.set()
                return await asyncio.shield(completed)

            def terminate(self) -> None:
                actions.append("terminate")

            def kill(self) -> None:
                actions.append("kill")
                self.returncode = -9
                if not completed.done():
                    completed.set_result((b"", b""))

        process = FakeProcess()

        async def create_subprocess(*_args: object, **_kwargs: object) -> FakeProcess:
            return process

        def killpg(_pid: int, value: signal.Signals) -> None:
            if value == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg, raising=False)
        task = asyncio.create_task(environment._run_docker_compose_command(["exec", "main", "sleep", "30"]))
        await asyncio.wait_for(communicating.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        return actions

    assert asyncio.run(run_cancelled()) == ["terminate", "kill"]


@pytest.mark.parametrize("interrupt_mode", ["cancel", "timeout"])
def test_interrupted_exec_stops_main_container_before_reaping_host_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_mode: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-remote-cancellation-test"
    environment.environment_name = "secure-remote-cancellation-test"
    environment.environment_dir = tmp_path
    environment.default_user = None
    environment.task_env_config = SimpleNamespace(workdir=None, env={})
    environment._persistent_env = {}
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def run_cancelled() -> tuple[list[str], list[tuple[object, ...]]]:
        actions: list[str] = []
        commands: list[tuple[object, ...]] = []
        communicating = asyncio.Event()
        completed: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()

        class OriginalProcess:
            pid = 4545
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                communicating.set()
                return await asyncio.shield(completed)

        class CleanupProcess:
            pid = 4546
            returncode = 0

            def __init__(self, action: str) -> None:
                self.action = action

            async def communicate(self) -> tuple[bytes, bytes]:
                actions.append(self.action)
                return b"", b""

        original = OriginalProcess()

        async def create_subprocess(*args: object, **_kwargs: object) -> OriginalProcess | CleanupProcess:
            commands.append(args)
            if len(commands) == 1:
                return original
            rendered = " ".join(str(arg) for arg in args)
            return CleanupProcess("container-stop" if " stop " in f" {rendered} " else "container-remove")

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == original.pid
            actions.append(f"host-{value.name.lower()}")
            if value == signal.SIGKILL:
                original.returncode = -9
                if not completed.done():
                    completed.set_result((b"", b""))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        task = asyncio.create_task(
            environment.exec(
                "sleep 30",
                env={"NVIDIA_API_KEY": "credential-for-cancellation-test"},
                timeout_sec=0.01 if interrupt_mode == "timeout" else None,
            )
        )
        await asyncio.wait_for(communicating.wait(), timeout=1)
        if interrupt_mode == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        else:
            with pytest.raises(RuntimeError, match="timed out"):
                await asyncio.wait_for(task, timeout=1)
        return actions, commands

    actions, commands = asyncio.run(run_cancelled())

    assert len(commands) == 3
    rendered_original = " ".join(str(arg) for arg in commands[0])
    rendered_cleanup = " ".join(str(arg) for arg in commands[1])
    assert ".skillevaluator-exec-" not in rendered_original
    assert "SKILLEVALUATOR_EXEC_TOKEN" not in rendered_original
    assert rendered_cleanup.endswith("stop --timeout 0 main")
    assert " ".join(str(arg) for arg in commands[2]).endswith("rm --force --stop --volumes main")
    assert actions.index("container-stop") < actions.index("host-sigterm")
    assert actions.index("container-remove") < actions.index("host-sigterm")


def test_exec_cancelled_during_process_creation_still_stops_main_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-creation-cancellation-test"
    environment.environment_name = "secure-creation-cancellation-test"
    environment.environment_dir = tmp_path
    environment.default_user = None
    environment.task_env_config = SimpleNamespace(workdir=None, env={})
    environment._persistent_env = {}
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def run_cancelled() -> tuple[list[str], list[tuple[object, ...]]]:
        actions: list[str] = []
        commands: list[tuple[object, ...]] = []
        creation_started = asyncio.Event()
        release_creation = asyncio.Event()
        stop_started = asyncio.Event()
        release_stop = asyncio.Event()
        completed: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()

        class OriginalProcess:
            pid = 4645
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                return await asyncio.shield(completed)

        class CleanupProcess:
            pid = 4646
            returncode = 0

            def __init__(self, action: str) -> None:
                self.action = action

            async def communicate(self) -> tuple[bytes, bytes]:
                actions.append(f"{self.action}-started")
                if self.action == "container-stop":
                    stop_started.set()
                    await release_stop.wait()
                actions.append(f"{self.action}-finished")
                return b"", b""

        original = OriginalProcess()

        async def create_subprocess(*args: object, **_kwargs: object) -> OriginalProcess | CleanupProcess:
            commands.append(args)
            if len(commands) == 1:
                creation_started.set()
                await release_creation.wait()
                return original
            rendered = " ".join(str(arg) for arg in args)
            return CleanupProcess("container-stop" if " stop " in f" {rendered} " else "container-remove")

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == original.pid
            actions.append(f"host-{value.name.lower()}")
            if value == signal.SIGKILL:
                original.returncode = -9
                if not completed.done():
                    completed.set_result((b"", b""))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        task = asyncio.create_task(environment.exec("sleep 30", env={"NVIDIA_API_KEY": "creation-secret"}))
        await asyncio.wait_for(creation_started.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release_creation.set()
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        task.cancel()
        release_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        return actions, commands

    actions, commands = asyncio.run(run_cancelled())

    assert len(commands) == 3
    assert " ".join(str(arg) for arg in commands[1]).endswith("stop --timeout 0 main")
    assert " ".join(str(arg) for arg in commands[2]).endswith("rm --force --stop --volumes main")
    assert actions.index("container-remove-finished") < actions.index("host-sigterm")


@pytest.mark.parametrize("interrupt_mode", ["cancel", "timeout"])
def test_interrupted_exec_fails_closed_when_main_container_containment_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_mode: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-stop-failure-test"
    environment.environment_name = "secure-stop-failure-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def run_timeout() -> tuple[list[tuple[object, ...]], list[signal.Signals]]:
        commands: list[tuple[object, ...]] = []
        signals: list[signal.Signals] = []
        communicating = asyncio.Event()
        completed: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()

        class OriginalProcess:
            pid = 4745
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                communicating.set()
                return await asyncio.shield(completed)

        class FailedStopProcess:
            pid = 4746
            returncode = 1

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"stop failed", b""

        original = OriginalProcess()

        async def create_subprocess(*args: object, **_kwargs: object) -> OriginalProcess | FailedStopProcess:
            commands.append(args)
            return original if len(commands) == 1 else FailedStopProcess()

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == original.pid
            signals.append(value)
            if value == signal.SIGKILL:
                original.returncode = -9
                if not completed.done():
                    completed.set_result((b"", b""))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        task = asyncio.create_task(
            environment._run_docker_compose_command(
                ["exec", "main", "sleep", "30"],
                timeout_sec=0.01 if interrupt_mode == "timeout" else None,
                stop_main_on_interrupt=True,
            )
        )
        await asyncio.wait_for(communicating.wait(), timeout=1)
        if interrupt_mode == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        with pytest.raises(RuntimeError, match="main task container containment could not be confirmed"):
            await task
        return commands, signals

    commands, signals = asyncio.run(run_timeout())

    assert len(commands) >= 2
    assert " ".join(str(arg) for arg in commands[1]).endswith("stop --timeout 0 main")
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_compose_process_cleanup_remains_bounded_when_communication_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_CANCEL_SECONDS", 0.01, raising=False)

    async def run_cleanup() -> bool:
        started = asyncio.Event()
        release = asyncio.Event()

        async def stubborn_communication() -> tuple[bytes, bytes]:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return b"", b""

        communication = asyncio.create_task(stubborn_communication())
        await started.wait()

        class FakeProcess:
            pid = 4444
            returncode = None

        monkeypatch.setattr(secure_docker_environment.os, "killpg", lambda *_args: None, raising=False)
        cleanup = asyncio.create_task(
            secure_docker_environment._terminate_process_tree(
                FakeProcess(),  # type: ignore[arg-type]
                communication,
                preserve_cancellation=False,
            )
        )
        done, _pending = await asyncio.wait({cleanup}, timeout=0.1)
        finished_within_bound = cleanup in done
        release.set()
        await communication
        await cleanup
        return finished_within_bound

    assert asyncio.run(run_cleanup()) is True


@pytest.mark.integration
@pytest.mark.parametrize("stop_mode", ["cancel", "timeout"])
def test_real_docker_interrupted_exec_stops_task_container_only(
    tmp_path: Path,
    stop_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Container-root marker forgery cannot defeat a host-authoritative stop."""
    docker_info = subprocess.run(
        ["docker", "info", "--format", "{{.OSType}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if docker_info.returncode != 0 or docker_info.stdout.strip() != "linux":
        pytest.skip("requires a running Linux Docker daemon")

    from skillevaluator.tier3.harbor import secure_docker_environment

    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.2)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.2)

    target_dir = tmp_path / "target"
    unrelated_dir = tmp_path / "unrelated"
    target_dir.mkdir()
    unrelated_dir.mkdir()
    target_compose_path = target_dir / "docker-compose.yaml"
    target_compose_path.write_text(
        "services:\n"
        "  main:\n"
        "    image: python:3.13-slim\n"
        '    command: ["sh", "-c", "trap : TERM INT; sleep infinity & wait"]\n'
        "  helper:\n"
        "    image: python:3.13-slim\n"
        '    command: ["sh", "-c", "trap : TERM INT; sleep infinity & wait"]\n',
        encoding="utf-8",
    )
    unrelated_compose_path = unrelated_dir / "docker-compose.yaml"
    unrelated_compose_path.write_text(
        "services:\n"
        "  main:\n"
        "    image: python:3.13-slim\n"
        '    command: ["sh", "-c", "trap : TERM INT; sleep infinity & wait"]\n',
        encoding="utf-8",
    )
    target_project = f"skillevaluator-cancel-target-{uuid.uuid4().hex[:10]}"
    unrelated_project = f"skillevaluator-cancel-unrelated-{uuid.uuid4().hex[:10]}"
    target_compose = [
        "docker",
        "compose",
        "--project-name",
        target_project,
        "--project-directory",
        str(target_dir),
        "-f",
        str(target_compose_path),
    ]
    unrelated_compose = [
        "docker",
        "compose",
        "--project-name",
        unrelated_project,
        "--project-directory",
        str(unrelated_dir),
        "-f",
        str(unrelated_compose_path),
    ]

    def cleanup_projects() -> None:
        subprocess.run(
            [*target_compose, "down", "--remove-orphans", "--volumes"],
            check=False,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            [*unrelated_compose, "down", "--remove-orphans", "--volumes"],
            check=False,
            capture_output=True,
            timeout=60,
        )

    request.addfinalizer(cleanup_projects)
    subprocess.run([*target_compose, "up", "-d", "--wait"], check=True, timeout=60)
    subprocess.run([*unrelated_compose, "up", "-d", "--wait"], check=True, timeout=60)

    target_main_id = subprocess.run(
        [*target_compose, "ps", "-q", "main"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    target_helper_id = subprocess.run(
        [*target_compose, "ps", "-q", "helper"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    unrelated_main_id = subprocess.run(
        [*unrelated_compose, "ps", "-q", "main"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    assert target_main_id and target_helper_id and unrelated_main_id

    class ComposeOnlyEnvironment(SkillEvaluatorSecureDockerEnvironment):
        @property
        def _docker_compose_paths(self) -> list[Path]:
            return [target_compose_path]

        async def upload_file(self, source_path: Path | str, target_path: str) -> None:
            await self._run_docker_compose_command(
                ["cp", str(Path(source_path).resolve()), f"main:{target_path}"],
                check=True,
            )

    environment = object.__new__(ComposeOnlyEnvironment)
    environment.session_id = target_project
    environment.environment_name = target_project
    environment.environment_dir = target_dir
    environment.default_user = None
    environment.task_env_config = SimpleNamespace(workdir=None, env={})
    environment._persistent_env = {}
    environment._is_windows_container = False
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: dict(os.environ), environment)
    remote_pid_path = f"/tmp/skillevaluator-test-{uuid.uuid4().hex}.pid"
    attack_status_path = f"/tmp/skillevaluator-attack-{uuid.uuid4().hex}.txt"
    credential = "credential-must-not-outlive-cancelled-agent-command"
    real_create_subprocess = asyncio.create_subprocess_exec
    exec_clients: list[asyncio.subprocess.Process] = []

    async def capture_exec_client(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await real_create_subprocess(*args, **kwargs)
        if remote_pid_path in " ".join(str(arg) for arg in args):
            exec_clients.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_exec_client)

    normal_result = asyncio.run(
        environment.exec_with_sensitive_env(
            "printf 'normal-output\\n'; printf 'normal-error\\n' >&2; "
            "printf 'control-token=%s\\n' \"${SKILLEVALUATOR_EXEC_TOKEN-unset}\"; exit 7",
            env={"NVIDIA_API_KEY": credential},
        )
    )
    assert normal_result.return_code == 7
    assert set((normal_result.stdout or "").splitlines()) == {
        "normal-output",
        "normal-error",
        "control-token=unset",
    }
    assert credential not in (normal_result.stdout or "")
    assert (
        subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", target_main_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        == "true"
    )

    malicious_command = (
        "marker=$(find /tmp -maxdepth 1 -type f -name '.skillevaluator-exec-*' -print -quit 2>/dev/null || true); "
        "token=''; "
        "for proc_env in /proc/[0-9]*/environ; do "
        '[ -r "$proc_env" ] || continue; '
        "candidate=$(tr '\\000' '\\n' < \"$proc_env\" 2>/dev/null "
        "| sed -n 's/^SKILLEVALUATOR_EXEC_TOKEN=//p' | head -n 1); "
        'if [ -n "$candidate" ]; then token=$candidate; break; fi; '
        "done; "
        'if [ -n "$marker" ] && [ -n "$token" ]; then '
        'setsid env SKILLEVALUATOR_EXEC_TOKEN="$token" '
        "sh -c 'trap \"\" TERM INT HUP; sleep 300 & wait' & decoy=$!; "
        'printf \'%s\\n\' "$decoy" > "$marker"; '
        f"printf 'legacy-control-forged\\n' > {attack_status_path}; "
        "else "
        f"printf 'no-in-container-control\\n' > {attack_status_path}; "
        "fi; "
        "trap '' TERM INT HUP; "
        "sleep 300 & child=$!; "
        f"printf '%s\\n' \"$child\" > {remote_pid_path}; "
        'wait "$child"'
    )

    async def exercise() -> str:
        task = asyncio.create_task(
            environment.exec(
                malicious_command,
                env={"NVIDIA_API_KEY": credential},
                timeout_sec=1 if stop_mode == "timeout" else None,
            )
        )
        for _ in range(100):
            probe = subprocess.run(
                [*target_compose, "exec", "-T", "main", "test", "-s", remote_pid_path, "-a", "-s", attack_status_path],
                check=False,
                capture_output=True,
                timeout=5,
            )
            if probe.returncode == 0:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("adversarial remote command did not finish setup")

        attack_status = subprocess.run(
            [*target_compose, "exec", "-T", "main", "cat", attack_status_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        assert exec_clients, "did not capture the credential-bearing Compose exec client"
        os.killpg(exec_clients[0].pid, signal.SIGSTOP)
        if stop_mode == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=15)
        else:
            with pytest.raises(RuntimeError, match="timed out"):
                await asyncio.wait_for(task, timeout=15)
        return attack_status

    try:
        attack_status = asyncio.run(exercise())
        target_container_after_interrupt = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", target_main_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        helper_running_after_interrupt = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", target_helper_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        unrelated_running_after_interrupt = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", unrelated_main_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    finally:
        cleanup_projects()

    leaked_containers = "\n".join(
        subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", f"label=com.docker.compose.project={project}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        for project in (target_project, unrelated_project)
    ).strip()
    leaked_networks = "\n".join(
        subprocess.run(
            ["docker", "network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        for project in (target_project, unrelated_project)
    ).strip()

    assert (target_container_after_interrupt.returncode, attack_status) == (
        1,
        "no-in-container-control",
    ), (
        "interrupted credential-bearing task was not contained: "
        f"container_inspect_status={target_container_after_interrupt.returncode}, attack_status={attack_status}"
    )
    assert helper_running_after_interrupt == "true", "stopping main affected another service in the same project"
    assert unrelated_running_after_interrupt == "true", "stopping main affected an unrelated Compose project"
    assert leaked_containers == ""
    assert leaked_networks == ""


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
