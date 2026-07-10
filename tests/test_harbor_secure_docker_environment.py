# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
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
        remote_control=None,
    ) -> ExecResult:
        del self, check, timeout_sec, remote_control
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


def test_exec_cancellation_terminates_scoped_remote_group_before_host_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

            async def communicate(self) -> tuple[bytes, bytes]:
                actions.append("remote-cleanup")
                return b"", b""

        original = OriginalProcess()

        async def create_subprocess(*args: object, **_kwargs: object) -> OriginalProcess | CleanupProcess:
            commands.append(args)
            return original if len(commands) == 1 else CleanupProcess()

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
            environment.exec("sleep 30", env={"NVIDIA_API_KEY": "credential-for-cancellation-test"})
        )
        await asyncio.wait_for(communicating.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        return actions, commands

    actions, commands = asyncio.run(run_cancelled())

    assert len(commands) == 2
    rendered_original = " ".join(str(arg) for arg in commands[0])
    rendered_cleanup = " ".join(str(arg) for arg in commands[1])
    assert ".skillevaluator-exec-" in rendered_original
    assert "SKILLEVALUATOR_EXEC_TOKEN" in rendered_original
    assert "exec -u root main bash -c" in rendered_cleanup
    assert "kill --" in rendered_cleanup
    assert actions.index("remote-cleanup") < actions.index("host-sigterm")


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
def test_real_docker_exec_stops_only_its_remote_process_tree(
    tmp_path: Path,
    stop_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """A stopped Compose client must not leave its credential-bearing exec alive."""
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

    compose_path = tmp_path / "docker-compose.yaml"
    compose_path.write_text(
        "services:\n"
        "  main:\n"
        "    image: python:3.13-slim\n"
        '    command: ["sh", "-c", "trap : TERM INT; sleep infinity & wait"]\n',
        encoding="utf-8",
    )
    project_name = f"skillevaluator-remote-cancel-{uuid.uuid4().hex[:10]}"
    compose = [
        "docker",
        "compose",
        "--project-name",
        project_name,
        "--project-directory",
        str(tmp_path),
        "-f",
        str(compose_path),
    ]
    subprocess.run([*compose, "up", "-d", "--wait"], check=True, timeout=60)

    def cleanup_compose() -> None:
        subprocess.run([*compose, "down", "--remove-orphans", "--volumes"], check=False, timeout=60)

    request.addfinalizer(cleanup_compose)

    class ComposeOnlyEnvironment(SkillEvaluatorDockerEnvironment):
        @property
        def _docker_compose_paths(self) -> list[Path]:
            return [compose_path]

    environment = object.__new__(ComposeOnlyEnvironment)
    environment.session_id = project_name
    environment.environment_name = project_name
    environment.environment_dir = tmp_path
    environment.default_user = None
    environment.task_env_config = SimpleNamespace(workdir=None, env={})
    environment._persistent_env = {}
    environment._is_windows_container = False
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: dict(os.environ), environment)
    remote_pid_path = f"/tmp/skillevaluator-test-{uuid.uuid4().hex}.pid"
    unrelated_pid_path = f"/tmp/skillevaluator-unrelated-{uuid.uuid4().hex}.pid"
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
        environment.exec(
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

    subprocess.run(
        [
            *compose,
            "exec",
            "-T",
            "-d",
            "main",
            "sh",
            "-c",
            f"echo $$ > {unrelated_pid_path}; exec sleep 300",
        ],
        check=True,
        timeout=5,
    )
    for _ in range(100):
        unrelated_pid_result = subprocess.run(
            [*compose, "exec", "-T", "main", "cat", unrelated_pid_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if unrelated_pid_result.returncode == 0 and unrelated_pid_result.stdout.strip():
            unrelated_pid = int(unrelated_pid_result.stdout.strip())
            break
        time.sleep(0.05)
    else:
        pytest.fail("unrelated container process did not publish its PID")

    tampered_marker = f"/tmp/.skillevaluator-exec-tampered-{uuid.uuid4().hex}"
    subprocess.run(
        [*compose, "exec", "-T", "main", "sh", "-c", f"printf '%s\\n' {unrelated_pid} > {tampered_marker}"],
        check=True,
        timeout=5,
    )
    tampered_cleanup = subprocess.run(
        [
            *compose,
            "exec",
            "-T",
            "-u",
            "root",
            "main",
            "bash",
            "-c",
            secure_docker_environment._REMOTE_TERMINATE_SCRIPT,
            "skillevaluator-tamper-test",
            tampered_marker,
            "token-not-owned-by-the-unrelated-process",
        ],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert tampered_cleanup.returncode == 4
    subprocess.run(
        [*compose, "exec", "-T", "main", "rm", "-f", "--", tampered_marker],
        check=True,
        timeout=5,
    )

    async def exercise() -> int:
        task = asyncio.create_task(
            environment.exec(
                f"trap '' TERM INT HUP; sleep 300 & child=$!; echo $child > {remote_pid_path}; wait $child",
                env={"NVIDIA_API_KEY": credential},
                timeout_sec=1 if stop_mode == "timeout" else None,
            )
        )
        for _ in range(100):
            probe = subprocess.run(
                [*compose, "exec", "-T", "main", "test", "-s", remote_pid_path],
                check=False,
                capture_output=True,
                timeout=5,
            )
            if probe.returncode == 0:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("remote command did not publish its PID")

        remote_pid = int(
            subprocess.run(
                [*compose, "exec", "-T", "main", "cat", remote_pid_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
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
        return remote_pid

    try:
        remote_pid = asyncio.run(exercise())
        process_probe = subprocess.run(
            [*compose, "exec", "-T", "main", "sh", "-c", f"kill -0 {remote_pid}"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        marker_probe = subprocess.run(
            [
                *compose,
                "exec",
                "-T",
                "main",
                "sh",
                "-c",
                "find /tmp -maxdepth 1 -name '.skillevaluator-exec-*' -print -quit",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        unrelated_probe = subprocess.run(
            [*compose, "exec", "-T", "main", "sh", "-c", f"kill -0 {unrelated_pid}"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        assert process_probe.returncode != 0, f"remote PID {remote_pid} survived {stop_mode}"
        assert unrelated_probe.returncode == 0, "scoped cleanup killed an unrelated container process"
        assert marker_probe.stdout.strip() == ""
    finally:
        cleanup_compose()


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
