# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker subprocess argv must never contain evaluator or agent credentials."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_secure_docker_exec_hands_environment_over_by_file_without_argv_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "skillevaluator.tier3.harbor.secure_docker_environment"
    assert importlib.util.find_spec(module_name) is not None, "secure Docker environment module is missing"
    module = importlib.import_module(module_name)
    environment = object.__new__(module.SkillEvaluatorSecureDockerEnvironment)
    environment._persistent_env = {"PERSISTENT_TOKEN": "persistent-secret-value"}
    environment.default_user = "1000"
    environment.task_env_config = SimpleNamespace(workdir="/workspace")
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])

    uploaded: list[tuple[str, str]] = []
    docker_commands: list[list[str]] = []

    async def fake_upload_file(source_path: Path | str, target_path: str) -> None:
        uploaded.append((Path(source_path).read_text(encoding="utf-8"), target_path))

    async def fake_run(
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        stop_main_on_interrupt: bool = False,
    ):
        del check, timeout_sec
        assert stop_main_on_interrupt is ("if ! ." in " ".join(command))
        docker_commands.append(command)
        return SimpleNamespace(stdout="ok", stderr=None, return_code=0)

    monkeypatch.setattr(environment, "upload_file", fake_upload_file)
    monkeypatch.setattr(environment, "_run_docker_compose_command", fake_run)

    secret = "nvidia-secret-value-for-test"
    result = asyncio.run(
        environment.exec(
            "printf done",
            env={"NVIDIA_API_KEY": secret, "VISIBLE_SETTING": "enabled"},
        )
    )

    assert result.return_code == 0
    assert uploaded and secret in uploaded[0][0]
    assert "persistent-secret-value" in uploaded[0][0]
    rendered_argv = "\n".join("\0".join(command) for command in docker_commands)
    assert secret not in rendered_argv
    assert "persistent-secret-value" not in rendered_argv
    assert "NVIDIA_API_KEY=" not in rendered_argv
    assert all("-e" not in command for command in docker_commands)
    assert any("rm -f" in part for command in docker_commands for part in command)


def test_secure_docker_exec_redacts_persistent_and_per_call_credentials_from_all_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_docker_environment")
    environment = object.__new__(module.SkillEvaluatorSecureDockerEnvironment)
    persistent_secret = "persistent-credential-for-redaction"
    per_call_secret = "per-call-credential-for-redaction"
    environment._persistent_env = {"PERSISTENT_TOKEN": persistent_secret}
    environment.default_user = "1000"
    environment.task_env_config = SimpleNamespace(workdir="/workspace")
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])
    environment.session_id = "secure-redaction-test"
    environment.environment_name = "secure-redaction-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._compose_env_vars = lambda **_kwargs: {"PATH": "/usr/bin"}

    async def fake_upload_file(_source_path: Path | str, _target_path: str) -> None:
        return None

    class FakeProcess:
        returncode = 0

        def __init__(self, *, expose_secrets: bool) -> None:
            self._expose_secrets = expose_secrets

        async def communicate(self) -> tuple[bytes, bytes]:
            if not self._expose_secrets:
                return b"", b""
            output = f"stdout {persistent_secret} {per_call_secret}".encode()
            error = f"stderr {per_call_secret} {persistent_secret}".encode()
            return output, error

    async def create_subprocess(*args: object, **_kwargs: object) -> FakeProcess:
        rendered = " ".join(str(arg) for arg in args)
        return FakeProcess(expose_secrets="if ! ." in rendered)

    monkeypatch.setattr(environment, "upload_file", fake_upload_file)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(environment.exec("printf done", env={"NVIDIA_API_KEY": per_call_secret}))

    rendered = f"{result.stdout}\n{result.stderr}"
    assert persistent_secret not in rendered
    assert per_call_secret not in rendered
    assert rendered.count("[REDACTED]") == 4


def test_nvidia_build_docker_command_uses_secure_environment_and_bridge() -> None:
    from skillevaluator.tier3.harbor import runner

    agent_import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"

    command = runner.build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="secure-build",
        env_mode="docker",
        agent_import_path=agent_import_path,
    )

    assert command[command.index("--agent-import-path") + 1] == agent_import_path
    assert command[command.index("--environment-import-path") + 1] == runner.SECURE_DOCKER_ENV_IMPORT_PATH
    assert "-a" not in command
    assert "--env" not in command


def test_secure_docker_exec_cleans_remote_handoff_when_upload_reports_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    del tmp_path
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_docker_environment")
    environment = object.__new__(module.SkillEvaluatorSecureDockerEnvironment)
    environment._persistent_env = {}
    environment.default_user = "1000"
    environment.task_env_config = SimpleNamespace(workdir="/workspace")
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])
    removed: list[str] = []

    async def upload_then_fail(_source_path: Path | str, _target_path: str) -> None:
        raise ConnectionError("docker cp disconnected after creating the file")

    async def remove_handoff(remote_path: str) -> None:
        removed.append(remote_path)

    monkeypatch.setattr(environment, "upload_file", upload_then_fail)
    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)

    with pytest.raises(ConnectionError, match="docker cp disconnected"):
        asyncio.run(environment.exec("true", env={"NVIDIA_API_KEY": "secret-for-test"}))

    assert len(removed) == 1


def test_secure_docker_exec_repeated_cancellation_does_not_interrupt_handoff_cleanup(
    monkeypatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_docker_environment")
    environment = object.__new__(module.SkillEvaluatorSecureDockerEnvironment)
    environment._persistent_env = {}
    environment.default_user = "1000"
    environment.task_env_config = SimpleNamespace(workdir="/workspace")
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])

    async def run_cancelled() -> bool:
        upload_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = False

        async def blocked_upload(_source_path: Path | str, _target_path: str) -> None:
            upload_started.set()
            await asyncio.Event().wait()

        async def remove_handoff(_remote_path: str) -> None:
            nonlocal cleanup_finished
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished = True

        monkeypatch.setattr(environment, "upload_file", blocked_upload)
        monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)

        task = asyncio.create_task(environment.exec("true", env={"NVIDIA_API_KEY": "secret-for-test"}))
        await asyncio.wait_for(upload_started.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        task.cancel()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        return cleanup_finished

    assert asyncio.run(run_cancelled()) is True


def test_secure_docker_exec_fails_closed_when_final_secret_cleanup_fails(monkeypatch) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_docker_environment")
    environment = object.__new__(module.SkillEvaluatorSecureDockerEnvironment)
    environment._persistent_env = {}
    environment.default_user = "1000"
    environment.task_env_config = SimpleNamespace(workdir="/workspace")
    environment._platform = SimpleNamespace(exec_shell_args=lambda command: ["bash", "-c", command])

    async def fake_upload_file(_source_path: Path | str, _target_path: str) -> None:
        return None

    async def fake_run(
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        stop_main_on_interrupt: bool = False,
    ):
        del check, timeout_sec
        assert stop_main_on_interrupt is ("if ! ." in " ".join(command))
        return SimpleNamespace(stdout="ok", stderr=None, return_code=0)

    async def failed_cleanup(_remote_path: str) -> None:
        raise ConnectionError("container cleanup unavailable")

    monkeypatch.setattr(environment, "upload_file", fake_upload_file)
    monkeypatch.setattr(environment, "_run_docker_compose_command", fake_run)
    monkeypatch.setattr(environment, "_remove_handoff", failed_cleanup)

    with pytest.raises(RuntimeError, match="could not confirm removal"):
        asyncio.run(environment.exec("true", env={"NVIDIA_API_KEY": "secret-for-test"}))


def test_secure_docker_remove_handoff_rejects_nonzero_docker_result(monkeypatch) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_docker_environment")
    environment = object.__new__(module.SkillEvaluatorSecureDockerEnvironment)

    async def failed_run(command: list[str], check: bool = True, timeout_sec: int | None = None):
        del command, check, timeout_sec
        return SimpleNamespace(stdout="", stderr="rm failed", return_code=1)

    monkeypatch.setattr(environment, "_run_docker_compose_command", failed_run)

    with pytest.raises(RuntimeError, match="Docker environment handoff removal failed"):
        asyncio.run(environment._remove_handoff("/tmp/secret-handoff"))


def test_harbor_subprocess_receives_only_file_backed_nvidia_sentinel(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import runner

    secret = "nvidia-real-secret-value-for-test"
    observed_key_files: list[Path] = []

    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        assert secret not in command
        assert secret not in environment.values()
        assert environment["NVIDIA_API_KEY"] == runner._NVIDIA_BUILD_FILE_SENTINEL
        key_file = Path(environment["SKILLEVALUATOR_NVIDIA_API_KEY_FILE"])
        observed_key_files.append(key_file)
        assert key_file.read_text(encoding="utf-8") == secret
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_validate_harbor_job_result", lambda *_args, **_kwargs: (True, ""))

    ok, detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="opencode",
        job_name="secure-build",
        env_mode="docker",
        model="nvidia/nvidia/nemotron-3-nano-30b-a3b",
        jobs_dir=tmp_path / "jobs",
        run_env={
            "SKILL_EVAL_LLM_PROVIDER": "nv_build",
            "NVIDIA_API_KEY": secret,
        },
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert (ok, detail) == (True, "")
    assert observed_key_files and all(not path.exists() for path in observed_key_files)
