# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker subprocess argv must never contain evaluator or agent credentials."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_secure_docker_exec_streams_environment_without_host_file_or_argv_values(
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

    del tmp_path
    uploaded: list[tuple[str, str]] = []
    docker_commands: list[list[str]] = []
    handoff_payloads: list[str] = []

    async def fake_upload_file(source_path: Path | str, target_path: str) -> None:
        uploaded.append((Path(source_path).read_text(encoding="utf-8"), target_path))

    async def fake_run(
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        stdin_bytes: bytes | None = None,
        redact_values: set[str] | None = None,
        stop_main_on_interrupt: bool = False,
    ):
        del check, timeout_sec
        assert stop_main_on_interrupt is ("if ! ." in " ".join(command))
        if stdin_bytes is not None:
            handoff_payloads.append(stdin_bytes.decode("utf-8"))
            assert redact_values is not None
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
    assert uploaded == []
    assert len(handoff_payloads) == 1
    assert secret in handoff_payloads[0]
    assert "persistent-secret-value" in handoff_payloads[0]
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

        async def communicate(self, _input: bytes | None = None) -> tuple[bytes, bytes]:
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


def test_secure_docker_exec_cleans_remote_handoff_when_streaming_reports_failure(
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

    async def stream_then_fail(
        _command: list[str],
        _check: bool = True,
        _timeout_sec: int | None = None,
        *,
        stdin_bytes: bytes | None = None,
        **_kwargs,
    ) -> SimpleNamespace:
        if stdin_bytes is not None:
            raise ConnectionError("docker exec disconnected after creating the file")
        return SimpleNamespace(stdout="", stderr=None, return_code=0)

    async def remove_handoff(remote_path: str) -> None:
        removed.append(remote_path)

    monkeypatch.setattr(environment, "_run_docker_compose_command", stream_then_fail)
    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)

    with pytest.raises(ConnectionError, match="docker exec disconnected"):
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

        async def blocked_stream(
            _command: list[str],
            _check: bool = True,
            _timeout_sec: int | None = None,
            *,
            stdin_bytes: bytes | None = None,
            **_kwargs,
        ) -> SimpleNamespace:
            if stdin_bytes is not None:
                upload_started.set()
                await asyncio.Event().wait()
            return SimpleNamespace(stdout="", stderr=None, return_code=0)

        async def remove_handoff(_remote_path: str) -> None:
            nonlocal cleanup_finished
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished = True

        monkeypatch.setattr(environment, "_run_docker_compose_command", blocked_stream)
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
        stdin_bytes: bytes | None = None,
        redact_values: set[str] | None = None,
        stop_main_on_interrupt: bool = False,
    ):
        del check, timeout_sec, stdin_bytes, redact_values
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


def test_harbor_subprocess_receives_nvidia_key_only_over_stdin(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import runner

    secret = "nvidia-real-secret-value-for-test"

    def reject_temporary_directory(*_args, **_kwargs):
        raise AssertionError("NVIDIA key handoff must not create a temporary directory")

    monkeypatch.setattr(runner.tempfile, "TemporaryDirectory", reject_temporary_directory)

    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        assert secret not in command
        assert secret not in environment.values()
        assert environment["NVIDIA_API_KEY"] == runner._NVIDIA_BUILD_STDIN_SENTINEL
        assert environment[runner._NVIDIA_BUILD_KEY_STDIN_ENV] == "1"
        assert runner._NVIDIA_BUILD_KEY_FILE_ENV not in environment
        assert kwargs["input"] == secret
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
    handoff = runner._nvidia_build_key_handoff(
        {"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": secret},
        env_mode="docker",
    )
    assert secret not in repr(handoff)

    other_secret = "non-nvidia-provider-secret-value-for-test"
    other_handoff = runner._nvidia_build_key_handoff(
        {"SKILL_EVAL_LLM_PROVIDER": "openai", "OPENAI_API_KEY": other_secret},
        env_mode="docker",
    )
    assert other_handoff.subprocess_env["OPENAI_API_KEY"] == other_secret
    assert other_secret not in repr(other_handoff)


def test_harbor_subprocess_redacts_stdin_key_from_early_failure(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import runner

    secret = "nvidia-real-secret-value-for-test"

    def fake_run(command, **kwargs):
        assert kwargs["input"] == secret
        assert secret not in kwargs["env"].values()
        return SimpleNamespace(returncode=1, stdout="", stderr=f"provider rejected {secret}")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    ok, detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="opencode",
        job_name="secure-build",
        env_mode="docker",
        model="nvidia/nvidia/nemotron-3-nano-30b-a3b",
        jobs_dir=tmp_path / "jobs",
        run_env={"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": secret},
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert ok is False
    assert secret not in detail
    assert "redacted" in detail.lower()


def test_stdin_handoff_is_shared_by_docker_environment_and_nvidia_agent(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import local_agents, runner, secure_docker_environment, sensitive_stdin

    secret = "nvidia-real-secret-value-for-test"
    monkeypatch.setattr(sensitive_stdin._nvidia_build_key_cache, "value", sensitive_stdin._UNSET)
    monkeypatch.setattr(sensitive_stdin.sys, "stdin", io.StringIO(secret))
    monkeypatch.setenv("NVIDIA_API_KEY", runner._NVIDIA_BUILD_STDIN_SENTINEL)
    monkeypatch.setenv(runner._NVIDIA_BUILD_KEY_STDIN_ENV, "1")

    resolved = secure_docker_environment._host_handoff_environment(
        {"NVIDIA_API_KEY": runner._NVIDIA_BUILD_STDIN_SENTINEL}
    )
    agent_key = local_agents.SkillEvaluatorNvidiaBuildCodex._resolve_bridge_api_key()

    assert resolved["NVIDIA_API_KEY"] == secret
    assert agent_key == secret
    assert sensitive_stdin.sys.stdin.read() == ""


def test_secure_docker_preflight_consumes_stdin_key_before_docker_child(monkeypatch) -> None:
    from harbor.environments.docker.docker import DockerEnvironment

    from skillevaluator.tier3.harbor import runner, secure_docker_environment, sensitive_stdin

    secret = "nvidia-real-secret-value-for-test"
    observed: list[str] = []
    monkeypatch.setattr(sensitive_stdin._nvidia_build_key_cache, "value", sensitive_stdin._UNSET)
    monkeypatch.setattr(sensitive_stdin.sys, "stdin", io.StringIO(secret))
    monkeypatch.setenv("NVIDIA_API_KEY", runner._NVIDIA_BUILD_STDIN_SENTINEL)
    monkeypatch.setenv(runner._NVIDIA_BUILD_KEY_STDIN_ENV, "1")

    def docker_preflight(_cls) -> None:
        assert sensitive_stdin._nvidia_build_key_cache.value == secret
        observed.append(secret)
        assert sensitive_stdin.sys.stdin.read() == ""

    monkeypatch.setattr(DockerEnvironment, "preflight", classmethod(docker_preflight))

    secure_docker_environment.SkillEvaluatorSecureDockerEnvironment.preflight()

    assert observed == [secret]


def test_stdin_handoff_fails_closed_without_marker(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import runner, secure_docker_environment, sensitive_stdin

    monkeypatch.setattr(sensitive_stdin._nvidia_build_key_cache, "value", sensitive_stdin._UNSET)
    monkeypatch.setattr(sensitive_stdin.sys, "stdin", io.StringIO("must-not-be-read"))
    monkeypatch.delenv(runner._NVIDIA_BUILD_KEY_STDIN_ENV, raising=False)

    with pytest.raises(RuntimeError, match="SKILLEVALUATOR_NVIDIA_API_KEY_STDIN"):
        secure_docker_environment._host_handoff_environment({"NVIDIA_API_KEY": runner._NVIDIA_BUILD_STDIN_SENTINEL})

    assert sensitive_stdin.sys.stdin.read() == "must-not-be-read"


def test_non_handoff_environment_does_not_consume_stdin(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment, sensitive_stdin

    monkeypatch.setattr(sensitive_stdin._nvidia_build_key_cache, "value", sensitive_stdin._UNSET)
    monkeypatch.setattr(sensitive_stdin.sys, "stdin", io.StringIO("unrelated-input"))

    resolved = secure_docker_environment._host_handoff_environment({"NVIDIA_API_KEY": "ordinary-key"})

    assert resolved == {"NVIDIA_API_KEY": "ordinary-key"}
    assert sensitive_stdin.sys.stdin.read() == "unrelated-input"


def test_stdin_handoff_is_read_once_under_concurrency(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import runner, sensitive_stdin

    class CountingInput(io.StringIO):
        reads = 0

        def read(self, *args, **kwargs):
            self.reads += 1
            return super().read(*args, **kwargs)

    secret = "nvidia-real-secret-value-for-test"
    source = CountingInput(secret)
    monkeypatch.setattr(sensitive_stdin._nvidia_build_key_cache, "value", sensitive_stdin._UNSET)
    monkeypatch.setattr(sensitive_stdin.sys, "stdin", source)
    monkeypatch.setenv(runner._NVIDIA_BUILD_KEY_STDIN_ENV, "1")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: sensitive_stdin.read_nvidia_build_key_from_stdin(), range(32)))

    assert results == [secret] * 32
    assert source.reads == 1


def test_real_child_process_receives_nvidia_key_only_through_stdin() -> None:
    from skillevaluator.tier3.harbor import runner

    secret = "nvidia-real-secret-value-for-test"
    expected_digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    child = (
        "import hashlib; "
        "from skillevaluator.tier3.harbor.sensitive_stdin import read_nvidia_build_key_from_stdin; "
        "print(hashlib.sha256(read_nvidia_build_key_from_stdin().encode('utf-8')).hexdigest())"
    )
    environment = dict(os.environ)
    environment["NVIDIA_API_KEY"] = runner._NVIDIA_BUILD_STDIN_SENTINEL
    environment[runner._NVIDIA_BUILD_KEY_STDIN_ENV] = "1"
    environment.pop(runner._NVIDIA_BUILD_KEY_FILE_ENV, None)

    completed = subprocess.run(
        [sys.executable, "-c", child],
        input=secret,
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )

    assert completed.stdout.strip() == expected_digest
    assert completed.stderr == ""
    assert secret not in child
    assert secret not in environment.values()
    assert secret not in completed.stdout


def test_empty_stdin_handoff_fails_before_docker_preflight(monkeypatch) -> None:
    from harbor.environments.docker.docker import DockerEnvironment

    from skillevaluator.tier3.harbor import runner, secure_docker_environment, sensitive_stdin

    docker_preflight_called = False
    monkeypatch.setattr(sensitive_stdin._nvidia_build_key_cache, "value", sensitive_stdin._UNSET)
    monkeypatch.setattr(sensitive_stdin.sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("NVIDIA_API_KEY", runner._NVIDIA_BUILD_STDIN_SENTINEL)
    monkeypatch.setenv(runner._NVIDIA_BUILD_KEY_STDIN_ENV, "1")

    def docker_preflight(_cls) -> None:
        nonlocal docker_preflight_called
        docker_preflight_called = True

    monkeypatch.setattr(DockerEnvironment, "preflight", classmethod(docker_preflight))

    with pytest.raises(RuntimeError, match="stdin handoff is empty"):
        secure_docker_environment.SkillEvaluatorSecureDockerEnvironment.preflight()

    assert docker_preflight_called is False
