# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import io
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import tracemalloc
import uuid
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from harbor.environments.base import MAIN_SERVICE_NAME, ExecResult, ServiceOperationsUnsupportedError
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from skillevaluator.tier3.harbor import stream_redaction as stream_redaction_module
from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
from skillevaluator.tier3.harbor.runner import build_harbor_run_command
from skillevaluator.tier3.harbor.secure_docker_environment import (
    _REDACTION_SENTINEL_CANDIDATES,
    NVIDIA_BUILD_STDIN_SENTINEL,
    SECURE_DOCKER_ENV_IMPORT_PATH,
    SkillEvaluatorDockerEnvironment,
    SkillEvaluatorSecureDockerEnvironment,
    _collision_safe_redaction_marker,
    _compose_client_credential_values,
    _compose_interpolation_names,
    _redact,
    _secure_exec_arguments,
    _sidecar_environment_carriers,
    _signal_process_tree,
    _StreamingSecretRedactor,
)
from skillevaluator.tier3.harbor.sensitive_stdin import NVIDIA_BUILD_KEY_STDIN_ENV
from skillevaluator.tier3.harbor.stream_redaction import CommandOutputLimitError

_SENTINEL = "sentinel-never-visible-in-argv-or-files"


def _marker_for(*secrets: str) -> str:
    return _collision_safe_redaction_marker(secrets)


def _initialized_docker_environment(
    tmp_path: Path,
    *,
    environment_name: str = "secure-compose-test",
    persistent_env: dict[str, str] | None = None,
) -> SkillEvaluatorDockerEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    return SkillEvaluatorDockerEnvironment(
        environment_dir=environment_dir,
        environment_name=environment_name,
        session_id="secure-compose-test",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        persistent_env=persistent_env,
    )


def _initialized_secure_docker_environment(
    tmp_path: Path,
    *,
    persistent_env: dict[str, str] | None = None,
) -> SkillEvaluatorSecureDockerEnvironment:
    environment_dir = tmp_path / "secure-environment"
    environment_dir.mkdir(exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    return SkillEvaluatorSecureDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="secure-compose-public-exec-test",
        session_id="secure-compose-public-exec-test",
        trial_paths=TrialPaths(tmp_path / "secure-trial"),
        task_env_config=EnvironmentConfig(),
        persistent_env=persistent_env,
    )


class _BufferedComposeProcess:
    pid = 8841

    def __init__(
        self,
        *,
        stdout: bytes = b"buffered output",
        stderr: bytes | None = None,
        return_code: int = 0,
    ) -> None:
        self.returncode: int | None = return_code
        self._stdout = stdout
        self._stderr = stderr
        self.stdout = _ChunkStream([stdout] if stdout else [])
        self.stdin = _WritableStdin()
        self.communicate_inputs: list[bytes | None] = []
        self.wait_count = 0

    async def communicate(self, **kwargs: bytes | None) -> tuple[bytes, bytes | None]:
        assert set(kwargs) <= {"input"}
        self.communicate_inputs.append(kwargs.get("input"))
        return self._stdout, self._stderr

    async def wait(self) -> int:
        self.wait_count += 1
        return self.returncode or 0

    def terminate(self) -> None:
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.returncode = -signal.SIGKILL


class _WritableStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _ChunkStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _ChunkStream:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def read(self, _limit: int) -> bytes:
        try:
            return await self.__anext__()
        except StopAsyncIteration:
            return b""


class _FeedableStream:
    def __init__(self) -> None:
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()

    def feed_data(self, data: bytes) -> None:
        self._chunks.put_nowait(data)

    def feed_eof(self) -> None:
        self._chunks.put_nowait(None)

    async def read(self, _limit: int) -> bytes:
        chunk = await self._chunks.get()
        return b"" if chunk is None else chunk


class _ExitAwaitingStream:
    def __init__(self, process: _HangingComposeProcess) -> None:
        self._process = process

    async def read(self, _limit: int) -> bytes:
        self._process.started.set()
        await asyncio.shield(self._process.completion())
        return b""


class _HangingComposeProcess:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.started = asyncio.Event()
        self.stdin = _WritableStdin()
        self.stdout = _ExitAwaitingStream(self)
        self._completion: asyncio.Future[int] | None = None

    def completion(self) -> asyncio.Future[int]:
        if self._completion is None:
            self._completion = asyncio.get_running_loop().create_future()
        return self._completion

    def finish(self, return_code: int) -> None:
        self.returncode = return_code
        completion = self.completion()
        if not completion.done():
            completion.set_result(return_code)

    async def wait(self) -> int:
        return await asyncio.shield(self.completion())

    async def communicate(self, **_kwargs: bytes | None) -> tuple[bytes, None]:
        self.started.set()
        await asyncio.shield(self.completion())
        return b"", None

    def terminate(self) -> None:
        self.finish(-signal.SIGTERM)

    def kill(self) -> None:
        self.finish(-signal.SIGKILL)


class _StreamedComposeProcess:
    pid = 8842

    def __init__(self, chunks: list[bytes], *, return_code: int = 0) -> None:
        self.returncode: int | None = None
        self._exit_code = return_code
        self.stdout = _ChunkStream(chunks)
        self.stdin = _WritableStdin()
        self.wait_count = 0
        self.terminate_count = 0
        self.kill_count = 0

    async def wait(self) -> int:
        self.wait_count += 1
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -signal.SIGKILL


class _BufferedAndStreamedComposeProcess(_StreamedComposeProcess):
    def __init__(self, chunks: list[bytes], *, return_code: int = 0) -> None:
        super().__init__(chunks, return_code=return_code)
        self._buffered_stdout = b"".join(chunks)
        self.communicate_inputs: list[bytes | None] = []

    async def communicate(self, **kwargs: bytes | None) -> tuple[bytes, None]:
        assert set(kwargs) <= {"input"}
        self.communicate_inputs.append(kwargs.get("input"))
        self.returncode = self._exit_code
        return self._buffered_stdout, None


class _CallbackBaseError(BaseException):
    pass


def test_docker_streaming_accepts_newline_free_output_larger_than_reader_limit(
    tmp_path: Path,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    output_size = 200_000
    callbacks: list[str] = []

    async def on_output(text: str, stream: str) -> None:
        assert stream == "stdout"
        callbacks.append(text)

    async def exercise() -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            f"import sys; sys.stdout.buffer.write(b'x' * {output_size}); sys.stdout.buffer.flush()",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            return await environment._collect_streamed_output(
                process,
                timeout_sec=5,
                on_output=on_output,
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    result = asyncio.run(exercise())

    assert result.return_code == 0
    assert result.stdout == "x" * output_size
    assert "".join(callbacks) == result.stdout
    assert all(len(chunk.encode()) <= 64 * 1024 for chunk in callbacks)


@pytest.mark.parametrize("with_callback", [False, True])
def test_compose_output_limit_fails_closed_and_contains_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_callback: bool,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    overflow_value = b"synthetic-overflow-value"
    process = _BufferedAndStreamedComposeProcess([b"12345678", overflow_value])
    callback_chunks: list[str] = []
    containment_calls: list[asyncio.subprocess.Process] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedAndStreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    async def contain(
        contained_process: asyncio.subprocess.Process,
        communication: asyncio.Task[object],
        **_kwargs: object,
    ) -> None:
        containment_calls.append(contained_process)
        with contextlib.suppress(BaseException):
            await communication

    monkeypatch.setattr(stream_redaction_module, "MAX_COMMAND_OUTPUT_BYTES", 8, raising=False)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(environment, "_contain_main_and_reap_compose", contain)

    callback = on_output if with_callback else None
    with pytest.raises(CommandOutputLimitError, match=r"Command output exceeded the 8-byte safety limit") as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["version"],
                check=False,
                on_output=callback,
            )
        )

    assert containment_calls == [process]
    assert process.returncode is not None
    assert len("".join(callback_chunks).encode()) <= 8
    assert overflow_value.decode() not in "".join(callback_chunks)
    assert overflow_value.decode() not in str(caught.value)


def test_compose_bounded_path_uses_devnull_without_stdin_or_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _BufferedComposeProcess()
    captured: dict[str, object] = {}

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        captured.update(kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(environment._run_docker_compose_command(["version"], check=False))

    assert captured["stdin"] == asyncio.subprocess.DEVNULL
    assert process.communicate_inputs == []
    assert result == ExecResult(stdout="buffered output", stderr=None, return_code=0)


@pytest.mark.parametrize("stdin_data", [b"", b"\x00tar\xffpayload\nwith spaces\x00"])
def test_compose_stdin_reaches_bounded_subprocess_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdin_data: bytes,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _BufferedComposeProcess()
    captured: dict[str, object] = {}

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        captured.update(kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(
        environment._run_docker_compose_command(
            ["exec", "-T", "main", "tar", "-xf", "-"],
            check=False,
            stdin_data=stdin_data,
        )
    )

    assert captured["stdin"] == asyncio.subprocess.PIPE
    assert process.communicate_inputs == []
    assert bytes(process.stdin.data) == stdin_data
    assert result.return_code == 0


def test_compose_stream_callback_writes_and_closes_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([b"archive received\n"])
    captured: dict[str, object] = {}
    callback_chunks: list[str] = []
    stdin_data = b"\x00streamed\xffarchive\n"

    async def create_subprocess(*_args: object, **kwargs: object) -> _StreamedComposeProcess:
        captured.update(kwargs)
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(
        environment._run_docker_compose_command(
            ["exec", "-T", "main", "tar", "-xf", "-"],
            check=False,
            stdin_data=stdin_data,
            on_output=on_output,
        )
    )

    assert captured["stdin"] == asyncio.subprocess.PIPE
    assert bytes(process.stdin.data) == stdin_data
    assert process.stdin.closed is True
    assert callback_chunks == ["archive received\n"]
    assert result.stdout == "archive received\n"


def test_compose_stream_callback_receives_merged_complete_redacted_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_secret = "persistent-secret-value"
    scoped_secret = "scoped-secret-value"
    per_call_secret = "per-call-secret-value"
    environment = _initialized_docker_environment(
        tmp_path,
        persistent_env={"PERSISTENT_TOKEN": persistent_secret},
    )
    process = _StreamedComposeProcess(
        [
            f"stdout {persistent_secret}\n".encode(),
            f"stderr {scoped_secret}\n".encode(),
            f"tail {per_call_secret}\n".encode(),
        ]
    )
    captured: dict[str, object] = {}
    callbacks: list[tuple[str, str]] = []

    async def create_subprocess(*_args: object, **kwargs: object) -> _StreamedComposeProcess:
        captured.update(kwargs)
        return process

    async def on_output(text: str, stream: str) -> None:
        callbacks.append((text, stream))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_exec_env({"SCOPED_TOKEN": scoped_secret}):
            merged_environment = environment._merge_env({"PER_CALL_TOKEN": per_call_secret})
            assert merged_environment is not None
            return await environment._run_docker_compose_command(
                [
                    "exec",
                    "-e",
                    "PERSISTENT_TOKEN",
                    "-e",
                    "SCOPED_TOKEN",
                    "-e",
                    "PER_CALL_TOKEN",
                    "main",
                    "emit-output",
                ],
                check=False,
                on_output=on_output,
                env_overrides=merged_environment,
            )

    result = asyncio.run(exercise())

    assert captured["stderr"] == asyncio.subprocess.STDOUT
    child_environment = captured["env"]
    assert isinstance(child_environment, dict)
    assert child_environment["PERSISTENT_TOKEN"] == persistent_secret
    assert child_environment["SCOPED_TOKEN"] == scoped_secret
    assert child_environment["PER_CALL_TOKEN"] == per_call_secret
    assert {stream for _text, stream in callbacks} == {"stdout"}
    assert "".join(text for text, _stream in callbacks) == result.stdout
    marker = _marker_for(persistent_secret, scoped_secret, per_call_secret)
    assert result.stdout == f"stdout {marker}\nstderr {marker}\ntail {marker}\n"
    assert result.stderr is None
    assert result.return_code == 0


@pytest.mark.parametrize(
    ("environment_name", "credential_value"),
    [
        ("DOCKER_HOST", "tcp://compose-user:compose-password@docker.invalid:2376"),
        ("HTTPS_PROXY", "https://proxy-user:proxy-password@proxy.invalid:8443"),
        ("HTTPS_PROXY", "proxy-user:proxy-password@proxy.invalid:8443"),
        (
            "DOCKER_AUTH_CONFIG",
            '{"auths":{"registry.invalid":{"auth":"compose-registry-credential"}}}',
        ),
    ],
)
def test_compose_redacts_host_client_credentials_from_callback_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    credential_value: str,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([f"client failure: {credential_value}\n".encode()])
    callbacks: list[str] = []
    captured_environment: dict[str, str] = {}

    async def create_subprocess(*_args: object, **kwargs: object) -> _StreamedComposeProcess:
        captured_environment.update(kwargs["env"])
        return process

    async def on_output(text: str, _stream: str) -> None:
        callbacks.append(text)

    monkeypatch.setenv(environment_name, credential_value)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(
        environment._run_docker_compose_command(
            ["version"],
            check=False,
            on_output=on_output,
        )
    )

    callback_output = "".join(callbacks)
    assert captured_environment[environment_name] == credential_value
    assert callback_output == result.stdout
    assert credential_value not in callback_output
    assert credential_value not in (result.stdout or "")
    compose_credential_values = _compose_client_credential_values({environment_name: credential_value})
    assert _marker_for(*compose_credential_values) in callback_output


@pytest.mark.parametrize(
    ("proxy_uri", "diagnostic", "secrets"),
    [
        (
            "https://proxy-user:proxy-password@proxy.invalid:8443",
            "proxy auth failed for proxy-user with proxy-password\n",
            ("proxy-user", "proxy-password"),
        ),
        (
            "https://proxy%2Duser:proxy%2Dpassword@proxy.invalid:8443",
            "proxy auth failed for proxy-user with proxy-password\n",
            ("proxy-user", "proxy-password"),
        ),
        (
            "proxy-user:proxy-password@proxy.invalid:8443",
            "proxy auth failed for proxy-user with proxy-password\n",
            ("proxy-user", "proxy-password"),
        ),
    ],
)
def test_compose_redacts_host_proxy_credential_fragments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proxy_uri: str,
    diagnostic: str,
    secrets: tuple[str, ...],
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    split = max(1, len(diagnostic) // 2)
    process = _StreamedComposeProcess([diagnostic[:split].encode(), diagnostic[split:].encode()])

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    monkeypatch.setenv("HTTPS_PROXY", proxy_uri)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def on_output(_text: str, _stream: str) -> None:
        return None

    result = asyncio.run(environment._run_docker_compose_command(["version"], check=False, on_output=on_output))

    assert all(secret not in (result.stdout or "") for secret in secrets)


@pytest.mark.parametrize(
    ("environment_name", "credential_value"),
    [
        ("DOCKER_HOST", "tcp://compose-user:compose-password@docker.invalid:2376"),
        ("HTTPS_PROXY", "https://proxy-user:proxy-password@proxy.invalid:8443"),
        ("HTTPS_PROXY", "proxy-user:proxy-password@proxy.invalid:8443"),
        (
            "DOCKER_AUTH_CONFIG",
            '{"auths":{"registry.invalid":{"auth":"compose-registry-credential"}}}',
        ),
    ],
)
def test_compose_redacts_host_client_credentials_from_checked_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    credential_value: str,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _BufferedComposeProcess(
        stdout=f"client failure: {credential_value}\n".encode(),
        return_code=7,
    )

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        return process

    monkeypatch.setenv(environment_name, credential_value)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(environment._run_docker_compose_command(["version"], check=True))

    detail = str(caught.value)
    assert credential_value not in detail
    compose_credential_values = _compose_client_credential_values({environment_name: credential_value})
    assert _marker_for(*compose_credential_values) in detail


def test_compose_stream_callback_redacts_multiline_and_overlapping_secrets_across_reader_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multiline_secret = "FIRST-HALF\nSECOND-HALF"
    shorter_secret = "overlap-secret"
    longer_secret = "overlap-secret-tail"
    nested_shorter_secret = "NESTED-SECRET\n"
    nested_longer_secret = "NESTED-SECRET\nTAIL"
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([])
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        stdout = asyncio.StreamReader()
        stdout.feed_data(b"prefix FIRST-")
        stdout.feed_data(b"HALF\nSECOND-")
        stdout.feed_data(b"HALF overlap-secret-tail overlap-")
        stdout.feed_data(b"secret NESTED-SECRET\n")
        stdout.feed_data(b"TAIL suffix\n")
        stdout.feed_eof()
        process.stdout = stdout
        return await environment._run_docker_compose_command(
            ["exec", "main", "emit-output"],
            check=False,
            on_output=on_output,
            env_overrides={
                "MULTILINE_SECRET": multiline_secret,
                "SHORTER_SECRET": shorter_secret,
                "LONGER_SECRET": longer_secret,
                "NESTED_SHORTER_SECRET": nested_shorter_secret,
                "NESTED_LONGER_SECRET": nested_longer_secret,
            },
        )

    result = asyncio.run(exercise())
    callback_output = "".join(callback_chunks)
    marker = _marker_for(
        multiline_secret,
        shorter_secret,
        longer_secret,
        nested_shorter_secret,
        nested_longer_secret,
    )

    assert callback_output == f"prefix {marker} {marker} {marker} {marker} suffix\n"
    assert "FIRST-HALF" not in callback_output
    assert "SECOND-HALF" not in callback_output
    assert shorter_secret not in callback_output
    assert longer_secret not in callback_output
    assert "NESTED-SECRET" not in callback_output
    assert "TAIL" not in callback_output
    assert result.stdout == callback_output


def test_compose_stream_callback_flushes_incomplete_secret_prefix_at_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([b"ordinary output secret-"])
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(
        environment._run_docker_compose_command(
            ["exec", "main", "emit-output"],
            check=False,
            on_output=on_output,
            env_overrides={"SECRET": "secret-value"},
        )
    )

    assert "".join(callback_chunks) == result.stdout == "ordinary output secret-"


def test_streaming_and_buffered_redaction_union_is_deterministic_for_offset_overlaps() -> None:
    script = """
import json
from skillevaluator.tier3.harbor.secure_docker_environment import _StreamingSecretRedactor, _redact

secrets = {"abcdefgh", "ghijklmn"}
text = "abcdefghijklmn\\n"
redactor = _StreamingSecretRedactor(secrets)
streamed = redactor.feed(text) + redactor.finish()
print(json.dumps({"streamed": streamed, "buffered": _redact(text, secrets)}))
"""
    outputs: list[dict[str, str]] = []
    for seed in ("1", "2", "3", "4", "5", "6", "7", "8"):
        child_env = dict(os.environ)
        child_env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        outputs.append(json.loads(completed.stdout))

    expected = _marker_for("abcdefgh", "ghijklmn") + "\n"
    assert outputs == [{"streamed": expected, "buffered": expected}] * len(outputs)


def test_compose_stream_callback_and_result_match_for_offset_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([b"abcdefghijklmn\n"])
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(
        environment._run_docker_compose_command(
            ["exec", "main", "emit-output"],
            check=False,
            on_output=on_output,
            env_overrides={"LEFT_SECRET": "abcdefgh", "RIGHT_SECRET": "ghijklmn"},
        )
    )

    expected = _marker_for("abcdefgh", "ghijklmn") + "\n"
    assert "".join(callback_chunks) == result.stdout == expected


def test_compose_stream_callback_and_error_match_for_offset_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([b"abcdefghijklmn\n"], return_code=7)
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "main", "emit-output"],
                on_output=on_output,
                env_overrides={"LEFT_SECRET": "abcdefgh", "RIGHT_SECRET": "ghijklmn"},
            )
        )

    expected = _marker_for("abcdefgh", "ghijklmn") + "\n"
    assert "".join(callback_chunks) == expected
    assert f"Stdout: {expected}." in str(caught.value)
    assert f"abcdef{_marker_for('abcdefgh', 'ghijklmn')}" not in str(caught.value)


@pytest.mark.parametrize(
    ("secret_count", "output_size"),
    [(100, 100_000), (500, 20_000)],
)
def test_streaming_redactor_scale_is_linear_in_input(
    secret_count: int,
    output_size: int,
) -> None:
    secrets = {f"secret-{index:04d}-value" for index in range(secret_count)}
    selected_secret = f"secret-{secret_count - 1:04d}-value"
    prefix_size = output_size // 2
    text = "x" * prefix_size + selected_secret + "y" * (output_size - prefix_size)
    redactor = _StreamingSecretRedactor(secrets, _track_transitions=True)

    emitted: list[str] = []
    longest_secret = max(map(len, secrets))
    for index in range(0, len(text), 4093):
        emitted.append(redactor.feed(text[index : index + 4093]))
        assert len(redactor._pending) <= longest_secret - 1
    emitted.append(redactor.finish())
    streamed = "".join(emitted)

    assert streamed == _redact(text, secrets)
    assert selected_secret not in streamed
    assert redactor.match_transition_count <= 2 * len(text)
    assert redactor.match_work_count <= 10 * len(text)


@pytest.mark.parametrize(
    ("secret_length", "output_size"),
    [(1024, 20_000), (4096, 5_000)],
)
def test_streaming_redactor_repeated_prefix_scan_is_linear(
    secret_length: int,
    output_size: int,
) -> None:
    secret = "a" * (secret_length - 1) + "b"
    text = "a" * output_size
    redactor = _StreamingSecretRedactor({secret}, _track_transitions=True)

    assert redactor.feed(text) + redactor.finish() == text
    assert redactor.match_transition_count <= 2 * len(text)
    assert redactor.match_work_count <= 10 * len(text)


def test_streaming_redactor_does_not_starve_event_loop_on_repeated_prefix() -> None:
    async def exercise() -> float:
        redactor = _StreamingSecretRedactor(
            {"a" * 1023 + "b"},
            _track_transitions=True,
        )
        text = "a" * 20_000
        loop = asyncio.get_running_loop()
        heartbeat_gaps: list[float] = []
        keep_running = True

        async def heartbeat() -> None:
            previous = loop.time()
            while keep_running:
                await asyncio.sleep(0.01)
                now = loop.time()
                heartbeat_gaps.append(now - previous)
                previous = now

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.02)
        emitted = redactor.feed(text)
        await asyncio.sleep(0.02)
        keep_running = False
        await heartbeat_task
        assert emitted + redactor.finish() == text
        return max(heartbeat_gaps)

    assert asyncio.run(exercise()) < 1.0


def test_streaming_redactor_nested_terminal_matching_work_is_linear() -> None:
    secrets = {"a" * length for length in range(8, 1008)}
    text = "a" * 20_000
    started = time.perf_counter()
    redactor = _StreamingSecretRedactor(secrets, _track_transitions=True)

    output = redactor.feed(text) + redactor.finish()
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0
    assert output == _collision_safe_redaction_marker(secrets)
    assert redactor.match_work_count <= 10 * len(text)


def test_streaming_redactor_does_not_starve_event_loop_on_nested_terminals() -> None:
    async def exercise() -> float:
        redactor = _StreamingSecretRedactor(
            {"a" * length for length in range(8, 1008)},
            _track_transitions=True,
        )
        text = "a" * 20_000
        loop = asyncio.get_running_loop()
        heartbeat_gaps: list[float] = []
        keep_running = True

        async def heartbeat() -> None:
            previous = loop.time()
            while keep_running:
                await asyncio.sleep(0.01)
                now = loop.time()
                heartbeat_gaps.append(now - previous)
                previous = now

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.02)
        emitted = redactor.feed(text)
        await asyncio.sleep(0.02)
        keep_running = False
        await heartbeat_task
        assert emitted + redactor.finish() == _collision_safe_redaction_marker(
            {"a" * length for length in range(8, 1008)}
        )
        return max(heartbeat_gaps)

    assert asyncio.run(exercise()) < 1.0


def test_compose_stream_nested_secret_union_does_not_starve_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([b"a" * 20_000])
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> tuple[ExecResult, float]:
        loop = asyncio.get_running_loop()
        heartbeat_gaps: list[float] = []
        keep_running = True

        async def heartbeat() -> None:
            previous = loop.time()
            while keep_running:
                await asyncio.sleep(0.01)
                now = loop.time()
                heartbeat_gaps.append(now - previous)
                previous = now

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.02)
        result = await environment._run_docker_compose_command(
            ["exec", "main", "emit-output"],
            check=False,
            on_output=on_output,
            env_overrides={f"SECRET_{length}": "a" * length for length in range(8, 1008)},
        )
        await asyncio.sleep(0.02)
        keep_running = False
        await heartbeat_task
        return result, max(heartbeat_gaps)

    result, maximum_gap = asyncio.run(exercise())

    expected = _collision_safe_redaction_marker({"a" * length for length in range(8, 1008)})
    assert "".join(callback_chunks) == result.stdout == expected
    assert maximum_gap < 1.0


def _reference_union_redaction(text: str, secrets: set[str]) -> str:
    eligible = {secret for secret in secrets if secret and len(secret) >= 8}
    marker = _collision_safe_redaction_marker(eligible)
    coverage = [0] * (len(text) + 1)
    for secret in eligible:
        match_start = text.find(secret)
        while match_start >= 0:
            coverage[match_start] += 1
            coverage[match_start + len(secret)] -= 1
            match_start = text.find(secret, match_start + 1)

    redacted: list[str] = []
    active_matches = 0
    redaction_open = False
    for position, character in enumerate(text):
        active_matches += coverage[position]
        if active_matches:
            if not redaction_open:
                redacted.append(marker)
                redaction_open = True
        else:
            redaction_open = False
            redacted.append(character)
    return "".join(redacted)


@pytest.mark.parametrize(
    ("secrets", "text", "expected"),
    [
        ({"abcdefgh", "abcdefghij"}, "abcdefghij", "[REDACTED]"),
        ({"abcdefgh", "bcdefghijk"}, "abcdefghijk", "[REDACTED]"),
        ({"abcdefgh", "ghijklmn"}, "abcdefghijklmn", "[REDACTED]"),
        ({"aaaaaaaa", "aaaaaaaaaa"}, "a" * 12, "[REDACTED]"),
        ({"xxabcdefgh", "abcdefgh"}, "xxabcdefgh", "[REDACTED]"),
        ({"abcdefgh", "ijklmnop"}, "abcdefghijklmnop", "[REDACTED]"),
        ({"abcdefgh", "jklmnopq"}, "abcdefghXjklmnopq", "[REDACTED]X[REDACTED]"),
        (
            {"abcdefgh", "klmnopqr", "ghijklmnopqrst"},
            "abcdefghijklmnopqrst",
            "[REDACTED]",
        ),
        (
            {"unicode-🔑alpha\nβ", "🔑alpha\nβ"},
            "prefix unicode-🔑alpha\nβ suffix",
            "prefix [REDACTED] suffix",
        ),
    ],
)
def test_streaming_redactor_matches_union_reference_for_every_single_split(
    secrets: set[str],
    text: str,
    expected: str,
) -> None:
    marker = _collision_safe_redaction_marker(secrets)
    expected = expected.replace("[REDACTED]", marker)
    assert _reference_union_redaction(text, secrets) == expected
    longest_secret = max(map(len, secrets))

    for split in range(len(text) + 1):
        redactor = _StreamingSecretRedactor(secrets)
        streamed = redactor.feed(text[:split])
        assert len(redactor._pending) <= longest_secret - 1
        streamed += redactor.feed(text[split:])
        assert len(redactor._pending) <= longest_secret - 1
        streamed += redactor.finish()
        assert streamed == expected
        assert _redact(text, secrets) == expected


def test_streaming_redactor_matches_union_reference_for_randomized_chunks() -> None:
    randomizer = random.Random(0xAC022)
    alphabet = "abXYé🙂\n"

    for _case in range(100):
        secrets = {
            "".join(randomizer.choice(alphabet) for _character in range(randomizer.randint(8, 24)))
            for _secret in range(randomizer.randint(1, 20))
        }
        # Include nested and shared-prefix patterns on every run so overlapping
        # coverage and automaton failure behavior are exercised independently of chance.
        secrets.update({"aaaaaaaa", "aaaaaaaaaa", "abcdefgh", "abcdefghij"})
        parts = [
            "".join(randomizer.choice(alphabet) for _character in range(randomizer.randint(0, 30)))
            for _part in range(randomizer.randint(2, 8))
        ]
        selected = randomizer.sample(sorted(secrets), k=min(len(parts) - 1, len(secrets)))
        text = "".join(part + (selected[index] if index < len(selected) else "") for index, part in enumerate(parts))
        expected = _reference_union_redaction(text, secrets)
        longest_secret = max(map(len, secrets))
        redactor = _StreamingSecretRedactor(secrets, _track_transitions=True)
        emitted: list[str] = []
        position = 0
        while position < len(text):
            chunk_size = randomizer.randint(1, 19)
            emitted.append(redactor.feed(text[position : position + chunk_size]))
            position += chunk_size
            assert len(redactor._pending) <= longest_secret - 1
        emitted.append(redactor.finish())

        streamed = "".join(emitted)
        assert streamed == expected
        assert _redact(text, secrets) == expected
        for secret in secrets:
            assert secret not in streamed
        assert redactor.match_transition_count <= 2 * len(text)
        assert redactor.match_work_count <= 10 * len(text)


def test_collision_safe_marker_invariant_with_unicode_and_occupied_candidates() -> None:
    occupied_candidates = "".join(_REDACTION_SENTINEL_CANDIDATES)
    secrets = {
        f"{occupied_candidates}unicode-🔑alpha\nβ",
        "abcdefgh",
        "x[REDACTED]",
    }

    marker = _collision_safe_redaction_marker(secrets)
    sentinel = marker[0]
    minimum_secret_length = min(map(len, secrets))

    assert marker[-1] == sentinel
    assert all(sentinel not in secret for secret in secrets)
    assert all(secret not in marker for secret in secrets)
    for start in range(len(marker) - minimum_secret_length + 1):
        assert sentinel in marker[start : start + minimum_secret_length]
    for secret in secrets:
        for split in range(1, len(secret)):
            assert secret not in f"{secret[:split]}{marker}{secret[split:]}"


def test_collision_safe_marker_falls_back_to_an_absent_unicode_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    monkeypatch.setattr(
        secure_docker_environment,
        "_REDACTION_SENTINEL_CANDIDATES",
        (),
    )
    secret = "abcdefgh\ue000"

    marker = secure_docker_environment._collision_safe_redaction_marker({secret})

    assert marker[0] == "\ue001"
    assert marker[-1] == "\ue001"
    assert secret not in marker


def test_collision_safe_marker_fallback_never_emits_terminal_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    monkeypatch.setattr(
        secure_docker_environment,
        "_REDACTION_SENTINEL_CANDIDATES",
        (),
    )
    occupied_private_use_and_controls = "".join(
        chr(codepoint)
        for candidate_range in (
            range(0xE000, 0xF900),
            range(0xF0000, 0xFFFFE),
            range(0x100000, 0x10FFFE),
            range(1, 0x1B),
        )
        for codepoint in candidate_range
    )

    marker = secure_docker_environment._collision_safe_redaction_marker({occupied_private_use_and_controls})

    assert marker[0].isprintable()
    assert not marker[0].isspace()
    assert marker[-1] == marker[0]


def test_collision_safe_marker_exhaustion_fails_before_process_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    monkeypatch.setattr(
        secure_docker_environment,
        "_REDACTION_SENTINEL_CANDIDATES",
        (),
    )
    monkeypatch.setattr(
        secure_docker_environment,
        "unicodedata",
        SimpleNamespace(category=lambda _candidate: "Cc"),
    )
    occupied_private_use = "".join(
        chr(codepoint)
        for candidate_range in (
            range(0xE000, 0xF900),
            range(0xF0000, 0xFFFFE),
            range(0x100000, 0x10FFFE),
        )
        for codepoint in candidate_range
    )
    environment = _initialized_docker_environment(tmp_path)
    subprocess_created = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        nonlocal subprocess_created
        subprocess_created = True
        return _BufferedComposeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="Could not construct a collision-safe redaction marker") as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "main", "true"],
                env_overrides={"SECRET": occupied_private_use},
            )
        )

    assert not subprocess_created
    assert occupied_private_use not in str(caught.value)


@pytest.mark.parametrize("check", [True, False])
def test_compose_stream_callback_nonzero_check_semantics_are_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check: bool,
) -> None:
    secret = "nonzero-secret-value"
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([f"failure {secret}\n".encode()], return_code=7)
    callbacks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callbacks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    run = environment._run_docker_compose_command(
        ["exec", "main", "fail"],
        check=check,
        on_output=on_output,
        env_overrides={"SECRET": secret},
    )

    if check:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(run)
        assert secret not in str(caught.value)
        assert _marker_for(secret) in str(caught.value)
    else:
        result = asyncio.run(run)
        assert result.return_code == 7
        assert result.stdout == f"failure {_marker_for(secret)}\n"
    assert "".join(callbacks) == f"failure {_marker_for(secret)}\n"


def test_compose_stream_check_failure_redacts_replacement_token_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "REDACTED"
    environment = _initialized_docker_environment(
        tmp_path,
        environment_name=f"env-{secret}",
    )
    process = _StreamedComposeProcess([f"failure {secret}\n".encode()], return_code=7)
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "main", "fail", secret],
                on_output=on_output,
                env_overrides={"SECRET": secret},
            )
        )

    detail = str(caught.value)
    marker = _marker_for(secret)
    assert "".join(callback_chunks) == f"failure {marker}\n"
    assert secret not in "".join(callback_chunks)
    assert secret not in detail
    assert f"environment env-{marker}" in detail
    assert f"fail {marker}" in detail
    assert f"Stdout: failure {marker}\n." in detail
    assert "env-REDACTED" not in detail
    assert "fail REDACTED" not in detail
    assert "failure REDACTED" not in detail
    assert detail.count(marker) == 3


@pytest.mark.parametrize(
    ("secrets", "raw_output"),
    [
        (("REDACTED", "[REDACTED]"), "REDACTED [REDACTED]\n"),
        (("12345678", "x[REDACTED]"), "x12345678\n"),
    ],
)
def test_compose_stream_redaction_marker_cannot_disclose_or_synthesize_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secrets: tuple[str, ...],
    raw_output: str,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    callbacks: list[list[str]] = [[], []]
    callback_index = 0

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return _StreamedComposeProcess([raw_output.encode()], return_code=7)

    async def on_output(text: str, _stream: str) -> None:
        callbacks[callback_index].append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    environment_overrides = {f"SECRET_{index}": secret for index, secret in enumerate(secrets)}

    async def exercise() -> tuple[ExecResult, RuntimeError]:
        nonlocal callback_index
        result = await environment._run_docker_compose_command(
            ["exec", "main", "fail"],
            check=False,
            on_output=on_output,
            env_overrides=environment_overrides,
        )
        callback_index = 1
        with pytest.raises(RuntimeError) as caught:
            await environment._run_docker_compose_command(
                ["exec", "main", "fail"],
                on_output=on_output,
                env_overrides=environment_overrides,
            )
        return result, caught.value

    result, error = asyncio.run(exercise())
    check_false_callback = "".join(callbacks[0])
    check_true_callback = "".join(callbacks[1])

    assert check_false_callback == result.stdout
    assert check_true_callback == result.stdout
    for rendered in (check_false_callback, result.stdout or "", str(error)):
        for secret in secrets:
            assert secret not in rendered


@pytest.mark.parametrize(
    "error_type",
    [TimeoutError, LookupError, _CallbackBaseError, asyncio.CancelledError],
)
def test_compose_stream_callback_exception_is_propagated_after_process_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    secret = "callback-secret-value"
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/" + "noncredential-socket-path-" * 4)
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([f"output {secret}\n".encode(), b"unread tail\n"])
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def failing_callback(text: str, _stream: str) -> None:
        callback_chunks.append(text)
        raise error_type("stream consumer failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    with pytest.raises(error_type, match="stream consumer failed") as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "main", "emit-output"],
                check=False,
                on_output=failing_callback,
                env_overrides={"SECRET": secret},
            )
        )

    assert callback_chunks == [f"output {_marker_for(secret)}"]
    assert secret not in str(caught.value)
    assert process.returncode is not None
    assert process.wait_count >= 1


def test_compose_stream_external_cancellation_reaps_cooperative_callback_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([b"callback starts\n"])
    callback_started = asyncio.Event()
    callback_reaped = asyncio.Event()
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_CANCEL_SECONDS", 0.01)

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def cooperative_callback(_text: str, _stream: str) -> None:
        callback_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            callback_reaped.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> None:
        task = asyncio.create_task(
            environment._run_docker_compose_command(
                ["exec", "main", "emit-output"],
                check=False,
                on_output=cooperative_callback,
            )
        )
        await asyncio.wait_for(callback_started.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        await asyncio.wait_for(callback_reaped.wait(), timeout=1)
        await asyncio.sleep(0)
        current = asyncio.current_task()
        assert all(candidate is current or candidate.done() for candidate in asyncio.all_tasks())

    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
@pytest.mark.parametrize(
    "error_type",
    [TimeoutError, _CallbackBaseError, asyncio.CancelledError],
)
def test_real_subprocess_callback_failure_repeat_preserves_primary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    real_create_subprocess = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []
    callback_chunks: list[str] = []
    secret = "real-subprocess-callback-secret"

    async def create_real_subprocess(*_args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await real_create_subprocess(
            sys.executable,
            "-c",
            "import sys, time; print(sys.argv[1], flush=True); time.sleep(60)",
            secret,
            stdin=kwargs["stdin"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            start_new_session=kwargs["start_new_session"],
        )
        processes.append(process)
        return process

    async def failing_callback(text: str, _stream: str) -> None:
        callback_chunks.append(text)
        raise error_type("real callback failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_real_subprocess)

    async def exercise() -> None:
        try:
            for _index in range(10):
                with pytest.raises(error_type, match="real callback failure") as caught:
                    await environment._run_docker_compose_command(
                        ["exec", "main", "emit-output"],
                        check=False,
                        on_output=failing_callback,
                        env_overrides={"SECRET": secret},
                    )
                assert secret not in str(caught.value)
                assert processes[-1].returncode is not None
        finally:
            for process in processes:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()

    asyncio.run(exercise())

    assert len(processes) == 10
    assert callback_chunks == [_marker_for(secret)] * 10


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_signal_process_tree_suppresses_permission_race_only_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(pid=99881, returncode=None)
    monkeypatch.setattr(os, "killpg", lambda *_args: (_ for _ in ()).throw(PermissionError("denied")))

    def missing_process(_pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", missing_process)
    _signal_process_tree(process, signal.SIGTERM)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    with pytest.raises(PermissionError, match="denied"):
        _signal_process_tree(process, signal.SIGTERM)  # type: ignore[arg-type]


def test_callback_primary_exception_retains_cleanup_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _StreamedComposeProcess([b"callback output\n"])

    async def create_subprocess(*_args: object, **_kwargs: object) -> _StreamedComposeProcess:
        return process

    async def failing_callback(_text: str, _stream: str) -> None:
        raise _CallbackBaseError("primary callback failure")

    async def cleanup_then_fail(
        _process: object,
        communication: asyncio.Task[object],
        *,
        contain_service_on_interrupt: str | None = None,
        stop_main_on_interrupt: bool,
    ) -> None:
        assert contain_service_on_interrupt is None
        del stop_main_on_interrupt
        await communication
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(environment, "_contain_main_and_reap_compose", cleanup_then_fail)

    with pytest.raises(_CallbackBaseError, match="primary callback failure") as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "main", "emit-output"],
                check=False,
                on_output=failing_callback,
            )
        )

    assert isinstance(caught.value.__cause__, PermissionError)
    assert "cleanup denied" in str(caught.value.__cause__)
    assert any("cleanup or container containment/restoration also failed" in note for note in caught.value.__notes__)


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

    assert "--agent-import-path" not in command
    assert "--environment-import-path" not in command
    assert command[command.index("--agent") + 1] == "opencode"
    assert command[command.index("--env") + 1] == SECURE_DOCKER_ENV_IMPORT_PATH


@pytest.mark.parametrize("secret", ["x", "hunter2", "public-exec-callback-secret"])
def test_public_exec_streams_through_harbor_scoped_output_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secret: str,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    process = _BufferedAndStreamedComposeProcess(
        [f"output {secret}\n".encode(), b"tail\n"],
    )
    callback_chunks: list[tuple[str, str]] = []
    captured: dict[str, object] = {}

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedAndStreamedComposeProcess:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return process

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(on_output):
            return await environment.exec(
                "emit-output",
                env={"PUBLIC_EXEC_TOKEN": secret},
            )

    result = asyncio.run(exercise())
    marker = _collision_safe_redaction_marker({secret}, include_short=True)
    expected = f"output {marker}\ntail\n"

    assert "".join(text for text, _stream in callback_chunks) == result.stdout == expected
    assert {stream for _text, stream in callback_chunks} == {"stdout"}
    rendered_arguments = [str(argument) for argument in captured["args"]]
    if len(secret) >= 8:
        assert all(secret not in argument for argument in rendered_arguments)
    else:
        assert secret not in rendered_arguments
    assert isinstance(captured["env"], dict)
    assert captured["env"]["PUBLIC_EXEC_TOKEN"] == secret


@pytest.mark.parametrize("service", [None, MAIN_SERVICE_NAME])
def test_service_exec_for_main_delegates_to_secure_public_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service: str | None,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    captured: dict[str, object] = {}
    expected = ExecResult(stdout="secure-main\n", stderr=None, return_code=0)

    async def secure_exec(
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        captured.update(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )
        return expected

    monkeypatch.setattr(environment, "exec", secure_exec)

    result = asyncio.run(
        environment.service_exec(
            "main-command",
            service=service,
            cwd="/main-cwd",
            env={"MAIN_TOKEN": "main-explicit-secret"},
            timeout_sec=17,
            user=1200,
        )
    )

    assert result == expected
    assert captured == {
        "command": "main-command",
        "cwd": "/main-cwd",
        "env": {"MAIN_TOKEN": "main-explicit-secret"},
        "timeout_sec": 17,
        "user": 1200,
    }


def test_sidecar_service_exec_keeps_values_off_argv_and_isolates_main_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_secret = "main-persistent-secret-75124"
    task_secret = "main-task-secret-86235"
    scoped_secret = "main-scoped-secret-97346"
    sidecar_secret = "sidecar-explicit-secret-08457"
    reused_name_secret = "sidecar-reused-name-secret-19568"
    main_secrets = {persistent_secret, task_secret, scoped_secret}
    sidecar_secrets = {sidecar_secret, reused_name_secret}
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"PERSISTENT_TOKEN": persistent_secret},
    )
    environment.default_user = 4321
    environment.task_env_config = EnvironmentConfig(workdir="/main-only-workdir")
    environment._compose_task_env = {
        "TASK_TOKEN": task_secret,
        "TASK_WRAPPED": f"prefix:{persistent_secret}:suffix",
    }
    monkeypatch.setenv("SCOPED_TOKEN", scoped_secret)
    monkeypatch.setenv("HOST_WRAPPED_MAIN_TOKEN", f"prefix:{task_secret}:suffix")
    monkeypatch.setenv("NVIDIA_API_KEY", NVIDIA_BUILD_STDIN_SENTINEL)
    monkeypatch.setenv(NVIDIA_BUILD_KEY_STDIN_ENV, "1")
    monkeypatch.setenv("SKILLEVALUATOR_NVIDIA_API_KEY_FILE", "/tmp/main-only-nvidia-key")

    process = _BufferedAndStreamedComposeProcess(
        [
            f"stdout {sidecar_secret}\n".encode(),
            f"stderr {reused_name_secret}\n".encode(),
        ],
        return_code=9,
    )
    captured: dict[str, object] = {}
    callback_chunks: list[tuple[str, str]] = []

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedAndStreamedComposeProcess:
        captured["args"] = args
        captured["env"] = dict(kwargs["env"])
        return process

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with (
            environment.scoped_exec_env({"SCOPED_TOKEN": scoped_secret}),
            environment.scoped_output_callback(on_output),
        ):
            return await environment.service_exec(
                "printf sidecar-output; printf sidecar-error >&2; exit 9",
                service="helper",
                env={
                    "SIDECAR_TOKEN": sidecar_secret,
                    # Reusing a main-only name is intentional and must retain
                    # the explicit sidecar value after base-env filtering.
                    "PERSISTENT_TOKEN": reused_name_secret,
                },
            )

    result = asyncio.run(exercise())
    marker = _collision_safe_redaction_marker(sidecar_secrets)
    expected = f"stdout {marker}\nstderr {marker}\n"
    arguments = [str(argument) for argument in captured["args"]]  # type: ignore[index]
    process_environment = captured["env"]

    assert isinstance(process_environment, dict)
    exec_arguments = arguments[arguments.index("exec") :]
    carrier_names = {name for name in process_environment if name.startswith("SKILLEVALUATOR_SIDECAR_ENV_")}
    assert len(carrier_names) == 2
    assert exec_arguments[0] == "exec"
    assert exec_arguments[1:5:2] == ["-e", "-e"]
    assert set(exec_arguments[2:6:2]) == carrier_names
    assert exec_arguments[-6:-3] == ["helper", "sh", "-c"]
    assert exec_arguments[-2:] == ["sh", "printf sidecar-output; printf sidecar-error >&2; exit 9"]
    wrapper = exec_arguments[-3]
    assert all(f"export {name}=" in wrapper for name in ("SIDECAR_TOKEN", "PERSISTENT_TOKEN"))
    assert all(carrier in wrapper for carrier in carrier_names)
    assert 'exec /bin/sh -c "$1"' in wrapper
    assert "bash" not in exec_arguments[-6:]
    assert "-w" not in arguments
    assert "-u" not in arguments
    assert all(secret not in " ".join(arguments) for secret in main_secrets | sidecar_secrets)
    assert "SIDECAR_TOKEN" not in process_environment
    assert "PERSISTENT_TOKEN" not in process_environment
    assert {process_environment[name] for name in carrier_names} == sidecar_secrets
    assert (
        not {
            "TASK_TOKEN",
            "TASK_WRAPPED",
            "SCOPED_TOKEN",
            "HOST_WRAPPED_MAIN_TOKEN",
            "NVIDIA_API_KEY",
            NVIDIA_BUILD_KEY_STDIN_ENV,
            "SKILLEVALUATOR_NVIDIA_API_KEY_FILE",
        }
        & process_environment.keys()
    )
    assert all(
        secret not in value
        for value in process_environment.values()
        if isinstance(value, str)
        for secret in main_secrets
    )
    assert "".join(text for text, _stream in callback_chunks) == result.stdout == expected
    assert {stream for _text, stream in callback_chunks} == {"stdout"}
    assert result.return_code == 9


@pytest.mark.parametrize("secret", ["x", "hunter2", "abcdefgh"])
def test_sidecar_sensitive_named_values_redact_exact_short_and_long_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secret: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    process = _BufferedAndStreamedComposeProcess(
        [f"before|{secret}|after\n".encode()],
        return_code=7,
    )
    callback_chunks: list[str] = []
    captured_arguments: tuple[object, ...] = ()

    async def create_subprocess(
        *args: object,
        **_kwargs: object,
    ) -> _BufferedAndStreamedComposeProcess:
        nonlocal captured_arguments
        captured_arguments = args
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(on_output):
            return await environment.service_exec(
                "emit-sensitive-output",
                service="helper",
                env={"API_TOKEN": secret},
            )

    result = asyncio.run(exercise())
    marker = _collision_safe_redaction_marker({secret}, include_short=True)

    assert "".join(callback_chunks) == result.stdout == f"before|{marker}|after\n"
    assert secret not in (result.stdout or "")
    assert secret not in captured_arguments
    assert result.return_code == 7


def test_sidecar_exec_redacts_schemeless_proxy_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    process = _BufferedAndStreamedComposeProcess(
        [b"proxy rejected sidecar-user with sidecar-password\n"],
        return_code=7,
    )
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedAndStreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(on_output):
            return await environment.service_exec(
                "emit-proxy-failure",
                service="helper",
                env={"HTTPS_PROXY": "sidecar-user:sidecar-password@proxy.invalid:8443"},
            )

    result = asyncio.run(exercise())
    rendered = "".join(callback_chunks) + (result.stdout or "") + (result.stderr or "")

    assert result.return_code == 7
    assert "sidecar-user" not in rendered
    assert "sidecar-password" not in rendered


def test_sidecar_main_only_redaction_values_include_proxy_components(tmp_path: Path) -> None:
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={
            "HTTPS_PROXY": "main-user:main-password@proxy.invalid:8443",
        },
    )

    names, values = environment._main_only_compose_environment()

    assert "HTTPS_PROXY" in names
    assert {"main-user", "main-password"} <= values


def test_sidecar_service_exec_uses_only_explicit_workdir_and_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment.default_user = 4321
    environment.task_env_config = EnvironmentConfig(workdir="/main-only-workdir")
    captured: dict[str, object] = {}

    async def capture(
        command: list[str],
        check: bool = True,
        timeout_sec: float | None = None,
        stdin_data: bytes | None = None,
        on_output: object | None = None,
        **kwargs: object,
    ) -> ExecResult:
        captured.update(
            command=command,
            check=check,
            timeout_sec=timeout_sec,
            stdin_data=stdin_data,
            on_output=on_output,
            kwargs=kwargs,
        )
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    monkeypatch.setattr(environment, "_run_docker_compose_command", capture)

    result = asyncio.run(
        environment.service_exec(
            "pwd",
            service="helper",
            cwd="/sidecar-workdir",
            timeout_sec=23,
            user=2222,
        )
    )

    assert result.return_code == 0
    assert captured["command"] == [
        "exec",
        "-w",
        "/sidecar-workdir",
        "-u",
        "2222",
        "--",
        "helper",
        "sh",
        "-c",
        "pwd",
    ]
    assert captured["check"] is False
    assert captured["timeout_sec"] == 23
    assert captured["stdin_data"] is None
    assert captured["kwargs"] == {
        "sidecar_env_carriers": {},
        "additional_secret_values": {
            NVIDIA_BUILD_STDIN_SENTINEL,
            "skillevaluator-file-backed-nvidia-key",
        },
        "compose_env_excluded_names": {
            "NVIDIA_API_KEY",
            NVIDIA_BUILD_KEY_STDIN_ENV,
            "SKILLEVALUATOR_NVIDIA_API_KEY_FILE",
        },
        "compose_env_excluded_values": {
            NVIDIA_BUILD_STDIN_SENTINEL,
            "skillevaluator-file-backed-nvidia-key",
        },
        "use_sidecar_compose_model": True,
        "contain_service_on_interrupt": "helper",
        "stop_main_on_interrupt": False,
    }


def test_sidecar_filter_removes_all_shadowed_main_values_before_explicit_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_value = "shadowed-persistent-main-value"
    task_value = "shadowed-task-main-value"
    scoped_value = "shadowed-scoped-main-value"
    sidecar_value = "intentional-sidecar-shared-value"
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"SHARED_TOKEN": persistent_value},
    )
    environment._compose_task_env = {"SHARED_TOKEN": task_value}
    monkeypatch.setenv("WRAPPED_PERSISTENT", f"prefix:{persistent_value}:suffix")
    monkeypatch.setenv("WRAPPED_TASK", f"prefix:{task_value}:suffix")
    monkeypatch.setenv("WRAPPED_SCOPED", f"prefix:{scoped_value}:suffix")
    captured_environment: dict[str, str] = {}

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        captured_environment.update(kwargs["env"])
        return _BufferedComposeProcess(stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_exec_env({"SHARED_TOKEN": scoped_value}):
            return await environment.service_exec(
                "true",
                service="helper",
                env={"SHARED_TOKEN": sidecar_value},
            )

    result = asyncio.run(exercise())

    assert result.return_code == 0
    assert "SHARED_TOKEN" not in captured_environment
    carriers = {
        name: value for name, value in captured_environment.items() if name.startswith("SKILLEVALUATOR_SIDECAR_ENV_")
    }
    assert set(carriers.values()) == {sidecar_value}
    assert not {"WRAPPED_PERSISTENT", "WRAPPED_TASK", "WRAPPED_SCOPED"} & captured_environment.keys()
    assert all(
        main_value not in value
        for value in captured_environment.values()
        for main_value in (persistent_value, task_value, scoped_value)
    )


def test_sidecar_service_name_is_validated_and_option_terminated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    commands: list[list[str]] = []

    async def capture(command: list[str], **_kwargs: object) -> ExecResult:
        commands.append(command)
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    monkeypatch.setattr(environment, "_run_docker_compose_command", capture)

    result = asyncio.run(environment.service_exec("true", service="-T"))

    assert result.return_code == 0
    assert commands == [["exec", "--", "-T", "sh", "-c", "true"]]

    for invalid_service in ("", "helper/name", "helper name", "helper\x00name"):
        with pytest.raises(ValueError, match="Invalid Docker Compose service name"):
            asyncio.run(environment.service_exec("true", service=invalid_service))
    assert len(commands) == 1


def test_sidecar_invalid_target_environment_fails_before_spawn_without_rendering_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    secret = "invalid-target-name-secret-value"
    spawned = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        nonlocal spawned
        spawned = True
        return _BufferedComposeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(ValueError, match="Invalid environment variable name") as caught:
        asyncio.run(
            environment.service_exec(
                "true",
                service="helper",
                env={"INVALID-NAME": secret},
            )
        )

    assert secret not in str(caught.value)
    assert spawned is False


def test_same_sidecar_execs_are_serialized_before_target_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)

    async def exercise() -> bool:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()
        call_count = 0

        async def controlled_run(_command: list[str], **_kwargs: object) -> ExecResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_started.set()
                await release_first.wait()
                return ExecResult(stdout="first", stderr=None, return_code=0)
            second_started.set()
            return ExecResult(stdout="second", stderr=None, return_code=0)

        monkeypatch.setattr(environment, "_run_docker_compose_command", controlled_run)
        first = asyncio.create_task(environment.service_exec("first", service="helper"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = asyncio.create_task(environment.service_exec("second", service="helper"))
        await asyncio.sleep(0)
        overlapped = second_started.is_set()
        release_first.set()
        assert (await first).stdout == "first"
        assert (await second).stdout == "second"
        return overlapped

    assert asyncio.run(exercise()) is False


def test_cross_sidecar_callback_lock_cycle_fails_fast_without_deadlock(
    tmp_path: Path,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)

    async def exercise() -> list[tuple[str, str]]:
        both_outer_locks = asyncio.Event()
        entered = 0
        outcomes: list[tuple[str, str]] = []

        async def nested(outer: str, inner: str) -> None:
            nonlocal entered
            async with environment._sidecar_operation(outer):
                entered += 1
                if entered == 2:
                    both_outer_locks.set()
                await both_outer_locks.wait()
                try:
                    async with environment._sidecar_operation(inner):
                        outcomes.append((outer, "entered"))
                except RuntimeError:
                    outcomes.append((outer, "rejected"))

        await asyncio.wait_for(
            asyncio.gather(
                nested("helper-a", "helper-b"),
                nested("helper-b", "helper-a"),
            ),
            timeout=1,
        )
        return outcomes

    assert sorted(asyncio.run(exercise())) == [
        ("helper-a", "entered"),
        ("helper-b", "rejected"),
    ]


def test_cross_environment_sidecar_lock_cycle_fails_fast_without_deadlock(
    tmp_path: Path,
) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    environments = [
        _initialized_secure_docker_environment(tmp_path / "one"),
        _initialized_secure_docker_environment(tmp_path / "two"),
    ]
    lower, higher = sorted(environments, key=id)

    async def exercise() -> list[str]:
        both_outer_locks = asyncio.Event()
        entered = 0
        outcomes: list[str] = []

        async def nested(
            outer_environment: SkillEvaluatorSecureDockerEnvironment,
            outer_service: str,
            inner_environment: SkillEvaluatorSecureDockerEnvironment,
            inner_service: str,
            label: str,
        ) -> None:
            nonlocal entered
            async with outer_environment._sidecar_operation(outer_service):
                entered += 1
                if entered == 2:
                    both_outer_locks.set()
                await both_outer_locks.wait()
                try:
                    async with inner_environment._sidecar_operation(inner_service):
                        outcomes.append(f"{label}:entered")
                except RuntimeError:
                    outcomes.append(f"{label}:rejected")

        await asyncio.wait_for(
            asyncio.gather(
                nested(lower, "helper-a", higher, "helper-b", "ascending"),
                nested(higher, "helper-b", lower, "helper-a", "descending"),
            ),
            timeout=1,
        )
        return outcomes

    assert sorted(asyncio.run(exercise())) == [
        "ascending:entered",
        "descending:rejected",
    ]


def test_raw_service_resolution_uses_stdout_ids_and_ignores_stderr_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    identity = "1" * 64
    captured_commands: list[list[str]] = []

    async def run(command: list[str], **_kwargs: object) -> ExecResult:
        captured_commands.append(command)
        if command[:2] == ["container", "ls"]:
            return ExecResult(
                stdout=f"{identity}\n",
                stderr="time=warning msg=obsolete compose version\n",
                return_code=0,
            )
        return ExecResult(
            stdout=(
                f"{identity}\tsecure-compose-public-exec-test\thelper\t1\tFalse\t"
                f"{'c' * 64}\ttrue\tfalse\tfalse\trunning\tnone\n"
            ),
            stderr="inspection warning\n",
            return_code=0,
        )

    monkeypatch.setattr(environment, "_run_trusted_docker_command", run)

    assert asyncio.run(environment._raw_service_container_ids("helper")) == (identity,)
    assert captured_commands[0][:2] == ["container", "ls"]
    assert "label=com.docker.compose.oneoff=False" in captured_commands[0]
    assert "label=com.docker.compose.config-hash" in captured_commands[0]


def test_raw_service_resolution_rejects_warning_or_malformed_stdout_before_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    captured_commands: list[list[str]] = []

    async def run(command: list[str], **_kwargs: object) -> ExecResult:
        captured_commands.append(command)
        return ExecResult(
            stdout="warning on stdout\n",
            stderr=None,
            return_code=0,
        )

    monkeypatch.setattr(environment, "_run_trusted_docker_command", run)

    with pytest.raises(RuntimeError, match="invalid service container identity"):
        asyncio.run(environment._raw_service_container_ids("helper"))

    assert len(captured_commands) == 1
    assert captured_commands[0][:2] == ["container", "ls"]


@pytest.mark.parametrize(
    "malformation",
    ["reversed", "duplicate-number", "invalid-running", "wrong-service"],
)
def test_raw_service_resolution_rejects_malformed_or_mislabeled_inspect_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    identities = ("8" * 64, "9" * 64)
    captured_commands: list[list[str]] = []

    def state_line(identity: str, number: int) -> str:
        service = "observer" if malformation == "wrong-service" and identity == identities[0] else "helper"
        running = "unknown" if malformation == "invalid-running" and identity == identities[0] else "true"
        if malformation == "duplicate-number":
            number = 1
        return (
            f"{identity}\tsecure-compose-public-exec-test\t{service}\t{number}\tFalse\t"
            f"{'d' * 64}\t{running}\tfalse\tfalse\trunning\tnone"
        )

    async def run(command: list[str], **_kwargs: object) -> ExecResult:
        captured_commands.append(command)
        if command[:2] == ["container", "ls"]:
            return ExecResult(
                stdout="\n".join(identities) + "\n",
                stderr=None,
                return_code=0,
            )
        lines = [state_line(identities[0], 1), state_line(identities[1], 2)]
        if malformation == "reversed":
            lines.reverse()
        return ExecResult(
            stdout="\n".join(lines) + "\n",
            stderr=None,
            return_code=0,
        )

    monkeypatch.setattr(environment, "_run_trusted_docker_command", run)

    with pytest.raises(RuntimeError, match=r"invalid container state|duplicate service container numbers"):
        asyncio.run(environment._raw_service_container_ids("helper"))

    assert len(captured_commands) == 2
    assert captured_commands[0][:2] == ["container", "ls"]
    assert captured_commands[1][:2] == ["container", "inspect"]


def test_same_sidecar_callback_reentry_fails_fast_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    call_count = 0

    async def emit_once(
        _command: list[str],
        on_output: object | None = None,
        **_kwargs: object,
    ) -> ExecResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert callable(on_output)
            await on_output("outer-output\n", "stdout")
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    async def reenter(_text: str, _stream: str) -> None:
        await environment.service_exec("nested", service="helper")

    monkeypatch.setattr(environment, "_run_docker_compose_command", emit_once)

    async def exercise() -> None:
        with (
            environment.scoped_output_callback(reenter),
            pytest.raises(RuntimeError, match=r"reentrant sidecar operation.*helper"),
        ):
            await asyncio.wait_for(
                environment.service_exec("outer", service="helper"),
                timeout=1,
            )

    asyncio.run(exercise())
    assert call_count == 1


@pytest.mark.parametrize("nested_operation", ["stop", "download-file", "download-dir"])
def test_sidecar_callback_reentrant_lifecycle_operation_fails_fast_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_operation: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    call_count = 0

    async def emit_once(
        _command: list[str],
        on_output: object | None = None,
        **_kwargs: object,
    ) -> ExecResult:
        nonlocal call_count
        call_count += 1
        assert callable(on_output)
        await on_output("outer-output\n", "stdout")
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    async def reenter(_text: str, _stream: str) -> None:
        if nested_operation == "stop":
            await environment.stop_service("helper")
        elif nested_operation == "download-file":
            await environment.service_download_file(
                "/tmp/source",
                tmp_path / "target-file",
                service="helper",
            )
        else:
            await environment.service_download_dir(
                "/tmp/source",
                tmp_path / "target-dir",
                service="helper",
            )

    monkeypatch.setattr(environment, "_run_docker_compose_command", emit_once)

    async def exercise() -> None:
        with (
            environment.scoped_output_callback(reenter),
            pytest.raises(RuntimeError, match=r"reentrant sidecar operation.*helper"),
        ):
            await asyncio.wait_for(
                environment.service_exec("outer", service="helper"),
                timeout=0.2,
            )

    asyncio.run(exercise())
    assert call_count == 1


def test_inactive_sidecar_reentry_marker_does_not_poison_callback_background_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    background_tasks: list[asyncio.Task[ExecResult]] = []
    call_count = 0

    async def emit_once(
        _command: list[str],
        on_output: object | None = None,
        **_kwargs: object,
    ) -> ExecResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert callable(on_output)
            await on_output("outer-output\n", "stdout")
        return ExecResult(stdout=f"call-{call_count}", stderr=None, return_code=0)

    async def spawn_background(_text: str, _stream: str) -> None:
        async def after_outer_finishes() -> ExecResult:
            await asyncio.sleep(0.01)
            return await environment.service_exec("later", service="helper")

        background_tasks.append(asyncio.create_task(after_outer_finishes()))

    monkeypatch.setattr(environment, "_run_docker_compose_command", emit_once)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(spawn_background):
            outer = await environment.service_exec("outer", service="helper")
        assert outer.stdout == "call-1"
        assert len(background_tasks) == 1
        return await asyncio.wait_for(background_tasks[0], timeout=1)

    assert asyncio.run(exercise()).stdout == "call-2"


def test_sidecar_compose_client_restores_operational_host_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = tmp_path / "sidecar-client-env.json"
    bin_dir = tmp_path / "host-bin"
    bin_dir.mkdir()
    docker_path = bin_dir / "docker"
    main_secret = "main-only-compose-client-secret-87235"
    docker_path.write_text(
        f"""#!{sys.executable}
import json
import os

main_secret = {main_secret!r}
with open({str(audit_path)!r}, "w", encoding="utf-8") as audit:
    json.dump({{
        "PATH": os.environ.get("PATH"),
        "HOME": os.environ.get("HOME"),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST"),
        "leaked_names": sorted(name for name in os.environ if name.startswith("MAIN_ONLY")),
        "leaked_values": sorted(name for name, value in os.environ.items() if main_secret in value),
    }}, audit)
""",
        encoding="utf-8",
    )
    docker_path.chmod(0o700)
    host_home = str(tmp_path / "host-home")
    host_docker = "unix:///safe-host-docker.sock"
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HOME", host_home)
    monkeypatch.setenv("DOCKER_HOST", host_docker)
    monkeypatch.setenv("MAIN_ONLY_WRAPPED_HOST", f"prefix:{main_secret}:suffix")
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={
            "PATH": "/main-only-bin",
            "HOME": "/main-only-home",
            "DOCKER_HOST": "unix:///main-only-docker.sock",
            "MAIN_ONLY_TOKEN": main_secret,
        },
    )

    result = asyncio.run(environment.service_exec("true", service="helper"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert result.return_code == 0
    assert audit == {
        "PATH": str(bin_dir),
        "HOME": host_home,
        "DOCKER_HOST": host_docker,
        "leaked_names": [],
        "leaked_values": [],
    }


def test_raw_docker_host_baseline_retains_windows_home_controls_without_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    monkeypatch.delenv("HOME", raising=False)
    windows_home = {
        "USERPROFILE": r"C:\Users\trusted",
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\trusted",
    }
    for name, value in windows_home.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("COMPOSE_FILE", r"C:\task-controlled\compose.yaml")

    raw_environment = environment._trusted_docker_client_environment()

    assert {name: raw_environment[name] for name in windows_home} == windows_home
    assert "HOME" not in raw_environment
    assert "COMPOSE_FILE" not in raw_environment


def test_main_secure_handoff_restores_host_control_environment_for_every_compose_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = tmp_path / "main-control-audit.jsonl"
    bin_dir = tmp_path / "trusted-main-bin"
    bin_dir.mkdir()
    docker_path = bin_dir / "docker"
    docker_path.write_text(
        f"""#!{sys.executable}
import json
import os
import sys

with open({str(audit_path)!r}, "a", encoding="utf-8") as audit:
    audit.write(json.dumps({{
        "PATH": os.environ.get("PATH"),
        "HOME": os.environ.get("HOME"),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST"),
        "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG"),
        "COMPOSE_FILE": os.environ.get("COMPOSE_FILE"),
        "COMPOSE_ENV_FILES": os.environ.get("COMPOSE_ENV_FILES"),
        "COMPOSE_DISABLE_ENV_FILE": os.environ.get("COMPOSE_DISABLE_ENV_FILE"),
    }}) + "\\n")
sys.stdin.buffer.read()
""",
        encoding="utf-8",
    )
    docker_path.chmod(0o700)
    trusted_environment = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path / "trusted-main-home"),
        "DOCKER_HOST": "unix:///trusted-main-docker.sock",
        "DOCKER_CONFIG": str(tmp_path / "trusted-main-docker-config"),
        "COMPOSE_FILE": str(tmp_path / "trusted-main-compose.yaml"),
    }
    target_environment = {
        "PATH": "/main-target-bin",
        "HOME": "/main-target-home",
        "DOCKER_HOST": "tcp://main-target.invalid:2376",
        "DOCKER_CONFIG": "/main-target-docker-config",
        "COMPOSE_FILE": "/main-target-compose.yaml",
    }
    for name, value in trusted_environment.items():
        monkeypatch.setenv(name, value)
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env=target_environment,
    )

    result = asyncio.run(environment.exec("true"))
    audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    expected_client_environment = {
        **trusted_environment,
        "COMPOSE_FILE": None,
        "COMPOSE_ENV_FILES": None,
        "COMPOSE_DISABLE_ENV_FILE": "1",
    }

    assert result.return_code == 0
    assert len(audits) == 4
    assert all(audit == expected_client_environment for audit in audits)
    assert all(
        target_value not in value
        for audit in audits
        for value in audit.values()
        for target_value in target_environment.values()
        if value is not None
    )


def test_sidecar_filter_preserves_harbor_infra_over_user_name_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    user_collision = "main-user-reserved-infra-secret"
    environment._compose_task_env = {"MAIN_IMAGE_NAME": user_collision}
    expected_infra_value = environment._compose_infra_env_vars()["MAIN_IMAGE_NAME"]
    captured_environment: dict[str, str] = {}

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        captured_environment.update(kwargs["env"])
        return _BufferedComposeProcess(stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(environment.service_exec("true", service="helper"))

    assert result.return_code == 0
    assert captured_environment["MAIN_IMAGE_NAME"] == expected_infra_value
    assert all(user_collision not in value for value in captured_environment.values())


def test_sidecar_compose_client_drops_unrelated_host_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    unrelated_credentials = {
        "AWS_SECRET_ACCESS_KEY": "unrelated-host-aws-secret",
        "GITHUB_TOKEN": "unrelated-host-github-token",
        "OPENAI_API_KEY": "unrelated-host-openai-secret",
    }
    for name, value in unrelated_credentials.items():
        monkeypatch.setenv(name, value)
    captured_environment: dict[str, str] = {}

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        captured_environment.update(kwargs["env"])
        return _BufferedComposeProcess(stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(environment.service_exec("true", service="helper"))

    assert result.return_code == 0
    assert not unrelated_credentials.keys() & captured_environment.keys()
    assert all(
        credential not in value
        for credential in unrelated_credentials.values()
        for value in captured_environment.values()
    )


def test_sidecar_retains_only_structurally_required_nonsecret_compose_task_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    compose_path = environment.environment_dir / "docker-compose.yaml"
    compose_path.write_text(
        "services:\n  helper:\n    image: ${HELPER_IMAGE:?required}\n",
        encoding="utf-8",
    )
    helper_image = "python:3.13-slim"
    main_secret = "compose-model-main-api-secret"
    monkeypatch.setenv(NVIDIA_BUILD_KEY_STDIN_ENV, "1")
    environment._compose_task_env = {
        "HELPER_IMAGE": helper_image,
        "MAIN_API_TOKEN": main_secret,
        "UNREFERENCED_SETTING": "not-needed-by-compose",
    }
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedComposeProcess:
        rendered = tuple(str(argument) for argument in args)
        calls.append((rendered, dict(kwargs["env"])))
        return _BufferedComposeProcess(stdout=b"sidecar-ok")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(environment.service_exec("true", service="helper"))

    assert result.stdout == "sidecar-ok"
    assert len(calls) == 1
    exec_arguments, exec_environment = calls[0]
    assert helper_image not in " ".join(exec_arguments)
    assert exec_environment["HELPER_IMAGE"] == helper_image
    assert "MAIN_API_TOKEN" not in exec_environment
    assert "UNREFERENCED_SETTING" not in exec_environment
    assert all(main_secret not in value for value in exec_environment.values())


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("$PLAIN", {"PLAIN"}),
        ("prefix ${BRACED} suffix", {"BRACED"}),
        ("${DEFAULT:-fallback} ${REQUIRED:?required}", {"DEFAULT", "REQUIRED"}),
        ("$$ESCAPED $${ALSO_ESCAPED}", set()),
        ("${OUTER:-${INNER:-fallback}}", {"OUTER", "INNER"}),
    ],
)
def test_compose_interpolation_scanner_handles_compose_forms(
    content: str,
    expected: set[str],
) -> None:
    assert _compose_interpolation_names(content) == expected


def test_compose_model_parser_excludes_comments_and_mapping_keys_and_follows_safe_inputs(
    tmp_path: Path,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    included = environment.environment_dir / "included.yaml"
    extended = environment.environment_dir / "extended.yaml"
    included.write_text(
        "services:\n  included:\n    image: ${INCLUDED_IMAGE:?required}\n",
        encoding="utf-8",
    )
    extended.write_text(
        "services:\n  base:\n    image: ${EXTENDED_IMAGE:?required}\n",
        encoding="utf-8",
    )
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "# ${COMMENT_ONLY:?must-not-count}\n"
        "include:\n  - included.yaml\n"
        "services:\n"
        "  helper:\n"
        "    extends:\n      file: extended.yaml\n      service: base\n"
        "    image: ${HELPER_IMAGE:?required}\n"
        "    labels:\n"
        "      ${LITERAL_MAPPING_KEY}: fixed\n"
        "      used: ${MAPPING_VALUE:?required}\n"
        "      equal-list: !override\n"
        "        - ${EQUAL_LIST_KEY}=value\n"
        "    volumes: !reset &tagged_values\n"
        "      - ${TAGGED_VALUE}:/data\n"
        "  anchor-user:\n"
        "    image: alpine:3.20\n"
        "    volumes: *tagged_values\n",
        encoding="utf-8",
    )

    names = environment._compose_model_interpolation_names()

    assert names >= {
        "HELPER_IMAGE",
        "MAPPING_VALUE",
        "EQUAL_LIST_KEY",
        "TAGGED_VALUE",
        "INCLUDED_IMAGE",
        "EXTENDED_IMAGE",
    }
    assert not {"COMMENT_ONLY", "LITERAL_MAPPING_KEY"} & names


def test_docker_start_rejects_project_dotenv_before_parent_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    (environment.environment_dir / ".env").write_text(
        "HELPER_IMAGE=alpine:3.20\n",
        encoding="utf-8",
    )
    parent_started = False

    async def parent_start(_self: object, force_build: bool) -> None:
        nonlocal parent_started
        assert force_build is False
        parent_started = True

    monkeypatch.setattr(DockerEnvironment, "start", parent_start)

    with pytest.raises(RuntimeError, match=r"Docker Compose project \.env files are not supported"):
        asyncio.run(environment.start(force_build=False))

    assert parent_started is False


def test_compose_model_parser_bounds_recursive_aliases_and_node_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    compose_path = environment.environment_dir / "docker-compose.yaml"
    compose_path.write_text(
        "x-loop: &loop\n  - *loop\nservices:\n  helper:\n    image: ${HELPER_IMAGE:?required}\n",
        encoding="utf-8",
    )

    assert "HELPER_IMAGE" in environment._compose_model_interpolation_names()

    monkeypatch.setattr(secure_docker_environment, "_MAX_COMPOSE_MODEL_NODES", 1)
    with pytest.raises(RuntimeError, match="could not inspect Docker Compose"):
        environment._compose_model_interpolation_names()


def test_compose_model_parser_rejects_non_regular_include_without_blocking(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable on this platform")
    environment = _initialized_secure_docker_environment(tmp_path)
    include_path = environment.environment_dir / "blocking-include.yaml"
    os.mkfifo(include_path)
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "include:\n  - blocking-include.yaml\nservices:\n  helper:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="could not inspect Docker Compose"):
        environment._compose_model_interpolation_names()

    assert time.monotonic() - started < 1


def test_compose_model_parser_rejects_include_dotenv_and_custom_project_directory(
    tmp_path: Path,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    included_directory = environment.environment_dir / "included"
    included_directory.mkdir()
    (included_directory / "compose.yaml").write_text(
        "services:\n  helper:\n    image: ${HELPER_IMAGE:?required}\n",
        encoding="utf-8",
    )
    (included_directory / ".env").write_text(
        "API_TOKEN=must-not-enter-compose-client\n",
        encoding="utf-8",
    )
    compose_path = environment.environment_dir / "docker-compose.yaml"
    compose_path.write_text(
        "include:\n  - included/compose.yaml\nservices:\n  main:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="could not inspect Docker Compose"):
        environment._compose_model_interpolation_names()

    (included_directory / ".env").unlink()
    compose_path.write_text(
        "include:\n"
        "  - path: included/compose.yaml\n"
        "    project_directory: included\n"
        "services:\n  main:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="could not inspect Docker Compose"):
        environment._compose_model_interpolation_names()


def test_compose_model_parser_uses_project_directory_for_override_extends(
    tmp_path: Path,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    override_directory = environment.environment_dir / "overrides"
    override_directory.mkdir()
    override_path = override_directory / "override.yaml"
    override_path.write_text(
        "services:\n  helper:\n    extends:\n      file: common.yaml\n      service: base\n",
        encoding="utf-8",
    )
    (environment.environment_dir / "common.yaml").write_text(
        "services:\n  base:\n    image: ${PROJECT_BASE_IMAGE:?required}\n",
        encoding="utf-8",
    )
    (override_directory / "common.yaml").write_text(
        "services:\n  base:\n    image: ${SHADOW_IMAGE:?must-not-count}\n",
        encoding="utf-8",
    )
    environment.extra_docker_compose_paths = [override_path]

    names = environment._compose_model_interpolation_names()

    assert "PROJECT_BASE_IMAGE" in names
    assert "SHADOW_IMAGE" not in names


def test_sidecar_retains_structurally_required_nonsecret_host_compose_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    compose_path = environment.environment_dir / "docker-compose.yaml"
    compose_path.write_text(
        "services:\n  helper:\n    image: alpine:3.20\n"
        "    ports:\n"
        "      - target: 80\n"
        '        published: "${CUSTOM_PORT:?required}"\n'
        '        host_ip: "${CUSTOM_HOST_IP:?required}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CUSTOM_PORT", "43127")
    monkeypatch.setenv("CUSTOM_HOST_IP", "127.0.0.1")
    captured_environments: list[dict[str, str]] = []

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        captured_environments.append(dict(kwargs["env"]))
        return _BufferedComposeProcess(stdout=b"ok")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(environment.service_exec("true", service="helper"))

    assert result.stdout == "ok"
    assert len(captured_environments) == 1
    assert captured_environments[0]["CUSTOM_PORT"] == "43127"
    assert captured_environments[0]["CUSTOM_HOST_IP"] == "127.0.0.1"


@pytest.mark.parametrize(
    ("required_name", "required_value"),
    [
        ("API_TOKEN", "compose-required-api-token-secret"),
        ("SIDECAR_API_KEY", "short7"),
        ("DOCKER_HOST", "tcp://compose-target.invalid:2376"),
        ("HELPER_IMAGE", "prefix:compose-wrapped-secret:suffix"),
    ],
)
def test_sidecar_rejects_protected_compose_interpolation_before_user_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required_name: str,
    required_value: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    compose_path = environment.environment_dir / "docker-compose.yaml"
    compose_path.write_text(
        f"services:\n  helper:\n    image: ${{{required_name}:?required}}\n",
        encoding="utf-8",
    )
    environment._compose_task_env = {required_name: required_value}
    if required_name == "HELPER_IMAGE":
        environment._compose_task_env["MAIN_API_TOKEN"] = "compose-wrapped-secret"
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedComposeProcess:
        rendered = tuple(str(argument) for argument in args)
        calls.append((rendered, dict(kwargs["env"])))
        return _BufferedComposeProcess(stdout=b"must-not-spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(
        RuntimeError,
        match=r"requires protected execution state|cannot (?:override|use) host client controls",
    ) as caught:
        asyncio.run(environment.service_exec("true", service="helper"))

    assert calls == []
    assert required_value not in str(caught.value)


def test_sidecar_rejects_host_docker_auth_interpolation_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    docker_auth = "host-docker-auth-config-secret"
    monkeypatch.setenv("DOCKER_AUTH_CONFIG", docker_auth)
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  helper:\n    image: ${DOCKER_AUTH_CONFIG:?required}\n",
        encoding="utf-8",
    )
    spawned = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError("host Docker authorization must fail before spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="requires protected execution state") as caught:
        asyncio.run(environment.service_exec("true", service="helper"))

    assert docker_auth not in str(caught.value)
    assert spawned is False


def test_sidecar_rejects_compose_value_wrapping_short_sensitive_main_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_secret = "x"
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"API_TOKEN": short_secret},
    )
    monkeypatch.setenv("HELPER_IMAGE", f"alpine:{short_secret}")
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  helper:\n    image: ${HELPER_IMAGE:?required}\n",
        encoding="utf-8",
    )
    spawned = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError("short sensitive wrapper must fail before spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="requires protected execution state"):
        asyncio.run(environment.service_exec("true", service="helper"))

    assert spawned is False


@pytest.mark.parametrize("wrapper", ["{}", "prefix:{}:suffix"])
def test_sidecar_rejects_compose_value_reusing_other_main_only_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrapper: str,
) -> None:
    main_only_value = "non-sensitive-main-only-build-reference"
    required_value = wrapper.format(main_only_value)
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"BUILD_REF": main_only_value},
    )
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  helper:\n    image: ${CUSTOM_IMAGE:?required}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUSTOM_IMAGE", required_value)

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        raise AssertionError("protected Compose interpolation must fail before spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="requires protected execution state") as caught:
        asyncio.run(environment.service_exec("true", service="helper"))

    assert main_only_value not in str(caught.value)
    assert required_value not in str(caught.value)


def test_sidecar_rejects_effective_compose_value_wrapping_shadowed_same_name_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadowed_value = "shadowed-same-name-main-only-value"
    effective_value = f"prefix:{shadowed_value}:suffix"
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"BUILD_REF": effective_value},
    )
    environment._compose_task_env = {"BUILD_REF": shadowed_value}
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  helper:\n    image: ${BUILD_REF:?required}\n",
        encoding="utf-8",
    )
    spawned = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        nonlocal spawned
        spawned = True
        return _BufferedComposeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="requires protected execution state") as caught:
        asyncio.run(environment.service_exec("true", service="helper"))

    assert shadowed_value not in str(caught.value)
    assert effective_value not in str(caught.value)
    assert spawned is False


def test_sidecar_control_env_uses_carriers_without_redirecting_compose_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = tmp_path / "sidecar-control-env.json"
    bin_dir = tmp_path / "trusted-host-bin"
    bin_dir.mkdir()
    docker_path = bin_dir / "docker"
    docker_path.write_text(
        f"""#!{sys.executable}
import json
import os
import sys

with open({str(audit_path)!r}, "w", encoding="utf-8") as audit:
    json.dump({{
        "argv": sys.argv[1:],
        "PATH": os.environ.get("PATH"),
        "HOME": os.environ.get("HOME"),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST"),
        "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG"),
        "COMPOSE_FILE": os.environ.get("COMPOSE_FILE"),
        "COMPOSE_ENV_FILES": os.environ.get("COMPOSE_ENV_FILES"),
        "COMPOSE_DISABLE_ENV_FILE": os.environ.get("COMPOSE_DISABLE_ENV_FILE"),
        "carriers": {{
            name: value
            for name, value in os.environ.items()
            if name.startswith("SKILLEVALUATOR_SIDECAR_ENV_")
        }},
    }}, audit)
""",
        encoding="utf-8",
    )
    docker_path.chmod(0o700)
    host_home = str(tmp_path / "trusted-host-home")
    host_docker = "unix:///trusted-host-docker.sock"
    host_docker_config = str(tmp_path / "trusted-docker-config")
    host_compose_file = str(tmp_path / "trusted-compose.yaml")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HOME", host_home)
    monkeypatch.setenv("DOCKER_HOST", host_docker)
    monkeypatch.setenv("DOCKER_CONFIG", host_docker_config)
    monkeypatch.setenv("COMPOSE_FILE", host_compose_file)
    compose_env_file = tmp_path / "hostile-compose.env"
    compose_env_file.write_text("HOSTILE_TOKEN=must-not-be-loaded\n", encoding="utf-8")
    monkeypatch.setenv("COMPOSE_ENV_FILES", str(compose_env_file))
    environment = _initialized_secure_docker_environment(tmp_path)
    target_environment = {
        "PATH": "/sidecar-only-bin",
        "HOME": "/sidecar-only-home",
        "DOCKER_HOST": "tcp://sidecar-only.invalid:2376",
        "DOCKER_CONFIG": "/sidecar-only-docker-config",
        "COMPOSE_FILE": "/sidecar-only-compose.yaml",
        "NORMAL_TOKEN": "sidecar 'quoted' $dollar\nsecond-line-secret",
        "EMPTY_VALUE": "",
    }

    result = asyncio.run(
        environment.service_exec(
            "printf control-env",
            service="helper",
            env=target_environment,
        )
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rendered_argv = " ".join(audit["argv"])

    assert result.return_code == 0
    assert audit["PATH"] == str(bin_dir)
    assert audit["HOME"] == host_home
    assert audit["DOCKER_HOST"] == host_docker
    assert audit["DOCKER_CONFIG"] == host_docker_config
    assert audit["COMPOSE_FILE"] is None
    assert audit["COMPOSE_ENV_FILES"] is None
    assert audit["COMPOSE_DISABLE_ENV_FILE"] == "1"
    assert set(audit["carriers"].values()) == set(target_environment.values())
    assert len(audit["carriers"]) == len(target_environment)
    assert all(target_value not in rendered_argv for target_value in target_environment.values() if target_value)
    assert all(f"export {target_name}=" in rendered_argv for target_name in target_environment)
    assert 'exec /bin/sh -c "$1"' in rendered_argv
    assert audit["argv"][-2:] == ["sh", "printf control-env"]
    assert all(target_name not in audit["carriers"] for target_name in target_environment)


@pytest.mark.parametrize("collision_source", ["target", "base"])
def test_sidecar_carriers_retry_target_and_base_environment_name_collisions(
    monkeypatch: pytest.MonkeyPatch,
    collision_source: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    first_id = "A" * 32
    second_id = "B" * 32
    first_carrier = f"SKILLEVALUATOR_SIDECAR_ENV_{first_id}_0"
    second_carrier = f"SKILLEVALUATOR_SIDECAR_ENV_{second_id}_0"
    generated_ids = iter((first_id, second_id))
    monkeypatch.setattr(
        secure_docker_environment.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(generated_ids)),
    )
    target_name = first_carrier if collision_source == "target" else "TARGET_TOKEN"
    reserved_names = {first_carrier} if collision_source == "base" else set()

    arguments, carriers, wrapper = _sidecar_environment_carriers(
        {target_name: "collision-safe-sidecar-value"},
        reserved_names=reserved_names,
    )

    assert arguments == ["-e", second_carrier]
    assert carriers == {second_carrier: "collision-safe-sidecar-value"}
    assert wrapper is not None
    assert f"export {target_name}=" in wrapper
    assert f"unset {second_carrier}" in wrapper


@pytest.mark.parametrize("required_name", ["PATH", "DOCKER_HOST", "MAIN_IMAGE_NAME"])
def test_sidecar_fails_closed_when_trusted_client_value_wraps_main_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required_name: str,
) -> None:
    secret = "protected-main-secret-inside-client-control"
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"MAIN_ONLY_TOKEN": secret},
    )
    spawned = False

    if required_name == "MAIN_IMAGE_NAME":
        infrastructure = environment._compose_infra_env_vars()

        def poisoned_infrastructure() -> dict[str, str]:
            return {
                **infrastructure,
                required_name: f"prefix:{secret}:suffix",
            }

        monkeypatch.setattr(environment, "_compose_infra_env_vars", poisoned_infrastructure)
    else:
        monkeypatch.setenv(required_name, f"prefix:{secret}:suffix")

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        nonlocal spawned
        spawned = True
        return _BufferedComposeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="contains protected execution state") as caught:
        asyncio.run(environment.service_exec("true", service="helper"))

    assert required_name in str(caught.value)
    assert secret not in str(caught.value)
    assert spawned is False


def test_sidecar_fails_closed_when_harbor_infra_wraps_main_stdin_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    infrastructure = environment._compose_infra_env_vars()
    spawned = False

    monkeypatch.setattr(
        environment,
        "_compose_infra_env_vars",
        lambda: {
            **infrastructure,
            "MAIN_IMAGE_NAME": f"prefix:{NVIDIA_BUILD_STDIN_SENTINEL}:suffix",
        },
    )

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        nonlocal spawned
        spawned = True
        return _BufferedComposeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="contains protected execution state") as caught:
        asyncio.run(environment.service_exec("true", service="helper"))

    assert NVIDIA_BUILD_STDIN_SENTINEL not in str(caught.value)
    assert spawned is False


def test_service_stop_and_sidecar_downloads_scrub_main_compose_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_secret = "service-lifecycle-persistent-secret"
    task_secret = "service-lifecycle-task-secret"
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"MAIN_ONLY_PERSISTENT": persistent_secret},
    )
    environment._compose_task_env = {"MAIN_ONLY_TASK": task_secret}
    monkeypatch.setenv("SERVICE_LIFECYCLE_WRAPPED", f"prefix:{persistent_secret}:{task_secret}:suffix")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedComposeProcess:
        calls.append((tuple(str(argument) for argument in args), dict(kwargs["env"])))
        return _BufferedComposeProcess(stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> None:
        await environment.stop_service(MAIN_SERVICE_NAME)
        await environment.service_download_file(
            "/tmp/sidecar-file",
            tmp_path / "downloaded-file",
            service="helper",
        )
        await environment.service_download_dir(
            "/tmp/sidecar-dir",
            tmp_path / "downloaded-dir",
            service="helper",
        )

    asyncio.run(exercise())

    assert len(calls) == 3
    assert calls[0][0][1:3] == ("container", "ls")
    assert "label=com.docker.compose.service=main" in calls[0][0]
    assert calls[1][0][-4:] == (
        "cp",
        "--",
        "helper:/tmp/sidecar-file",
        str(tmp_path / "downloaded-file"),
    )
    assert calls[2][0][-4:] == (
        "cp",
        "--",
        "helper:/tmp/sidecar-dir/.",
        str(tmp_path / "downloaded-dir"),
    )
    for _arguments, process_environment in calls:
        assert (
            not {
                "MAIN_ONLY_PERSISTENT",
                "MAIN_ONLY_TASK",
                "SERVICE_LIFECYCLE_WRAPPED",
            }
            & process_environment.keys()
        )
        assert all(
            secret not in value for value in process_environment.values() for secret in (persistent_secret, task_secret)
        )


@pytest.mark.parametrize("target_service", [MAIN_SERVICE_NAME, "helper"])
def test_service_stop_uses_label_scoped_raw_docker_when_compose_model_requires_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_service: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    protected_value = "protected-stop-api-token"
    trusted_host_controls = {
        "DOCKER_HOST": "unix:///trusted-host-docker.sock",
        "DOCKER_CONFIG": str(tmp_path / "trusted-host-docker-config"),
    }
    for name, value in trusted_host_controls.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("COMPOSE_FILE", "/trusted/compose-must-not-reach-raw-docker.yaml")
    environment._compose_task_env = {
        "API_TOKEN": protected_value,
        "PATH": "/task-controlled-path",
        "HOME": "/task-controlled-home",
        "DOCKER_HOST": "tcp://task-controlled.invalid:2376",
        "DOCKER_CONFIG": "/task-controlled-docker-config",
        "COMPOSE_FILE": "/task-controlled-compose.yaml",
        "PATH_OVERLAP": os.environ["PATH"],
    }
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n    image: alpine:3.20\n"
        "    environment:\n      API_TOKEN: ${API_TOKEN:?required}\n"
        "  helper:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )
    project = "secure-compose-public-exec-test"
    main_id = "a" * 64
    helper_id = "b" * 64
    containers = {
        main_id: {"project": project, "service": "main", "number": 1, "running": True},
        helper_id: {"project": project, "service": "helper", "number": 1, "running": True},
    }
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    class RawDockerProcess:
        pid = 8123

        def __init__(self, arguments: tuple[str, ...]) -> None:
            self.arguments = arguments
            self.returncode: int | None = 0

        async def communicate(self, **_kwargs: bytes | None) -> tuple[bytes, bytes]:
            arguments = list(self.arguments[1:])
            if arguments[:2] == ["container", "ls"]:
                filters = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--filter"]
                project_filter = next(value.split("=", 2)[2] for value in filters if ".project=" in value)
                service_filter = next(value.split("=", 2)[2] for value in filters if ".service=" in value)
                matches = [
                    container_id
                    for container_id, state in containers.items()
                    if state["project"] == project_filter and state["service"] == service_filter
                ]
                stdout = "\n".join(matches) + ("\n" if matches else "")
                return stdout.encode(), b"warning on stderr"
            if arguments[:2] == ["container", "inspect"]:
                identities = arguments[arguments.index("--") + 1 :]
                states = [
                    "\t".join(
                        (
                            identity,
                            str(containers[identity]["project"]),
                            str(containers[identity]["service"]),
                            str(containers[identity]["number"]),
                            "False",
                            "c" * 64,
                            str(containers[identity]["running"]).lower(),
                            "false",
                            "false",
                            "running" if containers[identity]["running"] else "exited",
                            "none",
                        )
                    )
                    for identity in identities
                    if identity in containers
                ]
                if len(states) != len(identities):
                    self.returncode = 1
                stdout = "\n".join(states) + ("\n" if states else "")
                return stdout.encode(), b""
            if arguments[:2] in (["container", "stop"], ["container", "kill"]):
                for identity in arguments[arguments.index("--") + 1 :]:
                    containers[identity]["running"] = False
                return b"", b""
            raise AssertionError(f"unexpected raw Docker command: {arguments!r}")

        def terminate(self) -> None:
            self.returncode = -signal.SIGTERM

        def kill(self) -> None:
            self.returncode = -signal.SIGKILL

    async def create_subprocess(*args: object, **kwargs: object) -> RawDockerProcess:
        rendered = tuple(str(argument) for argument in args)
        process_environment = dict(kwargs["env"])
        calls.append((rendered, process_environment))
        assert "compose" not in rendered
        assert protected_value not in process_environment.values()
        assert "COMPOSE_FILE" not in process_environment
        assert process_environment["PATH"] == os.environ["PATH"]
        assert process_environment["HOME"] == os.environ["HOME"]
        assert {name: process_environment[name] for name in trusted_host_controls} == trusted_host_controls
        return RawDockerProcess(rendered)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    asyncio.run(environment.stop_service(target_service))

    assert containers[main_id]["running"] is (target_service != MAIN_SERVICE_NAME)
    assert containers[helper_id]["running"] is (target_service != "helper")
    assert calls
    assert all(
        f"label=com.docker.compose.project={project}" in arguments for arguments, _env in calls if "ls" in arguments
    )
    assert all(
        f"label=com.docker.compose.service={target_service}" in arguments
        for arguments, _env in calls
        if "ls" in arguments
    )
    assert all(
        "label=com.docker.compose.oneoff=False" in arguments and "label=com.docker.compose.config-hash" in arguments
        for arguments, _env in calls
        if "ls" in arguments
    )


def test_service_stop_rejects_unknown_service_before_raw_docker_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    spawned = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError("unknown service must fail before spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="unknown Docker Compose service 'does-not-exist'"):
        asyncio.run(environment.stop_service("does-not-exist"))

    assert spawned is False


def test_service_stop_accepts_service_declared_by_trusted_compose_include(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    (environment.environment_dir / "included.yaml").write_text(
        "services:\n  included-helper:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "include:\n  - included.yaml\nservices:\n  helper:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )
    calls: list[tuple[object, ...]] = []

    async def create_subprocess(*args: object, **_kwargs: object) -> _BufferedComposeProcess:
        calls.append(args)
        return _BufferedComposeProcess(stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    asyncio.run(environment.stop_service("included-helper"))

    assert len(calls) == 1
    assert "label=com.docker.compose.service=included-helper" in calls[0]


def test_raw_sidecar_containment_kills_survivor_and_restores_only_running_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    project = "secure-compose-public-exec-test"
    first_id, second_id, initially_stopped_id = ("1" * 64, "2" * 64, "3" * 64)
    containers: dict[str, dict[str, object]] = {
        first_id: {"number": 1, "running": True, "health": None},
        second_id: {"number": 2, "running": True, "health": "healthy"},
        initially_stopped_id: {"number": 3, "running": False, "health": None},
    }
    actions: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    restore_poll_count = 0
    restoration_started = False

    async def ids(service: str) -> tuple[str, ...]:
        assert service == "helper"
        return first_id, second_id, initially_stopped_id

    async def states(
        identities: tuple[str, ...],
        *,
        service: str,
    ) -> dict[str, object]:
        nonlocal restore_poll_count
        assert service == "helper"
        if restoration_started and identities == (
            first_id,
            second_id,
            initially_stopped_id,
        ):
            restore_poll_count += 1
        rendered: dict[str, object] = {}
        for identity in identities:
            state = containers[identity]
            running = bool(state["running"])
            health = state["health"]
            if identity == second_id and restoration_started and running:
                health = "starting" if restore_poll_count == 1 else "healthy"
            rendered[identity] = secure_docker_environment._RawContainerState(
                identity=identity,
                project=project,
                service=service,
                container_number=int(state["number"]),
                running=running,
                paused=False,
                restarting=False,
                status="running" if running else "exited",
                health_status=health,
            )
        return rendered

    async def action(command: list[str], identities: tuple[str, ...]) -> bool:
        nonlocal restoration_started
        actions.append((tuple(command), tuple(identities)))
        if command[:2] == ["container", "stop"]:
            # Model a stop race: replica 1 survives and must be killed.
            containers[second_id]["running"] = False
        elif command[:2] == ["container", "kill"]:
            for identity in identities:
                containers[identity]["running"] = False
        elif command[:2] == ["container", "start"]:
            restoration_started = True
            for identity in identities:
                containers[identity]["running"] = True
        else:
            raise AssertionError(f"unexpected raw action: {command!r}")
        return True

    monkeypatch.setattr(environment, "_raw_service_container_ids", ids)
    monkeypatch.setattr(environment, "_raw_container_states", states)
    monkeypatch.setattr(environment, "_raw_docker_action", action)

    async def exercise() -> tuple[object, bool]:
        with secure_docker_environment._raw_lifecycle_deadline_scope():
            snapshot = await environment._contain_sidecar_service("helper")
            assert containers[initially_stopped_id]["running"] is False
            restored = await environment._restore_sidecar_service(
                "helper",
                snapshot=snapshot,
            )
            return snapshot, restored

    snapshot, restored = asyncio.run(exercise())

    assert snapshot == secure_docker_environment._RawServiceSnapshot(
        all_identities=(first_id, second_id, initially_stopped_id),
        running_identities=(first_id, second_id),
    )
    assert restored is True
    assert actions == [
        (("container", "stop", "--timeout", "0"), (first_id, second_id)),
        (("container", "kill", "--signal", "SIGKILL"), (first_id,)),
        (("container", "start"), (first_id, second_id)),
    ]
    assert containers[initially_stopped_id]["running"] is False
    assert restore_poll_count == 2


def test_main_containment_uses_rm_fallback_when_state_inspection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    identity = "4" * 64
    removed = False
    actions: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    async def validated_ids(_service: str) -> tuple[str, ...]:
        raise RuntimeError("malformed Docker state")

    async def filtered_ids(service: str) -> tuple[str, ...]:
        assert service == MAIN_SERVICE_NAME
        return () if removed else (identity,)

    async def action(command: list[str], identities: set[str]) -> bool:
        nonlocal removed
        actions.append((tuple(command), tuple(sorted(identities))))
        assert command == ["container", "rm", "--force", "--volumes"]
        removed = True
        return True

    monkeypatch.setattr(environment, "_raw_service_container_ids", validated_ids)
    monkeypatch.setattr(environment, "_raw_filtered_service_container_ids", filtered_ids)
    monkeypatch.setattr(environment, "_raw_docker_action", action)

    asyncio.run(environment._contain_main_container())

    assert removed is True
    assert actions == [
        (("container", "rm", "--force", "--volumes"), (identity,)),
    ]


def test_sidecar_restore_rejects_replacement_container_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    original_id = "5" * 64
    initially_stopped_id = "6" * 64
    replacement_id = "7" * 64
    snapshot = secure_docker_environment._RawServiceSnapshot(
        all_identities=(original_id, initially_stopped_id),
        running_identities=(original_id,),
    )
    action_called = False

    async def ids(_service: str) -> tuple[str, ...]:
        return original_id, initially_stopped_id, replacement_id

    async def action(_command: list[str], _identities: tuple[str, ...]) -> bool:
        nonlocal action_called
        action_called = True
        return True

    monkeypatch.setattr(environment, "_raw_service_container_ids", ids)
    monkeypatch.setattr(environment, "_raw_docker_action", action)

    assert (
        asyncio.run(
            environment._restore_sidecar_service(
                "helper",
                snapshot=snapshot,
            )
        )
        is False
    )
    assert action_called is False


def test_main_service_downloads_restore_host_controls_and_harbor_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = tmp_path / "main-download-client-audit.jsonl"
    bin_dir = tmp_path / "trusted-download-bin"
    bin_dir.mkdir()
    docker_path = bin_dir / "docker"
    docker_path.write_text(
        f"""#!{sys.executable}
import json
import os

with open({str(audit_path)!r}, "a", encoding="utf-8") as audit:
    audit.write(json.dumps({{
        "PATH": os.environ.get("PATH"),
        "HOME": os.environ.get("HOME"),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST"),
        "MAIN_IMAGE_NAME": os.environ.get("MAIN_IMAGE_NAME"),
        "HARBOR_CONTAINER_NAME": os.environ.get("HARBOR_CONTAINER_NAME"),
    }}) + "\\n")
""",
        encoding="utf-8",
    )
    docker_path.chmod(0o700)
    trusted_controls = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path / "trusted-download-home"),
        "DOCKER_HOST": "unix:///trusted-download-docker.sock",
    }
    for name, value in trusted_controls.items():
        monkeypatch.setenv(name, value)
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={
            "PATH": "/main-download-target-bin",
            "HOME": "/main-download-target-home",
            "DOCKER_HOST": "tcp://main-download-target.invalid:2376",
            "MAIN_IMAGE_NAME": "main-download-user-infra-collision",
            "HARBOR_CONTAINER_NAME": "main-download-user-harbor-collision",
        },
    )
    environment._windows_container_name = "trusted-harbor-container-name"
    trusted_infrastructure = environment._compose_infra_env_vars()["MAIN_IMAGE_NAME"]

    async def exercise() -> None:
        await environment.service_download_file(
            "/tmp/source-file",
            tmp_path / "target-file",
        )
        await environment.service_download_dir(
            "/tmp/source-dir",
            tmp_path / "target-dir",
            service=MAIN_SERVICE_NAME,
        )

    asyncio.run(exercise())
    audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

    assert len(audits) == 2
    assert all(
        audit
        == {
            **trusted_controls,
            "MAIN_IMAGE_NAME": trusted_infrastructure,
            "HARBOR_CONTAINER_NAME": "trusted-harbor-container-name",
        }
        for audit in audits
    )


def test_windows_main_downloads_use_scrubbed_docker_env_and_validated_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_secret = "windows-main-transfer-secret-value"
    unrelated_secret = "windows-unrelated-host-secret-value"
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"MAIN_TOKEN": main_secret},
    )
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    monkeypatch.setenv("UNRELATED_API_TOKEN", unrelated_secret)
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
    docker_call_count = 0

    class TransferProcess:
        pid = 7643

        def __init__(
            self,
            arguments: tuple[str, ...],
            process_environment: dict[str, str],
            archive_payload: bytes,
        ) -> None:
            self.arguments = arguments
            self.process_environment = process_environment
            self.returncode: int | None = None
            self.stdout = _ChunkStream([archive_payload])
            self.stderr = _ChunkStream([b"successful docker warning must not enter tar bytes"])

        async def wait(self) -> int:
            self.returncode = 0
            return self.returncode

        async def communicate(self, **_kwargs: bytes | None) -> tuple[bytes, bytes]:
            raise AssertionError("Windows container archives must not be buffered by communicate()")

    async def create_subprocess(*args: object, **kwargs: object) -> TransferProcess:
        nonlocal docker_call_count
        assert "env" in kwargs
        assert Path(str(args[0])).name == "docker"
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        docker_call_count += 1
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            if docker_call_count == 1:
                payload = b"\x00windows-file\xff"
                member = tarfile.TarInfo("./payload.bin")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                directory = tarfile.TarInfo("./nested")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                payload = b"\x00windows-dir\xfe"
                member = tarfile.TarInfo("./nested/value.bin")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        archive_payload = archive_buffer.getvalue()
        arguments = tuple(str(argument) for argument in args)
        process_environment = dict(kwargs["env"])
        calls.append((arguments, process_environment))
        return TransferProcess(
            arguments,
            process_environment,
            archive_payload,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    downloaded_file = tmp_path / "downloaded.bin"
    downloaded_dir = tmp_path / "downloaded-dir"
    downloaded_file.write_bytes(b"old-file")
    (downloaded_dir / "nested").mkdir(parents=True)
    (downloaded_dir / "nested" / "value.bin").write_bytes(b"old-directory-file")
    (downloaded_dir / "preserved.bin").write_bytes(b"preserved")

    async def exercise() -> None:
        await environment.service_download_file(
            "/remote/payload.bin",
            downloaded_file,
        )
        await environment.service_download_dir(
            "/remote/tree",
            downloaded_dir,
            service=MAIN_SERVICE_NAME,
        )

    asyncio.run(exercise())

    assert downloaded_file.read_bytes() == b"\x00windows-file\xff"
    assert (downloaded_dir / "nested" / "value.bin").read_bytes() == b"\x00windows-dir\xfe"
    assert (downloaded_dir / "preserved.bin").read_bytes() == b"preserved"
    assert len(calls) == 2
    assert [Path(arguments[0]).name for arguments, _env in calls] == [
        "docker",
        "docker",
    ]
    assert calls[0][0][1:5] == (
        "exec",
        "--",
        "harbor-secure-windows",
        "tar",
    )
    assert calls[0][0][-4:] == ("-C", "/remote", "--", "payload.bin")
    assert calls[1][0][-4:] == ("-C", "/remote/tree", "--", ".")
    for _arguments, process_environment in calls:
        assert "MAIN_TOKEN" not in process_environment
        assert "UNRELATED_API_TOKEN" not in process_environment
        assert all(
            secret not in value for secret in (main_secret, unrelated_secret) for value in process_environment.values()
        )


@pytest.mark.parametrize(
    ("source_path", "expected_parent"),
    (
        ("C:/payload.bin", "C:/"),
        (r"C:\payload.bin", "C:/"),
        ("/payload.bin", "/"),
        ("payload.bin", "."),
    ),
)
def test_windows_file_download_preserves_container_source_root_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_path: str,
    expected_parent: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    calls: list[tuple[str, tuple[str, ...]]] = []

    async def download_dir(
        source_dir: str,
        target_dir: Path | str,
        **kwargs: object,
    ) -> None:
        archive_members = kwargs["archive_members"]
        assert isinstance(archive_members, tuple)
        calls.append((source_dir, archive_members))
        (Path(target_dir) / "payload.bin").write_bytes(b"payload")

    monkeypatch.setattr(environment, "_secure_windows_download_dir", download_dir)
    target = tmp_path / "downloaded.bin"

    asyncio.run(environment.service_download_file(source_path, target))

    assert calls == [(expected_parent, ("payload.bin",))]
    assert target.read_bytes() == b"payload"


@pytest.mark.parametrize("source_path", ["", "C:/", "C:\\", "/", "bad\x00name"])
def test_windows_file_download_rejects_invalid_container_source_path(
    tmp_path: Path,
    source_path: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"

    with pytest.raises(RuntimeError, match="download path is invalid"):
        asyncio.run(environment.service_download_file(source_path, tmp_path / "target.bin"))


def test_windows_transfer_streams_large_binary_output_without_communicate(tmp_path: Path) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "large-transfer.bin"
    output_size = 16 * 1024 * 1024 + 137

    tracemalloc.start()
    asyncio.run(
        environment._run_trusted_transfer_command(
            [
                sys.executable,
                "-c",
                f"import sys; sys.stdout.buffer.write(bytes(range(256)) * {output_size // 256} + "
                f"bytes(range({output_size % 256})))",
            ],
            process_environment=dict(os.environ),
            protected_values=set(),
            output_path=output_path,
            idle_timeout_sec=5,
        )
    )
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    observed_digest = hashlib.sha256()
    with output_path.open("rb") as output:
        while chunk := output.read(1024 * 1024):
            observed_digest.update(chunk)
    expected_digest = hashlib.sha256()
    for _ in range(output_size // 256):
        expected_digest.update(bytes(range(256)))
    expected_digest.update(bytes(range(output_size % 256)))

    assert output_path.stat().st_size == output_size
    assert observed_digest.hexdigest() == expected_digest.hexdigest()
    assert peak_bytes < 12 * 1024 * 1024


def test_windows_transfer_removes_output_if_diagnostic_spool_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "unstarted-transfer.tar"
    spawned = False

    def fail_temporary_file(**_kwargs: object) -> object:
        raise OSError("injected diagnostic spool failure")

    async def create_subprocess(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError

    monkeypatch.setattr(secure_docker_environment.tempfile, "TemporaryFile", fail_temporary_file)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(OSError, match="injected diagnostic spool failure"):
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values=set(),
                output_path=output_path,
            )
        )

    assert spawned is False
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("stdout_bytes", "stderr_bytes", "should_fail"),
    (
        (11, 0, True),
        (0, 11, True),
        (6, 5, True),
        (6, 4, False),
    ),
)
def test_windows_transfer_enforces_combined_temporary_disk_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout_bytes: int,
    stderr_bytes: int,
    should_fail: bool,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "budgeted-transfer.tar"

    class TransferProcess:
        pid = 7648

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _ChunkStream([b"o" * stdout_bytes])
            self.stderr = _ChunkStream([b"e" * stderr_bytes])

        async def wait(self) -> int:
            self.returncode = 0
            return self.returncode

    async def create_subprocess(*_args: object, **kwargs: object) -> TransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return TransferProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_DISK_RESERVE_BYTES", 0)
    monkeypatch.setattr(
        secure_docker_environment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10),
    )

    async def transfer() -> None:
        await environment._run_trusted_transfer_command(
            ["docker", "version"],
            process_environment={"PATH": os.environ["PATH"]},
            protected_values=set(),
            output_path=output_path,
        )

    if should_fail:
        with pytest.raises(RuntimeError, match="exceeded its temporary disk budget"):
            asyncio.run(transfer())
        assert not output_path.exists()
    else:
        asyncio.run(transfer())
        assert output_path.read_bytes() == b"o" * stdout_bytes


def test_windows_artifact_disk_reserve_does_not_ratchet_between_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    initial_free = 2 * 1024 * 1024 * 1024
    later_free = 1024 * 1024 * 1024
    free_values = iter((initial_free, later_free, later_free))
    monkeypatch.setattr(
        secure_docker_environment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=next(free_values)),
    )

    reserve = secure_docker_environment._windows_artifact_filesystem_reserve(tmp_path)
    assert reserve.minimum_free_bytes == initial_free // 20
    exact_later_budget = later_free - reserve.minimum_free_bytes

    secure_docker_environment._require_windows_artifact_resources(
        tmp_path,
        exact_later_budget,
        required_entries=0,
        minimum_free_bytes=reserve.minimum_free_bytes,
        purpose="test phase",
    )
    with pytest.raises(RuntimeError, match="insufficient disk space"):
        secure_docker_environment._require_windows_artifact_resources(
            tmp_path,
            exact_later_budget + 1,
            required_entries=0,
            minimum_free_bytes=reserve.minimum_free_bytes,
            purpose="test phase",
        )


@pytest.mark.parametrize(
    ("same_filesystem", "expected_publication_reserve"),
    ((True, 20), (False, 10)),
)
def test_windows_download_reuses_or_separates_filesystem_reserves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_filesystem: bool,
    expected_publication_reserve: int,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    target = tmp_path / "target"
    captured: dict[str, int] = {}
    reserves = iter(
        (
            secure_docker_environment._WindowsFilesystemReserve(identity=1, minimum_free_bytes=10),
            secure_docker_environment._WindowsFilesystemReserve(
                identity=1 if same_filesystem else 2,
                minimum_free_bytes=20,
            ),
        )
    )

    async def transfer(_command: list[str], **kwargs: object) -> None:
        captured["transfer"] = int(kwargs["minimum_free_bytes"])  # type: ignore[arg-type]
        Path(kwargs["output_path"]).write_bytes(b"unused")  # type: ignore[arg-type]

    def extract(_archive: Path, _target: Path, **kwargs: object) -> object:
        captured["extraction"] = int(kwargs["minimum_free_bytes"])  # type: ignore[arg-type]
        return secure_docker_environment._WindowsArtifactUsage(file_bytes=0, entries=1)

    def require_resources(_path: Path, _required_bytes: int, **kwargs: object) -> None:
        captured["publication"] = int(kwargs["minimum_free_bytes"])  # type: ignore[arg-type]

    monkeypatch.setattr(secure_docker_environment, "_windows_artifact_filesystem_reserve", lambda _path: next(reserves))
    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    monkeypatch.setattr(secure_docker_environment, "_extract_regular_tar_archive", extract)
    monkeypatch.setattr(secure_docker_environment, "_require_windows_artifact_resources", require_resources)
    monkeypatch.setattr(secure_docker_environment, "copytree_secure", lambda *_args, **_kwargs: None)

    asyncio.run(environment.service_download_dir("C:/artifacts", target))

    assert captured == {
        "transfer": 20,
        "extraction": 20,
        "publication": expected_publication_reserve,
    }


def test_windows_large_file_download_extracts_and_publishes_with_bounded_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    payload_path = tmp_path / "payload.bin"
    payload_size = 16 * 1024 * 1024 + 113
    block = bytes(range(256)) * 4096
    with payload_path.open("wb") as payload_file:
        remaining = payload_size
        while remaining:
            chunk = block[: min(remaining, len(block))]
            payload_file.write(chunk)
            remaining -= len(chunk)
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        archive.add(payload_path, arcname="payload.bin", recursive=False)

    async def transfer(_command: list[str], **kwargs: object) -> None:
        shutil.copyfile(archive_path, Path(kwargs["output_path"]))  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    downloaded = tmp_path / "downloaded.bin"
    downloaded.write_bytes(b"old")

    tracemalloc.start()
    asyncio.run(environment.service_download_file("/remote/payload.bin", downloaded))
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    source_digest = hashlib.sha256()
    downloaded_digest = hashlib.sha256()
    with payload_path.open("rb") as source, downloaded.open("rb") as destination:
        while chunk := source.read(1024 * 1024):
            source_digest.update(chunk)
        while chunk := destination.read(1024 * 1024):
            downloaded_digest.update(chunk)

    assert downloaded.stat().st_size == payload_size
    assert downloaded_digest.hexdigest() == source_digest.hexdigest()
    assert peak_bytes < 12 * 1024 * 1024


@pytest.mark.parametrize("operation", ["file", "directory"])
def test_windows_download_rejects_insufficient_publication_filesystem_before_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        payload = b"new"
        member = tarfile.TarInfo("./payload.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_payload = archive_buffer.getvalue()

    async def transfer(_command: list[str], **kwargs: object) -> None:
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_ENTRY_DISK_BYTES", 1)
    target = tmp_path / ("downloaded.bin" if operation == "file" else "downloaded-dir")
    if operation == "file":
        target.write_bytes(b"old")
        target_budget = 7  # new stage (4) plus existing rollback (4), minus one
    else:
        target.mkdir()
        (target / "sentinel.bin").write_bytes(b"old")
        target_budget = 14  # new tree (5) plus two existing snapshots (2 * 5), minus one

    def disk_budget(path: Path) -> tuple[int, int]:
        if Path(path) == target.parent:
            return target_budget, target_budget
        return 1024 * 1024, 1024 * 1024

    monkeypatch.setattr(secure_docker_environment, "_windows_artifact_disk_budget", disk_budget)

    if operation == "file":
        with pytest.raises(RuntimeError, match=r"insufficient disk space.*file publication"):
            asyncio.run(environment.service_download_file("/remote/payload.bin", target))
        assert target.read_bytes() == b"old"
    else:
        with pytest.raises(RuntimeError, match=r"insufficient disk space.*directory publication"):
            asyncio.run(environment.service_download_dir("/remote/tree", target))
        assert [child.name for child in target.iterdir()] == ["sentinel.bin"]
        assert (target / "sentinel.bin").read_bytes() == b"old"


def test_windows_file_publication_charges_missing_destination_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        payload = b"new"
        member = tarfile.TarInfo("./payload.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_payload = archive_buffer.getvalue()

    async def transfer(_command: list[str], **kwargs: object) -> None:
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_ENTRY_DISK_BYTES", 1)
    target = tmp_path / "missing-one" / "missing-two" / "downloaded.bin"

    def disk_budget(path: Path) -> tuple[int, int]:
        if Path(path) == target.parent:
            return 5, 5  # file stage (4) plus two missing parents (2), minus one
        return 1024 * 1024, 1024 * 1024

    monkeypatch.setattr(secure_docker_environment, "_windows_artifact_disk_budget", disk_budget)

    with pytest.raises(RuntimeError, match=r"insufficient disk space.*file publication"):
        asyncio.run(environment.service_download_file("/remote/payload.bin", target))

    assert not (tmp_path / "missing-one").exists()


def test_windows_file_publication_enospc_preserves_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_copy, secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        payload = b"replacement"
        member = tarfile.TarInfo("./payload.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_payload = archive_buffer.getvalue()

    async def transfer(_command: list[str], **kwargs: object) -> None:
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    original_copy = secure_copy.copy_file_secure

    def copy_with_enospc(*args: object, **kwargs: object) -> None:
        original_write = secure_copy.os.write
        injected = False

        def fail_after_partial_write(descriptor: int, data: bytes | memoryview) -> int:
            nonlocal injected
            written = original_write(descriptor, data)
            if not injected:
                injected = True
                raise OSError(errno.ENOSPC, "injected publication disk exhaustion")
            return written

        with monkeypatch.context() as copy_patch:
            copy_patch.setattr(secure_copy.os, "write", fail_after_partial_write)
            original_copy(*args, **kwargs)

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    monkeypatch.setattr(secure_docker_environment, "copy_file_secure", copy_with_enospc)
    target = tmp_path / "downloaded.bin"
    target.write_bytes(b"old")

    with pytest.raises(RuntimeError, match="could not be copied safely"):
        asyncio.run(environment.service_download_file("/remote/payload.bin", target))

    assert target.read_bytes() == b"old"
    assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith(".downloaded.bin")) == []


def test_windows_transfer_timeout_reaps_process_and_removes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "partial-transfer.tar"
    exited = asyncio.Event()
    signalled: list[signal.Signals] = []

    class BlockedTransferProcess:
        pid = 7645

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _FeedableStream()
            self.stderr = _FeedableStream()

        async def wait(self) -> int:
            await exited.wait()
            assert self.returncode is not None
            return self.returncode

    process = BlockedTransferProcess()

    async def create_subprocess(*_args: object, **kwargs: object) -> BlockedTransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        process.stdout.feed_data(b"partial-untrusted-archive")
        return process

    def signal_process_tree(_process: object, value: signal.Signals) -> None:
        signalled.append(value)
        process.returncode = -int(value)
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        exited.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_signal_process_tree", signal_process_tree)

    with pytest.raises(RuntimeError, match="secure Windows container transfer command timed out while idle"):
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values=set(),
                output_path=output_path,
                idle_timeout_sec=0.01,
            )
        )

    assert signalled == [signal.SIGTERM]
    assert not output_path.exists()


def test_windows_transfer_idle_timeout_resets_while_output_progresses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "progressing-transfer.tar"
    exited = asyncio.Event()
    producer: asyncio.Task[None] | None = None

    class ProgressingTransferProcess:
        pid = 7647

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _FeedableStream()
            self.stderr = _FeedableStream()

        async def wait(self) -> int:
            await exited.wait()
            assert self.returncode is not None
            return self.returncode

    process = ProgressingTransferProcess()

    async def create_subprocess(*_args: object, **kwargs: object) -> ProgressingTransferProcess:
        nonlocal producer
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE

        async def produce() -> None:
            for index in range(8):
                process.stdout.feed_data(bytes([index]))
                await asyncio.sleep(0.01)
            process.stdout.feed_eof()
            process.stderr.feed_eof()
            process.returncode = 0
            exited.set()

        producer = asyncio.create_task(produce())
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> None:
        await environment._run_trusted_transfer_command(
            ["docker", "version"],
            process_environment={"PATH": os.environ["PATH"]},
            protected_values=set(),
            output_path=output_path,
            idle_timeout_sec=0.025,
        )
        assert producer is not None
        await producer

    asyncio.run(exercise())

    assert output_path.read_bytes() == bytes(range(8))


def test_windows_transfer_growing_output_exceeding_budget_reaps_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "growing-transfer.tar"
    exited = asyncio.Event()
    signalled: list[signal.Signals] = []
    producer_tasks: list[asyncio.Task[None]] = []

    class GrowingTransferProcess:
        pid = 7651

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _FeedableStream()
            self.stderr = _FeedableStream()

        async def wait(self) -> int:
            await exited.wait()
            assert self.returncode is not None
            return self.returncode

    process = GrowingTransferProcess()

    async def create_subprocess(*_args: object, **kwargs: object) -> GrowingTransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE

        async def produce() -> None:
            while not exited.is_set():
                process.stdout.feed_data(b"grow")
                await asyncio.sleep(0.002)

        producer_tasks.append(asyncio.create_task(produce()))
        return process

    def signal_process_tree(_process: object, value: signal.Signals) -> None:
        signalled.append(value)
        process.returncode = -int(value)
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        exited.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_signal_process_tree", signal_process_tree)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TRANSFER_POLL_SECONDS", 0.001)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_DISK_RESERVE_BYTES", 0)
    monkeypatch.setattr(
        secure_docker_environment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10),
    )

    with pytest.raises(RuntimeError, match="exceeded its temporary disk budget"):
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values=set(),
                output_path=output_path,
            )
        )

    assert signalled == [signal.SIGTERM]
    assert all(task.done() for task in producer_tasks)
    assert not output_path.exists()


def test_windows_transfer_total_timeout_reaps_continuously_progressing_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "trickling-transfer.tar"
    exited = asyncio.Event()
    signalled: list[signal.Signals] = []
    producer_tasks: list[asyncio.Task[None]] = []

    class TricklingTransferProcess:
        pid = 7652

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _FeedableStream()
            self.stderr = _FeedableStream()

        async def wait(self) -> int:
            await exited.wait()
            assert self.returncode is not None
            return self.returncode

    process = TricklingTransferProcess()

    async def create_subprocess(*_args: object, **kwargs: object) -> TricklingTransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE

        async def produce() -> None:
            while not exited.is_set():
                process.stdout.feed_data(b"x")
                await asyncio.sleep(0.002)

        producer_tasks.append(asyncio.create_task(produce()))
        return process

    def signal_process_tree(_process: object, value: signal.Signals) -> None:
        signalled.append(value)
        process.returncode = -int(value)
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        exited.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_signal_process_tree", signal_process_tree)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TRANSFER_POLL_SECONDS", 0.001)

    with pytest.raises(RuntimeError, match="exceeded its total time limit"):
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values=set(),
                output_path=output_path,
                idle_timeout_sec=0.02,
                total_timeout_sec=0.01,
            )
        )

    assert signalled == [signal.SIGTERM]
    assert all(task.done() for task in producer_tasks)
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("timeout_name", "timeout_value"),
    (
        ("idle_timeout_sec", 0.0),
        ("idle_timeout_sec", -1.0),
        ("idle_timeout_sec", float("nan")),
        ("idle_timeout_sec", float("inf")),
        ("total_timeout_sec", 0.0),
        ("total_timeout_sec", -1.0),
        ("total_timeout_sec", float("nan")),
        ("total_timeout_sec", float("inf")),
    ),
)
def test_windows_transfer_rejects_nonpositive_or_nonfinite_timeouts(
    tmp_path: Path,
    timeout_name: str,
    timeout_value: float,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)

    with pytest.raises(ValueError, match="must be a positive finite value"):
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values=set(),
                output_path=tmp_path / "invalid-timeout.tar",
                **{timeout_name: timeout_value},
            )
        )


def test_windows_transfer_rejects_process_completion_after_total_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "late-completion.tar"
    signalled: list[signal.Signals] = []

    class LateTransferProcess:
        pid = 7653

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _ChunkStream([])
            self.stderr = _ChunkStream([])

        async def wait(self) -> int:
            await asyncio.sleep(0.02)
            self.returncode = 0
            return self.returncode

    process = LateTransferProcess()

    async def create_subprocess(*_args: object, **_kwargs: object) -> LateTransferProcess:
        return process

    def signal_process_tree(_process: object, value: signal.Signals) -> None:
        signalled.append(value)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_signal_process_tree", signal_process_tree)

    with pytest.raises(RuntimeError, match="exceeded its total time limit"):
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values=set(),
                output_path=output_path,
                total_timeout_sec=0.01,
            )
        )

    assert signalled == [signal.SIGTERM]
    assert not output_path.exists()


def test_windows_transfer_repeated_cancellation_reaps_process_and_removes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "cancelled-transfer.tar"
    started = asyncio.Event()
    exited = asyncio.Event()
    signalled: list[signal.Signals] = []

    class BlockedTransferProcess:
        pid = 7646

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _FeedableStream()
            self.stderr = _FeedableStream()

        async def wait(self) -> int:
            await exited.wait()
            assert self.returncode is not None
            return self.returncode

    process = BlockedTransferProcess()

    async def create_subprocess(*_args: object, **kwargs: object) -> BlockedTransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        process.stdout.feed_data(b"partial-cancelled-archive")
        started.set()
        return process

    def signal_process_tree(_process: object, value: signal.Signals) -> None:
        signalled.append(value)
        process.returncode = -int(value)
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        exited.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_signal_process_tree", signal_process_tree)

    async def exercise() -> None:
        transfer = asyncio.create_task(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values=set(),
                output_path=output_path,
            )
        )
        await started.wait()
        transfer.cancel()
        await asyncio.sleep(0)
        transfer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await transfer

    asyncio.run(exercise())

    assert signalled == [signal.SIGTERM]
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("member_type", "member_name", "link_name"),
    (
        (tarfile.SYMTYPE, "./payload.bin", "/host/outside.txt"),
        (tarfile.LNKTYPE, "./payload.bin", "./other.bin"),
        (tarfile.FIFOTYPE, "./payload.bin", ""),
        (tarfile.REGTYPE, "../payload.bin", ""),
        (tarfile.REGTYPE, "/payload.bin", ""),
    ),
)
def test_windows_main_download_rejects_unsafe_container_archive_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_type: bytes,
    member_name: str,
    link_name: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    outside = tmp_path / "outside.txt"
    outside.write_text("host-only-content", encoding="utf-8")

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        member = tarfile.TarInfo(member_name)
        member.type = member_type
        member.linkname = str(outside) if member_type == tarfile.SYMTYPE else link_name
        if member_type == tarfile.REGTYPE:
            payload = b"container-controlled-content"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        else:
            archive.addfile(member)
    archive_payload = archive_buffer.getvalue()

    async def transfer(command: list[str], **kwargs: object) -> None:
        assert command[0] == "docker"
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    downloaded = tmp_path / "downloaded.bin"

    with pytest.raises(RuntimeError, match="unsafe Windows container download archive"):
        asyncio.run(environment.service_download_file("/remote/payload.bin", downloaded))

    assert not downloaded.exists()
    assert outside.read_text(encoding="utf-8") == "host-only-content"


def test_windows_tar_member_rejects_many_leading_current_components_before_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    class NoFrontPopParts(list[str]):
        def pop(self, index: int = -1) -> str:
            if index == 0:
                raise AssertionError("front-pop path normalization is quadratic")
            return super().pop(index)

    class ManyLeadingCurrentComponents(str):
        __slots__ = ()

        def split(self, *args: object, **kwargs: object) -> NoFrontPopParts:
            return NoFrontPopParts(super().split(*args, **kwargs))

    member = tarfile.TarInfo(ManyLeadingCurrentComponents("./" * 100_000 + "payload.bin"))
    monkeypatch.setattr(
        tarfile,
        "data_filter",
        lambda _candidate, _target: pytest.fail("unbounded path reached tarfile filter"),
    )

    with pytest.raises(ValueError):
        secure_docker_environment._validated_windows_tar_member(member, tmp_path)


def test_windows_tar_member_accepts_bounded_leading_current_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    member = tarfile.TarInfo(
        "./" * secure_docker_environment._WINDOWS_TAR_MAX_LEADING_CURRENT_COMPONENTS + "payload.bin"
    )
    monkeypatch.setattr(tarfile, "data_filter", lambda candidate, _target: candidate)

    validated = secure_docker_environment._validated_windows_tar_member(member, tmp_path)

    assert validated is not None
    assert validated.canonical_parts == ("payload.bin",)


@pytest.mark.parametrize(
    "member_names",
    (
        ("./payload.bin", "./payload.bin"),
        ("./Readme", "./README"),
        ("./café", "./cafe\N{COMBINING ACUTE ACCENT}"),
        ("./file", "./file/child"),
        ("./file/child", "./file"),
        ("./name:alternate-stream",),
        ("./CON.txt",),
        ("./CONIN$",),
        ("./CONOUT$.log",),
        ("./COM1",),
        ("./COM9.log",),
        ("./LPT1",),
        ("./LPT9.txt",),
        ("./COM¹",),
        ("./LPT³.txt",),
        ("./trailing.",),
        ("./trailing-space ",),
    ),
)
def test_windows_main_download_rejects_aliases_reserved_names_and_file_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_names: tuple[str, ...],
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for index, member_name in enumerate(member_names):
            payload = f"payload-{index}".encode()
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    archive_payload = archive_buffer.getvalue()

    async def transfer(_command: list[str], **kwargs: object) -> None:
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    target = tmp_path / "existing-target"
    target.mkdir()
    sentinel = target / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")

    with pytest.raises(RuntimeError, match="unsafe Windows container download archive"):
        asyncio.run(environment.service_download_dir("/remote/tree", target))

    assert list(target.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("member_name", ("./COM0", "./COM0.txt", "./LPT0", "./LPT0.log"))
def test_windows_tar_member_accepts_nonreserved_zero_suffixed_device_names(
    tmp_path: Path,
    member_name: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    member = tarfile.TarInfo(member_name)

    validated = secure_docker_environment._validated_windows_tar_member(member, tmp_path)

    assert validated is not None
    assert validated.kind == "file"


@pytest.mark.parametrize("archive_kind", ["truncated", "compressed"])
def test_windows_main_download_rejects_malformed_or_compressed_archive_before_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_kind: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    archive_buffer = io.BytesIO()
    mode = "w:gz" if archive_kind == "compressed" else "w"
    with tarfile.open(fileobj=archive_buffer, mode=mode) as archive:
        payload = b"container-content"
        member = tarfile.TarInfo("./payload.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_payload = archive_buffer.getvalue()
    if archive_kind == "truncated":
        archive_payload = archive_payload[:700]

    async def transfer(_command: list[str], **kwargs: object) -> None:
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    target = tmp_path / "existing-target"
    target.mkdir()
    sentinel = target / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")

    with pytest.raises(RuntimeError, match="unsafe Windows container download archive"):
        asyncio.run(environment.service_download_dir("/remote/tree", target))

    assert list(target.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("bound", ["members", "paths", "components", "metadata", "disk"])
def test_windows_main_download_enforces_archive_resource_bounds_before_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    archive_buffer = io.BytesIO()
    long_name = "./" + "long-name-" * 20 + "payload.bin"
    with tarfile.open(fileobj=archive_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        names = (
            ("./first.bin", "./second.bin")
            if bound == "members"
            else (
                "./nested/payload.bin"
                if bound == "components"
                else long_name
                if bound in {"paths", "metadata"}
                else "./payload.bin",
            )
        )
        for name in names:
            payload = b"four"
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    archive_payload = archive_buffer.getvalue()

    if bound == "members":
        monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TAR_MAX_MEMBERS", 1)
    elif bound == "paths":
        monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TAR_MAX_PATH_BYTES", 8)
    elif bound == "components":
        monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TAR_MAX_PATH_COMPONENTS", 1)
    elif bound == "metadata":
        monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TAR_MAX_EXTENSION_BYTES", 8)
    else:
        monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_DISK_RESERVE_BYTES", 0)
        monkeypatch.setattr(
            secure_docker_environment.shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(free=3),
        )

    async def transfer(_command: list[str], **kwargs: object) -> None:
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    target = tmp_path / "existing-target"
    target.mkdir()
    sentinel = target / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")

    expected_error = (
        "insufficient disk space.*archive extraction"
        if bound == "disk"
        else "unsafe Windows container download archive"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        asyncio.run(environment.service_download_dir("/remote/tree", target))

    assert list(target.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize(
    "extension_type",
    (
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.SOLARIS_XHDTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
    ),
)
def test_windows_tar_prescan_bounds_every_allocating_extension_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extension_type: bytes,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "metadata-extension.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        payload = b"1 path=payload.bin\n" + b"x" * 64
        member = tarfile.TarInfo("pax-metadata")
        member.type = extension_type
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TAR_MAX_EXTENSION_BYTES", 8)

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_rejects_global_pax_even_below_metadata_limit(
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "global-pax.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        member = tarfile.TarInfo("global-pax")
        member.type = tarfile.XGLTYPE
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
        for index in range(32):
            payload = b""
            regular = tarfile.TarInfo(f"payload-{index}.bin")
            regular.size = len(payload)
            archive.addfile(regular, io.BytesIO(payload))

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


@pytest.mark.parametrize(
    "sparse_keyword",
    (
        "GNU.sparse.map",
        "GNU.sparse.size",
        "GNU.sparse.name",
        "GNU.sparse.realsize",
        "GNU.sparse.major",
        "GNU.sparse.minor",
    ),
)
def test_windows_tar_prescan_rejects_local_pax_gnu_sparse_metadata(
    tmp_path: Path,
    sparse_keyword: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "gnu-sparse-pax.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        payload = b"1\n0\n0\n"
        member = tarfile.TarInfo("payload.bin")
        member.pax_headers = {sparse_keyword: "1"}
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_rejects_gnu_sparse_v1_before_payload_parsing(
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "gnu-sparse-v1.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        payload = (b"0\n0\n" * 100_000) + b"0\n"
        member = tarfile.TarInfo("payload.bin")
        member.pax_headers = {
            "GNU.sparse.major": "1",
            "GNU.sparse.minor": "0",
        }
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_rejects_pax_size_tunneling_hidden_sparse_metadata(
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    hidden_archive = io.BytesIO()
    with tarfile.open(fileobj=hidden_archive, mode="w", format=tarfile.PAX_FORMAT) as archive:
        sparse_map = (b"0\n0\n" * 100_000) + b"0\n"
        member = tarfile.TarInfo("hidden-sparse.bin")
        member.pax_headers = {
            "GNU.sparse.major": "1",
            "GNU.sparse.minor": "0",
        }
        member.size = len(sparse_map)
        archive.addfile(member, io.BytesIO(sparse_map))

    archive_path = tmp_path / "pax-size-tunnel.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        size_override = b"10 size=0\n"
        pax = tarfile.TarInfo("size-override")
        pax.type = tarfile.XHDTYPE
        pax.size = len(size_override)
        archive.addfile(pax, io.BytesIO(size_override))

        hidden_payload = hidden_archive.getvalue()
        carrier = tarfile.TarInfo("carrier.bin")
        carrier.size = len(hidden_payload)
        archive.addfile(carrier, io.BytesIO(hidden_payload))

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_rejects_pax_size_tunneling_hidden_global_pax(
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    hidden_archive = io.BytesIO()
    with tarfile.open(fileobj=hidden_archive, mode="w") as archive:
        global_pax_payload = b"18 comment=hidden\n"
        global_pax = tarfile.TarInfo("hidden-global-pax")
        global_pax.type = tarfile.XGLTYPE
        global_pax.size = len(global_pax_payload)
        archive.addfile(global_pax, io.BytesIO(global_pax_payload))
        regular = tarfile.TarInfo("payload.bin")
        regular.size = 0
        archive.addfile(regular, io.BytesIO())

    archive_path = tmp_path / "pax-size-global-tunnel.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        size_override = b"10 size=0\n"
        pax = tarfile.TarInfo("size-override")
        pax.type = tarfile.XHDTYPE
        pax.size = len(size_override)
        archive.addfile(pax, io.BytesIO(size_override))

        hidden_payload = hidden_archive.getvalue()
        carrier = tarfile.TarInfo("carrier.bin")
        carrier.size = len(hidden_payload)
        archive.addfile(carrier, io.BytesIO(hidden_payload))

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_bounds_local_pax_record_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "many-pax-records.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo("payload.bin")
        member.pax_headers = {"comment": "one", "atime": "two"}
        member.size = 0
        archive.addfile(member, io.BytesIO())

    monkeypatch.setattr(
        secure_docker_environment,
        "_WINDOWS_TAR_MAX_PAX_RECORDS_PER_HEADER",
        1,
    )

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_resets_pax_record_count_between_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "separate-pax-records.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for index in range(2):
            member = tarfile.TarInfo(f"payload-{index}.bin")
            member.pax_headers = {"comment": str(index)}
            member.size = 0
            archive.addfile(member, io.BytesIO())

    monkeypatch.setattr(
        secure_docker_environment,
        "_WINDOWS_TAR_MAX_PAX_RECORDS_PER_HEADER",
        1,
    )
    secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_accepts_bounded_ordinary_local_pax_metadata(
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "ordinary-pax.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        payload = b"content"
        member = tarfile.TarInfo("payload.bin")
        member.pax_headers = {"comment": "ordinary metadata"}
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_follows_bounded_local_pax_size_override(
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    payload = b"content"
    size_override = b"9 size=7\n"
    pax = tarfile.TarInfo("size-override")
    pax.type = tarfile.XHDTYPE
    pax.size = len(size_override)
    carrier = tarfile.TarInfo("payload.bin")
    carrier.size = 0
    archive_bytes = (
        pax.tobuf(format=tarfile.USTAR_FORMAT)
        + size_override.ljust(512, b"\0")
        + carrier.tobuf(format=tarfile.USTAR_FORMAT)
        + payload.ljust(512, b"\0")
        + b"\0" * 1024
    )
    archive_path = tmp_path / "pax-size-override.tar"
    archive_path.write_bytes(archive_bytes)

    secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)
    with tarfile.open(archive_path, mode="r:") as archive:
        member = archive.next()
        assert member is not None
        assert member.size == len(payload)
        extracted = archive.extractfile(member)
        assert extracted is not None
        assert extracted.read() == payload


def test_windows_tar_prescan_bounds_zero_byte_extension_chains(
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "extension-chain.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for index in range(secure_docker_environment._WINDOWS_TAR_MAX_EXTENSION_CHAIN + 1):
            extension = tarfile.TarInfo(f"pax-{index}")
            extension.type = tarfile.XHDTYPE
            extension.size = 0
            archive.addfile(extension, io.BytesIO())
        regular = tarfile.TarInfo("payload.bin")
        regular.size = 0
        archive.addfile(regular, io.BytesIO())

    with pytest.raises(ValueError):
        secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_tar_prescan_resets_extension_chain_after_regular_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "separated-extensions.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for index in range(2):
            extension = tarfile.TarInfo(f"pax-{index}")
            extension.type = tarfile.XHDTYPE
            extension.size = 0
            archive.addfile(extension, io.BytesIO())
            regular = tarfile.TarInfo(f"payload-{index}.bin")
            regular.size = 0
            archive.addfile(regular, io.BytesIO())

    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TAR_MAX_EXTENSION_CHAIN", 1)
    secure_docker_environment._prescan_uncompressed_tar_archive(archive_path)


def test_windows_archive_budget_counts_zero_byte_entries_before_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True
    environment._windows_container_name = "harbor-secure-windows"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        for index in range(3):
            member = tarfile.TarInfo(f"./empty-{index}.bin")
            member.size = 0
            archive.addfile(member, io.BytesIO())
    archive_payload = archive_buffer.getvalue()

    async def transfer(_command: list[str], **kwargs: object) -> None:
        Path(kwargs["output_path"]).write_bytes(archive_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(environment, "_run_trusted_transfer_command", transfer)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_ENTRY_DISK_BYTES", 10)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_DISK_RESERVE_BYTES", 0)
    monkeypatch.setattr(
        secure_docker_environment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=39),
    )
    target = tmp_path / "existing-target"
    target.mkdir()
    sentinel = target / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")

    with pytest.raises(RuntimeError, match=r"insufficient disk space.*archive extraction"):
        asyncio.run(environment.service_download_dir("/remote/tree", target))

    assert list(target.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize(("available_bytes", "should_succeed"), ((34, True), (33, False)))
def test_windows_archive_extraction_enforces_exact_estimated_byte_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_bytes: int,
    should_succeed: bool,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "boundary.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        payload = b"four"
        member = tarfile.TarInfo("./nested/payload.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    target = tmp_path / "extracted"
    target.mkdir()
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_ENTRY_DISK_BYTES", 10)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_DISK_RESERVE_BYTES", 0)
    monkeypatch.setattr(
        secure_docker_environment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=available_bytes),
    )

    if should_succeed:
        usage = secure_docker_environment._extract_regular_tar_archive(archive_path, target)
        assert usage.estimated_disk_bytes == available_bytes
        assert (target / "nested" / "payload.bin").read_bytes() == b"four"
    else:
        with pytest.raises(RuntimeError, match="insufficient disk space"):
            secure_docker_environment._extract_regular_tar_archive(archive_path, target)
        assert list(target.iterdir()) == []


@pytest.mark.parametrize("resource", ["entries", "inodes"])
def test_windows_archive_rejects_implicit_directory_resource_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    archive_path = tmp_path / "implicit-directories.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        member = tarfile.TarInfo("./one/two/payload.bin")
        member.size = 0
        archive.addfile(member, io.BytesIO())
    target = tmp_path / "extracted"
    target.mkdir()
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_ARTIFACT_DISK_RESERVE_BYTES", 0)
    if resource == "entries":
        monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TAR_MAX_FILESYSTEM_ENTRIES", 3)
        expected_error = "unsafe Windows container download archive"
    else:
        monkeypatch.setattr(secure_docker_environment, "_windows_artifact_inode_budget", lambda _path: 3)
        expected_error = "insufficient filesystem entries"

    with pytest.raises(RuntimeError, match=expected_error):
        secure_docker_environment._extract_regular_tar_archive(archive_path, target)

    assert list(target.iterdir()) == []


def test_windows_transfer_error_redacts_exact_short_sensitive_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    secret = "x"
    output_path = tmp_path / "failed-transfer.tar"

    class FailedTransferProcess:
        pid = 7644

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _ChunkStream([])
            self.stderr = _ChunkStream([("discarded-prefix-" * 20 + f"|{secret}|transfer failed").encode()])

        async def wait(self) -> int:
            self.returncode = 9
            return self.returncode

    async def create_subprocess(
        *_args: object,
        **kwargs: object,
    ) -> FailedTransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return FailedTransferProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TRANSFER_STDERR_MAX_BYTES", 64)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values={secret},
                output_path=output_path,
            )
        )

    assert secret not in str(caught.value)
    assert "diagnostics omitted" in str(caught.value)
    assert len(str(caught.value)) < 256
    assert not output_path.exists()


def test_windows_transfer_truncated_error_cannot_expose_secret_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    secret = "SYNTHETIC_SECRET_ABCDEF"
    output_path = tmp_path / "failed-transfer.tar"

    class FailedTransferProcess:
        pid = 7649

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _ChunkStream([])
            self.stderr = _ChunkStream([f"prefix|{secret}|FAIL".encode()])

        async def wait(self) -> int:
            self.returncode = 9
            return self.returncode

    async def create_subprocess(*_args: object, **kwargs: object) -> FailedTransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return FailedTransferProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TRANSFER_STDERR_MAX_BYTES", 16)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values={secret},
                output_path=output_path,
            )
        )

    rendered = str(caught.value)
    assert secret not in rendered
    assert "CRET_ABCDEF" not in rendered
    assert "diagnostics omitted" in rendered
    assert not output_path.exists()


def test_windows_transfer_truncated_error_with_repeated_secret_cannot_expose_earlier_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    secret = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567"
    leaked_suffix = "RSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567"
    output_path = tmp_path / "failed-transfer.tar"

    class FailedTransferProcess:
        pid = 7654

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _ChunkStream([])
            self.stderr = _ChunkStream([("X" + secret + "|" + secret + "!" * 8).encode()])

        async def wait(self) -> int:
            self.returncode = 9
            return self.returncode

    async def create_subprocess(*_args: object, **_kwargs: object) -> FailedTransferProcess:
        return FailedTransferProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment, "_WINDOWS_TRANSFER_STDERR_MAX_BYTES", 64)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={"PATH": os.environ["PATH"]},
                protected_values={secret},
                output_path=output_path,
            )
        )

    rendered = str(caught.value)
    assert secret not in rendered
    assert leaked_suffix not in rendered
    assert "diagnostics omitted" in rendered
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("environment_name", "secret"),
    (
        ("DOCKER_AUTH_CONFIG", "SYNTHETIC_DOCKER_AUTH_SECRET"),
        ("DOCKER_HOST", "tcp://synthetic-user:synthetic-pass@example.invalid:2376"),
        ("DOCKER_HOST", "tcp://synthetic-user:synthetic-pass@[malformed"),
        ("HTTPS_PROXY", "synthetic-user:synthetic-pass@example.invalid:8080"),
    ),
)
def test_windows_transfer_error_redacts_sensitive_docker_client_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    secret: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    output_path = tmp_path / "failed-transfer.tar"

    class FailedTransferProcess:
        pid = 7650

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _ChunkStream([])
            self.stderr = _ChunkStream([f"docker auth failed: {secret}".encode()])

        async def wait(self) -> int:
            self.returncode = 9
            return self.returncode

    async def create_subprocess(*_args: object, **kwargs: object) -> FailedTransferProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return FailedTransferProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_trusted_transfer_command(
                ["docker", "version"],
                process_environment={
                    "PATH": os.environ["PATH"],
                    environment_name: secret,
                },
                protected_values=set(),
                output_path=output_path,
            )
        )

    assert secret not in str(caught.value)
    assert _collision_safe_redaction_marker({secret}, include_short=True) in str(caught.value)
    assert not output_path.exists()


@pytest.mark.parametrize("operation", ["file", "dir"])
def test_service_download_rejects_empty_service_instead_of_crossing_into_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    spawned = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        nonlocal spawned
        spawned = True
        return _BufferedComposeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(ValueError, match="Invalid Docker Compose service name"):
        if operation == "file":
            asyncio.run(
                environment.service_download_file(
                    "/tmp/source",
                    tmp_path / "target",
                    service="",
                )
            )
        else:
            asyncio.run(
                environment.service_download_dir(
                    "/tmp/source",
                    tmp_path / "target",
                    service="",
                )
            )

    assert spawned is False


def test_sidecar_service_exec_rejects_windows_with_public_harbor_error(tmp_path: Path) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    environment._is_windows_container = True

    with pytest.raises(ServiceOperationsUnsupportedError, match="requested service: 'helper'"):
        asyncio.run(environment.service_exec("echo unsupported", service="helper"))


def test_sidecar_callback_base_exception_reaps_client_without_stopping_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    secret = "sidecar-callback-base-error-secret"
    callback_text: list[str] = []
    main_containment_calls = 0
    sidecar_containment_calls: list[str] = []
    child_processes: list[asyncio.subprocess.Process] = []
    real_create_subprocess = asyncio.create_subprocess_exec

    async def create_host_subprocess(*_args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await real_create_subprocess(
            sys.executable,
            "-c",
            f"print({secret!r})",
            **kwargs,
        )
        child_processes.append(process)
        return process

    async def contain_main() -> None:
        nonlocal main_containment_calls
        main_containment_calls += 1

    async def contain_sidecar(service: str) -> None:
        sidecar_containment_calls.append(service)

    callback_error = _CallbackBaseError("sidecar callback failed")

    async def on_output(text: str, _stream: str) -> None:
        callback_text.append(text)
        raise callback_error

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_host_subprocess)
    monkeypatch.setattr(environment, "_contain_main_container", contain_main)
    monkeypatch.setattr(environment, "_contain_sidecar_service", contain_sidecar, raising=False)

    async def exercise() -> None:
        with environment.scoped_output_callback(on_output):
            with pytest.raises(_CallbackBaseError, match="sidecar callback failed") as caught:
                await environment.service_exec(
                    "emit-sidecar-secret",
                    service="helper",
                    env={"SIDECAR_TOKEN": secret},
                )
            assert caught.value is callback_error

    asyncio.run(exercise())

    assert callback_text == [f"{_marker_for(secret)}\n"]
    assert main_containment_calls == 0
    assert sidecar_containment_calls == ["helper"]
    assert len(child_processes) == 1
    assert child_processes[0].returncode is not None


@pytest.mark.parametrize("reap_fails", [False, True])
def test_sidecar_containment_reaps_expired_client_before_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reap_fails: bool,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    actions: list[str] = []

    contained_snapshot = secure_docker_environment._RawServiceSnapshot(
        all_identities=("1" * 64,),
        running_identities=("1" * 64,),
    )

    async def contain(service: str) -> object:
        assert service == "helper"
        actions.append("contained")
        return contained_snapshot

    async def reap(
        _process: object,
        _communication: asyncio.Task[object],
        *,
        preserve_cancellation: bool,
    ) -> None:
        assert preserve_cancellation is False
        actions.append("reaped")
        if reap_fails:
            raise PermissionError("host client reap denied")

    async def restore(service: str, *, snapshot: object) -> bool:
        assert service == "helper"
        assert snapshot == contained_snapshot
        assert actions == ["contained", "reaped"]
        actions.append("restored")
        return True

    monkeypatch.setattr(environment, "_contain_sidecar_service", contain)
    monkeypatch.setattr(environment, "_restore_sidecar_service", restore)
    monkeypatch.setattr(secure_docker_environment, "_terminate_process_tree", reap)

    async def exercise() -> None:
        communication = asyncio.create_task(asyncio.sleep(0))
        if reap_fails:
            with pytest.raises(RuntimeError, match="containment and restoration") as caught:
                await environment._contain_main_and_reap_compose(
                    SimpleNamespace(),  # type: ignore[arg-type]
                    communication,
                    contain_service_on_interrupt="helper",
                    stop_main_on_interrupt=False,
                )
            assert isinstance(caught.value.__cause__, PermissionError)
        else:
            await environment._contain_main_and_reap_compose(
                SimpleNamespace(),  # type: ignore[arg-type]
                communication,
                contain_service_on_interrupt="helper",
                stop_main_on_interrupt=False,
            )

    asyncio.run(exercise())

    assert actions == (["contained", "reaped"] if reap_fails else ["contained", "reaped", "restored"])


@pytest.mark.parametrize("interrupt_mode", ["timeout", "cancel"])
def test_sidecar_interrupt_reaps_client_without_stopping_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_mode: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.05)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.05)
    environment = _initialized_secure_docker_environment(tmp_path)
    main_containment_calls = 0
    sidecar_containment_calls: list[str] = []
    child_processes: list[asyncio.subprocess.Process] = []
    child_created = asyncio.Event()
    real_create_subprocess = asyncio.create_subprocess_exec

    async def create_host_subprocess(*_args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await real_create_subprocess(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            **kwargs,
        )
        child_processes.append(process)
        child_created.set()
        return process

    async def contain_main() -> None:
        nonlocal main_containment_calls
        main_containment_calls += 1

    async def contain_sidecar(service: str) -> None:
        sidecar_containment_calls.append(service)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_host_subprocess)
    monkeypatch.setattr(environment, "_contain_main_container", contain_main)
    monkeypatch.setattr(environment, "_contain_sidecar_service", contain_sidecar, raising=False)

    async def exercise() -> None:
        task = asyncio.create_task(
            environment.service_exec(
                "sleep 30",
                service="helper",
                timeout_sec=0.02 if interrupt_mode == "timeout" else None,
            )
        )
        await asyncio.wait_for(child_created.wait(), timeout=1)
        if interrupt_mode == "timeout":
            with pytest.raises(RuntimeError, match="timed out"):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())

    assert main_containment_calls == 0
    assert sidecar_containment_calls == ["helper"]
    assert len(child_processes) == 1
    assert child_processes[0].returncode is not None


def test_upload_file_compose_cp_failure_forwards_exact_tar_bytes_to_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    source = tmp_path / "binary-payload.bin"
    source.write_bytes(b"\x00tar-fallback\xff\nwith spaces\x00")
    expected_tar = environment._platform._tar_file(source, "uploaded-payload.bin")
    calls: list[tuple[list[str], bool, bytes | None]] = []

    async def run_compose(
        command: list[str],
        check: bool = True,
        timeout_sec: float | None = None,
        stdin_data: bytes | None = None,
        on_output: object | None = None,
        **_kwargs: object,
    ) -> ExecResult:
        del timeout_sec, on_output
        calls.append((command, check, stdin_data))
        if command[0] == "cp":
            raise RuntimeError("compose cp deliberately unavailable")
        if "test" in command:
            return ExecResult(stdout=None, stderr=None, return_code=1)
        return ExecResult(stdout=None, stderr=None, return_code=0)

    monkeypatch.setattr(environment, "_run_docker_compose_command", run_compose)

    asyncio.run(environment.upload_file(source, "/tmp/uploaded-payload.bin"))

    assert calls[0][0][0] == "cp"
    tar_calls = [call for call in calls if "tar" in call[0]]
    assert len(tar_calls) == 1
    assert tar_calls[0][0] == [
        "exec",
        "-T",
        "-u",
        "root",
        MAIN_SERVICE_NAME,
        "tar",
        "-xf",
        "-",
        "-C",
        "/tmp",
    ]
    assert tar_calls[0][1] is True
    assert tar_calls[0][2] == expected_tar


def test_secure_public_exec_without_environment_still_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    process = _BufferedAndStreamedComposeProcess([b"plain output\n"])
    callback_chunks: list[str] = []

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedAndStreamedComposeProcess:
        return process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(on_output):
            return await environment.exec("emit-plain-output")

    result = asyncio.run(exercise())

    assert "".join(callback_chunks) == result.stdout == "plain output\n"
    assert process.communicate_inputs == []


def test_secure_public_exec_streams_redacted_handoff_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_secret = "secure-persistent-callback-secret"
    per_call_secret = "secure-per-call-callback-secret"
    scoped_secret = "secure-scoped-callback-secret"
    secrets = {persistent_secret, per_call_secret, scoped_secret}
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"PERSISTENT_TOKEN": persistent_secret},
    )
    callback_events: list[tuple[str, str, str]] = []
    subprocess_commands: list[tuple[object, ...]] = []
    handoff_process = _BufferedComposeProcess(stdout=b"", return_code=0)
    main_process = _BufferedAndStreamedComposeProcess(
        [
            f"persistent {persistent_secret}\n".encode(),
            f"per-call {per_call_secret}\n".encode(),
            f"scoped {scoped_secret}\n".encode(),
        ],
        return_code=7,
    )

    async def remove_handoff(_remote_path: str) -> None:
        return None

    async def create_subprocess(
        *args: object,
        **_kwargs: object,
    ) -> _BufferedComposeProcess | _BufferedAndStreamedComposeProcess:
        subprocess_commands.append(args)
        if 'umask 077; cat > "$1"' in args:
            return handoff_process
        if "chmod" in args or "chown" in args:
            return _BufferedComposeProcess(stdout=b"", return_code=0)
        return main_process

    async def outer_callback(text: str, stream: str) -> None:
        callback_events.append(("outer", text, stream))

    async def inner_callback(text: str, stream: str) -> None:
        callback_events.append(("inner", text, stream))

    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with (
            environment.scoped_exec_env({"SCOPED_TOKEN": scoped_secret}),
            environment.scoped_output_callback(outer_callback),
            environment.scoped_output_callback(inner_callback),
        ):
            return await environment.exec(
                "emit-handoff-output",
                env={"PER_CALL_TOKEN": per_call_secret},
            )

    result = asyncio.run(exercise())
    marker = _collision_safe_redaction_marker(secrets)
    expected = f"persistent {marker}\nper-call {marker}\nscoped {marker}\n"
    outer_chunks = [(text, stream) for label, text, stream in callback_events if label == "outer"]
    inner_chunks = [(text, stream) for label, text, stream in callback_events if label == "inner"]
    callback_output = "".join(text for text, _stream in outer_chunks)

    assert callback_output == "".join(text for text, _stream in inner_chunks) == result.stdout == expected
    assert outer_chunks == inner_chunks
    assert [label for label, _text, _stream in callback_events] == [
        label for _chunk in outer_chunks for label in ("outer", "inner")
    ]
    assert result.return_code == 7
    assert {stream for _text, stream in outer_chunks} == {"stdout"}
    assert handoff_process.communicate_inputs == []
    handoff_script = bytes(handoff_process.stdin.data).decode("utf-8")
    assert all(secret in handoff_script for secret in secrets)
    rendered_commands = "\n".join(" ".join(str(argument) for argument in command) for command in subprocess_commands)
    for secret in secrets:
        assert secret not in callback_output
        assert secret not in (result.stdout or "")
        assert secret not in rendered_commands


@pytest.mark.parametrize(
    "proxy_uri",
    [
        "https://secure%2Duser:secure%2Dpassword@proxy.invalid:8443",
        "secure-user:secure-password@proxy.invalid:8443",
    ],
    ids=["percent-encoded", "schemeless"],
)
def test_secure_public_exec_redacts_detached_proxy_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proxy_uri: str,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    handoff_process = _BufferedComposeProcess(stdout=b"", return_code=0)
    main_process = _BufferedAndStreamedComposeProcess(
        [b"proxy rejected secure-user with secure-password\n"],
        return_code=7,
    )
    callback_chunks: list[str] = []

    async def remove_handoff(_remote_path: str) -> None:
        return None

    async def create_subprocess(
        *args: object,
        **_kwargs: object,
    ) -> _BufferedComposeProcess | _BufferedAndStreamedComposeProcess:
        if 'umask 077; cat > "$1"' in args or "chmod" in args or "chown" in args:
            return handoff_process
        return main_process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(on_output):
            return await environment.exec(
                "emit-proxy-failure",
                env={"HTTPS_PROXY": proxy_uri},
            )

    result = asyncio.run(exercise())
    rendered = "".join(callback_chunks) + (result.stdout or "") + (result.stderr or "")

    assert result.return_code == 7
    assert "secure-user" not in rendered
    assert "secure-password" not in rendered


def test_secure_public_exec_output_limit_redacts_detached_proxy_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    diagnostic = b"ordinary-output-" * 10 + b"proxy rejected secure-limit-user with secure-limit-password\n"
    handoff_process = _BufferedComposeProcess(stdout=b"", return_code=0)
    main_process = _BufferedAndStreamedComposeProcess([diagnostic, b"overflow"])
    callback_chunks: list[str] = []

    async def remove_handoff(_remote_path: str) -> None:
        return None

    async def create_subprocess(
        *args: object,
        **_kwargs: object,
    ) -> _BufferedComposeProcess | _BufferedAndStreamedComposeProcess:
        if 'umask 077; cat > "$1"' in args or "chmod" in args or "chown" in args:
            return handoff_process
        return main_process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    async def contain(
        contained_process: asyncio.subprocess.Process,
        communication: asyncio.Task[object],
        **_kwargs: object,
    ) -> None:
        contained_process.terminate()
        with contextlib.suppress(BaseException):
            await communication

    monkeypatch.setattr(stream_redaction_module, "MAX_COMMAND_OUTPUT_BYTES", len(diagnostic), raising=False)
    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)
    monkeypatch.setattr(environment, "_contain_main_and_reap_compose", contain)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(on_output):
            return await environment.exec(
                "emit-proxy-failure",
                env={
                    "HTTPS_PROXY": "https://secure-limit-user:secure-limit-password@proxy.invalid:8443",
                },
            )

    with pytest.raises(CommandOutputLimitError) as caught:
        asyncio.run(exercise())

    rendered = "".join(callback_chunks) + str(caught.value)
    assert "ordinary-output" in "".join(callback_chunks)
    assert "secure-limit-user" not in rendered
    assert "secure-limit-password" not in rendered


def test_secure_public_exec_timeout_redacts_detached_proxy_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_secure_docker_environment(tmp_path)
    handoff_process = _BufferedComposeProcess(stdout=b"", return_code=0)
    callback_chunks: list[str] = []

    class OutputThenHangingProcess(_HangingComposeProcess):
        def __init__(self) -> None:
            super().__init__(pid=8844)
            stream = _FeedableStream()
            stream.feed_data(b"proxy rejected secure-timeout-user with secure-timeout-password\n")
            self.stdout = stream

        def terminate(self) -> None:
            self.stdout.feed_eof()
            super().terminate()

    main_process = OutputThenHangingProcess()

    async def remove_handoff(_remote_path: str) -> None:
        return None

    async def create_subprocess(
        *args: object,
        **_kwargs: object,
    ) -> _BufferedComposeProcess | OutputThenHangingProcess:
        if 'umask 077; cat > "$1"' in args or "chmod" in args or "chown" in args:
            return handoff_process
        return main_process

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    async def contain(
        contained_process: OutputThenHangingProcess,
        communication: asyncio.Task[object],
        **_kwargs: object,
    ) -> None:
        contained_process.terminate()
        with contextlib.suppress(BaseException):
            await communication

    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)
    monkeypatch.setattr(environment, "_contain_main_and_reap_compose", contain)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> ExecResult:
        with environment.scoped_output_callback(on_output):
            return await environment.exec(
                "emit-proxy-failure",
                env={
                    "HTTPS_PROXY": "https://secure-timeout-user:secure-timeout-password@proxy.invalid:8443",
                },
                timeout_sec=0.05,
            )

    with pytest.raises(RuntimeError, match="timed out") as caught:
        asyncio.run(exercise())

    rendered = "".join(callback_chunks) + str(caught.value)
    assert "proxy rejected" in "".join(callback_chunks)
    assert "secure-timeout-user" not in rendered
    assert "secure-timeout-password" not in rendered


def test_secure_handoff_scope_scrubs_compose_client_environment_and_resets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_secret = "persistent-context-scope-secret"
    per_call_secret = "per-call-context-scope-secret"
    scoped_secret = "scoped-context-scope-secret"
    override_secret = "explicit-override-context-secret"
    handoff_secrets = {persistent_secret, per_call_secret, scoped_secret}
    all_secrets = handoff_secrets | {override_secret}
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"PERSISTENT_TOKEN": persistent_secret},
    )
    environment._compose_task_env = {
        "COMPOSE_DIRECT": per_call_secret,
        "COMPOSE_WRAPPED": f"prefix:{persistent_secret}:suffix",
    }
    monkeypatch.setenv("PER_CALL_TOKEN", "host-value-hidden-by-active-name")
    monkeypatch.setenv("SCOPED_TOKEN", "another-host-value-hidden-by-active-name")
    monkeypatch.setenv("INHERITED_WRAPPED", f"prefix:{scoped_secret}:suffix")

    subprocess_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    callback_chunks: list[str] = []
    handoff_processes: list[_BufferedComposeProcess] = []

    async def create_subprocess(*args: object, **kwargs: object):
        subprocess_calls.append((args, kwargs))
        rendered = " ".join(str(argument) for argument in args)
        if "emit-context-scope-output" in rendered:
            raw_output = " ".join(sorted(all_secrets)) + "\n"
            return _BufferedAndStreamedComposeProcess([raw_output.encode()])

        process = _BufferedComposeProcess(stdout=b"")
        handoff_processes.append(process)
        return process

    async def exec_with_explicit_overrides(
        self,
        command: str,
        *,
        cwd: str | None,
        timeout_sec: int | None,
        user: str | int | None,
        secret_values: set[str] | None = None,
    ) -> ExecResult:
        del cwd, timeout_sec, user
        return await self._run_docker_compose_command(
            ["exec", "main", command],
            check=False,
            on_output=self._output_callback(),
            env_overrides={
                "EXPLICIT_OVERRIDE": override_secret,
                "EXPLICIT_OVERRIDE_WRAPPED": f"prefix:{override_secret}:suffix",
            },
            additional_secret_values=secret_values,
            stop_main_on_interrupt=True,
        )

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    monkeypatch.setattr(
        environment,
        "_exec_without_environment",
        MethodType(exec_with_explicit_overrides, environment),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> tuple[ExecResult, dict[str, str]]:
        with (
            environment.scoped_exec_env({"SCOPED_TOKEN": scoped_secret}),
            environment.scoped_output_callback(on_output),
        ):
            result = await environment.exec(
                "emit-context-scope-output",
                env={"PER_CALL_TOKEN": per_call_secret},
            )

        # Outside the secure handoff scope, retain the Harbor compatibility
        # behavior: Compose gets its ordinary environment and explicit
        # overrides. This also proves that the successful exec reset its scope.
        await environment._run_docker_compose_command(
            ["version"],
            check=False,
            env_overrides={"EXPLICIT_OVERRIDE": override_secret},
        )
        return result, subprocess_calls[-1][1]["env"]  # type: ignore[return-value]

    result, post_scope_environment = asyncio.run(exercise())

    handoff_inputs = [bytes(process.stdin.data) for process in handoff_processes if process.stdin.data]
    assert handoff_inputs and all(secret.encode() in handoff_inputs[0] for secret in handoff_secrets)
    assert "".join(callback_chunks) == result.stdout
    for secret in all_secrets:
        assert secret not in "".join(callback_chunks)
        assert secret not in (result.stdout or "")

    scoped_calls = subprocess_calls[:-1]
    assert len(scoped_calls) == 4  # stdin setup, chmod, command, final removal
    for arguments, kwargs in scoped_calls:
        process_environment = kwargs["env"]
        assert isinstance(process_environment, dict)
        assert not {"PERSISTENT_TOKEN", "PER_CALL_TOKEN", "SCOPED_TOKEN"} & process_environment.keys()
        assert "EXPLICIT_OVERRIDE" not in process_environment
        assert "EXPLICIT_OVERRIDE_WRAPPED" not in process_environment
        assert all(
            secret not in value
            for value in process_environment.values()
            if isinstance(value, str)
            for secret in all_secrets
        )
        rendered_arguments = " ".join(str(argument) for argument in arguments)
        assert all(secret not in rendered_arguments for secret in all_secrets)

    assert post_scope_environment["PERSISTENT_TOKEN"] == persistent_secret
    assert post_scope_environment["COMPOSE_DIRECT"] == per_call_secret
    assert post_scope_environment["INHERITED_WRAPPED"] == f"prefix:{scoped_secret}:suffix"
    assert post_scope_environment["EXPLICIT_OVERRIDE"] == override_secret


def test_secure_handoff_scope_scrubs_real_compose_client_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "real-child-scope-secret-marker-74891"
    audit_path = tmp_path / "compose-client-commands.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_path = bin_dir / "docker"
    docker_path.write_text(
        f"""#!{sys.executable}
import os
import sys

bad_names = {{"REAL_SCOPE_TOKEN"}} & os.environ.keys()
bad_values = [name for name, value in os.environ.items() if "scope-secret-marker" in value]
with open({str(audit_path)!r}, "a", encoding="utf-8") as audit:
    audit.write(" ".join(sys.argv[1:]) + "\\n")
sys.stdin.buffer.read()
if bad_names or bad_values:
    print("secure handoff leaked into compose client: " + ",".join(sorted(bad_names | set(bad_values))))
    raise SystemExit(91)
""",
        encoding="utf-8",
    )
    docker_path.chmod(0o700)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("REAL_SCOPE_TOKEN", secret)
    monkeypatch.setenv("REAL_INHERITED_WRAPPER", f"prefix:{secret}:suffix")

    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"REAL_SCOPE_TOKEN": secret},
    )
    environment._compose_task_env = {"REAL_COMPOSE_WRAPPER": f"prefix:{secret}:suffix"}

    result = asyncio.run(environment.exec("true"))

    assert result.return_code == 0
    commands = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 4
    assert any('umask 077; cat > "$1"' in command for command in commands)
    assert any(" chmod 600 " in f" {command} " for command in commands)
    assert any("bash -c" in command and "if ! ." in command for command in commands)
    assert any(" rm -f -- " in f" {command} " for command in commands)


def test_concurrent_secure_handoff_scopes_are_isolated_and_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_a = "concurrent-secure-scope-secret-alpha"
    secret_b = "concurrent-secure-scope-secret-bravo"
    monkeypatch.setenv("CONCURRENT_TOKEN_A", secret_a)
    monkeypatch.setenv("CONCURRENT_TOKEN_B", secret_b)
    environment = _initialized_secure_docker_environment(tmp_path)
    setup_barrier = asyncio.Event()
    setup_count = 0
    calls: list[dict[str, object]] = []
    remote_secrets: dict[str, str] = {}

    class CapturingStdin(_WritableStdin):
        def __init__(self, record: dict[str, object]) -> None:
            super().__init__()
            self._record = record

        async def drain(self) -> None:
            nonlocal setup_count
            stdin_data = bytes(self.data)
            self._record["stdin"] = stdin_data
            if stdin_data:
                rendered = " ".join(str(argument) for argument in self._record["args"])
                remote_path = next(
                    token for token in rendered.split() if token.startswith("/tmp/.skillevaluator-exec-env-")
                )
                decoded = stdin_data.decode()
                active_secret = secret_a if secret_a in decoded else secret_b
                remote_secrets[remote_path] = active_secret
                setup_count += 1
                if setup_count == 2:
                    setup_barrier.set()
                await setup_barrier.wait()

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedComposeProcess:
        record = {"args": args, "env": dict(kwargs["env"])}
        calls.append(record)
        process = _BufferedComposeProcess(stdout=b"")
        process.stdin = CapturingStdin(record)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> dict[str, str]:
        results = await asyncio.gather(
            environment.exec("true", env={"CONCURRENT_TOKEN_A": secret_a}),
            environment.exec("true", env={"CONCURRENT_TOKEN_B": secret_b}),
        )
        assert [result.return_code for result in results] == [0, 0]
        await environment._run_docker_compose_command(["version"], check=False)
        return calls[-1]["env"]  # type: ignore[return-value]

    post_scope_environment = asyncio.run(exercise())

    assert len(remote_secrets) == 2
    scoped_calls = calls[:-1]
    assert len(scoped_calls) == 8
    for record in scoped_calls:
        rendered = " ".join(str(argument) for argument in record["args"])
        remote_path = next(path for path in remote_secrets if path in rendered)
        active_secret = remote_secrets[remote_path]
        inactive_secret = secret_b if active_secret == secret_a else secret_a
        process_environment = record["env"]
        assert isinstance(process_environment, dict)
        assert all(active_secret not in value for value in process_environment.values() if isinstance(value, str))
        assert all(inactive_secret not in value for value in process_environment.values() if isinstance(value, str))

    assert post_scope_environment["CONCURRENT_TOKEN_A"] == secret_a
    assert post_scope_environment["CONCURRENT_TOKEN_B"] == secret_b


def test_secure_handoff_scope_does_not_reinsert_required_compose_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "required-compose-dependency-secret"
    monkeypatch.setenv("REQUIRED_HANDOFF_TOKEN", secret)
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"REQUIRED_HANDOFF_TOKEN": secret},
    )
    captured_environments: list[dict[str, str]] = []

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        process_environment = dict(kwargs["env"])
        captured_environments.append(process_environment)
        if "REQUIRED_HANDOFF_TOKEN" not in process_environment:
            return _BufferedComposeProcess(
                stdout=b"required variable REQUIRED_HANDOFF_TOKEN is missing",
                return_code=17,
            )
        return _BufferedComposeProcess(stdout=b"", return_code=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError, match="required variable REQUIRED_HANDOFF_TOKEN is missing"):
        asyncio.run(environment.exec("never-runs"))

    assert captured_environments
    for process_environment in captured_environments:
        assert "REQUIRED_HANDOFF_TOKEN" not in process_environment
        assert all(secret not in value for value in process_environment.values())


@pytest.mark.parametrize("handoff_mode", ["stdin", "file"])
def test_secure_handoff_scope_scrubs_resolved_nvidia_key_from_compose_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handoff_mode: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    resolved_secret = f"resolved-{handoff_mode}-nvidia-secret-93641"
    if handoff_mode == "stdin":
        requested_value = NVIDIA_BUILD_STDIN_SENTINEL
        monkeypatch.setattr(
            secure_docker_environment,
            "read_nvidia_build_key_from_stdin",
            lambda: resolved_secret,
        )
    else:
        requested_value = secure_docker_environment._NVIDIA_BUILD_FILE_SENTINEL
        key_path = tmp_path / "nvidia-api-key"
        key_path.write_text(resolved_secret, encoding="utf-8")
        monkeypatch.setenv(secure_docker_environment._NVIDIA_BUILD_KEY_FILE_ENV, str(key_path))

    monkeypatch.setenv("RESOLVED_NVIDIA_WRAPPER", f"prefix:{resolved_secret}:suffix")
    environment = _initialized_secure_docker_environment(tmp_path)
    calls: list[tuple[dict[str, str], _BufferedComposeProcess]] = []

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        process = _BufferedComposeProcess(stdout=b"")
        calls.append((dict(kwargs["env"]), process))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(environment.exec("true", env={"NVIDIA_API_KEY": requested_value}))

    assert result.return_code == 0
    assert len(calls) == 4
    assert any(resolved_secret.encode() in bytes(process.stdin.data) for _env, process in calls)
    for process_environment, _process in calls:
        assert "NVIDIA_API_KEY" not in process_environment
        assert all(resolved_secret not in value for value in process_environment.values())


def test_secure_handoff_setup_error_redacts_merged_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_secret = "secure-setup-persistent-secret"
    per_call_secret = "secure-setup-per-call-secret"
    secrets = {persistent_secret, per_call_secret}
    environment = _initialized_secure_docker_environment(
        tmp_path,
        persistent_env={"PERSISTENT_TOKEN": persistent_secret},
    )
    subprocess_commands: list[tuple[object, ...]] = []
    subprocess_environments: list[dict[str, str]] = []
    monkeypatch.setenv("SETUP_ERROR_WRAPPER", f"prefix:{per_call_secret}:suffix")

    async def remove_handoff(_remote_path: str) -> None:
        return None

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedComposeProcess:
        subprocess_commands.append(args)
        subprocess_environments.append(dict(kwargs["env"]))
        return _BufferedComposeProcess(
            stdout=f"setup failed {persistent_secret} {per_call_secret}\n".encode(),
            return_code=7,
        )

    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> RuntimeError:
        try:
            await environment.exec(
                "never-runs",
                env={"PER_CALL_TOKEN": per_call_secret},
            )
        except RuntimeError as caught:
            await environment._run_docker_compose_command(["version"], check=False)
            return caught
        raise AssertionError("secure setup failure did not propagate")

    caught = asyncio.run(exercise())
    detail = str(caught)
    marker = _collision_safe_redaction_marker(secrets)
    assert f"setup failed {marker} {marker}\n" in detail
    for secret in secrets:
        assert secret not in detail
        assert secret not in "\n".join(
            " ".join(str(argument) for argument in command) for command in subprocess_commands
        )
    for process_environment in subprocess_environments[:-1]:
        assert all(secret not in value for value in process_environment.values() for secret in secrets)
    assert subprocess_environments[-1]["PERSISTENT_TOKEN"] == persistent_secret
    assert subprocess_environments[-1]["SETUP_ERROR_WRAPPER"] == f"prefix:{per_call_secret}:suffix"


def test_secure_handoff_marker_exhaustion_fails_before_container_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    monkeypatch.setattr(
        secure_docker_environment,
        "_REDACTION_SENTINEL_CANDIDATES",
        (),
    )
    monkeypatch.setattr(
        secure_docker_environment,
        "unicodedata",
        SimpleNamespace(category=lambda _candidate: "Cc"),
    )
    occupied_private_use = "".join(
        chr(codepoint)
        for candidate_range in (
            range(0xE000, 0xF900),
            range(0xF0000, 0xFFFFE),
            range(0x100000, 0x10FFFE),
        )
        for codepoint in candidate_range
    )
    environment = _initialized_secure_docker_environment(tmp_path)
    handoff_started = False
    cleanup_attempted = False

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        nonlocal handoff_started
        handoff_started = True
        return _BufferedComposeProcess()

    async def remove_handoff(_remote_path: str) -> None:
        nonlocal cleanup_attempted
        cleanup_attempted = True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)

    with pytest.raises(RuntimeError, match="Could not construct a collision-safe redaction marker") as caught:
        asyncio.run(
            environment.exec(
                "never-runs",
                env={"SECRET": occupied_private_use},
            )
        )

    assert not handoff_started
    assert not cleanup_attempted
    assert occupied_private_use not in str(caught.value)


@pytest.mark.parametrize("secret", ["x", "hunter2", "secure-public-callback-failure-secret"])
@pytest.mark.parametrize("error_type", [_CallbackBaseError, asyncio.CancelledError])
def test_secure_public_exec_preserves_scoped_callback_failure_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
    secret: str,
) -> None:
    monkeypatch.setenv("CALLBACK_FAILURE_WRAPPER", f"prefix:{secret}:suffix")
    environment = _initialized_secure_docker_environment(tmp_path)
    main_process = _BufferedAndStreamedComposeProcess(
        [f"output {secret}\n".encode(), b"unread tail\n"],
    )
    callback_errors: list[BaseException] = []
    callback_chunks: list[str] = []
    containment_calls: list[bool] = []
    removed_handoffs: list[str] = []
    subprocess_environments: list[dict[str, str]] = []
    handoff_process = _BufferedComposeProcess(stdout=b"", return_code=0)

    async def remove_handoff(remote_path: str) -> None:
        removed_handoffs.append(remote_path)

    async def create_subprocess(
        *args: object,
        **kwargs: object,
    ) -> _BufferedComposeProcess | _BufferedAndStreamedComposeProcess:
        subprocess_environments.append(dict(kwargs["env"]))
        if "version" in args:
            return _BufferedComposeProcess(stdout=b"", return_code=0)
        if 'umask 077; cat > "$1"' in args:
            return handoff_process
        if "chmod" in args or "chown" in args:
            return _BufferedComposeProcess(stdout=b"", return_code=0)
        return main_process

    async def contain_and_reap(
        _process: object,
        communication: asyncio.Task[object],
        *,
        contain_service_on_interrupt: str | None = None,
        stop_main_on_interrupt: bool,
    ) -> None:
        assert contain_service_on_interrupt is None
        containment_calls.append(stop_main_on_interrupt)
        await communication

    async def failing_callback(text: str, _stream: str) -> None:
        callback_chunks.append(text)
        error = error_type(f"callback rejected {text}")
        callback_errors.append(error)
        raise error

    monkeypatch.setattr(environment, "_remove_handoff", remove_handoff)
    monkeypatch.setattr(environment, "_contain_main_and_reap_compose", contain_and_reap)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> BaseException:
        try:
            with environment.scoped_output_callback(failing_callback):
                await environment.exec(
                    "emit-output",
                    env={"SECRET": secret},
                )
        except BaseException as caught:
            assert isinstance(caught, error_type)
            await environment._run_docker_compose_command(["version"], check=False)
            return caught
        raise AssertionError("callback failure did not propagate")

    caught = asyncio.run(exercise())

    assert callback_errors and caught is callback_errors[0]
    marker = _collision_safe_redaction_marker({secret}, include_short=True)
    expected_callback = f"output {marker}" + ("\n" if len(secret) == 1 else "")
    assert callback_chunks == [expected_callback]
    assert str(caught) == f"callback rejected {callback_chunks[0]}"
    assert secret not in str(caught)
    assert containment_calls == [True]
    assert len(removed_handoffs) == 1
    assert main_process.returncode is not None
    for process_environment in subprocess_environments[:-1]:
        for value in process_environment.values():
            if len(secret) >= 8:
                assert secret not in value
            else:
                assert value != secret
    assert subprocess_environments[-1]["CALLBACK_FAILURE_WRAPPER"] == f"prefix:{secret}:suffix"


def test_secure_handoff_scope_resets_after_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secure-cleanup-failure-scope-secret"
    monkeypatch.setenv("CLEANUP_FAILURE_WRAPPER", f"prefix:{secret}:suffix")
    environment = _initialized_secure_docker_environment(tmp_path)
    subprocess_environments: list[dict[str, str]] = []

    async def create_subprocess(*_args: object, **kwargs: object) -> _BufferedComposeProcess:
        subprocess_environments.append(dict(kwargs["env"]))
        return _BufferedComposeProcess(stdout=b"", return_code=0)

    original_remove_handoff = environment._remove_handoff

    async def failed_remove_handoff(remote_path: str) -> None:
        await original_remove_handoff(remote_path)
        raise RuntimeError("forced final handoff cleanup failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(environment, "_remove_handoff", failed_remove_handoff)

    async def exercise() -> RuntimeError:
        try:
            await environment.exec("true", env={"CLEANUP_SECRET": secret})
        except RuntimeError as caught:
            await environment._run_docker_compose_command(["version"], check=False)
            return caught
        raise AssertionError("final handoff cleanup failure did not propagate")

    caught = asyncio.run(exercise())

    assert "could not confirm removal of Docker environment handoff" in str(caught)
    assert len(subprocess_environments) == 5
    assert all(
        secret not in value
        for process_environment in subprocess_environments[:-1]
        for value in process_environment.values()
    )
    assert subprocess_environments[-1]["CLEANUP_FAILURE_WRAPPER"] == f"prefix:{secret}:suffix"


@pytest.mark.parametrize("interrupt_mode", ["timeout", "cancel"])
def test_secure_handoff_scope_covers_containment_and_resets_after_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_mode: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    secret = f"secure-{interrupt_mode}-containment-scope-secret"
    monkeypatch.setenv("INTERRUPT_SCOPE_WRAPPER", f"prefix:{secret}:suffix")
    environment = _initialized_secure_docker_environment(tmp_path)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)
    subprocess_calls: list[tuple[tuple[object, ...], dict[str, str]]] = []
    hanging_process = _HangingComposeProcess(pid=7049)

    async def create_subprocess(*args: object, **kwargs: object):
        subprocess_calls.append((args, dict(kwargs["env"])))
        if "never-ending-scope-command" in " ".join(str(argument) for argument in args):
            return hanging_process
        return _BufferedComposeProcess(stdout=b"", return_code=0)

    def killpg(pid: int, value: signal.Signals) -> None:
        assert pid == hanging_process.pid
        hanging_process.finish(-value)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)

    async def worker() -> tuple[BaseException, dict[str, str]]:
        try:
            await environment.exec(
                "never-ending-scope-command",
                env={"INTERRUPT_SECRET": secret},
                timeout_sec=0.01 if interrupt_mode == "timeout" else None,
            )
        except BaseException as caught:
            await environment._run_docker_compose_command(["version"], check=False)
            return caught, subprocess_calls[-1][1]
        raise AssertionError("interrupted secure command did not propagate")

    async def exercise() -> tuple[BaseException, dict[str, str]]:
        task = asyncio.create_task(worker())
        await asyncio.wait_for(hanging_process.started.wait(), timeout=1)
        if interrupt_mode == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        return await asyncio.wait_for(task, timeout=1)

    caught, post_scope_environment = asyncio.run(exercise())

    if interrupt_mode == "timeout":
        assert isinstance(caught, RuntimeError)
        assert "timed out" in str(caught)
    else:
        assert isinstance(caught, asyncio.CancelledError)
    assert len(subprocess_calls) == 6
    scoped_calls = subprocess_calls[:-1]
    rendered_commands = [" ".join(str(argument) for argument in arguments) for arguments, _env in scoped_calls]
    assert any(
        "container ls" in command and "label=com.docker.compose.service=main" in command
        for command in rendered_commands
    )
    assert any(" rm -f -- " in f" {command} " for command in rendered_commands)
    assert all(
        secret not in value
        for _arguments, process_environment in scoped_calls
        for value in process_environment.values()
    )
    assert post_scope_environment["INTERRUPT_SCOPE_WRAPPER"] == f"prefix:{secret}:suffix"


def test_runner_additional_secret_values_redact_callback_result_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "handoff-persistent-secret",
        "persistent-secret",
    }
    additional_secret_values = [
        "handoff-persistent-secret",
        "persistent-secret",
        "handoff-persistent-secret",
    ]
    raw_output = " ".join(sorted(secrets)) + "\n"
    environment = _initialized_docker_environment(tmp_path)
    callback_chunks: list[list[str]] = [[], []]
    callback_index = 0
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedAndStreamedComposeProcess:
        captured.append((args, kwargs))
        return _BufferedAndStreamedComposeProcess(
            [raw_output.encode()],
            return_code=7,
        )

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks[callback_index].append(text)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    async def exercise() -> tuple[ExecResult, RuntimeError]:
        nonlocal callback_index
        result = await environment._run_docker_compose_command(
            ["exec", "main", "emit-output"],
            check=False,
            on_output=on_output,
            additional_secret_values=additional_secret_values,
        )
        callback_index = 1
        with pytest.raises(RuntimeError) as caught:
            await environment._run_docker_compose_command(
                ["exec", "main", "emit-output"],
                on_output=on_output,
                additional_secret_values=additional_secret_values,
            )
        return result, caught.value

    result, error = asyncio.run(exercise())
    marker = _collision_safe_redaction_marker(secrets)
    expected = f"{marker} {marker}\n"

    assert "".join(callback_chunks[0]) == result.stdout == expected
    assert "".join(callback_chunks[1]) == expected
    assert f"Stdout: {expected}." in str(error)
    for rendered in (result.stdout or "", "".join(callback_chunks[0]), "".join(callback_chunks[1]), str(error)):
        for secret in secrets:
            assert secret not in rendered
    for arguments, kwargs in captured:
        rendered_arguments = " ".join(str(argument) for argument in arguments)
        rendered_environment = "\n".join(f"{name}={value}" for name, value in kwargs["env"].items())
        for secret in secrets:
            assert secret not in rendered_arguments
            assert secret not in rendered_environment


def test_exec_uses_name_only_argv_and_subprocess_override(tmp_path: Path) -> None:
    environment = _initialized_docker_environment(
        tmp_path,
        persistent_env={"DATABASE_URL": "old-value"},
    )
    captured: dict[str, object] = {}

    async def _capture(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        on_output: object | None = None,
        *,
        env_overrides=None,
        additional_secret_values=None,
        exact_secret_values=None,
        stop_main_on_interrupt: bool = False,
    ) -> ExecResult:
        del self, check, timeout_sec
        captured["command"] = command
        captured["env"] = env_overrides
        captured["on_output"] = on_output
        captured["additional_secret_values"] = additional_secret_values
        captured["exact_secret_values"] = exact_secret_values
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
    assert captured["on_output"] is None
    assert captured["exact_secret_values"] == {
        _SENTINEL,
        "new-value",
    }
    assert captured["additional_secret_values"] is None
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
    environment = _initialized_docker_environment(tmp_path)
    environment._compose_env_vars = MethodType(
        lambda _self, **_kwargs: {"PATH": "/usr/bin"},
        environment,
    )
    captured: dict[str, object] = {}

    async def _create_subprocess(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return _BufferedComposeProcess(
            stdout=f"failure included {_SENTINEL}".encode(),
            return_code=7,
        )

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
    assert _marker_for(_SENTINEL) in str(caught.value)


def test_compose_check_false_redacts_success_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        return _BufferedComposeProcess(
            stdout=f"stdout {_SENTINEL}\nstderr {_SENTINEL}".encode(),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    result = asyncio.run(
        environment._run_docker_compose_command(
            ["exec", "-e", "DATABASE_URL", "main", "true"],
            check=False,
            env_overrides={"DATABASE_URL": _SENTINEL},
        )
    )

    marker = _marker_for(_SENTINEL)
    assert result.stdout == f"stdout {marker}\nstderr {marker}"
    assert result.stderr is None


def test_compose_stdin_handoff_redacts_secret_without_argv_or_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(SkillEvaluatorDockerEnvironment)
    environment.session_id = "secure-stdin-test"
    environment.environment_name = "secure-stdin-test"
    environment.environment_dir = tmp_path
    environment._resources_compose_path = None
    environment._env_compose_path = None
    environment._mounts_compose_path = None
    environment._use_prebuilt = True
    environment._is_windows_container = False
    environment.extra_docker_compose_paths = []
    environment._network_policy = SimpleNamespace(network_mode="public")
    environment._enable_egress_control = False
    environment._egress_control_services_compose_path = None
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    captured: dict[str, object] = {}

    process = _BufferedComposeProcess(
        stdout=f"failure included {_SENTINEL}".encode(),
        return_code=9,
    )

    async def create_subprocess(*args: object, **kwargs: object) -> _BufferedComposeProcess:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["stdin"] = kwargs["stdin"]
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["exec", "-T", "main", "true"],
                stdin_data=b"private-stream-payload",
                additional_secret_values={_SENTINEL},
            )
        )

    assert bytes(process.stdin.data) == b"private-stream-payload"
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
    assert _redact("token=abcdefgh", {"abcdefgh"}) == f"token={_marker_for('abcdefgh')}"


def test_compose_redacts_long_secrets_without_rewriting_short_env_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    long_secret = "abcdefgh"

    async def create_subprocess(*_args: object, **_kwargs: object) -> _BufferedComposeProcess:
        return _BufferedComposeProcess(
            stdout=f"http://127.0.0.1:41927\nstderr {long_secret}".encode(),
        )

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

    assert result.stdout == f"http://127.0.0.1:41927\nstderr {_marker_for(long_secret)}"
    assert result.stderr is None


@pytest.mark.parametrize("secret", ["x", "hunter2", "abcdefgh"])
def test_compose_exact_sensitive_values_redact_check_errors_at_every_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secret: str,
) -> None:
    environment = _initialized_docker_environment(tmp_path)

    async def create_subprocess(
        *_args: object,
        **_kwargs: object,
    ) -> _BufferedComposeProcess:
        return _BufferedComposeProcess(
            stdout=f"failure|{secret}|\n".encode(),
            return_code=9,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            environment._run_docker_compose_command(
                ["version"],
                check=True,
                exact_secret_values={secret},
            )
        )

    marker = _collision_safe_redaction_marker({secret}, include_short=True)
    assert secret not in str(caught.value)
    assert marker in str(caught.value)


def test_compose_cancellation_reaps_process_tree_even_when_repeated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_docker_environment(tmp_path)
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01, raising=False)

    async def run_cancelled() -> list[str]:
        actions: list[str] = []
        process = _HangingComposeProcess(pid=4343)

        async def create_subprocess(*_args: object, **_kwargs: object) -> _HangingComposeProcess:
            return process

        def killpg(_pid: int, value: signal.Signals) -> None:
            if value == signal.SIGTERM:
                actions.append("terminate")
            else:
                actions.append("kill")
                process.finish(-signal.SIGKILL)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg, raising=False)
        task = asyncio.create_task(environment._run_docker_compose_command(["exec", "main", "sleep", "30"]))
        await asyncio.wait_for(process.started.wait(), timeout=1)
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

    environment = _initialized_docker_environment(tmp_path)
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def run_cancelled() -> tuple[list[str], list[tuple[object, ...]]]:
        actions: list[str] = []
        commands: list[tuple[object, ...]] = []
        original = _HangingComposeProcess(pid=4545)

        async def create_subprocess(*args: object, **_kwargs: object) -> _HangingComposeProcess:
            commands.append(args)
            return original

        async def contain_main() -> None:
            actions.extend(("container-stop", "container-remove"))

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == original.pid
            actions.append(f"host-{value.name.lower()}")
            if value == signal.SIGKILL:
                original.finish(-signal.SIGKILL)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(environment, "_contain_main_container", contain_main)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        task = asyncio.create_task(
            environment.exec(
                "sleep 30",
                env={"NVIDIA_API_KEY": "credential-for-cancellation-test"},
                timeout_sec=0.01 if interrupt_mode == "timeout" else None,
            )
        )
        await asyncio.wait_for(original.started.wait(), timeout=1)
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

    assert len(commands) == 1
    rendered_original = " ".join(str(arg) for arg in commands[0])
    assert ".skillevaluator-exec-" not in rendered_original
    assert "SKILLEVALUATOR_EXEC_TOKEN" not in rendered_original
    assert actions.index("container-stop") < actions.index("host-sigterm")
    assert actions.index("container-remove") < actions.index("host-sigterm")


def test_exec_cancelled_during_process_creation_still_stops_main_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_docker_environment(tmp_path)
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
        original = _HangingComposeProcess(pid=4645)

        async def create_subprocess(*args: object, **_kwargs: object) -> _HangingComposeProcess:
            commands.append(args)
            creation_started.set()
            await release_creation.wait()
            return original

        async def contain_main() -> None:
            actions.append("container-stop-started")
            stop_started.set()
            await release_stop.wait()
            actions.extend(("container-stop-finished", "container-remove-finished"))

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == original.pid
            actions.append(f"host-{value.name.lower()}")
            if value == signal.SIGKILL:
                original.finish(-signal.SIGKILL)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(environment, "_contain_main_container", contain_main)
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

    assert len(commands) == 1
    assert actions.index("container-remove-finished") < actions.index("host-sigterm")


@pytest.mark.parametrize("interrupt_mode", ["cancel", "timeout"])
def test_interrupted_exec_fails_closed_when_main_container_containment_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_mode: str,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_docker_environment(tmp_path)
    environment._compose_env_vars = MethodType(lambda _self, **_kwargs: {"PATH": "/usr/bin"}, environment)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def run_timeout() -> tuple[list[tuple[object, ...]], list[signal.Signals]]:
        commands: list[tuple[object, ...]] = []
        signals: list[signal.Signals] = []
        original = _HangingComposeProcess(pid=4745)

        async def create_subprocess(*args: object, **_kwargs: object) -> _HangingComposeProcess:
            commands.append(args)
            return original

        async def contain_main() -> None:
            raise PermissionError("raw Docker containment denied")

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == original.pid
            signals.append(value)
            if value == signal.SIGKILL:
                original.finish(-signal.SIGKILL)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(environment, "_contain_main_container", contain_main)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        task = asyncio.create_task(
            environment._run_docker_compose_command(
                ["exec", "main", "sleep", "30"],
                timeout_sec=0.01 if interrupt_mode == "timeout" else None,
                stop_main_on_interrupt=True,
            )
        )
        await asyncio.wait_for(original.started.wait(), timeout=1)
        if interrupt_mode == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
        else:
            with pytest.raises(RuntimeError, match="timed out") as caught:
                await task
        assert caught.value.__cause__ is not None
        assert "containment" in str(caught.value.__cause__)
        assert any("containment could not be confirmed" in note for note in caught.value.__notes__)
        return commands, signals

    commands, signals = asyncio.run(run_timeout())

    assert len(commands) == 1
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
        with pytest.raises(RuntimeError, match="could not confirm Docker client process termination"):
            await cleanup
        return finished_within_bound

    assert asyncio.run(run_cleanup()) is True


def test_raw_docker_creation_cancellation_resolves_process_race_and_reaps_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def exercise() -> list[signal.Signals]:
        creation_started = asyncio.Event()
        release_creation = asyncio.Event()
        communication_done: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()
        signals: list[signal.Signals] = []

        class RawProcess:
            pid = 8451
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                return await asyncio.shield(communication_done)

        process = RawProcess()

        async def create_subprocess(*_args: object, **_kwargs: object) -> RawProcess:
            creation_started.set()
            try:
                await release_creation.wait()
            except asyncio.CancelledError:
                await release_creation.wait()
            return process

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == process.pid
            signals.append(value)
            if value == signal.SIGKILL:
                process.returncode = -9
                if not communication_done.done():
                    communication_done.set_result((b"", b""))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        task = asyncio.create_task(environment._run_trusted_docker_command(["version"]))
        await asyncio.wait_for(creation_started.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release_creation.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert process.returncode == -9
        return signals

    assert asyncio.run(exercise()) == [signal.SIGTERM, signal.SIGKILL]


def test_raw_docker_creation_timeout_is_total_deadline_bounded_with_late_reaper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    monkeypatch.setattr(secure_docker_environment, "_RAW_DOCKER_COMMAND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_RAW_LIFECYCLE_TOTAL_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def exercise() -> tuple[float, list[signal.Signals]]:
        creation_started = asyncio.Event()
        creation_cancelled = asyncio.Event()
        release_creation = asyncio.Event()
        process_reaped = asyncio.Event()
        communication_done: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()
        signals: list[signal.Signals] = []

        class RawProcess:
            pid = 8453
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                return await asyncio.shield(communication_done)

        process = RawProcess()

        async def create_subprocess(*_args: object, **_kwargs: object) -> RawProcess:
            creation_started.set()
            try:
                await release_creation.wait()
            except asyncio.CancelledError:
                creation_cancelled.set()
                await release_creation.wait()
            return process

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == process.pid
            signals.append(value)
            if value == signal.SIGKILL:
                process.returncode = -9
                if not communication_done.done():
                    communication_done.set_result((b"", b""))
                process_reaped.set()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        started_at = asyncio.get_running_loop().time()
        with (
            secure_docker_environment._raw_lifecycle_deadline_scope(),
            pytest.raises(RuntimeError, match="trusted Docker client creation timed out") as caught,
        ):
            await environment._run_trusted_docker_command(["version"])
        elapsed = asyncio.get_running_loop().time() - started_at
        assert creation_started.is_set()
        assert creation_cancelled.is_set()
        assert caught.value.__cause__ is not None
        assert any("late-process reaper" in note for note in caught.value.__cause__.__notes__)
        release_creation.set()
        await asyncio.wait_for(process_reaped.wait(), timeout=1)
        await asyncio.sleep(0)
        return elapsed, signals

    elapsed, signals = asyncio.run(exercise())

    assert elapsed < 0.1
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_raw_docker_command_timeout_kills_and_reaps_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import secure_docker_environment

    environment = _initialized_secure_docker_environment(tmp_path)
    monkeypatch.setattr(secure_docker_environment, "_RAW_DOCKER_COMMAND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_TERMINATE_SECONDS", 0.01)
    monkeypatch.setattr(secure_docker_environment, "_COMPOSE_KILL_SECONDS", 0.01)

    async def exercise() -> list[signal.Signals]:
        communication_done: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()
        signals: list[signal.Signals] = []

        class RawProcess:
            pid = 8452
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                return await asyncio.shield(communication_done)

        process = RawProcess()

        async def create_subprocess(*_args: object, **_kwargs: object) -> RawProcess:
            return process

        def killpg(pid: int, value: signal.Signals) -> None:
            assert pid == process.pid
            signals.append(value)
            if value == signal.SIGKILL:
                process.returncode = -9
                if not communication_done.done():
                    communication_done.set_result((b"", b""))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(secure_docker_environment.os, "killpg", killpg)
        with pytest.raises(RuntimeError, match="trusted Docker client command timed out"):
            await environment._run_trusted_docker_command(["version"])
        assert process.returncode == -9
        return signals

    assert asyncio.run(exercise()) == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.integration
def test_real_docker_main_sidecar_stdin_streaming_and_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    docker_info = subprocess.run(
        ["docker", "info", "--format", "{{.OSType}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if docker_info.returncode != 0 or docker_info.stdout.strip() != "linux":
        pytest.skip("requires a running Linux Docker daemon")

    environment_dir = tmp_path / "real-sidecar-environment"
    environment_dir.mkdir()
    helper_image = "alpine:3.20"
    safe_host_config = "safe-host-config-43127"
    compose_path = environment_dir / "docker-compose.yaml"
    compose_content = (
        'version: "3.8"\n'
        "services:\n"
        "  helper:\n"
        "    image: ${HELPER_IMAGE:?required}\n"
        "    environment:\n"
        "      SAFE_HOST_CONFIG: ${SAFE_HOST_CONFIG:?required}\n"
        '    command: ["sh", "-c", "trap : TERM INT; while :; do sleep 3600; done"]\n'
        "  observer:\n"
        "    image: alpine:3.20\n"
        '    command: ["sh", "-c", "trap : TERM INT; while :; do sleep 3600; done"]\n'
    )
    protected_compose_content = compose_content.replace(
        "      SAFE_HOST_CONFIG: ${SAFE_HOST_CONFIG:?required}\n",
        "      SAFE_HOST_CONFIG: ${SAFE_HOST_CONFIG:?required}\n      API_TOKEN: ${API_TOKEN:?required}\n",
    )
    compose_path.write_text(compose_content, encoding="utf-8")
    project = f"skillevaluator-sidecar-{uuid.uuid4().hex[:10]}"
    cleanup_compose = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--project-directory",
        str(environment_dir),
        "-f",
        str(compose_path),
    ]

    def emergency_cleanup() -> None:
        compose_path.write_text(compose_content, encoding="utf-8")
        subprocess.run(
            [*cleanup_compose, "down", "--remove-orphans", "--volumes"],
            check=False,
            capture_output=True,
            env={
                **os.environ,
                "HELPER_IMAGE": helper_image,
                "SAFE_HOST_CONFIG": safe_host_config,
            },
            timeout=60,
        )

    request.addfinalizer(emergency_cleanup)

    main_persistent_secret = "real-main-persistent-secret-21679"
    main_task_secret = "real-main-task-secret-32780"
    main_scoped_secret = "real-main-scoped-secret-43891"
    main_exec_secret = "real-main-exec-secret-54902"
    sidecar_secret = "real-sidecar-explicit-secret-65013"
    sidecar_reused_secret = "real-sidecar-reused-secret-76124"
    sidecar_control_environment = {
        "PATH": "/sidecar-only-bin",
        "HOME": "/sidecar-only-home",
        "DOCKER_HOST": "tcp://sidecar-only.invalid:2376",
        "DOCKER_CONFIG": "/sidecar-only-docker-config",
        "COMPOSE_FILE": "/sidecar-only-compose.yaml",
        "NORMAL_TOKEN": "real 'quoted' $dollar\nsecond-line-secret",
        "EMPTY_VALUE": "",
    }
    monkeypatch.setenv("REAL_SCOPED_ONLY", main_scoped_secret)
    monkeypatch.setenv("REAL_MAIN_WRAPPED", f"prefix:{main_persistent_secret}:suffix")
    monkeypatch.setenv("NVIDIA_API_KEY", NVIDIA_BUILD_STDIN_SENTINEL)
    monkeypatch.setenv(NVIDIA_BUILD_KEY_STDIN_ENV, "1")
    monkeypatch.setenv("SKILLEVALUATOR_NVIDIA_API_KEY_FILE", "/tmp/real-main-only-key")
    monkeypatch.setenv("SAFE_HOST_CONFIG", safe_host_config)
    monkeypatch.delenv("API_TOKEN", raising=False)

    environment = SkillEvaluatorSecureDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="real-sidecar-security",
        session_id=project,
        trial_paths=TrialPaths(tmp_path / "real-sidecar-trial"),
        task_env_config=EnvironmentConfig(
            docker_image="python:3.13-slim",
            workdir="/tmp",
            env={
                "REAL_TASK_ONLY": main_task_secret,
                "HELPER_IMAGE": helper_image,
            },
        ),
        persistent_env={"REAL_PERSISTENT_ONLY": main_persistent_secret},
    )
    real_create_subprocess = asyncio.create_subprocess_exec
    all_compose_clients: list[tuple[tuple[object, ...], dict[str, str], asyncio.subprocess.Process]] = []
    sidecar_clients: list[tuple[tuple[object, ...], dict[str, str], asyncio.subprocess.Process]] = []
    restore_compose_after_raw_sidecar_start = False

    async def capture_compose_clients(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        nonlocal restore_compose_after_raw_sidecar_start
        process = await real_create_subprocess(*args, **kwargs)
        all_compose_clients.append((args, dict(kwargs["env"]), process))
        rendered = tuple(str(argument) for argument in args)
        if "exec" in rendered and "helper" in rendered:
            sidecar_clients.append((args, dict(kwargs["env"]), process))
        if restore_compose_after_raw_sidecar_start and len(rendered) > 2 and rendered[1:3] == ("container", "start"):
            # Keep the model hostile through raw resolution, containment, and
            # restart spawn, then restore it before a serialized waiter enters.
            compose_path.write_text(compose_content, encoding="utf-8")
            restore_compose_after_raw_sidecar_start = False
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_compose_clients)

    async def exercise() -> None:
        nonlocal restore_compose_after_raw_sidecar_start
        started = False
        try:
            await environment.start(force_build=False)
            started = True

            main_callback: list[tuple[str, str]] = []

            async def on_main_output(text: str, stream: str) -> None:
                main_callback.append((text, stream))

            with environment.scoped_output_callback(on_main_output):
                main_result = await environment.service_exec(
                    "printf 'main-out:%s\\n' \"$REAL_MAIN_EXEC\"; printf 'main-error:%s\\n' \"$REAL_MAIN_EXEC\" >&2",
                    service=MAIN_SERVICE_NAME,
                    env={"REAL_MAIN_EXEC": main_exec_secret},
                )
            main_marker = _collision_safe_redaction_marker(
                {
                    main_persistent_secret,
                    main_exec_secret,
                    helper_image,
                    safe_host_config,
                },
            )
            assert "".join(text for text, _stream in main_callback) == main_result.stdout
            assert set((main_result.stdout or "").splitlines()) == {
                f"main-out:{main_marker}",
                f"main-error:{main_marker}",
            }
            assert {stream for _text, stream in main_callback} == {"stdout"}

            sidecar_callback: list[tuple[str, str]] = []

            async def on_sidecar_output(text: str, stream: str) -> None:
                sidecar_callback.append((text, stream))

            sidecar_command = (
                "printf 'sidecar-out:%s:%s:%s:%s:%s:%s\\n' \"$REAL_SIDECAR\" "
                '"$REAL_PERSISTENT_ONLY" "${REAL_TASK_ONLY-unset}" "${REAL_SCOPED_ONLY-unset}" '
                '"${HELPER_IMAGE-unset}" "$SAFE_HOST_CONFIG"; '
                "printf 'sidecar-error:%s\\n' \"$REAL_SIDECAR\" >&2; exit 7"
            )
            control_hash_command = (
                "path_hash=$(printf '%s' \"$PATH\" | /bin/busybox sha256sum); path_hash=${path_hash%% *}; "
                "home_hash=$(printf '%s' \"$HOME\" | /bin/busybox sha256sum); home_hash=${home_hash%% *}; "
                "host_hash=$(printf '%s' \"$DOCKER_HOST\" | /bin/busybox sha256sum); host_hash=${host_hash%% *}; "
                "config_hash=$(printf '%s' \"$DOCKER_CONFIG\" | /bin/busybox sha256sum); config_hash=${config_hash%% *}; "
                "compose_hash=$(printf '%s' \"$COMPOSE_FILE\" | /bin/busybox sha256sum); compose_hash=${compose_hash%% *}; "
                "normal_hash=$(printf '%s' \"$NORMAL_TOKEN\" | /bin/busybox sha256sum); normal_hash=${normal_hash%% *}; "
                "empty_hash=$(printf '%s' \"$EMPTY_VALUE\" | /bin/busybox sha256sum); empty_hash=${empty_hash%% *}; "
                "carrier=gone; /bin/busybox env | /bin/busybox grep -q '^SKILLEVALUATOR_SIDECAR_ENV_' "
                "&& carrier=found; "
                "printf 'control:%s:%s:%s:%s:%s:%s:%s carrier=%s\\n' "
                '"$path_hash" "$home_hash" "$host_hash" "$config_hash" "$compose_hash" "$normal_hash" '
                '"$empty_hash" "$carrier"'
            )
            sidecar_command = sidecar_command.removesuffix("; exit 7") + "; " + control_hash_command + "; exit 7"
            with (
                environment.scoped_exec_env({"REAL_SCOPED_ONLY": main_scoped_secret}),
                environment.scoped_output_callback(on_sidecar_output),
            ):
                sidecar_result = await environment.service_exec(
                    sidecar_command,
                    service="helper",
                    env={
                        "REAL_SIDECAR": sidecar_secret,
                        "REAL_PERSISTENT_ONLY": sidecar_reused_secret,
                        **sidecar_control_environment,
                    },
                )
            sidecar_marker = _collision_safe_redaction_marker(
                {
                    sidecar_secret,
                    sidecar_reused_secret,
                    helper_image,
                    safe_host_config,
                    *sidecar_control_environment.values(),
                },
            )
            assert "".join(text for text, _stream in sidecar_callback) == sidecar_result.stdout
            assert set((sidecar_result.stdout or "").splitlines()) == {
                f"sidecar-out:{sidecar_marker}:{sidecar_marker}:unset:unset:unset:{sidecar_marker}",
                f"sidecar-error:{sidecar_marker}",
                "control:"
                + ":".join(hashlib.sha256(value.encode()).hexdigest() for value in sidecar_control_environment.values())
                + " carrier=gone",
            }
            assert sidecar_result.return_code == 7
            assert {stream for _text, stream in sidecar_callback} == {"stdout"}
            sidecar_identity = await environment.service_exec(
                'printf \'%s:%s\' "$PWD" "$(id -u)"',
                service="helper",
                cwd="/tmp",
                user=0,
            )
            assert sidecar_identity.stdout == "/tmp:0"

            binary_payload = b"\x00real-binary-stdin\xff\nwith spaces\x00"
            binary_result = await environment._run_docker_compose_command(
                [
                    "exec",
                    "-T",
                    MAIN_SERVICE_NAME,
                    "python",
                    "-c",
                    "import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())",
                ],
                check=True,
                stdin_data=binary_payload,
            )
            assert binary_result.stdout == hashlib.sha256(binary_payload).hexdigest() + "\n"

            upload_source = tmp_path / "real-upload-payload.bin"
            upload_payload = b"\x00real-tar-upload\xff\nwith spaces\x00"
            upload_source.write_bytes(upload_payload)
            original_run = environment._run_docker_compose_command
            cp_failed = False

            async def force_tar_fallback(
                command: list[str],
                *args: object,
                **kwargs: object,
            ) -> ExecResult:
                nonlocal cp_failed
                if command and command[0] == "cp" and not cp_failed:
                    cp_failed = True
                    raise RuntimeError("force real tar-stream fallback")
                return await original_run(command, *args, **kwargs)

            with monkeypatch.context() as upload_patch:
                upload_patch.setattr(environment, "_run_docker_compose_command", force_tar_fallback)
                await environment.upload_file(upload_source, "/tmp/real-uploaded-payload.bin")
            assert cp_failed is True
            upload_result = await environment.service_exec(
                'python -c \'import hashlib; print(hashlib.sha256(open("/tmp/real-uploaded-payload.bin", "rb").read()).hexdigest())\'',
                service=MAIN_SERVICE_NAME,
            )
            assert upload_result.stdout == hashlib.sha256(upload_payload).hexdigest() + "\n"

            main_download_payload = b"\x00main-download\xff"
            helper_download_payload = b"\x00helper-download\xfe"
            main_download_script = (
                "from pathlib import Path; "
                f'Path("/tmp/main-download.bin").write_bytes({main_download_payload!r}); '
                'Path("/tmp/main-tree/nested").mkdir(parents=True, exist_ok=True); '
                f'Path("/tmp/main-tree/nested/value.bin").write_bytes({main_download_payload!r})'
            )
            await environment.service_exec(
                f"python -c {shlex.quote(main_download_script)}",
                service=MAIN_SERVICE_NAME,
            )
            await environment.service_exec(
                "mkdir -p /tmp/helper-tree/nested; "
                "printf '\\000helper-download\\376' > /tmp/helper-download.bin; "
                "printf '\\000helper-download\\376' > /tmp/helper-tree/nested/value.bin",
                service="helper",
            )
            main_download_file = tmp_path / "main-downloaded.bin"
            helper_download_file = tmp_path / "helper-downloaded.bin"
            main_download_dir = tmp_path / "main-downloaded-tree"
            helper_download_dir = tmp_path / "helper-downloaded-tree"
            await environment.service_download_file(
                "/tmp/main-download.bin",
                main_download_file,
            )
            await environment.service_download_file(
                "/tmp/helper-download.bin",
                helper_download_file,
                service="helper",
            )
            await environment.service_download_dir(
                "/tmp/main-tree",
                main_download_dir,
            )
            await environment.service_download_dir(
                "/tmp/helper-tree",
                helper_download_dir,
                service="helper",
            )
            assert main_download_file.read_bytes() == main_download_payload
            assert helper_download_file.read_bytes() == helper_download_payload
            assert (main_download_dir / "nested" / "value.bin").read_bytes() == main_download_payload
            assert (helper_download_dir / "nested" / "value.bin").read_bytes() == helper_download_payload

            def container_id(service: str) -> str:
                return subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-q",
                        "--filter",
                        f"label=com.docker.compose.project={project}",
                        "--filter",
                        f"label=com.docker.compose.service={service}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()

            main_container_id = container_id(MAIN_SERVICE_NAME)
            observer_container_id = container_id("observer")
            assert main_container_id and observer_container_id

            def container_generation(service: str) -> tuple[str, str]:
                identity = container_id(service)
                started_at = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.StartedAt}}", identity],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
                return identity, started_at

            async def probe_interrupted_process(pid_path: str) -> ExecResult:
                return await environment.service_exec(
                    "process_state=gone; secret_state=gone; "
                    "for proc_cmdline in /proc/[0-9]*/cmdline; do "
                    '[ -r "$proc_cmdline" ] || continue; '
                    "if tr '\\000' ' ' < \"$proc_cmdline\" 2>/dev/null "
                    '| grep -Fq -- "$PROBE_MARKER"; then process_state=alive; break; fi; done; '
                    "for proc_env in /proc/[0-9]*/environ; do "
                    '[ -r "$proc_env" ] || continue; '
                    "if tr '\\000' '\\n' < \"$proc_env\" 2>/dev/null "
                    "| grep -q '^INTERRUPT_SECRET='; then secret_state=found; break; fi; done; "
                    'printf \'process=%s secret=%s\' "$process_state" "$secret_state"',
                    service="helper",
                    env={"PROBE_MARKER": pid_path},
                )

            async def exercise_callback_interrupt(
                hostile_command: str,
                interrupt_secret: str,
                pid_path: str,
            ) -> ExecResult:
                callback_error = _CallbackBaseError("real sidecar callback failure")
                callback_output: list[str] = []

                async def fail_callback(text: str, _stream: str) -> None:
                    callback_output.append(text)
                    raise callback_error

                async def run_with_callback() -> ExecResult:
                    with environment.scoped_output_callback(fail_callback):
                        return await environment.service_exec(
                            hostile_command,
                            service="helper",
                            env={"INTERRUPT_SECRET": interrupt_secret},
                        )

                with pytest.raises(
                    _CallbackBaseError,
                    match="real sidecar callback failure",
                ) as caught:
                    await run_with_callback()
                assert caught.value is callback_error
                assert callback_output
                assert all(interrupt_secret not in chunk for chunk in callback_output)
                return await probe_interrupted_process(pid_path)

            for interrupt_mode in ("timeout", "cancel", "callback"):
                interrupt_secret = f"real-sidecar-{interrupt_mode}-secret-{uuid.uuid4().hex}"
                pid_path = f"/tmp/sidecar-{interrupt_mode}-{uuid.uuid4().hex}.pid"
                helper_generation_before = container_generation("helper")
                hostile_command = (
                    "trap '' TERM INT HUP; sleep 300 & child=$!; "
                    f"printf '%s' \"$child\" > {shlex.quote(pid_path)}; "
                    "printf 'callback-output-boundary-that-exceeds-secret-buffer-length-0123456789\\n'; "
                    'wait "$child"'
                )

                if interrupt_mode == "timeout":
                    interrupted = asyncio.create_task(
                        environment.service_exec(
                            hostile_command,
                            service="helper",
                            env={"INTERRUPT_SECRET": interrupt_secret},
                            timeout_sec=0.5,
                        )
                    )
                    await asyncio.sleep(0.15)
                    compose_path.write_text(
                        protected_compose_content,
                        encoding="utf-8",
                    )
                    restore_compose_after_raw_sidecar_start = True
                    concurrent_probe = asyncio.create_task(probe_interrupted_process(pid_path))
                    with pytest.raises(RuntimeError, match="timed out"):
                        await interrupted
                    assert restore_compose_after_raw_sidecar_start is False
                    probe_result = await concurrent_probe
                elif interrupt_mode == "cancel":
                    interrupted = asyncio.create_task(
                        environment.service_exec(
                            hostile_command,
                            service="helper",
                            env={"INTERRUPT_SECRET": interrupt_secret},
                        )
                    )
                    await asyncio.sleep(0.15)
                    concurrent_probe = asyncio.create_task(probe_interrupted_process(pid_path))
                    interrupted.cancel()
                    await asyncio.sleep(0.05)
                    interrupted.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await interrupted
                    probe_result = await concurrent_probe
                else:
                    probe_result = await exercise_callback_interrupt(
                        hostile_command,
                        interrupt_secret,
                        pid_path,
                    )

                assert probe_result.stdout == "process=gone secret=gone"
                helper_generation_after = container_generation("helper")
                assert helper_generation_after[0] == helper_generation_before[0]
                assert helper_generation_after[1] != helper_generation_before[1]
                assert container_id(MAIN_SERVICE_NAME) == main_container_id
                assert container_id("observer") == observer_container_id
                main_alive = await environment.service_exec(
                    f"printf main-alive-after-{interrupt_mode}",
                    service=MAIN_SERVICE_NAME,
                )
                helper_alive = await environment.service_exec(
                    f"printf helper-alive-after-{interrupt_mode}",
                    service="helper",
                )
                assert main_alive.stdout == f"main-alive-after-{interrupt_mode}"
                assert helper_alive.stdout == f"helper-alive-after-{interrupt_mode}"

            compose_path.write_text(
                protected_compose_content,
                encoding="utf-8",
            )
            try:
                await environment.stop_service(MAIN_SERVICE_NAME)
            finally:
                compose_path.write_text(compose_content, encoding="utf-8")
            assert container_id(MAIN_SERVICE_NAME) == ""
            assert container_id("observer") == observer_container_id
            post_stop_helper_download = tmp_path / "post-stop-helper-download.bin"
            await environment.service_download_file(
                "/tmp/helper-download.bin",
                post_stop_helper_download,
                service="helper",
            )
            assert post_stop_helper_download.read_bytes() == helper_download_payload
        finally:
            compose_path.write_text(compose_content, encoding="utf-8")
            if started:
                await environment.stop(delete=True)

    asyncio.run(exercise())

    sidecar_env_calls = [
        (arguments, process_environment)
        for arguments, process_environment, _process in sidecar_clients
        if "REAL_SIDECAR" in " ".join(str(argument) for argument in arguments)
    ]
    assert len(sidecar_env_calls) == 1
    sidecar_arguments, sidecar_process_environment = sidecar_env_calls[0]
    rendered_sidecar_arguments = " ".join(str(argument) for argument in sidecar_arguments)
    explicit_sidecar_values = {
        sidecar_secret,
        sidecar_reused_secret,
        *sidecar_control_environment.values(),
    }
    carrier_environment = {
        name: value
        for name, value in sidecar_process_environment.items()
        if name.startswith("SKILLEVALUATOR_SIDECAR_ENV_")
    }
    assert sidecar_secret not in rendered_sidecar_arguments
    assert sidecar_reused_secret not in rendered_sidecar_arguments
    assert set(carrier_environment.values()) == explicit_sidecar_values
    assert len(carrier_environment) == 2 + len(sidecar_control_environment)
    assert "REAL_SIDECAR" not in sidecar_process_environment
    assert sidecar_process_environment.get("REAL_PERSISTENT_ONLY") != sidecar_reused_secret
    for control_name, target_value in sidecar_control_environment.items():
        assert sidecar_process_environment.get(control_name) != target_value
    assert (
        not {
            "REAL_TASK_ONLY",
            "REAL_SCOPED_ONLY",
            "REAL_MAIN_WRAPPED",
            "NVIDIA_API_KEY",
            NVIDIA_BUILD_KEY_STDIN_ENV,
            "SKILLEVALUATOR_NVIDIA_API_KEY_FILE",
        }
        & sidecar_process_environment.keys()
    )
    for main_secret in {main_persistent_secret, main_task_secret, main_scoped_secret}:
        assert all(main_secret not in value for value in sidecar_process_environment.values())
    assert sidecar_clients
    assert all(process.returncode is not None for _args, _env, process in sidecar_clients)
    containment_clients = [
        (arguments, process_environment, process)
        for arguments, process_environment, process in all_compose_clients
        if len(arguments) > 2
        and str(arguments[1]) == "container"
        and str(arguments[2]) in {"stop", "kill", "rm", "start"}
    ]
    assert containment_clients
    for arguments, process_environment, process in containment_clients:
        rendered_arguments = " ".join(str(argument) for argument in arguments)
        assert all(value not in rendered_arguments for value in explicit_sidecar_values if value)
        assert not any(name.startswith("SKILLEVALUATOR_SIDECAR_ENV_") for name in process_environment)
        assert all(value not in process_environment.values() for value in explicit_sidecar_values if value)
        assert process.returncode is not None
    leaked_containers = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", f"label=com.docker.compose.project={project}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    leaked_networks = subprocess.run(
        ["docker", "network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    assert leaked_containers == ""
    assert leaked_networks == ""


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

    environment = ComposeOnlyEnvironment(
        environment_dir=target_dir,
        environment_name=target_project,
        session_id=target_project,
        trial_paths=TrialPaths(tmp_path / "interrupted-main-trial"),
        task_env_config=EnvironmentConfig(docker_image="python:3.13-slim"),
    )
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
