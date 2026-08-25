# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
from skillevaluator.tier3.harbor.runner import build_harbor_run_command
from skillevaluator.tier3.harbor.secure_docker_environment import (
    _REDACTION_SENTINEL_CANDIDATES,
    NVIDIA_BUILD_STDIN_SENTINEL,
    SECURE_DOCKER_ENV_IMPORT_PATH,
    SkillEvaluatorDockerEnvironment,
    SkillEvaluatorSecureDockerEnvironment,
    _collision_safe_redaction_marker,
    _redact,
    _secure_exec_arguments,
    _signal_process_tree,
    _StreamingSecretRedactor,
)

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


def test_compose_buffered_path_uses_devnull_without_stdin_or_callback(
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
    assert process.communicate_inputs == [None]
    assert result == ExecResult(stdout="buffered output", stderr=None, return_code=0)


@pytest.mark.parametrize("stdin_data", [b"", b"\x00tar\xffpayload\nwith spaces\x00"])
def test_compose_stdin_reaches_buffered_subprocess_byte_for_byte(
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
    assert process.communicate_inputs == [stdin_data]
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
        stop_main_on_interrupt: bool,
    ) -> None:
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
    assert any("cleanup or main-container containment also failed" in note for note in caught.value.__notes__)


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


def test_public_exec_streams_through_harbor_scoped_output_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "public-exec-callback-secret"
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
    marker = _marker_for(secret)
    expected = f"output {marker}\ntail\n"

    assert "".join(text for text, _stream in callback_chunks) == result.stdout == expected
    assert {stream for _text, stream in callback_chunks} == {"stdout"}
    assert secret not in " ".join(str(argument) for argument in captured["args"])
    assert isinstance(captured["env"], dict)
    assert captured["env"]["PUBLIC_EXEC_TOKEN"] == secret


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
    assert len(handoff_process.communicate_inputs) == 1
    handoff_script = (handoff_process.communicate_inputs[0] or b"").decode("utf-8")
    assert all(secret in handoff_script for secret in secrets)
    rendered_commands = "\n".join(" ".join(str(argument) for argument in command) for command in subprocess_commands)
    for secret in secrets:
        assert secret not in callback_output
        assert secret not in (result.stdout or "")
        assert secret not in rendered_commands


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
    handoff_inputs: list[bytes] = []

    async def create_subprocess(*args: object, **kwargs: object):
        subprocess_calls.append((args, kwargs))
        rendered = " ".join(str(argument) for argument in args)
        if "emit-context-scope-output" in rendered:
            raw_output = " ".join(sorted(all_secrets)) + "\n"
            return _BufferedAndStreamedComposeProcess([raw_output.encode()])

        process = _BufferedComposeProcess(stdout=b"")
        original_communicate = process.communicate

        async def capture_communicate(**communicate_kwargs: bytes | None):
            stdin_value = communicate_kwargs.get("input")
            if stdin_value is not None:
                handoff_inputs.append(stdin_value)
            return await original_communicate(**communicate_kwargs)

        process.communicate = capture_communicate  # type: ignore[method-assign]
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
with open(os.environ["COMPOSE_CLIENT_AUDIT"], "a", encoding="utf-8") as audit:
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
    monkeypatch.setenv("COMPOSE_CLIENT_AUDIT", str(audit_path))
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

    class Process:
        returncode = 0

        def __init__(self, record: dict[str, object]) -> None:
            self._record = record

        async def communicate(self, **kwargs: bytes | None) -> tuple[bytes, None]:
            nonlocal setup_count
            stdin_data = kwargs.get("input")
            self._record["stdin"] = stdin_data
            if stdin_data is not None:
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
            return b"", None

    async def create_subprocess(*args: object, **kwargs: object) -> Process:
        record = {"args": args, "env": dict(kwargs["env"])}
        calls.append(record)
        return Process(record)

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
        assert any(inactive_secret in value for value in process_environment.values() if isinstance(value, str))

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
    calls: list[tuple[dict[str, str], bytes | None]] = []

    class Process:
        returncode = 0

        def __init__(self, process_environment: dict[str, str]) -> None:
            self._process_environment = process_environment

        async def communicate(self, **kwargs: bytes | None) -> tuple[bytes, None]:
            calls.append((self._process_environment, kwargs.get("input")))
            return b"", None

    async def create_subprocess(*_args: object, **kwargs: object) -> Process:
        return Process(dict(kwargs["env"]))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = asyncio.run(environment.exec("true", env={"NVIDIA_API_KEY": requested_value}))

    assert result.return_code == 0
    assert len(calls) == 4
    assert any(stdin_data and resolved_secret.encode() in stdin_data for _env, stdin_data in calls)
    for process_environment, _stdin_data in calls:
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


@pytest.mark.parametrize("error_type", [_CallbackBaseError, asyncio.CancelledError])
def test_secure_public_exec_preserves_scoped_callback_failure_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    secret = "secure-public-callback-failure-secret"
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
        stop_main_on_interrupt: bool,
    ) -> None:
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
    assert callback_chunks == [f"output {_marker_for(secret)}"]
    assert str(caught) == f"callback rejected {callback_chunks[0]}"
    assert secret not in str(caught)
    assert containment_calls == [True]
    assert len(removed_handoffs) == 1
    assert main_process.returncode is not None
    assert all(
        secret not in value
        for process_environment in subprocess_environments[:-1]
        for value in process_environment.values()
    )
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
    main_communicating = asyncio.Event()
    main_completed: asyncio.Future[tuple[bytes, None]] | None = None

    class HangingProcess:
        pid = 7049
        returncode: int | None = None

        async def communicate(self, **_kwargs: bytes | None) -> tuple[bytes, None]:
            nonlocal main_completed
            main_communicating.set()
            main_completed = asyncio.get_running_loop().create_future()
            return await asyncio.shield(main_completed)

    hanging_process = HangingProcess()

    async def create_subprocess(*args: object, **kwargs: object):
        subprocess_calls.append((args, dict(kwargs["env"])))
        if "never-ending-scope-command" in " ".join(str(argument) for argument in args):
            return hanging_process
        return _BufferedComposeProcess(stdout=b"", return_code=0)

    def killpg(pid: int, value: signal.Signals) -> None:
        assert pid == hanging_process.pid
        hanging_process.returncode = -value
        if main_completed is not None and not main_completed.done():
            main_completed.set_result((b"", None))

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
        await asyncio.wait_for(main_communicating.wait(), timeout=1)
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
    assert len(subprocess_calls) == 7
    scoped_calls = subprocess_calls[:-1]
    rendered_commands = [" ".join(str(argument) for argument in arguments) for arguments, _env in scoped_calls]
    assert any(command.endswith("stop --timeout 0 main") for command in rendered_commands)
    assert any(command.endswith("rm --force --stop --volumes main") for command in rendered_commands)
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
        stop_main_on_interrupt: bool = False,
    ) -> ExecResult:
        del self, check, timeout_sec
        captured["command"] = command
        captured["env"] = env_overrides
        captured["on_output"] = on_output
        captured["additional_secret_values"] = additional_secret_values
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
    assert _marker_for(_SENTINEL) in str(caught.value)


def test_compose_check_false_redacts_success_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
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

    marker = _marker_for(_SENTINEL)
    assert result.stdout == f"stdout {marker}"
    assert result.stderr == f"stderr {marker}"


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

    class Process:
        returncode = 9

        async def communicate(self, **kwargs: bytes | None) -> tuple[bytes, None]:
            assert set(kwargs) <= {"input"}
            captured["input"] = kwargs.get("input")
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
                stdin_data=b"private-stream-payload",
                additional_secret_values={_SENTINEL},
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
    assert _redact("token=abcdefgh", {"abcdefgh"}) == f"token={_marker_for('abcdefgh')}"


def test_compose_redacts_long_secrets_without_rewriting_short_env_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _initialized_docker_environment(tmp_path)
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
    assert result.stderr == f"stderr {_marker_for(long_secret)}"


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

    environment = _initialized_docker_environment(tmp_path)
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

    environment = _initialized_docker_environment(tmp_path)
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
    environment._output_callbacks = contextvars.ContextVar("integration_output_callbacks", default=())
    environment._exec_env_overlays = contextvars.ContextVar("integration_exec_env_overlays", default=())
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
