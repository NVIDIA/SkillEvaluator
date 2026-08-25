# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker environments that keep per-exec credentials out of process argv.

The pinned Harbor release serializes ``exec(env=...)`` values as
``docker compose exec -e NAME=value``. Process arguments are host-visible, so
the compatibility backend passes only names on argv and values through the
compose subprocess environment. SkillEvaluator's selected backend is stronger:
it receives the parent NVIDIA credential through stdin, then transfers values
through a short-lived container handoff removed before the requested command.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
import re
import shlex
import signal
import unicodedata
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.environments.base import ExecResult, OutputCallback, OutputStream
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name

from skillevaluator.tier3.harbor.sensitive_stdin import (
    NVIDIA_BUILD_STDIN_SENTINEL,
    read_nvidia_build_key_from_stdin,
)

SECURE_DOCKER_ENV_IMPORT_PATH = (
    "skillevaluator.tier3.harbor.secure_docker_environment:SkillEvaluatorSecureDockerEnvironment"
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NVIDIA_BUILD_FILE_SENTINEL = "skillevaluator-file-backed-nvidia-key"
_NVIDIA_BUILD_KEY_FILE_ENV = "SKILLEVALUATOR_NVIDIA_API_KEY_FILE"
# Match llm_judge / local_environment: short env values like "1" must not
# become substring secrets or loopback origins such as 127.0.0.1 break.
_MIN_EXACT_SECRET_LENGTH = 8
_COMPOSE_TERMINATE_SECONDS = 5.0
_COMPOSE_KILL_SECONDS = 5.0
_COMPOSE_CANCEL_SECONDS = 0.1
_MAIN_CONTAINER_STOP_TIMEOUT_SECONDS = 8
_REDACTION_LABEL = "[REDACTED]"
_REDACTION_SENTINEL_CANDIDATES = ("␟", "␞", "␝", "␜", "")


@dataclass(frozen=True, slots=True, repr=False)
class _SecureHandoffScope:
    environment_names: frozenset[str]
    secret_values: frozenset[str]


_SECURE_HANDOFF_SCOPES: contextvars.ContextVar[tuple[_SecureHandoffScope, ...]] = contextvars.ContextVar(
    "skillevaluator_secure_docker_handoff_scopes",
    default=(),
)


class _ComposeCommandTimeout(Exception):
    """Internal marker that cannot collide with callback exception types."""


class _SecretTrieNode:
    __slots__ = ("children", "failure", "max_terminal_length", "terminal_length")

    def __init__(self) -> None:
        self.children: dict[str, _SecretTrieNode] = {}
        self.failure = self
        self.max_terminal_length = 0
        self.terminal_length = 0


def _eligible_secret_values(secret_values: Iterable[str]) -> list[str]:
    return sorted({value for value in secret_values if value and len(value) >= _MIN_EXACT_SECRET_LENGTH})


def _collision_safe_redaction_marker(secret_values: Iterable[str]) -> str:
    """Build a marker that cannot contain or join into an eligible secret."""
    secrets = _eligible_secret_values(secret_values)
    if not secrets:
        return _REDACTION_LABEL

    sentinel: str | None = None
    for candidate in _REDACTION_SENTINEL_CANDIDATES:
        if all(candidate not in secret for secret in secrets):
            sentinel = candidate
            break

    if sentinel is None:
        used_characters: set[str] = set()
        for secret in secrets:
            used_characters.update(secret)

        # Private-use scalars have no standardized control semantics. If they
        # are all occupied, accept only Unicode letters, numbers, punctuation,
        # or symbols; controls, separators, combining marks, and surrogates
        # are unsafe as callback and diagnostic boundaries.
        private_use_ranges = (
            range(0xE000, 0xF900),
            range(0xF0000, 0xFFFFE),
            range(0x100000, 0x10FFFE),
        )
        for candidate_range in private_use_ranges:
            for codepoint in candidate_range:
                candidate = chr(codepoint)
                if candidate not in used_characters:
                    sentinel = candidate
                    break
            if sentinel is not None:
                break

        if sentinel is None:
            scalar_ranges = (range(1, 0xD800), range(0xE000, 0x110000))
            for scalar_range in scalar_ranges:
                for codepoint in scalar_range:
                    candidate = chr(codepoint)
                    if candidate not in used_characters and unicodedata.category(candidate)[0] in "LNPS":
                        sentinel = candidate
                        break
                if sentinel is not None:
                    break

    if sentinel is None:
        # No marker can satisfy the absent-character invariant. Fail without
        # rendering any secret value; callers construct this before spawning.
        raise RuntimeError("Could not construct a collision-safe redaction marker")

    minimum_secret_length = min(map(len, secrets))
    chunk_length = minimum_secret_length - 1
    # Sentinel boundaries prevent a secret from bridging raw text and the
    # marker. Splitting the readable label keeps every sentinel-free run below
    # the shortest eligible secret length, so no secret can live in the marker.
    label_chunks = [
        _REDACTION_LABEL[index : index + chunk_length] for index in range(0, len(_REDACTION_LABEL), chunk_length)
    ]
    return sentinel + sentinel.join(label_chunks) + sentinel


class _StreamingSecretRedactor:
    """Redact the union of secret-match spans using Aho-Corasick.

    Collapsing every connected covered span deliberately redacts more than
    leftmost-longest replacement when secrets overlap. Each automaton state
    stores only its longest terminal suffix, because that interval covers all
    shorter matches ending at the same position.
    """

    def __init__(
        self,
        secret_values: Iterable[str],
        *,
        _replacement: str | None = None,
        _track_transitions: bool = False,
    ) -> None:
        secrets = _eligible_secret_values(secret_values)
        self._root = _SecretTrieNode()
        for secret in secrets:
            node = self._root
            for character in secret:
                child = node.children.get(character)
                if child is None:
                    child = _SecretTrieNode()
                    node.children[character] = child
                node = child
            node.terminal_length = len(secret)

        failure_queue = deque(self._root.children.values())
        for child in failure_queue:
            child.failure = self._root
            child.max_terminal_length = child.terminal_length
        while failure_queue:
            node = failure_queue.popleft()
            for character, child in node.children.items():
                failure = node.failure
                while failure is not self._root and character not in failure.children:
                    failure = failure.failure
                child.failure = failure.children.get(character, self._root)
                child.max_terminal_length = max(
                    child.terminal_length,
                    child.failure.max_terminal_length,
                )
                failure_queue.append(child)

        self._has_secrets = bool(secrets)
        self._max_secret_length = max(map(len, secrets), default=0)
        self._state = self._root
        self._pending: deque[str] = deque()
        self._pending_start = 0
        self._processed = 0
        self._coverage: deque[tuple[int, int]] = deque()
        self._redaction_open = False
        self._replacement = _collision_safe_redaction_marker(secrets) if _replacement is None else _replacement
        self._track_transitions = _track_transitions
        self._match_transition_count = 0
        self._match_work_count = 0

    @property
    def match_transition_count(self) -> int:
        return self._match_transition_count

    @property
    def match_work_count(self) -> int:
        """Return instrumented scan, coverage, and commit operations."""
        return self._match_work_count

    def _record_work(self) -> None:
        if self._track_transitions:
            self._match_work_count += 1

    def _add_coverage(self, start: int, end: int) -> None:
        merged_start = start
        while True:
            self._record_work()
            if not self._coverage or self._coverage[-1][1] < merged_start:
                break
            previous_start, _previous_end = self._coverage.pop()
            merged_start = min(merged_start, previous_start)
            self._record_work()
        self._coverage.append((merged_start, end))
        self._record_work()

    def _advance(self, character: str) -> None:
        while True:
            if self._track_transitions:
                self._match_transition_count += 1
                self._match_work_count += 1
            child = self._state.children.get(character)
            if child is not None:
                self._state = child
                break
            if self._state is self._root:
                break
            self._state = self._state.failure

        match_end = self._processed + 1
        match_length = self._state.max_terminal_length
        self._record_work()
        if match_length:
            self._add_coverage(match_end - match_length, match_end)
        self._processed = match_end

    def _position_is_covered(self, position: int) -> bool:
        while True:
            self._record_work()
            if not self._coverage or self._coverage[0][1] > position:
                break
            self._coverage.popleft()
            self._record_work()
        self._record_work()
        return bool(self._coverage and self._coverage[0][0] <= position < self._coverage[0][1])

    def _drain(self, *, final: bool) -> str:
        if final:
            safe_end = self._processed
        else:
            safe_end = self._processed - self._max_secret_length + 1

        emitted: list[str] = []
        while self._pending_start < safe_end:
            character = self._pending.popleft()
            if self._position_is_covered(self._pending_start):
                if not self._redaction_open:
                    emitted.append(self._replacement)
                    self._redaction_open = True
            else:
                self._redaction_open = False
                emitted.append(character)
            self._pending_start += 1
            self._record_work()

        return "".join(emitted)

    def feed(self, text: str, *, final: bool = False) -> str:
        """Return safe output, retaining at most one maximum-pattern window."""
        if not self._has_secrets:
            self._processed += len(text)
            self._pending_start = self._processed
            return text

        emitted: list[str] = []
        for character in text:
            self._pending.append(character)
            self._advance(character)
            safe_output = self._drain(final=False)
            if safe_output:
                emitted.append(safe_output)
        if final:
            final_output = self._drain(final=True)
            if final_output:
                emitted.append(final_output)
        return "".join(emitted)

    def finish(self) -> str:
        """Flush the final suffix once no later chunk can complete a match."""
        return self.feed("", final=True)


async def _await_task_uninterruptibly(
    task: asyncio.Task[Any],
    *,
    preserve_cancellation: bool = True,
) -> Any:
    """Await process cleanup to completion despite repeated cancellation."""
    cancellation_requested = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancellation_requested = True
            if task.done():
                result = task.result()
                break
    if cancellation_requested and preserve_cancellation:
        raise asyncio.CancelledError
    return result


def _validate_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    """Validate environment names and values without rendering secret values."""
    validated: dict[str, str] = {}
    for name, value in (environment or {}).items():
        if not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        if not isinstance(value, str):
            raise ValueError(f"Environment variable {name!r} must have a string value")
        if "\x00" in value:
            raise ValueError(f"Environment variable {name!r} contains a NUL byte")
        validated[name] = value
    return validated


def _secure_exec_arguments(
    environment: Mapping[str, str] | None,
) -> tuple[list[str], dict[str, str]]:
    """Put env names on argv and every value in the child process env."""
    subprocess_environment = _validate_environment(environment)
    arguments = [part for name in subprocess_environment for part in ("-e", name)]
    return arguments, subprocess_environment


def _redact(
    text: str | None,
    secret_values: set[str],
    *,
    replacement: str | None = None,
) -> str | None:
    if text is None:
        return None
    redactor = _StreamingSecretRedactor(secret_values, _replacement=replacement)
    return redactor.feed(text) + redactor.finish()


def _redact_result(
    result: ExecResult,
    secret_values: set[str],
    *,
    replacement: str | None = None,
) -> ExecResult:
    return ExecResult(
        stdout=_redact(result.stdout, secret_values, replacement=replacement),
        stderr=_redact(result.stderr, secret_values, replacement=replacement),
        return_code=result.return_code,
    )


def _signal_process_tree(process: asyncio.subprocess.Process, value: signal.Signals) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, value)
        except ProcessLookupError:
            return
        except PermissionError:
            # macOS can report EPERM instead of ESRCH if the process-group
            # leader exits between the returncode check and killpg(). Suppress
            # only when a second liveness check proves that PID is gone; a live
            # process with a genuine permission failure must still fail closed.
            if process.returncode is not None:
                return
            try:
                os.getpgid(process.pid)
            except ProcessLookupError:
                return
            raise
    elif value == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


async def _terminate_process_tree(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[Any],
    *,
    preserve_cancellation: bool,
) -> None:
    async def reap() -> None:
        _signal_process_tree(process, signal.SIGTERM)
        done, _pending = await asyncio.wait(
            {communication},
            timeout=_COMPOSE_TERMINATE_SECONDS,
        )
        if communication in done and process.returncode is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                communication.result()
            return

        _signal_process_tree(process, signal.SIGKILL)
        done, _pending = await asyncio.wait(
            {communication},
            timeout=_COMPOSE_KILL_SECONDS,
        )
        if communication in done:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                communication.result()
        else:
            communication.cancel()
            done, _pending = await asyncio.wait({communication}, timeout=_COMPOSE_CANCEL_SECONDS)
            if communication in done:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    communication.result()
            # asyncio cannot forcibly terminate a coroutine that deliberately
            # suppresses CancelledError. Keep this wait bounded; cooperative
            # callbacks are owned and reaped above, while a hostile callback can
            # only be left for event-loop shutdown after ignoring cancellation.

    cleanup = asyncio.create_task(reap())
    await _await_task_uninterruptibly(cleanup, preserve_cancellation=preserve_cancellation)


def _host_handoff_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Resolve a private NVIDIA Build sentinel without putting its value in argv."""
    resolved = _validate_environment(environment)
    if resolved.get("NVIDIA_API_KEY") == NVIDIA_BUILD_STDIN_SENTINEL:
        resolved["NVIDIA_API_KEY"] = read_nvidia_build_key_from_stdin()
        return resolved
    if resolved.get("NVIDIA_API_KEY") != _NVIDIA_BUILD_FILE_SENTINEL:
        return resolved
    key_file = os.environ.get(_NVIDIA_BUILD_KEY_FILE_ENV, "").strip()
    if not key_file:
        raise RuntimeError(f"{_NVIDIA_BUILD_KEY_FILE_ENV} is required for NVIDIA Build Docker runs")
    try:
        api_key = Path(key_file).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("NVIDIA Build key handoff file is unavailable") from exc
    if not api_key:
        raise RuntimeError("NVIDIA Build key handoff file is empty")
    resolved["NVIDIA_API_KEY"] = api_key
    return resolved


def _render_environment_script(environment: Mapping[str, str]) -> str:
    """Render a sourceable script after validating every name and value."""
    validated = _validate_environment(environment)
    lines = [f"export {name}={shlex.quote(value)}" for name, value in sorted(validated.items())]
    return "\n".join(lines) + "\n"


class SkillEvaluatorDockerEnvironment(DockerEnvironment):
    """Pinned Harbor compatibility backend with host-visible argv safety."""

    @classmethod
    def preflight(cls) -> None:
        """Consume the private stdin handoff before Docker can inherit it."""
        if os.environ.get("NVIDIA_API_KEY", "").strip() == NVIDIA_BUILD_STDIN_SENTINEL:
            read_nvidia_build_key_from_stdin()
        super().preflight()

    async def _contain_main_container(self) -> None:
        """Stop and remove this Compose project's task container from the trusted host."""
        stopped = False
        try:
            result = await self._run_docker_compose_command(
                ["stop", "--timeout", "0", "main"],
                check=False,
                timeout_sec=_MAIN_CONTAINER_STOP_TIMEOUT_SECONDS,
            )
            stopped = result.return_code == 0
        except Exception:
            pass

        if not stopped:
            try:
                result = await self._run_docker_compose_command(
                    ["kill", "--signal", "SIGKILL", "main"],
                    check=False,
                    timeout_sec=_MAIN_CONTAINER_STOP_TIMEOUT_SECONDS,
                )
                stopped = result.return_code == 0
            except Exception:
                pass

        # Removal destroys a handoff that cancellation may have interrupted
        # before the in-container wrapper could unlink it. ``--stop`` is also
        # the final host-authoritative fallback if stop/kill was inconclusive.
        try:
            result = await self._run_docker_compose_command(
                ["rm", "--force", "--stop", "--volumes", "main"],
                check=False,
                timeout_sec=_MAIN_CONTAINER_STOP_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise RuntimeError("could not confirm main task container containment") from exc
        if result.return_code != 0:
            detail = "after a confirmed stop" if stopped else "after inconclusive stop and kill attempts"
            raise RuntimeError(
                f"could not confirm main task container containment (removal status {result.return_code} {detail})"
            )

    async def _contain_main_and_reap_compose(
        self,
        process: asyncio.subprocess.Process,
        communication: asyncio.Task[Any],
        *,
        stop_main_on_interrupt: bool,
    ) -> None:
        containment_error: BaseException | None = None
        if stop_main_on_interrupt:
            try:
                await self._contain_main_container()
            except BaseException as exc:
                containment_error = exc
        await _terminate_process_tree(process, communication, preserve_cancellation=False)
        if containment_error is not None:
            raise RuntimeError("could not confirm main task container containment") from containment_error

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        merged_environment = self._merge_env(env)
        environment_args, subprocess_environment = _secure_exec_arguments(merged_environment)

        exec_command = ["exec"]
        effective_cwd = cwd or self.task_env_config.workdir
        if effective_cwd:
            exec_command.extend(["-w", effective_cwd])
        exec_command.extend(environment_args)
        if user is not None:
            exec_command.extend(["-u", str(user)])
        exec_command.append("main")
        exec_command.extend(self._platform.exec_shell_args(command))

        return await self._run_docker_compose_command(
            exec_command,
            check=False,
            timeout_sec=timeout_sec,
            on_output=self._output_callback(),
            env_overrides=subprocess_environment,
            stop_main_on_interrupt=True,
        )

    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: float | None = None,
        stdin_data: bytes | None = None,
        on_output: OutputCallback | None = None,
        *,
        env_overrides: Mapping[str, str] | None = None,
        additional_secret_values: Iterable[str] | None = None,
        stop_main_on_interrupt: bool = False,
    ) -> ExecResult:
        """Run compose with sensitive values only in child env or stdin."""
        full_command = [
            "docker",
            "compose",
            "--project-name",
            _sanitize_docker_compose_project_name(self.session_id),
            "--project-directory",
            str(self.environment_dir.resolve().absolute()),
        ]
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(command)

        active_handoff_scopes = _SECURE_HANDOFF_SCOPES.get()
        effective_env_overrides = (
            _validate_environment(env_overrides) if active_handoff_scopes else dict(env_overrides or {})
        )
        active_handoff_names: set[str] = set()
        active_handoff_values: set[str] = set()
        for handoff_scope in active_handoff_scopes:
            active_handoff_names.update(handoff_scope.environment_names)
            active_handoff_values.update(handoff_scope.secret_values)
        if active_handoff_scopes:
            active_handoff_names.update(effective_env_overrides)
            active_handoff_values.update(_eligible_secret_values(effective_env_overrides.values()))

        process_environment = self._compose_env_vars(include_os_env=True)
        process_environment.update(effective_env_overrides)
        if active_handoff_scopes:
            process_environment = {
                name: value
                for name, value in process_environment.items()
                if name not in active_handoff_names
                and not (
                    isinstance(value, str) and any(secret_value in value for secret_value in active_handoff_values)
                )
            }

        secret_values = set(_eligible_secret_values(effective_env_overrides.values()))
        secret_values.update(_eligible_secret_values(additional_secret_values or ()))
        secret_values.update(active_handoff_values)
        redaction_marker = _collision_safe_redaction_marker(secret_values)
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *full_command,
                env=process_environment,
                stdin=(asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        )
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            process = await _await_task_uninterruptibly(creation, preserve_cancellation=False)
            communication = asyncio.create_task(process.communicate())
            cleanup = asyncio.create_task(
                self._contain_main_and_reap_compose(
                    process,
                    communication,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    "main task container containment could not be confirmed during cancellation"
                ) from exc
            raise

        callback_error: asyncio.Future[BaseException] | None = None

        if on_output is None:

            async def collect_buffered_output() -> ExecResult:
                if stdin_data is None:
                    stdout_bytes, stderr_bytes = await process.communicate()
                else:
                    stdout_bytes, stderr_bytes = await process.communicate(input=stdin_data)
                stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
                stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
                return ExecResult(
                    stdout=stdout,
                    stderr=stderr,
                    return_code=process.returncode or 0,
                )

            communication = asyncio.create_task(collect_buffered_output())
        else:
            callback_error = asyncio.get_running_loop().create_future()
            stream_redactor = _StreamingSecretRedactor(
                secret_values,
                _replacement=redaction_marker,
            )

            async def emit_redacted_output(text: str, stream: OutputStream) -> None:
                if callback_error.done():
                    await asyncio.sleep(0)
                    return
                try:
                    await on_output(text, stream)
                except BaseException as exc:
                    if not callback_error.done():
                        callback_error.set_result(exc)
                    # Do not re-raise into Harbor's collector: it would race its
                    # immediate-process termination against our process-group and
                    # optional main-container cleanup. The outer owner wakes on
                    # callback_error and performs containment exactly once.
                    await asyncio.sleep(0)

            async def redacted_callback(text: str, stream: OutputStream) -> None:
                if callback_error.done():
                    await asyncio.sleep(0)
                    return
                redacted_text = stream_redactor.feed(text)
                if redacted_text:
                    await emit_redacted_output(redacted_text, stream)

            async def collect_streamed_output() -> ExecResult:
                try:
                    result = await self._collect_streamed_output(
                        process,
                        timeout_sec=None,
                        stdin_data=stdin_data,
                        on_output=redacted_callback,
                    )
                    if not callback_error.done():
                        final_output = stream_redactor.finish()
                        if final_output:
                            await emit_redacted_output(final_output, "stdout")
                    return result
                except BaseException as exc:
                    if not callback_error.done():
                        callback_error.set_result(exc)
                    # The outer owner performs process-group cleanup before it
                    # propagates the original collector/callback exception.
                    return ExecResult(
                        stdout=None,
                        stderr=None,
                        return_code=process.returncode or 0,
                    )

            communication = asyncio.create_task(collect_streamed_output())

        async def cleanup_preserving_primary(primary_error: BaseException) -> None:
            cleanup = asyncio.create_task(
                self._contain_main_and_reap_compose(
                    process,
                    communication,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except BaseException as cleanup_error:
                primary_error.add_note(
                    "Docker Compose cleanup or main-container containment also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                raise primary_error from cleanup_error

        callback_failure: BaseException | None = None
        try:
            waitables: set[asyncio.Future[Any] | asyncio.Task[Any]] = {communication}
            if callback_error is not None:
                waitables.add(callback_error)
            done, _pending = await asyncio.wait(
                waitables,
                timeout=timeout_sec or None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise _ComposeCommandTimeout
            if callback_error is not None and callback_error in done:
                callback_failure = callback_error.result()
            else:
                result = await asyncio.shield(communication)
        except _ComposeCommandTimeout:
            cleanup = asyncio.create_task(
                self._contain_main_and_reap_compose(
                    process,
                    communication,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Command timed out after {timeout_sec} seconds; main task container containment could not be confirmed"
                ) from exc
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._contain_main_and_reap_compose(
                    process,
                    communication,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    "main task container containment could not be confirmed during cancellation"
                ) from exc
            raise
        except BaseException as exc:
            await cleanup_preserving_primary(exc)
            raise

        if callback_failure is not None:
            # Kept outside the try/except above so a callback's deliberate
            # CancelledError is not mistaken for cancellation of this caller.
            await cleanup_preserving_primary(callback_failure)
            raise callback_failure

        if check and result.return_code != 0:
            detail = (
                f"Docker compose command failed for environment {self.environment_name}. "
                f"Command: {' '.join(full_command)}. Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. Stderr: {result.stderr}."
            )
            raise RuntimeError(
                _redact(
                    detail,
                    secret_values,
                    replacement=redaction_marker,
                )
            )
        return _redact_result(
            result,
            secret_values,
            replacement=redaction_marker,
        )


class SkillEvaluatorSecureDockerEnvironment(SkillEvaluatorDockerEnvironment):
    """Stream exec environments into short-lived container-only files."""

    async def _exec_without_environment(
        self,
        command: str,
        *,
        cwd: str | None,
        timeout_sec: int | None,
        user: str | int | None,
        secret_values: set[str] | None = None,
    ) -> ExecResult:
        exec_command = ["exec"]
        effective_cwd = cwd or self.task_env_config.workdir
        if effective_cwd:
            exec_command.extend(["-w", effective_cwd])
        if user is not None:
            exec_command.extend(["-u", str(user)])
        exec_command.append("main")
        exec_command.extend(self._platform.exec_shell_args(command))
        return await self._run_docker_compose_command(
            exec_command,
            check=False,
            timeout_sec=timeout_sec,
            on_output=self._output_callback(),
            additional_secret_values=secret_values,
            stop_main_on_interrupt=True,
        )

    async def _remove_handoff(self, remote_path: str) -> None:
        result = await self._run_docker_compose_command(
            ["exec", "-u", "root", "main", "rm", "-f", "--", remote_path],
            check=False,
        )
        if result.return_code != 0:
            raise RuntimeError("Docker environment handoff removal failed")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        merged = self._merge_env(env)
        if not merged:
            return await self._exec_without_environment(
                command,
                cwd=cwd,
                timeout_sec=timeout_sec,
                user=user,
            )

        merged = _host_handoff_environment(merged)
        secret_values = set(_eligible_secret_values(merged.values()))
        if secret_values:
            # Fail before writing or uploading a handoff if no structurally
            # safe output marker can represent these values.
            _collision_safe_redaction_marker(secret_values)
        remote_path = f"/tmp/.skillevaluator-exec-env-{uuid.uuid4().hex}.sh"
        handoff_scope = _SecureHandoffScope(
            environment_names=frozenset(merged),
            secret_values=frozenset(secret_values),
        )
        scope_token = _SECURE_HANDOFF_SCOPES.set((*_SECURE_HANDOFF_SCOPES.get(), handoff_scope))
        primary_error: BaseException | None = None
        try:
            await self._run_docker_compose_command(
                [
                    "exec",
                    "-T",
                    "-u",
                    "root",
                    "main",
                    "sh",
                    "-c",
                    'umask 077; cat > "$1"',
                    "sh",
                    remote_path,
                ],
                check=True,
                stdin_data=_render_environment_script(merged).encode("utf-8"),
                additional_secret_values=secret_values,
            )

            if user is None:
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chmod", "600", remote_path],
                    check=True,
                    additional_secret_values=secret_values,
                )
            else:
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chown", "--", str(user), remote_path],
                    check=True,
                    additional_secret_values=secret_values,
                )
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chmod", "600", remote_path],
                    check=True,
                    additional_secret_values=secret_values,
                )
            quoted_path = shlex.quote(remote_path)
            wrapped = (
                f"if ! . {quoted_path}; then rm -f -- {quoted_path}; exit 126; fi; "
                f"if ! rm -f -- {quoted_path}; then exit 126; fi; {command}"
            )
            return await self._exec_without_environment(
                wrapped,
                cwd=cwd,
                timeout_sec=timeout_sec,
                user=user,
                secret_values=secret_values,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                cleanup = asyncio.create_task(self._remove_handoff(remote_path))
                try:
                    await _await_task_uninterruptibly(
                        cleanup,
                        preserve_cancellation=primary_error is None,
                    )
                except Exception as cleanup_error:
                    message = f"could not confirm removal of Docker environment handoff {remote_path}"
                    if primary_error is not None:
                        if hasattr(primary_error, "add_note"):
                            primary_error.add_note(f"{message}: {cleanup_error}")
                    else:
                        raise RuntimeError(message) from cleanup_error
            finally:
                _SECURE_HANDOFF_SCOPES.reset(scope_token)

    async def exec_with_sensitive_env(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute with values streamed into a private, container-only handoff."""
        return await self.exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )
