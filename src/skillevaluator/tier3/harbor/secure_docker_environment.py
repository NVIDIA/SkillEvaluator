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
import shutil
import signal
import stat
import tempfile
import unicodedata
import uuid
from collections import deque
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from harbor.environments.base import (
    MAIN_SERVICE_NAME,
    ExecResult,
    OutputCallback,
    OutputStream,
    ServiceOperationsUnsupportedError,
)
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name

from skillevaluator.tier3.harbor.sensitive_stdin import (
    NVIDIA_BUILD_KEY_STDIN_ENV,
    NVIDIA_BUILD_STDIN_SENTINEL,
    read_nvidia_build_key_from_stdin,
)

SECURE_DOCKER_ENV_IMPORT_PATH = (
    "skillevaluator.tier3.harbor.secure_docker_environment:SkillEvaluatorSecureDockerEnvironment"
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_NAME_PREFIX_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMPOSE_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|KEY|PAT|TOKEN|SECRET|PASS(?:WORD)?|"
    r"CREDENTIALS?|AUTH(?:ORIZATION)?|BEARER|COOKIE|SESSION|CERT(?:IFICATE)?|DSN|"
    r"CONNECTION(?:_STRING)?|(?:PRE)?SIGNED_?URL|SAS_?URL|CREDENTIAL_?URL|DATABASE_?URL)(?:_|$)",
    re.IGNORECASE,
)
_NVIDIA_BUILD_FILE_SENTINEL = "skillevaluator-file-backed-nvidia-key"
_NVIDIA_BUILD_KEY_FILE_ENV = "SKILLEVALUATOR_NVIDIA_API_KEY_FILE"
# Match llm_judge / local_environment: short env values like "1" must not
# become substring secrets or loopback origins such as 127.0.0.1 break.
_MIN_EXACT_SECRET_LENGTH = 8
_COMPOSE_TERMINATE_SECONDS = 5.0
_COMPOSE_KILL_SECONDS = 5.0
_COMPOSE_CANCEL_SECONDS = 0.1
_RAW_DOCKER_COMMAND_TIMEOUT_SECONDS = 3.0
_RAW_LIFECYCLE_TOTAL_TIMEOUT_SECONDS = 30.0
_SIDECAR_ENV_CARRIER_PREFIX = "SKILLEVALUATOR_SIDECAR_ENV_"
_MAX_COMPOSE_MODEL_FILES = 64
_MAX_COMPOSE_MODEL_BYTES = 8 * 1024 * 1024
_MAX_COMPOSE_MODEL_NODES = 100_000
_MAX_COMPOSE_MODEL_DEPTH = 128
_REDACTION_LABEL = "[REDACTED]"
_REDACTION_SENTINEL_CANDIDATES = ("␟", "␞", "␝", "␜", "")


@dataclass(frozen=True, slots=True, repr=False)
class _SecureHandoffScope:
    environment_names: frozenset[str]
    secret_values: frozenset[str]


@dataclass(slots=True, eq=False)
class _SidecarOperation:
    environment_identity: int
    service: str
    compose_model_environment: dict[str, str]
    active: bool = True


@dataclass(frozen=True, slots=True)
class _RawContainerState:
    identity: str
    project: str
    service: str
    container_number: int
    running: bool
    paused: bool
    restarting: bool
    status: str
    health_status: str | None


@dataclass(frozen=True, slots=True)
class _RawServiceSnapshot:
    all_identities: tuple[str, ...]
    running_identities: tuple[str, ...]


_SECURE_HANDOFF_SCOPES: contextvars.ContextVar[tuple[_SecureHandoffScope, ...]] = contextvars.ContextVar(
    "skillevaluator_secure_docker_handoff_scopes",
    default=(),
)
_SIDECAR_EXEC_OPERATIONS: contextvars.ContextVar[tuple[_SidecarOperation, ...]] = contextvars.ContextVar(
    "skillevaluator_sidecar_exec_operations",
    default=(),
)
_RAW_LIFECYCLE_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "skillevaluator_raw_docker_lifecycle_deadline",
    default=None,
)


class _ComposeCommandTimeout(Exception):
    """Internal marker that cannot collide with callback exception types."""


class _ComposeModelLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves Compose override-tagged values."""


def _construct_compose_override_value(
    loader: _ComposeModelLoader,
    node: yaml.Node,
) -> object:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise yaml.constructor.ConstructorError(
        None,
        None,
        "unsupported Docker Compose override value",
        node.start_mark,
    )


for _compose_override_tag in ("!reset", "!override"):
    _ComposeModelLoader.add_constructor(
        _compose_override_tag,
        _construct_compose_override_value,
    )


class _SecretTrieNode:
    __slots__ = ("children", "failure", "max_terminal_length", "terminal_length")

    def __init__(self) -> None:
        self.children: dict[str, _SecretTrieNode] = {}
        self.failure = self
        self.max_terminal_length = 0
        self.terminal_length = 0


def _eligible_secret_values(
    secret_values: Iterable[str],
    *,
    include_short: bool = False,
) -> list[str]:
    return sorted(
        {value for value in secret_values if value and (include_short or len(value) >= _MIN_EXACT_SECRET_LENGTH)}
    )


def _sensitive_environment_values(environment: Mapping[str, str]) -> set[str]:
    """Return exact values whose component-aware names mark them sensitive."""
    return {value for name, value in environment.items() if value and _SENSITIVE_ENV_NAME_RE.search(name)}


def _value_contains_protected_value(value: str, protected_value: str) -> bool:
    """Match protected values inside structurally retained Compose values."""
    return protected_value in value


def _collision_safe_redaction_marker(
    secret_values: Iterable[str],
    *,
    include_short: bool = False,
) -> str:
    """Build a marker that cannot contain or join into an eligible secret."""
    secrets = _eligible_secret_values(secret_values, include_short=include_short)
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
    if minimum_secret_length == 1:
        return sentinel
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
        _include_short: bool = False,
    ) -> None:
        secrets = _eligible_secret_values(
            secret_values,
            include_short=_include_short,
        )
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
        self._replacement = (
            _collision_safe_redaction_marker(secrets, include_short=True) if _replacement is None else _replacement
        )
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


def _validate_compose_service_name(service: str) -> str:
    if not isinstance(service, str) or not _COMPOSE_SERVICE_NAME_RE.fullmatch(service):
        raise ValueError(f"Invalid Docker Compose service name: {service!r}")
    return service


def _compose_interpolation_names(content: str) -> set[str]:
    """Extract Compose variable names while respecting ``$$`` escapes."""
    names: set[str] = set()
    index = 0
    while index < len(content):
        if content[index] != "$":
            index += 1
            continue
        if index + 1 < len(content) and content[index + 1] == "$":
            index += 2
            continue
        name_start = index + 1
        if name_start < len(content) and content[name_start] == "{":
            name_start += 1
        match = _ENV_NAME_PREFIX_RE.match(content, name_start)
        if match is None:
            index += 1
            continue
        names.add(match.group())
        index = match.end()
    return names


def _sidecar_environment_carriers(
    environment: Mapping[str, str] | None,
    *,
    reserved_names: Iterable[str],
) -> tuple[list[str], dict[str, str], str | None]:
    """Map target env names through unpredictable client-safe carriers."""
    validated = _validate_environment(environment)
    if not validated:
        return [], {}, None

    occupied_names = set(reserved_names) | set(validated)
    while True:
        invocation_id = uuid.uuid4().hex.upper()
        carrier_names = [f"{_SIDECAR_ENV_CARRIER_PREFIX}{invocation_id}_{index}" for index in range(len(validated))]
        if occupied_names.isdisjoint(carrier_names):
            break

    carrier_environment = dict(zip(carrier_names, validated.values(), strict=True))
    environment_args = [part for carrier in carrier_names for part in ("-e", carrier)]
    exports = [
        f'export {target_name}="${{{carrier_name}?missing sidecar environment carrier}}"'
        for target_name, carrier_name in zip(validated, carrier_names, strict=True)
    ]
    wrapper = "; ".join(
        (
            *exports,
            f"unset {' '.join(carrier_names)}",
            'exec /bin/sh -c "$1"',
        )
    )
    return environment_args, carrier_environment, wrapper


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
    include_short: bool = False,
) -> str | None:
    if text is None:
        return None
    redactor = _StreamingSecretRedactor(
        secret_values,
        _replacement=replacement,
        _include_short=include_short,
    )
    return redactor.feed(text) + redactor.finish()


def _redact_result(
    result: ExecResult,
    secret_values: set[str],
    *,
    replacement: str | None = None,
    include_short: bool = False,
) -> ExecResult:
    return ExecResult(
        stdout=_redact(
            result.stdout,
            secret_values,
            replacement=replacement,
            include_short=include_short,
        ),
        stderr=_redact(
            result.stderr,
            secret_values,
            replacement=replacement,
            include_short=include_short,
        ),
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


def _force_kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Force-kill without evaluating POSIX-only signal constants on Windows."""
    if os.name == "posix":
        _signal_process_tree(process, signal.SIGKILL)
    elif process.returncode is None:
        process.kill()


@contextlib.contextmanager
def _raw_lifecycle_deadline_scope() -> Iterator[None]:
    """Share one monotonic deadline across containment, reap, and restore."""
    if _RAW_LIFECYCLE_DEADLINE.get() is not None:
        yield
        return
    deadline = asyncio.get_running_loop().time() + _RAW_LIFECYCLE_TOTAL_TIMEOUT_SECONDS
    token = _RAW_LIFECYCLE_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _RAW_LIFECYCLE_DEADLINE.reset(token)


def _bounded_cleanup_timeout(
    maximum: float,
    *,
    allow_expired_reap: bool = False,
) -> float:
    deadline = _RAW_LIFECYCLE_DEADLINE.get()
    if deadline is None:
        return maximum
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        if allow_expired_reap:
            # A raw lifecycle deadline must never prevent the local Docker
            # client from receiving SIGKILL and a final bounded reap attempt.
            return max(0.001, min(maximum, _COMPOSE_CANCEL_SECONDS))
        raise RuntimeError("Docker lifecycle cleanup deadline expired")
    # Never return zero: several asyncio and Docker timeout APIs interpret it
    # as an unlimited wait.
    return max(0.001, min(maximum, remaining))


async def _terminate_process_tree(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[Any],
    *,
    preserve_cancellation: bool,
) -> None:
    async def reap() -> None:
        _signal_process_tree(process, signal.SIGTERM)
        try:
            terminate_timeout = _bounded_cleanup_timeout(_COMPOSE_TERMINATE_SECONDS)
        except RuntimeError:
            terminate_timeout = 0
        done, _pending = await asyncio.wait({communication}, timeout=terminate_timeout)
        if communication in done and process.returncode is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                communication.result()
            return

        _force_kill_process_tree(process)
        done, _pending = await asyncio.wait(
            {communication},
            timeout=_bounded_cleanup_timeout(
                _COMPOSE_KILL_SECONDS,
                allow_expired_reap=True,
            ),
        )
        if communication in done:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                communication.result()
        else:
            communication.cancel()
            done, _pending = await asyncio.wait(
                {communication},
                timeout=_bounded_cleanup_timeout(
                    _COMPOSE_CANCEL_SECONDS,
                    allow_expired_reap=True,
                ),
            )
            if communication in done:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    communication.result()
            # asyncio cannot forcibly terminate a coroutine that deliberately
            # suppresses CancelledError. Keep this wait bounded; cooperative
            # callbacks are owned and reaped above, while a hostile callback can
            # only be left for event-loop shutdown after ignoring cancellation.
        if process.returncode is None:
            raise RuntimeError("could not confirm Docker client process termination")

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

    def _trusted_docker_client_environment(self) -> dict[str, str]:
        """Build a host-only Docker CLI environment without Compose/task state."""
        trusted_environment: dict[str, str] = {}
        for name, value in os.environ.items():
            if name.upper().startswith("COMPOSE_") or not self._is_trusted_compose_client_host_name(name):
                continue
            # A task override with the same name must not remove the genuine
            # host Docker control. The value here comes only from os.environ;
            # coincidental byte overlap with attacker-chosen task values cannot
            # be allowed to disable host-authoritative containment.
            trusted_environment[name] = value
        if "PATH" not in trusted_environment:
            raise RuntimeError("trusted Docker client PATH is unavailable")
        return trusted_environment

    async def _run_trusted_docker_command(
        self,
        command: list[str],
    ) -> ExecResult:
        """Run a bounded raw-Docker command under the scrubbed host baseline."""
        process_environment = self._trusted_docker_client_environment()
        docker_executable = shutil.which(
            "docker",
            path=process_environment.get("PATH"),
        )
        if docker_executable is None:
            raise RuntimeError("trusted Docker client executable was not found")
        full_command = [docker_executable, *command]
        creation_timeout = _bounded_cleanup_timeout(_RAW_DOCKER_COMMAND_TIMEOUT_SECONDS)
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *full_command,
                env=process_environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        )
        try:
            process = await asyncio.wait_for(
                asyncio.shield(creation),
                timeout=creation_timeout,
            )
        except (TimeoutError, asyncio.CancelledError) as primary_error:

            async def reap_late_creation(
                completed_creation: asyncio.Task[asyncio.subprocess.Process],
            ) -> None:
                try:
                    late_process = completed_creation.result()
                except BaseException:
                    return
                late_communication = asyncio.create_task(late_process.communicate())
                await _terminate_process_tree(
                    late_process,
                    late_communication,
                    preserve_cancellation=False,
                )

            def schedule_late_reap(
                completed_creation: asyncio.Task[asyncio.subprocess.Process],
            ) -> None:
                late_cleanup = asyncio.create_task(reap_late_creation(completed_creation))

                def report_late_cleanup_failure(
                    completed_cleanup: asyncio.Task[None],
                ) -> None:
                    try:
                        completed_cleanup.result()
                    except BaseException as exc:
                        asyncio.get_running_loop().call_exception_handler(
                            {
                                "message": "late Docker client creation cleanup failed",
                                "exception": exc,
                                "task": completed_cleanup,
                            }
                        )

                late_cleanup.add_done_callback(report_late_cleanup_failure)

            async def cancel_creation_race() -> tuple[
                asyncio.subprocess.Process | None,
                bool,
            ]:
                creation.cancel()
                try:
                    cancellation_timeout = _bounded_cleanup_timeout(_COMPOSE_CANCEL_SECONDS)
                except RuntimeError:
                    cancellation_timeout = 0
                done, _pending = await asyncio.wait(
                    {creation},
                    timeout=cancellation_timeout,
                )
                if creation not in done:
                    creation.add_done_callback(schedule_late_reap)
                    return None, False
                try:
                    return creation.result(), True
                except BaseException:
                    return None, True

            cancellation = asyncio.create_task(cancel_creation_race())
            process, creation_resolved = await _await_task_uninterruptibly(
                cancellation,
                preserve_cancellation=False,
            )
            if process is not None:
                communication = asyncio.create_task(process.communicate())
                await _terminate_process_tree(
                    process,
                    communication,
                    preserve_cancellation=False,
                )
            if not creation_resolved:
                primary_error.add_note(
                    "Docker client creation cancellation remained pending past the cleanup deadline; "
                    "a late-process reaper was installed"
                )
            if isinstance(primary_error, TimeoutError):
                raise RuntimeError("trusted Docker client creation timed out") from primary_error
            raise

        communication = asyncio.create_task(process.communicate())
        try:
            communication_timeout = _bounded_cleanup_timeout(_RAW_DOCKER_COMMAND_TIMEOUT_SECONDS)
            done, _pending = await asyncio.wait(
                {communication},
                timeout=communication_timeout,
            )
            if communication not in done:
                raise _ComposeCommandTimeout
            stdout_bytes, stderr_bytes = communication.result()
        except BaseException as primary_error:
            await _terminate_process_tree(
                process,
                communication,
                preserve_cancellation=False,
            )
            if isinstance(primary_error, _ComposeCommandTimeout):
                raise RuntimeError("trusted Docker client command timed out") from primary_error
            raise
        return ExecResult(
            stdout=(stdout_bytes.decode(errors="replace") if stdout_bytes else None),
            stderr=(stderr_bytes.decode(errors="replace") if stderr_bytes else None),
            return_code=process.returncode or 0,
        )

    async def _raw_filtered_service_container_ids(self, service: str) -> tuple[str, ...]:
        """Resolve exact IDs from Docker's authoritative Compose-label index."""
        service = _validate_compose_service_name(service)
        project = _sanitize_docker_compose_project_name(self.session_id)
        result = await self._run_trusted_docker_command(
            [
                "container",
                "ls",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--filter",
                f"label=com.docker.compose.service={service}",
                "--filter",
                "label=com.docker.compose.oneoff=False",
                "--filter",
                "label=com.docker.compose.config-hash",
            ]
        )
        if result.return_code != 0:
            raise RuntimeError("could not resolve Docker service containers")
        identities = tuple(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
        if len(set(identities)) != len(identities) or any(
            not re.fullmatch(r"[0-9a-f]{64}", identity, re.IGNORECASE) for identity in identities
        ):
            raise RuntimeError("Docker returned an invalid service container identity")
        return identities

    async def _raw_service_container_ids(self, service: str) -> tuple[str, ...]:
        """Resolve and validate every managed container for one service."""
        service = _validate_compose_service_name(service)
        identities = await self._raw_filtered_service_container_ids(service)
        states = await self._raw_container_states(identities, service=service)
        if len({state.container_number for state in states.values()}) != len(states):
            raise RuntimeError("Docker returned duplicate service container numbers")
        return tuple(
            state.identity
            for state in sorted(
                states.values(),
                key=lambda state: state.container_number,
            )
        )

    async def _raw_container_states(
        self,
        identities: Iterable[str],
        *,
        service: str,
    ) -> dict[str, _RawContainerState]:
        service = _validate_compose_service_name(service)
        validated_identities = tuple(identities)
        if not validated_identities:
            return {}
        if any(not re.fullmatch(r"[0-9a-f]{64}", identity, re.IGNORECASE) for identity in validated_identities):
            raise RuntimeError("invalid Docker container identity")
        result = await self._run_trusted_docker_command(
            [
                "container",
                "inspect",
                "--format",
                (
                    '{{.Id}}\t{{index .Config.Labels "com.docker.compose.project"}}'
                    '\t{{index .Config.Labels "com.docker.compose.service"}}'
                    '\t{{index .Config.Labels "com.docker.compose.container-number"}}'
                    '\t{{index .Config.Labels "com.docker.compose.oneoff"}}'
                    '\t{{index .Config.Labels "com.docker.compose.config-hash"}}'
                    "\t{{.State.Running}}\t{{.State.Paused}}"
                    "\t{{.State.Restarting}}\t{{.State.Status}}"
                    '\t{{with index .State "Health"}}{{.Status}}{{else}}none{{end}}'
                ),
                "--",
                *validated_identities,
            ]
        )
        if result.return_code != 0:
            raise RuntimeError("could not inspect Docker service containers")
        lines = (result.stdout or "").splitlines()
        if len(lines) != len(validated_identities):
            raise RuntimeError("Docker returned incomplete container state")
        project = _sanitize_docker_compose_project_name(self.session_id)
        states: dict[str, _RawContainerState] = {}
        for identity, line in zip(validated_identities, lines, strict=True):
            fields = line.split("\t")
            if len(fields) != 11:
                raise RuntimeError("Docker returned invalid container state")
            (
                rendered_identity,
                rendered_project,
                rendered_service,
                rendered_number,
                rendered_oneoff,
                rendered_config_hash,
                rendered_running,
                rendered_paused,
                rendered_restarting,
                rendered_status,
                rendered_health,
            ) = fields
            if (
                rendered_identity.lower() != identity.lower()
                or rendered_project != project
                or rendered_service != service
                or rendered_oneoff != "False"
                or re.fullmatch(r"[0-9a-f]{64}", rendered_config_hash, re.IGNORECASE) is None
                or re.fullmatch(r"[1-9][0-9]*", rendered_number) is None
                or rendered_running not in {"true", "false"}
                or rendered_paused not in {"true", "false"}
                or rendered_restarting not in {"true", "false"}
                or rendered_status
                not in {
                    "created",
                    "running",
                    "paused",
                    "restarting",
                    "removing",
                    "exited",
                    "dead",
                }
                or rendered_health not in {"none", "starting", "healthy", "unhealthy"}
            ):
                raise RuntimeError("Docker returned invalid container state")
            states[identity] = _RawContainerState(
                identity=identity,
                project=rendered_project,
                service=rendered_service,
                container_number=int(rendered_number),
                running=rendered_running == "true",
                paused=rendered_paused == "true",
                restarting=rendered_restarting == "true",
                status=rendered_status,
                health_status=(None if rendered_health == "none" else rendered_health),
            )
        return states

    async def _raw_docker_action(
        self,
        action: list[str],
        identities: Iterable[str],
    ) -> bool:
        validated_identities = tuple(identities)
        if not validated_identities:
            return True
        result = await self._run_trusted_docker_command([*action, "--", *validated_identities])
        return result.return_code == 0

    async def _stop_raw_service_containers(
        self,
        service: str,
        *,
        remove: bool,
        require_existing: bool,
    ) -> _RawServiceSnapshot:
        service = _validate_compose_service_name(service)
        stop_failure: BaseException | None = None
        try:
            initial_identities = await self._raw_service_container_ids(service)
            initial_states = await self._raw_container_states(
                initial_identities,
                service=service,
            )
        except BaseException as exc:
            if not remove:
                raise
            stop_failure = exc
            initial_identities = await self._raw_filtered_service_container_ids(service)
            initial_states = {}
        if require_existing and not initial_identities:
            raise RuntimeError(f"could not resolve a container for sidecar service {service!r}")
        if not initial_identities:
            return _RawServiceSnapshot((), ())
        restore_identities = tuple(
            identity
            for identity in initial_identities
            if identity in initial_states and initial_states[identity].running
        )

        try:
            if stop_failure is not None:
                raise stop_failure
            for action in (
                ["container", "stop", "--timeout", "0"],
                ["container", "kill", "--signal", "SIGKILL"],
            ):
                current_identities = await self._raw_service_container_ids(service)
                states = await self._raw_container_states(
                    current_identities,
                    service=service,
                )
                running_identities = tuple(identity for identity, state in states.items() if state.running)
                if not running_identities:
                    break
                await self._raw_docker_action(action, running_identities)

            current_identities = await self._raw_service_container_ids(service)
            states = await self._raw_container_states(
                current_identities,
                service=service,
            )
            if any(state.running or state.restarting or state.paused for state in states.values()):
                raise RuntimeError(f"could not confirm Docker service {service!r} stopped")
        except BaseException as exc:
            if not remove:
                raise
            stop_failure = exc

        if remove:
            removal_failure: BaseException | None = stop_failure
            removal_candidates = set(initial_identities)
            for _attempt in range(2):
                try:
                    removal_candidates = set(await self._raw_filtered_service_container_ids(service))
                except BaseException as exc:
                    removal_failure = exc
                if not removal_candidates:
                    break
                try:
                    await self._raw_docker_action(
                        ["container", "rm", "--force", "--volumes"],
                        removal_candidates,
                    )
                except BaseException as exc:
                    removal_failure = exc
            try:
                remaining_identities = await self._raw_filtered_service_container_ids(service)
            except BaseException as exc:
                raise RuntimeError(f"could not confirm Docker service {service!r} removal") from exc
            if remaining_identities:
                error = RuntimeError(f"could not confirm Docker service {service!r} removal")
                if removal_failure is not None:
                    raise error from removal_failure
                raise error
        return _RawServiceSnapshot(initial_identities, restore_identities)

    async def _contain_main_container(self) -> None:
        """Remove only this project's main container after an interrupted exec."""
        await self._stop_raw_service_containers(
            MAIN_SERVICE_NAME,
            remove=True,
            require_existing=False,
        )

    async def _restore_sidecar_service(
        self,
        service: str,
        *,
        snapshot: _RawServiceSnapshot,
    ) -> bool:
        service = _validate_compose_service_name(service)
        identities = snapshot.running_identities
        if not identities:
            return False
        current_identities = set(await self._raw_service_container_ids(service))
        if current_identities != set(snapshot.all_identities):
            return False
        states = await self._raw_container_states(
            snapshot.all_identities,
            service=service,
        )
        if any(state.running or state.restarting for state in states.values()):
            return False
        if not await self._raw_docker_action(["container", "start"], identities):
            return False

        while True:
            if set(await self._raw_service_container_ids(service)) != set(snapshot.all_identities):
                return False
            states = await self._raw_container_states(
                snapshot.all_identities,
                service=service,
            )
            ready = True
            for identity in snapshot.all_identities:
                state = states[identity]
                if identity not in snapshot.running_identities:
                    if state.running or state.paused or state.restarting:
                        return False
                    continue
                if state.health_status == "unhealthy" or state.status in {"created", "removing", "exited", "dead"}:
                    return False
                if (
                    not state.running
                    or state.paused
                    or state.restarting
                    or state.status != "running"
                    or state.health_status not in {None, "healthy"}
                ):
                    ready = False
            if ready and len(states) == len(snapshot.all_identities):
                return True
            await asyncio.sleep(_bounded_cleanup_timeout(0.1))

    async def _contain_sidecar_service(self, service: str) -> _RawServiceSnapshot:
        """Stop only the target sidecar and retain its exact IDs for restart."""
        service = _validate_compose_service_name(service)
        if service == MAIN_SERVICE_NAME:
            raise ValueError("sidecar containment cannot target the main service")
        return await self._stop_raw_service_containers(
            service,
            remove=False,
            require_existing=True,
        )

    async def _contain_main_and_reap_compose(
        self,
        process: asyncio.subprocess.Process,
        communication: asyncio.Task[Any],
        *,
        contain_service_on_interrupt: str | None = None,
        stop_main_on_interrupt: bool,
    ) -> None:
        with _raw_lifecycle_deadline_scope():
            await self._contain_main_and_reap_compose_within_deadline(
                process,
                communication,
                contain_service_on_interrupt=contain_service_on_interrupt,
                stop_main_on_interrupt=stop_main_on_interrupt,
            )

    async def _contain_main_and_reap_compose_within_deadline(
        self,
        process: asyncio.subprocess.Process,
        communication: asyncio.Task[Any],
        *,
        contain_service_on_interrupt: str | None,
        stop_main_on_interrupt: bool,
    ) -> None:
        containment_error: BaseException | None = None
        reap_error: BaseException | None = None
        restoration_error: BaseException | None = None
        sidecar_snapshot: _RawServiceSnapshot | None = None
        if stop_main_on_interrupt and contain_service_on_interrupt is not None:
            raise ValueError("only one interrupt-containment target may be configured")
        if stop_main_on_interrupt:
            try:
                await self._contain_main_container()
            except BaseException as exc:
                containment_error = exc
        elif contain_service_on_interrupt is not None:
            try:
                sidecar_snapshot = await self._contain_sidecar_service(contain_service_on_interrupt)
            except BaseException as exc:
                containment_error = exc
        try:
            await _terminate_process_tree(
                process,
                communication,
                preserve_cancellation=False,
            )
        except BaseException as exc:
            reap_error = exc

        if (
            containment_error is None
            and reap_error is None
            and contain_service_on_interrupt is not None
            and sidecar_snapshot is not None
        ):
            try:
                restored = await self._restore_sidecar_service(
                    contain_service_on_interrupt,
                    snapshot=sidecar_snapshot,
                )
                if not restored:
                    raise RuntimeError(f"sidecar service {contain_service_on_interrupt!r} could not be restored")
            except BaseException as exc:
                restoration_error = exc

        failures = [error for error in (containment_error, reap_error, restoration_error) if error is not None]
        if failures:
            target = (
                "main task container" if stop_main_on_interrupt else f"sidecar service {contain_service_on_interrupt!r}"
            )
            error = RuntimeError(f"could not confirm {target} containment and restoration")
            for additional_error in failures[1:]:
                error.add_note(
                    f"Additional containment cleanup failure: {type(additional_error).__name__}: {additional_error}"
                )
            raise error from failures[0]

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
        exact_secret_values = _sensitive_environment_values(merged_environment)

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
            **({"exact_secret_values": exact_secret_values} if exact_secret_values else {}),
        )

    def _main_only_compose_environment(self) -> tuple[set[str], set[str]]:
        """Return main-only names and values that a sidecar client must not inherit."""
        main_names: set[str] = set()
        main_values: set[str] = set()

        def include(environment: Mapping[str, str]) -> None:
            main_names.update(environment)
            main_values.update(_eligible_secret_values(environment.values()))
            main_values.update(_sensitive_environment_values(environment))

        include(getattr(self, "_compose_task_env", {}))
        include(getattr(self, "_persistent_env", {}))
        for scoped_environment in self._exec_env_overlays.get():
            include(scoped_environment)

        active_handoff_scopes = _SECURE_HANDOFF_SCOPES.get()
        for handoff_scope in active_handoff_scopes:
            main_names.update(handoff_scope.environment_names)
            main_values.update(
                _eligible_secret_values(
                    handoff_scope.secret_values,
                    include_short=True,
                )
            )

        sentinel_names = {
            "NVIDIA_API_KEY",
            NVIDIA_BUILD_KEY_STDIN_ENV,
            _NVIDIA_BUILD_KEY_FILE_ENV,
        }
        for name in sentinel_names:
            value = os.environ.get(name)
            if value is not None:
                main_values.update(_eligible_secret_values((value,)))
                if name != NVIDIA_BUILD_KEY_STDIN_ENV:
                    main_values.update(_eligible_secret_values((value,), include_short=True))

        main_values.update(
            _eligible_secret_values(
                (
                    NVIDIA_BUILD_STDIN_SENTINEL,
                    _NVIDIA_BUILD_FILE_SENTINEL,
                )
            )
        )
        return main_names | sentinel_names, main_values

    def _sidecar_exec_lock(self, service: str) -> asyncio.Lock:
        locks = getattr(self, "_skillevaluator_sidecar_exec_locks", None)
        if locks is None:
            locks = {}
            self._skillevaluator_sidecar_exec_locks = locks
        lock = locks.get(service)
        if lock is None:
            lock = asyncio.Lock()
            locks[service] = lock
        return lock

    def _compose_model_metadata(self) -> tuple[set[str], set[str]]:
        """Inspect Compose interpolation and service keys with one hardened walk.

        Compose's own ``config --no-interpolate`` rejects valid unresolved
        values in typed fields (for example ``ports[].host_ip``), so it cannot
        safely serve as metadata discovery. Parse YAML values instead and fail
        closed for dynamic or external include/extends inputs.
        """
        environment_root = self.environment_dir.resolve()
        root_paths = [path.resolve() for path in self._docker_compose_paths]
        pending_models = [(path, environment_root) for path in root_paths]
        trusted_roots = {environment_root, *(path.parent for path in root_paths)}
        inspected_models: set[tuple[Path, Path]] = set()
        interpolation_names: set[str] = set()
        declared_services: set[str] = set()
        total_bytes = 0

        def is_trusted_path(path: Path) -> bool:
            return any(path.is_relative_to(root) for root in trusted_roots)

        def resolve_model_path(raw_path: object, *, relative_to: Path) -> Path:
            if not isinstance(raw_path, str) or "$" in raw_path:
                raise RuntimeError("could not inspect Docker Compose interpolation inputs")
            referenced_path = (relative_to / raw_path).resolve()
            if not is_trusted_path(referenced_path):
                raise RuntimeError("could not inspect Docker Compose interpolation inputs")
            return referenced_path

        def reject_include_dotenv(project_directory: Path) -> None:
            if (project_directory / ".env").exists():
                raise RuntimeError("could not inspect Docker Compose interpolation inputs")

        while pending_models:
            path, project_directory = pending_models.pop()
            model_identity = (path, project_directory)
            if model_identity in inspected_models:
                continue
            if not is_trusted_path(path):
                raise RuntimeError("could not inspect Docker Compose interpolation inputs")
            inspected_models.add(model_identity)
            if len(inspected_models) > _MAX_COMPOSE_MODEL_FILES:
                raise RuntimeError("could not inspect Docker Compose interpolation inputs")
            try:
                flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(path, flags)
                with os.fdopen(descriptor, "rb") as compose_file:
                    if not stat.S_ISREG(os.fstat(compose_file.fileno()).st_mode):
                        raise RuntimeError("could not inspect Docker Compose interpolation inputs")
                    remaining_bytes = _MAX_COMPOSE_MODEL_BYTES - total_bytes
                    content = compose_file.read(remaining_bytes + 1)
                if len(content) > remaining_bytes:
                    raise RuntimeError("could not inspect Docker Compose interpolation inputs")
                total_bytes += len(content)
                # _ComposeModelLoader subclasses yaml.SafeLoader and only adds
                # bounded-node accounting; arbitrary object construction stays disabled.
                model = yaml.load(  # nosec B506
                    content.decode("utf-8"),
                    Loader=_ComposeModelLoader,
                )
            except (OSError, RecursionError, UnicodeDecodeError, yaml.YAMLError) as exc:
                raise RuntimeError("could not inspect Docker Compose interpolation inputs") from exc
            if not isinstance(model, Mapping):
                raise RuntimeError("could not inspect Docker Compose interpolation inputs")

            values = [(value, 1) for value in model.values()]
            visited_containers: set[int] = {id(model)}
            inspected_nodes = 0
            while values:
                value, depth = values.pop()
                inspected_nodes += 1
                if inspected_nodes > _MAX_COMPOSE_MODEL_NODES or depth > _MAX_COMPOSE_MODEL_DEPTH:
                    raise RuntimeError("could not inspect Docker Compose interpolation inputs")
                if isinstance(value, str):
                    interpolation_names.update(_compose_interpolation_names(value))
                elif isinstance(value, Mapping):
                    if id(value) in visited_containers:
                        continue
                    visited_containers.add(id(value))
                    # Compose interpolates YAML values, not mapping keys.
                    values.extend((nested_value, depth + 1) for nested_value in value.values())
                elif isinstance(value, list):
                    if id(value) in visited_containers:
                        continue
                    visited_containers.add(id(value))
                    values.extend((nested_value, depth + 1) for nested_value in value)

            includes = model.get("include", [])
            if not isinstance(includes, list):
                includes = [includes]
            for include in includes:
                if isinstance(include, str):
                    include_path = resolve_model_path(
                        include,
                        relative_to=project_directory,
                    )
                    reject_include_dotenv(include_path.parent)
                    pending_models.append((include_path, include_path.parent))
                    continue
                if (
                    not isinstance(include, Mapping)
                    or "env_file" in include
                    or include.get("project_directory") is not None
                ):
                    raise RuntimeError("could not inspect Docker Compose interpolation inputs")
                include_paths = include.get("path")
                if not isinstance(include_paths, list):
                    include_paths = [include_paths]
                resolved_include_paths = [
                    resolve_model_path(
                        include_path,
                        relative_to=project_directory,
                    )
                    for include_path in include_paths
                ]
                if not resolved_include_paths:
                    raise RuntimeError("could not inspect Docker Compose interpolation inputs")
                included_project_directory = resolved_include_paths[0].parent
                reject_include_dotenv(included_project_directory)
                pending_models.extend(
                    (include_path, included_project_directory) for include_path in resolved_include_paths
                )

            services = model.get("services", {})
            if not isinstance(services, Mapping):
                raise RuntimeError("could not inspect Docker Compose interpolation inputs")
            for service_name, service in services.items():
                if not isinstance(service_name, str):
                    raise RuntimeError("could not inspect Docker Compose interpolation inputs")
                declared_services.add(_validate_compose_service_name(service_name))
                if not isinstance(service, Mapping):
                    continue
                extends = service.get("extends")
                if not isinstance(extends, Mapping) or "file" not in extends:
                    continue
                extends_path = resolve_model_path(
                    extends["file"],
                    relative_to=project_directory,
                )
                pending_models.append((extends_path, project_directory))

        return interpolation_names, declared_services

    def _compose_model_interpolation_names(self) -> set[str]:
        return self._compose_model_metadata()[0]

    def _declared_compose_service_names(self) -> set[str]:
        return self._compose_model_metadata()[1]

    async def _required_compose_model_environment_names(self) -> set[str]:
        """Return available, non-infrastructure names required by the model."""
        available_names = set(self._compose_env_vars(include_os_env=True))
        possible_names = self._compose_model_interpolation_names()
        infrastructure_names = set(self._compose_infra_env_vars())
        if self._windows_container_name:
            infrastructure_names.add("HARBOR_CONTAINER_NAME")
        return {name for name in possible_names & available_names if name not in infrastructure_names}

    def _protected_main_environment_values(self) -> set[str]:
        protected_values = {
            NVIDIA_BUILD_STDIN_SENTINEL,
            _NVIDIA_BUILD_FILE_SENTINEL,
        }
        exact_protected_values: set[str] = set()

        def include(environment: Mapping[str, str]) -> None:
            exact_protected_values.update(
                value
                for name, value in environment.items()
                if name != NVIDIA_BUILD_KEY_STDIN_ENV and value and _SENSITIVE_ENV_NAME_RE.search(name)
            )

        include(os.environ)
        include(getattr(self, "_compose_task_env", {}))
        include(getattr(self, "_persistent_env", {}))
        for scoped_environment in self._exec_env_overlays.get():
            include(scoped_environment)
        for name in (
            "NVIDIA_API_KEY",
            NVIDIA_BUILD_KEY_STDIN_ENV,
            _NVIDIA_BUILD_KEY_FILE_ENV,
        ):
            value = os.environ.get(name)
            if value:
                protected_values.add(value)
                if name != NVIDIA_BUILD_KEY_STDIN_ENV:
                    exact_protected_values.add(value)
        return {
            *_eligible_secret_values(protected_values),
            *exact_protected_values,
        }

    def _other_main_environment_values(
        self,
        retained_name: str,
        retained_value: str,
    ) -> set[str]:
        protected_values: set[str] = set()

        def include(environment: Mapping[str, str]) -> None:
            for name, value in environment.items():
                if not value or (name == retained_name and value == retained_value):
                    continue
                if len(value) >= _MIN_EXACT_SECRET_LENGTH or _SENSITIVE_ENV_NAME_RE.search(name):
                    protected_values.add(value)

        include(getattr(self, "_compose_task_env", {}))
        include(getattr(self, "_persistent_env", {}))
        for scoped_environment in self._exec_env_overlays.get():
            include(scoped_environment)
        return set(
            _eligible_secret_values(
                protected_values,
                include_short=True,
            )
        )

    async def _sidecar_compose_model_environment(self) -> dict[str, str]:
        """Retain only non-sensitive task values structurally required by Compose."""
        possible_names = await self._required_compose_model_environment_names()
        if not possible_names:
            return {}

        required_names = possible_names
        compose_environment = _validate_environment(self._compose_env_vars(include_os_env=True))
        protected_values = self._protected_main_environment_values()
        retained_environment: dict[str, str] = {}
        for name in sorted(required_names):
            value = compose_environment[name]
            if _SENSITIVE_ENV_NAME_RE.search(name):
                raise RuntimeError(f"Docker Compose interpolation variable {name!r} requires protected execution state")
            if self._is_compose_client_operational_name(name):
                raise RuntimeError(f"Docker Compose interpolation variable {name!r} cannot use host client controls")
            if any(
                _value_contains_protected_value(value, protected_value)
                for protected_value in (protected_values | self._other_main_environment_values(name, value))
                if protected_value
            ):
                raise RuntimeError(f"Docker Compose interpolation variable {name!r} requires protected execution state")
            retained_environment[name] = value
        return retained_environment

    @contextlib.asynccontextmanager
    async def _sidecar_operation(
        self,
        service: str,
        *,
        discover_compose_model: bool = True,
    ) -> AsyncIterator[None]:
        """Serialize a service while rejecting callback reentry before waiting."""
        service = _validate_compose_service_name(service)
        active_operations = [operation for operation in _SIDECAR_EXEC_OPERATIONS.get() if operation.active]
        if any(
            operation.environment_identity == id(self) and operation.service == service
            for operation in active_operations
        ):
            raise RuntimeError(f"reentrant sidecar operation for service {service!r} is not supported")
        operation_key = (id(self), service)
        if active_operations and operation_key <= max(
            (operation.environment_identity, operation.service) for operation in active_operations
        ):
            raise RuntimeError("nested sidecar operation violates deterministic lock ordering")

        async with self._sidecar_exec_lock(service):
            compose_model_environment = (
                await self._sidecar_compose_model_environment() if discover_compose_model else {}
            )
            operation = _SidecarOperation(
                environment_identity=id(self),
                service=service,
                compose_model_environment=compose_model_environment,
            )
            token = _SIDECAR_EXEC_OPERATIONS.set((*_SIDECAR_EXEC_OPERATIONS.get(), operation))
            try:
                yield
            finally:
                operation.active = False
                _SIDECAR_EXEC_OPERATIONS.reset(token)

    @contextlib.contextmanager
    def _compose_environment_scrub_scope(
        self,
        environment_names: Iterable[str],
        secret_values: Iterable[str],
    ) -> Iterator[None]:
        scope = _SecureHandoffScope(
            environment_names=frozenset(environment_names),
            secret_values=frozenset(_eligible_secret_values(secret_values, include_short=True)),
        )
        token = _SECURE_HANDOFF_SCOPES.set((*_SECURE_HANDOFF_SCOPES.get(), scope))
        try:
            yield
        finally:
            _SECURE_HANDOFF_SCOPES.reset(token)

    async def stop_service(self, service: str) -> None:
        service = _validate_compose_service_name(service)
        async with self._sidecar_operation(
            service,
            discover_compose_model=False,
        ):
            if service not in self._declared_compose_service_names():
                raise RuntimeError(f"unknown Docker Compose service {service!r}")
            with _raw_lifecycle_deadline_scope():
                await self._stop_raw_service_containers(
                    service,
                    remove=False,
                    require_existing=False,
                )

    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        service = MAIN_SERVICE_NAME if service is None else service
        service = _validate_compose_service_name(service)
        excluded_names, excluded_values = self._main_only_compose_environment()

        if service == MAIN_SERVICE_NAME and self._is_windows_container:
            with tempfile.TemporaryDirectory() as temporary_directory:
                await self._secure_windows_download_dir(
                    str(Path(source_path).parent).replace("\\", "/"),
                    temporary_directory,
                    excluded_names=excluded_names,
                    excluded_values=excluded_values,
                )
                downloaded = Path(temporary_directory) / Path(source_path).name
                if not downloaded.is_file():
                    raise RuntimeError("requested file was not present in the Windows container download")
                shutil.copy2(downloaded, target_path)
            return

        async def download() -> None:
            if service == MAIN_SERVICE_NAME:
                await self._run_docker_compose_command(
                    ["cp", "--", f"{service}:{source_path}", str(target_path)],
                    check=True,
                    additional_secret_values=excluded_values,
                    compose_env_excluded_names=excluded_names,
                    compose_env_excluded_values=excluded_values,
                    use_sidecar_compose_model=True,
                )
                return
            self._sidecar_platform(service)
            await self._run_docker_compose_command(
                ["cp", "--", f"{service}:{source_path}", str(target_path)],
                check=True,
                additional_secret_values=excluded_values,
                compose_env_excluded_names=excluded_names,
                compose_env_excluded_values=excluded_values,
                use_sidecar_compose_model=True,
            )

        async with self._sidecar_operation(service):
            await download()

    async def service_download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        service = MAIN_SERVICE_NAME if service is None else service
        service = _validate_compose_service_name(service)
        excluded_names, excluded_values = self._main_only_compose_environment()

        if service == MAIN_SERVICE_NAME and self._is_windows_container:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            await self._secure_windows_download_dir(
                source_dir,
                target_dir,
                excluded_names=excluded_names,
                excluded_values=excluded_values,
            )
            return

        async def download() -> None:
            if service == MAIN_SERVICE_NAME:
                await self._run_docker_compose_command(
                    ["cp", "--", f"{service}:{source_dir}/.", str(target_dir)],
                    check=True,
                    additional_secret_values=excluded_values,
                    compose_env_excluded_names=excluded_names,
                    compose_env_excluded_values=excluded_values,
                    use_sidecar_compose_model=True,
                )
                return
            self._sidecar_platform(service)
            await self._run_docker_compose_command(
                ["cp", "--", f"{service}:{source_dir}/.", str(target_dir)],
                check=True,
                additional_secret_values=excluded_values,
                compose_env_excluded_names=excluded_names,
                compose_env_excluded_values=excluded_values,
                use_sidecar_compose_model=True,
            )

        async with self._sidecar_operation(service):
            await download()

    async def _run_trusted_transfer_command(
        self,
        command: list[str],
        *,
        process_environment: Mapping[str, str],
        protected_values: set[str],
        stdin_data: bytes | None = None,
    ) -> bytes:
        executable = shutil.which(command[0])
        if executable is None:
            raise RuntimeError(f"required host transfer executable {command[0]!r} was not found")
        full_command = [executable, *command[1:]]
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *full_command,
                env=dict(process_environment),
                stdin=(asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        )
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            process = await _await_task_uninterruptibly(creation, preserve_cancellation=False)
            communication = asyncio.create_task(process.communicate(input=stdin_data))
            await _terminate_process_tree(process, communication, preserve_cancellation=False)
            raise

        communication = asyncio.create_task(process.communicate(input=stdin_data))
        try:
            stdout, stderr = await asyncio.shield(communication)
        except asyncio.CancelledError:
            await _terminate_process_tree(process, communication, preserve_cancellation=False)
            raise
        if process.returncode != 0:
            detail = b"\n".join(part for part in (stdout, stderr) if part).decode(errors="replace")
            raise RuntimeError(
                _redact(
                    "secure Windows container transfer command failed: " + detail,
                    protected_values,
                    include_short=True,
                )
            )
        return stdout or b""

    async def _secure_windows_download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        excluded_names: set[str],
        excluded_values: set[str],
    ) -> None:
        container_name = self._windows_container_name
        if not container_name or not _COMPOSE_SERVICE_NAME_RE.fullmatch(container_name):
            raise RuntimeError("Windows container transfer target is invalid")
        process_environment = self._trusted_compose_client_environment(
            excluded_names=excluded_names,
            excluded_values=excluded_values,
        )
        tar_data = await self._run_trusted_transfer_command(
            [
                "docker",
                "exec",
                "--",
                container_name,
                "tar",
                "cf",
                "-",
                "-C",
                source_dir,
                ".",
            ],
            process_environment=process_environment,
            protected_values=excluded_values,
        )
        await self._run_trusted_transfer_command(
            ["tar", "xf", "-", "-C", str(target_dir)],
            process_environment=process_environment,
            protected_values=excluded_values,
            stdin_data=tar_data,
        )

    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute in main securely, or in an isolated POSIX sidecar."""
        if service is None or service == MAIN_SERVICE_NAME:
            return await self.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user=user,
            )
        if self._is_windows_container:
            raise ServiceOperationsUnsupportedError(
                f"Per-service operations are not supported for Windows containers (requested service: {service!r})."
            )
        service = _validate_compose_service_name(service)
        return await self._secure_compose_exec(
            command,
            service=service,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def _secure_compose_exec(
        self,
        command: str,
        *,
        service: str,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str | int | None,
    ) -> ExecResult:
        """Run a sidecar command without exposing values or main-only state."""
        service = _validate_compose_service_name(service)
        validated_environment = _validate_environment(env)
        async with self._sidecar_operation(service):
            reserved_names = set(self._compose_env_vars(include_os_env=True))
            environment_args, carrier_environment, environment_wrapper = _sidecar_environment_carriers(
                validated_environment,
                reserved_names=reserved_names,
            )
            carrier_names = set(carrier_environment)
            secret_values = set(_eligible_secret_values(validated_environment.values()))
            exact_secret_values = _sensitive_environment_values(validated_environment)
            secret_values.update(exact_secret_values)
            main_names, main_values = self._main_only_compose_environment()
            with self._compose_environment_scrub_scope(
                main_names | set(validated_environment) | carrier_names,
                main_values | secret_values,
            ):
                excluded_names, excluded_values = self._main_only_compose_environment()
                exec_command = ["exec"]
                if cwd:
                    exec_command.extend(["-w", cwd])
                exec_command.extend(environment_args)
                if user is not None:
                    exec_command.extend(["-u", str(user)])
                exec_command.extend(["--", service, "sh", "-c"])
                if environment_wrapper is None:
                    exec_command.append(command)
                else:
                    exec_command.extend([environment_wrapper, "sh", command])

                return await self._run_docker_compose_command(
                    exec_command,
                    check=False,
                    timeout_sec=timeout_sec,
                    on_output=self._output_callback(),
                    sidecar_env_carriers=carrier_environment,
                    additional_secret_values=excluded_values,
                    compose_env_excluded_names=excluded_names,
                    compose_env_excluded_values=excluded_values,
                    use_sidecar_compose_model=True,
                    contain_service_on_interrupt=service,
                    stop_main_on_interrupt=False,
                    **({"exact_secret_values": exact_secret_values} if exact_secret_values else {}),
                )

    @staticmethod
    def _is_compose_client_operational_name(name: str) -> bool:
        normalized_name = name.upper()
        return (
            normalized_name
            in {
                "PATH",
                "HOME",
                "USER",
                "LOGNAME",
                "TMPDIR",
                "TMP",
                "TEMP",
                "XDG_CONFIG_HOME",
                "XDG_RUNTIME_DIR",
                "LANG",
                "LANGUAGE",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "PATHEXT",
                "USERPROFILE",
                "HOMEDRIVE",
                "HOMEPATH",
                "APPDATA",
                "LOCALAPPDATA",
                "PROGRAMDATA",
                "SSH_AUTH_SOCK",
                "SSH_AGENT_PID",
                "DBUS_SESSION_BUS_ADDRESS",
                "GNUPGHOME",
                "TERM",
                "COLORTERM",
                "NO_COLOR",
            }
            or normalized_name.startswith(("DOCKER_", "COMPOSE_", "LC_"))
            or normalized_name.endswith("_PROXY")
        )

    @classmethod
    def _is_trusted_compose_client_host_name(cls, name: str) -> bool:
        normalized_name = name.upper()
        if normalized_name.startswith("COMPOSE_"):
            return normalized_name in {
                "COMPOSE_ANSI",
                "COMPOSE_HTTP_TIMEOUT",
                "COMPOSE_IGNORE_ORPHANS",
                "COMPOSE_PARALLEL_LIMIT",
                "COMPOSE_PROGRESS",
                "COMPOSE_STATUS_STDOUT",
            }
        return cls._is_compose_client_operational_name(name)

    def _trusted_compose_client_environment(
        self,
        *,
        excluded_names: set[str],
        excluded_values: set[str],
    ) -> dict[str, str]:
        """Build a host/Harbor baseline without main or sidecar target state."""
        # Values originate exclusively from the host operational allowlist or
        # Harbor-generated infrastructure. Incidental short caller-chosen byte
        # overlap must not disable or replace those authoritative controls.
        del excluded_names
        infrastructure = self._compose_infra_env_vars()
        trusted_environment = {
            name: value for name, value in os.environ.items() if self._is_trusted_compose_client_host_name(name)
        }
        trusted_environment.update(infrastructure)
        if self._windows_container_name:
            trusted_environment["HARBOR_CONTAINER_NAME"] = self._windows_container_name

        for name, value in trusted_environment.items():
            if any(
                value == protected_value
                or (len(protected_value) >= _MIN_EXACT_SECRET_LENGTH and protected_value in value)
                for protected_value in excluded_values
            ):
                raise RuntimeError(
                    f"trusted Docker Compose client environment variable {name!r} contains protected execution state"
                )
        return trusted_environment

    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: float | None = None,
        stdin_data: bytes | None = None,
        on_output: OutputCallback | None = None,
        *,
        env_overrides: Mapping[str, str] | None = None,
        sidecar_env_carriers: Mapping[str, str] | None = None,
        additional_secret_values: Iterable[str] | None = None,
        exact_secret_values: Iterable[str] | None = None,
        compose_env_excluded_names: Iterable[str] | None = None,
        compose_env_excluded_values: Iterable[str] | None = None,
        use_sidecar_compose_model: bool = False,
        contain_service_on_interrupt: str | None = None,
        stop_main_on_interrupt: bool = False,
    ) -> ExecResult:
        """Run compose with sensitive values only in child env or stdin."""
        if stop_main_on_interrupt and contain_service_on_interrupt is not None:
            raise ValueError("only one interrupt-containment target may be configured")
        if contain_service_on_interrupt is not None:
            contain_service_on_interrupt = _validate_compose_service_name(contain_service_on_interrupt)
            if contain_service_on_interrupt == MAIN_SERVICE_NAME:
                raise ValueError("sidecar containment cannot target the main service")
        containment_target = (
            "main task container"
            if stop_main_on_interrupt
            else (
                f"sidecar service {contain_service_on_interrupt!r}"
                if contain_service_on_interrupt is not None
                else "Docker Compose client"
            )
        )
        docker_executable = shutil.which("docker") or "docker"
        full_command = [
            docker_executable,
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
        effective_env_overrides = _validate_environment(env_overrides)
        carrier_overrides = _validate_environment(sidecar_env_carriers)
        if carrier_overrides and (
            not active_handoff_scopes
            or any(not name.startswith(_SIDECAR_ENV_CARRIER_PREFIX) for name in carrier_overrides)
        ):
            raise ValueError("sidecar environment carriers require an active secure sidecar scope")
        active_handoff_names: set[str] = set()
        active_handoff_values: set[str] = set()
        for handoff_scope in active_handoff_scopes:
            active_handoff_names.update(handoff_scope.environment_names)
            active_handoff_values.update(handoff_scope.secret_values)
        if active_handoff_scopes:
            active_handoff_names.update(effective_env_overrides)
            active_handoff_values.update(_eligible_secret_values(effective_env_overrides.values()))
            active_handoff_names.update(carrier_overrides)
            active_handoff_values.update(_eligible_secret_values(carrier_overrides.values()))

        isolate_compose_base = compose_env_excluded_names is not None or bool(active_handoff_scopes)
        compose_model_environment: dict[str, str] = {}
        if isolate_compose_base:
            active_operation = next(
                (
                    operation
                    for operation in reversed(_SIDECAR_EXEC_OPERATIONS.get())
                    if operation.active and operation.environment_identity == id(self)
                ),
                None,
            )
            if active_operation is not None:
                compose_model_environment = active_operation.compose_model_environment
            elif use_sidecar_compose_model:
                raise RuntimeError("sidecar Compose model requested outside a protected operation")
            else:
                compose_model_environment = await self._sidecar_compose_model_environment()
        if isolate_compose_base:
            excluded_names = set(compose_env_excluded_names or ()) | active_handoff_names
            excluded_values = set(
                _eligible_secret_values(
                    compose_env_excluded_values or (),
                    include_short=True,
                )
            )
            excluded_values.update(active_handoff_values)
            process_environment = self._trusted_compose_client_environment(
                excluded_names=excluded_names,
                excluded_values=excluded_values,
            )
            # Never let Compose auto-load project .env files or a host-provided
            # COMPOSE_ENV_FILES path across this isolation boundary.
            process_environment["COMPOSE_DISABLE_ENV_FILE"] = "1"
            # Only collision-checked internal carriers may cross an active
            # scrub scope. Arbitrary overrides remain scrubbed as in Task 4.
            process_environment.update(compose_model_environment)
            process_environment.update(carrier_overrides)
        else:
            process_environment = self._compose_env_vars(include_os_env=True)
            process_environment.update(effective_env_overrides)

        secret_values = set(_eligible_secret_values(effective_env_overrides.values()))
        secret_values.update(_eligible_secret_values(carrier_overrides.values()))
        secret_values.update(_eligible_secret_values(compose_model_environment.values()))
        secret_values.update(
            _eligible_secret_values(
                additional_secret_values or (),
                include_short=True,
            )
        )
        secret_values.update(active_handoff_values)
        secret_values.update(_eligible_secret_values(exact_secret_values or (), include_short=True))
        redaction_marker = _collision_safe_redaction_marker(
            secret_values,
            include_short=True,
        )
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
        except asyncio.CancelledError as primary_error:
            process = await _await_task_uninterruptibly(creation, preserve_cancellation=False)
            communication = asyncio.create_task(process.communicate())
            cleanup = asyncio.create_task(
                self._contain_main_and_reap_compose(
                    process,
                    communication,
                    contain_service_on_interrupt=contain_service_on_interrupt,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"{containment_target} containment could not be confirmed during cancellation: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                raise primary_error from cleanup_error
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
                _include_short=True,
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
                    contain_service_on_interrupt=contain_service_on_interrupt,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except BaseException as cleanup_error:
                primary_error.add_note(
                    "Docker Compose cleanup or container containment/restoration also failed: "
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
            primary_error = RuntimeError(f"Command timed out after {timeout_sec} seconds")
            cleanup = asyncio.create_task(
                self._contain_main_and_reap_compose(
                    process,
                    communication,
                    contain_service_on_interrupt=contain_service_on_interrupt,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup)
            except Exception as cleanup_error:
                primary_error.add_note(
                    f"{containment_target} containment could not be confirmed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                raise primary_error from cleanup_error
            raise primary_error from None
        except asyncio.CancelledError as primary_error:
            cleanup = asyncio.create_task(
                self._contain_main_and_reap_compose(
                    process,
                    communication,
                    contain_service_on_interrupt=contain_service_on_interrupt,
                    stop_main_on_interrupt=stop_main_on_interrupt,
                )
            )
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"{containment_target} containment could not be confirmed during cancellation: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                raise primary_error from cleanup_error
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
                    include_short=True,
                )
            )
        return _redact_result(
            result,
            secret_values,
            replacement=redaction_marker,
            include_short=True,
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
            **({"exact_secret_values": secret_values} if secret_values else {}),
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
        secret_values.update(_sensitive_environment_values(merged))
        if secret_values:
            # Fail before writing or uploading a handoff if no structurally
            # safe output marker can represent these values.
            _collision_safe_redaction_marker(secret_values, include_short=True)
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
                exact_secret_values=secret_values,
            )

            if user is None:
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chmod", "600", remote_path],
                    check=True,
                    additional_secret_values=secret_values,
                    exact_secret_values=secret_values,
                )
            else:
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chown", "--", str(user), remote_path],
                    check=True,
                    additional_secret_values=secret_values,
                    exact_secret_values=secret_values,
                )
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chmod", "600", remote_path],
                    check=True,
                    additional_secret_values=secret_values,
                    exact_secret_values=secret_values,
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
