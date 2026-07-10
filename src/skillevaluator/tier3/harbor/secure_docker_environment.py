# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker environments that keep per-exec credentials out of process argv.

The pinned Harbor release serializes ``exec(env=...)`` values as
``docker compose exec -e NAME=value``. Process arguments are host-visible, so
the compatibility backend passes only names on argv and values through the
compose subprocess environment. SkillEvaluator's selected backend is stronger:
it transfers values through a short-lived, file-backed handoff and removes the
container copy before running the requested command.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import shlex
import signal
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name

SECURE_DOCKER_ENV_IMPORT_PATH = (
    "skillevaluator.tier3.harbor.secure_docker_environment:SkillEvaluatorSecureDockerEnvironment"
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NVIDIA_BUILD_FILE_SENTINEL = "skillevaluator-file-backed-nvidia-key"
_NVIDIA_BUILD_KEY_FILE_ENV = "SKILLEVALUATOR_NVIDIA_API_KEY_FILE"
_COMPOSE_TERMINATE_SECONDS = 5.0
_COMPOSE_KILL_SECONDS = 5.0
_COMPOSE_CANCEL_SECONDS = 0.1
_REMOTE_CLEANUP_TIMEOUT_SECONDS = 8

_REMOTE_GROUP_SCRIPT = r"""
child=
trap ':' TERM INT HUP
unset SKILLEVALUATOR_EXEC_TOKEN
"$@" &
child=$!
status=0
while true; do
    if wait "$child"; then status=0; else status=$?; fi
    if ! kill -0 "$child" 2>/dev/null; then break; fi
done
exit "$status"
""".strip()

_REMOTE_CONTROLLER_SCRIPT = r"""
marker=$1
token=$2
group_script=$3
shift 3
umask 077
temporary="${marker}.tmp.$$"
cleanup() { rm -f -- "$temporary" "$marker"; }
trap cleanup EXIT
trap ':' TERM INT HUP
if ! command -v setsid >/dev/null 2>&1; then
    exit 125
fi
SKILLEVALUATOR_EXEC_TOKEN=$token
export SKILLEVALUATOR_EXEC_TOKEN
setsid sh -c "$group_script" skillevaluator-managed-group "$@" &
managed_pid=$!
printf '%s\n' "$managed_pid" > "$temporary"
chmod 600 "$temporary"
mv -f -- "$temporary" "$marker"
if wait "$managed_pid"; then status=0; else status=$?; fi
exit "$status"
""".strip()

_REMOTE_TERMINATE_SCRIPT = r"""
marker=$1
token=$2
attempt=0
while [[ ! -s "$marker" && $attempt -lt 100 ]]; do
    sleep 0.05
    attempt=$((attempt + 1))
done
if [[ ! -s "$marker" ]]; then
    exit 3
fi
IFS= read -r managed_pid < "$marker"
if [[ ! "$managed_pid" =~ ^[0-9]+$ ]] || ((managed_pid <= 1)); then
    exit 4
fi
verify_managed_group() {
    [[ -r "/proc/$managed_pid/environ" ]] || return 1
    tr '\000' '\n' < "/proc/$managed_pid/environ" \
        | grep -Fqx -- "SKILLEVALUATOR_EXEC_TOKEN=$token" || return 1
    local stat_tail
    stat_tail=$(sed 's/^.*) //' "/proc/$managed_pid/stat") || return 1
    local fields
    read -ra fields <<< "$stat_tail"
    [[ "${fields[2]:-}" == "$managed_pid" ]]
}
if ! verify_managed_group; then
    if ! kill -0 "$managed_pid" 2>/dev/null; then
        rm -f -- "$marker" "${marker}.tmp."*
        exit 0
    fi
    exit 4
fi
kill -- "-$managed_pid" 2>/dev/null || true
attempt=0
while kill -0 "$managed_pid" 2>/dev/null && ((attempt < 20)); do
    sleep 0.05
    attempt=$((attempt + 1))
done
if kill -0 "$managed_pid" 2>/dev/null; then
    verify_managed_group || exit 4
    kill -KILL -- "-$managed_pid" 2>/dev/null || true
fi
attempt=0
while kill -0 "$managed_pid" 2>/dev/null && ((attempt < 20)); do
    sleep 0.05
    attempt=$((attempt + 1))
done
rm -f -- "$marker" "${marker}.tmp."*
if kill -0 "$managed_pid" 2>/dev/null; then
    exit 5
fi
""".strip()


@dataclass(frozen=True)
class _RemoteProcessControl:
    marker_path: str
    token: str


def _new_remote_process_control() -> _RemoteProcessControl:
    return _RemoteProcessControl(
        marker_path=f"/tmp/.skillevaluator-exec-{uuid.uuid4().hex}",
        token=secrets.token_urlsafe(32),
    )


def _managed_exec_arguments(command: list[str], control: _RemoteProcessControl) -> list[str]:
    """Run one remote command under an identifiable, isolated process group."""
    return [
        "sh",
        "-c",
        _REMOTE_CONTROLLER_SCRIPT,
        "skillevaluator-managed-exec",
        control.marker_path,
        control.token,
        _REMOTE_GROUP_SCRIPT,
        *command,
    ]


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


def _redact(text: str | None, secret_values: set[str]) -> str | None:
    if text is None:
        return None
    redacted = text
    for value in sorted((value for value in secret_values if value), key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _redact_result(result: ExecResult, secret_values: set[str]) -> ExecResult:
    return ExecResult(
        stdout=_redact(result.stdout, secret_values),
        stderr=_redact(result.stderr, secret_values),
        return_code=result.return_code,
    )


def _signal_process_tree(process: asyncio.subprocess.Process, value: signal.Signals) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, value)
    elif value == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


async def _terminate_process_tree(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]],
    *,
    preserve_cancellation: bool,
) -> None:
    async def reap() -> None:
        _signal_process_tree(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(communication), timeout=_COMPOSE_TERMINATE_SECONDS)
            return
        except TimeoutError:
            pass

        _signal_process_tree(process, signal.SIGKILL)
        try:
            await asyncio.wait_for(asyncio.shield(communication), timeout=_COMPOSE_KILL_SECONDS)
        except TimeoutError:
            communication.cancel()
            done, _pending = await asyncio.wait({communication}, timeout=_COMPOSE_CANCEL_SECONDS)
            if communication in done:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    communication.result()

    cleanup = asyncio.create_task(reap())
    await _await_task_uninterruptibly(cleanup, preserve_cancellation=preserve_cancellation)


def _file_backed_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Resolve the private NVIDIA Build sentinel without putting its value in argv."""
    resolved = _validate_environment(environment)
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

    def _remote_process_control(self) -> _RemoteProcessControl | None:
        """Return scoped Linux process control; Windows keeps Harbor's native path."""
        if getattr(self, "_is_windows_container", False):
            return None
        return _new_remote_process_control()

    async def _terminate_remote_process(self, control: _RemoteProcessControl) -> None:
        result = await self._run_docker_compose_command(
            [
                "exec",
                "-u",
                "root",
                "main",
                "bash",
                "-c",
                _REMOTE_TERMINATE_SCRIPT,
                "skillevaluator-terminate-exec",
                control.marker_path,
                control.token,
            ],
            check=False,
            timeout_sec=_REMOTE_CLEANUP_TIMEOUT_SECONDS,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"could not confirm scoped Docker command termination (cleanup status {result.return_code})"
            )

    async def _stop_compose_and_remote_process(
        self,
        process: asyncio.subprocess.Process,
        communication: asyncio.Task[tuple[bytes, bytes]],
        control: _RemoteProcessControl | None,
    ) -> None:
        remote_error: BaseException | None = None
        if control is not None:
            try:
                await self._terminate_remote_process(control)
            except BaseException as exc:
                remote_error = exc
        await _terminate_process_tree(process, communication, preserve_cancellation=False)
        if remote_error is not None:
            raise RuntimeError("could not confirm termination of the remote Docker command") from remote_error

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
        remote_control = self._remote_process_control()
        shell_arguments = self._platform.exec_shell_args(command)
        exec_command.extend(
            _managed_exec_arguments(shell_arguments, remote_control) if remote_control is not None else shell_arguments
        )

        return await self._run_docker_compose_command(
            exec_command,
            check=False,
            timeout_sec=timeout_sec,
            env_overrides=subprocess_environment,
            remote_control=remote_control,
        )

    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        env_overrides: Mapping[str, str] | None = None,
        remote_control: _RemoteProcessControl | None = None,
    ) -> ExecResult:
        """Run compose with sensitive exec overrides only in the child env."""
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

        process_environment = self._compose_env_vars(include_os_env=True)
        process_environment.update(env_overrides or {})
        secret_values = {value for value in (env_overrides or {}).values() if value}
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *full_command,
                env=process_environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        )
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError as cancellation:
            process = await _await_task_uninterruptibly(creation, preserve_cancellation=False)
            communication = asyncio.create_task(process.communicate())
            cleanup = asyncio.create_task(self._stop_compose_and_remote_process(process, communication, remote_control))
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except RuntimeError as exc:
                cancellation.add_note(f"Remote Docker command termination could not be confirmed: {exc}")
            raise

        communication = asyncio.create_task(process.communicate())
        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=timeout_sec,
                )
            else:
                stdout_bytes, stderr_bytes = await asyncio.shield(communication)
        except TimeoutError:
            cleanup = asyncio.create_task(self._stop_compose_and_remote_process(process, communication, remote_control))
            try:
                await _await_task_uninterruptibly(cleanup)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Command timed out after {timeout_sec} seconds; remote termination could not be confirmed"
                ) from exc
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
        except asyncio.CancelledError as cancellation:
            cleanup = asyncio.create_task(self._stop_compose_and_remote_process(process, communication, remote_control))
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except RuntimeError as exc:
                cancellation.add_note(f"Remote Docker command termination could not be confirmed: {exc}")
            raise

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        result = _redact_result(
            ExecResult(stdout=stdout, stderr=stderr, return_code=process.returncode or 0),
            secret_values,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"Docker compose command failed for environment {self.environment_name}. "
                f"Command: {' '.join(full_command)}. Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. Stderr: {result.stderr}."
            )
        return result


class SkillEvaluatorSecureDockerEnvironment(SkillEvaluatorDockerEnvironment):
    """Transfer exec environments through short-lived container-only files."""

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
        remote_control = self._remote_process_control()
        shell_arguments = self._platform.exec_shell_args(command)
        exec_command.extend(
            _managed_exec_arguments(shell_arguments, remote_control) if remote_control is not None else shell_arguments
        )
        result = await self._run_docker_compose_command(
            exec_command,
            check=False,
            timeout_sec=timeout_sec,
            remote_control=remote_control,
        )
        return _redact_result(result, secret_values or set())

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

        merged = _file_backed_environment(merged)
        remote_path = f"/tmp/.skillevaluator-exec-env-{uuid.uuid4().hex}.sh"
        primary_error: BaseException | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="skillevaluator-docker-env-") as temp_dir:
                host_path = Path(temp_dir) / "environment.sh"
                host_path.write_text(_render_environment_script(merged), encoding="utf-8")
                host_path.chmod(0o600)
                await self.upload_file(host_path, remote_path)

            if user is None:
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chmod", "600", remote_path],
                    check=True,
                )
            else:
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chown", "--", str(user), remote_path],
                    check=True,
                )
                await self._run_docker_compose_command(
                    ["exec", "-u", "root", "main", "chmod", "600", remote_path],
                    check=True,
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
                secret_values={value for value in merged.values() if value},
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
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
