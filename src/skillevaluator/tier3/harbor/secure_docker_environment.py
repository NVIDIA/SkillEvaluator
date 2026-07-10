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
import os
import re
import shlex
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name

SECURE_DOCKER_ENV_IMPORT_PATH = (
    "skillevaluator.tier3.harbor.secure_docker_environment:SkillEvaluatorSecureDockerEnvironment"
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NVIDIA_BUILD_FILE_SENTINEL = "skillevaluator-file-backed-nvidia-key"
_NVIDIA_BUILD_KEY_FILE_ENV = "SKILLEVALUATOR_NVIDIA_API_KEY_FILE"


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
            env_overrides=subprocess_environment,
        )

    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        env_overrides: Mapping[str, str] | None = None,
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
        process = await asyncio.create_subprocess_exec(
            *full_command,
            env=process_environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.communicate(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.communicate()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        result = ExecResult(stdout=stdout, stderr=stderr, return_code=process.returncode or 0)
        if check and result.return_code != 0:
            safe_stdout = _redact(result.stdout, secret_values)
            safe_stderr = _redact(result.stderr, secret_values)
            raise RuntimeError(
                f"Docker compose command failed for environment {self.environment_name}. "
                f"Command: {' '.join(full_command)}. Return code: {result.return_code}. "
                f"Stdout: {safe_stdout}. Stderr: {safe_stderr}."
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
    ) -> ExecResult:
        exec_command = ["exec"]
        effective_cwd = cwd or self.task_env_config.workdir
        if effective_cwd:
            exec_command.extend(["-w", effective_cwd])
        if user is not None:
            exec_command.extend(["-u", str(user)])
        exec_command.append("main")
        exec_command.extend(self._platform.exec_shell_args(command))
        return await self._run_docker_compose_command(exec_command, check=False, timeout_sec=timeout_sec)

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
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self._remove_handoff(remote_path)
            except Exception as cleanup_error:
                message = f"could not confirm removal of Docker environment handoff {remote_path}"
                if primary_error is not None:
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(f"{message}: {cleanup_error}")
                else:
                    raise RuntimeError(message) from cleanup_error
