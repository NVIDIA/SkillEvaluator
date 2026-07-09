# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker environment that keeps per-exec credentials out of process argv.

Harbor 0.13.2 serializes ``exec(env=...)`` values as ``docker compose exec
-e NAME=value``. Process arguments are host-visible, so SkillEvaluator routes
all values through the compose subprocess environment and passes only
``-e NAME`` on argv. This class is intentionally tied to the pinned Harbor
version and can be removed after the behavior is fixed upstream.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name

SECURE_DOCKER_ENV_IMPORT_PATH = (
    "skillevaluator.tier3.harbor.secure_docker_environment:SkillEvaluatorDockerEnvironment"
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _secure_exec_arguments(
    environment: Mapping[str, str] | None,
) -> tuple[list[str], dict[str, str]]:
    """Put env names on argv and every value in the child process env."""
    arguments: list[str] = []
    subprocess_environment: dict[str, str] = {}
    for name, value in (environment or {}).items():
        if not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        if not isinstance(value, str):
            raise ValueError(f"Environment variable {name!r} must have a string value")
        if "\x00" in value:
            raise ValueError(f"Environment variable {name!r} contains a NUL byte")
        arguments.extend(["-e", name])
        subprocess_environment[name] = value
    return arguments, subprocess_environment


def _redact(text: str | None, secret_values: set[str]) -> str | None:
    if text is None:
        return None
    redacted = text
    for value in sorted((value for value in secret_values if value), key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted


class SkillEvaluatorDockerEnvironment(DockerEnvironment):
    """Pinned Harbor Docker backend with host-visible argv credential safety."""

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
