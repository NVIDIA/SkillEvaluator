# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted host execution backend for Harbor local mode."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

from harbor.environments.base import BaseEnvironment, ExecResult, OutputCallback, OutputStream
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType

from skillevaluator.tier3.harbor import local_sandbox
from skillevaluator.tier3.harbor.local_runtime import (
    LOCAL_RUNTIME_AGENTS,
    default_runtime_root,
    local_subprocess_env,
    runtime_bin_dirs,
    runtime_command_roots,
    validate_runtime_root,
)
from skillevaluator.tier3.harbor.progress import secret_values_from_environment
from skillevaluator.tier3.harbor.secure_copy import copytree_secure
from skillevaluator.tier3.harbor.stream_redaction import CommandOutputByteBudget, StreamingLogRedactor
from skillevaluator.tier3.output_provenance import output_provenance_key_path

_SAFE_HOST_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "SKILLEVALUATOR_RUNTIME_DIR",
    }
)
_BLOCKED_COMMAND_ENV_NAMES = frozenset(
    {
        "BASHOPTS",
        "BASH_ENV",
        "CDPATH",
        "CLASSPATH",
        "ENV",
        "GCONV_PATH",
        "IFS",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "LOCPATH",
        "NLSPATH",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PERL5LIB",
        "PERL5OPT",
        "RUBYLIB",
        "RUBYOPT",
        "SHELLOPTS",
        "ZDOTDIR",
        "_JAVA_OPTIONS",
    }
)
_BLOCKED_COMMAND_ENV_PREFIXES = ("DYLD_", "LD_", "PYTHON")
_INNER_ENV_BOOTSTRAP = """
import json
import os
import sys

environment = json.load(sys.stdin)
command = sys.argv[1:]
os.execvpe(command[0], command, environment)
"""
# Evaluator/provider credentials are not seeded into every skill child by
# default. Agents and verifiers receive credentials per-exec (Harbor agent env
# / task env blocks); this ambient fallback is opt-in only. Covers the public
# provider env vars (NVIDIA Build / OpenAI / Anthropic).
_LIVE_AGENT_KEYS = frozenset(
    {
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    }
)
_SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|KEY|PAT|TOKEN|SECRET|PASS(?:WORD)?|"
    r"CREDENTIALS?|AUTH(?:ORIZATION)?|BEARER|COOKIE|SESSION|CERT(?:IFICATE)?|DSN|"
    r"CONNECTION(?:_STRING)?|(?:PRE)?SIGNED_?URL|SAS_?URL|CREDENTIAL_?URL|DATABASE_?URL)(?:_|$)",
    re.IGNORECASE,
)
_LEGACY_SECRET_ENV_NAME_RE = re.compile(
    r"(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH)",
    re.IGNORECASE,
)
# Compact credential names remain high-signal even without underscore token
# boundaries. Deliberately omit a bare KEY suffix so ordinary names such as
# MONKEY and KEYBOARD do not cause short public values to be rewritten.
_COMPACT_SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:^|_)[A-Z0-9]*(?:API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|TOKEN|SECRET|PASSWORD|"
    r"CREDENTIALS?|AUTHENTICATION|AUTHORIZATION|BEARER)(?:_|$)",
    re.IGNORECASE,
)
# Before Harbor 0.22, local output used the broad legacy name matcher above.
# Retain protection for credential-sized values while avoiding corruption from
# common short variables such as MONKEY=banana and KEYBOARD=clacky.
_MIN_LEGACY_SECRET_VALUE_LENGTH = 8
_SHELL_WRITE_REDIRECT_RE = re.compile(r"(?:^|\s)(?:\d?>{1,2}|&>)\s*([^\s;&|]+)")
_SHELL_WRITE_COMMAND_RE = re.compile(r"(?:^|[;&|]\s*)(?:tee|touch|mkdir|cp|mv)\b(?P<args>[^;&|]*)")
_BACKGROUND_AMPERSAND_RE = re.compile(r"(?<![&>])&(?![&>])")
_DETACHED_PROCESS_COMMANDS = frozenset({"setsid", "nohup", "daemon", "disown"})
_SHELL_COMMANDS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_COMMAND_PREFIXES = frozenset({"command", "do", "elif", "env", "exec", "if", "then", "until", "while"})
_REAP_TERM_SECONDS = 1.0
_REAP_KILL_SECONDS = 1.0
_REAP_CANCEL_SECONDS = 0.1
_CREATION_CANCEL_SECONDS = 0.1
_STOP_CLEANUP_SECONDS = _REAP_TERM_SECONDS + _REAP_KILL_SECONDS + _REAP_CANCEL_SECONDS + 0.2
_PATH_START_BOUNDARY_RE = r"(?<![A-Za-z0-9_.-])"
_PATH_BOUNDARY_RE = r"(?=$|[\s'\";&|<>])"
_HOST_HOME_PREFIX_RE = r"(?:~|\$HOME|\$\{HOME\}|/Users/[^\s/;'\"&|<>]+|/home/[^\s/;'\"&|<>]+|/root)"
_SENSITIVE_HOME_DIR_RE = re.compile(
    rf"{_PATH_START_BOUNDARY_RE}(?P<path>{_HOST_HOME_PREFIX_RE}/(?:\.ssh|\.aws|\.gnupg|\.kube|\.azure|\.oci)"
    rf"(?:/[^\s'\";&|<>]+)*){_PATH_BOUNDARY_RE}"
)
_SENSITIVE_HOME_FILE_RE = re.compile(
    rf"{_PATH_START_BOUNDARY_RE}(?P<path>{_HOST_HOME_PREFIX_RE}/(?:\.netrc|\.git-credentials|\.pypirc|\.npmrc|Work/\.env))"
    rf"{_PATH_BOUNDARY_RE}"
)
_SENSITIVE_HOME_SUBPATH_RE = re.compile(
    rf"{_PATH_START_BOUNDARY_RE}(?P<path>{_HOST_HOME_PREFIX_RE}/(?:\.config/gcloud|\.docker/config\.json|\.docker/run/docker\.sock|\.huggingface/token)"
    rf"(?:/[^\s'\";&|<>]+)*){_PATH_BOUNDARY_RE}"
)
_SENSITIVE_ABSOLUTE_PATH_RE = re.compile(
    rf"{_PATH_START_BOUNDARY_RE}(?P<path>/(?:etc/(?:shadow|sudoers)|var/run/docker\.sock)){_PATH_BOUNDARY_RE}"
)
_SENSITIVE_HOST_PATH_RES = (
    _SENSITIVE_HOME_DIR_RE,
    _SENSITIVE_HOME_FILE_RE,
    _SENSITIVE_HOME_SUBPATH_RE,
    _SENSITIVE_ABSOLUTE_PATH_RE,
)


def _credential_uri_environment_values(environment: dict[str, str]) -> set[str]:
    """Extract only credential URI/proxy values beyond local's name policy."""
    candidates = {
        name: value
        for name, value in environment.items()
        if value and (name.upper().endswith("_PROXY") or "://" in value)
    }
    return secret_values_from_environment(candidates)


class _StreamCallbackOutput:
    """Own per-stream redaction state until the exec outcome is known."""

    def __init__(
        self,
        callback: OutputCallback,
        callback_error: asyncio.Future[BaseException],
        secret_values: set[str],
    ) -> None:
        self._callback = callback
        self._callback_error = callback_error
        self._redactors = {
            "stdout": StreamingLogRedactor(secret_values),
            "stderr": StreamingLogRedactor(secret_values),
        }
        self._raw_output: dict[OutputStream, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        self._delivery_cancelled = False
        self._active_deliveries: set[asyncio.Task[None]] = set()

    def append_raw(self, chunk: bytes, stream: OutputStream) -> None:
        self._raw_output[stream].extend(chunk)

    def raw_output(self, stream: OutputStream) -> bytes:
        return bytes(self._raw_output[stream])

    def abandon_delivery(self) -> None:
        """Prevent timeout cleanup from re-entering a stuck callback."""
        self._delivery_cancelled = True

    async def _emit(self, text: str, stream: OutputStream) -> None:
        if not text or self._callback_error.done() or self._delivery_cancelled:
            return

        async def invoke_callback() -> None:
            try:
                await self._callback(text, stream)
            except asyncio.CancelledError as exc:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    # Cleanup is cancelling this delivery task. A deliberate
                    # CancelledError raised by callback code has a zero
                    # cancellation count and remains the callback failure.
                    raise
                if not self._callback_error.done():
                    self._callback_error.set_result(exc)
                await asyncio.sleep(0)
            except BaseException as exc:
                if not self._callback_error.done():
                    self._callback_error.set_result(exc)
                await asyncio.sleep(0)

        delivery = asyncio.create_task(invoke_callback())
        self._active_deliveries.add(delivery)

        def retire_delivery(completed: asyncio.Task[None]) -> None:
            self._active_deliveries.discard(completed)
            with contextlib.suppress(BaseException):
                completed.result()

        delivery.add_done_callback(retire_delivery)
        try:
            await asyncio.shield(delivery)
        except asyncio.CancelledError:
            # Cancellation of the collector/finalizer must explicitly target
            # the shielded callback task. Repeating cancellation handles a
            # callback that performs async cleanup after its first cancel.
            self._delivery_cancelled = True
            cancellation = asyncio.create_task(_cancel_task_repeatedly(delivery, timeout=_REAP_CANCEL_SECONDS))
            await _await_task_uninterruptibly(cancellation, preserve_cancellation=False)
            raise

    async def feed(self, text: str, stream: OutputStream) -> None:
        await self._emit(self._redactors[stream].feed(text), stream)

    async def finish(self, *, stderr_suffix: str = "") -> None:
        await self._emit(self._redactors["stdout"].finish(), "stdout")
        if stderr_suffix:
            await self._emit(self._redactors["stderr"].feed(stderr_suffix), "stderr")
        await self._emit(self._redactors["stderr"].finish(), "stderr")


async def _await_task_uninterruptibly(
    task: asyncio.Task[Any],
    *,
    preserve_cancellation: bool = True,
) -> Any:
    """Await a cleanup task to completion despite repeated cancellation."""
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


async def _cancel_task_repeatedly(task: asyncio.Future[Any], *, timeout: float) -> bool:
    """Bound cleanup even when a coroutine suppresses its first cancellation."""
    deadline = asyncio.get_running_loop().time() + timeout
    retry_interval = min(0.01, timeout / 4) if timeout > 0 else 0
    while not task.done():
        task.cancel()
        # Give the cancellation target a chance to catch the injected error
        # before deciding whether another cancellation is necessary.
        await asyncio.sleep(0)
        if task.done():
            break
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        await asyncio.wait({task}, timeout=min(retry_interval, remaining))
    if task.done():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()
    return task.done()


def _looks_like_path_token(token: str) -> bool:
    return token.startswith(("/", "~/", "$"))


def _unquoted_shell_text(command: str) -> str:
    output: list[str] = []
    quote = ""
    escaped = False
    for char in command:
        if escaped:
            output.append(" ")
            escaped = False
            continue
        if char == "\\" and quote != "'":
            output.append(" ")
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            output.append(" ")
            continue
        if char in {"'", '"'}:
            quote = char
            output.append(" ")
            continue
        output.append(char)
    return "".join(output)


def _contains_detached_process_launcher(command: str, *, _depth: int = 0) -> bool:
    """Detect common direct/nested shell launchers without treating arguments as commands.

    This is advisory defense in depth, not a process-isolation boundary: a
    script or native program can still call ``setsid(2)`` without spelling a
    launcher in Harbor's command string.
    """
    if _depth > 3:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False

    command_position = True
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token and set(token) <= set(";&|()"):
            command_position = True
            index += 1
            continue
        if not command_position:
            index += 1
            continue

        name = Path(token).name
        if name == "env":
            index += 1
            while index < len(tokens) and (
                tokens[index].startswith("-")
                or ("=" in tokens[index] and not tokens[index].startswith(("/", "./", "../")))
            ):
                index += 1
            continue
        if name == "nice":
            index += 1
            if index < len(tokens) and tokens[index] in {"-n", "--adjustment"}:
                index += 2
            elif index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            continue
        if name in _COMMAND_PREFIXES or ("=" in token and not token.startswith(("/", "./", "../"))):
            index += 1
            continue
        if name in _DETACHED_PROCESS_COMMANDS:
            return True
        if name in _SHELL_COMMANDS:
            segment_end = next(
                (
                    offset
                    for offset in range(index + 1, len(tokens))
                    if tokens[offset] and set(tokens[offset]) <= set(";&|()")
                ),
                len(tokens),
            )
            for offset in range(index + 1, segment_end - 1):
                option = tokens[offset]
                is_command_option = option == "-c" or (
                    option.startswith("-") and not option.startswith("--") and "c" in option[1:]
                )
                if is_command_option and _contains_detached_process_launcher(tokens[offset + 1], _depth=_depth + 1):
                    return True
        command_position = False
        index += 1
    return False


def _contains_background_command(command: str, *, _depth: int = 0) -> bool:
    """Detect background operators in direct and nested shell command strings."""
    if _BACKGROUND_AMPERSAND_RE.search(_unquoted_shell_text(command)):
        return True
    if _depth > 3:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False

    command_position = True
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token and set(token) <= set(";&|()"):
            command_position = True
            index += 1
            continue
        if not command_position:
            index += 1
            continue
        name = Path(token).name
        if name == "env":
            index += 1
            while index < len(tokens) and (
                tokens[index].startswith("-")
                or ("=" in tokens[index] and not tokens[index].startswith(("/", "./", "../")))
            ):
                index += 1
            continue
        if name == "nice":
            index += 1
            if index < len(tokens) and tokens[index] in {"-n", "--adjustment"}:
                index += 2
            elif index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            continue
        if name in _COMMAND_PREFIXES or ("=" in token and not token.startswith(("/", "./", "../"))):
            index += 1
            continue
        if name in _SHELL_COMMANDS:
            segment_end = next(
                (
                    offset
                    for offset in range(index + 1, len(tokens))
                    if tokens[offset] and set(tokens[offset]) <= set(";&|()")
                ),
                len(tokens),
            )
            for offset in range(index + 1, segment_end - 1):
                option = tokens[offset]
                is_command_option = option == "-c" or (
                    option.startswith("-") and not option.startswith("--") and "c" in option[1:]
                )
                if is_command_option and _contains_background_command(tokens[offset + 1], _depth=_depth + 1):
                    return True
        command_position = False
        index += 1
    return False


class SkillEvaluatorLocalEnvironment(BaseEnvironment):
    """Run Harbor tasks on the host, confined by an OS-level sandbox.

    Commands execute under bubblewrap on Linux with writes confined to the run
    directory. Network egress defaults on for live agent calls and can be
    disabled with ``allow_net``. macOS Seatbelt is semi-trusted: strict reads
    are opt-in via ``SKILLEVALUATOR_LOCAL_STRICT_READS=1``. Common detached
    shell launch patterns are rejected as defense in depth, but Seatbelt has
    no PID namespace and cannot guarantee cleanup of script/native detachment.
    Docker remains the supported macOS backend for arbitrary untrusted code.
    """

    def __init__(
        self,
        *args,
        runtime_agent: str,
        runtime_root: str | None = None,
        working_dir: str | None = None,
        sandbox_mode: str | None = None,
        allow_net: str | bool | None = None,
        inherit_agent_keys: str | bool | None = None,
        strict_reads: str | bool | None = None,
        **kwargs,
    ):
        if runtime_agent not in LOCAL_RUNTIME_AGENTS:
            supported = ", ".join(LOCAL_RUNTIME_AGENTS)
            raise ValueError(f"runtime_agent must be one of: {supported}")
        self._runtime_root = validate_runtime_root(runtime_root or default_runtime_root())
        self._runtime_agent = runtime_agent
        self._working_dir_override = Path(working_dir).expanduser() if working_dir else None
        self._sandbox_mode = local_sandbox.resolve_mode(sandbox_mode)
        # Egress defaults ON: local mode exists to run a live agent, and the
        # agent CLI's model call needs the network. Writes and reads stay
        # confined, so open egress is the accepted semi-trusted boundary;
        # set SKILLEVALUATOR_LOCAL_ALLOW_NET=0 (or allow_net=false) to airgap a
        # skill that must not reach the network.
        self._allow_net = local_sandbox.coerce_flag(allow_net, env_var=local_sandbox.ALLOW_NET_ENV, default=True)
        self._inherit_agent_keys = local_sandbox.coerce_flag(
            inherit_agent_keys, env_var=local_sandbox.INHERIT_AGENT_KEYS_ENV
        )
        self._strict_reads = local_sandbox.coerce_flag(strict_reads, env_var=local_sandbox.STRICT_READS_ENV)
        self._sandbox: local_sandbox.Sandbox | None = None
        self._active_processes: dict[
            asyncio.subprocess.Process,
            asyncio.Task[tuple[bytes, bytes]] | None,
        ] = {}
        self._active_process_secret_values: dict[asyncio.subprocess.Process, set[str]] = {}
        self._pending_creations: set[asyncio.Task[asyncio.subprocess.Process]] = set()
        self._creation_secret_values: dict[asyncio.Task[asyncio.subprocess.Process], set[str]] = {}
        self._creation_cleanups: dict[
            asyncio.Task[asyncio.subprocess.Process],
            asyncio.Task[None],
        ] = {}
        self._creation_cleanup_errors: list[set[str]] = []
        self._stop_requested = False
        super().__init__(*args, **kwargs)
        base_dir = self._working_dir_override or (self.trial_paths.trial_dir / "local-environment")
        self._root = base_dir.resolve()
        self._workspace = self._root / "workspace"
        self._tests = self._root / "tests"
        self._solution = self._root / "solution"
        self._installed_agent = self._root / "installed-agent"
        self._tmp = self._root / "tmp"
        self._home = self._root / "home"
        self.default_user = None

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    @classmethod
    def preflight(cls) -> None:
        return None

    def _validate_definition(self) -> None:
        if not self.environment_dir.exists():
            raise FileNotFoundError(f"Environment directory does not exist: {self.environment_dir}")
        for name in ("docker-compose.yaml", "docker-compose.yml"):
            if (self.environment_dir / name).exists():
                raise ValueError("Docker Compose sidecars are unsupported in Harbor local mode.")

    async def start(self, force_build: bool = False) -> None:
        _ = force_build
        if (
            self._pending_creations
            or self._creation_secret_values
            or self._creation_cleanups
            or self._creation_cleanup_errors
            or self._active_processes
            or self._active_process_secret_values
        ):
            raise RuntimeError("Cannot start local environment while process cleanup is still pending")
        self._stop_requested = False
        self.trial_paths.mkdir()
        for path in (
            self._root,
            self._workspace,
            self._workspace / "skills",
            self._workspace / "input",
            self._tests,
            self.trial_paths.agent_dir,
            self.trial_paths.verifier_dir,
            self.trial_paths.artifacts_dir,
            self._solution,
            self._installed_agent,
            self._tmp,
            self._home,
            self._home / ".local" / "bin",
            self._home / ".claude" / "skills",
            self._home / ".agents" / "skills",
            self._home / ".config" / "opencode" / "skills",
            self._home / ".codex",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._copy_environment_bundle()
        self._sandbox = local_sandbox.detect(self._sandbox_mode)
        plan = self._sandbox.plan
        self.logger.info("local mode isolation: %s (%s)", plan.strength, plan.reason)
        if plan.backend == "none":
            self.logger.warning("local mode is NOT kernel-sandboxed; advisory guardrails only: %s", plan.reason)

    async def stop(self, delete: bool) -> None:
        self._stop_requested = True
        pending_creations = tuple(self._pending_creations)
        for creation in pending_creations:
            self._schedule_creation_cleanup(creation)
        if pending_creations:
            await asyncio.wait(
                pending_creations,
                timeout=_CREATION_CANCEL_SECONDS,
            )

        async def wait_for_resolved_creation_cleanups() -> None:
            cleanup_tasks = tuple(
                cleanup
                for creation, cleanup in self._creation_cleanups.items()
                if creation.done() and not cleanup.done()
            )
            if cleanup_tasks:
                await asyncio.wait(cleanup_tasks, timeout=_STOP_CLEANUP_SECONDS)
            # Let cleanup callbacks retire their creation/mapping entries
            # before the final containment check.
            await asyncio.sleep(0)

        await wait_for_resolved_creation_cleanups()

        for proc, communication in tuple(self._active_processes.items()):
            try:
                await self._terminate_process_tree(proc, communication)
            except BaseException:
                # Keep the handle and protected values for a redacted retry.
                continue
            self._release_active_process(proc)

        # A pending creation can resolve while known active processes are
        # being reaped. Give its already-tracked cleanup the same bounded
        # opportunity before deciding whether deletion is safe.
        await wait_for_resolved_creation_cleanups()

        if (
            not self._pending_creations
            and not self._creation_secret_values
            and not self._creation_cleanups
            and not self._active_processes
        ):
            # Every process whose earlier cleanup failed has now been reaped
            # through its retained active-process handle.
            self._creation_cleanup_errors.clear()

        unresolved_creations = tuple(creation for creation in self._pending_creations if not creation.done())
        outstanding_cleanups = tuple(cleanup for cleanup in self._creation_cleanups.values() if not cleanup.done())
        if (
            unresolved_creations
            or self._creation_secret_values
            or outstanding_cleanups
            or self._creation_cleanup_errors
            or self._active_processes
        ):
            details: list[str] = []
            if unresolved_creations:
                details.append("process creation containment remains unresolved")
            if outstanding_cleanups:
                details.append("late process cleanup remains unresolved")
            if self._creation_cleanup_errors:
                details.append("creation cleanup failed")
            if self._active_processes:
                details.append("active process cleanup remains unresolved")
            diagnostic = "Local environment stop could not confirm process creation containment before the deadline" + (
                f" ({'; '.join(details)})" if details else ""
            )
            cleanup_secret_values = set().union(
                *self._creation_cleanup_errors,
                *self._creation_secret_values.values(),
                *self._active_process_secret_values.values(),
            )
            raise RuntimeError(self._redact_output(diagnostic, cleanup_secret_values))

        if delete and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)

    def _schedule_creation_cleanup(
        self,
        creation: asyncio.Task[asyncio.subprocess.Process],
        *,
        secret_values: set[str] | None = None,
    ) -> asyncio.Task[None]:
        if secret_values:
            self._creation_secret_values.setdefault(creation, set()).update(secret_values)
        existing = self._creation_cleanups.get(creation)
        if existing is not None:
            return existing

        async def reap_created_process() -> None:
            try:
                process = await asyncio.shield(creation)
            except BaseException:
                return
            communication = asyncio.create_task(process.communicate())
            self._active_processes[process] = communication
            self._active_process_secret_values[process] = set(self._creation_secret_values.get(creation, set()))
            try:
                await self._terminate_process_tree(process, communication)
            except BaseException:
                # Retain ownership so stop() can retry containment.
                raise
            self._release_active_process(process)

        cleanup = asyncio.create_task(reap_created_process())
        self._creation_cleanups[creation] = cleanup

        def finish_cleanup(completed: asyncio.Task[None]) -> None:
            self._pending_creations.discard(creation)
            self._creation_cleanups.pop(creation, None)
            protected_values = self._creation_secret_values.pop(creation, set())
            try:
                completed.result()
            except BaseException:
                self._creation_cleanup_errors.append(protected_values)
                diagnostic = self._redact_output(
                    "Local process creation cleanup failed",
                    protected_values,
                )
                with contextlib.suppress(RuntimeError):
                    loop = asyncio.get_running_loop()
                    loop.call_exception_handler(
                        {
                            "message": diagnostic,
                            "exception": RuntimeError(diagnostic),
                        }
                    )

        cleanup.add_done_callback(finish_cleanup)
        return cleanup

    def _release_active_process(self, process: asyncio.subprocess.Process) -> None:
        self._active_processes.pop(process, None)
        self._active_process_secret_values.pop(process, None)

    async def prepare_logs_for_host(self) -> None:
        return None

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        target = self._resolve_path(target_path)
        if not self._path_is_within_allowed_local_roots(target):
            raise ValueError(f"local mode upload target is outside the run directory: {target_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self._rewrite_uploaded_script(target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        target = self._resolve_path(target_dir)
        if not self._path_is_within_allowed_local_roots(target):
            raise ValueError(f"local mode upload target is outside the run directory: {target_dir}")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        if source.exists():
            copytree_secure(source, target, dirs_exist_ok=True)
        self._rewrite_uploaded_scripts(target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self._resolve_path(source_path)
        if not self._path_is_within_allowed_local_roots(source):
            raise ValueError(f"local mode download source is outside the run directory: {source_path}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = self._resolve_path(source_dir)
        if not self._path_is_within_allowed_local_roots(source):
            raise ValueError(f"local mode download source is outside the run directory: {source_dir}")
        target = Path(target_dir)
        if target.exists():
            shutil.rmtree(target)
        copytree_secure(source, target, dirs_exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = user
        output_callback = self._output_callback()
        output_secret_values = self._output_secret_values(env, {})

        async def blocked(reason: str) -> ExecResult:
            diagnostic = self._redact_output(f"Local mode command blocked: {reason}", output_secret_values)
            self.logger.warning("%s", diagnostic)
            if output_callback is not None:
                await output_callback(diagnostic, "stderr")
            return ExecResult(stdout="", stderr=diagnostic, return_code=126)

        if self._stop_requested:
            return await blocked("environment shutdown is in progress")

        rewritten = self._rewrite_command(command)
        try:
            workdir = self._resolve_path(cwd) if cwd else self._workspace
        except ValueError as exc:
            return await blocked(str(exc))
        if not self._path_is_within_allowed_local_roots(workdir):
            return await blocked(f"cwd {workdir} is outside the local run directory.")
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            exec_env = self._exec_env(env)
        except ValueError as exc:
            return await blocked(str(exc))
        output_secret_values.update(self._output_secret_values(env, exec_env))
        guardrail_reason = self._local_command_guardrail_reason(command, rewritten, exec_env)
        if guardrail_reason:
            return await blocked(guardrail_reason)

        sandbox = self._sandbox
        if sandbox is None:
            sandbox = self._sandbox = local_sandbox.detect(self._sandbox_mode)
        env_payload = json.dumps(exec_env).encode("utf-8")
        # Resource limits are part of confinement, so honor the trusted escape
        # hatch: sandbox_mode=off means "unconstrained host run" and must not
        # impose the CPU/NOFILE caps (they would SIGXCPU a long trusted run).
        apply_limits = os.name == "posix" and self._sandbox_mode != "off"
        # macOS Seatbelt defaults to a HOME-denylist for compatibility. The
        # strict profile is available for callers that need deny-all reads and
        # receives only the selected runtime/system exceptions.
        #
        # The strict profile includes visible runtime aliases and can be
        # selected explicitly.
        bootstrap_interpreter = Path(sys.executable)
        if sandbox.plan.backend == "seatbelt":
            # sandbox-exec can reject relocatable or symlinked venv launchers
            # while they resolve their own path, even when the profile permits
            # the venv and its target. This bootstrap imports only stdlib
            # modules before execing bash, so launch the canonical base
            # interpreter instead of broadening the profile's read roots.
            bootstrap_interpreter = bootstrap_interpreter.resolve()
        argv = sandbox.wrap(
            [str(bootstrap_interpreter), "-I", "-c", _INNER_ENV_BOOTSTRAP, "bash", "-c", rewritten],
            workdir=workdir,
            write_roots=self._allowed_write_roots(),
            home=self._home,
            tmp=self._tmp,
            allow_net=self._allow_net,
            extra_ro=self._runtime_ro_binds(),
            strict_reads=self._strict_reads,
            deny_reads=(output_provenance_key_path(),),
        )
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workdir),
                env=self._launcher_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=local_sandbox.apply_rlimits if apply_limits else None,
                # Descendants inherit this POSIX process group, including through
                # sandbox launchers such as macOS Seatbelt.
                start_new_session=os.name == "posix",
            )
        )
        self._pending_creations.add(creation)
        self._creation_secret_values[creation] = set(output_secret_values)
        try:
            proc = await asyncio.shield(creation)
        except asyncio.CancelledError as primary_error:
            # Cancellation may arrive after the OS process exists but before
            # asyncio returns its handle. Bound how long the caller waits for
            # an uncooperative creation coroutine. The tracked cleanup owns
            # the eventual handle and remains visible to stop().
            cleanup = self._schedule_creation_cleanup(creation)
            resolution = asyncio.create_task(asyncio.wait({creation}, timeout=_CREATION_CANCEL_SECONDS))
            done, _pending = await _await_task_uninterruptibly(
                resolution,
                preserve_cancellation=False,
            )
            if creation in done:
                safe_cleanup_error: RuntimeError | None = None
                try:
                    await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
                except BaseException as cleanup_error:
                    safe_cleanup_error = self._redacted_cleanup_error(cleanup_error, output_secret_values)
                if safe_cleanup_error is not None:
                    self._raise_primary_with_cleanup(
                        primary_error,
                        safe_cleanup_error,
                        output_secret_values,
                        note_prefix="Local process-tree cleanup also failed during creation cancellation",
                    )
            else:
                primary_error.add_note(
                    self._redact_output(
                        "Local process creation cancellation remained pending past the cleanup deadline; "
                        "a tracked late-process reaper remains active",
                        output_secret_values,
                    )
                )
            raise
        except BaseException:
            self._pending_creations.discard(creation)
            self._creation_secret_values.pop(creation, None)
            raise

        if self._stop_requested:
            cleanup = self._schedule_creation_cleanup(creation)
            safe_cleanup_error: RuntimeError | None = None
            try:
                await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except BaseException as cleanup_error:
                safe_cleanup_error = self._redacted_cleanup_error(cleanup_error, output_secret_values)
            if safe_cleanup_error is not None:
                raise safe_cleanup_error from None
            raise RuntimeError("Local process creation completed during environment shutdown")
        self._pending_creations.discard(creation)
        self._creation_secret_values.pop(creation, None)
        callback_error: asyncio.Future[BaseException] | None = None
        callback_output: _StreamCallbackOutput | None = None
        if output_callback is not None:
            callback_error = asyncio.get_running_loop().create_future()
            callback_output = _StreamCallbackOutput(output_callback, callback_error, output_secret_values)
        communication = asyncio.create_task(
            self._collect_streamed_output(
                proc,
                env_payload,
                callback_output,
            )
        )
        self._active_processes[proc] = communication
        self._active_process_secret_values[proc] = set(output_secret_values)
        process_contained = False

        async def terminate_preserving_primary(primary_error: BaseException) -> tuple[bytes, bytes]:
            nonlocal process_contained
            cleanup = asyncio.create_task(self._terminate_process_tree(proc, communication))
            safe_cleanup_error: RuntimeError | None = None
            try:
                result = await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
            except BaseException as cleanup_error:
                safe_cleanup_error = self._redacted_cleanup_error(cleanup_error, output_secret_values)
            if callback_output is not None and not communication.done():
                callback_output.abandon_delivery()
            if safe_cleanup_error is not None:
                self._raise_primary_with_cleanup(
                    primary_error,
                    safe_cleanup_error,
                    output_secret_values,
                    note_prefix="Local process-tree cleanup also failed",
                )
            process_contained = True
            return result

        async def cleanup_process_tree() -> tuple[bytes, bytes]:
            """Contain the group while preserving cancellation as primary."""
            nonlocal process_contained
            cleanup = asyncio.create_task(self._terminate_process_tree(proc, communication))
            safe_cleanup_error: RuntimeError | None = None
            try:
                result = await asyncio.shield(cleanup)
            except asyncio.CancelledError as primary_error:
                try:
                    result = await _await_task_uninterruptibly(cleanup, preserve_cancellation=False)
                except BaseException as cleanup_error:
                    safe_cleanup_error = self._redacted_cleanup_error(cleanup_error, output_secret_values)
                if callback_output is not None and not communication.done():
                    callback_output.abandon_delivery()
                if safe_cleanup_error is not None:
                    self._raise_primary_with_cleanup(
                        primary_error,
                        safe_cleanup_error,
                        output_secret_values,
                        note_prefix="Local process-tree cleanup also failed",
                    )
                process_contained = True
                self._release_active_process(proc)
                raise
            except BaseException as cleanup_error:
                safe_cleanup_error = self._redacted_cleanup_error(cleanup_error, output_secret_values)
            if callback_output is not None and not communication.done():
                callback_output.abandon_delivery()
            if safe_cleanup_error is not None:
                raise safe_cleanup_error from None
            process_contained = True
            self._release_active_process(proc)
            return result

        callback_failure: BaseException | None = None
        timed_out = False
        try:
            try:
                waitables: set[asyncio.Future[Any] | asyncio.Task[Any]] = {communication}
                if callback_error is not None:
                    waitables.add(callback_error)
                done, _pending = await asyncio.wait(
                    waitables,
                    timeout=timeout_sec,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    timed_out = True
                elif callback_error is not None and callback_error in done:
                    callback_failure = callback_error.result()
                else:
                    stdout_b, stderr_b = await asyncio.shield(communication)
            except asyncio.CancelledError as primary_error:
                await terminate_preserving_primary(primary_error)
                raise
            except BaseException as primary_error:
                await terminate_preserving_primary(primary_error)
                raise

            if callback_failure is None and callback_error is not None and callback_error.done():
                callback_failure = callback_error.result()
            if callback_failure is not None:
                await terminate_preserving_primary(callback_failure)
                raise callback_failure

            if timed_out:
                stdout_b, stderr_b = await cleanup_process_tree()
                if callback_output is not None:
                    stdout_b = callback_output.raw_output("stdout")
                    stderr_b = callback_output.raw_output("stderr")
                if callback_error is not None and callback_error.done():
                    callback_failure = callback_error.result()
                    raise callback_failure
                raw_stdout = stdout_b.decode(errors="replace")
                raw_stderr = stderr_b.decode(errors="replace")
                diagnostic = "Timed out" if not raw_stderr or raw_stderr.endswith("\n") else "\nTimed out"
                if callback_output is not None:
                    callback_finish = asyncio.create_task(callback_output.finish(stderr_suffix=diagnostic))

                    async def cancel_callback_finish(*, preserve_cancellation: bool) -> bool:
                        callback_output.abandon_delivery()
                        cancellation = asyncio.create_task(
                            _cancel_task_repeatedly(callback_finish, timeout=_REAP_CANCEL_SECONDS)
                        )
                        return bool(
                            await _await_task_uninterruptibly(
                                cancellation,
                                preserve_cancellation=preserve_cancellation,
                            )
                        )

                    try:
                        done, _pending = await asyncio.wait(
                            {callback_finish},
                            timeout=_REAP_CANCEL_SECONDS,
                        )
                    except asyncio.CancelledError:
                        await cancel_callback_finish(preserve_cancellation=False)
                        raise
                    if callback_finish not in done:
                        if not await cancel_callback_finish(preserve_cancellation=True):
                            raise RuntimeError(
                                self._redact_output(
                                    "Local callback cleanup remained pending past the deadline",
                                    output_secret_values,
                                )
                            )
                    else:
                        callback_finish.result()
                    if callback_error is not None and callback_error.done():
                        raise callback_error.result()
                stdout = self._redact_output(raw_stdout, output_secret_values)
                timeout_stderr = self._redact_output(raw_stderr + diagnostic, output_secret_values)
                return ExecResult(
                    stdout=stdout,
                    stderr=timeout_stderr,
                    return_code=124,
                )

            # proc.wait()/communicate() only proves that the launcher exited;
            # a same-group descendant may have closed its inherited streams
            # and survived. Contain the group immediately, before an
            # arbitrarily slow final callback creates a PID-reuse window.
            await cleanup_process_tree()

            if callback_output is not None:
                await callback_output.finish()
                if callback_error is not None and callback_error.done():
                    callback_failure = callback_error.result()
                    raise callback_failure

            return ExecResult(
                stdout=self._redact_output(stdout_b.decode(errors="replace"), output_secret_values),
                stderr=self._redact_output(stderr_b.decode(errors="replace"), output_secret_values),
                return_code=int(proc.returncode or 0),
            )
        finally:
            if process_contained:
                self._release_active_process(proc)

    @staticmethod
    async def _collect_streamed_output(
        proc: asyncio.subprocess.Process,
        stdin_data: bytes,
        callback_output: _StreamCallbackOutput | None,
    ) -> tuple[bytes, bytes]:
        """Drain both streams within one combined hard raw-byte budget."""
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise RuntimeError("local subprocess pipe invariant violated")
        output_budget = CommandOutputByteBudget()

        async def write_stdin() -> None:
            try:
                proc.stdin.write(stdin_data)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                # Match asyncio's communicate(): an early child exit is
                # represented by its return code, not a host-side pipe error.
                pass
            finally:
                proc.stdin.close()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    await proc.stdin.wait_closed()

        async def drain_stream(
            reader: asyncio.StreamReader,
            stream: OutputStream,
        ) -> bytes:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            output = bytearray()

            while chunk := await reader.read(64 * 1024):
                output_budget.consume(chunk)
                output.extend(chunk)
                if callback_output is not None:
                    callback_output.append_raw(chunk, stream)
                    if text := decoder.decode(chunk):
                        await callback_output.feed(text, stream)
                # A callback coroutine is allowed to complete synchronously;
                # still give command timeout/cancellation and the other stream
                # a scheduling point after each bounded read.
                await asyncio.sleep(0)
            if callback_output is not None and (text := decoder.decode(b"", final=True)):
                await callback_output.feed(text, stream)
            return bytes(output)

        stdin_task = asyncio.create_task(write_stdin())
        stdout_task = asyncio.create_task(drain_stream(proc.stdout, "stdout"))
        stderr_task = asyncio.create_task(drain_stream(proc.stderr, "stderr"))
        wait_task = asyncio.create_task(proc.wait())
        tasks = (stdin_task, stdout_task, stderr_task, wait_task)
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return stdout_task.result(), stderr_task.result()

    async def exec_with_sensitive_env(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute with values delivered through the existing stdin-only bootstrap."""
        return await self.exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    @staticmethod
    async def _terminate_process_tree(
        proc: asyncio.subprocess.Process,
        communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
    ) -> tuple[bytes, bytes]:
        async def reap() -> tuple[bytes, bytes]:
            active_communication = communication
            if active_communication is None:
                active_communication = asyncio.create_task(proc.communicate())
            process_wait: asyncio.Task[tuple[bytes, bytes]] | None = None

            def send(sig: signal.Signals) -> None:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        return
                    except PermissionError:
                        if proc.returncode is not None:
                            return
                        try:
                            os.getpgid(proc.pid)
                        except ProcessLookupError:
                            return
                        raise
                elif proc.returncode is None:
                    if sig == signal.SIGTERM:
                        proc.terminate()
                    else:
                        proc.kill()

            async def wait_for_process_group_exit(seconds: float) -> None:
                if os.name != "posix" or not isinstance(proc, asyncio.subprocess.Process):
                    return
                deadline = asyncio.get_running_loop().time() + seconds
                while True:
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        return
                    except PermissionError:
                        return
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        return
                    await asyncio.sleep(min(0.01, remaining))

            async def bounded_wait(seconds: float) -> tuple[bytes, bytes] | None:
                nonlocal process_wait
                try:
                    return await asyncio.wait_for(asyncio.shield(active_communication), timeout=seconds)
                except TimeoutError:
                    return None
                except BaseException:
                    # A collector failure is the caller's primary error, but
                    # cleanup still has to confirm process exit. Fall back to
                    # waiting on the process handle without re-reading pipes.
                    if not hasattr(proc, "wait"):
                        raise
                    if process_wait is None:

                        async def wait_for_process() -> tuple[bytes, bytes]:
                            await proc.wait()
                            return b"", b""

                        process_wait = asyncio.create_task(wait_for_process())
                    try:
                        return await asyncio.wait_for(asyncio.shield(process_wait), timeout=seconds)
                    except TimeoutError:
                        return None

            send(signal.SIGTERM)
            term_output = await bounded_wait(_REAP_TERM_SECONDS)
            if os.name != "posix" and term_output is not None:
                return term_output

            # On POSIX the launcher can exit and close its pipes while a
            # same-group descendant ignores SIGTERM.  Escalate to the original
            # group immediately even when communication already completed;
            # this minimizes the process-group-ID reuse window and prevents a
            # successful collector from being mistaken for containment.
            send(signal.SIGKILL)
            group_exit = asyncio.create_task(wait_for_process_group_exit(_REAP_KILL_SECONDS))
            kill_output = term_output
            if kill_output is None:
                kill_output = await bounded_wait(_REAP_KILL_SECONDS)
            await group_exit
            if kill_output is not None:
                return kill_output

            if not await _cancel_task_repeatedly(active_communication, timeout=_REAP_CANCEL_SECONDS):
                raise RuntimeError("Local output cleanup remained pending past the reap deadline")
            if process_wait is not None and not process_wait.done():
                process_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await process_wait
            return b"", b""

        cleanup = asyncio.create_task(reap())
        return await _await_task_uninterruptibly(cleanup)

    def _launcher_env(self) -> dict[str, str]:
        """Return the minimal environment visible before confinement starts."""
        return {
            "PATH": self._local_host_env().get("PATH", os.defpath),
            "HOME": str(self._home),
            "TMPDIR": str(self._tmp),
        }

    def _copy_environment_bundle(self) -> None:
        skills_src = self.environment_dir / "skills"
        if skills_src.is_dir():
            self._copy_dir_contents(skills_src, self._workspace / "skills")
            for target in (
                self._home / ".claude" / "skills",
                self._home / ".agents" / "skills",
                self._home / ".config" / "opencode" / "skills",
            ):
                self._copy_dir_contents(skills_src, target)

        input_src = self.environment_dir / "input"
        if input_src.is_dir():
            self._copy_dir_contents(input_src, self._workspace / "input")

        repo_src = self.environment_dir / "repo"
        if repo_src.is_dir():
            self._copy_dir_contents(repo_src, self._workspace / "repo")

        linked_root_src = self.environment_dir / "repo-linked-root"
        if linked_root_src.is_dir():
            self._copy_dir_contents(linked_root_src, self._workspace)

        codex_config = self.environment_dir / "codex-config" / "config.toml"
        if codex_config.is_file():
            shutil.copy2(codex_config, self._home / ".codex" / "config.toml")

    @staticmethod
    def _copy_dir_contents(source: Path, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        copytree_secure(source, target, dirs_exist_ok=True)

    def _exec_env(self, env: dict[str, str] | None) -> dict[str, str]:
        base = {
            "HOME": str(self._home),
            "TMPDIR": str(self._tmp),
            "HARBOR_WORKSPACE_DIR": str(self._workspace),
            "HARBOR_TESTS_DIR": str(self._tests),
            "HARBOR_LOGS_DIR": str(self.trial_paths.trial_dir),
            "HARBOR_AGENT_LOGS_DIR": str(self.trial_paths.agent_dir),
            "HARBOR_VERIFIER_DIR": str(self.trial_paths.verifier_dir),
            "HARBOR_ARTIFACTS_DIR": str(self.trial_paths.artifacts_dir),
            "HARBOR_SOLUTION_DIR": str(self._solution),
            "HARBOR_INSTALLED_AGENT_DIR": str(self._installed_agent),
            "HARBOR_SKILLS_DIR": str(self._workspace / "skills"),
            "HARBOR_INPUT_DIR": str(self._workspace / "input"),
            "HARBOR_ENTRY_JSON": str(self._tests / "entry.json"),
            "HARBOR_REWARD_JSON": str(self.trial_paths.reward_json_path),
            "HARBOR_REWARD_TXT": str(self.trial_paths.reward_text_path),
            "HARBOR_CUSTOM_REWARD_JSON": str(self.trial_paths.verifier_dir / "custom_reward.json"),
            "HARBOR_GRADER": str(self._tests / "grader.py"),
            "HARBOR_GRADER_SH": str(self._tests / "grader.sh"),
        }
        host_env = self._local_host_env(inherit_agent_keys=self._inherit_agent_keys)
        merged = local_subprocess_env(
            runtime_root=self._runtime_root,
            runtime_agents=[self._runtime_agent],
            base_env=host_env,
        )
        merged["PATH"] = self._path_with_evaluator_python(merged.get("PATH", ""))
        merged.update(base)
        persistent = self._merge_env(env) or {}
        merged.update(self._rewrite_env_values(self._filter_command_env(persistent, protected=set(base))))
        merged.update(base)
        return merged

    @staticmethod
    def _local_host_env(*, inherit_agent_keys: bool = False) -> dict[str, str]:
        """Return only the host env values local mode is allowed to inherit.

        Evaluator credentials (``_LIVE_AGENT_KEYS``) are excluded unless the
        caller opted in: agents and verifiers get credentials per-exec, so a
        hostile skill command must not find them in its ambient environment.
        """
        allowed = _SAFE_HOST_ENV | (_LIVE_AGENT_KEYS if inherit_agent_keys else frozenset())
        env = {key: value for key, value in os.environ.items() if key in allowed}
        env.setdefault("PATH", os.defpath)
        return env

    def _runtime_ro_binds(self) -> list[Path]:
        """Read-only mounts the sandbox needs: managed agent CLIs + evaluator python.

        Strict mode publishes the visible and canonical interpreter files plus
        Python's stdlib/site-library roots; it must not expose an entire prefix
        or bin directory. Compatibility mode keeps the broader historical
        prefix/parent/site roots for unusual interpreter layouts.
        """
        strict_reads = getattr(self, "_strict_reads", False)
        visible_executable = Path(sys.executable).expanduser().absolute()
        executable = visible_executable.resolve()
        if strict_reads:
            import sysconfig

            python_paths = sysconfig.get_paths()
            candidates = [visible_executable, executable]
            candidates.extend(
                Path(path)
                for name in ("stdlib", "platstdlib", "purelib", "platlib")
                if (path := python_paths.get(name))
            )
            library_dir = sysconfig.get_config_var("LIBDIR")
            library_name = sysconfig.get_config_var("LDLIBRARY")
            if library_dir and library_name:
                candidates.append(Path(library_dir) / library_name)
            # A Homebrew framework launcher links this exact image outside the
            # stdlib tree.  Publish the image, never its Cellar/prefix parent.
            framework_version = next(
                (parent for parent in executable.parents if parent.parent.name == "Versions"),
                None,
            )
            if framework_version is not None:
                candidates.append(framework_version / "Python")
                candidates.append(framework_version / "lib" / f"python{framework_version.name}")
        else:
            import site

            candidates = [
                Path(sys.prefix),
                Path(sys.exec_prefix),
                Path(sys.base_prefix),
                Path(sys.base_exec_prefix),
                executable.parent,  # the bin/ dir
                executable.parent.parent,  # the usual install prefix
            ]
            with contextlib.suppress(Exception):
                candidates.extend(Path(p) for p in site.getsitepackages())
            with contextlib.suppress(Exception):
                candidates.append(Path(site.getusersitepackages()))
        candidates.extend(runtime_command_roots([self._runtime_agent], runtime_root=self._runtime_root))
        binds: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            raw = candidate.expanduser().absolute()
            visible = raw.parent.resolve() / raw.name
            path = visible if visible.is_symlink() else visible.resolve()
            if path.exists() and path not in seen:
                seen.add(path)
                binds.append(path)
        return binds

    def _local_command_guardrail_reason(
        self,
        command: str,
        rewritten_command: str,
        env: dict[str, str],
    ) -> str:
        """Advisory defense-in-depth checks with friendly error messages.

        The security boundary is the OS sandbox in ``local_sandbox``; these
        string-level checks exist to fail obvious mistakes fast and explain
        why, not to contain a hostile command.
        """
        command_one_line = " ".join(command.split())
        if re.search(r"(?:^|[;&|]\s*)rm\s+-[^\n;&|]*r[^\n;&|]*f[^\n;&|]*\s+/(?:\s|$)", command_one_line):
            return "refusing destructive rm -rf / in trusted-host local mode."
        if re.search(r"(?:^|[;&|]\s*)(?:curl|wget)\b[^;&]*\|\s*(?:sudo\s+)?(?:bash|sh)\b", command_one_line):
            return "refusing curl/wget piped directly into a shell in trusted-host local mode."

        sensitive_path = self._sensitive_host_path_reference(command)
        if sensitive_path:
            return f"refusing access to sensitive host path {sensitive_path}."

        if self._background_command(command):
            return (
                "background commands are unsupported in local mode because processes cannot survive between sandbox "
                "invocations; use Docker or a cloud environment."
            )

        if _contains_detached_process_launcher(command):
            return (
                "common detached process launchers (setsid, nohup, daemon, or disown) are unsupported in local "
                "mode; use Docker or a cloud environment for services or untrusted code."
            )

        unsafe_target = self._unsafe_write_target(rewritten_command, exec_env=env)
        if unsafe_target:
            return f"write target {unsafe_target} is outside the local run directory."
        return ""

    def _sensitive_host_path_reference(self, command: str) -> str:
        for pattern in _SENSITIVE_HOST_PATH_RES:
            match = pattern.search(command)
            if match:
                return match.group("path")
        return ""

    @staticmethod
    def _background_command(command: str) -> bool:
        return _contains_background_command(command)

    def _unsafe_write_target(self, command: str, *, exec_env: dict[str, str]) -> Path | None:
        for token in self._shell_write_redirect_targets(command):
            target = self._shell_token_to_path(token, exec_env=exec_env)
            if target and not self._path_is_within_allowed_write_roots(target):
                return target

        for match in _SHELL_WRITE_COMMAND_RE.finditer(command):
            command_name = match.group(0).lstrip(";&| ").split(maxsplit=1)[0]
            for token in self._write_command_targets(command_name, match.group("args")):
                target = self._shell_token_to_path(token, exec_env=exec_env)
                if target and not self._path_is_within_allowed_write_roots(target):
                    return target
        return None

    @staticmethod
    def _shell_write_redirect_targets(command: str) -> list[str]:
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return [match.group(1) for match in _SHELL_WRITE_REDIRECT_RE.finditer(command)]

        targets: list[str] = []
        for index, token in enumerate(tokens[:-1]):
            if token in {">", ">>", ">|", "&>", "&>>"} or (
                token == ">&" and not tokens[index + 1].isdigit() and tokens[index + 1] != "-"
            ):
                targets.append(tokens[index + 1])
        return targets

    @staticmethod
    def _pathlike_tokens(args: str) -> list[str]:
        return [token for token in SkillEvaluatorLocalEnvironment._shell_tokens(args) if _looks_like_path_token(token)]

    @staticmethod
    def _shell_tokens(args: str) -> list[str]:
        try:
            return shlex.split(args)
        except ValueError:
            return args.split()

    def _write_command_targets(self, command_name: str, args: str) -> list[str]:
        tokens = self._shell_tokens(args)
        if command_name in {"cp", "mv"}:
            for index, token in enumerate(tokens):
                if token in {"-t", "--target-directory"} and index + 1 < len(tokens):
                    return [tokens[index + 1]]
                if token.startswith("--target-directory="):
                    return [token.split("=", 1)[1]]
            pathlike = [token for token in tokens if _looks_like_path_token(token)]
            return pathlike[-1:]  # source operands are reads; only the final operand is the destination.
        return [token for token in tokens if _looks_like_path_token(token)]

    def _shell_token_to_path(self, token: str, *, exec_env: dict[str, str]) -> Path | None:
        token = token.strip().strip("'\"")
        if not token:
            return None
        env_path = self._env_token_to_path(token, exec_env)
        if env_path is not None:
            return env_path
        if token.startswith("$HOME/"):
            return self._home / token[len("$HOME/") :]
        if token.startswith("${HOME}/"):
            return self._home / token[len("${HOME}/") :]
        if token.startswith("~/"):
            return self._home / token[2:]
        if token.startswith("$TMPDIR/"):
            return self._tmp / token[len("$TMPDIR/") :]
        if token.startswith("${TMPDIR}/"):
            return self._tmp / token[len("${TMPDIR}/") :]
        path = Path(token)
        if not path.is_absolute():
            return None
        return path

    def _env_token_to_path(self, token: str, exec_env: dict[str, str]) -> Path | None:
        if token.startswith("${"):
            end = token.find("}")
            if end < 0:
                return None
            name = token[2:end]
            suffix = token[end + 1 :]
        elif token.startswith("$"):
            match = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)(.*)", token)
            if not match:
                return None
            name = match.group(1)
            suffix = match.group(2)
        else:
            return None

        value = exec_env.get(name)
        if not value:
            return None
        base = Path(value)
        if not base.is_absolute():
            return None
        if not suffix:
            return base
        if suffix.startswith("/"):
            return base / suffix[1:]
        return None

    def _path_is_within_allowed_write_roots(self, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path.absolute()
        if str(resolved) in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
            return True
        return any(self._is_relative_to(resolved, root) for root in self._allowed_write_roots())

    def _path_is_within_allowed_local_roots(self, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path.absolute()
        return any(self._is_relative_to(resolved, root) for root in self._allowed_write_roots())

    def _allowed_write_roots(self) -> tuple[Path, ...]:
        return (
            self._root,
            self.trial_paths.trial_dir,
        )

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            return False

    def _output_secret_values(self, env: dict[str, str] | None, exec_env: dict[str, str]) -> set[str]:
        merged = self._merge_env(env) or {}
        secret_values = {
            value
            for key, value in merged.items()
            if value
            and (
                _SENSITIVE_ENV_NAME_RE.search(key)
                or _COMPACT_SENSITIVE_ENV_NAME_RE.search(key)
                or (len(value) >= _MIN_LEGACY_SECRET_VALUE_LENGTH and _LEGACY_SECRET_ENV_NAME_RE.search(key))
            )
        }
        secret_values.update(
            value
            for key, value in exec_env.items()
            if value
            and (
                _SENSITIVE_ENV_NAME_RE.search(key)
                or _COMPACT_SENSITIVE_ENV_NAME_RE.search(key)
                or (len(value) >= _MIN_LEGACY_SECRET_VALUE_LENGTH and _LEGACY_SECRET_ENV_NAME_RE.search(key))
            )
        )
        secret_values.update(_credential_uri_environment_values(merged))
        secret_values.update(_credential_uri_environment_values(exec_env))
        return secret_values

    @staticmethod
    def _redact_output(text: str, secret_values: set[str]) -> str:
        redactor = StreamingLogRedactor(secret_values)
        return redactor.feed(text) + redactor.finish()

    def _redacted_cleanup_error(self, error: BaseException, secret_values: set[str]) -> RuntimeError:
        """Return a fresh cleanup error with no reference to the raw exception."""
        error_type = type(error).__name__
        try:
            summary = f"{error_type}: {error}"
        except BaseException:
            summary = f"{error_type}: cleanup detail unavailable"
        return RuntimeError(self._redact_output(summary, secret_values))

    def _raise_primary_with_cleanup(
        self,
        primary_error: BaseException,
        cleanup_error: RuntimeError,
        secret_values: set[str],
        *,
        note_prefix: str,
    ) -> NoReturn:
        note = self._redact_output(
            f"{note_prefix}: {cleanup_error}",
            secret_values,
        )
        primary_error.add_note(note)
        raise primary_error from cleanup_error

    def _path_with_evaluator_python(self, path: str) -> str:
        """Ensure local verifier scripts use the evaluator's Python runtime."""
        parts = [piece for piece in path.split(os.pathsep) if piece]
        python_bin = str(Path(sys.executable).resolve().parent)
        if python_bin in parts:
            return os.pathsep.join(parts)

        runtime_prefix = {str(path) for path in runtime_bin_dirs(self._runtime_root, agents=[self._runtime_agent])}
        insert_at = 0
        while insert_at < len(parts) and parts[insert_at] in runtime_prefix:
            insert_at += 1
        parts.insert(insert_at, python_bin)
        return os.pathsep.join(parts)

    def _rewrite_env_values(self, env: dict[str, str]) -> dict[str, str]:
        """Map Harbor container paths that are passed through environment values."""
        return {key: self._rewrite_raw_paths(value) for key, value in env.items()}

    @staticmethod
    def _filter_command_env(env: dict[str, str], *, protected: set[str]) -> dict[str, str]:
        """Reject process-control values and drop protected path overrides."""
        out: dict[str, str] = {}
        for key, value in env.items():
            normalized = key.upper()
            if normalized in _BLOCKED_COMMAND_ENV_NAMES or normalized.startswith(_BLOCKED_COMMAND_ENV_PREFIXES):
                # Docker tasks deliberately reset loader-controlled variables
                # to the empty string.  In local mode, absence is the safer
                # equivalent; reject every non-empty value and drop the reset.
                if value == "":
                    continue
                raise ValueError(
                    f"environment variable {key} can execute or alter code before confinement and is not allowed"
                )
            if key in protected or key in {"HOME", "TMPDIR", "PATH", "PWD"}:
                continue
            if key.startswith("HARBOR_") and key != "HARBOR_DECLARED_PORTS":
                continue
            out[key] = value
        return out

    def _path_map(self) -> list[tuple[str, Path]]:
        return [
            ("/logs/agent", self.trial_paths.agent_dir),
            ("/logs/verifier", self.trial_paths.verifier_dir),
            ("/logs/artifacts", self.trial_paths.artifacts_dir),
            ("/logs", self.trial_paths.trial_dir),
            ("/workspace", self._workspace),
            ("/tests", self._tests),
            ("/solution", self._solution),
            ("/installed-agent", self._installed_agent),
        ]

    def _rewrite_command(self, command: str) -> str:
        rewritten = command
        for remote, local in self._path_map():
            rewritten = self._replace_shell_path(rewritten, remote, local)
        return rewritten

    def _rewrite_raw_paths(self, value: str) -> str:
        path_map = dict(self._path_map())
        if not path_map:
            return value

        # Match all container roots in one pass so a container-looking segment
        # in a generated local path is not rewritten a second time. Longer
        # roots must win (for example, /logs/agent before /logs).
        remote_roots = sorted(path_map, key=len, reverse=True)
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_.\-/])(?P<root>{'|'.join(re.escape(root) for root in remote_roots)})"
            r"(?P<separator>/|(?=$|[^A-Za-z0-9_.-]))"
        )

        def replace(match: re.Match[str]) -> str:
            separator = os.sep if match.group("separator") else ""
            return f"{path_map[match.group('root')]}{separator}"

        return pattern.sub(replace, value)

    @staticmethod
    def _replace_shell_path(command: str, remote: str, local: Path) -> str:
        """Replace a container path while preserving shell token boundaries."""
        output: list[str] = []
        index = 0
        quote = ""
        local_text = str(local)
        shell_boundaries = "/ \t\r\n'\";&|<>"
        while index < len(command):
            if command.startswith(remote, index) and (
                index + len(remote) == len(command) or command[index + len(remote)] in shell_boundaries
            ):
                if quote == "'":
                    replacement = local_text.replace("'", "'\"'\"'")
                elif quote == '"':
                    replacement = (
                        local_text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
                    )
                else:
                    replacement = shlex.quote(local_text)
                output.append(replacement)
                index += len(remote)
                continue

            char = command[index]
            output.append(char)
            if char == "\\" and quote != "'" and index + 1 < len(command):
                index += 1
                output.append(command[index])
            elif char in {"'", '"'}:
                if not quote:
                    quote = char
                elif quote == char:
                    quote = ""
            index += 1
        return "".join(output)

    def _rewrite_uploaded_scripts(self, target: Path) -> None:
        if target.is_file():
            self._rewrite_uploaded_script(target)
            return
        if not target.is_dir():
            return
        for file_path in target.rglob("*"):
            self._rewrite_uploaded_script(file_path)

    def _rewrite_uploaded_script(self, target: Path) -> None:
        if not target.is_file() or target.suffix not in {".bash", ".py", ".sh"}:
            return
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return
        rewritten = self._rewrite_raw_paths(text) if target.suffix == ".py" else self._rewrite_command(text)
        if rewritten != text:
            # copy2 preserves published template modes; restore owner-write on
            # the local run copy before rewriting container paths in place.
            if not os.access(target, os.W_OK):
                target.chmod(target.stat().st_mode | 0o200)
            target.write_text(rewritten, encoding="utf-8")

    def _resolve_path(self, path: str | Path | None) -> Path:
        if path is None:
            return self._workspace
        raw = str(path)
        for remote, local in self._path_map():
            if raw == remote:
                return local
            if raw.startswith(remote + "/"):
                return local / raw[len(remote) + 1 :]
        if raw.startswith("~/"):
            return self._home / raw[2:]
        candidate = Path(raw)
        if candidate.is_absolute():
            if not self._path_is_within_allowed_local_roots(candidate):
                raise ValueError(f"path {raw} is outside the local run directory")
            return candidate
        return self._workspace / candidate
