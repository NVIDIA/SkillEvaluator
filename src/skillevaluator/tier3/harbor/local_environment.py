# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted host execution backend for Harbor local mode."""

from __future__ import annotations

import asyncio
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
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
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
from skillevaluator.tier3.harbor.secret_redaction import redact_secrets_in_log_line
from skillevaluator.tier3.harbor.secure_copy import copytree_secure

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
_SECRET_ENV_NAME_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE)
_SHELL_WRITE_REDIRECT_RE = re.compile(r"(?:^|\s)(?:\d?>{1,2}|&>)\s*([^\s;&|]+)")
_SHELL_WRITE_COMMAND_RE = re.compile(r"(?:^|[;&|]\s*)(?:tee|touch|mkdir|cp|mv)\b(?P<args>[^;&|]*)")
_BACKGROUND_AMPERSAND_RE = re.compile(r"(?<![&>])&(?![&>])")
_DETACHED_PROCESS_COMMANDS = frozenset({"setsid", "nohup", "daemon", "disown"})
_SHELL_COMMANDS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_COMMAND_PREFIXES = frozenset({"command", "do", "elif", "env", "exec", "if", "then", "until", "while"})
_REAP_TERM_SECONDS = 1.0
_REAP_KILL_SECONDS = 1.0
_REAP_CANCEL_SECONDS = 0.1
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
        for proc, communication in tuple(self._active_processes.items()):
            await self._terminate_process_tree(proc, communication)
        self._active_processes.clear()
        if delete and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)

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
        rewritten = self._rewrite_command(command)
        try:
            workdir = self._resolve_path(cwd) if cwd else self._workspace
        except ValueError as exc:
            return ExecResult(stdout="", stderr=f"Local mode command blocked: {exc}", return_code=126)
        if not self._path_is_within_allowed_local_roots(workdir):
            return ExecResult(
                stdout="",
                stderr=f"Local mode command blocked: cwd {workdir} is outside the local run directory.",
                return_code=126,
            )
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            exec_env = self._exec_env(env)
        except ValueError as exc:
            self.logger.warning("Local mode command blocked: %s", exc)
            return ExecResult(
                stdout="",
                stderr=f"Local mode command blocked: {exc}",
                return_code=126,
            )
        guardrail_reason = self._local_command_guardrail_reason(command, rewritten, exec_env)
        if guardrail_reason:
            self.logger.warning("Local mode command blocked: %s", guardrail_reason)
            return ExecResult(
                stdout="",
                stderr=f"Local mode command blocked: {guardrail_reason}",
                return_code=126,
            )

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
        try:
            proc = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # A second cancellation must not propagate into ``creation`` after
            # the OS process exists but before asyncio returns its handle.
            proc = await _await_task_uninterruptibly(creation, preserve_cancellation=False)
            await self._terminate_process_tree(proc)
            raise
        self._active_processes[proc] = None
        communication = asyncio.create_task(proc.communicate(input=env_payload))
        self._active_processes[proc] = communication
        try:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=timeout_sec,
                )
            except TimeoutError:
                stdout_b, stderr_b = await self._terminate_process_tree(proc, communication)
                stdout = self._redact_output(stdout_b.decode(errors="replace"), exec_env)
                stderr = self._redact_output(stderr_b.decode(errors="replace"), exec_env)
                return ExecResult(
                    stdout=stdout,
                    stderr=(stderr + "\nTimed out").strip(),
                    return_code=124,
                )
            except asyncio.CancelledError:
                await self._terminate_process_tree(proc, communication)
                raise

            return ExecResult(
                stdout=self._redact_output(stdout_b.decode(errors="replace"), exec_env),
                stderr=self._redact_output(stderr_b.decode(errors="replace"), exec_env),
                return_code=int(proc.returncode or 0),
            )
        finally:
            self._active_processes.pop(proc, None)

    @staticmethod
    async def _terminate_process_tree(
        proc: asyncio.subprocess.Process,
        communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
    ) -> tuple[bytes, bytes]:
        async def reap() -> tuple[bytes, bytes]:
            active_communication = communication
            if active_communication is None:
                active_communication = asyncio.create_task(proc.communicate())

            def send(sig: signal.Signals) -> None:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, sig)
                elif proc.returncode is None:
                    if sig == signal.SIGTERM:
                        proc.terminate()
                    else:
                        proc.kill()

            async def bounded_wait(seconds: float) -> tuple[bytes, bytes] | None:
                try:
                    return await asyncio.wait_for(asyncio.shield(active_communication), timeout=seconds)
                except TimeoutError:
                    return None

            send(signal.SIGTERM)
            if output := await bounded_wait(_REAP_TERM_SECONDS):
                return output
            send(signal.SIGKILL)
            if output := await bounded_wait(_REAP_KILL_SECONDS):
                return output

            active_communication.cancel()
            done, _pending = await asyncio.wait({active_communication}, timeout=_REAP_CANCEL_SECONDS)
            if active_communication in done:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    active_communication.result()
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

    @staticmethod
    def _redact_output(text: str, env: dict[str, str]) -> str:
        secret_values = [
            value for key, value in env.items() if _SECRET_ENV_NAME_RE.search(key) and value and len(value) >= 8
        ]
        return redact_secrets_in_log_line(text, extra_secret_values=secret_values)

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
