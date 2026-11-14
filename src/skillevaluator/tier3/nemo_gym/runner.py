# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Tier 3 cases on NeMo Gym instead of Harbor.

SkillEvaluator still turns ``evals.json`` into task folders with its own verifier
(``tests/test.sh``). NeMo Gym then owns execution: ``gym eval run <tasks>`` starts its
Harbor-format environment server, model server and agent harness, runs each case in a
Gym sandbox and grades it with that verifier. The rollouts are normalized back into the
trial layout the shared collector reads, so scores, lift and reports are unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.progress import (
    MIN_EXACT_SECRET_CHARS,
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    redact_progress_detail,
    secret_values_from_environment,
)
from skillevaluator.tier3.nemo_gym.backend import (
    DEFAULT_RUN_TIMEOUT_S,
    MCP_SERVERS_ERROR,
    MODEL_API_KEY_ENV,
    MODEL_URL_ENV,
    NATIVE_MCP_SERVERS_ERROR,
    NEMO_GYM_AGENTS,
    NEMO_GYM_BACKEND,
    NEMO_GYM_BIN_ENV,
    RUN_TIMEOUT_ENV,
    mcp_servers_declared,
    model_url_credential_error,
    native_mcp_servers_declared,
    nemo_gym_bin,
    nemo_gym_capability_errors,
    nemo_gym_cli_error,
    nemo_gym_harbor_only_option_errors,
    nemo_gym_sandbox_error,
    nemo_gym_timeout_s,
    nemo_gym_version,
    uses_native_harbor_tasks,
)
from skillevaluator.tier3.nemo_gym.results import read_rollout_failures, read_rollouts, write_job_from_rollouts

__all__ = ["MODEL_API_KEY_ENV", "MODEL_URL_ENV", "RUN_TIMEOUT_ENV"]

SANDBOX_ENV = "SKILL_EVAL_NEMO_GYM_SANDBOX"
HOST_ENV = "SKILL_EVAL_NEMO_GYM_HOST"


_SKILLS_DIR_LINE = re.compile(r"^skills_dir\s*=.*\n?", re.MULTILINE)


def stage_for_gym(
    arm_dir: Path,
    verifier_env: dict[str, str] | None = None,
    *,
    timeout_multiplier: float = 1.0,
    resource_overrides: dict[str, int] | None = None,
) -> None:
    """Adapt Harbor-format tasks for Gym's agents and verifier.

    Drops Harbor's ``skills_dir`` task hook, which Gym's CLI agents reject. The task image
    already carries the skills at each agent's native location (``/root/.agents/skills``,
    ``/root/.claude/skills``, ...), so the with-skill arm is unchanged and the baseline image
    still has none. Adds ``verifier_env`` (``${NAME}`` templates) to ``[verifier.env]``: Harbor
    passes the judge overrides at job level, and Gym has no job-level verifier env. Scales the
    agent, verifier and environment-build timeouts by ``timeout_multiplier``, as Harbor's
    ``--timeout-multiplier`` does, because Gym enforces the task's own timeouts. Writes the
    ``--override-cpus``/``--override-memory-mb`` values into ``[environment]``, which Harbor
    applies at job level. Raises ``ValueError`` for a task that declares MCP servers.
    """
    for task_toml in arm_dir.glob("*/task.toml"):
        text = task_toml.read_text(encoding="utf-8")
        if (tomllib.loads(text).get("environment") or {}).get("mcp_servers"):
            raise ValueError(f"{task_toml.parent.name}: {NATIVE_MCP_SERVERS_ERROR}")
        staged = _SKILLS_DIR_LINE.sub("", text)
        existing = set((tomllib.loads(staged).get("verifier") or {}).get("env") or {})
        lines = "".join(
            f"{json.dumps(name)} = {json.dumps(value)}\n"
            for name, value in sorted((verifier_env or {}).items())
            if name not in existing
        )
        if lines:
            header = re.search(r"^\[verifier\.env\][ \t]*\n", staged, re.MULTILINE)
            if header:
                staged = staged[: header.end()] + lines + staged[header.end() :]
            else:
                staged = staged.rstrip("\n") + "\n\n[verifier.env]\n" + lines
        if timeout_multiplier != 1.0:
            staged = _scaled_task_timeouts(staged, timeout_multiplier, task_file=task_toml)
            # Harbor's multiplier also scales the verifier's 600 s default.
            staged = _set_table_numbers(
                staged,
                "verifier",
                {"timeout_sec": _DEFAULT_VERIFIER_TIMEOUT_SEC * timeout_multiplier},
                only_missing=True,
                task_file=task_toml,
            )
        if resource_overrides:
            if "memory_mb" in resource_overrides:
                # Harbor's deprecated `memory = "4G"` would conflict with the overriding memory_mb.
                staged = _drop_table_key(staged, "environment", "memory", task_file=task_toml)
            staged = _set_table_numbers(staged, "environment", resource_overrides, task_file=task_toml)
        if staged != text:
            task_toml.write_text(staged, encoding="utf-8")


_TIMEOUT_FIELDS = {
    "agent": "timeout_sec",
    "verifier": "timeout_sec",
    "environment": "build_timeout_sec",
    "steps.agent": "timeout_sec",
    "steps.verifier": "timeout_sec",
}
# Harbor's default environment build timeout, which its multiplier also scales when a task names none.
_DEFAULT_BUILD_TIMEOUT_SEC = 600.0
# Harbor's VerifierConfig.timeout_sec default.
_DEFAULT_VERIFIER_TIMEOUT_SEC = 600.0
_TRIPLE_QUOTES = ('"' * 3, "'" * 3)
_TABLE_RE = re.compile(r"^[ \t]*\[{1,2}[ \t]*([A-Za-z0-9_.-]+)[ \t]*\]{1,2}[ \t]*(?:#.*)?$")
_NUMBER_RE = re.compile(r"^([ \t]*)([A-Za-z_]+)([ \t]*=[ \t]*)([0-9][0-9_.eE+-]*)([ \t]*(?:#.*)?)$")


def _scaled_task_timeouts(text: str, multiplier: float, *, task_file: Path) -> str:
    """Scale a staged task.toml's timeouts by Harbor's multiplier, checking nothing else changed."""
    from skillevaluator.tier3.harbor.runner import _scaled_harbor_timeout

    lines = text.split("\n")
    table = ""
    open_string: str | None = None
    build_timeout_seen = False
    environment_end: int | None = None
    for index, line in enumerate(lines):
        if open_string is not None:
            if line.count(open_string) % 2:
                open_string = None
            continue
        header = _TABLE_RE.match(line)
        if header:
            if table == "environment" and environment_end is None:
                environment_end = index
            table = header.group(1)
            continue
        number = _NUMBER_RE.match(line)
        if number and _TIMEOUT_FIELDS.get(table) == number.group(2):
            field = f"{table}.{number.group(2)}"
            scaled = _scaled_harbor_timeout(number.group(4), multiplier, task_file=task_file, field=field)
            build_timeout_seen = build_timeout_seen or table == "environment"
            lines[index] = f"{number.group(1)}{number.group(2)}{number.group(3)}{scaled!r}{number.group(5)}"
            continue
        open_string = next((quote for quote in ('"""', "'''") if line.count(quote) % 2), None)
    if table == "environment" and environment_end is None:
        environment_end = len(lines)
    expected = tomllib.loads(text)
    _scale_timeout_values(expected, multiplier)
    if not build_timeout_seen and environment_end is not None:
        while environment_end > 0 and not lines[environment_end - 1].strip():
            environment_end -= 1
        lines.insert(environment_end, f"build_timeout_sec = {_DEFAULT_BUILD_TIMEOUT_SEC * multiplier!r}")
        expected["environment"]["build_timeout_sec"] = _DEFAULT_BUILD_TIMEOUT_SEC * multiplier
    staged = "\n".join(lines)
    if tomllib.loads(staged) != expected:
        raise ValueError(f"{task_file}: could not apply --timeout-multiplier to the task timeouts")
    return staged


_KEY_RE = re.compile(r"^[ \t]*([A-Za-z0-9_-]+)[ \t]*=")


def _drop_table_key(text: str, table_name: str, key: str, *, task_file: Path) -> str:
    """Remove a single-line ``key = value`` from one top-level task.toml table, checking nothing else changed."""
    expected = tomllib.loads(text)
    table_data = expected.get(table_name)
    if not isinstance(table_data, dict) or key not in table_data:
        return text
    del table_data[key]
    lines = text.split("\n")
    table = ""
    for index, line in enumerate(lines):
        header = _TABLE_RE.match(line)
        if header:
            table = header.group(1)
            continue
        match = _KEY_RE.match(line)
        if table == table_name and match and match.group(1) == key:
            del lines[index]
            break
    staged = "\n".join(lines)
    if tomllib.loads(staged) != expected:
        raise ValueError(f"{task_file}: could not replace [{table_name}] {key} with the override")
    return staged


def _set_table_numbers(
    text: str, table_name: str, values: dict[str, Any], *, only_missing: bool = False, task_file: Path
) -> str:
    """Set numeric keys of one top-level task.toml table, adding the keys or the table when absent."""
    lines = text.split("\n")
    table = ""
    open_string: str | None = None
    seen: set[str] = set()
    table_end: int | None = None
    for index, line in enumerate(lines):
        if open_string is not None:
            if line.count(open_string) % 2:
                open_string = None
            continue
        header = _TABLE_RE.match(line)
        if header:
            if table == table_name and table_end is None:
                table_end = index
            table = header.group(1)
            continue
        number = _NUMBER_RE.match(line)
        if table == table_name and number and number.group(2) in values:
            seen.add(number.group(2))
            if not only_missing:
                value = values[number.group(2)]
                lines[index] = f"{number.group(1)}{number.group(2)}{number.group(3)}{value!r}{number.group(5)}"
            continue
        open_string = next((quote for quote in _TRIPLE_QUOTES if line.count(quote) % 2), None)
    if table == table_name and table_end is None:
        table_end = len(lines)
    expected = tomllib.loads(text)
    current = expected.setdefault(table_name, {})
    if not isinstance(current, dict):
        raise ValueError(f"{task_file}: [{table_name}] is not a table")
    missing = {key: value for key, value in values.items() if key not in current and key not in seen}
    for key, value in values.items():
        if key in missing or not only_missing:
            current[key] = value
    added = [f"{key} = {value!r}" for key, value in missing.items()]
    if added:
        if table_end is None:
            body = lines[:-1] if lines and lines[-1] == "" else lines
            lines = [*body, "", f"[{table_name}]", *added, ""]
        else:
            while table_end > 0 and not lines[table_end - 1].strip():
                table_end -= 1
            lines[table_end:table_end] = added
    staged = "\n".join(lines)
    if tomllib.loads(staged) != expected:
        raise ValueError(f"{task_file}: could not set [{table_name}] {', '.join(values)} in the task")
    return staged


def _scale_timeout_values(data: dict[str, Any], multiplier: float) -> None:
    tables = [(data, section) for section in ("agent", "verifier", "environment")]
    tables += [
        (step, section)
        for step in data.get("steps") or []
        if isinstance(step, dict)
        for section in ("agent", "verifier")
    ]
    for parent, section in tables:
        key = "steps." + section if parent is not data else section
        table = parent.get(section)
        if isinstance(table, dict) and _TIMEOUT_FIELDS[key] in table:
            table[_TIMEOUT_FIELDS[key]] = float(table[_TIMEOUT_FIELDS[key]]) * multiplier


def _task_case_ids(arm_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for task_dir in sorted(p for p in arm_dir.iterdir() if (p / "task.toml").is_file()):
        metadata = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8")).get("metadata") or {}
        entry_id = metadata.get("entry_id")
        mapping[task_dir.name] = entry_id if isinstance(entry_id, str) and entry_id else task_dir.name
    return mapping


def _model_endpoint(model: str | None) -> tuple[str, str, str]:
    """Return ``(model, base_url, api_key)`` for Gym's OpenAI-compatible policy model server."""
    from skillevaluator.provider_config import resolve_llm_provider

    url = os.environ.get(MODEL_URL_ENV, "").strip()
    key = os.environ.get(MODEL_API_KEY_ENV, "").strip()
    if url and not key:
        # Never send the evaluator's provider key to a policy server it was not issued for.
        raise ValueError(f"{MODEL_URL_ENV} is set, so also set {MODEL_API_KEY_ENV} (use EMPTY for a keyless server)")
    resolved_model = model
    if not url or not key or not resolved_model:
        provider = resolve_llm_provider()
        url = url or (provider.base_url or "")
        key = key or (provider.api_key or "")
        resolved_model = resolved_model or provider.model
    if not url or not key or not resolved_model:
        raise ValueError(
            f"--backend nemo-gym needs a model, an OpenAI-compatible base URL and an API key "
            f"(--model, {MODEL_URL_ENV}, {MODEL_API_KEY_ENV}, or the SkillEvaluator LLM provider settings)"
        )
    return resolved_model, url, key


def verifier_judge_env(skill_path: Path, grading_mode: str, env_mode: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(task verifier env templates, Gym process env)`` carrying the judge settings.

    Mirrors the Harbor runner's staging. Gym has no job-level env layer, so the job-level
    judge overrides are folded into the task templates, which the Gym environment server
    resolves from its own environment; the values never enter a task folder or row.
    """
    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
    from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
    from skillevaluator.tier3.harbor.runner import (
        _VERIFIER_JUDGE_CONTROL_ENV_VARS,
        _VERIFIER_JUDGE_MODEL_ENV_VARS,
        _job_judge_subprocess_env,
        _job_judge_verifier_env,
        _provider_environment,
        _resolve_runtime_env,
    )

    try:
        provider = resolve_llm_provider()
        config, _config_path = load_evals_config(skill_path)
    except (ProviderConfigurationError, EvalsConfigError) as exc:
        raise ValueError(str(exc)) from exc
    runtime_env, runtime_errors = _resolve_runtime_env(
        (config.get("harbor") or {}).get("runtime_env"), env_mode=env_mode
    )
    if runtime_errors:
        raise ValueError("; ".join(runtime_errors))
    provider_env = _provider_environment(provider)
    values = {**runtime_env, **provider_env}
    templates = {
        name: f"${{{name}}}"
        for name in values
        if name not in _VERIFIER_JUDGE_MODEL_ENV_VARS
        and not (grading_mode == "custom_only" and name in _VERIFIER_JUDGE_CONTROL_ENV_VARS)
    }
    templates.update(_job_judge_verifier_env(provider_env, grading_mode))
    return templates, {**values, **_job_judge_subprocess_env(provider_env, grading_mode)}


def gym_eval_command(
    gym: str,
    arm_dir: Path,
    *,
    agent: str,
    model: str,
    model_url: str,
    rollouts_path: Path,
    n_attempts: int,
    n_concurrent: int | None,
    sandbox: str,
    inherit_sandbox_environment: bool = False,
) -> list[str]:
    """Build the ``gym eval run`` command for one agent and arm; the API key is read from the environment."""
    command = [
        gym,
        "eval",
        "run",
        str(arm_dir),
        "--agent",
        NEMO_GYM_AGENTS[agent],
        "--sandbox",
        sandbox,
        "--model-type",
        "openai_model",
        "--model",
        model,
        "--model-url",
        model_url,
        "--output",
        str(rollouts_path),
        f"+policy_api_key=${{oc.env:{MODEL_API_KEY_ENV}}}",
        f"+num_repeats={n_attempts}",
        "+resume_from_cache=false",
    ]
    if n_concurrent:
        command += ["--concurrency", str(n_concurrent)]
    if inherit_sandbox_environment:
        # Gym's Codex agent starts Codex with only its own variables unless told to keep the sandbox's.
        gym_agent = NEMO_GYM_AGENTS[agent]
        command.append(f"++{gym_agent}.responses_api_agents.{gym_agent}.inherit_sandbox_environment=true")
    host = os.environ.get(HOST_ENV, "").strip()
    if host:
        command.append(f"+default_host={host}")
    return command


# Package-index settings, and the credentials for them, that Gym's own ``uv sync`` of its servers reads.
_GYM_HOST_ENV_VARS = frozenset(
    {"NETRC", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL", "UV_INDEX", "UV_INDEX_URL", "UV_KEYRING_PROVIDER"}
)
_UV_INDEX_CREDENTIAL_ENV = re.compile(r"UV_INDEX_[A-Z0-9_]+_(?:USERNAME|PASSWORD)")
_GYM_SHUTDOWN_GRACE_SECONDS = 30.0
_GYM_LOG_DRAIN_SECONDS = 5.0


class GymLaunchError(OSError):
    """``gym eval run`` could not be started."""


class GymCleanupError(RuntimeError):
    """``gym eval run`` could not be confirmed stopped, so its servers or sandboxes may still be running."""


_GYM_EARLIER_RUN_NOT_STOPPED = "an earlier NeMo Gym run could not be confirmed stopped"


@contextlib.contextmanager
def _signals_raise_keyboard_interrupt() -> Any:
    """While Gym runs, turn SIGTERM/SIGHUP into one KeyboardInterrupt so Gym's cleanup still runs."""
    import threading

    names = [name for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)]
    if threading.current_thread() is not threading.main_thread() or not names:
        yield
        return
    raised = False

    def _handler(_signum: int, _frame: object) -> None:
        nonlocal raised
        if not raised:
            raised = True
            raise KeyboardInterrupt

    previous = {name: signal.signal(getattr(signal, name), _handler) for name in names}
    try:
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


def _run_gym(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Any,
    timeout_s: float,
    secret_values: set[str] | None = None,
    reporter: ProgressReporter | None = None,
) -> int | None:
    """Run ``gym eval run`` in its own process group; return its exit code, or None on timeout.

    Gym's output reaches ``log`` through the secret redactor. On timeout, Ctrl-C, SIGTERM or
    SIGHUP the group gets SIGTERM and a grace period, so Gym's own shutdown stops its servers
    and removes its sandboxes, then always SIGKILL. A second Ctrl-C skips the rest of the grace.
    If that stop fails, one last bounded SIGKILL and reap follows; a group still not confirmed
    stopped raises ``GymCleanupError``, so the caller never starts another launch beside it.
    """
    import codecs
    import threading

    from skillevaluator.tier3.harbor.runner import (
        _HARBOR_RUN_CANCEL_POLL_SECONDS,
        _HARBOR_RUN_REAP_SECONDS,
        _terminate_harbor_process_tree,
    )
    from skillevaluator.tier3.harbor.stream_redaction import StreamingLogRedactor

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise GymLaunchError(str(exc)) from exc
    redactor = StreamingLogRedactor(secret_values or set())

    def _pump() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream = process.stdout
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            for chunk in iter(lambda: stream.read1(65536), b""):
                log.write(redactor.feed(decoder.decode(chunk)))
                log.flush()
            log.write(redactor.feed(decoder.decode(b"", final=True)) + redactor.finish())
            log.flush()

    reader = threading.Thread(target=_pump, name="nemo-gym-log", daemon=True)
    reader.start()

    def _kill_group() -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)

    def _group_is_gone() -> bool:
        # Signal 0 finds no member once the group is empty; killed members get a moment to be reaped.
        deadline = time.monotonic() + _HARBOR_RUN_REAP_SECONDS
        while True:
            # As PID 1 or a subreaper we adopt Gym's orphans, and on Linux signal 0 still finds a zombie,
            # so reap our own dead members first. The leader is already reaped, so Popen keeps its status.
            with contextlib.suppress(ChildProcessError):
                while os.waitpid(-process.pid, os.WNOHANG)[0]:
                    pass
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(_HARBOR_RUN_CANCEL_POLL_SECONDS)

    def _confirm_stopped(cause: Exception) -> None:
        # The normal stop failed: one last bounded SIGKILL and reap before giving up on the group.
        _kill_group()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_HARBOR_RUN_REAP_SECONDS)
        if process.poll() is None or not _group_is_gone():
            raise GymCleanupError(
                f"NeMo Gym (process group {process.pid}) could not be confirmed stopped: {cause}. "
                "Its servers or sandboxes may still be running; stop them before another run."
            ) from cause

    def _stop() -> None:
        # A second interrupt anywhere in here still SIGKILLs the group before it propagates.
        try:
            if reporter is not None:
                reporter.emit(
                    ProgressEvent(
                        stage="cleanup",
                        state="running",
                        detail="Stopping NeMo Gym; press Ctrl+C again to skip up to 30s of sandbox cleanup",
                    )
                )
            # Let a leader that is already exiting be reaped, so macOS reports ESRCH rather than EPERM.
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_HARBOR_RUN_CANCEL_POLL_SECONDS)
            try:
                try:
                    _terminate_harbor_process_tree(process, grace_seconds=_GYM_SHUTDOWN_GRACE_SECONDS)
                except PermissionError:
                    # macOS EPERM race: the leader was exiting while its group was signalled.
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=_HARBOR_RUN_CANCEL_POLL_SECONDS)
                    if process.poll() is None:
                        raise
                    _kill_group()
            except (OSError, RuntimeError) as exc:
                _confirm_stopped(exc)
        except KeyboardInterrupt:
            _kill_group()
            process.wait()
            raise

    try:
        with _signals_raise_keyboard_interrupt():
            try:
                return process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _stop()
                return None
            except BaseException as interrupt:
                # A failed stop during Ctrl-C/SIGTERM must not replace the interrupt and let the next arm launch.
                try:
                    _stop()
                except Exception as stop_error:
                    if isinstance(stop_error, GymCleanupError) and reporter is not None:
                        # The interrupt still ends the run; the progress output says Gym may be running.
                        with contextlib.suppress(Exception):
                            reporter.emit(ProgressEvent(stage="cleanup", state="failed", detail=str(stop_error)))
                    raise interrupt from stop_error
                raise
    finally:
        reader.join(timeout=_GYM_LOG_DRAIN_SECONDS)


def _redacted_row(row: dict[str, Any], secrets: set[str]) -> dict[str, Any]:
    """Redact the failure text a rollout row carries before it reaches trial and result files."""
    redacted = dict(row)
    for name in ("failure", "failure_reason", "failure_kind"):
        value = redacted.get(name)
        if isinstance(value, dict):
            redacted[name] = {
                key: redact_progress_detail(item, secret_values=secrets) if isinstance(item, str) else item
                for key, item in value.items()
            }
        elif isinstance(value, str):
            redacted[name] = redact_progress_detail(value, secret_values=secrets)
    return redacted


def gym_launch_environment(credentials: dict[str, str]) -> dict[str, str]:
    """Return Gym's process environment: the host variables a Harbor backend may receive, plus its credentials.

    Relative host paths (Docker TLS, CA bundles, ...) are anchored to the caller's directory,
    because Gym runs in the run directory.
    """
    from skillevaluator.tier3.harbor.runner import (
        _HARBOR_BASE_ENV_VARS,
        _HARBOR_ENV_MODE_VARS,
        _HARBOR_HOST_PATH_ENV_VARS,
        _HARBOR_HOST_PATH_LIST_ENV_VARS,
        _absolute_host_path,
    )

    allowed = _HARBOR_BASE_ENV_VARS | _HARBOR_ENV_MODE_VARS["docker"] | _GYM_HOST_ENV_VARS
    host_env = {
        name: value for name, value in os.environ.items() if name in allowed or _UV_INDEX_CREDENTIAL_ENV.fullmatch(name)
    }
    for name in _HARBOR_HOST_PATH_ENV_VARS & host_env.keys():
        host_env[name] = _absolute_host_path(host_env[name])
    for name in _HARBOR_HOST_PATH_LIST_ENV_VARS & host_env.keys():
        host_env[name] = os.pathsep.join(_absolute_host_path(entry) for entry in host_env[name].split(os.pathsep))
    return {**host_env, **credentials}


def _refresh_adc_credentials(
    credentials: dict[str, str], *, policy_uses_provider_key: bool, timeout_s: float
) -> tuple[dict[str, str], float]:
    """Mint a fresh Vertex ADC token for one Gym launch and cap the launch to its lifetime.

    Mirrors the Harbor runner: the token is refreshed before every launch, reused as the
    policy key when the policy has no key of its own, and the launch timeout is
    capped by ``compute_adc_job_timeout``. Raises ``RuntimeError``/``ValueError`` on failure.
    """
    from skillevaluator.provider_config import (
        CREDENTIAL_EXPIRY_ENV,
        CREDENTIAL_SOURCE_ADC,
        CREDENTIAL_SOURCE_ENV,
        _get_google_access_token,
        _is_vertex_openapi_endpoint,
        _normalize_expiry_epoch,
        compute_adc_job_timeout,
        get_adc_token_expiry,
        refresh_host_vertex_adc_environment,
    )
    from skillevaluator.tier3.harbor.runner import VERTEX_ADC_JOB_TIMEOUT_ENV

    refreshed = dict(credentials)
    fresh = refresh_host_vertex_adc_environment(
        refreshed, fallback_env=os.environ, fail_on_expired=True, token_getter=_get_google_access_token
    )
    if fresh:
        refreshed["OPENAI_API_KEY"] = fresh
        if policy_uses_provider_key:
            refreshed[MODEL_API_KEY_ENV] = fresh
    source = refreshed.get(CREDENTIAL_SOURCE_ENV) or os.environ.get(CREDENTIAL_SOURCE_ENV)
    base_url = refreshed.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if source == CREDENTIAL_SOURCE_ADC and _is_vertex_openapi_endpoint(base_url):
        expiry = get_adc_token_expiry(refreshed.get("OPENAI_API_KEY")) or _normalize_expiry_epoch(
            refreshed.get(CREDENTIAL_EXPIRY_ENV) or os.environ.get(CREDENTIAL_EXPIRY_ENV)
        )
        raw_timeout = os.environ.get(VERTEX_ADC_JOB_TIMEOUT_ENV, "").strip()
        timeout_s = min(
            timeout_s, compute_adc_job_timeout(expiry_epoch=expiry, configured_timeout_sec=raw_timeout or None)
        )
    return refreshed, timeout_s


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def run_nemo_gym_eval(
    *,
    skill_path: Path,
    agents: list[str],
    skip_baseline: bool,
    n_attempts: int | None,
    pass_threshold: float | None,
    n_concurrent: int | None,
    model: str | None,
    agent_models: dict[str, str] | None,
    skill_workspace_mode: str | None,
    include_skills: list[Path] | None,
    grading_mode: str | None,
    output_dir: Path,
    env_mode: str,
    timeout_multiplier: float | None,
    evaluated_source: dict[str, str] | None,
    override_cpus: int | None,
    override_memory_mb: int | None,
    repo_context_exclude_paths: tuple[Path, ...] = (),
    progress_reporter: ProgressReporter | None = None,
    skip_with_skill: bool = False,
    max_cases: int | None = None,
    max_agents: int | None = None,
    stop_on_pass: bool | None = None,
    agent_runtime_preflight: bool | None = None,
    custom_dockerfile_mode: str | None = None,
    copy_repo: bool = False,
    tolerate_baseline_timeouts: bool | None = None,
) -> dict[str, Any]:
    """Run the selected arms on NeMo Gym and return the shared results mapping."""
    from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
    from skillevaluator.publication_identity import publication_target_from_path
    from skillevaluator.source_identity import normalized_evaluated_source
    from skillevaluator.tier3.case_ids import validate_case_ids
    from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
    from skillevaluator.tier3.harbor.adapter import (
        build_eval_base_image,
        find_evals_file,
        generate_harbor_tasks,
        stage_native_harbor_tasks,
    )
    from skillevaluator.tier3.harbor.artifact_retention import HarborArtifactLifecycle
    from skillevaluator.tier3.harbor.collector import _score_definition_for_policy, collect_harbor_results
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, score_policy_for_metrics
    from skillevaluator.tier3.harbor.progress import Tier3RunPlan
    from skillevaluator.tier3.harbor.report_data import dataset_snapshot_manifest
    from skillevaluator.tier3.harbor.runner import (
        _MISSING_EVALS_DATASET_ERROR,
        _VERIFIER_JUDGE_CONTROL_ENV_VARS,
        _VERIFIER_JUDGE_MODEL_ENV_VARS,
        _apply_retention_outcome,
        _decoy_workspace_skills,
        _judge_model_config,
        _limit_staged_cases,
        _merge_launch_execution_errors,
        _native_entry_id,
        _persist_dataset_truth,
        _private_decoy_workspace_snapshot,
        _provider_environment,
        _reserve_run_dir,
        _resolve_runtime_env,
        _runtime_context_is_bound_to_publication_target,
        _validated_pass_threshold,
        _workspace_skills,
        _write_final_result,
    )
    from skillevaluator.tier3.results_location import publish_latest_results

    started_at = time.monotonic()
    publication_target = publication_target_from_path(skill_path)
    reporter = progress_reporter or NullProgressReporter()

    def _config_failure(detail: str) -> dict[str, Any]:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail=detail))
        return {"error": [detail]}

    gym = nemo_gym_bin()
    if gym is None:
        raise ValueError(
            f"--backend nemo-gym needs a verified, pinned Gym task-folder CLI: set {NEMO_GYM_BIN_ENV} "
            "to its `gym` executable; public nemo-gym 0.7.0 is incompatible and the candidate source still needs runtime qualification"
        )
    if skip_baseline and skip_with_skill:
        return _config_failure("skip_baseline and skip_with_skill are mutually exclusive (no arm would run)")
    if cli_error := nemo_gym_cli_error(gym):
        return _config_failure(cli_error)
    try:
        provider = resolve_llm_provider()
        config, config_path = load_evals_config(skill_path)
    except (ProviderConfigurationError, EvalsConfigError) as exc:
        return _config_failure(str(exc))

    # Unset options resolve from evals/config.yml exactly as on Harbor; anything Gym cannot honor is rejected.
    harbor_config = config.get("harbor") or {}
    grading_config = config.get("grading") or {}
    workspace_config = config.get("skill_workspace") or {}
    attempts = n_attempts if n_attempts is not None else harbor_config.get("n_attempts", 1)
    concurrency = n_concurrent if n_concurrent is not None else harbor_config.get("n_concurrent", 4)
    max_cases = max_cases if max_cases is not None else harbor_config.get("max_cases")
    effective_grading = grading_mode or grading_config.get("mode", "default")
    workspace_mode = skill_workspace_mode or workspace_config.get("mode", "isolated")
    effective_timeout = (
        timeout_multiplier if timeout_multiplier is not None else harbor_config.get("timeout_multiplier", 1.0)
    )
    errors = nemo_gym_capability_errors(
        agents=agents,
        env_mode=env_mode,
        grading_mode=effective_grading,
        stop_on_pass=bool(stop_on_pass if stop_on_pass is not None else harbor_config.get("stop_on_pass", False)),
        custom_dockerfile_mode=custom_dockerfile_mode or harbor_config.get("custom_dockerfile_mode"),
        agent_runtime_preflight=bool(
            agent_runtime_preflight
            if agent_runtime_preflight is not None
            else harbor_config.get("agent_runtime_preflight", False)
        ),
    )
    # A flag overrides its config key, as on Harbor.
    config_options = dict(harbor_config)
    if tolerate_baseline_timeouts is not None:
        config_options.pop("tolerate_baseline_timeouts", None)
    errors += nemo_gym_harbor_only_option_errors(config_options, from_config=True)
    errors += nemo_gym_harbor_only_option_errors({"tolerate_baseline_timeouts": tolerate_baseline_timeouts})
    # Gym runs agents one at a time, so the cap always holds; it is still validated as on Harbor.
    effective_max_agents = max_agents if max_agents is not None else harbor_config.get("max_agents", 1)
    if isinstance(effective_max_agents, bool) or not isinstance(effective_max_agents, int) or effective_max_agents < 1:
        errors.append("max_agents must be >= 1")
    from skillevaluator.tier3.harbor.runner import _validated_timeout_multiplier

    try:
        effective_timeout = _validated_timeout_multiplier(effective_timeout)
    except ValueError as exc:
        errors.append(str(exc))
    if not _is_positive_int(attempts):
        errors.append("n_attempts must be >= 1")
    if not _is_positive_int(concurrency):
        errors.append("n_concurrent must be >= 1")
    if max_cases is not None and not _is_positive_int(max_cases):
        errors.append("max_cases must be >= 1")
    configured_task_source = harbor_config.get("task_source", "auto")
    if configured_task_source not in ("auto", "evals_json", "native_harbor"):
        errors.append("harbor.task_source must be auto, evals_json, or native_harbor")
    task_source = "native_harbor" if uses_native_harbor_tasks(skill_path, configured_task_source) else "evals_json"
    if task_source == "native_harbor" and not (skill_path / "evals" / "harbor").is_dir():
        errors.append("No native Harbor task source found at evals/harbor.")
    if task_source == "evals_json" and find_evals_file(skill_path) is None:
        errors.append(_MISSING_EVALS_DATASET_ERROR)
    if mcp_servers_declared(skill_path):
        errors.append(MCP_SERVERS_ERROR)
    if task_source == "native_harbor" and native_mcp_servers_declared(skill_path):
        errors.append(NATIVE_MCP_SERVERS_ERROR)
    pre_agent_setup = list(harbor_config.get("pre_agent_setup") or [])
    if pre_agent_setup and not skip_baseline:
        if skip_with_skill:
            errors.append(
                "harbor.pre_agent_setup/setup_commands cannot run in a baseline-only run (--skip-with-skill); "
                "the baseline arm must stay isolated from skill-owned setup commands"
            )
        else:
            errors.append(
                "harbor.pre_agent_setup/setup_commands cannot run in a paired evaluation; "
                "use --skip-baseline for a with-skill-only run"
            )
    agent_workdir = harbor_config.get("agent_workdir")
    if agent_workdir is not None:
        from skillevaluator.tier3.harbor.adapter import _validated_agent_workdir

        try:
            agent_workdir = _validated_agent_workdir(agent_workdir)
        except ValueError as exc:
            errors.append(f"harbor.{exc}")
    resource_config = dict(harbor_config.get("resources") or {})
    if "storage_mb" in resource_config:
        errors.append("harbor.resources.storage_mb is not supported with --backend nemo-gym")
    include_values = [*workspace_config.get("include", []), *(include_skills or [])]
    if include_values and workspace_mode != "group":
        errors.append("include_skills requires skill_workspace.mode=group")
    for name, value in (("cpus", override_cpus), ("memory_mb", override_memory_mb)):
        if value is not None and value <= 0:
            errors.append(f"--override-{name.replace('_', '-')} must be greater than 0")
    timeout_s = DEFAULT_RUN_TIMEOUT_S
    try:
        timeout_s = nemo_gym_timeout_s()
    except ValueError as exc:
        errors.append(str(exc))
    if sandbox_error := nemo_gym_sandbox_error():
        errors.append(sandbox_error)
    if errors:
        return _config_failure("; ".join(errors))
    try:
        threshold = _validated_pass_threshold(
            pass_threshold if pass_threshold is not None else harbor_config.get("pass_threshold", 0.5)
        )
        decoy_manifest: dict[str, Any] | None = None
        selected_decoys: list[Path] = []
        if "decoy_manifest" in workspace_config:
            workspace_skills, selected_decoys, decoy_manifest = _decoy_workspace_skills(
                skill_path, workspace_config, workspace_mode, include_skills
            )
        else:
            workspace_skills = _workspace_skills(skill_path, include_values if workspace_mode == "group" else [])
        workspace_source_paths = [*workspace_skills, *selected_decoys]
    except ValueError as exc:
        return _config_failure(str(exc))

    gym_version = nemo_gym_version(gym)
    sandbox = os.environ.get(SANDBOX_ENV, "docker").strip() or "docker"
    execution_backend = {"name": NEMO_GYM_BACKEND, "version": gym_version, "sandbox": sandbox}
    resolved_models: dict[str, dict[str, str]] = {}
    endpoints: dict[str, tuple[str, str, str]] = {}
    # _model_endpoint falls back to the provider key exactly when the policy has none of its own.
    policy_uses_provider_key = not os.environ.get(MODEL_API_KEY_ENV, "").strip()
    config_agents = harbor_config.get("agents") or {}
    for agent in agents:
        cli_model = (agent_models or {}).get(agent) or model
        config_model = (config_agents.get(agent) or {}).get("model")
        endpoints[agent] = _model_endpoint(cli_model or config_model)
        source = "CLI" if cli_model else "evals/config.yml" if config_model else "public provider default"
        resolved_models[agent] = {"agent": agent, "model": endpoints[agent][0], "source": source}
        if url_error := model_url_credential_error(endpoints[agent][1]):
            return _config_failure(url_error)

    # Judge credentials reach only the verifier: tasks carry ${NAME} templates that the
    # Gym environment server resolves from its own process environment. Resolved before the
    # run folder is reserved, so a bad harbor.runtime_env leaves nothing behind, as on Harbor.
    try:
        judge_verifier_env, judge_process_env = verifier_judge_env(skill_path, effective_grading, env_mode)
    except ValueError as exc:
        return _config_failure(str(exc))
    reporter.set_secret_values(set(judge_process_env.values()))
    # harbor.runtime_env reaches the agent as ${NAME} templates, resolved from Gym's environment.
    runtime_values, _runtime_errors = _resolve_runtime_env(harbor_config.get("runtime_env"), env_mode=env_mode)
    agent_runtime_env = {name: f"${{{name}}}" for name in runtime_values}

    run_dir = _reserve_run_dir(output_dir, datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    # As on Harbor, the results root never enters a --copy-repo or base-image build context.
    repo_context_exclude_paths = (Path(output_dir), *(Path(path) for path in repo_context_exclude_paths))
    tasks_root = run_dir / "_harbor-tasks"
    jobs_dir = run_dir / "_harbor-jobs"
    gym_root = run_dir / "_nemo-gym"
    # Staged tasks and raw trial files can hold trajectories with secrets: removed as on Harbor
    # without --harbor-keep-jobs. Gym's own launch logs and rollouts stay in _nemo-gym/.
    with HarborArtifactLifecycle([jobs_dir, tasks_root], keep_requested=False) as lifecycle:
        gym_root.mkdir(parents=True, exist_ok=True)
        arms = [
            (arm, with_skill)
            for arm, with_skill, skipped in (("with", True, skip_with_skill), ("without", False, skip_baseline))
            if not skipped
        ]
        task_resources = {
            **resource_config,
            **{
                key: value
                for key, value in (("cpus", override_cpus), ("memory_mb", override_memory_mb))
                if value is not None
            },
        }
        reporter.emit(ProgressEvent(stage="configuration", state="ready", detail="evaluation config validated"))
        judge_job_env = {
            name: value
            for name, value in judge_verifier_env.items()
            if name in _VERIFIER_JUDGE_CONTROL_ENV_VARS | _VERIFIER_JUDGE_MODEL_ENV_VARS
        }
        launch_errors: list[str] = []
        case_ids: list[str] = []
        case_selection: dict[str, Any] | None = None
        # Stage every agent and arm before the first launch, so all arms see the same skill inputs.
        planned: list[tuple[str, str, Path, dict[str, str]]] = []
        staged_task_paths: list[Path] = []
        base_image_mode = harbor_config.get("base_image_mode", "disabled")
        base_image = ""
        if base_image_mode != "disabled":
            reporter.emit(
                ProgressEvent(
                    stage="docker-images",
                    state="running",
                    detail=f"preparing shared eval base image ({base_image_mode})",
                )
            )
            base_image = build_eval_base_image(
                skill_path.resolve(),
                None,
                workspace_skill_paths=workspace_source_paths,
                excluded_roots=repo_context_exclude_paths,
                force_rebuild=base_image_mode == "rebuild",
            )
            reporter.emit(
                ProgressEvent(stage="docker-images", state="complete", detail=f"eval base image ready: {base_image}")
                if base_image
                else ProgressEvent(
                    stage="docker-images",
                    state="degraded",
                    detail="base image build failed; falling back to per-task Dockerfiles",
                )
            )
        # Every arm stages the same verified decoy bytes, as on Harbor.
        with (
            _private_decoy_workspace_snapshot(selected_decoys, decoy_manifest)
            if decoy_manifest is not None
            else contextlib.nullcontext([])
        ) as frozen_decoys:
            staged_workspace_skills = [*workspace_skills, *frozen_decoys]
            for agent in agents:
                for arm, with_skill in arms:
                    arm_dir = tasks_root / agent / arm
                    emit_tasks = stage_native_harbor_tasks if task_source == "native_harbor" else generate_harbor_tasks
                    task_paths = emit_tasks(
                        skill_path,
                        arm_dir,
                        with_skill=with_skill,
                        workspace_skill_paths=staged_workspace_skills,
                        workspace_mode=workspace_mode,
                        grading_mode=effective_grading,
                        verifier_env=judge_verifier_env or None,
                        repo_context_exclude_paths=repo_context_exclude_paths,
                        task_resources=task_resources or None,
                        runtime_env=agent_runtime_env or None,
                        arm_suffix=f"-{arm}-skill" if len(arms) == 2 else "",
                        pre_agent_setup=pre_agent_setup or None,
                        agent_workdir=agent_workdir,
                        copy_repo=copy_repo,
                        base_image=base_image,
                        custom_dockerfile_mode=custom_dockerfile_mode
                        or harbor_config.get("custom_dockerfile_mode", "rebase"),
                    )
                    if max_cases is not None:
                        selectors = validate_case_ids(task.name for task in task_paths)
                        selector_case_ids = validate_case_ids(_native_entry_id(task) for task in task_paths)
                        case_selection = _limit_staged_cases(
                            selectors,
                            dict(zip(selectors, selector_case_ids, strict=True)),
                            task_paths,
                            max_cases=max_cases,
                        )
                    stage_for_gym(
                        arm_dir,
                        judge_job_env,
                        timeout_multiplier=effective_timeout,
                        # Generated tasks already carry them; native tasks get Harbor's job-level override.
                        resource_overrides={
                            key: value
                            for key, value in (("cpus", override_cpus), ("memory_mb", override_memory_mb))
                            if value is not None and task_source == "native_harbor"
                        },
                    )
                    task_case_ids = _task_case_ids(arm_dir)
                    case_ids = case_ids or sorted(set(task_case_ids.values()))
                    staged_task_paths += [task for task in task_paths if task.is_dir()]
                    planned.append((agent, arm, arm_dir, task_case_ids))
        # The same plan a Harbor run starts its reporter with; output_dir lets a case view poll the rollouts.
        reporter.start(
            Tier3RunPlan(
                skill_name=skill_path.name,
                environment=env_mode,
                agents=tuple(agents),
                agent_models=tuple((agent, endpoints[agent][0]) for agent in agents),
                provider=provider.provider,
                case_count=len(case_ids),
                attempts=attempts,
                baseline=not skip_baseline,
                with_skill=not skip_with_skill,
                concurrency=concurrency,
                timeout_multiplier=float(effective_timeout),
                matrix_trials=len(case_ids) * attempts * len(planned),
                output_dir=str(run_dir),
            )
        )
        runtime_context_bound = _runtime_context_is_bound_to_publication_target(
            skill_path,
            staged_task_paths,
            reference_skills_dir=None,
            workspace_skill_paths=workspace_source_paths,
        )
        # The same immutable dataset identity a Harbor run records, taken before any agent runs.
        dataset_warning: str | None = None
        try:
            dataset_truth = _persist_dataset_truth(run_dir, fallback_task_ids=case_ids)
        except Exception as exc:
            dataset_truth = None
            dataset_warning = f"Dataset snapshot was not recorded: {type(exc).__name__}"

        # Set once a Gym group cannot be confirmed stopped. Nothing else is launched beside it, but the
        # arms that finished are still collected, so the failed run keeps their scores.
        cleanup_error: GymCleanupError | None = None
        for agent in agents:
            reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="running", detail="NeMo Gym"))
            agent_model, model_url, api_key = endpoints[agent]
            errors_before = len(launch_errors)
            blocked_before_agent = cleanup_error is not None
            for planned_agent, arm, arm_dir, task_case_ids in planned:
                if planned_agent != agent:
                    continue
                arm_errors: list[str] = []
                secrets = {value for value in judge_process_env.values() if len(value) >= MIN_EXACT_SECRET_CHARS}
                rollouts_path = gym_root / f"{agent}-{arm}" / "rollouts.jsonl"
                log_path = rollouts_path.with_name("gym.log")
                if cleanup_error is not None:
                    arm_errors.append(f"gym eval run was not started: {_GYM_EARLIER_RUN_NOT_STOPPED}")
                else:
                    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        credentials, arm_timeout = _refresh_adc_credentials(
                            {**judge_process_env, MODEL_API_KEY_ENV: api_key},
                            policy_uses_provider_key=policy_uses_provider_key,
                            timeout_s=timeout_s,
                        )
                    except (RuntimeError, ValueError) as exc:
                        arm_errors.append(f"Vertex AI ADC credentials could not be refreshed: {exc}")
                    else:
                        launch_env = gym_launch_environment(credentials)
                        secrets |= {
                            value
                            for value in credentials.values()
                            if len(value) >= MIN_EXACT_SECRET_CHARS and value != "EMPTY"
                        }
                        secrets |= secret_values_from_environment(launch_env)
                        reporter.set_secret_values(secrets)
                        command = gym_eval_command(
                            gym,
                            arm_dir,
                            agent=agent,
                            model=agent_model,
                            model_url=model_url,
                            rollouts_path=rollouts_path,
                            n_attempts=attempts,
                            n_concurrent=concurrency,
                            sandbox=sandbox,
                            inherit_sandbox_environment=agent == "codex" and sandbox == "docker",
                        )
                        reporter.emit(
                            ProgressEvent(stage=f"agent:{agent}", state="running", detail=f"NeMo Gym {arm}-skill arm")
                        )
                        with log_path.open("w", encoding="utf-8") as log:
                            try:
                                returncode = _run_gym(
                                    command,
                                    cwd=rollouts_path.parent,
                                    env=launch_env,
                                    log=log,
                                    timeout_s=arm_timeout,
                                    secret_values=secrets,
                                    reporter=reporter,
                                )
                            except GymLaunchError as exc:
                                arm_errors.append(f"gym eval run could not start: {exc}")
                            except GymCleanupError as exc:
                                # Only the timeout path raises it; an interrupt re-raises KeyboardInterrupt.
                                cleanup_error = exc
                                arm_errors += [f"gym eval run timed out after {arm_timeout:.0f}s", str(exc)]
                            else:
                                if returncode is None:
                                    arm_errors.append(f"gym eval run timed out after {arm_timeout:.0f}s")
                                elif returncode != 0:
                                    arm_errors.append(f"gym eval run exited {returncode}; see {log_path}")
                rows = [_redacted_row(row, secrets) for row in read_rollouts(rollouts_path, arm_errors)]
                failures = {
                    task: [redact_progress_detail(reason, secret_values=secrets) for reason in reasons]
                    for task, reasons in read_rollout_failures(rollouts_path, arm_errors).items()
                }
                arm_errors += write_job_from_rollouts(
                    rows,
                    jobs_dir / f"{skill_path.name}-{agent}-{arm}",
                    eval_name=f"{agent}__{agent_model}",
                    task_case_ids=task_case_ids,
                    n_attempts=attempts,
                    failures=failures,
                )
                launch_errors += [
                    f"{agent} {arm}-skill: {redact_progress_detail(error, secret_values=secrets)}"
                    for error in arm_errors
                ]
            if cleanup_error is not None:
                # The row says that Gym may still be running, or why this agent never started.
                detail = f"not started: {_GYM_EARLIER_RUN_NOT_STOPPED}" if blocked_before_agent else str(cleanup_error)
                reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="failed", detail=detail))
            else:
                reporter.emit(
                    ProgressEvent(
                        stage=f"agent:{agent}",
                        state="failed",
                        detail=f"one or more NeMo Gym runs failed; see {gym_root}/{agent}-*/gym.log",
                    )
                    if len(launch_errors) > errors_before
                    else ProgressEvent(stage=f"agent:{agent}", state="complete")
                )

        reporter.emit(ProgressEvent(stage="collection", state="running"))
        results = collect_harbor_results(
            skill_name=skill_path.name,
            agents=agents,
            output_dir=run_dir,
            jobs_dir=jobs_dir,
            skip_baseline=skip_baseline,
            skip_with_skill=skip_with_skill,
            n_attempts=attempts,
            pass_threshold=threshold,
            expected_cases=len(case_ids),
            expected_case_ids=case_ids,
            expected_trials=len(case_ids) * attempts,
            env_mode=env_mode,
            agent_models=resolved_models,
            launch_errors=launch_errors,
            grading_mode=effective_grading,
        )
        if launch_errors:
            _merge_launch_execution_errors(results, launch_errors)
        if cleanup_error is not None:
            # Also a warning: the summary shows warnings even when it folds failure details behind --verbose.
            results.setdefault("warnings", []).append(str(cleanup_error))
        reporter.emit(ProgressEvent(stage="collection", state="complete", detail="NeMo Gym results collected"))

        # The same completion contract as a Harbor run, so result discovery (`view`, compare) finds Gym runs.
        try:
            config_file = str(config_path.relative_to(skill_path)) if config_path else "none"
        except ValueError:
            config_file = config_path.name if config_path else "none"
        run_config: dict[str, Any] = {
            "config_file": config_file,
            "harbor": {
                "environment": {"value": env_mode, "source": "CLI"},
                "environment_kwargs": {"keys": [], "sources": {}},
                "n_attempts": attempts,
                "stop_on_pass": False,
                "n_concurrent": concurrency,
                "timeout_multiplier": effective_timeout,
                "base_image_mode": base_image_mode,
                "custom_dockerfile_mode": custom_dockerfile_mode
                or harbor_config.get("custom_dockerfile_mode", "rebase"),
                "jobs_retained": False,
            },
            "provider": {"name": provider.provider, "model": provider.model},
            "judge": _judge_model_config(provider, _provider_environment(provider), effective_grading),
            "task_source": task_source,
            "grading": {"mode": effective_grading},
            "agents": resolved_models,
            "evaluated_source": normalized_evaluated_source(evaluated_source),
            "execution_backend": execution_backend,
        }
        if max_cases is not None:
            run_config["harbor"]["max_cases"] = max_cases
        if case_selection is not None:
            run_config["case_selection"] = case_selection
        if decoy_manifest is not None:
            run_config["skill_workspace"] = {"mode": workspace_mode, "decoy_manifest": decoy_manifest}
        selected_metrics = tuple(results.get("metrics", DEFAULT_METRICS))
        attempt_policy = dict(results.get("attempt_policy") or {})
        score_policy = results.get("score_policy") or attempt_policy.get("score_policy")
        if not isinstance(score_policy, str) or not score_policy.strip():
            score_policy = score_policy_for_metrics(selected_metrics)
        attempt_policy.update(
            {
                "max_attempts": attempts,
                "pass_threshold": float(threshold),
                "stop_on_pass": False,
                "score_policy": score_policy,
                "score_definition": attempt_policy.get("score_definition")
                or _score_definition_for_policy(selected_metrics, score_policy),
            }
        )
        if dataset_warning:
            results.setdefault("warnings", []).append(dataset_warning)
        if dataset_truth is not None:
            results.update(
                {
                    "evaluated_at": datetime.now(UTC).isoformat(),
                    "evaluator_version": dataset_truth["evaluator_version"],
                    "dataset_snapshot": dataset_snapshot_manifest(dataset_truth),
                    "dataset_snapshot_path": str(run_dir / "dataset_snapshot.json"),
                    "dataset_summary": dataset_truth["dataset_summary"],
                    "dataset_digest": dataset_truth["dataset_digest"],
                    "dataset_digest_algorithm": dataset_truth["dataset_digest_algorithm"],
                }
            )
        run_end_target = publication_target_from_path(skill_path) if runtime_context_bound else None
        if not runtime_context_bound:
            # Included skills or repo context the digest does not cover were staged into the tasks.
            results["publication_target"] = None
            results["publication_target_conflict"] = {
                "run_start": dict(publication_target) if publication_target is not None else None,
                "run_end": None,
                "reason_code": "runtime_context_outside_publication_target",
            }
        elif run_end_target == publication_target:
            results["publication_target"] = dict(publication_target) if publication_target is not None else None
        else:
            results["publication_target"] = None
            results["publication_target_conflict"] = {
                "run_start": dict(publication_target) if publication_target is not None else None,
                "run_end": dict(run_end_target) if run_end_target is not None else None,
            }
        result_path = run_dir / "result.json"
        results.update(
            {
                "skill_name": skill_path.name,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "result_path": str(result_path),
                "execution_backend": execution_backend,
                "nemo_gym_dir": str(gym_root),
                "attempt_policy": attempt_policy,
                "run_config": run_config,
            }
        )
        if evaluated_source:
            results["evaluated_source"] = dict(evaluated_source)
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        try:
            results["report_path"] = str(
                render_agent_eval_html_report(skill_path, run_dir, env_mode=env_mode, engine_result=results)
            )
            results["report_status"] = "complete"
        except Exception as exc:
            results.setdefault("warnings", []).append(f"HTML report was not generated: {type(exc).__name__}")
            results["report_status"] = "degraded"
        outcome = lifecycle.finalize()
    _apply_retention_outcome(results, outcome=outcome, jobs_dir=jobs_dir)
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    results["duration_seconds"] = round(time.monotonic() - started_at, 3)
    final = _write_final_result(result_path, results)
    publish_latest_results(output_dir, run_dir.name)
    return final
