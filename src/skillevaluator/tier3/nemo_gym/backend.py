# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 execution backend selection and NeMo Gym capability checks."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

HARBOR_BACKEND = "harbor"
NEMO_GYM_BACKEND = "nemo-gym"
EXECUTION_BACKENDS = (HARBOR_BACKEND, NEMO_GYM_BACKEND)
DEFAULT_EXECUTION_BACKEND = HARBOR_BACKEND

NEMO_GYM_BIN_ENV = "SKILL_EVAL_NEMO_GYM_BIN"
MODEL_API_KEY_ENV = "SKILL_EVAL_NEMO_GYM_MODEL_API_KEY"
MODEL_URL_ENV = "SKILL_EVAL_NEMO_GYM_MODEL_URL"
RUN_TIMEOUT_ENV = "SKILL_EVAL_NEMO_GYM_TIMEOUT_S"
DEFAULT_RUN_TIMEOUT_S = 21600.0
_GYM_HELP_TIMEOUT_S = 15.0
_GYM_HELP_MAX_OUTPUT_BYTES = 256 * 1024
NEMO_GYM_AGENTS = {
    "codex": "codex_agent",
    "claude-code": "claude_code_agent",
    "opencode": "opencode_agent",
}
NEMO_GYM_ENV_MODES = frozenset({"docker"})
NEMO_GYM_GRADING_MODES = frozenset({"default", "default_plus_custom", "custom_only"})

_ALIASES = {
    "harbor": HARBOR_BACKEND,
    "nemo-gym": NEMO_GYM_BACKEND,
    "nemo_gym": NEMO_GYM_BACKEND,
    "nemogym": NEMO_GYM_BACKEND,
    "gym": NEMO_GYM_BACKEND,
}


def normalize_backend(value: str | None) -> str:
    """Return the canonical backend name or raise ``ValueError``."""
    if value is None or not str(value).strip():
        return DEFAULT_EXECUTION_BACKEND
    key = str(value).strip().casefold()
    if key not in _ALIASES:
        choices = ", ".join(EXECUTION_BACKENDS)
        raise ValueError(f"Unknown Tier 3 execution backend {value!r}; choose one of: {choices}")
    return _ALIASES[key]


def nemo_gym_bin() -> str | None:
    """Locate the ``gym`` CLI, preferring the pinned worker named by ``SKILL_EVAL_NEMO_GYM_BIN``."""
    configured = os.environ.get(NEMO_GYM_BIN_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser()
        return str(path.absolute()) if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which("gym")
    return str(Path(found).absolute()) if found else None


def _gym_interpreters(gym_bin: str) -> list[Path]:
    """Return candidate interpreters for the Gym install: the script's shebang, then its sibling Python."""
    script = Path(gym_bin).resolve()
    candidates: list[Path] = []
    try:
        with script.open("rb") as handle:
            first_line = handle.read(512).split(b"\n", 1)[0].decode("utf-8", "replace")
    except OSError:
        first_line = ""
    if first_line.startswith("#!"):
        parts = first_line[2:].strip().split()
        if parts and Path(parts[0]).name.startswith("python"):
            candidates.append(Path(parts[0]))
    candidates += [script.with_name("python"), script.with_name("python.exe")]
    return [candidate for candidate in candidates if candidate.is_file()]


def nemo_gym_version(gym_bin: str) -> str | None:
    """Return the installed NeMo Gym version from the worker's own interpreter.

    ``-P`` keeps the caller's directory off ``sys.path``, so a ``nemo_gym`` module in the
    skill under evaluation is never imported, and the probe gets only Gym's launch environment.
    """
    from skillevaluator.tier3.nemo_gym.runner import gym_launch_environment

    for python in _gym_interpreters(gym_bin):
        try:
            completed = subprocess.run(
                [str(python), "-P", "-c", "import nemo_gym; print(nemo_gym.__version__)"],
                cwd=python.parent,
                env=gym_launch_environment({}),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        version = completed.stdout.strip().splitlines()
        if completed.returncode == 0 and version:
            return version[-1]
    return None


# Candidate public source interface; native runtime qualification remains required.
NEMO_GYM_SOURCE = "git+https://github.com/NVIDIA-NeMo/Gym@792fa142f58277aecff05d402061d647e9a92acf"


def nemo_gym_cli_error(gym_bin: str) -> str | None:
    """Check the task-folder CLI contract without starting servers or passing policy credentials.

    A version string alone does not identify this interface: public Gym 0.7.0 uses
    JSONL inputs and lacks the task-folder / sandbox command this adapter requires.
    This necessary interface check does not certify an installation's runtime support.
    """
    from skillevaluator.tier3.harbor.runner import _HarborRunTimeoutError, _run_bounded_harbor_process
    from skillevaluator.tier3.nemo_gym.runner import _signals_raise_keyboard_interrupt, gym_launch_environment

    try:
        # Reuse bounded streaming and isolated process-group cleanup, not Harbor execution.
        # Both output streams share one raw byte budget; no subprocess output enters errors.
        with _signals_raise_keyboard_interrupt():
            completed = _run_bounded_harbor_process(
                [gym_bin, "eval", "run", "--help"],
                cwd=Path(gym_bin).resolve().parent,
                env=gym_launch_environment({}),
                stdin_text=None,
                timeout_seconds=_GYM_HELP_TIMEOUT_S,
                max_output_bytes=_GYM_HELP_MAX_OUTPUT_BYTES,
                diagnostic_tail_chars=_GYM_HELP_MAX_OUTPUT_BYTES,
                secret_values=set(),
            )
    except _HarborRunTimeoutError:
        return (
            f"NeMo Gym CLI compatibility check failed: `gym eval run --help` timed out after {_GYM_HELP_TIMEOUT_S:g}s"
        )
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return "NeMo Gym CLI compatibility check failed: `gym eval run --help` could not complete safely"
    if completed.output_exceeded:
        return "NeMo Gym CLI compatibility check failed: `gym eval run --help` exceeded its combined output limit"
    if completed.returncode:
        return f"NeMo Gym CLI compatibility check failed: `gym eval run --help` exited {completed.returncode}"
    help_text = completed.output_tail
    flags = ("--sandbox", "--agent", "--model-type", "--model", "--model-url", "--output", "--concurrency")
    missing = [flag for flag in flags if not re.search(rf"(?<![\w-]){re.escape(flag)}(?=[\s,=]|$)", help_text)]
    if not _declares_task_positional(help_text):
        missing.append("positional tasks directory")
    if missing:
        return (
            f"NeMo Gym CLI is incompatible with this experimental task-folder adapter; missing: {', '.join(missing)}. "
            f"Public nemo-gym 0.7.0 is incompatible; candidate source {NEMO_GYM_SOURCE} still needs runtime qualification. "
            "No Harbor fallback was started."
        )
    return None


def _declares_task_positional(help_text: str) -> bool:
    """Whether argparse help lists a task-folder positional under ``positional arguments:``.

    Only entries at argparse's two-space item indent count, so wrapped option prose that happens
    to start with ``tasks`` (as ``--only-tasks`` does at 80 columns) is not mistaken for one.
    """
    section = re.search(r"(?ms)^positional arguments:\n(.*?)(?=^\S|\Z)", help_text)
    return bool(
        section and re.search(r"(?m)^ {2}(?:tasks|task_dir|tasks_dir|task_folder|TARGET)(?:\s|$)", section.group(1))
    )


_UNSUPPORTED_RUN_OPTIONS = (
    ("reuse_baseline", "--reuse-baseline"),
    ("compare_to", "--compare-to"),
    ("max_agent_tokens", "--max-agent-tokens"),
    ("max_judge_calls", "--max-judge-calls"),
    ("max_attempts_total", "--max-attempts-total"),
    ("max_cost_usd", "--max-cost-usd"),
    ("price_file", "--price-file"),
)
_UNSUPPORTED_BUDGET_CONFIG = ("max_agent_tokens", "max_judge_calls", "max_attempts_total", "max_cost_usd", "price_file")
# Off is Gym's own behavior, so only an enabled option is rejected.
_UNSUPPORTED_ENABLED_OPTIONS = (("tolerate_baseline_timeouts", "--tolerate-baseline-timeouts"),)


def nemo_gym_capability_errors(
    *,
    agents: list[str],
    env_mode: str,
    grading_mode: str | None,
    harbor_keep_jobs: bool = False,
    stop_on_pass: bool = False,
    custom_dockerfile_mode: str | None = None,
    copy_repo: bool = False,
    environment_kwargs: dict[str, str] | None = None,
    override_storage_mb: int | None = None,
    agent_runtime_preflight: bool = False,
    timeout_multiplier: float | None = None,
    run_options: dict[str, Any] | None = None,
) -> list[str]:
    """Reject configurations the NeMo Gym backend cannot run equivalently, before anything starts."""
    errors = nemo_gym_harbor_only_option_errors(run_options or {})
    if os.name != "posix":
        errors.append("--backend nemo-gym needs a POSIX host (Linux or macOS); NeMo Gym does not run on Windows")
    unsupported = [agent for agent in agents if agent not in NEMO_GYM_AGENTS]
    if unsupported:
        supported = ", ".join(sorted(NEMO_GYM_AGENTS))
        errors.append(f"--backend nemo-gym supports agents {supported}; not supported: {', '.join(unsupported)}")
    if env_mode not in NEMO_GYM_ENV_MODES:
        errors.append(f"--backend nemo-gym supports --env-mode docker only; got {env_mode!r}")
    if grading_mode and grading_mode not in NEMO_GYM_GRADING_MODES:
        errors.append(f"--backend nemo-gym does not support grading mode {grading_mode!r}")
    if harbor_keep_jobs:
        errors.append("--harbor-keep-jobs applies to the Harbor backend only")
    if stop_on_pass:
        errors.append("--stop-on-pass is not supported with --backend nemo-gym yet")
    if custom_dockerfile_mode not in (None, "rebase", "preserve"):
        errors.append(f"unsupported --custom-dockerfile-mode {custom_dockerfile_mode!r}")
    if environment_kwargs:
        errors.append("--environment-kwarg applies to the Harbor environment providers only")
    if override_storage_mb is not None:
        errors.append("--override-storage-mb is not supported with --backend nemo-gym")
    if agent_runtime_preflight:
        errors.append("--agent-runtime-preflight applies to the Harbor backend only")
    if timeout_multiplier is not None:
        from skillevaluator.tier3.harbor.runner import _validated_timeout_multiplier

        try:
            _validated_timeout_multiplier(timeout_multiplier)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def nemo_gym_harbor_only_option_errors(options: Mapping[str, object], *, from_config: bool = False) -> list[str]:
    """Reject baseline reuse, version comparison, budgets and baseline-timeout tolerance until Gym can honor them."""
    errors: list[str] = []
    for name, flag in _UNSUPPORTED_RUN_OPTIONS:
        # Zero is an explicit limit, not an unset option.
        if options.get(name) is not None:
            label = f"harbor.{name}" if from_config else flag
            errors.append(f"{label} is not supported with --backend nemo-gym yet")
    for name, flag in _UNSUPPORTED_ENABLED_OPTIONS:
        if options.get(name):
            label = f"harbor.{name}" if from_config else flag
            errors.append(f"{label} is not supported with --backend nemo-gym yet")
    return errors


def nemo_gym_timeout_s() -> float:
    """Return ``SKILL_EVAL_NEMO_GYM_TIMEOUT_S`` in seconds, or raise ``ValueError``."""
    raw = os.environ.get(RUN_TIMEOUT_ENV, "").strip()
    try:
        timeout_s = float(raw) if raw else DEFAULT_RUN_TIMEOUT_S
    except ValueError:
        timeout_s = math.nan
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{RUN_TIMEOUT_ENV} must be a finite number of seconds greater than 0")
    return timeout_s


def nemo_gym_sandbox_error() -> str | None:
    """Return an error when ``SKILL_EVAL_NEMO_GYM_SANDBOX`` names a sandbox the backend does not support."""
    sandbox = os.environ.get("SKILL_EVAL_NEMO_GYM_SANDBOX", "").strip()
    if sandbox and sandbox not in NEMO_GYM_ENV_MODES:
        return f"SKILL_EVAL_NEMO_GYM_SANDBOX={sandbox} is not supported; NeMo Gym runs Tier 3 cases in docker only"
    return None


def nemo_gym_settings_errors() -> list[str]:
    """Reject invalid ``SKILL_EVAL_NEMO_GYM_*`` settings, before any dataset work."""
    errors: list[str] = []
    if sandbox_error := nemo_gym_sandbox_error():
        errors.append(sandbox_error)
    try:
        nemo_gym_timeout_s()
    except ValueError as exc:
        errors.append(str(exc))
    url = os.environ.get(MODEL_URL_ENV, "").strip()
    if url and not os.environ.get(MODEL_API_KEY_ENV, "").strip():
        errors.append(f"{MODEL_URL_ENV} is set, so also set {MODEL_API_KEY_ENV} (use EMPTY for a keyless server)")
    return errors


def model_url_credential_error(url: str) -> str | None:
    """Return an error when a policy URL carries userinfo, which would land on Gym's command line."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.username or parts.password:
        return (
            "the NeMo Gym policy model URL must not embed a username or password; "
            f"put the credential in {MODEL_API_KEY_ENV} instead"
        )
    return None


def uses_native_harbor_tasks(skill_path: Path, task_source: object) -> bool:
    """Return True when the Tier 3 source resolves to native ``evals/harbor/`` tasks, as on Harbor."""
    if task_source == "native_harbor":
        return True
    if task_source not in (None, "auto"):
        return False
    from skillevaluator.tier3.harbor.adapter import find_evals_file

    try:
        has_evals_file = find_evals_file(skill_path) is not None
    except (OSError, ValueError):
        return False
    return not has_evals_file and (skill_path / "evals" / "harbor").is_dir()


def mcp_servers_declared(skill_path: Path) -> bool:
    """Return True when ``evals/environment/mcp_servers.toml`` declares any MCP server."""
    from skillevaluator.tier3.harbor.adapter import _load_mcp_servers

    if not (skill_path / "evals" / "environment" / "mcp_servers.toml").is_file():
        return False
    try:
        return bool(_load_mcp_servers(skill_path))
    except Exception:
        return True


MCP_SERVERS_ERROR = "evals/environment/mcp_servers.toml is not supported with --backend nemo-gym yet"
NATIVE_MCP_SERVERS_ERROR = (
    "[[environment.mcp_servers]] in a native task.toml is not supported with --backend nemo-gym yet"
)


def native_mcp_servers_declared(skill_path: Path) -> bool:
    """Return True when a native ``evals/harbor/*/task.toml`` declares ``[[environment.mcp_servers]]``."""
    import tomllib

    for task_toml in sorted((skill_path / "evals" / "harbor").glob("*/task.toml")):
        try:
            environment = tomllib.loads(task_toml.read_text(encoding="utf-8")).get("environment") or {}
        except (OSError, ValueError):
            continue
        if isinstance(environment, dict) and environment.get("mcp_servers"):
            return True
    return False


def nemo_gym_config_errors(
    skill_path: Path,
    *,
    stop_on_pass: bool | None = None,
    timeout_multiplier: float | None = None,
    agent_runtime_preflight: bool | None = None,
    custom_dockerfile_mode: str | None = None,
    run_options: dict[str, Any] | None = None,
) -> list[str]:
    """Reject ``evals/config.yml`` settings the NeMo Gym backend cannot honor, before any dataset work.

    A setting also given on the command line is skipped here: the CLI value wins, and
    ``nemo_gym_capability_errors`` already rejects an unsupported CLI value.
    """
    from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config

    try:
        config, _config_path = load_evals_config(skill_path)
    except EvalsConfigError as exc:
        return [str(exc)]
    harbor = config.get("harbor") or {}
    unsupported_config = {
        name: harbor.get(name)
        for name in (*_UNSUPPORTED_BUDGET_CONFIG, *(name for name, _flag in _UNSUPPORTED_ENABLED_OPTIONS))
        if (run_options or {}).get(name) is None
    }
    errors = nemo_gym_harbor_only_option_errors(unsupported_config, from_config=True)
    if stop_on_pass is None and harbor.get("stop_on_pass"):
        errors.append("harbor.stop_on_pass is not supported with --backend nemo-gym yet")
    if agent_runtime_preflight is None and harbor.get("agent_runtime_preflight"):
        errors.append("harbor.agent_runtime_preflight applies to the Harbor backend only")
    if mcp_servers_declared(skill_path):
        errors.append(MCP_SERVERS_ERROR)
    if uses_native_harbor_tasks(skill_path, harbor.get("task_source", "auto")) and native_mcp_servers_declared(
        skill_path
    ):
        errors.append(NATIVE_MCP_SERVERS_ERROR)
    if "storage_mb" in (harbor.get("resources") or {}):
        errors.append("harbor.resources.storage_mb is not supported with --backend nemo-gym")
    return errors
