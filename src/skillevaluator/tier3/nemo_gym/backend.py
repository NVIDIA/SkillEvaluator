# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 execution backend selection and NeMo Gym capability checks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HARBOR_BACKEND = "harbor"
NEMO_GYM_BACKEND = "nemo-gym"
EXECUTION_BACKENDS = (HARBOR_BACKEND, NEMO_GYM_BACKEND)
DEFAULT_EXECUTION_BACKEND = HARBOR_BACKEND

NEMO_GYM_BIN_ENV = "SKILL_EVAL_NEMO_GYM_BIN"
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
        first_line = script.read_bytes()[:512].split(b"\n", 1)[0].decode("utf-8", "replace")
    except OSError:
        first_line = ""
    if first_line.startswith("#!"):
        parts = first_line[2:].strip().split()
        if parts and Path(parts[0]).name.startswith("python"):
            candidates.append(Path(parts[0]))
    candidates += [script.with_name("python"), script.with_name("python.exe")]
    return [candidate for candidate in candidates if candidate.is_file()]


def nemo_gym_version(gym_bin: str) -> str | None:
    """Return the installed NeMo Gym version from the worker's own interpreter."""
    for python in _gym_interpreters(gym_bin):
        try:
            completed = subprocess.run(
                [str(python), "-c", "import nemo_gym; print(nemo_gym.__version__)"],
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
) -> list[str]:
    """Reject configurations the NeMo Gym backend cannot run equivalently, before anything starts."""
    errors: list[str] = []
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
    if custom_dockerfile_mode not in (None, "rebase"):
        errors.append("--backend nemo-gym supports --custom-dockerfile-mode rebase only")
    if copy_repo:
        errors.append("--copy-repo is not supported with --backend nemo-gym yet")
    if environment_kwargs:
        errors.append("--environment-kwarg applies to the Harbor environment providers only")
    if override_storage_mb is not None:
        errors.append("--override-storage-mb is not supported with --backend nemo-gym")
    if agent_runtime_preflight:
        errors.append("--agent-runtime-preflight applies to the Harbor backend only")
    if timeout_multiplier not in (None, 1, 1.0):
        errors.append("--timeout-multiplier is not supported with --backend nemo-gym yet")
    return errors


def nemo_gym_config_errors(
    skill_path: Path,
    *,
    stop_on_pass: bool | None = None,
    timeout_multiplier: float | None = None,
    agent_runtime_preflight: bool | None = None,
    custom_dockerfile_mode: str | None = None,
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
    workspace = config.get("skill_workspace") or {}
    errors: list[str] = []
    if stop_on_pass is None and harbor.get("stop_on_pass"):
        errors.append("harbor.stop_on_pass is not supported with --backend nemo-gym yet")
    if timeout_multiplier is None and harbor.get("timeout_multiplier", 1.0) not in (1, 1.0):
        errors.append("harbor.timeout_multiplier is not supported with --backend nemo-gym yet")
    if agent_runtime_preflight is None and harbor.get("agent_runtime_preflight"):
        errors.append("harbor.agent_runtime_preflight applies to the Harbor backend only")
    if harbor.get("base_image_mode", "disabled") != "disabled":
        errors.append("harbor.base_image_mode reuse/rebuild applies to the Harbor backend only")
    if custom_dockerfile_mode is None and harbor.get("custom_dockerfile_mode") not in (None, "rebase"):
        errors.append("--backend nemo-gym supports --custom-dockerfile-mode rebase only")
    if harbor.get("task_source", "auto") == "native_harbor":
        errors.append("harbor.task_source native_harbor is not supported with --backend nemo-gym yet")
    errors += [
        f"harbor.{key} is not supported with --backend nemo-gym yet"
        for key in ("agent_workdir", "pre_agent_setup")
        if harbor.get(key)
    ]
    if "storage_mb" in (harbor.get("resources") or {}):
        errors.append("harbor.resources.storage_mb is not supported with --backend nemo-gym")
    if workspace.get("decoy_manifest"):
        errors.append("skill_workspace.decoy_manifest is not supported with --backend nemo-gym yet")
    return errors
