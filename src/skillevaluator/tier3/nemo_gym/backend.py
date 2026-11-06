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
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which("gym")


def nemo_gym_version(gym_bin: str) -> str | None:
    """Return the installed NeMo Gym version from the worker's own interpreter."""
    python = Path(gym_bin).with_name("python")
    if not python.is_file():
        return None
    try:
        completed = subprocess.run(
            [str(python), "-c", "import nemo_gym; print(nemo_gym.__version__)"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    version = completed.stdout.strip().splitlines()
    return version[-1] if completed.returncode == 0 and version else None


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
    return errors
