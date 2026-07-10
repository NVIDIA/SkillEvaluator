# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared option models for the in-process evaluation service.

Both the CLI and the API construct these option objects and hand them to
:class:`skillevaluator.evaluation.service.EvaluationService`, guaranteeing that
the two surfaces drive the Tier 3 engine with identical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class EvaluationOptions:
    """Options for a Tier 3 live agent evaluation run.

    Field names mirror :func:`skillevaluator.tier3.commands.evaluate` keyword
    arguments so the service can forward them without translation drift.
    """

    skill_path: Path
    agents: str = "codex"
    env_mode: str = "docker"
    skip_baseline: bool = False
    n_attempts: int | None = None
    pass_threshold: float | None = None
    stop_on_pass: bool | None = None
    n_concurrent: int | None = None
    max_agents: int | None = None
    model: str | None = None
    agent_model: tuple[str, ...] = ()
    custom_dockerfile_mode: str | None = None
    skill_workspace_mode: str | None = None
    include_skills: tuple[Path, ...] = ()
    copy_repo: bool = False
    grading_mode: str | None = None
    results_dir: Path | None = None
    harbor_keep_jobs: bool = False
    agent_runtime_preflight: bool | None = None
    timeout_multiplier: float | None = None
    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None

    def engine_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments (excluding ``skill_path``) for the engine."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "skill_path"}


@dataclass
class DatasetOptions:
    """Options for synthetic dataset creation.

    Field names mirror :func:`skillevaluator.tier3.commands.create_dataset`.
    """

    skill_path: Path
    full: bool = False
    no_llm: bool = False
    dry_run: bool = False
    force: bool = False
    prompt: Path | None = None
    refine: bool = False
    from_results: Path | None = None
    results_dir: Path | None = None

    def engine_kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "skill_path"}
