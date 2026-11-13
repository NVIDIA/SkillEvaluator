# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared option models for the in-process evaluation service.

Both the CLI and the API construct these option objects and hand them to
:class:`skillevaluator.evaluation.service.EvaluationService`, guaranteeing that
the two surfaces drive the Tier 3 engine with identical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class EvaluationOptions:
    """Options for a Tier 3 live agent evaluation run.

    Field names mirror :func:`skillevaluator.tier3.commands.evaluate` keyword
    arguments so the service can forward them without translation drift.
    """

    skill_path: Path
    agents: str | None = None
    env_mode: str = "docker"
    skip_baseline: bool = False
    # Baseline-only runs (for example harness calibration) skip the with-skill
    # arm. Mutually exclusive with ``skip_baseline``.
    skip_with_skill: bool = False
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
    repo_context_exclude_paths: tuple[Path, ...] = ()
    environment_kwarg: tuple[str, ...] = ()
    # Supplied by the orchestration input, never inferred from repository state:
    # the tree that runs the evaluator is not the tree being evaluated.
    evaluated_source: dict[str, str] | None = None
    environment_kwargs: dict[str, str] = field(default_factory=dict)
    # Appended so legacy positional construction keeps its field order.
    max_cases: int | None = None
    reuse_baseline: str | None = None
    compare_to: str | None = None
    max_agent_tokens: int | None = None
    max_judge_calls: int | None = None
    max_attempts_total: int | None = None
    max_cost_usd: float | None = None
    price_file: Path | None = None
    tolerate_baseline_timeouts: bool | None = None
    # Tier 3 execution backend ("harbor" or "nemo-gym"); appended to keep positional order stable.
    backend: str = "harbor"

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
    grounding: str = "auto"
    refine: bool = False
    from_results: Path | None = None
    results_dir: Path | None = None

    def engine_kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "skill_path"}


@dataclass(frozen=True)
class PluginOptions:
    """Plugin runtime controls; unset values defer to authored configuration."""

    plugin_load: str | None = None
    lift_mode: str | None = None
    probe_mcp: bool | None = None
    probe_mcp_env: tuple[str, ...] = ()
    trial_retries: int | None = None
    repo_root: Path | None = None
    allowed_private_hosts: tuple[str, ...] = ()
    hook_allowed_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from skillevaluator.validators.plugin_policy import normalize_plugin_allowlist

        object.__setattr__(
            self,
            "allowed_private_hosts",
            normalize_plugin_allowlist(self.allowed_private_hosts, "mcp.allowed_private_hosts", 255),
        )
        object.__setattr__(
            self, "hook_allowed_urls", normalize_plugin_allowlist(self.hook_allowed_urls, "hooks.allowed_urls", 2048)
        )
        for name, choices in (
            ("plugin_load", {"wrapper", "native", "auto"}),
            ("lift_mode", {"effectiveness", "integration", "both", "ladder"}),
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or value not in choices):
                raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
        if self.probe_mcp is not None and type(self.probe_mcp) is not bool:
            raise ValueError("probe_mcp must be a boolean")
        if self.trial_retries is not None and (type(self.trial_retries) is not int or not 0 <= self.trial_retries <= 5):
            raise ValueError("trial_retries must be an integer between 0 and 5")
        if (
            not isinstance(self.probe_mcp_env, tuple)
            or len(self.probe_mcp_env) > 64
            or any(not isinstance(value, str) or len(value) > 512 for value in self.probe_mcp_env)
        ):
            raise ValueError("probe_mcp_env must contain at most 64 bounded environment grants")
