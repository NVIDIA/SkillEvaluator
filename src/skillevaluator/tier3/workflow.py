# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation for the direct Tier 3 workflow, sharing the expert evaluator."""

from __future__ import annotations

import stat
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import click


def _require_regular_file(path: Path, *, optional: bool = False) -> None:
    """Reject linked or special authored files before a parser can open them."""
    from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if optional:
            return
        raise ValueError(f"Required skill file is missing: {path}") from None
    if stat_is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Skill input must be a regular non-linked file: {path}")


def _validate_skill_root(skill_path: Path) -> None:
    from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

    metadata = skill_path.lstat()
    if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Skill root must be a real directory: {skill_path}")
    _require_regular_file(skill_path / "SKILL.md")


def _reject_legacy_datasets(skill_path: Path) -> None:
    """Refuse legacy discovery before it can escape the evaluator snapshot."""
    from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS
    from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

    legacy = skill_path / "eval"
    try:
        metadata = legacy.lstat()
    except FileNotFoundError:
        return
    migrate = (
        "Direct Tier 3 requires evals/evals.*; migrate legacy eval/dataset.* "
        "to evals/evals.* before running this workflow."
    )
    if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(migrate)
    for extension in DATASET_EXTENSIONS:
        try:
            (legacy / f"dataset{extension}").lstat()
        except FileNotFoundError:
            continue
        raise ValueError(migrate)


@contextmanager
def _source_snapshot(skill_path: Path) -> Iterator[Path]:
    """Read authored evaluator inputs only through the engine's secure snapshot."""
    from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS
    from skillevaluator.tier3.evals_config import CONFIG_FILENAMES
    from skillevaluator.tier3.harbor.adapter import private_evaluator_skill_snapshot

    _reject_legacy_datasets(skill_path)
    evals_dir = skill_path / "evals"
    if evals_dir.is_symlink() or (evals_dir.exists() and not evals_dir.is_dir()):
        raise ValueError(f"evals directory must be a real directory inside the skill: {evals_dir}")
    if not evals_dir.exists():
        yield skill_path
        return
    # The engine snapshot selects one dataset/config; refuse ambiguous authored
    # inputs before that selection. Inspect names only, never their contents.
    for label, names in (
        ("datasets", tuple(f"evals{extension}" for extension in DATASET_EXTENSIONS)),
        ("config files", CONFIG_FILENAMES),
    ):
        present = [path for name in names if (path := evals_dir / name).exists() or path.is_symlink()]
        if len(present) > 1:
            raise ValueError(f"multiple eval {label} found: " + ", ".join(str(path) for path in present))
    with private_evaluator_skill_snapshot(skill_path) as snapshot:
        yield snapshot


def _preflight_options(skill_path: Path, params: dict[str, Any]) -> None:
    """Reject known configuration errors before generating a paid dataset."""
    from skillevaluator.cli import _evaluated_source_from_options
    from skillevaluator.provider_config import resolve_llm_provider
    from skillevaluator.tier3.commands import parse_agent_model_overrides, resolve_agents, validate_agents
    from skillevaluator.tier3.evals_config import _validate_config, load_evals_config
    from skillevaluator.tier3.harbor.runner import (
        _model_for_agent,
        _resolve_agent_runtime_plan,
        _resolve_runtime_env,
        _workspace_skills,
    )

    provider = resolve_llm_provider()
    config, config_path = load_evals_config(skill_path)
    effective = deepcopy(config)
    effective.setdefault("schema_version", 1)
    harbor = effective.setdefault("harbor", {})
    for name in (
        "n_attempts",
        "pass_threshold",
        "stop_on_pass",
        "n_concurrent",
        "max_agents",
        "timeout_multiplier",
        "agent_runtime_preflight",
        "custom_dockerfile_mode",
    ):
        if params.get(name) is not None:
            harbor[name] = params[name]
    for resource in ("cpus", "memory_mb", "storage_mb"):
        if params.get(f"override_{resource}") is not None:
            harbor.setdefault("resources", {})[resource] = params[f"override_{resource}"]
    for cli_name, group in (("grading_mode", "grading"), ("skill_workspace_mode", "skill_workspace")):
        if params.get(cli_name) is not None:
            effective.setdefault(group, {})["mode"] = params[cli_name]
    _validate_config(effective, config_path or skill_path / "evals" / "config.yml")
    if harbor.get("stop_on_pass", False) and harbor.get("n_attempts", 1) == 1:
        raise ValueError("stop_on_pass requires n_attempts > 1")

    workspace = effective.get("skill_workspace", {})
    # Match the expert callback: CLI paths are relative to the caller, while
    # authored config paths remain relative to the target skill's parent.
    cli_includes = [path.resolve() for path in params["include_skills"]]
    include_values = [*workspace.get("include", []), *cli_includes]
    if include_values and workspace.get("mode", "isolated") != "group":
        raise ValueError("include_skills requires skill_workspace.mode=group")
    _workspace_skills(params["skill_path"].resolve(), include_values)

    agents = resolve_agents(params["agents"], provider=provider.provider)
    if not agents:
        raise ValueError("Select at least one agent with --agents.")
    # The cloned expert callback receives an explicit, canonical selection so
    # preparation, prerequisite checks, progress, and execution cannot diverge.
    params["agents"] = ",".join(agents)
    unknown = validate_agents(agents)
    if unknown:
        raise ValueError("Unknown agent(s): " + ", ".join(unknown))
    overrides = parse_agent_model_overrides(params["agent_model"])
    unselected = sorted(set(overrides) - set(agents))
    if unselected:
        raise ValueError("--agent-model provided for agent(s) not selected by --agents: " + ", ".join(unselected))
    resolution = {
        agent: _model_for_agent(
            agent,
            cli_model=(overrides.get(agent) or [params["model"]])[0],
            config_agents=harbor.get("agents", {}),
            provider=provider,
        )
        for agent in agents
    }
    from skillevaluator.tier3.commands import parse_environment_kwargs

    parsed_environment_kwargs = parse_environment_kwargs(params.get("environment_kwargs", ()))
    resolved_environment_kwargs = {
        **(harbor.get("environment_kwargs") or {}),
        **parsed_environment_kwargs,
    }
    runtime_env, errors = _resolve_runtime_env(harbor.get("runtime_env"))
    if errors:
        raise ValueError("; ".join(errors))
    _resolve_agent_runtime_plan(
        provider=provider,
        agents=agents,
        models={agent: value[0] for agent, value in resolution.items()},
        configured_runtime_env=runtime_env,
        env_mode=params["env_mode"],
        model_sources={agent: value[1] for agent, value in resolution.items()},
        environment_kwargs=resolved_environment_kwargs,
    )
    _evaluated_source_from_options(
        params["evaluated_source_repository"],
        params["evaluated_source_revision"],
        params["evaluator_container_revision"],
    )


def _preflight_environment(params: dict[str, Any]) -> None:
    """Check installed runtimes and backend configuration without an agent run."""
    from skillevaluator.tier3.commands import parse_agents, parse_environment_kwargs
    from skillevaluator.tier3.harbor.runner import _check_prerequisites

    # These Harbor preflights perform remote authentication RPCs. Leave those
    # probes in the execution engine instead of running them before generation.
    if params["env_mode"] in {"cwsandbox", "wandb", "langsmith"}:
        return
    parsed_environment_kwargs = parse_environment_kwargs(params.get("environment_kwargs", ()))
    errors = _check_prerequisites(
        env_mode=params["env_mode"],
        agents=parse_agents(params["agents"]),
        environment_kwargs=parsed_environment_kwargs,
    )
    if errors:
        raise ValueError("; ".join(errors))


def _prepare_dataset(skill_path: Path, *, source_path: Path, progress: str = "auto") -> None:
    """Reuse valid authored sources; create one starter case only when absent."""
    from skillevaluator.cli import _ensure_autopilot_dataset
    from skillevaluator.tier3.commands import _validate_existing_scaffold_state
    from skillevaluator.tier3.dataset_utils import find_eval_file
    from skillevaluator.tier3.evals_spec import validate_tier3_source

    evals_dir = source_path / "evals"
    _validate_existing_scaffold_state(source_path, evals_dir)
    source, checks = validate_tier3_source(source_path)
    native_source = evals_dir / "harbor"
    missing = source == "missing" or (source == "evals_json" and find_eval_file(source_path) is None)
    generated = missing and not native_source.exists()
    if generated:
        _require_regular_file(skill_path / "SKILL.md")
        _require_regular_file(skill_path / "evals" / "EVAL.md", optional=True)
        _ensure_autopilot_dataset(skill_path, progress=progress)
        with _source_snapshot(skill_path) as generated_source:
            _validate_existing_scaffold_state(generated_source, generated_source / "evals")
            source, checks = validate_tier3_source(generated_source)
    errors = [f"{check.path}: {check.message}" for check in checks if check.status in {"error", "missing"}]
    if errors:
        raise ValueError("Invalid Tier 3 evaluation source: " + "; ".join(errors))
    if not generated:
        click.echo("Tier 3: reusing the existing evaluation source unchanged.", err=True)


def build_tier3_workflow(evaluate_command: click.Command) -> click.Command:
    """Clone the expert command with automatic dataset preparation enabled.

    Keeping its parameters and callback preserves evaluation options, result
    rendering, and failure exit semantics without another evaluation engine.
    """
    command = copy(evaluate_command)
    command.params = [copy(param) for param in evaluate_command.params]
    command.hidden = False
    command.help = (
        "Run Tier 3 live agent evaluation. Reuse an existing valid dataset or "
        "create one starter case when the dataset is missing, then compare "
        "execution with and without the skill. Review generated cases before "
        "using them as a benchmark."
    )
    for param in command.params:
        if param.name == "autopilot":
            param.hidden = True
        elif param.name == "skill_path" and isinstance(param.type, click.Path):
            param.type = copy(param.type)
            param.type.resolve_path = False
    original_callback = evaluate_command.callback
    if original_callback is None:
        raise ValueError("The Tier 3 evaluation command must have a callback.")

    def workflow(**params: Any) -> Any:
        try:
            _validate_skill_root(params["skill_path"])
            params["skill_path"] = params["skill_path"].resolve()
            with _source_snapshot(params["skill_path"]) as source_path:
                _preflight_options(source_path, params)
                _preflight_environment(params)
                _prepare_dataset(params["skill_path"], source_path=source_path, progress=params["progress"])
        except ImportError as exc:
            raise click.ClickException(
                "Tier 3 dependencies are not installed. Install skillevaluator[tier3] and try again."
            ) from exc
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        params["autopilot"] = False
        return original_callback(**params)

    command.callback = workflow
    return command
