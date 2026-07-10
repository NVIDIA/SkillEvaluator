# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-skill eval execution config.

``evals/config.yml`` is intentionally separate from the eval dataset.  The
dataset says what to evaluate; this config says how SkillEvaluator should run Harbor.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.tier3.harbor import canonical_agent_name

CONFIG_FILENAMES = ("config.yml", "config.yaml")
HARBOR_CUSTOM_DOCKERFILE_MODES = {"preserve", "rebase"}
HARBOR_BASE_IMAGE_MODES = {"reuse", "rebuild", "disabled"}
SKILL_WORKSPACE_MODES = {"isolated", "group"}
# Legacy grading-mode spellings stay accepted API surface; loading normalizes
# them so the engine only ever sees the current names.
GRADING_MODE_ALIASES = {
    "aces_default": "default",
    "aces_plus_custom": "default_plus_custom",
}
GRADING_MODES = {"default", "default_plus_custom", "custom_only", *GRADING_MODE_ALIASES}

_TOP_LEVEL_KEYS = {"schema_version", "harbor", "skill_workspace", "grading"}
_HARBOR_KEYS = {
    "task_source",
    "custom_dockerfile_mode",
    "base_image_mode",
    "n_attempts",
    "pass_threshold",
    "stop_on_pass",
    "n_concurrent",
    "max_agents",
    "timeout_multiplier",
    "agent_runtime_preflight",
    "agent_workdir",
    "resources",
    "runtime_env",
    "pre_agent_setup",
    "passthrough_env",
    "setup_commands",
    "agents",
}
HARBOR_TASK_SOURCES = {"auto", "evals_json", "native_harbor"}
_AGENT_KEYS = {"model"}
_RESOURCE_KEYS = {"cpus", "memory_mb", "storage_mb"}
_SKILL_WORKSPACE_KEYS = {"mode", "include"}
_GRADING_KEYS = {"mode"}
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EvalsConfigError(ValueError):
    """Raised when ``evals/config.yml`` is present but invalid."""


def find_evals_config(skill_path: Path) -> Path | None:
    """Return the first supported evals config file for a skill, if present."""
    evals_dir = skill_path / "evals"
    for name in CONFIG_FILENAMES:
        candidate = evals_dir / name
        if candidate.exists():
            return candidate
    return None


def load_evals_config(skill_path: Path) -> tuple[dict[str, Any], Path | None]:
    """Load and validate ``evals/config.yml`` or ``evals/config.yaml``.

    Missing config is not an error.  Returned dictionaries contain only keys
    supplied by the config file, so callers can preserve CLI/config/default
    precedence explicitly.
    """
    config_path = find_evals_config(skill_path)
    if config_path is None:
        return {}, None

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise EvalsConfigError(f"{config_path}: invalid YAML: {e}") from e
    except OSError as e:
        raise EvalsConfigError(f"{config_path}: cannot read config: {e}") from e

    if raw is None:
        raise EvalsConfigError(f"{config_path}: config must not be empty")
    if not isinstance(raw, dict):
        raise EvalsConfigError(f"{config_path}: top-level config must be a mapping")

    return _validate_config(raw, config_path), config_path


def _validate_config(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    unknown_top = set(raw) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise EvalsConfigError(f"{config_path}: unknown top-level key(s): {', '.join(sorted(unknown_top))}")

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise EvalsConfigError(f"{config_path}: schema_version must be 1")

    out: dict[str, Any] = {"schema_version": 1}
    harbor_raw = raw.get("harbor")
    if harbor_raw is not None:
        if not isinstance(harbor_raw, dict):
            raise EvalsConfigError(f"{config_path}: harbor must be a mapping")

        unknown_harbor = set(harbor_raw) - _HARBOR_KEYS
        if unknown_harbor:
            raise EvalsConfigError(f"{config_path}: unknown harbor key(s): {', '.join(sorted(unknown_harbor))}")

        harbor: dict[str, Any] = {}
        if "task_source" in harbor_raw:
            harbor["task_source"] = _enum(
                harbor_raw["task_source"],
                HARBOR_TASK_SOURCES,
                config_path,
                "harbor.task_source",
            )
        if "custom_dockerfile_mode" in harbor_raw:
            harbor["custom_dockerfile_mode"] = _enum(
                harbor_raw["custom_dockerfile_mode"],
                HARBOR_CUSTOM_DOCKERFILE_MODES,
                config_path,
                "harbor.custom_dockerfile_mode",
            )
        if "base_image_mode" in harbor_raw:
            harbor["base_image_mode"] = _enum(
                harbor_raw["base_image_mode"],
                HARBOR_BASE_IMAGE_MODES,
                config_path,
                "harbor.base_image_mode",
            )
        if "n_attempts" in harbor_raw:
            harbor["n_attempts"] = _int_at_least(harbor_raw["n_attempts"], 1, config_path, "harbor.n_attempts")
        if "pass_threshold" in harbor_raw:
            harbor["pass_threshold"] = _float_between(
                harbor_raw["pass_threshold"], 0.0, 1.0, config_path, "harbor.pass_threshold"
            )
        if "stop_on_pass" in harbor_raw:
            harbor["stop_on_pass"] = _bool(harbor_raw["stop_on_pass"], config_path, "harbor.stop_on_pass")
        if "n_concurrent" in harbor_raw:
            harbor["n_concurrent"] = _int_at_least(harbor_raw["n_concurrent"], 1, config_path, "harbor.n_concurrent")
        if "max_agents" in harbor_raw:
            harbor["max_agents"] = _int_at_least(harbor_raw["max_agents"], 1, config_path, "harbor.max_agents")
        if "timeout_multiplier" in harbor_raw:
            harbor["timeout_multiplier"] = _float_greater_than(
                harbor_raw["timeout_multiplier"], 0.0, config_path, "harbor.timeout_multiplier"
            )
        if "agent_runtime_preflight" in harbor_raw:
            harbor["agent_runtime_preflight"] = _bool(
                harbor_raw["agent_runtime_preflight"],
                config_path,
                "harbor.agent_runtime_preflight",
            )
        if "agent_workdir" in harbor_raw:
            harbor["agent_workdir"] = _non_empty_string(
                harbor_raw["agent_workdir"],
                config_path,
                "harbor.agent_workdir",
            )
        if "resources" in harbor_raw:
            harbor["resources"] = _resources(harbor_raw["resources"], config_path)
        runtime_env_value = _aliased_harbor_value(
            harbor_raw,
            canonical="runtime_env",
            alias="passthrough_env",
            config_path=config_path,
        )
        if runtime_env_value is not None:
            harbor["runtime_env"] = _runtime_env(runtime_env_value, config_path)
        pre_agent_setup_value = _aliased_harbor_value(
            harbor_raw,
            canonical="pre_agent_setup",
            alias="setup_commands",
            config_path=config_path,
        )
        if pre_agent_setup_value is not None:
            harbor["pre_agent_setup"] = _pre_agent_setup(pre_agent_setup_value, config_path)
        if "agents" in harbor_raw:
            harbor["agents"] = _agents(harbor_raw["agents"], config_path)

        out["harbor"] = harbor

    skill_workspace_raw = raw.get("skill_workspace")
    if skill_workspace_raw is not None:
        out["skill_workspace"] = _skill_workspace(skill_workspace_raw, config_path)

    grading_raw = raw.get("grading")
    if grading_raw is not None:
        out["grading"] = _grading(grading_raw, config_path)

    return out


def _aliased_harbor_value(
    harbor_raw: dict[str, Any],
    *,
    canonical: str,
    alias: str,
    config_path: Path,
) -> Any:
    if canonical in harbor_raw and alias in harbor_raw:
        raise EvalsConfigError(
            f"{config_path}: harbor.{canonical} and harbor.{alias} are aliases; use only harbor.{canonical}"
        )
    if canonical in harbor_raw:
        return harbor_raw[canonical]
    if alias in harbor_raw:
        return harbor_raw[alias]
    return None


def _enum(value: Any, allowed: set[str], config_path: Path, field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise EvalsConfigError(f"{config_path}: {field} must be one of: {', '.join(sorted(allowed))}")
    return value


def _bool(value: Any, config_path: Path, field: str) -> bool:
    if not isinstance(value, bool):
        raise EvalsConfigError(f"{config_path}: {field} must be true or false")
    return value


def _int_at_least(value: Any, minimum: int, config_path: Path, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvalsConfigError(f"{config_path}: {field} must be an integer")
    if value < minimum:
        raise EvalsConfigError(f"{config_path}: {field} must be >= {minimum}")
    return value


def _float_between(value: Any, minimum: float, maximum: float, config_path: Path, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvalsConfigError(f"{config_path}: {field} must be a number")
    value = float(value)
    if not minimum <= value <= maximum:
        raise EvalsConfigError(f"{config_path}: {field} must be between {minimum} and {maximum}")
    return value


def _float_greater_than(value: Any, minimum: float, config_path: Path, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvalsConfigError(f"{config_path}: {field} must be a number")
    value = float(value)
    if value <= minimum:
        raise EvalsConfigError(f"{config_path}: {field} must be > {minimum}")
    return value


def _agents(value: Any, config_path: Path) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise EvalsConfigError(f"{config_path}: harbor.agents must be a mapping")

    agents: dict[str, dict[str, str]] = {}
    authored_names: dict[str, str] = {}
    for agent_name, agent_cfg in value.items():
        if not isinstance(agent_name, str) or not agent_name:
            raise EvalsConfigError(f"{config_path}: harbor.agents keys must be non-empty strings")
        if not isinstance(agent_cfg, dict):
            raise EvalsConfigError(f"{config_path}: harbor.agents.{agent_name} must be a mapping")

        canonical_name = canonical_agent_name(agent_name)
        if canonical_name in agents:
            previous = authored_names[canonical_name]
            raise EvalsConfigError(
                f"{config_path}: harbor.agents.{previous} and harbor.agents.{agent_name} "
                f"refer to the same agent ({canonical_name}); use only {canonical_name}"
            )

        unknown_agent = set(agent_cfg) - _AGENT_KEYS
        if unknown_agent:
            raise EvalsConfigError(
                f"{config_path}: unknown key(s) under harbor.agents.{agent_name}: {', '.join(sorted(unknown_agent))}"
            )

        model = agent_cfg.get("model")
        if model is None:
            agents[canonical_name] = {}
        elif not isinstance(model, str) or not model.strip():
            raise EvalsConfigError(f"{config_path}: harbor.agents.{agent_name}.model must be a non-empty string")
        else:
            agents[canonical_name] = {"model": model.strip()}
        authored_names[canonical_name] = agent_name

    return agents


def _non_empty_string(value: Any, config_path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalsConfigError(f"{config_path}: {field} must be a non-empty string")
    return value.strip()


def _resources(value: Any, config_path: Path) -> dict[str, int]:
    if not isinstance(value, dict):
        raise EvalsConfigError(f"{config_path}: harbor.resources must be a mapping")

    unknown = set(value) - _RESOURCE_KEYS
    if unknown:
        raise EvalsConfigError(f"{config_path}: unknown key(s) under harbor.resources: {', '.join(sorted(unknown))}")

    resources: dict[str, int] = {}
    for key in ("cpus", "memory_mb", "storage_mb"):
        if key in value:
            resources[key] = _int_at_least(value[key], 1, config_path, f"harbor.resources.{key}")
    return resources


def _validate_env_name(name: Any, config_path: Path, field: str) -> str:
    if not isinstance(name, str) or not _ENV_NAME_RE.match(name):
        raise EvalsConfigError(f"{config_path}: {field} entries must be valid environment variable names")
    return name


def _runtime_env(value: Any, config_path: Path) -> dict[str, str]:
    """Normalize Harbor runtime env config to Harbor's ``environment.env`` shape."""
    field = "harbor.runtime_env"
    if isinstance(value, list):
        out: dict[str, str] = {}
        for idx, item in enumerate(value):
            name = _validate_env_name(item, config_path, f"{field}[{idx}]")
            out[name] = f"${{{name}}}"
        return out

    if isinstance(value, dict):
        out = {}
        for raw_name, raw_template in value.items():
            name = _validate_env_name(raw_name, config_path, field)
            if not isinstance(raw_template, str) or not raw_template.strip():
                raise EvalsConfigError(f"{config_path}: {field}.{name} must be a non-empty string")
            out[name] = raw_template
        return out

    raise EvalsConfigError(f"{config_path}: {field} must be a list or mapping")


def _pre_agent_setup(value: Any, config_path: Path) -> list[str]:
    field = "harbor.pre_agent_setup"
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise EvalsConfigError(f"{config_path}: {field} must be a string or list")

    commands: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise EvalsConfigError(f"{config_path}: {field}[{idx}] must be a non-empty string")
        commands.append(item.strip())
    return commands


def _string_list(value: Any, config_path: Path, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvalsConfigError(f"{config_path}: {field} must be a list")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise EvalsConfigError(f"{config_path}: {field}[{idx}] must be a non-empty string")
        out.append(item)
    return out


def _skill_workspace(value: Any, config_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvalsConfigError(f"{config_path}: skill_workspace must be a mapping")

    unknown = set(value) - _SKILL_WORKSPACE_KEYS
    if unknown:
        raise EvalsConfigError(f"{config_path}: unknown skill_workspace key(s): {', '.join(sorted(unknown))}")

    out: dict[str, Any] = {}
    if "mode" in value:
        out["mode"] = _enum(value["mode"], SKILL_WORKSPACE_MODES, config_path, "skill_workspace.mode")
    if "include" in value:
        out["include"] = _string_list(value["include"], config_path, "skill_workspace.include")
    return out


def _grading(value: Any, config_path: Path) -> dict[str, str]:
    if not isinstance(value, dict):
        raise EvalsConfigError(f"{config_path}: grading must be a mapping")

    unknown = set(value) - _GRADING_KEYS
    if unknown:
        raise EvalsConfigError(f"{config_path}: unknown grading key(s): {', '.join(sorted(unknown))}")

    out: dict[str, str] = {}
    if "mode" in value:
        mode = _enum(value["mode"], GRADING_MODES, config_path, "grading.mode")
        out["mode"] = GRADING_MODE_ALIASES.get(mode, mode)
    return out
