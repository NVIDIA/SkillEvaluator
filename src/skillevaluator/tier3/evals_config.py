# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-skill eval execution config.

``evals/config.yml`` is intentionally separate from the eval dataset.  The
dataset says what to evaluate; this config says how SkillEvaluator should run Harbor.
"""

from __future__ import annotations

import json
import math
import re
from itertools import pairwise
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
_URI_AUTHORITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{0,31}://(?P<authority>[^\s/?#]*)")
_SCHEMELESS_CREDENTIAL_AUTHORITY_RE = re.compile(r"[^\s/:@]+:[^\s/@]+@[^\s/?#]+")
_REFERENCE_KWARG_NAMES_BY_ENV_MODE = {
    "ack": frozenset({"image_pull_secret"}),
    "cwsandbox": frozenset({"secrets"}),
    "daytona": frozenset({"secrets"}),
    "modal": frozenset({"registry_secret", "secrets"}),
    "skypilot": frozenset({"secrets"}),
    "wandb": frozenset({"secrets"}),
}
_CWSANDBOX_SECRET_REFERENCE_KEYS = frozenset({"env_var", "field", "name", "store"})
_KUBERNETES_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_MAX_ENVIRONMENT_KWARGS_JSON_BYTES = 64 * 1024
_MAX_ENVIRONMENT_KWARGS_DEPTH = 32
_MAX_ENVIRONMENT_KWARGS_NODES = 4096
_MAX_ENVIRONMENT_REFERENCE_BYTES = 512


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


def _is_sensitive_environment_kwarg_name(name: str) -> bool:
    """Recognize secret-bearing names across snake, kebab, and camel case."""
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    tokens = tuple(part for part in re.sub(r"[^A-Za-z0-9]+", "_", normalized).lower().split("_") if part)
    if any(
        token
        in {
            "auth",
            "authorization",
            "authentication",
            "credential",
            "credentials",
            "passwd",
            "password",
            "secret",
            "secrets",
            "token",
        }
        for token in tokens
    ):
        return True
    if any(
        pair in {("api", "key"), ("access", "key"), ("private", "key"), ("secret", "key")} for pair in pairwise(tokens)
    ):
        return True
    compact = "".join(tokens)
    return compact.endswith(
        (
            "apikey",
            "accesskey",
            "privatekey",
            "secretkey",
            "auth",
            "authorization",
            "authentication",
            "credential",
            "credentials",
            "passwd",
            "password",
            "secret",
            "secrets",
            "token",
        )
    )


def _contains_credential_bearing_uri(value: str) -> bool:
    """Reject URI userinfo without parsing or rendering the supplied value."""
    if any("@" in match.group("authority") for match in _URI_AUTHORITY_RE.finditer(value)):
        return True
    return _SCHEMELESS_CREDENTIAL_AUTHORITY_RE.search(value) is not None


def _safe_environment_kwarg_path(path: tuple[str | int, ...]) -> str:
    """Describe a value location without echoing attacker-controlled nested keys."""
    if not path:
        return "mapping"
    root = path[0]
    if not isinstance(root, str) or not _ENV_NAME_RE.fullmatch(root):
        return "mapping value"
    return root if len(path) == 1 else f"{root} nested value"


def _environment_kwarg_shape_error(value: Any) -> str | None:
    """Validate bounded JSON shape iteratively so cycles/deep values fail closed."""
    stack: list[tuple[bool, Any, tuple[str | int, ...], int]] = [(True, value, (), 0)]
    active_containers: set[int] = set()
    node_count = 0
    while stack:
        entering, current, path, depth = stack.pop()
        if not entering:
            active_containers.remove(id(current))
            continue
        node_count += 1
        if node_count > _MAX_ENVIRONMENT_KWARGS_NODES:
            return f"must contain at most {_MAX_ENVIRONMENT_KWARGS_NODES} JSON values"
        if depth > _MAX_ENVIRONMENT_KWARGS_DEPTH:
            return f"must nest at most {_MAX_ENVIRONMENT_KWARGS_DEPTH} levels"
        if isinstance(current, dict | list):
            identity = id(current)
            if identity in active_containers:
                return "must not contain cyclic values"
            active_containers.add(identity)
            stack.append((False, current, path, depth))
            if isinstance(current, dict):
                items = list(current.items())
                for raw_key, item in reversed(items):
                    if not isinstance(raw_key, str):
                        return f"{_safe_environment_kwarg_path(path)} keys must be strings"
                    stack.append((True, item, (*path, raw_key), depth + 1))
            else:
                for index in range(len(current) - 1, -1, -1):
                    stack.append((True, current[index], (*path, index), depth + 1))
            continue
        if isinstance(current, str):
            if _contains_credential_bearing_uri(current):
                return (
                    f"{_safe_environment_kwarg_path(path)} contains a credential-bearing URI; "
                    "pass credentials through the host environment instead"
                )
            continue
        if current is None or isinstance(current, bool | int):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                return f"{_safe_environment_kwarg_path(path)} must be a finite JSON number"
            continue
        return f"{_safe_environment_kwarg_path(path)} must contain only JSON-compatible values"
    return None


def _secret_reference_name_error(value: Any, *, label: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{label} must be a non-empty secret reference name"
    if value != value.strip() or any(ord(character) < 32 or ord(character) == 127 for character in value):
        return f"{label} must be a trimmed secret reference name without control characters"
    if len(value.encode("utf-8")) > _MAX_ENVIRONMENT_REFERENCE_BYTES:
        return f"{label} must encode to at most {_MAX_ENVIRONMENT_REFERENCE_BYTES} bytes"
    if _contains_credential_bearing_uri(value):
        return f"{label} must be a secret reference name, not a credential-bearing URI"
    return None


def _reference_kwarg_error(env_mode: str | None, name: str, value: Any) -> str | None:
    """Validate Harbor fields whose values name provider-managed secrets."""
    if name == "image_pull_secret":
        if env_mode != "ack":
            return "image_pull_secret is supported only for Harbor environment 'ack'"
        if not isinstance(value, str) or len(value.encode("utf-8")) > 253:
            return "image_pull_secret must be a Kubernetes DNS-subdomain Secret name"
        labels = value.split(".")
        if not labels or any(len(label) > 63 or not _KUBERNETES_DNS_LABEL_RE.fullmatch(label) for label in labels):
            return "image_pull_secret must be a Kubernetes DNS-subdomain Secret name"
        return None
    if env_mode in {"modal", "skypilot"} and name == "secrets":
        if not isinstance(value, list):
            return f"{name} must be a list of provider secret reference names for Harbor environment '{env_mode}'"
        for item in value:
            if error := _secret_reference_name_error(item, label=name):
                return error
        return None
    if env_mode == "modal" and name == "registry_secret":
        return _secret_reference_name_error(value, label=name)
    if env_mode == "daytona" and name == "secrets":
        if not isinstance(value, dict):
            return "secrets must map sandbox environment variable names to Daytona organization secret names"
        for target_name, secret_name in value.items():
            if not isinstance(target_name, str) or not _ENV_NAME_RE.fullmatch(target_name):
                return "secrets keys must be valid sandbox environment variable names"
            if error := _secret_reference_name_error(secret_name, label="secrets value"):
                return error
        return None
    if env_mode in {"cwsandbox", "wandb"} and name == "secrets":
        if not isinstance(value, list):
            return f"secrets must be a list of provider secret reference mappings for Harbor environment '{env_mode}'"
        for item in value:
            if not isinstance(item, dict) or not item:
                return "secrets entries must be non-empty provider secret reference mappings"
            if set(item) - _CWSANDBOX_SECRET_REFERENCE_KEYS:
                return "secrets entries contain unsupported provider secret reference fields"
            for field_name, field_value in item.items():
                if field_name == "env_var":
                    if not isinstance(field_value, str) or not _ENV_NAME_RE.fullmatch(field_value):
                        return "secrets env_var fields must be valid environment variable names"
                elif error := _secret_reference_name_error(field_value, label=f"secrets {field_name} field"):
                    return error
        return None
    return f"{name} is secret-bearing; pass credentials through the host environment instead"


def _environment_kwarg_secret_policy_error(value: dict[str, Any], *, env_mode: str | None) -> str | None:
    stack: list[tuple[dict[str, Any] | list[Any], tuple[str | int, ...]]] = [(value, ())]
    allowed_mode_references = _REFERENCE_KWARG_NAMES_BY_ENV_MODE.get(env_mode or "", frozenset())
    while stack:
        current, path = stack.pop()
        if isinstance(current, dict):
            for raw_key, item in current.items():
                item_path = (*path, raw_key)
                is_top_level_reference = not path and (raw_key in allowed_mode_references)
                if is_top_level_reference:
                    if error := _reference_kwarg_error(env_mode, raw_key, item):
                        return error
                    continue
                if _is_sensitive_environment_kwarg_name(raw_key):
                    return (
                        f"{_safe_environment_kwarg_path(item_path)} is secret-bearing; "
                        "pass credentials through the host environment instead"
                    )
                if isinstance(item, dict | list):
                    stack.append((item, item_path))
        else:
            for index, item in enumerate(current):
                if isinstance(item, dict | list):
                    stack.append((item, (*path, index)))
    return None


def validate_environment_kwargs(value: Any, *, env_mode: str | None = None) -> dict[str, Any]:
    """Validate non-secret Harbor constructor kwargs for safe argv forwarding."""
    if not isinstance(value, dict):
        raise ValueError("must be a mapping")
    for raw_name in value:
        if not isinstance(raw_name, str) or not _ENV_NAME_RE.fullmatch(raw_name):
            raise ValueError("keys must be valid Python keyword names")
    if error := _environment_kwarg_shape_error(value):
        raise ValueError(error)
    if error := _environment_kwarg_secret_policy_error(value, env_mode=env_mode):
        raise ValueError(error)
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    except (RecursionError, TypeError, ValueError):
        raise ValueError("must contain only JSON-compatible values") from None
    if len(encoded.encode("utf-8")) > _MAX_ENVIRONMENT_KWARGS_JSON_BYTES:
        raise ValueError(f"must encode to at most {_MAX_ENVIRONMENT_KWARGS_JSON_BYTES} bytes")
    return dict(value)


def parse_environment_kwarg_overrides(
    values: tuple[str, ...] | list[str],
    *,
    env_mode: str | None = None,
) -> dict[str, Any]:
    """Parse repeatable CLI ``KEY=VALUE`` arguments using Harbor's value rules."""
    parsed: dict[str, Any] = {}
    for index, raw in enumerate(values, start=1):
        if "=" not in raw:
            raise ValueError(f"Invalid --environment-kwarg entry {index}: expected KEY=VALUE")
        name, raw_value = raw.split("=", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = {"True": True, "False": False, "None": None}.get(raw_value, raw_value)
        except RecursionError:
            raise ValueError(f"Invalid --environment-kwarg entry {index}: JSON value nests too deeply") from None
        parsed[name] = value
    try:
        return validate_environment_kwargs(parsed, env_mode=env_mode)
    except ValueError as exc:
        raise ValueError(f"Invalid --environment-kwarg: {exc}") from None


def encode_environment_kwarg(name: str, value: Any) -> str:
    """Encode one validated kwarg so Harbor's parser preserves its JSON type."""
    return f"{name}={json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(',', ':'))}"


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
