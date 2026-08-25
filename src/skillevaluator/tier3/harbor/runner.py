# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Harbor runner for live agent skill evaluation."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from queue import Empty, SimpleQueue
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from skillevaluator import __version__
from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.provider_config import (
    ProviderConfig,
    ProviderConfigurationError,
    _normalize_anthropic_base_url,
    resolve_llm_provider,
)
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.harbor.adapter import (
    _VERIFIER_JUDGE_MODEL_ENV_VARS,
    _prevalidate_baseline_skill_candidates,
    build_eval_base_image,
    find_evals_file,
    generate_harbor_tasks,
    private_evaluator_skill_snapshot,
    stage_native_harbor_tasks,
    validate_output_provenance_key_location,
    validate_results_root_location,
)
from skillevaluator.tier3.harbor.artifact_retention import HarborArtifactLifecycle, RetentionOutcome
from skillevaluator.tier3.harbor.collector import (
    collect_harbor_results,
    harbor_job_passed,
    validate_harbor_job_result,
)
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, score_definition
from skillevaluator.tier3.harbor.progress import (
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    Tier3RunPlan,
    redact_progress_detail,
    safe_progress_reporter,
    secret_values_from_environment,
)
from skillevaluator.tier3.harbor.report_data import (
    build_dataset_snapshot,
    load_staged_harbor_dataset,
)
from skillevaluator.tier3.harbor.secure_copy import copytree_secure
from skillevaluator.tier3.harbor.secure_docker_environment import SECURE_DOCKER_ENV_IMPORT_PATH
from skillevaluator.tier3.harbor.sensitive_stdin import (
    NVIDIA_BUILD_KEY_STDIN_ENV as _NVIDIA_BUILD_KEY_STDIN_ENV,
)
from skillevaluator.tier3.harbor.sensitive_stdin import (
    NVIDIA_BUILD_STDIN_SENTINEL as _NVIDIA_BUILD_STDIN_SENTINEL,
)
from skillevaluator.tier3.output_provenance import (
    mark_generated_output_root,
    remove_generated_output_root_if_owned,
    remove_output_reservation_if_identity_matches,
    write_output_file_atomically,
)
from skillevaluator.tier3.results_location import publish_latest_results
from skillevaluator.tier3_environments import DEFAULT_ENV_MODE, ENV_MODE_LOCAL, HARBOR_ENV_MODES

logger = logging.getLogger(__name__)


def _persist_dataset_truth(run_dir: Path, *, fallback_task_ids: list[str]) -> dict[str, Any]:
    """Persist immutable dataset and evaluator identity before staging cleanup."""
    entries = load_staged_harbor_dataset(run_dir)
    if not entries:
        entries = [{"id": task_id} for task_id in fallback_task_ids]
    snapshot = build_dataset_snapshot(entries, evaluator_version=__version__)
    target = run_dir / "dataset_snapshot.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=run_dir,
        prefix=".dataset_snapshot.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(snapshot, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return snapshot


_NVIDIA_BUILD_FILE_SENTINEL = "skillevaluator-file-backed-nvidia-key"
_NVIDIA_BUILD_KEY_FILE_ENV = "SKILLEVALUATOR_NVIDIA_API_KEY_FILE"
_NVIDIA_BUILD_BRIDGED_AGENT_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _reserve_run_dir(results_root: Path, timestamp: str) -> Path:
    """Atomically reserve and authenticate a unique run directory."""
    results_root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        run_id = f"{timestamp}_{os.getpid()}_{uuid4().hex[:12]}"
        run_dir = results_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        reservation_metadata = run_dir.lstat()
        reservation_identity = reservation_metadata.st_dev, reservation_metadata.st_ino
        try:
            mark_generated_output_root(run_dir)
        except Exception:
            if not remove_generated_output_root_if_owned(run_dir, expected_identity=reservation_identity):
                remove_output_reservation_if_identity_matches(run_dir, reservation_identity)
            raise
        return run_dir
    raise RuntimeError("Could not reserve a unique Tier 3 run directory")


_HARBOR_BASE_ENV_VARS = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
        "XDG_RUNTIME_DIR",
    }
)
_HARBOR_ENV_MODE_VARS = {
    "docker": frozenset(
        {
            "DOCKER_API_VERSION",
            "DOCKER_CERT_PATH",
            "DOCKER_CONFIG",
            "DOCKER_CONTEXT",
            "DOCKER_HOST",
            "DOCKER_TLS_VERIFY",
        }
    ),
    "daytona": frozenset(
        {
            "DAYTONA_API_KEY",
            "DAYTONA_API_URL",
            "DAYTONA_JWT_TOKEN",
            "DAYTONA_ORGANIZATION_ID",
            "DAYTONA_TARGET",
        }
    ),
    "e2b": frozenset({"E2B_API_KEY"}),
    "modal": frozenset({"MODAL_ENVIRONMENT", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}),
    "runloop": frozenset({"RUNLOOP_API_KEY"}),
    "langsmith": frozenset(
        {
            "LANGCHAIN_API_KEY",
            "LANGSMITH_API_KEY",
            "LANGSMITH_ENDPOINT",
            "LANGSMITH_PROFILE",
            "LANGSMITH_SANDBOX_API_URL",
        }
    ),
    "gke": frozenset(
        {"CLOUDSDK_CONFIG", "GCP_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "KUBECONFIG"}
    ),
    "novita": frozenset({"NOVITA_API_KEY", "NOVITA_API_URL", "NOVITA_BASE_URL", "NOVITA_DOMAIN"}),
    "islo": frozenset({"ISLO_API_KEY", "ISLO_API_URL", "ISLO_COMPUTE_URL"}),
    "tensorlake": frozenset({"TENSORLAKE_API_KEY"}),
    "cwsandbox": frozenset({"CWSANDBOX_API_KEY"}),
    "wandb": frozenset({"WANDB_API_KEY", "WANDB_BASE_URL"}),
    "use-computer": frozenset(
        {"USE_COMPUTER_API_KEY", "USE_COMPUTER_HOST", "USE_COMPUTER_SNAPSHOT", "USE_COMPUTER_VERSION"}
    ),
}
_BEDROCK_HOST_ENV_VARS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SDK_LOAD_CONFIG",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }
)
_RUNTIME_ENV_HOST_CONTROL_NAMES = (
    frozenset(
        {
            "ALL_PROXY",
            "BASHOPTS",
            "BASH_ENV",
            "CDPATH",
            "CLAUDE_CODE_DISABLE_POLICY_SKILLS",
            "CLAUDE_CONFIG_DIR",
            "CLASSPATH",
            "COMSPEC",
            "CODEX_HOME",
            "ENV",
            "GCONV_PATH",
            "GEMINI_CLI_HOME",
            "HOME",
            "HOSTALIASES",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "IFS",
            "JAVA_TOOL_OPTIONS",
            "LOCPATH",
            "LUA_CPATH",
            "LUA_INIT",
            "LUA_PATH",
            "NLSPATH",
            "NO_PROXY",
            "OPENCODE_CONFIG_DIR",
            "PATHEXT",
            "PATH",
            "PERL5LIB",
            "PERL5OPT",
            "REQUESTS_CA_BUNDLE",
            "RES_OPTIONS",
            "RUBYOPT",
            "RUBYLIB",
            "SHELLOPTS",
            "SSLKEYLOGFILE",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "SSH_AUTH_SOCK",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "USERPROFILE",
            "WINDIR",
            "XDG_CONFIG_HOME",
            "XDG_RUNTIME_DIR",
            "ZDOTDIR",
            "_JAVA_OPTIONS",
        }
    )
    | _BEDROCK_HOST_ENV_VARS
    | _VERIFIER_JUDGE_MODEL_ENV_VARS
    | frozenset().union(*_HARBOR_ENV_MODE_VARS.values())
)
_RUNTIME_ENV_HOST_CONTROL_PREFIXES = (
    "BASH_FUNC_",
    "COMPOSE_",
    "DOCKER_",
    "DYLD_",
    "GIT_",
    "HARBOR_",
    "LD_",
    "NODE_",
    "OTEL_",
    "PIP_",
    "PYTHON",
    "SKILL_EVAL_",
    "SKILLEVALUATOR_",
    "UV_",
)
_OPERATOR_OWNED_AGENT_ENV = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    }
)


@dataclass(frozen=True)
class AgentRuntimePlan:
    """One agent's immutable model, credential, and Harbor environment plan."""

    agent: str
    model: str
    provider: ProviderConfig
    staged_env: Mapping[str, str]
    subprocess_env: Mapping[str, str]


def _harbor_bin() -> str:
    """Return the Harbor executable installed with the active interpreter."""
    candidate = Path(os.sys.executable).parent / "harbor"
    return str(candidate) if candidate.exists() else (shutil.which("harbor") or "harbor")


def _harbor_supports_yes() -> bool:
    """Harbor 0.13.2, the supported Tier 3 dependency, accepts ``--yes``."""
    return True


def format_harbor_view_command(jobs_dir: Path | str, *, multiline: bool = False) -> str:
    """Return the portable command for inspecting retained Harbor artifacts."""
    path = shlex.quote(str(jobs_dir))
    command = "skillevaluator tier3 harbor-view"
    return f"{command} {path}" if not multiline else f"{command} \\\n  {path}"


@dataclass(frozen=True)
class _NvidiaBuildKeyHandoff:
    """Sanitized child environment plus an optional stdin credential payload."""

    subprocess_env: dict[str, str] = field(repr=False)
    stdin_text: str | None = field(default=None, repr=False, compare=False)


def _nvidia_build_key_handoff(
    run_env: Mapping[str, str],
    *,
    env_mode: str,
) -> _NvidiaBuildKeyHandoff:
    """Replace the host Build key with a stdin-backed sentinel."""
    subprocess_env = dict(run_env)
    subprocess_env.pop(_NVIDIA_BUILD_KEY_FILE_ENV, None)
    subprocess_env.pop(_NVIDIA_BUILD_KEY_STDIN_ENV, None)
    api_key = subprocess_env.get("NVIDIA_API_KEY", "")
    if (
        env_mode == "docker"
        and subprocess_env.get("SKILL_EVAL_LLM_PROVIDER") == "nv_build"
        and api_key
        and api_key not in {_NVIDIA_BUILD_FILE_SENTINEL, _NVIDIA_BUILD_STDIN_SENTINEL}
    ):
        subprocess_env["NVIDIA_API_KEY"] = _NVIDIA_BUILD_STDIN_SENTINEL
        subprocess_env[_NVIDIA_BUILD_KEY_STDIN_ENV] = "1"
        return _NvidiaBuildKeyHandoff(subprocess_env, api_key)
    return _NvidiaBuildKeyHandoff(subprocess_env)


def build_harbor_run_command(
    *,
    dataset_path: str | Path,
    agent: str,
    job_name: str,
    env_mode: str,
    n_attempts: int = 1,
    n_concurrent: int = 4,
    model: str | None = None,
    jobs_dir: Path | None = None,
    timeout_multiplier: float = 1.0,
    disable_verification: bool = False,
    include_task_names: list[str] | None = None,
    override_cpus: int | None = None,
    override_memory_mb: int | None = None,
    override_storage_mb: int | None = None,
    agent_import_path: str | None = None,
    verifier_env: Mapping[str, str] | None = None,
) -> list[str]:
    """Build a Harbor invocation for a built-in environment type or local mode."""
    if env_mode not in HARBOR_ENV_MODES:
        raise ValueError(f"env_mode must be one of: {', '.join(sorted(HARBOR_ENV_MODES))}")
    if agent_import_path and env_mode not in {"docker", ENV_MODE_LOCAL}:
        raise ValueError("agent_import_path is supported only with --env docker or local")

    command = [
        _harbor_bin(),
        "run",
        "--job-name",
        job_name,
        "--n-attempts",
        str(n_attempts),
        "--n-concurrent",
        str(n_concurrent),
        "-p",
        str(dataset_path),
    ]
    if env_mode == ENV_MODE_LOCAL:
        # Local mode is a custom SkillEvaluator environment + agent wrappers,
        # dispatched via import paths (not Harbor's --env), with sandbox knobs
        # passed as environment-kwargs (--ek). Harbor's create_agent_from_config
        # prefers the agent NAME when both -a and --agent-import-path are set, so
        # local mode passes ONLY --agent-import-path (its wrapper skips the
        # Debian apt-get bootstrap the stock agent runs) and never -a.
        from skillevaluator.tier3.harbor import LOCAL_AGENT_IMPORT_PATHS, LOCAL_ENV_IMPORT_PATH, local_sandbox
        from skillevaluator.tier3.harbor.local_runtime import default_runtime_root

        agent_import_path = agent_import_path or LOCAL_AGENT_IMPORT_PATHS.get(agent)
        if not agent_import_path:
            raise ValueError(f"--env-mode local does not support agent: {agent}")
        command.extend(["--agent-import-path", agent_import_path])
        command.extend(["--environment-import-path", LOCAL_ENV_IMPORT_PATH])
        command.extend(["--ek", f"runtime_root={default_runtime_root()}"])
        command.extend(["--ek", f"runtime_agent={agent}"])
        command.extend(["--ek", f"sandbox_mode={local_sandbox.resolve_mode(None)}"])
        command.extend(
            [
                "--ek",
                f"allow_net={str(local_sandbox.coerce_flag(None, env_var=local_sandbox.ALLOW_NET_ENV, default=True)).lower()}",
            ]
        )
        command.extend(
            [
                "--ek",
                f"strict_reads={str(local_sandbox.coerce_flag(None, env_var=local_sandbox.STRICT_READS_ENV)).lower()}",
            ]
        )
        command.extend(
            [
                "--ek",
                f"inherit_agent_keys={str(local_sandbox.coerce_flag(None, env_var=local_sandbox.INHERIT_AGENT_KEYS_ENV)).lower()}",
            ]
        )
    elif env_mode == "docker":
        if agent_import_path:
            command.extend(["--agent-import-path", agent_import_path])
        else:
            command.extend(["-a", agent])
        command.extend(["--environment-import-path", SECURE_DOCKER_ENV_IMPORT_PATH])
    else:
        command.extend(["-a", agent, "--env", env_mode])
    if jobs_dir is not None:
        command.extend(["--jobs-dir", str(jobs_dir)])
    if disable_verification:
        command.append("--disable-verification")
    for task_name in include_task_names or []:
        command.extend(["--include-task-name", task_name])
    if model:
        command.extend(["--model", model])
    if timeout_multiplier != 1.0:
        command.extend(["--timeout-multiplier", str(timeout_multiplier)])
    if override_cpus is not None:
        command.extend(["--override-cpus", str(override_cpus)])
    if override_memory_mb is not None:
        command.extend(["--override-memory-mb", str(override_memory_mb)])
    if override_storage_mb is not None:
        command.extend(["--override-storage-mb", str(override_storage_mb)])
    for name, value in sorted((verifier_env or {}).items()):
        command.extend(["--verifier-env", f"{name}={value}"])
    if _harbor_supports_yes():
        command.append("--yes")
    return command


def _provider_environment(config: ProviderConfig) -> dict[str, str]:
    """Build evaluator-owned verifier variables from provider config and host overrides."""
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": config.provider,
        "SKILL_EVAL_LLM_MODEL": config.model,
    }
    environment.update(
        {name: value for name in _VERIFIER_JUDGE_MODEL_ENV_VARS if (value := os.environ.get(name, "").strip())}
    )
    if config.provider == "anthropic":
        environment["ANTHROPIC_API_KEY"] = config.api_key or ""
        if config.base_url:
            environment["ANTHROPIC_BASE_URL"] = config.base_url
    elif config.provider == "bedrock":
        environment["AWS_REGION"] = config.region or "us-west-2"
        environment.update({name: os.environ[name] for name in _BEDROCK_HOST_ENV_VARS if os.environ.get(name)})
    elif config.provider == "nv_build":
        environment["NVIDIA_API_KEY"] = config.api_key or ""
    else:
        environment["OPENAI_API_KEY"] = config.api_key or ""
        environment["OPENAI_BASE_URL"] = config.base_url or ""
    return {name: value for name, value in environment.items() if value}


def _local_agent_credentials(config: ProviderConfig) -> dict[str, str]:
    """Map the resolved provider to the env vars local-mode agent CLIs read.

    opencode/codex read OPENAI_API_KEY/OPENAI_BASE_URL; claude-code reads
    ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL. NVIDIA Build is OpenAI-compatible, so
    it maps to the OPENAI_* pair pointing at its base URL.
    """
    if config.provider == "anthropic":
        env = {"ANTHROPIC_API_KEY": config.api_key or ""}
        if config.base_url:
            env["ANTHROPIC_BASE_URL"] = config.base_url
    else:  # openai, nv_build, or any OpenAI-compatible provider
        env = {"OPENAI_API_KEY": config.api_key or "", "OPENAI_BASE_URL": config.base_url or ""}
    return {name: value for name, value in env.items() if value}


def _validate_agent_provider_credentials(
    provider: ProviderConfig,
    agents: list[str],
    agent_runtime_env: dict[str, str],
    agent_model_sources: dict[str, str] | None = None,
    *,
    env_mode: str = DEFAULT_ENV_MODE,
    agent_models: Mapping[str, str] | None = None,
) -> list[str]:
    """Reject provider-to-agent combinations that cannot use the selected API."""
    model_sources = agent_model_sources or {}
    models = agent_models or {}

    opencode_model = models.get("opencode")
    expected_opencode_provider = {
        "anthropic": "anthropic",
        "nv_build": "nvidia",
        "openai": "openai",
        "openai-compatible": "openai",
    }.get(provider.provider)
    if (
        "opencode" in agents
        and opencode_model
        and expected_opencode_provider
        and "/" in opencode_model
        and opencode_model.split("/", maxsplit=1)[0].casefold() != expected_opencode_provider
    ):
        return [
            "OpenCode's provider-qualified model must match the evaluator provider so each agent route uses "
            "only its selected provider credential."
        ]

    if provider.provider != "nv_build":
        supported_agents = {
            "openai": {"claude-code", "codex", "opencode"},
            "openai-compatible": {"claude-code", "codex", "opencode"},
            "anthropic": {"claude-code", "codex", "opencode"},
            "bedrock": {"claude-code"},
        }.get(provider.provider, set())
        unsupported = [agent for agent in agents if agent not in supported_agents]
        if unsupported:
            return [
                f"{provider.provider} does not support live agent(s): {', '.join(unsupported)}. "
                "Choose a compatible evaluator provider and agent."
            ]
        if env_mode == ENV_MODE_LOCAL and provider.provider == "anthropic" and "opencode" in agents:
            return ["anthropic with opencode does not support local mode; use Docker/cloud or select claude-code."]
        if env_mode == ENV_MODE_LOCAL and provider.provider == "bedrock":
            return ["bedrock live agents do not support local mode; use Docker or a supported cloud backend."]
        if provider.provider == "bedrock" and "claude-code" in agents:
            has_bearer = bool(agent_runtime_env.get("AWS_BEARER_TOKEN_BEDROCK", "").strip())
            has_access_pair = bool(
                agent_runtime_env.get("AWS_ACCESS_KEY_ID", "").strip()
                and agent_runtime_env.get("AWS_SECRET_ACCESS_KEY", "").strip()
            )
            if not has_bearer and not has_access_pair:
                return [
                    "bedrock with claude-code requires an explicit AWS access-key pair or "
                    "AWS_BEARER_TOKEN_BEDROCK for the agent environment."
                ]

        if provider.provider in {"openai", "openai-compatible"} and "claude-code" in agents:
            if not agent_runtime_env.get("ANTHROPIC_API_KEY", "").strip():
                return [
                    "claude-code with the OpenAI evaluator provider requires an independent ANTHROPIC_API_KEY "
                    "in the operator host environment."
                ]
            if model_sources.get("claude-code", "public provider default") == "public provider default":
                return [
                    "claude-code needs an explicit Anthropic model when OpenAI is the evaluator provider; "
                    "set --agent-model claude-code=MODEL or harbor.agents.claude-code.model."
                ]

        if provider.provider == "anthropic":
            if "opencode" in agents and model_sources.get("opencode", "public provider default") == (
                "public provider default"
            ):
                return [
                    "opencode needs an explicit provider-qualified model when Anthropic is the evaluator provider; "
                    "set --agent-model opencode=PROVIDER/MODEL or harbor.agents.opencode.model."
                ]
            if "codex" in agents:
                openai_key = agent_runtime_env.get("OPENAI_API_KEY", "").strip()
                openai_base_url = agent_runtime_env.get("OPENAI_BASE_URL", "").strip()
                if not openai_key or not openai_base_url:
                    return [
                        "codex with the Anthropic evaluator provider requires independent OPENAI_API_KEY and "
                        "OPENAI_BASE_URL values in the operator host environment."
                    ]
                if model_sources.get("codex", "public provider default") == "public provider default":
                    return [
                        "codex needs an explicit OpenAI-compatible model when Anthropic is the evaluator provider; "
                        "set --agent-model codex=MODEL or harbor.agents.codex.model."
                    ]
        return []

    unsupported = [agent for agent in agents if agent not in {"claude-code", "codex", "opencode"}]
    if unsupported:
        return [
            "nv_build does not support live agent(s): "
            + ", ".join(unsupported)
            + ". Choose opencode, claude-code, or codex."
        ]

    if env_mode == ENV_MODE_LOCAL:
        from skillevaluator.tier3.harbor import local_sandbox

        if not local_sandbox.coerce_flag(None, env_var=local_sandbox.ALLOW_NET_ENV, default=True):
            return [
                "NVIDIA Build local agents require network access; unset SKILLEVALUATOR_LOCAL_ALLOW_NET or set it to 1."
            ]

    if env_mode in {"docker", ENV_MODE_LOCAL}:
        for agent in agents:
            model = models.get(agent, "")
            raw_model = model.removeprefix("nvidia/") if agent == "opencode" else model
            is_explicit = model_sources.get(agent) in {"CLI", "evals/config.yml"}
            if is_explicit and model and ("/" not in raw_model or raw_model.startswith("/") or raw_model.endswith("/")):
                return [
                    f"{agent} with NVIDIA Build requires a full NVIDIA Build catalog model ID "
                    "in publisher/model form; native provider model names are not routed by the compatibility bridge."
                ]
        # Codex and Claude Code use the in-container compatibility bridge;
        # local mode uses an authenticated in-process host bridge. OpenCode
        # continues to use NVIDIA Build's native provider adapter.
        return []

    if "claude-code" in agents:
        if not agent_runtime_env.get("ANTHROPIC_API_KEY", "").strip():
            return [
                "claude-code with NVIDIA Build requires an independent ANTHROPIC_API_KEY in the agent runtime "
                "environment; NVIDIA_API_KEY is not an Anthropic credential."
            ]
        model_source = (agent_model_sources or {}).get("claude-code", "public provider default")
        if model_source == "public provider default":
            return [
                "claude-code needs an explicit Anthropic model when NVIDIA Build is the evaluator provider; "
                "set --agent-model claude-code=MODEL or harbor.agents.claude-code.model."
            ]

    if "codex" not in agents:
        return []

    # NVIDIA Build exposes /v1/responses, but only for basic function tools — it
    # rejects codex-cli's namespace/multi-agent tool schema (`unified_exec`), so
    # codex cannot complete a run against NVIDIA Build. codex needs a full
    # OpenAI-compatible Responses provider; require the user to supply one.
    openai_key = agent_runtime_env.get("OPENAI_API_KEY", "").strip()
    openai_base_url = agent_runtime_env.get("OPENAI_BASE_URL", "").rstrip("/")
    if not openai_key or not openai_base_url or openai_base_url == (provider.base_url or "").rstrip("/"):
        return [
            "codex requires a full OpenAI Responses API credential — NVIDIA Build's /responses does not "
            "support codex's tool schema. Set OPENAI_API_KEY + OPENAI_BASE_URL to an OpenAI-compatible "
            "Responses provider (e.g. https://api.openai.com/v1) in the operator's host environment for Codex."
        ]

    model_source = (agent_model_sources or {}).get("codex", "public provider default")
    if model_source == "public provider default":
        return [
            "codex needs an explicit OpenAI-compatible model when NVIDIA Build is the evaluator provider; "
            "set --agent-model codex=MODEL or harbor.agents.codex.model."
        ]
    return []


def _check_prerequisites(
    env_mode: str = DEFAULT_ENV_MODE,
    agents: list[str] | None = None,
) -> list[str]:
    """Check Harbor and the selected environment (built-in or local mode)."""
    if env_mode not in HARBOR_ENV_MODES:
        return [f"Unsupported Harbor environment '{env_mode}'. Choose one of: {', '.join(sorted(HARBOR_ENV_MODES))}"]
    if env_mode == ENV_MODE_LOCAL:
        from skillevaluator.tier3.harbor import local_sandbox

        try:
            local_sandbox.require_supported_platform()
        except local_sandbox.SandboxUnavailable as exc:
            return [str(exc)]
    executable = _harbor_bin()
    if executable == "harbor" and shutil.which(executable) is None:
        return [
            "harbor CLI not found. Reinstall with the Tier 3 extra: "
            'uv tool install "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
        ]

    if env_mode == ENV_MODE_LOCAL:
        # Local mode is a host sandbox, not a Harbor-native backend: verify the
        # OS sandbox is usable and the requested agent CLIs are installed.
        from skillevaluator.tier3.harbor import local_sandbox
        from skillevaluator.tier3.harbor.local_runtime import ensure_local_runtimes

        try:
            sandbox = local_sandbox.detect(local_sandbox.resolve_mode(None))
        except local_sandbox.SandboxUnavailable as exc:
            return [str(exc)]
        except ValueError as exc:
            return [f"Invalid local sandbox configuration: {exc}"]
        from skillevaluator.tier3.harbor.local_runtime import validate_local_agents

        selected_agents = agents or []
        unsupported = validate_local_agents(selected_agents)
        if unsupported:
            return [f"Local mode supports only claude-code, codex, opencode. Unsupported: {', '.join(unsupported)}."]
        try:
            strict_reads = local_sandbox.coerce_flag(None, env_var=local_sandbox.STRICT_READS_ENV)
            return ensure_local_runtimes(selected_agents, sandbox=sandbox, strict_reads=strict_reads)
        except ValueError as exc:
            return [f"Invalid local runtime configuration: {exc}"]

    if env_mode == "docker":
        try:
            compose = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"Docker Compose v2 is required for Tier 3 Docker mode: {exc}"]
        if compose.returncode != 0:
            detail = (compose.stderr or compose.stdout).strip()
            suffix = f": {detail}" if detail else ""
            return [f"Docker Compose v2 is required for Tier 3 Docker mode{suffix}"]

    try:
        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.environment_type import EnvironmentType

        EnvironmentFactory.run_preflight(EnvironmentType(env_mode))
    except ImportError as exc:
        return [
            f"Harbor environment '{env_mode}' needs optional dependencies: {exc}. "
            "Install the matching Harbor environment extra."
        ]
    except SystemExit as exc:
        detail = " ".join(str(exc).split()) or "preflight exited without a diagnostic"
        return [f"Harbor environment '{env_mode}' is not ready: {detail}"]
    except Exception as exc:
        return [f"Harbor environment '{env_mode}' is not ready: {exc}"]
    return []


def _is_operator_owned_runtime_name(name: str) -> bool:
    normalized = name.upper()
    return (
        normalized in _RUNTIME_ENV_HOST_CONTROL_NAMES
        or normalized in _OPERATOR_OWNED_AGENT_ENV
        or normalized.startswith(_RUNTIME_ENV_HOST_CONTROL_PREFIXES)
    )


def _resolve_runtime_env(templates: dict[str, str] | None) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    errors: list[str] = []
    for name, template in (templates or {}).items():
        if _is_operator_owned_runtime_name(name):
            errors.append(f"harbor.runtime_env.{name} controls the host process and is not allowed")
            continue
        template_value = str(template)
        dollar_references = {
            braced or plain
            for braced, plain in re.findall(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)\b",
                template_value,
            )
        }
        percent_references = set(re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)%", template_value))
        references = dollar_references | percent_references
        owned_references = sorted(reference for reference in references if _is_operator_owned_runtime_name(reference))
        if owned_references:
            errors.append(
                f"harbor.runtime_env.{name} references operator-owned credential(s): " + ", ".join(owned_references)
            )
            continue
        value = os.path.expandvars(template_value)
        if "$" in value:
            errors.append(f"harbor.runtime_env.{name} references an unset environment variable")
        else:
            resolved[name] = value
    return resolved, errors


def _selected_host_environment(names: set[str] | frozenset[str], source: Mapping[str, str]) -> dict[str, str]:
    return {name: source[name] for name in names if source.get(name)}


def _harbor_subprocess_environment(
    *,
    env_mode: str,
    provider: ProviderConfig,
    configured_runtime_env: Mapping[str, str],
    provider_env: Mapping[str, str],
    agent: str | None = None,
    agent_model: str | None = None,
) -> dict[str, str]:
    """Build Harbor's minimal host environment without ambient secrets."""
    host_env = os.environ
    environment = _selected_host_environment(_HARBOR_BASE_ENV_VARS, host_env)
    environment.update(_selected_host_environment(_HARBOR_ENV_MODE_VARS.get(env_mode, frozenset()), host_env))
    if provider.provider == "bedrock":
        environment.update(_selected_host_environment(_BEDROCK_HOST_ENV_VARS, host_env))
    environment.update(configured_runtime_env)
    environment.update(provider_env)
    if env_mode == ENV_MODE_LOCAL:
        from skillevaluator.tier3.harbor.local_runtime import local_subprocess_env

        local_credentials = _local_agent_credentials(provider)
        if provider.provider == "nv_build" and agent == "opencode" and (agent_model or "").startswith("nvidia/"):
            # OpenCode's NVIDIA adapter reads OPENAI_* internally. Override an
            # independent Codex pair for this Harbor subprocess only; each
            # selected local agent receives its own environment below.
            environment.pop("ANTHROPIC_API_KEY", None)
            environment.pop("ANTHROPIC_BASE_URL", None)
            environment.update(local_credentials)
        elif provider.provider == "nv_build" and agent in {"codex", "claude-code"}:
            # The trusted Harbor parent keeps NVIDIA_API_KEY for the verifier
            # and in-process bridge. Vendor children receive only the bridge's
            # per-trial capability token, never ambient native credentials.
            for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
                environment.pop(name, None)
        else:
            # Never synthesize the missing half of a configured independent
            # OpenAI pair from NVIDIA Build. Shared preflight rejects partial
            # Codex credentials before Harbor starts.
            configured_openai = {
                name for name in configured_runtime_env if name in {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
            }
            for name, value in local_credentials.items():
                if provider.provider == "nv_build" and configured_openai and name.startswith("OPENAI_"):
                    continue
                environment.setdefault(name, value)
        environment = local_subprocess_env(runtime_agents=[agent] if agent else None, base_env=environment)
    return environment


def _independent_anthropic_agent_credentials() -> dict[str, str]:
    """Resolve and validate a host-owned Anthropic credential pair."""
    credentials = {
        name: os.environ.get(name, "") for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL") if os.environ.get(name)
    }
    if base_url := credentials.get("ANTHROPIC_BASE_URL"):
        normalized_base_url = _normalize_anthropic_base_url(
            base_url,
            variable="ANTHROPIC_BASE_URL",
        )
        if normalized_base_url is None:
            credentials.pop("ANTHROPIC_BASE_URL")
        else:
            credentials["ANTHROPIC_BASE_URL"] = normalized_base_url
    return credentials


def _judge_model_config(
    provider: ProviderConfig,
    provider_env: Mapping[str, str],
    grading_mode: str,
) -> dict[str, str | bool]:
    """Describe the configured standard-grading judge before any provider fallback."""
    if grading_mode == "custom_only":
        return {"enabled": False}
    for name in ("LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"):
        if model := provider_env.get(name):
            return {
                "enabled": True,
                "provider": provider.provider,
                "model": model,
                "source": name,
                "override_applied": True,
            }
    return {
        "enabled": True,
        "provider": provider.provider,
        "model": provider.model,
        "source": (
            "SKILL_EVAL_LLM_MODEL" if os.environ.get("SKILL_EVAL_LLM_MODEL", "").strip() else "provider default"
        ),
        "override_applied": False,
    }


def _job_judge_override(provider_env: Mapping[str, str], grading_mode: str) -> tuple[str, str] | None:
    """Return the selected dedicated host override name and value, if enabled."""
    if grading_mode == "custom_only":
        return None
    for name in ("LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"):
        if value := provider_env.get(name):
            return name, value
    return None


def _job_judge_verifier_env(provider_env: Mapping[str, str], grading_mode: str) -> dict[str, str]:
    """Return placeholder-based judge overrides for Harbor's verifier job layer."""
    selected = _job_judge_override(provider_env, grading_mode)
    if selected is None:
        return {}
    source, _value = selected
    # Harbor resolves every task-authored placeholder before verifier startup.
    # Override both spellings from the selected host source so a stale alias
    # cannot fail resolution or survive task/step/job environment merging.
    return dict.fromkeys(sorted(_VERIFIER_JUDGE_MODEL_ENV_VARS), f"${{{source}}}")


def _job_judge_subprocess_env(provider_env: Mapping[str, str], grading_mode: str) -> dict[str, str]:
    """Make both aliases resolvable while Harbor constructs verifier environments."""
    selected = _job_judge_override(provider_env, grading_mode)
    if selected is None:
        return {}
    _source, value = selected
    return dict.fromkeys(_VERIFIER_JUDGE_MODEL_ENV_VARS, value)


def _agent_credentials(
    *,
    provider: ProviderConfig,
    agent: str,
    env_mode: str,
) -> dict[str, str]:
    """Resolve operator-owned credentials for exactly one agent runtime."""
    if provider.provider == "nv_build":
        if agent == "opencode":
            if env_mode == ENV_MODE_LOCAL:
                return _local_agent_credentials(provider)
            return {"NVIDIA_API_KEY": provider.api_key or ""}
        if env_mode in {"docker", ENV_MODE_LOCAL} and agent in {"claude-code", "codex"}:
            # The Docker bridge wrapper reads the evaluator credential from
            # the Harbor parent handoff; the vendor CLI receives only a local
            # sentinel and must not inherit NVIDIA_API_KEY in task env.
            return {}
        if agent == "claude-code":
            return _independent_anthropic_agent_credentials()
        if agent == "codex":
            return {
                name: os.environ.get(name, "") for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if os.environ.get(name)
            }
        return {}

    if provider.provider in {"openai", "openai-compatible"} and agent == "claude-code":
        return _independent_anthropic_agent_credentials()
    if provider.provider == "anthropic" and agent == "codex":
        return {
            name: os.environ.get(name, "") for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if os.environ.get(name)
        }

    if provider.provider == "anthropic" and agent in {"claude-code", "opencode"}:
        return {
            name: value
            for name, value in {
                "ANTHROPIC_API_KEY": provider.api_key or "",
                "ANTHROPIC_BASE_URL": provider.base_url or "",
            }.items()
            if value
        }
    if provider.provider in {"openai", "openai-compatible"} and agent in {"codex", "opencode"}:
        return {
            name: value
            for name, value in {
                "OPENAI_API_KEY": provider.api_key or "",
                "OPENAI_BASE_URL": provider.base_url or "",
            }.items()
            if value
        }
    if provider.provider == "bedrock" and agent == "claude-code":
        credentials = {
            name: value for name, value in _provider_environment(provider).items() if name.startswith("AWS_") and value
        }
        credentials["CLAUDE_CODE_USE_BEDROCK"] = "1"
        return credentials
    return {}


def _agent_provider_config(
    *,
    evaluator_provider: ProviderConfig,
    agent: str,
    model: str,
    credentials: Mapping[str, str],
    env_mode: str,
) -> ProviderConfig:
    """Describe the API provider the selected agent will actually call."""
    if evaluator_provider.provider in {"openai", "openai-compatible"} and agent == "claude-code":
        resolved_model = model.removeprefix("anthropic/")
        return ProviderConfig(
            provider="anthropic",
            model=resolved_model,
            api_key=credentials.get("ANTHROPIC_API_KEY"),
            base_url=credentials.get("ANTHROPIC_BASE_URL"),
            litellm_model=f"anthropic/{resolved_model}",
        )
    if evaluator_provider.provider == "anthropic" and agent == "codex":
        resolved_model = model.removeprefix("openai/")
        return ProviderConfig(
            provider="openai-compatible",
            model=resolved_model,
            api_key=credentials.get("OPENAI_API_KEY"),
            base_url=credentials.get("OPENAI_BASE_URL"),
            litellm_model=f"openai/{resolved_model}",
        )
    if (
        evaluator_provider.provider == "nv_build"
        and agent == "claude-code"
        and env_mode
        not in {
            "docker",
            ENV_MODE_LOCAL,
        }
    ):
        resolved_model = model.removeprefix("anthropic/")
        return ProviderConfig(
            provider="anthropic",
            model=resolved_model,
            api_key=credentials.get("ANTHROPIC_API_KEY"),
            base_url=credentials.get("ANTHROPIC_BASE_URL"),
            litellm_model=f"anthropic/{resolved_model}",
        )
    if (
        evaluator_provider.provider == "nv_build"
        and agent == "codex"
        and env_mode
        not in {
            "docker",
            ENV_MODE_LOCAL,
        }
    ):
        resolved_model = model.removeprefix("openai/")
        return ProviderConfig(
            provider="openai-compatible",
            model=resolved_model,
            api_key=credentials.get("OPENAI_API_KEY"),
            base_url=credentials.get("OPENAI_BASE_URL"),
            litellm_model=f"openai/{resolved_model}",
        )
    if agent == "opencode":
        runtime_namespaces = {
            "anthropic": "anthropic/",
            "nv_build": "nvidia/",
            "openai": "openai/",
            "openai-compatible": "openai/",
        }
        resolved_model = model.removeprefix(runtime_namespaces.get(evaluator_provider.provider, ""))
    else:
        resolved_model = model
    default_prefix = "anthropic" if evaluator_provider.provider == "anthropic" else evaluator_provider.provider
    litellm_prefix = getattr(evaluator_provider, "litellm_model", f"{default_prefix}/{resolved_model}").partition("/")[
        0
    ]
    return ProviderConfig(
        provider=evaluator_provider.provider,
        model=resolved_model,
        api_key=evaluator_provider.api_key,
        base_url=evaluator_provider.base_url,
        litellm_model=f"{litellm_prefix}/{resolved_model}",
        region=getattr(evaluator_provider, "region", None),
    )


def _resolve_agent_runtime_plan(
    *,
    provider: ProviderConfig,
    agents: list[str],
    models: Mapping[str, str],
    configured_runtime_env: Mapping[str, str],
    env_mode: str,
    model_sources: Mapping[str, str] | None = None,
) -> dict[str, AgentRuntimePlan]:
    """Resolve the single credential plan used by staging and execution.

    Skill-owned configuration may add non-credential runtime values, but agent
    and provider credentials always come from the operator's selected provider
    or host environment. This prevents a skill from replacing a credential or
    routing a trusted key to an attacker-controlled endpoint.
    """
    collisions = sorted(_OPERATOR_OWNED_AGENT_ENV.intersection(configured_runtime_env))
    if collisions:
        names = ", ".join(collisions)
        raise ValueError(f"harbor.runtime_env contains operator-owned credential name(s): {names}")

    # Dedicated judge aliases are job-scoped. Standard grading adds the
    # selected alias at launch time, while custom-only grading must not retain
    # it in the reusable Harbor parent environment.
    provider_env = {
        name: value
        for name, value in _provider_environment(provider).items()
        if name not in _VERIFIER_JUDGE_MODEL_ENV_VARS
    }
    plans: dict[str, AgentRuntimePlan] = {}
    for agent in agents:
        credentials = _agent_credentials(provider=provider, agent=agent, env_mode=env_mode)
        validation_env = {**configured_runtime_env, **credentials}
        credential_errors = _validate_agent_provider_credentials(
            provider,
            [agent],
            validation_env,
            dict(model_sources or {}),
            env_mode=env_mode,
            agent_models={agent: models[agent]},
        )
        if credential_errors:
            raise ValueError(credential_errors[0])

        subprocess_env = _harbor_subprocess_environment(
            env_mode=env_mode,
            provider=provider,
            configured_runtime_env=configured_runtime_env,
            provider_env=provider_env,
            agent=agent,
            agent_model=models[agent],
        )
        subprocess_env.update(credentials)
        staged = {name: f"${{{name}}}" for name in (*configured_runtime_env, *credentials)}
        plans[agent] = AgentRuntimePlan(
            agent=agent,
            model=models[agent],
            provider=_agent_provider_config(
                evaluator_provider=provider,
                agent=agent,
                model=models[agent],
                credentials=credentials,
                env_mode=env_mode,
            ),
            staged_env=MappingProxyType(staged),
            subprocess_env=MappingProxyType(subprocess_env),
        )
    return plans


def _is_skill_dir(path: Path) -> bool:
    return path.is_dir() and (path / "SKILL.md").is_file()


def _workspace_skills(skill_path: Path, values: list[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in values:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = skill_path.parent / candidate
        candidate = candidate.resolve()
        options = (
            [candidate]
            if _is_skill_dir(candidate)
            else sorted(path for path in candidate.iterdir() if _is_skill_dir(path))
            if candidate.is_dir()
            else []
        )
        if not options:
            raise ValueError(f"Included skill path is not a skill or skill directory: {raw}")
        for option in options:
            if option != skill_path and option not in seen:
                resolved.append(option)
                seen.add(option)
    return resolved


def _task_timeout_plan(task_roots: list[Path], timeout_multiplier: float) -> float | None:
    """Return the largest staged agent timeout after applying Harbor scaling."""
    timeouts: list[float] = []
    for root in task_roots:
        for task_file in root.glob("*/task.toml"):
            try:
                data = tomllib.loads(task_file.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                continue
            agent = data.get("agent") if isinstance(data, dict) else None
            value = agent.get("timeout_sec") if isinstance(agent, dict) else None
            if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
                timeouts.append(float(value))
    return round(max(timeouts) * timeout_multiplier, 3) if timeouts else None


def _model_for_agent(
    agent: str,
    *,
    cli_model: str | None,
    config_agents: dict[str, Any],
    provider: ProviderConfig,
) -> tuple[str, str]:
    if cli_model:
        selected, source = cli_model, "CLI"
    else:
        configured = config_agents.get(agent, {}) if isinstance(config_agents, dict) else {}
        if isinstance(configured, dict) and configured.get("model"):
            selected, source = str(configured["model"]), "evals/config.yml"
        else:
            selected, source = provider.model, "public provider default"
    if agent in {"codex", "claude-code"} and provider.provider == "nv_build" and source == "public provider default":
        # Nano is the cost-conscious default for Build itself, but in real
        # bridged tool loops it failed to execute the target skill. Super is
        # the smallest verified default for these compatibility bridges;
        # explicit overrides remain exact.
        selected = _NVIDIA_BUILD_BRIDGED_AGENT_DEFAULT_MODEL
    if agent == "opencode":
        namespace = {
            "anthropic": "anthropic",
            "nv_build": "nvidia",
            "openai": "openai",
            "openai-compatible": "openai",
        }.get(provider.provider)
        if namespace and source == "public provider default":
            selected = f"{namespace}/{selected}"
    return selected, source


def _nvidia_build_agent_import_path(provider: ProviderConfig, agent: str, env_mode: str) -> str | None:
    """Return the environment-specific NVIDIA Build compatibility wrapper."""
    if provider.provider != "nv_build":
        return None
    from skillevaluator.tier3.harbor.local_agents import (
        NVIDIA_BUILD_AGENT_IMPORT_PATHS,
        NVIDIA_BUILD_LOCAL_AGENT_IMPORT_PATHS,
    )

    if env_mode == "docker":
        return NVIDIA_BUILD_AGENT_IMPORT_PATHS.get(agent)
    if env_mode == ENV_MODE_LOCAL:
        return NVIDIA_BUILD_LOCAL_AGENT_IMPORT_PATHS.get(agent)
    return None


def _run_harbor(
    *,
    dataset: Path,
    agent: str,
    job_name: str,
    env_mode: str,
    model: str,
    jobs_dir: Path,
    run_env: dict[str, str],
    n_attempts: int,
    n_concurrent: int,
    timeout_multiplier: float,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
    agent_import_path: str | None = None,
    verifier_env: Mapping[str, str] | None = None,
    expected_trials: int | None = None,
    expected_total_trials: int | None = None,
    include_task_names: list[str] | None = None,
) -> tuple[bool, str]:
    command = build_harbor_run_command(
        dataset_path=dataset,
        agent=agent,
        job_name=job_name,
        env_mode=env_mode,
        n_attempts=n_attempts,
        n_concurrent=n_concurrent,
        model=model,
        jobs_dir=jobs_dir,
        timeout_multiplier=timeout_multiplier,
        include_task_names=include_task_names,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        override_storage_mb=override_storage_mb,
        agent_import_path=agent_import_path,
        verifier_env=verifier_env,
    )
    try:
        handoff = _nvidia_build_key_handoff(run_env, env_mode=env_mode)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            input=handoff.stdin_text,
            env=handoff.subprocess_env,
            timeout=7200,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode == 0:
        return _validate_harbor_job_result(
            jobs_dir,
            job_name,
            expected_trials=expected_trials,
            expected_total_trials=expected_total_trials,
        )
    output = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
    detail = output[-2000:] or f"harbor run exited {result.returncode}"
    return False, redact_progress_detail(detail, secret_values=set(run_env.values()))


def _validate_harbor_job_result(
    jobs_dir: Path,
    job_name: str,
    *,
    expected_trials: int | None = None,
    expected_total_trials: int | None = None,
) -> tuple[bool, str]:
    """Require Harbor's persisted trial state to be complete and error-free."""
    return validate_harbor_job_result(
        jobs_dir / job_name / "result.json",
        expected_trials=expected_trials,
        expected_total_trials=expected_total_trials,
    )


def _job_passed(job_dir: Path, pass_threshold: float) -> bool:
    """Use collector-authoritative logical-attempt semantics for early stop."""
    return harbor_job_passed(job_dir, pass_threshold)


def _attempt_job_stats(
    job_dir: Path,
) -> tuple[int, int, int, dict[str, tuple[int, int, dict[str, dict[str, list[str]]]]]] | None:
    """Read one per-attempt Harbor job result for merging; ``None`` when unreadable."""
    try:
        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(result, dict):
        return None
    total = result.get("n_total_trials")
    stats = result.get("stats")
    if not isinstance(total, int) or isinstance(total, bool) or not isinstance(stats, dict):
        return None
    if any(key in stats for key in ("n_completed_trials", "n_errored_trials")):
        completed = stats.get("n_completed_trials")
        errored = stats.get("n_errored_trials")
    else:
        completed = stats.get("n_trials")
        errored = stats.get("n_errors")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (completed, errored)):
        return None

    evals_out: dict[str, tuple[int, int, dict[str, dict[str, list[str]]]]] = {}
    evals = stats.get("evals")
    if isinstance(evals, dict):
        for eval_name, eval_stats in evals.items():
            if not isinstance(eval_stats, dict):
                continue
            n_trials = eval_stats.get("n_trials")
            n_errors = eval_stats.get("n_errors")
            reward_stats = eval_stats.get("reward_stats")
            per_metric: dict[str, dict[str, list[str]]] = {}
            if isinstance(reward_stats, dict):
                for metric, buckets in reward_stats.items():
                    if not isinstance(buckets, dict):
                        continue
                    per_metric[str(metric)] = {
                        str(bucket): [str(name) for name in names]
                        for bucket, names in buckets.items()
                        if isinstance(names, list)
                    }
            evals_out[str(eval_name)] = (
                n_trials if isinstance(n_trials, int) and not isinstance(n_trials, bool) else 0,
                n_errors if isinstance(n_errors, int) and not isinstance(n_errors, bool) else 0,
                per_metric,
            )
    return total, completed, errored, evals_out


def _job_path_is_link_or_reparse(path: Path, metadata: os.stat_result) -> bool:
    """Return whether an attempt-job root is a symlink, junction, or reparse point."""
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag):
        return True
    is_junction = getattr(path, "is_junction", None)
    if not callable(is_junction):
        return False
    try:
        return bool(is_junction())
    except (OSError, RuntimeError):
        return True


def _job_root_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _merge_attempt_jobs(job_dirs: list[Path], aggregate_dir: Path) -> None:
    """Merge per-attempt Harbor jobs into the job directory shape collection expects.

    Trial directories are copied under attempt-qualified names and the
    per-attempt Harbor ``result.json`` statistics are combined so the merged
    job still satisfies :func:`validate_harbor_job_result`.
    """
    aggregate_path = Path(os.path.abspath(aggregate_dir))  # noqa: PTH100 -- compare lexical publication roots
    source_paths: list[tuple[str, Path, Path, tuple[int, int, int, int, int, int]]] = []
    for job_dir in job_dirs:
        job_path = Path(os.path.abspath(job_dir))  # noqa: PTH100 -- reject overlap before temp creation
        if not os.path.lexists(job_path):
            continue
        try:
            metadata = job_path.lstat()
        except OSError as exc:
            raise ValueError(f"cannot inspect attempt Harbor job root: {job_path}") from exc
        if _job_path_is_link_or_reparse(job_path, metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"attempt Harbor job root must be a non-linked directory: {job_path}")
        try:
            job_resolved = job_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"cannot resolve attempt Harbor job root: {job_path}") from exc
        aggregate_resolved = aggregate_path.resolve(strict=False)
        if (
            aggregate_resolved == job_resolved
            or aggregate_resolved.is_relative_to(job_resolved)
            or job_resolved.is_relative_to(aggregate_resolved)
        ):
            raise ValueError("aggregate Harbor job directory must not overlap an attempt job directory")
        source_paths.append((job_path.name, job_path, job_resolved, _job_root_fingerprint(metadata)))

    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{aggregate_path.name}-merge-",
        dir=aggregate_path.parent,
    ) as private_root_raw:
        private_root = Path(private_root_raw)
        snapshot_root = private_root / "attempt-jobs"
        snapshot_root.mkdir()
        staged_aggregate = private_root / "aggregate"
        staged_aggregate.mkdir()

        snapshots: list[tuple[str, Path]] = []
        for index, (job_name, job_path, job_resolved, expected_fingerprint) in enumerate(source_paths):
            snapshot = snapshot_root / f"{index:04d}-{job_name}"
            try:
                before = job_path.lstat()
            except OSError as exc:
                raise ValueError(f"attempt Harbor job root changed before snapshot: {job_path}") from exc
            if _job_path_is_link_or_reparse(job_path, before) or _job_root_fingerprint(before) != expected_fingerprint:
                raise ValueError(f"attempt Harbor job root changed before snapshot: {job_path}")
            copytree_secure(job_path, snapshot, allowed_root=job_resolved)
            try:
                after = job_path.lstat()
            except OSError as exc:
                raise ValueError(f"attempt Harbor job root changed during snapshot: {job_path}") from exc
            if _job_path_is_link_or_reparse(job_path, after) or _job_root_fingerprint(after) != expected_fingerprint:
                raise ValueError(f"attempt Harbor job root changed during snapshot: {job_path}")
            snapshots.append((job_name, snapshot))

        total_trials = 0
        completed_trials = 0
        errored_trials = 0
        merged_evals: dict[str, dict[str, Any]] = {}
        for job_name, job_dir in snapshots:
            renamed: dict[str, str] = {}
            for child in sorted(job_dir.iterdir()):
                if not child.is_dir():
                    continue
                dest = staged_aggregate / f"{job_name}__{child.name}"
                suffix = 2
                while dest.exists():
                    dest = staged_aggregate / f"{job_name}__{child.name}-{suffix}"
                    suffix += 1
                copytree_secure(child, dest, allowed_root=job_dir)
                renamed[child.name] = dest.name

            stats = _attempt_job_stats(job_dir)
            if stats is None:
                continue
            job_total, job_completed, job_errored, job_evals = stats
            total_trials += job_total
            completed_trials += job_completed
            errored_trials += job_errored
            for eval_name, (eval_trials, eval_errors, reward_stats) in job_evals.items():
                merged = merged_evals.setdefault(eval_name, {"n_trials": 0, "n_errors": 0, "reward_stats": {}})
                merged["n_trials"] += eval_trials
                merged["n_errors"] += eval_errors
                for metric, buckets in reward_stats.items():
                    merged_buckets = merged["reward_stats"].setdefault(metric, {})
                    for bucket, trial_names in buckets.items():
                        merged_buckets.setdefault(bucket, []).extend(
                            renamed.get(name, f"{job_name}__{name}") for name in trial_names
                        )

        (staged_aggregate / "result.json").write_text(
            json.dumps(
                {
                    "n_total_trials": total_trials,
                    "stats": {
                        "n_trials": completed_trials,
                        "n_errors": errored_trials,
                        "evals": merged_evals,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        copytree_secure(
            staged_aggregate,
            aggregate_path,
            replace_existing=aggregate_path.exists(),
            allowed_root=private_root,
        )


def _run_stop_on_pass_variant(
    *,
    skill_name: str,
    agent: str,
    variant: str,
    dataset: Path,
    task_names: list[str],
    env_mode: str,
    model: str,
    jobs_dir: Path,
    run_env: dict[str, str],
    n_attempts: int,
    pass_threshold: float,
    timeout_multiplier: float,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
    agent_import_path: str | None = None,
    verifier_env: Mapping[str, str] | None = None,
) -> list[str]:
    """Run each case one attempt at a time, stopping its attempts on first pass."""
    errors: list[str] = []
    attempt_job_dirs: list[Path] = []
    for task_name in task_names:
        for attempt in range(1, n_attempts + 1):
            job_name = f"{skill_name}-{agent}-{variant}-{task_name}-attempt{attempt:03d}"
            ok, detail = _run_harbor(
                dataset=dataset,
                agent=agent,
                job_name=job_name,
                env_mode=env_mode,
                model=model,
                jobs_dir=jobs_dir,
                run_env=run_env,
                n_attempts=1,
                n_concurrent=1,
                timeout_multiplier=timeout_multiplier,
                override_cpus=override_cpus,
                override_memory_mb=override_memory_mb,
                override_storage_mb=override_storage_mb,
                agent_import_path=agent_import_path,
                verifier_env=verifier_env,
                expected_trials=1,
                include_task_names=[task_name],
            )
            job_dir = jobs_dir / job_name
            attempt_job_dirs.append(job_dir)
            if not ok:
                errors.append(f"{agent} {variant}-skill Harbor run failed: {task_name} attempt {attempt}: {detail}")
                continue
            if _job_passed(job_dir, pass_threshold):
                break
    _merge_attempt_jobs(attempt_job_dirs, jobs_dir / f"{skill_name}-{agent}-{variant}")
    return errors


def _run_agent_pair(
    *,
    skill_name: str,
    agent: str,
    model: str,
    env_mode: str,
    with_skill: Path,
    baseline: Path | None,
    jobs_dir: Path,
    run_env: dict[str, str],
    n_attempts: int,
    n_concurrent: int,
    timeout_multiplier: float,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
    expected_trials: int,
    agent_import_path: str | None = None,
    stop_on_pass: bool = False,
    pass_threshold: float = 0.50,
    task_names: list[str] | None = None,
    verifier_env: Mapping[str, str] | None = None,
) -> list[str]:
    jobs = [("with", with_skill)]
    if baseline is not None:
        jobs.append(("without", baseline))
    if stop_on_pass:
        # A later attempt is launched only after the previous one scored, so
        # stop-on-pass runs each condition sequentially, one attempt at a time.
        sequential_errors: list[str] = []
        for variant, dataset in jobs:
            sequential_errors.extend(
                _run_stop_on_pass_variant(
                    skill_name=skill_name,
                    agent=agent,
                    variant=variant,
                    dataset=dataset,
                    task_names=list(task_names or []),
                    env_mode=env_mode,
                    model=model,
                    jobs_dir=jobs_dir,
                    run_env=run_env,
                    n_attempts=n_attempts,
                    pass_threshold=pass_threshold,
                    timeout_multiplier=timeout_multiplier,
                    override_cpus=override_cpus,
                    override_memory_mb=override_memory_mb,
                    override_storage_mb=override_storage_mb,
                    agent_import_path=agent_import_path,
                    verifier_env=verifier_env,
                )
            )
        return sequential_errors
    # The advertised concurrency is one per-agent trial budget. Split it
    # across concurrently running conditions instead of multiplying it by two.
    worker_count = min(len(jobs), n_concurrent)
    if worker_count == len(jobs):
        concurrency_per_job, extra_slots = divmod(n_concurrent, len(jobs))
        job_concurrency = [concurrency_per_job + (1 if index < extra_slots else 0) for index in range(len(jobs))]
    else:
        job_concurrency = [1] * len(jobs)
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_harbor,
                dataset=dataset,
                agent=agent,
                job_name=f"{skill_name}-{agent}-{variant}",
                env_mode=env_mode,
                model=model,
                jobs_dir=jobs_dir,
                run_env=run_env,
                n_attempts=n_attempts,
                n_concurrent=condition_concurrency,
                timeout_multiplier=timeout_multiplier,
                override_cpus=override_cpus,
                override_memory_mb=override_memory_mb,
                override_storage_mb=override_storage_mb,
                agent_import_path=agent_import_path,
                verifier_env=verifier_env,
                expected_trials=expected_trials,
            ): variant
            for (variant, dataset), condition_concurrency in zip(jobs, job_concurrency, strict=True)
        }
        for future in as_completed(futures):
            ok, detail = future.result()
            if not ok:
                errors.append(f"{agent} {futures[future]}-skill Harbor run failed: {detail}")
    return errors


class _RunProgressLifecycle:
    """Track orchestrator stages and guarantee one terminal run event."""

    def __init__(
        self,
        reporter: ProgressReporter,
        *,
        inherited_active_stages: tuple[str, ...] = (),
    ) -> None:
        self._reporter = reporter
        self._active_stages = dict.fromkeys(inherited_active_stages)
        self._run_finished = False
        self._output_dir: str | None = None
        self._result_path: str | None = None
        self._report_path: str | None = None

    @property
    def is_active(self) -> bool:
        return self._reporter.is_active

    @property
    def output_dir(self) -> str | None:
        return self._output_dir

    def start(self, plan: Tier3RunPlan) -> None:
        self._remember_artifacts(
            output_dir=plan.output_dir,
            result_path=plan.result_path,
            report_path=plan.report_path,
        )
        self._reporter.start(plan)

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None:
        self._reporter.set_secret_values(values)

    def emit(self, event: ProgressEvent) -> None:
        self._remember_artifacts(
            output_dir=event.output_dir,
            result_path=event.result_path,
            report_path=event.report_path,
        )
        if event.stage == "run-finished":
            if self._run_finished:
                return
            self._run_finished = True
        elif event.state == "running":
            self._active_stages[event.stage] = None
        else:
            self._active_stages.pop(event.stage, None)
        self._reporter.emit(event)

    def heartbeat(self) -> None:
        self._reporter.heartbeat()

    def close(self) -> None:
        self._reporter.close()

    def fail_unfinished(self) -> None:
        """Fail every open stage and finish the run without masking its error."""
        for stage in tuple(self._active_stages):
            self.emit(
                ProgressEvent(
                    stage=stage,
                    state="failed",
                    detail="unexpected failure interrupted this stage",
                )
            )
        self.emit(
            ProgressEvent(
                stage="run-finished",
                state="failed",
                detail="Tier 3 evaluation failed unexpectedly",
                output_dir=self._output_dir,
                result_path=self._existing_file(self._result_path),
                report_path=self._existing_file(self._report_path),
            )
        )

    def finish_result(self, result: Mapping[str, Any]) -> None:
        """Emit the terminal event for expected early-return failures too."""
        if self._run_finished:
            return
        raw_errors = result.get("error") or result.get("execution_errors") or []
        if isinstance(raw_errors, str):
            errors = [raw_errors]
        elif isinstance(raw_errors, list):
            errors = [str(error) for error in raw_errors if str(error).strip()]
        else:
            errors = [str(raw_errors)] if raw_errors else []
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        failed = bool(errors) or result.get("execution_status") not in {None, "succeeded"}
        state = "failed" if failed else "degraded" if warnings else "complete"
        detail = errors[0] if errors else str(warnings[0]) if warnings else "Tier 3 evaluation finished"
        for stage in tuple(self._active_stages):
            self.emit(ProgressEvent(stage=stage, state="failed" if failed else "complete", detail=detail))
        self.emit(
            ProgressEvent(
                stage="run-finished",
                state=state,
                detail=detail,
                output_dir=str(result.get("run_dir") or self._output_dir or "") or None,
                result_path=self._existing_file(str(result.get("result_path") or self._result_path or "") or None),
                report_path=self._existing_file(str(result.get("report_path") or self._report_path or "") or None),
            )
        )

    def _remember_artifacts(
        self,
        *,
        output_dir: str | None,
        result_path: str | None,
        report_path: str | None,
    ) -> None:
        self._output_dir = output_dir or self._output_dir
        self._result_path = result_path or self._result_path
        self._report_path = report_path or self._report_path

    @staticmethod
    def _existing_file(path: str | None) -> str | None:
        return path if path is not None and Path(path).is_file() else None


def _run_harbor_eval_impl(
    skill_path: Path,
    agents: list[str],
    *,
    skip_baseline: bool = False,
    n_attempts: int | None = None,
    pass_threshold: float | None = None,
    stop_on_pass: bool | None = None,
    n_concurrent: int | None = None,
    max_agents: int | None = None,
    model: str | None = None,
    agent_models: dict[str, str | list[str]] | None = None,
    custom_dockerfile_mode: str | None = None,
    skill_workspace_mode: str | None = None,
    include_skills: list[str | Path] | None = None,
    copy_repo: bool = False,
    grading_mode: str | None = None,
    reference_skills_dir: Path | None = None,
    output_dir: Path | None = None,
    keep_harbor_jobs: bool = False,
    agent_runtime_preflight: bool | None = None,
    env_mode: str = DEFAULT_ENV_MODE,
    env_mode_source: str = "CLI",
    timeout_multiplier: float | None = None,
    override_cpus: int | None = None,
    override_memory_mb: int | None = None,
    override_storage_mb: int | None = None,
    progress_reporter: ProgressReporter | None = None,
    _evaluator_skill_path: Path | None = None,
    _monotonic_start: float | None = None,
) -> dict[str, Any]:
    """Run a public Harbor evaluation with and without the target skill."""
    forwarded = dict(locals()) if _evaluator_skill_path is None else None
    started_at = _monotonic_start if _monotonic_start is not None else time.monotonic()
    reporter = safe_progress_reporter(progress_reporter or NullProgressReporter())
    if env_mode not in HARBOR_ENV_MODES:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="unsupported environment"))
        return {"error": [f"env_mode must be one of: {', '.join(sorted(HARBOR_ENV_MODES))}"]}
    if not agents:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="no agents selected"))
        return {"error": ["At least one Harbor agent is required."]}
    if env_mode == ENV_MODE_LOCAL:
        from skillevaluator.tier3.harbor import local_sandbox

        try:
            local_sandbox.require_supported_platform()
        except local_sandbox.SandboxUnavailable as exc:
            reporter.emit(ProgressEvent(stage="configuration", state="failed", detail=str(exc)))
            return {"error": [str(exc)]}

    if _evaluator_skill_path is None:
        assert forwarded is not None
        forwarded.pop("skill_path")
        forwarded.pop("agents")
        with ExitStack() as snapshot_stack:
            try:
                evaluator_skill_path = snapshot_stack.enter_context(private_evaluator_skill_snapshot(skill_path))
            except (OSError, ValueError) as exc:
                reporter.emit(ProgressEvent(stage="configuration", state="failed", detail=str(exc)))
                return {"error": [str(exc)]}
            forwarded["_evaluator_skill_path"] = evaluator_skill_path
            forwarded["_monotonic_start"] = started_at
            return _run_harbor_eval_impl(skill_path, agents, **forwarded)

    evaluator_skill_path = _evaluator_skill_path

    try:
        provider = resolve_llm_provider()
        config, config_path = load_evals_config(evaluator_skill_path)
    except (ProviderConfigurationError, EvalsConfigError) as exc:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail=str(exc)))
        return {"error": [str(exc)]}

    harbor_config = config.get("harbor", {})
    workspace_config = config.get("skill_workspace", {})
    grading_config = config.get("grading", {})
    n_attempts = n_attempts if n_attempts is not None else harbor_config.get("n_attempts", 1)
    pass_threshold = pass_threshold if pass_threshold is not None else harbor_config.get("pass_threshold", 0.5)
    stop_on_pass = stop_on_pass if stop_on_pass is not None else harbor_config.get("stop_on_pass", False)
    n_concurrent = n_concurrent if n_concurrent is not None else harbor_config.get("n_concurrent", 4)
    max_agents = max_agents if max_agents is not None else harbor_config.get("max_agents", len(agents))
    timeout_multiplier = (
        timeout_multiplier if timeout_multiplier is not None else harbor_config.get("timeout_multiplier", 1.0)
    )
    agent_runtime_preflight = (
        agent_runtime_preflight
        if agent_runtime_preflight is not None
        else harbor_config.get("agent_runtime_preflight", True)
    )
    grading_mode = grading_mode or grading_config.get("mode", "default")
    workspace_mode = skill_workspace_mode or workspace_config.get("mode", "isolated")
    dockerfile_mode = custom_dockerfile_mode or harbor_config.get("custom_dockerfile_mode", "rebase")
    # The public engine ships self-contained per-task Dockerfiles by default;
    # ``reuse``/``rebuild`` opt into the shared pre-built eval base image.
    base_image_mode = harbor_config.get("base_image_mode", "disabled")
    task_source = harbor_config.get("task_source", "auto")

    if not isinstance(n_attempts, int) or n_attempts < 1:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="invalid attempt count"))
        return {"error": ["n_attempts must be >= 1"]}
    if stop_on_pass and n_attempts == 1:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="invalid attempt policy"))
        return {"error": ["stop_on_pass requires n_attempts > 1"]}
    if not isinstance(n_concurrent, int) or n_concurrent < 1:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="invalid concurrency"))
        return {"error": ["n_concurrent must be >= 1"]}
    if not isinstance(max_agents, int) or max_agents < 1:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="invalid agent concurrency"))
        return {"error": ["max_agents must be >= 1"]}
    if not isinstance(pass_threshold, (int, float)) or not 0 <= float(pass_threshold) <= 1:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="invalid pass threshold"))
        return {"error": ["pass_threshold must be between 0.0 and 1.0"]}
    if grading_mode not in {"default", "default_plus_custom", "custom_only"}:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="invalid grading mode"))
        return {"error": ["grading.mode must be default, default_plus_custom, or custom_only"]}
    if workspace_mode not in {"isolated", "group"}:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail="invalid workspace mode"))
        return {"error": ["skill_workspace.mode must be isolated or group"]}

    reporter.emit(ProgressEvent(stage="configuration", state="ready", detail="evaluation config validated"))
    reporter.emit(ProgressEvent(stage="model-resolution", state="running"))

    agent_models_config = harbor_config.get("agents", {})
    agent_models = agent_models or {}
    model_resolution: dict[str, dict[str, str]] = {}
    for agent in agents:
        override = agent_models.get(agent)
        if isinstance(override, list):
            override = override[0] if override else None
        selected, source = _model_for_agent(
            agent,
            cli_model=str(override or model or "") or None,
            config_agents=agent_models_config,
            provider=provider,
        )
        model_resolution[agent] = {"agent": agent, "model": selected, "source": source}

    provider_env = _provider_environment(provider)
    configured_runtime_env, runtime_errors = _resolve_runtime_env(harbor_config.get("runtime_env"))
    reporter.set_secret_values(secret_values_from_environment(provider_env) | set(configured_runtime_env.values()))
    reporter.emit(ProgressEvent(stage="model-resolution", state="complete", detail="agent models resolved"))
    reporter.start(
        Tier3RunPlan(
            skill_name=skill_path.name,
            environment=env_mode,
            agents=tuple(agents),
            agent_models=tuple((agent, model_resolution[agent]["model"]) for agent in agents),
            provider=provider.provider,
            attempts=n_attempts,
            baseline=not skip_baseline,
            concurrency=n_concurrent,
            max_agents=max_agents,
            timeout_multiplier=float(timeout_multiplier),
        )
    )

    reporter.emit(ProgressEvent(stage="environment-preflight", state="running", detail=env_mode))
    prereq_errors = _check_prerequisites(env_mode=env_mode, agents=agents)
    if prereq_errors:
        reporter.emit(ProgressEvent(stage="environment-preflight", state="failed", detail="; ".join(prereq_errors)))
        return {"error": prereq_errors}
    reporter.emit(ProgressEvent(stage="environment-preflight", state="complete", detail=env_mode))

    # Resolve the effective source before constructing credential-probe targets.
    # Native Harbor tasks can select the standard-grader judge model at task or
    # step scope, so the provider fallback is not necessarily the runtime model.
    evals_exists = find_evals_file(evaluator_skill_path) is not None
    native_exists = (evaluator_skill_path / "evals" / "harbor").exists()
    if task_source == "auto":
        task_source = "evals_json" if evals_exists else "native_harbor" if native_exists else ""
    if task_source == "evals_json" and not evals_exists:
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail="evaluation dataset missing"))
        return {"error": ["No evals/evals.json found. Run create-eval-dataset or add a dataset."]}
    if task_source == "native_harbor" and not native_exists:
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail="native Harbor tasks missing"))
        return {"error": ["No native Harbor task source found at evals/harbor."]}
    if task_source not in {"evals_json", "native_harbor"}:
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail="invalid task source"))
        return {"error": ["harbor.task_source must be auto, evals_json, or native_harbor"]}

    reporter.emit(ProgressEvent(stage="credential-validation", state="running"))
    if runtime_errors:
        reporter.emit(ProgressEvent(stage="credential-validation", state="failed", detail="; ".join(runtime_errors)))
        return {"error": runtime_errors}
    try:
        runtime_plans = _resolve_agent_runtime_plan(
            provider=provider,
            agents=agents,
            models={agent: details["model"] for agent, details in model_resolution.items()},
            configured_runtime_env=configured_runtime_env,
            env_mode=env_mode,
            model_sources={agent: details["source"] for agent, details in model_resolution.items()},
        )
    except ValueError as exc:
        reporter.emit(ProgressEvent(stage="credential-validation", state="failed", detail=str(exc)))
        return {"error": [str(exc)]}
    nvidia_build_agent_import_paths = {
        agent: import_path
        for agent in agents
        if (import_path := _nvidia_build_agent_import_path(provider, agent, env_mode)) is not None
    }
    runtime_secret_values = set().union(
        *(secret_values_from_environment(plan.subprocess_env) for plan in runtime_plans.values())
    )

    # Probe each exact agent route before reserving output space, building images,
    # or staging tasks. The runtime-preflight module imports runner helpers, so
    # this import must remain lazy to avoid a module cycle.
    from skillevaluator.tier3.harbor.runtime_preflight import (
        CredentialProbeDisposition,
        credential_probe_disposition,
        probe_model,
    )

    probe_targets: dict[
        tuple[str, str, str | None, str | None, str | None],
        tuple[ProviderConfig, list[str]],
    ] = {}
    probe_degraded: list[str] = []
    credential_validation_targets: list[dict[str, Any]] = []
    judge_config = _judge_model_config(provider, provider_env, grading_mode)

    def add_probe_target(label: str, selected_provider: ProviderConfig) -> None:
        route_key = (
            selected_provider.provider,
            selected_provider.model,
            selected_provider.api_key,
            selected_provider.base_url,
            selected_provider.region,
        )
        target = probe_targets.get(route_key)
        if target is None:
            probe_targets[route_key] = (selected_provider, [label])
        else:
            target[1].append(label)

    for agent in agents:
        add_probe_target(agent, runtime_plans[agent].provider)

    if grading_mode != "custom_only":
        if task_source == "native_harbor" and not bool(judge_config["override_applied"]):
            detail = (
                "native Harbor resolves the effective judge model at runtime; "
                "task or step verifier env may supersede the configured fallback"
            )
            probe_degraded.append(f"standard grader: {detail}")
            judge_config["catalog_verification"] = "inconclusive"
            judge_config["effective_model_source"] = "native_harbor_runtime"
            credential_validation_targets.append(
                {
                    "labels": ["standard grader"],
                    "provider": provider.provider,
                    "model": None,
                    "fallback_model": provider.model,
                    "status": "inconclusive",
                    "detail": detail,
                }
            )
        else:
            judge_model = str(judge_config["model"])
            litellm_model = str(getattr(provider, "litellm_model", f"{provider.provider}/{provider.model}"))
            litellm_prefix = litellm_model.partition("/")[0] or provider.provider
            add_probe_target(
                "standard grader",
                ProviderConfig(
                    provider=provider.provider,
                    model=judge_model,
                    api_key=provider.api_key,
                    base_url=provider.base_url,
                    litellm_model=f"{litellm_prefix}/{judge_model}",
                    region=getattr(provider, "region", None),
                    credential_env=getattr(provider, "credential_env", None),
                    base_url_env=getattr(provider, "base_url_env", None),
                ),
            )

    runtime_secret_values.update(
        selected_provider.api_key for selected_provider, _labels in probe_targets.values() if selected_provider.api_key
    )
    reporter.set_secret_values(runtime_secret_values)

    with ThreadPoolExecutor(max_workers=min(len(probe_targets), 4)) as probe_pool:
        probe_futures = {
            route_key: probe_pool.submit(probe_model, selected_provider)
            for route_key, (selected_provider, _labels) in probe_targets.items()
        }

    probe_errors: list[str] = []
    for route_key, (selected_provider, selected_labels) in probe_targets.items():
        label = ", ".join(selected_labels)
        try:
            probe = probe_futures[route_key].result()
        except Exception as exc:
            safe_detail = f"model catalog probe failed: {type(exc).__name__}"
            probe_degraded.append(f"{label}: {safe_detail}")
            credential_validation_targets.append(
                {
                    "labels": list(selected_labels),
                    "provider": selected_provider.provider,
                    "model": selected_provider.model,
                    "status": "degraded",
                    "detail": safe_detail,
                }
            )
            if "standard grader" in selected_labels:
                judge_config["catalog_verification"] = "degraded"
            continue

        safe_detail = redact_progress_detail(probe.detail, secret_values=runtime_secret_values)
        disposition = credential_probe_disposition(selected_provider, probe)
        if probe.ok and disposition == CredentialProbeDisposition.DEGRADED:
            safe_detail = "model catalog access does not verify runtime credentials for this endpoint"
        credential_validation_targets.append(
            {
                "labels": list(selected_labels),
                "provider": selected_provider.provider,
                "model": selected_provider.model,
                "status": disposition.value,
                "detail": safe_detail,
            }
        )
        if "standard grader" in selected_labels:
            judge_config["catalog_verification"] = disposition.value
        if disposition == CredentialProbeDisposition.FATAL:
            probe_errors.append(f"{label} provider verification failed: {safe_detail}")
        elif disposition == CredentialProbeDisposition.DEGRADED:
            probe_degraded.append(f"{label}: {safe_detail}")

    if probe_errors:
        reporter.emit(
            ProgressEvent(
                stage="credential-validation",
                state="failed",
                detail="; ".join(probe_errors),
            )
        )
        return {"error": probe_errors}
    if probe_degraded:
        reporter.emit(
            ProgressEvent(
                stage="credential-validation",
                state="degraded",
                detail=(
                    "credential configuration resolved; live catalog verification inconclusive: "
                    f"{'; '.join(probe_degraded)}; continuing to evaluation"
                ),
            )
        )
    else:
        reporter.emit(
            ProgressEvent(
                stage="credential-validation",
                state="complete",
                detail="credentials and selected models verified",
            )
        )
    run_config = {
        "config_file": str(config_path.relative_to(evaluator_skill_path)) if config_path else "none",
        "harbor": {
            "environment": {"value": env_mode, "source": env_mode_source},
            "n_attempts": n_attempts,
            "stop_on_pass": bool(stop_on_pass),
            "n_concurrent": n_concurrent,
            "timeout_multiplier": timeout_multiplier,
            "base_image_mode": base_image_mode,
            "jobs_retained": keep_harbor_jobs,
        },
        "provider": {"name": provider.provider, "model": provider.model},
        "judge": judge_config,
        "credential_validation": {
            "status": "degraded" if probe_degraded else "verified",
            "targets": credential_validation_targets,
        },
        "task_source": task_source,
        "grading": {"mode": grading_mode},
        "agents": model_resolution,
    }
    verifier_env = {**configured_runtime_env, **provider_env}
    staged_verifier_env = {name: f"${{{name}}}" for name in verifier_env if name not in _VERIFIER_JUDGE_MODEL_ENV_VARS}
    job_judge_verifier_env = _job_judge_verifier_env(provider_env, grading_mode)
    job_judge_subprocess_env = _job_judge_subprocess_env(provider_env, grading_mode)

    include_values = [*workspace_config.get("include", []), *(include_skills or [])]
    if include_values and workspace_mode != "group":
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail="invalid included skills"))
        return {"error": ["include_skills requires skill_workspace.mode=group"]}
    try:
        workspace_skills = _workspace_skills(skill_path.resolve(), include_values if workspace_mode == "group" else [])
    except ValueError as exc:
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail=str(exc)))
        return {"error": [str(exc)]}

    root = Path(output_dir) if output_dir is not None else skill_path / "evals" / "results"
    try:
        validate_output_provenance_key_location(
            skill_path,
            root,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skills,
        )
        validate_results_root_location(
            skill_path,
            root,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skills,
        )
    except ValueError as exc:
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail=str(exc)))
        return {"error": [str(exc)]}
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    try:
        run_dir = _reserve_run_dir(root, timestamp)
    except (OSError, RuntimeError, ValueError) as exc:
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail=str(exc)))
        return {"error": [str(exc)]}
    run_id = run_dir.name
    jobs_dir = run_dir / "_harbor-jobs"
    tasks_dir = run_dir / "_harbor-tasks"
    result_path = run_dir / "result.json"
    report_path: Path | None = None

    def _emit_run_finished(state: str, detail: str, *, include_artifacts: bool = True) -> None:
        reporter.emit(
            ProgressEvent(
                stage="run-finished",
                state=state,
                detail=detail,
                output_dir=str(run_dir) if include_artifacts else None,
                result_path=str(result_path) if include_artifacts and result_path.is_file() else None,
                report_path=(
                    str(report_path)
                    if include_artifacts and report_path is not None and report_path.is_file()
                    else None
                ),
            )
        )

    def _persist_pre_execution_failure(errors: list[str]) -> dict[str, Any]:
        """Retain redacted probe provenance for failures after run reservation."""
        failed_result: dict[str, Any] = {
            "skill_name": skill_path.name,
            "execution_status": "failed",
            "execution_errors": errors,
            "error": errors,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "harbor_jobs_dir": str(jobs_dir),
            "harbor_jobs_retained": jobs_dir.is_dir(),
            "duration_seconds": round(time.monotonic() - started_at, 3),
            "result_path": str(result_path),
            "agents": {},
            "run_config": run_config,
        }
        write_output_file_atomically(
            run_dir / "run_config.json",
            json.dumps(run_config, indent=2).encode("utf-8"),
        )
        write_output_file_atomically(result_path, json.dumps(failed_result, indent=2).encode("utf-8"))
        return failed_result

    reservation_identity: tuple[int, int] | None = None
    try:
        reservation_metadata = run_dir.lstat()
        reservation_identity = reservation_metadata.st_dev, reservation_metadata.st_ino
        jobs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="failed", detail=str(exc)))
        if reservation_identity is not None:
            remove_generated_output_root_if_owned(run_dir, expected_identity=reservation_identity)
        _emit_run_finished("failed", "Harbor jobs directory could not be created", include_artifacts=False)
        return {"error": [str(exc)]}

    emitter = stage_native_harbor_tasks if task_source == "native_harbor" else generate_harbor_tasks
    resource_config = harbor_config.get("resources", {})
    use_base_image = env_mode == "docker" and base_image_mode != "disabled"
    base_image = ""
    if use_base_image:
        reporter.emit(
            ProgressEvent(
                stage="docker-images",
                state="running",
                detail=f"preparing shared eval base image ({base_image_mode})",
            )
        )
        base_image = build_eval_base_image(
            skill_path.resolve(),
            reference_skills_dir,
            workspace_skill_paths=workspace_skills,
            evaluator_skill_path=evaluator_skill_path,
            excluded_roots=(root,),
            force_rebuild=base_image_mode == "rebuild",
        )
        if base_image:
            reporter.emit(
                ProgressEvent(stage="docker-images", state="complete", detail=f"eval base image ready: {base_image}")
            )
        else:
            reporter.emit(
                ProgressEvent(
                    stage="docker-images",
                    state="degraded",
                    detail="base image build failed; falling back to per-task Dockerfiles",
                )
            )
    agent_task_dirs: dict[str, tuple[Path, Path | None]] = {}
    expected_task_names: list[str] | None = None
    reporter.emit(
        ProgressEvent(
            stage="with-skill-tasks",
            state="running",
            output_dir=str(run_dir),
            result_path=str(result_path),
        )
    )
    staging_failure_stage = "with-skill-tasks"
    try:
        for agent in agents:
            with_dir = tasks_dir / agent / "with"
            without_dir = None if skip_baseline else tasks_dir / agent / "without"
            task_paths = emitter(
                skill_path,
                with_dir,
                with_skill=True,
                reference_skills_dir=reference_skills_dir,
                workspace_skill_paths=workspace_skills,
                workspace_mode=workspace_mode,
                grading_mode=grading_mode,
                base_image=base_image,
                custom_dockerfile_mode=dockerfile_mode,
                copy_repo=copy_repo,
                repo_context_exclude_paths=(root,),
                runtime_env=dict(runtime_plans[agent].staged_env),
                verifier_env=staged_verifier_env,
                pre_agent_setup=harbor_config.get("pre_agent_setup", []),
                task_resources=resource_config,
                agent_workdir=harbor_config.get("agent_workdir"),
                evaluator_skill_path=evaluator_skill_path,
            )
            task_names = [task.name for task in task_paths]
            if expected_task_names is None:
                expected_task_names = task_names
            elif task_names != expected_task_names:
                raise ValueError(f"Generated task cases differ for agent {agent}")
            agent_task_dirs[agent] = (with_dir, without_dir)
        reporter.emit(ProgressEvent(stage="with-skill-tasks", state="ready", detail="task inputs staged"))
        if not skip_baseline:
            reporter.emit(ProgressEvent(stage="baseline-tasks", state="running"))
            staging_failure_stage = "baseline-tasks"
            baseline_alias_validation = _prevalidate_baseline_skill_candidates(
                skill_path,
                reference_skills_dir,
                workspace_skills,
                excluded_roots=(root,),
            )
        else:
            baseline_alias_validation = None
        for agent in agents:
            without_dir = agent_task_dirs[agent][1]
            if without_dir is not None:
                emitter(
                    skill_path,
                    without_dir,
                    with_skill=False,
                    reference_skills_dir=reference_skills_dir,
                    workspace_skill_paths=workspace_skills,
                    workspace_mode=workspace_mode,
                    grading_mode=grading_mode,
                    base_image=base_image,
                    custom_dockerfile_mode=dockerfile_mode,
                    copy_repo=copy_repo,
                    repo_context_exclude_paths=(root,),
                    runtime_env=dict(runtime_plans[agent].staged_env),
                    verifier_env=staged_verifier_env,
                    pre_agent_setup=harbor_config.get("pre_agent_setup", []),
                    task_resources=resource_config,
                    agent_workdir=harbor_config.get("agent_workdir"),
                    evaluator_skill_path=evaluator_skill_path,
                    _baseline_alias_validation=baseline_alias_validation,
                )
        if not skip_baseline:
            reporter.emit(ProgressEvent(stage="baseline-tasks", state="ready", detail="baseline inputs staged"))
        else:
            reporter.emit(ProgressEvent(stage="baseline-tasks", state="skipped", detail="baseline disabled"))
    except (OSError, ValueError) as exc:
        reporter.emit(ProgressEvent(stage=staging_failure_stage, state="failed", detail=str(exc)))
        return _persist_pre_execution_failure([str(exc)])

    task_names = expected_task_names or []
    expected_trials = len(task_names) * n_attempts
    variants = 1 if skip_baseline else 2
    matrix_trials = expected_trials * len(agents) * variants
    preflight_trials = len(agents) if agent_runtime_preflight else 0
    task_timeout_seconds = _task_timeout_plan(
        [paths[0] for paths in agent_task_dirs.values()],
        float(timeout_multiplier),
    )
    reporter.start(
        Tier3RunPlan(
            skill_name=skill_path.name,
            environment=env_mode,
            agents=tuple(agents),
            agent_models=tuple((agent, model_resolution[agent]["model"]) for agent in agents),
            provider=provider.provider,
            task_count=len(task_names),
            case_count=len(task_names),
            attempts=n_attempts,
            baseline=not skip_baseline,
            concurrency=n_concurrent,
            max_agents=max_agents,
            timeout_multiplier=float(timeout_multiplier),
            matrix_trials=matrix_trials,
            preflight_trials=preflight_trials,
            total_containers=matrix_trials + preflight_trials,
            task_timeout_seconds=task_timeout_seconds,
            output_dir=str(run_dir),
            result_path=str(result_path),
        )
    )
    if env_mode == ENV_MODE_LOCAL:
        reporter.emit(ProgressEvent(stage="docker-images", state="skipped", detail="local environment selected"))
    elif not use_base_image:
        # The shared base image branch already emitted its terminal stage event.
        reporter.emit(
            ProgressEvent(
                stage="docker-images",
                state="delegated",
                detail="image preparation delegated to Harbor during task execution",
            )
        )
    if agent_runtime_preflight:
        from skillevaluator.tier3.harbor.runtime_preflight import run_agent_runtime_preflight

        reporter.emit(ProgressEvent(stage="agent-runtime-preflight", state="running"))
        preflight_errors: list[str] = []
        for agent in agents:
            preflight = run_agent_runtime_preflight(
                dataset=agent_task_dirs[agent][0],
                agent=agent,
                model=model_resolution[agent]["model"],
                env_mode=env_mode,
                jobs_dir=jobs_dir,
                run_env={**runtime_plans[agent].subprocess_env, **job_judge_subprocess_env},
                timeout_multiplier=float(timeout_multiplier),
                override_cpus=override_cpus,
                override_memory_mb=override_memory_mb,
                override_storage_mb=override_storage_mb,
                agent_import_path=nvidia_build_agent_import_paths.get(agent),
            )
            if not preflight.ok:
                preflight_errors.append(f"{agent} runtime preflight failed: {preflight.detail}")
        if preflight_errors:
            detail = "; ".join(preflight_errors)
            reporter.emit(ProgressEvent(stage="agent-runtime-preflight", state="failed", detail=detail))
            failed_result = _persist_pre_execution_failure(preflight_errors)
            _emit_run_finished("failed", "agent runtime preflight failed")
            return failed_result
        reporter.emit(
            ProgressEvent(
                stage="agent-runtime-preflight",
                state="complete",
                detail=f"{len(agents)} agent runtime(s) started successfully",
            )
        )
    else:
        reporter.emit(ProgressEvent(stage="agent-runtime-preflight", state="skipped", detail="disabled by operator"))
    errors: list[str] = []
    started_agents: SimpleQueue[str] = SimpleQueue()

    def _execute_agent(agent: str) -> list[str]:
        started_agents.put(agent)
        return _run_agent_pair(
            skill_name=skill_path.name,
            agent=agent,
            model=model_resolution[agent]["model"],
            env_mode=env_mode,
            with_skill=agent_task_dirs[agent][0],
            baseline=agent_task_dirs[agent][1],
            jobs_dir=jobs_dir,
            run_env={**runtime_plans[agent].subprocess_env, **job_judge_subprocess_env},
            n_attempts=n_attempts,
            n_concurrent=n_concurrent,
            timeout_multiplier=float(timeout_multiplier),
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            override_storage_mb=override_storage_mb,
            agent_import_path=nvidia_build_agent_import_paths.get(agent),
            expected_trials=expected_trials,
            stop_on_pass=bool(stop_on_pass),
            pass_threshold=float(pass_threshold),
            task_names=task_names,
            verifier_env=job_judge_verifier_env,
        )

    active_agents: set[str] = set()
    unexpected_worker_error: Exception | None = None

    def _emit_started_agents() -> None:
        while True:
            try:
                agent = started_agents.get_nowait()
            except Empty:
                return
            active_agents.add(agent)
            variants = "with-skill" if skip_baseline else "with-skill + baseline"
            reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="running", detail=variants))

    with ThreadPoolExecutor(max_workers=min(max_agents, len(agents))) as executor:
        futures = {executor.submit(_execute_agent, agent): agent for agent in agents}
        pending = set(futures)
        while pending:
            _emit_started_agents()
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            _emit_started_agents()
            for future in done:
                agent = futures[future]
                active_agents.discard(agent)
                try:
                    agent_errors = future.result()
                except Exception as exc:
                    reporter.emit(
                        ProgressEvent(
                            stage=f"agent:{agent}",
                            state="failed",
                            detail="agent worker failed unexpectedly",
                        )
                    )
                    unexpected_worker_error = unexpected_worker_error or exc
                    continue
                errors.extend(agent_errors)
                if agent_errors:
                    reporter.emit(
                        ProgressEvent(
                            stage=f"agent:{agent}",
                            state="failed",
                            detail="one or more Harbor jobs failed; inspect retained artifacts",
                        )
                    )
                else:
                    reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="complete"))

    if unexpected_worker_error is not None:
        for agent in sorted(active_agents):
            reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="failed", detail="agent execution interrupted"))
        _emit_run_finished("failed", "agent execution failed")
        raise unexpected_worker_error

    reporter.emit(ProgressEvent(stage="collection", state="running"))
    try:
        results = collect_harbor_results(
            skill_name=skill_path.name,
            agents=agents,
            output_dir=run_dir,
            jobs_dir=jobs_dir,
            skip_baseline=skip_baseline,
            n_attempts=n_attempts,
            pass_threshold=float(pass_threshold),
            stop_on_pass=bool(stop_on_pass),
            expected_cases=len(task_names),
            expected_case_ids=task_names,
            # Early-stopped cases legitimately use fewer trials than the
            # n_attempts maximum; per-case coverage is validated instead.
            expected_trials=None if stop_on_pass else expected_trials,
            env_mode=env_mode,
            agent_models=model_resolution,
            launch_errors=errors,
        )
    except Exception:
        reporter.emit(ProgressEvent(stage="collection", state="failed", detail="result collection failed"))
        _emit_run_finished("failed", "result collection failed")
        raise
    reporter.emit(ProgressEvent(stage="collection", state="complete", detail="Harbor results collected"))
    dataset_truth = _persist_dataset_truth(run_dir, fallback_task_ids=task_names)
    results.update(
        {
            "skill_name": skill_path.name,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "harbor_jobs_dir": str(jobs_dir),
            "harbor_jobs_retained": keep_harbor_jobs,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "evaluator_version": dataset_truth["evaluator_version"],
            "dataset_snapshot": dataset_truth,
            "dataset_snapshot_path": str(run_dir / "dataset_snapshot.json"),
            "dataset_summary": dataset_truth["dataset_summary"],
            "dataset_digest": dataset_truth["dataset_digest"],
            "dataset_digest_algorithm": dataset_truth["dataset_digest_algorithm"],
            "run_config": run_config,
            "attempt_policy": {
                "max_attempts": n_attempts,
                "pass_threshold": float(pass_threshold),
                "stop_on_pass": bool(stop_on_pass),
                "score_definition": score_definition(tuple(results.get("metrics", DEFAULT_METRICS))),
            },
        }
    )
    _finalize_harbor_artifacts(
        run_dir_value=run_dir,
        keep_requested=keep_harbor_jobs,
        result=results,
    )
    if errors:
        execution_errors = list(
            dict.fromkeys([*(str(error) for error in results.get("execution_errors", [])), *errors])
        )
        results["execution_status"] = "failed"
        results["execution_errors"] = execution_errors
        results["error"] = execution_errors
    reporter.emit(ProgressEvent(stage="report", state="running"))
    try:
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    except Exception:
        reporter.emit(ProgressEvent(stage="report", state="failed", detail="run configuration write failed"))
        _emit_run_finished("failed", "report artifacts could not be written")
        raise

    report_warning: str | None = None
    try:
        candidate_report_path = render_agent_eval_html_report(
            skill_path,
            run_dir,
            env_mode=env_mode,
            engine_result=results,
        )
        if candidate_report_path.is_file():
            report_path = candidate_report_path
            results["report_path"] = str(report_path)
        else:
            report_warning = "HTML report was not generated: report file is missing"
    except Exception as exc:
        report_warning = f"HTML report was not generated: {exc}"
    if report_warning:
        results.setdefault("warnings", []).append(report_warning)
        results["report_status"] = "degraded"
    else:
        results["report_status"] = "complete"
    results["duration_seconds"] = round(time.monotonic() - started_at, 3)

    try:
        write_output_file_atomically(result_path, json.dumps(results, indent=2).encode("utf-8"))
    except Exception:
        reporter.emit(ProgressEvent(stage="report", state="failed", detail="result write failed"))
        _emit_run_finished("failed", "report artifacts could not be written")
        raise
    if report_warning:
        reporter.emit(ProgressEvent(stage="report", state="degraded", detail=report_warning))
    else:
        reporter.emit(ProgressEvent(stage="report", state="complete", detail="result and HTML reports written"))

    publish_latest_results(root, run_id)
    return results


def _apply_retention_outcome(
    result: dict[str, Any],
    *,
    outcome: RetentionOutcome,
    jobs_dir: Path,
) -> None:
    """Apply actual artifact filesystem truth to the returned result."""
    result["harbor_jobs_dir"] = str(jobs_dir)
    result["harbor_jobs_retained"] = jobs_dir.is_dir()
    result["harbor_jobs_retention_reason"] = outcome.reason
    if outcome.warning:
        warning = f"Harbor artifact cleanup failed: {outcome.warning}"
        warnings = result.setdefault("warnings", [])
        if isinstance(warnings, list) and warning not in warnings:
            warnings.append(warning)
    run_config = result.get("run_config")
    harbor_config = run_config.get("harbor") if isinstance(run_config, dict) else None
    if isinstance(harbor_config, dict):
        harbor_config["jobs_retained"] = jobs_dir.is_dir()


def _finalize_harbor_artifacts(
    *,
    run_dir_value: object,
    keep_requested: bool,
    result: dict[str, Any] | None,
) -> None:
    """Finalize transient paths and persist corrected metadata when available."""
    if not run_dir_value:
        return
    run_dir = Path(str(run_dir_value))
    if not run_dir.is_dir():
        return
    jobs_dir = run_dir / "_harbor-jobs"
    tasks_dir = run_dir / "_harbor-tasks"
    outcome = HarborArtifactLifecycle(
        [jobs_dir, tasks_dir],
        keep_requested=keep_requested,
    ).finalize()
    if result is None:
        return

    _apply_retention_outcome(result, outcome=outcome, jobs_dir=jobs_dir)
    result_path_value = result.get("result_path")
    result_path = Path(str(result_path_value)) if result_path_value else run_dir / "result.json"
    if result_path.is_file():
        write_output_file_atomically(result_path, json.dumps(result, indent=2).encode("utf-8"))
    run_config = result.get("run_config")
    run_config_path = run_dir / "run_config.json"
    if isinstance(run_config, dict) and run_config_path.is_file():
        run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")


@wraps(_run_harbor_eval_impl)
def run_harbor_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run Tier 3 with protected, coordinator-owned progress lifecycle."""
    reporter = safe_progress_reporter(kwargs.get("progress_reporter"))
    started_here = not reporter.is_active
    lifecycle = _RunProgressLifecycle(
        reporter,
        inherited_active_stages=() if started_here else ("configuration",),
    )
    kwargs["progress_reporter"] = lifecycle
    try:
        lifecycle.set_secret_values(secret_values_from_environment(os.environ))
        if started_here:
            lifecycle.start(
                Tier3RunPlan(
                    skill_name="pending",
                    environment=kwargs.get("env_mode", DEFAULT_ENV_MODE),
                    agents=(),
                    baseline=not kwargs.get("skip_baseline", False),
                    attempts=kwargs.get("n_attempts"),
                    concurrency=kwargs.get("n_concurrent"),
                    max_agents=kwargs.get("max_agents"),
                    timeout_multiplier=kwargs.get("timeout_multiplier"),
                )
            )
            lifecycle.emit(ProgressEvent(stage="configuration", state="running"))
        result = _run_harbor_eval_impl(*args, **kwargs)
        if "harbor_jobs_retention_reason" not in result:
            _finalize_harbor_artifacts(
                run_dir_value=result.get("run_dir") or lifecycle.output_dir,
                keep_requested=bool(kwargs.get("keep_harbor_jobs", False)),
                result=result,
            )
        lifecycle.finish_result(result)
        return result
    except BaseException:
        _finalize_harbor_artifacts(
            run_dir_value=lifecycle.output_dir,
            keep_requested=bool(kwargs.get("keep_harbor_jobs", False)),
            result=None,
        )
        try:
            lifecycle.fail_unfinished()
        except BaseException:
            logger.debug("Tier 3 progress terminalization failed", exc_info=True)
        raise
    finally:
        if started_here:
            lifecycle.close()
