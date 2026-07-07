# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Harbor runner for live agent skill evaluation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillevaluator.provider_config import ProviderConfig, ProviderConfigurationError, resolve_llm_provider
from skillevaluator.telemetry import record_agent_eval_summary
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.harbor.adapter import find_evals_file, generate_harbor_tasks, stage_native_harbor_tasks
from skillevaluator.tier3.harbor.collector import collect_harbor_results, validate_harbor_job_result
from skillevaluator.tier3.harbor.html_report import generate_html_report
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, score_definition
from skillevaluator.tier3_environments import DEFAULT_ENV_MODE, ENV_MODE_LOCAL, HARBOR_ENV_MODES

logger = logging.getLogger(__name__)

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
            "CLASSPATH",
            "COMSPEC",
            "ENV",
            "GCONV_PATH",
            "HOME",
            "HOSTALIASES",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "IFS",
            "JAVA_TOOL_OPTIONS",
            "LOCPATH",
            "NLSPATH",
            "NO_PROXY",
            "PATHEXT",
            "PATH",
            "PERL5LIB",
            "PERL5OPT",
            "REQUESTS_CA_BUNDLE",
            "RES_OPTIONS",
            "RUBYOPT",
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
            "XDG_RUNTIME_DIR",
            "_JAVA_OPTIONS",
        }
    )
    | _BEDROCK_HOST_ENV_VARS
    | frozenset().union(*_HARBOR_ENV_MODE_VARS.values())
)
_RUNTIME_ENV_HOST_CONTROL_PREFIXES = (
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
    "SKILLEVALUATOR_",
    "UV_",
)


def _harbor_bin() -> str:
    """Return the Harbor executable installed with the active interpreter."""
    candidate = Path(os.sys.executable).parent / "harbor"
    return str(candidate) if candidate.exists() else (shutil.which("harbor") or "harbor")


def _harbor_supports_yes() -> bool:
    """Harbor 0.13.2, the supported Tier 3 dependency, accepts ``--yes``."""
    return True


def format_harbor_view_command(jobs_dir: Path | str, *, multiline: bool = False) -> str:
    """Return the portable command for inspecting retained Harbor artifacts."""
    path = str(jobs_dir)
    return f"harbor view {path}" if not multiline else f"harbor view \\\n  {path}"


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
) -> list[str]:
    """Build a Harbor invocation for a built-in environment type or local mode."""
    if env_mode not in HARBOR_ENV_MODES:
        raise ValueError(f"env_mode must be one of: {', '.join(sorted(HARBOR_ENV_MODES))}")

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

        agent_import_path = LOCAL_AGENT_IMPORT_PATHS.get(agent)
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
    if _harbor_supports_yes():
        command.append("--yes")
    return command


def _provider_environment(config: ProviderConfig) -> dict[str, str]:
    """Map a public provider config to evaluator-owned verifier variables."""
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": config.provider,
        "SKILL_EVAL_LLM_MODEL": config.model,
    }
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
) -> list[str]:
    """Reject provider-to-agent combinations that cannot use the selected API."""
    if (
        env_mode != ENV_MODE_LOCAL
        and provider.provider == "nv_build"
        and "opencode" in agents
        and not agent_runtime_env.get("NVIDIA_API_KEY", "").strip()
    ):
        return [
            "opencode with NVIDIA Build requires NVIDIA_API_KEY in harbor.runtime_env so the agent container "
            "receives a credential."
        ]
    if provider.provider != "nv_build":
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
            "Responses provider (e.g. https://api.openai.com/v1) in harbor.runtime_env for Codex."
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


def _resolve_runtime_env(templates: dict[str, str] | None) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    errors: list[str] = []
    for name, template in (templates or {}).items():
        normalized_name = name.upper()
        if normalized_name in _RUNTIME_ENV_HOST_CONTROL_NAMES or normalized_name.startswith(
            _RUNTIME_ENV_HOST_CONTROL_PREFIXES
        ):
            errors.append(f"harbor.runtime_env.{name} controls the host process and is not allowed")
            continue
        value = os.path.expandvars(str(template))
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
        if (
            provider.provider == "nv_build"
            and agent == "opencode"
            and (agent_model or "").startswith("nvidia/")
        ):
            # OpenCode's NVIDIA adapter reads OPENAI_* internally. Override an
            # independent Codex pair for this Harbor subprocess only; each
            # selected local agent receives its own environment below.
            environment.pop("ANTHROPIC_API_KEY", None)
            environment.pop("ANTHROPIC_BASE_URL", None)
            environment.update(local_credentials)
        elif provider.provider == "nv_build" and agent == "codex":
            # Codex must receive only its configured independent Responses API
            # pair, never NVIDIA's OpenAI-compatible mapping or Claude's key.
            environment.pop("ANTHROPIC_API_KEY", None)
            environment.pop("ANTHROPIC_BASE_URL", None)
        elif provider.provider == "nv_build" and agent == "claude-code":
            # Claude must receive only its configured Anthropic credential;
            # keep the NVIDIA_API_KEY solely for verifier-side expansion.
            environment.pop("OPENAI_API_KEY", None)
            environment.pop("OPENAI_BASE_URL", None)
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
    environment["SKILLEVALUATOR_TELEMETRY_DISABLED"] = "true"
    return environment


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


def _model_for_agent(
    agent: str,
    *,
    cli_model: str | None,
    config_agents: dict[str, Any],
    provider: ProviderConfig,
) -> tuple[str, str]:
    if cli_model:
        return cli_model, "CLI"
    configured = config_agents.get(agent, {}) if isinstance(config_agents, dict) else {}
    if isinstance(configured, dict) and configured.get("model"):
        return str(configured["model"]), "evals/config.yml"
    selected = provider.model
    if agent == "opencode" and provider.provider == "nv_build" and not selected.startswith("nvidia/"):
        selected = f"nvidia/{selected}"
    return selected, "public provider default"


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
    expected_trials: int | None = None,
    expected_total_trials: int | None = None,
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
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        override_storage_mb=override_storage_mb,
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=run_env, timeout=7200, check=False)
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
    return False, output[-2000:] or f"harbor run exited {result.returncode}"


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
) -> list[str]:
    jobs = [("with", with_skill)]
    if baseline is not None:
        jobs.append(("without", baseline))
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
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
                n_concurrent=n_concurrent,
                timeout_multiplier=timeout_multiplier,
                override_cpus=override_cpus,
                override_memory_mb=override_memory_mb,
                override_storage_mb=override_storage_mb,
                expected_trials=expected_trials,
            ): variant
            for variant, dataset in jobs
        }
        for future in as_completed(futures):
            ok, detail = future.result()
            if not ok:
                errors.append(f"{agent} {futures[future]}-skill Harbor run failed: {detail}")
    return errors


def run_harbor_eval(
    skill_path: Path,
    agents: list[str],
    *,
    skip_baseline: bool = False,
    n_attempts: int | None = None,
    pass_threshold: float | None = None,
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
    env_mode: str = DEFAULT_ENV_MODE,
    env_mode_source: str = "CLI",
    timeout_multiplier: float | None = None,
    override_cpus: int | None = None,
    override_memory_mb: int | None = None,
    override_storage_mb: int | None = None,
) -> dict[str, Any]:
    """Run a public Harbor evaluation with and without the target skill."""
    if env_mode not in HARBOR_ENV_MODES:
        return {"error": [f"env_mode must be one of: {', '.join(sorted(HARBOR_ENV_MODES))}"]}
    if not agents:
        return {"error": ["At least one Harbor agent is required."]}

    try:
        provider = resolve_llm_provider()
        config, config_path = load_evals_config(skill_path)
    except (ProviderConfigurationError, EvalsConfigError) as exc:
        return {"error": [str(exc)]}

    harbor_config = config.get("harbor", {})
    workspace_config = config.get("skill_workspace", {})
    grading_config = config.get("grading", {})
    n_attempts = n_attempts if n_attempts is not None else harbor_config.get("n_attempts", 1)
    pass_threshold = pass_threshold if pass_threshold is not None else harbor_config.get("pass_threshold", 0.5)
    n_concurrent = n_concurrent if n_concurrent is not None else harbor_config.get("n_concurrent", 4)
    max_agents = max_agents if max_agents is not None else harbor_config.get("max_agents", len(agents))
    timeout_multiplier = (
        timeout_multiplier if timeout_multiplier is not None else harbor_config.get("timeout_multiplier", 1.0)
    )
    grading_mode = grading_mode or grading_config.get("mode", "default")
    workspace_mode = skill_workspace_mode or workspace_config.get("mode", "isolated")
    dockerfile_mode = custom_dockerfile_mode or harbor_config.get("custom_dockerfile_mode", "rebase")
    task_source = harbor_config.get("task_source", "auto")

    if not isinstance(n_attempts, int) or n_attempts < 1:
        return {"error": ["n_attempts must be >= 1"]}
    if not isinstance(n_concurrent, int) or n_concurrent < 1:
        return {"error": ["n_concurrent must be >= 1"]}
    if not isinstance(max_agents, int) or max_agents < 1:
        return {"error": ["max_agents must be >= 1"]}
    if not isinstance(pass_threshold, (int, float)) or not 0 <= float(pass_threshold) <= 1:
        return {"error": ["pass_threshold must be between 0.0 and 1.0"]}
    if grading_mode not in {"default", "default_plus_custom", "custom_only"}:
        return {"error": ["grading.mode must be default, default_plus_custom, or custom_only"]}
    if workspace_mode not in {"isolated", "group"}:
        return {"error": ["skill_workspace.mode must be isolated or group"]}

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

    prereq_errors = _check_prerequisites(env_mode=env_mode, agents=agents)
    if prereq_errors:
        return {"error": prereq_errors}

    provider_env = _provider_environment(provider)
    configured_runtime_env, runtime_errors = _resolve_runtime_env(harbor_config.get("runtime_env"))
    if runtime_errors:
        return {"error": runtime_errors}
    credential_errors = _validate_agent_provider_credentials(
        provider,
        agents,
        configured_runtime_env,
        {agent: details["source"] for agent, details in model_resolution.items()},
        env_mode=env_mode,
    )
    if credential_errors:
        return {"error": credential_errors}
    verifier_env = {**configured_runtime_env, **provider_env}
    staged_runtime_env = {name: f"${{{name}}}" for name in configured_runtime_env}
    staged_verifier_env = {name: f"${{{name}}}" for name in verifier_env}

    include_values = [*workspace_config.get("include", []), *(include_skills or [])]
    if include_values and workspace_mode != "group":
        return {"error": ["include_skills requires skill_workspace.mode=group"]}
    try:
        workspace_skills = _workspace_skills(skill_path.resolve(), include_values if workspace_mode == "group" else [])
    except ValueError as exc:
        return {"error": [str(exc)]}

    evals_exists = find_evals_file(skill_path) is not None
    native_exists = (skill_path / "evals" / "harbor").exists()
    if task_source == "auto":
        task_source = "evals_json" if evals_exists else "native_harbor" if native_exists else ""
    if task_source == "evals_json" and not evals_exists:
        return {"error": ["No evals/evals.json found. Run create-eval-dataset or add a dataset."]}
    if task_source == "native_harbor" and not native_exists:
        return {"error": ["No native Harbor task source found at evals/harbor."]}
    if task_source not in {"evals_json", "native_harbor"}:
        return {"error": ["harbor.task_source must be auto, evals_json, or native_harbor"]}

    root = output_dir or (skill_path / "evals" / "results")
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    jobs_dir = run_dir / "_harbor-jobs"
    tasks_dir = run_dir / "_harbor-tasks"
    baseline_dir = run_dir / "_harbor-tasks-baseline"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    emitter = stage_native_harbor_tasks if task_source == "native_harbor" else generate_harbor_tasks
    resource_config = harbor_config.get("resources", {})
    try:
        task_paths = emitter(
            skill_path,
            tasks_dir,
            with_skill=True,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skills,
            workspace_mode=workspace_mode,
            grading_mode=grading_mode,
            custom_dockerfile_mode=dockerfile_mode,
            copy_repo=copy_repo,
            runtime_env=staged_runtime_env,
            verifier_env=staged_verifier_env,
            pre_agent_setup=harbor_config.get("pre_agent_setup", []),
            task_resources=resource_config,
            agent_workdir=harbor_config.get("agent_workdir"),
        )
        if not skip_baseline:
            emitter(
                skill_path,
                baseline_dir,
                with_skill=False,
                reference_skills_dir=reference_skills_dir,
                workspace_skill_paths=workspace_skills,
                workspace_mode=workspace_mode,
                grading_mode=grading_mode,
                custom_dockerfile_mode=dockerfile_mode,
                copy_repo=copy_repo,
                runtime_env=staged_runtime_env,
                verifier_env=staged_verifier_env,
                pre_agent_setup=harbor_config.get("pre_agent_setup", []),
                task_resources=resource_config,
                agent_workdir=harbor_config.get("agent_workdir"),
            )
    except (FileNotFoundError, ValueError) as exc:
        return {"error": [str(exc)], "run_dir": str(run_dir)}

    task_names = [task.name for task in task_paths]
    expected_trials = len(task_names) * n_attempts
    agent_run_envs = {
        agent: _harbor_subprocess_environment(
            env_mode=env_mode,
            provider=provider,
            configured_runtime_env=configured_runtime_env,
            provider_env=provider_env,
            agent=agent,
            agent_model=model_resolution[agent]["model"],
        )
        for agent in agents
    }
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(max_agents, len(agents))) as executor:
        futures = {
            executor.submit(
                _run_agent_pair,
                skill_name=skill_path.name,
                agent=agent,
                model=model_resolution[agent]["model"],
                env_mode=env_mode,
                with_skill=tasks_dir,
                baseline=None if skip_baseline else baseline_dir,
                jobs_dir=jobs_dir,
                run_env=agent_run_envs[agent],
                n_attempts=n_attempts,
                n_concurrent=n_concurrent,
                timeout_multiplier=float(timeout_multiplier),
                override_cpus=override_cpus,
                override_memory_mb=override_memory_mb,
                override_storage_mb=override_storage_mb,
                expected_trials=expected_trials,
            ): agent
            for agent in agents
        }
        for future in as_completed(futures):
            errors.extend(future.result())

    results = collect_harbor_results(
        skill_name=skill_path.name,
        agents=agents,
        output_dir=run_dir,
        jobs_dir=jobs_dir,
        skip_baseline=skip_baseline,
        n_attempts=n_attempts,
        pass_threshold=float(pass_threshold),
        expected_cases=len(task_names),
        expected_case_ids=task_names,
        expected_trials=expected_trials,
        env_mode=env_mode,
        agent_models=model_resolution,
        launch_errors=errors,
    )
    run_config = {
        "config_file": str(config_path.relative_to(skill_path)) if config_path else "none",
        "harbor": {
            "environment": {"value": env_mode, "source": env_mode_source},
            "n_attempts": n_attempts,
            "n_concurrent": n_concurrent,
            "timeout_multiplier": timeout_multiplier,
            "jobs_retained": keep_harbor_jobs,
        },
        "provider": {"name": provider.provider, "model": provider.model},
        "task_source": task_source,
        "grading": {"mode": grading_mode},
        "agents": model_resolution,
    }
    results.update(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "harbor_jobs_dir": str(jobs_dir),
            "harbor_jobs_retained": keep_harbor_jobs,
            "run_config": run_config,
            "attempt_policy": {
                "max_attempts": n_attempts,
                "pass_threshold": float(pass_threshold),
                "score_definition": score_definition(tuple(results.get("metrics", DEFAULT_METRICS))),
            },
        }
    )
    if errors:
        execution_errors = list(
            dict.fromkeys([*(str(error) for error in results.get("execution_errors", [])), *errors])
        )
        results["execution_status"] = "failed"
        results["execution_errors"] = execution_errors
        results["error"] = execution_errors
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    try:
        generate_html_report(skill_path.name, run_dir, skill_path=skill_path)
    except Exception as exc:
        results.setdefault("warnings", []).append(f"HTML report was not generated: {exc}")
    (run_dir / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_id)
    except OSError:
        pass
    try:
        record_agent_eval_summary(
            runner="harbor",
            skill_name=skill_path.name,
            agents=agents,
            env_mode=env_mode,
            results=results,
            agent_models=model_resolution,
            duration_ms=0.0,
        )
    except Exception:
        logger.debug("Telemetry summary skipped", exc_info=True)

    if not keep_harbor_jobs:
        shutil.rmtree(jobs_dir, ignore_errors=True)
        shutil.rmtree(tasks_dir, ignore_errors=True)
        shutil.rmtree(baseline_dir, ignore_errors=True)
    return results
