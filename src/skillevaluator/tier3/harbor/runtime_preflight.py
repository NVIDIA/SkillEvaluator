# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded one-task Harbor smoke runs for agent runtime readiness."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from skillevaluator.model_catalog import ModelCatalogError, fetch_model_records
from skillevaluator.tier3.harbor.progress import redact_progress_detail
from skillevaluator.tier3.harbor.runner import _nvidia_build_key_handoff, build_harbor_run_command

if TYPE_CHECKING:
    from skillevaluator.provider_config import ProviderConfig

DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class PreflightResult:
    """Persistable outcome of a real, verification-disabled agent smoke."""

    ok: bool
    agent: str
    model: str
    detail: str
    job_name: str


@dataclass(frozen=True)
class ModelProbeResult:
    """Safe result from a provider model-catalog request."""

    ok: bool
    provider: str
    model: str
    detail: str


def _first_task_name(dataset: Path) -> str | None:
    for task_dir in sorted(path for path in dataset.iterdir() if path.is_dir() and not path.is_symlink()):
        if (task_dir / "task.toml").is_file():
            return task_dir.name
    return None


def _redact_detail(value: str, environment: Mapping[str, str]) -> str:
    secret_values = {item for item in environment.values() if len(item) >= 4}
    return redact_progress_detail(value, secret_values=secret_values)[-2000:]


def _first_trial_exception_detail(job_dir: Path) -> str:
    """Return a bounded first exception from Harbor's retained trial results."""
    try:
        result_paths = sorted(
            child / "result.json" for child in job_dir.iterdir() if child.is_dir() and (child / "result.json").is_file()
        )
    except OSError:
        return ""

    for result_path in result_paths:
        try:
            trial_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(trial_result, dict):
            continue
        candidates = [trial_result.get("exception_info")]
        step_results = trial_result.get("step_results")
        if isinstance(step_results, list):
            candidates.extend(
                step_result.get("exception_info") for step_result in step_results if isinstance(step_result, dict)
            )
        for exception_info in candidates:
            if not isinstance(exception_info, dict):
                continue
            exception_type = exception_info.get("exception_type")
            exception_message = exception_info.get("exception_message")
            parts = [
                part.strip() for part in (exception_type, exception_message) if isinstance(part, str) and part.strip()
            ]
            if parts:
                detail = " | ".join(" ".join(part.split()) for part in parts)
                return f"{result_path.parent.name}: {detail}"[:1500]
    return ""


def validate_harbor_agent_only_job_result(
    result_path: Path,
    *,
    expected_trials: int,
) -> tuple[bool, str]:
    """Validate a verification-disabled Harbor job and its agent result.

    Harbor 0.13.2 records an agent-only trial as completed at the job level,
    but intentionally leaves its evaluation trial and reward counts at zero.
    The per-trial result is therefore the proof that the agent actually ran.
    """
    if not isinstance(expected_trials, int) or isinstance(expected_trials, bool) or expected_trials <= 0:
        return False, f"Expected trial count is invalid: {expected_trials!r}"

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, f"Harbor exited successfully but did not produce {result_path}"
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Harbor produced an unreadable agent-only job result at {result_path}: {exc}"

    if not isinstance(result, dict):
        return False, f"Harbor agent-only job result at {result_path} is not a JSON object"
    total = result.get("n_total_trials")
    stats = result.get("stats")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0 or not isinstance(stats, dict):
        return False, f"Harbor agent-only job result at {result_path} is missing trial statistics"
    if total != expected_trials:
        return False, f"Harbor agent-only job declared {total} trials; expected {expected_trials}"

    counter_names = (
        "n_completed_trials",
        "n_errored_trials",
        "n_running_trials",
        "n_pending_trials",
        "n_cancelled_trials",
        "n_retries",
    )
    counters: dict[str, int] = {}
    for key in counter_names:
        value = stats.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False, f"Harbor agent-only job result has invalid {key}: {value!r}"
        counters[key] = value

    for key, label in (
        ("n_errored_trials", "errored"),
        ("n_running_trials", "running"),
        ("n_pending_trials", "pending"),
        ("n_cancelled_trials", "cancelled"),
    ):
        if counters[key]:
            detail = f"Harbor agent-only job did not complete successfully: {counters[key]} {label}"
            if key == "n_errored_trials" and (exception_detail := _first_trial_exception_detail(result_path.parent)):
                detail = f"{detail}; first trial: {exception_detail}"
            return False, detail
    completed = counters["n_completed_trials"]
    if completed != total:
        return False, f"Harbor agent-only job did not complete successfully: completed {completed}/{total} trials"

    evals = stats.get("evals")
    if not isinstance(evals, dict) or not evals:
        return False, "Harbor agent-only job result has no evaluation statistics"
    for eval_name, eval_stats in evals.items():
        if not isinstance(eval_stats, dict):
            return False, f"Harbor agent-only evaluation {eval_name!r} has invalid statistics"
        for key in ("n_trials", "n_errors"):
            value = eval_stats.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                return False, f"Harbor agent-only evaluation {eval_name!r} has invalid {key}: {value!r}"
        if eval_stats.get("reward_stats") != {}:
            return False, f"Harbor agent-only evaluation {eval_name!r} has invalid reward_stats"

    job_dir = result_path.parent
    try:
        trial_result_paths = sorted(
            child / "result.json" for child in job_dir.iterdir() if child.is_dir() and (child / "result.json").is_file()
        )
    except OSError as exc:
        return False, f"Harbor agent-only trial results at {job_dir} are unreadable: {exc}"
    if len(trial_result_paths) != expected_trials:
        return False, (
            f"Harbor agent-only job did not produce {expected_trials} trial result(s); found {len(trial_result_paths)}"
        )

    for trial_result_path in trial_result_paths:
        try:
            trial_result = json.loads(trial_result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return False, f"Harbor produced an unreadable trial result at {trial_result_path}: {exc}"
        if not isinstance(trial_result, dict):
            return False, f"Harbor trial result at {trial_result_path} is not a JSON object"
        if "exception_info" not in trial_result:
            return False, f"Harbor trial result at {trial_result_path} is missing exception_info"
        if trial_result["exception_info"] is not None:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} recorded an exception"
        agent_result = trial_result.get("agent_result")
        step_results = trial_result.get("step_results")
        if isinstance(agent_result, dict):
            if step_results is not None:
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} has mixed top-level and step agent results"
                )
            continue
        if agent_result is not None:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has invalid agent_result"

        if step_results is None:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has no agent result"
        if not isinstance(step_results, list):
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has invalid step_results"
        if not step_results:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has no step results"
        for step_index, step_result in enumerate(step_results, start=1):
            if not isinstance(step_result, dict):
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} has invalid step result {step_index}"
                )
            step_name = step_result.get("step_name")
            if not isinstance(step_name, str) or not step_name.strip():
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_index} has invalid step_name"
                )
            if "exception_info" not in step_result:
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_name!r} "
                    "is missing exception_info"
                )
            if step_result["exception_info"] is not None:
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_name!r} recorded an exception"
                )
            if not isinstance(step_result.get("agent_result"), dict):
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_name!r} has no agent result"
                )

    return True, ""


def probe_model(provider: ProviderConfig, *, timeout_seconds: float = 15.0) -> ModelProbeResult:
    """Verify that the selected provider catalog lists the requested model."""
    if provider.provider == "bedrock":
        try:
            response = boto3.client("bedrock", region_name=provider.region or "us-west-2").list_foundation_models()
        except (BotoCoreError, ClientError) as exc:
            return ModelProbeResult(
                False,
                provider.provider,
                provider.model,
                f"Bedrock model catalog request failed: {type(exc).__name__}",
            )
        summaries = response.get("modelSummaries") if isinstance(response, dict) else None
        available = {
            str(item["modelId"])
            for item in summaries or []
            if isinstance(item, dict) and isinstance(item.get("modelId"), str)
        }
        aliases = {provider.model}
        prefix, separator, unprefixed = provider.model.partition(".")
        if separator and prefix in {"apac", "eu", "global", "us"}:
            aliases.add(unprefixed)
        if aliases.isdisjoint(available):
            return ModelProbeResult(False, provider.provider, provider.model, f"model {provider.model} is not listed")
        return ModelProbeResult(True, provider.provider, provider.model, f"model {provider.model} is available")

    try:
        records = fetch_model_records(provider, timeout_seconds=timeout_seconds)
    except ModelCatalogError as exc:
        return ModelProbeResult(False, provider.provider, provider.model, str(exc))
    available = {record.id for record in records}
    if provider.model not in available:
        return ModelProbeResult(False, provider.provider, provider.model, f"model {provider.model} is not listed")
    return ModelProbeResult(True, provider.provider, provider.model, f"model {provider.model} is available")


def run_agent_runtime_preflight(
    *,
    dataset: Path,
    agent: str,
    model: str,
    env_mode: str,
    jobs_dir: Path,
    run_env: Mapping[str, str],
    timeout_multiplier: float = 1.0,
    timeout_seconds: int = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    override_cpus: int | None = None,
    override_memory_mb: int | None = None,
    override_storage_mb: int | None = None,
    agent_import_path: str | None = None,
) -> PreflightResult:
    """Start one real agent task and stop before the full A/B matrix."""
    task_name = _first_task_name(dataset)
    job_name = f"runtime-preflight-{agent}"
    if task_name is None:
        return PreflightResult(False, agent, model, "No staged tasks are available for runtime preflight.", job_name)

    command = build_harbor_run_command(
        dataset_path=dataset,
        agent=agent,
        job_name=job_name,
        env_mode=env_mode,
        n_attempts=1,
        n_concurrent=1,
        model=model,
        jobs_dir=jobs_dir,
        timeout_multiplier=timeout_multiplier,
        disable_verification=True,
        include_task_names=[task_name],
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        override_storage_mb=override_storage_mb,
        agent_import_path=agent_import_path,
    )
    try:
        with _nvidia_build_key_handoff(run_env, env_mode=env_mode) as subprocess_env:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=subprocess_env,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return PreflightResult(
            False,
            agent,
            model,
            f"Agent runtime preflight timed out after {timeout_seconds}s.",
            job_name,
        )
    except OSError as exc:
        return PreflightResult(False, agent, model, f"Agent runtime preflight could not start: {exc}", job_name)

    if completed.returncode != 0:
        output = "\n".join(part for part in (completed.stderr, completed.stdout) if part).strip()
        detail = _redact_detail(output, run_env) or f"harbor run exited {completed.returncode}"
        return PreflightResult(False, agent, model, detail, job_name)

    ok, detail = validate_harbor_agent_only_job_result(
        jobs_dir / job_name / "result.json",
        expected_trials=1,
    )
    return PreflightResult(ok, agent, model, _redact_detail(detail, run_env), job_name)
