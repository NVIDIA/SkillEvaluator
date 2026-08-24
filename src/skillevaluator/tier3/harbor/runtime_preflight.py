# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded one-task Harbor smoke runs for agent runtime readiness."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from skillevaluator.model_catalog import (
    ModelCatalogError,
    ModelCatalogFailureKind,
    fetch_anthropic_model_record,
    fetch_model_records,
)
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
    failure_kind: ModelCatalogFailureKind | None = None
    http_status: int | None = None


class CredentialProbeDisposition(StrEnum):
    """How Tier 3 should act on a model-catalog credential probe."""

    VERIFIED = "verified"
    FATAL = "fatal"
    DEGRADED = "degraded"


_BEDROCK_PROBE_SLOT = BoundedSemaphore(4)


def _is_native_catalog_endpoint(provider: ProviderConfig) -> bool:
    """Return whether catalog HTTP status has the provider's native meaning."""
    provider_name = provider.provider.casefold()
    if provider_name == "anthropic" and not provider.base_url:
        return True
    if not isinstance(provider.base_url, str) or provider.base_url != provider.base_url.strip():
        return False
    if "\\" in provider.base_url or any(character in provider.base_url for character in ("?", "#", ";")):
        return False
    try:
        endpoint = urlsplit(provider.base_url)
        port = endpoint.port
    except (TypeError, ValueError):
        return False
    if (
        endpoint.scheme.casefold() != "https"
        or endpoint.hostname is None
        or endpoint.username is not None
        or endpoint.password is not None
        or port not in {None, 443}
    ):
        return False

    if provider_name in {"openai", "openai-compatible"}:
        return endpoint.hostname.casefold() == "api.openai.com" and endpoint.path in {"/v1", "/v1/"}
    if provider_name == "nv_build":
        return endpoint.hostname.casefold() == "integrate.api.nvidia.com" and endpoint.path in {"/v1", "/v1/"}
    if provider_name == "anthropic":
        return endpoint.hostname.casefold() == "api.anthropic.com" and endpoint.path in {"", "/", "/v1", "/v1/"}
    return False


def _catalog_listing_is_authoritative(provider: ProviderConfig) -> bool:
    """Return whether absence from the listing proves the model is unusable."""
    # Anthropic aliases resolve through its single-model endpoint but are not
    # guaranteed to appear in the unique-ID listing. Bedrock's foundation-model
    # listing excludes other valid InvokeModel resources such as inference profiles.
    return provider.provider.casefold() in {"openai", "openai-compatible", "nv_build"} and (
        _is_native_catalog_endpoint(provider)
    )


def credential_probe_disposition(
    provider: ProviderConfig,
    probe: ModelProbeResult,
) -> CredentialProbeDisposition:
    """Classify a live catalog probe without rejecting compatible custom gateways."""
    if probe.ok:
        return CredentialProbeDisposition.VERIFIED

    raw_kind = getattr(probe, "failure_kind", None)
    try:
        failure_kind = ModelCatalogFailureKind(raw_kind) if raw_kind is not None else None
    except (TypeError, ValueError):
        failure_kind = ModelCatalogFailureKind.UNKNOWN

    if failure_kind == ModelCatalogFailureKind.AUTHENTICATION:
        return CredentialProbeDisposition.FATAL
    if failure_kind == ModelCatalogFailureKind.INVALID_CONFIGURATION:
        return CredentialProbeDisposition.FATAL
    if failure_kind == ModelCatalogFailureKind.MODEL_NOT_FOUND:
        return (
            CredentialProbeDisposition.FATAL
            if _is_native_catalog_endpoint(provider)
            else CredentialProbeDisposition.DEGRADED
        )
    if failure_kind == ModelCatalogFailureKind.AUTHORIZATION:
        if provider.provider == "bedrock":
            # ListFoundationModels permission is distinct from InvokeModel.
            return CredentialProbeDisposition.DEGRADED
        return (
            CredentialProbeDisposition.FATAL
            if _is_native_catalog_endpoint(provider)
            else CredentialProbeDisposition.DEGRADED
        )
    if failure_kind is None:
        # ``probe_model`` reached the catalog but did not find the selected model.
        return (
            CredentialProbeDisposition.FATAL
            if _catalog_listing_is_authoritative(provider)
            else CredentialProbeDisposition.DEGRADED
        )
    return CredentialProbeDisposition.DEGRADED


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


def _probe_bedrock_model(provider: ProviderConfig, *, timeout_seconds: float) -> ModelProbeResult:
    """Run one Bedrock catalog request inside the outer deadline worker."""
    try:
        request_config = BotoConfig(
            connect_timeout=timeout_seconds,
            read_timeout=timeout_seconds,
            retries={"max_attempts": 0},
        )
        response = (
            boto3.session.Session()
            .client(
                "bedrock",
                region_name=provider.region or "us-west-2",
                config=request_config,
            )
            .list_foundation_models()
        )
    except ClientError as exc:
        response = exc.response if isinstance(exc.response, dict) else {}
        error = response.get("Error") if isinstance(response.get("Error"), dict) else {}
        metadata = response.get("ResponseMetadata") if isinstance(response.get("ResponseMetadata"), dict) else {}
        error_code = str(error.get("Code") or "")
        http_status = metadata.get("HTTPStatusCode")
        if not isinstance(http_status, int) or isinstance(http_status, bool):
            http_status = None
        if http_status == 401 or error_code in {
            "ExpiredTokenException",
            "IncompleteSignature",
            "InvalidClientTokenId",
            "InvalidSignatureException",
            "UnrecognizedClientException",
        }:
            failure_kind = ModelCatalogFailureKind.AUTHENTICATION
        elif error_code in {"AccessDenied", "AccessDeniedException"}:
            failure_kind = ModelCatalogFailureKind.AUTHORIZATION
        elif error_code in {"ServiceUnavailable", "ServiceUnavailableException", "ThrottlingException"} or (
            isinstance(http_status, int) and (http_status == 429 or http_status >= 500)
        ):
            failure_kind = ModelCatalogFailureKind.UNAVAILABLE
        else:
            failure_kind = ModelCatalogFailureKind.UNKNOWN
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            f"Bedrock model catalog request failed: {type(exc).__name__}",
            failure_kind=failure_kind,
            http_status=http_status,
        )
    except BotoCoreError as exc:
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            f"Bedrock model catalog request failed: {type(exc).__name__}",
            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
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


def _probe_bedrock_model_with_deadline(
    provider: ProviderConfig,
    *,
    timeout_seconds: float,
) -> ModelProbeResult:
    """Bound session creation, credential providers, retries, and request I/O."""
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "model catalog timeout must be a positive number",
            failure_kind=ModelCatalogFailureKind.INVALID_CONFIGURATION,
        )

    result_queue: Queue[ModelProbeResult] = Queue(maxsize=1)
    deadline = monotonic() + timeout_seconds
    if not _BEDROCK_PROBE_SLOT.acquire(timeout=timeout_seconds):
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "Bedrock model catalog request timed out",
            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
        )

    def run_probe() -> None:
        try:
            try:
                result = _probe_bedrock_model(provider, timeout_seconds=remaining)
            except Exception as exc:
                result = ModelProbeResult(
                    False,
                    provider.provider,
                    provider.model,
                    f"Bedrock model catalog request failed: {type(exc).__name__}",
                    failure_kind=ModelCatalogFailureKind.UNKNOWN,
                )
            result_queue.put_nowait(result)
        finally:
            _BEDROCK_PROBE_SLOT.release()

    remaining = deadline - monotonic()
    if remaining <= 0:
        _BEDROCK_PROBE_SLOT.release()
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "Bedrock model catalog request timed out",
            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
        )

    Thread(target=run_probe, name="bedrock-model-catalog-probe", daemon=True).start()
    try:
        return result_queue.get(timeout=remaining)
    except Empty:
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "Bedrock model catalog request timed out",
            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
        )


def probe_model(provider: ProviderConfig, *, timeout_seconds: float = 15.0) -> ModelProbeResult:
    """Verify that the selected provider catalog lists the requested model."""
    if provider.provider == "bedrock":
        return _probe_bedrock_model_with_deadline(provider, timeout_seconds=timeout_seconds)

    try:
        records = fetch_model_records(provider, timeout_seconds=timeout_seconds)
    except ModelCatalogError as exc:
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            str(exc),
            failure_kind=exc.kind,
            http_status=exc.http_status,
        )
    available = {record.id for record in records}
    if provider.model not in available:
        if provider.provider == "anthropic" and _is_native_catalog_endpoint(provider):
            try:
                resolved = fetch_anthropic_model_record(
                    provider,
                    provider.model,
                    timeout_seconds=timeout_seconds,
                )
            except ModelCatalogError as exc:
                if exc.http_status == 404:
                    return ModelProbeResult(
                        False,
                        provider.provider,
                        provider.model,
                        f"model {provider.model} is not available",
                        failure_kind=ModelCatalogFailureKind.MODEL_NOT_FOUND,
                        http_status=404,
                    )
                return ModelProbeResult(
                    False,
                    provider.provider,
                    provider.model,
                    str(exc),
                    failure_kind=exc.kind,
                    http_status=exc.http_status,
                )
            return ModelProbeResult(
                True,
                provider.provider,
                provider.model,
                f"model {provider.model} resolves to {resolved.id}",
            )
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
