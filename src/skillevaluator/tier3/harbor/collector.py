# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor results collector -- reads Harbor job directories and consolidates
results into the evals/results/<agent>/ structure.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import unicodedata
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from harbor.models.trajectories import Trajectory

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts, get_skill_tool_calls
from skillevaluator.tier3.eval_core.checks import check_negative_case
from skillevaluator.tier3.harbor.metrics import (
    CUSTOM_ONLY_METRIC_SET,
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    LEGACY_METRIC_SET,
    LEGACY_METRICS,
    MAX_CUSTOM_METRICS,
    RESERVED_METRIC_NAMES,
    CustomMetricContractError,
    average_custom_metrics,
    average_metrics,
    custom_metric_contract_error,
    custom_metric_name_is_publishable,
    dimension_scores,
    extract_custom_metrics,
    metric_set_for_reward,
    metric_value,
    overall_score,
    rewards_have_mixed_metric_contracts,
    score_definition,
    score_value,
)
from skillevaluator.tier3.output_provenance import write_output_file_atomically
from skillevaluator.utils.redaction import contains_credential_value, redact_sensitive_data, redact_sensitive_text
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, stat_is_link_or_reparse

logger = logging.getLogger(__name__)

DISPLAY_METRICS = DEFAULT_METRICS
_LOGICAL_ATTEMPT_SENTINEL = object()
_CUSTOM_METRIC_CONTRACT_MARKER = "_skill_evaluator_custom_metric_contract_error"
TRUNCATED_AGGREGATE_ATTEMPT_PREFIX = "__skillevaluator_attempt"
_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1
DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES = 5 * 1024 * 1024
DIAGNOSTIC_ARTIFACT_HARD_MAX_BYTES = 64 * 1024 * 1024
REWARD_DIAGNOSTIC_STRING_MAX_CHARS = 8192
REWARD_METADATA_TEXT_MAX_CHARS = 512
REWARD_IDENTITY_TEXT_MAX_BYTES = 512
REWARD_JSON_MAX_DEPTH = 64
REWARD_JSON_MAX_NODES = 50_000
GENERATED_JSON_MAX_BYTES = 2 * 1024 * 1024
# Reserve room for collector-owned identity, model, and evidence annotations so
# a source reward that passes extraction remains readable after publication.
COLLECTED_REWARD_JSON_MAX_NODES = REWARD_JSON_MAX_NODES - 512
COLLECTED_REWARD_JSON_MAX_BYTES = GENERATED_JSON_MAX_BYTES - (256 * 1024)
ATIF_JSON_MAX_DEPTH = 64
ATIF_JSON_MAX_NODES = 50_000
PORTABLE_TRIAL_COMPONENT_MAX_UNITS = 240
PUBLISHED_CASE_DETAILS_MAX = 256
PUBLISHED_ATTEMPT_DETAILS_MAX = 512
PUBLISHED_ATTEMPT_DETAILS_PER_CASE_MAX = 8
PUBLISHED_CASE_ID_DIAGNOSTIC_SAMPLE_MAX = 32
PUBLISHED_FAILURE_DETAILS_MAX = 32
PUBLISHED_EXECUTION_ERRORS_MAX = 256
PUBLISHED_JOB_FAILURE_MAX_CHARS = 4096
UNSCOREABLE_NUMERIC_REWARD_REASON = "Reward metrics are incomplete or non-finite; trial was not scored"
UNSAFE_REWARD_STRUCTURE_REASON = "Reward payload exceeds safe structural limits; trial was not scored"
UNSAFE_REWARD_IDENTITY_REASON = "Reward identity violates the bounded publication contract; trial was not scored"
UNSAFE_CUSTOM_METRICS_REASON = "Custom metrics exceed the bounded publication contract; trial was not scored"
MALFORMED_HARBOR_REWARD_REASON = "Harbor verifier rewards contain nonnumeric metric values; trial was not scored"
UNSAFE_CUSTOM_METRIC_UNION_REASON = (
    "Custom metric union exceeds the per condition publication limit; condition was not scored"
)
MISSING_MULTI_STEP_REWARD_REASON = (
    "Authoritative multi-step verifier reward is missing; it was not reconstructed or scored"
)
TRIAL_DIAGNOSTIC_ARTIFACTS = ("result.json", "config.json", "exception.txt", "trial.log")
AGENT_LOG_ARTIFACTS = (
    "trajectory.json",
    "cursor-cli.txt",
    "claude-code.txt",
    "codex.txt",
    "aider.txt",
    "goose.txt",
    "mini-swe-agent.txt",
    "openhands.txt",
    "gemini-cli.txt",
    "cline.txt",
    "opencode.txt",
)
GENERATED_AGENT_ARTIFACTS = (
    "lift.json",
    "custom_lift.json",
    "pass_at_k_lift.json",
    "security_attribution.json",
    "findings.json",
)
GENERATED_CONDITION_DIRS = ("with-skill", "without-skill")
GENERATED_ROOT_ARTIFACTS = ("attempt_policy.json", "comparison.json")
_MAX_FAILED_JUDGE_SIDECARS = 64
_MAX_FAILED_JUDGE_STEP_PATHS_SCANNED = 256
_MAX_TRAJECTORY_REFERENCE_FILES = 64
_MAX_TRAJECTORY_REFERENCE_DEPTH = 16
_MAX_TRAJECTORY_REFERENCE_TOTAL_BYTES = 32 * 1024 * 1024
_MAX_TRAJECTORY_STEP_DIRECTORIES = 64
_ATIF_TOKEN_ID_FIELDS = frozenset({"completion_token_ids", "prompt_token_ids"})
_TRAJECTORY_NOT_PROVIDED = object()


def _is_aggregate_extra_token_key(key: str) -> bool:
    """Return true for final_metrics.extra token counters that should be summed."""
    return key == "reasoning_output_tokens" or (key.startswith("total_") and "token" in key)


_AGENT_RUNTIME_FAILURE_PATTERNS = (
    "API Error:",
    "AuthenticationError",
    "Unauthorized",
    "401",
    "404 Not Found",
    "405 Method Not Allowed",
    "invalid_api_key",
    "invalid api key",
    "Missing API key",
    "missing api key",
    "model_not_found",
    "model not found",
    "NotFoundError",
    "ProviderException",
    "ResourceExhausted",
    "context_management: Extra inputs are not permitted",
    "isApiErrorMessage",
    "Model Group Fallbacks=None",
)

_AGENT_RUNTIME_EXCEPTION_TYPES = {
    "NonZeroAgentExitCodeError",
    "AuthenticationError",
    "NotFoundError",
    "ProviderException",
}
_UNCONDITIONAL_AGENT_RUNTIME_EXCEPTION_TYPES = {
    "AgentAuthenticationError",
    "AgentTimeoutError",
    "ApiConnectionClosedError",
    "ApiInternalServerError",
    "ApiOverloadedError",
    "ApiProviderResourceNotFoundError",
    "ApiRateLimitError",
    "ApiResponseStalledError",
    "ApiUsageLimitError",
    "ContextWindowExceededError",
    "ModelNotFoundError",
    "NetworkConnectionError",
    "OutputTokenExceededError",
    "UnknownApiError",
}


def _agent_runtime_failure_pattern_start(value: str) -> int | None:
    for pattern in _AGENT_RUNTIME_FAILURE_PATTERNS:
        if pattern == "401":
            match = re.search(r"(?<![A-Za-z0-9_])401(?![A-Za-z0-9_])", value)
        elif (pattern[0].isalnum() or pattern[0] == "_") and (pattern[-1].isalnum() or pattern[-1] == "_"):
            match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(pattern)}(?![A-Za-z0-9_])", value)
        else:
            idx = value.find(pattern)
            if idx >= 0:
                return idx
            continue
        if match:
            return match.start()
    return None


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_safe_generated_output_path(
    path: Path,
    output_root: Path,
    *,
    follow_target: bool,
) -> None:
    """Reject cleanup targets that can escape the configured results directory."""
    lexical_root = output_root.absolute()
    lexical_path = path.absolute()
    try:
        resolved_root = output_root.resolve()
        resolved_parent = path.parent.resolve()
        target_within_root = not follow_target or _path_is_within(path.resolve(), resolved_root)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Refusing unsafe generated output path: {path}") from error
    if (
        lexical_path == lexical_root
        or not _path_is_within(lexical_path, lexical_root)
        or not _path_is_within(resolved_parent, resolved_root)
        or not target_within_root
    ):
        raise ValueError(f"Refusing to modify generated output outside results directory: {path}")


def _remove_generated_output_path(path: Path, output_root: Path) -> None:
    """Remove one known generated path without following a target symlink."""
    _assert_safe_generated_output_path(path, output_root, follow_target=False)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _write_generated_root_json(path: Path, output_root: Path, payload: Any) -> None:
    """Publish one bounded generated artifact without following replacements."""
    _assert_safe_generated_output_path(path, output_root, follow_target=False)
    _validate_generated_json_value(
        payload,
        max_depth=REWARD_JSON_MAX_DEPTH,
        max_nodes=REWARD_JSON_MAX_NODES,
        max_bytes=GENERATED_JSON_MAX_BYTES,
    )
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    write_output_file_atomically(path, encoded)


def _agent_generated_output_paths(agent_dir: Path) -> list[Path]:
    paths = [agent_dir / artifact for artifact in GENERATED_AGENT_ARTIFACTS]
    for condition in GENERATED_CONDITION_DIRS:
        condition_dir = agent_dir / condition
        paths.extend((condition_dir / "summary.json", condition_dir / "trials"))
    return paths


def _validate_agent_generated_outputs(agent_dir: Path, output_root: Path) -> None:
    _assert_safe_generated_output_path(agent_dir, output_root, follow_target=True)
    if agent_dir.is_symlink():
        raise ValueError(f"Refusing symlinked generated output directory: {agent_dir}")
    for condition in GENERATED_CONDITION_DIRS:
        condition_dir = agent_dir / condition
        if condition_dir.is_symlink():
            raise ValueError(f"Refusing symlinked generated output directory: {condition_dir}")
    for path in _agent_generated_output_paths(agent_dir):
        _assert_safe_generated_output_path(path, output_root, follow_target=False)


def _reset_agent_generated_outputs(agent_dir: Path, output_root: Path) -> None:
    """Clear only collector/report-owned artifacts before rebuilding an agent result."""
    _validate_agent_generated_outputs(agent_dir, output_root)
    agent_dir.mkdir(parents=True, exist_ok=True)
    for path in _agent_generated_output_paths(agent_dir):
        _remove_generated_output_path(path, output_root)


def _find_job_dir(jobs_dir: Path, job_name: str) -> Path | None:
    """Find the exact Harbor job directory produced for ``job_name``."""
    candidate = jobs_dir / job_name
    return candidate if candidate.is_dir() else None


def _safe_text(value: Any, *, max_len: int | None = 2048) -> str:
    text = redact_sensitive_text(str(value or ""))
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 14] + "...<truncated>"
    return text


def _safe_diagnostic_text(value: Any, *, max_len: int) -> str:
    """Return redacted, bounded text without terminal-control characters."""
    return re.sub(r"[\x00-\x1f\x7f]", " ", _safe_text(value, max_len=max_len)).strip()


def _published_job_failure(value: Any) -> str:
    """Return one validation/launch failure safe for generated results."""
    return _safe_diagnostic_text(value, max_len=PUBLISHED_JOB_FAILURE_MAX_CHARS)


def _identity_text_is_publishable(value: object) -> bool:
    """Return whether an identity can safely be used as a generated JSON key."""
    if not isinstance(value, str) or not value or value != value.strip() or not value.isprintable():
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        return False
    return len(encoded) <= REWARD_IDENTITY_TEXT_MAX_BYTES and not contains_credential_value(value)


def _published_trial_label(value: object, *, alias_ordinal: int | None = None) -> str:
    """Return a bounded value-only label; unsafe identities never become keys or paths."""
    if isinstance(value, str) and value and value.isprintable() and not contains_credential_value(value):
        try:
            if len(value.encode("utf-8")) <= REWARD_IDENTITY_TEXT_MAX_BYTES:
                return value
        except UnicodeError:
            pass
    if alias_ordinal is not None:
        return f"redacted-or-invalid-trial-{alias_ordinal:06d}"
    return "redacted-or-invalid-trial"


def _validated_expected_case_ids(expected_case_ids: list[str] | None) -> list[str]:
    """Validate caller-owned case identities before generated outputs are reset."""
    validated: list[str] = []
    seen: dict[str, str] = {}
    for case_id in expected_case_ids or []:
        if not _identity_text_is_publishable(case_id):
            raise ValueError("Expected case identity violates the bounded publication contract")
        collision_key = case_id.casefold()
        if collision_key in seen:
            raise ValueError(
                "Expected case identities must be unique without cross-platform collisions: "
                f"{case_id!r} conflicts with {seen[collision_key]!r}"
            )
        seen[collision_key] = case_id
        validated.append(case_id)
    return validated


def _validated_case_id_by_task_selector(
    case_id_by_task_selector: dict[str, str] | None,
    expected_case_ids: list[str],
) -> dict[str, str] | None:
    """Copy one trusted staged-selector mapping after rejecting ambiguity."""
    if case_id_by_task_selector is None:
        return None

    validated: dict[str, str] = {}
    seen_selectors: dict[str, str] = {}
    seen_case_ids: dict[str, str] = {}
    for selector, case_id in case_id_by_task_selector.items():
        if not _identity_text_is_publishable(selector) or not _identity_text_is_publishable(case_id):
            raise ValueError("Task selector mapping violates the bounded publication contract")
        selector_key = selector.casefold()
        if selector_key in seen_selectors:
            raise ValueError(
                "Task selector mapping contains duplicate or cross-platform colliding selectors: "
                f"{selector!r} conflicts with {seen_selectors[selector_key]!r}"
            )
        case_id_key = case_id.casefold()
        if case_id_key in seen_case_ids:
            raise ValueError(
                "Task selector mapping must contain unique logical case identities without cross-platform collisions: "
                f"{case_id!r} conflicts with {seen_case_ids[case_id_key]!r}"
            )
        seen_selectors[selector_key] = selector
        seen_case_ids[case_id_key] = case_id
        validated[selector] = case_id

    if expected_case_ids and set(validated.values()) != set(expected_case_ids):
        raise ValueError("Task selector mapping must match the expected logical case identities")
    return validated


def _sampled_case_id_diagnostic(label: str, case_ids: list[str]) -> str:
    """Render bounded case-coverage diagnostics while retaining an exact count."""
    if not case_ids:
        return ""
    sample = case_ids[:PUBLISHED_CASE_ID_DIAGNOSTIC_SAMPLE_MAX]
    rendered = ", ".join(sample)
    if len(case_ids) > len(sample):
        return f"{label} (showing {len(sample)} of {len(case_ids)}): {rendered}"
    return f"{label}: {rendered}"


def _public_failure_list(failures: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Return a bounded, redacted sample of trial-level failures."""
    public: list[dict[str, str]] = []
    for failure in (failures or [])[:PUBLISHED_FAILURE_DETAILS_MAX]:
        public.append(
            {
                "trial": _published_trial_label(failure.get("trial", "unknown trial")),
                "reason": _safe_diagnostic_text(failure.get("reason", "unknown error"), max_len=2048),
            }
        )
    return public


def _failure_list_metadata(prefix: str, failures: list[dict[str, str]] | None) -> dict[str, Any]:
    """Describe exact failure cardinality beside a bounded published sample."""
    total = len(failures or [])
    shown = min(total, PUBLISHED_FAILURE_DETAILS_MAX)
    return {
        f"{prefix}_total": total,
        f"{prefix}_shown": shown,
        f"{prefix}_truncated": shown < total,
    }


def _safe_evaluation_errors(value: Any) -> dict[str, str] | list[str] | str:
    """Normalize verifier diagnostics before persisting or displaying them."""
    if isinstance(value, dict):
        normalized: dict[str, str] = {}
        for metric, reason in list(value.items())[: len(DEFAULT_METRICS)]:
            safe_metric = _safe_diagnostic_text(metric, max_len=64) or "judge"
            safe_reason = _safe_diagnostic_text(reason, max_len=512)
            if safe_reason:
                normalized[safe_metric] = safe_reason
        return normalized
    if isinstance(value, list):
        return [
            safe_reason
            for reason in value[: len(DEFAULT_METRICS)]
            if (safe_reason := _safe_diagnostic_text(reason, max_len=512))
        ]
    return _safe_diagnostic_text(value, max_len=512)


def _bounded_reward_metadata_text(value: Any) -> str | None:
    """Return one bounded, redacted collector-owned metadata label."""
    if not isinstance(value, str) or not value:
        return None
    return _safe_diagnostic_text(value, max_len=REWARD_METADATA_TEXT_MAX_CHARS) or None


def _read_json(path: Path) -> Any:
    """Read one bounded regular JSON file through an anchored no-follow root."""
    try:
        with SecureRoot(path.parent) as secure_root:
            raw, _metadata = secure_root.read_bytes(Path(path.name), DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES)
        return json.loads(raw)
    except (SecurePathError, ValueError, OSError, RecursionError, UnicodeError):
        return None


def _read_bounded_text(path: Path, *, max_bytes: int = DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES) -> str | None:
    """Read bounded diagnostic text without following links or special files."""
    try:
        with SecureRoot(path.parent) as secure_root:
            raw, _metadata = secure_root.read_bytes(Path(path.name), max_bytes)
        return raw.decode("utf-8", errors="replace")
    except (SecurePathError, ValueError, OSError, RecursionError, UnicodeError):
        return None


def _is_collector_owned_agent_dir(agent_dir: Path) -> bool:
    """Recognize a prior collector directory without claiming arbitrary user content."""
    if agent_dir.is_symlink():
        return False
    for condition in GENERATED_CONDITION_DIRS:
        condition_dir = agent_dir / condition
        summary = condition_dir / "summary.json"
        if condition_dir.is_symlink() or summary.is_symlink():
            continue
        try:
            if not summary.is_file() or summary.stat().st_size > DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES:
                continue
        except OSError:
            continue
        data = _read_json(summary)
        if (
            isinstance(data, dict)
            and data.get("agent") == agent_dir.name
            and isinstance(data.get("scores"), dict)
            and data.get("execution_status") in {"failed", "skipped", "succeeded"}
        ):
            return True
    return False


def _collector_owned_agent_dirs(output_root: Path) -> list[Path]:
    try:
        with os.scandir(output_root) as entries:
            candidates = [
                Path(entry.path) for entry in entries if not entry.is_symlink() and entry.is_dir(follow_symlinks=False)
            ]
    except OSError:
        return []
    return sorted((path for path in candidates if _is_collector_owned_agent_dir(path)), key=lambda path: path.name)


def _prepare_generated_outputs(output_root: Path, agents: list[str]) -> None:
    """Validate the complete cleanup plan, then reset current and omitted generated outputs."""
    current_agent_dirs = [output_root / agent for agent in agents]
    planned_agent_dirs: dict[Path, Path] = {
        path.absolute(): path for path in [*current_agent_dirs, *_collector_owned_agent_dirs(output_root)]
    }
    root_artifacts = [output_root / artifact for artifact in GENERATED_ROOT_ARTIFACTS]

    for path in root_artifacts:
        _assert_safe_generated_output_path(path, output_root, follow_target=False)
    for path in planned_agent_dirs.values():
        _validate_agent_generated_outputs(path, output_root)

    for path in root_artifacts:
        _remove_generated_output_path(path, output_root)
    for path in planned_agent_dirs.values():
        _reset_agent_generated_outputs(path, output_root)


def validate_harbor_job_result(
    result_path: Path,
    *,
    expected_trials: int | None = None,
    expected_total_trials: int | None = None,
) -> tuple[bool, str]:
    """Validate Harbor's persisted aggregate trial state.

    Harbor returning zero is only subprocess success.  A usable result must
    account for every requested logical trial and include the trial names that
    contributed rewards.  This intentionally validates Harbor's public
    current ``stats.n_completed_trials`` / ``stats.evals`` schema (while still
    reading Harbor's migrated legacy counters) rather than accepting a shaped
    object containing only a top-level total.
    """
    if expected_trials is not None and expected_total_trials is not None and expected_trials != expected_total_trials:
        return False, "Conflicting expected trial counts were provided"
    if expected_trials is None:
        expected_trials = expected_total_trials

    if not result_path.exists():
        return False, f"Harbor exited successfully but did not produce {result_path}"
    result = _read_json(result_path)
    if result is None:
        return False, f"Harbor produced an unreadable job result at {result_path}"

    if not isinstance(result, dict):
        return False, f"Harbor job result at {result_path} is not a JSON object"
    total = result.get("n_total_trials")
    stats = result.get("stats")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0 or not isinstance(stats, dict):
        return False, f"Harbor job result at {result_path} is missing trial statistics"
    if total <= 0:
        return False, "Harbor completed with zero trials"
    if expected_trials is not None:
        if not isinstance(expected_trials, int) or isinstance(expected_trials, bool) or expected_trials <= 0:
            return False, f"Expected trial count is invalid: {expected_trials!r}"
        if total != expected_trials:
            return False, f"Harbor job declared {total} trials; expected {expected_trials}"

    current_counter_names = (
        "n_completed_trials",
        "n_errored_trials",
        "n_running_trials",
        "n_pending_trials",
        "n_cancelled_trials",
        "n_retries",
    )
    if any(key in stats for key in current_counter_names):
        current_counters: dict[str, int] = {}
        for key in current_counter_names:
            value = stats.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return False, f"Harbor job result has invalid {key}: {value!r}"
            current_counters[key] = value
        completed = current_counters["n_completed_trials"]
        errors = current_counters["n_errored_trials"]
        for key, label in (
            ("n_running_trials", "running"),
            ("n_pending_trials", "pending"),
            ("n_cancelled_trials", "cancelled"),
        ):
            if current_counters[key]:
                return False, (f"Harbor job did not complete successfully: {current_counters[key]} {label}")
    else:
        completed = stats.get("n_trials")
        errors = stats.get("n_errors")
        for key, value in (("n_trials", completed), ("n_errors", errors)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return False, f"Harbor job result has invalid {key}: {value!r}"
    if errors:
        return False, f"Harbor job did not complete successfully: {errors} errored"
    if completed != total:
        return False, f"Harbor job did not complete successfully: completed {completed}/{total} trials"

    evals = stats.get("evals")
    if not isinstance(evals, dict) or not evals:
        return False, "Harbor job result has no evaluation statistics"

    eval_trials = 0
    eval_errors = 0
    rewarded_trial_names: set[str] = set()
    for eval_name, eval_stats in evals.items():
        if not isinstance(eval_stats, dict):
            return False, f"Harbor evaluation {eval_name!r} has invalid statistics"
        n_trials = eval_stats.get("n_trials")
        n_errors = eval_stats.get("n_errors")
        for key, value in (("n_trials", n_trials), ("n_errors", n_errors)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return False, f"Harbor evaluation {eval_name!r} has invalid {key}: {value!r}"
        eval_trials += n_trials
        eval_errors += n_errors

        reward_stats = eval_stats.get("reward_stats")
        if not isinstance(reward_stats, dict):
            return False, f"Harbor evaluation {eval_name!r} has invalid reward_stats"
        for metric_stats in reward_stats.values():
            if not isinstance(metric_stats, dict):
                return False, f"Harbor evaluation {eval_name!r} has invalid reward statistics"
            metric_trial_names: list[str] = []
            for trial_names in metric_stats.values():
                if not isinstance(trial_names, list) or any(
                    not isinstance(name, str) or not name for name in trial_names
                ):
                    return False, f"Harbor evaluation {eval_name!r} has invalid rewarded trial names"
                metric_trial_names.extend(trial_names)
            if len(metric_trial_names) != len(set(metric_trial_names)):
                return False, f"Harbor evaluation {eval_name!r} has duplicate rewarded trial names"
            rewarded_trial_names.update(metric_trial_names)

    if eval_trials != total:
        return False, f"Harbor evaluation statistics account for {eval_trials}/{total} completed trials"
    if eval_errors != 0:
        return False, f"Harbor evaluation statistics contain {eval_errors} errored trials"
    if not rewarded_trial_names:
        return False, "Harbor job result has no scored trial names"
    if len(rewarded_trial_names) != total:
        return False, f"Harbor reward statistics cover {len(rewarded_trial_names)}/{total} trials"
    return True, ""


def _read_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _text_contains_agent_runtime_failure(text: str) -> str:
    def _reason_from_value(value: Any) -> str:
        if isinstance(value, str):
            idx = _agent_runtime_failure_pattern_start(value)
            if idx is not None:
                return value[idx:].strip()[:600]
            return ""
        if isinstance(value, dict):
            for item in value.values():
                reason = _reason_from_value(item)
                if reason:
                    return reason
        if isinstance(value, list):
            for item in value:
                reason = _reason_from_value(item)
                if reason:
                    return reason
        return ""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = _read_json_text(stripped)
        if parsed is not None:
            reason = _reason_from_value(parsed)
            if reason:
                return reason
        reason = _reason_from_value(stripped)
        if reason:
            return reason
    return ""


def _trajectory_agent_runtime_failure_reason(trajectory: Any) -> str:
    if not isinstance(trajectory, dict):
        return ""
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return ""
    final_metrics = trajectory.get("final_metrics") or {}
    total_prompt = final_metrics.get("total_prompt_tokens")
    total_completion = final_metrics.get("total_completion_tokens")
    tokenless = (total_prompt in (None, 0)) and (total_completion in (None, 0))

    for step in steps:
        if not isinstance(step, dict):
            continue
        message = str(step.get("message") or "")
        reason = _text_contains_agent_runtime_failure(message)
        if reason and tokenless:
            return reason
    return ""


def _exception_details(exception_info: Any) -> tuple[str, str]:
    """Return one Harbor exception type and bounded display reason."""
    if not isinstance(exception_info, dict):
        return "", ""

    exception_type = str(exception_info.get("exception_type") or "").strip()
    exception_message = str(exception_info.get("exception_message") or "").strip()
    if exception_type and exception_message:
        return exception_type, f"{exception_type}: {exception_message}"[:600]
    return exception_type, (exception_type or exception_message)[:600]


def _trial_exception_details(trial_dir: Path) -> tuple[str, str]:
    """Return the Harbor trial-root exception type and display reason, if present."""
    result = _read_json(trial_dir / "result.json")
    if not isinstance(result, dict):
        return "", ""
    return _exception_details(result.get("exception_info"))


def _trial_step_exception_details(trial_dir: Path) -> list[tuple[str, str]]:
    """Return ordered native multi-step exception types and display reasons."""
    result = _read_json(trial_dir / "result.json")
    if not isinstance(result, dict):
        return []
    step_results = result.get("step_results")
    if not isinstance(step_results, list):
        return []
    return [
        details
        for step in step_results
        if isinstance(step, dict)
        if (details := _exception_details(step.get("exception_info"))) != ("", "")
    ]


def _agent_log_runtime_failure_reason(
    trial_dir: Path,
    *,
    include_text_logs: bool = True,
) -> str:
    """Return the most specific agent log/runtime startup failure, if present."""
    for trajectory_path in (trial_dir / "agent" / "trajectory.json", trial_dir / "trajectory.json"):
        if trajectory_path.exists():
            reason = _trajectory_agent_runtime_failure_reason(_read_json(trajectory_path))
            if reason:
                return reason

    for path in (
        trial_dir / "agent" / "claude-code.txt",
        trial_dir / "claude-code.txt",
        trial_dir / "agent" / "codex.txt",
        trial_dir / "codex.txt",
        trial_dir / "agent" / "opencode.txt",
        trial_dir / "opencode.txt",
    ):
        text = _read_bounded_text(path)
        if text is None:
            continue
        for line in text.splitlines():
            parsed = _read_json_text(line.strip())
            if not isinstance(parsed, dict) or str(parsed.get("type") or "").casefold() != "error":
                continue
            reason = _text_contains_agent_runtime_failure(line)
            if reason:
                return reason.strip('"')
        if include_text_logs:
            reason = _text_contains_agent_runtime_failure(text)
            if reason:
                return reason

    return ""


def _agent_runtime_failure_reason(trial_dir: Path) -> str:
    """Return why a trial cannot produce a valid score."""
    exception_details = [
        _trial_exception_details(trial_dir),
        *_trial_step_exception_details(trial_dir),
    ]
    agent_reason = _agent_log_runtime_failure_reason(
        trial_dir,
        include_text_logs=any(reason for _exception_type, reason in exception_details),
    )
    if agent_reason:
        return agent_reason

    for exception_type, exception_reason in exception_details:
        if exception_type in _UNCONDITIONAL_AGENT_RUNTIME_EXCEPTION_TYPES:
            return exception_reason

        # Do not classify verifier/healthcheck/task exceptions as agent runtime failures.
        if (
            exception_type in _AGENT_RUNTIME_EXCEPTION_TYPES
            and exception_reason
            and _text_contains_agent_runtime_failure(exception_reason)
        ):
            return exception_reason

    return ""


def _is_agent_runtime_failure_trial(trial_dir: Path) -> bool:
    return bool(_agent_runtime_failure_reason(trial_dir))


def _read_failed_judge_sidecar(
    path: Path,
    *,
    trial_dir: Path,
    expected: os.stat_result,
) -> tuple[dict[str, Any] | None, str]:
    """Read one sidecar, distinguishing safe absence from an unreadable candidate."""
    read_failure = "Judge sidecar could not be read as a bounded regular JSON object; trial was not scored"
    try:
        relative = path.relative_to(trial_dir)
        with SecureRoot(trial_dir) as secure_root:
            raw, _metadata = secure_root.read_bytes(
                relative,
                DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES,
                expected=expected,
            )
        data = json.loads(raw)
    except (SecurePathError, ValueError, OSError, RecursionError, UnicodeError):
        return None, read_failure
    return (data, "") if isinstance(data, dict) else (None, read_failure)


def _failed_judge_sidecar_paths(trial_dir: Path) -> tuple[list[tuple[str, Path, os.stat_result]], str]:
    """Return bounded sidecar paths plus a safe reason when the scan is incomplete.

    The candidate bound applies to sidecars that actually exist. Empty verifier
    directories do not consume it, so large valid step graphs remain compatible.
    """
    candidates: list[tuple[str, Path, os.stat_result]] = []

    def inspect_candidate(
        verifier_dir: Path,
        *,
        location: str,
    ) -> tuple[tuple[Path, os.stat_result] | None, str]:
        try:
            verifier_metadata = verifier_dir.lstat()
        except FileNotFoundError:
            return None, ""
        except OSError:
            return None, "Judge sidecar scan could not inspect verifier artifacts; trial was not scored"
        if stat_is_link_or_reparse(verifier_metadata):
            return None, f"Judge sidecar scan refused a symlinked {location} verifier; trial was not scored"
        if not stat.S_ISDIR(verifier_metadata.st_mode):
            return None, "Judge sidecar scan found a non-directory verifier artifact; trial was not scored"
        sidecar_path = verifier_dir / "skill_evaluator_reward.json"
        try:
            sidecar_metadata = sidecar_path.lstat()
        except FileNotFoundError:
            return None, ""
        except OSError:
            return None, "Judge sidecar scan could not inspect verifier artifacts; trial was not scored"
        return (sidecar_path, sidecar_metadata), ""

    root_candidate, root_failure = inspect_candidate(trial_dir / "verifier", location="root")
    if root_failure:
        return candidates, root_failure
    if root_candidate is not None:
        candidates.append(("", *root_candidate))

    steps_dir = trial_dir / "steps"
    try:
        steps_metadata = steps_dir.lstat()
    except FileNotFoundError:
        return candidates, ""
    except OSError:
        return candidates, "Judge sidecar scan could not inspect step artifacts; trial was not scored"
    if stat_is_link_or_reparse(steps_metadata):
        return candidates, "Judge sidecar scan refused a symlinked steps directory; trial was not scored"
    if not stat.S_ISDIR(steps_metadata.st_mode):
        return candidates, "Judge sidecar scan found a non-directory steps artifact; trial was not scored"
    scan_failure = ""
    try:
        with os.scandir(steps_dir) as entries:
            for index, entry in enumerate(entries):
                if index >= _MAX_FAILED_JUDGE_STEP_PATHS_SCANNED:
                    scan_failure = (
                        "Judge sidecar scan exceeded the "
                        f"{_MAX_FAILED_JUDGE_STEP_PATHS_SCANNED}-entry safety limit; trial was not scored"
                    )
                    break
                try:
                    entry_metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    scan_failure = "Judge sidecar scan could not inspect step artifacts; trial was not scored"
                    break
                if stat_is_link_or_reparse(entry_metadata):
                    scan_failure = "Judge sidecar scan refused a symlinked step entry; trial was not scored"
                    break
                if not stat.S_ISDIR(entry_metadata.st_mode):
                    continue
                verifier_dir = Path(entry.path) / "verifier"
                candidate, candidate_failure = inspect_candidate(verifier_dir, location="step")
                if candidate_failure:
                    scan_failure = candidate_failure
                    break
                if candidate is None:
                    continue
                if len(candidates) >= _MAX_FAILED_JUDGE_SIDECARS:
                    scan_failure = (
                        "Judge sidecar scan exceeded the "
                        f"{_MAX_FAILED_JUDGE_SIDECARS}-candidate safety limit; trial was not scored"
                    )
                    break
                candidates.append((entry.name, *candidate))
    except FileNotFoundError:
        pass
    except OSError:
        scan_failure = "Judge sidecar scan could not inspect step artifacts; trial was not scored"
    return sorted(candidates, key=lambda item: (item[0], item[1].as_posix())), scan_failure


def _failed_judge_diagnostic(trial_dir: Path) -> dict[str, Any] | None:
    """Project failed judge sidecars into one safe, intrinsically unscoreable record."""
    errors: dict[str, str] = {}
    entry_id = ""
    sidecar_paths, scan_failure = _failed_judge_sidecar_paths(trial_dir)
    found_failure = bool(scan_failure)
    if scan_failure:
        errors["collector"] = scan_failure
    for step_name, path, expected in sidecar_paths:
        sidecar, read_failure = _read_failed_judge_sidecar(path, trial_dir=trial_dir, expected=expected)
        if read_failure:
            found_failure = True
            errors.setdefault("collector", read_failure)
            continue
        if not sidecar or str(sidecar.get("evaluation_status") or "").casefold() not in {"error", "failed"}:
            continue
        found_failure = True
        if not entry_id:
            entry_id = _safe_diagnostic_text(sidecar.get("entry_id"), max_len=256)
        safe_errors = _safe_evaluation_errors(sidecar.get("evaluation_errors"))
        if isinstance(safe_errors, dict):
            error_items = safe_errors.items()
        elif isinstance(safe_errors, list):
            error_items = ((f"judge_{index + 1}", reason) for index, reason in enumerate(safe_errors))
        elif safe_errors:
            error_items = (("judge", safe_errors),)
        else:
            error_items = ()
        safe_step = _safe_diagnostic_text(step_name, max_len=64)
        for metric, reason in error_items:
            key = f"{safe_step}.{metric}" if safe_step else str(metric)
            errors.setdefault(key, reason)
            if len(errors) >= len(DEFAULT_METRICS):
                break
        if len(errors) >= len(DEFAULT_METRICS):
            break

    if not found_failure:
        return None
    diagnostic: dict[str, Any] = {
        "metric_set": DEFAULT_METRIC_SET,
        "evaluation_status": "failed",
    }
    if entry_id:
        diagnostic["entry_id"] = entry_id
    if errors:
        diagnostic["evaluation_errors"] = errors
    return redact_sensitive_data(diagnostic, max_str_len=REWARD_DIAGNOSTIC_STRING_MAX_CHARS)


def _inspect_trial_directory(trial_dir: Path) -> tuple[str, str]:
    """Classify one job child with one lstat, without following links."""
    try:
        metadata = trial_dir.lstat()
    except OSError:
        return "missing", ""
    if stat_is_link_or_reparse(metadata):
        return "link", "Unsafe Harbor trial directory is a symlink or reparse point; trial was not scored"
    if not stat.S_ISDIR(metadata.st_mode):
        return "other", ""
    return "directory", ""


def _unsafe_trial_directory_reason(trial_dir: Path) -> str:
    """Return a bounded reason when an expected Harbor trial root is unsafe."""
    kind, reason = _inspect_trial_directory(trial_dir)
    if kind == "directory":
        return ""
    if kind == "link":
        return reason
    return "Unsafe Harbor trial directory could not be inspected; trial was not scored"


def _trial_failure_reason(trial_dir: Path) -> str:
    """Return the failure recorded for any incomplete Harbor trial."""
    if unsafe_reason := _unsafe_trial_directory_reason(trial_dir):
        return unsafe_reason
    _, exception_reason = _trial_exception_details(trial_dir)
    if exception_reason:
        if diagnostic := _failed_judge_diagnostic(trial_dir):
            return _unscoreable_reward_reason(diagnostic)
        return exception_reason
    exception_file = trial_dir / "exception.txt"
    exception_text = _read_bounded_text(exception_file)
    if exception_text is None:
        return ""
    lines = [line.strip() for line in exception_text.splitlines()]
    reason = next((line for line in reversed(lines) if line), "")
    if not reason:
        return ""
    if diagnostic := _failed_judge_diagnostic(trial_dir):
        return _unscoreable_reward_reason(diagnostic)
    return f"HarborTrialError: {reason}"[:600]


def _extract_trial_failures(job_dir: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for ordinal, trial_dir in enumerate(sorted(job_dir.iterdir()), start=1):
        trial_label = _published_trial_label(trial_dir.name, alias_ordinal=ordinal)
        kind, unsafe_reason = _inspect_trial_directory(trial_dir)
        if kind == "link":
            failures.append({"trial": trial_label, "reason": unsafe_reason})
            continue
        if kind != "directory":
            continue
        reason = _trial_failure_reason(trial_dir)
        if reason:
            failures.append({"trial": trial_label, "reason": redact_sensitive_text(reason)})
    return failures


def _can_preserve_partial_rewards(job_dir: Path, trial_failures: list[dict[str, str]]) -> bool:
    """Return whether every aggregate job error maps to a concrete failed trial."""
    result = _read_json(job_dir / "result.json")
    stats = result.get("stats") if isinstance(result, dict) else None
    if not isinstance(stats, dict):
        return False

    current_schema = "n_errored_trials" in stats
    errors = stats.get("n_errored_trials" if current_schema else "n_errors")
    completed = stats.get("n_completed_trials" if current_schema else "n_trials")
    total = result.get("n_total_trials")
    if (
        not isinstance(errors, int)
        or isinstance(errors, bool)
        or errors <= 0
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or completed != total
    ):
        return False
    if current_schema and any(stats.get(key) for key in ("n_running_trials", "n_pending_trials", "n_cancelled_trials")):
        return False

    failed_trials = {str(failure.get("trial") or "") for failure in trial_failures}
    failed_trials.discard("")
    return len(failed_trials) >= errors


def _extract_agent_runtime_failures(job_dir: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for ordinal, trial_dir in enumerate(sorted(job_dir.iterdir()), start=1):
        kind, _reason = _inspect_trial_directory(trial_dir)
        if kind != "directory":
            continue
        reason = _agent_runtime_failure_reason(trial_dir)
        if reason:
            failures.append(
                {
                    "trial": _published_trial_label(trial_dir.name, alias_ordinal=ordinal),
                    "reason": redact_sensitive_text(reason),
                }
            )
    return failures


def _diagnostic_artifact_max_bytes() -> int:
    raw = os.environ.get("SKILLEVALUATOR_HARBOR_DIAGNOSTIC_ARTIFACT_MAX_BYTES")
    if not raw:
        return DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.debug("Ignoring invalid SKILLEVALUATOR_HARBOR_DIAGNOSTIC_ARTIFACT_MAX_BYTES=%r", raw)
        return DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES
    return min(max(0, value), DIAGNOSTIC_ARTIFACT_HARD_MAX_BYTES)


def _without_atif_token_id_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_atif_token_id_fields(item)
            for key, item in value.items()
            if str(key) not in _ATIF_TOKEN_ID_FIELDS
        }
    if isinstance(value, list):
        return [_without_atif_token_id_fields(item) for item in value]
    return value


def _redacted_trajectory_data(value: Any) -> dict[str, Any] | None:
    """Redact one ATIF document without changing schema field types."""
    try:
        validated = _validated_trajectory_dict(value)
        without_token_ids = _without_atif_token_id_fields(validated)
        return _validated_trajectory_dict(redact_sensitive_data(without_token_ids))
    except (_TrajectoryMergeError, RecursionError, MemoryError):
        return None


def _redacted_artifact_text(src: Path, text: str) -> str | None:
    if src.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            if src.name.startswith("trajectory"):
                safe_data = _redacted_trajectory_data(data)
                if safe_data is None:
                    return None
            else:
                safe_data = redact_sensitive_data(data)
            # Compact encoding avoids indentation-driven amplification for
            # deeply nested but otherwise valid diagnostic JSON.
            return json.dumps(safe_data, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        except (json.JSONDecodeError, ValueError, RecursionError, MemoryError):
            return None
    try:
        return redact_sensitive_text(text)
    except (RecursionError, MemoryError):
        return None


def _write_artifact_manifest(trial_out: Path, manifest: dict[str, Any]) -> None:
    try:
        write_output_file_atomically(
            trial_out / "artifact_manifest.json",
            json.dumps(redact_sensitive_data(manifest), indent=2).encode("utf-8"),
        )
    except (OSError, ValueError) as e:
        logger.debug("Failed to write Harbor artifact manifest %s: %s", trial_out, e)


def _write_redacted_text_copy(
    src: Path,
    dest: Path,
    *,
    source_root: Path | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Copy a Harbor text artifact while masking common credential shapes."""
    max_bytes = _diagnostic_artifact_max_bytes()
    try:
        path_metadata = src.lstat()
    except OSError as e:
        logger.debug("Failed to stat Harbor artifact %s: %s", src, e)
        return False, {"name": src.name, "reason": "stat_failed"}
    if (
        stat_is_link_or_reparse(path_metadata)
        or not stat.S_ISREG(path_metadata.st_mode)
        or getattr(path_metadata, "st_nlink", 1) != 1
    ):
        return False, {"name": src.name, "reason": "not_regular_file"}

    read_root = source_root or src.parent
    try:
        relative = src.relative_to(read_root)
        with SecureRoot(read_root) as secure_root:
            raw, metadata = secure_root.read_bytes(relative, max_bytes, expected=path_metadata)
    except SecurePathError as e:
        if e.code == "file_size_limit":
            return False, {
                "name": src.name,
                "reason": "exceeds_max_bytes",
                "size_bytes": e.metadata.get("actual_bytes", path_metadata.st_size),
                "max_bytes": max_bytes,
            }
        logger.debug("Refusing unsafe Harbor artifact %s: %s", src, e)
        return False, {"name": src.name, "reason": "not_regular_file"}
    except (OSError, OverflowError, MemoryError, ValueError) as e:
        logger.debug("Failed to read Harbor artifact %s: %s", src, e)
        return False, {"name": src.name, "reason": "read_failed"}

    size_bytes = metadata.st_size
    if len(raw) > max_bytes:
        return False, {
            "name": src.name,
            "reason": "exceeds_max_bytes",
            "size_bytes": len(raw),
            "max_bytes": max_bytes,
        }
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        logger.debug("Failed to read Harbor artifact %s: %s", src, e)
        return False, {"name": src.name, "reason": "read_failed", "size_bytes": size_bytes}
    redacted = _redacted_artifact_text(src, text)
    if redacted is None:
        return False, {"name": src.name, "reason": "invalid_json", "size_bytes": size_bytes}
    payload = redacted.encode("utf-8")
    if len(payload) > max_bytes:
        return False, {
            "name": src.name,
            "reason": "exceeds_max_bytes",
            "size_bytes": len(payload),
            "max_bytes": max_bytes,
        }
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_output_file_atomically(dest, payload)
    except (OSError, ValueError) as e:
        logger.debug("Failed to write Harbor artifact %s: %s", dest, e)
        return False, {"name": src.name, "reason": "write_failed", "size_bytes": size_bytes}
    return True, {"name": src.name, "size_bytes": size_bytes}


def _copy_trial_artifacts(
    trial_dir: Path,
    trial_out: Path,
    *,
    include_root_trajectory: bool = True,
) -> list[str]:
    copied: list[str] = []
    manifest: dict[str, Any] = {"copied": [], "skipped": []}
    for artifact_name in TRIAL_DIAGNOSTIC_ARTIFACTS:
        src = trial_dir / artifact_name
        if src.exists() or src.is_symlink():
            ok, record = _write_redacted_text_copy(src, trial_out / artifact_name, source_root=trial_dir)
            if ok:
                copied.append(artifact_name)
                manifest["copied"].append(record)
            elif record:
                manifest["skipped"].append(record)

    agent_logs = trial_dir / "agent"
    try:
        agent_logs_metadata = agent_logs.lstat()
    except FileNotFoundError:
        agent_logs_metadata = None
    except OSError:
        manifest["skipped"].append({"name": "agent", "reason": "stat_failed"})
        agent_logs_metadata = None
    if agent_logs_metadata is not None and (
        stat_is_link_or_reparse(agent_logs_metadata) or not stat.S_ISDIR(agent_logs_metadata.st_mode)
    ):
        manifest["skipped"].append({"name": "agent", "reason": "not_regular_directory"})
    elif agent_logs_metadata is not None:
        for artifact_name in AGENT_LOG_ARTIFACTS:
            if artifact_name == "trajectory.json" and not include_root_trajectory:
                continue
            src = agent_logs / artifact_name
            if src.exists() or src.is_symlink():
                ok, record = _write_redacted_text_copy(src, trial_out / artifact_name, source_root=trial_dir)
                if ok:
                    copied.append(artifact_name)
                    manifest["copied"].append(record)
                elif record:
                    manifest["skipped"].append(record)
    if manifest["skipped"]:
        _write_artifact_manifest(trial_out, manifest)
    return copied


def _record_skipped_trajectory(trial_out: Path, reason: str) -> None:
    """Persist a value-free explanation when canonical trajectory output is omitted."""
    manifest_path = trial_out / "artifact_manifest.json"
    existing = _read_json(manifest_path)
    manifest = existing if isinstance(existing, dict) else {"copied": [], "skipped": []}
    copied = manifest.get("copied")
    skipped = manifest.get("skipped")
    if not isinstance(copied, list):
        manifest["copied"] = []
    if not isinstance(skipped, list):
        skipped = []
        manifest["skipped"] = skipped
    record = {"name": "trajectory.json", "reason": reason}
    if record not in skipped:
        skipped.append(record)
    _write_artifact_manifest(trial_out, manifest)


def _trial_error_summary(trial_dir: Path) -> dict[str, Any]:
    result_file = trial_dir / "result.json"
    summary: dict[str, Any] = {}
    result = _read_json(result_file)
    if isinstance(result, dict):
        for key in ("task_id", "task_name", "trial_name", "started_at", "finished_at"):
            value = result.get(key)
            if value not in (None, ""):
                summary[key] = value

        agent_info = result.get("agent_info") if isinstance(result.get("agent_info"), dict) else {}
        config = result.get("config") if isinstance(result.get("config"), dict) else {}
        config_agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        model = agent_info.get("model_name") or config_agent.get("model_name")
        if model:
            summary["model"] = model

        exception_info = result.get("exception_info")
        if isinstance(exception_info, dict) and exception_info:
            error = {
                "type": exception_info.get("exception_type"),
                "message": _safe_text(exception_info.get("exception_message")),
                "occurred_at": exception_info.get("occurred_at"),
            }
            summary["error"] = {k: v for k, v in error.items() if v not in (None, "")}

    if "error" not in summary:
        exception_file = trial_dir / "exception.txt"
        exception_text = _read_bounded_text(exception_file)
        if exception_text is not None:
            lines = [line.strip() for line in exception_text.splitlines() if line.strip()]
            if lines:
                summary["error"] = {
                    "type": "HarborTrialError",
                    "message": _safe_text(lines[-1]),
                }

    return summary


def _write_bounded_failure_artifact(path: Path, failure: dict[str, Any]) -> None:
    """Write a redacted failure record inside the generated JSON envelope."""
    try:
        safe_failure = redact_sensitive_data(failure, max_str_len=REWARD_METADATA_TEXT_MAX_CHARS)
        _validate_generated_json_value(
            safe_failure,
            max_depth=REWARD_JSON_MAX_DEPTH,
            max_nodes=REWARD_JSON_MAX_NODES,
            max_bytes=GENERATED_JSON_MAX_BYTES,
        )
    except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
        raw_error = failure.get("error")
        error = raw_error if isinstance(raw_error, dict) else {}
        safe_failure = {
            "status": "unscored",
            "trial": _published_trial_label(failure.get("trial", "unknown trial")),
            "agent": _safe_diagnostic_text(failure.get("agent", "unknown"), max_len=256),
            "variant": _safe_diagnostic_text(failure.get("variant", "unknown"), max_len=64),
            "artifacts": [
                _safe_diagnostic_text(item, max_len=128)
                for item in (failure.get("artifacts") if isinstance(failure.get("artifacts"), list) else [])[:64]
            ],
            "error": {
                "type": _safe_diagnostic_text(error.get("type", "HarborTrialError"), max_len=128),
                "message": _safe_diagnostic_text(
                    error.get("message", "Failure metadata exceeded safe artifact limits"),
                    max_len=512,
                ),
            },
            "diagnostic_truncated": True,
            "diagnostic_truncated_reason": "failure metadata exceeded safe artifact limits",
        }
        if str(failure.get("evaluation_status") or "").casefold() in {"error", "failed"}:
            safe_failure["evaluation_status"] = "failed"
        if failure.get("evaluation_errors") not in (None, ""):
            safe_failure["evaluation_errors"] = _safe_evaluation_errors(failure["evaluation_errors"])
    encoded = json.dumps(
        safe_failure,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    write_output_file_atomically(path, encoded)


def _looks_like_trial_dir(path: Path) -> bool:
    return any((path / name).exists() for name in TRIAL_DIAGNOSTIC_ARTIFACTS) or (path / "agent").exists()


def _save_unscored_trials(
    rewards: list[dict[str, Any]],
    trials_dir: Path,
    job_dir: Path | None,
    *,
    agent: str,
    variant: str,
    agent_model: str | None = None,
    agent_model_source: str | None = None,
    persisted_names: dict[str, str] | None = None,
) -> None:
    if job_dir is None or not job_dir.exists():
        return
    agent = _bounded_reward_metadata_text(agent) or "unknown"
    agent_model = _bounded_reward_metadata_text(agent_model)
    agent_model_source = _bounded_reward_metadata_text(agent_model_source)

    scored_trials = {str(reward.get("_trial_root_name") or reward.get("_trial_name") or "") for reward in rewards}
    if persisted_names is None:
        _scored_names, persisted_names = _persisted_trial_layout(rewards, job_dir)
    for trial_src in sorted(job_dir.iterdir()):
        kind, unsafe_reason = _inspect_trial_directory(trial_src)
        if kind == "link":
            trial_out = trials_dir / persisted_names.get(trial_src.name, "unknown")
            trial_out.mkdir(parents=True, exist_ok=True)
            _write_bounded_failure_artifact(
                trial_out / "failure.json",
                {
                    "status": "unscored",
                    "trial": trial_out.name,
                    "agent": agent,
                    "variant": variant,
                    "artifacts": [],
                    "error": {"type": "UnsafeHarborTrial", "message": unsafe_reason},
                },
            )
            continue
        if kind != "directory":
            continue
        if trial_src.name in scored_trials or not _looks_like_trial_dir(trial_src):
            continue

        trial_out = trials_dir / persisted_names.get(trial_src.name, "unknown")
        trial_out.mkdir(parents=True, exist_ok=True)
        copied = _copy_trial_artifacts(trial_src, trial_out, include_root_trajectory=False)
        materialized_trajectory, trajectory_reason = _materialized_trial_trajectory(trial_src, None)
        safe_trajectory = (
            _redacted_trajectory_data(materialized_trajectory) if materialized_trajectory is not None else None
        )
        if safe_trajectory is not None:
            (trial_out / "trajectory.json").write_text(
                json.dumps(
                    safe_trajectory,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            copied.append("trajectory.json")
        else:
            if materialized_trajectory is not None:
                trajectory_reason = "trajectory_redaction_or_validation_failed"
            _record_skipped_trajectory(trial_out, trajectory_reason or "trajectory_unavailable")
            copied.append("artifact_manifest.json")
        judge_diagnostic = _failed_judge_diagnostic(trial_src)
        if judge_diagnostic is not None:
            judge_diagnostic.update({"agent": agent, "variant": variant})
            if agent_model:
                judge_diagnostic["model"] = agent_model
            if agent_model_source:
                judge_diagnostic["model_source"] = agent_model_source
            (trial_out / "reward.json").write_text(
                json.dumps(judge_diagnostic, indent=2),
                encoding="utf-8",
            )
            copied.append("reward.json")
        failure = {
            "status": "unscored",
            "trial": trial_out.name,
            "agent": agent,
            "variant": variant,
            "artifacts": copied,
        }
        if agent_model:
            failure["model"] = agent_model
        if agent_model_source:
            failure["model_source"] = agent_model_source
        if judge_diagnostic is not None:
            failure["evaluation_status"] = "failed"
            if "evaluation_errors" in judge_diagnostic:
                failure["evaluation_errors"] = judge_diagnostic["evaluation_errors"]
        error_summary = _trial_error_summary(trial_src)
        if agent_model:
            error_summary.pop("model", None)
        failure.update(error_summary)
        failure_file = trial_out / "failure.json"
        try:
            _write_bounded_failure_artifact(failure_file, failure)
        except (OSError, TypeError, ValueError) as e:
            logger.debug("Failed to write Harbor failure artifact %s: %s", failure_file, e)


def _reward_trial_context(reward_file: Path) -> tuple[Path, str, str | None]:
    """Return ``(trial_root, trial_name, step_name)`` for a Harbor reward file.

    Harbor single-step tasks write ``<trial>/verifier/reward.json``. Native
    multi-step tasks may write ``<trial>/steps/<step>/verifier/reward.json``.
    Keep the real trial root for artifacts while making the persisted result
    name unique per step.
    """
    verifier_dir = reward_file.parent
    reward_parent = verifier_dir.parent
    if reward_parent.parent.name == "steps":
        step_name = reward_parent.name
        trial_root = reward_parent.parent.parent
        return trial_root, f"{trial_root.name}__{step_name}", step_name
    return reward_parent, reward_parent.name, None


def _reward_trajectory_path(trial_root: Path, step_name: str | None) -> Path:
    if step_name:
        step_traj = trial_root / "steps" / step_name / "agent" / "trajectory.json"
        if step_traj.exists():
            return step_traj
    root_traj = trial_root / "agent" / "trajectory.json"
    if root_traj.exists():
        return root_traj
    step_trajs = _ordered_step_trajectory_paths(trial_root)
    if step_trajs:
        return step_trajs[-1]
    return root_traj


def _ordered_step_trajectory_paths(trial_root: Path) -> list[Path]:
    steps_dir = trial_root / "steps"
    try:
        steps_metadata = steps_dir.lstat()
    except OSError:
        return []
    if stat_is_link_or_reparse(steps_metadata) or not stat.S_ISDIR(steps_metadata.st_mode):
        return []

    safe_paths: dict[str, Path] = {}
    scanned_entries = 0
    try:
        with os.scandir(steps_dir) as entries:
            for entry in entries:
                scanned_entries += 1
                if scanned_entries > _MAX_TRAJECTORY_STEP_DIRECTORIES:
                    return []
                try:
                    entry_metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat_is_link_or_reparse(entry_metadata) or not stat.S_ISDIR(entry_metadata.st_mode):
                    continue
                agent_dir = Path(entry.path) / "agent"
                try:
                    agent_metadata = agent_dir.lstat()
                except OSError:
                    continue
                if stat_is_link_or_reparse(agent_metadata) or not stat.S_ISDIR(agent_metadata.st_mode):
                    continue
                trajectory = agent_dir / "trajectory.json"
                try:
                    trajectory_metadata = trajectory.lstat()
                except OSError:
                    continue
                if stat_is_link_or_reparse(trajectory_metadata) or not stat.S_ISREG(trajectory_metadata.st_mode):
                    continue
                safe_paths[entry.name] = trajectory
    except OSError:
        return []

    ordered_names: list[str] = []
    seen_names: set[str] = set()
    result = _read_json(trial_root / "result.json")
    if isinstance(result, dict):
        step_results = result.get("step_results")
        if isinstance(step_results, list):
            for step in step_results:
                if isinstance(step, dict):
                    step_name = step.get("step_name")
                    if isinstance(step_name, str) and step_name and step_name not in seen_names:
                        ordered_names.append(step_name)
                        seen_names.add(step_name)

    ordered_paths: list[Path] = []
    seen: set[Path] = set()
    for step_name in ordered_names:
        path = safe_paths.get(step_name)
        if path is not None:
            ordered_paths.append(path)
            seen.add(path)

    for path in sorted(safe_paths.values()):
        if path not in seen:
            ordered_paths.append(path)
    return ordered_paths


def _trajectory_dict_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _trajectory_step_merge_identity(
    step: dict[str, Any],
    *,
    copied_context: bool = False,
) -> dict[str, Any]:
    """Return semantic step content without collector-owned identity fields."""
    identity = {
        key: value
        for key, value in step.items()
        if key not in {"is_copied_context", "step_id"} and not (copied_context and key == "metrics")
    }
    extra = identity.get("extra")
    if isinstance(extra, dict):
        identity["extra"] = {
            key: value
            for key, value in extra.items()
            if key not in {"harbor_step_name", "harbor_original_step_id"} and not (copied_context and key == "note")
        }
        if not identity["extra"]:
            identity.pop("extra")
    return identity


def _is_cumulative_trajectory_continuation(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """Return true only when ATIF copied-context markers prove continuation."""
    previous_steps = _trajectory_dict_steps(previous)
    current_steps = _trajectory_dict_steps(current)
    if not previous_steps or not current_steps:
        return False
    copied_prefix_length = _copied_context_prefix_length(current_steps)
    if copied_prefix_length:
        if copied_prefix_length > len(previous_steps):
            raise _TrajectoryMergeError("copied-context prefix exceeds previous trajectory")
        if copied_prefix_length == len(current_steps):
            raise _TrajectoryMergeError("copied-context continuation has no new steps")
        for previous_step, current_step in zip(
            previous_steps[-copied_prefix_length:],
            current_steps[:copied_prefix_length],
            strict=True,
        ):
            if _trajectory_step_merge_identity(previous_step, copied_context=True) != (
                _trajectory_step_merge_identity(current_step, copied_context=True)
            ):
                raise _TrajectoryMergeError("copied-context prefix does not match previous trajectory suffix")
        return True

    previous_session = previous.get("session_id")
    current_session = current.get("session_id")
    if (
        not isinstance(previous_session, str)
        or not previous_session
        or current_session != previous_session
        or previous_session == "copilot-cli"
        or len(current_steps) <= len(previous_steps)
    ):
        return False
    return all(
        _trajectory_step_merge_identity(previous_step) == _trajectory_step_merge_identity(current_step)
        for previous_step, current_step in zip(
            previous_steps,
            current_steps[: len(previous_steps)],
            strict=True,
        )
    )


def _copied_context_prefix_length(steps: list[dict[str, Any]]) -> int:
    """Return the leading copied-context count, rejecting non-prefix markers."""
    prefix_length = 0
    saw_new_step = False
    for step in steps:
        if step.get("is_copied_context") is True:
            if saw_new_step:
                raise _TrajectoryMergeError("copied-context steps must form a prefix")
            prefix_length += 1
        else:
            saw_new_step = True
    return prefix_length


class _TrajectoryMergeError(ValueError):
    """A multi-file ATIF trajectory cannot be materialized without data loss."""


@dataclass
class _TrajectoryReferenceState:
    raw_cache: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    materialized_cache: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    resolved_ids: dict[str, str] = dataclass_field(default_factory=dict)
    reference_ordinals: dict[str, int] = dataclass_field(default_factory=dict)
    generated_continuation_fingerprints: dict[str, str] = dataclass_field(default_factory=dict)
    active: set[str] = dataclass_field(default_factory=set)
    file_count: int = 0
    total_bytes: int = 0


def _normalized_trajectory_reference(reference: Any) -> tuple[Path, str]:
    if not isinstance(reference, str) or not reference.strip() or len(reference) > 512 or "\x00" in reference:
        raise _TrajectoryMergeError("invalid trajectory reference")
    # Harbor resolves this value as an exact child path. Preserve significant
    # leading/trailing whitespace for the source lookup; normalization is only
    # for traversal checks and the cache key.
    raw = reference
    try:
        raw.encode("utf-8")
    except UnicodeError as exc:
        raise _TrajectoryMergeError("invalid trajectory reference encoding") from exc
    platform_path = PureWindowsPath(raw) if os.name == "nt" else PurePosixPath(raw)
    if platform_path.drive or platform_path.is_absolute():
        raise _TrajectoryMergeError("absolute trajectory reference")
    if "://" in raw or any(part in {"", ".", ".."} for part in platform_path.parts):
        raise _TrajectoryMergeError("unsafe trajectory reference")
    # Match Harbor's native ``agent_dir / reference`` lookup exactly.  In
    # particular, POSIX treats a backslash as a literal filename character;
    # rewriting it to ``/`` can select a different trajectory than Harbor did.
    relative = Path(raw)
    key = platform_path.as_posix().casefold() if os.name == "nt" else platform_path.as_posix()
    return relative, key


def _validate_generated_json_structure(
    value: Any,
    *,
    max_depth: int,
    max_nodes: int,
) -> None:
    """Reject trees that the bounded report loader cannot traverse."""
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("JSON node count exceeds limit")
        if not isinstance(current, dict | list):
            continue
        if depth > max_depth:
            raise ValueError("JSON depth exceeds limit")
        if nodes + len(current) > max_nodes:
            raise ValueError("JSON node count exceeds limit")
        children = current.values() if isinstance(current, dict) else current
        stack.extend((child, depth + 1) for child in children)


def _validate_generated_json_value(
    value: Any,
    *,
    max_depth: int,
    max_nodes: int,
    max_bytes: int,
) -> None:
    """Validate structural, numeric, and encoded-size browser safety."""
    _validate_generated_json_structure(value, max_depth=max_depth, max_nodes=max_nodes)
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, bool) or current is None or isinstance(current, str):
            continue
        elif isinstance(current, int):
            if abs(current) > _MAX_JSON_SAFE_INTEGER:
                raise ValueError("browser-unsafe JSON integer")
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError("non-finite JSON number")
        else:
            raise TypeError("value is not JSON serializable")
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("encoded JSON exceeds limit")


def _validated_trajectory_dict(data: Any) -> dict[str, Any]:
    try:
        # Bound traversal before deepcopy so hostile ATIF cannot amplify work
        # before validation begins.
        _validate_generated_json_structure(
            data,
            max_depth=ATIF_JSON_MAX_DEPTH,
            max_nodes=ATIF_JSON_MAX_NODES,
        )
        candidate = copy.deepcopy(data)
        if isinstance(candidate, dict):
            final_metrics = candidate.get("final_metrics")
            if isinstance(final_metrics, dict):
                for key in (
                    "total_prompt_tokens",
                    "total_completion_tokens",
                    "total_cached_tokens",
                    "total_steps",
                ):
                    value = final_metrics.get(key)
                    if key in final_metrics and (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        or value > _MAX_JSON_SAFE_INTEGER
                    ):
                        final_metrics.pop(key, None)
                cost = final_metrics.get("total_cost_usd")
                if "total_cost_usd" in final_metrics:
                    try:
                        cost_is_finite = (
                            isinstance(cost, int | float) and not isinstance(cost, bool) and math.isfinite(cost)
                        )
                    except OverflowError:
                        cost_is_finite = False
                    if not cost_is_finite:
                        final_metrics.pop("total_cost_usd", None)
                metric_extra = final_metrics.get("extra")
                if isinstance(metric_extra, dict):
                    for key, value in list(metric_extra.items()):
                        if _is_aggregate_extra_token_key(str(key)) and (
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value < 0
                            or value > _MAX_JSON_SAFE_INTEGER
                        ):
                            metric_extra.pop(key, None)
            steps = candidate.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    llm_call_count = step.get("llm_call_count")
                    if "llm_call_count" in step and (
                        not isinstance(llm_call_count, int)
                        or isinstance(llm_call_count, bool)
                        or llm_call_count < 0
                        or llm_call_count > _MAX_JSON_SAFE_INTEGER
                    ):
                        step.pop("llm_call_count", None)
                    step_metrics = step.get("metrics")
                    if not isinstance(step_metrics, dict):
                        continue
                    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                        value = step_metrics.get(key)
                        if key in step_metrics and (
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value < 0
                            or value > _MAX_JSON_SAFE_INTEGER
                        ):
                            step_metrics.pop(key, None)
                    cost = step_metrics.get("cost_usd")
                    if "cost_usd" in step_metrics:
                        try:
                            cost_is_finite = (
                                isinstance(cost, int | float) and not isinstance(cost, bool) and math.isfinite(cost)
                            )
                        except OverflowError:
                            cost_is_finite = False
                        if not cost_is_finite:
                            step_metrics.pop("cost_usd", None)
                    step_extra = step_metrics.get("extra")
                    if isinstance(step_extra, dict):
                        for key, value in list(step_extra.items()):
                            if _is_aggregate_extra_token_key(str(key)) and (
                                not isinstance(value, int)
                                or isinstance(value, bool)
                                or value < 0
                                or value > _MAX_JSON_SAFE_INTEGER
                            ):
                                step_extra.pop(key, None)
        _validate_generated_json_value(
            candidate,
            max_depth=ATIF_JSON_MAX_DEPTH,
            max_nodes=ATIF_JSON_MAX_NODES,
            max_bytes=GENERATED_JSON_MAX_BYTES,
        )
        validated = Trajectory.model_validate(candidate).to_json_dict()
        _validate_generated_json_value(
            validated,
            max_depth=ATIF_JSON_MAX_DEPTH,
            max_nodes=ATIF_JSON_MAX_NODES,
            max_bytes=GENERATED_JSON_MAX_BYTES,
        )
        return validated
    except (TypeError, ValueError, RecursionError, UnicodeError, MemoryError) as exc:
        raise _TrajectoryMergeError("invalid ATIF trajectory") from exc


def _trajectory_reference_cache_key(agent_dir: Path, reference_key: str) -> str:
    return f"{agent_dir.absolute()}\0{reference_key}"


def _read_referenced_trajectory(
    agent_dir: Path,
    reference: Any,
    state: _TrajectoryReferenceState,
    *,
    count_against_reference_budget: bool,
) -> tuple[dict[str, Any], str]:
    relative, key = _normalized_trajectory_reference(reference)
    cache_key = _trajectory_reference_cache_key(agent_dir, key)
    if cached := state.raw_cache.get(cache_key):
        return copy.deepcopy(cached), key
    if count_against_reference_budget and state.file_count >= _MAX_TRAJECTORY_REFERENCE_FILES:
        raise _TrajectoryMergeError("trajectory reference count exceeded")
    try:
        with SecureRoot(agent_dir) as secure_root:
            raw, _metadata = secure_root.read_bytes(relative, DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES)
    except (OSError, SecurePathError) as exc:
        raise _TrajectoryMergeError("unreadable trajectory reference") from exc
    if count_against_reference_budget:
        state.file_count += 1
    state.total_bytes += len(raw)
    if state.total_bytes > _MAX_TRAJECTORY_REFERENCE_TOTAL_BYTES:
        raise _TrajectoryMergeError("trajectory reference bytes exceeded")
    try:
        data = json.loads(raw)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise _TrajectoryMergeError("invalid trajectory JSON") from exc
    validated = _validated_trajectory_dict(data)
    state.raw_cache[cache_key] = validated
    return copy.deepcopy(validated), key


def _agent_identity(agent: Any) -> dict[str, Any] | None:
    if not isinstance(agent, dict):
        return None
    return {key: agent.get(key) for key in ("name", "version", "model_name", "tool_definitions")}


def _combined_notes(*values: Any) -> str | None:
    notes: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in notes:
            notes.append(value)
    return "\n\n".join(notes) or None


def _canonical_trajectory_sha256(trajectory: dict[str, Any]) -> str:
    redacted = _redacted_trajectory_data(trajectory)
    if redacted is None:
        raise _TrajectoryMergeError("trajectory identity cannot be safely canonicalized")
    canonical = json.dumps(
        redacted,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _raw_trajectory_sha256(trajectory: dict[str, Any]) -> str:
    try:
        canonical = json.dumps(
            trajectory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise _TrajectoryMergeError("trajectory fingerprint cannot be safely canonicalized") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_trusted_materialized_cumulative_continuation(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    state: _TrajectoryReferenceState,
) -> bool:
    """Recognize native resume across collector-built explicit continuation chains."""
    if _is_cumulative_trajectory_continuation(previous, current):
        return True
    for trajectory in (previous, current):
        trajectory_id = trajectory.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            return False
        if state.generated_continuation_fingerprints.get(trajectory_id) != _raw_trajectory_sha256(trajectory):
            return False
    previous_steps = _trajectory_dict_steps(previous)
    current_steps = _trajectory_dict_steps(current)
    if not previous_steps or len(current_steps) <= len(previous_steps):
        return False
    return all(
        _trajectory_step_merge_identity(previous_step) == _trajectory_step_merge_identity(current_step)
        for previous_step, current_step in zip(
            previous_steps,
            current_steps[: len(previous_steps)],
            strict=True,
        )
    )


def _synthetic_trajectory_content_sha256(trajectory: dict[str, Any]) -> str:
    """Hash redacted semantic content without source identifiers or references."""
    redacted = _redacted_trajectory_data(trajectory)
    if redacted is None:
        raise _TrajectoryMergeError("trajectory identity cannot be safely canonicalized")
    redacted.pop("continued_trajectory_ref", None)
    redacted.pop("session_id", None)
    redacted.pop("trajectory_id", None)
    root_extra = redacted.get("extra")
    if isinstance(root_extra, dict):
        for key in ("harbor_continuation", "harbor_multi_step", "harbor_parent_scope"):
            root_extra.pop(key, None)
    canonical = json.dumps(
        redacted,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _merged_embedded_subagents(*collections: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    for collection in collections:
        if collection is None:
            continue
        if not isinstance(collection, list):
            raise _TrajectoryMergeError("invalid embedded subagent collection")
        for item in collection:
            validated = _validated_trajectory_dict(item)
            trajectory_id = validated.get("trajectory_id")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise _TrajectoryMergeError("embedded subagent lacks trajectory_id")
            canonical = json.dumps(validated, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            previous = identities.get(trajectory_id)
            if previous is not None and previous != canonical:
                raise _TrajectoryMergeError("conflicting embedded subagent trajectory_id")
            if previous is None:
                identities[trajectory_id] = canonical
                merged.append(validated)
    return merged


def _remap_parent_subagent_scope(
    trajectory: dict[str, Any],
    *,
    namespace: str,
) -> dict[str, Any]:
    """Give one parent's embedded IDs a deterministic combined-parent scope."""
    scoped = copy.deepcopy(trajectory)
    embedded = _merged_embedded_subagents(scoped.get("subagent_trajectories"))
    if not embedded:
        return scoped

    aliases: dict[str, str] = {}
    remapped: list[dict[str, Any]] = []
    for child_index, child in enumerate(embedded):
        original_id = str(child["trajectory_id"])
        identity = {
            "namespace": namespace,
            "child_index": child_index,
            "content_sha256": _synthetic_trajectory_content_sha256(child),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:20]
        scoped_id = f"skillevaluator-scoped-subagent-{digest}"
        aliases[original_id] = scoped_id
        child_extra = child.get("extra")
        if not isinstance(child_extra, dict):
            child_extra = {}
        prior_scope = copy.deepcopy(child_extra.get("harbor_parent_scope"))
        scope_provenance: dict[str, Any] = {
            "original_trajectory_id": redact_sensitive_text(original_id),
            "parent_scope": namespace,
        }
        if prior_scope is not None:
            scope_provenance["prior_scope"] = prior_scope
        child_extra["harbor_parent_scope"] = scope_provenance
        child["extra"] = child_extra
        child["trajectory_id"] = scoped_id
        remapped.append(_validated_trajectory_dict(child))

    for step in _trajectory_dict_steps(scoped):
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        results = observation.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            refs = result.get("subagent_trajectory_ref")
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                trajectory_id = ref.get("trajectory_id")
                if isinstance(trajectory_id, str) and trajectory_id in aliases:
                    ref["trajectory_id"] = aliases[trajectory_id]
    scoped["subagent_trajectories"] = remapped
    return _validated_trajectory_dict(scoped)


def _continuation_source_provenance(
    trajectory: dict[str, Any],
    *,
    trusted_continuation_fingerprints: dict[str, str],
) -> tuple[
    int,
    list[str],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    extra = trajectory.get("extra")
    continuation = extra.get("harbor_continuation") if isinstance(extra, dict) else None
    trajectory_id = trajectory.get("trajectory_id")
    trusted_fingerprint = (
        trusted_continuation_fingerprints.get(trajectory_id) if isinstance(trajectory_id, str) else None
    )
    if (
        trusted_fingerprint is not None
        and trusted_fingerprint == _raw_trajectory_sha256(trajectory)
        and isinstance(continuation, dict)
    ):
        count = continuation.get("segment_count")
        sessions = continuation.get("source_session_ids")
        trajectory_ids = continuation.get("source_trajectory_ids")
        root_extras = continuation.get("source_root_extra")
        agent_extras = continuation.get("source_agent_extra")
        final_metrics_extras = continuation.get("source_final_metrics_extra")
        schema_versions = continuation.get("source_schema_versions")
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 1
            and isinstance(sessions, list)
            and all(isinstance(value, str) and value for value in sessions)
            and isinstance(trajectory_ids, list)
            and all(isinstance(value, str) and value for value in trajectory_ids)
            and isinstance(root_extras, list)
            and all(isinstance(value, dict) for value in root_extras)
            and isinstance(agent_extras, list)
            and all(isinstance(value, dict) for value in agent_extras)
            and isinstance(final_metrics_extras, list)
            and all(isinstance(value, dict) for value in final_metrics_extras)
            and isinstance(schema_versions, list)
            and all(isinstance(value, str) and value for value in schema_versions)
        ):
            return (
                count,
                list(sessions),
                list(trajectory_ids),
                copy.deepcopy(root_extras),
                copy.deepcopy(agent_extras),
                copy.deepcopy(final_metrics_extras),
                list(schema_versions),
            )
    session_id = trajectory.get("session_id")
    agent = trajectory.get("agent")
    agent_extra = agent.get("extra") if isinstance(agent, dict) else None
    final_metrics = trajectory.get("final_metrics")
    final_metrics_extra = final_metrics.get("extra") if isinstance(final_metrics, dict) else None
    schema_version = trajectory.get("schema_version")
    return (
        1,
        [session_id] if isinstance(session_id, str) and session_id else [],
        [trajectory_id] if isinstance(trajectory_id, str) and trajectory_id else [],
        [copy.deepcopy(extra)] if isinstance(extra, dict) else [],
        [copy.deepcopy(agent_extra)] if isinstance(agent_extra, dict) else [],
        [copy.deepcopy(final_metrics_extra)] if isinstance(final_metrics_extra, dict) else [],
        [schema_version] if isinstance(schema_version, str) and schema_version else [],
    )


def _combine_continuation_trajectories(
    base: dict[str, Any],
    continuation: dict[str, Any],
    *,
    state: _TrajectoryReferenceState,
) -> dict[str, Any]:
    if _agent_identity(base.get("agent")) != _agent_identity(continuation.get("agent")):
        raise _TrajectoryMergeError("continuation agent mismatch")
    base_steps = _trajectory_dict_steps(base)
    continuation_steps = _trajectory_dict_steps(continuation)
    copied_prefix_length = _copied_context_prefix_length(continuation_steps)
    if copied_prefix_length:
        appended_steps = continuation_steps[copied_prefix_length:]
    else:
        appended_steps = continuation_steps

    base_namespace = "continuation-base"
    continuation_namespace = "continuation-next"
    scoped_base = _remap_parent_subagent_scope(base, namespace=base_namespace)
    scoped_continuation = _remap_parent_subagent_scope(
        continuation,
        namespace=continuation_namespace,
    )
    base_steps = _trajectory_dict_steps(scoped_base)
    continuation_steps = _trajectory_dict_steps(scoped_continuation)
    if copied_prefix_length:
        appended_steps = continuation_steps[copied_prefix_length:]
    else:
        appended_steps = continuation_steps

    combined = copy.deepcopy(scoped_base)
    combined["schema_version"] = "ATIF-v1.7"
    combined["agent"] = copy.deepcopy(scoped_continuation["agent"])
    combined["steps"] = [copy.deepcopy(step) for step in (*base_steps, *appended_steps)]
    for index, step in enumerate(combined["steps"], start=1):
        step["step_id"] = index
    sessions = [value for value in (base.get("session_id"), continuation.get("session_id")) if value]
    if sessions and len(set(sessions)) == 1:
        combined["session_id"] = sessions[0]
    else:
        combined.pop("session_id", None)
    if "final_metrics" in continuation:
        combined["final_metrics"] = copy.deepcopy(continuation["final_metrics"])
        if isinstance(combined["final_metrics"], dict):
            combined["final_metrics"]["total_steps"] = len(combined["steps"])
    else:
        combined.pop("final_metrics", None)
    notes = _combined_notes(base.get("notes"), continuation.get("notes"))
    if notes:
        combined["notes"] = notes
    else:
        combined.pop("notes", None)
    subagents = _merged_embedded_subagents(
        scoped_base.get("subagent_trajectories"),
        scoped_continuation.get("subagent_trajectories"),
    )
    if subagents:
        combined["subagent_trajectories"] = subagents
    else:
        combined.pop("subagent_trajectories", None)
    combined.pop("continued_trajectory_ref", None)
    (
        base_count,
        base_sessions,
        base_trajectory_ids,
        base_root_extras,
        base_agent_extras,
        base_final_metrics_extras,
        base_schema_versions,
    ) = _continuation_source_provenance(
        base,
        trusted_continuation_fingerprints=state.generated_continuation_fingerprints,
    )
    (
        continuation_count,
        continuation_sessions,
        continuation_trajectory_ids,
        continuation_root_extras,
        continuation_agent_extras,
        continuation_final_metrics_extras,
        continuation_schema_versions,
    ) = _continuation_source_provenance(
        continuation,
        trusted_continuation_fingerprints=state.generated_continuation_fingerprints,
    )
    continuation_identity = {
        "base_sha256": _canonical_trajectory_sha256(base),
        "continuation_sha256": _canonical_trajectory_sha256(continuation),
    }
    continuation_digest = hashlib.sha256(
        json.dumps(continuation_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    combined["trajectory_id"] = f"skillevaluator-continuation-{continuation_digest}"
    combined["extra"] = {
        "harbor_continuation": {
            "segment_count": base_count + continuation_count,
            "source_session_ids": [*base_sessions, *continuation_sessions],
            "source_trajectory_ids": [*base_trajectory_ids, *continuation_trajectory_ids],
            "source_root_extra": [*base_root_extras, *continuation_root_extras],
            "source_agent_extra": [*base_agent_extras, *continuation_agent_extras],
            "source_final_metrics_extra": [
                *base_final_metrics_extras,
                *continuation_final_metrics_extras,
            ],
            "source_schema_versions": [*base_schema_versions, *continuation_schema_versions],
        }
    }
    return _validated_trajectory_dict(combined)


def _mint_embedded_trajectory_id(
    reference_key: str,
    trajectory: dict[str, Any],
    *,
    reference_ordinal: int,
) -> str:
    identity = {
        "reference": redact_sensitive_text(reference_key),
        "reference_ordinal": reference_ordinal,
        "content_sha256": _canonical_trajectory_sha256(trajectory),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"skillevaluator-subagent-{digest}"


def _materialized_trajectory_source_ids(
    trajectory: dict[str, Any],
    *,
    state: _TrajectoryReferenceState,
) -> set[str]:
    values = {str(trajectory_id)} if (trajectory_id := trajectory.get("trajectory_id")) else set()
    trusted_fingerprint = state.generated_continuation_fingerprints.get(str(trajectory_id))
    if trusted_fingerprint != _raw_trajectory_sha256(trajectory):
        return values
    extra = trajectory.get("extra")
    continuation = extra.get("harbor_continuation") if isinstance(extra, dict) else None
    source_ids = continuation.get("source_trajectory_ids") if isinstance(continuation, dict) else None
    if isinstance(source_ids, list):
        values.update(str(value) for value in source_ids if isinstance(value, str) and value)
    return values


def _materialize_embedded_trajectory(
    trajectory: dict[str, Any],
    *,
    agent_dir: Path,
    state: _TrajectoryReferenceState,
    depth: int,
) -> dict[str, Any]:
    if depth > _MAX_TRAJECTORY_REFERENCE_DEPTH:
        raise _TrajectoryMergeError("trajectory reference depth exceeded")
    materialized = copy.deepcopy(trajectory)
    continued_ref = materialized.pop("continued_trajectory_ref", None)
    combined_continuation = False
    if continued_ref:
        continuation, _continuation_key = _materialize_trajectory_file(
            agent_dir,
            continued_ref,
            state=state,
            depth=depth + 1,
        )
        materialized = _combine_continuation_trajectories(materialized, continuation, state=state)
        combined_continuation = True
    materialized = _resolve_subagent_trajectory_refs(
        materialized,
        agent_dir=agent_dir,
        state=state,
        depth=depth + 1,
    )
    if combined_continuation:
        state.generated_continuation_fingerprints[materialized["trajectory_id"]] = _raw_trajectory_sha256(materialized)
    return materialized


def _resolve_subagent_trajectory_refs(
    trajectory: dict[str, Any],
    *,
    agent_dir: Path,
    state: _TrajectoryReferenceState,
    depth: int,
) -> dict[str, Any]:
    if depth > _MAX_TRAJECTORY_REFERENCE_DEPTH:
        raise _TrajectoryMergeError("trajectory reference depth exceeded")
    source_embedded = _merged_embedded_subagents(trajectory.get("subagent_trajectories"))
    embedded: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    materialized_sources: list[tuple[int, str, dict[str, Any]]] = []
    for source_index, source in enumerate(source_embedded):
        source_id = str(source["trajectory_id"])
        materialized = _materialize_embedded_trajectory(
            source,
            agent_dir=agent_dir,
            state=state,
            depth=depth,
        )
        materialized_id = materialized.get("trajectory_id")
        if not isinstance(materialized_id, str) or not materialized_id:
            raise _TrajectoryMergeError("materialized embedded subagent lacks trajectory_id")
        if state.generated_continuation_fingerprints.get(materialized_id) == _raw_trajectory_sha256(materialized):
            identity = {
                "embedded_ordinal": source_index,
                "content_sha256": _canonical_trajectory_sha256(materialized),
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest()[:20]
            materialized_id = f"skillevaluator-embedded-continuation-{digest}"
            materialized["trajectory_id"] = materialized_id
            materialized = _validated_trajectory_dict(materialized)
            state.generated_continuation_fingerprints[materialized_id] = _raw_trajectory_sha256(materialized)
        materialized_sources.append((source_index, source_id, materialized))

    redacted_id_counts: dict[str, int] = {}
    for _source_index, _source_id, materialized in materialized_sources:
        safe_id = redact_sensitive_text(str(materialized["trajectory_id"]))
        redacted_id_counts[safe_id] = redacted_id_counts.get(safe_id, 0) + 1

    for source_index, source_id, materialized in materialized_sources:
        materialized_id = str(materialized["trajectory_id"])
        safe_id = redact_sensitive_text(materialized_id)
        if redacted_id_counts[safe_id] > 1:
            was_generated_continuation = state.generated_continuation_fingerprints.get(
                materialized_id
            ) == _raw_trajectory_sha256(materialized)
            identity = {
                "embedded_ordinal": source_index,
                "content_sha256": _synthetic_trajectory_content_sha256(materialized),
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest()[:20]
            materialized_id = f"skillevaluator-redacted-subagent-{digest}"
            materialized["trajectory_id"] = materialized_id
            materialized = _validated_trajectory_dict(materialized)
            if was_generated_continuation:
                state.generated_continuation_fingerprints[materialized_id] = _raw_trajectory_sha256(materialized)
        aliases[source_id] = materialized_id
        existing = by_id.get(materialized_id)
        if existing is not None and existing != materialized:
            raise _TrajectoryMergeError("conflicting embedded subagent trajectory_id")
        if existing is None:
            by_id[materialized_id] = materialized
            embedded.append(materialized)

    refs_to_resolve: list[dict[str, Any]] = []
    for step in _trajectory_dict_steps(trajectory):
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        results = observation.get("results")
        if not isinstance(results, list):
            raise _TrajectoryMergeError("invalid trajectory observation results")
        for result in results:
            if not isinstance(result, dict):
                raise _TrajectoryMergeError("invalid trajectory observation result")
            refs = result.get("subagent_trajectory_ref")
            if refs is None:
                continue
            if not isinstance(refs, list):
                raise _TrajectoryMergeError("invalid subagent trajectory references")
            for ref in refs:
                if not isinstance(ref, dict):
                    raise _TrajectoryMergeError("invalid subagent trajectory reference")
                refs_to_resolve.append(ref)

    explicit_path_ids: dict[str, str] = {}
    for ref in refs_to_resolve:
        trajectory_path = ref.get("trajectory_path")
        trajectory_id = ref.get("trajectory_id")
        if not trajectory_path or not isinstance(trajectory_id, str):
            continue
        _relative, reference_key = _normalized_trajectory_reference(trajectory_path)
        reference_cache_key = _trajectory_reference_cache_key(agent_dir, reference_key)
        proposed_id = aliases.get(trajectory_id, trajectory_id)
        prior = explicit_path_ids.get(reference_cache_key)
        if prior is not None and prior != proposed_id:
            raise _TrajectoryMergeError("conflicting trajectory_id aliases for referenced file")
        explicit_path_ids[reference_cache_key] = proposed_id

    for ref in refs_to_resolve:
        trajectory_path = ref.get("trajectory_path")
        trajectory_id = ref.get("trajectory_id")
        embedded_id = aliases.get(trajectory_id, trajectory_id) if isinstance(trajectory_id, str) else None
        if embedded_id in by_id:
            # ATIF allows both keys and recommends preferring an available
            # embedded document over an external sidecar. Remember the path
            # alias so a later path-only reference resolves to this same ID.
            if trajectory_path:
                _relative, reference_key = _normalized_trajectory_reference(trajectory_path)
                reference_cache_key = _trajectory_reference_cache_key(agent_dir, reference_key)
                prior_resolved_id = state.resolved_ids.get(reference_cache_key)
                if prior_resolved_id not in (None, embedded_id):
                    raise _TrajectoryMergeError("conflicting trajectory_id aliases for referenced file")
                state.resolved_ids[reference_cache_key] = embedded_id
            ref["trajectory_id"] = embedded_id
            ref.pop("trajectory_path", None)
        elif trajectory_path:
            _relative, reference_key = _normalized_trajectory_reference(trajectory_path)
            reference_cache_key = _trajectory_reference_cache_key(agent_dir, reference_key)
            remembered_id = state.resolved_ids.get(reference_cache_key)
            candidate_id = remembered_id or explicit_path_ids.get(reference_cache_key)
            if candidate_id is not None and candidate_id in by_id:
                proposed_id = embedded_id or explicit_path_ids.get(reference_cache_key)
                if proposed_id not in (None, candidate_id):
                    raise _TrajectoryMergeError("conflicting trajectory_id aliases for referenced file")
                state.resolved_ids[reference_cache_key] = candidate_id
                ref["trajectory_id"] = candidate_id
                ref.pop("trajectory_path", None)
                continue
            external, reference_key = _materialize_trajectory_file(
                agent_dir,
                trajectory_path,
                state=state,
                depth=depth + 1,
            )
            external_id = external.get("trajectory_id")
            generated_external_id = bool(
                isinstance(external_id, str)
                and state.generated_continuation_fingerprints.get(external_id) == _raw_trajectory_sha256(external)
            )
            if (
                trajectory_id
                and external_id
                and not generated_external_id
                and trajectory_id not in _materialized_trajectory_source_ids(external, state=state)
            ):
                raise _TrajectoryMergeError("subagent trajectory_id does not match referenced file")
            reference_cache_key = _trajectory_reference_cache_key(agent_dir, reference_key)
            prior_resolved_id = state.resolved_ids.get(reference_cache_key)
            proposed_id = (
                trajectory_id
                or explicit_path_ids.get(reference_cache_key)
                or (None if generated_external_id else external_id)
            )
            if prior_resolved_id is not None and proposed_id not in (None, prior_resolved_id):
                raise _TrajectoryMergeError("conflicting trajectory_id aliases for referenced file")
            reference_ordinal = state.reference_ordinals.setdefault(
                reference_cache_key,
                len(state.reference_ordinals),
            )
            resolved_id = (
                prior_resolved_id
                or proposed_id
                or _mint_embedded_trajectory_id(
                    reference_key,
                    external,
                    reference_ordinal=reference_ordinal,
                )
            )
            state.resolved_ids[reference_cache_key] = resolved_id
            external["trajectory_id"] = resolved_id
            external = _validated_trajectory_dict(external)
            if generated_external_id:
                state.generated_continuation_fingerprints[resolved_id] = _raw_trajectory_sha256(external)
            existing = by_id.get(resolved_id)
            if existing is not None and existing != external:
                raise _TrajectoryMergeError("conflicting referenced subagent trajectory_id")
            if existing is None:
                by_id[resolved_id] = external
                embedded.append(external)
            ref["trajectory_id"] = resolved_id
            ref.pop("trajectory_path", None)
        else:
            raise _TrajectoryMergeError("unresolved embedded subagent trajectory_id")
    if embedded:
        trajectory["subagent_trajectories"] = embedded
    else:
        trajectory.pop("subagent_trajectories", None)
    return _validated_trajectory_dict(trajectory)


def _materialize_trajectory_file(
    agent_dir: Path,
    reference: Any,
    *,
    state: _TrajectoryReferenceState | None = None,
    depth: int = 0,
    count_against_reference_budget: bool = True,
) -> tuple[dict[str, Any], str]:
    if depth > _MAX_TRAJECTORY_REFERENCE_DEPTH:
        raise _TrajectoryMergeError("trajectory reference depth exceeded")
    state = state or _TrajectoryReferenceState()
    _relative, key = _normalized_trajectory_reference(reference)
    cache_key = _trajectory_reference_cache_key(agent_dir, key)
    if cached := state.materialized_cache.get(cache_key):
        return copy.deepcopy(cached), key
    if cache_key in state.active:
        raise _TrajectoryMergeError("trajectory reference cycle")
    state.active.add(cache_key)
    try:
        trajectory, _key = _read_referenced_trajectory(
            agent_dir,
            reference,
            state,
            count_against_reference_budget=count_against_reference_budget,
        )
        continued_ref = trajectory.pop("continued_trajectory_ref", None)
        combined_continuation = False
        if continued_ref:
            continuation, _continuation_key = _materialize_trajectory_file(
                agent_dir,
                continued_ref,
                state=state,
                depth=depth + 1,
            )
            trajectory = _combine_continuation_trajectories(trajectory, continuation, state=state)
            combined_continuation = True
        trajectory = _resolve_subagent_trajectory_refs(
            trajectory,
            agent_dir=agent_dir,
            state=state,
            depth=depth,
        )
        if combined_continuation:
            state.generated_continuation_fingerprints[trajectory["trajectory_id"]] = _raw_trajectory_sha256(trajectory)
        state.materialized_cache[cache_key] = trajectory
        return copy.deepcopy(trajectory), key
    finally:
        state.active.discard(cache_key)


def _trial_resumes_step_trajectories(trial_root: Path) -> bool:
    """Read Harbor 0.22's serialized agent resume flag without coercion."""
    result = _read_json(trial_root / "result.json")
    if not isinstance(result, dict):
        return False
    config = result.get("config")
    if not isinstance(config, dict):
        return False
    agent = config.get("agent")
    return isinstance(agent, dict) and agent.get("resume_trajectory") is True


def _valid_step_result_names(value: Any, *, max_count: int | None = None) -> list[str] | None:
    """Return structurally authoritative Harbor step names, or ``None``."""
    if not isinstance(value, list) or not value or (max_count is not None and len(value) > max_count):
        return None
    names: list[str] = []
    seen_names: set[str] = set()
    for step in value:
        if not isinstance(step, dict):
            return None
        step_name = step.get("step_name")
        if not isinstance(step_name, str) or not step_name.strip() or step_name in seen_names:
            return None
        names.append(step_name)
        seen_names.add(step_name)
    return names


def _is_serialized_harbor_trial_result(result: dict[str, Any]) -> bool:
    """Recognize Harbor's complete TrialResult envelope, not legacy fragments."""
    config = result.get("config")
    return (
        isinstance(result.get("trial_uri"), str)
        and isinstance(result.get("task_checksum"), str)
        and isinstance(result.get("agent_info"), dict)
        and isinstance(config, dict)
        and isinstance(config.get("task"), dict)
    )


def _expected_step_trajectory_names(trial_root: Path) -> list[str] | None:
    result = _read_json(trial_root / "result.json")
    if not isinstance(result, dict):
        return None
    return _valid_step_result_names(
        result.get("step_results"),
        max_count=_MAX_TRAJECTORY_STEP_DIRECTORIES,
    )


def _merged_step_trajectory(trial_root: Path) -> dict[str, Any] | None:
    """Merge a complete Harbor multi-step ATIF set or fail closed."""
    try:
        (trial_root / "agent" / "trajectory.json").lstat()
    except FileNotFoundError:
        pass
    except OSError:
        return None
    else:
        # Root and native step trajectories are contradictory authorities.
        return None
    expected_names = _expected_step_trajectory_names(trial_root)
    paths = _ordered_step_trajectory_paths(trial_root)
    discovered_names = [path.parent.parent.name for path in paths]
    if expected_names is None or discovered_names != expected_names:
        return None

    try:
        trajectories: list[tuple[str, dict[str, Any]]] = []
        reference_state = _TrajectoryReferenceState()
        for step_name, path in zip(expected_names, paths, strict=True):
            trajectory, _reference_key = _materialize_trajectory_file(
                path.parent,
                path.name,
                state=reference_state,
                count_against_reference_budget=False,
            )
            trajectories.append((step_name, trajectory))
        if not trajectories:
            return None

        agent_identity = _agent_identity(trajectories[0][1].get("agent"))
        if agent_identity is None or any(
            _agent_identity(trajectory.get("agent")) != agent_identity for _, trajectory in trajectories[1:]
        ):
            raise _TrajectoryMergeError("multi-step agent mismatch")

        merged_steps: list[dict[str, Any]] = []
        scoped_trajectories: list[tuple[str, dict[str, Any]]] = []
        resume_trajectory = _trial_resumes_step_trajectories(trial_root)
        continuation_flags: list[bool] = []
        previous_trajectory: dict[str, Any] | None = None
        for source_index, (step_name, trajectory) in enumerate(trajectories):
            is_continuation = bool(
                resume_trajectory
                and previous_trajectory is not None
                and _is_trusted_materialized_cumulative_continuation(
                    previous_trajectory,
                    trajectory,
                    state=reference_state,
                )
            )
            continuation_flags.append(is_continuation)
            scoped_trajectory = _remap_parent_subagent_scope(
                trajectory,
                namespace=f"harbor-step:{source_index}",
            )
            scoped_trajectories.append((step_name, scoped_trajectory))
            steps = _trajectory_dict_steps(scoped_trajectory)
            if is_continuation:
                copied_prefix_length = _copied_context_prefix_length(steps)
                previous_length = len(_trajectory_dict_steps(previous_trajectory)) if previous_trajectory else 0
                steps = steps[copied_prefix_length or previous_length :]
            for step in steps:
                merged_step = copy.deepcopy(step)
                original_step_id = merged_step.get("step_id")
                merged_step["step_id"] = len(merged_steps) + 1
                extra = merged_step.get("extra")
                if not isinstance(extra, dict):
                    extra = {}
                extra["harbor_step_name"] = step_name
                if original_step_id not in (None, ""):
                    extra["harbor_original_step_id"] = original_step_id
                else:
                    extra.pop("harbor_original_step_id", None)
                merged_step["extra"] = extra
                merged_steps.append(merged_step)
            previous_trajectory = trajectory
        if not merged_steps:
            raise _TrajectoryMergeError("empty merged trajectory")

        source_provenance = [
            {
                "step_name": step_name,
                "schema_version": trajectory.get("schema_version"),
                "session_id": trajectory.get("session_id"),
                "trajectory_id": trajectory.get("trajectory_id"),
                "root_extra": copy.deepcopy(trajectory.get("extra")),
                "agent_extra": copy.deepcopy(
                    agent.get("extra") if isinstance((agent := trajectory.get("agent")), dict) else None
                ),
                "final_metrics_extra": copy.deepcopy(
                    final_metrics.get("extra")
                    if isinstance((final_metrics := trajectory.get("final_metrics")), dict)
                    else None
                ),
            }
            for step_name, trajectory in trajectories
        ]
        source_identity = [
            {
                "source_index": source_index,
                "step_name": redact_sensitive_text(step_name),
                "is_continuation": continuation_flags[source_index],
                "content_sha256": _canonical_trajectory_sha256(trajectory),
            }
            for source_index, (step_name, trajectory) in enumerate(trajectories)
        ]
        digest = hashlib.sha256(
            json.dumps(source_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:20]
        merged: dict[str, Any] = {
            "schema_version": "ATIF-v1.7",
            "session_id": f"skillevaluator-multistep-{digest}",
            "trajectory_id": f"skillevaluator-multistep-{digest}",
            "agent": copy.deepcopy(trajectories[-1][1]["agent"]),
            "steps": merged_steps,
            "final_metrics": _merge_trajectory_final_metrics(
                [trajectory for _, trajectory in trajectories],
                continuation_flags=continuation_flags,
                total_steps=len(merged_steps),
            ),
            "extra": {
                "harbor_multi_step": {
                    "step_count": len(expected_names),
                    "step_names": expected_names,
                    "source_trajectories": source_provenance,
                }
            },
        }
        notes = _combined_notes(
            *(
                f"[{step_name}] {note}"
                for step_name, trajectory in trajectories
                if isinstance((note := trajectory.get("notes")), str) and note
            )
        )
        if notes:
            merged["notes"] = notes
        subagents = _merged_embedded_subagents(
            *(trajectory.get("subagent_trajectories") for _, trajectory in scoped_trajectories)
        )
        if subagents:
            merged["subagent_trajectories"] = subagents
        return _validated_trajectory_dict(merged)
    except (OSError, SecurePathError, _TrajectoryMergeError, RecursionError):
        logger.debug("Harbor multi-step trajectory could not be materialized", exc_info=True)
        return None


def _merge_trajectory_final_metrics(
    trajectories: list[dict[str, Any]],
    *,
    continuation_flags: list[bool],
    total_steps: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in (
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
    ):
        value = _sum_trajectory_metric_segments(
            trajectories,
            continuation_flags=continuation_flags,
            key=key,
        )
        if value is not None:
            metrics[key] = value

    metrics["total_steps"] = total_steps
    terminal_final_metrics = trajectories[-1].get("final_metrics") if trajectories else None
    last_final_metrics = terminal_final_metrics if isinstance(terminal_final_metrics, dict) else {}
    last_extra = last_final_metrics.get("extra") if isinstance(last_final_metrics, dict) else None
    extra = {
        key: copy.deepcopy(last_extra[key])
        for key in ("finish_reason",)
        if isinstance(last_extra, dict) and key in last_extra
    }
    extra_token_keys = sorted(
        {
            str(key)
            for trajectory in trajectories
            if isinstance(final_metrics := trajectory.get("final_metrics"), dict)
            if isinstance(step_extra := final_metrics.get("extra"), dict)
            for key in step_extra
            if _is_aggregate_extra_token_key(str(key))
        }
    )
    for key in list(extra):
        if _is_aggregate_extra_token_key(str(key)):
            extra.pop(key, None)
    for key in extra_token_keys:
        value = _sum_trajectory_metric_segments(
            trajectories,
            continuation_flags=continuation_flags,
            key=key,
            from_extra=True,
        )
        if value is not None:
            extra[key] = value
    extra["harbor_multi_step"] = True
    metrics["extra"] = extra
    return metrics


def _sum_trajectory_metric_segments(
    trajectories: list[dict[str, Any]],
    *,
    continuation_flags: list[bool],
    key: str,
    from_extra: bool = False,
) -> int | float | None:
    """Sum independent fragments while taking the latest cumulative value per resume chain."""
    total: int | float = 0
    segment_value: int | float | None = None
    segment_known = False
    for index, trajectory in enumerate(trajectories):
        if index == 0 or not continuation_flags[index]:
            if index > 0:
                if not segment_known or segment_value is None:
                    return None
                total += segment_value
            segment_value = None
            segment_known = False

        final_metrics = trajectory.get("final_metrics")
        source = final_metrics.get("extra") if from_extra and isinstance(final_metrics, dict) else final_metrics
        value = source.get(key) if isinstance(source, dict) else None
        try:
            is_finite_number = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        except OverflowError:
            is_finite_number = False
        if is_finite_number:
            segment_value = value
            segment_known = True
        else:
            segment_value = None
            segment_known = False

    if not segment_known or segment_value is None:
        return None
    return total + segment_value


def _merge_reward_sidecars(data: dict[str, Any], verifier_dir: Path) -> None:
    """Merge SkillEvaluator-rich sidecars back into Harbor's numeric-only reward payload."""
    skill_evaluator_reward = _read_json(verifier_dir / "skill_evaluator_reward.json")
    if isinstance(skill_evaluator_reward, dict):
        sidecar_failed = str(skill_evaluator_reward.get("evaluation_status") or "").casefold() in {
            "error",
            "failed",
        }
        if sidecar_failed:
            data["evaluation_status"] = "failed"
            if "evaluation_errors" in skill_evaluator_reward:
                data["evaluation_errors"] = _safe_evaluation_errors(skill_evaluator_reward["evaluation_errors"])
            else:
                data.pop("evaluation_errors", None)
        for key, value in skill_evaluator_reward.items():
            if sidecar_failed and key in {"evaluation_status", "evaluation_errors"}:
                continue
            if key in data and isinstance(data.get(key), int | float) and not isinstance(data.get(key), bool):
                continue
            if key == "evaluation_errors":
                value = _safe_evaluation_errors(value)
            data.setdefault(key, value)

    custom_reward = _read_json(verifier_dir / "custom_reward.json")
    if not isinstance(custom_reward, dict):
        return

    if custom_metric_contract_error(custom_reward):
        data["evaluation_status"] = "failed"
        data["evaluation_errors"] = _merge_bounded_evaluation_errors(
            {"collector": UNSAFE_CUSTOM_METRICS_REASON},
            data.get("evaluation_errors"),
        )
        return

    custom_metrics = extract_custom_metrics(custom_reward)
    if custom_metrics:
        existing = data.get("custom_metrics")
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(custom_metrics)
        data["custom_metrics"] = merged

    custom_details = custom_reward.get("details")
    if isinstance(custom_details, dict):
        details = data.get("details")
        if not isinstance(details, dict):
            details = {}
        for metric, detail in custom_details.items():
            if metric in custom_metrics and metric not in details:
                details[metric] = detail
        if details:
            data["details"] = details
        safe_custom_details = {
            str(metric): detail for metric, detail in custom_details.items() if str(metric) in custom_metrics
        }
        if safe_custom_details:
            data["custom_details"] = safe_custom_details

    for key in ("entry_id", "error"):
        value = custom_reward.get(key)
        if value not in (None, ""):
            data.setdefault(key, value)


def _merge_trial_evaluation_failures(data: dict[str, Any], trial_dir: Path) -> None:
    """Preserve judge-failure diagnostics when Harbor supplies an aggregate reward."""
    # Pure custom-only verifiers do not run the standard Tier-3 LLM judge and
    # must not inherit its sidecar scan limits merely because step directories
    # exist. Default and default-plus-custom rewards carry canonical metrics.
    if not _standard_reward_metrics(data):
        return
    diagnostic = _failed_judge_diagnostic(trial_dir)
    if diagnostic is None:
        return

    merged_errors = _merge_bounded_evaluation_errors(
        diagnostic.get("evaluation_errors"),
        data.get("evaluation_errors"),
    )

    data["evaluation_status"] = "failed"
    if merged_errors:
        data["evaluation_errors"] = merged_errors
    else:
        data.pop("evaluation_errors", None)


def _merge_bounded_evaluation_errors(*values: Any) -> dict[str, str]:
    """Merge diagnostics in priority order while retaining the public display bound."""
    merged: dict[str, str] = {}
    for value in values:
        safe_errors = _safe_evaluation_errors(value)
        if isinstance(safe_errors, dict):
            items = safe_errors.items()
        elif isinstance(safe_errors, list):
            items = ((f"judge_{index + 1}", reason) for index, reason in enumerate(safe_errors))
        elif safe_errors:
            items = (("judge", safe_errors),)
        else:
            items = ()
        for metric, reason in items:
            merged.setdefault(str(metric), str(reason))
            if len(merged) >= len(DEFAULT_METRICS):
                return merged
    return merged


def _standard_reward_metrics(
    rewards: dict[str, Any],
    *,
    inherited_metrics: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return required standard metrics, preserving explicit child metric-set declarations."""
    nested_metrics = rewards.get("metrics")
    metric_names = {str(name) for name in rewards}
    if isinstance(nested_metrics, dict):
        metric_names.update(str(name) for name in nested_metrics)
    declared_metric_set = str(rewards.get("metric_set") or rewards.get("metric_set_version") or "")
    if declared_metric_set and declared_metric_set not in {DEFAULT_METRIC_SET, LEGACY_METRIC_SET}:
        return ()
    if not declared_metric_set and not metric_names.intersection(DEFAULT_METRICS):
        return ()

    _, expected_metrics = metric_set_for_reward(rewards)
    if not declared_metric_set and inherited_metrics:
        return inherited_metrics
    if expected_metrics:
        return expected_metrics
    # An unversioned all-non-finite standard shape can otherwise look
    # custom-only because its numeric ``overall`` is the sole finite value.
    return DEFAULT_METRICS if "security" in metric_names else LEGACY_METRICS


def _physical_steps_layout_present(trial_root: Path) -> bool:
    """Return whether a trial has any physical steps entry, failing closed on I/O errors."""
    try:
        (trial_root / "steps").lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _reward_claims_metric(rewards: dict[str, Any], metric: str) -> bool:
    if metric in rewards:
        return True
    nested = rewards.get("metrics")
    return isinstance(nested, dict) and metric in nested


def _standard_aggregate_matches_harbor_strategy(
    root_rewards: dict[str, Any],
    step_results: list[dict[str, Any]],
    root_metrics: tuple[str, ...],
) -> bool:
    """Recognize Harbor's FINAL or missing-as-zero MEAN aggregate semantics."""
    root_values = {metric: metric_value(root_rewards, metric) for metric in root_metrics}
    if any(value is None for value in root_values.values()):
        return False

    final_rewards: dict[str, Any] | None = None
    if step_results:
        final_verifier = step_results[-1].get("verifier_result")
        candidate = final_verifier.get("rewards") if isinstance(final_verifier, dict) else None
        if isinstance(candidate, dict):
            final_rewards = candidate
    if final_rewards is not None and all(
        (value := metric_value(final_rewards, metric)) is not None
        and math.isclose(value, root_values[metric], rel_tol=1e-9, abs_tol=1e-9)
        for metric in root_metrics
    ):
        return True

    verifier_rewards: list[dict[str, Any]] = []
    for step in step_results:
        verifier_result = step.get("verifier_result")
        if verifier_result is None:
            continue
        if not isinstance(verifier_result, dict):
            return False
        rewards = verifier_result.get("rewards")
        if rewards is None:
            verifier_rewards.append({})
        elif isinstance(rewards, dict):
            verifier_rewards.append(rewards)
        else:
            return False
    if not verifier_rewards:
        return False

    for metric in root_metrics:
        values: list[float] = []
        for rewards in verifier_rewards:
            if not _reward_claims_metric(rewards, metric):
                values.append(0.0)
                continue
            value = metric_value(rewards, metric)
            if value is None:
                return False
            values.append(value)
        mean = sum(values) / len(verifier_rewards)
        if not math.isclose(mean, root_values[metric], rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def _constituent_default_reward_failure(result: dict[str, Any], trial_root: Path | None = None) -> str:
    """Return a safe failure when a standard step reward cannot support its aggregate."""
    root_verifier = result.get("verifier_result")
    root_rewards = root_verifier.get("rewards") if isinstance(root_verifier, dict) else None
    root_failed = str(result.get("evaluation_status") or "").casefold() in {"error", "failed"}
    if isinstance(root_verifier, dict):
        root_failed = root_failed or str(root_verifier.get("evaluation_status") or "").casefold() in {
            "error",
            "failed",
        }
    if isinstance(root_rewards, dict):
        root_failed = root_failed or str(root_rewards.get("evaluation_status") or "").casefold() in {
            "error",
            "failed",
        }
    if root_failed:
        return "Authoritative verifier reward is failed; it was not scored"

    step_results = result.get("step_results")
    # Harbor serializes ``None`` for a successful single-step trial. A physical
    # steps layout makes that sentinel ambiguous, so retain fail-closed handling.
    if (
        step_results is None
        and "step_results" in result
        and trial_root is not None
        and _physical_steps_layout_present(trial_root)
    ):
        return "Authoritative verifier result has malformed constituent steps; it was not scored"
    if step_results is not None and not isinstance(step_results, list):
        return "Authoritative verifier result has malformed constituent steps; it was not scored"
    if not isinstance(step_results, list):
        return ""
    if _valid_step_result_names(step_results) is None:
        return "Authoritative verifier result has malformed constituent steps; it was not scored"
    if _is_serialized_harbor_trial_result(result) and not isinstance(root_rewards, dict):
        return MISSING_MULTI_STEP_REWARD_REASON

    root_metrics = _standard_reward_metrics(root_rewards) if isinstance(root_rewards, dict) else ()
    if root_metrics and not _standard_aggregate_matches_harbor_strategy(root_rewards, step_results, root_metrics):
        return (
            "Constituent default rewards do not match Harbor's final or mean aggregate semantics; "
            "the authoritative aggregate was not scored"
        )

    for index, step in enumerate(step_results, start=1):
        if not isinstance(step, dict):
            if root_metrics:
                return (
                    f"Constituent default reward for step {index} is incomplete, non-finite, or failed; "
                    "the authoritative aggregate was not scored"
                )
            continue
        step_name = _safe_diagnostic_text(step.get("step_name"), max_len=64) or str(index)
        verifier_result = step.get("verifier_result")
        step_failed = ("exception_info" in step and step.get("exception_info") is not None) or str(
            step.get("evaluation_status") or ""
        ).casefold() in {
            "error",
            "failed",
        }
        if not isinstance(verifier_result, dict):
            if step_failed:
                return (
                    f"Constituent default reward for step {step_name} is incomplete, non-finite, or failed; "
                    "the authoritative aggregate was not scored"
                )
            continue
        rewards = verifier_result.get("rewards")
        failed_status = step_failed or str(verifier_result.get("evaluation_status") or "").casefold() in {
            "error",
            "failed",
        }
        if isinstance(rewards, dict):
            failed_status = failed_status or str(rewards.get("evaluation_status") or "").casefold() in {
                "error",
                "failed",
            }
        if failed_status or (root_metrics and rewards is not None and not isinstance(rewards, dict)):
            return (
                f"Constituent default reward for step {step_name} is incomplete, non-finite, or failed; "
                "the authoritative aggregate was not scored"
            )
        if not isinstance(rewards, dict):
            continue

        expected_metrics = _standard_reward_metrics(
            rewards,
            inherited_metrics=root_metrics,
        )
        if not expected_metrics:
            continue
        if all(
            not _reward_claims_metric(rewards, metric) or metric_value(rewards, metric) is not None
            for metric in expected_metrics
        ):
            continue

        return (
            f"Constituent default reward for step {step_name} is incomplete, non-finite, or failed; "
            "the authoritative aggregate was not scored"
        )
    return ""


def _constituent_custom_metric_failure(result: dict[str, Any]) -> str:
    """Validate custom metric bounds without reconstructing Harbor's root reward."""
    reward_rows: list[dict[str, Any]] = []
    root_verifier = result.get("verifier_result")
    root_rewards = root_verifier.get("rewards") if isinstance(root_verifier, dict) else None
    if isinstance(root_rewards, dict):
        reward_rows.append(root_rewards)

    step_results = result.get("step_results")
    if isinstance(step_results, list):
        for step in step_results:
            if not isinstance(step, dict):
                continue
            verifier_result = step.get("verifier_result")
            rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
            if isinstance(rewards, dict):
                reward_rows.append(rewards)

    custom_names: set[str] = set()
    for rewards in reward_rows:
        for raw_name, raw_value in rewards.items():
            name = str(raw_name)
            if isinstance(raw_value, int | float) and not isinstance(raw_value, bool):
                continue
            if name == "metrics" and isinstance(raw_value, dict):
                nested_standard_metrics_are_valid = all(
                    str(metric) in DEFAULT_METRICS
                    and score_value(value.get("score") if isinstance(value, dict) else value) is not None
                    for metric, value in raw_value.items()
                )
                if nested_standard_metrics_are_valid:
                    continue
            if name in RESERVED_METRIC_NAMES and name not in {"custom_metrics", "metrics"}:
                continue
            if name.startswith("_"):
                continue
            return MALFORMED_HARBOR_REWARD_REASON
        if custom_metric_contract_error(rewards):
            return UNSAFE_CUSTOM_METRICS_REASON
        custom_names.update(extract_custom_metrics(rewards))
        if len(custom_names) > MAX_CUSTOM_METRICS:
            return UNSAFE_CUSTOM_METRICS_REASON
    return ""


def _merge_constituent_default_reward_failure(
    data: dict[str, Any],
    result: dict[str, Any],
    trial_root: Path | None = None,
) -> None:
    """Make an aggregate unscoreable when one of its constituents is invalid."""
    reasons = (
        _constituent_default_reward_failure(result, trial_root),
        _constituent_custom_metric_failure(result),
    )
    reason = "; ".join(item for item in reasons if item)
    if not reason:
        return
    data["evaluation_status"] = "failed"
    data["evaluation_errors"] = _merge_bounded_evaluation_errors(
        {"collector": reason},
        data.get("evaluation_errors"),
    )


def _extract_rewards(
    job_dir: Path,
    case_id_by_task_selector: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Extract reward.json from all trials in a job directory."""
    rewards: list[dict[str, Any]] = []
    scored_trial_roots: set[Path] = set()
    authoritative_trial_roots: set[Path] = set()

    # Native multi-step Harbor tasks can persist an authoritative aggregate at
    # the trial root in addition to one reward file per step. Materialize that
    # single logical row first so both averages and pass@k use the same score.
    for result_file in sorted(job_dir.glob("*/result.json")):
        trial_dir = result_file.parent
        if _trial_failure_reason(trial_dir) or _is_agent_runtime_failure_trial(trial_dir):
            continue
        result = _read_json(result_file)
        if not isinstance(result, dict) or not isinstance(result.get("step_results"), list):
            continue
        verifier_result = result.get("verifier_result")
        if not isinstance(verifier_result, dict) or not isinstance(verifier_result.get("rewards"), dict):
            continue
        data = _reward_from_harbor_result(result)
        if not data:
            continue
        _merge_constituent_default_reward_failure(data, result, trial_dir)
        _merge_trial_evaluation_failures(data, trial_dir)
        trial_name = str(result.get("trial_name") or trial_dir.name)
        data["_trial_name"] = trial_name
        data["_trial_root_name"] = trial_dir.name
        data["_started_at"] = result.get("started_at")
        _apply_harbor_result_case_identity(data, result, case_id_by_task_selector)
        traj_file = _reward_trajectory_path(trial_dir, None)
        if traj_file.exists():
            data["_has_trajectory"] = True
        data = _fail_closed_invalid_reward_numbers(data)
        rewards.append(data)
        authoritative_trial_roots.add(trial_dir)

    for reward_file in sorted(job_dir.rglob("reward.json")):
        if reward_file.parent.name == "verifier":
            try:
                trial_dir, trial_name, step_name = _reward_trial_context(reward_file)
                if _trial_failure_reason(trial_dir) or _is_agent_runtime_failure_trial(trial_dir):
                    continue
                if trial_dir in authoritative_trial_roots:
                    continue
                data = _read_json(reward_file)
                if not isinstance(data, dict):
                    logger.warning("Ignoring invalid or oversized Harbor reward: %s", reward_file)
                    continue
                _merge_reward_sidecars(data, reward_file.parent)
                _merge_trial_evaluation_failures(data, trial_dir)
                if _trial_failure_reason(trial_dir) or _is_agent_runtime_failure_trial(trial_dir):
                    logger.debug(
                        "Skipping reward for failed Harbor trial: %s",
                        trial_dir,
                    )
                    continue
                data["_trial_name"] = trial_name
                data["_trial_root_name"] = trial_dir.name
                if step_name:
                    data["_step_name"] = step_name
                result_file = trial_dir / "result.json"
                result: dict[str, Any] = {}
                if result_file.exists():
                    loaded_result = _read_json(result_file)
                    if isinstance(loaded_result, dict):
                        result = loaded_result
                        _merge_constituent_default_reward_failure(data, result, trial_dir)
                        data["_started_at"] = result.get("started_at")
                _apply_harbor_result_case_identity(data, result, case_id_by_task_selector)
                traj_file = _reward_trajectory_path(trial_dir, step_name)
                if traj_file.exists():
                    data["_has_trajectory"] = True
                data = _fail_closed_invalid_reward_numbers(data)
                rewards.append(data)
                scored_trial_roots.add(trial_dir)
            except OSError as e:
                logger.warning("Failed to read %s: %s", reward_file, e)

    for result_file in sorted(job_dir.glob("*/result.json")):
        trial_dir = result_file.parent
        if (
            trial_dir in authoritative_trial_roots
            or trial_dir in scored_trial_roots
            or _trial_failure_reason(trial_dir)
            or _is_agent_runtime_failure_trial(trial_dir)
        ):
            continue
        result = _read_json(result_file)
        if not isinstance(result, dict):
            continue
        # Older Harbor fragments can contain only embedded step verifier rows,
        # with no root aggregate and no physical reward sidecars. Preserve the
        # rows as separate diagnostics so row-specific metric-set semantics and
        # logical-overall weighting survive collection and report reloads.
        step_names = (
            None if _is_serialized_harbor_trial_result(result) else _valid_step_result_names(result.get("step_results"))
        )
        embedded_rows: list[tuple[str, dict[str, Any]]] = []
        if step_names is not None:
            for step_name, step in zip(step_names, result["step_results"], strict=True):
                step_rewards = _harbor_result_rewards({"step_results": [step]})
                data = (
                    _reward_from_harbor_result({"verifier_result": {"rewards": step_rewards}}) if step_rewards else None
                )
                if data:
                    embedded_rows.append((step_name, data))
        if embedded_rows:
            for step_name, data in embedded_rows:
                _merge_constituent_default_reward_failure(data, result, trial_dir)
                _merge_trial_evaluation_failures(data, trial_dir)
                trial_name = str(result.get("trial_name") or trial_dir.name)
                data["_trial_name"] = trial_name
                data["_trial_root_name"] = trial_dir.name
                data["_step_name"] = step_name
                data["_started_at"] = result.get("started_at")
                _apply_harbor_result_case_identity(data, result, case_id_by_task_selector)
                traj_file = _reward_trajectory_path(trial_dir, step_name)
                if traj_file.exists():
                    data["_has_trajectory"] = True
                rewards.append(_fail_closed_invalid_reward_numbers(data))
            continue
        data = _reward_from_harbor_result(result)
        if not data:
            continue
        _merge_constituent_default_reward_failure(data, result, trial_dir)
        _merge_trial_evaluation_failures(data, trial_dir)
        trial_name = str(result.get("trial_name") or trial_dir.name)
        data["_trial_name"] = trial_name
        data["_trial_root_name"] = trial_dir.name
        data["_started_at"] = result.get("started_at")
        _apply_harbor_result_case_identity(data, result, case_id_by_task_selector)
        traj_file = _reward_trajectory_path(trial_dir, None)
        if traj_file.exists():
            data["_has_trajectory"] = True
        data = _fail_closed_invalid_reward_numbers(data)
        rewards.append(data)
    return rewards


def _finite_reward_number(value: Any) -> float | None:
    """Convert a Harbor numeric reward without propagating overflow or non-finite values."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


class _RewardStructureLimitError(ValueError):
    """Raised when a reward cannot be copied safely for generated output."""


def _normalized_reward_numbers(
    value: Any,
    *,
    _depth: int = 1,
    _node_budget: list[int] | None = None,
    _max_nodes: int = COLLECTED_REWARD_JSON_MAX_NODES,
) -> tuple[Any, bool]:
    """Return JSON-safe reward data and whether an invalid number was replaced."""
    node_budget = _node_budget if _node_budget is not None else [0]
    node_budget[0] += 1
    if node_budget[0] > _max_nodes:
        raise _RewardStructureLimitError("reward node count exceeds limit")
    if isinstance(value, bool):
        return value, False
    if isinstance(value, int):
        if abs(value) > _MAX_JSON_SAFE_INTEGER:
            return None, True
        return value, False
    if isinstance(value, float):
        if _finite_reward_number(value) is None:
            return None, True
        return value, False
    if isinstance(value, dict):
        if _depth > REWARD_JSON_MAX_DEPTH:
            raise _RewardStructureLimitError("reward depth exceeds limit")
        invalid = False
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_item, item_invalid = _normalized_reward_numbers(
                item,
                _depth=_depth + 1,
                _node_budget=node_budget,
                _max_nodes=_max_nodes,
            )
            normalized[str(key)] = normalized_item
            invalid = invalid or item_invalid
        return normalized, invalid
    if isinstance(value, list):
        if _depth > REWARD_JSON_MAX_DEPTH:
            raise _RewardStructureLimitError("reward depth exceeds limit")
        invalid = False
        normalized_items: list[Any] = []
        for item in value:
            normalized_item, item_invalid = _normalized_reward_numbers(
                item,
                _depth=_depth + 1,
                _node_budget=node_budget,
                _max_nodes=_max_nodes,
            )
            normalized_items.append(normalized_item)
            invalid = invalid or item_invalid
        return normalized_items, invalid
    return value, False


def _structural_limit_reward(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only shallow identifiers when a reward payload is unsafe to traverse."""
    safe: dict[str, Any] = {}
    for key in (
        "_trial_name",
        "_trial_root_name",
        "_step_name",
        "_started_at",
        "entry_id",
        "trial_id",
        "agent",
        "model",
        "model_source",
    ):
        value = data.get(key)
        if isinstance(value, str):
            safe_value = _safe_diagnostic_text(value, max_len=512)
            if safe_value:
                safe[key] = safe_value
    if isinstance(data.get("_has_trajectory"), bool):
        safe["_has_trajectory"] = data["_has_trajectory"]
    safe["evaluation_status"] = "failed"
    safe["evaluation_errors"] = {"collector": UNSAFE_REWARD_STRUCTURE_REASON}
    return safe


def _fail_closed_invalid_reward_numbers(
    data: dict[str, Any],
    *,
    max_nodes: int = COLLECTED_REWARD_JSON_MAX_NODES,
    max_bytes: int = COLLECTED_REWARD_JSON_MAX_BYTES,
) -> dict[str, Any]:
    """Replace invalid numbers and attach a bounded diagnostic to the reward."""
    try:
        normalized, invalid = _normalized_reward_numbers(data, _max_nodes=max_nodes)
        encoded = json.dumps(
            normalized,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > max_bytes:
            raise _RewardStructureLimitError("reward bytes exceed generated-artifact limit")
    except (_RewardStructureLimitError, TypeError, ValueError, RecursionError, MemoryError):
        return _structural_limit_reward(data)
    if not isinstance(normalized, dict):
        return data
    if custom_metric_contract_error(normalized):
        normalized["evaluation_status"] = "failed"
        normalized["evaluation_errors"] = _merge_bounded_evaluation_errors(
            {"collector": UNSAFE_CUSTOM_METRICS_REASON},
            normalized.get("evaluation_errors"),
        )
    if invalid:
        normalized["evaluation_status"] = "failed"
        normalized["evaluation_errors"] = _merge_bounded_evaluation_errors(
            {"collector": UNSCOREABLE_NUMERIC_REWARD_REASON},
            normalized.get("evaluation_errors"),
        )
    return normalized


def _reward_from_harbor_result(result: dict[str, Any]) -> dict[str, Any] | None:
    root_verifier = result.get("verifier_result")
    root_rewards = root_verifier.get("rewards") if isinstance(root_verifier, dict) else None
    if (
        _is_serialized_harbor_trial_result(result)
        and _valid_step_result_names(result.get("step_results")) is not None
        and not isinstance(root_rewards, dict)
    ):
        return {
            "evaluation_status": "failed",
            "evaluation_errors": {"collector": MISSING_MULTI_STEP_REWARD_REASON},
            "details": {"harbor_rewards": {}},
        }

    harbor_rewards = _harbor_result_rewards(result)
    if not harbor_rewards:
        return None

    data: dict[str, Any] = {}
    custom_metrics: dict[str, float] = {}
    safe_harbor_rewards: dict[str, Any] = {}
    invalid_numeric_reward = False
    for key, value in harbor_rewards.items():
        if key == _CUSTOM_METRIC_CONTRACT_MARKER:
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            safe_harbor_rewards[str(key)] = value
            continue
        score = _finite_reward_number(value)
        safe_harbor_rewards[str(key)] = value if score is not None else None
        if score is None:
            invalid_numeric_reward = True
            continue
        if key in DEFAULT_METRICS:
            data[key] = score
        elif key == "overall":
            data["overall"] = score
        elif key == "reward":
            data.setdefault("overall", score)
            custom_metrics[key] = score
        else:
            custom_metrics[key] = score

    metric_set = harbor_rewards.get("metric_set") or harbor_rewards.get("metric_set_version")
    if isinstance(metric_set, str) and metric_set:
        data["metric_set"] = metric_set

    if invalid_numeric_reward:
        data["evaluation_status"] = "failed"
        data["evaluation_errors"] = {"collector": UNSCOREABLE_NUMERIC_REWARD_REASON}
    if harbor_rewards.get(_CUSTOM_METRIC_CONTRACT_MARKER):
        data["evaluation_status"] = "failed"
        data["evaluation_errors"] = {"collector": UNSAFE_CUSTOM_METRICS_REASON}
    if not any(not k.startswith("_") for k in data) and not custom_metrics:
        return None
    if custom_metrics:
        data["custom_metrics"] = custom_metrics
    data["details"] = {"harbor_rewards": safe_harbor_rewards}
    return data


def _average_multistep_custom_metrics(rewards: list[dict[str, Any]]) -> dict[str, float]:
    """Mirror Harbor 0.22 MEAN aggregation by zero-filling missing custom keys."""
    if not rewards:
        return {}
    names: set[str] = set()
    extracted_rows: list[dict[str, float]] = []
    for reward in rewards:
        if reason := custom_metric_contract_error(reward):
            raise CustomMetricContractError(reason)
        extracted = extract_custom_metrics(reward)
        names.update(extracted)
        if len(names) > MAX_CUSTOM_METRICS:
            raise CustomMetricContractError("Custom metric union exceeds the per condition publication limit")
        extracted_rows.append(extracted)
    denominator = len(extracted_rows)
    return {name: round(sum(row.get(name, 0.0) for row in extracted_rows) / denominator, 4) for name in sorted(names)}


def _harbor_result_rewards(result: dict[str, Any]) -> dict[str, Any] | None:
    verifier_result = result.get("verifier_result")
    if isinstance(verifier_result, dict):
        rewards = verifier_result.get("rewards")
        if isinstance(rewards, dict):
            return rewards

    step_reward_rows: list[dict[str, Any]] = []
    step_results = result.get("step_results")
    if isinstance(step_results, list):
        for step in step_results:
            if not isinstance(step, dict):
                continue
            step_verifier = step.get("verifier_result")
            if not isinstance(step_verifier, dict):
                continue
            step_rewards = step_verifier.get("rewards")
            if isinstance(step_rewards, dict):
                step_reward_rows.append(step_rewards)
    if not step_reward_rows:
        return None

    aggregated: dict[str, Any] = {}
    for key in sorted({str(key) for rewards in step_reward_rows for key in rewards}):
        raw_values = [
            rewards[key]
            for rewards in step_reward_rows
            if isinstance(rewards.get(key), int | float) and not isinstance(rewards.get(key), bool)
        ]
        values = [_finite_reward_number(value) for value in raw_values]
        if any(value is None for value in values):
            # Preserve one invalid numeric long enough for the shared reward
            # normalizer to mark the whole artifact unscoreable and replace it
            # with strict-JSON ``null``. Silently dropping only the bad step
            # would let a partial average pass.
            aggregated[key] = next(
                raw_value for raw_value, value in zip(raw_values, values, strict=True) if value is None
            )
        elif values:
            aggregated[key] = sum(values) / len(values)

    try:
        aggregated.update(_average_multistep_custom_metrics(step_reward_rows))
    except CustomMetricContractError:
        return {
            "metric_set": CUSTOM_ONLY_METRIC_SET,
            _CUSTOM_METRIC_CONTRACT_MARKER: True,
        }

    # Classify each row before accepting canonical metric names. An explicitly
    # custom-only step may carry arbitrary keys, including reserved names, but
    # those names must never be reclassified as SkillEvaluator-owned scores.
    standard_scores, metric_set, _active_metrics = average_metrics(step_reward_rows)
    has_standard_contract = any(_standard_reward_metrics(rewards) for rewards in step_reward_rows)
    if not has_standard_contract and any(
        _finite_reward_number(aggregated.get(name)) is not None for name in ("overall", "reward")
    ):
        metric_set = CUSTOM_ONLY_METRIC_SET
    for metric in DEFAULT_METRICS:
        aggregated.pop(metric, None)
    aggregated.update(standard_scores)
    aggregated["metric_set"] = metric_set
    if (logical_overall := _average_overall(step_reward_rows)) is not None:
        aggregated["overall"] = logical_overall
    return aggregated or None


def _task_selector_from_harbor_result(result: dict[str, Any]) -> str:
    """Return one consistent staged selector, never ``[task].name``."""
    task_paths: list[str] = []
    task_id = result.get("task_id")
    if isinstance(task_id, dict):
        task_path = task_id.get("path")
        if isinstance(task_path, str) and task_path.strip():
            task_paths.append(task_path.strip())

    config = result.get("config")
    if isinstance(config, dict):
        task = config.get("task")
        if isinstance(task, dict):
            task_path = task.get("path")
            if isinstance(task_path, str) and task_path.strip():
                task_paths.append(task_path.strip())

    if not task_paths or any(task_path != task_paths[0] for task_path in task_paths):
        return ""
    return Path(task_paths[0]).name


def _entry_id_from_harbor_result(
    result: dict[str, Any],
    case_id_by_task_selector: dict[str, str] | None = None,
) -> str:
    if case_id_by_task_selector is None:
        # Preserve the legacy collector contract for direct callers and older
        # artifacts: Harbor's task_name is the logical/display identity they
        # supplied. New runner paths pass an explicit trusted selector map and
        # never use this authored field as logical truth.
        task_name = result.get("task_name")
        if isinstance(task_name, str) and task_name.strip():
            return task_name.strip().rsplit("/", 1)[-1]

    selector = _task_selector_from_harbor_result(result)
    if selector:
        return case_id_by_task_selector.get(selector, "") if case_id_by_task_selector is not None else selector

    # A trusted mapping deliberately keeps authored ``[task].name`` separate
    # from logical identity. Harbor 0.22 normally persists ``task_id.path``;
    # if it is absent, fail coverage instead of guessing from a display name.
    if case_id_by_task_selector is not None:
        return ""

    return ""


def _positive_attempt_ordinal(value: object) -> int | None:
    """Return a positive integer attempt ordinal without accepting booleans."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        ordinal = int(value)
        return ordinal if ordinal > 0 else None
    return None


def _structural_attempt_ordinal(trial_root_name: object, task_selector: str) -> int | None:
    """Read an ordinal only from runner-owned or exact legacy selector structure."""
    if not isinstance(trial_root_name, str) or not trial_root_name or not task_selector:
        return None

    # stop_on_pass aggregates are named ``<job>-<selector>-attemptNNN__<child>``.
    # Consume the complete trusted selector before reading the runner-owned
    # suffix so attempt-like text inside any authored identity is inert.
    aggregate_marker = f"-{task_selector}-attempt"
    marker_index = trial_root_name.rfind(aggregate_marker)
    if marker_index >= 0:
        tail = trial_root_name[marker_index + len(aggregate_marker) :]
        ordinal_text, separator, child_name = tail.partition("__")
        if separator and child_name:
            return _positive_attempt_ordinal(ordinal_text)

    # Long aggregate names carry the anchored runner attempt behind an exact
    # digest suffix because the complete selector marker may not fit within a
    # portable filesystem component. Harbor 0.22 native trial names use a
    # seven-character ShortUUID and cannot produce this aggregate shape.
    truncated_match = re.fullmatch(
        rf".+{re.escape(TRUNCATED_AGGREGATE_ATTEMPT_PREFIX)}0*(?P<ordinal>[1-9][0-9]*)__[0-9a-f]{{16}}",
        trial_root_name,
        flags=re.IGNORECASE,
    )
    if truncated_match:
        return _positive_attempt_ordinal(truncated_match.group("ordinal"))

    # Harbor 0.13-era local trials used ``<selector>_attemptNNN``. Matching the
    # complete trusted selector keeps compatibility without guessing from an
    # arbitrary occurrence of ``attempt`` in the selector or display name.
    legacy_marker = f"{task_selector}_attempt"
    if trial_root_name.startswith(legacy_marker):
        return _positive_attempt_ordinal(trial_root_name[len(legacy_marker) :])
    return None


def _apply_harbor_result_case_identity(
    data: dict[str, Any],
    result: dict[str, Any],
    case_id_by_task_selector: dict[str, str] | None,
) -> None:
    """Apply trusted staged identity, or retain legacy reward fallback."""
    data.pop("_attempt_ordinal", None)
    data.pop("_trusted_task_selector", None)
    task_selector = _task_selector_from_harbor_result(result)
    if task_selector:
        data["_trusted_task_selector"] = task_selector
        attempt_ordinal = _structural_attempt_ordinal(data.get("_trial_root_name"), task_selector)
        if attempt_ordinal is not None:
            data["_attempt_ordinal"] = attempt_ordinal
    entry_id = _entry_id_from_harbor_result(result, case_id_by_task_selector)
    if case_id_by_task_selector is not None:
        if entry_id:
            data["entry_id"] = entry_id
            data.pop("_trusted_case_identity_unresolved", None)
        else:
            # A grader-authored identity is not authoritative. If Harbor's
            # persisted selector is missing, unknown, or internally
            # inconsistent, make the reward diagnostic-only so coverage fails
            # instead of allowing it to impersonate an expected case.
            data.pop("entry_id", None)
            data["_trusted_case_identity_unresolved"] = True
    elif entry_id and not data.get("entry_id"):
        data["entry_id"] = entry_id


def _overall_score(reward: dict[str, Any]) -> float | None:
    if reward.get("_logical_attempt_sentinel") is _LOGICAL_ATTEMPT_SENTINEL:
        return _finite_reward_number(reward.get("_logical_overall"))
    return overall_score(reward)


def _sanitize_reward_metric_surfaces(reward: dict[str, Any]) -> dict[str, Any]:
    """Omit unsafe metric names without creating redaction aliases."""
    contract_failed = custom_metric_contract_error(reward) is not None
    sanitized = dict(reward)

    for field in ("custom_metrics", "metrics"):
        raw_metrics = reward.get(field)
        if not isinstance(raw_metrics, dict):
            continue
        sanitized_metrics: dict[str, Any] = {}
        for raw_name, value in raw_metrics.items():
            name = str(raw_name)
            if name in RESERVED_METRIC_NAMES:
                sanitized_metrics[name] = value
                continue
            candidate = value.get("score") if isinstance(value, dict) else value
            if not contract_failed and custom_metric_name_is_publishable(name) and score_value(candidate) is not None:
                sanitized_metrics[name] = value
        sanitized[field] = sanitized_metrics

    for raw_name, value in list(reward.items()):
        name = str(raw_name)
        if name in RESERVED_METRIC_NAMES or name.startswith("_"):
            continue
        if not custom_metric_name_is_publishable(name):
            sanitized.pop(raw_name, None)
            continue
        candidate = value.get("score") if isinstance(value, dict) else value
        if score_value(candidate) is None:
            continue
        if contract_failed:
            sanitized.pop(raw_name, None)

    details = reward.get("details")
    if isinstance(details, dict):
        sanitized["details"] = {
            str(raw_name): detail
            for raw_name, detail in details.items()
            if str(raw_name) in RESERVED_METRIC_NAMES or custom_metric_name_is_publishable(str(raw_name))
        }
    safe_details = sanitized.get("details")
    harbor_rewards = details.get("harbor_rewards") if isinstance(details, dict) else None
    if isinstance(harbor_rewards, dict):
        safe_harbor_rewards: dict[str, Any] = {}
        for raw_name, value in harbor_rewards.items():
            name = str(raw_name)
            if name in {"custom_metrics", "metrics"} and isinstance(value, dict):
                safe_harbor_rewards[name] = {
                    str(raw_metric): metric_value
                    for raw_metric, metric_value in value.items()
                    if str(raw_metric) in RESERVED_METRIC_NAMES
                    or (
                        not contract_failed
                        and custom_metric_name_is_publishable(str(raw_metric))
                        and score_value(metric_value.get("score") if isinstance(metric_value, dict) else metric_value)
                        is not None
                    )
                }
                continue
            if (
                name in RESERVED_METRIC_NAMES
                or name in {"reward", "metric_set", "metric_set_version"}
                or (not contract_failed and custom_metric_name_is_publishable(name))
            ):
                safe_harbor_rewards[name] = value
        safe_details = dict(safe_details) if isinstance(safe_details, dict) else {}
        safe_details["harbor_rewards"] = safe_harbor_rewards
        sanitized["details"] = safe_details

    custom_details = reward.get("custom_details")
    if isinstance(custom_details, dict):
        custom_metric_names = set(extract_custom_metrics(reward)) if not contract_failed else set()
        safe_custom_details = {
            str(raw_name): detail for raw_name, detail in custom_details.items() if str(raw_name) in custom_metric_names
        }
        if safe_custom_details:
            sanitized["custom_details"] = safe_custom_details
        else:
            sanitized.pop("custom_details", None)
    return sanitized


def _reward_publication_projection_is_safe(reward: dict[str, Any]) -> bool:
    """Check the same redacted envelope used by persisted reward artifacts."""
    clean_reward = _sanitize_reward_metric_surfaces(
        {key: value for key, value in reward.items() if not key.startswith("_")}
    )
    diagnostic_reward = (
        str(clean_reward.get("evaluation_status") or "").casefold() in {"error", "failed"}
        or overall_score(clean_reward) is None
    )
    max_str_len = REWARD_DIAGNOSTIC_STRING_MAX_CHARS if diagnostic_reward else None
    try:
        safe_reward = redact_sensitive_data(
            clean_reward,
            max_str_len=max_str_len,
        )
        _restore_custom_metric_scores(clean_reward, safe_reward, max_str_len=max_str_len)
        _restore_custom_metric_details(clean_reward, safe_reward, max_str_len=max_str_len)
        normalized, _invalid = _normalized_reward_numbers(
            safe_reward,
            _max_nodes=COLLECTED_REWARD_JSON_MAX_NODES,
        )
        _validate_generated_json_value(
            normalized,
            max_depth=REWARD_JSON_MAX_DEPTH,
            max_nodes=COLLECTED_REWARD_JSON_MAX_NODES,
            max_bytes=COLLECTED_REWARD_JSON_MAX_BYTES,
        )
    except (
        _RewardStructureLimitError,
        MemoryError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        return False
    return isinstance(normalized, dict)


def _mark_reward_collection_failure(reward: dict[str, Any], reason: str) -> None:
    reward["evaluation_status"] = "failed"
    reward["evaluation_errors"] = _merge_bounded_evaluation_errors(
        {"collector": reason},
        reward.get("evaluation_errors"),
    )


def _partition_scoreable_rewards(
    rewards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Separate complete finite rewards from diagnostic-only reward artifacts."""
    scoreable: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    failed_trials: set[str] = set()
    for reward in rewards:
        if not _reward_identity_is_publishable(reward):
            _mark_reward_collection_failure(reward, UNSAFE_REWARD_IDENTITY_REASON)
        elif custom_metric_contract_error(reward):
            _mark_reward_collection_failure(reward, UNSAFE_CUSTOM_METRICS_REASON)
        elif not _reward_publication_projection_is_safe(reward):
            _mark_reward_collection_failure(reward, UNSAFE_REWARD_STRUCTURE_REASON)
        evaluation_failed = str(reward.get("evaluation_status") or "").casefold() in {"error", "failed"}
        if not evaluation_failed and overall_score(reward) is not None:
            scoreable.append(reward)
            continue
        raw_trial = str(reward.get("_trial_name") or reward.get("_trial_root_name") or "unknown trial")
        if raw_trial in failed_trials:
            continue
        failed_trials.add(raw_trial)
        failures.append(
            {
                "trial": _published_trial_label(raw_trial, alias_ordinal=len(failures) + 1),
                "reason": _unscoreable_reward_reason(reward),
            }
        )

    custom_names = {name for reward in scoreable for name in extract_custom_metrics(reward)}
    if len(custom_names) > MAX_CUSTOM_METRICS:
        for reward in scoreable:
            _mark_reward_collection_failure(reward, UNSAFE_CUSTOM_METRIC_UNION_REASON)
            raw_trial = str(reward.get("_trial_name") or reward.get("_trial_root_name") or "unknown trial")
            if raw_trial in failed_trials:
                continue
            failed_trials.add(raw_trial)
            failures.append(
                {
                    "trial": _published_trial_label(raw_trial, alias_ordinal=len(failures) + 1),
                    "reason": UNSAFE_CUSTOM_METRIC_UNION_REASON,
                }
            )
        scoreable = []
    return scoreable, failures


def _unscoreable_reward_reason(reward: dict[str, Any]) -> str:
    """Return a bounded, redacted diagnostic for an unscoreable reward."""
    fallback = UNSCOREABLE_NUMERIC_REWARD_REASON
    if str(reward.get("evaluation_status") or "").casefold() not in {"error", "failed"}:
        return fallback

    errors = reward.get("evaluation_errors")
    parts: list[str] = []
    if isinstance(errors, dict):
        for metric, reason in list(errors.items())[: len(DEFAULT_METRICS)]:
            safe_metric = _safe_diagnostic_text(metric, max_len=64)
            safe_reason = _safe_diagnostic_text(reason, max_len=512)
            if safe_reason:
                parts.append(f"{safe_metric or 'judge'}: {safe_reason}")
    elif isinstance(errors, list):
        parts.extend(_safe_diagnostic_text(reason, max_len=512) for reason in errors[: len(DEFAULT_METRICS)])
        parts = [part for part in parts if part]
    elif errors not in (None, ""):
        safe_reason = _safe_diagnostic_text(errors, max_len=512)
        if safe_reason:
            parts.append(safe_reason)

    if not parts:
        return fallback
    return _safe_diagnostic_text("Required judge evaluation failed: " + "; ".join(parts), max_len=2048)


def _strip_attempt_suffix(value: str) -> str:
    """Remove SkillEvaluator per-attempt suffixes from a task/case identifier."""
    return re.sub(r"(?:[-_])attempt\d+$", "", value)


def _reward_identity_is_publishable(reward: dict[str, Any]) -> bool:
    """Validate the effective case identity before score aggregation."""
    if reward.get("_trusted_case_identity_unresolved") is True:
        return False
    entry_id = reward.get("entry_id")
    if entry_id not in (None, ""):
        return _identity_text_is_publishable(entry_id)
    trial_name = reward.get("_trial_name")
    if not isinstance(trial_name, str) or not trial_name:
        return False
    return _identity_text_is_publishable(trial_name.split("__", 1)[0])


def _canonical_case_id(value: str, expected_case_ids: set[str] | None = None) -> str:
    if not _identity_text_is_publishable(value):
        return ""
    if expected_case_ids and value in expected_case_ids:
        return value
    stripped = _strip_attempt_suffix(value)
    if expected_case_ids and stripped in expected_case_ids:
        return stripped
    generated_prefix_stripped = stripped.removeprefix("skillevaluator-")
    if expected_case_ids and generated_prefix_stripped in expected_case_ids:
        return generated_prefix_stripped
    return stripped


def _entry_id(reward: dict[str, Any], expected_case_ids: set[str] | None = None) -> str:
    if reward.get("_trusted_case_identity_unresolved") is True:
        return "unknown"
    if isinstance(reward.get("entry_id"), str) and reward["entry_id"]:
        return _canonical_case_id(reward["entry_id"], expected_case_ids)
    trial_name = reward.get("_trial_name")
    if trial_name:
        return _canonical_case_id(trial_name.split("__", 1)[0], expected_case_ids)
    return "unknown"


def _attempt_sort_key(reward: dict[str, Any]) -> tuple[int, int | str, str, str]:
    """Sort attempts by explicit attempt label, then Harbor start time."""
    trial_name = str(reward.get("_trial_name") or "")
    attempt_ordinal = _attempt_ordinal(reward)
    if attempt_ordinal is not None:
        return (0, attempt_ordinal, str(reward.get("_started_at") or ""), trial_name)
    started_at = str(reward.get("_started_at") or "")
    return (1 if started_at else 2, started_at, "", trial_name)


def _attempt_ordinal(reward: dict[str, Any]) -> int | None:
    """Return a carried ordinal or one unambiguous legacy suffix."""
    if (attempt_ordinal := _positive_attempt_ordinal(reward.get("_attempt_ordinal"))) is not None:
        return attempt_ordinal
    if reward.get("_trusted_task_selector") is not None:
        return None
    entry_id = reward.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        return None
    for key in ("_trial_root_name", "_trial_name"):
        trial_name = str(reward.get(key) or "")
        if trial_name == entry_id:
            continue
        match = re.fullmatch(
            r"(?P<base>.+?)(?:__|_)attempt0*(?P<ordinal>\d+)",
            trial_name,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        # Harbor 0.13 trial names embedded the logical task name immediately
        # before ``_attemptNNN`` (sometimes behind a generated job prefix).
        # Requiring that relationship preserves those artifacts without
        # interpreting an authored selector/display name such as
        # ``selector_attempt2`` as attempt two for an unrelated logical case.
        base = match.group("base")
        if base == entry_id or base.endswith((f"-{entry_id}", f"_{entry_id}")):
            return _positive_attempt_ordinal(match.group("ordinal"))
    return None


def _logical_attempt_rewards(rewards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse multi-step reward rows to one metric-bearing logical trial."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, reward in enumerate(rewards):
        root = str(reward.get("_trial_root_name") or reward.get("_trial_name") or f"__row_{index}")
        grouped.setdefault(root, []).append(reward)

    logical: list[dict[str, Any]] = []
    for root, rows in grouped.items():
        authoritative = next((row for row in rows if not row.get("_step_name")), None)
        if authoritative is not None or len(rows) == 1:
            logical.append(authoritative if authoritative is not None else rows[0])
            continue
        first = rows[0]
        standard_scores, metric_set, metrics = average_metrics(rows)
        custom_scores = _average_multistep_custom_metrics(rows)
        logical_reward: dict[str, Any] = {
            "entry_id": first.get("entry_id"),
            "_trial_name": root,
            "_trial_root_name": root,
            "_started_at": first.get("_started_at"),
            "_logical_attempt_sentinel": _LOGICAL_ATTEMPT_SENTINEL,
        }
        if first.get("_trusted_task_selector") is not None:
            logical_reward["_trusted_task_selector"] = first["_trusted_task_selector"]
        if (attempt_ordinal := _attempt_ordinal(first)) is not None:
            logical_reward["_attempt_ordinal"] = attempt_ordinal
        if metric_set:
            logical_reward["metric_set"] = metric_set
        for metric in metrics:
            if metric in standard_scores:
                logical_reward[metric] = standard_scores[metric]
        logical_reward.update(custom_scores)
        if custom_scores:
            logical_reward["custom_metrics"] = custom_scores
        if (overall := _average_overall(rows)) is not None:
            logical_reward["overall"] = overall
            logical_reward["_logical_overall"] = overall
        if any(row.get("_has_trajectory") for row in rows):
            logical_reward["_has_trajectory"] = True
        logical.append(logical_reward)
    return logical


def harbor_job_passed(job_dir: Path, pass_threshold: float) -> bool:
    """Return whether a complete logical attempt meets the pass threshold.

    This deliberately shares collection's failure filtering and multi-step
    reward precedence. A root Harbor ``result.json`` reward is authoritative;
    step rewards are averaged only when Harbor did not persist one.
    """
    job_ok, _ = validate_harbor_job_result(job_dir / "result.json")
    trial_failures = _extract_trial_failures(job_dir)
    if not job_ok and not _can_preserve_partial_rewards(job_dir, trial_failures):
        return False
    rewards, _ = _partition_scoreable_rewards(_extract_rewards(job_dir))
    return any(
        (score := _overall_score(reward)) is not None and score >= pass_threshold
        for reward in _logical_attempt_rewards(rewards)
    )


def _wilson_score_interval(successes: int, total: int) -> dict[str, Any] | None:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if total <= 0 or successes < 0 or successes > total:
        return None

    # Standard-normal 97.5th percentile for a two-sided 95% interval.
    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total) / denominator
    return {
        "method": "wilson_score",
        "confidence_level": 0.95,
        # At the mathematical endpoints, ``center +/- margin`` can leave a
        # one-ulp residual on the wrong side of the observed proportion.  Pin
        # those endpoints exactly so the reported interval contains 0 and 1.
        "lower": 0.0 if successes == 0 else max(0.0, center - margin),
        "upper": 1.0 if successes == total else min(1.0, center + margin),
    }


def _mcnemar_exact_probability(with_only_pass: int, without_only_pass: int) -> Fraction:
    """Return the exact two-sided McNemar probability as a rational number."""
    discordant = with_only_pass + without_only_pass
    if discordant == 0:
        return Fraction(1, 1)

    tail = min(with_only_pass, without_only_pass)
    # Either balanced split covers at least half of the symmetric binomial
    # distribution, so the doubled tail is exactly one after clamping.  This
    # also avoids constructing thousands of huge coefficients for an obvious
    # result such as 10,000 versus 10,000 discordant outcomes.
    if tail == discordant // 2:
        return Fraction(1, 1)

    coefficient = 1
    lower_tail = 1
    for count in range(1, tail + 1):
        coefficient = coefficient * (discordant - count + 1) // count
        lower_tail += coefficient
    return min(Fraction(1, 1), Fraction(2 * lower_tail, 1 << discordant))


def _probability_float(probability: Fraction) -> float | None:
    """Return a nonzero float approximation, or None when conversion underflows."""
    approximate = float(probability)
    return approximate if approximate > 0.0 or probability == 0 else None


def _probability_text(probability: Fraction) -> str:
    """Return a bounded nonzero decimal representation for an exact probability."""
    approximate = _probability_float(probability)
    # Positive subnormal floats have too few significant bits for the requested
    # decimal precision, so format them from the exact ratio below as well.
    if approximate is not None and (probability == 0 or approximate >= sys.float_info.min):
        return format(approximate, ".10g")

    # ``Decimal(numerator) / Decimal(denominator)`` both scales with the full
    # integer width and underflows at Decimal's default Emin.  For probabilities
    # already below binary-float range, derive a scientific mantissa from the
    # integer logarithms instead.  The work and output size are bounded by the
    # operands' bit lengths, and a nonzero Fraction can never be rendered as 0.
    log10_probability = (math.log2(probability.numerator) - math.log2(probability.denominator)) * math.log10(2.0)
    exponent = math.floor(log10_probability)
    mantissa = 10.0 ** (log10_probability - exponent)
    mantissa_text = format(mantissa, ".10g")
    if mantissa_text == "10":
        mantissa_text = "1"
        exponent += 1
    return f"{mantissa_text}e{exponent:+d}"


_MAX_EXACT_PROBABILITY_INTEGER_DIGITS = 4_300
_MAX_UNPAIRED_CASE_ID_SAMPLE = 64


def _decimal_digit_count(value: int) -> int:
    """Count base-10 digits without invoking CPython's guarded int-to-string path."""
    value = abs(value)
    if value == 0:
        return 1

    estimate = max(1, int((value.bit_length() - 1) * math.log10(2)) + 1)
    lower_bound = 10 ** (estimate - 1)
    while value < lower_bound:
        estimate -= 1
        lower_bound //= 10
    while value >= lower_bound * 10:
        estimate += 1
        lower_bound *= 10
    return estimate


def _probability_exact(probability: Fraction) -> str | None:
    """Return a bounded reduced rational, or None when decimal rendering is unsafe."""
    active_limit = sys.get_int_max_str_digits()
    render_limit = (
        min(_MAX_EXACT_PROBABILITY_INTEGER_DIGITS, active_limit)
        if active_limit > 0
        else _MAX_EXACT_PROBABILITY_INTEGER_DIGITS
    )
    if max(_decimal_digit_count(probability.numerator), _decimal_digit_count(probability.denominator)) > (render_limit):
        return None
    try:
        if probability.denominator == 1:
            return str(probability.numerator)
        return f"{probability.numerator}/{probability.denominator}"
    except ValueError:
        # The interpreter-wide limit can change between the digit check and
        # rendering. Treat that race like any other bounded omission.
        return None


def _exact_probability_fields(field: str, probability: Fraction) -> dict[str, Any]:
    """Serialize exact probability diagnostics without changing process-wide safety limits."""
    exact = _probability_exact(probability)
    fields: dict[str, Any] = {field: exact, f"{field}_omitted": exact is None}
    if exact is None:
        fields[f"{field}_omitted_reason"] = "decimal_digit_limit"
    return fields


def _mcnemar_exact_p_value(with_only_pass: int, without_only_pass: int) -> float | None:
    """Return the two-sided exact McNemar p-value, or None on float underflow."""
    return _probability_float(_mcnemar_exact_probability(with_only_pass, without_only_pass))


def _minimum_attainable_mcnemar_probability(discordant: int) -> Fraction:
    """Return the smallest two-sided exact probability attainable for this pair count."""
    if discordant <= 1:
        return Fraction(1, 1)
    return Fraction(2, 1 << discordant)


def _minimum_attainable_mcnemar_p_value(discordant: int) -> float | None:
    """Return the smallest two-sided exact p-value attainable for this pair count."""
    return _probability_float(_minimum_attainable_mcnemar_probability(discordant))


def _pass_rate_delta(with_skill: dict[str, Any], without_skill: dict[str, Any]) -> float:
    """Return the legacy delta contract derived from persisted rounded rates."""
    return round(float(with_skill.get("rate", 0.0) or 0.0) - float(without_skill.get("rate", 0.0) or 0.0), 4)


def _count_derived_pass_rate_delta(with_skill: dict[str, Any], without_skill: dict[str, Any]) -> float:
    """Return the corrected arm-level delta derived directly from pass counts."""
    with_total = int(with_skill.get("total_cases", 0) or 0)
    without_total = int(without_skill.get("total_cases", 0) or 0)
    if with_total > 0 and without_total > 0:
        with_rate = int(with_skill.get("passed_cases", 0) or 0) / with_total
        without_rate = int(without_skill.get("passed_cases", 0) or 0) / without_total
        return round(with_rate - without_rate, 4)
    return _pass_rate_delta(with_skill, without_skill)


def _paired_pass_comparison(
    with_skill: dict[str, Any],
    without_skill: dict[str, Any],
) -> dict[str, Any]:
    """Compare pass@k outcomes for matching cases across both evaluation arms."""
    with_cases = with_skill.get("_pairing_cases", with_skill.get("cases"))
    without_cases = without_skill.get("_pairing_cases", without_skill.get("cases"))
    if not isinstance(with_cases, dict) or not isinstance(without_cases, dict):
        return {"pairing_status": "unavailable", "paired_cases": 0}

    with_expected = {str(case_id): case for case_id, case in with_cases.items() if not case.get("extra_case")}
    without_expected = {str(case_id): case for case_id, case in without_cases.items() if not case.get("extra_case")}
    common_ids = sorted(set(with_expected) & set(without_expected))
    with_only_ids = sorted(set(with_expected) - set(without_expected))
    without_only_ids = sorted(set(without_expected) - set(with_expected))

    with_total = int(with_skill.get("total_cases", 0) or 0)
    without_total = int(without_skill.get("total_cases", 0) or 0)
    unidentified_with = max(0, with_total - len(with_expected))
    unidentified_without = max(0, without_total - len(without_expected))
    complete = (
        bool(common_ids)
        and not with_only_ids
        and not without_only_ids
        and unidentified_with == 0
        and unidentified_without == 0
        and len(common_ids) == with_total == without_total
    )

    outcomes = {
        "both_pass": 0,
        "with_skill_only_pass": 0,
        "without_skill_only_pass": 0,
        "neither_pass": 0,
    }
    for case_id in common_ids:
        with_passed = bool(with_expected[case_id].get("passed"))
        without_passed = bool(without_expected[case_id].get("passed"))
        if with_passed and without_passed:
            outcomes["both_pass"] += 1
        elif with_passed:
            outcomes["with_skill_only_pass"] += 1
        elif without_passed:
            outcomes["without_skill_only_pass"] += 1
        else:
            outcomes["neither_pass"] += 1

    paired_cases = len(common_ids)
    paired_delta = (
        (outcomes["with_skill_only_pass"] - outcomes["without_skill_only_pass"]) / paired_cases
        if paired_cases
        else None
    )
    result: dict[str, Any] = {
        "pairing_status": "complete" if complete else ("partial" if common_ids else "unavailable"),
        "paired_cases": paired_cases,
        "with_skill_unpaired_case_count": len(with_only_ids),
        "without_skill_unpaired_case_count": len(without_only_ids),
        "with_skill_unpaired_case_ids": with_only_ids[:_MAX_UNPAIRED_CASE_ID_SAMPLE],
        "without_skill_unpaired_case_ids": without_only_ids[:_MAX_UNPAIRED_CASE_ID_SAMPLE],
        "with_skill_unpaired_case_ids_truncated": len(with_only_ids) > _MAX_UNPAIRED_CASE_ID_SAMPLE,
        "without_skill_unpaired_case_ids_truncated": len(without_only_ids) > _MAX_UNPAIRED_CASE_ID_SAMPLE,
        "with_skill_unidentified_cases": unidentified_with,
        "without_skill_unidentified_cases": unidentified_without,
        **outcomes,
        "discordant_cases": outcomes["with_skill_only_pass"] + outcomes["without_skill_only_pass"],
        "paired_rate_delta": paired_delta,
    }
    if complete:
        discordant = result["discordant_cases"]
        exact_probability = _mcnemar_exact_probability(
            outcomes["with_skill_only_pass"],
            outcomes["without_skill_only_pass"],
        )
        minimum_attainable_probability = _minimum_attainable_mcnemar_probability(discordant)
        exact_probability_float = _probability_float(exact_probability)
        minimum_attainable_float = _probability_float(minimum_attainable_probability)
        exact_probability_text = _probability_text(exact_probability)
        minimum_attainable_text = (
            exact_probability_text
            if minimum_attainable_probability == exact_probability
            else _probability_text(minimum_attainable_probability)
        )
        result["mcnemar_exact"] = {
            "method": "two_sided_exact_binomial",
            "null_hypothesis": "equal marginal pass probabilities",
            "p_value": exact_probability_float,
            "p_value_text": exact_probability_text,
            **_exact_probability_fields("p_value_exact", exact_probability),
            "p_value_numeric_underflow": exact_probability_float is None,
            "minimum_attainable_p_value": minimum_attainable_float,
            "minimum_attainable_p_value_text": minimum_attainable_text,
            **_exact_probability_fields(
                "minimum_attainable_p_value_exact",
                minimum_attainable_probability,
            ),
            "minimum_attainable_p_value_numeric_underflow": minimum_attainable_float is None,
            "resolution_limited_at_alpha_0_05": minimum_attainable_probability > Fraction(1, 20),
        }
    return result


def _public_pass_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove collector-only pairing state from a pass summary."""
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def _pass_summary(
    rewards: list[dict[str, Any]],
    *,
    n_attempts: int,
    pass_threshold: float,
    stop_on_pass: bool = False,
    expected_cases: int | None,
    expected_case_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize pass@k using SkillEvaluator continuous reward scores."""
    expected_ids = _validated_expected_case_ids(expected_case_ids)
    expected_id_set = set(expected_ids) if expected_ids else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for reward in _logical_attempt_rewards(rewards):
        if _overall_score(reward) is None:
            continue
        grouped.setdefault(_entry_id(reward, expected_id_set), []).append(reward)

    cases: dict[str, Any] = {}
    pairing_cases: dict[str, dict[str, bool]] = {}
    passed_cases = 0
    attempts_used = 0
    extra_case_ids: list[str] = []

    case_order = expected_ids or sorted(grouped)
    if expected_ids:
        extra_case_ids = sorted(entry_id for entry_id in grouped if entry_id not in expected_id_set)
        case_order = [*case_order, *extra_case_ids]

    published_case_ids = set(case_order[:PUBLISHED_CASE_DETAILS_MAX])
    published_attempt_details = 0

    for entry_id in case_order:
        attempts = grouped.get(entry_id, [])
        attempt_rows: list[dict[str, Any]] = []
        best_score: float | None = None
        first_pass_attempt: int | None = None
        for idx, reward in enumerate(sorted(attempts, key=_attempt_sort_key), start=1):
            overall = _overall_score(reward)
            if overall is None:
                continue
            score = round(overall, 4)
            passed = score >= pass_threshold
            if passed and first_pass_attempt is None:
                first_pass_attempt = idx
            best_score = score if best_score is None else max(best_score, score)
            if (
                entry_id in published_case_ids
                and len(attempt_rows) < PUBLISHED_ATTEMPT_DETAILS_PER_CASE_MAX
                and published_attempt_details < PUBLISHED_ATTEMPT_DETAILS_MAX
            ):
                attempt_rows.append(
                    {
                        "attempt": idx,
                        "trial": _published_trial_label(reward.get("_trial_name", "")),
                        "score": score,
                        "passed": passed,
                    }
                )
                published_attempt_details += 1

        case_passed = first_pass_attempt is not None
        is_expected_case = expected_id_set is None or entry_id in expected_id_set
        if case_passed and is_expected_case:
            passed_cases += 1
        if is_expected_case:
            attempts_used += len(attempts)
        unscored = max(0, n_attempts - len(attempts))
        skipped = unscored if stop_on_pass and case_passed else 0
        missing = 0 if skipped else unscored

        pairing_cases[entry_id] = {"passed": case_passed, "extra_case": not is_expected_case}
        if entry_id in published_case_ids:
            cases[entry_id] = {
                "passed": case_passed,
                "first_pass_attempt": first_pass_attempt,
                "attempts_used": len(attempts),
                "attempts_skipped": skipped,
                "attempts_missing": missing,
                "best_score": round(best_score, 4) if best_score is not None else None,
                "attempts": attempt_rows,
                "attempt_details_total": len(attempts),
                "attempt_details_shown": len(attempt_rows),
                "attempt_details_truncated": len(attempt_rows) < len(attempts),
            }
            if not is_expected_case:
                cases[entry_id]["extra_case"] = True

    if expected_ids:
        total_cases = len(expected_ids)
    elif expected_cases is not None:
        total_cases = expected_cases
    else:
        total_cases = len(grouped)
    failed_cases = max(0, total_cases - passed_cases)
    rate = round(passed_cases / total_cases, 4) if total_cases else 0.0
    rate_interval = _wilson_score_interval(passed_cases, total_cases)

    return {
        "k": n_attempts,
        "pass_threshold": pass_threshold,
        "stop_on_pass": stop_on_pass,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "total_cases": total_cases,
        "rate": rate,
        "rate_interval": rate_interval,
        "attempts_used": attempts_used,
        "max_attempts_possible": total_cases * n_attempts,
        "avg_attempts_used": round(attempts_used / total_cases, 4) if total_cases else 0.0,
        "extra_case_count": len(extra_case_ids),
        "extra_cases": extra_case_ids[:PUBLISHED_CASE_ID_DIAGNOSTIC_SAMPLE_MAX],
        "extra_cases_truncated": len(extra_case_ids) > PUBLISHED_CASE_ID_DIAGNOSTIC_SAMPLE_MAX,
        "case_details_total": len(case_order),
        "case_details_shown": len(cases),
        "case_details_truncated": len(cases) < len(case_order),
        "case_details_limit": PUBLISHED_CASE_DETAILS_MAX,
        "cases": cases,
        "_pairing_cases": pairing_cases,
    }


def _compute_lift(
    with_scores: dict[str, float],
    without_scores: dict[str, float],
) -> dict[str, Any]:
    """Compute skill lift (with-skill minus without-skill) per metric."""
    lift: dict[str, Any] = {}
    metrics = tuple(m for m in DISPLAY_METRICS if m in with_scores and m in without_scores)
    for metric in metrics:
        w = with_scores[metric]
        wo = without_scores[metric]
        delta = round(w - wo, 4)
        lift[metric] = {
            "with_skill": w,
            "without_skill": wo,
            "delta": delta,
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
        }
    if metrics in {DISPLAY_METRICS, LEGACY_METRICS}:
        overall_with = sum(with_scores[m] for m in metrics) / len(metrics)
        overall_without = sum(without_scores[m] for m in metrics) / len(metrics)
        lift["overall"] = {
            "with_skill": round(overall_with, 4),
            "without_skill": round(overall_without, 4),
            "delta": round(overall_with - overall_without, 4),
        }
    return lift


def _average_overall(rewards: list[dict[str, Any]]) -> float | None:
    """Average the pass/lift overall score across reward payloads."""
    values = [_overall_score(reward) for reward in rewards]
    if not values or any(value is None for value in values):
        return None
    return round(sum(value for value in values if value is not None) / len(values), 4)


def _compute_custom_lift(
    with_custom_scores: dict[str, float],
    without_custom_scores: dict[str, float],
    with_rewards: list[dict[str, Any]],
    without_rewards: list[dict[str, Any]],
    *,
    include_overall: bool = False,
) -> dict[str, Any]:
    """Compute lift for user-owned custom metrics."""
    lift: dict[str, Any] = {}

    if include_overall:
        w = _average_overall(with_rewards)
        wo = _average_overall(without_rewards)
        if w is not None and wo is not None:
            delta = round(w - wo, 4)
            lift["overall"] = {
                "with_skill": w,
                "without_skill": wo,
                "delta": delta,
                "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
            }

    for metric in sorted(set(with_custom_scores) & set(without_custom_scores)):
        w = with_custom_scores[metric]
        wo = without_custom_scores[metric]
        delta = round(w - wo, 4)
        lift[metric] = {
            "with_skill": w,
            "without_skill": wo,
            "delta": delta,
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
        }

    return lift


def _security_score_findings(reward: dict[str, Any]) -> list[dict[str, Any]]:
    details = reward.get("details", {})
    security = details.get("security", {}) if isinstance(details, dict) else {}
    findings = security.get("findings", []) if isinstance(security, dict) else []
    return [f for f in findings if isinstance(f, dict) and f.get("score_impact")]


def _security_finding_signature(finding: dict[str, Any]) -> tuple[str, str]:
    text = str(finding.get("evidence") or finding.get("message") or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return str(finding.get("type") or "unknown"), text[:160]


def _safe_trial_path_component(value: Any) -> str:
    """Return a portable single path component or an empty string."""
    raw_component = str(value or "")
    component = raw_component.strip()
    invalid_characters = '<>:"/\\|?*\x00'
    stem = component.split(".", 1)[0].rstrip(" .").casefold()
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
        *(f"com{index}" for index in "¹²³"),
        *(f"lpt{index}" for index in "¹²³"),
    }
    try:
        utf8_bytes = len(component.encode("utf-8"))
        utf16_units = len(component.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        return ""
    if (
        not component
        or component != raw_component
        or component in {".", ".."}
        or utf8_bytes > PORTABLE_TRIAL_COMPONENT_MAX_UNITS
        or utf16_units > PORTABLE_TRIAL_COMPONENT_MAX_UNITS
        or any(
            ord(character) < 32 or ord(character) == 127 or character in invalid_characters for character in component
        )
        or component.endswith(".")
        or stem in reserved
        or contains_credential_value(component)
    ):
        return ""
    return component


def _safe_trial_source_component(value: Any) -> str:
    """Return an exact safe child name without applying output normalization."""
    if not isinstance(value, str):
        return ""
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return ""
    if os.name == "nt":
        windows_path = PureWindowsPath(value)
        if windows_path.drive or windows_path.is_absolute() or len(windows_path.parts) != 1:
            return ""
    return value


def _persisted_trial_name(reward: dict[str, Any]) -> tuple[str, str]:
    """Derive output and source names only from physical Harbor path components."""
    trial_root_name = _safe_trial_source_component(reward.get("_trial_root_name")) or "unknown"
    output_root_name = _safe_trial_path_component(trial_root_name) or "unknown"
    step_name = _safe_trial_path_component(reward.get("_step_name"))
    return (f"{output_root_name}__{step_name}" if step_name else output_root_name), trial_root_name


def _portable_trial_name_key(value: str) -> str:
    """Normalize one output name for case-insensitive and Win32-compatible collision checks."""
    return unicodedata.normalize("NFC", value.rstrip(" .").casefold())


def _persisted_trial_names(
    rewards: list[dict[str, Any]],
    job_dir: Path | None,
) -> list[tuple[str, str]]:
    """Resolve distinct physical reward identities to distinct output directories."""
    scored_names, _unscored_names = _persisted_trial_layout(rewards, job_dir)
    return scored_names


def _persisted_trial_layout(
    rewards: list[dict[str, Any]],
    job_dir: Path | None,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Allocate portable output names across scored and unscored physical trials."""
    entries = [
        (
            (
                str(reward.get("_trial_root_name") or ""),
                str(reward.get("_step_name") or ""),
            ),
            *_persisted_trial_name(reward),
        )
        for reward in rewards
    ]
    scored_roots = {trial_root_name for _identity, _legacy_name, trial_root_name in entries}
    unscored_sources: list[str] = []
    if job_dir is not None:
        with contextlib.suppress(OSError):
            for child in sorted(job_dir.iterdir()):
                kind, _unsafe_reason = _inspect_trial_directory(child)
                if child.name not in scored_roots and (
                    kind == "link" or (kind == "directory" and _looks_like_trial_dir(child))
                ):
                    unscored_sources.append(child.name)

    preferred_names = [legacy_name for _identity, legacy_name, _trial_root_name in entries]
    unsafe_preferred_indices = {
        index
        for index, (identity, legacy_name, _trial_root_name) in enumerate(entries)
        if not _safe_trial_path_component(identity[0])
        or (identity[1] and not _safe_trial_path_component(identity[1]))
        or not _safe_trial_path_component(legacy_name)
    }
    for source_name in unscored_sources:
        preferred = _safe_trial_path_component(source_name)
        if not preferred:
            unsafe_preferred_indices.add(len(preferred_names))
        preferred_names.append(preferred or "unknown")
    indices_by_name: dict[str, list[int]] = {}
    for index, preferred_name in enumerate(preferred_names):
        indices_by_name.setdefault(_portable_trial_name_key(preferred_name), []).append(index)

    conflicting_indices = {
        index for indices in indices_by_name.values() if len(indices) > 1 for index in indices
    } | unsafe_preferred_indices
    resolved_names = list(preferred_names)
    used_name_keys = set(indices_by_name)
    suffix = 1
    for index in range(len(preferred_names)):
        if index not in conflicting_indices:
            continue
        while True:
            resolved_name = f"skillevaluator-trial-collision-{suffix:06d}"
            suffix += 1
            resolved_key = _portable_trial_name_key(resolved_name)
            if resolved_key not in used_name_keys:
                break
        resolved_names[index] = resolved_name
        used_name_keys.add(resolved_key)

    scored_names = [
        (resolved_names[index], trial_root_name)
        for index, (_identity, _legacy_name, trial_root_name) in enumerate(entries)
    ]
    unscored_names = {
        source_name: resolved_names[len(entries) + index] for index, source_name in enumerate(unscored_sources)
    }
    return scored_names, unscored_names


def _annotate_security_attribution(
    with_rewards: list[dict[str, Any]],
    without_rewards: list[dict[str, Any]],
    *,
    baseline_run: bool = True,
) -> dict[str, Any]:
    """Annotate with-skill security findings with baseline-aware attribution."""
    baseline_by_case: dict[str, list[dict[str, Any]]] = {}
    for reward in without_rewards:
        baseline_by_case.setdefault(_entry_id(reward), []).extend(_security_score_findings(reward))

    summary = {
        "likely_skill_related": 0,
        "likely_baseline_prompt_or_environment": 0,
        "skill_may_have_improved_safety": 0,
        "ambiguous_with_skill_only": 0,
        "unknown_no_baseline": 0,
        "cases": {},
    }
    seen_cases: set[str] = set()

    for reward in with_rewards:
        entry_id = _entry_id(reward)
        details = reward.get("details")
        if not isinstance(details, dict):
            continue
        security = details.get("security")
        if not isinstance(security, dict):
            continue

        with_findings = _security_score_findings(reward)
        baseline_findings = baseline_by_case.get(entry_id, [])
        baseline_signatures = {_security_finding_signature(f) for f in baseline_findings}

        case_status = "safe"
        if with_findings:
            case_status = "with_skill_unsafe"
            attribution_plan: list[tuple[str, str]] = []
            for finding in with_findings:
                signature = _security_finding_signature(finding)
                if not baseline_run:
                    attribution = "unknown_no_baseline"
                    explanation = (
                        "No without-skill baseline was run, so SkillEvaluator cannot tell whether this "
                        "unsafe behavior is skill-related or natural agent behavior."
                    )
                    summary["unknown_no_baseline"] += 1
                elif signature in baseline_signatures:
                    attribution = "likely_baseline_prompt_or_environment"
                    explanation = (
                        "Unsafe behavior also appeared in the without-skill baseline for this case, "
                        "so this is less likely to be caused solely by the target skill."
                    )
                    summary["likely_baseline_prompt_or_environment"] += 1
                elif finding.get("target_skill_used_before"):
                    attribution = "likely_skill_related"
                    explanation = (
                        "Unsafe behavior appeared only in the with-skill run and the target skill "
                        "was used before the unsafe action."
                    )
                    summary["likely_skill_related"] += 1
                else:
                    attribution = "ambiguous_with_skill_only"
                    explanation = (
                        "Unsafe behavior appeared only in the with-skill run, but the trajectory did "
                        "not show target-skill use before the unsafe action."
                    )
                    summary["ambiguous_with_skill_only"] += 1
                attribution_plan.append((attribution, explanation))

            first_attribution, first_explanation = attribution_plan[0]
            projected_details: dict[str, Any] | None = None
            for projection in ("full", "labels", "aggregate"):
                try:
                    candidate_reward = copy.deepcopy(reward)
                    candidate_details = candidate_reward.get("details")
                    candidate_security = (
                        candidate_details.get("security") if isinstance(candidate_details, dict) else None
                    )
                    if not isinstance(candidate_security, dict):
                        raise ValueError("security detail projection disappeared")
                    candidate_findings = _security_score_findings(candidate_reward)
                    if projection != "aggregate":
                        if len(candidate_findings) != len(attribution_plan):
                            raise ValueError("security finding projection changed cardinality")
                        for candidate_finding, (attribution, explanation) in zip(
                            candidate_findings,
                            attribution_plan,
                            strict=True,
                        ):
                            candidate_finding["attribution"] = attribution
                            if projection == "full":
                                candidate_finding["attribution_explanation"] = explanation
                    candidate_security["attribution"] = first_attribution
                    candidate_security["attribution_explanation"] = first_explanation
                    if projection == "labels":
                        candidate_security["attribution_completeness"] = (
                            "Per-finding attribution labels were retained; repeated explanations were omitted "
                            "to stay within artifact limits."
                        )
                    elif projection == "aggregate":
                        candidate_security["attribution_completeness"] = (
                            "Per-finding attribution was omitted because the expanded reward would exceed "
                            "artifact limits."
                        )
                    if not _reward_publication_projection_is_safe(candidate_reward):
                        raise ValueError("security attribution projection exceeds publication limits")
                except (MemoryError, RecursionError, TypeError, ValueError):
                    continue
                projected_details = candidate_details
                break
            if projected_details is not None:
                reward["details"] = projected_details
        elif baseline_findings:
            case_status = "baseline_unsafe_with_skill_safe"
            explanation = (
                "The without-skill baseline showed unsafe behavior for this case, while the with-skill run did not."
            )
            for include_derived_finding in (True, False):
                try:
                    candidate_reward = copy.deepcopy(reward)
                    candidate_details = candidate_reward.get("details")
                    candidate_security = (
                        candidate_details.get("security") if isinstance(candidate_details, dict) else None
                    )
                    if not isinstance(candidate_security, dict):
                        raise ValueError("security detail projection disappeared")
                    if include_derived_finding:
                        candidate_findings = candidate_security.get("findings")
                        if not isinstance(candidate_findings, list):
                            candidate_findings = []
                            candidate_security["findings"] = candidate_findings
                        candidate_findings.append(
                            {
                                "type": "skill_reduced_unsafe_behavior",
                                "severity": "info",
                                "message": "Baseline had unsafe agent action, but with-skill run did not",
                                "evidence": (
                                    f"Without-skill baseline contained {len(baseline_findings)} "
                                    "score-impacting security finding(s)."
                                ),
                                "source": "baseline_comparison",
                                "score_impact": False,
                                "attribution": "skill_may_have_improved_safety",
                                "attribution_explanation": explanation,
                            }
                        )
                    else:
                        candidate_security["attribution_completeness"] = (
                            "The derived comparison finding was omitted to stay within artifact limits."
                        )
                    candidate_security["attribution"] = "skill_may_have_improved_safety"
                    candidate_security["attribution_explanation"] = explanation
                    if not _reward_publication_projection_is_safe(candidate_reward):
                        raise ValueError("security improvement projection exceeds publication limits")
                except (MemoryError, RecursionError, TypeError, ValueError):
                    continue
                reward["details"] = candidate_details
                break
            summary["skill_may_have_improved_safety"] += 1

        seen_cases.add(entry_id)
        if entry_id in summary["cases"] or len(summary["cases"]) < PUBLISHED_CASE_DETAILS_MAX:
            summary["cases"][entry_id] = {
                "status": case_status,
                "with_skill_findings": len(with_findings),
                "baseline_findings": len(baseline_findings),
            }

    summary["case_details_total"] = len(seen_cases)
    summary["case_details_shown"] = len(summary["cases"])
    summary["case_details_truncated"] = len(summary["cases"]) < len(seen_cases)
    summary["case_details_limit"] = PUBLISHED_CASE_DETAILS_MAX
    return summary


def _is_standard_skill_execution_reward(reward: dict[str, Any]) -> bool:
    """Return whether the reward carries a usable canonical standard score."""
    standard_metric_sets = {DEFAULT_METRIC_SET, LEGACY_METRIC_SET}
    declared_metric_set = reward.get("metric_set") or reward.get("metric_set_version")
    if declared_metric_set and str(declared_metric_set) not in standard_metric_sets:
        return False
    metric_set, metrics = metric_set_for_reward(reward)
    return (
        metric_set in standard_metric_sets
        and "skill_execution" in metrics
        and metric_value(reward, "skill_execution") is not None
    )


def _trajectory_skill_invoked(trajectory: Any, skill_name: str, *, depth: int = 0) -> bool | None:
    """Derive target invocation from executed root and referenced subagent paths."""
    if not isinstance(trajectory, dict):
        return None
    if depth > _MAX_TRAJECTORY_REFERENCE_DEPTH:
        return None
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps or any(not isinstance(step, dict) for step in steps):
        return None
    agent_steps = [
        step for step in steps if step.get("source") == "agent" and step.get("is_copied_context") is not True
    ]
    results: list[bool | None] = []
    if agent_steps:
        for step in agent_steps:
            tool_calls = step.get("tool_calls")
            if tool_calls is None:
                # ATIF 1.7 makes tool_calls optional. A normal agent message
                # without calls contributes no routing evidence, but it must
                # not hide a later, well-formed invocation.
                continue
            if not isinstance(tool_calls, list):
                results.append(None)
                break
            if any(
                not isinstance(tool_call, dict)
                or not isinstance(tool_call.get("function_name"), str)
                or not tool_call["function_name"].strip()
                or not isinstance(tool_call.get("arguments"), dict)
                for tool_call in tool_calls
            ):
                results.append(None)
                break
        else:
            try:
                agent_trajectory = {**trajectory, "steps": agent_steps}
                tool_calls = extract_tool_calls_as_dicts(agent_trajectory)
                skill_tool_names = get_skill_tool_calls(agent_trajectory)
                negative_check = check_negative_case(tool_calls, skill_name, skill_tool_names=skill_tool_names)
            except (AttributeError, TypeError, ValueError):
                results.append(None)
            else:
                passed = negative_check.get("passed")
                results.append(not passed if isinstance(passed, bool) else None)
    else:
        results.append(None)

    embedded = trajectory.get("subagent_trajectories")
    if embedded is None:
        embedded_by_id: dict[str, dict[str, Any]] = {}
    elif not isinstance(embedded, list):
        return True if True in results else None
    else:
        embedded_by_id = {}
        for child in embedded:
            if not isinstance(child, dict):
                return True if True in results else None
            trajectory_id = child.get("trajectory_id")
            if not isinstance(trajectory_id, str) or not trajectory_id or trajectory_id in embedded_by_id:
                return True if True in results else None
            embedded_by_id[trajectory_id] = child

    referenced_ids: list[str] = []
    seen_ids: set[str] = set()
    for step in steps:
        if step.get("is_copied_context") is True:
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        observation_results = observation.get("results")
        if not isinstance(observation_results, list):
            return True if True in results else None
        for observation_result in observation_results:
            if not isinstance(observation_result, dict):
                return True if True in results else None
            refs = observation_result.get("subagent_trajectory_ref")
            if refs is None:
                continue
            if not isinstance(refs, list):
                return True if True in results else None
            for ref in refs:
                if not isinstance(ref, dict):
                    return True if True in results else None
                trajectory_id = ref.get("trajectory_id")
                if not isinstance(trajectory_id, str) or trajectory_id not in embedded_by_id:
                    results.append(None)
                    continue
                if trajectory_id not in seen_ids:
                    seen_ids.add(trajectory_id)
                    referenced_ids.append(trajectory_id)
    results.extend(
        _trajectory_skill_invoked(embedded_by_id[trajectory_id], skill_name, depth=depth + 1)
        for trajectory_id in referenced_ids
    )
    if True in results:
        return True
    if None in results:
        return None
    return False


def _authoritative_step_names(trial_root: Path) -> tuple[bool, list[str] | None]:
    """Return ordered authoritative Harbor step names, if this is multi-step."""
    result_path = trial_root / "result.json"
    steps_path = trial_root / "steps"
    try:
        result_path.lstat()
        result_artifact_present = True
    except FileNotFoundError:
        result_artifact_present = False
    except OSError:
        return True, None
    try:
        steps_path.lstat()
        steps_layout_present = True
    except FileNotFoundError:
        steps_layout_present = False
    except OSError:
        return True, None

    result = _read_json(result_path)
    if not isinstance(result, dict):
        return (True, None) if result_artifact_present or steps_layout_present else (False, None)
    if "step_results" not in result:
        if steps_layout_present:
            return True, None
        return False, None
    step_results = result.get("step_results")
    if step_results is None:
        # Single-step Harbor trials serialize an explicit null. A physical
        # steps layout still makes that shape ambiguous, so keep it fail-closed.
        return (True, None) if steps_layout_present else (False, None)
    names = _valid_step_result_names(
        step_results,
        max_count=_MAX_TRAJECTORY_STEP_DIRECTORIES,
    )
    return True, names


def _materialized_trial_trajectory(
    trial_root: Path,
    step_name: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Select and materialize the trajectory authorized by Harbor topology."""
    is_multi_step, authoritative_names = _authoritative_step_names(trial_root)
    root_path = trial_root / "agent" / "trajectory.json"
    try:
        root_path.lstat()
    except FileNotFoundError:
        root_present = False
    except OSError:
        return None, "trajectory_stat_failed"
    else:
        root_present = True

    if is_multi_step:
        if authoritative_names is None:
            return None, "invalid_or_incomplete_multi_step_topology"
        if root_present:
            return None, "contradictory_root_and_multi_step_trajectories"
        paths = _ordered_step_trajectory_paths(trial_root)
        discovered_names = [path.parent.parent.name for path in paths]
        if discovered_names != authoritative_names:
            return None, "incomplete_or_unexpected_multi_step_trajectories"
        if step_name:
            if step_name not in authoritative_names:
                return None, "reward_step_not_in_authoritative_topology"
            selected = paths[authoritative_names.index(step_name)]
            try:
                trajectory, _reference_key = _materialize_trajectory_file(selected.parent, selected.name)
            except (OSError, SecurePathError, _TrajectoryMergeError, RecursionError):
                return None, "invalid_step_trajectory"
            return trajectory, None
        merged = _merged_step_trajectory(trial_root)
        if merged is None:
            return None, "incomplete_or_invalid_multi_step_trajectory"
        return merged, None

    if step_name:
        return None, "unexpected_step_reward_for_single_step_trial"
    if not root_present:
        return None, "missing_single_step_trajectory"
    try:
        trajectory, _reference_key = _materialize_trajectory_file(root_path.parent, root_path.name)
    except (OSError, SecurePathError, _TrajectoryMergeError, RecursionError):
        return None, "invalid_single_step_trajectory"
    return trajectory, None


def _trusted_trial_skill_invoked(
    trial_root: Path,
    step_name: str | None,
    skill_name: str,
) -> bool | None:
    """Derive trusted invocation from the topology-authorized materialized ATIF."""
    trajectory, _reason = _materialized_trial_trajectory(trial_root, step_name)
    return _trajectory_skill_invoked(trajectory, skill_name)


def _add_trusted_invocation_evidence(
    clean_reward: dict[str, Any],
    source_reward: dict[str, Any],
    trial_root: Path | None,
    skill_name: str,
    *,
    trajectory: dict[str, Any] | None | object = _TRAJECTORY_NOT_PROVIDED,
) -> None:
    """Replace verifier-authored routing evidence on standard rewards only."""
    if not _is_standard_skill_execution_reward(source_reward):
        return
    for key in ("skill_invoked", "routing_passed", "invocation_evidence_source"):
        clean_reward.pop(key, None)
    if trial_root is None:
        return
    invoked = (
        _trajectory_skill_invoked(trajectory, skill_name)
        if trajectory is not _TRAJECTORY_NOT_PROVIDED
        else _trusted_trial_skill_invoked(trial_root, source_reward.get("_step_name"), skill_name)
    )
    if invoked is None:
        return
    if invoked is True:
        clean_reward["skill_invoked"] = True
    elif invoked is False:
        clean_reward["skill_invoked"] = False
    else:
        return
    clean_reward["invocation_evidence_source"] = "trajectory"


def _can_restore_custom_metric_name(value: str) -> bool:
    """Allow safe names plus narrow, explicitly metric-shaped secret terms."""
    return custom_metric_name_is_publishable(value)


def _restore_custom_metric_scores(
    source_reward: dict[str, Any],
    safe_reward: dict[str, Any],
    *,
    max_str_len: int | None = None,
) -> None:
    """Restore only finite numeric values recognized by the custom-metric schema."""
    for field in ("custom_metrics", "metrics"):
        source_metrics = source_reward.get(field)
        safe_metrics = safe_reward.get(field)
        if not isinstance(source_metrics, dict) or not isinstance(safe_metrics, dict):
            continue
        for raw_name, raw_value in source_metrics.items():
            name = str(raw_name)
            if name not in safe_metrics or not _can_restore_custom_metric_name(name):
                continue
            score = extract_custom_metrics({field: {name: raw_value}}).get(name)
            if score is None:
                continue
            if isinstance(raw_value, dict):
                redacted_value = redact_sensitive_data(raw_value, max_str_len=max_str_len)
                safe_value = redacted_value if isinstance(redacted_value, dict) else {}
                safe_metrics[name] = safe_value
                safe_value["score"] = score
            else:
                safe_metrics[name] = score

    for name, score in extract_custom_metrics(source_reward).items():
        if name not in source_reward or name not in safe_reward:
            continue
        raw_value = source_reward[name]
        if isinstance(raw_value, dict):
            redacted_value = redact_sensitive_data(raw_value, max_str_len=max_str_len)
            safe_value = redacted_value if isinstance(redacted_value, dict) else {}
            safe_value["score"] = score
            safe_reward[name] = safe_value
        else:
            safe_reward[name] = score


def _restore_custom_metric_details(
    source_reward: dict[str, Any],
    safe_reward: dict[str, Any],
    *,
    max_str_len: int | None = None,
) -> None:
    """Preserve safe metric-keyed evidence while redacting nested secrets."""
    custom_names = set(extract_custom_metrics(source_reward))
    source_custom_details = source_reward.get("custom_details")
    if isinstance(source_custom_details, dict):
        safe_custom_details = {
            str(raw_name): redact_sensitive_data(detail, max_str_len=max_str_len)
            for raw_name, detail in source_custom_details.items()
            if str(raw_name) in custom_names
        }
        if safe_custom_details:
            safe_reward["custom_details"] = safe_custom_details
        else:
            safe_reward.pop("custom_details", None)

    source_details = source_reward.get("details")
    safe_details = safe_reward.get("details")
    if not isinstance(source_details, dict) or not isinstance(safe_details, dict):
        return
    for raw_name, detail in source_details.items():
        name = str(raw_name)
        if name in custom_names:
            safe_details[name] = redact_sensitive_data(detail, max_str_len=max_str_len)


def _strict_json_numbers(value: Any, *, max_nodes: int = COLLECTED_REWARD_JSON_MAX_NODES) -> Any:
    """Replace non-finite floats before strict generated-artifact serialization."""
    try:
        normalized, _invalid = _normalized_reward_numbers(value, _max_nodes=max_nodes)
    except (_RewardStructureLimitError, RecursionError, MemoryError):
        if isinstance(value, dict):
            return _structural_limit_reward(value)
        return None
    return normalized


def _save_trials(
    rewards: list[dict[str, Any]],
    trials_dir: Path,
    job_dir: Path | None,
    *,
    skill_name: str,
    agent: str,
    variant: str,
    agent_model: str | None = None,
    agent_model_source: str | None = None,
) -> None:
    """Save per-trial reward.json and trajectory.json into the results directory."""
    agent = _bounded_reward_metadata_text(agent) or "unknown"
    agent_model = _bounded_reward_metadata_text(agent_model)
    agent_model_source = _bounded_reward_metadata_text(agent_model_source)
    trials_dir.mkdir(parents=True, exist_ok=True)
    persisted_names, unscored_names = _persisted_trial_layout(rewards, job_dir)
    for reward, (trial_name, trial_root_name) in zip(rewards, persisted_names, strict=True):
        trial_out = trials_dir / trial_name
        trial_out.mkdir(parents=True, exist_ok=True)
        trial_src = job_dir / trial_root_name if job_dir else None
        materialized_traj: dict[str, Any] | None = None
        trajectory_reason: str | None = None
        if trial_src:
            materialized_traj, trajectory_reason = _materialized_trial_trajectory(
                trial_src,
                reward.get("_step_name"),
            )
        safe_materialized_traj = _redacted_trajectory_data(materialized_traj) if materialized_traj else None
        if materialized_traj is not None and safe_materialized_traj is None:
            trajectory_reason = "trajectory_redaction_or_validation_failed"
        if safe_materialized_traj is None:
            reward.setdefault(
                "_trajectory_summary",
                {"readable": False, "reason": trajectory_reason or "trajectory_unavailable"},
            )
        elif "_trajectory_summary" not in reward:
            reward["_trajectory_summary"] = _summarize_trajectory(materialized_traj)

        clean_reward = {k: v for k, v in reward.items() if not k.startswith("_")}
        if safe_materialized_traj is None and trajectory_reason not in {
            None,
            "missing_single_step_trajectory",
        }:
            warning = "Trajectory artifact omitted because it exceeded safety or validation limits."
            warnings = clean_reward.get("warnings")
            if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
                warnings = []
                clean_reward["warnings"] = warnings
            if warning not in warnings:
                warnings.append(warning)
        _add_trusted_invocation_evidence(
            clean_reward,
            reward,
            trial_src,
            skill_name,
            trajectory=materialized_traj if safe_materialized_traj is not None else None,
        )
        # Persist the physical attempt identity so bounded report readers can keep
        # fallback multi-step rows for diagnostics without weighting a logical
        # Harbor trial once per step.  This also populates the canonical report's
        # existing trial_id field instead of inventing a second report schema.
        clean_reward["trial_id"] = _published_trial_label(trial_root_name)
        if not clean_reward.get("entry_id"):
            clean_reward["entry_id"] = _entry_id(reward)
        clean_reward["agent"] = agent
        if agent_model:
            clean_reward["model"] = agent_model
        if agent_model_source:
            clean_reward["model_source"] = agent_model_source
        if "evaluation_errors" in clean_reward:
            clean_reward["evaluation_errors"] = _safe_evaluation_errors(clean_reward["evaluation_errors"])
        clean_reward = _sanitize_reward_metric_surfaces(clean_reward)
        diagnostic_reward = (
            str(clean_reward.get("evaluation_status") or "").casefold() in {"error", "failed"}
            or overall_score(clean_reward) is None
        )
        max_str_len = REWARD_DIAGNOSTIC_STRING_MAX_CHARS if diagnostic_reward else None
        safe_reward = redact_sensitive_data(
            clean_reward,
            max_str_len=max_str_len,
        )
        _restore_custom_metric_scores(clean_reward, safe_reward, max_str_len=max_str_len)
        _restore_custom_metric_details(clean_reward, safe_reward, max_str_len=max_str_len)
        safe_reward = _strict_json_numbers(safe_reward, max_nodes=REWARD_JSON_MAX_NODES)
        if isinstance(safe_reward, dict):
            safe_reward = _fail_closed_invalid_reward_numbers(
                safe_reward,
                max_nodes=REWARD_JSON_MAX_NODES,
                max_bytes=GENERATED_JSON_MAX_BYTES,
            )
        (trial_out / "reward.json").write_text(
            json.dumps(
                safe_reward,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        if trial_src:
            _copy_trial_artifacts(trial_src, trial_out, include_root_trajectory=False)
        if safe_materialized_traj:
            (trial_out / "trajectory.json").write_text(
                json.dumps(
                    safe_materialized_traj,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        elif trial_src:
            _record_skipped_trajectory(trial_out, trajectory_reason or "trajectory_unavailable")

    _save_unscored_trials(
        rewards,
        trials_dir,
        job_dir,
        agent=agent,
        variant=variant,
        agent_model=agent_model,
        agent_model_source=agent_model_source,
        persisted_names=unscored_names,
    )


def _summarize_trajectory_file(path: Path) -> dict[str, Any]:
    """Return safe trajectory metadata without raw prompts, outputs, or arguments."""
    data = _read_json(path)
    if not isinstance(data, dict):
        return {"readable": False}

    return _summarize_trajectory(data)


def _summarize_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    """Return safe trajectory metadata without raw prompts, outputs, or arguments."""
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        return {"readable": True, "steps": 0, "tool_calls": 0, "tool_names": []}

    tool_names: list[str] = []
    tool_calls = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        calls = step.get("tool_calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            tool_calls += 1
            name = call.get("function_name") or call.get("name") or call.get("tool_name")
            if name:
                tool_names.append(str(name))

    unique_tool_names = sorted(dict.fromkeys(tool_names))
    return {
        "readable": True,
        "steps": len(steps),
        "tool_calls": tool_calls,
        "unique_tools": len(unique_tool_names),
        "tool_names": unique_tool_names[:20],
    }


def _condition_execution_summary(
    rewards: list[dict[str, Any]],
    *,
    expected_case_ids: list[str] | None,
    expected_cases: int | None,
    n_attempts: int,
    job_failure: str,
    runtime_failures: list[dict[str, str]] | None = None,
    reward_failures: list[dict[str, str]] | None = None,
    skipped: bool = False,
    stop_on_pass: bool = False,
    pass_threshold: float = 0.50,
) -> dict[str, Any]:
    """Describe whether a Harbor condition produced complete logical attempts.

    Native multi-step tasks may emit several reward rows for one Harbor trial.
    ``_trial_root_name`` is therefore the attempt identity; the case id alone
    is not sufficient and raw reward-row count would over-count those tasks.
    Early-stopped cases require attempts only through their first passing trial.
    """
    expected_ids = _validated_expected_case_ids(expected_case_ids)
    expected_count = len(expected_ids) if expected_ids else int(expected_cases or 0)
    if skipped:
        return {
            "execution_status": "skipped",
            "execution_errors": [],
            "execution_error_details_total": 0,
            "execution_error_details_shown": 0,
            "execution_error_details_truncated": False,
            "expected_attempts": 0,
            "scored_attempts": 0,
            **_failure_list_metadata("runtime_failure_details", runtime_failures),
            **_failure_list_metadata("reward_failure_details", reward_failures),
        }

    errors: list[str] = [job_failure] if job_failure else []
    public_runtime_failures = _public_failure_list(runtime_failures)
    public_reward_failures = _public_failure_list(reward_failures)
    errors.extend(
        "Agent runtime failed in "
        f"{_published_trial_label(failure.get('trial', 'unknown trial'))}: "
        f"{_safe_diagnostic_text(failure.get('reason', 'unknown error'), max_len=2048)}"
        for failure in public_runtime_failures
    )
    errors.extend(
        "Unscoreable reward in "
        f"{_safe_diagnostic_text(failure.get('trial', 'unknown trial'), max_len=256)}: "
        f"{_safe_diagnostic_text(failure.get('reason', 'unknown error'), max_len=2048)}"
        for failure in public_reward_failures
    )
    if len(runtime_failures or []) > len(public_runtime_failures):
        errors.append(
            "Agent runtime failure details were truncated "
            f"(showing {len(public_runtime_failures)} of {len(runtime_failures or [])})"
        )
    if len(reward_failures or []) > len(public_reward_failures):
        errors.append(
            "Unscoreable reward details were truncated "
            f"(showing {len(public_reward_failures)} of {len(reward_failures or [])})"
        )
    expected_set = set(expected_ids)
    logical_passed: dict[str, bool] = {}
    for reward in _logical_attempt_rewards(rewards):
        score = _overall_score(reward)
        if score is not None:
            logical_passed[str(reward.get("_trial_root_name") or reward.get("_trial_name") or "")] = (
                score >= pass_threshold
            )
    roots: dict[str, dict[str, Any]] = {}
    for reward in rewards:
        root = str(reward.get("_trial_root_name") or "").strip()
        published_root = _published_trial_label(root)
        case_id = _entry_id(reward, expected_set or None)
        step_name = str(reward.get("_step_name") or "").strip()
        if not root:
            errors.append("A scored reward is missing its Harbor trial root name")
            continue
        if not case_id or case_id == "unknown":
            errors.append(f"Scored trial {published_root!r} has no case identifier")
            continue
        score = overall_score(reward)
        if score is None:
            errors.append(f"Scored trial {published_root!r} has incomplete or non-finite reward metrics")
            continue
        existing = roots.get(root)
        if existing is None:
            roots[root] = {
                "case_id": case_id,
                "steps": {step_name} if step_name else set(),
                "reward": reward,
                "passed": logical_passed.get(root, score >= pass_threshold),
                "attempt_ordinal": _attempt_ordinal(reward),
            }
            continue
        if existing["case_id"] != case_id:
            errors.append(f"Harbor trial {published_root!r} maps to multiple cases")
        elif not step_name or step_name in existing["steps"]:
            errors.append(f"Harbor trial {published_root!r} has duplicate reward rows")
        else:
            existing["steps"].add(step_name)

    by_case: dict[str, list[dict[str, Any]]] = {}
    for root_data in roots.values():
        by_case.setdefault(str(root_data["case_id"]), []).append(root_data)
    for attempts in by_case.values():
        attempts.sort(key=lambda item: _attempt_sort_key(item["reward"]))

    def _case_attempt_coverage(case_id: str, attempts: list[dict[str, Any]]) -> tuple[int, bool, bool]:
        explicit = [item["attempt_ordinal"] for item in attempts if item["attempt_ordinal"] is not None]
        all_explicit = len(explicit) == len(attempts) and bool(attempts)
        if explicit and len(explicit) != len(attempts):
            errors.append(f"Scored case {case_id!r} mixes explicit and implicit attempt labels")
        if len(explicit) != len(set(explicit)):
            errors.append(f"Scored case {case_id!r} has duplicate attempt ordinals")

        required = n_attempts
        if stop_on_pass:
            first_pass = next(
                ((index, item) for index, item in enumerate(attempts, start=1) if item["passed"]),
                None,
            )
            if first_pass is not None:
                observed_index, passed_attempt = first_pass
                required = int(passed_attempt["attempt_ordinal"] or observed_index)
        if required > n_attempts:
            errors.append(f"Scored case {case_id!r} has an attempt ordinal above configured maximum {n_attempts}")

        if all_explicit:
            expected_ordinals = set(range(1, required + 1))
            actual_ordinals = set(explicit)
            return required, bool(expected_ordinals - actual_ordinals), bool(actual_ordinals - expected_ordinals)
        return required, len(attempts) < required, len(attempts) > required

    missing: list[str] = []
    excess: list[str] = []
    expected_attempts = 0 if stop_on_pass else expected_count * n_attempts
    case_ids = expected_ids or sorted(by_case)
    for case_id in case_ids:
        required, case_missing, case_excess = _case_attempt_coverage(case_id, by_case.get(case_id, []))
        if stop_on_pass:
            expected_attempts += required
        if case_missing:
            missing.append(case_id)
        if case_excess:
            excess.append(case_id)

    if expected_ids:
        unexpected = sorted(case_id for case_id in by_case if case_id not in expected_set)
        if unexpected:
            errors.append(_sampled_case_id_diagnostic("Unexpected scored cases", unexpected))
    else:
        if expected_count and len(by_case) != expected_count:
            errors.append(f"Scored case coverage is {len(by_case)}/{expected_count}")
        if stop_on_pass and expected_count > len(by_case):
            expected_attempts += (expected_count - len(by_case)) * n_attempts

    if missing:
        errors.append(_sampled_case_id_diagnostic("Missing scored attempts for cases", sorted(missing)))
    if excess:
        errors.append(_sampled_case_id_diagnostic("Excess scored attempts for cases", sorted(excess)))

    scored_attempts = len(roots)
    if scored_attempts != expected_attempts:
        errors.append(f"Scored attempt coverage is {scored_attempts}/{expected_attempts}")
    all_errors = list(
        dict.fromkeys(
            safe_error for error in errors if error if (safe_error := _safe_diagnostic_text(error, max_len=4096))
        )
    )
    errors = all_errors[:PUBLISHED_EXECUTION_ERRORS_MAX]
    return {
        "execution_status": "failed" if all_errors else "succeeded",
        "execution_errors": errors,
        "execution_error_details_total": len(all_errors),
        "execution_error_details_shown": len(errors),
        "execution_error_details_truncated": len(errors) < len(all_errors),
        "expected_attempts": expected_attempts,
        "scored_attempts": scored_attempts,
        **_failure_list_metadata("runtime_failure_details", runtime_failures),
        **_failure_list_metadata("reward_failure_details", reward_failures),
    }


def _aggregate_execution(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate execution status while retaining hidden child error counts.

    Published error strings are deduplicated for display, but the total is an
    occurrence count summed from child summaries. Hidden child strings cannot
    be compared for uniqueness, so collapsing the total to the visible sample
    would falsely erase declared diagnostics.
    """
    active = [summary for summary in summaries if summary.get("execution_status") != "skipped"]
    all_errors = list(
        dict.fromkeys(str(error) for summary in active for error in summary.get("execution_errors", []) if error)
    )
    errors = all_errors[:PUBLISHED_EXECUTION_ERRORS_MAX]
    declared_total = 0
    child_truncated = False
    for summary in active:
        raw_errors = summary.get("execution_errors")
        visible_count = len([error for error in raw_errors if error]) if isinstance(raw_errors, list) else 0
        raw_total = summary.get("execution_error_details_total")
        child_total = (
            raw_total
            if isinstance(raw_total, int)
            and not isinstance(raw_total, bool)
            and 0 <= raw_total <= _MAX_JSON_SAFE_INTEGER
            else visible_count
        )
        child_total = max(child_total, visible_count)
        was_truncated = summary.get("execution_error_details_truncated") is True
        if was_truncated and child_total <= visible_count:
            child_total = min(_MAX_JSON_SAFE_INTEGER, visible_count + 1)
        declared_total = min(_MAX_JSON_SAFE_INTEGER, declared_total + child_total)
        child_truncated = child_truncated or was_truncated
    if not active:
        status = "skipped"
    elif declared_total or any(summary.get("execution_status") != "succeeded" for summary in active):
        status = "failed"
    else:
        status = "succeeded"
    return {
        "execution_status": status,
        "execution_errors": errors,
        "execution_error_details_total": declared_total,
        "execution_error_details_shown": len(errors),
        "execution_error_details_truncated": child_truncated or len(errors) < declared_total,
        "expected_attempts": sum(int(summary.get("expected_attempts", 0) or 0) for summary in active),
        "scored_attempts": sum(int(summary.get("scored_attempts", 0) or 0) for summary in active),
    }


def collect_harbor_results(
    skill_name: str,
    agents: list[str],
    output_dir: Path,
    jobs_dir: Path,
    *,
    skip_baseline: bool = False,
    n_attempts: int = 1,
    pass_threshold: float = 0.50,
    stop_on_pass: bool = False,
    expected_cases: int | None = None,
    expected_case_ids: list[str] | None = None,
    case_id_by_task_selector: dict[str, str] | None = None,
    expected_trials: int | None = None,
    expected_total_trials: int | None = None,
    env_mode: str | None = None,
    agent_models: dict[str, dict[str, str]] | None = None,
    launch_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Collect results from Harbor jobs into evals/results/<agent>/ structure.

    Returns a dict with per-agent scores, lift, and a cross-agent comparison.
    """
    if expected_trials is not None and expected_total_trials is not None and expected_trials != expected_total_trials:
        raise ValueError("Conflicting expected trial counts were provided")
    if expected_trials is None:
        expected_trials = expected_total_trials
    expected_case_ids = _validated_expected_case_ids(expected_case_ids)
    case_id_by_task_selector = _validated_case_id_by_task_selector(
        case_id_by_task_selector,
        expected_case_ids,
    )

    all_results: dict[str, Any] = {
        "agents": {},
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DISPLAY_METRICS),
        "attempt_policy": {
            "max_attempts": n_attempts,
            "pass_threshold": pass_threshold,
            "stop_on_pass": stop_on_pass,
            "score_definition": score_definition(DISPLAY_METRICS),
        },
    }

    _prepare_generated_outputs(output_dir, agents)

    for agent in agents:
        model_info = agent_models.get(agent, {}) if agent_models else {}
        agent_model = _bounded_reward_metadata_text(model_info.get("model"))
        agent_model_source = _bounded_reward_metadata_text(model_info.get("source"))
        agent_dir = output_dir / agent

        with_job_name = f"{skill_name}-{agent}-with"
        with_job_dir = _find_job_dir(jobs_dir, with_job_name)

        with_collected_rewards: list[dict[str, Any]] = []
        with_rewards: list[dict[str, Any]] = []
        with_logical_rewards: list[dict[str, Any]] = []
        with_mixed_metric_contracts = False
        with_scores: dict[str, float] = {}
        with_custom_scores: dict[str, float] = {}
        with_pass: dict[str, Any] = {}
        with_runtime_failures: list[dict[str, str]] = []
        with_trial_failures: list[dict[str, str]] = []
        with_job_failure = ""
        with_execution: dict[str, Any] = {}

        if with_job_dir:
            with_job_ok, with_job_failure = validate_harbor_job_result(
                with_job_dir / "result.json",
                expected_trials=expected_trials,
            )
            with_job_failure = _published_job_failure(with_job_failure)
            with_runtime_failures = _extract_agent_runtime_failures(with_job_dir)
            with_trial_failures = _extract_trial_failures(with_job_dir)
            preserve_partial = _can_preserve_partial_rewards(with_job_dir, with_trial_failures)
            with_collected_rewards = (
                _extract_rewards(with_job_dir, case_id_by_task_selector) if with_job_ok or preserve_partial else []
            )
            with_rewards, invalid_score_failures = _partition_scoreable_rewards(with_collected_rewards)
            with_logical_rewards = _logical_attempt_rewards(with_rewards)
            with_mixed_metric_contracts = rewards_have_mixed_metric_contracts(with_rewards)
            with_trial_failures.extend(invalid_score_failures)
            with_scores, with_metric_set, with_metrics = average_metrics(with_logical_rewards)
            all_results["metric_set"] = with_metric_set
            all_results["metrics"] = list(with_metrics)
            all_results["attempt_policy"]["score_definition"] = score_definition(with_metrics)
            with_custom_scores = average_custom_metrics(with_logical_rewards)
            with_pass = _pass_summary(
                with_logical_rewards,
                n_attempts=n_attempts,
                pass_threshold=pass_threshold,
                stop_on_pass=stop_on_pass,
                expected_cases=expected_cases,
                expected_case_ids=expected_case_ids,
            )
            with_execution = _condition_execution_summary(
                with_rewards,
                expected_case_ids=expected_case_ids,
                expected_cases=expected_cases,
                n_attempts=n_attempts,
                job_failure=with_job_failure,
                runtime_failures=with_runtime_failures,
                reward_failures=with_trial_failures,
                stop_on_pass=stop_on_pass,
                pass_threshold=pass_threshold,
            )
            if with_execution["execution_status"] != "succeeded":
                with_scores = {}
                with_custom_scores = {}
                with_pass = {}
            with_overall_score = (
                _average_overall(with_logical_rewards) if with_execution["execution_status"] == "succeeded" else None
            )
            _save_trials(
                with_collected_rewards,
                agent_dir / "with-skill" / "trials",
                with_job_dir,
                skill_name=skill_name,
                agent=agent,
                variant="with_skill",
                agent_model=agent_model,
                agent_model_source=agent_model_source,
            )
            _write_generated_root_json(
                agent_dir / "with-skill" / "summary.json",
                output_dir,
                {
                    "agent": agent,
                    "model": agent_model,
                    "model_source": agent_model_source,
                    "scores": with_scores,
                    "custom_scores": with_custom_scores,
                    "overall_score": with_overall_score,
                    "metric_set": with_metric_set,
                    "metrics": list(with_metrics),
                    "dimensions": dimension_scores(with_scores),
                    "num_trials": len(with_logical_rewards),
                    "num_reward_rows": len(with_collected_rewards),
                    "mixed_metric_contracts": with_mixed_metric_contracts,
                    "pass_at_k": _public_pass_summary(with_pass),
                    **with_execution,
                    "job_failure": with_job_failure,
                    "trial_failures": _public_failure_list(with_trial_failures),
                    **_failure_list_metadata("trial_failure_details", with_trial_failures),
                },
            )
            logger.debug(
                "Agent %s with-skill: %d trials, scores=%s",
                agent,
                len(with_logical_rewards),
                with_scores,
            )
        else:
            with_job_failure = f"No Harbor job found for {with_job_name}"
            logger.warning("No Harbor job found for %s (with-skill)", with_job_name)
            prefix = f"{agent} with-skill Harbor run failed: "
            with_job_failure = next(
                (error.removeprefix(prefix) for error in (launch_errors or []) if error.startswith(prefix)),
                f"Harbor job directory was not created: {with_job_name}",
            )
            with_job_failure = _published_job_failure(with_job_failure)
            summary_dir = agent_dir / "with-skill"
            summary_dir.mkdir(parents=True, exist_ok=True)
            _write_generated_root_json(
                summary_dir / "summary.json",
                output_dir,
                {
                    "agent": agent,
                    "model": agent_model,
                    "model_source": agent_model_source,
                    "scores": {},
                    "custom_scores": {},
                    "overall_score": None,
                    "metric_set": DEFAULT_METRIC_SET,
                    "metrics": list(DISPLAY_METRICS),
                    "dimensions": {},
                    "num_trials": 0,
                    "num_reward_rows": 0,
                    "mixed_metric_contracts": False,
                    "pass_at_k": {},
                    "job_failure": with_job_failure,
                    "trial_failures": [],
                },
            )

        if not with_execution:
            with_execution = _condition_execution_summary(
                with_rewards,
                expected_case_ids=expected_case_ids,
                expected_cases=expected_cases,
                n_attempts=n_attempts,
                job_failure=with_job_failure,
                runtime_failures=with_runtime_failures,
                stop_on_pass=stop_on_pass,
                pass_threshold=pass_threshold,
            )
        if with_job_dir is None:
            summary_dir = agent_dir / "with-skill"
            summary_dir.mkdir(parents=True, exist_ok=True)
            _write_generated_root_json(
                summary_dir / "summary.json",
                output_dir,
                {
                    "agent": agent,
                    "model": agent_model,
                    "model_source": agent_model_source,
                    "scores": {},
                    "custom_scores": {},
                    "overall_score": None,
                    "metrics": [],
                    "dimensions": {},
                    "num_trials": 0,
                    "num_reward_rows": 0,
                    "mixed_metric_contracts": False,
                    "pass_at_k": {},
                    **with_execution,
                    "job_failure": with_job_failure,
                    "trial_failures": [],
                },
            )

        without_collected_rewards: list[dict[str, Any]] = []
        without_rewards: list[dict[str, Any]] = []
        without_logical_rewards: list[dict[str, Any]] = []
        without_mixed_metric_contracts = False
        without_scores: dict[str, float] = {}
        without_custom_scores: dict[str, float] = {}
        without_pass: dict[str, Any] = {}
        without_runtime_failures: list[dict[str, str]] = []
        without_trial_failures: list[dict[str, str]] = []
        without_job_failure = ""
        without_execution: dict[str, Any] = {}
        without_job_dir: Path | None = None
        if not skip_baseline:
            without_job_name = f"{skill_name}-{agent}-without"
            without_job_dir = _find_job_dir(jobs_dir, without_job_name)

            if without_job_dir:
                without_job_ok, without_job_failure = validate_harbor_job_result(
                    without_job_dir / "result.json",
                    expected_trials=expected_trials,
                )
                without_job_failure = _published_job_failure(without_job_failure)
                without_runtime_failures = _extract_agent_runtime_failures(without_job_dir)
                without_trial_failures = _extract_trial_failures(without_job_dir)
                preserve_partial = _can_preserve_partial_rewards(without_job_dir, without_trial_failures)
                without_collected_rewards = (
                    _extract_rewards(without_job_dir, case_id_by_task_selector)
                    if without_job_ok or preserve_partial
                    else []
                )
                without_rewards, invalid_score_failures = _partition_scoreable_rewards(without_collected_rewards)
                without_logical_rewards = _logical_attempt_rewards(without_rewards)
                without_mixed_metric_contracts = rewards_have_mixed_metric_contracts(without_rewards)
                without_trial_failures.extend(invalid_score_failures)
                without_scores, without_metric_set, without_metrics = average_metrics(without_logical_rewards)
                without_custom_scores = average_custom_metrics(without_logical_rewards)
                without_pass = _pass_summary(
                    without_logical_rewards,
                    n_attempts=n_attempts,
                    pass_threshold=pass_threshold,
                    stop_on_pass=stop_on_pass,
                    expected_cases=expected_cases,
                    expected_case_ids=expected_case_ids,
                )
                without_execution = _condition_execution_summary(
                    without_rewards,
                    expected_case_ids=expected_case_ids,
                    expected_cases=expected_cases,
                    n_attempts=n_attempts,
                    job_failure=without_job_failure,
                    runtime_failures=without_runtime_failures,
                    reward_failures=without_trial_failures,
                    stop_on_pass=stop_on_pass,
                    pass_threshold=pass_threshold,
                )
                if without_execution["execution_status"] != "succeeded":
                    without_scores = {}
                    without_custom_scores = {}
                    without_pass = {}
                without_overall_score = (
                    _average_overall(without_logical_rewards)
                    if without_execution["execution_status"] == "succeeded"
                    else None
                )
                _save_trials(
                    without_collected_rewards,
                    agent_dir / "without-skill" / "trials",
                    without_job_dir,
                    skill_name=skill_name,
                    agent=agent,
                    variant="without_skill",
                    agent_model=agent_model,
                    agent_model_source=agent_model_source,
                )
                _write_generated_root_json(
                    agent_dir / "without-skill" / "summary.json",
                    output_dir,
                    {
                        "agent": agent,
                        "model": agent_model,
                        "model_source": agent_model_source,
                        "scores": without_scores,
                        "custom_scores": without_custom_scores,
                        "overall_score": without_overall_score,
                        "metric_set": without_metric_set,
                        "metrics": list(without_metrics),
                        "dimensions": dimension_scores(without_scores),
                        "num_trials": len(without_logical_rewards),
                        "num_reward_rows": len(without_collected_rewards),
                        "mixed_metric_contracts": without_mixed_metric_contracts,
                        "pass_at_k": _public_pass_summary(without_pass),
                        **without_execution,
                        "job_failure": without_job_failure,
                        "trial_failures": _public_failure_list(without_trial_failures),
                        **_failure_list_metadata("trial_failure_details", without_trial_failures),
                    },
                )
                logger.debug(
                    "Agent %s without-skill: %d trials, scores=%s",
                    agent,
                    len(without_logical_rewards),
                    without_scores,
                )
            else:
                without_job_failure = f"No Harbor job found for {without_job_name}"
                logger.warning("No Harbor job found for %s (without-skill)", without_job_name)
                prefix = f"{agent} without-skill Harbor run failed: "
                without_job_failure = next(
                    (error.removeprefix(prefix) for error in (launch_errors or []) if error.startswith(prefix)),
                    f"Harbor job directory was not created: {without_job_name}",
                )
                without_job_failure = _published_job_failure(without_job_failure)
                summary_dir = agent_dir / "without-skill"
                summary_dir.mkdir(parents=True, exist_ok=True)
                _write_generated_root_json(
                    summary_dir / "summary.json",
                    output_dir,
                    {
                        "agent": agent,
                        "model": agent_model,
                        "model_source": agent_model_source,
                        "scores": {},
                        "custom_scores": {},
                        "overall_score": None,
                        "metric_set": DEFAULT_METRIC_SET,
                        "metrics": list(DISPLAY_METRICS),
                        "dimensions": {},
                        "num_trials": 0,
                        "num_reward_rows": 0,
                        "mixed_metric_contracts": False,
                        "pass_at_k": {},
                        "job_failure": without_job_failure,
                        "trial_failures": [],
                    },
                )

        if not without_execution:
            without_execution = _condition_execution_summary(
                without_rewards,
                expected_case_ids=expected_case_ids,
                expected_cases=expected_cases,
                n_attempts=n_attempts,
                job_failure=without_job_failure,
                runtime_failures=without_runtime_failures,
                skipped=skip_baseline,
                stop_on_pass=stop_on_pass,
                pass_threshold=pass_threshold,
            )
        if not skip_baseline and without_job_dir is None:
            summary_dir = agent_dir / "without-skill"
            summary_dir.mkdir(parents=True, exist_ok=True)
            _write_generated_root_json(
                summary_dir / "summary.json",
                output_dir,
                {
                    "agent": agent,
                    "model": agent_model,
                    "model_source": agent_model_source,
                    "scores": {},
                    "custom_scores": {},
                    "overall_score": None,
                    "metrics": [],
                    "dimensions": {},
                    "num_trials": 0,
                    "num_reward_rows": 0,
                    "mixed_metric_contracts": False,
                    "pass_at_k": {},
                    **without_execution,
                    "job_failure": without_job_failure,
                    "trial_failures": [],
                },
            )

        lift: dict[str, Any] = {}
        paired_execution_succeeded = (
            with_execution.get("execution_status") == "succeeded"
            and without_execution.get("execution_status") == "succeeded"
        )
        if paired_execution_succeeded and with_scores and without_scores:
            lift = _compute_lift(with_scores, without_scores)
            _write_generated_root_json(agent_dir / "lift.json", output_dir, lift)

        custom_lift: dict[str, Any] = {}
        if (
            paired_execution_succeeded
            and with_logical_rewards
            and without_logical_rewards
            and (with_custom_scores or without_custom_scores or (not with_scores and not without_scores))
        ):
            custom_lift = _compute_custom_lift(
                with_custom_scores,
                without_custom_scores,
                with_logical_rewards,
                without_logical_rewards,
                include_overall=not with_scores and not without_scores,
            )
            if custom_lift:
                _write_generated_root_json(agent_dir / "custom_lift.json", output_dir, custom_lift)

        pass_lift: dict[str, Any] = {}
        if paired_execution_succeeded and with_pass and without_pass:
            pass_lift = {
                "with_skill": with_pass.get("rate", 0.0),
                "without_skill": without_pass.get("rate", 0.0),
                "delta": _pass_rate_delta(with_pass, without_pass),
                "count_derived_delta": _count_derived_pass_rate_delta(with_pass, without_pass),
                "passed_cases_delta": int(with_pass.get("passed_cases", 0)) - int(without_pass.get("passed_cases", 0)),
                "paired_comparison": _paired_pass_comparison(with_pass, without_pass),
            }
            _write_generated_root_json(agent_dir / "pass_at_k_lift.json", output_dir, pass_lift)

        security_attribution: dict[str, Any] = {}
        attribution_execution_succeeded = with_execution.get("execution_status") == "succeeded" and (
            skip_baseline or without_execution.get("execution_status") == "succeeded"
        )
        if attribution_execution_succeeded and with_rewards:
            security_attribution = _annotate_security_attribution(
                with_rewards,
                without_rewards,
                baseline_run=not skip_baseline,
            )
            _write_generated_root_json(
                agent_dir / "security_attribution.json",
                output_dir,
                security_attribution,
            )
            if with_job_dir:
                _save_trials(
                    with_rewards,
                    agent_dir / "with-skill" / "trials",
                    with_job_dir,
                    skill_name=skill_name,
                    agent=agent,
                    variant="with_skill",
                    agent_model=agent_model,
                    agent_model_source=agent_model_source,
                )

        agent_execution = _aggregate_execution([with_execution, without_execution])
        all_results["agents"][agent] = {
            "model": agent_model,
            "model_source": agent_model_source,
            "model_resolution": {
                "model": agent_model,
                "source": agent_model_source,
            },
            "with_skill": with_scores,
            "without_skill": without_scores,
            "custom_with_skill": with_custom_scores,
            "custom_without_skill": without_custom_scores,
            "dimensions_with_skill": dimension_scores(with_scores),
            "dimensions_without_skill": dimension_scores(without_scores),
            "lift": lift,
            "custom_lift": custom_lift,
            "pass_at_k": {
                "with_skill": _public_pass_summary(with_pass),
                "without_skill": _public_pass_summary(without_pass),
                "lift": pass_lift,
            },
            "security_attribution": security_attribution,
            "agent_runtime_failures": {
                "with_skill": _public_failure_list(with_runtime_failures),
                "without_skill": _public_failure_list(without_runtime_failures),
            },
            "trial_failures": {
                "with_skill": _public_failure_list(with_trial_failures),
                "without_skill": _public_failure_list(without_trial_failures),
            },
            "failure_detail_metadata": {
                "with_skill_runtime": _failure_list_metadata("details", with_runtime_failures),
                "without_skill_runtime": _failure_list_metadata("details", without_runtime_failures),
                "with_skill_trials": _failure_list_metadata("details", with_trial_failures),
                "without_skill_trials": _failure_list_metadata("details", without_trial_failures),
            },
            "job_failures": {
                "with_skill": with_job_failure,
                "without_skill": without_job_failure,
            },
            "conditions": {
                "with_skill": with_execution,
                "without_skill": without_execution,
            },
            **agent_execution,
            "num_trials_with": len(with_logical_rewards),
            "num_trials_without": len(without_logical_rewards) if not skip_baseline else 0,
            "output_dir": str(agent_dir.resolve()),
        }

    _write_generated_root_json(output_dir / "attempt_policy.json", output_dir, all_results["attempt_policy"])

    if len(agents) > 1:
        comparison = _build_comparison(all_results["agents"])
        all_results["comparison"] = comparison
        _write_generated_root_json(output_dir / "comparison.json", output_dir, comparison)

    top_execution = _aggregate_execution(list(all_results["agents"].values()))
    all_results.update(top_execution)
    if top_execution["execution_errors"]:
        all_results["error"] = list(top_execution["execution_errors"])

    return all_results


def _build_comparison(agents_data: dict[str, Any]) -> dict[str, Any]:
    """Build cross-agent comparison table."""
    comparison: dict[str, Any] = {"metrics": {}}

    metric_names = []
    for data in agents_data.values():
        for metric in data.get("with_skill", {}):
            if metric not in metric_names:
                metric_names.append(metric)
    if not metric_names:
        metric_names = list(DISPLAY_METRICS)

    for metric in metric_names:
        comparison["metrics"][metric] = {}
        for agent, data in agents_data.items():
            succeeded = data.get("execution_status") == "succeeded"
            with_skill = data.get("with_skill", {}).get(metric) if succeeded else None
            without_skill = data.get("without_skill", {}).get(metric) if succeeded else None
            lift = data.get("lift", {}).get(metric, {}).get("delta") if succeeded else None
            comparison["metrics"][metric][agent] = {
                "with_skill": with_skill
                if isinstance(with_skill, int | float) and not isinstance(with_skill, bool)
                else None,
                "without_skill": (
                    without_skill
                    if isinstance(without_skill, int | float) and not isinstance(without_skill, bool)
                    else None
                ),
                "lift": lift if isinstance(lift, int | float) and not isinstance(lift, bool) else None,
            }

    return comparison
