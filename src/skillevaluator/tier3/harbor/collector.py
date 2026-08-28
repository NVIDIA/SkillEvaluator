# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor results collector -- reads Harbor job directories and consolidates
results into the evals/results/<agent>/ structure.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts, get_skill_tool_calls
from skillevaluator.tier3.eval_core.checks import check_negative_case
from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    LEGACY_METRIC_SET,
    LEGACY_METRICS,
    average_custom_metrics,
    average_metrics,
    dimension_scores,
    extract_custom_metrics,
    metric_set_for_reward,
    metric_value,
    overall_score,
    score_definition,
)
from skillevaluator.tier3.output_provenance import write_output_file_atomically
from skillevaluator.utils.redaction import is_sensitive_key, redact_sensitive_data, redact_sensitive_text
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, stat_is_link_or_reparse

logger = logging.getLogger(__name__)

DISPLAY_METRICS = DEFAULT_METRICS
DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES = 5 * 1024 * 1024
DIAGNOSTIC_ARTIFACT_HARD_MAX_BYTES = 64 * 1024 * 1024
REWARD_DIAGNOSTIC_STRING_MAX_CHARS = 8192
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
    "AgentTimeoutError",
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
    """Publish one root artifact without following unsafe replacements."""
    _assert_safe_generated_output_path(path, output_root, follow_target=False)
    write_output_file_atomically(path, json.dumps(payload, indent=2).encode("utf-8"))


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
    """Find a Harbor job directory by name."""
    candidate = jobs_dir / job_name
    if candidate.exists():
        return candidate
    for d in sorted(jobs_dir.iterdir(), reverse=True):
        if d.is_dir() and job_name in d.name:
            return d
    return None


def _safe_text(value: Any, *, max_len: int | None = 2048) -> str:
    text = redact_sensitive_text(str(value or ""))
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 14] + "...<truncated>"
    return text


def _safe_diagnostic_text(value: Any, *, max_len: int) -> str:
    """Return redacted, bounded text without terminal-control characters."""
    return re.sub(r"[\x00-\x1f\x7f]", " ", _safe_text(value, max_len=max_len)).strip()


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


def _trial_exception_details(trial_dir: Path) -> tuple[str, str]:
    """Return the Harbor trial exception type and display reason, if present."""
    result = _read_json(trial_dir / "result.json")
    if not isinstance(result, dict):
        return "", ""
    exception_info = result.get("exception_info")
    if not isinstance(exception_info, dict):
        return "", ""

    exception_type = str(exception_info.get("exception_type") or "").strip()
    exception_message = str(exception_info.get("exception_message") or "").strip()
    if exception_type and exception_message:
        return exception_type, f"{exception_type}: {exception_message}"[:600]
    return exception_type, (exception_type or exception_message)[:600]


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
    exception_type, exception_reason = _trial_exception_details(trial_dir)
    agent_reason = _agent_log_runtime_failure_reason(
        trial_dir,
        include_text_logs=(
            exception_type in _AGENT_RUNTIME_EXCEPTION_TYPES
            or exception_type in _UNCONDITIONAL_AGENT_RUNTIME_EXCEPTION_TYPES
        ),
    )
    if agent_reason:
        return agent_reason

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
    for trial_dir in sorted(job_dir.iterdir()):
        kind, unsafe_reason = _inspect_trial_directory(trial_dir)
        if kind == "link":
            failures.append({"trial": trial_dir.name, "reason": unsafe_reason})
            continue
        if kind != "directory":
            continue
        reason = _trial_failure_reason(trial_dir)
        if reason:
            failures.append({"trial": trial_dir.name, "reason": redact_sensitive_text(reason)})
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
    for trial_dir in sorted(job_dir.iterdir()):
        kind, _reason = _inspect_trial_directory(trial_dir)
        if kind != "directory":
            continue
        reason = _agent_runtime_failure_reason(trial_dir)
        if reason:
            failures.append({"trial": trial_dir.name, "reason": redact_sensitive_text(reason)})
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


def _redacted_artifact_text(src: Path, text: str) -> str | None:
    if src.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            safe_data = redact_sensitive_data(data)
            # Compact encoding avoids indentation-driven amplification for
            # deeply nested but otherwise valid diagnostic JSON.
            return json.dumps(safe_data, separators=(",", ":"), ensure_ascii=False)
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


def _copy_trial_artifacts(trial_dir: Path, trial_out: Path) -> list[str]:
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
) -> None:
    if job_dir is None or not job_dir.exists():
        return

    scored_trials = {str(reward.get("_trial_root_name") or reward.get("_trial_name") or "") for reward in rewards}
    for trial_src in sorted(job_dir.iterdir()):
        kind, unsafe_reason = _inspect_trial_directory(trial_src)
        if kind == "link":
            trial_out = trials_dir / trial_src.name
            trial_out.mkdir(parents=True, exist_ok=True)
            write_output_file_atomically(
                trial_out / "failure.json",
                json.dumps(
                    {
                        "status": "unscored",
                        "trial": trial_src.name,
                        "agent": agent,
                        "variant": variant,
                        "artifacts": [],
                        "error": {"type": "UnsafeHarborTrial", "message": unsafe_reason},
                    },
                    indent=2,
                ).encode("utf-8"),
            )
            continue
        if kind != "directory":
            continue
        if trial_src.name in scored_trials or not _looks_like_trial_dir(trial_src):
            continue

        trial_out = trials_dir / trial_src.name
        trial_out.mkdir(parents=True, exist_ok=True)
        copied = _copy_trial_artifacts(trial_src, trial_out)
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
            "trial": trial_src.name,
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
            failure_file.write_text(
                json.dumps(redact_sensitive_data(failure), indent=2),
                encoding="utf-8",
            )
        except OSError as e:
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
    try:
        with os.scandir(steps_dir) as entries:
            for entry in entries:
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
    result = _read_json(trial_root / "result.json")
    if isinstance(result, dict):
        step_results = result.get("step_results")
        if isinstance(step_results, list):
            for step in step_results:
                if isinstance(step, dict):
                    step_name = step.get("step_name")
                    if isinstance(step_name, str) and step_name and step_name not in ordered_names:
                        ordered_names.append(step_name)

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


def _merged_step_trajectory(trial_root: Path) -> dict[str, Any] | None:
    """Merge Harbor multi-step ATIF fragments into one collected trajectory."""
    trajectories: list[tuple[str, dict[str, Any]]] = []
    for path in _ordered_step_trajectory_paths(trial_root):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        steps = data.get("steps")
        if not isinstance(steps, list):
            continue
        step_name = path.parent.parent.name
        trajectories.append((step_name, data))

    if not trajectories:
        return None

    merged = copy.deepcopy(trajectories[0][1])
    merged_steps: list[dict[str, Any]] = []
    for step_name, trajectory in trajectories:
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            merged_step = copy.deepcopy(step)
            original_step_id = merged_step.get("step_id")
            merged_step["step_id"] = len(merged_steps) + 1
            extra = merged_step.get("extra")
            if not isinstance(extra, dict):
                extra = {}
            extra.setdefault("harbor_step_name", step_name)
            if original_step_id not in (None, ""):
                extra.setdefault("harbor_original_step_id", original_step_id)
            merged_step["extra"] = extra
            merged_steps.append(merged_step)

    if not merged_steps:
        return None

    step_names = [name for name, _ in trajectories]
    merged["steps"] = merged_steps
    merged["schema_version"] = str(trajectories[0][1].get("schema_version") or merged.get("schema_version") or "")
    merged["agent"] = trajectories[0][1].get("agent") or merged.get("agent")
    merged_extra = merged.get("extra")
    if not isinstance(merged_extra, dict):
        merged_extra = {}
    merged_extra["harbor_multi_step"] = {
        "step_count": len(step_names),
        "step_names": step_names,
    }
    merged["extra"] = merged_extra

    merged["final_metrics"] = _merge_trajectory_final_metrics(
        [trajectory for _, trajectory in trajectories],
        total_steps=len(merged_steps),
    )
    return merged


def _merge_trajectory_final_metrics(
    trajectories: list[dict[str, Any]],
    *,
    total_steps: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("total_prompt_tokens", "total_completion_tokens", "total_cached_tokens"):
        values = [
            final_metrics.get(key)
            for trajectory in trajectories
            if isinstance(final_metrics := trajectory.get("final_metrics"), dict)
        ]
        numeric = [value for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
        if numeric:
            metrics[key] = sum(numeric)

    metrics["total_steps"] = total_steps
    last_final_metrics = next(
        (
            trajectory.get("final_metrics")
            for trajectory in reversed(trajectories)
            if isinstance(trajectory.get("final_metrics"), dict)
        ),
        {},
    )
    last_extra = last_final_metrics.get("extra") if isinstance(last_final_metrics, dict) else None
    extra = copy.deepcopy(last_extra) if isinstance(last_extra, dict) else {}
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
    for key in extra_token_keys:
        numeric_values: list[int | float] = []
        for trajectory in trajectories:
            final_metrics = trajectory.get("final_metrics")
            if not isinstance(final_metrics, dict):
                continue
            step_extra = final_metrics.get("extra")
            if not isinstance(step_extra, dict):
                continue
            value = step_extra.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                numeric_values.append(value)
        if numeric_values:
            extra[key] = sum(numeric_values)
    extra["harbor_multi_step"] = True
    metrics["extra"] = extra
    return metrics


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
        data["custom_details"] = custom_details

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
    if declared_metric_set not in {DEFAULT_METRIC_SET, LEGACY_METRIC_SET} and not metric_names.intersection(
        DEFAULT_METRICS
    ):
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

    root_metrics = _standard_reward_metrics(root_rewards) if isinstance(root_rewards, dict) else ()

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
        if failed_status or (root_metrics and (not isinstance(rewards, dict) or not rewards)):
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
        if all(metric_value(rewards, metric) is not None for metric in expected_metrics):
            continue

        return (
            f"Constituent default reward for step {step_name} is incomplete, non-finite, or failed; "
            "the authoritative aggregate was not scored"
        )
    return ""


def _merge_constituent_default_reward_failure(
    data: dict[str, Any],
    result: dict[str, Any],
    trial_root: Path | None = None,
) -> None:
    """Make an aggregate unscoreable when one of its standard constituents is invalid."""
    reason = _constituent_default_reward_failure(result, trial_root)
    if not reason:
        return
    data["evaluation_status"] = "failed"
    data["evaluation_errors"] = _merge_bounded_evaluation_errors(
        {"collector": reason},
        data.get("evaluation_errors"),
    )


def _extract_rewards(job_dir: Path) -> list[dict[str, Any]]:
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
        if not data.get("entry_id"):
            entry_id = _entry_id_from_harbor_result(result)
            if entry_id:
                data["entry_id"] = entry_id
        traj_file = _reward_trajectory_path(trial_dir, None)
        if traj_file.exists():
            data["_has_trajectory"] = True
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
                if result_file.exists():
                    result = _read_json(result_file)
                    if isinstance(result, dict):
                        _merge_constituent_default_reward_failure(data, result, trial_dir)
                        data["_started_at"] = result.get("started_at")
                        if not data.get("entry_id"):
                            entry_id = _entry_id_from_harbor_result(result)
                            if entry_id:
                                data["entry_id"] = entry_id
                traj_file = _reward_trajectory_path(trial_dir, step_name)
                if traj_file.exists():
                    data["_has_trajectory"] = True
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
        data = _reward_from_harbor_result(result)
        if not data:
            continue
        _merge_constituent_default_reward_failure(data, result, trial_dir)
        _merge_trial_evaluation_failures(data, trial_dir)
        trial_name = str(result.get("trial_name") or trial_dir.name)
        data["_trial_name"] = trial_name
        data["_trial_root_name"] = trial_dir.name
        data["_started_at"] = result.get("started_at")
        if not data.get("entry_id"):
            entry_id = _entry_id_from_harbor_result(result)
            if entry_id:
                data["entry_id"] = entry_id
        traj_file = _reward_trajectory_path(trial_dir, None)
        if traj_file.exists():
            data["_has_trajectory"] = True
        rewards.append(data)
    return rewards


def _reward_from_harbor_result(result: dict[str, Any]) -> dict[str, Any] | None:
    harbor_rewards = _harbor_result_rewards(result)
    if not harbor_rewards:
        return None

    data: dict[str, Any] = {}
    custom_metrics: dict[str, float] = {}
    for key, value in harbor_rewards.items():
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        score = float(value)
        if key in DEFAULT_METRICS:
            data[key] = score
        elif key == "overall":
            data["overall"] = score
        elif key == "reward":
            data.setdefault("overall", score)
            custom_metrics[key] = score
        else:
            custom_metrics[key] = score

    if not any(not k.startswith("_") for k in data) and not custom_metrics:
        return None
    if custom_metrics:
        data["custom_metrics"] = custom_metrics
    data["details"] = {"harbor_rewards": harbor_rewards}
    return data


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
        values = [
            float(rewards[key])
            for rewards in step_reward_rows
            if isinstance(rewards.get(key), int | float) and not isinstance(rewards.get(key), bool)
        ]
        if values:
            aggregated[key] = sum(values) / len(values)

    aggregated.update(average_custom_metrics(step_reward_rows))

    standard_rows = [rewards for rewards in step_reward_rows if _standard_reward_metrics(rewards)]
    active_metrics: tuple[str, ...] = ()
    if any(_standard_reward_metrics(rewards) == DEFAULT_METRICS for rewards in standard_rows):
        active_metrics = DEFAULT_METRICS
    elif standard_rows:
        active_metrics = LEGACY_METRICS
    for metric in active_metrics:
        values = [value for rewards in standard_rows if (value := metric_value(rewards, metric)) is not None]
        if values:
            aggregated[metric] = sum(values) / len(values)
    return aggregated or None


def _entry_id_from_harbor_result(result: dict[str, Any]) -> str:
    task_name = result.get("task_name")
    if isinstance(task_name, str) and task_name.strip():
        return task_name.strip().rsplit("/", 1)[-1]

    task_id = result.get("task_id")
    if isinstance(task_id, dict):
        task_path = task_id.get("path")
        if isinstance(task_path, str) and task_path.strip():
            return Path(task_path).name

    config = result.get("config")
    if isinstance(config, dict):
        task = config.get("task")
        if isinstance(task, dict):
            task_path = task.get("path")
            if isinstance(task_path, str) and task_path.strip():
                return Path(task_path).name

    return ""


def _overall_score(reward: dict[str, Any]) -> float | None:
    return overall_score(reward)


def _partition_scoreable_rewards(
    rewards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Separate complete finite rewards from diagnostic-only reward artifacts."""
    scoreable: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    failed_trials: set[str] = set()
    for reward in rewards:
        evaluation_failed = str(reward.get("evaluation_status") or "").casefold() in {"error", "failed"}
        if not evaluation_failed and overall_score(reward) is not None:
            scoreable.append(reward)
            continue
        trial = str(reward.get("_trial_name") or reward.get("_trial_root_name") or "unknown trial")
        if trial in failed_trials:
            continue
        failed_trials.add(trial)
        failures.append(
            {
                "trial": trial,
                "reason": _unscoreable_reward_reason(reward),
            }
        )
    return scoreable, failures


def _unscoreable_reward_reason(reward: dict[str, Any]) -> str:
    """Return a bounded, redacted diagnostic for an unscoreable reward."""
    fallback = "Reward metrics are incomplete or non-finite; trial was not scored"
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


def _canonical_case_id(value: str, expected_case_ids: set[str] | None = None) -> str:
    value = str(value or "").strip()
    if not value:
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
    if reward.get("entry_id"):
        return _canonical_case_id(str(reward["entry_id"]), expected_case_ids)
    trial_name = str(reward.get("_trial_name") or "")
    if trial_name:
        return _canonical_case_id(trial_name.split("__", 1)[0], expected_case_ids)
    return "unknown"


def _attempt_sort_key(reward: dict[str, Any]) -> tuple[int, int | str, str, str]:
    """Sort attempts by explicit attempt label, then Harbor start time."""
    trial_name = str(reward.get("_trial_name") or "")
    match = re.search(r"attempt(\d+)", trial_name)
    if match:
        return (0, int(match.group(1)), str(reward.get("_started_at") or ""), trial_name)
    started_at = str(reward.get("_started_at") or "")
    return (1 if started_at else 2, started_at, "", trial_name)


def _attempt_ordinal(reward: dict[str, Any]) -> int | None:
    """Return an explicit Harbor attempt ordinal when the trial names carry one."""
    for key in ("_trial_root_name", "_trial_name"):
        match = re.search(r"attempt0*(\d+)", str(reward.get(key) or ""), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
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
        custom_scores = average_custom_metrics(rows)
        logical_reward: dict[str, Any] = {
            "entry_id": first.get("entry_id"),
            "_trial_name": root,
            "_trial_root_name": root,
            "_started_at": first.get("_started_at"),
        }
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
    with_cases = with_skill.get("cases")
    without_cases = without_skill.get("cases")
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
    expected_ids = list(dict.fromkeys(str(case_id) for case_id in (expected_case_ids or []) if str(case_id)))
    expected_id_set = set(expected_ids) if expected_ids else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for reward in _logical_attempt_rewards(rewards):
        if _overall_score(reward) is None:
            continue
        grouped.setdefault(_entry_id(reward, expected_id_set), []).append(reward)

    cases: dict[str, Any] = {}
    passed_cases = 0
    attempts_used = 0
    extra_cases: list[str] = []

    case_order = expected_ids or sorted(grouped)
    if expected_ids:
        extra_cases = sorted(entry_id for entry_id in grouped if entry_id not in expected_id_set)
        case_order = [*case_order, *extra_cases]

    for entry_id in case_order:
        attempts = grouped.get(entry_id, [])
        attempt_rows = []
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
            attempt_rows.append(
                {
                    "attempt": idx,
                    "trial": reward.get("_trial_name", ""),
                    "score": score,
                    "passed": passed,
                }
            )

        case_passed = first_pass_attempt is not None
        is_expected_case = expected_id_set is None or entry_id in expected_id_set
        if case_passed and is_expected_case:
            passed_cases += 1
        if is_expected_case:
            attempts_used += len(attempts)
        unscored = max(0, n_attempts - len(attempts))
        skipped = unscored if stop_on_pass and case_passed else 0
        missing = 0 if skipped else unscored

        cases[entry_id] = {
            "passed": case_passed,
            "first_pass_attempt": first_pass_attempt,
            "attempts_used": len(attempts),
            "attempts_skipped": skipped,
            "attempts_missing": missing,
            "best_score": round(best_score, 4) if best_score is not None else None,
            "attempts": attempt_rows,
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
        "extra_cases": extra_cases,
        "cases": cases,
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
    values = [overall_score(reward) for reward in rewards]
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
    component = str(value or "").strip()
    if (
        not component
        or component in {".", ".."}
        or any(character in component for character in ("/", "\\", ":", "\x00"))
        or any(ord(character) < 32 or ord(character) == 127 for character in component)
    ):
        return ""
    return component


def _persisted_trial_name(reward: dict[str, Any]) -> tuple[str, str]:
    """Derive output and source names only from physical Harbor path components."""
    trial_root_name = _safe_trial_path_component(reward.get("_trial_root_name")) or "unknown"
    step_name = _safe_trial_path_component(reward.get("_step_name"))
    return (f"{trial_root_name}__{step_name}" if step_name else trial_root_name), trial_root_name


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
                finding["attribution"] = attribution
                finding["attribution_explanation"] = explanation
            security["attribution"] = with_findings[0].get("attribution")
            security["attribution_explanation"] = with_findings[0].get("attribution_explanation")
        elif baseline_findings:
            case_status = "baseline_unsafe_with_skill_safe"
            security.setdefault("findings", []).append(
                {
                    "type": "skill_reduced_unsafe_behavior",
                    "severity": "info",
                    "message": "Baseline had unsafe agent action, but with-skill run did not",
                    "evidence": "; ".join(str(f.get("message", "")) for f in baseline_findings[:2]),
                    "source": "baseline_comparison",
                    "score_impact": False,
                    "attribution": "skill_may_have_improved_safety",
                    "attribution_explanation": (
                        "The without-skill baseline showed unsafe behavior for this case, while the "
                        "with-skill run did not."
                    ),
                }
            )
            security["attribution"] = "skill_may_have_improved_safety"
            security["attribution_explanation"] = (
                "The without-skill baseline showed unsafe behavior for this case, while the with-skill run did not."
            )
            summary["skill_may_have_improved_safety"] += 1

        summary["cases"][entry_id] = {
            "status": case_status,
            "with_skill_findings": len(with_findings),
            "baseline_findings": len(baseline_findings),
        }

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


def _trajectory_skill_invoked(trajectory: Any, skill_name: str) -> bool | None:
    """Derive target invocation from one readable ATIF trajectory."""
    if not isinstance(trajectory, dict):
        return None
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps or any(not isinstance(step, dict) for step in steps):
        return None
    agent_steps = [step for step in steps if step.get("source") == "agent"]
    if not agent_steps:
        return None
    for step in agent_steps:
        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list):
            return None
        for tool_call in tool_calls:
            if (
                not isinstance(tool_call, dict)
                or not isinstance(tool_call.get("function_name"), str)
                or not tool_call["function_name"].strip()
                or not isinstance(tool_call.get("arguments"), dict)
            ):
                return None
    try:
        agent_trajectory = {**trajectory, "steps": agent_steps}
        tool_calls = extract_tool_calls_as_dicts(agent_trajectory)
        skill_tool_names = get_skill_tool_calls(agent_trajectory)
        negative_check = check_negative_case(tool_calls, skill_name, skill_tool_names=skill_tool_names)
    except (AttributeError, TypeError, ValueError):
        return None
    passed = negative_check.get("passed")
    return not passed if isinstance(passed, bool) else None


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
    if not isinstance(step_results, list) or not step_results:
        return True, None

    names: list[str] = []
    for step in step_results:
        if not isinstance(step, dict):
            return True, None
        step_name = step.get("step_name")
        if not isinstance(step_name, str) or not step_name or step_name in names:
            return True, None
        names.append(step_name)
    return True, names


def _trusted_trial_skill_invoked(
    trial_root: Path,
    step_name: str | None,
    skill_name: str,
) -> bool | None:
    """Derive trusted invocation without falling back across logical steps."""
    if step_name:
        safe_step_paths = {path.parent.parent.name: path for path in _ordered_step_trajectory_paths(trial_root)}
        trajectory_path = safe_step_paths.get(step_name)
        return _trajectory_skill_invoked(_read_json(trajectory_path), skill_name) if trajectory_path else None

    is_multi_step, authoritative_names = _authoritative_step_names(trial_root)
    if is_multi_step:
        if authoritative_names is None:
            return None
        safe_step_paths = {path.parent.parent.name: path for path in _ordered_step_trajectory_paths(trial_root)}
        saw_unknown = False
        for authoritative_name in authoritative_names:
            trajectory_path = safe_step_paths.get(authoritative_name)
            invoked = _trajectory_skill_invoked(_read_json(trajectory_path), skill_name) if trajectory_path else None
            if invoked is True:
                return True
            if invoked is None:
                saw_unknown = True
        return None if saw_unknown else False

    return _trajectory_skill_invoked(_read_json(trial_root / "agent" / "trajectory.json"), skill_name)


def _add_trusted_invocation_evidence(
    clean_reward: dict[str, Any],
    source_reward: dict[str, Any],
    trial_root: Path | None,
    skill_name: str,
) -> None:
    """Replace verifier-authored routing evidence on standard rewards only."""
    if not _is_standard_skill_execution_reward(source_reward):
        return
    for key in ("skill_invoked", "routing_passed", "invocation_evidence_source"):
        clean_reward.pop(key, None)
    if trial_root is None:
        return
    invoked = _trusted_trial_skill_invoked(trial_root, source_reward.get("_step_name"), skill_name)
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
    if not is_sensitive_key(value):
        return True
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", camel_split).strip("_").lower()
    parts = tuple(part for part in normalized.split("_") if part)
    return (
        len(parts) == 2
        and parts[0] in {"auth", "secret", "token"}
        and parts[1]
        in {
            "accuracy",
            "compliance",
            "count",
            "coverage",
            "efficiency",
            "handling",
            "leakage",
            "precision",
            "quality",
            "rate",
            "ratio",
            "recall",
            "safety",
            "score",
            "usage",
        }
    )


def _restore_custom_metric_scores(source_reward: dict[str, Any], safe_reward: dict[str, Any]) -> None:
    """Restore only finite numeric values recognized by the custom-metric schema."""
    for field in ("custom_metrics", "metrics"):
        source_metrics = source_reward.get(field)
        safe_metrics = safe_reward.get(field)
        if not isinstance(source_metrics, dict) or not isinstance(safe_metrics, dict):
            continue
        for raw_name, raw_value in source_metrics.items():
            name = str(raw_name)
            if not _can_restore_custom_metric_name(name):
                continue
            score = extract_custom_metrics({field: {name: raw_value}}).get(name)
            if score is None:
                continue
            if isinstance(raw_value, dict):
                safe_value = safe_metrics.get(name)
                if not isinstance(safe_value, dict):
                    safe_value = {}
                    safe_metrics[name] = safe_value
                safe_value["score"] = score
            else:
                safe_metrics[name] = score

    # Legacy custom-only rewards allowed this report metric at the top level.
    token_efficiency = extract_custom_metrics({"token_efficiency": source_reward.get("token_efficiency")}).get(
        "token_efficiency"
    )
    if token_efficiency is not None:
        safe_reward["token_efficiency"] = token_efficiency


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
    trials_dir.mkdir(parents=True, exist_ok=True)
    for reward in rewards:
        trial_name, trial_root_name = _persisted_trial_name(reward)
        trial_out = trials_dir / trial_name
        trial_out.mkdir(parents=True, exist_ok=True)
        trial_src = job_dir / trial_root_name if job_dir else None
        src_traj = _reward_trajectory_path(trial_src, reward.get("_step_name")) if trial_src else None
        merged_traj = (
            _merged_step_trajectory(trial_src)
            if trial_src and not reward.get("_step_name") and not (trial_src / "agent" / "trajectory.json").exists()
            else None
        )
        if merged_traj and "_trajectory_summary" not in reward:
            reward["_trajectory_summary"] = _summarize_trajectory(merged_traj)
        elif src_traj and src_traj.exists() and "_trajectory_summary" not in reward:
            reward["_trajectory_summary"] = _summarize_trajectory_file(src_traj)

        clean_reward = {k: v for k, v in reward.items() if not k.startswith("_")}
        _add_trusted_invocation_evidence(clean_reward, reward, trial_src, skill_name)
        # Persist the physical attempt identity so bounded report readers can keep
        # fallback multi-step rows for diagnostics without weighting a logical
        # Harbor trial once per step.  This also populates the canonical report's
        # existing trial_id field instead of inventing a second report schema.
        clean_reward["trial_id"] = trial_root_name
        if not clean_reward.get("entry_id"):
            clean_reward["entry_id"] = _entry_id(reward)
        clean_reward["agent"] = agent
        if agent_model:
            clean_reward["model"] = agent_model
        if agent_model_source:
            clean_reward["model_source"] = agent_model_source
        if "evaluation_errors" in clean_reward:
            clean_reward["evaluation_errors"] = _safe_evaluation_errors(clean_reward["evaluation_errors"])
        diagnostic_reward = (
            str(clean_reward.get("evaluation_status") or "").casefold() in {"error", "failed"}
            or overall_score(clean_reward) is None
        )
        safe_reward = redact_sensitive_data(
            clean_reward,
            max_str_len=REWARD_DIAGNOSTIC_STRING_MAX_CHARS if diagnostic_reward else None,
        )
        _restore_custom_metric_scores(clean_reward, safe_reward)
        (trial_out / "reward.json").write_text(json.dumps(safe_reward, indent=2), encoding="utf-8")

        if trial_src:
            _copy_trial_artifacts(trial_src, trial_out)
        if merged_traj:
            (trial_out / "trajectory.json").write_text(
                json.dumps(redact_sensitive_data(merged_traj), indent=2),
                encoding="utf-8",
            )
        elif src_traj and src_traj.exists():
            _write_redacted_text_copy(src_traj, trial_out / "trajectory.json", source_root=trial_src)

    _save_unscored_trials(
        rewards,
        trials_dir,
        job_dir,
        agent=agent,
        variant=variant,
        agent_model=agent_model,
        agent_model_source=agent_model_source,
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
    expected_ids = list(dict.fromkeys(str(case_id) for case_id in (expected_case_ids or []) if str(case_id)))
    expected_count = len(expected_ids) if expected_ids else int(expected_cases or 0)
    if skipped:
        return {
            "execution_status": "skipped",
            "execution_errors": [],
            "expected_attempts": 0,
            "scored_attempts": 0,
        }

    errors: list[str] = [job_failure] if job_failure else []
    errors.extend(
        f"Agent runtime failed in {failure.get('trial', 'unknown trial')}: {failure.get('reason', 'unknown error')}"
        for failure in (runtime_failures or [])
    )
    errors.extend(
        "Unscoreable reward in "
        f"{_safe_diagnostic_text(failure.get('trial', 'unknown trial'), max_len=256)}: "
        f"{_safe_diagnostic_text(failure.get('reason', 'unknown error'), max_len=2048)}"
        for failure in (reward_failures or [])
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
        case_id = _entry_id(reward, expected_set or None)
        step_name = str(reward.get("_step_name") or "").strip()
        if not root:
            errors.append("A scored reward is missing its Harbor trial root name")
            continue
        if not case_id or case_id == "unknown":
            errors.append(f"Scored trial {root!r} has no case identifier")
            continue
        score = overall_score(reward)
        if score is None:
            errors.append(f"Scored trial {root!r} has incomplete or non-finite reward metrics")
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
            errors.append(f"Harbor trial {root!r} maps to multiple cases")
        elif not step_name or step_name in existing["steps"]:
            errors.append(f"Harbor trial {root!r} has duplicate reward rows")
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
            errors.append("Unexpected scored cases: " + ", ".join(unexpected))
    else:
        if expected_count and len(by_case) != expected_count:
            errors.append(f"Scored case coverage is {len(by_case)}/{expected_count}")
        if stop_on_pass and expected_count > len(by_case):
            expected_attempts += (expected_count - len(by_case)) * n_attempts

    if missing:
        errors.append("Missing scored attempts for cases: " + ", ".join(sorted(missing)))
    if excess:
        errors.append("Excess scored attempts for cases: " + ", ".join(sorted(excess)))

    scored_attempts = len(roots)
    if scored_attempts != expected_attempts:
        errors.append(f"Scored attempt coverage is {scored_attempts}/{expected_attempts}")
    errors = list(dict.fromkeys(error for error in errors if error))
    return {
        "execution_status": "failed" if errors else "succeeded",
        "execution_errors": errors,
        "expected_attempts": expected_attempts,
        "scored_attempts": scored_attempts,
    }


def _aggregate_execution(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    active = [summary for summary in summaries if summary.get("execution_status") != "skipped"]
    errors = [str(error) for summary in active for error in summary.get("execution_errors", []) if error]
    if not active:
        status = "skipped"
    elif errors or any(summary.get("execution_status") != "succeeded" for summary in active):
        status = "failed"
    else:
        status = "succeeded"
    return {
        "execution_status": status,
        "execution_errors": list(dict.fromkeys(errors)),
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
        agent_model = model_info.get("model")
        agent_model_source = model_info.get("source")
        agent_dir = output_dir / agent

        with_job_name = f"{skill_name}-{agent}-with"
        with_job_dir = _find_job_dir(jobs_dir, with_job_name)

        with_collected_rewards: list[dict[str, Any]] = []
        with_rewards: list[dict[str, Any]] = []
        with_logical_rewards: list[dict[str, Any]] = []
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
            with_runtime_failures = _extract_agent_runtime_failures(with_job_dir)
            with_trial_failures = _extract_trial_failures(with_job_dir)
            preserve_partial = _can_preserve_partial_rewards(with_job_dir, with_trial_failures)
            with_collected_rewards = _extract_rewards(with_job_dir) if with_job_ok or preserve_partial else []
            with_rewards, invalid_score_failures = _partition_scoreable_rewards(with_collected_rewards)
            with_logical_rewards = _logical_attempt_rewards(with_rewards)
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
            (agent_dir / "with-skill" / "summary.json").write_text(
                json.dumps(
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
                        "num_trials": len(with_rewards),
                        "pass_at_k": with_pass,
                        **with_execution,
                        "job_failure": with_job_failure,
                        "trial_failures": with_trial_failures,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.debug("Agent %s with-skill: %d trials, scores=%s", agent, len(with_rewards), with_scores)
        else:
            with_job_failure = f"No Harbor job found for {with_job_name}"
            logger.warning("No Harbor job found for %s (with-skill)", with_job_name)
            prefix = f"{agent} with-skill Harbor run failed: "
            with_job_failure = next(
                (error.removeprefix(prefix) for error in (launch_errors or []) if error.startswith(prefix)),
                f"Harbor job directory was not created: {with_job_name}",
            )
            summary_dir = agent_dir / "with-skill"
            summary_dir.mkdir(parents=True, exist_ok=True)
            (summary_dir / "summary.json").write_text(
                json.dumps(
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
                        "pass_at_k": {},
                        "job_failure": with_job_failure,
                        "trial_failures": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
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
            (summary_dir / "summary.json").write_text(
                json.dumps(
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
                        "pass_at_k": {},
                        **with_execution,
                        "job_failure": with_job_failure,
                        "trial_failures": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        without_collected_rewards: list[dict[str, Any]] = []
        without_rewards: list[dict[str, Any]] = []
        without_logical_rewards: list[dict[str, Any]] = []
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
                without_runtime_failures = _extract_agent_runtime_failures(without_job_dir)
                without_trial_failures = _extract_trial_failures(without_job_dir)
                preserve_partial = _can_preserve_partial_rewards(without_job_dir, without_trial_failures)
                without_collected_rewards = (
                    _extract_rewards(without_job_dir) if without_job_ok or preserve_partial else []
                )
                without_rewards, invalid_score_failures = _partition_scoreable_rewards(without_collected_rewards)
                without_logical_rewards = _logical_attempt_rewards(without_rewards)
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
                (agent_dir / "without-skill" / "summary.json").write_text(
                    json.dumps(
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
                            "num_trials": len(without_rewards),
                            "pass_at_k": without_pass,
                            **without_execution,
                            "job_failure": without_job_failure,
                            "trial_failures": without_trial_failures,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.debug(
                    "Agent %s without-skill: %d trials, scores=%s",
                    agent,
                    len(without_rewards),
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
                summary_dir = agent_dir / "without-skill"
                summary_dir.mkdir(parents=True, exist_ok=True)
                (summary_dir / "summary.json").write_text(
                    json.dumps(
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
                            "pass_at_k": {},
                            "job_failure": without_job_failure,
                            "trial_failures": [],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
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
            (summary_dir / "summary.json").write_text(
                json.dumps(
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
                        "pass_at_k": {},
                        **without_execution,
                        "job_failure": without_job_failure,
                        "trial_failures": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        lift: dict[str, Any] = {}
        paired_execution_succeeded = (
            with_execution.get("execution_status") == "succeeded"
            and without_execution.get("execution_status") == "succeeded"
        )
        if paired_execution_succeeded and with_scores and without_scores:
            lift = _compute_lift(with_scores, without_scores)
            (agent_dir / "lift.json").write_text(json.dumps(lift, indent=2), encoding="utf-8")

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
                (agent_dir / "custom_lift.json").write_text(json.dumps(custom_lift, indent=2), encoding="utf-8")

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
            (agent_dir / "pass_at_k_lift.json").write_text(json.dumps(pass_lift, indent=2), encoding="utf-8")

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
            (agent_dir / "security_attribution.json").write_text(
                json.dumps(security_attribution, indent=2), encoding="utf-8"
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
                "with_skill": with_pass,
                "without_skill": without_pass,
                "lift": pass_lift,
            },
            "security_attribution": security_attribution,
            "agent_runtime_failures": {
                "with_skill": with_runtime_failures,
                "without_skill": without_runtime_failures,
            },
            "trial_failures": {
                "with_skill": with_trial_failures,
                "without_skill": without_trial_failures,
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
            "num_trials_with": len(with_rewards),
            "num_trials_without": len(without_rewards) if not skip_baseline else 0,
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
