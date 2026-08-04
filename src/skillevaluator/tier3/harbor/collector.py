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
import os
import re
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    LEGACY_METRICS,
    average_custom_metrics,
    average_metrics,
    dimension_scores,
    extract_custom_metrics,
    overall_score,
    score_definition,
)
from skillevaluator.utils.redaction import redact_sensitive_data, redact_sensitive_text

logger = logging.getLogger(__name__)

DISPLAY_METRICS = DEFAULT_METRICS
DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES = 5 * 1024 * 1024
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


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


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

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, f"Harbor exited successfully but did not produce {result_path}"
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Harbor produced an unreadable job result at {result_path}: {exc}"

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
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
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
        include_text_logs=bool(exception_reason),
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


def _trial_failure_reason(trial_dir: Path) -> str:
    """Return the failure recorded for any incomplete Harbor trial."""
    _, exception_reason = _trial_exception_details(trial_dir)
    if exception_reason:
        return exception_reason
    exception_file = trial_dir / "exception.txt"
    if not exception_file.exists():
        return ""
    try:
        lines = [line.strip() for line in exception_file.read_text(encoding="utf-8", errors="replace").splitlines()]
    except OSError:
        return ""
    reason = next((line for line in reversed(lines) if line), "")
    return f"HarborTrialError: {reason}"[:600] if reason else ""


def _extract_trial_failures(job_dir: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
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
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
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
    return max(0, value)


def _redacted_artifact_text(src: Path, text: str) -> str:
    if src.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return redact_sensitive_text(text)
        return json.dumps(redact_sensitive_data(data), indent=2)
    return redact_sensitive_text(text)


def _write_artifact_manifest(trial_out: Path, manifest: dict[str, Any]) -> None:
    try:
        (trial_out / "artifact_manifest.json").write_text(
            json.dumps(redact_sensitive_data(manifest), indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.debug("Failed to write Harbor artifact manifest %s: %s", trial_out, e)


def _write_redacted_text_copy(src: Path, dest: Path) -> tuple[bool, dict[str, Any] | None]:
    """Copy a Harbor text artifact while masking common credential shapes."""
    max_bytes = _diagnostic_artifact_max_bytes()
    try:
        size_bytes = src.stat().st_size
    except OSError as e:
        logger.debug("Failed to stat Harbor artifact %s: %s", src, e)
        return False, {"name": src.name, "reason": "stat_failed"}
    if max_bytes and size_bytes > max_bytes:
        return False, {
            "name": src.name,
            "reason": "exceeds_max_bytes",
            "size_bytes": size_bytes,
            "max_bytes": max_bytes,
        }
    try:
        text = src.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Failed to read Harbor artifact %s: %s", src, e)
        return False, {"name": src.name, "reason": "read_failed", "size_bytes": size_bytes}
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_redacted_artifact_text(src, text), encoding="utf-8")
    except OSError as e:
        logger.debug("Failed to write Harbor artifact %s: %s", dest, e)
        return False, {"name": src.name, "reason": "write_failed", "size_bytes": size_bytes}
    return True, {"name": src.name, "size_bytes": size_bytes}


def _copy_trial_artifacts(trial_dir: Path, trial_out: Path) -> list[str]:
    copied: list[str] = []
    manifest: dict[str, Any] = {"copied": [], "skipped": []}
    for artifact_name in TRIAL_DIAGNOSTIC_ARTIFACTS:
        src = trial_dir / artifact_name
        if src.exists():
            ok, record = _write_redacted_text_copy(src, trial_out / artifact_name)
            if ok:
                copied.append(artifact_name)
                manifest["copied"].append(record)
            elif record:
                manifest["skipped"].append(record)

    agent_logs = trial_dir / "agent"
    for artifact_name in AGENT_LOG_ARTIFACTS:
        src = agent_logs / artifact_name
        if src.exists():
            ok, record = _write_redacted_text_copy(src, trial_out / artifact_name)
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
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to read %s: %s", result_file, e)
            result = {}
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
        if exception_file.exists():
            try:
                lines = [
                    line.strip()
                    for line in exception_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.strip()
                ]
            except OSError as e:
                logger.debug("Failed to read %s: %s", exception_file, e)
                lines = []
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
        if not trial_src.is_dir() or trial_src.name in scored_trials or not _looks_like_trial_dir(trial_src):
            continue

        trial_out = trials_dir / trial_src.name
        trial_out.mkdir(parents=True, exist_ok=True)
        copied = _copy_trial_artifacts(trial_src, trial_out)
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
    if not steps_dir.exists():
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
        path = steps_dir / step_name / "agent" / "trajectory.json"
        if path.exists():
            ordered_paths.append(path)
            seen.add(path)

    for path in sorted(steps_dir.glob("*/agent/trajectory.json")):
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
        for key, value in skill_evaluator_reward.items():
            if key in data and isinstance(data.get(key), int | float) and not isinstance(data.get(key), bool):
                continue
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


def _extract_rewards(job_dir: Path) -> list[dict[str, Any]]:
    """Extract reward.json from all trials in a job directory."""
    rewards: list[dict[str, Any]] = []
    scored_trial_roots: set[Path] = set()
    for reward_file in sorted(job_dir.rglob("reward.json")):
        if reward_file.parent.name == "verifier":
            try:
                data = json.loads(reward_file.read_text(encoding="utf-8"))
                _merge_reward_sidecars(data, reward_file.parent)
                trial_dir, trial_name, step_name = _reward_trial_context(reward_file)
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
                    try:
                        result = json.loads(result_file.read_text(encoding="utf-8"))
                        data["_started_at"] = result.get("started_at")
                        if not data.get("entry_id"):
                            entry_id = _entry_id_from_harbor_result(result)
                            if entry_id:
                                data["entry_id"] = entry_id
                    except (json.JSONDecodeError, OSError) as e:
                        logger.debug("Failed to read %s: %s", result_file, e)
                traj_file = _reward_trajectory_path(trial_dir, step_name)
                if traj_file.exists():
                    data["_has_trajectory"] = True
                rewards.append(data)
                scored_trial_roots.add(trial_dir)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", reward_file, e)

    for result_file in sorted(job_dir.glob("*/result.json")):
        trial_dir = result_file.parent
        if (
            trial_dir in scored_trial_roots
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
        if overall_score(reward) is not None:
            scoreable.append(reward)
            continue
        trial = str(reward.get("_trial_name") or reward.get("_trial_root_name") or "unknown trial")
        if trial in failed_trials:
            continue
        failed_trials.add(trial)
        failures.append(
            {
                "trial": trial,
                "reason": "Reward metrics are incomplete or non-finite; trial was not scored",
            }
        )
    return scoreable, failures


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
    """Collapse multi-step reward rows to one pass@k score per Harbor trial root."""
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
        scores = [_overall_score(row) for row in rows]
        logical.append(
            {
                "entry_id": first.get("entry_id"),
                "overall": (
                    sum(score for score in scores if score is not None) / len(scores)
                    if all(score is not None for score in scores)
                    else None
                ),
                "_trial_name": root,
                "_trial_root_name": root,
                "_started_at": first.get("_started_at"),
            }
        )
    return logical


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

    return {
        "k": n_attempts,
        "pass_threshold": pass_threshold,
        "stop_on_pass": stop_on_pass,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "total_cases": total_cases,
        "rate": rate,
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


def _save_trials(
    rewards: list[dict[str, Any]],
    trials_dir: Path,
    job_dir: Path | None,
    *,
    agent: str,
    variant: str,
    agent_model: str | None = None,
    agent_model_source: str | None = None,
) -> None:
    """Save per-trial reward.json and trajectory.json into the results directory."""
    trials_dir.mkdir(parents=True, exist_ok=True)
    for reward in rewards:
        trial_name = reward.get("_trial_name", "unknown")
        trial_out = trials_dir / trial_name
        trial_out.mkdir(parents=True, exist_ok=True)
        trial_root_name = reward.get("_trial_root_name", trial_name)
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
        if not clean_reward.get("entry_id"):
            clean_reward["entry_id"] = _entry_id(reward)
        clean_reward["agent"] = agent
        if agent_model:
            clean_reward["model"] = agent_model
        if agent_model_source:
            clean_reward["model_source"] = agent_model_source
        (trial_out / "reward.json").write_text(json.dumps(clean_reward, indent=2), encoding="utf-8")

        if trial_src:
            _copy_trial_artifacts(trial_src, trial_out)
        if merged_traj:
            (trial_out / "trajectory.json").write_text(
                json.dumps(redact_sensitive_data(merged_traj), indent=2),
                encoding="utf-8",
            )
        elif src_traj and src_traj.exists():
            _write_redacted_text_copy(src_traj, trial_out / "trajectory.json")

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
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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

    for agent in agents:
        model_info = agent_models.get(agent, {}) if agent_models else {}
        agent_model = model_info.get("model")
        agent_model_source = model_info.get("source")
        agent_dir = output_dir / agent
        agent_dir.mkdir(parents=True, exist_ok=True)

        with_job_name = f"{skill_name}-{agent}-with"
        with_job_dir = _find_job_dir(jobs_dir, with_job_name)

        with_rewards: list[dict[str, Any]] = []
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
            with_rewards = _extract_rewards(with_job_dir) if with_job_ok or preserve_partial else []
            with_rewards, invalid_score_failures = _partition_scoreable_rewards(with_rewards)
            with_trial_failures.extend(invalid_score_failures)
            with_scores, with_metric_set, with_metrics = average_metrics(with_rewards)
            all_results["metric_set"] = with_metric_set
            all_results["metrics"] = list(with_metrics)
            all_results["attempt_policy"]["score_definition"] = score_definition(with_metrics)
            with_custom_scores = average_custom_metrics(with_rewards)
            with_pass = _pass_summary(
                with_rewards,
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
                stop_on_pass=stop_on_pass,
                pass_threshold=pass_threshold,
            )
            _save_trials(
                with_rewards,
                agent_dir / "with-skill" / "trials",
                with_job_dir,
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

        without_rewards: list[dict[str, Any]] = []
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
                without_rewards = _extract_rewards(without_job_dir) if without_job_ok or preserve_partial else []
                without_rewards, invalid_score_failures = _partition_scoreable_rewards(without_rewards)
                without_trial_failures.extend(invalid_score_failures)
                without_scores, without_metric_set, without_metrics = average_metrics(without_rewards)
                without_custom_scores = average_custom_metrics(without_rewards)
                without_pass = _pass_summary(
                    without_rewards,
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
                    stop_on_pass=stop_on_pass,
                    pass_threshold=pass_threshold,
                )
                _save_trials(
                    without_rewards,
                    agent_dir / "without-skill" / "trials",
                    without_job_dir,
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
                    "Agent %s without-skill: %d trials, scores=%s", agent, len(without_rewards), without_scores
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
        if with_scores and without_scores:
            lift = _compute_lift(with_scores, without_scores)
            (agent_dir / "lift.json").write_text(json.dumps(lift, indent=2), encoding="utf-8")

        custom_lift: dict[str, Any] = {}
        if (
            with_rewards
            and without_rewards
            and (with_custom_scores or without_custom_scores or (not with_scores and not without_scores))
        ):
            custom_lift = _compute_custom_lift(
                with_custom_scores,
                without_custom_scores,
                with_rewards,
                without_rewards,
                include_overall=not with_scores and not without_scores,
            )
            if custom_lift:
                (agent_dir / "custom_lift.json").write_text(json.dumps(custom_lift, indent=2), encoding="utf-8")

        pass_lift: dict[str, Any] = {}
        if with_pass and without_pass:
            pass_lift = {
                "with_skill": with_pass.get("rate", 0.0),
                "without_skill": without_pass.get("rate", 0.0),
                "delta": round(with_pass.get("rate", 0.0) - without_pass.get("rate", 0.0), 4),
                "passed_cases_delta": int(with_pass.get("passed_cases", 0)) - int(without_pass.get("passed_cases", 0)),
            }
            (agent_dir / "pass_at_k_lift.json").write_text(json.dumps(pass_lift, indent=2), encoding="utf-8")

        security_attribution: dict[str, Any] = {}
        if with_rewards:
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

    (output_dir / "attempt_policy.json").write_text(
        json.dumps(all_results["attempt_policy"], indent=2), encoding="utf-8"
    )

    if len(agents) > 1:
        comparison = _build_comparison(all_results["agents"])
        all_results["comparison"] = comparison
        (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

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
