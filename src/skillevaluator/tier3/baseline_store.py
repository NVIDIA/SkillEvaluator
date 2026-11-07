# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparison identities and validated, retained baseline evidence."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from skillevaluator.tier3.results_location import ordered_run_directories, run_directory_sort_key

_NON_COMPARISON_FIELDS = frozenset(
    {
        "n_concurrent",
        "jobs_retained",
        "credential_validation",
        "evaluated_source",
        "config_file",
        "private_comparisons",
        "baseline",
        "reuse_baseline",
        "compare_to",
        "prior_version_comparison",
    }
)
BASELINE_ANNOTATION_KEYS = (
    "_trial_name",
    "_trial_root_name",
    "_step_name",
    "_started_at",
    "_attempt_ordinal",
    "_trusted_task_selector",
    "_arm_suffix",
    "_authoritative_entry_id",
    "_result_entry_id",
    "_trusted_case_identity_unresolved",
    "_judge_models",
    "_trajectory_summary",
)


def measured_condition_usage(condition_dir: Path) -> dict[str, float | None]:
    """Sum observed trajectory counters; missing measurements remain unavailable."""
    fields = ("total_prompt_tokens", "total_completion_tokens", "total_cached_tokens", "total_cost_usd")
    values: dict[str, list[float]] = {field: [] for field in fields}
    seen = set()
    complete = True
    for reward_path in sorted((condition_dir / "trials").glob("*/reward.json")):
        try:
            reward = json.loads(reward_path.read_text(encoding="utf-8"))
            trial_id = reward["trial_id"]
            if trial_id in seen:
                continue
            seen.add(trial_id)
            trajectory = json.loads(reward_path.with_name("trajectory.json").read_text(encoding="utf-8"))
            metrics = trajectory["final_metrics"]
            for field in fields:
                value = metrics.get(field)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    values[field].append(value)
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            complete = False
    return {
        field: sum(counters) if complete and counters and len(counters) == len(seen) else None
        for field, counters in values.items()
    }


def comparison_identity(run_config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Keep unknown configuration relevant by default, separating skill identity."""

    def relevant(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: relevant(item) for key, item in value.items() if key not in _NON_COMPARISON_FIELDS}
        if isinstance(value, list):
            return [relevant(item) for item in value]
        return value

    config = relevant(run_config)
    workspace = config.get("skill_workspace", {})
    skill = {
        "evaluated_source": copy.deepcopy(run_config.get("evaluated_source")),
        "publication_target": copy.deepcopy(result.get("publication_target")),
    }
    if isinstance(workspace, dict) and "decoy_manifest" in workspace:
        skill["workspace_content"] = workspace.pop("decoy_manifest")
    return {
        "config": config,
        "dataset_digest": result.get("dataset_digest"),
        "expected_case_ids": sorted(result.get("expected_case_ids", [])),
        "attempt_policy": copy.deepcopy(result.get("attempt_policy")),
        "score_policy": result.get("score_policy"),
        "evaluator_version": result.get("evaluator_version"),
        "skill": skill,
    }


def comparison_mismatches(current: dict[str, Any], source: dict[str, Any]) -> list[str]:
    """Explain all baseline-relevant differences, ignoring skill revisions."""
    reasons: list[str] = []
    current_config, source_config = current["config"], source["config"]
    for agent, model in current_config.get("agents", {}).items():
        old = source_config.get("agents", {}).get(agent)
        if old != model:
            reasons.append(f"model changed for {agent}: {old} -> {model}")
    added = sorted(set(current.get("expected_case_ids", [])) - set(source.get("expected_case_ids", [])))
    removed = sorted(set(source.get("expected_case_ids", [])) - set(current.get("expected_case_ids", [])))
    if current.get("dataset_digest") != source.get("dataset_digest") or added or removed:
        reasons.append(f"dataset changed (cases added: {added[:16]}; removed: {removed[:16]})")
    for field in ("attempt_policy", "score_policy", "evaluator_version"):
        if field == "score_policy" and current.get(field) is None:
            continue
        if current.get(field) != source.get(field):
            reasons.append(f"{field.replace('_', ' ')} changed: {source.get(field)} -> {current.get(field)}")
    for field in sorted(set(current_config) | set(source_config)):
        if field == "agents":
            continue
        if current_config.get(field) != source_config.get(field):
            label = {
                "harbor": "environment/attempt policy",
                "judge": "judge/scorer",
                "provider": "judge provider/model",
            }.get(field, field)
            reasons.append(f"{label} changed: {source_config.get(field)} -> {current_config.get(field)}")
    return reasons


def load_baseline_rows(run_dir: Path, agent: str) -> list[dict[str, Any]]:
    """Load the retained without-skill arm."""
    return load_condition_rows(run_dir, agent, "without_skill")


def load_condition_rows(run_dir: Path, agent: str, arm: str) -> list[dict[str, Any]]:
    """Restore collector-owned annotations from their public persistence envelope."""
    if arm not in {"with_skill", "without_skill"}:
        raise ValueError("Unknown retained condition")
    paths = sorted((run_dir / agent / arm.replace("_", "-") / "trials").glob("*/reward.json"))
    if not paths:
        raise ValueError(f"{agent}: missing {arm.replace('_', '-')} reward rows")
    rows = []
    for path in paths:
        try:
            if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError("linked baseline evidence")
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("oversized baseline evidence")
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{agent}: corrupt reward.json ({path.parent.name})") from exc
        if not isinstance(row, dict) or not isinstance(row.get("baseline_evidence"), dict):
            raise ValueError(f"{agent}: reward.json lacks reusable case/attempt evidence")
        evidence = row.pop("baseline_evidence")
        if evidence.get("version") != 1 or not isinstance(evidence.get("annotations"), dict):
            raise ValueError(f"{agent}: unsupported baseline evidence")
        annotations = evidence["annotations"]
        if set(annotations) - set(BASELINE_ANNOTATION_KEYS) or not annotations.get("_trial_root_name"):
            raise ValueError(f"{agent}: invalid baseline case/attempt annotations")
        row.update(annotations)
        row["_authoritative_entry_id"] = True
        rows.append(row)
    return rows


def baseline_eligibility(
    run_dir: Path, agents: list[str], identity: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Require a fresh, complete baseline under the current result contract."""
    return condition_eligibility(run_dir, agents, identity, "without_skill", require_fresh=True)


def condition_eligibility(
    run_dir: Path,
    agents: list[str],
    identity: dict[str, Any],
    arm: str,
    *,
    require_fresh: bool = False,
    collected_result: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate a retained arm, or the runner's not-yet-published collection."""
    from skillevaluator.tier3.harbor.collector import _condition_execution_summary, _partition_scoreable_rewards

    reasons: list[str] = []
    try:
        result = (
            collected_result
            if collected_result is not None
            else json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        )
        if not isinstance(result, dict):
            raise ValueError("invalid result")
        config = (
            result.get("run_config")
            if collected_result is not None
            else json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        )
        if not isinstance(config, dict) or not isinstance(result, dict):
            raise ValueError("invalid result/config")
    except (OSError, ValueError):
        return ["missing or corrupt result/config"], {}
    if collected_result is None and run_directory_sort_key(run_dir, require_completed_result=True) is None:
        return ["run is not completed under the current result contract"], result
    baseline = result.get("baseline", config.get("baseline", {}))
    if require_fresh and (not isinstance(baseline, dict) or baseline.get("mode") != "fresh"):
        reasons.append("source run reused a baseline; select the original fresh run")
    source_identity = result.get("comparison_identity")
    if not isinstance(source_identity, dict):
        reasons.append("run lacks a persisted comparison identity")
    else:
        recorded = comparison_identity(config, {})["config"]
        cases = source_identity.get("expected_case_ids")
        if (
            source_identity.get("config") != recorded
            or not isinstance(cases, list)
            or not all(isinstance(case, str) for case in cases)
        ):
            reasons.append("source comparison identity disagrees with run configuration")
        else:
            reasons.extend(comparison_mismatches(identity, source_identity))
        if source_identity.get("dataset_digest") != result.get("dataset_digest") or source_identity.get(
            "evaluator_version"
        ) != result.get("evaluator_version"):
            reasons.append("source comparison identity disagrees with dataset/evaluator evidence")
        if source_identity.get("score_policy") != result.get("score_policy"):
            reasons.append("source comparison identity disagrees with score policy evidence")
        policy = result.get("attempt_policy", {})
        if source_identity.get("attempt_policy") != {
            key: policy.get(key) for key in ("max_attempts", "pass_threshold", "stop_on_pass")
        }:
            reasons.append("source comparison identity disagrees with attempt policy evidence")
    for agent in agents:
        payload = result.get("agents", {}).get(agent, {})
        condition = payload.get("conditions", {}).get(arm, {})
        if condition.get("execution_status") != "succeeded":
            reasons.append(f"{agent}: {arm.replace('_', '-')} condition did not succeed")
        if (
            payload.get("job_failures", {}).get(arm)
            or payload.get("trial_failures", {}).get(arm)
            or payload.get("agent_runtime_failures", {}).get(arm)
            or payload.get("historical_job_failures", {}).get(arm)
            or payload.get("recovered_trial_failures", {}).get(arm)
        ):
            reasons.append(f"{agent}: baseline has job, trial or runtime failures")
        try:
            summary = json.loads((run_dir / agent / arm.replace("_", "-") / "summary.json").read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("invalid baseline summary")
            if (
                summary.get("execution_status") != "succeeded"
                or summary.get("job_failure")
                or summary.get("trial_failures")
            ):
                reasons.append(f"{agent}: baseline summary is incomplete or failed")
            rows = load_condition_rows(run_dir, agent, arm)
            scored, failures = _partition_scoreable_rewards(rows)
            policy = identity["attempt_policy"]
            execution = _condition_execution_summary(
                scored,
                expected_case_ids=identity["expected_case_ids"],
                expected_cases=len(identity["expected_case_ids"]),
                n_attempts=policy["max_attempts"],
                job_failure="",
                stop_on_pass=policy["stop_on_pass"],
                pass_threshold=policy["pass_threshold"],
                reward_failures=failures,
            )
            if failures or execution["execution_status"] != "succeeded":
                reasons.append(f"{agent}: baseline reward coverage is incomplete or unscoreable")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            reasons.append(f"{agent}: missing or corrupt baseline evidence: {exc}")
    return reasons, result


def select_baseline(root: Path, requested: str, agents: list[str], identity: dict[str, Any]) -> dict[str, Any]:
    """Select fresh evidence, or return bounded rejection diagnostics for latest."""
    if requested != "latest" and (Path(requested).name != requested or requested in {".", ".."}):
        raise ValueError("Baseline run must be a run id under the results root")
    candidates = ordered_run_directories(root, completed_only=True) if requested == "latest" else [root / requested]
    rejected = []
    for candidate in candidates:
        reasons, result = baseline_eligibility(candidate, agents, identity)
        if reasons:
            if requested != "latest":
                raise ValueError(f"Cannot reuse baseline {requested}: " + "; ".join(reasons))
            if len(rejected) < 16:
                rejected.append({"run_id": candidate.name, "reasons": [reason[:1024] for reason in reasons[:16]]})
            continue
        return {
            "mode": "reused",
            "source_run_id": candidate.name,
            "source_evaluated_at": result.get("evaluated_at"),
            "source_run_dir": str(candidate),
            "source_identity": copy.deepcopy(result["comparison_identity"]),
        }
    return {"mode": "fresh", "rejected_candidates": rejected, "reuse_reason": "no compatible completed fresh baseline"}


def baseline_description(baseline: dict[str, Any]) -> str:
    """Render a compact provenance line shared by reports."""
    if baseline.get("mode") == "reused":
        return f"Baseline: reused from run {baseline.get('source_run_id')} ({baseline.get('source_evaluated_at')})"
    if baseline.get("mode") == "fresh" and baseline.get("reuse_reason"):
        return f"Baseline: fresh (reuse not possible: {baseline['reuse_reason']})"
    return f"Baseline: {baseline.get('mode', 'fresh')}"
