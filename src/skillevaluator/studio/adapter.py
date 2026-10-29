# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project native evaluator results for Studio without inventing evaluations.

This adapter reads only the evaluator's contained, published trial artifacts.
Scores, lift and quality decisions stay owned by the existing evaluator. Missing
measurements and unidentified cases remain explicitly unavailable.
"""

from __future__ import annotations

import json
import math
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from skillevaluator.utils.secure_fs import SecureRoot, stat_is_link_or_reparse

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TRIALS = 1000
MAX_ARTIFACTS = 4000
_ARMS = {"with_skill": "with-skill", "without_skill": "without-skill"}
_TRIAL_ARTIFACTS = ("reward.json", "failure.json", "trajectory.json", "result.json", "artifact_manifest.json")


def is_native_custom_only(skill_path: Path) -> bool:
    """Mirror the native task/custom-only dataset exemption without Harbor."""
    from skillevaluator.tier3.evals_config import load_evals_config

    try:
        config, _path = load_evals_config(skill_path)
    except (OSError, ValueError):
        return False
    harbor = config.get("harbor")
    grading = config.get("grading")
    source = skill_path / "evals" / "harbor"
    return (
        isinstance(harbor, Mapping)
        and isinstance(grading, Mapping)
        and harbor.get("task_source") == "native_harbor"
        and grading.get("mode") == "custom_only"
        and source.is_dir()
        and not source.is_symlink()
    )


def finite_number(value: Any) -> int | float | None:
    """Return an actual finite JSON number; booleans are not measurements."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def json_safe(value: Any, *, secret_values: tuple[str, ...] = ()) -> Any:
    """Make a strict, redacted JSON value for browser display and persistence."""
    from skillevaluator.utils.redaction import redact_sensitive_data

    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 60:
            return "<depth limit>"
        if isinstance(item, Mapping):
            return {clean(str(key), depth + 1): clean(child, depth + 1) for key, child in item.items()}
        if isinstance(item, list | tuple):
            return [clean(child, depth + 1) for child in item]
        if isinstance(item, str):
            for secret in secret_values:
                if secret:
                    item = item.replace(secret, "<redacted>")
            return item
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int | float):
            return finite_number(item)
        if isinstance(item, Path):
            return str(item)
        return str(item)

    original = clean(value)
    redacted = clean(redact_sensitive_data(original))

    def retain_measurements(source: Any, target: Any) -> None:
        if isinstance(source, dict) and isinstance(target, dict):
            for key, item in source.items():
                if key not in target:
                    continue
                # Native comparison counters are measurements, never credentials.
                if (
                    key.startswith("resources.tokens.")
                    and isinstance(item, dict)
                    and set(item) == {"before", "after", "delta"}
                    and all(v is None or finite_number(v) is not None for v in item.values())
                ):
                    target[key] = item
                elif (
                    key == "tokens"
                    and isinstance(item, dict)
                    and set(item) == {"with_skill", "baseline", "delta"}
                    and (item["delta"] is None or finite_number(item["delta"]) is not None)
                ):
                    measures = [item[arm] for arm in ("with_skill", "baseline")]
                    if all(
                        isinstance(m, dict)
                        and set(m) == {"status", "measured_trials", "trials", "total"}
                        and m["status"] in {"complete", "unavailable"}
                        and all(v is None or finite_number(v) is not None for k, v in m.items() if k != "status")
                        for m in measures
                    ):
                        target[key] = item
                elif key in target:
                    retain_measurements(item, target[key])
        elif isinstance(source, list) and isinstance(target, list):
            for left, right in zip(source, target, strict=True):
                retain_measurements(left, right)

    retain_measurements(original, redacted)
    return redacted


def read_json(path: Path, *, root: Path, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Read bounded JSON without following links anywhere below a pinned root."""
    relative = path.absolute().relative_to(root.absolute())
    with SecureRoot(root) as secure_root:
        payload, _metadata = secure_root.read_bytes(relative, max_bytes)
    return json.loads(payload, parse_constant=_invalid_number)


def _invalid_number(token: str) -> None:
    raise ValueError(f"Non-finite JSON number: {token}")


def _optional_json(path: Path, root: Path, notices: list[str]) -> Any:
    try:
        return read_json(path, root=root)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as exc:
        # SecureRoot wraps a missing selected artifact in SecurePathError.
        # Optional failure/trajectory/findings files may legitimately be absent.
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        notices.append(f"Could not read {path.relative_to(root).as_posix()}: {type(exc).__name__}")
        return None


def _directories(path: Path, *, root: Path) -> list[Path]:
    """List actual contained directories; selected-file reads remain anchored."""
    try:
        path.absolute().relative_to(root.absolute())
        for parent in (path, *path.parents):
            if parent == root.parent:
                break
            metadata = parent.lstat()
            if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                return []
        return sorted(
            child
            for child in path.iterdir()
            if stat.S_ISDIR((metadata := child.lstat()).st_mode) and not stat_is_link_or_reparse(metadata)
        )
    except (OSError, ValueError):
        return []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _final_response(trajectory: Any) -> str | None:
    from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import atif_content_has_text, atif_content_text

    if not isinstance(trajectory, Mapping):
        return None
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, Mapping) or step.get("source", step.get("role")) not in ("agent", "assistant"):
            continue
        if step.get("tool_calls"):
            continue
        message = step.get("message", step.get("content"))
        if isinstance(message, str | list) and atif_content_has_text(message):
            return atif_content_text(message)
    return None


def _metrics(reward: Mapping[str, Any]) -> dict[str, int | float]:
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, extract_custom_metrics, metric_value, score_value

    metrics = {metric: value for metric in DEFAULT_METRICS if (value := metric_value(dict(reward), metric)) is not None}
    if (overall := score_value(reward.get("overall"))) is not None:
        metrics["overall"] = overall
    for name, value in extract_custom_metrics(dict(reward)).items():
        if finite_number(value) is not None:
            metrics.setdefault(name, value)
    return metrics


def _checks(reward: Mapping[str, Any], evidence: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Preserve explicit native predicates; numeric scores are not hard passes."""
    details = reward.get("details")
    if not isinstance(details, Mapping):
        return []
    checks: list[dict[str, Any]] = []

    def escape(part: Any) -> str:
        return str(part).replace("~", "~0").replace("/", "~1")

    def add(name: str, detail: Any, pointer: str, depth: int = 0) -> None:
        if not isinstance(detail, Mapping):
            return
        if depth > 8:
            return
        passed = detail.get("passed", detail.get("achieved"))
        status = (
            "passed"
            if passed is True
            else "failed"
            if passed is False
            else "scored"
            if finite_number(detail.get("score")) is not None
            else "unknown"
        )
        refs = [{**ref, "json_pointer": pointer} for ref in evidence if ref.get("kind") == "reward"]
        native_refs = detail.get("evidence_refs")
        if isinstance(native_refs, list):
            refs.extend(dict(ref) for ref in native_refs if isinstance(ref, Mapping))
        if any(key in detail for key in ("passed", "achieved", "score", "reason", "summary")):
            checks.append(
                {
                    "name": name,
                    "status": status,
                    "score": finite_number(detail.get("score")),
                    "reason": _text(detail.get("reason")) or _text(detail.get("summary")),
                    "evidence": refs,
                }
            )
        criteria = detail.get("criteria")
        if isinstance(criteria, Mapping):
            for criterion, outcome in criteria.items():
                if isinstance(outcome, bool):
                    add(
                        f"{name}: {criterion}",
                        {"passed": outcome, "reason": detail.get("reason")},
                        f"{pointer}/criteria/{escape(criterion)}",
                        depth + 1,
                    )
        results = detail.get("results")
        if isinstance(results, list):
            for index, result in enumerate(results):
                if isinstance(result, Mapping):
                    add(
                        f"{name}: step {result.get('step', index + 1)}", result, f"{pointer}/results/{index}", depth + 1
                    )
        for child_name, child in detail.items():
            if child_name in {"criteria", "results", "evidence_refs"} or not isinstance(child, Mapping):
                continue
            if child_name in {"checks", "details"}:
                for check_name, check in child.items():
                    if isinstance(check, Mapping):
                        add(f"{name}: {check_name}", check, f"{pointer}/{child_name}/{escape(check_name)}", depth + 1)
            else:
                # Native execution/efficiency details have direct child maps
                # (activation, script_execution, error_recovery, routing, ...).
                add(f"{name}: {child_name}", child, f"{pointer}/{escape(child_name)}", depth + 1)

    for metric, detail in details.items():
        add(str(metric), detail, f"/details/{escape(metric)}")
    return checks


def normalize_trial(
    reward: Mapping[str, Any],
    *,
    agent: str,
    arm: str,
    failure: Mapping[str, Any] | None = None,
    trajectory: Any = None,
    trial_id: str | None = None,
    evidence: list[dict[str, str]] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Normalize one physical trial without guessing its case or pass threshold."""
    from skillevaluator.tier3.harbor.metrics import overall_score

    failure = failure or {}
    evidence = evidence or []
    case_id = _text(reward.get("entry_id")) or _text(reward.get("case_id")) or _text(failure.get("entry_id"))
    identity = _text(reward.get("trial_id")) or trial_id or _text(failure.get("trial"))
    error = failure.get("error") or reward.get("error") or reward.get("evaluation_errors")
    recorded_status = str(reward.get("status", "")).lower()
    failed = (
        bool(failure.get("error"))
        or str(reward.get("evaluation_status", "")).lower() in {"failed", "error"}
        or recorded_status in {"failed", "error"}
        or bool(reward.get("evaluation_errors"))
    )
    metrics = _metrics(reward)
    status = (
        "failed"
        if failed
        else "scored"
        if not failure and recorded_status != "unscored" and overall_score(dict(reward)) is not None
        else "unscored"
    )
    trial = {
        "id": identity,
        "physical_trial_id": identity,
        "agent": agent,
        "arm": arm,
        "case_id": case_id,
        "identity_status": "recorded" if case_id is not None else "unknown",
        "status": status,
        "metrics": metrics if status == "scored" else {},
        "diagnostic_metrics": metrics if status != "scored" else {},
        "checks": _checks(reward, evidence),
        "final_response": _text(reward.get("final_response")) or _final_response(trajectory),
        "error": error,
        "evidence": evidence,
    }
    if reward.get("attempt") is not None:
        trial["attempt"] = reward["attempt"]
    return case_id, trial


def normalize_evaluation(raw: Mapping[str, Any], *, job_dir: Path, skill_path: Path) -> dict[str, Any]:
    """Render a native run, preserving its unknowns and original score policies."""
    from skillevaluator.tier3.dataset_utils import find_eval_file, load_dataset_entries

    notices: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    unidentified: list[dict[str, Any]] = []
    dataset_file = find_eval_file(skill_path)
    if dataset_file is not None:
        try:
            for entry in load_dataset_entries(dataset_file):
                case_id = _text(entry.get("id"))
                if case_id is not None:
                    cases[case_id] = {
                        "id": case_id,
                        "question": entry.get("question"),
                        "arms": {arm: [] for arm in _ARMS},
                    }
        except (OSError, ValueError, RecursionError):
            notices.append("The authored dataset could not be loaded for case labels.")
    artifacts: list[dict[str, str]] = []

    def artifact(path: Path, kind: str) -> dict[str, str] | None:
        if len(artifacts) >= MAX_ARTIFACTS:
            return None
        try:
            relative = path.absolute().relative_to(job_dir.absolute())
            with SecureRoot(job_dir) as secure_root:
                secure_root.read_bytes(relative, MAX_JSON_BYTES)
        except (OSError, ValueError):
            return None
        ref = {"name": path.name, "path": relative.as_posix(), "kind": kind}
        artifacts.append(ref)
        return ref

    def add_trial(case_id: str | None, trial: dict[str, Any], arm: str) -> None:
        if case_id is None:
            case = {"id": None, "question": None, "identity_status": "unknown", "arms": {item: [] for item in _ARMS}}
            case["arms"][arm].append(trial)
            unidentified.append(case)
            return
        case = cases.setdefault(case_id, {"id": case_id, "question": None, "arms": {item: [] for item in _ARMS}})
        case["arms"][arm].append(trial)

    run_dir: Path | None = None
    recorded_dir = raw.get("run_dir")
    if isinstance(recorded_dir, str):
        candidate = Path(recorded_dir)
        if candidate.is_absolute() and candidate != job_dir:
            try:
                candidate.relative_to(job_dir)
                if candidate in _directories(candidate.parent, root=job_dir):
                    run_dir = candidate
            except ValueError:
                notices.append("Recorded run artifacts were outside this job and were not read.")

    agents = raw.get("agents")
    agents = agents if isinstance(agents, Mapping) else {}
    suggestions: list[dict[str, Any]] = []
    found_suggestions = False
    count = 0
    for agent, agent_info in agents.items():
        if not isinstance(agent_info, Mapping):
            continue
        for arm, variant in _ARMS.items():
            # Native in-process results normally keep trials on disk. Also accept
            # explicit trial lists from native consolidated reports for fixtures.
            trial_key = "trials" if arm == "with_skill" else "trials_baseline"
            inline = agent_info.get(trial_key)
            if isinstance(inline, list):
                for reward in inline[: MAX_TRIALS - count]:
                    if isinstance(reward, Mapping):
                        case_id, trial = normalize_trial(reward, agent=str(agent), arm=arm)
                        add_trial(case_id, trial, arm)
                        count += 1
                continue
            if run_dir is None or str(agent) not in {path.name for path in _directories(run_dir, root=job_dir)}:
                continue
            agent_dir = run_dir / str(agent)
            artifact(agent_dir / variant / "summary.json", "summary")
            for trial_dir in _directories(agent_dir / variant / "trials", root=job_dir):
                if count >= MAX_TRIALS:
                    notices.append(f"Trial display limited to {MAX_TRIALS} artifacts; native results remain available.")
                    break
                reward = _optional_json(trial_dir / "reward.json", job_dir, notices)
                failure = _optional_json(trial_dir / "failure.json", job_dir, notices)
                result = _optional_json(trial_dir / "result.json", job_dir, notices)
                if isinstance(result, Mapping) and result.get("exception_info") and not failure:
                    failure = {"error": result["exception_info"]}
                trajectory = _optional_json(trial_dir / "trajectory.json", job_dir, notices)
                refs = [
                    ref
                    for name in _TRIAL_ARTIFACTS
                    if (ref := artifact(trial_dir / name, name.removesuffix(".json"))) is not None
                ]
                case_id, trial = normalize_trial(
                    reward if isinstance(reward, Mapping) else {},
                    agent=str(agent),
                    arm=arm,
                    failure=failure if isinstance(failure, Mapping) else None,
                    trajectory=trajectory,
                    trial_id=trial_dir.name,
                    evidence=refs,
                )
                add_trial(case_id, trial, arm)
                count += 1
        if run_dir is not None:
            findings_path = run_dir / str(agent) / "findings.json"
            findings = _optional_json(findings_path, job_dir, notices)
            if isinstance(findings, Mapping):
                found_suggestions = True
                artifact(findings_path, "findings")
                items = findings.get("suggestions_v2") or findings.get("suggestions")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, Mapping):
                            suggestions.append({**item, "agent": str(agent), "mode": findings.get("suggestion_mode")})
                        elif isinstance(item, str):
                            suggestions.append(
                                {
                                    "suggestion": item,
                                    "agent": str(agent),
                                    "mode": findings.get("suggestion_mode"),
                                    "evidence_refs": [],
                                }
                            )
    if run_dir is not None:
        for name in ("result.json", "run_config.json", "dataset_snapshot.json", "attempt_policy.json", "report.html"):
            artifact(run_dir / name, "report" if name.endswith("html") else "native")
    only_agent = next(iter(agents.values())) if len(agents) == 1 else None
    only_agent = only_agent if isinstance(only_agent, Mapping) else {}
    lift = only_agent.get("lift")
    overall_lift = lift.get("overall") if isinstance(lift, Mapping) else None
    succeeded = raw.get("execution_status") == "succeeded"
    signals = None
    if run_dir is not None:
        from skillevaluator.tier3.improvement_signals import load_signals

        try:
            packet = load_signals(run_dir)
            signals = {key: packet[key] for key in ("schema_version", "dataset_digest", "evidence_issues", "agents")}
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            notices.append(
                f"Improvement evidence is unavailable: {type(exc).__name__}. Retained native reports remain available."
            )
    return {
        "kind": "evaluation",
        "execution_status": raw.get("execution_status", "unknown"),
        "error": raw.get("error") or raw.get("execution_errors") or None,
        "summary": {
            "expected_attempts": finite_number(raw.get("expected_attempts")),
            "scored_attempts": finite_number(raw.get("scored_attempts")),
            "with_skill": finite_number(only_agent.get("overall_with_skill")) if succeeded else None,
            "without_skill": finite_number(only_agent.get("overall_without_skill")) if succeeded else None,
            "lift": finite_number(overall_lift.get("delta"))
            if succeeded and isinstance(overall_lift, Mapping)
            else None,
            "precision": raw.get("precision"),
            "precision_note": "Uncertainty is not estimated by this UI; inspect the native comparison and coverage.",
        },
        "cases": [*cases.values(), *unidentified],
        "suggestions": suggestions,
        "suggestions_status": "available" if found_suggestions else "not_generated",
        "optimizer": {
            "available": True,
            "reason": "Evidence-backed proposals are available in Improve. Candidates require fresh evaluation and native comparison.",
        },
        "improvement_signals": signals,
        "artifacts": artifacts,
        "notes": list(dict.fromkeys(notices)),
        "raw": dict(raw),
    }
