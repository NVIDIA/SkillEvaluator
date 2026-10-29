# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence for skill edits and comparison of explicit, retained evaluation runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from skillevaluator.tier3.eval_core.llm_judge import _configured_secret_values, _redact_configured_credentials
from skillevaluator.tier3.harbor.activation import summarize_skill_activation
from skillevaluator.tier3.harbor.metrics import (
    extract_custom_metrics,
    metric_set_for_reward,
    metric_value,
    overall_score,
)
from skillevaluator.tier3.harbor.report import _bounded_json_text, _detail_for_finding, _extract_findings
from skillevaluator.tier3.harbor.report_data import load_agent_data, load_dataset_snapshot, logical_trial_reward_groups
from skillevaluator.tier3.improve import RESERVED_SPLITS, retained_task_roots, snapshot_tree
from skillevaluator.utils.redaction import redact_sensitive_data


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _values(value: Any) -> Any:
    """Remove configuration provenance labels, retaining effective values."""
    if isinstance(value, dict):
        if "value" in value:
            return _values(value["value"])
        return {key: _values(item) for key, item in value.items() if key not in {"source", "occurrence"}}
    if isinstance(value, list):
        return [_values(item) for item in value]
    return value


def _attempt(reward: dict, agent: str) -> dict:
    evaluation_errors = {key: reward[key] for key in ("error", "evaluation_errors") if reward.get(key)}
    evaluation_failed = bool(evaluation_errors) or str(reward.get("evaluation_status") or "").casefold() in {
        "error", "failed"
    }
    metric_set, metrics = metric_set_for_reward(reward)
    scores = {name: metric_value(reward, name) for name in metrics}
    scores["overall"] = overall_score(reward)
    scores.update(extract_custom_metrics(reward))
    if evaluation_failed:
        scores = {}  # Error-envelope sentinel values are not measured performance.
    findings = []
    # The CLI's summary cutoff must not discard partial failures from edit evidence.
    for finding in _extract_findings([reward], passing_score=1.0) if not evaluation_failed else []:
        if finding["severity"] != "ok":
            # Display summaries can omit the causal error; retain bounded grader detail separately.
            detail = _redact_configured_credentials(
                json.dumps(redact_sensitive_data(_detail_for_finding(reward, finding["metric"]))),
                extra_secret_values=tuple(json.dumps(value)[1:-1] for value in _configured_secret_values()),
            )
            excerpt = _bounded_json_text(detail, max_encoded_bytes=16 * 1024)
            findings.append(
                {
                    **finding,
                    "id": f"{agent}:{reward['entry_id']}:{reward.get('trial_id')}:{finding['metric']}",
                    "diagnostic": {"text": excerpt, "truncated": excerpt != detail},
                }
            )
    details = reward.get("details")
    details = details if isinstance(details, dict) else {}
    return {
        "trial_id": reward.get("trial_id"),
        "evaluation_status": "failed" if evaluation_failed else reward.get("evaluation_status"),
        "evaluation_errors": evaluation_errors,
        "skill_activation": summarize_skill_activation([{}] if evaluation_failed or not metrics else [reward]),
        "scores": {name: value for name, value in scores.items() if value is not None},
        "score_contract": {"metric_set": metric_set, "score_policy": reward.get("score_policy")},
        "findings": findings,
        **({"agent_observations": [
            {**observation, "id": f"{agent}:{reward['entry_id']}:{reward.get('trial_id')}:agent_observation"}
            for observation in reward["_agent_observations"]
        ]} if reward.get("_agent_observations") and not evaluation_failed else {}),
        "missing_scores": [
            name
            for name in (*metrics, *(("overall",) if metrics or "overall" in reward else ()))
            if scores.get(name) is None
        ],
        "usage": reward.get("_traj", {}),
        "judges": {
            name: {
                key: detail.get(key) for key in ("judge_model_used", "judge_reasoning_effort", "judge_fallback_used")
            }
            for name, detail in details.items()
            if isinstance(detail, dict) and detail.get("judge_model_used")
        },
    }


def _verification_context(run_dir: Path, skill_name: str | None) -> dict:
    """Freeze all staged task inputs and graders except the evaluated skill itself."""
    skill_directories = {task / "environment" / "skills" for task in retained_task_roots(run_dir)}

    def ignore(directory: str, names: list[str]) -> list[str]:
        return [skill_name] if Path(directory) in skill_directories and skill_name in names else []

    return snapshot_tree(run_dir / "_harbor-tasks", ignore=ignore, include_generated=True)


def _runtime_context(run_dir: Path) -> dict:
    """Compare effective resource/backend options and actual agent versions."""
    context = {}
    for path in sorted((run_dir / "_harbor-jobs").glob("*/config.json")):
        config = json.loads(path.read_text())
        effective = {
            key: config.get(key) for key in ("environment", "agents", "timeout_multiplier", "n_concurrent_trials")
        }
        versions = set()
        for trial in path.parent.glob("*/result.json"):
            agent = json.loads(trial.read_text()).get("agent_info") or {}
            versions.add(json.dumps({key: agent.get(key) for key in ("name", "version", "model_info")}, sort_keys=True))
        context[path.parent.name] = {
            # Hash settings rather than disclosing potentially private backend values.
            "settings_digest": hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest(),
            "agents": [json.loads(value) for value in sorted(versions)],
        }
    return context


def _execution_evidence(run_dir: Path, agent: str, trial_id: Any, *, entry_id: str | None = None) -> dict:
    """Read only the identified public trial; never trust a reward's embedded text/path."""
    from skillevaluator.tier3.harbor.report_data import _read_bounded_bytes
    from skillevaluator.tier3.retained_evidence import summarize_trajectory

    unavailable = {"status": "unavailable", "reason": "Retained trajectory identity or content is unavailable"}
    if any(not isinstance(value, str) or value in {"", ".", ".."} or Path(value).name != value
           for value in (agent, trial_id)):
        return unavailable
    path = run_dir / agent / "with-skill" / "trials" / trial_id / "trajectory.json"
    # Check every component. A contained-looking leaf may sit under a directory
    # link to held-out data or to an unrelated host file.
    if path.resolve() != path or not path.is_relative_to(run_dir):
        return unavailable
    raw = _read_bounded_bytes(path, [], artifact="improvement_trajectory")
    if raw is None:
        return unavailable
    try:
        identity_raw = _read_bounded_bytes(path.with_name("reward.json"), [], artifact="improvement_reward_identity")
        identity = json.loads(identity_raw) if identity_raw else {}
        if entry_id is None or identity.get("entry_id") != entry_id or identity.get("trial_id") != trial_id:
            return unavailable
        trajectory = json.loads(raw)
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
            return unavailable
        # Final reports have their own explicit opt-in channel. Tool returns here
        # must not bypass that source boundary through the shared formatter.
        summary = summarize_trajectory(trajectory, include_final_response=False)
    except (ValueError, TypeError, AttributeError, UnicodeError):
        return unavailable
    return {
        "status": "available",
        "source": str(path.relative_to(run_dir)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "summary": summary,
        "artifact_comparison": "unavailable_without_retained_original_and_delivered_pairs",
    }


def load_signals(run_dir: Path, *, include_agent_observations: bool = False) -> dict:
    """Read one explicit run; retain incomplete status rather than inventing scores.

    Trials and failures use the same loader as the existing HTML report. Missing
    metrics stay missing. Dataset prompts and ground-truth fields are omitted;
    grader diagnostics may contain expected/observed results. Optional final agent
    reports remain explicitly unverified observations.
    """
    from skillevaluator.evaluation.tier3_report import _normalize_trials
    from skillevaluator.tier3.harbor.report_data import metrics_for_condition
    from skillevaluator.tier3.harbor.resource_usage import resource_comparison

    run_dir = run_dir.expanduser().resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    result = json.loads((run_dir / "result.json").read_text())
    snapshot = load_dataset_snapshot(run_dir)
    if snapshot is None:
        raise ValueError("A valid retained dataset snapshot is required")
    config = json.loads((run_dir / "run_config.json").read_text())
    entries = {
        str(
            next((case[key] for key in ("id", "entry_id", "case_id", "task_id") if case.get(key) is not None), "")
        ): case
        for case in snapshot["dataset"]
    }
    agents = {}
    evidence_issues = [
        f"{case_id}: draft expectations require independent oracle qualification and a new evaluation"
        for case_id, case in entries.items()
        if case.get("authoring_status") == "draft_requires_oracle_review"
    ]
    for name, info in load_agent_data(run_dir, include_agent_observations=include_agent_observations).items():
        baseline_condition = info.get("conditions", {}).get("without_skill") or (
            result.get("agents", {}).get(name, {}).get("conditions", {}).get("without_skill")
        )
        baseline_expected = bool(baseline_condition) and baseline_condition.get("execution_status") != "skipped"
        cases = {
            case_id: {
                "entry_id": case_id,
                "split": case.get("split", "discovery"),
                **{
                    key: case[key]
                    for key in ("authoring_status", "unresolved_requirements")
                    if key in case
                },
                "attempts": [],
                "baseline_attempts": [],
            }
            for case_id, case in entries.items()
        }
        observed_cases = set()
        for key, field in (("rewards", "attempts"), ("rewards_baseline", "baseline_attempts")):
            if any(len(group) > 1 for group in logical_trial_reward_groups(info.get(key, []))):
                evidence_issues.append(
                    f"{name}: multistep reward rows need logical-attempt comparison; automatic improvement is unsupported"
                )
            if info.get(f"{key}_complete") is not True and (key == "rewards" or baseline_expected):
                evidence_issues.append(f"{name}: retained {key} coverage is incomplete")
            for reward in info.get(key, []):
                case_id = str(reward["entry_id"])
                observed_cases.add(case_id)
                case = cases.setdefault(
                    case_id,
                    {
                        "entry_id": case_id,
                        "split": entries.get(case_id, {}).get("split", "discovery"),
                        "attempts": [],
                        "baseline_attempts": [],
                    },
                )
                attempt = _attempt(reward, name)
                if field == "attempts" and case_id in entries and str(case["split"]).lower().strip() not in RESERVED_SPLITS:
                    from skillevaluator.tier3.artifact_evidence import load_artifact_evidence

                    attempt["execution_evidence"] = _execution_evidence(
                        run_dir, name, attempt["trial_id"], entry_id=case_id,
                    )
                    attempt["artifact_evidence"] = load_artifact_evidence(
                        run_dir, name, attempt["trial_id"], entry_id=case_id,
                        declarations=entries[case_id].get("artifact_pairs"),
                    )
                if field == "baseline_attempts" or str(case["split"]).lower().strip() in RESERVED_SPLITS:
                    attempt.pop("agent_observations", None)
                if field == "baseline_attempts":
                    attempt.pop("skill_activation")
                if attempt["evaluation_status"] == "failed":
                    evidence_issues.append(f"{name}/{case_id}/{attempt['trial_id']}: reward reports a grading error")
                elif attempt["missing_scores"]:
                    evidence_issues.append(f"{name}/{case_id}: required scores are missing")
                case[field].append(attempt)
        if observed_cases != set(entries):
            evidence_issues.append(f"{name}: retained cases differ from the evaluated dataset")
        for case in cases.values():
            case["scoring_status"] = {
                arm: "error"
                if any(attempt["evaluation_status"] == "failed" for attempt in case[field])
                else "scored" if case[field] else "unscored"
                for arm, field in (("with_skill", "attempts"), ("without_skill", "baseline_attempts"))
            }
        agents[name] = {
            "model": config.get("agents", {}).get(name, {}).get("model"),
            "execution_status": info.get("execution_status"),
            "execution_errors": info.get("execution_errors", []),
            "expected_attempts": info.get("expected_attempts"),
            "scored_attempts": info.get("scored_attempts"),
            "skill_activation": info.get("skill_activation") or summarize_skill_activation([]),
            "aggregate": {
                key: info.get(key, {})
                for key in (
                    "with_skill",
                    "without_skill",
                    "overall_with_skill",
                    "overall_without_skill",
                    "lift",
                    "custom_lift",
                    "custom_with_skill",
                    "custom_without_skill",
                    "dimensions_with_skill",
                    "dimensions_without_skill",
                    "cost",
                )
            },
            "cases": list(cases.values()),
        }
        if any(
            attempt["evaluation_status"] == "failed"
            for case in cases.values()
            for field in ("attempts", "baseline_attempts")
            for attempt in case[field]
        ):
            # Existing summaries may include an error sentinel in their averages.
            agents[name]["scored_attempts"] = None
            agents[name]["skill_activation"] = summarize_skill_activation([])
            for key in agents[name]["aggregate"]:
                if key != "cost":
                    agents[name]["aggregate"][key] = None
        agents[name]["aggregate"]["resources"] = resource_comparison(
            _normalize_trials(info.get("rewards", []), metrics_for_condition(info, "with_skill")),
            _normalize_trials(info.get("rewards_baseline", []), metrics_for_condition(info, "without_skill")),
            with_complete=info.get("rewards_complete") is True,
            baseline_complete=info.get("rewards_baseline_complete") is True,
        )
    harbor = _values(config.get("harbor", {}))
    if not agents or not entries or "" in entries:
        evidence_issues.append("Agent or dataset case identity is missing")
    for key in ("stream", "subprocess_timeout_seconds", "peak_trials", "peak_sandbox_pods"):
        harbor.pop(key, None)
    context = _verification_context(run_dir, result.get("skill_name"))
    runtime = _runtime_context(run_dir)
    if not runtime or any(
        len(item["agents"]) != 1 or not item["agents"][0].get("version") for item in runtime.values()
    ):
        evidence_issues.append("Retained runtime configuration or agent versions are missing")
    if not any("/tests/" in path and entry["kind"] == "file" for path, entry in context.items()):
        evidence_issues.append("Retained grader evidence is missing")
    if any(entry["kind"] in {"symlink", "other"} for entry in context.values()):
        evidence_issues.append("Retained evaluation context contains unresolved links or special files")
    packet = redact_sensitive_data(
        {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "dataset_digest": snapshot.get("dataset_digest"),
            "execution_status": result.get("execution_status"),
            "evidence_issues": evidence_issues,
            "contract": {
                "evaluator_version": result.get("evaluator_version"),
                "harbor": harbor,
                "grading": _values(config.get("grading")),
                "provider": _values(config.get("provider")),
                "judge": _values(config.get("judge")),
                "env_mode": result.get("observability", {}).get("env_mode"),
                "skill_workspace": _values(config.get("skill_workspace")),
                "runtime_fallback": result.get("runtime_fallback"),
                "verification_context": context,
                "runtime": runtime,
            },
            "agents": agents,
        }
    )
    for name, agent in packet["agents"].items():
        # This fixed-schema measurement was validated by resource_comparison;
        # redact its contents without mistaking the container name for a credential.
        agent["aggregate"]["resources"]["tokens"] = redact_sensitive_data(
            agents[name]["aggregate"]["resources"]["tokens"]
        )
    return packet


def _flatten(values: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif _number(value) or value is None:
            flat[name] = value
    return flat


def _deltas(before: dict, after: dict) -> dict:
    return {
        key: {
            "before": before.get(key),
            "after": after.get(key),
            "delta": round(after[key] - before[key], 6)
            if _number(before.get(key)) and _number(after.get(key))
            else None,
        }
        for key in sorted(before.keys() | after.keys())
    }


def _case_metrics(case: dict) -> dict:
    attempts = case["attempts"]
    names = {key for attempt in attempts for key in attempt["scores"]}
    # Do not average a metric over only the convenient subset of attempts.
    return {
        name: mean(attempt["scores"][name] for attempt in attempts)
        for name in names
        if all(name in attempt["scores"] for attempt in attempts)
    }


def compare_signals(before: dict, after: dict, *, outcome_metrics: tuple[str, ...] = ()) -> dict:
    """Expose measured changes; accept only comparable, complete evidence.

    Outcome metrics are explicitly chosen independent checks, not inferred from
    metric names. Score-only gains are labeled as such. This is a measured sample
    comparison, not a statistical significance claim or automatic publication.
    """
    issues = [*before.get("evidence_issues", []), *after.get("evidence_issues", [])]
    if not before.get("dataset_digest") or before.get("dataset_digest") != after.get("dataset_digest"):
        issues.append("Dataset identity is missing or changed")
    if before.get("contract") != after.get("contract"):
        issues.append("Evaluation configuration or retained grader changed")
    if before.get("execution_status") != "succeeded" or after.get("execution_status") != "succeeded":
        issues.append("Evaluation execution is incomplete")
    if not before["agents"] or before["agents"].keys() != after["agents"].keys():
        issues.append("Agent coverage is empty or changed")
    agents = {}
    improved = regressed = outcome_improved = False
    for name in sorted(before["agents"].keys() & after["agents"].keys()):
        old, new = before["agents"][name], after["agents"][name]
        if not old.get("model") or old["model"] != new.get("model"):
            issues.append(f"{name}: agent model is missing or changed")
        if any(item.get("execution_status") != "succeeded" for item in (old, new)):
            issues.append(f"{name}: incomplete agent execution")
        old_cases = {case["entry_id"]: case for case in old["cases"]}
        new_cases = {case["entry_id"]: case for case in new["cases"]}
        if not old_cases or old_cases.keys() != new_cases.keys():
            issues.append(f"{name}: case coverage is empty or changed")
        cases = []
        for case_id in sorted(old_cases.keys() & new_cases.keys()):
            left, right = old_cases[case_id], new_cases[case_id]
            for arm in ("attempts", "baseline_attempts"):
                if len(left[arm]) != len(right[arm]) or (arm == "attempts" and not left[arm]):
                    issues.append(f"{name}/{case_id}: {arm} coverage changed or empty")
                judge_sets = [
                    {json.dumps(a.get("judges", {}), sort_keys=True) for a in case[arm]} for case in (left, right)
                ]
                if judge_sets[0] != judge_sets[1]:
                    issues.append(f"{name}/{case_id}: judge configuration changed")
                if any(len(judges) > 1 for judges in judge_sets):
                    issues.append(f"{name}/{case_id}: attempts used different judge configurations")
                policies = [
                    {json.dumps(a.get("score_contract", {}), sort_keys=True) for a in case[arm]}
                    for case in (left, right)
                ]
                if policies[0] != policies[1]:
                    issues.append(f"{name}/{case_id}: score definition changed")
            old_scores, new_scores = _case_metrics(left), _case_metrics(right)
            if old_scores.keys() != new_scores.keys() or not old_scores:
                issues.append(f"{name}/{case_id}: score coverage missing or changed")
            outcomes_available = bool(outcome_metrics) and all(
                metric in old_scores and metric in new_scores for metric in outcome_metrics
            )
            if outcome_metrics and not outcomes_available:
                issues.append(f"{name}/{case_id}: required outcome check missing")
            deltas = _deltas(old_scores, new_scores)
            gains = [key for key, value in deltas.items() if value["delta"] is not None and value["delta"] > 0]
            losses = [key for key, value in deltas.items() if value["delta"] is not None and value["delta"] < 0]
            improved |= bool(gains)
            regressed |= bool(losses)
            outcome_improved |= bool(set(gains) & set(outcome_metrics))
            findings = {
                label: sum(len(a["findings"]) for a in case["attempts"])
                if case["attempts"] and all(a["scores"] for a in case["attempts"])
                else None
                for label, case in (("before", left), ("after", right))
            }
            cases.append(
                {
                    "entry_id": case_id,
                    "split": left.get("split"),
                    "attempts": len(left["attempts"]),
                    "status": "unscored"
                    if not old_scores or not new_scores
                    else "regressed"
                    if losses
                    else "improved"
                    if gains
                    else "unchanged",
                    "metrics": deltas,
                    "outcome_gains": [key for key in gains if key in outcome_metrics] if outcomes_available else None,
                    "outcome_regressions": [key for key in losses if key in outcome_metrics] if outcomes_available else None,
                    "other_metric_declines": [key for key in losses if key not in outcome_metrics],
                    "findings": {
                        **findings,
                        "delta": findings["after"] - findings["before"]
                        if findings["after"] is not None and findings["before"] is not None
                        else None,
                    },
                }
            )
        agents[name] = {
            "cases": cases,
            "aggregate": _deltas(_flatten(old["aggregate"]), _flatten(new["aggregate"])),
            "skill_activation": {
                label: item.get("skill_activation") or summarize_skill_activation([])
                for label, item in (("before", old), ("after", new))
            },
        }
    verdict = (
        "abstain"
        if issues
        else "regression"
        if regressed
        else "outcome_improvement"
        if outcome_improved
        else "signal_improvement"
        if improved
        else "no_demonstrated_benefit"
    )
    return {
        "schema_version": 1,
        "before": before["run_dir"],
        "after": after["run_dir"],
        "verdict": verdict,
        "issues": list(dict.fromkeys(issues)),
        "outcome_metrics": list(outcome_metrics),
        "agents": agents,
        "interpretation": "Observed sample changes, not proof the skill edit caused them. Verify skill use and repeat on independent cases before claiming generalization.",
    }


def compare_runs(before: Path, after: Path, *, outcome_metrics: tuple[str, ...] = ()) -> dict:
    return compare_signals(load_signals(before), load_signals(after), outcome_metrics=outcome_metrics)


def comparison_markdown(comparison: dict) -> str:
    def display(value: Any) -> str:
        return str(value) if value is not None else "unavailable"

    lines = [
        "# Skill improvement comparison",
        "",
        f"Result: **{comparison['verdict']}**",
        "",
        comparison["interpretation"],
        "",
        "Resource tokens count prompt + completion (cache included); seconds measure agent execution. Both exclude setup, judging and proposing.",
        "",
    ]
    lines.extend(f"- {issue}" for issue in comparison["issues"])
    for name, agent in comparison["agents"].items():
        activation = agent["skill_activation"]
        lines.extend(
            [
                "", f"## {name}", "",
                f"Skill activation: **{activation['before'].get('status', 'unknown')}** → **{activation['after'].get('status', 'unknown')}**.",
                "",
            ]
        )
        for case in agent["cases"]:
            assessment = (
                f"Outcome gains: {', '.join(case['outcome_gains']) or 'none'}; "
                f"outcome regressions: {', '.join(case['outcome_regressions']) or 'none'}."
                if case.get("outcome_gains") is not None
                else "Outcome assessment unavailable: select an independent outcome metric measured in both runs."
            )
            lines.append(
                f"- {case['entry_id']}: {assessment} Other metric declines: {', '.join(case.get('other_metric_declines', [])) or 'none'}."
            )
        lines.extend(["", "| Case | Signal | Before | After | Change |", "| --- | --- | ---: | ---: | ---: |"])
        for case in agent["cases"]:
            for metric, values in {**case["metrics"], "finding count": case["findings"]}.items():
                lines.append(
                    f"| {case['entry_id']} | {metric} | {display(values['before'])} | {display(values['after'])} | {display(values['delta'])} |"
                )
        lines.extend(["", "| Aggregate signal | Before | After | Change |", "| --- | ---: | ---: | ---: |"])
        for metric, values in agent["aggregate"].items():
            if ".sources." in metric:
                continue  # Dimension weights are provenance, not observed performance.
            lines.append(
                f"| {metric} | {display(values['before'])} | {display(values['after'])} | {display(values['delta'])} |"
            )
    return "\n".join(lines) + "\n"
