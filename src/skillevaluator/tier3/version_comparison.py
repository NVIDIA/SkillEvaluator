# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only comparisons of retained with-skill evidence."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from skillevaluator.tier3.baseline_store import comparison_identity, condition_eligibility, load_condition_rows
from skillevaluator.tier3.results_location import ordered_run_directories


def retained_run(root: Path, requested: str) -> Path:
    """Resolve one retained run without following run-directory aliases."""
    if requested == "latest":
        candidates = ordered_run_directories(root, completed_only=True)
        if not candidates:
            raise ValueError("No completed retained runs found")
        return candidates[0]
    if Path(requested).name != requested or requested in {".", ".."}:
        raise ValueError("Run must be a retained run id under the results root")
    candidate = root / requested
    if candidate.is_symlink() or candidate.resolve().parent != root.resolve():
        raise ValueError("Linked run directories are not comparison sources")
    return candidate


def _read_result(run_dir: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Invalid retained result")
    return result


def _skill_revision(skill: dict[str, Any]) -> tuple[Any, Any]:
    source = skill.get("evaluated_source") if isinstance(skill.get("evaluated_source"), dict) else {}
    target = skill.get("publication_target") if isinstance(skill.get("publication_target"), dict) else {}
    return source.get("revision"), target.get("skill_digest")


def _eligible(
    run_dir: Path,
    identity: dict[str, Any],
    *,
    collected_result: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    try:
        agents = list(identity["config"]["agents"])
        reasons, result = condition_eligibility(
            run_dir, agents, identity, "with_skill", collected_result=collected_result
        )
        recorded = result.get("comparison_identity", {})
        if set(recorded.get("config", {}).get("agents", {})) != set(agents):
            reasons.append("agent selection changed")
        config = (
            collected_result["run_config"]
            if collected_result is not None
            else json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        )
        if recorded.get("skill") != comparison_identity(config, result)["skill"]:
            reasons.append("recorded skill identity disagrees with source evidence")
        return reasons, result
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"Missing or corrupt with-skill comparison evidence: {str(exc)[:512]}"], {}


def _provenance(current_identity: dict[str, Any], source: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    current_skill = copy.deepcopy(current_identity.get("skill", {}))
    prior_skill = copy.deepcopy(source.get("comparison_identity", {}).get("skill", {}))
    current_skill = current_skill if isinstance(current_skill, dict) else {}
    prior_skill = prior_skill if isinstance(prior_skill, dict) else {}
    identical = _skill_revision(current_skill) == _skill_revision(prior_skill)
    return {
        "source_run_id": source_dir.name,
        "source_evaluated_at": source.get("evaluated_at"),
        "source_run_dir": str(source_dir),
        "source_skill_identity": prior_skill,
        "current_skill_identity": current_skill,
        "identical_skill": identical,
    }


def select_prior_version(
    root: Path, requested: str, identity: dict[str, Any], *, current_run_dir: Path
) -> dict[str, Any]:
    """Check a source before execution; latest excludes the current skill revision."""
    candidates = (
        ordered_run_directories(root, completed_only=True) if requested == "latest" else [retained_run(root, requested)]
    )
    rejected = []
    for candidate in candidates:
        if candidate.resolve() == current_run_dir.resolve():
            continue
        reasons, result = _eligible(candidate, identity)
        provenance = _provenance(identity, result, candidate)
        if requested == "latest" and (
            provenance["identical_skill"] or not any(_skill_revision(identity.get("skill", {})))
        ):
            reasons.append("skill identity is unchanged or unrecorded; not a different skill revision")
        if reasons:
            if requested != "latest":
                raise ValueError(f"Cannot compare to {requested}: " + "; ".join(reasons))
            if len(rejected) < 16:
                rejected.append({"run_id": candidate.name, "reasons": [reason[:1024] for reason in reasons[:16]]})
            continue
        return {"status": "available", "reasons": [], "rejected_candidates": rejected, **provenance}
    return {
        "status": "unavailable",
        "reasons": ["No compatible completed run of a different skill revision"],
        "current_skill_identity": copy.deepcopy(identity.get("skill", {})),
        "rejected_candidates": rejected,
    }


def compare_versions(
    current_dir: Path, prior_dir: Path, *, collected_result: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compute paired version deltas without changing either retained run."""
    from skillevaluator.tier3.harbor.collector import (
        _compute_custom_lift,
        _compute_lift,
        _logical_attempt_rewards,
        _score_policy_for_rewards,
        average_custom_metrics,
        average_metrics,
    )

    unavailable: dict[str, Any] = {"status": "unavailable", "reasons": [], "agents": {}}
    try:
        current = collected_result if collected_result is not None else _read_result(current_dir)
        identity = current["comparison_identity"]
        current_reasons, _ = _eligible(current_dir, identity, collected_result=collected_result)
        prior_reasons, source = _eligible(prior_dir, identity)
        unavailable.update(_provenance(identity, source, prior_dir))
        reasons = [
            *(f"current: {reason}" for reason in current_reasons),
            *(f"prior: {reason}" for reason in prior_reasons),
        ]
        if reasons:
            return {**unavailable, "reasons": reasons}
        agents = {}
        for agent in identity["config"]["agents"]:
            current_rows = _logical_attempt_rewards(load_condition_rows(current_dir, agent, "with_skill"))
            prior_rows = _logical_attempt_rewards(load_condition_rows(prior_dir, agent, "with_skill"))
            current_scores, _, current_metrics = average_metrics(current_rows)
            prior_scores, _, prior_metrics = average_metrics(prior_rows)
            if current_metrics != prior_metrics:
                raise ValueError(f"{agent}: scored metric set changed")
            current_policy, _ = _score_policy_for_rewards(current_rows, current_metrics)
            prior_policy, _ = _score_policy_for_rewards(prior_rows, prior_metrics)
            if current_policy != prior_policy or current_policy != identity["score_policy"]:
                raise ValueError(f"{agent}: conflicting observed score policies")
            lift = _compute_lift(
                current_scores,
                prior_scores,
                current_rows,
                prior_rows,
                identity["expected_case_ids"],
                score_policy=current_policy,
            )
            custom_lift = _compute_custom_lift(
                average_custom_metrics(current_rows),
                average_custom_metrics(prior_rows),
                current_rows,
                prior_rows,
                include_overall=not current_metrics and not prior_metrics,
            )
            overall = lift.get("overall") or custom_lift.get("overall")
            if overall is None:
                raise ValueError(f"{agent}: no comparable paired overall score")

            def deltas(values):
                return {
                    key: {"current": value["with_skill"], "prior": value["without_skill"], "delta": value["delta"]}
                    for key, value in values.items()
                    if key != "overall" and isinstance(value, dict) and "delta" in value
                }

            agents[agent] = {
                "current_overall": overall["with_skill"],
                "prior_overall": overall["without_skill"],
                "delta": overall["delta"],
                "regression": overall["delta"] < 0,
                "metrics": deltas(lift),
                "custom_metrics": deltas(custom_lift),
                "pairing": lift.get("basis", {}).get("pairing"),
            }
        if not agents:
            raise ValueError("No selected agents with comparable scores")
        return {
            **unavailable,
            "status": "available",
            "agents": agents,
            "reasons": ["Skill identity is unchanged; comparison is not a version change"]
            if unavailable["identical_skill"]
            else [],
            "regression": any(agent["regression"] for agent in agents.values()),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {**unavailable, "reasons": [f"Version comparison unavailable: {str(exc)[:1024]}"]}


def compare_retained_runs(root: Path, current: str, prior: str) -> dict[str, Any]:
    """Resolve read-only command selectors and use the live comparison path."""
    try:
        current_dir = retained_run(root, current)
        current_result = _read_result(current_dir)
        if prior == "latest":
            selection = select_prior_version(
                root, prior, current_result["comparison_identity"], current_run_dir=current_dir
            )
            if selection["status"] != "available":
                return selection
            prior_dir = Path(selection["source_run_dir"])
        else:
            prior_dir = retained_run(root, prior)
        return compare_versions(current_dir, prior_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"status": "unavailable", "reasons": [str(exc)], "agents": {}}


def version_comparison_lines(comparison: dict[str, Any]) -> list[str]:
    """Small shared report section, independent of no-skill lift."""
    if not isinstance(comparison, dict):
        return ["Versus prior version: unavailable (invalid comparison)"]
    if comparison.get("status") != "available":
        reasons = "; ".join(str(reason) for reason in comparison.get("reasons", []))
        return [f"Versus prior version: unavailable ({reasons})"]
    agents = comparison.get("agents")
    if (
        not isinstance(agents, dict)
        or not agents
        or any(
            not isinstance(values, dict)
            or any(
                type(values.get(key)) not in (int, float) or not math.isfinite(values[key])
                for key in ("prior_overall", "current_overall", "delta")
            )
            for values in agents.values()
        )
    ):
        return ["Versus prior version: unavailable (invalid scores)"]
    skill = comparison.get("source_skill_identity")
    revision, digest = _skill_revision(skill if isinstance(skill, dict) else {})
    prefix = f"Versus prior version (run {comparison.get('source_run_id')}, rev {revision or digest or 'unrecorded'})"
    flag = "; identical skill, not a version change" if comparison.get("identical_skill") else ""
    return [
        f"{prefix} · {agent}: overall {values['prior_overall']:.2f} → {values['current_overall']:.2f} "
        f"({values['delta']:+.2f}{', regression' if values['delta'] < 0 else ''}{flag})"
        for agent, values in agents.items()
    ]
