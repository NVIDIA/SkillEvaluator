# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed Studio paths around the native, evidence-backed improvement API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.studio.adapter import read_json
from skillevaluator.studio.store import IDENTIFIER, safe_file

if TYPE_CHECKING:
    from skillevaluator.studio.store import Workspace


def managed_job(workspace: Workspace, job_id: str, action: str | None = None) -> tuple[Path, dict, dict, dict]:
    if not isinstance(job_id, str) or not IDENTIFIER.fullmatch(job_id):
        raise ValueError("Choose a saved run in this workspace.")
    folder = safe_file(workspace.jobs, job_id)
    metadata = read_json(folder / "metadata.json", root=workspace.jobs)
    if not isinstance(metadata, dict) or metadata.get("id") != job_id or (action and metadata.get("action") != action):
        raise ValueError("The selected run has the wrong identity or action.")
    request = read_json(folder / "request.json", root=workspace.jobs)
    result = read_json(folder / "result.json", root=workspace.jobs) if (folder / "result.json").exists() else {}
    if not isinstance(request, dict) or not isinstance(result, dict):
        raise ValueError("The selected run has malformed saved data.")
    return folder, metadata, request, result


def job_source(folder: Path, request: dict) -> Path:
    try:
        relative = Path(request["skill_path"]).relative_to(folder)
        source = safe_file(folder, relative.as_posix())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The saved run has no contained source snapshot.") from exc
    if not (source / "SKILL.md").is_file():
        raise ValueError("The saved source snapshot is unavailable.")
    return source


def evaluation_run(workspace: Workspace, job_id: str, *, skill_id: str | None = None) -> Path:
    folder, metadata, _request, result = managed_job(workspace, job_id, "evaluate")
    if skill_id is not None and metadata.get("skill_id") != skill_id:
        raise ValueError("The selected evaluation belongs to a different skill.")
    if result.get("kind") != "evaluation":
        raise ValueError("The selected run has no retained evaluation result.")
    raw = result.get("raw")
    pointer = raw.get("run_dir") if isinstance(raw, dict) else None
    try:
        relative = Path(pointer).relative_to(folder)
        run = safe_file(folder, relative.as_posix())
    except (TypeError, ValueError) as exc:
        raise ValueError("The selected evaluation has no contained native run directory.") from exc
    if not safe_file(run, "result.json").is_file():
        raise ValueError("Retained native evaluation artifacts are unavailable.")
    return run


def same_package(left: Path, right: Path) -> bool:
    from skillevaluator.tier3.improve import snapshot_tree

    return snapshot_tree(left, include_generated=True) == snapshot_tree(right, include_generated=True)


def copy_package(source: Path, destination: Path) -> None:
    from skillevaluator.tier3.harbor.secure_copy import copytree_secure
    from skillevaluator.tier3.improve import snapshot_tree

    if any(item["kind"] not in {"file", "dir"} for item in snapshot_tree(source, include_generated=True).values()):
        raise ValueError("Skill packages must contain only regular files and directories.")
    copytree_secure(source, destination, allowed_root=source)


def readiness(workspace: Workspace, job_id: str, skill_id: str | None = None) -> dict[str, Any]:
    from skillevaluator.reporting.dashboard_improvements import _runtime_digest, _source_matches
    from skillevaluator.tier3.improve import discovery_packet
    from skillevaluator.tier3.improvement_signals import load_signals

    reasons: list[str] = []
    summary = {"cases": 0, "findings": 0, "observations": 0}
    source_matches = False
    try:
        folder, metadata, request, result = managed_job(workspace, job_id, "evaluate")
        selected = skill_id or metadata["skill_id"]
        run = evaluation_run(workspace, job_id, skill_id=selected)
        source = job_source(folder, request)
        source_matches = same_package(source, workspace.directory(selected))
        if not source_matches:
            reasons.append("The skill or dataset changed after this run. Evaluate the current source again.")
        signals = load_signals(run, include_agent_observations=True)
        reasons.extend(signals.get("evidence_issues", []))
        if metadata.get("status") != "succeeded" or result.get("execution_status") != "succeeded":
            reasons.append("Select a completed successful evaluation.")
        if not signals.get("agents") or any(
            a.get("execution_status") != "succeeded" for a in signals["agents"].values()
        ):
            reasons.append("Complete successful agent evidence is required.")
        identities = {(name, case["entry_id"]) for name, agent in signals["agents"].items() for case in agent["cases"]}
        if not _source_matches(run, source.name, _runtime_digest(source), identities):
            reasons.append("The retained evaluated source is missing or does not match this skill.")
        for agent in discovery_packet(signals, include_agent_observations=True)["agents"].values():
            for case in agent["cases"]:
                summary["cases"] += 1
                summary["findings"] += sum(len(a["findings"]) for a in case["attempts"])
                summary["observations"] += sum(len(a.get("agent_observations", [])) for a in case["attempts"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        reasons.append(str(exc))
    return {
        "eligible": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "source_matches": source_matches,
        "signals_summary": summary,
    }


def proposal(workspace: Workspace, proposal_job_id: str) -> tuple[Path, dict, dict]:
    folder, metadata, _request, result = managed_job(workspace, proposal_job_id, "improve")
    if metadata.get("status") != "succeeded" or result.get("kind") != "improvement":
        raise ValueError("Select a completed improvement proposal.")
    return folder / "improvement", metadata, result


def validation_reference(
    workspace: Workspace, proposal_job_id: str, candidate_id: str, before_job_id: str | None
) -> tuple[str, dict, Path]:
    from skillevaluator.reporting.dashboard_improvements import _runtime_digest, _source_matches, load_skill_change
    from skillevaluator.tier3.improvement_signals import load_signals

    root, metadata, result = proposal(workspace, proposal_job_id)
    if result.get("candidate_id") != candidate_id:
        raise ValueError("This candidate does not belong to the selected proposal.")
    view = load_skill_change(root)
    if view["issues"] or not view["receipt"]["candidate_ready"]:
        raise ValueError("The retained proposal or candidate needs review before evaluation.")
    native_candidate = root / "candidate" / view["receipt"]["evaluated_skill_name"]
    if _runtime_digest(workspace.directory(candidate_id)) != _runtime_digest(native_candidate):
        raise ValueError("The candidate changed after this proposal. Keep that edit separate or create a new proposal.")
    before_job_id = before_job_id or result["discovery_job_id"]
    run = evaluation_run(workspace, before_job_id, skill_id=metadata["skill_id"])
    folder, before, request, _ = managed_job(workspace, before_job_id, "evaluate")
    signals = load_signals(run)
    identities = {(name, case["entry_id"]) for name, agent in signals["agents"].items() for case in agent["cases"]}
    if (
        before.get("status") != "succeeded"
        or signals.get("execution_status") != "succeeded"
        or any(agent.get("execution_status") != "succeeded" for agent in signals["agents"].values())
        or signals["evidence_issues"]
        or not _source_matches(run, native_candidate.name, view["receipt"]["runtime_digest"], identities)
    ):
        raise ValueError("Choose a complete original validation run with the proposal's original runtime source.")
    return before_job_id, before, job_source(folder, request)


def freeze_evaluation(reference: Path, destination: Path) -> None:
    import shutil

    for path in destination.iterdir():
        if path.name.casefold() in {"eval", "evals"}:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for path in reference.iterdir():
        if path.name.casefold() in {"eval", "evals"}:
            if not path.is_dir():
                raise ValueError("Evaluation sources must be regular directories.")
            copy_package(path, destination / path.name)


def comparison(workspace: Workspace, request: dict) -> dict[str, Any]:
    from skillevaluator.reporting.dashboard_improvements import load_skill_change

    root, metadata, result = proposal(workspace, request.get("proposal_job_id"))
    if not result.get("candidate_id"):
        raise ValueError("This proposal has no changed candidate to evaluate.")
    before = evaluation_run(workspace, request.get("before_job_id"), skill_id=metadata["skill_id"])
    after = evaluation_run(workspace, request.get("after_job_id"), skill_id=result["candidate_id"])
    metrics = outcome_metrics(request.get("outcome_metrics", []))
    view = load_skill_change(root, before, after, outcome_metrics=tuple(metrics))
    return {"kind": "improvement_comparison", "proposal_view": view}


def outcome_metrics(metrics: Any) -> list[str]:
    if (
        not isinstance(metrics, list)
        or len(metrics) > 32
        or any(not isinstance(x, str) or not x.strip() or len(x) > 200 for x in metrics)
    ):
        raise ValueError("Select explicit outcome metric names as a list.")
    return list(dict.fromkeys(metrics))


def prepare(workspace: Workspace, request: dict, skill: Path, job_dir: Path) -> dict[str, Any]:
    from skillevaluator.reporting.dashboard_improvements import load_skill_change
    from skillevaluator.tier3.improve import prepare_improvement

    options = request["improvement"]
    edits_path = None
    if options["mode"] == "deterministic":
        edits_path = job_dir / "reviewed-edits.json"
        edits_path.write_text(json.dumps(options["edits"]), encoding="utf-8")
    run = evaluation_run(workspace, request["discovery_job_id"], skill_id=request["skill_id"])
    receipt = prepare_improvement(
        skill,
        run,
        job_dir / "improvement",
        mode=options["mode"],
        edits_path=edits_path,
        model=request["options"]["model"] if options["mode"] == "llm" else None,
        include_agent_observations=options["include_agent_observations"],
    )
    view = load_skill_change(job_dir / "improvement")
    candidate_id = None
    if receipt["candidate_ready"] and receipt["status"] == "candidate_created" and not view["issues"]:
        candidate = workspace.import_candidate(
            request["skill_id"],
            job_dir / "improvement" / "candidate" / skill.name,
            proposal_job_id=job_dir.name,
            discovery_job_id=request["discovery_job_id"],
        )
        candidate_id = candidate["id"]
    return {
        "kind": "improvement",
        "receipt": receipt,
        "proposal_view": view,
        "candidate_id": candidate_id,
        "proposal_job_id": job_dir.name,
        "discovery_job_id": request["discovery_job_id"],
    }
