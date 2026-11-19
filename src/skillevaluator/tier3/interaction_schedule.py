# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case-blocked scheduling with retained, independently checked engine runs."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.evaluation import EvaluationOptions, EvaluationService
from skillevaluator.tier3.case_ids import validate_case_ids
from skillevaluator.tier3.decoy_identity import decoy_source_copy_ignore, decoy_source_digest
from skillevaluator.tier3.harbor.secure_copy import copytree_secure
from skillevaluator.tier3.interactions import (
    IncomparableInteraction,
    _case_contract,
    _summarize_case_interactions,
    compare_configurations,
    interaction_scores,
)

if TYPE_CHECKING:
    from skillevaluator.tier3.harbor.progress import ProgressReporter


def validate_interleaved_budgets(source: Path, options: EvaluationOptions) -> None:
    """Reject run-wide caps before separate steps could reset their controllers."""
    from skillevaluator.tier3.evals_config import load_evals_config

    config, _ = load_evals_config(source)
    harbor = config.get("harbor", {})
    limits = []
    for name in ("max_agent_tokens", "max_judge_calls", "max_attempts_total", "max_cost_usd"):
        override = getattr(options, name)
        if (override if override is not None else harbor.get(name)) is not None:
            limits.append(name)
    if limits:
        raise ValueError(
            f"Interleaved execution does not support run-wide budgets ({', '.join(limits)}); use sequential execution"
        )


def interleaved_entries(source: Path) -> list[dict[str, Any]]:
    """Validate the complete frozen structured suite before preparing any runs."""
    from skillevaluator.tier3.dataset_utils import find_eval_file, load_dataset_entries
    from skillevaluator.tier3.evals_spec import validate_tier3_source

    kind, _ = validate_tier3_source(source)
    dataset = find_eval_file(source)
    if kind != "evals_json" or dataset is None:
        raise ValueError(
            "Interleaved execution requires a JSON, JSONL or YAML dataset; use sequential for native tasks"
        )
    entries = load_dataset_entries(dataset)
    identities = validate_case_ids(entry.get("id") for entry in entries)
    if not identities:
        raise ValueError("Interleaved execution requires at least one reviewed case")
    return [{**entry, "id": identity} for entry, identity in zip(entries, identities, strict=True)]


def prepare_case_sources(
    entries: Sequence[dict[str, Any]], frozen: Sequence[Path], evaluator: Path, root: Path
) -> dict[str, list[Path]]:
    """Keep skill bytes and dataset provenance fixed while selecting each case."""
    from skillevaluator.tier3.dataset_utils import find_eval_file

    dataset = find_eval_file(evaluator)
    document = None
    if dataset is not None and dataset.suffix.lower() == ".json":
        document = json.loads(dataset.read_text(encoding="utf-8"))
    elif dataset is not None and dataset.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        document = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    sources = {}
    for ordinal, entry in enumerate(entries):
        selected_document: Any = [entry]
        if isinstance(document, dict):
            if isinstance(document.get("evals"), list):
                selected_document = {**document, "evals": [entry]}
            elif isinstance(document.get("cases"), list):
                selected_document = {**document, "cases": [entry]}
            else:
                selected_document = entry
        selected = []
        for index, skill in enumerate(frozen):
            destination = root / f"case-{ordinal}" / f"skill-{index}" / skill.name
            copytree_secure(skill, destination, allowed_root=skill, ignore=decoy_source_copy_ignore(skill))
            if decoy_source_digest(destination) != decoy_source_digest(skill):
                raise ValueError("Frozen skill changed while preparing interleaved cases")
            copytree_secure(evaluator / "evals", destination / "evals", allowed_root=evaluator)
            # JSON takes precedence over the original supported filename. The
            # original source and fixtures are retained in the experiment root.
            (destination / "evals" / "evals.json").write_text(json.dumps(selected_document) + "\n", encoding="utf-8")
            selected.append(destination)
        sources[entry["id"]] = selected
    return sources


def interaction_schedule(
    case_ids: Sequence[str], configuration_ids: Sequence[str], *, n_attempts: int, seed: int
) -> list[dict[str, Any]]:
    """Rotate a seeded arm order in each complete case/attempt block."""
    cases = sorted(validate_case_ids(case_ids))
    if type(n_attempts) is not int or n_attempts < 1:
        raise ValueError("n_attempts must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("schedule_seed must be an integer between 0 and 4294967295")
    order = list(configuration_ids)
    if len(order) < 4 or len(set(order)) != len(order) or not cases:
        raise ValueError("Expected reviewed cases and distinct baseline, singleton and combined configurations")
    random.Random(seed).shuffle(order)
    schedule = []
    for attempt in range(1, n_attempts + 1):
        for case_index, case_id in enumerate(cases):
            offset = ((attempt - 1) * len(cases) + case_index) % len(order)
            for config_id in [*order[offset:], *order[:offset]]:
                schedule.append(
                    {"case_id": case_id, "attempt": attempt, "configuration": config_id, "status": "pending"}
                )
    return schedule


def _identity(result: Mapping[str, Any], agent: str) -> str:
    config = result["run_config"]
    return json.dumps(
        {
            "model": result["agents"][agent]["model"],
            "judge": {key: value for key, value in config["judge"].items() if key != "catalog_verification"},
            "grading": config["grading"],
            "environment": config["harbor"]["environment"],
            "evaluator_version": result["evaluator_version"],
        },
        sort_keys=True,
    )


def compare_interleaved(
    configurations: Sequence[Mapping[str, Any]], *, case_ids: Sequence[str], metric: str, n_attempts: int
) -> dict[str, Any]:
    """Require every planned pair, then apply the ordinary checks in each block."""
    expected = {(case, attempt) for case in case_ids for attempt in range(1, n_attempts + 1)}
    indexed = []
    failures = []
    for config in configurations:
        runs = config.get("runs", [])
        index = {}
        for run in runs:
            pair = (run.get("case_id"), run.get("attempt"))
            if pair in index or pair not in expected or type(pair[1]) is not int:
                failures.append(f"{config['id']}: duplicate or unexpected scheduled run")
            index[pair] = run
            if "error" in run or "result" not in run:
                failures.append(f"{config['id']} {pair}: {run.get('error') or 'Missing run result'}")
        if set(index) != expected:
            failures.append(f"{config['id']}: incomplete scheduled case/attempt coverage")
        indexed.append(index)
    if failures or len(configurations) < 4:
        return {"status": "incomplete", "reasons": failures or ["Missing configurations"], "agents": {}}

    scores: dict[str, dict[str, list[dict[str, Any]]]] = {}
    evidence: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    identities, contracts, scorers, errors, seen_paths = {}, {}, {}, {}, {}
    for case_id, attempt in sorted(expected):
        block = [
            {"id": config["id"], "condition": config["condition"], "result": index[case_id, attempt]["result"]}
            for config, index in zip(configurations, indexed, strict=True)
        ]
        checked = compare_configurations(block, metric=metric, n_attempts=1)
        if checked["status"] == "incomplete":
            return checked
        for agent, score in checked["agents"].items():
            try:
                if score["status"] != "complete":
                    raise IncomparableInteraction("; ".join(score.get("reasons", [])))
                if set(score["cases"]) != {case_id}:
                    raise IncomparableInteraction("Run evidence does not match its scheduled case")
                result = block[0]["result"]
                identity = _identity(result, agent)
                if identity != identities.setdefault(agent, identity):
                    raise IncomparableInteraction("Agent, scorer, environment or evaluator changed between blocks")
                contract = _case_contract(result, Path(result["run_dir"]))[case_id]
                if contract != contracts.setdefault(case_id, contract):
                    raise IncomparableInteraction("Reviewed case changed between attempts")
                scores.setdefault(agent, {}).setdefault(case_id, []).append(score)
                for config_id, cases in score["evidence"].items():
                    rows = evidence.setdefault(agent, {}).setdefault(config_id, {}).setdefault(case_id, [])
                    for row in cases[case_id]:
                        scorer = json.dumps(row["scorer"], sort_keys=True)
                        if scorer != scorers.setdefault((agent, case_id), scorer):
                            raise IncomparableInteraction(
                                f"Outcome scorer differs across configurations or attempts: {case_id}"
                            )
                        path = str(Path(row["reward_path"]).resolve())
                        paths = seen_paths.setdefault((agent, config_id), set())
                        if path in paths:
                            raise IncomparableInteraction("One retained reward was reused for multiple scheduled runs")
                        paths.add(path)
                        rows.append({**row, "scheduled_attempt": attempt})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.setdefault(agent, []).append(f"{case_id} attempt {attempt}: {exc}")
    agents = {}
    for agent in sorted(set(scores) | set(errors)):
        cases = scores.get(agent, {})
        if agent in errors or set(cases) != set(case_ids) or any(len(rows) != n_attempts for rows in cases.values()):
            agents[agent] = {"status": "incomparable", "reasons": errors.get(agent, ["Incomplete agent coverage"])}
            continue
        per_case = {}
        for case_id, attempts in cases.items():
            baseline = math.fsum(row["baseline"] for row in attempts) / n_attempts
            combined = math.fsum(row["combined"] for row in attempts) / n_attempts
            individuals = [
                math.fsum(row["individuals"][index] for row in attempts) / n_attempts
                for index in range(len(attempts[0]["individuals"]))
            ]
            per_case[case_id] = interaction_scores(baseline, individuals, combined)
        agents[agent] = _summarize_case_interactions(per_case, evidence[agent])
    complete = bool(agents) and all(score["status"] == "complete" for score in agents.values())
    return {"status": "complete" if complete else "incomparable", "agents": agents}


def execute_interleaved(
    report: dict[str, Any],
    plan: Sequence[tuple[str, int, tuple[Path, ...], str]],
    entries: Sequence[dict[str, Any]],
    options: EvaluationOptions,
    *,
    frozen: Sequence[Path],
    evaluator: Path,
    seed: int,
    on_configuration: Callable[[str], None] | None,
    progress_reporter: ProgressReporter | None,
) -> dict[str, Any]:
    """Run one real engine invocation per scheduled configuration/case/attempt."""
    from skillevaluator.tier3.harbor.progress import redact_progress_detail, secret_values_from_environment

    configs = {
        name: {"id": name, "condition": condition, "skills": [] if name == "baseline" else [index], "runs": []}
        for name, index, _, condition in plan
    }
    configs["combined"]["skills"] = list(range(len(report["skills"])))
    report["configurations"] = list(configs.values())
    entries_by_id = {entry["id"]: entry for entry in entries}
    report["schedule"] = interaction_schedule(
        list(entries_by_id), list(configs), n_attempts=report["n_attempts"], seed=seed
    )
    root = Path(report["run_dir"])
    report_path = root / "interaction.json"
    by_name = {name: (index, bool(companions), condition) for name, index, companions, condition in plan}

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    service = EvaluationService()
    active_case = None
    selected: Sequence[Path] = ()
    with ExitStack() as staging:
        for ordinal, step in enumerate(report["schedule"]):
            name, case_id = step["configuration"], step["case_id"]
            index, combined, condition = by_name[name]
            if case_id != active_case:
                staging.close()
                case_root = Path(
                    staging.enter_context(tempfile.TemporaryDirectory(prefix="skillevaluator-interaction-case-"))
                )
                selected = prepare_case_sources([entries_by_id[case_id]], frozen, evaluator, case_root)[case_id]
                active_case = case_id
            run = {"case_id": case_id, "attempt": step["attempt"]}
            configs[name]["runs"].append(run)
            step["status"] = "running"
            save()
            if on_configuration:
                on_configuration(f"{name} ({case_id}, attempt {step['attempt']})")
            try:
                result = service.evaluate(
                    replace(
                        options,
                        skill_path=selected[index],
                        results_dir=root / f"step-{ordinal:06d}" / name,
                        skip_baseline=condition == "with-skill",
                        skip_with_skill=condition == "without-skill",
                        n_attempts=1,
                        stop_on_pass=False,
                        skill_workspace_mode="group" if combined else "isolated",
                        include_skills=tuple(selected[1:]) if combined else (),
                    ),
                    progress_reporter=progress_reporter,
                )
                run["result"] = result
                if failure := service.failure_reason(result):
                    run["error"] = failure
                step["status"] = "failed" if "error" in run else "complete"
            except KeyboardInterrupt:
                step["status"] = "interrupted"
                save()
                raise
            except Exception as exc:
                message = redact_progress_detail(str(exc), secret_values=secret_values_from_environment(os.environ))
                run["error"] = message or type(exc).__name__
                step["status"] = "failed"
            save()
    return compare_interleaved(
        report["configurations"], case_ids=list(entries_by_id), metric=report["metric"], n_attempts=report["n_attempts"]
    )
