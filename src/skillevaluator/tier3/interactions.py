# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, fixed-source experiments measuring departure from additive skill gains."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.evaluation import EvaluationOptions, EvaluationService
from skillevaluator.tier3.decoy_identity import (
    DECOY_SOURCE_DIGEST_ALGORITHM,
    decoy_source_copy_ignore,
    decoy_source_digest,
)
from skillevaluator.tier3.harbor.secure_copy import copytree_secure, tree_content_fingerprint_secure

if TYPE_CHECKING:
    from skillevaluator.tier3.harbor.progress import ProgressReporter

OUTCOME_METRICS = ("goal_accuracy", "accuracy", "overall")
_RUNTIME_ENTRY_FIELDS = frozenset({"has_skill", "evaluated_skill", "skill_workspace_mode", "workspace_skill_names"})


class IncomparableInteraction(ValueError):
    """Evidence cannot support an interaction estimate."""


def interaction_scores(baseline: float, individuals: Sequence[float], combined: float) -> dict[str, Any]:
    """Calculate the signed interaction from comparable outcome means."""
    if len(individuals) < 2:
        raise ValueError("Interaction evaluation requires at least two skills")
    for score in (baseline, *individuals, combined):
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Outcome scores must be finite numbers between 0 and 1")
    lift = combined - baseline
    individual_lifts = [score - baseline for score in individuals]
    interaction = lift - math.fsum(individual_lifts)
    if math.isclose(interaction, 0.0, abs_tol=1e-12):
        interaction = 0.0
    return {
        "baseline": baseline,
        "individuals": list(individuals),
        "combined": combined,
        "individual_lifts": individual_lifts,
        "combined_lift": lift,
        "interaction": interaction,
        "classification": "positive" if interaction > 0 else "negative" if interaction < 0 else "zero",
    }


def _read_json(path: Path, root: Path) -> Any:
    from skillevaluator.tier3.harbor.adapter import _read_regular_evals_file

    return json.loads(_read_regular_evals_file(path, allowed_root=root, max_bytes=16 * 1024 * 1024))


def _case_contract(result: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    snapshot = _read_json(run_dir / "dataset_snapshot.json", run_dir)
    entries = snapshot.get("dataset") if isinstance(snapshot, dict) else None
    if not isinstance(entries, list) or not entries:
        raise IncomparableInteraction("Missing retained dataset cases")
    cases = {}
    for entry in entries:
        case_id = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(case_id, str) or not case_id or case_id in cases:
            raise IncomparableInteraction("Missing or duplicate dataset case identity")
        cases[case_id] = {key: value for key, value in entry.items() if key not in _RUNTIME_ENTRY_FIELDS}
    selection = result.get("run_config", {}).get("case_selection", {})
    if not isinstance(selection, Mapping) or selection.get("skipped", 0):
        raise IncomparableInteraction("Partial task suites cannot support an interaction estimate")
    return cases


def _outcome_rows(
    run_dir: Path, agent: str, condition: str, metric: str, *, custom_only: bool
) -> dict[str, list[dict[str, Any]]]:
    from skillevaluator.tier3.harbor.collector import _reward_judge_models
    from skillevaluator.tier3.harbor.metrics import metric_is_not_applicable, metric_value, overall_score

    trials = run_dir / agent / condition / "trials"
    rows: dict[str, list[dict[str, Any]]] = {}
    seen_trials: set[str] = set()
    for path in sorted(trials.glob("*/reward.json")):
        reward = _read_json(path, run_dir)
        if not isinstance(reward, dict) or reward.get("evaluation_status") in {"error", "failed"}:
            raise IncomparableInteraction("Unscored trial in retained evidence")
        case_id, trial_id = reward.get("entry_id"), reward.get("trial_id")
        if not isinstance(case_id, str) or not isinstance(trial_id, str) or not case_id or not trial_id:
            raise IncomparableInteraction("Missing trial or case identity")
        if trial_id in seen_trials:
            raise IncomparableInteraction("Multiple reward rows for one logical attempt; no comparable outcome")
        seen_trials.add(trial_id)
        value = overall_score(reward) if metric == "overall" else metric_value(reward, metric)
        if (
            value is None
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            or (metric != "overall" and metric_is_not_applicable(reward, metric))
        ):
            raise IncomparableInteraction(f"Missing, N/A, or invalid {metric} outcome for {case_id}")
        models = sorted(_reward_judge_models(reward, metric))
        if not custom_only and len(models) != 1:
            raise IncomparableInteraction(f"Unknown or mixed outcome scorer for {case_id}")
        details = reward.get("details")
        detail = details.get(metric) if isinstance(details, dict) else None
        scorer = {
            "judge_models": models,
            "method": detail.get("method") if isinstance(detail, dict) else None,
            "metric_set": reward.get("metric_set"),
            "score_policy": reward.get("score_policy"),
        }
        rows.setdefault(case_id, []).append(
            {"score": value, "trial_id": trial_id, "reward_path": str(path), "scorer": scorer}
        )
    return rows


def compare_configurations(
    configurations: Sequence[Mapping[str, Any]], *, metric: str, n_attempts: int
) -> dict[str, Any]:
    """Require complete per-case, per-agent evidence before calculating a score.

    Every case has equal weight; attempts are averaged within a case first.
    Configuration order is baseline, singleton skills, combined skills.
    """
    if len(configurations) < 4:
        raise ValueError("Expected baseline, at least two individuals, and combined evidence")
    failures = [
        f"{config['id']}: {config.get('error') or 'Configuration failed or returned no result'}"
        for config in configurations
        if "error" in config or "result" not in config
    ]
    if failures:
        return {"status": "incomplete", "reasons": failures, "agents": {}}
    results = [config["result"] for config in configurations]
    failures = [
        f"{config['id']}: {failure}"
        for config, result in zip(configurations, results, strict=True)
        if (failure := EvaluationService.failure_reason(result))
    ]
    if failures:
        return {"status": "incomplete", "reasons": failures, "agents": {}}
    if any(not isinstance(result.get("agents"), Mapping) for result in results):
        return {"status": "incomplete", "reasons": ["Missing agent results"], "agents": {}}
    agent_names = sorted(set().union(*(set(result.get("agents", {})) for result in results)))
    compared = {}
    for agent in agent_names:
        try:
            signatures, case_contracts, outcomes = [], [], []
            for config, result in zip(configurations, results, strict=True):
                failure = EvaluationService.failure_reason(result)
                if failure:
                    raise IncomparableInteraction(failure)
                info = result.get("agents", {}).get(agent)
                if not isinstance(info, dict) or not info.get("model"):
                    raise IncomparableInteraction(f"Missing agent/model identity: {agent}")
                run_dir = Path(result["run_dir"])
                run_config = result.get("run_config", {})
                if not isinstance(run_config, Mapping):
                    raise IncomparableInteraction("Missing run configuration identity")
                harbor = run_config.get("harbor", {})
                if not isinstance(harbor, Mapping):
                    raise IncomparableInteraction("Missing attempt configuration")
                if harbor.get("n_attempts") != n_attempts or harbor.get("stop_on_pass") is not False:
                    raise IncomparableInteraction("Unequal attempt budgets or early stopping")
                grading = run_config.get("grading", {})
                if not isinstance(grading, Mapping):
                    raise IncomparableInteraction("Missing grading configuration")
                custom_only = grading.get("mode") == "custom_only"
                if (metric == "overall") != custom_only:
                    raise IncomparableInteraction(
                        "Use overall only for custom_only grading; otherwise select an outcome metric"
                    )
                judge_identity = run_config.get("judge")
                if isinstance(judge_identity, Mapping):
                    # Each arm probes the model catalog independently.
                    judge_identity = {k: v for k, v in judge_identity.items() if k != "catalog_verification"}
                if not judge_identity or not harbor.get("environment") or not result.get("evaluator_version"):
                    raise IncomparableInteraction("Missing scorer, environment, or evaluator identity")
                signatures.append(
                    (
                        info["model"],
                        judge_identity,
                        grading,
                        harbor.get("environment"),
                        result.get("evaluator_version"),
                    )
                )
                cases = _case_contract(result, run_dir)
                rows = _outcome_rows(run_dir, agent, config["condition"], metric, custom_only=custom_only)
                if rows.keys() != cases.keys() or any(len(attempts) != n_attempts for attempts in rows.values()):
                    raise IncomparableInteraction("Incomplete or unequal per-case attempt coverage")
                case_contracts.append(cases)
                outcomes.append(rows)
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise IncomparableInteraction("Agent model, scorer, environment, or evaluator identities differ")
            if any(cases != case_contracts[0] for cases in case_contracts[1:]):
                raise IncomparableInteraction("Task suite identities or case contents differ")
            per_case = {}
            for case_id in sorted(case_contracts[0]):
                scorers = {json.dumps(row["scorer"], sort_keys=True) for rows in outcomes for row in rows[case_id]}
                if len(scorers) != 1:
                    raise IncomparableInteraction(
                        f"Outcome scorer differs across configurations or attempts: {case_id}"
                    )
                means = [math.fsum(row["score"] for row in rows[case_id]) / n_attempts for rows in outcomes]
                per_case[case_id] = interaction_scores(means[0], means[1:-1], means[-1])
            means = [
                math.fsum(math.fsum(row["score"] for row in rows[case]) / n_attempts for case in per_case)
                / len(per_case)
                for rows in outcomes
            ]
            compared[agent] = {
                "status": "complete",
                **interaction_scores(means[0], means[1:-1], means[-1]),
                "cases": per_case,
                "evidence": {config["id"]: rows for config, rows in zip(configurations, outcomes, strict=True)},
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            compared[agent] = {"status": "incomparable", "reasons": [str(exc)]}
    if not compared:
        return {"status": "incomplete", "reasons": ["No agent results were retained"], "agents": {}}
    complete = all(agent["status"] == "complete" for agent in compared.values())
    return {"status": "complete" if complete else "incomparable", "agents": compared}


def _selected_skills(skills: Sequence[Path]) -> list[Path]:
    from skillevaluator.tier3.harbor.runner import _workspace_skill_name
    from skillevaluator.tier3.workflow import _validate_skill_root

    if len(skills) < 2:
        raise ValueError("Interaction evaluation requires at least two selected skills")
    selected, paths, names = [], set(), set()
    for path in skills:
        _validate_skill_root(path)
        path = path.resolve()
        name = _workspace_skill_name(path).casefold()
        if path in paths or name in names or path.name.casefold() in {item.name.casefold() for item in selected}:
            raise ValueError("Selected skills must have distinct paths, directory names, and skill names")
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths):
            raise ValueError("Selected skill roots must not overlap")
        selected.append(path)
        paths.add(path)
        names.add(name)
    return selected


def _validate_source(source: Path, metric: str) -> None:
    from skillevaluator.tier3.commands import _validate_existing_scaffold_state
    from skillevaluator.tier3.evals_config import load_evals_config
    from skillevaluator.tier3.evals_spec import validate_tier3_source

    _validate_existing_scaffold_state(source, source / "evals")
    _, checks = validate_tier3_source(source)
    errors = [check.message for check in checks if check.status in {"error", "missing"}]
    if errors:
        raise ValueError("Invalid fixed evaluation source: " + "; ".join(errors))
    config, _ = load_evals_config(source)
    workspace = config.get("skill_workspace", {})
    if workspace.get("include") or "decoy_manifest" in workspace:
        raise ValueError(
            "Interaction skills are explicit; remove skill_workspace.include/decoy_manifest from the source"
        )
    if config.get("harbor", {}).get("max_cases"):
        raise ValueError("Interaction evaluation requires the full reviewed suite; remove harbor.max_cases")
    if config.get("harbor", {}).get("pre_agent_setup"):
        raise ValueError("Interaction evaluation requires a no-skill baseline; remove harbor.pre_agent_setup")
    custom_only = config.get("grading", {}).get("mode") == "custom_only"
    if metric not in OUTCOME_METRICS or (metric == "overall") != custom_only:
        raise ValueError("Select goal_accuracy or accuracy for standard grading, or overall for custom_only grading")


def run_interaction(
    skills: Sequence[Path],
    *,
    eval_source: Path,
    options: EvaluationOptions,
    metric: str = "goal_accuracy",
    on_configuration: Callable[[str], None] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Freeze reviewed inputs once, then delegate every arm to the Tier 3 service."""
    from skillevaluator.tier3.evals_config import parse_environment_kwarg_overrides
    from skillevaluator.tier3.harbor.adapter import (
        private_evaluator_skill_snapshot,
        validate_output_provenance_key_location,
    )
    from skillevaluator.tier3.harbor.runner import _reserve_run_dir
    from skillevaluator.tier3.results_location import ENV_RESULTS_DIR
    from skillevaluator.tier3.workflow import _validate_skill_root

    selected = _selected_skills(skills)
    _validate_skill_root(eval_source)
    source = eval_source.resolve()
    if not (source / "evals").is_dir():
        raise ValueError(f"No evaluator source directory found at {source / 'evals'}")
    parse_environment_kwarg_overrides(options.environment_kwarg, env_mode=options.env_mode)
    if (
        options.include_skills
        or options.copy_repo
        or options.skip_baseline
        or options.skip_with_skill
        or options.max_cases
    ):
        raise ValueError("Interaction evaluation controls skill selection and requires full, uncopied task suites")
    attempts = options.n_attempts if options.n_attempts is not None else 1
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("n_attempts must be a positive integer")
    root = Path(options.results_dir or os.environ.get(ENV_RESULTS_DIR) or source.parent / "interaction-results")
    root = root.expanduser().resolve()
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in [source, *selected]):
        raise ValueError("Save interaction results outside the evaluation source and selected skill roots")
    validate_output_provenance_key_location(source, root, reference_skills_dir=None, workspace_skill_paths=selected)
    service = EvaluationService()
    with (
        private_evaluator_skill_snapshot(source) as evaluator,
        tempfile.TemporaryDirectory(prefix="skillevaluator-interaction-") as directory,
    ):
        _validate_source(evaluator, metric)
        frozen, identities = [], []
        for index, path in enumerate(selected):
            isolated_root = Path(directory) / f"skill-{index + 1}"
            isolated_root.mkdir()
            destination = isolated_root / path.name
            digest = decoy_source_digest(path)
            if digest is None:
                raise ValueError(f"Unsafe or unstable selected skill: {path}")
            copytree_secure(path, destination, allowed_root=path, ignore=decoy_source_copy_ignore(path))
            if decoy_source_digest(destination) != digest:
                raise ValueError(f"Skill source changed while freezing: {path}")
            identities.append({"path": str(path), "name": path.name, "source_digest": digest})
            copytree_secure(evaluator / "evals", destination / "evals", allowed_root=evaluator)
            frozen.append(destination)
        run_dir = _reserve_run_dir(root, datetime.now(UTC).strftime("interaction_%Y%m%d_%H%M%S"))
        copytree_secure(evaluator / "evals", run_dir / "source" / "evals", allowed_root=evaluator)
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "skill_interaction",
            "definition": "[P(S) - P(empty)] - sum(P(skill) - P(empty) for skill in S)",
            "interpretation": "Total departure from additive individual gains; not an isolated higher-order interaction.",
            "metric": metric,
            "n_attempts": attempts,
            "source": {"path": str(source), "fingerprint": tree_content_fingerprint_secure(evaluator / "evals")},
            "skill_digest_algorithm": DECOY_SOURCE_DIGEST_ALGORITHM,
            "skills": identities,
            "run_dir": str(run_dir),
            "configurations": [],
            "status": "incomplete",
        }
        report_path = run_dir / "interaction.json"
        plan = [("baseline", 0, (), "without-skill")]
        plan.extend((f"individual-{index + 1}", index, (), "with-skill") for index in range(len(frozen)))
        plan.append(("combined", 0, tuple(frozen[1:]), "with-skill"))
        try:
            for config_id, index, companions, condition in plan:
                config: dict[str, Any] = {
                    "id": config_id,
                    "condition": condition,
                    "skills": []
                    if config_id == "baseline"
                    else [index]
                    if config_id != "combined"
                    else list(range(len(frozen))),
                }
                report["configurations"].append(config)
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                if on_configuration:
                    on_configuration(config_id)
                try:
                    result = service.evaluate(
                        replace(
                            options,
                            skill_path=frozen[index],
                            results_dir=run_dir / config_id,
                            skip_baseline=condition == "with-skill",
                            skip_with_skill=condition == "without-skill",
                            n_attempts=attempts,
                            stop_on_pass=False,
                            skill_workspace_mode="group" if companions else "isolated",
                            include_skills=companions,
                        ),
                        progress_reporter=progress_reporter,
                    )
                    config["result"] = result
                    if failure := service.failure_reason(result):
                        config["error"] = failure
                except Exception as exc:
                    from skillevaluator.tier3.harbor.progress import (
                        redact_progress_detail,
                        secret_values_from_environment,
                    )

                    redacted = redact_progress_detail(
                        str(exc), secret_values=secret_values_from_environment(os.environ)
                    )
                    config["error"] = redacted or type(exc).__name__
            report.update(compare_configurations(report["configurations"], metric=metric, n_attempts=attempts))
        except KeyboardInterrupt:
            report["reasons"] = ["Interaction evaluation interrupted before all configurations completed"]
            raise
        finally:
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
