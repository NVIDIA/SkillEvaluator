# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Harbor-shaped fixtures for the f90 lift regression tests.

Each helper writes what a real run leaves on disk: per-trial ``reward.json``
plus the verifier's ``skill_evaluator_reward.json`` sidecar (N/A markers live
there), a trial ``result.json`` (with ``exception_info`` for a failed trial),
and the job ``result.json``. The tests then run the real collector, statistics,
report payload and reporters on it. Values come from the proof examples
(check-13, check-14, check-20) in a small synthetic form.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS

SKILL = "demo"
NA = None  # the verifier recorded the metric as not applicable
SKILL_METRICS = ("skill_execution", "skill_efficiency")
MEMBERS = ["alpha", "beta"]


def metrics(value: float, *, skill: bool = True, **overrides: float | None) -> dict[str, float | None]:
    """All six default metrics at *value*; ``skill=False`` makes the skill metrics N/A (no skill in the arm)."""
    scores: dict[str, float | None] = dict.fromkeys(DEFAULT_METRICS, value)
    if not skill:
        scores.update(dict.fromkeys(SKILL_METRICS, NA))
    scores.update(overrides)
    return scores


def _verifier_outputs(case_id: str, scores: Mapping[str, float | None], *, has_skill: bool) -> tuple[dict, dict]:
    details: dict[str, Any] = {}
    numeric: dict[str, Any] = {}
    sidecar: dict[str, Any] = {"metric_set": DEFAULT_METRIC_SET, "entry_id": case_id, "has_skill": has_skill}
    for metric in DEFAULT_METRICS:
        value = scores.get(metric)
        sidecar[metric] = value
        if value is None:
            details[metric] = {"score": None, "status": "not_applicable", "reason": "N/A"}
        else:
            numeric[metric] = value
            details[metric] = {"score": value}
    sidecar["details"] = details
    scored = [value for value in scores.values() if value is not None]
    numeric["overall"] = round(sum(scored) / len(scored), 4)
    numeric["entry_id"] = case_id
    return numeric, sidecar


def write_job(
    jobs_dir: Path,
    agent: str,
    variant: str,
    cases: Mapping[str, list[Mapping[str, float | None] | None]],
) -> None:
    """Write one Harbor job; a ``None`` attempt is a trial that timed out (no reward)."""
    job_dir = jobs_dir / f"{SKILL}-{agent}-{variant}"
    names: list[str] = []
    errors = 0
    for case_id, attempts in cases.items():
        for index, scores in enumerate(attempts, start=1):
            name = f"{case_id}__attempt{index:03d}"
            names.append(name)
            trial = job_dir / name
            (trial / "verifier").mkdir(parents=True)
            result: dict[str, Any] = {"trial_name": name, "task_name": case_id}
            if scores is None:
                errors += 1
                result["exception_info"] = {"exception_type": "AgentTimeoutError", "exception_message": "timeout"}
            else:
                numeric, sidecar = _verifier_outputs(case_id, scores, has_skill=variant == "with")
                (trial / "verifier" / "reward.json").write_text(json.dumps(numeric), encoding="utf-8")
                (trial / "verifier" / "skill_evaluator_reward.json").write_text(json.dumps(sidecar), encoding="utf-8")
            (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(names),
                "stats": {
                    "n_trials": len(names),
                    "n_errors": errors,
                    "evals": {
                        f"{agent}__model": {
                            "n_trials": len(names),
                            "n_errors": errors,
                            "reward_stats": {"reward": {"0.5": names}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def collect(
    tmp_path: Path,
    arms: Mapping[str, Mapping[str, Mapping[str, list[Any]]]],
    *,
    n_attempts: int,
    stop_on_pass: bool = False,
    plugin_signals: Any = None,
) -> dict[str, Any]:
    """Write every ``{agent: {variant: cases}}`` job and run the real collector."""
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    jobs = tmp_path / "jobs"
    case_ids: list[str] = []
    for agent, variants in arms.items():
        for variant, cases in variants.items():
            write_job(jobs, agent, variant, cases)
            case_ids.extend(case for case in cases if case not in case_ids)
    first = next(iter(arms.values()))
    return collect_harbor_results(
        skill_name=SKILL,
        agents=list(arms),
        output_dir=tmp_path / "results",
        jobs_dir=jobs,
        skip_baseline="without" not in first,
        sum_of_parts_arm="sumofparts" in first,
        n_attempts=n_attempts,
        stop_on_pass=stop_on_pass,
        expected_cases=len(case_ids),
        expected_case_ids=case_ids,
        expected_trials=None if stop_on_pass else n_attempts * len(case_ids),
        plugin_signals=plugin_signals,
    )


def run_config(lift_mode: str = "effectiveness", *, members: list[str] | None = None) -> dict[str, Any]:
    staged = list(MEMBERS if members is None else members)
    return {
        "eval_target": {"kind": "plugin"},
        "skill_workspace": {
            "mode": "group",
            "staged_skills": staged,
            "include": [f"/stage/{name}" for name in staged],
            "baseline_includes_workspace_skills": lift_mode == "integration",
            "sum_of_parts_arm": lift_mode == "both",
        },
        "lift_mode": {"requested": lift_mode, "effective": lift_mode, "integration_skip_reason": None},
    }


def provenance(lift_mode: str = "effectiveness", **extra: Any) -> dict[str, Any]:
    return {
        "plugin_name": SKILL,
        "requested_lift_mode": lift_mode,
        "effective_lift_mode": lift_mode,
        "partial": False,
        "cross_component_case_count": 1,
        "integration_evidence_ready": True,
        "evaluated_member_skills": list(MEMBERS),
        **extra,
    }


def payload(tmp_path: Path, lift_mode: str = "effectiveness", **provenance_extra: Any) -> dict[str, Any]:
    """Build the report payload from the collected run directory, as the reporters do."""
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    built = build_agent_eval_payload(
        SKILL,
        load_agent_data(tmp_path / "results"),
        run_config=run_config(lift_mode),
        plugin_provenance=provenance(lift_mode, **provenance_extra),
        use_llm_judge=False,
    )
    assert built is not None
    return built


def result_for(built: Mapping[str, Any]) -> Any:
    """Wrap a payload in the ValidationResult shape every reporter reads."""
    from skillevaluator.models import ValidationResult

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Run live agent evaluation")
    result.metadata["agent_eval"] = dict(built)
    result.metadata["gating"] = {"tier": 3, "blocking": False}
    return result


def benchmark(built: Mapping[str, Any]) -> str:
    from skillevaluator.reporting import BenchmarkReporter

    reporter = BenchmarkReporter(include_timestamp=False, skill_name=SKILL, content_type="plugin")
    return reporter.render_all([result_for(built)])


def markdown(built: Mapping[str, Any]) -> str:
    from skillevaluator.reporting.markdown import MarkdownReporter

    return MarkdownReporter(include_timestamp=False).render_all([result_for(built)])


def html(built: Mapping[str, Any]) -> str:
    from skillevaluator.reporting.html import HTMLReporter

    return HTMLReporter().render_all([result_for(built)])


def cli(built: Mapping[str, Any]) -> str:
    from rich.console import Console

    from skillevaluator.reporting.cli import CLIReporter

    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=False, color_system=None)
    CLIReporter._print_agent_eval_tables(dict(built), console)
    return buffer.getvalue()
