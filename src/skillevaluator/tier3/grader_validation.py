# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification of the shipped verifier against independent, versioned labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from skillevaluator.provider_config import resolve_llm_provider
from skillevaluator.tier3.eval_core.atif_helpers import (
    extract_tool_calls_as_dicts,
    get_final_response,
)

SCHEMA = "skillevaluator.grader-qualification/v1"
CATEGORIES = {"positive", "negative", "edge", "failure_recovery", "unavailable_evidence", "non_use", "baseline"}
METRICS = ("security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check")
VERDICTS = ("pass", "fail", "not_applicable", "error", "insufficient_evidence")
DEFAULT_THRESHOLDS = dict.fromkeys(METRICS, 1.0) | {"accuracy": 0.8, "overall": 0.5}
TEMPLATE = Path(__file__).parent / "harbor" / "templates" / "eval.py"
REFERENCE_SUITE = Path(__file__).parent / "grader_reference"
LIMITATIONS = (
    "Results describe this suite, scorer and model only. A fixed case count or score band is not "
    "evidence of statistical sufficiency or universal grader reliability. Repeats of the same case "
    "are dependent; attempt-level Wilson intervals do not account for case clustering."
)
ENTRY_FIELDS = {
    "id",
    "question",
    "ground_truth",
    "expected_behavior",
    "expected_skill",
    "expected_script",
    "should_trigger",
    "has_skill",
    "evaluated_skill",
    "acceptable_skills",
    "acceptable_alternates",
    "skill_workspace_mode",
    "workspace_skill_names",
}


def _object(value, allowed, required, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    unknown, missing = value.keys() - allowed, required - value.keys()
    if unknown or missing:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}; missing fields {sorted(missing)}")


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected nonempty string")


def _relative(root, value):
    _text(value, "trajectory")
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError("trajectory must stay inside suite directory")
    if not path.is_file():
        raise ValueError(f"trajectory file not found: {value}")
    return path


def load_suite(directory):
    """Strictly validate a suite and re-verify its executable labels."""
    root = Path(directory).resolve()
    suite = json.loads((root / "suite.json").read_text())
    _object(suite, {"schema", "suite", "thresholds", "cases"}, {"schema", "suite", "cases"}, "suite")
    if suite["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    _object(suite["suite"], {"name", "version", "description"}, {"name", "version", "description"}, "suite metadata")
    for key, value in suite["suite"].items():
        _text(value, f"suite.{key}")
    thresholds = suite.get("thresholds", {})
    _object(thresholds, set(DEFAULT_THRESHOLDS), set(), "thresholds")
    for key, value in thresholds.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"thresholds.{key}: expected number in [0,1]")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("cases must be a nonempty list")
    ids = set()
    for case in suite["cases"]:
        fields = {"id", "split", "category", "source", "entry", "trajectory", "expected", "label"}
        _object(case, fields, fields, "case")
        case_id = case["id"]
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case_id):
            raise ValueError("case id must contain only letters, digits, hyphens and underscores")
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        if not isinstance(case["split"], str) or case["split"] not in {"assessment", "tuning"}:
            raise ValueError(f"{case_id}: invalid split")
        if not isinstance(case["category"], str) or case["category"] not in CATEGORIES:
            raise ValueError(f"{case_id}: unknown category")
        _object(case["source"], {"kind", "origin", "trace_id"}, {"kind", "origin", "trace_id"}, "source")
        for value in case["source"].values():
            _text(value, "source")
        entry = case["entry"]
        _object(entry, ENTRY_FIELDS, {"question"}, "entry")
        for key in (
            "question",
            "ground_truth",
            "expected_skill",
            "expected_script",
            "evaluated_skill",
            "id",
            "skill_workspace_mode",
        ):
            if key in entry and not isinstance(entry[key], str):
                raise ValueError(f"entry.{key}: expected string")
        for key in ("should_trigger", "has_skill"):
            if key in entry and not isinstance(entry[key], bool):
                raise ValueError(f"entry.{key}: expected boolean")
        for key in ("expected_behavior", "acceptable_skills", "acceptable_alternates", "workspace_skill_names"):
            if key in entry and (not isinstance(entry[key], list) or any(not isinstance(v, str) for v in entry[key])):
                raise ValueError(f"entry.{key}: expected list of strings")
        expected = case["expected"]
        _object(expected, {"overall", "metrics"}, {"overall", "metrics"}, "expected")
        if not isinstance(expected["overall"], str) or expected["overall"] not in {
            "pass",
            "fail",
            "insufficient_evidence",
        }:
            raise ValueError("invalid expected overall")
        _object(expected["metrics"], set(METRICS), set(), "expected.metrics")
        for metric, label in expected["metrics"].items():
            if metric == "behavior_check" and isinstance(label, list):
                if len(label) != len(entry.get("expected_behavior", [])) or any(type(v) is not bool for v in label):
                    raise ValueError("behavior label length must match expected_behavior and contain booleans")
            elif not isinstance(label, str) or label not in set(VERDICTS) - {"error"}:
                raise ValueError(f"invalid expected metric label: {metric}")
        label = case["label"]
        fields = {"method", "reviewed_by", "reviewed_on", "rationale", "checks"}
        _object(label, fields, fields, "label")
        if not isinstance(label["method"], str) or label["method"] not in {"executable", "reviewed"}:
            raise ValueError("label.method must be executable or reviewed")
        for key in ("reviewed_by", "reviewed_on", "rationale"):
            _text(label[key], f"label.{key}")
        try:
            datetime.strptime(label["reviewed_on"], "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError as exc:
            raise ValueError("reviewed_on must be YYYY-MM-DD") from exc
        trajectory = None
        if case["trajectory"] is not None:
            trajectory = json.loads(_relative(root, case["trajectory"]).read_text())
            if (
                not isinstance(trajectory, dict)
                or not isinstance(trajectory.get("steps"), list)
                or not trajectory["steps"]
                or any(not isinstance(step, dict) for step in trajectory["steps"])
            ):
                raise ValueError(f"{case_id}: trajectory must have nonempty ATIF steps")
        elif case["category"] != "unavailable_evidence" or expected["overall"] != "insufficient_evidence":
            raise ValueError("null trajectory requires unavailable_evidence with insufficient_evidence label")
        verify_checks(label, trajectory, case_id)
    return suite


def verify_checks(label, trajectory, case_id):
    """Assert only recorded evidence; never ask a grader to establish a label."""
    checks = label["checks"]
    if not isinstance(checks, list) or (label["method"] == "executable" and not checks):
        raise ValueError(f"{case_id}: executable labels require checks")
    for check in checks:
        _object(check, {"type", "value", "path"}, {"type", "value"}, "label check")
        _text(check["value"], "check.value")
        kind = check["type"]
        if not isinstance(kind, str) or kind not in {
            "final_response_contains",
            "final_response_not_contains",
            "tool_result_contains",
            "file_written_contains",
        }:
            raise ValueError(f"unknown executable check: {kind}")
        if trajectory is None:
            raise ValueError("executable check needs trajectory")
        if kind.startswith("final_response"):
            found = check["value"] in get_final_response(trajectory)
            passed = not found if kind.endswith("not_contains") else found
        elif kind == "tool_result_contains":
            observations = [step.get("observation", {}) for step in trajectory["steps"]]
            passed = check["value"] in json.dumps(observations, ensure_ascii=False)
        else:
            _text(check.get("path"), "check.path")
            passed = any(
                call.get("action", "").lower() in {"write", "write_file"}
                and call.get("action_input", {}).get("file_path") == check["path"]
                and check["value"] in call.get("action_input", {}).get("content", "")
                for call in extract_tool_calls_as_dicts(trajectory)
            )
        if not passed:
            raise ValueError(f"{case_id}: executable label check failed: {kind}")


def rate(numerator, denominator):
    """A proportion and Wilson 95% bounds; undefined when denominator is zero."""
    if not denominator:
        return {"numerator": numerator, "denominator": 0, "rate": None, "wilson_95": None}
    p, z = numerator / denominator, 1.959963984540054
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": p,
        "wilson_95": [max(0, center - half), min(1, center + half)],
    }


def classify(reward, metric, thresholds):
    if str(reward.get("error", "")).startswith("No trajectory"):
        return "insufficient_evidence"
    detail = reward.get("details", {}).get(metric, {})
    if detail.get("status") in {"not_applicable", "error"}:
        return detail["status"]
    if reward.get("_process_error") or (metric == "overall" and reward.get("evaluation_status") == "failed"):
        return "error"
    score = reward.get(metric)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return "error"
    return "pass" if score >= thresholds[metric] else "fail"


def observations(case, reward, thresholds):
    result = []
    labels = {"overall": case["expected"]["overall"], **case["expected"]["metrics"]}
    for metric, expected in labels.items():
        observed = classify(reward, metric, thresholds)
        score = reward.get(metric)
        if isinstance(expected, list):
            result.append(
                {
                    "metric": metric,
                    "judgment": f"{case['id']}:{metric}",
                    "expected": "pass" if all(expected) else "fail",
                    "observed": observed,
                    "score": score,
                }
            )
            steps = reward.get("details", {}).get(metric, {}).get("results", [])
            for index, passed in enumerate(expected):
                verdict = observed
                step_score = None
                if observed in {"pass", "fail"}:
                    step = next((s for s in steps if s.get("step") == index + 1), {})
                    step_score = step.get("passed")
                    verdict = ("pass" if step_score else "fail") if type(step_score) is bool else "error"
                result.append(
                    {
                        "metric": "behavior_steps",
                        "judgment": f"{case['id']}:{metric}:{index + 1}",
                        "expected": "pass" if passed else "fail",
                        "observed": verdict,
                        "score": int(step_score) if type(step_score) is bool else None,
                    }
                )
        else:
            result.append(
                {
                    "metric": metric,
                    "judgment": f"{case['id']}:{metric}",
                    "expected": expected,
                    "observed": observed,
                    "score": score,
                }
            )
    return result


def summarize(rows):
    matrix = {expected: dict.fromkeys(VERDICTS, 0) for expected in VERDICTS if expected != "error"}
    groups = defaultdict(list)
    for row in rows:
        matrix[row["expected"]][row["observed"]] += 1
        groups[row["judgment"]].append(row)
    known = [row for row in rows if row["expected"] in {"pass", "fail"}]
    scored_fail = sum(matrix["fail"][v] for v in ("pass", "fail"))
    scored_pass = sum(matrix["pass"][v] for v in ("pass", "fail"))
    errors = sum(row["observed"] == "error" for row in known)
    insufficient = sum(row["observed"] == "insufficient_evidence" for row in known)
    repeated = [group for group in groups.values() if len(group) > 1]
    spreads = {}
    for judgment, group in groups.items():
        scores = [
            r["score"]
            for r in group
            if isinstance(r["score"], (int, float)) and not isinstance(r["score"], bool) and math.isfinite(r["score"])
        ]
        spreads[judgment] = max(scores) - min(scores) if scores else None
    return {
        "attempts": len(rows),
        "judgments": len(groups),
        "cases": len({r["judgment"].split(":")[0] for r in rows}),
        "expected_class_counts": dict(Counter(r["expected"] for r in rows)),
        "confusion_matrix": matrix,
        "false_pass_rate": rate(matrix["fail"]["pass"], scored_fail),
        "false_fail_rate": rate(matrix["pass"]["fail"], scored_pass),
        "unjudgeable_rate": rate(errors + insufficient, len(known)),
        "errors": errors,
        "insufficient_evidence": insufficient,
        "disagreement": sum(r["expected"] != r["observed"] for r in rows),
        "expected_insufficient": {
            "attempts": sum(matrix["insufficient_evidence"].values()),
            "correctly_abstained": matrix["insufficient_evidence"]["insufficient_evidence"],
            "errors": matrix["insufficient_evidence"]["error"],
            "confident_verdict": sum(matrix["insufficient_evidence"][v] for v in ("pass", "fail")),
        },
        "variability": rate(sum(len({r["observed"] for r in group}) > 1 for group in repeated), len(repeated)),
        "score_spread_by_judgment": spreads,
    }


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _attempt(root, case, repeat, output, thresholds, timeout):
    directory = output / case["id"] / f"attempt-{repeat}"
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "entry.json", case["entry"])
    env = os.environ.copy()
    paths = {
        "ENTRY_JSON": directory / "entry.json",
        "ATIF_PATH": directory / "trajectory.json",
        "REWARD_JSON": directory / "reward.json",
        "REWARD_TXT": directory / "reward.txt",
        "VERIFIER_DIR": directory,
        "LOGS_DIR": directory,
        "SKILL_EVALUATOR_REWARD_JSON": directory / "skill_evaluator_reward.json",
    }
    if case["trajectory"] is not None:
        (directory / "trajectory.json").write_bytes(_relative(root, case["trajectory"]).read_bytes())
    env.update({f"HARBOR_{key}": str(value) for key, value in paths.items()})
    try:
        process = subprocess.run(
            [sys.executable, str(TEMPLATE)], env=env, capture_output=True, timeout=timeout, check=False
        )
        code = process.returncode
    except subprocess.TimeoutExpired:
        code = -1
    reward = {}
    for filename in ("reward.json", "skill_evaluator_reward.json"):
        path = directory / filename
        if path.exists():
            try:
                reward.update(json.loads(path.read_text()))
            except (ValueError, OSError):
                code = -1
    if code and reward.get("evaluation_status") != "failed":
        reward["_process_error"] = True
    rows = observations(case, reward, thresholds)
    _write(directory / "expected-vs-observed.json", rows)
    if not (directory / "judge_attempts.json").exists():
        _write(directory / "judge_attempts.json", [])
    return {"case": case["id"], "repeat": repeat, "returncode": code, "reward": reward, "observations": rows}


def qualify(root, suite, cases, output, repeats, concurrency, timeout, split, provider):
    thresholds = DEFAULT_THRESHOLDS | suite.get("thresholds", {})
    tasks = [(case, repeat) for case in cases for repeat in range(1, repeats + 1)]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        attempts = list(executor.map(lambda task: _attempt(root, *task, output, thresholds, timeout), tasks))
    rows = [row for attempt in attempts for row in attempt["observations"]]
    summaries = {}
    for metric in (*METRICS, "behavior_steps", "overall"):
        subset = [row for row in rows if row["metric"] == metric]
        if subset:
            summaries[metric] = summarize(subset) | {
                "kind": "deterministic"
                if metric in METRICS[:3]
                else "model_judgment"
                if metric != "overall"
                else "composite"
            }
    digest = hashlib.sha256()
    files = ["suite.json", *sorted({c["trajectory"] for c in suite["cases"] if c["trajectory"]})]
    for filename in files:
        data = (root / filename).read_bytes()
        digest.update(filename.encode() + b"\0" + len(data).to_bytes(8, "big") + data)
    rewards = [attempt["reward"] for attempt in attempts]
    details = [
        detail for reward in rewards for detail in reward.get("details", {}).values() if isinstance(detail, dict)
    ]
    safe_env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "SKILL_EVAL_JUDGE_MODEL",
            "SKILL_EVAL_JUDGE_FALLBACK_MODEL",
            "SKILL_EVAL_JUDGE_REASONING_EFFORT",
            "SKILL_EVAL_GOAL_ACCURACY_METHOD",
            "SKILL_EVAL_LLM_JUDGE_BUDGET_SEC",
            "SKILL_EVAL_LLM_MAX_RETRIES",
            "SKILL_EVAL_LLM_RETRY_BASE_DELAY",
            "SKILL_EVAL_LLM_RETRY_MAX_DELAY",
            "SKILL_EVAL_ACCURACY_BUDGET",
            "SKILL_EVAL_GOAL_ACCURACY_BUDGET",
            "SKILL_EVAL_BEHAVIOR_CHECK_BUDGET",
            "SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT",
        }
    }
    result = {
        "schema": SCHEMA,
        "suite": suite["suite"],
        "metrics": summaries,
        "coverage": {
            "cases": len(cases),
            "attempts": len(attempts),
            "categories": dict(Counter(c["category"] for c in cases)),
            "splits": dict(Counter(c["split"] for c in cases)),
            "missing_categories": sorted(CATEGORIES - {c["category"] for c in cases}),
            "cases_with_trajectory": sum(c["trajectory"] is not None for c in cases),
            "truncated_judgments": sum(d.get("omitted", {}).get("truncated") is True for d in details),
            "evidence_judgments": sum("omitted" in d for d in details),
        },
        "provenance": {
            "suite_sha256": digest.hexdigest(),
            "hash_files": files,
            "split": split,
            "case_ids": [c["id"] for c in cases],
            "skillevaluator_version": version("skillevaluator"),
            "verifier_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
            "contracts": {
                key: sorted({r[key] for r in rewards if key in r})
                for key in ("judge_contract", "metric_set", "score_policy")
            },
            "judge_models_used": sorted({d["judge_model_used"] for d in details if d.get("judge_model_used")}),
            "goal_methods": sorted(
                {r.get("details", {}).get("goal_accuracy", {}).get("method") for r in rewards} - {None}
            ),
            "provider": provider.provider,
            "model": provider.model,
            "judge_environment": safe_env,
            "thresholds": thresholds,
            "repeats": repeats,
            "concurrency": concurrency,
            "timeout": timeout,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "attempts": attempts,
        "limitations": LIMITATIONS,
    }
    if split == "all":
        result["by_split"] = {
            name: {
                metric: summarize(
                    [
                        row
                        for attempt in attempts
                        if attempt["case"] in {c["id"] for c in cases if c["split"] == name}
                        for row in attempt["observations"]
                        if row["metric"] == metric
                    ]
                )
                | {"kind": summaries[metric]["kind"]}
                for metric in summaries
            }
            for name in ("assessment", "tuning")
        }
        # Mixed splits have no aggregate headline or gates.
        result["metrics"] = {}
    _write(output / "grader-validation.json", result)
    return result


def gate_failures(result, gates):
    summaries = result["metrics"] or result.get("by_split", {}).get("assessment", {})
    return [
        name
        for name, limit in gates.items()
        if limit is not None
        and any(summary[name]["rate"] is not None and summary[name]["rate"] > limit for summary in summaries.values())
    ]


@click.command("validate-grader")
@click.argument("suite_dir", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("--repeats", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--split", type=click.Choice(["assessment", "tuning", "all"]), default="assessment", show_default=True)
@click.option("--case", "case_ids", multiple=True)
@click.option(
    "-o", "--output", type=click.Path(path_type=Path, file_okay=False), default=Path("grader-validation-output")
)
@click.option("--json", "as_json", is_flag=True)
@click.option("--check-suite", is_flag=True)
@click.option("--concurrency", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0, min_open=True), default=300.0, show_default=True)
@click.option("--max-false-pass-rate", type=click.FloatRange(0, 1))
@click.option("--max-false-fail-rate", type=click.FloatRange(0, 1))
@click.option("--max-unjudgeable-rate", type=click.FloatRange(0, 1))
def validate_grader(
    suite_dir,
    repeats,
    split,
    case_ids,
    output,
    as_json,
    check_suite,
    concurrency,
    timeout,
    max_false_pass_rate,
    max_false_fail_rate,
    max_unjudgeable_rate,
):
    """Qualify the shipped Tier 3 verifier against independently labelled cases.

    SUITE_DIR defaults to the bundled reference suite when omitted.
    """
    suite_dir = REFERENCE_SUITE if suite_dir is None else suite_dir
    try:
        suite = load_suite(suite_dir)
        unknown = set(case_ids) - {case["id"] for case in suite["cases"]}
        if unknown:
            raise ValueError(f"unknown case IDs: {sorted(unknown)}")
        cases = [
            case
            for case in suite["cases"]
            if (split == "all" or case["split"] == split) and (not case_ids or case["id"] in case_ids)
        ]
        if not cases:
            raise ValueError("no cases selected")
        if check_suite:
            payload = {"status": "valid", "cases": len(cases), "split": split}
            click.echo(
                json.dumps(payload)
                if as_json
                else f"Suite valid: {len(cases)} {split} cases; executable labels verified (no LLM calls)."
            )
            return
        provider = resolve_llm_provider()
        output = output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory must be empty (prevents stale attempt evidence)")
        output.mkdir(parents=True, exist_ok=True)
        result = qualify(suite_dir.resolve(), suite, cases, output, repeats, concurrency, timeout, split, provider)
    except (ValueError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc
    gates = {
        "false_pass_rate": max_false_pass_rate,
        "false_fail_rate": max_false_fail_rate,
        "unjudgeable_rate": max_unjudgeable_rate,
    }
    failures = gate_failures(result, gates)
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        table = Table(title="Grader qualification")
        for header in ("Metric", "False passes", "False fails", "Unjudgeable", "Judgments / cases"):
            table.add_column(header)
        summaries = result["metrics"] or result["by_split"]["assessment"]
        for metric, summary in summaries.items():
            cells = [f"{summary[key]['numerator']}/{summary[key]['denominator']}" for key in gates]
            table.add_row(metric, *cells, f"{summary['judgments']} / {summary['cases']}")
        Console().print(table)
        click.echo(f"Split: {split}; report: {output / 'grader-validation.json'}")
        click.echo(LIMITATIONS)
    if failures:
        if not as_json:
            click.echo(f"Gates exceeded: {', '.join(failures)}", err=True)
        raise click.exceptions.Exit(1)
