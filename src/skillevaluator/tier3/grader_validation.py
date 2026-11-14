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
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from skillevaluator import __version__
from skillevaluator.provider_config import resolve_llm_provider
from skillevaluator.tier3.eval_core.atif_helpers import (
    extract_tool_calls_as_dicts,
    get_final_response,
)
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import atif_content_text

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
        try:
            verify_checks(label, trajectory, case_id)
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"{case_id}: malformed trajectory evidence") from exc
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
            passed = any(
                check["value"] in atif_content_text(result.get("content"))
                for step in trajectory["steps"]
                if isinstance(step, dict)
                for observation in [step.get("observation")]
                if isinstance(observation, dict)
                for results in [observation.get("results")]
                if isinstance(results, list)
                for result in results
                if isinstance(result, dict)
            )
        else:
            _text(check.get("path"), "check.path")
            safe_steps = []
            for step in trajectory["steps"]:
                if not isinstance(step, dict):
                    continue
                calls = step.get("tool_calls")
                safe_calls = []
                if isinstance(calls, list):
                    for call in calls:
                        if (
                            isinstance(call, dict)
                            and isinstance(call.get("function_name"), str)
                            and isinstance(call.get("arguments") or {}, dict)
                        ):
                            safe_calls.append(call)
                safe_steps.append(
                    {
                        **step,
                        "tool_calls": safe_calls,
                    }
                )
            safe_trajectory = {**trajectory, "steps": safe_steps}
            passed = False
            for call in extract_tool_calls_as_dicts(safe_trajectory):
                if not isinstance(call, dict):
                    continue
                action = call.get("action")
                action_input = call.get("action_input")
                if (
                    isinstance(action, str)
                    and action.lower() in {"write", "write_file"}
                    and isinstance(action_input, dict)
                    and action_input.get("file_path") == check["path"]
                    and isinstance(action_input.get("content"), str)
                    and check["value"] in action_input["content"]
                ):
                    passed = True
                    break
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
            expected_score = round(sum(expected) / len(expected), 4) if expected else 1.0
            result.append(
                {
                    "metric": metric,
                    "judgment": f"{case['id']}:{metric}",
                    "expected": "pass" if expected_score >= thresholds[metric] else "fail",
                    "observed": observed,
                    "score": score,
                }
            )
            detail = reward.get("details", {}).get(metric, {})
            steps = detail.get("results", [])
            for index, passed in enumerate(expected):
                verdict = observed
                step_score = None
                if observed in {"pass", "fail"}:
                    step = next((s for s in steps if s.get("step") == index + 1), {})
                    step_score = step.get("passed")
                    verdict = ("pass" if step_score else "fail") if type(step_score) is bool else "error"
                    expectations = detail.get("expectations", [])
                    if (
                        "passed" in step
                        and step_score is None
                        and type(step.get("step")) is int
                        and isinstance(expectations, list)
                        and any(
                            isinstance(item, dict)
                            and type(item.get("step")) is int
                            and item["step"] == index + 1
                            and item.get("status") == "not_checked"
                            for item in expectations
                        )
                    ):
                        verdict = "insufficient_evidence"
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


def _attempt(root, case, repeat, output, thresholds, timeout, provider_env=None):
    directory = output / case["id"] / f"attempt-{repeat}"
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "entry.json", case["entry"])
    env = os.environ.copy()
    env.update(provider_env or {})
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


def _verifier_git_provenance(paths):
    """Disclose source identity only when every actual verifier file is tracked."""
    unknown = {"verifier_git_commit": None, "verifier_git_dirty": None}
    paths = [path.resolve() for path in paths]
    if not paths:
        return unknown

    def git(directory, *args):
        # Fixed commands and explicit paths only; never read remotes or credentials.
        result = subprocess.run(
            ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(directory), *args],
            env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            capture_output=True,
            timeout=2,
            check=False,
        )
        if result.returncode or len(result.stdout) > 256 * 1024:
            raise ValueError("Git source identity unavailable")
        return result.stdout.decode("utf-8").strip()

    try:
        repository = Path(git(paths[0].parent, "rev-parse", "--show-toplevel")).resolve()
        names = [path.relative_to(repository).as_posix() for path in paths]
        tracked = [name for name in git(repository, "ls-files", "-z", "--", *names).split("\0") if name]
        if set(tracked) != set(names):
            return unknown
        commit = git(repository, "rev-parse", "--verify", "HEAD")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            return unknown
        dirty = bool(git(repository, "status", "--porcelain", "--untracked-files=no", "--", *names))
        return {"verifier_git_commit": commit, "verifier_git_dirty": dirty}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return unknown


def qualify(
    root,
    suite,
    cases,
    output,
    repeats,
    concurrency,
    timeout,
    split,
    provider,
    provider_env,
    judge_config,
    qualification_policy=None,
):
    from skillevaluator.tier3.grader_policy import evaluate_qualification, preflight_qualification

    thresholds = DEFAULT_THRESHOLDS | suite.get("thresholds", {})
    if qualification_policy is not None and split != "assessment":
        raise ValueError("qualification policies require --split assessment")
    preflight = (
        preflight_qualification(cases, thresholds, qualification_policy) if qualification_policy is not None else None
    )
    not_run = preflight is not None and preflight["status"] != "ready"
    tasks = [] if not_run else [(case, repeat) for case in cases for repeat in range(1, repeats + 1)]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        attempts = list(
            executor.map(lambda task: _attempt(root, *task, output, thresholds, timeout, provider_env), tasks)
        )
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
    from skillevaluator.tier3.harbor.adapter import (
        _VERIFIER_BUDGET_ENV_VARS,
        _VERIFIER_FILES,
        _VERIFIER_JUDGE_CONTROL_ENV_VARS,
        _VERIFIER_RETRY_ENV_VARS,
    )

    safe_env = {
        key: value
        for key, value in provider_env.items()
        if key in _VERIFIER_BUDGET_ENV_VARS | _VERIFIER_RETRY_ENV_VARS | _VERIFIER_JUDGE_CONTROL_ENV_VARS
    }
    source_root = Path(__file__).resolve().parents[2]
    verifier_file_sha256 = {}
    verifier_digest = hashlib.sha256()
    verifier_paths = [path.resolve() for path in _VERIFIER_FILES]
    for path in sorted(verifier_paths, key=lambda item: item.relative_to(source_root).as_posix()):
        name = path.relative_to(source_root).as_posix()
        content = path.read_bytes()
        verifier_file_sha256[name] = hashlib.sha256(content).hexdigest()
        verifier_digest.update(name.encode() + b"\0" + len(content).to_bytes(8, "big") + content)
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
            "skillevaluator_version": __version__,
            "verifier_sha256": verifier_digest.hexdigest(),
            "verifier_file_sha256": verifier_file_sha256,
            **_verifier_git_provenance(verifier_paths),
            "contracts": {
                key: sorted({r[key] for r in rewards if key in r})
                for key in ("judge_contract", "metric_set", "score_policy")
            },
            "judge_models_used": sorted({d["judge_model_used"] for d in details if d.get("judge_model_used")}),
            "goal_methods": sorted(
                {r.get("details", {}).get("goal_accuracy", {}).get("method") for r in rewards} - {None}
            ),
            "provider": provider.provider if provider is not None else None,
            "model": provider.model if provider is not None else None,
            "judge_config": judge_config,
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
    if qualification_policy is not None:
        result["execution_status"] = "not_run" if not_run else "completed"
        result["qualification"] = preflight if not_run else evaluate_qualification(result, cases, qualification_policy)
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
@click.option(
    "--qualification-policy",
    type=click.Path(path_type=Path),
    help="Explicit assessment coverage and risk policy JSON file.",
)
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
    qualification_policy,
):
    """Qualify the shipped Tier 3 verifier against independently labelled cases.

    SUITE_DIR defaults to the bundled reference suite when omitted.
    """
    suite_dir = REFERENCE_SUITE if suite_dir is None else suite_dir
    try:
        from skillevaluator.tier3.grader_policy import load_qualification_policy, preflight_qualification

        for name, value in {
            "timeout": timeout,
            "max-false-pass-rate": max_false_pass_rate,
            "max-false-fail-rate": max_false_fail_rate,
            "max-unjudgeable-rate": max_unjudgeable_rate,
        }.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"--{name} must be finite")
        if qualification_policy is not None and any(
            value is not None for value in (max_false_pass_rate, max_false_fail_rate, max_unjudgeable_rate)
        ):
            raise ValueError("cannot combine --qualification-policy with --max-*-rate gates; put limits in the policy")
        policy = load_qualification_policy(qualification_policy) if qualification_policy is not None else None
        if policy is not None and split != "assessment":
            raise ValueError("qualification policies require --split assessment")
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
        preflight = (
            preflight_qualification(cases, DEFAULT_THRESHOLDS | suite.get("thresholds", {}), policy)
            if policy is not None
            else None
        )
        if check_suite:
            payload = {"status": "valid", "cases": len(cases), "split": split}
            if preflight is not None:
                payload["qualification"] = preflight
            click.echo(
                json.dumps(payload)
                if as_json
                else f"Suite valid: {len(cases)} {split} cases; executable labels verified (no LLM calls)."
            )
            if preflight is not None:
                if not as_json:
                    _print_qualification(preflight)
                if preflight["status"] != "ready":
                    raise click.exceptions.Exit(1)
            return
        output = output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory must be empty (prevents stale attempt evidence)")
        provider, provider_env, judge_config = None, {}, {}
        if preflight is None or preflight["status"] == "ready":
            provider = resolve_llm_provider()
            from skillevaluator.tier3.harbor.runner import _judge_model_config, _provider_environment

            provider_env = _provider_environment(provider)
            judge_config = _judge_model_config(provider, provider_env, "default")
        output.mkdir(parents=True, exist_ok=True)
        result = qualify(
            suite_dir.resolve(),
            suite,
            cases,
            output,
            repeats,
            concurrency,
            timeout,
            split,
            provider,
            provider_env,
            judge_config,
            qualification_policy=policy,
        )
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
        tables = (
            [(name.title(), summaries) for name, summaries in result["by_split"].items()]
            if split == "all"
            else [("Grader qualification", result["metrics"])]
        )
        for title, summaries in tables:
            table = Table(title=title if split != "all" else f"{title} grader qualification")
            for header in ("Metric", "False passes", "False fails", "Unjudgeable", "Judgments / cases"):
                table.add_column(header)
            for metric, summary in summaries.items():
                cells = [f"{summary[key]['numerator']}/{summary[key]['denominator']}" for key in gates]
                table.add_row(metric, *cells, f"{summary['judgments']} / {summary['cases']}")
            Console().print(table)
        click.echo(f"Split: {split}; report: {output / 'grader-validation.json'}")
        click.echo(LIMITATIONS)
        if "qualification" in result:
            _print_qualification(result["qualification"])
    policy_failed = "qualification" in result and result["qualification"]["status"] != "passed"
    if failures or policy_failed:
        if failures and not as_json:
            click.echo(f"Gates exceeded: {', '.join(failures)}", err=True)
        raise click.exceptions.Exit(1)


def _print_qualification(decision):
    click.echo(f"Policy {decision['policy_id']}: {decision['status']} ({decision['phase']})")
    for reason in decision["reasons"]:
        click.echo(f"  {reason['message']}")
