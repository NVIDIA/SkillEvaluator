# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

from skillevaluator.tier3 import grader_policy as gp
from skillevaluator.tier3 import grader_validation as gv


def policy(**changes):
    return {
        "schema": "skillevaluator.grader-qualification-policy/v1",
        "policy_id": "experimental-test-v1",
        "required_metrics": ["accuracy"],
        "min_class_cases": {"pass": 1, "fail": 1},
        "min_category_cases": {"positive": 1, "negative": 1},
        "max_rates": {"false_pass_rate": 0.1, "false_fail_rate": 0.1, "unjudgeable_rate": 0.05},
    } | changes


def cases():
    return [
        {
            "id": "p",
            "split": "assessment",
            "category": "positive",
            "expected": {"overall": "pass", "metrics": {"accuracy": "pass"}},
        },
        {
            "id": "f",
            "split": "assessment",
            "category": "negative",
            "expected": {"overall": "fail", "metrics": {"accuracy": "fail"}},
        },
    ]


def report(selected, observed=None, repeats=1):
    attempts = []
    for case in selected:
        for repeat in range(repeats):
            rows = gv.observations(case, {}, gv.DEFAULT_THRESHOLDS)
            for row in rows:
                row["observed"] = observed or row["expected"]
            attempts.append({"case": case["id"], "repeat": repeat + 1, "observations": rows})
    metrics = {
        m: gv.summarize([r for a in attempts for r in a["observations"] if r["metric"] == m])
        for m in ("accuracy", "overall", "behavior_check", "behavior_steps")
    }
    return {"attempts": attempts, "metrics": metrics, "provenance": {"thresholds": gv.DEFAULT_THRESHOLDS}}


def test_policy_identity_and_label_only_readiness(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy(), indent=2))
    loaded = gp.load_qualification_policy(path)
    decision = gp.preflight_qualification(cases(), gv.DEFAULT_THRESHOLDS, loaded)
    assert decision["status"] == "ready"
    assert decision["phase"] == "preflight"
    assert decision["policy"] == policy()
    assert (
        decision["policy_sha256"]
        == hashlib.sha256(
            json.dumps(policy(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        ).hexdigest()
    )
    assert decision["metrics"]["accuracy"]["expected_class_counts"] == {"pass": 1, "fail": 1}
    assert decision["metrics"]["accuracy"]["scored_class_counts"] is None
    assert decision["metrics"]["accuracy"]["rate_checks"] is None
    assert decision["categories"]["positive"]["observed_cases"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(extra=True),
        lambda p: p.update(schema="v2"),
        lambda p: p.update(policy_id="../unsafe"),
        lambda p: p.update(policy_id="é"),
        lambda p: p.update(policy_id="a" * 129),
        lambda p: p.update(description=""),
        lambda p: p.update(description="x" * 1025),
        lambda p: p.update(required_metrics=[]),
        lambda p: p.update(required_metrics=["accuracy", "accuracy"]),
        lambda p: p.update(required_metrics=["unknown"]),
        lambda p: p.update(required_metrics=[{}]),
        lambda p: p.update(min_class_cases={"pass": True, "fail": 1}),
        lambda p: p.update(min_class_cases={"pass": 1}),
        lambda p: p["min_class_cases"].update(fail=0),
        lambda p: p.update(min_category_cases={"unknown": 1}),
        lambda p: p.update(min_category_cases={"positive": False}),
        lambda p: p["max_rates"].update(false_pass_rate=True),
        lambda p: p["max_rates"].update(false_pass_rate=1.1),
        lambda p: p["max_rates"].update(false_pass_rate=10**400),
        lambda p: p["max_rates"].update(false_pass_rate=float("nan")),
        lambda p: p["max_rates"].update(false_pass_rate=float("inf")),
        lambda p: p["max_rates"].pop("false_fail_rate"),
    ],
)
def test_reject_invalid_policy(tmp_path, mutation):
    value = policy()
    mutation(value)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        gp.load_qualification_policy(path)


@pytest.mark.parametrize("content", ["{", '"scalar"', "[" * 150 + "]" * 150, " " * (256 * 1024 + 1)])
def test_bounded_policy_read(tmp_path, content):
    path = tmp_path / "policy.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        gp.load_qualification_policy(path)


def test_nonregular_policy(tmp_path):
    with pytest.raises(ValueError):
        gp.load_qualification_policy(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no FIFO creation")
def test_fifo_policy_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        gp.load_qualification_policy(fifo)


def test_missing_required_metric_class_category_and_selected_cases():
    p = policy(required_metrics=["accuracy", "security"], min_category_cases={"baseline": 1})
    decision = gp.preflight_qualification(cases()[:1], gv.DEFAULT_THRESHOLDS, p)
    assert decision["status"] == "insufficient_evidence"
    assert {r["code"] for r in decision["reasons"]} >= {
        "missing_metric",
        "expected_class_coverage",
        "expected_category_coverage",
    }
    assert decision["scope"]["case_ids"] == ["p"]
    assert gp.preflight_qualification(cases(), gv.DEFAULT_THRESHOLDS, policy())["status"] == "ready"


def test_evaluated_counts_and_rate_equality():
    selected = cases()
    value = report(selected, repeats=50)
    decision = gp.evaluate_qualification(value, selected, policy(max_rates=dict.fromkeys(policy()["max_rates"], 0)))
    assert decision["status"] == "passed"
    assert decision["phase"] == "evaluated"
    assert decision["metrics"]["accuracy"]["scored_class_counts"] == {"pass": 1, "fail": 1}
    for check in decision["metrics"]["accuracy"]["rate_checks"].values():
        assert check["met"] is True
        assert check["measurement"]["wilson_95"] is not None
    stricter = policy(min_class_cases={"pass": 2, "fail": 2})
    assert gp.evaluate_qualification(value, selected, stricter)["status"] == "insufficient_evidence"


@pytest.mark.parametrize("name", ["false_pass_rate", "false_fail_rate", "unjudgeable_rate"])
def test_each_exceeded_rate_fails(name):
    value = report(cases())
    value["metrics"]["accuracy"][name] = gv.rate(1, 2)
    p = policy(max_rates=dict.fromkeys(policy()["max_rates"], 0.5))
    assert gp.evaluate_qualification(value, cases(), p)["status"] == "passed"
    p["max_rates"][name] = 0.49
    decision = gp.evaluate_qualification(value, cases(), p)
    assert decision["status"] == "failed"
    assert any(r["code"] == "rate_exceeded" and r["rate"] == name for r in decision["reasons"])


@pytest.mark.parametrize("observed", ["error", "insufficient_evidence", "not_applicable"])
def test_unscored_cannot_satisfy_class_or_category(observed):
    decision = gp.evaluate_qualification(report(cases(), observed), cases(), policy())
    assert decision["status"] == "insufficient_evidence"
    assert decision["metrics"]["accuracy"]["scored_class_counts"] == {"pass": 0, "fail": 0}
    assert decision["categories"]["positive"]["observed_cases"] == 0
    assert {r["code"] for r in decision["reasons"]} >= {
        "scored_class_coverage",
        "observed_category_coverage",
        "missing_rate",
    }


@pytest.mark.parametrize("missing", ["metric", "rate", "zero"])
def test_missing_metric_or_rate_never_passes(missing):
    value = report(cases())
    if missing == "metric":
        value["metrics"].pop("accuracy")
    elif missing == "rate":
        value["metrics"]["accuracy"].pop("false_pass_rate")
    else:
        value["metrics"]["accuracy"]["false_pass_rate"] = gv.rate(0, 0)
    assert gp.evaluate_qualification(value, cases(), policy())["status"] == "insufficient_evidence"


def test_many_behavior_steps_are_distinct_cases_per_class():
    selected = cases()[:1]
    selected[0]["expected"]["metrics"] = {"behavior_check": [True, False] * 50}
    value = report(selected, repeats=50)
    p = policy(required_metrics=["behavior_steps"], min_category_cases={})
    decision = gp.evaluate_qualification(value, selected, p)
    assert decision["status"] == "passed"
    assert decision["metrics"]["behavior_steps"]["expected_class_counts"] == {"pass": 1, "fail": 1}
    assert decision["metrics"]["behavior_steps"]["scored_class_counts"] == {"pass": 1, "fail": 1}
    p["min_class_cases"] = {"pass": 2, "fail": 2}
    assert gp.evaluate_qualification(value, selected, p)["status"] == "insufficient_evidence"


def test_tuning_observations_cannot_fill_assessment_coverage():
    selected = cases()
    tuning = copy.deepcopy(selected[1])
    tuning.update(id="tuning", split="tuning")
    value = report([selected[0], tuning])
    with pytest.raises(ValueError, match="scope"):
        gp.evaluate_qualification(value, selected, policy())
    with pytest.raises(ValueError, match="assessment"):
        gp.preflight_qualification([tuning], gv.DEFAULT_THRESHOLDS, policy())


def test_duplicate_json_fields_rejected(tmp_path):
    path = tmp_path / "policy.json"
    content = json.dumps(policy()).replace('"false_pass_rate": 0.1', '"false_pass_rate": 0.9, "false_pass_rate": 0.1')
    path.write_text(content)
    with pytest.raises(ValueError, match="duplicate"):
        gp.load_qualification_policy(path)


def test_behavior_aggregate_expected_class_uses_suite_threshold():
    selected = cases()
    for case in selected:
        case["expected"]["metrics"] = {"behavior_check": [True, False] if case["id"] == "p" else [False, False]}
    decision = gp.preflight_qualification(
        selected, gv.DEFAULT_THRESHOLDS | {"behavior_check": 0.5}, policy(required_metrics=["behavior_check"])
    )
    assert decision["status"] == "ready"
    assert decision["metrics"]["behavior_check"]["expected_class_counts"] == {"pass": 1, "fail": 1}


def test_experimental_example_intentionally_fails_bundled_label_coverage():
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "examples/grader-qualification-policy.json"
    value = gp.load_qualification_policy(example)
    assert set(value["required_metrics"]) == {*gv.METRICS, "behavior_steps", "overall"}
    assert set(value["min_category_cases"]) == gv.CATEGORIES
    selected = [case for case in gv.load_suite(gv.REFERENCE_SUITE)["cases"] if case["split"] == "assessment"]
    decision = gp.preflight_qualification(selected, gv.DEFAULT_THRESHOLDS, value)
    assert decision["status"] == "insufficient_evidence"
    assert {(r["metric"], r["class"]) for r in decision["reasons"]} == {
        ("security", "fail"),
        ("skill_efficiency", "fail"),
    }


def test_policy_read_without_nonblocking_flag(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()))
    monkeypatch.delattr(gp.os, "O_NONBLOCK", raising=False)
    assert gp.load_qualification_policy(path) == policy()


@pytest.mark.parametrize("label", ["not_applicable", "insufficient_evidence"])
@pytest.mark.parametrize("observed", ["matching", "error", "pass", "fail", "other_abstention"])
def test_category_observed_coverage_distinguishes_correct_expected_abstentions(label, observed):
    selected = cases()
    unavailable = {
        "id": "u",
        "split": "assessment",
        "category": "unavailable_evidence",
        "expected": {"overall": "insufficient_evidence", "metrics": {"accuracy": label}},
    }
    selected.append(unavailable)
    value = report(selected)
    verdict = (
        label
        if observed == "matching"
        else ("not_applicable" if label == "insufficient_evidence" else "insufficient_evidence")
        if observed == "other_abstention"
        else observed
    )
    for row in value["attempts"][-1]["observations"]:
        if row["metric"] == "accuracy":
            row["observed"] = verdict
    value["metrics"]["accuracy"] = gv.summarize(
        [row for attempt in value["attempts"] for row in attempt["observations"] if row["metric"] == "accuracy"]
    )
    criteria = policy(min_category_cases={"unavailable_evidence": 1}, max_rates=dict.fromkeys(gp.RATES, 0))
    decision = gp.evaluate_qualification(value, selected, criteria)
    assert decision["status"] == ("passed" if observed == "matching" else "insufficient_evidence")
    category = decision["categories"]["unavailable_evidence"]
    assert category["observed_cases"] == int(observed == "matching")
    assert category["observed_met"] is (observed == "matching")
    assert "scored_cases" not in category and "scored_met" not in category
    assert decision["metrics"]["accuracy"]["scored_class_counts"] == {"pass": 1, "fail": 1}
    for check in decision["metrics"]["accuracy"]["rate_checks"].values():
        assert check["measurement"]["rate"] == 0


def test_broader_report_cannot_dilute_selected_case_false_passes():
    selected = cases()
    extras = [dict(copy.deepcopy(selected[1]), id=f"extra-{i}") for i in range(20)]
    value = report(selected + extras)
    selected_fail = value["attempts"][1]
    for row in selected_fail["observations"]:
        row["observed"] = "pass"
    value["metrics"]["accuracy"] = gv.summarize(
        [row for attempt in value["attempts"] for row in attempt["observations"] if row["metric"] == "accuracy"]
    )
    assert value["metrics"]["accuracy"]["false_pass_rate"]["rate"] == pytest.approx(1 / 21)
    assert gp.evaluate_qualification(value, selected + extras, policy())["status"] == "passed"
    with pytest.raises(ValueError, match="scope"):
        gp.evaluate_qualification(value, selected, policy())
    exact = report(selected)
    exact["attempts"][1] = selected_fail
    exact["metrics"]["accuracy"] = gv.summarize(
        [row for attempt in exact["attempts"] for row in attempt["observations"] if row["metric"] == "accuracy"]
    )
    decision = gp.evaluate_qualification(exact, selected, policy())
    assert decision["status"] == "failed"
    assert decision["metrics"]["accuracy"]["rate_checks"]["false_pass_rate"]["measurement"]["rate"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["provenance"].update(case_ids=["p", "f", "extra"]),
        lambda r: r["provenance"].update(case_ids=["p"]),
        lambda r: r["provenance"].update(case_ids=["p", "f", "p"]),
        lambda r: r["provenance"].update(split="tuning"),
        lambda r: r["attempts"].pop(),
        lambda r: r["attempts"][0].update(case="extra"),
        lambda r: r["attempts"][0]["observations"][0].update(judgment="extra:overall"),
    ],
)
def test_report_provenance_and_attempt_scope_must_agree(mutation):
    value = report(cases())
    mutation(value)
    with pytest.raises(ValueError, match="scope"):
        gp.evaluate_qualification(value, cases(), policy())
