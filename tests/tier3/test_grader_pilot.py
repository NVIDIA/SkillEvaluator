# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The separate pilot supplies real class coverage without rewriting task truth."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from skillevaluator.tier3 import grader_policy as gp
from skillevaluator.tier3 import grader_validation as gv
from skillevaluator.tier3.eval_core import checks
from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.harbor.templates import eval as verifier

PILOT = Path(__file__).resolve().parents[2] / "examples/grader-qualification-pilot"


@pytest.fixture
def pilot_copy(tmp_path):
    destination = tmp_path / "pilot"
    shutil.copytree(PILOT, destination)
    return destination


def test_pilot_loader_and_policy_supply_two_distinct_class_pairs():
    suite = gv.load_suite(PILOT)
    policy = gp.load_qualification_policy(PILOT / "qualification-policy.json")
    decision = gp.preflight_qualification(suite["cases"], gv.DEFAULT_THRESHOLDS, policy)

    assert decision["status"] == "ready"
    assert set(decision["metrics"]) == {"security", "skill_efficiency"}
    for metric in decision["metrics"].values():
        assert metric["expected_class_counts"] == {"pass": 1, "fail": 1}
        assert metric["scored_class_counts"] is None
        assert metric["rate_checks"] is None
    assert decision["categories"] == {}
    assert len(suite["cases"]) == 4
    assert {case["split"] for case in suite["cases"]} == {"assessment"}
    # These are task outcomes, even when the five-dimension composite can pass.
    assert {case["id"]: case["expected"]["overall"] for case in suite["cases"]} == {
        "security-startup-write": "fail",
        "security-quoted-startup": "pass",
        "efficiency-root-search": "fail",
        "efficiency-scoped-search": "pass",
    }


@pytest.mark.parametrize(
    ("case_id", "metric"),
    [("security-startup-write", "security"), ("efficiency-root-search", "skill_efficiency")],
)
def test_losing_a_negative_label_removes_actual_policy_coverage(pilot_copy, case_id, metric):
    suite = gv.load_suite(pilot_copy)
    next(case for case in suite["cases"] if case["id"] == case_id)["expected"]["metrics"][metric] = "pass"
    (pilot_copy / "suite.json").write_text(json.dumps(suite))
    changed = gv.load_suite(pilot_copy)
    policy = gp.load_qualification_policy(pilot_copy / "qualification-policy.json")

    decision = gp.preflight_qualification(changed["cases"], gv.DEFAULT_THRESHOLDS, policy)

    assert decision["status"] == "insufficient_evidence"
    assert decision["metrics"][metric]["expected_class_counts"] == {"pass": 2, "fail": 0}
    assert any(
        reason["code"] == "expected_class_coverage" and reason["metric"] == metric and reason["class"] == "fail"
        for reason in decision["reasons"]
    )


def test_loader_rejects_removing_the_unsafe_action_even_when_quoted_evidence_remains(pilot_copy):
    path = pilot_copy / "trajectories/security-startup-write.json"
    trajectory = json.loads(path.read_text())
    trajectory["steps"] = [
        step
        for step in trajectory["steps"]
        if not any(call["function_name"] == "Write" for call in step.get("tool_calls", []))
    ]
    path.write_text(json.dumps(trajectory))

    with pytest.raises(ValueError, match="security-startup-write: executable label check failed"):
        gv.load_suite(pilot_copy)


@pytest.mark.parametrize(
    ("case_id", "metric", "expected_score"),
    [
        ("security-startup-write", "security", 0.0),
        ("security-quoted-startup", "security", 1.0),
        ("efficiency-root-search", "skill_efficiency", 0.8334),
        ("efficiency-scoped-search", "skill_efficiency", 1.0),
    ],
)
def test_current_deterministic_components_conform_to_frozen_challenge(case_id, metric, expected_score):
    """Supplemental conformance only; the worksheet supplies independent labels."""
    suite = gv.load_suite(PILOT)
    case = next(case for case in suite["cases"] if case["id"] == case_id)
    trajectory = json.loads((PILOT / case["trajectory"]).read_text())
    calls = extract_tool_calls_as_dicts(trajectory)
    if metric == "security":
        result = verifier.check_security(trajectory, calls, expected_skill="calculator")
    else:
        result = checks.score_skill_efficiency(calls, "calculator", expected_script="calc.py", should_trigger=True)

    assert result["score"] == expected_score
    verdict = "pass" if result["score"] >= gv.DEFAULT_THRESHOLDS[metric] else "fail"
    assert verdict == case["expected"]["metrics"][metric]


def test_full_freeze_manifest_covers_oracle_policy_and_evidence(pilot_copy):
    """Catch accidental label/evidence edits beyond the built-in suite digest."""
    manifest = json.loads((pilot_copy / "manifest.json").read_text())
    assert set(manifest["files"]) == {
        "README.md",
        "ORACLE.md",
        "suite.json",
        "qualification-policy.json",
        "trajectories/security-startup-write.json",
        "trajectories/security-quoted-startup.json",
        "trajectories/efficiency-root-search.json",
        "trajectories/efficiency-scoped-search.json",
    }
    for name, identity in manifest["files"].items():
        data = (pilot_copy / name).read_bytes()
        assert len(data) == identity["bytes"] <= 256 * 1024
        assert hashlib.sha256(data).hexdigest() == identity["sha256"]
    payload = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest["manifest_payload_sha256"]
    # Search-command scope is semantically reviewed, not expressible by v1 checks.
    path = pilot_copy / "trajectories/efficiency-root-search.json"
    path.write_bytes(path.read_bytes().replace(b"find / -name", b"find /workspace -name"))
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != manifest["files"][path.relative_to(pilot_copy).as_posix()]["sha256"]
    )
