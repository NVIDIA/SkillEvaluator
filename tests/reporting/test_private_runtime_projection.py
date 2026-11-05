# SPDX-License-Identifier: Apache-2.0
"""Private views preserve evidence strength and the existing numerical basis."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy

import pytest


def api():
    return importlib.import_module("skillevaluator.reporting.plugin_runtime")


def sample():
    return {
        "run_dir": "/retained/run-42",
        "execution_status": "succeeded",
        "runtime_evidence": {
            "evidence_complete": True,
            "source": {"manifest_sha256": "a" * 64, "source_tree_sha256": "b" * 64},
            "mcp_proof": {"lookup": {"status": "declared", "tools": [], "detail": "probe not requested"}},
            "errors": [],
        },
        "native_evidence": {
            "claude-code": {
                "mode": "native",
                "trials": 5,
                "loaded": [{"type": "skill", "name": "member", "evidence": "startup event"}],
                "listed": [{"type": "rule", "name": "policy", "evidence": "setup listing"}],
                "staged": [{"type": "mcp", "name": "lookup", "evidence": "staged only"}],
                "not_loaded": [{"type": "hook", "name": "startup", "reason": "permission refused"}],
            }
        },
        "agents": {
            "claude-code": {
                "plugin_signals_summary": {
                    "with_skill": {
                        "n_trials": 5,
                        "n_missing_trials": 0,
                        "n_missing_trajectory": 0,
                        "evidence_complete": True,
                        "routing": {"status": "scored", "n_scored": 5, "f1": 0.8},
                        "arguments": {"status": "not_applicable", "n_scored": 0, "pass_rate": None},
                        "activation_coverage": {
                            "declared": ["Skill:member", "MCP:lookup"],
                            "exercised": ["Skill:member"],
                            "unverified": ["MCP:lookup"],
                            "exercise_rate": {"Skill:member": 1.0, "MCP:lookup": 0.0},
                        },
                        "mcp_calls": {"by_server": {"lookup": {"total": 1, "succeeded": 0, "failed": 1, "unknown": 0}}},
                    }
                }
            }
        },
        "private_comparisons": {
            "status": "recorded",
            "errors": [],
            "agents": {
                "claude-code": {
                    "lift_ladder": {
                        "partial": False,
                        "n_cases": 5,
                        "steps": {
                            "skill": {
                                "estimate": 0.2,
                                "ci_low": 0.1,
                                "ci_high": 0.3,
                                "n_cases": 5,
                                "precision": "adequate",
                            }
                        },
                        "excluded_cases": [{"case_id": "excluded", "reason": "mixed judges"}],
                    },
                    "integration_claim": {"eligible": False, "verdict": "inconclusive", "reason": "load unverified"},
                    "loose_parts": {"measured": False, "not_measured_reason": "unsupported adapter"},
                    "cost": {
                        "with_skill": {
                            "total_tokens": 0,
                            "total_usd": None,
                            "successes": 0,
                            "tokens_per_success": None,
                            "usd_per_success": None,
                        }
                    },
                    "context_cost_measured": {
                        "status": "unavailable",
                        "delta_tokens_mean": None,
                        "n_pairs": 0,
                        "partial": False,
                        "excluded": {"hosted_search": 0, "missing_tokens": 5},
                        "reason": "missing first-turn tokens",
                    },
                }
            },
        },
    }


def sections(payload):
    return {section["key"]: section for section in payload["sections"]}


def test_projection_keeps_staging_loading_unknowns_and_recorded_statistics_separate():
    result = sample()
    before = deepcopy(result)
    report = api().project_plugin_runtime(result, provenance={"plugin": "fixture", "complete": True})
    payload = report.payload()
    assert payload["run_id"] == "run-42" and payload["execution_status"] == "succeeded"
    assert payload["name"] == "fixture" and payload["schema_version"] == 1
    assert payload["source"]["manifest_sha256"] == "a" * 64
    rows = sections(payload)
    assert {row[4] for row in rows["loading"]["rows"]} == {"loaded", "listed", "staged", "not_loaded"}
    assert any("permission refused" in row for row in rows["loading"]["rows"])
    assert ["claude-code", "with_skill", "MCP:lookup", "unverified", 0.0] in rows["coverage"]["rows"]
    assert ["claude-code", "skill", 0.2, 0.1, 0.3, 5, "adequate", "no"] in rows["comparisons"]["rows"]
    assert ["claude-code", "with_skill", 0, None, 0, None, None] in rows["cost"]["rows"]
    assert any("load unverified" in row for row in rows["eligibility"]["rows"])
    assert any("unsupported adapter" in row for row in rows["eligibility"]["rows"])
    assert any("mixed judges" in row for row in rows["exclusions"]["rows"])
    assert result == before
    payload["source"].clear()
    assert report.payload()["source"]


def complete_sample():
    """A fixture with no limitation of its own, so each check is observable."""
    result = sample()
    native = result["native_evidence"]["claude-code"]
    native.pop("listed")
    native.pop("staged")
    return result


def diagnostics(payload):
    return sections(payload)["diagnostics"]["rows"]


def test_complete_baseline_has_no_evidence_limitations():
    payload = api().project_plugin_runtime(complete_sample()).payload()
    assert payload["status"] == "complete" and payload["evidence_complete"] is True
    assert diagnostics(payload) == []


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("runtime", ["runtime", "Runtime evidence is unavailable or incomplete"]),
        ("arm", ["claude-code/with_skill", "Runtime behavior evidence is incomplete"]),
    ],
)
def test_successful_execution_with_missing_evidence_is_not_a_complete_report(scope, expected):
    result = complete_sample()
    summary = result["agents"]["claude-code"]["plugin_signals_summary"]["with_skill"]
    summary.update(n_missing_trials=2, n_missing_trajectory=3)
    if scope == "runtime":
        result["runtime_evidence"]["evidence_complete"] = False
    else:
        summary["evidence_complete"] = False
    payload = api().project_plugin_runtime(result).payload()
    assert payload["execution_status"] == "succeeded"
    assert payload["status"] == "incomplete" and payload["evidence_complete"] is False
    assert diagnostics(payload) == [expected]
    signals = sections(payload)["signals"]["rows"]
    assert ["claude-code", "with_skill", "n_missing_trials", "recorded", 2, None, None] in signals
    assert ["claude-code", "with_skill", "n_missing_trajectory", "recorded", 3, None, None] in signals


def test_execution_error_alias_is_not_listed_twice():
    result = complete_sample()
    result["execution_errors"] = ["claude-code: launch failed", "codex: launch failed"]
    # The engine keeps ``error`` as a compatibility copy of execution_errors.
    result["error"] = list(result["execution_errors"])
    payload = api().project_plugin_runtime(result).payload()
    assert diagnostics(payload) == [
        ["execution", "claude-code: launch failed"],
        ["execution", "codex: launch failed"],
    ]
    legacy = complete_sample()
    legacy["error"] = ["legacy launch failure"]
    assert diagnostics(api().project_plugin_runtime(legacy).payload()) == [["execution", "legacy launch failure"]]


def test_ordinary_result_has_no_private_projection():
    assert api().project_plugin_runtime({"execution_status": "succeeded", "agents": {}}) is None


def test_projection_sanitizes_all_identifiers_and_does_not_accept_invalid_numbers():
    result = sample()
    bad = "\x1b[31m<script>secret</script> token=fixture-not-a-real-secret\x00"
    result["runtime_evidence"]["errors"] = [bad]
    result["private_comparisons"]["agents"]["claude-code"]["cost"]["with_skill"]["total_tokens"] = float("nan")
    payload = api().project_plugin_runtime(result, provenance={"plugin": bad}).payload()
    text = json.dumps(payload, allow_nan=False)
    assert "fixture-not-a-real-secret" not in text
    assert "\\u001b" not in text and "\\u0000" not in text
    assert sections(payload)["cost"]["rows"][0][2] is None


def test_large_coverage_has_explicit_omissions_with_bounded_output():
    result = sample()
    coverage = result["agents"]["claude-code"]["plugin_signals_summary"]["with_skill"]["activation_coverage"]
    coverage["declared"] = [f"Skill:member-{i}" for i in range(10000)]
    payload = api().project_plugin_runtime(result).payload()
    section = sections(payload)["coverage"]
    assert payload["truncated"] is True
    assert section["total"] == 10000 and section["omitted"] == 10000 - len(section["rows"])
    assert len(json.dumps(payload).encode()) <= api().MAX_OUTPUT_BYTES
    assert sum(len(section["rows"]) for section in payload["sections"]) <= 256


def test_recorded_comparison_basis_and_attempt_coverage_survive_the_view():
    result = sample()
    result["agents"]["claude-code"]["conditions"] = {
        "with_skill": {
            "execution_status": "succeeded",
            "expected_cases": 5,
            "scored_cases": 5,
            "expected_attempts": 10,
            "scored_attempts": 10,
        }
    }
    stats = result["private_comparisons"]["agents"]["claude-code"]
    stats["lift_ladder"].update(basis="common scored dimensions", bootstrap={"confidence": 0.95, "resamples": 2000})
    stats["lift_uncertainty"] = {"integration": {"estimate": 0.4, "n_cases": 5}}
    payload = api().project_plugin_runtime(result).payload()
    views = sections(payload)
    assert ["claude-code", "with_skill", "succeeded", 5, 5, 10, 10] in views["conditions"]["rows"]
    assert any("common scored dimensions" in row for row in views["run"]["rows"])
    assert any(0.95 in row for row in views["run"]["rows"])
    assert any(row[1] == "Bundle lift (full minus member skills)" for row in views["comparisons"]["rows"])


def test_staging_only_native_observations_stay_incomplete_despite_readable_trajectories():
    result = sample()
    payload = api().project_plugin_runtime(result).payload()
    assert payload["evidence_complete"] is False


@pytest.mark.parametrize("expected,skipped", [({}, 0), ({"expected_order": [["command:review", "skill:member"]]}, 1)])
def test_unsupported_signal_expectations_retain_their_skipped_count(expected, skipped):
    from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals, summarize_plugin_signals

    signal = compute_plugin_signals(
        {"steps": [], "agent": {"name": "codex"}},
        expected,
        declared={"command": ["review"], "skill": ["member"]},
    )
    summary = summarize_plugin_signals([signal])
    summary["evidence_complete"] = True
    result = {
        "execution_status": "succeeded",
        "runtime_evidence": {"evidence_complete": True},
        "agents": {"codex": {"plugin_signals_summary": {"with_skill": summary}}},
    }
    payload = api().project_plugin_runtime(result).payload()
    section = sections(payload)["signals"]
    assert "Skipped" in section["columns"]
    row = next(row for row in section["rows"] if row[2] == "order")
    assert row[section["columns"].index("Skipped")] == skipped
    assert any("unsupported" in note for note in section["notes"])


def test_load_inventory_bounds_report_all_withheld_rows():
    result = sample()
    native = result["native_evidence"]["claude-code"]
    native.clear()
    native["loaded"] = [{"type": "skill", "name": f"member-{i}"} for i in range(10000)]
    payload = api().project_plugin_runtime(result).payload()
    section = sections(payload)["loading"]
    assert section["total"] == 10000
    assert section["omitted"] == 10000 - len(section["rows"])


@pytest.mark.parametrize("value", [None, [], True, "unavailable"])
def test_malformed_optional_evidence_is_explicit_and_does_not_crash(value):
    result = sample()
    result.update(native_evidence=value, private_comparisons=value)
    result["agents"]["claude-code"]["plugin_signals_summary"] = value
    payload = api().project_plugin_runtime(result).payload()
    assert payload["status"] == "incomplete"


@pytest.mark.parametrize("value", [42, {}, True, "unexpected"])
def test_malformed_coverage_lists_cannot_invent_activation(value):
    result = complete_sample()
    coverage = result["agents"]["claude-code"]["plugin_signals_summary"]["with_skill"]["activation_coverage"]
    coverage.update(exercised=value, unavailable=value)
    payload = api().project_plugin_runtime(result).payload()
    assert all(row[3] == "unverified" for row in sections(payload)["coverage"]["rows"])
    assert payload["status"] == "incomplete"
    assert diagnostics(payload) == [
        ["claude-code/with_skill", "Malformed exercised coverage evidence"],
        ["claude-code/with_skill", "Malformed unavailable coverage evidence"],
    ]


def test_case_coverage_uses_real_condition_summaries_for_every_arm():
    from skillevaluator.tier3.harbor.collector import _condition_execution_summary
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS

    rewards = [
        {
            "entry_id": "recorded",
            "_trial_root_name": f"recorded__attempt{attempt}",
            "metric_set": DEFAULT_METRIC_SET,
            **dict.fromkeys(DEFAULT_METRICS, 0.8),
        }
        for attempt in (1, 2)
    ]
    condition = _condition_execution_summary(
        rewards,
        expected_case_ids=["recorded", "missing"],
        expected_cases=2,
        n_attempts=2,
        job_failure="",
    )
    result = sample()
    result["agents"]["claude-code"]["conditions"] = dict.fromkeys(
        ("with_skill", "without_skill", "sum_of_parts", "loose_parts"), condition
    )
    rows = sections(api().project_plugin_runtime(result).payload())["conditions"]["rows"]
    assert rows == [
        ["claude-code", "with_skill", "failed", 2, 1, 4, 2],
        ["claude-code", "without_skill", "failed", 2, 1, 4, 2],
        ["claude-code", "sum_of_parts", "failed", 2, 1, 4, 2],
        ["claude-code", "loose_parts", "failed", 2, 1, 4, 2],
    ]


@pytest.mark.parametrize("member_skills", [False, True])
def test_pairwise_policy_exclusions_survive_without_a_loose_parts_ladder(member_skills):
    from skillevaluator.tier3.harbor.metrics import (
        DEFAULT_METRIC_SET,
        DEFAULT_METRICS,
        LEGACY_METRIC_SET,
        LEGACY_METRICS,
    )
    from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation, build_agent_statistics

    def trial(case, value, *, legacy=False):
        return TrialObservation(
            case,
            {
                "entry_id": case,
                "metric_set": LEGACY_METRIC_SET if legacy else DEFAULT_METRIC_SET,
                **dict.fromkeys(LEGACY_METRICS if legacy else DEFAULT_METRICS, value),
            },
        )

    arms = {
        "with_skill": ArmObservations("succeeded", (trial("kept", 0.8), trial("excluded-policy", 0.8))),
        "without_skill": ArmObservations("succeeded", (trial("kept", 0.4), trial("excluded-policy", 0.4, legacy=True))),
    }
    if member_skills:
        arms["sum_of_parts"] = ArmObservations(
            "succeeded", (trial("kept", 0.6), trial("excluded-policy", 0.6, legacy=True))
        )
    stats = build_agent_statistics(
        arms,
        expected_case_ids=["kept", "excluded-policy"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=member_skills,
    )
    assert stats["lift_ladder"] is None
    # Repeated retained entries must not create duplicate displayed exclusions.
    stats["comparison_eligibility"]["effectiveness"]["score_policy_mismatches"] *= 2
    result = sample()
    result["private_comparisons"]["agents"]["claude-code"] = stats
    excluded = sections(api().project_plugin_runtime(result).payload())["exclusions"]
    expected = [["claude-code", "excluded-policy", "Effectiveness lift (full minus baseline): score_policy_mismatch"]]
    if member_skills:
        expected.append(
            ["claude-code", "excluded-policy", "Bundle lift (full minus member skills): score_policy_mismatch"]
        )
    assert excluded["rows"] == expected
    assert excluded["total"] == len(expected) and excluded["omitted"] == 0


def test_pairwise_policy_exclusion_bounds_count_every_withheld_entry():
    result = sample()
    stats = result["private_comparisons"]["agents"]["claude-code"]
    stats["lift_ladder"] = None
    stats["comparison_eligibility"] = {
        "effectiveness": {"score_policy_mismatches": [f"case-{i}" for i in range(10000)]}
    }
    payload = api().project_plugin_runtime(result).payload()
    excluded = sections(payload)["exclusions"]
    assert excluded["total"] == 10000
    assert excluded["rows"] and excluded["omitted"] == 10000 - len(excluded["rows"])
    assert payload["truncated"] is True
