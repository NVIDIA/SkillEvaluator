# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting.dashboard_data import (
    comparison_rows,
    dedupe_rows,
    load_dashboard_path,
    parse_dashboard_report,
)


def _report() -> dict:
    def trial(prompt: int, completion: int, seconds: float, cost: float) -> dict:
        return {
            "entry_id": "case-a",
            "trial_id": "case-a__random",
            "overall": 0.8,
            "tokens": {"prompt": prompt, "completion": completion, "cached": 0},
            "agent_seconds": seconds,
            # Tier 3 canonical JSON does not produce cost_usd; it is kept to
            # exercise the reader for hand-imported payloads.
            "cost_usd": cost,
        }

    return {
        "schema_version": "2.0",
        "skill_name": "calculator",
        "run_id": "run-1",
        "dataset_digest": "sha256:dataset",
        "attempt_policy": {"max_attempts": 1},
        "agents": {
            "claude-code": {
                "name": "claude-code",
                "model": "model-a",
                "execution_status": "succeeded",
                "with_skill": 0.8,
                "baseline": 0.6,
                "num_trials": 1,
                "num_trials_baseline": 1,
                "conditions": {
                    arm: {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
                    for arm in ("with_skill", "without_skill")
                },
                "trials": [trial(120, 30, 8, 0.02)],
                "trials_baseline": [trial(100, 20, 10, 0.01)],
                "pass_at_k": {"with_skill": {"rate": 1.0}, "without_skill": {"rate": 0.0}},
            }
        },
    }


def test_with_without_differences_and_explicit_features() -> None:
    report = _report()
    report["dashboard_metadata"] = {"harness_features": ["rules", "memory"]}
    data = parse_dashboard_report(report, "report.json")
    assert len(data.rows) == 2
    assert data.rows[0]["features"] == "memory, rules"
    comparison = comparison_rows(data.rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["comparison_reason"] == ""
    assert comparison["score_delta"] == pytest.approx(0.2)
    assert comparison["total_tokens_delta"] == 30
    assert comparison["total_tokens_percent"] == 25
    assert comparison["duration_seconds_delta"] == -2
    assert comparison["cost_usd_percent"] == 100


@pytest.mark.parametrize("lift", [-0.1, 0.0, None])
def test_score_delta_preserves_canonical_judge_aware_lift(lift: float | None) -> None:
    report = _report()
    report["agents"]["claude-code"]["lift"] = lift
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["with_score"] == 0.8
    assert comparison["without_score"] == 0.6
    assert comparison["score_delta"] == lift
    assert comparison["pass_rate_delta"] == 1.0
    assert comparison["total_tokens_delta"] == 30


def test_single_trial_coverage_label_is_singular() -> None:
    assert parse_dashboard_report(_report(), "r").rows[0]["coverage"].startswith("1 trial (complete)")


def test_authoritative_validator_results_exclude_top_level_mirror() -> None:
    first, second = _report(), _report()
    second["skill_name"] = "second-skill"
    wrapper = {"tier3": first, "results": [{"tier3": first}, {"metadata": {"agent_eval": second}}]}
    data = parse_dashboard_report(wrapper, "combined.json")
    assert len(data.rows) == 4
    assert {row["skill"] for row in data.rows} == {"calculator", "second-skill"}
    assert len(comparison_rows(data.rows)) == 2
    assert parse_dashboard_report({"skills": [wrapper]}, "combined.json").rows == data.rows


def test_partial_missing_metrics_never_become_zero_or_partial_total() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field in ("trials", "trials_baseline"):
        agent[field][0]["tokens"].pop("completion")
        agent[field][0].pop("cost_usd")
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["total_tokens"] is None
    assert data.rows[0]["cost_usd"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["prompt_tokens_delta"] == 20
    assert comparison["total_tokens_delta"] is None
    assert comparison["cost_usd_delta"] is None
    assert "cost 0/1" in data.rows[0]["coverage"]


def test_pre_v050_zero_token_counters_are_unavailable() -> None:
    report = _report()
    report["evaluator_version"] = "0.4.0"
    report["agents"]["claude-code"]["trials_baseline"][0]["tokens"] = {"prompt": 0, "completion": 0, "cached": 0}
    data = parse_dashboard_report(report, "legacy.json")
    with_row, without_row = data.rows
    assert without_row["prompt_tokens"] is None
    assert without_row["total_tokens"] is None
    assert with_row["total_tokens"] == 150
    assert with_row["cached_tokens"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["score_delta"] == pytest.approx(0.2)
    assert comparison["total_tokens_delta"] is None
    assert any("before v0.5.0" in warning for warning in data.warnings)
    report.pop("evaluator_version")
    report["summary"] = {"evaluator_version": "0.4.1"}
    assert parse_dashboard_report(report, "legacy.json").rows[1]["total_tokens"] is None
    # Current, missing, or unparsable versions keep recorded zeros.
    for version in ("0.5.0", None, "not-a-version", "9" * 5000):
        report["summary"]["evaluator_version"] = version
        data = parse_dashboard_report(report, "current.json")
        assert data.rows[1]["total_tokens"] == 0
        assert not data.warnings


def test_canonical_missing_attempt_keeps_usage_totals_unknown() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["conditions"]["with_skill"].update(expected_attempts=2, scored_attempts=1, execution_status="failed")
    row = parse_dashboard_report(report, "report.json").rows[0]
    assert row["total_tokens"] is None
    assert row["duration_seconds"] is None
    assert "tokens 1/2" in row["coverage"]


def test_canonical_missing_trial_counts_each_declared_trial_once() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["num_trials"] = 10
    agent["conditions"]["with_skill"].update(expected_attempts=10, scored_attempts=10)
    agent["trials"] = [
        {**copy.deepcopy(agent["trials"][0]), "entry_id": f"case-{index}", "trial_id": f"case-{index}__trial0"}
        for index in range(9)
    ]
    row = parse_dashboard_report(report, "report.json").rows[0]
    assert row["total_tokens"] is None
    assert "tokens 9/10" in row["coverage"]
    # Without case identities the shortfall still keeps totals unknown.
    for trial in agent["trials"]:
        trial.pop("entry_id")
    row = parse_dashboard_report(report, "report.json").rows[0]
    assert row["total_tokens"] is None
    assert "tokens 9/10" in row["coverage"]


def test_per_trial_usage_normalizes_trial_counts() -> None:
    per_trial = []
    for count in (2, 4):
        report = _report()
        agent = report["agents"]["claude-code"]
        for field, count_field, arm in (
            ("trials", "num_trials", "with_skill"),
            ("trials_baseline", "num_trials_baseline", "without_skill"),
        ):
            agent[field] = [
                {**copy.deepcopy(agent[field][0]), "entry_id": f"case-{index}", "trial_id": f"case-{index}__trial0"}
                for index in range(count)
            ]
            agent[count_field] = count
            agent["conditions"][arm].update(expected_attempts=count, scored_attempts=count)
        rows = parse_dashboard_report(report, f"trials-{count}.json").rows
        # Totals grow with the trial count; per-trial means do not.
        assert [row["total_tokens"] for row in rows] == [150 * count, 120 * count]
        per_trial.append([(row["total_tokens_per_trial"], row["duration_seconds_per_trial"]) for row in rows])
    assert per_trial == [[(150.0, 8.0), (120.0, 10.0)]] * 2


def test_per_trial_usage_is_blank_when_any_trial_is_unmeasured() -> None:
    report = _repeated_attempts_report(("case-a__trial0", "case-a__trial1"))
    report["agents"]["claude-code"]["trials"][1]["tokens"].pop("completion")
    with_row, without_row = parse_dashboard_report(report, "report.json").rows
    assert with_row["total_tokens"] is None
    assert with_row["total_tokens_per_trial"] is None
    assert with_row["duration_seconds_per_trial"] == 8.0
    assert without_row["total_tokens_per_trial"] == 120.0


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", 10**30])
@pytest.mark.parametrize("field", ["declared", "expected", "both_attempt_counts"])
def test_present_invalid_coverage_counts_never_become_inferred_complete_counts(value: object, field: str) -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    if field == "declared":
        agent["num_trials"] = value
    else:
        agent["conditions"]["with_skill"]["expected_attempts"] = value
        if field == "both_attempt_counts":
            agent["conditions"]["with_skill"]["scored_attempts"] = value
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["score"] is None
    assert data.rows[0]["total_tokens"] is None
    assert data.rows[0]["duration_seconds"] is None
    assert any("invalid trial coverage counts" in warning for warning in data.warnings)
    comparison = comparison_rows(data.rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["score_delta"] is None


def test_unrecorded_legacy_coverage_counts_preserve_observed_trial_pairing() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field in ("num_trials", "num_trials_baseline"):
        agent.pop(field)
    for condition in agent["conditions"].values():
        condition.pop("expected_attempts")
        condition["scored_attempts"] = None
    assert comparison_rows(parse_dashboard_report(report, "legacy.json").rows)[0]["comparison_status"] == "comparable"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("failed", "without-skill arm status failed"),
        ("missing_trial", "without-skill trial coverage incomplete"),
        ("truncated", "trial data truncated"),
        ("different_case", "different cases or attempts"),
        ("missing_attempt", "without-skill trial coverage incomplete"),
        ("invalid_count", "invalid trial counts"),
    ],
)
def test_incomplete_or_unmatched_arms_block_all_deltas(mutation: str, reason: str) -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    if mutation == "failed":
        agent["conditions"]["without_skill"]["execution_status"] = "failed"
    elif mutation == "missing_trial":
        agent["trials_baseline"] = []
    elif mutation == "truncated":
        report["report_truncation"] = {"truncated": True}
    elif mutation == "different_case":
        agent["trials_baseline"][0]["entry_id"] = "case-b"
    elif mutation == "missing_attempt":
        agent["conditions"]["without_skill"]["expected_attempts"] = 2
    elif mutation == "invalid_count":
        agent["num_trials_baseline"] = -1
    comparison = comparison_rows(parse_dashboard_report(report, "same.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["comparison_reason"] == reason
    assert comparison["score_delta"] is None
    assert comparison["total_tokens_delta"] is None


@pytest.mark.parametrize(
    ("field", "reason"),
    [("_dataset_digest", "different dataset"), ("_attempt_policy", "different attempt policy")],
)
def test_tautological_pair_checks_have_reasons(field: str, reason: str) -> None:
    rows = parse_dashboard_report(_report(), "same.json").rows
    rows[1][field] = "other"
    comparison = comparison_rows(rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["comparison_reason"] == reason
    assert comparison["score_delta"] is None


def test_disjoint_runs_features_and_duplicate_conditions_cannot_pair() -> None:
    data = parse_dashboard_report(_report(), "first.json")
    rows = copy.deepcopy(data.rows)
    rows[1]["source"] = "second.json"
    comparisons = comparison_rows(rows)
    assert all(row["comparison_status"] == "missing condition" for row in comparisons)
    assert [row["comparison_reason"] for row in comparisons] == [
        "no without-skill arm for this run and configuration",
        "no with-skill arm for this run and configuration",
    ]
    rows = copy.deepcopy(data.rows)
    rows[1]["features"] = "rules"
    comparisons = comparison_rows(rows)
    assert all(row["comparison_status"] == "missing condition" for row in comparisons)
    assert all(row["comparison_reason"].startswith("no ") for row in comparisons)
    duplicate = comparison_rows([*data.rows, data.rows[0]])[0]
    assert duplicate["comparison_status"] == "ambiguous duplicate conditions"
    assert duplicate["comparison_reason"] == "same arm loaded more than once"
    assert duplicate["score_delta"] is None


def test_dedupe_rows_drops_only_reloaded_rows() -> None:
    rows = parse_dashboard_report(_report(), "x").rows
    deduped, dropped = dedupe_rows(rows + copy.deepcopy(rows))
    assert (deduped, dropped) == (rows, 2)
    # Identical embedded reports in one file are separate occurrences, not reloads.
    assert dedupe_rows(parse_dashboard_report([_report(), _report()], "x").rows)[1] == 0
    assert comparison_rows(deduped)[0]["comparison_status"] == "comparable"


@pytest.mark.parametrize("scope", ["reports", "agents"])
def test_status_filter_cannot_pair_different_report_or_agent_occurrences(scope: str) -> None:
    first, second = _report(), _report()
    first_agent = first["agents"]["claude-code"]
    second_agent = second["agents"]["claude-code"]
    first_agent["conditions"]["without_skill"]["execution_status"] = "failed"
    second_agent["conditions"]["with_skill"]["execution_status"] = "failed"
    if scope == "reports":
        payload = [first, second]
    else:
        first["agents"]["second-occurrence"] = second_agent
        payload = first
    rows = [row for row in parse_dashboard_report(payload, "combined.json").rows if row["status"] == "succeeded"]
    comparisons = comparison_rows(rows)
    assert len(comparisons) == 2
    assert all(row["comparison_status"] == "missing condition" for row in comparisons)
    assert all(row["score_delta"] is None for row in comparisons)


def test_display_identical_feature_labels_cannot_pair_different_configurations() -> None:
    report = _report()
    conditions = report["agents"]["claude-code"]["conditions"]
    conditions["with_skill"]["dashboard_metadata"] = {"harness_features": ["memory, rules"]}
    conditions["without_skill"]["dashboard_metadata"] = {"harness_features": ["memory", "rules"]}
    rows = parse_dashboard_report(report, "report.json").rows
    assert rows[0]["features"] == rows[1]["features"]
    assert all(row["comparison_status"] == "missing condition" for row in comparison_rows(rows))


def test_zero_attempt_identity_and_conflicting_attempt_fields() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["trials"][0]["attempt"] = 0
    agent["trials_baseline"][0]["attempt"] = "000"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"] == "comparable"
    agent["trials"][0]["attempt_index"] = 2
    agent["trials_baseline"][0]["attempt"] = 2
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["score_delta"] is None


def test_long_numeric_attempt_field_is_normalized_without_integer_conversion() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    # Longer than int()'s default 4300-digit limit for decimal strings.
    agent["trials"][0]["attempt"] = "0" * 10 + "9" * 5000
    agent["trials_baseline"][0]["attempt"] = "9" * 5000
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "comparable"
    agent["trials_baseline"][0]["attempt"] = "9" * 4999 + "8"
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["comparison_reason"] == "different cases or attempts"


def test_zero_baseline_and_nonfinite_values() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["trials_baseline"][0]["cost_usd"] = 0
    agent["trials"][0]["agent_seconds"] = float("inf")
    agent["trials"][0]["tokens"]["cached"] = True
    report["dashboard_metadata"] = {"harness_features": {"invalid": float("nan")}}
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["features"] == "unknown"
    assert data.rows[0]["cached_tokens"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["cost_usd_delta"] == 0.02
    assert comparison["cost_usd_percent"] is None
    assert comparison["duration_seconds_delta"] is None


def test_invalid_agent_seconds_does_not_fall_back_to_legacy_duration() -> None:
    report = _report()
    trial = report["agents"]["claude-code"]["trials"][0]
    trial.update(agent_seconds=-1, duration_seconds=5)
    row = parse_dashboard_report(report, "report.json").rows[0]
    assert row["duration_seconds"] is None
    assert "time 0/1" in row["coverage"]


def test_legacy_trial_duration_seconds_is_still_read() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field in ("trials", "trials_baseline"):
        agent[field][0]["duration_seconds"] = agent[field][0].pop("agent_seconds")
    data = parse_dashboard_report(report, "report.json")
    assert [row["duration_seconds"] for row in data.rows] == [8.0, 10.0]


def test_malformed_status_and_aggregate_overflow_fail_closed() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["execution_status"] = ["succeeded"]
    agent["conditions"] = {}
    assert all(row["status"] == "unknown" for row in parse_dashboard_report(report, "report.json").rows)
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["trials"][0]["tokens"] = {"prompt": 1e308, "completion": 1e308}
    row = parse_dashboard_report(report, "report.json").rows[0]
    assert row["total_tokens"] is None


def _repeated_attempts_report(trial_ids: tuple[str, ...]) -> dict:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field, count_field, arm in (
        ("trials", "num_trials", "with_skill"),
        ("trials_baseline", "num_trials_baseline", "without_skill"),
    ):
        agent[field] = [{**copy.deepcopy(agent[field][0]), "trial_id": trial_id} for trial_id in trial_ids]
        agent[count_field] = len(trial_ids)
        agent["conditions"][arm].update(expected_attempts=len(trial_ids), scored_attempts=len(trial_ids))
    return report


def test_repeated_attempts_pair_by_case_with_distinct_trial_ids() -> None:
    report = _repeated_attempts_report(("case-a__trial0", "case-a__trial1"))
    agent = report["agents"]["claude-code"]
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["score_delta"] == pytest.approx(0.2)
    # Harbor's attempt names are random, so arms pair by case count, not by name.
    agent["trials_baseline"][1]["trial_id"] = "case-a__trial2"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"] == "comparable"
    for field in ("trials", "trials_baseline"):
        for index, trial in enumerate(agent[field]):
            trial["trial_id"] = f"case-a__unrelated-{index}"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"] == "comparable"

    report = _repeated_attempts_report(("case-a__trial0", "case-a__trial1"))
    report["agents"]["claude-code"]["trials"][1]["trial_id"] = ""
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None

    report = _repeated_attempts_report(("case-a__trial0", "case-a__trial1"))
    report["agents"]["claude-code"]["trials"][1].update(entry_id="case-b", trial_id="case-a__trial0")
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None

    report = _repeated_attempts_report(("case-a__trial0", "case-a__trial1"))
    agent = report["agents"]["claude-code"]
    agent["trials_baseline"].pop()
    agent["num_trials_baseline"] = 1
    agent["conditions"]["without_skill"].update(expected_attempts=1, scored_attempts=1)
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["score_delta"] is None

    report = _repeated_attempts_report(("case-a__trial0", "case-a__trial1"))
    agent = report["agents"]["claude-code"]
    for field, attempts in (("trials", (1, 2)), ("trials_baseline", (1, 3))):
        for trial, attempt in zip(agent[field], attempts, strict=True):
            trial["attempt"] = attempt
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["score_delta"] is None

    report = _repeated_attempts_report(("case-a__trial0", "case-a__trial1"))
    report["agents"]["claude-code"]["trials"][0]["attempt"] = 1
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None


def test_authored_case_id_containing_attempt_text_is_not_parsed() -> None:
    report = _repeated_attempts_report(("retry-attempt1__aaa", "retry-attempt1__bbb"))
    for field in ("trials", "trials_baseline"):
        for trial in report["agents"]["claude-code"][field]:
            trial["entry_id"] = "retry-attempt1"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"] == "comparable"


def test_multistep_rewards_keep_logical_scores_without_ambiguous_resource_sums() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field, count_field in (("trials", "num_trials"), ("trials_baseline", "num_trials_baseline")):
        agent[field].append(copy.deepcopy(agent[field][0]))
        agent[count_field] = 2
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["trials"] == 1
    assert data.rows[0]["total_tokens"] is None
    assert data.rows[0]["duration_seconds"] is None
    assert data.rows[0]["cost_usd"] is None
    assert any("resource ownership is ambiguous" in warning for warning in data.warnings)
    comparison = comparison_rows(data.rows)[0]
    assert comparison["score_delta"] == pytest.approx(0.2)
    assert comparison["total_tokens_delta"] is None
    # Different counters do not establish whether each row measures one step
    # or mirrors the same physical trial's trajectory.
    agent["trials"][1]["tokens"]["prompt"] = 10
    assert parse_dashboard_report(report, "report.json").rows[0]["total_tokens"] is None


@pytest.mark.parametrize("conflict", ["different_attempt", "conflicting_fields", "invalid_attempt"])
def test_duplicate_physical_trials_require_consistent_attempt_identity(conflict: str) -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field, count in (("trials", "num_trials"), ("trials_baseline", "num_trials_baseline")):
        agent[field][0]["attempt"] = 1
        agent[field].append(copy.deepcopy(agent[field][0]))
        agent[count] = 2
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"] == "comparable"
    duplicate = agent["trials"][1]
    if conflict == "different_attempt":
        duplicate["attempt"] = 2
    elif conflict == "conflicting_fields":
        duplicate["attempt_index"] = 2
    else:
        duplicate["attempt"] = True
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["score_delta"] is None
    assert comparison["pass_rate_delta"] is None


def test_explicit_adverse_row_has_no_invented_baseline_delta() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent.update(adverse=0.2, trials_adverse=copy.deepcopy(agent["trials"]), num_trials_adverse=1)
    agent["conditions"]["adverse"] = {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
    data = parse_dashboard_report(report, "report.json")
    assert [row["condition"] for row in data.rows] == ["with_skill", "without_skill", "adverse"]
    assert data.rows[2]["score"] == 0.2
    assert len(comparison_rows(data.rows)) == 1


@pytest.mark.parametrize("omitted", ["evidence_entries", "raw_trial_rewards", "non_best_agent_details"])
def test_diagnostic_truncation_preserves_complete_trial_comparisons(omitted: str) -> None:
    report = _report()
    report["report_truncation"] = {"truncated": True, "omitted": {omitted: 50}}
    data = parse_dashboard_report(report, "report.json")
    assert data.warnings
    assert data.rows[0]["score"] == 0.8
    assert comparison_rows(data.rows)[0]["comparison_status"] == "comparable"


def test_pruned_non_best_agent_reports_declared_trials_and_reason() -> None:
    report = _report()
    report["best_agent"] = "best"
    report["agents"]["best"] = copy.deepcopy(report["agents"]["claude-code"])
    agent = report["agents"]["claude-code"]
    agent.update(num_trials=5, num_trials_baseline=5, conditions={}, trials=[], trials_baseline=[])
    report["report_truncation"] = {"truncated": True, "omitted": {"non_best_agent_details": 12}}
    data = parse_dashboard_report(report, "report.json")
    rows = [row for row in data.rows if row["_agent_key"] == "claude-code"]
    assert [row["trials"] for row in rows] == [5, 5]
    assert all(row["score"] is None and "tokens 0/5" in row["coverage"] for row in rows)
    comparisons = {row["_agent_key"]: row for row in comparison_rows(data.rows)}
    assert comparisons["best"]["comparison_status"] == "comparable"
    assert comparisons["claude-code"]["comparison_status"] == "incomplete or mismatched coverage"
    assert comparisons["claude-code"]["comparison_reason"] == (
        "trial details pruned from embedded report; load the retained run directory"
    )
    assert comparisons["claude-code"]["score_delta"] is None


def test_artifact_loading_truncation_blocks_comparisons() -> None:
    report = _report()
    report["report_truncation"] = {
        "truncated": True,
        "omitted": {"evidence_entries": 50},
        "artifact_loading": [{"code": "trial_limit", "artifact": "with-skill", "limit": 512}],
    }
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None


_TRAJECTORY_BYTES = {"code": "json_bytes", "artifact": "trajectory", "limit": 2 * 1024 * 1024}


@pytest.mark.parametrize(
    ("marker", "status"),
    [
        ({"truncated": True, "omitted": {}, "artifact_loading": [_TRAJECTORY_BYTES]}, "comparable"),
        (
            {
                "truncated": True,
                "omitted": {},
                "artifact_loading": [
                    _TRAJECTORY_BYTES,
                    {"code": "json_bytes", "artifact": "reward", "limit": 2 * 1024 * 1024},
                ],
            },
            "incomplete or mismatched coverage",
        ),
        (
            {"truncated": True, "artifact_loading": [{"artifact": "with-skill", "code": "trial_limit"}]},
            "incomplete or mismatched coverage",
        ),
        (
            {"truncated": True, "artifact_loading": [{"artifact": "mystery", "code": "json_bytes"}]},
            "incomplete or mismatched coverage",
        ),
        (
            {"truncated": True, "artifact_loading": [{"artifact": ["trajectory"], "code": "json_bytes"}]},
            "incomplete or mismatched coverage",
        ),
        ({"truncated": True}, "incomplete or mismatched coverage"),
        ({"truncated": True, "artifact_loading": _TRAJECTORY_BYTES}, "incomplete or mismatched coverage"),
    ],
)
def test_only_trajectory_loader_bounds_keep_trial_coverage(marker: dict, status: str) -> None:
    report = _report()
    report["report_truncation"] = marker
    data = parse_dashboard_report(report, "report.json")
    assert data.warnings
    assert comparison_rows(data.rows)[0]["comparison_status"] == status


def test_trajectory_reason_does_not_hide_missing_trial() -> None:
    report = _report()
    report["agents"]["claude-code"]["trials_baseline"] = []
    report["report_truncation"] = {"truncated": True, "omitted": {}, "artifact_loading": [_TRAJECTORY_BYTES]}
    status = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"]
    assert status == "incomplete or mismatched coverage"


def test_bad_input_is_diagnostic(tmp_path: Path) -> None:
    malformed = tmp_path / "broken.json"
    malformed.write_text("{", encoding="utf-8")
    assert load_dashboard_path(malformed).warnings
    assert load_dashboard_path(tmp_path / "missing").warnings


def test_directory_skips_symlinks(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    good = tmp_path / "report.json"
    good.write_text(json.dumps(_report()), encoding="utf-8")
    try:
        (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are unavailable on this host")
    data = load_dashboard_path(tmp_path)
    assert len(data.rows) == 2
    assert any("cannot read" in warning for warning in data.warnings)


def test_catalog_index_loads_authoritative_local_reports_with_containment(tmp_path: Path) -> None:
    reports = tmp_path / "calculator"
    reports.mkdir()
    (reports / "current.json").write_text(json.dumps(_report()), encoding="utf-8")
    index = tmp_path / "catalog-summary.json"
    index.write_text(
        json.dumps(
            {
                "skills": [
                    {"report_dir": "calculator", "json_report": "current.json"},
                    {"report_dir": "..", "json_report": "outside.json"},
                ]
            }
        ),
        encoding="utf-8",
    )
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert data.warnings


def _native_run(path: Path) -> None:
    metrics = ["security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check"]
    path.mkdir()
    run_config = {"agents": {"opencode": {"agent": "opencode", "model": "m"}}}
    # Like the runner, result.json embeds the config that run_config.json holds.
    (path / "result.json").write_text(
        json.dumps({"skill_name": "native-skill", "run_id": "native-run", "run_config": run_config}), encoding="utf-8"
    )
    (path / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    for variant, input_tokens in (("with-skill", 200), ("without-skill", 100)):
        condition = path / "opencode" / variant
        trial = condition / "trials" / "case-a__random"
        trial.mkdir(parents=True)
        (condition / "summary.json").write_text(
            json.dumps(
                {
                    "agent": "opencode",
                    "model": "m",
                    "scores": dict.fromkeys(metrics, 0.8),
                    "metrics": metrics,
                    "num_trials": 1,
                    "execution_status": "succeeded",
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
            ),
            encoding="utf-8",
        )
        (trial / "reward.json").write_text(
            json.dumps(
                {
                    **dict.fromkeys(metrics, 0.8),
                    "entry_id": "case-a",
                    "trial_id": "case-a__random",
                    "overall": 0.8,
                }
            ),
            encoding="utf-8",
        )
        (trial / "trajectory.json").write_text(
            json.dumps(
                {
                    "steps": [],
                    "final_metrics": {"total_prompt_tokens": input_tokens, "total_completion_tokens": 10},
                }
            ),
            encoding="utf-8",
        )
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "agent_result": {"n_input_tokens": input_tokens, "n_output_tokens": 10, "cost_usd": None},
                    "agent_execution": {"started_at": "2026-10-01T00:00:00Z", "finished_at": "2026-10-01T00:00:10Z"},
                }
            ),
            encoding="utf-8",
        )


def test_native_results_use_real_usage_and_prune_nested_mirrors(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    data = load_dashboard_path(tmp_path)
    assert len(data.rows) == 2
    assert data.rows[0]["skill"] == "native-skill"
    assert data.rows[0]["prompt_tokens"] == 200
    assert data.rows[0]["duration_seconds"] == 10
    assert data.rows[0]["cost_usd"] is None
    assert data.rows[0]["cached_tokens"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["total_tokens_delta"] == 100
    assert load_dashboard_path(run / "result.json").rows == load_dashboard_path(run).rows


def test_native_per_trial_usage_uses_retained_measurements(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    rows = load_dashboard_path(run).rows
    assert [row["total_tokens_per_trial"] for row in rows] == [210.0, 110.0]
    assert [row["duration_seconds_per_trial"] for row in rows] == [10.0, 10.0]
    assert [row["cost_usd_per_trial"] for row in rows] == [None, None]


def _canonical_export(run: Path) -> dict:
    result = agent_eval_result_from_directory(Path("native-skill"), run, use_llm_judge=False)
    assert result is not None
    return json.loads(json.dumps(result.metadata["agent_eval"]))


def test_canonical_export_agent_seconds_matches_retained_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    canonical = parse_dashboard_report(_canonical_export(run), "r.json")
    retained = load_dashboard_path(run)
    assert (
        [row["duration_seconds"] for row in canonical.rows]
        == [row["duration_seconds"] for row in retained.rows]
        == [10.0, 10.0]
    )
    assert all("time 1/1" in row["coverage"] for row in canonical.rows)
    comparison = comparison_rows(canonical.rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["duration_seconds_delta"] == 0.0


def test_unreadable_trajectory_keeps_retained_time_and_token_fallback(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    trajectory = run / "opencode" / "with-skill" / "trials" / "case-a__random" / "trajectory.json"
    trajectory.write_text("{not json", encoding="utf-8")
    retained = load_dashboard_path(run)
    canonical = parse_dashboard_report(_canonical_export(run), "r.json")
    assert retained.rows[0]["duration_seconds"] == canonical.rows[0]["duration_seconds"] == 10.0
    assert comparison_rows(retained.rows)[0]["duration_seconds_delta"] == 0.0
    # Harbor's agent_result counters stand in for the unreadable trajectory totals.
    assert retained.rows[0]["total_tokens"] == 210
    assert any("cannot read trajectory" in warning for warning in retained.warnings)


def test_retained_run_ignores_export_zero_token_notice(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    engine_result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    engine_result["evaluator_version"] = "0.4.0"
    (run / "result.json").write_text(json.dumps(engine_result), encoding="utf-8")
    for trajectory_path in run.glob("opencode/*/trials/*/trajectory.json"):
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        trajectory["final_metrics"]["total_cached_tokens"] = 0
        trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    assert any("before v0.5.0" in warning for warning in parse_dashboard_report(_canonical_export(run), "r").warnings)
    data = load_dashboard_path(run)
    # Retained trajectories hold recorded counters, so their zeros stay measured.
    assert [row["cached_tokens"] for row in data.rows] == [0, 0]
    assert not any("before v0.5.0" in warning for warning in data.warnings)


def test_retained_naive_timestamps_follow_canonical_timezone_policy(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    for result_path in run.glob("*/*/trials/*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["agent_execution"] = {"started_at": "2026-10-01T00:00:00", "finished_at": "2026-10-01T00:00:10"}
        result_path.write_text(json.dumps(result), encoding="utf-8")
    rows = load_dashboard_path(run).rows
    assert len(rows) == 2
    assert all(row["duration_seconds"] is None for row in rows)
    assert all("time 0/1" in row["coverage"] for row in rows)
    agent = _canonical_export(run)["agents"]["opencode"]
    assert all("agent_seconds" not in trial for trial in agent["trials"] + agent["trials_baseline"])


def test_oversized_trajectory_keeps_scores_comparable(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    trajectory = run / "opencode" / "with-skill" / "trials" / "case-a__random" / "trajectory.json"
    # Larger than the report loader's 2 MiB JSON bound, below the dashboard's own.
    trajectory.write_text(json.dumps({"steps": [], "padding": "x" * (3 * 1024 * 1024)}), encoding="utf-8")
    comparison = comparison_rows(load_dashboard_path(run).rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["score_delta"] == 0
    canonical = _canonical_export(run)
    assert canonical["report_truncation"]["artifact_loading"][0]["artifact"] == "trajectory"
    rows = parse_dashboard_report(canonical, "r.json").rows
    assert comparison_rows(rows)[0]["comparison_status"] == "comparable"
    assert rows[0]["condition"] == "with_skill"
    assert rows[0]["total_tokens"] is None


def test_retained_run_keeps_non_best_agent_trials_beyond_embedded_budget(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.evaluation import tier3_report

    run = tmp_path / "run"
    _native_run(run)
    shutil.copytree(run / "opencode", run / "codex")
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    config["agents"]["codex"] = {"agent": "codex", "model": "m"}
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    for summary_path in run.glob("codex/*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["agent"] = "codex"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    # A budget just below the full payload prunes only the non-best agent's details.
    budget = tier3_report._serialized_payload_size(_canonical_export(run)) - 200
    monkeypatch.setattr(tier3_report, "_MAX_EMBEDDED_REPORT_BYTES", budget)
    canonical = _canonical_export(run)
    assert canonical["report_truncation"]["omitted"]["non_best_agent_details"]
    pruned = next(name for name in canonical["agents"] if name != canonical["best_agent"])

    rows = load_dashboard_path(run).rows
    assert len(rows) == 4
    assert all(row["trials"] == 1 and row["score"] == 0.8 and row["total_tokens"] is not None for row in rows)
    assert all(row["comparison_status"] == "comparable" for row in comparison_rows(rows))
    # The bounded canonical export still fails closed for the pruned agent.
    comparison = next(
        row for row in comparison_rows(parse_dashboard_report(canonical, "r.json").rows) if row["_agent_key"] == pruned
    )
    assert comparison["coverage"] == "1 with / 1 without"
    assert comparison["comparison_reason"].startswith("trial details pruned from embedded report")


def test_catalog_native_and_mixed_references_are_loaded_without_cycles(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    index = tmp_path / "catalog.json"
    reference = {"report_dir": "run", "json_report": "result.json"}
    index.write_text(json.dumps({"skills": [reference]}), encoding="utf-8")
    assert load_dashboard_path(index).rows == load_dashboard_path(run).rows
    index.write_text(json.dumps({"skills": [{"tier3": _report()}, reference]}), encoding="utf-8")
    assert len(load_dashboard_path(index).rows) == 4
    index.write_text(
        json.dumps({"skills": [reference, {"report_dir": ".", "json_report": "catalog.json"}]}), encoding="utf-8"
    )
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert any("cyclic" in warning for warning in data.warnings)


def test_nested_catalog_references_load_each_report_once(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    reference = {"report_dir": ".", "json_report": "report.json"}
    (tmp_path / "child.json").write_text(json.dumps({"skills": [reference] * 10}), encoding="utf-8")
    child = {"report_dir": ".", "json_report": "child.json"}
    index = tmp_path / "catalog.json"
    index.write_text(json.dumps({"skills": [child] * 10}), encoding="utf-8")
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert comparison_rows(data.rows)[0]["comparison_status"] == "comparable"
    assert not data.warnings


def test_catalog_file_budget_is_shared_across_nested_references(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.reporting import dashboard_data

    monkeypatch.setattr(dashboard_data, "_MAX_FILES", 3)
    index = tmp_path / "catalog.json"
    (tmp_path / "one.json").write_text(json.dumps(_report()), encoding="utf-8")
    second = _report()
    second["skill_name"] = "second-skill"
    (tmp_path / "two.json").write_text(json.dumps(second), encoding="utf-8")
    (tmp_path / "child.json").write_text(
        json.dumps({"skills": [{"report_dir": ".", "json_report": name} for name in ("one.json", "two.json")]}),
        encoding="utf-8",
    )
    index.write_text(json.dumps({"skills": [{"report_dir": ".", "json_report": "child.json"}]}), encoding="utf-8")
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert any("catalog limit" in warning for warning in data.warnings)


def test_directory_enumeration_stops_before_materializing_all_entries(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.reporting import dashboard_data

    real_scandir = dashboard_data.os.scandir
    scanned = []

    class CountingScanner:
        def __enter__(self):
            self.scanner = real_scandir(tmp_path)
            return self

        def __exit__(self, *args):
            self.scanner.close()

        def __iter__(self):
            for entry in self.scanner:
                scanned.append(entry.name)
                yield entry

    for index in range(20):
        (tmp_path / f"report-{index}.json").write_text(json.dumps(_report()), encoding="utf-8")
    monkeypatch.setattr(dashboard_data, "_MAX_PATHS", 4)
    monkeypatch.setattr(dashboard_data.os, "scandir", lambda _path: CountingScanner())
    data = load_dashboard_path(tmp_path)
    assert len(scanned) == 5
    assert len(data.rows) == 8
    assert any("discovery limit" in warning for warning in data.warnings)


def test_native_trial_bounds_mirror_report_data() -> None:
    from skillevaluator.reporting import dashboard_data
    from skillevaluator.tier3.harbor import report_data

    assert dashboard_data._MAX_NATIVE_TRIALS == report_data._MAX_TRIALS_PER_CONDITION
    assert dashboard_data._MAX_NATIVE_TRIAL_ENTRIES == report_data._MAX_TRIAL_PATHS_SCANNED


def test_native_trial_discovery_limit_keeps_usage_unavailable(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.reporting import dashboard_data

    run = tmp_path / "run"
    _native_run(run)
    condition = run / "opencode" / "with-skill"
    original = condition / "trials" / "case-a__random"
    for index in (1, 2):
        trial_dir = shutil.copytree(original, original.with_name(f"case-a__trial{index}"))
        reward_path = trial_dir / "reward.json"
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        reward["trial_id"] = trial_dir.name
        reward_path.write_text(json.dumps(reward), encoding="utf-8")
    summary_path = condition / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(num_trials=3, expected_attempts=3, scored_attempts=3)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    # Within the bound all three trials are measured.
    assert next(row for row in load_dashboard_path(run).rows if row["condition"] == "with_skill")["total_tokens"] == 630
    monkeypatch.setattr(dashboard_data, "_MAX_NATIVE_TRIALS", 2)
    data = load_dashboard_path(run)
    row = next(row for row in data.rows if row["condition"] == "with_skill")
    assert any("trial discovery limit reached; usage totals are unavailable." in warning for warning in data.warnings)
    assert row["total_tokens"] is None
    assert row["duration_seconds"] is None


def test_local_json_over_limit_is_a_notice_and_other_reports_load(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.reporting import dashboard_data

    monkeypatch.setattr(dashboard_data, "_MAX_FILE_BYTES", 2048)
    content = json.dumps(_report())
    exact = tmp_path / "exact.json"
    exact.write_text(content.ljust(2048), encoding="utf-8")
    oversized = tmp_path / "oversized.json"
    oversized.write_text(content.ljust(2049), encoding="utf-8")
    assert (exact.stat().st_size, oversized.stat().st_size) == (2048, 2049)
    assert len(load_dashboard_path(exact).rows) == 2
    assert any("16 MiB" in warning for warning in load_dashboard_path(oversized).warnings)
    data = load_dashboard_path(tmp_path)
    assert {row["source"] for row in data.rows} == {str(exact.resolve())}
    assert len(data.rows) == 2
    assert any(str(oversized.resolve()) in warning and "16 MiB" in warning for warning in data.warnings)


def test_directory_file_budget_counts_malformed_reports(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.reporting import dashboard_data

    monkeypatch.setattr(dashboard_data, "_MAX_FILES", 3)
    for index in range(20):
        (tmp_path / f"broken-{index}.json").write_text("{", encoding="utf-8")
    data = load_dashboard_path(tmp_path)
    assert sum("cannot read report" in warning for warning in data.warnings) == 3
    assert any("file limit" in warning for warning in data.warnings)


def test_cyclic_input_catalog_and_native_symlinks_are_diagnostics(tmp_path: Path) -> None:
    loop = tmp_path / "loop.json"
    try:
        loop.symlink_to(loop)
    except OSError:
        pytest.skip("Symlinks are unavailable on this host")
    assert load_dashboard_path(loop).warnings
    index = tmp_path / "catalog.json"
    index.write_text(json.dumps({"skills": [{"report_dir": ".", "json_report": "loop.json"}]}), encoding="utf-8")
    assert load_dashboard_path(index).warnings
    run = tmp_path / "run"
    _native_run(run)
    trials = run / "opencode" / "with-skill" / "trials"
    shutil.rmtree(trials)
    trials.symlink_to(trials, target_is_directory=True)
    data = load_dashboard_path(run)
    assert data.rows
    assert any("symlink" in warning for warning in data.warnings)
    assert next(row for row in data.rows if row["condition"] == "with_skill")["total_tokens"] is None


def test_uploaded_native_engine_json_needs_retained_artifacts() -> None:
    raw_result = {
        "skill_name": "native-skill",
        "run_id": "native-run",
        "agents": {
            "opencode": {
                "with_skill": {"security": 0.8},
                "without_skill": {"security": 0.6},
                "num_trials_with": 1,
                "num_trials_without": 1,
            }
        },
    }
    data = parse_dashboard_report(raw_result, "result.json")
    assert data.rows == []
    assert any("retained run directory" in warning for warning in data.warnings)


@pytest.mark.parametrize("malformation", ["summary_status", "run_config_agents"])
def test_malformed_native_schema_is_diagnostic_and_preserves_other_sources(tmp_path: Path, malformation: str) -> None:
    run = tmp_path / "bad-run"
    _native_run(run)
    if malformation == "summary_status":
        path = run / "opencode" / "with-skill" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["execution_status"] = ["succeeded"]
        path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        (run / "run_config.json").write_text(json.dumps({"agents": [{"agent": "opencode"}]}), encoding="utf-8")
    assert load_dashboard_path(run / "result.json").warnings
    (tmp_path / "valid.json").write_text(json.dumps(_report()), encoding="utf-8")
    data = load_dashboard_path(tmp_path)
    assert len(data.rows) == 2
    assert {row["skill"] for row in data.rows} == {"calculator"}
    assert data.warnings


def test_native_missing_attempt_does_not_publish_partial_usage_as_total(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    for variant in ("with-skill", "without-skill"):
        summary_path = run / "opencode" / variant / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(num_trials=2, expected_attempts=2, scored_attempts=2)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    data = load_dashboard_path(run)
    assert all(row["score"] is None and row["total_tokens"] is None for row in data.rows)
    assert all(row["duration_seconds"] is None for row in data.rows)
    assert all("tokens 1/2" in row["coverage"] and "time 1/2" in row["coverage"] for row in data.rows)


def test_native_generated_stop_on_pass_names_preserve_attempt_sets(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    for variant, label in (("with-skill", "with"), ("without-skill", "without")):
        condition = run / "opencode" / variant
        first_name = f"demo-opencode-{label}-case-a-attempt001__case-a__random1"
        second_name = f"demo-opencode-{label}-case-a-attempt002__case-a__random2"
        original = condition / "trials" / "case-a__random"
        first = original.rename(original.with_name(first_name))
        second = first.with_name(second_name)
        shutil.copytree(first, second)
        for trial_dir in (first, second):
            reward_path = trial_dir / "reward.json"
            reward = json.loads(reward_path.read_text(encoding="utf-8"))
            reward["trial_id"] = trial_dir.name
            reward_path.write_text(json.dumps(reward), encoding="utf-8")
        summary_path = condition / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(num_trials=2, expected_attempts=2, scored_attempts=2)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    data = load_dashboard_path(run)
    assert all(row["trials"] == 2 and row["score"] == 0.8 for row in data.rows)
    assert comparison_rows(data.rows)[0]["comparison_status"] == "comparable"


def test_native_different_condition_models_cannot_pair(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    baseline_summary = run / "opencode" / "without-skill" / "summary.json"
    summary = json.loads(baseline_summary.read_text(encoding="utf-8"))
    summary["model"] = "different-model"
    baseline_summary.write_text(json.dumps(summary), encoding="utf-8")
    comparisons = comparison_rows(load_dashboard_path(run).rows)
    assert len(comparisons) == 2
    assert all(row["comparison_status"] == "missing condition" for row in comparisons)


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", 10**30])
@pytest.mark.parametrize("field", ["num_trials", "expected_attempts", "scored_attempts"])
def test_native_invalid_coverage_counts_cannot_be_hidden_by_canonical_normalization(
    tmp_path: Path, field: str, value: object
) -> None:
    run = tmp_path / "run"
    _native_run(run)
    summary_path = run / "opencode" / "with-skill" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    data = load_dashboard_path(run)
    row = next(row for row in data.rows if row["condition"] == "with_skill")
    assert row["score"] is None
    assert row["total_tokens"] is None
    assert row["duration_seconds"] is None
    assert any("invalid trial coverage counts" in warning for warning in data.warnings)
    assert comparison_rows(data.rows)[0]["score_delta"] is None


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="Needs POSIX permissions enforced for this user")
def test_unreadable_condition_trials_keep_other_rows(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    trials = run / "opencode" / "with-skill" / "trials"
    trials.chmod(0)
    try:
        data = load_dashboard_path(run)
    finally:
        trials.chmod(0o755)
    assert len(data.rows) == 2
    assert next(row for row in data.rows if row["condition"] == "with_skill")["total_tokens"] is None
    assert next(row for row in data.rows if row["condition"] == "without_skill")["total_tokens"] == 110
    assert any("cannot read retained trials" in warning for warning in data.warnings)


@pytest.mark.parametrize("artifact", ["trials", "trajectory.json", "result.json", "reward.json"])
def test_native_usage_rejects_intermediate_directory_and_file_symlinks(tmp_path: Path, artifact: str) -> None:
    run = tmp_path / "run"
    _native_run(run)
    trials = run / "opencode" / "with-skill" / "trials"
    target = trials if artifact == "trials" else trials / "case-a__random" / artifact
    outside = tmp_path / ("outside-trials" if artifact == "trials" else artifact)
    target.rename(outside)
    try:
        target.symlink_to(outside, target_is_directory=artifact == "trials")
    except OSError:
        pytest.skip("Symlinks are unavailable on this host")
    data = load_dashboard_path(run)
    row = next(row for row in data.rows if row["condition"] == "with_skill")
    if artifact == "trajectory.json":
        # A skipped trajectory loses only its own totals; result.json still measures the trial.
        assert row["total_tokens"] == 210
        assert row["duration_seconds"] == 10
    else:
        assert row["total_tokens"] is None
        assert row["duration_seconds"] is None
    assert any("symlink" in warning for warning in data.warnings)


def test_multistep_native_cost_time_and_tokens_require_every_step(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    trial_result = run / "opencode" / "with-skill" / "trials" / "case-a__random" / "result.json"
    result = json.loads(trial_result.read_text(encoding="utf-8"))
    step = {
        "agent_result": {"n_input_tokens": 50, "n_output_tokens": 5, "n_cache_tokens": 0, "cost_usd": 0.1},
        "agent_execution": {"started_at": "2026-10-01T00:00:00Z", "finished_at": "2026-10-01T00:00:03Z"},
    }
    result["step_results"] = [step, copy.deepcopy(step)]
    trial_result.write_text(json.dumps(result), encoding="utf-8")
    row = load_dashboard_path(run).rows[0]
    assert row["prompt_tokens"] == 100
    assert row["duration_seconds"] == 6
    assert row["cost_usd"] == pytest.approx(0.2)
    result["step_results"][1]["agent_result"].pop("n_input_tokens")
    result["step_results"][1]["agent_result"].pop("cost_usd")
    trial_result.write_text(json.dumps(result), encoding="utf-8")
    row = load_dashboard_path(run).rows[0]
    assert row["prompt_tokens"] is None
    assert row["total_tokens"] is None
    assert row["cost_usd"] is None


def test_native_duplicate_reward_rows_require_authoritative_step_scope(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    condition = run / "opencode" / "with-skill"
    first = condition / "trials" / "case-a__random"
    second = first.with_name("case-a__random__finish")
    shutil.copytree(first, second)
    summary_path = condition / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # The collector counts one logical trial and records its two reward rows.
    summary["num_reward_rows"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for trial_dir in (first, second):
        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        step = {
            "step_name": "prepare",
            "agent_result": {"n_input_tokens": 50, "n_output_tokens": 5, "n_cache_tokens": 0, "cost_usd": 0.1},
            "agent_execution": result["agent_execution"],
        }
        result["step_results"] = [step, {**copy.deepcopy(step), "step_name": "finish"}]
        result_path.write_text(json.dumps(result), encoding="utf-8")
    data = load_dashboard_path(run)
    assert not any("resource ownership is ambiguous" in warning for warning in data.warnings)
    row = data.rows[0]
    assert row["score"] == 0.8
    assert row["total_tokens"] == 110
    assert row["duration_seconds"] == 20
    assert row["cost_usd"] == pytest.approx(0.2)
    for trial_dir in (first, second):
        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.pop("step_results")
        result_path.write_text(json.dumps(result), encoding="utf-8")
    data = load_dashboard_path(run)
    assert data.rows[0]["score"] == 0.8
    assert data.rows[0]["total_tokens"] is None
    assert data.rows[0]["duration_seconds"] is None
    assert any("resource ownership is ambiguous" in warning for warning in data.warnings)


def test_retained_feature_metadata_precedence_summary_over_agent_over_root(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "dashboard_metadata": {"harness_features": ["root"]},
                "agents": {
                    "opencode": {
                        "agent": "opencode",
                        "model": "m",
                        "dashboard_metadata": {"harness_features": ["agent"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    summary_path = run / "opencode" / "without-skill" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["dashboard_metadata"] = {"harness_features": ["summary"]}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    rows = {row["condition"]: row for row in load_dashboard_path(run).rows}

    assert rows["with_skill"]["features"] == "agent"
    assert rows["without_skill"]["features"] == "summary"
    assert all(item["comparison_status"] == "missing condition" for item in comparison_rows(list(rows.values())))
    # An unreadable run_config.json falls back to the copy in result.json.
    (run / "run_config.json").write_text("{", encoding="utf-8")
    data = load_dashboard_path(run)
    assert [row["features"] for row in data.rows] == ["unknown", "summary"]
    assert any("cannot read run config" in warning for warning in data.warnings)


def test_empty_feature_list_and_string_and_mapping_forms() -> None:
    labels = []
    for value in ([], "rules on", {"memory": False, "rules": True}):
        report = _report()
        report["dashboard_metadata"] = {"harness_features": value}
        labels.append(parse_dashboard_report(report, "r.json").rows[0]["features"])

    assert labels == ["none (recorded)", "rules on", '{"memory": false, "rules": true}']
