# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from skillevaluator.reporting.dashboard_data import comparison_rows, load_dashboard_path, parse_dashboard_report


def _report() -> dict:
    def trial(prompt: int, completion: int, seconds: float, cost: float) -> dict:
        return {
            "entry_id": "case-a",
            "trial_id": "case-a__random",
            "overall": 0.8,
            "tokens": {"prompt": prompt, "completion": completion, "cached": 0},
            "duration_seconds": seconds,
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
    assert comparison["score_delta"] == pytest.approx(0.2)
    assert comparison["total_tokens_delta"] == 30
    assert comparison["total_tokens_percent"] == 25
    assert comparison["duration_seconds_delta"] == -2
    assert comparison["cost_usd_percent"] == 100


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


def test_canonical_missing_attempt_keeps_usage_totals_unknown() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["conditions"]["with_skill"].update(expected_attempts=2, scored_attempts=1, execution_status="failed")
    row = parse_dashboard_report(report, "report.json").rows[0]
    assert row["total_tokens"] is None
    assert row["duration_seconds"] is None
    assert "tokens 1/2" in row["coverage"]


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
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


@pytest.mark.parametrize("mutation", ["failed", "missing_trial", "truncated", "different_case", "missing_attempt"])
def test_incomplete_or_unmatched_arms_block_all_deltas(mutation: str) -> None:
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
    comparison = comparison_rows(parse_dashboard_report(report, "same.json").rows)[0]
    assert comparison["comparison_status"] == "incomplete or mismatched coverage"
    assert comparison["score_delta"] is None
    assert comparison["total_tokens_delta"] is None


def test_disjoint_runs_features_and_duplicate_conditions_cannot_pair() -> None:
    data = parse_dashboard_report(_report(), "first.json")
    rows = copy.deepcopy(data.rows)
    rows[1]["source"] = "second.json"
    assert all(row["comparison_status"] == "missing condition" for row in comparison_rows(rows))
    rows = copy.deepcopy(data.rows)
    rows[1]["features"] = "rules"
    assert all(row["comparison_status"] == "missing condition" for row in comparison_rows(rows))
    duplicate = comparison_rows([*data.rows, data.rows[0]])[0]
    assert duplicate["comparison_status"] == "ambiguous duplicate conditions"
    assert duplicate["score_delta"] is None


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


def test_long_numeric_attempt_suffix_is_normalized_without_integer_conversion() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field in ("trials", "trials_baseline"):
        agent[field][0]["trial_id"] = "case-a__attempt" + "9" * 5000
    comparison = comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]
    assert comparison["comparison_status"] == "comparable"


def test_zero_baseline_and_nonfinite_values() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    agent["trials_baseline"][0]["cost_usd"] = 0
    agent["trials"][0]["duration_seconds"] = float("inf")
    agent["trials"][0]["tokens"]["cached"] = True
    report["dashboard_metadata"] = {"harness_features": {"invalid": float("nan")}}
    data = parse_dashboard_report(report, "report.json")
    assert data.rows[0]["features"] == "unknown"
    assert data.rows[0]["cached_tokens"] is None
    comparison = comparison_rows(data.rows)[0]
    assert comparison["cost_usd_delta"] == 0.02
    assert comparison["cost_usd_percent"] is None
    assert comparison["duration_seconds_delta"] is None


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


def test_repeated_attempts_require_recorded_identity_and_match_exact_set() -> None:
    report = _report()
    agent = report["agents"]["claude-code"]
    for field, count_field, arm in (
        ("trials", "num_trials", "with_skill"),
        ("trials_baseline", "num_trials_baseline", "without_skill"),
    ):
        agent[field].append(copy.deepcopy(agent[field][0]))
        agent[field][0]["trial_id"] = "case-a__attempt1__abc"
        agent[field][1]["trial_id"] = "case-a__attempt2__def"
        agent[count_field] = 2
        agent["conditions"][arm].update(expected_attempts=2, scored_attempts=2)
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["comparison_status"] == "comparable"
    agent["trials_baseline"][1]["trial_id"] = "case-a__attempt3__def"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None
    for field in ("trials", "trials_baseline"):
        for index, trial in enumerate(agent[field]):
            trial["trial_id"] = f"case-a__unrelated-{index}"
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None


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


def test_artifact_loading_truncation_blocks_comparisons() -> None:
    report = _report()
    report["report_truncation"] = {
        "truncated": True,
        "omitted": {"evidence_entries": 50},
        "artifact_loading": [{"code": "trial_limit", "artifact": "with-skill", "limit": 512}],
    }
    assert comparison_rows(parse_dashboard_report(report, "report.json").rows)[0]["score_delta"] is None


def test_bad_input_is_diagnostic_and_directory_skips_symlinks(tmp_path: Path) -> None:
    malformed = tmp_path / "broken.json"
    malformed.write_text("{", encoding="utf-8")
    assert load_dashboard_path(malformed).warnings
    assert load_dashboard_path(tmp_path / "missing").warnings
    good = tmp_path / "report.json"
    good.write_text(json.dumps(_report()), encoding="utf-8")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    data = load_dashboard_path(tmp_path)
    assert len(data.rows) == 2
    assert any("cannot read" in warning for warning in data.warnings)


def test_catalog_index_loads_authoritative_local_reports_with_containment(tmp_path: Path) -> None:
    reports = tmp_path / "calculator"
    reports.mkdir()
    (reports / "current.json").write_text(json.dumps(_report()))
    index = tmp_path / "catalog-summary.json"
    index.write_text(
        json.dumps(
            {
                "skills": [
                    {"report_dir": "calculator", "json_report": "current.json"},
                    {"report_dir": "..", "json_report": "outside.json"},
                ]
            }
        )
    )
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert data.warnings


def _native_run(path: Path) -> None:
    metrics = ["security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check"]
    path.mkdir()
    (path / "result.json").write_text(json.dumps({"skill_name": "native-skill", "run_id": "native-run"}))
    (path / "run_config.json").write_text(json.dumps({"agents": {"opencode": {"agent": "opencode", "model": "m"}}}))
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
            )
        )
        (trial / "reward.json").write_text(
            json.dumps(
                {
                    **dict.fromkeys(metrics, 0.8),
                    "entry_id": "case-a",
                    "trial_id": "case-a__random",
                    "overall": 0.8,
                }
            )
        )
        (trial / "trajectory.json").write_text(
            json.dumps(
                {
                    "steps": [],
                    "final_metrics": {"total_prompt_tokens": input_tokens, "total_completion_tokens": 10},
                }
            )
        )
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "agent_result": {"n_input_tokens": input_tokens, "n_output_tokens": 10, "cost_usd": None},
                    "agent_execution": {"started_at": "2026-10-01T00:00:00Z", "finished_at": "2026-10-01T00:00:10Z"},
                }
            )
        )


def test_native_results_use_real_usage_and_prune_nested_mirrors(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    (run / "report.json").write_text(json.dumps(_report()))
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


def test_catalog_native_and_mixed_references_are_loaded_without_cycles(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    index = tmp_path / "catalog.json"
    reference = {"report_dir": "run", "json_report": "result.json"}
    index.write_text(json.dumps({"skills": [reference]}))
    assert load_dashboard_path(index).rows == load_dashboard_path(run).rows
    index.write_text(json.dumps({"skills": [{"tier3": _report()}, reference]}))
    assert len(load_dashboard_path(index).rows) == 4
    index.write_text(json.dumps({"skills": [reference, {"report_dir": ".", "json_report": "catalog.json"}]}))
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert any("cyclic" in warning for warning in data.warnings)


def test_nested_catalog_references_load_each_report_once(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(json.dumps(_report()))
    reference = {"report_dir": ".", "json_report": "report.json"}
    (tmp_path / "child.json").write_text(json.dumps({"skills": [reference] * 10}))
    child = {"report_dir": ".", "json_report": "child.json"}
    index = tmp_path / "catalog.json"
    index.write_text(json.dumps({"skills": [child] * 10}))
    data = load_dashboard_path(index)
    assert len(data.rows) == 2
    assert comparison_rows(data.rows)[0]["comparison_status"] == "comparable"
    assert not data.warnings


def test_catalog_file_budget_is_shared_across_nested_references(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.reporting import dashboard_data

    monkeypatch.setattr(dashboard_data, "_MAX_FILES", 3)
    index = tmp_path / "catalog.json"
    (tmp_path / "one.json").write_text(json.dumps(_report()))
    second = _report()
    second["skill_name"] = "second-skill"
    (tmp_path / "two.json").write_text(json.dumps(second))
    (tmp_path / "child.json").write_text(
        json.dumps({"skills": [{"report_dir": ".", "json_report": name} for name in ("one.json", "two.json")]})
    )
    index.write_text(json.dumps({"skills": [{"report_dir": ".", "json_report": "child.json"}]}))
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
        (tmp_path / f"report-{index}.json").write_text(json.dumps(_report()))
    monkeypatch.setattr(dashboard_data, "_MAX_PATHS", 4)
    monkeypatch.setattr(dashboard_data.os, "scandir", lambda _path: CountingScanner())
    data = load_dashboard_path(tmp_path)
    assert len(scanned) == 5
    assert len(data.rows) == 8
    assert any("discovery limit" in warning for warning in data.warnings)


def test_directory_file_budget_counts_malformed_reports(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.reporting import dashboard_data

    monkeypatch.setattr(dashboard_data, "_MAX_FILES", 3)
    for index in range(20):
        (tmp_path / f"broken-{index}.json").write_text("{")
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
    index.write_text(json.dumps({"skills": [{"report_dir": ".", "json_report": "loop.json"}]}))
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
        summary = json.loads(path.read_text())
        summary["execution_status"] = ["succeeded"]
        path.write_text(json.dumps(summary))
    else:
        (run / "run_config.json").write_text(json.dumps({"agents": [{"agent": "opencode"}]}))
    assert load_dashboard_path(run / "result.json").warnings
    (tmp_path / "valid.json").write_text(json.dumps(_report()))
    data = load_dashboard_path(tmp_path)
    assert len(data.rows) == 2
    assert {row["skill"] for row in data.rows} == {"calculator"}
    assert data.warnings


def test_native_missing_attempt_does_not_publish_partial_usage_as_total(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    for variant in ("with-skill", "without-skill"):
        summary_path = run / "opencode" / variant / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(num_trials=2, expected_attempts=2, scored_attempts=2)
        summary_path.write_text(json.dumps(summary))
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
            reward = json.loads(reward_path.read_text())
            reward["trial_id"] = trial_dir.name
            reward_path.write_text(json.dumps(reward))
        summary_path = condition / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(num_trials=2, expected_attempts=2, scored_attempts=2)
        summary_path.write_text(json.dumps(summary))
    data = load_dashboard_path(run)
    assert all(row["trials"] == 2 and row["score"] == 0.8 for row in data.rows)
    assert comparison_rows(data.rows)[0]["comparison_status"] == "comparable"


def test_native_different_condition_models_cannot_pair(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    baseline_summary = run / "opencode" / "without-skill" / "summary.json"
    summary = json.loads(baseline_summary.read_text())
    summary["model"] = "different-model"
    baseline_summary.write_text(json.dumps(summary))
    comparisons = comparison_rows(load_dashboard_path(run).rows)
    assert len(comparisons) == 2
    assert all(row["comparison_status"] == "missing condition" for row in comparisons)


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
@pytest.mark.parametrize("field", ["num_trials", "expected_attempts", "scored_attempts"])
def test_native_invalid_coverage_counts_cannot_be_hidden_by_canonical_normalization(
    tmp_path: Path, field: str, value: object
) -> None:
    run = tmp_path / "run"
    _native_run(run)
    summary_path = run / "opencode" / "with-skill" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary[field] = value
    summary_path.write_text(json.dumps(summary))
    data = load_dashboard_path(run)
    row = next(row for row in data.rows if row["condition"] == "with_skill")
    assert row["score"] is None
    assert row["total_tokens"] is None
    assert row["duration_seconds"] is None
    assert any("invalid trial coverage counts" in warning for warning in data.warnings)
    assert comparison_rows(data.rows)[0]["score_delta"] is None


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
    assert row["total_tokens"] is None
    assert row["duration_seconds"] is None
    assert any("symlink" in warning for warning in data.warnings)


def test_multistep_native_cost_time_and_tokens_require_every_step(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _native_run(run)
    trial_result = run / "opencode" / "with-skill" / "trials" / "case-a__random" / "result.json"
    result = json.loads(trial_result.read_text())
    step = {
        "agent_result": {"n_input_tokens": 50, "n_output_tokens": 5, "n_cache_tokens": 0, "cost_usd": 0.1},
        "agent_execution": {"started_at": "2026-10-01T00:00:00Z", "finished_at": "2026-10-01T00:00:03Z"},
    }
    result["step_results"] = [step, copy.deepcopy(step)]
    trial_result.write_text(json.dumps(result))
    row = load_dashboard_path(run).rows[0]
    assert row["prompt_tokens"] == 100
    assert row["duration_seconds"] == 6
    assert row["cost_usd"] == pytest.approx(0.2)
    result["step_results"][1]["agent_result"].pop("n_input_tokens")
    result["step_results"][1]["agent_result"].pop("cost_usd")
    trial_result.write_text(json.dumps(result))
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
    summary = json.loads(summary_path.read_text())
    summary["num_trials"] = 2
    summary_path.write_text(json.dumps(summary))
    for trial_dir in (first, second):
        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text())
        step = {
            "step_name": "prepare",
            "agent_result": {"n_input_tokens": 50, "n_output_tokens": 5, "n_cache_tokens": 0, "cost_usd": 0.1},
            "agent_execution": result["agent_execution"],
        }
        result["step_results"] = [step, {**copy.deepcopy(step), "step_name": "finish"}]
        result_path.write_text(json.dumps(result))
    row = load_dashboard_path(run).rows[0]
    assert row["score"] == 0.8
    assert row["total_tokens"] == 110
    assert row["duration_seconds"] == 20
    assert row["cost_usd"] == pytest.approx(0.2)
    for trial_dir in (first, second):
        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text())
        result.pop("step_results")
        result_path.write_text(json.dumps(result))
    data = load_dashboard_path(run)
    assert data.rows[0]["score"] == 0.8
    assert data.rows[0]["total_tokens"] is None
    assert data.rows[0]["duration_seconds"] is None
    assert any("resource ownership is ambiguous" in warning for warning in data.warnings)
