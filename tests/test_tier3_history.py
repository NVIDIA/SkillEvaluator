# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from skillevaluator.cli import cli
from skillevaluator.tier3.baseline_store import comparison_identity
from skillevaluator.tier3.harbor.runner import _reserve_run_dir
from skillevaluator.tier3.history import collect_history, print_history


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _run(
    root: Path,
    hour: int,
    *,
    score=0.8,
    model="model-a",
    policy="policy-a",
    revision="a",
    status="succeeded",
    baseline="fresh",
):
    name = f"20260101_{hour:02d}0000"
    directory = root / name
    config = {
        "agents": {"opencode": {"model": model, "source": "explicit"}},
        "harbor": {"environment": "local"},
        "judge": {"model": "judge-a"},
        "evaluated_source": {"commit": revision * 40},
    }
    condition = {
        "execution_status": "succeeded",
        "scored_cases": 1,
        "expected_cases": 1,
        "scored_attempts": 1,
        "expected_attempts": 1,
    }
    result = {
        "run_id": name,
        "evaluated_at": f"2026-01-01T{hour:02d}:00:00+00:00",
        "execution_status": status,
        "dataset_digest": "sha256:cases",
        "expected_case_ids": ["case-001"],
        "evaluator_version": "1",
        "attempt_policy": {"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        "score_policy": policy,
        "baseline": {"mode": baseline},
        "publication_target": {"skill_digest": f"sha256:{revision}"},
        "agents": {
            "opencode": {
                "model": model,
                "overall_with_skill": score,
                "overall_without_skill": 0.5,
                "score_policy_with_skill": policy,
                "score_policy_without_skill": policy,
                "conditions": {"with_skill": copy.deepcopy(condition), "without_skill": copy.deepcopy(condition)},
                "lift": {"overall": {"with_skill": score, "without_skill": 0.5, "delta": round(score - 0.5, 4)}},
            }
        },
    }
    result["comparison_identity"] = comparison_identity(config, result)
    _write(directory / "run_config.json", config)
    _write(directory / "result.json", result)
    return directory


def _update(directory, update):
    path = directory / "result.json"
    result = json.loads(path.read_text())
    update(result)
    _write(path, result)


def test_chronology_revision_changes_and_lift_trend(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    _run(root, 3, score=0.9, revision="c", baseline="reused")
    _run(root, 1)
    _run(root, 2, score=0.5, revision="b")
    rows = collect_history(skill)["rows"]
    assert [r["run_id"] for r in rows] == ["20260101_010000", "20260101_020000", "20260101_030000"]
    assert [r["lift_delta"] for r in rows] == [0.3, 0.0, 0.4]
    assert [r["lift_change"] for r in rows] == [None, -0.3, 0.4]
    assert all(r["segment"] == 1 and not r["config_changed"] for r in rows)
    assert rows[-1]["baseline_mode"] == "reused"
    assert rows[-1]["coverage"]["with_skill"] == {"scored_cases": 1, "expected_cases": 1}


@pytest.mark.parametrize("change", ["model", "policy"])
def test_segment_break_stops_trend_across_configuration_change(tmp_path, change):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    _run(root, 1)
    _run(root, 2, **({"model": "model-b"} if change == "model" else {"policy": "policy-b"}))
    rows = collect_history(skill)["rows"]
    assert rows[1]["segment"] == 2 and rows[1]["config_changed"] is True
    assert rows[1]["lift_change"] is None
    assert ("model changed" if change == "model" else "score policy changed") in str(rows[1]["comparison_mismatches"])


@pytest.mark.parametrize(
    "failure",
    ["failed", "incomplete", "corrupt", "missing", "skipped", "baseline_only", "coverage", "unscored", "invalid_delta"],
)
def test_unavailable_lift_is_never_zero(tmp_path, failure):
    skill = tmp_path / "demo"
    d = _run(skill / "evals/results", 1, score=0.5)
    if failure == "corrupt":
        (d / "result.json").write_text("{")
    elif failure == "missing":
        (d / "result.json").unlink()
    else:

        def update(r):
            a = r["agents"]["opencode"]
            if failure in {"failed", "incomplete"}:
                r["execution_status"] = failure
            elif failure == "skipped":
                r["baseline"]["mode"] = "skipped"
                a["conditions"]["without_skill"]["execution_status"] = "skipped"
            elif failure == "baseline_only":
                a["conditions"]["with_skill"]["execution_status"] = "skipped"
            elif failure == "coverage":
                a["conditions"]["with_skill"]["scored_cases"] = 0
            elif failure == "unscored":
                a["overall_with_skill"] = None
            else:
                a["lift"]["overall"]["delta"] = True

        _update(d, update)
    row = collect_history(skill)["rows"][0]
    assert row["lift_delta"] is None
    assert row["lift_status"] == "unavailable" and row["lift_reasons"]


def test_unknown_old_identity_does_not_create_a_trend(tmp_path):
    skill = tmp_path / "demo"
    first = _run(skill / "evals/results", 1)
    second = _run(skill / "evals/results", 2)
    for d in (first, second):

        def update(r):
            r.pop("comparison_identity")
            r.pop("evaluator_version")
            r.pop("baseline")

        _update(d, update)
    rows = collect_history(skill)["rows"]
    assert rows[0]["skill_revision"] == "a" * 40
    assert rows[0]["baseline_mode"] is None
    assert rows[1]["lift_change"] is None
    assert rows[1]["identity_status"] == "unknown"
    assert rows[1]["config_changed"] is False


def test_external_root_configured_root_and_legacy_fallback(monkeypatch, tmp_path):
    skill = tmp_path / "demo"
    _run(skill / "evals/results", 1)
    _run(tmp_path / "external/demo", 2)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "external"))
    assert collect_history(skill)["rows"][0]["run_id"] == "20260101_020000"
    _run(tmp_path / "explicit/demo", 3)
    assert collect_history(skill, tmp_path / "explicit")["rows"][0]["run_id"] == "20260101_030000"
    monkeypatch.delenv("SKILLEVALUATOR_RESULTS_DIR")
    assert collect_history(skill, tmp_path / "empty")["rows"][0]["run_id"] == "20260101_010000"


@pytest.mark.parametrize("suffix", ["json", "csv", "md"])
def test_cli_exports_and_read_only_results(tmp_path, suffix):
    skill = tmp_path / "demo"
    root = tmp_path / "external/demo"
    _run(root, 1)
    _run(root, 2, model="model-b")
    skill.mkdir()

    def snapshot():
        return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}

    before = snapshot()
    output = tmp_path / f"history.{suffix}"
    args = ["tier3", "history", str(skill), "--results-dir", str(root.parent)]
    result = CliRunner().invoke(cli, [*args, "--json", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["rows"]) == 2
    if suffix == "json":
        assert json.loads(output.read_text())["rows"][1]["segment"] == 2
    elif suffix == "csv":
        assert list(csv.DictReader(io.StringIO(output.read_text())))[1]["config_changed"] == "True"
    else:
        assert "**Segment 2 break" in output.read_text()
    assert snapshot() == before
    table = CliRunner().invoke(cli, args)
    assert table.exit_code == 0, table.output
    assert "BREAK" in table.output and "model changed" in table.output
    assert json.loads(CliRunner().invoke(cli, [*args, "--agent", "absent", "--json"]).output)["rows"] == []
    refused = CliRunner().invoke(cli, [*args, "--output", str(root / "overwrite.md")])
    assert refused.exit_code != 0 and "outside" in refused.output
    assert snapshot() == before


def test_agent_segments_are_independent(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    for hour in (1, 2):
        directory = _run(root, hour)

        def update(r):
            r["agents"]["codex"] = copy.deepcopy(r["agents"]["opencode"])

        _update(directory, update)
    rows = collect_history(skill)["rows"]
    assert [r["agent"] for r in rows] == ["codex", "opencode", "codex", "opencode"]
    opencode = collect_history(skill, agent="opencode")["rows"]
    assert opencode[1]["lift_change"] == 0.0


def _set_codex(directory: Path, model: str | None, *, status: str = "succeeded") -> None:
    config_path = directory / "run_config.json"
    result_path = directory / "result.json"
    config = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())
    if model is None:
        config["agents"].pop("codex", None)
        result["agents"].pop("codex", None)
    else:
        config["agents"]["codex"] = {"model": model, "source": "explicit"}
        codex = copy.deepcopy(result["agents"]["opencode"])
        codex["model"] = model
        codex["execution_status"] = status
        result["agents"]["codex"] = codex
        result["agents"]["opencode"]["execution_status"] = "succeeded"
    result["comparison_identity"] = comparison_identity(config, result)
    _write(config_path, config)
    _write(result_path, result)


def test_agent_lift_status_is_independent_of_run_status(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    succeeded = _run(root, 1)
    failed = _run(root, 2, status="failed")
    _set_codex(succeeded, "codex-a")
    _set_codex(failed, "codex-a", status="failed")

    opencode = collect_history(skill, agent="opencode")["rows"]
    codex = collect_history(skill, agent="codex")["rows"]

    assert opencode[1]["status"] == "failed"
    assert opencode[1]["lift_delta"] == 0.3
    assert codex[1]["lift_delta"] is None
    assert "agent status is failed" in codex[1]["lift_reasons"]


def test_agent_segments_ignore_other_agents_changes(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    first, second, third = (_run(root, hour) for hour in (1, 2, 3))
    _set_codex(first, "codex-a")
    _set_codex(second, "codex-b")
    _set_codex(third, None)

    rows = collect_history(skill, agent="opencode")["rows"]

    assert [row["segment"] for row in rows] == [1, 1, 1]
    assert not any(row["config_changed"] for row in rows)
    assert rows[-1]["lift_change"] == 0.0


def test_with_and_without_columns_use_recorded_lift_operands(tmp_path):
    skill = tmp_path / "demo"
    run = _run(skill / "evals/results", 1, score=0.6)
    result = json.loads((run / "result.json").read_text())
    result["agents"]["opencode"]["lift"]["overall"] = {
        "with_skill": 0.6667,
        "without_skill": 0.5667,
        "delta": 0.1,
    }
    _write(run / "result.json", result)

    row = collect_history(skill)["rows"][0]

    assert row["with_skill_score"] == 0.6667
    assert row["without_skill_score"] == 0.5667


def test_csv_export_disables_newline_translation(monkeypatch, tmp_path):
    skill = tmp_path / "demo"
    _run(skill / "evals/results", 1)
    output = tmp_path / "history.csv"
    original_write_text = Path.write_text
    observed = {}

    def write_text(path, data, encoding=None, errors=None, newline=None):
        if path == output:
            observed["newline"] = newline
        return original_write_text(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", write_text)
    result = CliRunner().invoke(
        cli,
        ["tier3", "history", str(skill), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert observed["newline"] == ""


def test_history_sanitizes_terminal_controls_in_cells_and_reasons(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    _run(root, 1)
    _run(root, 2, model="model\x1b]8;;https://example.invalid\x1b\\")
    output = io.StringIO()

    print_history(collect_history(skill), Console(file=output, width=300))

    assert "\x1b" not in output.getvalue()
    assert "BREAK: segment" in output.getvalue()


def test_legacy_completed_run_has_legacy_status_not_missing_result_error(tmp_path):
    from test_results_location_legacy_compatibility import (
        _write_legacy_run_config,
        _write_successful_summary,
    )

    skill = tmp_path / "demo"
    legacy = skill / "evals/results/20260709_120000"
    legacy.mkdir(parents=True)
    _write_legacy_run_config(legacy, "opencode")
    _write_successful_summary(legacy)

    row = collect_history(skill)["rows"][0]

    assert row["status"] == "legacy"
    assert "legacy run: no run-level result.json; case coverage not recorded" in row["lift_reasons"]
    assert not any("No such file or directory" in reason for reason in row["lift_reasons"])


def test_new_run_reservation_keeps_previous_runs(tmp_path):
    original = _run(tmp_path, 1)
    before = (original / "result.json").read_bytes()
    new = _reserve_run_dir(tmp_path, "20260101_020000")
    assert new.is_dir() and original.is_dir()
    assert (original / "result.json").read_bytes() == before


def test_available_trend_skips_unavailable_rows_in_same_segment(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    _run(root, 1)
    _run(root, 2, status="failed")
    _run(root, 3, score=0.9)
    rows = collect_history(skill)["rows"]
    assert rows[1]["lift_delta"] is None
    assert rows[2]["lift_change"] == 0.1
    assert all(row["segment"] == 1 for row in rows)


def test_partial_failed_run_still_exposes_known_configuration_change(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    _run(root, 1)
    failed = _run(root, 2, status="failed")
    result = json.loads((failed / "result.json").read_text())
    for key in ("comparison_identity", "dataset_digest", "expected_case_ids", "attempt_policy", "score_policy"):
        result.pop(key)
    result["agents"] = {}
    _write(failed / "result.json", result)
    config = json.loads((failed / "run_config.json").read_text())
    config["harbor"]["n_attempts"] = 2
    _write(failed / "run_config.json", config)
    rows = collect_history(skill)["rows"]
    assert rows[1]["config_changed"] and rows[1]["segment"] == 2
    assert "n_attempts" in str(rows[1]["comparison_mismatches"])
    assert "dataset changed" not in str(rows[1]["comparison_mismatches"])
    assert rows[1]["lift_delta"] is None


def test_per_agent_policy_is_recovered_when_global_policy_is_unknown(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    for hour, policy in ((1, "policy-a"), (2, "policy-b")):
        directory = _run(root, hour, policy=policy)

        def update(r):
            r["score_policy"] = None
            r["comparison_identity"]["score_policy"] = None

        _update(directory, update)
    rows = collect_history(skill)["rows"]
    assert rows[1]["config_changed"] and rows[1]["score_policy"] == "policy-b"
    assert "score policy changed" in str(rows[1]["comparison_mismatches"])


def test_persisted_comparison_config_remains_authoritative(tmp_path):
    skill = tmp_path / "demo"
    root = skill / "evals/results"
    _run(root, 1)
    later = _run(root, 2)
    config = json.loads((later / "run_config.json").read_text())
    config["report_metadata"] = {"source_run_id": "another-run"}
    _write(later / "run_config.json", config)
    rows = collect_history(skill)["rows"]
    assert rows[1]["config_changed"] is False
    assert rows[1]["lift_change"] == 0.0


def test_table_keeps_full_model_and_dataset_case_count(tmp_path):
    skill = tmp_path / "demo"
    model = "provider/long-model-name-with-distinct-version"
    _run(skill / "evals/results", 1, model=model)
    output = io.StringIO()
    print_history(collect_history(skill), Console(file=output, width=300))
    assert model in output.getvalue()
    assert "sha256:cases (1)" in output.getvalue()


def test_empty_history_and_invalid_export(tmp_path):
    skill = tmp_path / "demo"
    skill.mkdir()
    result = CliRunner().invoke(cli, ["tier3", "history", str(skill), "--json"])
    assert result.exit_code == 0 and json.loads(result.output)["rows"] == []
    invalid = CliRunner().invoke(cli, ["tier3", "history", str(skill), "--output", str(tmp_path / "bad.txt")])
    assert invalid.exit_code != 0 and "must end in" in invalid.output


def test_corrupt_config_and_linked_artifacts_do_not_escape(tmp_path):
    skill = tmp_path / "demo"
    d = _run(skill / "evals/results", 1)
    (d / "run_config.json").unlink()
    try:
        (d / "run_config.json").symlink_to(d / "result.json")
    except OSError:
        pytest.skip("symlinks unavailable")
    row = collect_history(skill)["rows"][0]
    assert row["lift_delta"] is None and "run_config.json" in str(row["lift_reasons"])
