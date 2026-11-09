# SPDX-License-Identifier: Apache-2.0
"""Public plugin commands share runtime options and safely replay saved views."""

import io
import json

import pytest
from click.testing import CliRunner
from rich.console import Console

from skillevaluator.cli import cli
from skillevaluator.cli_plugin import emit_plugin_run_reports
from skillevaluator.reporting.plugin_runtime import project_plugin_runtime
from skillevaluator.reporting.plugin_runtime_io import save_plugin_runtime_report
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.results_location import publish_latest_results


def test_public_plugin_commands_expose_shared_runtime_options():
    runner = CliRunner()
    for command in (["validate"], ["tier3", "evaluate-plugin"]):
        result = runner.invoke(cli, [*command, "--help"])
        assert result.exit_code == 0, result.output
        for flag in ("--plugin-load", "--lift-mode", "--probe-mcp", "--probe-mcp-env", "--trial-retries"):
            assert flag in result.output


def test_saved_run_replays_six_formats_without_the_source(tmp_path, monkeypatch):
    run_dir = tmp_path / "saved-run"
    run_dir.mkdir()
    report = project_plugin_runtime(
        {"run_dir": str(run_dir), "execution_status": "succeeded"},
        provenance={"plugin": "replayed-public", "runtime_evidence": {}},
    )
    assert report is not None
    save_plugin_runtime_report(report, run_dir)
    from skillevaluator.evaluation.service import EvaluationService

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not invoke the execution service")

    monkeypatch.setattr(EvaluationService, "evaluate", forbidden)
    output = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        ["tier3", "report-plugin", str(run_dir), "-r", "cli,json,html,markdown,sarif,benchmark", "-o", str(output)],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "replayed-public" in result.output
    assert {path.name for path in output.iterdir()} == {
        "plugin-report.json",
        "plugin-report.html",
        "plugin-report.md",
        "plugin-report.sarif.json",
        "BENCHMARK.md",
    }
    payload = json.loads((output / "plugin-report.json").read_text(encoding="utf-8"))
    assert payload["plugin_runtime"] == report.payload()


def test_replay_missing_or_invalid_snapshot_fails_before_writing(tmp_path):
    output = tmp_path / "reports"
    for payload in (None, '{"schema_version": 100}'):
        if payload is not None:
            (tmp_path / "plugin_runtime_report.json").write_text(payload, encoding="utf-8")
        result = CliRunner().invoke(cli, ["tier3", "report-plugin", str(tmp_path), "-r", "json", "-o", str(output)])
        assert result.exit_code != 0
        assert "unavailable" in result.output.lower()
        assert not output.exists()


@pytest.mark.parametrize("report_format", ["json", "benchmark"])
def test_live_report_destination_is_rejected_before_execution(tmp_path, monkeypatch, report_format):
    from skillevaluator.evaluation.service import EvaluationService

    manifest = tmp_path / ".claude-plugin/plugin.json"
    manifest.parent.mkdir()
    manifest.write_text('{"name":"public-demo"}', encoding="utf-8")
    seen = []

    def engine(*args, **kwargs):
        seen.append((args, kwargs))
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate_plugin", engine)
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "evaluate-plugin",
            str(tmp_path),
            "--agents",
            "claude-code",
            "-r",
            report_format,
            "-o",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "outside" in result.output
    assert seen == []


def _live_report_source(tmp_path):
    source = tmp_path / "demo"
    manifest = source / ".claude-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"name":"demo"}', encoding="utf-8")
    return source


def _scored_run(results, run_id, score):
    run_dir = results / "demo" / run_id
    summary = run_dir / "opencode/with-skill/summary.json"
    summary.parent.mkdir(parents=True)
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id, "agents": {}}), encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(DEFAULT_METRICS, score),
                "metrics": list(DEFAULT_METRICS),
                "num_trials": 1,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _live_json_report(source, engine_result, results, output):
    report = project_plugin_runtime(engine_result, provenance={"plugin": "demo", "runtime_evidence": {}})
    assert report is not None
    engine_result["plugin_runtime_report"] = report.payload()
    emit_plugin_run_reports(
        source,
        engine_result,
        report_formats=("json",),
        output_dir=output,
        results_dir=results,
        env_mode="docker",
        console=Console(file=io.StringIO()),
    )
    return json.loads((output / "plugin-report.json").read_text(encoding="utf-8"))


def test_live_reports_use_returned_run_when_latest_moves(tmp_path):
    source = _live_report_source(tmp_path)
    results = tmp_path / "results"
    current = _scored_run(results, "20261110_010001", 0.2)
    _scored_run(results, "20261110_010002", 0.9)
    publish_latest_results(results / "demo", "20261110_010002")
    # Newer completed output represents another execution publishing latest
    # before the current command has emitted its requested report formats.
    payload = _live_json_report(
        source,
        {"execution_status": "succeeded", "run_id": current.name, "run_dir": str(current)},
        results,
        tmp_path / "reports",
    )
    assert payload["tier3"]["run_id"] == "20261110_010001"
    assert payload["tier3"]["overall_score"] == pytest.approx(0.2)
    assert payload["plugin_runtime"]["run_id"] == "20261110_010001"


@pytest.mark.parametrize(
    "execution_status,returned_run",
    [("failed", None), ("succeeded", None), ("failed", "missing"), ("failed", "empty")],
)
def test_live_reports_without_current_scores_do_not_reuse_previous_run(tmp_path, execution_status, returned_run):
    source = _live_report_source(tmp_path)
    results = tmp_path / "results"
    _scored_run(results, "20261110_010001", 0.9)
    engine_result = {"execution_status": execution_status}
    if execution_status == "failed":
        engine_result["error"] = "Fixture setup failed"
    if returned_run is not None:
        run_dir = results / "demo" / returned_run
        engine_result["run_dir"] = str(run_dir)
        if returned_run == "empty":
            run_dir.mkdir()
    payload = _live_json_report(source, engine_result, results, tmp_path / "reports")
    assert "tier3" not in payload
    assert payload["overall_passed"] is False
    assert "canonical-score-summary" in payload["incomplete_scans"]
    assert payload["plugin_runtime"]["execution_status"] == execution_status
