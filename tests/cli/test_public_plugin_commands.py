# SPDX-License-Identifier: Apache-2.0
"""Public plugin commands share runtime options and safely replay saved views."""

import json

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.reporting.plugin_runtime import project_plugin_runtime
from skillevaluator.reporting.plugin_runtime_io import save_plugin_runtime_report


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
    payload = json.loads((output / "plugin-report.json").read_text())
    assert payload["plugin_runtime"] == report.payload()


def test_replay_missing_or_invalid_snapshot_fails_before_writing(tmp_path):
    output = tmp_path / "reports"
    for payload in (None, '{"schema_version": 100}'):
        if payload is not None:
            (tmp_path / "plugin_runtime_report.json").write_text(payload)
        result = CliRunner().invoke(cli, ["tier3", "report-plugin", str(tmp_path), "-r", "json", "-o", str(output)])
        assert result.exit_code != 0
        assert "unavailable" in result.output.lower()
        assert not output.exists()


@pytest.mark.parametrize("report_format", ["json", "benchmark"])
def test_live_report_destination_is_rejected_before_execution(tmp_path, monkeypatch, report_format):
    from skillevaluator.evaluation.service import EvaluationService

    manifest = tmp_path / ".claude-plugin/plugin.json"
    manifest.parent.mkdir()
    manifest.write_text('{"name":"public-demo"}')
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
