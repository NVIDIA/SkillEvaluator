# SPDX-License-Identifier: Apache-2.0
"""Public plugin commands share runtime options and safely replay saved views."""

import io
import json
from pathlib import Path

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
    card = (output / "BENCHMARK.md").read_text(encoding="utf-8")
    assert card.startswith("# Plugin Benchmark: replayed-public\n")
    assert "- Skill: `replayed-public`" in card


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


@pytest.mark.parametrize("manifest_input", [False, True], ids=["root", "manifest"])
@pytest.mark.parametrize("unsafe", ["config-link", "evals-link", "root-link", "oversized", "malformed"])
def test_plugin_cli_preflight_refuses_unsafe_config_without_unbounded_reads(
    tmp_path, monkeypatch, manifest_input, unsafe
):
    from skillevaluator.evaluation.service import EvaluationService
    from skillevaluator.tier3.plugin_workflow import MAX_CONFIG_BYTES

    source = _live_report_source(tmp_path)
    evals = source / "evals"
    evals.mkdir()
    config = evals / "config.yml"
    secret = "LOCAL_SECRET_DO_NOT_PRINT"
    malformed = f"schema_version: [{secret}\n"
    if unsafe == "config-link":
        outside = tmp_path / "outside-private.yml"
        outside.write_text(malformed, encoding="utf-8")
        config.symlink_to(outside)
    elif unsafe == "evals-link":
        outside = tmp_path / "outside-evals"
        evals.rmdir()
        outside.mkdir()
        (outside / "config.yml").write_text(malformed, encoding="utf-8")
        evals.symlink_to(outside, target_is_directory=True)
    elif unsafe == "root-link":
        config.write_text(malformed, encoding="utf-8")
        linked = tmp_path / "linked-plugin"
        linked.symlink_to(source, target_is_directory=True)
        source = linked
    elif unsafe == "oversized":
        config.write_text("schema_version: 1\n#" + "x" * MAX_CONFIG_BYTES, encoding="utf-8")
    else:
        config.write_text(malformed, encoding="utf-8")

    unbounded_reads = []
    read_text = Path.read_text

    def observe_read(path, *args, **kwargs):
        if path.name in {"config.yml", "config.yaml"}:
            unbounded_reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observe_read)
    monkeypatch.setattr(
        EvaluationService, "evaluate_plugin", lambda *_a, **_kw: pytest.fail("Unsafe config reached execution")
    )
    target = source / ".claude-plugin/plugin.json" if manifest_input else source
    result = CliRunner().invoke(
        cli, ["tier3", "evaluate-plugin", str(target), "--agents", "opencode", "--progress", "off"]
    )

    assert result.exit_code == 2, (result.output, result.exception)
    assert "bounded plugin eval configuration" in result.output
    assert secret not in result.output
    assert unbounded_reads == []


@pytest.mark.parametrize("manifest_input", [False, True], ids=["root", "manifest"])
def test_plugin_cli_preflight_enforces_authored_cost_requirement(tmp_path, monkeypatch, manifest_input):
    from skillevaluator.evaluation.service import EvaluationService

    source = _live_report_source(tmp_path)
    (source / "evals").mkdir()
    (source / "evals/config.yml").write_text("schema_version: 1\nharbor:\n  max_cost_usd: 0.5\n", encoding="utf-8")
    monkeypatch.setattr(
        EvaluationService, "evaluate_plugin", lambda *_a, **_kw: pytest.fail("Invalid budget reached execution")
    )
    target = source / ".claude-plugin/plugin.json" if manifest_input else source
    result = CliRunner().invoke(
        cli, ["tier3", "evaluate-plugin", str(target), "--agents", "opencode", "--progress", "off"]
    )

    assert result.exit_code == 2, (result.output, result.exception)
    assert "max_cost_usd requires harbor.price_file" in result.output


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


def _combined_plugin_result(source, engine_result, results, monkeypatch):
    from skillevaluator.cli import _run_agent_eval_or_skip
    from skillevaluator.evaluation import PluginOptions, tier3_report
    from skillevaluator.evaluation.service import EvaluationService

    report = project_plugin_runtime(engine_result, provenance={"plugin": "demo", "runtime_evidence": {}})
    engine_result["plugin_runtime_report"] = report.payload()

    def execute(*_args, **_kwargs):
        # Another execution finishes before combined validation normalizes this one.
        publish_latest_results(results / "demo", "20261110_010002")
        return engine_result

    normalize = tier3_report.agent_eval_result_from_directory

    def without_judge(*args, **kwargs):
        # Keep artifact selection and normalization real; no optional provider calls.
        kwargs["use_llm_judge"] = False
        return normalize(*args, **kwargs)

    monkeypatch.setattr(EvaluationService, "evaluate_plugin", execute)
    monkeypatch.setattr(tier3_report, "agent_eval_result_from_directory", without_judge)
    return _run_agent_eval_or_skip(
        source,
        agents="opencode",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=None,
        results_dir=results,
        validate_source=False,
        plugin_options=PluginOptions(),
    )


@pytest.mark.parametrize("partial", [False, True], ids=["succeeded", "partial-failure"])
def test_combined_plugin_report_uses_returned_run_when_latest_moves(tmp_path, monkeypatch, partial):
    source = _live_report_source(tmp_path)
    results = tmp_path / "results"
    current = _scored_run(results, "20261110_010001", 0.2)
    _scored_run(results, "20261110_010002", 0.9)
    engine = {
        "execution_status": "failed" if partial else "succeeded",
        "agents": {"opencode": {"with_skill": {}}},
        "run_id": current.name,
        "run_dir": str(current),
    }
    if partial:
        engine["execution_errors"] = ["Another agent crashed", "Another agent crashed"]

    result = _combined_plugin_result(source, engine, results, monkeypatch)

    assert result.metadata["agent_eval"]["run_id"] == "20261110_010001"
    assert result.metadata["agent_eval"]["overall_score"] == pytest.approx(0.2)
    assert result.metadata["plugin_runtime_report"]["run_id"] == "20261110_010001"
    if partial:
        assert result.passed is False
        assert result.errors == ["Another agent crashed"]
        assert result.metadata["agent_eval"]["execution_status"] != "skipped"


@pytest.mark.parametrize("returned_run", [None, "missing", "empty", "malformed"])
def test_combined_plugin_report_without_exact_scores_cannot_borrow_latest(tmp_path, monkeypatch, returned_run):
    source = _live_report_source(tmp_path)
    results = tmp_path / "results"
    _scored_run(results, "20261110_010002", 0.9)
    engine = {"execution_status": "succeeded"}
    if returned_run is not None:
        run_dir = results / "demo" / "20261110_010001"
        engine.update(run_id=run_dir.name, run_dir=str(run_dir))
        if returned_run == "empty":
            run_dir.mkdir()
        elif returned_run == "malformed":
            run_dir = _scored_run(results, run_dir.name, 0.2)
            (run_dir / "opencode/with-skill/summary.json").write_text("{broken", encoding="utf-8")

    result = _combined_plugin_result(source, engine, results, monkeypatch)

    assert result.passed is False
    assert result.metadata["agent_eval"]["execution_status"] == "skipped"
    assert result.metadata["agent_eval"].get("overall_score") is None
    assert result.metadata["agent_eval"].get("run_id") != "20261110_010002"
    assert result.metadata["plugin_runtime_report"] == engine["plugin_runtime_report"]


def _provenance(complete):
    reasons = [] if complete else ["rule reference 'github::other/lib::rules::style.md' was not staged (external)"]
    return {"plugin": "demo", "complete": complete, "incomplete_reasons": reasons}


@pytest.mark.parametrize("complete", [True, False], ids=["complete", "incomplete"])
def test_combined_plugin_result_is_incomplete_when_the_package_is(tmp_path, monkeypatch, complete):
    source = _live_report_source(tmp_path)
    results = tmp_path / "results"
    current = _scored_run(results, "20261110_010001", 0.2)
    engine = {
        "execution_status": "succeeded",
        "agents": {"opencode": {"with_skill": {}}},
        "run_id": current.name,
        "run_dir": str(current),
        "plugin_provenance": _provenance(complete),
    }

    result = _combined_plugin_result(source, engine, results, monkeypatch)

    # Scores stay visible; an incomplete package never counts as a complete plugin evaluation.
    assert result.metadata["agent_eval"]["overall_score"] == pytest.approx(0.2)
    assert ("plugin-components" in result.incomplete_scans) is not complete


@pytest.mark.parametrize("complete", [True, False], ids=["complete", "incomplete"])
def test_live_reports_are_incomplete_when_the_package_is(tmp_path, complete):
    source = _live_report_source(tmp_path)
    results = tmp_path / "results"
    current = _scored_run(results, "20261110_010001", 0.5)
    engine_result = {
        "execution_status": "succeeded",
        "run_id": current.name,
        "run_dir": str(current),
        "plugin_provenance": _provenance(complete),
    }

    payload = _live_json_report(source, engine_result, results, tmp_path / "reports")

    assert payload["tier3"]["overall_score"] == pytest.approx(0.5)
    assert ("plugin-components" in payload["incomplete_scans"]) is not complete


@pytest.mark.parametrize("scored", [True, False], ids=["scored", "no-scores"])
def test_live_benchmark_card_names_the_plugin(tmp_path, scored):
    source = _live_report_source(tmp_path)
    results = tmp_path / "results"
    engine_result = {"execution_status": "failed", "error": "Fixture setup failed"}
    if scored:
        run_dir = _scored_run(results, "20261110_010001", 0.5)
        engine_result = {"execution_status": "succeeded", "run_id": run_dir.name, "run_dir": str(run_dir)}
    report = project_plugin_runtime(engine_result, provenance={"plugin": "demo", "runtime_evidence": {}})
    engine_result["plugin_runtime_report"] = report.payload()
    output = tmp_path / "reports"
    emit_plugin_run_reports(
        source,
        engine_result,
        report_formats=("benchmark",),
        output_dir=output,
        results_dir=results,
        env_mode="docker",
        console=Console(file=io.StringIO()),
    )
    # Without a score summary the card has no other source for the name and falls back to "skill".
    card = (output / "BENCHMARK.md").read_text(encoding="utf-8")
    assert card.startswith("# Plugin Benchmark: demo\n")
    assert "- Skill: `demo`" in card


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
