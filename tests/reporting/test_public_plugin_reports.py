# SPDX-License-Identifier: Apache-2.0
"""Public report hooks share the bounded plugin views with saved-run replay."""

import json

import pytest

from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting import (
    BenchmarkReporter,
    CLIReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    SARIFReporter,
)
from skillevaluator.reporting.plugin_runtime import project_plugin_runtime


@pytest.fixture
def plugin_result():
    result = ValidationResult(validator_name="Plugin checks")
    result.metadata["plugin"] = {
        "name": "public-demo",
        "component_inventory": {
            "complete": True,
            "components": [{"type": "skill", "name": "member", "path": "skills/member"}],
        },
    }
    return result


@pytest.mark.parametrize(
    "reporter_class", [CLIReporter, JSONReporter, HTMLReporter, MarkdownReporter, SARIFReporter, BenchmarkReporter]
)
def test_public_report_hook_and_saved_files_include_static_evidence(tmp_path, plugin_result, reporter_class):
    reporter = reporter_class()
    output = reporter.render_report([plugin_result])
    if reporter.name == "json":
        assert json.loads(output)["plugin_static"]["name"] == "public-demo"
    elif reporter.name == "sarif":
        assert json.loads(output)["runs"][0]["properties"]["pluginStatic"]["name"] == "public-demo"
    else:
        assert "public-demo" in output
        assert "Plugin static evidence" in output
    target = tmp_path / ("report" + reporter.get_file_extension())
    reporter.save([plugin_result], target)
    assert "public-demo" in target.read_text(encoding="utf-8")


def test_runtime_snapshot_is_validated_without_mutating_metadata(plugin_result):
    report = project_plugin_runtime(
        {"run_dir": "/retained-public-run", "execution_status": "succeeded"},
        provenance={"plugin": "public-demo", "runtime_evidence": {}},
    )
    assert report is not None
    snapshot = report.payload()
    plugin_result.metadata["plugin_runtime_report"] = snapshot
    before = json.dumps(snapshot, sort_keys=True)
    output = json.loads(JSONReporter().render_report([plugin_result]))
    assert output["plugin_runtime"]["run_id"] == "retained-public-run"
    assert json.dumps(snapshot, sort_keys=True) == before


def test_ordinary_reports_keep_the_existing_rendering_exactly():
    results = [ValidationResult(validator_name="Skill checks")]
    for reporter in (JSONReporter(include_timestamp=False), MarkdownReporter(include_timestamp=False)):
        assert reporter.render_report(results) == reporter.render_all(results)


def test_malformed_runtime_snapshot_cannot_be_rendered_as_valid_evidence(plugin_result):
    plugin_result.metadata["plugin_runtime_report"] = {"schema_version": 900}
    with pytest.raises(ValueError, match="schema"):
        JSONReporter().render_report([plugin_result])
