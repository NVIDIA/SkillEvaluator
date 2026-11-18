# SPDX-License-Identifier: Apache-2.0
"""The public passive hook composes retained evidence before rendering HTML."""

import json
import re
from copy import deepcopy

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter, JUnitReporter
from skillevaluator.reporting.plugin_runtime import project_plugin_runtime
from skillevaluator.reporting.plugin_static import project_plugin_static


class RecordingHTMLReporter(HTMLReporter):
    def __init__(self):
        super().__init__(include_timestamp=False, content_label="Plugin")
        self.calls = []

    def render_all(self, results, *, plugin_static=None, plugin_runtime=None):
        self.calls.append((results, plugin_static, plugin_runtime))
        return super().render_all(results, plugin_static=plugin_static, plugin_runtime=plugin_runtime)


@pytest.mark.parametrize("entrypoint", ["render_plugin_report", "augment_plugin_report"])
@pytest.mark.parametrize("with_static", [False, True], ids=["runtime", "static-and-runtime"])
def test_retained_runtime_html_is_rendered_once_with_both_snapshots(entrypoint, with_static, monkeypatch):
    from skillevaluator.reporting import plugin_reports, plugin_runtime_io

    # Rendering a retained result must not try to recover the absent source/run.
    def unavailable(*args, **kwargs):
        raise AssertionError("HTML composition must use the retained snapshot")

    monkeypatch.setattr(plugin_runtime_io, "load_plugin_runtime_report", unavailable)
    runtime = project_plugin_runtime(
        {"run_dir": "/absent/saved-run", "execution_status": "succeeded"},
        provenance={"plugin": '<img src=x onerror="alert(1)">', "runtime_evidence": {}},
    ).payload()
    metadata = {"plugin_runtime_report": runtime}
    if with_static:
        metadata["plugin"] = {
            "name": "retained-plugin",
            "component_inventory": {"complete": True, "components": []},
        }
    results = [ValidationResult(validator_name="SCHEMA", metadata=metadata)]
    before = deepcopy(results[0].metadata)
    static = project_plugin_static(results)
    reporter = RecordingHTMLReporter()
    if entrypoint == "render_plugin_report":
        document = plugin_reports.render_plugin_report(reporter, results)
    else:
        document = plugin_reports.augment_plugin_report(reporter, "previously rendered document", results)

    assert len(reporter.calls) == 1
    assert reporter.calls[0][0] is results
    assert reporter.calls[0][1] == (static.payload() if static is not None else None)
    assert reporter.calls[0][2] == runtime
    encoded = re.search(r'<script\b[^>]*id="report-data"[^>]*>(.*?)</script>', document, flags=re.DOTALL)
    assert encoded
    exported = json.loads(encoded[1])
    assert exported["plugin_runtime"] == runtime
    if with_static:
        assert exported["plugin_static"]["name"] == "retained-plugin"
        assert document.count('id="plugin-static-inventory"') == 1
    else:
        assert "plugin_static" not in exported
    assert document.count('id="plugin-runtime-loading"') == 1
    assert "Plugin evaluation" in document
    assert '<img src=x onerror="alert(1)">' not in document
    assert results[0].metadata == before


@pytest.mark.parametrize("entrypoint", ["render_plugin_report", "augment_plugin_report"])
def test_invalid_retained_runtime_is_rejected_before_html_rendering(entrypoint):
    from skillevaluator.reporting import plugin_reports

    results = [ValidationResult(validator_name="SCHEMA", metadata={"plugin_runtime_report": {"schema_version": 99}})]
    reporter = RecordingHTMLReporter()
    with pytest.raises(ValueError, match="runtime report schema"):
        if entrypoint == "render_plugin_report":
            plugin_reports.render_plugin_report(reporter, results)
        else:
            plugin_reports.augment_plugin_report(reporter, "previously rendered document", results)
    assert reporter.calls == []


@pytest.mark.parametrize("with_runtime", [False, True], ids=["static", "static-and-runtime"])
def test_junit_keeps_plugin_evidence_out_of_the_xml(with_runtime):
    from skillevaluator.reporting import plugin_reports

    metadata = {"plugin": {"name": "retained-plugin", "component_inventory": {"complete": True, "components": []}}}
    if with_runtime:
        metadata["plugin_runtime_report"] = project_plugin_runtime(
            {"run_dir": "/absent/saved-run", "execution_status": "succeeded"},
            provenance={"plugin": "retained-plugin", "runtime_evidence": {}},
        ).payload()
    results = [ValidationResult(validator_name="SCHEMA", metadata=metadata)]
    assert project_plugin_static(results) is not None
    reporter = JUnitReporter()
    plain = reporter.render_all(results)

    # Neither the static nor the runtime adapter adds to JUnit, and neither raises.
    assert plugin_reports.augment_plugin_report(reporter, None, results) == plain
    assert plugin_reports.augment_plugin_report(reporter, plain, results) == plain
    assert plugin_reports.render_plugin_report(reporter, results) == plain
