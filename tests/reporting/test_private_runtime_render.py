# SPDX-License-Identifier: Apache-2.0
"""Private runtime adapters preserve shared evidence and inert display text."""

from __future__ import annotations

import html
import importlib
import json
import subprocess
import sys
from copy import deepcopy

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
from skillevaluator.reporting.plugin_static_render import render_plugin_static_reports


class RetainedReport:
    """Exercise rendering before the independent projection is available."""

    def __init__(self, payload: dict) -> None:
        self.serialized = json.dumps(payload)
        self.reads = 0

    def payload(self) -> dict:
        self.reads += 1
        return json.loads(self.serialized)


def renderers():
    return importlib.import_module("skillevaluator.reporting.plugin_runtime_render")


def reporters():
    return [
        CLIReporter(),
        JSONReporter(include_timestamp=False),
        MarkdownReporter(include_timestamp=False),
        HTMLReporter(include_timestamp=False),
        SARIFReporter(include_timestamp=False),
        BenchmarkReporter(include_timestamp=False),
    ]


def payload():
    return {
        "schema_version": 1,
        "name": "runtime-demo",
        "status": "incomplete",
        "run_id": "retained-run-7",
        "execution_status": "succeeded",
        "evidence_complete": False,
        "source": {"manifest_sha256": "a" * 64, "source_tree_sha256": "b" * 64},
        "truncated": True,
        "sections": [
            {
                "key": "coverage",
                "title": "Activation coverage",
                "columns": ["Agent", "Arm", "Component", "State", "Calls", "Cost"],
                "rows": [["claude-code", "with_skill", "docs-server", "unverified", 0, None]],
                "total": 3,
                "omitted": 2,
                "notes": ["One trajectory is missing; costs were not recorded."],
            },
            {
                "key": "comparison",
                "title": "Comparison uncertainty",
                "columns": ["Comparison", "Estimate", "Precision", "Reason"],
                "rows": [["Wrapper effect", 0.2, "insufficient", "Only one paired case"]],
                "total": 1,
                "omitted": 0,
                "notes": [],
            },
        ],
    }


def bundle(*, static=False):
    result = ValidationResult(validator_name="Private checks")
    if static:
        result.metadata["plugin"] = {
            "name": "static-demo",
            "component_inventory": {
                "complete": True,
                "components": [{"type": "skill", "name": "member", "path": "skills/member"}],
            },
        }
    return [result]


def test_all_six_formats_compose_static_and_runtime_without_losing_shared_facts():
    report = RetainedReport(payload())
    output = renderers().render_plugin_runtime_reports(bundle(static=True), report=report, reporters=iter(reporters()))

    assert set(output) == {"cli", "json", "markdown", "html", "sarif", "benchmark"}
    expected = payload()
    assert json.loads(output["json"])["plugin_runtime"] == expected
    assert json.loads(output["sarif"])["runs"][0]["properties"]["pluginRuntime"] == expected
    assert json.loads(output["json"])["plugin_static"]["name"] == "static-demo"
    for name in ("cli", "markdown", "html", "benchmark"):
        document = html.unescape(output[name]).replace("\\_", "_")
        for marker in (
            "Plugin runtime evidence",
            "runtime-demo",
            "retained-run-7",
            "Execution status: succeeded",
            "Evidence status: incomplete",
            "Advisory observations do not change scores or validation gates.",
            "docs-server",
            "unverified",
            "Wrapper effect",
            "Only one paired case",
            "2 more rows omitted (3 total).",
            "static-demo",
        ):
            assert marker in document, (name, marker)
    assert 'id="plugin-runtime-evidence"' in output["html"]
    assert 'class="contributor-table plugin-runtime-table"' in output["html"]
    assert output["html"].count('id="plugin-static-evidence"') == 1
    assert output["html"].count('id="plugin-runtime-evidence"') == 1
    assert report.reads == 1


@pytest.mark.parametrize("static", [False, True])
def test_absent_runtime_report_keeps_existing_outputs_exactly(static):
    results = bundle(static=static)
    report_list = reporters()
    expected = render_plugin_static_reports(results, reporters=report_list)

    assert renderers().render_plugin_runtime_reports(results, report=None, reporters=iter(report_list)) == expected


@pytest.mark.parametrize("force_color", [False, True])
def test_runtime_markup_stays_inert_in_html_markdown_and_rich(monkeypatch, force_color):
    monkeypatch.setenv("FORCE_COLOR", "1" if force_color else "0")
    raw = payload()
    raw["name"] = '<script>alert("name")</script>'
    raw["run_id"] = '<img src=x onerror="alert(1)">'
    raw["sections"][0]["title"] = "[heading](javascript:alert(1))"
    raw["sections"][0]["columns"][0] = "[label](javascript:alert(1))"
    raw["sections"][0]["rows"][0][2] = "[click](javascript:alert(1)) | `code` [bold red]text[/]"
    raw["sections"][0]["notes"] = ['<iframe src="javascript:alert(1)"></iframe>']
    output = renderers().render_plugin_runtime_reports(bundle(), report=RetainedReport(raw), reporters=reporters())

    fragment = output["html"].split('id="plugin-runtime-evidence"', 1)[1].split("</main>", 1)[0]
    assert "<script>" not in fragment and "<img " not in fragment and "<iframe " not in fragment
    assert "&lt;script&gt;" in fragment and "&lt;img " in fragment and "&lt;iframe " in fragment
    for name in ("markdown", "benchmark"):
        runtime = output[name].split("## Plugin runtime evidence", 1)[1]
        assert "<script>" not in runtime and "<img " not in runtime and "<iframe " not in runtime
        assert "[click](javascript:alert(1))" not in runtime
        assert "[heading](javascript:alert(1))" not in runtime
        assert "[label](javascript:alert(1))" not in runtime
        assert "| `code`" not in runtime
    runtime_cli = output["cli"].split("Plugin runtime evidence", 1)[1]
    assert "[bold red]text[/]" in runtime_cli
    assert "\x1b[" not in runtime_cli


def test_unknown_and_zero_remain_distinct_and_rendering_does_not_mutate_inputs():
    results = bundle(static=True)
    before = deepcopy(results)
    raw = payload()
    report = RetainedReport(raw)
    encoded = report.serialized
    output = renderers().render_plugin_runtime_reports(results, report=report, reporters=reporters())

    assert results == before and report.serialized == encoded
    assert json.loads(output["json"])["plugin_runtime"]["sections"][0]["rows"][0][-2:] == [0, None]
    for name in ("cli", "markdown", "html", "benchmark"):
        assert "unknown" in output[name]
    assert "| 0 | unknown |" in output["markdown"]
    assert "<td>0</td><td>unknown</td>" in output["html"]


def test_runtime_rendering_does_not_collect_source_or_network_evidence(monkeypatch):
    import socket

    from skillevaluator.plugin_paths import PluginRootReader

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime report rendering cannot collect new evidence")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(PluginRootReader, "read_text", forbidden)

    output = renderers().render_plugin_runtime_reports(
        bundle(), report=RetainedReport(payload()), reporters=reporters()
    )
    assert json.loads(output["json"])["plugin_runtime"]["execution_status"] == "succeeded"


def test_runtime_adapter_import_and_render_need_no_optional_runtime_dependencies():
    script = """
import json, sys
import skillevaluator.reporting
from skillevaluator.models.result import ValidationResult
before = set(sys.modules)
from skillevaluator.reporting.plugin_runtime_render import render_plugin_runtime_reports
class Report:
    def payload(self):
        return {'schema_version': 1, 'name': 'offline', 'status': 'incomplete',
                'run_id': 'saved', 'execution_status': 'succeeded', 'evidence_complete': False,
                'source': {}, 'truncated': False, 'sections': []}
output = render_plugin_runtime_reports([ValidationResult(validator_name='retained')],
    report=Report(), reporters=[skillevaluator.reporting.JSONReporter(include_timestamp=False)])
assert json.loads(output['json'])['plugin_runtime']['name'] == 'offline'
new = set(sys.modules) - before
forbidden = ('litellm', 'anthropic', 'openai', 'harbor', 'skillevaluator.inference',
             'skillevaluator.tier3', 'skillevaluator.validators')
assert not sorted(name for name in new if name.startswith(forbidden)), sorted(new)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
