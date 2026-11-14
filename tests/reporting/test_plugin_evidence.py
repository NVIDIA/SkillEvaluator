# SPDX-License-Identifier: Apache-2.0
"""Passive plugin sections enter native tabs without changing retained facts."""

import importlib
import json
from copy import deepcopy

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import JSONReporter
from skillevaluator.reporting.plugin_runtime import project_plugin_runtime
from skillevaluator.reporting.plugin_static import project_plugin_static
from skillevaluator.reporting.retained import RetainedReportError, retained_results


def api():
    return importlib.import_module("skillevaluator.reporting.plugin_evidence")


def static_report():
    result = ValidationResult(
        validator_name="Private checks",
        metadata={
            "plugin": {
                "name": "passive-demo",
                "component_inventory": {
                    "complete": True,
                    "components": [{"type": "skill", "name": "member", "path": "skills/member"}],
                },
                "catalog_skill_similarity": {
                    "status": "compared",
                    "catalog_entries": 2,
                    "matches": [{"skill": "member", "match": "catalog-member", "similarity": 0.8}],
                },
                "context_cost": {"by_component": [], "by_harness": []},
            }
        },
    )
    return project_plugin_static([result])


def runtime_report():
    return project_plugin_runtime(
        {
            "run_dir": "/gone/retained-run",
            "execution_status": "failed",
            "runtime_evidence": {"evidence_complete": False},
        },
        provenance={"plugin": "passive-demo"},
    )


def test_sections_are_owned_by_native_tabs_and_snapshots_stay_exact():
    static, runtime = static_report(), runtime_report()
    raw_static, raw_runtime = static.payload(), runtime.payload()
    original = deepcopy((raw_static, raw_runtime))
    evidence = api().build_plugin_evidence(static=raw_static, runtime=raw_runtime)
    assert [section["key"] for section in evidence["tier1"]] == ["inventory", "context", "context_harnesses"]
    assert [section["key"] for section in evidence["tier2"]] == ["catalog_skills"]
    assert {section["key"] for section in evidence["overview"]} == {"eligibility", "conditions"}
    assert {section["key"] for section in evidence["agents"]} == {"loading", "coverage", "signals", "mcp", "mcp_calls"}
    assert {section["key"] for section in evidence["trials"]} == {"context", "comparisons", "cost"}
    assert {section["key"] for section in evidence["provenance"]} == {"run", "diagnostics", "exclusions"}
    assert evidence["static"] == raw_static and evidence["runtime"] == raw_runtime
    assert (raw_static, raw_runtime) == original
    sections = [
        section
        for slot in ("tier1", "tier2", "overview", "agents", "trials", "provenance")
        for section in evidence[slot]
    ]
    assert len(sections) == len(raw_static["sections"]) + len(raw_runtime["sections"])
    assert all(section["owner"] in {"static", "runtime"} for section in sections)
    evidence["static"]["sections"][0]["rows"].clear()
    evidence["tier1"][0]["rows"].clear()
    assert raw_static == original[0]
    assert api().build_plugin_evidence(static=static)["static"] == original[0]


@pytest.mark.parametrize("bad", [False, [], {"schema_version": 1}, {"schema_version": True}])
def test_malformed_payloads_are_rejected_without_guessing(bad):
    with pytest.raises(ValueError):
        api().build_plugin_evidence(static=bad)


@pytest.mark.parametrize("mutation", ["row_width", "nonfinite", "omission", "cycle", "oversized"])
def test_invalid_sections_are_rejected_before_rendering(mutation):
    raw = static_report().payload()
    section = raw["sections"][0]
    if mutation == "row_width":
        section["rows"][0].append("extra")
    elif mutation == "nonfinite":
        section["rows"][0][0] = float("nan")
    elif mutation == "omission":
        section["omitted"] = 30
    elif mutation == "cycle":
        section["rows"][0][0] = raw
    else:
        section["rows"][0][0] = "x" * 300000
    with pytest.raises(ValueError):
        api().build_plugin_evidence(static=raw)


@pytest.mark.parametrize("flag", [True, False])
def test_optional_skip_survives_json_retention_without_gate_changes(flag):
    result = ValidationResult(
        validator_name="Similarity Check",
        metadata={
            "gating": {"tier": 2, "blocking": False},
            "optional_skip": flag,
            "skipped": True,
            "skip_reason": "No catalog configured",
        },
    )
    reporter = JSONReporter(include_timestamp=False)
    raw = json.loads(reporter.render_all([result]))
    assert raw["results"][0]["optional_skip"] is flag
    restored = retained_results(raw)
    assert restored[0].metadata["optional_skip"] is flag
    assert json.loads(reporter.render_all(restored)) == raw


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_retained_optional_skip_requires_boolean(flag):
    raw = json.loads(JSONReporter(include_timestamp=False).render_all([ValidationResult(validator_name="SCHEMA")]))
    raw["results"][0]["optional_skip"] = flag
    with pytest.raises(RetainedReportError, match="optional_skip"):
        retained_results(raw)


def test_composition_supplies_both_snapshots_before_a_single_html_render():
    from skillevaluator.reporting import HTMLReporter
    from skillevaluator.reporting.plugin_runtime_render import render_plugin_runtime_reports

    class NativeBoundaryReporter(HTMLReporter):
        def __init__(self):
            super().__init__(include_timestamp=False)
            self.calls = []

        def render_all(self, results, *, plugin_static=None, plugin_runtime=None):
            self.calls.append((results, plugin_static, plugin_runtime))
            evidence = api().build_plugin_evidence(static=plugin_static, runtime=plugin_runtime)
            # A report need not have a closing main tag to receive its context.
            return json.dumps({"static": evidence["static"], "runtime": evidence["runtime"]})

    results = [
        ValidationResult(
            validator_name="Private checks",
            metadata={
                "plugin": {"name": "composed-static", "component_inventory": {"complete": True, "components": []}}
            },
        )
    ]
    reporter = NativeBoundaryReporter()
    runtime = runtime_report()
    output = render_plugin_runtime_reports(results, report=runtime, reporters=[reporter])
    decoded = json.loads(output["html"])
    assert decoded["static"]["name"] == "composed-static"
    assert decoded["runtime"] == runtime.payload()
    assert len(reporter.calls) == 1
    assert reporter.calls[0][0] is results


def test_native_macro_distinguishes_omitted_rows_from_shortened_text():
    from skillevaluator.reporting import HTMLReporter

    raw = static_report().payload()
    raw["truncated"] = True
    raw["sections"][0]["rows"][0][1] = "A" * 450
    raw["sections"][0]["total"] = 3
    raw["sections"][0]["omitted"] = 2
    before = deepcopy(raw)
    evidence = api().build_plugin_evidence(static=raw)
    env = HTMLReporter(include_timestamp=False)._env
    fragment = env.from_string(
        "{% from 'plugin_evidence.html.j2' import render_plugin_evidence %}"
        "{{ render_plugin_evidence(evidence, 'tier1') }}"
    ).render(evidence=evidence)
    assert "2 more rows omitted (3 total)" in fragment
    assert "Some text was shortened for display" in fragment
    assert "cannot be recovered" in fragment
    assert "A" * 450 not in fragment
    assert raw == before and evidence["static"] == before


def test_native_html_places_each_plugin_section_once_and_exports_exact_snapshots():
    import re
    from html.parser import HTMLParser

    from skillevaluator.reporting import HTMLReporter

    class Locations(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.parents = {}
            self.counts = {}

        def handle_starttag(self, tag, attrs):
            node_id = dict(attrs).get("id")
            if node_id:
                self.parents[node_id] = [item_id for _, item_id in self.stack if item_id]
                self.counts[node_id] = self.counts.get(node_id, 0) + 1
            if tag not in {
                "area",
                "base",
                "br",
                "col",
                "embed",
                "hr",
                "img",
                "input",
                "link",
                "meta",
                "param",
                "source",
                "track",
                "wbr",
            }:
                self.stack.append((tag, node_id))

        def handle_endtag(self, tag):
            for index in range(len(self.stack) - 1, -1, -1):
                if self.stack[index][0] == tag:
                    del self.stack[index:]
                    break

    static, runtime = static_report(), runtime_report()
    document = HTMLReporter(include_timestamp=False).render_all(
        [ValidationResult(validator_name="Private checks")], plugin_static=static, plugin_runtime=runtime
    )
    locations = Locations()
    locations.feed(document)
    for node_id, panel in (
        ("plugin-static-inventory", "tab-tier1"),
        ("plugin-static-catalog_skills", "tab-tier2"),
        ("plugin-static-context", "tab-tier1"),
        ("plugin-runtime-eligibility", "tier3-page-overview"),
        ("plugin-runtime-conditions", "tier3-page-overview"),
        ("plugin-runtime-loading", "tier3-page-agents"),
        ("plugin-runtime-comparisons", "tier3-page-trials"),
        ("plugin-runtime-diagnostics", "tier3-page-provenance"),
        ("plugin-runtime-exclusions", "tier3-page-provenance"),
    ):
        assert locations.counts.get(node_id) == 1, node_id
        assert panel in locations.parents[node_id], (node_id, locations.parents[node_id])
    encoded = re.search(r'<script\b[^>]*id="report-data"[^>]*>(.*?)</script>', document, flags=re.DOTALL)
    assert encoded
    exported = json.loads(encoded[1])
    assert exported["plugin_static"] == static.payload()
    assert exported["plugin_runtime"] == runtime.payload()
    assert exported["tier3"] is None


def test_static_only_sections_stay_visible_without_a_tier3_tab():
    from skillevaluator.reporting import HTMLReporter

    raw = static_report().payload()
    raw["sections"].append(
        {
            "key": "identity_conflicts",
            "title": "Source identity conflicts",
            "columns": ["Issue"],
            "rows": [["Retained source identities differ"]],
            "total": 1,
            "omitted": 0,
            "notes": [],
        }
    )
    evidence = api().build_plugin_evidence(static=raw)
    assert {section["key"] for section in evidence["tier1"]} == {
        "inventory",
        "context",
        "context_harnesses",
        "identity_conflicts",
    }
    assert not any(evidence[slot] for slot in ("overview", "agents", "trials", "provenance"))
    document = HTMLReporter(include_timestamp=False).render_all(
        [ValidationResult(validator_name="Private checks")], plugin_static=raw
    )
    assert 'id="tab-tier3"' not in document
    for section in raw["sections"]:
        assert document.count(f'id="plugin-static-{section["key"]}"') == 1
    assert "Retained source identities differ" in document
