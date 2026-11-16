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


class EvidenceMarkup:
    """Inspect disclosure ownership and summaries without a browser dependency."""

    def __init__(self, fragment):
        from html.parser import HTMLParser

        self.details = {}
        self.summaries = {}
        self.sources = 0
        self.text = []
        outer = self

        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.stack = []
                self.summary = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "details":
                    node_id = attrs.get("id")
                    outer.details[node_id] = {"attrs": attrs, "parents": list(self.stack)}
                    self.stack.append(node_id)
                elif tag == "summary":
                    self.summary = self.stack[-1]
                    outer.summaries[self.summary] = ""
                if "plugin-evidence-source" in attrs.get("class", "").split():
                    outer.sources += 1

            def handle_endtag(self, tag):
                if tag == "details":
                    self.stack.pop()
                elif tag == "summary":
                    self.summary = None

            def handle_data(self, text):
                outer.text.append(text)
                if self.summary is not None:
                    outer.summaries[self.summary] += text

        Parser().feed(fragment)


def evidence_fragment(evidence, slot):
    from skillevaluator.reporting import HTMLReporter

    return (
        HTMLReporter(include_timestamp=False)
        ._env.from_string(
            "{% from 'plugin_evidence.html.j2' import render_plugin_evidence %}"
            "{{ render_plugin_evidence(evidence, slot) }}"
        )
        .render(evidence=evidence, slot=slot)
    )


def test_passive_sections_close_and_group_neutral_empty_observations_with_stable_anchors():
    evidence = api().build_plugin_evidence(runtime=runtime_report())
    fragment = evidence_fragment(evidence, "agents")
    markup = EvidenceMarkup(fragment)
    assert all("open" not in detail["attrs"] for detail in markup.details.values())
    group = "plugin-runtime-empty-agents"
    assert group in markup.details
    assert "4 empty observations" in markup.summaries[group]
    for key in ("loading", "coverage", "mcp", "mcp_calls"):
        assert group in markup.details[f"plugin-runtime-{key}"]["parents"]
    # Method/scope notes are not classified as a neutral absence.
    assert group not in markup.details["plugin-runtime-signals"]["parents"]
    assert "Skipped expectations concern components unsupported in that arm." in fragment
    assert fragment.count("No retained observations for this section.") == 4


@pytest.mark.parametrize("owner,slot", [("static", "tier1"), ("runtime", "agents")])
def test_each_slot_has_one_compact_source_qualification_without_repeated_run_metadata(owner, slot):
    evidence = api().build_plugin_evidence(**{owner: static_report() if owner == "static" else runtime_report()})
    fragment = evidence_fragment(evidence, slot)
    markup = EvidenceMarkup(fragment)
    assert markup.sources == 1
    assert "passive-demo" in fragment
    if owner == "static":
        assert "Static snapshot" in fragment
        assert "runtime loading or behavior" in fragment
    else:
        assert "Evidence incomplete" in fragment
        assert "execution failed" in fragment
        assert "advisory" in fragment
        assert "retained-run" not in fragment
    assert "<p>Source:" not in fragment


def test_closed_summary_keeps_row_omission_and_display_shortening_distinct_and_notes_reachable():
    raw = static_report().payload()
    raw["truncated"] = True
    section = raw["sections"][0]
    section["rows"][0][1] = "A" * 450
    section["notes"] += ["Recorded scope limitation remains associated with inventory."]
    section["total"], section["omitted"] = 3, 2
    original = deepcopy(raw)
    evidence = api().build_plugin_evidence(static=raw)
    fragment = evidence_fragment(evidence, "tier1")
    markup = EvidenceMarkup(fragment)
    summary = markup.summaries["plugin-static-inventory"]
    assert "2 rows omitted" in summary
    assert "text shortened" in summary
    assert "Recorded scope limitation remains associated with inventory." in fragment
    assert "shortened before retention cannot be recovered" in fragment
    assert evidence["static"] == original == raw


def test_runtime_only_html_exports_exact_snapshot_and_retains_all_grouped_anchors():
    import re

    from skillevaluator.reporting import HTMLReporter

    raw = runtime_report().payload()
    original = deepcopy(raw)
    document = HTMLReporter(include_timestamp=False).render_all(
        [ValidationResult(validator_name="Private checks")], plugin_runtime=raw
    )
    for section in raw["sections"]:
        assert document.count(f'id="plugin-runtime-{section["key"]}"') == 1
    encoded = re.search(r'<script\b[^>]*id="report-data"[^>]*>(.*?)</script>', document, flags=re.DOTALL)
    exported = json.loads(encoded[1])
    assert exported["plugin_runtime"] == original == raw
    assert exported.get("plugin_static") is None


def test_empty_observations_with_omissions_or_scope_notes_keep_individual_qualifications():
    raw = runtime_report().payload()
    for section in raw["sections"]:
        if section["key"] == "loading":
            section["total"], section["omitted"] = 2, 2
        elif section["key"] == "mcp":
            section["notes"] = ["Source rejected; loading cannot be established from these observations."]
    evidence = api().build_plugin_evidence(runtime=raw)
    fragment = evidence_fragment(evidence, "agents")
    markup = EvidenceMarkup(fragment)
    group = "plugin-runtime-empty-agents"
    assert "2 empty observations" in markup.summaries[group]
    for key in ("loading", "mcp"):
        assert group not in markup.details[f"plugin-runtime-{key}"]["parents"]
    assert "2 rows omitted" in markup.summaries["plugin-runtime-loading"]
    assert "Source rejected; loading cannot be established from these observations." in fragment
    assert evidence["runtime"] == raw


def test_shortened_recorded_notes_remain_complete_in_canonical_export():
    import re

    from skillevaluator.reporting import HTMLReporter

    raw = static_report().payload()
    note = "Recorded limitation " + "x" * 450
    raw["sections"][0]["notes"] = [note]
    document = HTMLReporter(include_timestamp=False).render_all(
        [ValidationResult(validator_name="Private checks")], plugin_static=raw
    )
    markup = EvidenceMarkup(document)
    assert "text shortened" in markup.summaries["plugin-static-inventory"]
    encoded = re.search(r'<script\b[^>]*id="report-data"[^>]*>(.*?)</script>', document, flags=re.DOTALL)
    exported = json.loads(encoded[1])
    assert exported["plugin_static"] == raw
    assert exported["plugin_static"]["sections"][0]["notes"] == [note]


def test_footer_metadata_is_sanitized_independently_of_sections_and_canonical_snapshots():
    static, runtime = static_report().payload(), runtime_report().payload()
    static["sections"], runtime["sections"] = [], []
    static["name"] = "\x1b[31mStatic\x1b[0m\x00 \ud800\u202e"
    runtime["name"] = "Bearer syntheticcredential123"
    runtime["execution_status"] = "\x1b[32msucceeded\x1b[0m\x07"
    runtime["run_id"] = "run-\udfff\u202e\x00"
    runtime["source"] = {
        "manifest": "password=syntheticsecret",
        "manifest_sha256": "\x1b[31mdigest\x1b[0m\x07",
        "source_tree_sha256": "s" * 450,
    }
    original = deepcopy((static, runtime))
    evidence = api().build_plugin_evidence(static=static, runtime=runtime)
    metadata = evidence["metadata"]
    assert metadata["static"] == {
        "name": "Static \ufffd\\u202e",
        "status": static["status"],
        "truncated": static["truncated"],
        "text_shortened": False,
        "notice": None,
    }
    assert metadata["runtime"] == {
        "name": "Bearer <redacted>",
        "status": "incomplete",
        "truncated": runtime["truncated"],
        "execution_status": "succeeded",
        "run_id": "run-\ufffd\\u202e",
        "source": {
            "manifest": "password=<redacted>",
            "manifest_sha256": "digest",
            "source_tree_sha256": "s" * 299 + "…",
        },
        "text_shortened": True,
        "notice": "Some metadata text was shortened for display; the export preserves the retained snapshot.",
    }
    assert evidence["static"] == original[0] == static
    assert evidence["runtime"] == original[1] == runtime
    metadata["runtime"]["source"]["manifest"] = "changed display"
    assert evidence["runtime"] == original[1] == runtime


@pytest.mark.parametrize("owner", ["static", "runtime"])
def test_footer_metadata_uses_none_for_absent_owners_and_marks_only_display_shortening(owner):
    raw = (static_report() if owner == "static" else runtime_report()).payload()
    raw["sections"] = []
    raw["name"] = "bounded-name"
    raw["truncated"] = True
    evidence = api().build_plugin_evidence(**{owner: raw})
    other = "runtime" if owner == "static" else "static"
    assert evidence["metadata"][other] is None
    assert evidence["metadata"][owner]["truncated"] is True
    assert evidence["metadata"][owner]["text_shortened"] is False
    assert evidence["metadata"][owner]["notice"] is None
    assert evidence[owner] == raw
    assert api().build_plugin_evidence()["metadata"] == {"static": None, "runtime": None}
