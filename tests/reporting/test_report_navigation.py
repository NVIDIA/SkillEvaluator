# SPDX-License-Identifier: Apache-2.0
"""Navigation contracts on the actual rendered, standalone report."""

from html.parser import HTMLParser

import pytest
from browser_navigation_fixtures import render_fixtures

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting.html import HTMLReporter


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.stack = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        element = dict(attrs)
        element["tag"] = tag
        element["parents"] = tuple(self.stack)
        self.elements.append(element)
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
            self.stack.append(element)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break

    def by_id(self, name):
        return next(element for element in self.elements if element.get("id") == name)


def static_result():
    result = ValidationResult(validator_name="Schema Check")
    result.add_success("schema", "valid")
    return result


def agent_result():
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(
        "demo",
        {name: {"overall_with_skill": 0.8, "overall_without_skill": 0.4} for name in ['agent"[one]/雪', "agent.two"]},
    )
    return result


def test_displayed_absent_tiers_are_navigable_without_fabricated_checks():
    html = HTMLReporter(include_timestamp=False, requested_tiers=(1, 2), displayed_tiers=(1, 2, 3)).render_all(
        [static_result()]
    )
    doc = Document(html)
    tiles = [element for element in doc.elements if "tier-card" in element.get("class", "").split()]
    assert [tile["data-target-tab"] for tile in tiles] == ["tier1", "tier2", "tier3"]
    assert "Not run (not requested)" in html
    for tile in tiles:
        assert doc.by_id(tile["aria-controls"])["id"] == "tab-" + tile["data-target-tab"]
    assert "No check evidence was recorded for this tier." in html


def test_targeted_reports_keep_only_their_tier_navigation():
    doc = Document(HTMLReporter(include_timestamp=False).render_all([static_result()]))
    assert {element["data-target-tab"] for element in doc.elements if "data-target-tab" in element} == {"tier1"}


def test_evidence_destination_is_outside_hidden_tier_panels():
    doc = Document(HTMLReporter(include_timestamp=False).render_all([static_result(), agent_result()]))
    evidence = doc.by_id("report-evidence")
    assert not any("tab-content" in parent.get("class", "").split() for parent in evidence["parents"])
    assert sum(element.get("id") == "report-evidence" for element in doc.elements) == 1


def test_subtabs_and_unusual_agent_names_have_complete_relationships():
    html = HTMLReporter(include_timestamp=False).render_all([agent_result()])
    doc = Document(html)
    tabs = [element for element in doc.elements if element.get("role") == "tab"]
    assert len(tabs) == 8
    for tab in tabs:
        panel = doc.by_id(tab["aria-controls"])
        assert panel["role"] == "tabpanel"
        assert panel["aria-labelledby"] == tab["id"]
        assert tab["aria-selected"] in {"true", "false"}
        assert tab["tabindex"] == ("0" if tab["aria-selected"] == "true" else "-1")
    assert {tab["data-tier3-agent"] for tab in tabs if "data-tier3-agent" in tab} == {'agent"[one]/雪', "agent.two"}


def test_theme_control_includes_system_and_no_js_exposes_local_evidence():
    html = HTMLReporter(include_timestamp=False).render_all([agent_result()])
    doc = Document(html)
    assert doc.by_id("report-theme")["tag"] == "select"
    assert {element["value"] for element in doc.elements if element["tag"] == "option"} >= {"system", "light", "dark"}
    no_js = html.split("<noscript>", 1)[1].split("</noscript>", 1)[0]
    assert ".t3-page" in no_js and ".t3-agent-panel" in no_js


def test_overall_command_badge_is_prominent_and_agent_quality_is_named():
    source = static_result()
    source.add_error("Schema failed")
    html = HTMLReporter(include_timestamp=False).render_all([source, agent_result()])
    doc = Document(html)
    badge = next(element for element in doc.elements if "hero-command-status" in element.get("class", "").split())
    assert {"verdict-chip", "fail"} <= set(badge["class"].split())
    assert "Tier 3 quality: " in html.split('<div class="hero-side">', 1)[1].split("</div>", 1)[0]


def _gated(result, tier, blocking=True):
    result.metadata["gating"] = {"tier": tier, "blocking": blocking}
    return result


def _hero_reasons(html):
    import re

    return re.findall(r'<p class="hero-decision-reason">([^<]*)</p>', html)


def test_hero_reason_counts_advisory_failures_apart_from_the_required_result():
    passed = ValidationResult(validator_name="License Compliance")
    passed.add_success("license", "License found")
    advisory = ValidationResult(validator_name="Similarity Check")
    advisory.add_error("Similarity provider failed")
    html = HTMLReporter(include_timestamp=False).render_all([_gated(passed, 1), _gated(advisory, 2, blocking=False)])
    assert "Passed required checks" in html
    assert _hero_reasons(html) == [
        "1 advisory check(s) failed or incomplete; advisory checks do not change the required result."
    ]

    clean = ValidationResult(validator_name="License Compliance")
    clean.add_success("license", "License found")
    assert _hero_reasons(HTMLReporter(include_timestamp=False).render_all([_gated(clean, 1)])) == []

    required = ValidationResult(validator_name="Schema")
    required.add_error("Missing name")
    html = HTMLReporter(include_timestamp=False).render_all([_gated(required, 1)])
    assert _hero_reasons(html) == ["1 failed · 0 incomplete required check(s). See tier evidence for reasons."]


@pytest.mark.parametrize("tier", [1, 2, 3])
@pytest.mark.parametrize("incomplete", [False, True])
def test_hero_reason_keeps_advisory_results_separate_within_a_mixed_tier(tier, incomplete):
    required = ValidationResult(validator_name="Required check")
    required.add_success("required", "Required check passed")
    advisory = ValidationResult(validator_name="Advisory check")
    if incomplete:
        advisory.mark_scan_incomplete("optional comparison")
    else:
        advisory.add_error("Optional comparison failed")
    html = HTMLReporter(include_timestamp=False).render_all(
        [_gated(required, tier), _gated(advisory, tier, blocking=False)]
    )
    assert "Passed required checks" in html
    assert _hero_reasons(html) == [
        "1 advisory check(s) failed or incomplete; advisory checks do not change the required result."
    ]


def test_nojs_reports_have_explicit_native_report_theme():
    static = HTMLReporter(include_timestamp=False).render_all([static_result()])
    live = HTMLReporter(include_timestamp=False).render_all([agent_result()])
    assert 'data-default-theme="light" data-theme="light"' in static
    assert 'data-default-theme="dark" data-theme="dark"' in live
    assert '[data-theme="dark"]' in live


def test_browser_check_fixtures_render_every_target_the_checks_use(tmp_path):
    # browser_navigation_checks.py needs a browser; this keeps its fixtures usable without one.
    render_fixtures(tmp_path)
    pages = {name: (tmp_path / f"{name}.html").read_text(encoding="utf-8") for name in ("full", "stress", "absent")}
    documents = {name: Document(html) for name, html in pages.items()}

    def classes(element):
        return set((element.get("class") or "").split())

    full = documents["full"]
    ids = {element.get("id") for element in full.elements}
    assert {
        "t3-trials-table",
        "validator-1",
        "results-by-validator",
        "report-workspace",
        "run-metadata",
        "tier3-tab-overview",
        "tier3-tab-suggestions",
        "tier3-tab-provenance",
        "tier3-tab-agents",
    } <= ids
    for required in ({"t3-agent-anchor"}, {"t3-agent-tab-btn"}, {"skip-link"}, {"hero-command-status", "fail"}):
        assert any(required <= classes(element) for element in full.elements), required
    assert any("check-finding" in classes(element) and element.get("id") for element in full.elements)
    assert "Changes required" in pages["full"] and "Tier 3 quality: " in pages["full"]
    for name in ("full", "stress"):
        tiles = {
            element.get("data-target-tab") for element in documents[name].elements if "tier-card" in classes(element)
        }
        assert tiles == {"tier1", "tier2", "tier3"}, name
    skipped = [element for element in documents["absent"].elements if {"tier-card", "skipped"} <= classes(element)]
    assert len(skipped) == 2
