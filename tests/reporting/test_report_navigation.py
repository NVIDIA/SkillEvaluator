# SPDX-License-Identifier: Apache-2.0
"""Navigation contracts on the actual rendered, standalone report."""

from html.parser import HTMLParser

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
    assert {element["data-report-tier"] for element in doc.elements if "data-report-tier" in element} == {"tier1"}


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
