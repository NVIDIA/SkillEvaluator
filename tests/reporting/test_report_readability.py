# SPDX-License-Identifier: Apache-2.0
"""Concise reporting keeps unavailable measurements and evidence truthful."""

import json
import re
from copy import deepcopy
from html import unescape
from html.parser import HTMLParser

import pytest
from tests.reporting.browser_navigation_fixtures import _agent_result

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter


def _text(markup):
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def _reliability(markup):
    return _text(re.search(r'<table class="summary-table pass-at-k-table".*?</table>', markup, re.DOTALL)[0])


def _render(result):
    return HTMLReporter(include_timestamp=False).render_all([result])


class _DisclosureText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.closed = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.closed.append("open" not in dict(attrs))

    def handle_endtag(self, tag):
        if tag == "details":
            self.closed.pop()

    def handle_data(self, data):
        if not any(self.closed):
            self.text.append(data)


def test_comparison_keeps_measurements_visible_and_recorded_reasoning_expandable():
    result = _agent_result("demo", ["agent"])
    explanation = "Detailed recorded evaluator reasoning remains available for inspection."
    for dim in result.metadata["agent_eval"]["dimensions"]:
        dim["reasoning_bullets"] = [explanation]
    markup = _render(result)
    table = re.search(r'<table[^>]*id="tier3-dimension-table".*?</table>', markup, re.DOTALL)[0]
    visible = _DisclosureText()
    visible.feed(table)
    text = " ".join(visible.text)
    assert explanation not in text
    assert explanation in table
    for label in ("Security", "Correctness", "Discoverability", "Effectiveness", "Overall"):
        assert label in text
    assert "0.80" in text and "0.40" in text and "+0.40" in text


def test_attempt_coverage_does_not_invent_pass_rate_or_pass_counts():
    result = _agent_result("demo", ["agent"])
    payload = result.metadata["agent_eval"]
    agent = payload["agents"]["agent"]
    agent["pass_at_k"] = {
        "with_skill": {"attempts_used": 16, "max_attempts_possible": 18},
        "without_skill": {"attempts_used": 12, "max_attempts_possible": 18},
        "lift": {},
    }
    payload["pass_at_k"] = deepcopy(agent["pass_at_k"])
    original = deepcopy(payload)

    markup = _render(result)
    table = _reliability(markup)
    assert "0%" not in table
    assert "0 of 0" not in table
    assert "0 of 16" not in table
    assert "16 scored" in table and "12 scored" in table
    assert "N/A" in table
    exported = re.search(r'<script[^>]*id="tier3-full"[^>]*>(.*?)</script>', markup, re.DOTALL)[1]
    assert json.loads(exported) == original
    assert payload == original


def test_recorded_zero_reliability_is_preserved():
    result = _agent_result("demo", ["agent"])
    payload = result.metadata["agent_eval"]
    branch = {
        "rate": 0.0,
        "passed_cases": 0,
        "total_cases": 2,
        "attempts_used": 2,
        "cases": {name: {"attempts": [{"passed": False, "score": 0.0}]} for name in ["case-1", "case-2"]},
    }
    payload["agents"]["agent"]["pass_at_k"] = {"with_skill": branch}
    payload["pass_at_k"] = {"with_skill": deepcopy(branch)}
    table = _reliability(_render(result))
    assert "0%" in table
    assert "0 of 2" in table


@pytest.mark.parametrize("attempts", [None, [{"passed": True}], [{"passed": "false"}]])
def test_partial_or_untyped_attempt_outcomes_do_not_invent_a_pass_count(attempts):
    result = _agent_result("demo", ["agent"])
    payload = result.metadata["agent_eval"]
    case = {"attempts_used": 16}
    if attempts is not None:
        case["attempts"] = attempts
    branch = {"attempts_used": 16, "cases": {"case-1": case}}
    payload["agents"]["agent"]["pass_at_k"] = {"with_skill": branch}
    payload["pass_at_k"] = {"with_skill": deepcopy(branch)}
    table = _reliability(_render(result))
    assert "0 of 16" not in table and "1 of 16" not in table
    assert "N/A · 16 scored" in table


def test_agent_does_not_call_a_completed_baseline_absent_when_aggregate_is_missing():
    result = _agent_result("demo", ["agent"])
    agent = result.metadata["agent_eval"]["agents"]["agent"]
    agent["baseline"] = None
    agent["lift"] = None
    markup = _render(result)
    details = markup.split('id="tier3-page-agents"', 1)[1].split('id="tier3-page-dataset"', 1)[0]
    strip = details.split('class="t3-kpi-strip"', 1)[1].split('class="t3-dim-strip"', 1)[0]
    assert "Overall baseline unavailable" in _text(strip)
    assert "Skill Lift compares Overall" not in strip


def test_one_tier2_skip_notice_retains_both_affected_checks():
    results = [
        ValidationResult(validator_name=name, metadata={"skipped": True, "skip_reason": "No local catalog supplied"})
        for name in ["CONTEXT_DEDUP", "CONTEXT_OPTIMIZATION"]
    ]
    markup = HTMLReporter(include_timestamp=False).render_all(results)
    within = markup.split('id="tier2-within"', 1)[1].split('id="dedup-inter"', 1)[0]
    assert within.count("No local catalog supplied") == 1
    assert "CONTEXT_DEDUP" in within and "CONTEXT_OPTIMIZATION" in within


def test_unknown_tier2_scope_keeps_skipped_check_identity_and_reason():
    result = ValidationResult(
        validator_name="UNKNOWN_CATALOG_CHECK",
        metadata={"gating": {"tier": 2}, "skipped": True, "skip_reason": "Unique retained skip explanation"},
    )
    markup = _render(result)
    other = markup.split('id="tier2-other"', 1)[1].split("</section>", 1)[0]
    assert "Unique retained skip explanation" in other and "UNKNOWN_CATALOG_CHECK" in other
    ids = re.findall(r'\bid="([^"]+)"', markup)
    assert len(ids) == len(set(ids))


def test_harbor_display_labels_use_only_recorded_job_associations():
    from skillevaluator.reporting.harbor_viewer import harbor_job_display_links

    source = {
        "harbor_viewer": {
            "jobs": [
                {"name": "very-long-opaque-job-id", "url": "https://example.test/jobs/1"},
                {"name": "another-opaque-job-id", "url": "https://example.test/jobs/2"},
            ]
        },
        "agents": {"agent": {"trials": [{"harbor_viewer": {"job_url": "https://example.test/jobs/1"}}]}},
    }
    before = deepcopy(source)
    jobs = harbor_job_display_links(source)
    assert [job["display_label"] for job in jobs] == ["agent · With skill", "Job 2"]
    assert jobs[0]["name"] == "very-long-opaque-job-id"
    assert source == before
