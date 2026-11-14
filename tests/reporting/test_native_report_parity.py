# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for the native report and retained payload exports."""

import base64
import json
import re
from copy import deepcopy

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting.html import HTMLReporter


def _agent_result():
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(
        "demo",
        {"agent": {"overall_with_skill": 0.8, "overall_without_skill": 0.4}},
        use_llm_judge=False,
    )
    return result


def _embedded_payload(html):
    match = re.search(r'<script type="application/json" id="tier3-full"([^>]*)>(.*?)</script>', html, re.DOTALL)
    assert match
    raw = base64.b64decode(match[2]).decode() if 'data-encoding="base64"' in match[1] else match[2]
    return json.loads(raw)


def test_native_header_and_page_geometry_replace_sidebar_workspace():
    html = HTMLReporter(include_timestamp=False).render_all([ValidationResult(validator_name="Schema")])
    assert '<div class="logo-section">' in html
    assert ">Skill Evaluator</p>" in html
    assert '<aside class="report-sidebar"' not in html
    assert "body{font-family:ui-sans-serif" not in html
    assert ".container { max-width: 1200px; margin: 0 auto; padding: 2rem; }" in html
    assert "background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);" in html
    assert '<div class="dashboard">' in html
    assert "Validators Run" in html


def test_retained_canonical_export_is_distinct_from_fail_closed_display():
    result = _agent_result()
    raw = result.metadata["agent_eval"]
    raw.update(verdict="neutral", execution_status="failed", empty=[], nullable=None)
    raw["summary"]["execution_status"] = "failed"
    original = deepcopy(raw)
    result.add_error("Agent arm failed")
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert _embedded_payload(html) == original
    assert result.metadata["agent_eval"] == original
    assert HTMLReporter._tier3_report_data([result])["verdict"] != original["verdict"]
    assert "Tier 3 quality:" in html


def test_report_default_theme_is_dark_for_tier3_and_light_for_static():
    static = HTMLReporter(include_timestamp=False).render_all([ValidationResult(validator_name="Schema")])
    live = HTMLReporter(include_timestamp=False).render_all([_agent_result()])
    assert 'data-default-theme="light"' in static
    assert 'data-default-theme="dark"' in live
    assert "localStorage.getItem('skillevaluator-theme')" in live


def test_ci_report_suppresses_retained_payload_embedding():
    html = HTMLReporter(
        include_timestamp=False, ci_context={"canonical_json_url": "https://example.com/report.json"}
    ).render_all([_agent_result()])
    assert 'href="https://example.com/report.json"' in html
    assert 'id="report-data"' not in html
    assert 'id="tier3-full"' not in html
    assert "onclick=\"tier3DownloadJson('tier3-full'" not in html


def test_overview_and_insights_share_native_severity_cards_and_action_checks():
    result = _agent_result()
    payload = result.metadata["agent_eval"]
    payload.update(execution_status="succeeded", verdict="pass", expected_attempts=1, scored_attempts=1)
    payload["agents"]["agent"].update(execution_status="succeeded", expected_attempts=1, scored_attempts=1)
    payload["conclusions"] = [
        {"title": "Retained conclusion", "message": "Measured evidence", "severity": "pass", "source": "deterministic"}
    ]
    payload["recommendations"] = [
        {
            "title": "Retained action",
            "message": "Improve the instruction",
            "category": "Document",
            "source": "deterministic",
            "check": "Repeat the same case",
            "evidence_text": "case-one",
        }
    ]
    html = HTMLReporter(include_timestamp=False).render_all([result])
    overview = html.split('id="tier3-page-overview"', 1)[1].split('id="tier3-page-trials"', 1)[0]
    insights = html.split('id="tier3-page-suggestions"', 1)[1].split('id="tier3-page-provenance"', 1)[0]
    for section in (overview, insights):
        assert '<span class="t3-pill ok">pass</span>' in section
        assert '<span class="t3-sug-cat">Document</span>' in section
        assert "Check: Repeat the same case" in section
        assert "Evidence: case-one" in section


def test_native_tier2_intra_and_inter_tabs_keep_similarity_evidence():
    result = ValidationResult(validator_name="Similarity Check")
    result.add_success("catalog_compared", "Compared catalog entries", catalog_entry_count=5)
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert 'id="dedup-tab-intra"' in html
    assert 'id="dedup-tab-inter"' in html
    assert 'aria-controls="dedup-intra"' in html
    assert 'aria-controls="dedup-inter"' in html
    assert 'id="dedup-intra" class="dedup-subtab"' in html
    assert 'id="dedup-inter" class="dedup-subtab"' in html
    across = html.split('id="tier2-across"', 1)[1].split("</section>", 1)[0]
    assert "Catalog entries compared" in across
    assert "Compared catalog entries" in across


def test_failed_execution_actions_do_not_repeat_next_steps_or_expand_long_evidence():
    result = _agent_result()
    payload = result.metadata["agent_eval"]
    payload.update(execution_status="failed", verdict="fail")
    payload["next_steps"] = [
        {
            "kind": "Restore runner",
            "target": "evaluation",
            "problem": "A runner did not finish",
            "action": "Restore the runner and repeat the retained cases",
            "check": "Every case has a score",
            "evidence": "case-identity-" * 80,
            "strength": "hypothesis",
        }
    ]
    result.add_error("Runner failed")
    html = HTMLReporter(include_timestamp=False).render_all([result])
    for name, end in (("overview", "trials"), ("suggestions", "provenance")):
        section = html.split(f'id="tier3-page-{name}"', 1)[1].split(f'id="tier3-page-{end}"', 1)[0]
        assert re.sub(r"<[^>]+>", "", section).count("Restore the runner and repeat the retained cases") == 1
        assert "Next steps for the run:" not in section
        assert "Scope: evaluation" in section
        assert "Strength: hypothesis" in section
        disclosure = re.search(
            r'<details class="t3-eval-evidence"([^>]*)>.*?<summary>Evidence</summary>(.*?)</details>',
            section,
            re.DOTALL,
        )
        assert disclosure and "open" not in disclosure[1]
        assert payload["next_steps"][0]["evidence"] in disclosure[2]


def test_only_fully_represented_failed_steps_are_suppressed():
    from skillevaluator.reporting import html as html_module

    represented = {
        "action": "Rerun",
        "target": "agent",
        "check": "Complete",
        "evidence": "case-a",
        "strength": "hypothesis",
        "agent": "a",
    }
    different_evidence = {**represented, "evidence": "case-b"}
    different_scope = {**represented, "target": "dataset"}
    different_agent = {**represented, "agent": "b"}
    steps = [represented, different_evidence, different_scope, different_agent]
    view = {
        "next_steps": steps,
        "recommendations": [
            {
                "insight_kind": "execution_action",
                "message": "Rerun",
                "metric": "agent",
                "check": "Complete",
                "evidence_text": "case-a",
                "strength": "hypothesis",
                "agent": "a",
            }
        ],
    }
    assert html_module._unrepresented_insight_steps(view, failed=True) == steps[1:]
    assert html_module._unrepresented_insight_steps(view, failed=False) == steps


def test_failed_next_steps_keep_distinct_agent_case_and_condition_in_rendered_cards():
    result = _agent_result()
    payload = result.metadata["agent_eval"]
    payload.update(execution_status="failed", verdict="fail")
    common = {
        "kind": "Restore runner",
        "target": "evaluation",
        "problem": "Runner failed",
        "action": "Restore and rerun",
        "check": "Every attempt is scored",
        "evidence": "retained-log",
        "strength": "supported",
    }
    payload["next_steps"] = [
        {**common, "agent": "agent-one", "case_id": "case-one", "condition": "with_skill"},
        {**common, "agent": "agent-two", "case": "case-two", "condition": "without_skill"},
    ]
    original = deepcopy(payload)
    result.add_error("Runner failed")
    html = HTMLReporter(include_timestamp=False).render_all([result])
    for start, end in (("overview", "trials"), ("suggestions", "provenance")):
        section = html.split(f'id="tier3-page-{start}"', 1)[1].split(f'id="tier3-page-{end}"', 1)[0]
        text = re.sub(r"<[^>]+>", "", section)
        assert text.count("Restore and rerun") == 2
        for association in (
            "Agent: agent-one",
            "Agent: agent-two",
            "Case: case-one",
            "Case: case-two",
            "Condition: with_skill",
            "Condition: without_skill",
        ):
            assert association in text
        assert "Next steps for the run:" not in text
    assert _embedded_payload(html) == original
    assert payload == original
