# SPDX-License-Identifier: Apache-2.0
"""Single-owner report hierarchy keeps evidence and scoped outcomes reachable."""

import json
import re
from copy import deepcopy

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting.html import HTMLReporter


def _result(state="complete"):
    result = ValidationResult(validator_name="AGENT_EVAL")
    payload = build_agent_eval_payload(
        "demo", {"agent": {"overall_with_skill": 0.8, "overall_without_skill": 0.4}}, use_llm_judge=False
    )
    payload.update(execution_status="succeeded", expected_attempts=2, scored_attempts=2)
    payload["agents"]["agent"].update(execution_status="succeeded", expected_attempts=2, scored_attempts=2)
    payload["conclusions"] = [
        {"title": "Authored finding", "message": "Complete authored conclusion", "severity": "pass"}
    ]
    payload["recommendations"] = [
        {
            "title": "Recorded action",
            "message": "Complete authored recommendation",
            "check": "Repeat the retained case",
            "evidence_text": "case-one",
        }
    ]
    if state in ("incomplete", "source-rejected"):
        payload.update(
            overall_score=None,
            overall_lift=None,
            best_agent=None,
            agents={},
            execution_status="failed",
            scored_attempts=0,
        )
        result.add_error("Source rejected" if state == "source-rejected" else "Execution did not finish")
    elif state == "mixed":
        payload["agents"]["failed-agent"] = {
            "execution_status": "failed",
            "execution_errors": ["Runner failed"],
            "with_skill": None,
            "baseline": None,
        }
        payload.update(execution_status="failed", expected_attempts=4)
        result.add_error("Runner failed")
    elif state == "baseline-only":
        payload.update(mode="baseline_only", overall_lift=None)
    elif state == "custom":
        payload["metric_ids"] = ["custom_metric"]
    elif state == "skipped":
        payload.update(execution_status="skipped", overall_score=None, overall_lift=None, agents={})
    result.metadata["agent_eval"] = payload
    return result


def _page(html, name, next_name):
    return html.split(f'id="tier3-page-{name}"', 1)[1].split(f'id="tier3-page-{next_name}"', 1)[0]


@pytest.mark.parametrize(
    "state", ["complete", "incomplete", "mixed", "custom", "baseline-only", "skipped", "source-rejected"]
)
def test_overview_has_one_summary_owner_and_retains_full_insights(state):
    result = _result(state)
    original = deepcopy(result.metadata["agent_eval"])
    html = HTMLReporter(include_timestamp=False).render_all([result])
    overview = _page(html, "overview", "trials")
    insights = _page(html, "suggestions", "provenance")
    assert 'class="t3-sug-card ' not in overview
    assert len(re.findall(r'class="dashboard-card(?:[ "])', overview)) <= 3
    assert "Complete authored conclusion" not in overview
    if state not in ("incomplete", "mixed", "source-rejected", "skipped"):
        assert "Complete authored conclusion" in insights
        assert "Complete authored recommendation" in insights
        assert "Check: Repeat the retained case" in insights
        assert "Evidence: case-one" in insights
    assert 'id="tier3-next-steps-preview"' in overview
    assert 'href="#tier3-page-suggestions"' in overview
    assert 'class="nav-tabs mobile-tier-nav"' not in html
    assert html.count('aria-label="Tier navigation"') == 1
    assert 'class="theme-toggle"' not in html
    assert re.search(r'<details class="run-metadata" id="run-metadata">', html)
    assert result.metadata["agent_eval"] == original


def test_unscored_quality_uses_one_unavailable_state_with_coverage():
    html = HTMLReporter(include_timestamp=False).render_all([_result("incomplete")])
    overview = _page(html, "overview", "trials")
    assert overview.count('id="tier3-quality-unavailable"') == 1
    assert "Quality measurements unavailable" in overview
    assert "Coverage" in overview
    assert 'data-primary-metric="score"' not in overview
    assert 'data-primary-metric="lift"' not in overview
    assert "No successfully scored agent" not in overview


def test_recorded_secondary_outcomes_share_one_compact_table():
    result = _result()
    result.metadata["agent_eval"]["correct_answers"] = {
        "with_skill": {"correct": 1, "scored": 2},
        "without_skill": {"correct": 0, "scored": 2},
        "difference": 50,
        "difference_unit": "percentage_points",
    }
    html = HTMLReporter(include_timestamp=False).render_all([result])
    overview = _page(html, "overview", "trials")
    assert 'id="tier3-secondary-outcomes"' in overview
    assert "Goals met" in overview and "1/2" in overview and "+50.0 pp" in overview
    assert "Required Checks Passed" not in overview


def test_native_telemetry_uses_existing_chart_lifecycle_and_prepreview_input(monkeypatch):
    from skillevaluator.reporting import html as module
    from skillevaluator.reporting.telemetry_view import build_telemetry_view

    result = _result()
    totals = {"prompt_tokens": 120, "cached_tokens": 40, "uncached_prompt_tokens": 80, "completion_tokens": 20}
    scope = {
        "with_skill": {"totals": totals, "coverage": {"measured_attempts": 1, "status": "complete"}},
        "baseline": {"totals": totals, "coverage": {"measured_attempts": 1, "status": "complete"}},
    }
    result.metadata["agent_eval"]["token_usage"] = {"dataset": scope, "agents": {"agent": scope}}
    seen = []

    def project(payload):
        seen.append(deepcopy(payload.get("token_usage")))
        return build_telemetry_view(payload)

    monkeypatch.setattr(module, "build_telemetry_view", project)
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert seen == [result.metadata["agent_eval"]["token_usage"]]
    assert 'id="tier3-recorded-token-chart"' in html
    assert 'getElementById("tier3-recorded-token-chart")' in html
    assert "data: recordedTokenData" in html
    assert 'id="tier3-token-usage"' in _page(html, "trials", "agents")


def test_tier2_missing_catalog_metadata_does_not_make_unavailable_tile_grid():
    result = ValidationResult(validator_name="Similarity Check")
    result.add_success("catalog_compared", "Compared catalog entries")
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert 'class="tier2-scope-summary"' in html
    assert '<p class="dashboard-card-value">Unavailable</p>' not in html


def test_overview_coverage_names_both_planned_and_unscored_attempts():
    result = _result("incomplete")
    result.metadata["agent_eval"]["agents"] = {
        "agent": {
            "execution_status": "failed",
            "conditions": {
                "with_skill": {"execution_status": "failed", "scored_attempts": 16, "expected_attempts": 18},
                "without_skill": {"execution_status": "failed", "scored_attempts": 12, "expected_attempts": 18},
            },
        }
    }
    html = HTMLReporter(include_timestamp=False).render_all([result])
    overview = _page(html, "overview", "trials")
    assert "28 of 36 scored" in overview
    assert "8 unscored" in overview


def test_agent_details_owns_condition_coverage_beside_the_selected_agent():
    result = _result()
    result.metadata["agent_eval"]["agents"]["agent"]["conditions"] = {
        "with_skill": {"execution_status": "succeeded", "scored_attempts": 18, "expected_attempts": 18},
        "without_skill": {"execution_status": "failed", "scored_attempts": 12, "expected_attempts": 18},
    }
    html = HTMLReporter(include_timestamp=False).render_all([result])
    agents = _page(html, "agents", "dataset")
    assert 'class="t3-agent-coverage"' in agents
    assert "18 of 18 scored" in agents
    assert "12 of 18 scored" in agents
    assert "Without skill" in agents and "Failed" in agents


def test_artifacts_use_recorded_agent_condition_and_keep_exact_identifiers_in_diagnostics():
    result = _result()
    payload = result.metadata["agent_eval"]
    payload["harbor_viewer"] = {"jobs": [{"name": "raw_run_123", "url": "https://example.test/job/123"}]}
    payload["trials"] = [
        {
            "agent": "agent",
            "entry_id": "case-one",
            "trial_id": "trial-one",
            "harbor_viewer": {"job_url": "https://example.test/job/123"},
        }
    ]
    html = HTMLReporter(include_timestamp=False).render_all([result])
    overview = _page(html, "overview", "trials")
    diagnostics = html.split('id="tier3-page-provenance"', 1)[1].split("</section>", 1)[0]
    assert "Harbor artifacts" not in overview
    assert "agent · With skill" in diagnostics
    assert "Artifact identifiers" in diagnostics
    assert "raw_run_123" in diagnostics


def test_measured_resources_default_closed_and_full_methods_stay_on_demand():
    result = _result()
    html = HTMLReporter(include_timestamp=False).render_all([result])
    trials = _page(html, "trials", "agents")
    assert '<details class="t3-details" id="tier3-agent-resources">' in trials
    assert '<details class="t3-details" id="tier3-resource-methods">' in trials
    methods = trials.split('id="tier3-resource-methods"', 1)[1].split("</details>", 1)[0]
    assert (
        "Efficiency scores measure routing and productive tool use. These separate measurements show observed agent resource use, including failed task outcomes. Lower use alone does not establish improved task quality or Efficiency. Delta = with skill minus baseline; negative means less use. Judge and setup time are excluded. Deltas require complete execution evidence, complete telemetry and matching case counts in both arms. Pairing does not certify matched concurrency or budget, or judge-model agreement. Reused baseline measurements are historical, not fresh or independent evidence. LLM and tool seconds split the agent's trajectory time; inferred LLM time also includes harness and queue time."
        in methods
    )
    lead = trials.split("<summary>Measured Agent Resources</summary>", 1)[1].split('<div class="table-scroll">', 1)[0]
    assert "includes failed outcomes" in lead and "partial" in lead.lower() and "N/A" in lead
    assert "Pairing does not certify" not in lead


def test_public_token_accounting_folds_generic_methods_but_keeps_specific_notices_beside_rows():
    from skillevaluator.reporting.tier3_token_usage import COVERAGE_NOTE, REUSED_BASELINE_NOTE, TOKEN_USAGE_NOTE

    result = _result()
    arm = {
        "totals": {"total_tokens": 120},
        "coverage": {"status": "complete", "measured_attempts": 1, "expected_attempts": 1},
    }
    result.metadata["agent_eval"]["token_usage"] = {
        "agents": {"agent": {"with_skill": arm, "baseline": arm}},
        "baseline_reused": True,
        "case_rows_omitted": 3,
    }
    html = HTMLReporter(include_timestamp=False).render_all([result])
    trials = _page(html, "trials", "agents")
    assert '<details class="t3-details" id="tier3-token-accounting">' in trials
    accounting = trials.split('id="tier3-token-accounting"', 1)[1].split("</details>", 1)[0]
    assert TOKEN_USAGE_NOTE in accounting
    table = trials.split('<div id="t3-token-usage">', 1)[1].split('id="tier3-token-accounting"', 1)[0]
    assert "120" in table
    assert COVERAGE_NOTE in table
    assert REUSED_BASELINE_NOTE in table
    assert "3 case row(s) were left out" in table
    assert TOKEN_USAGE_NOTE not in table


def test_recorded_telemetry_loads_charts_without_quality_agent_entries():
    result = _result("incomplete")
    arm = {
        "totals": {"prompt_tokens": 120, "cached_tokens": 40, "uncached_prompt_tokens": 80, "completion_tokens": 20},
        "coverage": {"status": "complete", "measured_attempts": 1, "expected_attempts": 1},
    }
    result.metadata["agent_eval"]["token_usage"] = {"agents": {"recorded-agent": {"with_skill": arm, "baseline": arm}}}
    assert result.metadata["agent_eval"]["agents"] == {}
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert 'id="tier3-recorded-token-chart"' in html
    assert 'id="tier3-chart-runtime"' in html
    chart = json.loads(re.search(r"var recordedTokenData = (.*?);", html)[1])
    assert chart["labels"] == ["recorded-agent · With skill", "recorded-agent · Without skill"]
    assert chart["datasets"][0]["data"] == [80, 80]
