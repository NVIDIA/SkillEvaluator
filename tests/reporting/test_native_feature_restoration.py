# SPDX-License-Identifier: Apache-2.0
"""Native report capabilities stay visible without changing retained measurements."""

import re
from copy import deepcopy
from pathlib import Path

import pytest
from browser_navigation_fixtures import _agent_result

from skillevaluator import __version__
from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


@pytest.mark.parametrize("status", ["comparable", "partial"])
def test_comparison_table_keeps_partial_notes_without_repeating_normal_comparison_prose(status):
    result = _agent_result("synthetic", ["first-agent", "second-agent"])
    payload = result.metadata["agent_eval"]
    note = "The same full-comparison note should not repeat in every measured cell."
    dimensions = list(payload["dimensions"])
    for agent in payload["agents"].values():
        dimensions.extend(agent["dimensions"])
    for dimension in dimensions:
        dimension["comparison"] = {
            "status": status,
            "label": "Recorded comparison",
            "reason": note,
            "with_skill": 0.8,
            "baseline": 0.6,
            "lift": 0.15,
        }
    original = deepcopy(payload)
    html = _render(result)
    table = html.split('id="tier3-dimension-table"', 1)[1].split("</table>", 1)[0]
    assert (note in table) is (status == "partial")
    assert payload == original


def test_incomplete_baseline_quality_is_not_reported_as_a_baseline_that_never_ran():
    from test_report_clarity import _result

    result = _result("incomplete")
    payload = result.metadata["agent_eval"]
    payload["metric_ids"] = list(DEFAULT_METRICS)
    payload["agents"] = {
        "agent": {
            "execution_status": "failed",
            "conditions": {
                "with_skill": {"execution_status": "failed", "scored_attempts": 16, "expected_attempts": 18},
                "without_skill": {"execution_status": "failed", "scored_attempts": 12, "expected_attempts": 18},
            },
        }
    }
    original = deepcopy(payload)
    html = _render(result)
    table = html.split('id="tier3-dimension-table"', 1)[1].split("</table>", 1)[0]
    assert "baseline not run" not in table
    assert "baseline quality unavailable" in table
    assert "28 of 36 scored" in html
    assert payload == original


@pytest.mark.parametrize("arm", [None, "unknown", [], {"execution_status": "skipped"}])
def test_baseline_quality_hint_tolerates_legacy_arm_shapes_and_keeps_explicit_skips(arm):
    from test_report_browser import _edge_result

    result = _edge_result("missing-baseline")
    payload = result.metadata["agent_eval"]
    next(iter(payload["agents"].values()))["conditions"]["without_skill"] = arm
    original = deepcopy(payload)
    html = _render(result)
    table = html.split('id="tier3-dimension-table"', 1)[1].split("</table>", 1)[0]
    explicit_skip = isinstance(arm, dict) and arm.get("execution_status") == "skipped"
    assert ("baseline not run" in table) is explicit_skip
    if not explicit_skip:
        assert "baseline quality unavailable" in table
    assert payload == original


@pytest.mark.parametrize("invalid_dimension", [{}, {"id": ""}, {"id": "   "}])
def test_custom_metrics_do_not_create_a_blank_dimension_table_from_missing_peer_ids(invalid_dimension):
    from test_report_browser import _edge_result

    result = _edge_result("custom-only")
    payload = result.metadata["agent_eval"]
    payload["agents"]["peer-custom"] = deepcopy(next(iter(payload["agents"].values())))
    payload["agents"]["peer-custom"]["dimensions"] = [invalid_dimension]
    original = deepcopy(payload)
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert 'id="tier3-metric-comparison-table"' in html
    assert 'id="tier3-dimension-table"' not in html
    assert payload == original


def _render(result=None, **kwargs):
    result = result or _agent_result("synthetic", ["first-agent", "second-agent"])
    return HTMLReporter(include_timestamp=False, **kwargs).render_all([result])


def _overview(html):
    return html.split('id="tier3-page-overview"', 1)[1].split('id="tier3-page-trials"', 1)[0]


def test_recorded_tool_versions_are_visible_without_a_generated_timestamp():
    result = ValidationResult(validator_name="Schema Check")
    result.metadata["skillspector_version"] = "2.11.1"
    original = deepcopy(result)
    html = _render(result)
    header = html.split('<header class="main-header">', 1)[1].split("</header>", 1)[0]
    assert f"SkillEvaluator v{__version__}" in header
    assert "SkillSpector v2.11.1" in header
    assert "<details" not in header
    assert "Generated:" not in header
    assert result == original


def test_native_header_scale_is_not_replaced_by_the_compact_override():
    css = (Path(__file__).parents[2] / "src/skillevaluator/reporting/templates/report_ui.css.j2").read_text()
    assert not re.search(r"\.main-header\s+\.header-title\s*\{[^}]*font-size", css)
    assert not re.search(r"\.main-header\s*\{[^}]*padding:\s*\.7rem", css)
    assert ".main-header .header-subtitle-row { display:none; }" not in css


@pytest.mark.parametrize("multi", [False, True])
def test_harbor_results_and_analysis_are_visible_in_tile_and_overview(multi):
    result = _agent_result("synthetic", ["first-agent"])
    viewer = {"job_url": "https://example.test/job/one", "analysis_url": "https://example.test/analysis/one"}
    if multi:
        viewer = {
            "analysis_url": "https://example.test/analysis/run",
            "jobs": [
                {
                    "name": name,
                    "url": f"https://example.test/job/{name}",
                    "analysis_url": f"https://example.test/analysis/{name}",
                }
                for name in ("one", "two")
            ],
        }
    result.metadata["agent_eval"]["harbor_viewer"] = viewer
    original = deepcopy(result.metadata["agent_eval"])
    html = _render(result)
    tile = html.split('data-target-tab="tier3"', 1)[1].split("</nav>", 1)[0]
    urls = ["https://example.test/job/one", "https://example.test/analysis/one"]
    if multi:
        urls += [
            "https://example.test/job/two",
            "https://example.test/analysis/two",
            "https://example.test/analysis/run",
        ]
    for surface in (tile, _overview(html)):
        for url in urls:
            assert surface.count(f'href="{url}"') == 1
    assert "event.target===this" in tile
    assert result.metadata["agent_eval"] == original


def test_absent_harbor_links_have_a_concise_overview_state():
    overview = _overview(_render())
    assert 'id="tier3-harbor-unavailable"' in overview
    assert "No Harbor viewer or analysis links were recorded." in overview
    assert 'class="t3-harbor-link"' not in overview


def test_overview_has_one_native_dimension_table_with_cross_agent_context():
    overview = _overview(_render())
    table = overview.split('id="tier3-dimension-table"', 1)[1].split("</table>", 1)[0]
    assert "Baseline</th>" in table
    assert "second-agent</button>" in table
    assert "Overall</strong>" in table
    assert '<table class="summary-table" id="tier3-multi-agent-dimension-table"' not in overview
    wrapper = overview.split('id="tier3-multi-agent-dimension-table"', 1)[1]
    assert wrapper.index('id="tier3-dimension-table"') < wrapper.index("</table>")
    assert "Multi-Agent Dimension Lift</summary>" not in overview


def test_native_agent_scorecards_retain_scores_counts_and_details_navigation():
    result = _agent_result("synthetic", ["first-agent", "second-agent"])
    original = deepcopy(result.metadata["agent_eval"])
    overview = _overview(_render(result))
    assert 'id="tier3-agent-scorecards" open' in overview
    cards = re.findall(r'<div class="t3-agent-card t3-agent-deeplink"(.*?)</div>\s*</div>', overview, re.DOTALL)
    assert len(cards) == 2
    for name, card in zip(("first-agent", "second-agent"), cards, strict=True):
        assert name in card
        assert 'class="t3-agent-score"' in card
        assert "2 trials" in card
        assert "jumpToTier3Agent" in card
    assert result.metadata["agent_eval"] == original


def test_dataset_uses_native_generic_title_and_the_recorded_content_subject():
    result = _agent_result("synthetic", ["first-agent"])
    result.metadata["agent_eval"]["dataset"] = [{"id": "case-1", "prompt": "Summarize synthetic input."}]
    html = _render(result, content_label="Plugin")
    dataset = html.split('id="tier3-page-dataset"', 1)[1].split('id="tier3-page-suggestions"', 1)[0]
    assert "Evaluation Dataset</h2>" in dataset
    assert "AgentSkills Dataset" not in dataset
    assert "Plugin: <b>synthetic</b>" in dataset


def test_unavailable_overall_comparison_does_not_label_a_missing_measurement_fail():
    result = _agent_result("synthetic", ["first-agent"])
    result.metadata["agent_eval"].update(
        agents={},
        dimensions=[],
        overall_score=None,
        overall_lift=None,
        best_agent=None,
        execution_status="failed",
        metric_ids=list(DEFAULT_METRICS),
    )
    result.add_error("Recorded evaluation did not complete")
    original = deepcopy(result.metadata["agent_eval"])
    overview = _overview(_render(result))
    overall = re.search(r"<tr>\s*<td><strong>Overall</strong>.*?</tr>", overview, re.DOTALL)[0]
    assert "N/A" in overall
    assert "FAIL" not in overall and "PASS" not in overall
    assert result.metadata["agent_eval"] == original


def _extreme_result(scores):
    baseline = dict.fromkeys(DEFAULT_METRICS, 0.4)
    payload = build_agent_eval_payload(
        "synthetic",
        {
            "agent": {
                "execution_status": "succeeded",
                "expected_attempts": 2,
                "scored_attempts": 2,
                "with_skill": scores,
                "without_skill": baseline,
                "rewards": [{"trial_id": "with", "entry_id": "case-1", **scores}],
                "rewards_baseline": [{"trial_id": "base", "entry_id": "case-1", **baseline}],
                "num_trials": 1,
                "num_trials_baseline": 1,
            }
        },
        use_llm_judge=False,
    )
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = payload
    return result


def test_agent_details_extremes_use_recorded_dimensions_and_include_zero():
    scores = dict.fromkeys(DEFAULT_METRICS, 0.5)
    scores.update(security=0.9, accuracy=0.0, goal_accuracy=0.6, skill_efficiency=0.2)
    result = _extreme_result(scores)
    original = deepcopy(result.metadata["agent_eval"])
    html = _render(result)
    agents = html.split('id="tier3-page-agents"', 1)[1].split('id="tier3-page-dataset"', 1)[0]
    assert 'class="t3-kpi-label">Best Dimension</div>' in agents
    assert 'class="t3-kpi-label">Weakest Dimension</div>' in agents
    best = re.search(r'Best Dimension</div>\s*<div[^>]*>([^<]+)</div>\s*<div class="t3-kpi-sub">([^<]+)</div>', agents)
    weakest = re.search(
        r'Weakest Dimension</div>\s*<div[^>]*>([^<]+)</div>\s*<div class="t3-kpi-sub">([^<]+)</div>', agents
    )
    assert best.groups() == ("Security", "0.90")
    assert weakest.groups() == ("Correctness", "0.00")
    assert result.metadata["agent_eval"] == original


def test_agent_details_missing_dimensions_do_not_become_zero_extremes():
    result = _extreme_result(dict.fromkeys(DEFAULT_METRICS, None))
    html = _render(result)
    agents = html.split('id="tier3-page-agents"', 1)[1].split('id="tier3-page-dataset"', 1)[0]
    assert "Best Dimension</div>" not in agents
    assert "Weakest Dimension</div>" not in agents


@pytest.mark.parametrize("score", [None, 0.0])
def test_scorecard_distinguishes_missing_quality_from_a_recorded_zero(score):
    result = _extreme_result(dict.fromkeys(DEFAULT_METRICS, score))
    if score is None:
        result.metadata["agent_eval"]["mode"] = "baseline_only"
    original = deepcopy(result.metadata["agent_eval"])
    overview = _overview(_render(result))
    card = re.search(r'<div class="t3-agent-card t3-agent-deeplink".*?</div>\s*</div>', overview, re.DOTALL)[0]
    assert ('class="t3-mini-bar"' in card) is (score is not None)
    if score is None:
        assert "with-skill unavailable" in card
    else:
        assert re.search(r"width:0(?:\.0)?%", card)
    assert result.metadata["agent_eval"] == original


@pytest.mark.parametrize("status", ["partial", "unavailable"])
def test_recorded_partial_comparison_is_visible_and_unavailable_numbers_stay_absent(status):
    result = _agent_result("synthetic", ["first-agent"])
    result.metadata["agent_eval"]["dimensions"][0].update(
        partial=True,
        comparison={
            "status": status,
            "label": "Recorded paired comparison",
            "reason": "Recorded scope limitation",
            "with_skill": 0.8,
            "baseline": 0.6,
            "lift": 0.15,
        },
    )
    original = deepcopy(result.metadata["agent_eval"])
    overview = _overview(_render(result))
    assert ">partial</span>" in overview
    assert "Recorded paired comparison" in overview and "Recorded scope limitation" in overview
    assert ("0.60 → 0.80 (+0.15 lift)" in overview) is (status == "partial")
    assert result.metadata["agent_eval"] == original


@pytest.mark.parametrize("peer_score", [None, 0.0])
@pytest.mark.parametrize("recorded_not_applicable", [False, True])
def test_missing_best_dimensions_keep_each_peers_recorded_scope(peer_score, recorded_not_applicable):
    result = _agent_result("synthetic", ["best", "peer"])
    payload = result.metadata["agent_eval"]
    recorded_dimensions = deepcopy(payload["dimensions"])
    payload["dimensions"] = [dimension for dimension in recorded_dimensions if dimension["id"] == "correctness"]
    payload["not_applicable_dimensions"] = [
        {"id": dimension["id"], "reason": "Not configured for the best agent"}
        for dimension in recorded_dimensions
        if recorded_not_applicable and dimension["id"] != "correctness"
    ]
    payload["agents"]["best"]["dimensions"] = deepcopy(payload["dimensions"])
    payload["agents"]["best"]["metrics_with_skill"] = ["accuracy"]
    peer = payload["agents"]["peer"]
    peer.update(
        with_skill=peer_score, baseline=0.4, lift=None, execution_status="failed" if peer_score is None else "succeeded"
    )
    for dimension in peer["dimensions"]:
        dimension.update(score=peer_score, with_skill=peer_score, baseline=0.4, lift=None)
    assert payload["best_agent"] == "best"
    assert len(payload["not_applicable_dimensions"]) == (4 if recorded_not_applicable else 0)
    original = deepcopy(payload)
    table = _overview(_render(result)).split('id="tier3-dimension-table"', 1)[1].split("</table>", 1)[0]
    missing_rows = [
        row
        for row in re.findall(r"<tr>(.*?)</tr>", table, re.DOTALL)
        if ("Not applicable</span>" if recorded_not_applicable else "Not measured</span>") in row
    ]
    assert len(missing_rows) == 4
    for row in missing_rows:
        cells = re.findall(r"<td(?:\s[^>]*)?>(.*?)</td>", row, re.DOTALL)
        assert [cell.strip() for cell in cells[1:4]] == ["N/A", "N/A", "N/A"]
    peer_cells = re.findall(r'<span class="t3-mad-score"[^>]*>(.*?)</span>', table)
    assert peer_cells == (["N/A"] * 6 if peer_score is None else ["0.00"] * 6)
    assert table.count('class="t3-mad-bar"') == (0 if peer_score is None else 6)
    if peer_score is None:
        assert table.count("with-skill unavailable") == 6
    assert result.metadata["agent_eval"] == original
