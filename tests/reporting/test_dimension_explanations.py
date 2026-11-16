# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reports explain each dimension, its evidence and baseline, and why a score is missing."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import pytest
from scripts.ci import check_public_benchmarks as benchmark_gate
from tests.reporting.report_markup_assertions import text_content

from skillevaluator.constants import DIMENSION_HINTS, SIGNAL_EVIDENCE_SOURCES
from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
from skillevaluator.reporting import BenchmarkReporter, HTMLReporter
from skillevaluator.reporting.benchmark import _baseline_coverage, _dimension_cell
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

_SUCCEEDED = {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}


def _agent(*, baseline: bool = True, with_scores: dict[str, float | None] | None = None) -> dict:
    scores = {**dict.fromkeys(DEFAULT_METRICS, 0.9), **(with_scores or {})}
    reward = {**scores, "entry_id": "case-1", "trial_id": "case-1__with"}
    agent = {
        "execution_status": "succeeded",
        "expected_attempts": 2 if baseline else 1,
        "scored_attempts": 2 if baseline else 1,
        "num_trials": 1,
        "num_trials_baseline": int(baseline),
        "with_skill": scores,
        "rewards": [reward],
        "rewards_complete": True,
        "conditions": {"with_skill": _SUCCEEDED, "without_skill": {"execution_status": "skipped"}},
    }
    if baseline:
        baseline_scores = dict.fromkeys(DEFAULT_METRICS, 0.5)
        agent.update(
            without_skill=baseline_scores,
            rewards_baseline=[{**baseline_scores, "entry_id": "case-1", "trial_id": "case-1__without"}],
            rewards_baseline_complete=True,
        )
        agent["conditions"]["without_skill"] = _SUCCEEDED
    return agent


def _payload(agents: dict[str, dict]) -> dict:
    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)
    assert payload is not None
    return payload


def _render(payload: dict) -> tuple[str, str]:
    result = _validation_result_from_payload(payload)
    assert result is not None
    markdown = BenchmarkReporter(include_timestamp=False).render_all([result])
    html = HTMLReporter(include_timestamp=False).render_all([result])
    return markdown, html


def _row(markdown: str, label: str) -> str:
    glance = markdown.split("## Results at a Glance", 1)[1].split("## Tier Status", 1)[0]
    return next(line for line in glance.splitlines() if line.startswith(f"| {label} |"))


def _methodology_row(markdown: str, label: str) -> list[str]:
    methodology = markdown.split("## Scoring Methodology", 1)[1]
    line = next(line for line in methodology.splitlines() if line.startswith(f"| {label} |"))
    return [cell.strip() for cell in line.strip("|").split("|")]


def _dimension_table(html: str) -> str:
    match = re.search(r'<table class="summary-table" id="tier3-dimension-table">.*?</table>', html, re.DOTALL)
    assert match is not None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", match.group(0)))


def _dimension_row(html: str, label: str) -> str:
    table = re.search(r'<table[^>]*id="tier3-dimension-table".*?</table>', html, re.DOTALL)
    return next(
        row for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table[0], re.DOTALL) if f"<strong>{label}</strong>" in row
    )


def _dimension_definition(html: str, label: str) -> str:
    row = _dimension_row(html, label)
    name_cell = re.search(r"<td>(.*?)</td>", row, re.DOTALL)[1]
    match = re.search(
        r'<details[^>]*class="[^"]*t3-dimension-definition[^"]*"[^>]*>.*?</details>', name_cell, re.DOTALL
    )
    assert match is not None, f"{label} needs a definition beneath its name"
    assert name_cell.index(f"<strong>{label}</strong>") < match.start()
    return match[0]


def _assert_gate_accepts(markdown: str, tmp_path: Path) -> None:
    card = tmp_path / "BENCHMARK.md"
    card.write_text(markdown, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([card])
    assert offenders == []


def test_benchmark_methodology_explains_meaning_evidence_and_baseline(tmp_path: Path) -> None:
    markdown, _html = _render(_payload({"paired": _agent(), "solo": _agent(baseline=False)}))

    assert "| Dimension | What it measures | Scored signals | Evidence source | Baseline in this run |" in markdown
    security = _methodology_row(markdown, "Security")
    assert security[1] == f"Is it safe to use? {DIMENSION_HINTS['security']}"
    assert security[3] == "Deterministic checks of the agent's commands and outputs"
    assert security[4] == "Baseline for 1 of 2 agents"
    effectiveness = _methodology_row(markdown, "Effectiveness")
    assert effectiveness[3] == "An LLM judge of whether the user's goal was met; an LLM judge of the expected workflow"
    for term in ("**Overall verdict**", "**Uplift**", "**pass@k**", "**Quality Score** (Tier 1"):
        assert term in markdown
    assert (
        "**Uplift**: the displayed skill score minus the displayed no-skill baseline score. Published Skill Lift "
        "compares Overall only on the metrics and cases both arms scored, so it can differ. Uplift does not set the "
        "verdict."
    ) in markdown
    _assert_gate_accepts(markdown, tmp_path)


def test_benchmark_methodology_states_when_no_baseline_ran() -> None:
    markdown, _html = _render(_payload({"solo": _agent(baseline=False)}))

    assert _methodology_row(markdown, "Correctness")[4] == "No baseline score; uplift unavailable"
    assert _row(markdown, "Correctness").endswith("| 90.0% — baseline not run; uplift unavailable |")


def test_benchmark_verdict_methodology_includes_with_skill_required_checks() -> None:
    result = _validation_result_from_payload(_payload({"agent": _agent()}))
    assert result is not None
    markdown = BenchmarkReporter(include_timestamp=False).render_all([result])

    assert "with no with-skill case failing a required goal-accuracy or security check" in markdown
    assert (
        "A scored unmet goal or security issue in any attempt fails its case and that agent's quality "
        "verdict, even if all dimensions pass."
    ) in markdown
    assert "Baseline-arm required-check failures do not change this quality verdict." in markdown


def test_html_verdict_methodology_includes_with_skill_required_checks() -> None:
    result = _validation_result_from_payload(_payload({"agent": _agent()}))
    assert result is not None
    html = HTMLReporter(include_timestamp=False).render_all([result])

    assert "has no with-skill case failing a required goal-accuracy or security check" in html
    assert (
        "A scored unmet goal or security issue in any attempt fails its case and that agent's quality "
        "verdict, even if all dimensions pass."
    ) in html
    assert "Baseline-arm required-check failures do not change this quality verdict." in html


@pytest.mark.parametrize("scope", ["publication", "tier3"])
@pytest.mark.parametrize("baseline", [True, False])
def test_overall_row_methodology_describes_comparison_basis(scope: str, baseline: bool) -> None:
    result = _validation_result_from_payload(_payload({"agent": _agent(baseline=baseline)}))
    markdown = BenchmarkReporter(report_scope=scope, include_timestamp=False).render(result)

    assert (
        "**Overall** row: when comparable baseline scores are available, the two scores used for published "
        "Skill Lift, on the metrics and cases both arms scored comparably. Otherwise, the available arm "
        "Overall score(s). These can differ from the headline with-skill score."
    ) in markdown
    assert "**Overall** row: the equal-weight mean of the five dimension scores with the skill." not in markdown


def test_unscored_dimension_is_explained_not_zero(tmp_path: Path) -> None:
    markdown, html = _render(_payload({"agent": _agent(with_scores={"accuracy": None})}))

    assert _row(markdown, "Correctness") == (
        "| Correctness | Not scored with the skill — baseline 50.0%; uplift unavailable |"
    )
    assert "**Not scored** means a signal the dimension needs produced no score" in markdown
    _assert_gate_accepts(markdown, tmp_path)

    table = _dimension_table(html)
    assert "Correctness" in table
    row = _dimension_row(html, "Correctness")
    assert 'title="not scored with the skill"' in row
    assert 'title="with-skill unavailable"' in row
    assert re.search(r">0.50<|>\s*0.50\s*<", row)
    assert "No verdict" in row
    assert "With-skill score unavailable; no verdict was computed." in table


def test_missing_and_not_applicable_dimensions_have_reasons(tmp_path: Path) -> None:
    payload = _payload({"agent": _agent()})
    agent = payload["agents"]["agent"]
    agent["dimensions"] = [d for d in agent["dimensions"] if d["id"] not in {"discoverability", "efficiency"}]
    reason = "Not applicable: Skill Execution had nothing to judge against in any eval case"
    agent["not_applicable_dimensions"] = [{"id": "discoverability", "reason": reason}]
    payload["dimensions"] = deepcopy(agent["dimensions"])
    payload["not_applicable_dimensions"] = deepcopy(agent["not_applicable_dimensions"])

    markdown, html = _render(payload)

    assert _row(markdown, "Discoverability") == (
        "| Discoverability | Not applicable — no eval case gave its signals anything to judge |"
    )
    assert (
        _row(markdown, "Efficiency") == "| Efficiency | Not scored — a signal this dimension needs produced no score |"
    )
    _assert_gate_accepts(markdown, tmp_path)

    table = _dimension_table(html)
    assert "Discoverability" in table
    assert "N/A N/A N/A Not applicable" in table
    assert "Skill Efficiency" in table
    row = _dimension_row(html, "Skill Efficiency")
    for missing_reason in ("not scored with the skill", "baseline quality unavailable", "with-skill unavailable"):
        assert missing_reason in row
    assert "No verdict" in row
    assert "A signal this dimension needs produced no score." in table
    assert f"{reason}. It has no score, so the agent cannot PASS." in table
    assert "NEUTRAL" not in table


def test_not_applicable_dimension_row_does_not_contradict_a_failing_verdict() -> None:
    # A scored dimension below 0.40 fails the agent even when another dimension
    # is Not applicable, so the row names only the shared outcome.
    payload = _payload({"agent": _agent(with_scores={"skill_efficiency": 0.1})})
    assert payload["verdict"] == "fail"
    agent = payload["agents"]["agent"]
    agent["dimensions"] = [d for d in agent["dimensions"] if d["id"] != "correctness"]
    agent["not_applicable_dimensions"] = [{"id": "correctness", "reason": ""}]
    payload["dimensions"] = deepcopy(agent["dimensions"])
    payload["not_applicable_dimensions"] = deepcopy(agent["not_applicable_dimensions"])

    _markdown, html = _render(payload)

    assert "Tier 3 quality: Changes required" in re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    table = _dimension_table(html)
    row = table[table.index("N/A N/A N/A Not applicable") :].split("Overall", 1)[0]
    assert (
        "Not applicable: its signals had nothing to judge in any eval case. It has no score, so the agent cannot PASS."
    ) in row
    assert "verdict stays NEUTRAL" not in row


def test_html_dimension_table_names_evidence_and_baseline_status() -> None:
    _markdown, html = _render(_payload({"solo": _agent(baseline=False)}))

    table = _dimension_table(html)
    assert f"Evidence: {SIGNAL_EVIDENCE_SOURCES['security']}." in table.replace("&#39;", "'")
    row = _dimension_row(html, "Security")
    assert "0.90" in row and "PASS" in row
    assert 'title="baseline not run"' in row and 'title="no baseline score"' in row
    assert "N/A means no measurement, never a score of zero; the note under it says why." in html


def test_html_dimension_definitions_are_discoverable_beneath_each_dimension_name() -> None:
    payload = _payload({"agent": _agent()})
    # Older retained reports do not always record the hint mapping.
    payload.pop("dimension_hints")
    _markdown, html = _render(payload)

    assert 'id="tier3-dimension-definitions"' not in html
    for label, meaning in (
        ("Security", "Avoids unsafe operations, secret leakage, and unauthorized access."),
        ("Correctness", "Produces factually accurate output."),
        ("Discoverability", "Activates the right skill when relevant; ignores unrelated skills."),
        ("Effectiveness", "Reaches the user's goal and follows the expected behavior."),
        ("Skill Efficiency", "Uses the skill and tools productively without wasted work."),
    ):
        definition = _dimension_definition(html, label)
        assert f'<summary aria-label="Definition of {label}">Definition</summary>' in definition
        assert "open" not in definition.split(">", 1)[0]
        assert meaning in text_content(definition)
        # Reasoning retains scope and evidence without a duplicate definition.
        assert text_content(_dimension_row(html, label)).count(meaning) == 1


def test_html_dimension_definitions_remain_available_for_unscored_inapplicable_and_peer_only_rows() -> None:
    payload = _payload({"agent": _agent(baseline=False, with_scores={"accuracy": None})})
    agent = payload["agents"]["agent"]
    payload["agents"]["peer"] = deepcopy(agent)
    agent["dimensions"] = [d for d in agent["dimensions"] if d["id"] not in {"discoverability", "efficiency"}]
    agent["metrics_with_skill"] = [m for m in agent["metrics_with_skill"] if m != "skill_efficiency"]
    agent["not_applicable_dimensions"] = [{"id": "discoverability", "reason": "No relevant skill case"}]
    payload["dimensions"] = deepcopy(agent["dimensions"])
    payload["not_applicable_dimensions"] = deepcopy(agent["not_applicable_dimensions"])

    _markdown, html = _render(payload)

    assert "Produces factually accurate output." in text_content(_dimension_definition(html, "Correctness"))
    assert "Activates the right skill when relevant; ignores unrelated skills." in text_content(
        _dimension_definition(html, "Discoverability")
    )
    assert "Uses the skill and tools productively without wasted work." in text_content(
        _dimension_definition(html, "Skill Efficiency")
    )
    assert 'title="not scored with the skill"' in _dimension_row(html, "Correctness")
    assert "Not applicable" in _dimension_row(html, "Discoverability")
    assert "Not measured" in _dimension_row(html, "Skill Efficiency")
    assert "No measurement was recorded for the primary agent" in _dimension_row(html, "Skill Efficiency")


def test_html_dimension_definitions_escape_recorded_text_and_skip_missing_hints() -> None:
    payload = _payload({"agent": _agent()})
    recorded_hint = '<img src=x onerror="alert(1)"> & "recorded definition"'
    payload["dimension_hints"] = {"security": recorded_hint, "correctness": None}

    _markdown, html = _render(payload)

    definition = _dimension_definition(html, "Security")
    assert recorded_hint in text_content(definition)
    assert "<img" not in definition
    assert "&lt;img" in definition and "&amp;" in definition
    table = re.search(r'<table[^>]*id="tier3-dimension-table".*?</table>', html, re.DOTALL)[0]
    assert table.count('class="t3-table-detail t3-dimension-definition"') == 1
    assert recorded_hint not in text_content(_dimension_row(html, "Security").split("</td>", 1)[1])
    assert "None" not in definition


def test_html_number_key_separates_verdict_scores_lift_and_quality() -> None:
    _markdown, html = _render(_payload({"agent": _agent()}))

    key = html.split('id="tier3-number-key"', 1)[1].split("</details>", 1)[0]
    for term in ("Verdict", "Overall Score", "Skill Lift", "pass@k", "Quality Score", "Publication eligibility"):
        assert f"<strong>{term}</strong>" in key
    assert (
        "Skill Lift</strong>: compares Overall on the metrics and cases both arms scored, so it can differ "
        "from subtracting the displayed Overall scores."
    ) in key
    assert "not an input to any live-evaluation score" in key


def test_baseline_only_cells_show_the_baseline_without_a_skill_score(tmp_path: Path) -> None:
    payload = _payload({"agent": _agent()})
    payload["mode"] = "baseline_only"
    agent = payload["agents"]["agent"]
    agent["with_skill"] = None
    for dimension in agent["dimensions"]:
        dimension.update(with_skill=None, score=None, lift=None)

    markdown, _html = _render(payload)

    assert (
        _row(markdown, "Overall") == "| Overall | Baseline 50.0% — baseline-only run; no with-skill score or uplift |"
    )
    assert _row(markdown, "Security") == (
        "| Security | Baseline 50.0% — baseline-only run; no with-skill score or uplift |"
    )
    assert _methodology_row(markdown, "Security")[4] == "Baseline only; no with-skill arm ran"
    _assert_gate_accepts(markdown, tmp_path)


def test_html_does_not_add_unscored_rows_for_custom_only_metrics() -> None:
    payload = _payload({"agent": _agent()})
    payload["agents"]["agent"]["metrics_with_skill"] = ["custom_metric"]
    payload["agents"]["agent"]["dimensions"] = []
    payload["dimensions"] = []
    payload["not_applicable_dimensions"] = []

    _markdown, html = _render(payload)

    assert 'id="tier3-dimension-table"' not in html
    assert "t3-dimension-definition" not in html


def test_benchmark_missing_dimension_requires_a_successful_active_signal() -> None:
    active = {
        "execution_status": "succeeded",
        "dimensions": [],
        "metrics_with_skill": ["accuracy"],
    }
    assert _dimension_cell(active, "correctness") == "Not scored — a signal this dimension needs produced no score"
    assert _baseline_coverage({"agent": active}, "correctness", baseline_only=False) == (
        "No baseline score; uplift unavailable"
    )


def test_benchmark_does_not_blame_missing_signals_for_custom_failed_or_legacy_agents() -> None:
    agents = [
        {"execution_status": "succeeded", "dimensions": [], "metrics_with_skill": ["custom_metric"]},
        {"execution_status": "failed", "dimensions": [], "metrics_with_skill": ["accuracy"]},
        {"execution_status": "succeeded", "dimensions": []},
    ]
    for agent in agents:
        assert _dimension_cell(agent, "correctness") == "Not available"
        assert _baseline_coverage({"agent": agent}, "correctness", baseline_only=False) == "Not available"
