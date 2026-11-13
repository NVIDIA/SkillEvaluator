# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reports explain each dimension, its evidence and baseline, and why a score is missing."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from scripts.ci import check_public_benchmarks as benchmark_gate

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


def test_unscored_dimension_is_explained_not_zero(tmp_path: Path) -> None:
    markdown, html = _render(_payload({"agent": _agent(with_scores={"accuracy": None})}))

    assert _row(markdown, "Correctness") == (
        "| Correctness | Not scored with the skill — baseline 50.0%; uplift unavailable |"
    )
    assert "**Not scored** means a signal the dimension needs produced no score" in markdown
    _assert_gate_accepts(markdown, tmp_path)

    table = _dimension_table(html)
    assert "Correctness" in table
    assert "N/A not scored with the skill 0.50 N/A with-skill unavailable No verdict" in table
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
    assert "N/A not scored with the skill N/A baseline not scored N/A with-skill unavailable No verdict" in table
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
    row = table[table.index("N/A N/A N/A Not applicable") :]
    assert row.rstrip().endswith(
        "Not applicable: its signals had nothing to judge in any eval case. It has no score, so the agent cannot PASS."
    )
    for verdict in ("NEUTRAL", "FAIL", "INCONCLUSIVE"):
        assert verdict not in row


def test_reports_docs_name_only_the_shared_outcome_of_a_not_applicable_dimension() -> None:
    docs = " ".join((Path(__file__).resolve().parents[2] / "docs" / "reports.mdx").read_text(encoding="utf-8").split())
    sentences = [sentence for sentence in docs.split(". ") if "Not applicable" in sentence and "no score" in sentence]

    assert len(sentences) == 2
    for sentence in sentences:
        assert sentence.endswith("no score, so the agent cannot PASS")
        for verdict in ("NEUTRAL", "FAIL", "INCONCLUSIVE"):
            assert verdict not in sentence


def test_html_dimension_table_names_evidence_and_baseline_status() -> None:
    _markdown, html = _render(_payload({"solo": _agent(baseline=False)}))

    table = _dimension_table(html)
    assert f"Evidence: {SIGNAL_EVIDENCE_SOURCES['security']}." in table.replace("&#39;", "'")
    assert "0.90 N/A baseline not run N/A no baseline PASS" in table
    assert "N/A means no measurement, never a score of zero; the note under it says why." in html


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
