# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public report tables must preserve each agent's activation eligibility."""

from __future__ import annotations

import re

import pytest

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
from skillevaluator.reporting import BenchmarkReporter, HTMLReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def _agent(*, activated: bool | None, score: float = 0.4, baseline: bool = True) -> dict:
    scores = dict.fromkeys(DEFAULT_METRICS, score)
    scores["skill_execution"] = float(activated) if activated is not None else score
    reward = {**scores, "entry_id": "case", "trial_id": "case__attempt1", "details": {}}
    if activated is not None:
        reward["details"] = {"skill_execution": {"activation": {"passed": activated, "score": float(activated)}}}
    agent = {
        "execution_status": "succeeded",
        "expected_attempts": 2 if baseline else 1,
        "scored_attempts": 2 if baseline else 1,
        "num_trials": 1,
        "num_trials_baseline": int(baseline),
        "with_skill": scores,
        "rewards": [reward],
    }
    if baseline:
        baseline_scores = dict.fromkeys(DEFAULT_METRICS, 0.8)
        agent["without_skill"] = baseline_scores
        agent["rewards_baseline"] = [{**baseline_scores, "trial_id": "baseline__attempt1"}]
    else:
        agent["conditions"] = {"without_skill": {"execution_status": "skipped"}}
    return agent


def _render(agents: dict[str, dict]) -> tuple[dict, str, str]:
    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)
    assert payload is not None
    result = _validation_result_from_payload(payload)
    assert result is not None
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])
    html = HTMLReporter(include_timestamp=False).render_all([result])
    return payload, benchmark, html


def _benchmark_rows(benchmark: str) -> dict[str, list[str]]:
    section = benchmark.split("## Results at a Glance\n", 1)[1].split("\n## ", 1)[0]
    return {
        cells[0]: cells[1:]
        for line in section.splitlines()
        if line.startswith("| ")
        for cells in [[cell.strip() for cell in line.strip("|").split("|")]]
        if cells[0] != "Measure"
    }


@pytest.mark.parametrize("baseline", [True, False])
@pytest.mark.parametrize("score", [0.2, 0.9])
def test_benchmark_never_activated_scores_do_not_reconstruct_uplift(baseline: bool, score: float) -> None:
    payload, benchmark, _ = _render({"unactivated": _agent(activated=False, score=score, baseline=baseline)})
    agent = payload["agents"]["unactivated"]
    assert agent["lift"] is None
    assert agent["verdict"] == "inconclusive"
    rows = _benchmark_rows(benchmark)
    scores = {"Overall": agent["with_skill"], **{d["id"].title(): d["with_skill"] for d in agent["dimensions"]}}
    assert len(rows) == len(DIMENSION_MAPPING) + 1
    for measure, cells in rows.items():
        assert f"{scores[measure]:.0%}" in cells[0]
        assert "INCONCLUSIVE: skill never activated" in cells[0]
        assert "uplift unavailable" in cells[0]
        assert "points" not in cells[0]
        assert "baseline not run" not in cells[0]


def _mixed_agents() -> dict[str, dict]:
    return {
        "unactivated": _agent(activated=False, score=0.99),
        "activated": _agent(activated=True, score=0.5),
        "no-baseline": _agent(activated=True, score=0.7, baseline=False),
    }


def test_benchmark_mixed_agents_keep_inconclusive_missing_baseline_and_measured_lift_distinct() -> None:
    payload, benchmark, _ = _render(_mixed_agents())
    assert payload["agents"]["unactivated"]["with_skill"] > payload["agents"]["activated"]["with_skill"]
    for cells in _benchmark_rows(benchmark).values():
        by_agent = dict(zip(payload["agents"], cells, strict=True))
        assert "INCONCLUSIVE: skill never activated" in by_agent["unactivated"]
        assert "points" not in by_agent["unactivated"]
        assert "→" in by_agent["activated"] and "points" in by_agent["activated"]
        assert "INCONCLUSIVE" not in by_agent["activated"]
        assert "baseline not run; uplift unavailable" in by_agent["no-baseline"]


def test_html_mixed_agents_identify_activation_suppression_in_every_grid_cell() -> None:
    payload, _, html = _render(_mixed_agents())
    # The HTML grid interleaves one cell per agent for every dimension and Overall.
    labels = re.findall(r'<span class="t3-mad-lift"[^>]*>(.*?)</span>', html, flags=re.DOTALL)
    assert len(labels) == 3 * (len(DIMENSION_MAPPING) + 1)
    by_agent = {name: labels[index::3] for index, name in enumerate(payload["agents"])}
    assert by_agent["unactivated"] == ["inconclusive: skill never activated"] * (len(DIMENSION_MAPPING) + 1)
    assert all(re.fullmatch(r"[+-]\d+\.\d{2} lift", label) for label in by_agent["activated"])
    assert by_agent["no-baseline"] == ["no baseline lift"] * (len(DIMENSION_MAPPING) + 1)


def test_unknown_activation_keeps_legacy_comparisons() -> None:
    payload, benchmark, _ = _render({"legacy": _agent(activated=None)})
    assert payload["agents"]["legacy"]["skill_activation"]["status"] == "unknown"
    for cells in _benchmark_rows(benchmark).values():
        assert "→" in cells[0] and "points" in cells[0]
        assert "INCONCLUSIVE" not in cells[0]


@pytest.mark.parametrize("baseline", [True, False])
def test_html_grid_names_never_activation_with_or_without_baseline(baseline: bool) -> None:
    payload, _, html = _render(
        {
            "unactivated": _agent(activated=False, baseline=baseline),
            "activated": _agent(activated=True),
        }
    )
    labels = re.findall(r'<span class="t3-mad-lift"[^>]*>(.*?)</span>', html, flags=re.DOTALL)
    assert len(labels) == 2 * (len(DIMENSION_MAPPING) + 1)
    by_agent = {name: labels[index::2] for index, name in enumerate(payload["agents"])}
    assert by_agent["unactivated"] == ["inconclusive: skill never activated"] * (len(DIMENSION_MAPPING) + 1)
    assert all(re.fullmatch(r"[+-]\d+\.\d{2} lift", label) for label in by_agent["activated"])
