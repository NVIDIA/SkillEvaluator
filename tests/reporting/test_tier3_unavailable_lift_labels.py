# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recorded baseline without a comparable lift must not be labeled as missing."""

from __future__ import annotations

import re

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
from skillevaluator.reporting import HTMLReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, DEFAULT_SCORE_POLICY, LEGACY_SCORE_POLICY

_SUCCEEDED = {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}


def _reward(scores: dict[str, float], trial_id: str, judge: str) -> dict:
    details = {"accuracy": {"score": scores["accuracy"], "judge_model_used": judge}}
    return {**scores, "entry_id": "case-1", "trial_id": trial_id, "details": details}


def _agent(*, baseline: bool = True, judge_with: str = "primary") -> dict:
    with_scores = dict.fromkeys(DEFAULT_METRICS, 0.9)
    agent = {
        "execution_status": "succeeded",
        "expected_attempts": 2 if baseline else 1,
        "scored_attempts": 2 if baseline else 1,
        "num_trials": 1,
        "num_trials_baseline": int(baseline),
        "with_skill": with_scores,
        "rewards": [_reward(with_scores, "case-1__with", judge_with)],
        "rewards_complete": True,
        "conditions": {"with_skill": _SUCCEEDED},
    }
    if baseline:
        baseline_scores = dict.fromkeys(DEFAULT_METRICS, 0.5)
        agent.update(
            without_skill=baseline_scores,
            rewards_baseline=[_reward(baseline_scores, "case-1__without", "primary")],
            rewards_baseline_complete=True,
        )
        agent["conditions"]["without_skill"] = _SUCCEEDED
    else:
        agent["conditions"]["without_skill"] = {"execution_status": "skipped"}
    return agent


def _html(agents: dict[str, dict]) -> tuple[dict, str]:
    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)
    assert payload is not None
    result = _validation_result_from_payload(payload)
    assert result is not None
    return payload, HTMLReporter(include_timestamp=False).render_all([result])


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", fragment)).strip()


def _matrix_labels(html: str, agents: list[str]) -> dict[str, list[str]]:
    # The grid interleaves one cell per agent for every dimension, then Overall.
    labels = [_text(label) for label in re.findall(r'<span class="t3-mad-lift"[^>]*>(.*?)</span>', html, re.DOTALL)]
    assert len(labels) == len(agents) * (len(DIMENSION_MAPPING) + 1)
    return {name: labels[index :: len(agents)] for index, name in enumerate(agents)}


def _scorecard_labels(html: str) -> list[str]:
    pattern = r'<div class="t3-agent-score"[^>]*>.*?</div>\s*<div[^>]*>(.*?)</div>'
    return [_text(label) for label in re.findall(pattern, html, re.DOTALL)]


def _lift_kpi_labels(html: str) -> list[str]:
    pattern = r'<div class="t3-kpi-label">Skill Lift</div>.*?<div class="t3-kpi-sub">(.*?)</div>'
    return [_text(label) for label in re.findall(pattern, html, re.DOTALL)]


def test_mixed_judge_comparison_is_unavailable_not_missing_baseline() -> None:
    payload, html = _html({"matched": _agent(), "mixed": _agent(judge_with="fallback")})
    mixed = payload["agents"]["mixed"]
    correctness = next(dimension for dimension in mixed["dimensions"] if dimension["id"] == "correctness")
    assert (mixed["baseline"], mixed["lift"]) == (0.5, None)
    assert (correctness["baseline"], correctness["lift"]) == (0.5, None)

    labels = _matrix_labels(html, list(payload["agents"]))
    assert all(re.fullmatch(r"[+-]\d+\.\d{2} lift", label) for label in labels["matched"])
    dimension_labels = dict(zip(DIMENSION_MAPPING, labels["mixed"], strict=False))
    assert dimension_labels["correctness"] == "comparison unavailable"
    assert dimension_labels["security"] == "+0.40 lift"
    assert labels["mixed"][-1] == "comparison unavailable"
    assert _scorecard_labels(html) == ["+0.40 lift", "comparison unavailable"]
    assert _lift_kpi_labels(html) == ["positive", "comparison unavailable"]
    assert "no baseline" not in html


def test_score_policy_mismatch_is_unavailable_not_missing_baseline() -> None:
    agent = _agent()
    agent.update(
        score_policy_with_skill=DEFAULT_SCORE_POLICY,
        score_policy_without_skill=LEGACY_SCORE_POLICY,
        overall_with_skill=0.9,
        overall_without_skill=0.5,
    )
    payload, html = _html({"codex": agent})
    assert (payload["agents"]["codex"]["baseline"], payload["agents"]["codex"]["lift"]) == (0.5, None)
    assert "baseline 0.50" in html
    assert _scorecard_labels(html) == ["comparison unavailable"]
    assert _lift_kpi_labels(html) == ["comparison unavailable"]
    assert "no baseline" not in html


def test_skipped_baseline_keeps_no_baseline_labels() -> None:
    payload, html = _html({"compared": _agent(), "skipped": _agent(baseline=False)})
    assert payload["agents"]["skipped"]["baseline"] is None
    labels = _matrix_labels(html, list(payload["agents"]))
    assert labels["skipped"] == ["no baseline lift"] * (len(DIMENSION_MAPPING) + 1)
    assert _scorecard_labels(html) == ["+0.40 lift", "no baseline"]
    assert _lift_kpi_labels(html) == ["positive", "no baseline"]
    assert "comparison unavailable" not in html
