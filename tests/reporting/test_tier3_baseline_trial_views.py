# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 steps chart and per-evaluator drill-down show both evaluation conditions."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting import html as html_module
from skillevaluator.reporting.trial_charts import steps_by_case_chart
from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    LEGACY_METRIC_SET,
    LEGACY_METRICS,
)

METRICS = list(DEFAULT_METRICS)


def test_steps_chart_averages_retries_within_each_condition() -> None:
    chart = steps_by_case_chart(
        {"codex": {"trials_baseline": [{"entry_id": "case-1", "steps": 40}]}},
        [
            {"agent": "codex", "entry_id": "case-1", "steps": 71},
            {"agent": "codex", "entry_id": "case-1", "steps": 73},
        ],
    )

    assert chart == {
        "labels": ["case-1"],
        "datasets": [
            {"label": "codex · With skill", "data": [72.0]},
            {"label": "codex · Without skill", "data": [40.0]},
        ],
    }


def test_steps_chart_omits_unmeasured_series_and_keeps_case_gaps() -> None:
    chart = steps_by_case_chart(
        {
            "codex": {"trials_baseline": [{"entry_id": "baseline-only", "steps": 8}, {"entry_id": "no-telemetry"}]},
            "opencode": {"trials_baseline": []},
        },
        [
            {"agent": "codex", "entry_id": "with-only", "steps": 12},
            {"agent": "opencode", "entry_id": "unmeasured", "steps": None},
        ],
        with_label="With skill",
        without_label="Without skill",
    )

    assert chart == {
        "labels": ["with-only", "baseline-only", "no-telemetry"],
        "datasets": [
            {"label": "codex · With skill", "data": [12.0, None, None]},
            {"label": "codex · Without skill", "data": [None, 8.0, None]},
        ],
    }


def test_steps_chart_does_not_average_partially_measured_cases() -> None:
    chart = steps_by_case_chart(
        {},
        [
            {"agent": "codex", "entry_id": "partial", "steps": 10},
            {"agent": "codex", "entry_id": "partial"},
            {"agent": "codex", "entry_id": "complete", "steps": 4},
        ],
    )

    assert chart["datasets"] == [{"label": "codex · With skill", "data": [None, 4.0]}]


def test_steps_chart_rejects_non_counter_values() -> None:
    chart = steps_by_case_chart(
        {},
        [
            {"entry_id": "zero", "steps": 0},
            {"entry_id": "text", "steps": "12"},
            {"entry_id": "bool", "steps": True},
            {"entry_id": "negative", "steps": -1},
            {"entry_id": "nan", "steps": math.nan},
            {"entry_id": "overflow", "steps": 10**1000},
        ],
    )

    assert chart == {
        "labels": ["zero", "text", "bool", "negative", "nan", "overflow"],
        "datasets": [{"label": "(default) · With skill", "data": [0.0, None, None, None, None, None]}],
    }


def test_steps_chart_average_stays_finite_for_large_retries() -> None:
    chart = steps_by_case_chart(
        {},
        [
            {"agent": "codex", "entry_id": "large", "steps": 1e308},
            {"agent": "codex", "entry_id": "large", "steps": 1e308},
        ],
    )

    assert chart["datasets"][0]["data"] == [1e308]


def test_steps_chart_caps_cases_and_drops_series_measured_only_past_the_cap() -> None:
    chart = steps_by_case_chart(
        {"codex": {"trials_baseline": [{"entry_id": "case-3", "steps": 9}]}},
        [
            {"agent": "codex", "entry_id": "case-1", "steps": 1},
            {"agent": "codex", "entry_id": "case-2", "steps": 2},
        ],
        max_cases=1,
    )

    assert chart == {
        "labels": ["case-1"],
        "datasets": [{"label": "codex · With skill", "data": [1.0]}],
        "omitted_cases": 2,
    }


def _reward(trial_id: str, entry_id: str, *, score: float, steps: int, prompt: int) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "entry_id": entry_id,
        **dict.fromkeys(METRICS, score),
        "_traj": {"steps": steps, "prompt_tokens": prompt, "completion_tokens": 10, "cached_tokens": 0},
    }


def _payload(*, include_baseline: bool = True) -> dict[str, Any]:
    with_rewards = [_reward("case-1__attempt1", "case-1", score=0.0, steps=71, prompt=100)]
    baseline_rewards = (
        [_reward("case-1__baseline1", "case-1", score=1.0, steps=40, prompt=60)] if include_baseline else []
    )
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "expected_attempts": len(with_rewards) + len(baseline_rewards),
                "scored_attempts": len(with_rewards) + len(baseline_rewards),
                "with_skill": dict.fromkeys(METRICS, 0.0),
                "without_skill": dict.fromkeys(METRICS, 1.0) if include_baseline else {},
                "rewards": with_rewards,
                "rewards_baseline": baseline_rewards,
                "num_trials": len(with_rewards),
                "num_trials_baseline": len(baseline_rewards),
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None
    return payload


def _render(payload: dict[str, Any]) -> str:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    return HTMLReporter(include_timestamp=False).render_all([result])


def _drilldown_table(html: str) -> str:
    match = re.search(r'<table class="summary-table" id="t3-trials-table">(.*?)</table>', html, re.DOTALL)
    assert match is not None
    return match.group(1)


def _row(table: str, condition: str) -> str:
    match = re.search(rf"<tr>(?:(?!</tr>).)*<td>{condition}</td>.*?</tr>", table, re.DOTALL)
    assert match is not None
    return match.group(0)


def _cell(table: str, condition: str, column: str) -> str:
    headers = re.findall(r"<th>(.*?)</th>", table)
    cells = re.findall(r"<td[^>]*>(.*?)</td>", _row(table, condition), re.DOTALL)
    return re.sub(r"<[^>]+>", "", cells[headers.index(column)]).strip()


def _steps_chart(html: str) -> dict[str, Any]:
    match = re.search(r"function stepsByCaseDatasets\(\) \{.*?var chart = (.*?);\n", html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_trial_charts_include_agent_condition_series() -> None:
    html = _render(_payload())

    assert "Steps per Eval Case by Agent and Condition" in html
    assert _steps_chart(html) == {
        "labels": ["case-1"],
        "datasets": [
            {"label": "codex · With skill", "data": [71.0]},
            {"label": "codex · Without skill", "data": [40.0]},
        ],
    }


def test_drilldown_renders_with_skill_and_baseline_rows() -> None:
    html = _render(_payload())
    table = _drilldown_table(html)

    assert "<th>Agent</th><th>Condition</th><th>Entry</th><th>Overall</th>" in table
    assert table.count("<tr>") == 3
    assert ">0.00<" in _row(table, "With skill")
    assert re.search(r"<td>71</td>\s*<td>110</td>", _row(table, "With skill"))
    assert ">1.00<" in _row(table, "Without skill")
    assert re.search(r"<td>40</td>\s*<td>70</td>", _row(table, "Without skill"))
    assert 'data-tier3-filter-count="t3-trials-table">2 / 2' in html
    assert "<summary>Per-Evaluator Drill-Down</summary>" in html


def test_missing_baseline_shows_available_condition_only() -> None:
    html = _render(_payload(include_baseline=False))
    table = _drilldown_table(html)

    assert table.count("<tr>") == 2
    assert "<td>With skill</td>" in table
    assert "<td>Without skill</td>" not in table
    assert [dataset["label"] for dataset in _steps_chart(html)["datasets"]] == ["codex · With skill"]


def test_baseline_rows_render_without_with_skill_rows() -> None:
    payload = _payload()
    payload["trials"] = []
    payload["agents"]["codex"]["trials"] = []

    html = _render(payload)
    table = _drilldown_table(html)

    assert table.count("<tr>") == 2
    assert "<td>Without skill</td>" in table
    assert "No per-trial data available for this run." not in html
    assert [dataset["label"] for dataset in _steps_chart(html)["datasets"]] == ["codex · Without skill"]


def test_drilldown_adds_a_column_for_a_metric_only_baseline_rows_scored() -> None:
    # With-skill rewards use the legacy metric set, so the report's metric ids
    # do not list security; only the baseline rows scored it.
    with_rewards = [
        {"trial_id": "case-1__attempt1", "entry_id": "case-1", "metric_set": LEGACY_METRIC_SET}
        | dict.fromkeys(LEGACY_METRICS, 0.9)
    ]
    baseline_rewards = [
        {
            "trial_id": "case-1__baseline1",
            "entry_id": "case-1",
            "metric_set": DEFAULT_METRIC_SET,
            **dict.fromkeys(DEFAULT_METRICS, 0.9),
            "security": 0.0,
            "details": {"security": {"passed": False, "score": 0.0, "reason": "Baseline ran rm -rf /"}},
        }
    ]
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "expected_attempts": 2,
                "scored_attempts": 2,
                "with_skill": dict.fromkeys(LEGACY_METRICS, 0.9),
                "without_skill": {**dict.fromkeys(DEFAULT_METRICS, 0.9), "security": 0.0},
                "rewards": with_rewards,
                "rewards_baseline": baseline_rewards,
                "num_trials": 1,
                "num_trials_baseline": 1,
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None
    assert "security" not in payload["metric_ids"]

    table = _drilldown_table(_render(payload))

    headers = re.findall(r"<th>(.*?)</th>", table)
    assert headers[-4:] == ["Security", "Steps", "Tokens", "Warnings"]
    assert _cell(table, "Without skill", "Security") == "0.00"
    assert 'title="Security: Baseline ran rm -rf /"' in _row(table, "Without skill")
    assert _cell(table, "With skill", "Security") == "—"


def _na(reward: dict[str, Any], *metrics: str) -> dict[str, Any]:
    """Record *metrics* as N/A the way the Harbor verifier does: no score plus the sidecar marker."""
    details = reward.setdefault("details", {})
    for metric in metrics:
        reward[metric] = None
        details[metric] = {"score": None, "status": "not_applicable", "reason": "No reference"}
    return reward


def _raw_cell(table: str, condition: str, column: str) -> str:
    headers = re.findall(r"<th>(.*?)</th>", table)
    cells = re.findall(r"<td[^>]*>.*?</td>", _row(table, condition), re.DOTALL)
    return cells[headers.index(column)]


def test_drilldown_reads_na_for_not_applicable_trial_cells() -> None:
    # Each arm records a metric as N/A that the other arm scores, so both keep its column.
    with_reward = _na(_reward("case-1__attempt1", "case-1", score=0.9, steps=5, prompt=10), "accuracy")
    # A metric that is missing without the N/A marker is unscored, not N/A.
    with_reward["goal_accuracy"] = None
    baseline_reward = _na(
        _reward("case-1__baseline1", "case-1", score=0.5, steps=4, prompt=10), "skill_execution", "skill_efficiency"
    )
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "expected_attempts": 2,
                "scored_attempts": 2,
                "with_skill": {**dict.fromkeys(METRICS, 0.9), "accuracy": None},
                "without_skill": {**dict.fromkeys(METRICS, 0.5), "skill_execution": None, "skill_efficiency": None},
                "rewards": [with_reward],
                "rewards_baseline": [baseline_reward],
                "num_trials": 1,
                "num_trials_baseline": 1,
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None

    table = _drilldown_table(_render(payload))

    assert _cell(table, "With skill", "Accuracy") == "N/A"
    assert 'data-not-applicable="true"' in _raw_cell(table, "With skill", "Accuracy")
    assert 'title="Accuracy: Not applicable: No expected_output (legacy ground_truth) defined' in _raw_cell(
        table, "With skill", "Accuracy"
    )
    assert _cell(table, "With skill", "Goal Accuracy") == "—"
    assert "data-not-applicable" not in _raw_cell(table, "With skill", "Goal Accuracy")
    assert _cell(table, "With skill", "Skill Execution") == "0.90"
    for column in ("Skill Execution", "Skill Efficiency"):
        assert _cell(table, "Without skill", column) == "N/A"
        assert 'data-not-applicable="true"' in _raw_cell(table, "Without skill", column)
    assert 'title="Skill Execution: Not applicable: No applicable skill is available in this arm' in _raw_cell(
        table, "Without skill", "Skill Execution"
    )
    assert _cell(table, "Without skill", "Accuracy") == "0.50"


def test_drilldown_explains_with_skill_skill_metric_na_by_the_eval_case() -> None:
    # With the skill installed, a negative case leaves Efficiency N/A and a positive case scores it.
    # The baseline has no skill at all.
    with_rewards = [
        _na(_reward("negative__attempt1", "negative", score=0.9, steps=5, prompt=10), "skill_efficiency"),
        _reward("positive__attempt1", "positive", score=0.9, steps=5, prompt=10),
    ]
    baseline_rewards = [
        _na(_reward(f"{case}__baseline1", case, score=0.5, steps=4, prompt=10), "skill_execution", "skill_efficiency")
        for case in ("negative", "positive")
    ]
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "expected_attempts": 4,
                "scored_attempts": 4,
                "with_skill": dict.fromkeys(METRICS, 0.9),
                "without_skill": {**dict.fromkeys(METRICS, 0.5), "skill_execution": None, "skill_efficiency": None},
                "rewards": with_rewards,
                "rewards_baseline": baseline_rewards,
                "num_trials": 2,
                "num_trials_baseline": 2,
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None

    table = _drilldown_table(_render(payload))

    assert _cell(table, "With skill", "Skill Efficiency") == "N/A"
    assert (
        'title="Skill Efficiency: Not applicable: Negative case or no expected_skill defined, '
        'so there is no skill use to measure"'
    ) in _raw_cell(table, "With skill", "Skill Efficiency")
    assert _cell(table, "Without skill", "Skill Efficiency") == "N/A"
    assert 'title="Skill Efficiency: Not applicable: No applicable skill is available in this arm' in _raw_cell(
        table, "Without skill", "Skill Efficiency"
    )
    assert _cell(table, "With skill", "Skill Execution") == "0.90"


@pytest.mark.parametrize(
    ("recorded", "kept"),
    [
        (["security", "accuracy", "unknown"], ["accuracy"]),
        ("accuracy", []),
        ({"accuracy": True}, []),
    ],
    ids=["ineligible-names", "string", "mapping"],
)
def test_display_keeps_only_eligible_na_metric_names(recorded: object, kept: list[str]) -> None:
    payload = _payload()
    payload["agents"]["codex"]["trials_baseline"][0]["not_applicable_metrics"] = recorded
    payload["agents"]["codex"]["trials_baseline"][0]["scores"] = {}
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload

    data = HTMLReporter(include_timestamp=False)._tier3_report_data([result])

    assert data is not None
    assert data["agents"]["codex"]["trials_baseline"][0]["not_applicable_metrics"] == kept
    table = _drilldown_table(_render(payload))
    assert _cell(table, "Without skill", "Security") == "—"
    assert _cell(table, "Without skill", "Accuracy") == ("N/A" if kept else "—")


def test_drilldown_columns_include_metrics_scored_only_on_baseline_rows() -> None:
    payload = _payload()
    with_trial = payload["trials"][0]
    for score_key in ("scores", "baseline_scores", "lift_scores"):
        (with_trial.get(score_key) or {}).pop("accuracy", None)
    payload["agents"]["codex"]["trials_baseline"].append(
        {
            "trial_id": "baseline-only__attempt1",
            "entry_id": "baseline-only",
            "overall": 0.0,
            "scores": {"accuracy": 0.0},
        }
    )
    payload["agents"]["codex"]["trials_baseline"][0]["scores"].pop("accuracy", None)

    table = _drilldown_table(_render(payload))

    assert "<th>Accuracy</th>" in table
    baseline_row = re.search(r"<tr>(?:(?!</tr>).)*baseline-only.*?</tr>", table, re.DOTALL)
    assert baseline_row is not None
    assert ">0.00<" in baseline_row.group(0)


def test_missing_baseline_telemetry_renders_as_unavailable() -> None:
    payload = _payload()
    baseline_trial = payload["agents"]["codex"]["trials_baseline"][0]
    baseline_trial.pop("steps", None)
    baseline_trial.pop("tokens", None)

    html = _render(payload)

    assert re.search(r"<td>—</td>\s*<td>—</td>", _row(_drilldown_table(html), "Without skill"))
    assert [dataset["label"] for dataset in _steps_chart(html)["datasets"]] == ["codex · With skill"]


def test_legacy_trials_without_agent_use_default_label_in_table_and_chart() -> None:
    payload = _payload(include_baseline=False)
    payload["trials"][0].pop("agent", None)
    payload["agents"]["codex"]["trials"] = []

    html = _render(payload)
    table = _drilldown_table(html)

    assert "<td><strong>(default)</strong></td>" in table
    assert _steps_chart(html)["datasets"] == [{"label": "(default) · With skill", "data": [71.0]}]


@pytest.mark.parametrize(
    ("field", "value", "column"),
    [
        ("overall", "high", "Overall"),
        ("overall", 7.5, "Overall"),
        ("scores", {"accuracy": "0.5"}, "Accuracy"),
        ("scores", {"accuracy": 7.5}, "Accuracy"),
        ("scores", ["x"], "Accuracy"),
        ("steps", True, "Steps"),
        ("steps", -5, "Steps"),
        ("tokens", {"prompt": -100, "completion": 10}, "Tokens"),
    ],
)
def test_malformed_baseline_trial_values_render_as_unavailable(field: str, value: object, column: str) -> None:
    payload = _payload()
    payload["agents"]["codex"]["trials_baseline"][0][field] = value

    html = _render(payload)
    table = _drilldown_table(html)

    assert _cell(table, "Without skill", column) == "—"
    assert _cell(table, "With skill", column) != "—"
    if field == "steps":
        assert [dataset["label"] for dataset in _steps_chart(html)["datasets"]] == ["codex · With skill"]


def test_display_bounds_baseline_rows_without_adding_default_fields() -> None:
    payload = _payload()
    canonical_agent = payload["agents"]["codex"]
    canonical_agent["trials_baseline"][0]["warnings"] = "not-a-list"
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload

    data = HTMLReporter(include_timestamp=False)._tier3_report_data([result])

    assert data is not None
    agent = data["agents"]["codex"]
    # Baseline rows keep their recorded fields; no empty defaults grow the page.
    assert [set(trial) for trial in agent["trials_baseline"]] == [
        set(trial) for trial in canonical_agent["trials_baseline"]
    ]
    assert agent["trials_baseline"][0]["warnings"] == []
    # Nothing renders agents[*].trials, so it is left as recorded.
    assert agent["trials"] == canonical_agent["trials"]


def test_steps_chart_covers_every_trial_when_the_html_preview_is_bounded() -> None:
    # The bounded preview keeps 64 with-skill rows across all agents and blanks
    # strings once its character budget runs out; the chart must not follow it.
    infos: dict[str, Any] = {}
    for agent_index in range(3):
        arms = {
            arm: [
                _reward(f"{arm}-{agent_index}-{index}", f"case-{index:02d}", score=0.9, steps=10 + index, prompt=100)
                for index in range(60)
            ]
            for arm in ("rewards", "rewards_baseline")
        }
        infos[f"agent-{agent_index}"] = {
            "execution_status": "succeeded",
            "expected_attempts": 120,
            "scored_attempts": 120,
            "with_skill": dict.fromkeys(METRICS, 0.9),
            "without_skill": dict.fromkeys(METRICS, 0.9),
            **arms,
            "num_trials": 60,
            "num_trials_baseline": 60,
        }
    payload = build_agent_eval_payload("demo", infos, use_llm_judge=False)
    assert payload is not None
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    data = HTMLReporter(include_timestamp=False)._tier3_report_data([result])
    assert data is not None
    _preview, notice = html_module._bounded_tier3_preview(data)
    assert notice.get("trial_rows")

    chart = _steps_chart(_render(payload))

    assert chart["labels"] == [f"case-{index:02d}" for index in range(60)]
    assert [dataset["label"] for dataset in chart["datasets"]] == [
        f"agent-{agent_index} · {condition}"
        for agent_index in range(3)
        for condition in ("With skill", "Without skill")
    ]
    for dataset in chart["datasets"]:
        assert dataset["data"] == [float(10 + index) for index in range(60)]


def test_steps_chart_notes_cases_past_its_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    payload["trials"].append({**payload["trials"][0], "trial_id": "case-2__attempt1", "entry_id": "case-2"})
    monkeypatch.setattr(
        html_module,
        "steps_by_case_chart",
        lambda agents, trials: steps_by_case_chart(agents, trials, max_cases=1),
    )

    html = _render(payload)

    assert _steps_chart(html)["omitted_cases"] == 1
    assert "Showing the first 1 eval cases; 1 more are in the JSON Downloads." in html
