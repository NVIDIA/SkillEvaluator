# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy scores must be computed from the five outcome criteria in both runtimes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import load_harbor_eval_template

from skillevaluator.tier3.eval_core import llm_judge
from skillevaluator.tier3.harbor.collector import _compute_lift


@pytest.fixture(params=["host", "harbor"])
def accuracy_judge(request: pytest.FixtureRequest):
    if request.param == "host":
        return llm_judge
    return load_harbor_eval_template("accuracy_arithmetic_template")


def _criteria(passed: int) -> dict[str, bool]:
    return {key: index < passed for index, key in enumerate(sorted(llm_judge._ACCURACY_CRITERIA_KEYS))}


@pytest.mark.parametrize("passed", range(6))
def test_accuracy_derives_score_from_complete_criteria(monkeypatch, accuracy_judge, passed):
    payload = {"criteria": _criteria(passed), "score": 1 - passed / 5, "reason": "rubric verdicts"}
    monkeypatch.setattr(accuracy_judge, "call_public_llm", lambda *_args, **_kwargs: (json.dumps(payload), None))

    result = accuracy_judge.judge_accuracy("Summarize the CSV", "The total is 120", "The total is 120")

    assert result["score"] == pytest.approx(passed / 5)
    assert result["criteria"] == payload["criteria"]
    assert "error" not in result


def test_accuracy_criteria_grade_the_outcome_not_skill_use(accuracy_judge):
    # Skill use is scored by skill_execution. With no skill criterion, a correct
    # answer from the no-skill baseline is not capped below 1.0.
    outcome_criteria = {"ACTION_CORRECT", "FACTUALLY_ACCURATE", "TASK_ADDRESSED", "ACTIONABLE", "FORMAT_FOLLOWED"}
    schema = accuracy_judge.ACCURACY_JSON_SCHEMA

    assert set(accuracy_judge._ACCURACY_CRITERIA_KEYS) == outcome_criteria
    assert set(schema["properties"]["criteria"]["required"]) == outcome_criteria
    assert "score" not in schema["properties"]
    assert "score" not in schema["required"]


def test_tier3_docs_describe_criterion_derived_accuracy():
    docs = (Path(__file__).resolve().parents[1] / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")
    normalized = " ".join(docs.split())

    assert "passing criteria divided by five" in normalized
    assert "FORMAT_FOLLOWED" in normalized
    assert "falls back to the judge's numeric score" not in normalized
    assert "existing scoring contract" not in normalized


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"criteria": _criteria(3), "score": 0.6}, 0.6),
        ({"criteria": _criteria(3)}, 0.6),
        ({"criteria": _criteria(3), "score": 1.0}, 0.6),
    ],
)
def test_accuracy_ignores_the_judge_total(monkeypatch, accuracy_judge, payload, expected):
    monkeypatch.setattr(accuracy_judge, "call_public_llm", lambda *_args, **_kwargs: (json.dumps(payload), None))
    assert accuracy_judge.judge_accuracy("question", "reference", "answer")["score"] == expected


def test_accuracy_example_does_not_reverse_skill_lift(monkeypatch, accuracy_judge):
    # The shipped Harbor prompt used all five true with a numeric score of 0.8.
    payloads = iter(
        [
            {"criteria": _criteria(5), "score": 0.8},
            {"criteria": _criteria(5), "score": 0.8},
            {"criteria": _criteria(4), "score": 0.8},
            {"criteria": _criteria(5), "score": 1.0},
        ]
    )
    monkeypatch.setattr(accuracy_judge, "call_public_llm", lambda *_args, **_kwargs: (json.dumps(next(payloads)), None))
    scores = [accuracy_judge.judge_accuracy("question", "reference", "answer")["score"] for _ in range(4)]
    lift = _compute_lift({"accuracy": sum(scores[:2]) / 2}, {"accuracy": sum(scores[2:]) / 2})

    assert lift["accuracy"]["with_skill"] == 1.0
    assert lift["accuracy"]["without_skill"] == 0.9
    assert lift["accuracy"]["delta"] == 0.1
    assert lift["accuracy"]["direction"] == "up"
