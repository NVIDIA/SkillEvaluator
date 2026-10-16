# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy scores must implement the applicable-criterion rubric in both runtimes."""

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


@pytest.mark.parametrize("skill_verdict", [True, False])
@pytest.mark.parametrize("passed", range(5))
def test_negative_accuracy_scores_only_applicable_criteria(monkeypatch, accuracy_judge, passed, skill_verdict):
    # Negative cases drop SKILL_IDENTIFIED from numerator and denominator; the model total stays ignored.
    applicable = sorted(llm_judge._ACCURACY_CRITERIA_KEYS - {"SKILL_IDENTIFIED"})
    criteria = {key: index < passed for index, key in enumerate(applicable)}
    criteria["SKILL_IDENTIFIED"] = skill_verdict
    payload = {"criteria": criteria, "score": 0.1, "reason": "declined correctly"}
    monkeypatch.setattr(accuracy_judge, "call_public_llm", lambda *_args, **_kwargs: (json.dumps(payload), None))

    result = accuracy_judge.judge_accuracy("question", "reference", "answer", skill_identification_applicable=False)

    assert result["score"] == pytest.approx(passed / 4)
    assert result["criteria"] == criteria
    assert result["excluded_criteria"] == ["SKILL_IDENTIFIED"]
    assert result["scoring_policy"] == "applicable-accuracy-criteria/v1"


def test_tier3_docs_describe_criterion_derived_accuracy():
    docs = (Path(__file__).resolve().parents[1] / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")
    normalized = " ".join(docs.split())

    assert "passing applicable criteria divided by applicable criteria" in normalized
    assert "falls back to the judge's numeric score" in normalized
    assert "existing scoring contract" not in normalized


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"criteria": _criteria(3), "score": 0.6}, 0.6),
        ({"criteria": _criteria(3)}, 0.6),
        ({"score": 0.6}, 0.6),
    ],
)
def test_accuracy_preserves_consistent_and_legacy_payloads(monkeypatch, accuracy_judge, payload, expected):
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
