# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incomplete retained evidence must not authorize more budgeted attempts."""

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.run_budget import BudgetController, PriceBook


def _trajectory(root: Path, name: str, metrics: object) -> None:
    path = root / name / "agent" / "trajectory.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"final_metrics": metrics}), encoding="utf-8")


def _metrics() -> dict[str, object]:
    return {"total_prompt_tokens": 10, "total_completion_tokens": 5, "total_cost_usd": 0.2}


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("total_prompt_tokens", None),
        ("total_prompt_tokens", True),
        ("total_prompt_tokens", -10),
        ("total_prompt_tokens", 10**400),
        ("total_completion_tokens", None),
        ("total_cost_usd", None),
        ("total_cost_usd", -0.2),
        ("total_cost_usd", True),
        ("total_cost_usd", 10**400),
    ],
    ids=[
        "missing-prompt",
        "boolean-prompt",
        "negative-prompt",
        "oversized-prompt",
        "missing-completion",
        "missing-cost",
        "negative-cost",
        "boolean-cost",
        "oversized-cost",
    ],
)
def test_partial_trajectory_usage_stops_its_bounded_dimension(tmp_path: Path, field: str, invalid: object) -> None:
    _trajectory(tmp_path, "valid", _metrics())
    malformed = _metrics()
    malformed[field] = invalid
    _trajectory(tmp_path, "malformed", malformed)

    cost = field == "total_cost_usd"
    budget = (
        BudgetController(max_cost_usd=10, price_book=PriceBook({})) if cost else BudgetController(max_agent_tokens=100)
    )
    assert budget.admit(agent="a", arm="with", case_id="first", attempt=1)

    usage = runner._budget_usage(tmp_path)
    assert ("cost_usd" if cost else "input_tokens") not in usage
    if cost:
        assert usage["input_tokens"] == 20
    else:
        assert usage["cost_usd"] == pytest.approx(0.4)
    budget.record_usage(usage)

    dimension = "cost_usd" if cost else "agent_tokens"
    assert budget.metadata()["unmeasured_attempts"][dimension] == 1
    assert not budget.admit(agent="a", arm="with", case_id="next", attempt=1)
    assert budget.metadata()["reservation"]["in_flight_attempts"] == 0


@pytest.mark.parametrize("metrics", [None, [], {}])
def test_nonobject_or_missing_metrics_do_not_leave_a_complete_subtotal(tmp_path: Path, metrics: object) -> None:
    _trajectory(tmp_path, "valid", _metrics())
    _trajectory(tmp_path, "malformed", metrics)

    assert runner._budget_usage(tmp_path) == {}


@pytest.mark.parametrize("raw", [b"not JSON", b"\xff"], ids=["invalid-json", "invalid-utf8"])
def test_unreadable_trajectory_does_not_leave_a_complete_subtotal(tmp_path: Path, raw: bytes) -> None:
    _trajectory(tmp_path, "valid", _metrics())
    bad = tmp_path / "malformed" / "agent" / "trajectory.json"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(raw)

    assert runner._budget_usage(tmp_path) == {}


@pytest.mark.parametrize("raw", ["not JSON", "{}", '{"attempts": null}', '{"attempts": [null]}'])
@pytest.mark.parametrize("include_valid", [False, True])
def test_malformed_judge_evidence_is_not_zero_or_a_complete_subtotal(
    tmp_path: Path, raw: str, include_valid: bool
) -> None:
    bad = tmp_path / "malformed" / "judge_attempts.json"
    bad.parent.mkdir()
    bad.write_text(raw, encoding="utf-8")
    if include_valid:
        good = tmp_path / "valid" / "judge_attempts.json"
        good.parent.mkdir()
        good.write_text('{"attempts": [{"method": "llm"}]}', encoding="utf-8")
    budget = BudgetController(max_judge_calls=10)
    assert budget.admit(agent="a", arm="with", case_id="first", attempt=1)

    usage = runner._budget_usage(tmp_path)
    assert "judge_calls" not in usage
    budget.record_usage(usage)

    assert budget.metadata()["unmeasured_attempts"]["judge_calls"] == 1
    assert not budget.admit(agent="a", arm="with", case_id="next", attempt=1)


def test_complete_zero_usage_is_still_measured(tmp_path: Path) -> None:
    _trajectory(tmp_path, "zero", dict.fromkeys(_metrics(), 0))
    (tmp_path / "judge_attempts.json").write_text('{"attempts": []}', encoding="utf-8")

    assert runner._budget_usage(tmp_path) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "judge_calls": 0,
    }
