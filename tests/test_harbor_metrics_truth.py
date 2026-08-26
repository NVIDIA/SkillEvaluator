# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from fractions import Fraction

import pytest

from skillevaluator.tier3 import results_location
from skillevaluator.tier3.harbor import collector as collector_module
from skillevaluator.tier3.harbor.collector import (
    _compute_lift,
    _condition_execution_summary,
    _count_derived_pass_rate_delta,
    _mcnemar_exact_p_value,
    _mcnemar_exact_probability,
    _minimum_attainable_mcnemar_p_value,
    _paired_pass_comparison,
    _pass_rate_delta,
    _pass_summary,
    _probability_text,
    _public_pass_summary,
    _wilson_score_interval,
)
from skillevaluator.tier3.harbor.metrics import (
    CUSTOM_ONLY_METRIC_SET,
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    LEGACY_METRIC_SET,
    LEGACY_METRICS,
    MAX_CUSTOM_METRIC_NAME_BYTES,
    MAX_CUSTOM_METRICS,
    CustomMetricContractError,
    average_custom_metrics,
    average_metrics,
    custom_metric_contract_error,
    extract_custom_metrics,
    metric_set_for_reward,
    metric_value,
    overall_score,
    rewards_have_mixed_metric_contracts,
)


def test_custom_metric_names_filter_credentials_but_keep_explicit_metric_terms() -> None:
    credentials = [
        "sk-abcdefghijk",
        "ghp_" + ("a" * 36),
        "gho_" + ("a" * 36),
        "ghu_" + ("a" * 36),
        "ghs_" + ("a" * 36),
        "ghr_" + ("a" * 36),
        "qualityghp_" + ("a" * 36),
        "qualitygho_" + ("a" * 36) + "suffix",
        "ghs_123456789_" + ("a" * 32) + "." + ("b" * 32) + "." + ("c" * 32),
        "github_pat_" + ("a" * 30),
        "".join(("xoxb-", "1234567890-abcdefghijklmnopqrstuvwx")),  # noqa: FLY002
        "AIza" + ("A" * 35),
        "glpat-" + ("a" * 20),
    ]
    reward = {
        "custom_metrics": {
            **dict.fromkeys(credentials, 0.1),
            **dict.fromkeys((f"quality_{credential}" for credential in credentials), 0.1),
            "quality_nvapi-abcdefghijk": 0.1,
            "quality_crsr_0123456789abcdef": 0.1,
            "api_key_quality": 0.2,
            "quality": 0.3,
            "secret_handling": 0.4,
            "token_efficiency": 0.5,
        }
    }

    assert custom_metric_contract_error(reward) is None
    assert extract_custom_metrics(reward) == {
        "quality": 0.3,
        "secret_handling": 0.4,
        "token_efficiency": 0.5,
    }


@pytest.mark.parametrize(
    "exact",
    [
        "x" * MAX_CUSTOM_METRIC_NAME_BYTES,
        "é" * (MAX_CUSTOM_METRIC_NAME_BYTES // len("é".encode())),
    ],
    ids=("ascii", "multibyte"),
)
def test_custom_metric_contract_enforces_name_byte_boundary_without_aliasing(exact: str) -> None:
    oversized = exact + "x"

    assert custom_metric_contract_error({"custom_metrics": {exact: 1.0}}) is None
    assert extract_custom_metrics({"custom_metrics": {exact: 1.0}}) == {exact: 1.0}
    assert custom_metric_contract_error({"custom_metrics": {oversized: 1.0}}) is not None
    assert extract_custom_metrics({"custom_metrics": {oversized: 1.0}}) == {}


def test_custom_metric_contract_enforces_per_reward_and_union_cardinality() -> None:
    exact = {f"metric_{index:03d}": 1.0 for index in range(MAX_CUSTOM_METRICS)}
    oversized = {**exact, "one_too_many": 1.0}
    exact_plus_unsafe = {**exact, "quality_sk-abcdefghijk": 1.0}

    assert custom_metric_contract_error({"custom_metrics": exact}) is None
    assert len(extract_custom_metrics({"custom_metrics": exact})) == MAX_CUSTOM_METRICS
    assert custom_metric_contract_error({"custom_metrics": exact_plus_unsafe}) is None
    assert len(extract_custom_metrics({"custom_metrics": exact_plus_unsafe})) == MAX_CUSTOM_METRICS
    assert custom_metric_contract_error({"custom_metrics": oversized}) is not None
    with pytest.raises(CustomMetricContractError, match="per reward"):
        average_custom_metrics([{"custom_metrics": oversized}])

    left = {f"left_{index:03d}": 1.0 for index in range(MAX_CUSTOM_METRICS // 2 + 1)}
    right = {f"right_{index:03d}": 1.0 for index in range(MAX_CUSTOM_METRICS // 2 + 1)}
    with pytest.raises(CustomMetricContractError, match="per condition"):
        average_custom_metrics([{"custom_metrics": left}, {"custom_metrics": right}])


def test_explicit_custom_metrics_reject_reserved_name_collisions() -> None:
    reward = {"custom_metrics": {"security": 0.0, "quality": 1.0}, "overall": 1.0}

    assert "collides" in (custom_metric_contract_error(reward) or "")
    with pytest.raises(CustomMetricContractError, match="collides"):
        average_custom_metrics([reward])


@pytest.mark.parametrize("malformed", [None, 0.5, "quality", [0.5]])
def test_custom_metrics_container_must_be_an_object(malformed: object) -> None:
    reward = {"custom_metrics": malformed, "quality": 0.8, "overall": 0.8}

    assert "container" in (custom_metric_contract_error(reward) or "")
    assert extract_custom_metrics(reward) == {"quality": 0.8}
    with pytest.raises(CustomMetricContractError, match="container"):
        average_custom_metrics([reward])


def test_custom_metric_extraction_unions_explicit_nested_and_top_level_dict_scores() -> None:
    reward = {
        "custom_metrics": {"quality": 0.8},
        "metrics": {"coverage": {"score": 0.7}},
        "domain_score": {"score": 0.6},
        "api_key_quality": {"score": 0.9},
    }

    assert custom_metric_contract_error(reward) is None
    assert extract_custom_metrics(reward) == {
        "quality": 0.8,
        "coverage": 0.7,
        "domain_score": 0.6,
    }


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), 10**400])
def test_metric_inputs_reject_nonfinite_values(invalid: float | int) -> None:
    assert metric_value({"security": invalid}, "security") is None
    assert metric_value({"metrics": {"security": {"score": invalid}}}, "security") is None
    assert extract_custom_metrics({"custom_metrics": {"latency": invalid}}) == {}


@pytest.mark.parametrize("invalid", [-0.01, 1.01, 1e308])
def test_metric_inputs_reject_scores_outside_the_documented_unit_interval(invalid: float) -> None:
    reward = {"metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, invalid)}

    assert metric_value({"security": invalid}, "security") is None
    assert metric_value({"metrics": {"security": {"score": invalid}}}, "security") is None
    assert extract_custom_metrics({"custom_metrics": {"latency": invalid}}) == {}
    assert overall_score(reward) is None
    assert overall_score({"metric_set": CUSTOM_ONLY_METRIC_SET, "overall": invalid}) is None
    assert overall_score({**dict.fromkeys(DEFAULT_METRICS, invalid), "overall": 0.8}) is None


def test_average_metrics_omits_unavailable_canonical_metrics() -> None:
    scores, metric_set, metrics = average_metrics(
        [
            {
                "metric_set": DEFAULT_METRIC_SET,
                "security": 1.0,
                "accuracy": float("nan"),
            }
        ]
    )

    assert metric_set == DEFAULT_METRIC_SET
    assert metrics == DEFAULT_METRICS
    assert scores == {"security": 1.0}


def test_explicit_custom_metric_set_cannot_spoof_canonical_metrics() -> None:
    custom = {
        "metric_set": CUSTOM_ONLY_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 0.0),
        "overall": 0.25,
    }
    standard = {
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "overall": 1.0,
    }

    assert metric_set_for_reward(custom) == (CUSTOM_ONLY_METRIC_SET, ())
    assert overall_score(custom) == pytest.approx(0.25)
    scores, metric_set, metrics = average_metrics([custom, standard])
    assert metric_set == DEFAULT_METRIC_SET
    assert metrics == DEFAULT_METRICS
    assert scores == dict.fromkeys(DEFAULT_METRICS, 1.0)


@pytest.mark.parametrize(
    "rewards",
    [
        [
            {"metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, 1.0)},
            {"metric_set": LEGACY_METRIC_SET, **dict.fromkeys(LEGACY_METRICS, 1.0)},
        ],
        [
            {"metric_set": "domain-grader-v1", "overall": 1.0},
            {"metric_set": "domain-grader-v2", "overall": 1.0},
        ],
    ],
    ids=("default-v1-and-v2", "distinct-custom-contracts"),
)
def test_distinct_reward_metric_set_contracts_are_mixed(rewards: list[dict[str, object]]) -> None:
    assert rewards_have_mixed_metric_contracts(rewards) is True


def test_overall_score_requires_a_complete_finite_metric_set() -> None:
    complete = dict.fromkeys(DEFAULT_METRICS, 0.8)
    incomplete = dict(complete)
    incomplete.pop("behavior_check")
    invalid = dict(complete)
    invalid["behavior_check"] = float("inf")

    assert overall_score(complete) == pytest.approx(0.8)
    assert overall_score(incomplete) is None
    assert overall_score(invalid) is None
    assert overall_score({"metric_set": CUSTOM_ONLY_METRIC_SET, "overall": float("nan")}) is None


def test_lift_omits_unpaired_metrics_and_incomplete_overall() -> None:
    lift = _compute_lift(
        {"security": 1.0, "accuracy": 0.8},
        {"security": 0.5, "goal_accuracy": 0.4},
    )

    assert lift == {
        "security": {
            "with_skill": 1.0,
            "without_skill": 0.5,
            "delta": 0.5,
            "direction": "up",
        }
    }


def test_pass_summary_marks_incomplete_reward_unscored() -> None:
    reward = {
        "entry_id": "case-001",
        "_trial_name": "case-001__attempt1",
        "_trial_root_name": "trial-1",
        "security": 1.0,
    }

    summary = _pass_summary(
        [reward],
        n_attempts=1,
        pass_threshold=0.5,
        expected_cases=1,
        expected_case_ids=["case-001"],
    )

    case = summary["cases"]["case-001"]
    # An incomplete reward is not a scored attempt: it contributes no pass@k
    # attempt row and the attempt is reported as missing instead.
    assert case["attempts"] == []
    assert case["passed"] is False
    assert case["attempts_used"] == 0
    assert case["attempts_missing"] == 1
    assert case["best_score"] is None
    assert summary["attempts_used"] == 0
    assert math.isfinite(summary["rate"])


def test_large_case_sets_publish_bounded_samples_without_losing_exact_pairing() -> None:
    case_ids = [f"case-{index:05d}-{'x' * 80}" for index in range(20_000)]
    with_rewards = [
        {
            "entry_id": case_id,
            "_trial_name": f"trial-{index:05d}",
            "_trial_root_name": f"trial-{index:05d}",
            "metric_set": CUSTOM_ONLY_METRIC_SET,
            "overall": 1.0,
        }
        for index, case_id in enumerate(case_ids)
    ]
    without_rewards = [{**reward, "overall": 0.0} for reward in with_rewards]

    with_summary = _pass_summary(
        with_rewards,
        n_attempts=1,
        pass_threshold=0.5,
        expected_cases=len(case_ids),
        expected_case_ids=case_ids,
    )
    without_summary = _pass_summary(
        without_rewards,
        n_attempts=1,
        pass_threshold=0.5,
        expected_cases=len(case_ids),
        expected_case_ids=case_ids,
    )
    paired = _paired_pass_comparison(with_summary, without_summary)
    public = _public_pass_summary(with_summary)

    assert with_summary["passed_cases"] == len(case_ids)
    assert paired["pairing_status"] == "complete"
    assert paired["paired_cases"] == len(case_ids)
    assert paired["with_skill_only_pass"] == len(case_ids)
    assert public["case_details_total"] == len(case_ids)
    assert public["case_details_shown"] == collector_module.PUBLISHED_CASE_DETAILS_MAX
    assert public["case_details_truncated"] is True
    assert "_pairing_cases" not in public
    assert len(json.dumps(public, separators=(",", ":")).encode()) < collector_module.GENERATED_JSON_MAX_BYTES
    assert results_location._legacy_pass_at_k_is_complete(
        public,
        num_trials=len(case_ids),
        require_scored_attempt=True,
        expected_scored_attempts=len(case_ids),
    )


def test_large_missing_case_set_has_bounded_exact_execution_diagnostics() -> None:
    case_ids = [f"case-{index:05d}-{'x' * 80}" for index in range(20_000)]

    summary = _condition_execution_summary(
        [],
        expected_case_ids=case_ids,
        expected_cases=len(case_ids),
        n_attempts=1,
        job_failure="job did not produce trials",
    )
    encoded = json.dumps(summary, separators=(",", ":")).encode()

    assert summary["execution_status"] == "failed"
    assert summary["expected_attempts"] == len(case_ids)
    assert summary["scored_attempts"] == 0
    assert any("showing 32 of 20000" in error for error in summary["execution_errors"])
    assert len(encoded) < collector_module.GENERATED_JSON_MAX_BYTES


@pytest.mark.parametrize("failure_kind", ["runtime_failures", "reward_failures"])
def test_large_failure_sets_publish_bounded_samples_with_exact_counts(failure_kind: str) -> None:
    failures = [{"trial": f"trial-{index:05d}", "reason": "upstream failure " + ("x" * 600)} for index in range(4_096)]

    summary = _condition_execution_summary(
        [],
        expected_case_ids=[],
        expected_cases=0,
        n_attempts=1,
        job_failure="",
        **{failure_kind: failures},
    )
    encoded = json.dumps(summary, separators=(",", ":")).encode()
    prefix = "runtime_failure_details" if failure_kind == "runtime_failures" else "reward_failure_details"

    assert summary["execution_status"] == "failed"
    assert summary[f"{prefix}_total"] == len(failures)
    assert summary[f"{prefix}_shown"] == collector_module.PUBLISHED_FAILURE_DETAILS_MAX
    assert summary[f"{prefix}_truncated"] is True
    assert summary["execution_error_details_truncated"] is False
    assert any(
        f"showing {collector_module.PUBLISHED_FAILURE_DETAILS_MAX} of 4096" in error
        for error in summary["execution_errors"]
    )
    assert len(encoded) < collector_module.GENERATED_JSON_MAX_BYTES


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        (0, 4, (0.0, 0.4899)),
        (2, 4, (0.15, 0.85)),
        (4, 4, (0.5101, 1.0)),
    ],
)
def test_wilson_interval_reports_bounded_case_rate_uncertainty(
    successes: int,
    total: int,
    expected: tuple[float, float],
) -> None:
    interval = _wilson_score_interval(successes, total)

    assert interval is not None
    assert interval["method"] == "wilson_score"
    assert interval["confidence_level"] == 0.95
    assert interval["lower"] == pytest.approx(expected[0], abs=0.0001)
    assert interval["upper"] == pytest.approx(expected[1], abs=0.0001)


def test_wilson_interval_is_unavailable_without_cases() -> None:
    assert _wilson_score_interval(0, 0) is None


@pytest.mark.parametrize("total", [1, 2, 3, 10, 1_000, 100_000])
def test_wilson_interval_contains_observed_mathematical_endpoints_exactly(total: int) -> None:
    zero_passes = _wilson_score_interval(0, total)
    all_passes = _wilson_score_interval(total, total)

    assert zero_passes is not None
    assert all_passes is not None
    assert zero_passes["lower"] == 0.0
    assert zero_passes["lower"] <= 0.0 <= zero_passes["upper"]
    assert all_passes["upper"] == 1.0
    assert all_passes["lower"] <= 1.0 <= all_passes["upper"]


def test_wilson_interval_preserves_endpoint_uncertainty_beyond_four_decimal_places() -> None:
    zero_passes = _wilson_score_interval(0, 100_000)
    all_passes = _wilson_score_interval(100_000, 100_000)

    assert zero_passes is not None
    assert all_passes is not None
    assert zero_passes["lower"] == 0.0
    assert 0.0 < zero_passes["upper"] < 0.0001
    assert 0.9999 < all_passes["lower"] < 1.0
    assert all_passes["upper"] == 1.0


def test_mcnemar_exact_p_value_preserves_small_nonzero_results() -> None:
    assert _mcnemar_exact_p_value(32, 0) == pytest.approx(2**-31)
    assert _mcnemar_exact_p_value(32, 0) > 0.0


@pytest.mark.parametrize(
    ("discordant", "expected"),
    [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)],
)
def test_mcnemar_exact_reports_attainable_resolution(discordant: int, expected: float) -> None:
    assert _minimum_attainable_mcnemar_p_value(discordant) == expected


def test_pass_rate_delta_preserves_legacy_rate_contract_and_exposes_count_correction() -> None:
    with_skill = {"passed_cases": 2, "total_cases": 3, "rate": 0.6667}
    without_skill = {"passed_cases": 1, "total_cases": 3, "rate": 0.3333}

    assert _pass_rate_delta(with_skill, without_skill) == 0.3334
    assert _count_derived_pass_rate_delta(with_skill, without_skill) == 0.3333


@pytest.mark.parametrize(
    ("with_skill", "without_skill", "legacy_delta", "count_derived_delta"),
    [
        (
            {"passed_cases": 17, "total_cases": 160, "rate": 0.1062},
            {"passed_cases": 9, "total_cases": 160, "rate": 0.0563},
            0.0499,
            0.05,
        ),
        (
            {"passed_cases": 1, "total_cases": 160, "rate": 0.0063},
            {"passed_cases": 17, "total_cases": 160, "rate": 0.1062},
            -0.0999,
            -0.1,
        ),
    ],
)
def test_pass_rate_delta_threshold_compatibility(
    with_skill: dict[str, float | int],
    without_skill: dict[str, float | int],
    legacy_delta: float,
    count_derived_delta: float,
) -> None:
    assert _pass_rate_delta(with_skill, without_skill) == legacy_delta
    assert _count_derived_pass_rate_delta(with_skill, without_skill) == count_derived_delta


def test_balanced_large_exact_test_uses_constant_time_symmetric_tail() -> None:
    started = time.perf_counter()

    probability = _mcnemar_exact_probability(10_000, 10_000)

    assert probability == 1
    assert time.perf_counter() - started < 2.0


def test_paired_pass_comparison_preserves_case_direction_and_exact_test() -> None:
    with_skill = {
        "total_cases": 4,
        "cases": {
            "both": {"passed": True},
            "improved": {"passed": True},
            "regressed": {"passed": False},
            "neither": {"passed": False},
        },
    }
    without_skill = {
        "total_cases": 4,
        "cases": {
            "both": {"passed": True},
            "improved": {"passed": False},
            "regressed": {"passed": True},
            "neither": {"passed": False},
        },
    }

    paired = _paired_pass_comparison(with_skill, without_skill)

    assert paired["pairing_status"] == "complete"
    assert paired["paired_cases"] == 4
    assert paired["both_pass"] == 1
    assert paired["with_skill_only_pass"] == 1
    assert paired["without_skill_only_pass"] == 1
    assert paired["neither_pass"] == 1
    assert paired["paired_rate_delta"] == 0.0
    assert paired["mcnemar_exact"]["p_value"] == 1.0
    assert paired["mcnemar_exact"]["minimum_attainable_p_value"] == 0.5
    assert paired["mcnemar_exact"]["resolution_limited_at_alpha_0_05"] is True


def test_complete_pairing_delta_matches_count_derived_arm_delta() -> None:
    with_skill = {
        "passed_cases": 2,
        "total_cases": 3,
        "rate": 0.6667,
        "cases": {"a": {"passed": True}, "b": {"passed": True}, "c": {"passed": False}},
    }
    without_skill = {
        "passed_cases": 1,
        "total_cases": 3,
        "rate": 0.3333,
        "cases": {"a": {"passed": False}, "b": {"passed": True}, "c": {"passed": False}},
    }

    paired = _paired_pass_comparison(with_skill, without_skill)

    assert paired["paired_rate_delta"] == pytest.approx(1 / 3)
    assert paired["paired_rate_delta"] == pytest.approx(
        _count_derived_pass_rate_delta(with_skill, without_skill),
        abs=0.0001,
    )


@pytest.mark.parametrize(("skill_only", "direction"), [(True, 1), (False, -1)])
def test_paired_delta_preserves_small_nonzero_direction(skill_only: bool, direction: int) -> None:
    pair_count = 25_000
    case_ids = [f"case-{index}" for index in range(pair_count)]
    with_skill_cases = {case_id: {"passed": False} for case_id in case_ids}
    without_skill_cases = {case_id: {"passed": False} for case_id in case_ids}
    (with_skill_cases if skill_only else without_skill_cases)[case_ids[0]]["passed"] = True

    paired = _paired_pass_comparison(
        {"total_cases": pair_count, "cases": with_skill_cases},
        {"total_cases": pair_count, "cases": without_skill_cases},
    )

    assert paired["paired_rate_delta"] == pytest.approx(direction / pair_count)
    assert math.copysign(1.0, paired["paired_rate_delta"]) == direction


def test_mcnemar_exact_preserves_machine_readable_value_beyond_float_range() -> None:
    with_skill_cases = {f"case-{index}": {"passed": True} for index in range(1076)}
    without_skill_cases = {case_id: {"passed": False} for case_id in with_skill_cases}

    paired = _paired_pass_comparison(
        {"total_cases": 1076, "cases": with_skill_cases},
        {"total_cases": 1076, "cases": without_skill_cases},
    )
    exact = paired["mcnemar_exact"]

    assert exact["p_value"] is None
    assert exact["p_value_numeric_underflow"] is True
    assert exact["p_value_text"] != "0"
    assert exact["p_value_exact"].startswith("1/")
    assert exact["minimum_attainable_p_value"] is None
    assert exact["minimum_attainable_p_value_text"] != "0"
    assert exact["minimum_attainable_p_value_exact"] == exact["p_value_exact"]


def test_probability_text_is_fast_and_nonzero_beyond_decimal_exponent_range() -> None:
    probability = collector_module._minimum_attainable_mcnemar_probability(3_321_957)

    started = time.perf_counter()
    rendered = _probability_text(probability)
    elapsed = time.perf_counter() - started

    mantissa, separator, exponent = rendered.partition("e")
    assert separator == "e"
    assert float(mantissa) > 0.0
    assert int(exponent) < -1_000_000
    assert elapsed < 2.0


def test_complete_one_sided_pairing_formats_identical_probability_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Fraction] = []
    original = collector_module._probability_text

    def _counted_probability_text(probability: Fraction) -> str:
        calls.append(probability)
        return original(probability)

    monkeypatch.setattr(collector_module, "_probability_text", _counted_probability_text)
    with_skill_cases = {f"case-{index}": {"passed": True} for index in range(8)}
    without_skill_cases = {case_id: {"passed": False} for case_id in with_skill_cases}

    _paired_pass_comparison(
        {"total_cases": 8, "cases": with_skill_cases},
        {"total_cases": 8, "cases": without_skill_cases},
    )

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("discordant", "exact_is_serialized"),
    [(14_285, True), (14_286, False), (15_000, False)],
)
def test_large_exact_pairing_respects_integer_string_safety_limit(
    discordant: int,
    exact_is_serialized: bool,
) -> None:
    with_skill_cases = {f"case-{index}": {"passed": True} for index in range(discordant)}
    without_skill_cases = {case_id: {"passed": False} for case_id in with_skill_cases}

    paired = _paired_pass_comparison(
        {"total_cases": discordant, "cases": with_skill_cases},
        {"total_cases": discordant, "cases": without_skill_cases},
    )
    exact = paired["mcnemar_exact"]

    assert exact["p_value_text"] != "0"
    assert exact["minimum_attainable_p_value_text"] != "0"
    assert exact["p_value_exact_omitted"] is not exact_is_serialized
    assert exact["minimum_attainable_p_value_exact_omitted"] is not exact_is_serialized
    if exact_is_serialized:
        assert isinstance(exact["p_value_exact"], str)
        assert isinstance(exact["minimum_attainable_p_value_exact"], str)
    else:
        assert exact["p_value_exact"] is None
        assert exact["p_value_exact_omitted_reason"] == "decimal_digit_limit"
        assert exact["minimum_attainable_p_value_exact"] is None
        assert exact["minimum_attainable_p_value_exact_omitted_reason"] == "decimal_digit_limit"


def test_large_exact_pairing_respects_active_integer_string_limit_in_subprocess() -> None:
    code = """
from skillevaluator.tier3.harbor.collector import _paired_pass_comparison

pair_count = 2_200
with_skill_cases = {f"case-{index}": {"passed": True} for index in range(pair_count)}
without_skill_cases = {case_id: {"passed": False} for case_id in with_skill_cases}
paired = _paired_pass_comparison(
    {"total_cases": pair_count, "cases": with_skill_cases},
    {"total_cases": pair_count, "cases": without_skill_cases},
)
exact = paired["mcnemar_exact"]
assert exact["p_value_exact"] is None
assert exact["p_value_exact_omitted"] is True
assert exact["p_value_exact_omitted_reason"] == "decimal_digit_limit"
assert exact["minimum_attainable_p_value_exact"] is None
assert exact["minimum_attainable_p_value_exact_omitted"] is True
assert exact["minimum_attainable_p_value_exact_omitted_reason"] == "decimal_digit_limit"
"""
    environment = {**os.environ, "PYTHONINTMAXSTRDIGITS": "640"}

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_paired_pass_comparison_does_not_issue_exact_test_for_partial_pairing() -> None:
    paired = _paired_pass_comparison(
        {"total_cases": 2, "cases": {"shared": {"passed": True}, "with-only": {"passed": True}}},
        {"total_cases": 2, "cases": {"shared": {"passed": False}, "without-only": {"passed": False}}},
    )

    assert paired["pairing_status"] == "partial"
    assert paired["paired_cases"] == 1
    assert paired["with_skill_unpaired_case_count"] == 1
    assert paired["without_skill_unpaired_case_count"] == 1
    assert paired["with_skill_unpaired_case_ids"] == ["with-only"]
    assert paired["without_skill_unpaired_case_ids"] == ["without-only"]
    assert paired["with_skill_unpaired_case_ids_truncated"] is False
    assert paired["without_skill_unpaired_case_ids_truncated"] is False
    assert "mcnemar_exact" not in paired


def test_paired_pass_comparison_bounds_unpaired_case_id_diagnostics() -> None:
    case_count = 2_400
    with_cases = {f"with-{index:04d}": {"passed": True} for index in range(case_count)}
    without_cases = {f"without-{index:04d}": {"passed": False} for index in range(case_count)}

    paired = _paired_pass_comparison(
        {"total_cases": case_count, "cases": with_cases},
        {"total_cases": case_count, "cases": without_cases},
    )

    assert paired["pairing_status"] == "unavailable"
    assert paired["with_skill_unpaired_case_count"] == case_count
    assert paired["without_skill_unpaired_case_count"] == case_count
    assert len(paired["with_skill_unpaired_case_ids"]) == 64
    assert len(paired["without_skill_unpaired_case_ids"]) == 64
    assert paired["with_skill_unpaired_case_ids_truncated"] is True
    assert paired["without_skill_unpaired_case_ids_truncated"] is True


def test_condition_marks_incomplete_reward_as_unscored() -> None:
    reward = {
        "entry_id": "case-001",
        "_trial_root_name": "trial-1",
        "security": 1.0,
    }

    summary = _condition_execution_summary(
        [reward],
        expected_case_ids=["case-001"],
        expected_cases=1,
        n_attempts=1,
        job_failure="",
    )

    assert summary["execution_status"] == "failed"
    assert summary["scored_attempts"] == 0
    assert any("incomplete or non-finite" in error for error in summary["execution_errors"])
