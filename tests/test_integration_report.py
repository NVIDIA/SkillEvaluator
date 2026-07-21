# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skillevaluator.evaluation.tier3_report import _build_integration_report, _validation_result_from_payload


def test_integration_report_is_plugin_only_and_reconciles_operands() -> None:
    best = {
        "with_skill": 0.80,
        "baseline": 0.30,
        "sum_of_parts": 0.65,
        "integration_completeness": {"complete": True},
    }
    config = {
        "eval_target": {"kind": "plugin"},
        "skill_workspace": {
            "staged_skills": ["loader", "summarizer"],
            "baseline_includes_workspace_skills": False,
            "sum_of_parts_arm": True,
        },
    }
    report = _build_integration_report(best, config)
    assert report is not None
    assert report["integration_lift"] == 0.15
    assert report["verdict"] == "real_integration"
    assert report["report_only"] is True

    assert _build_integration_report(best, {**config, "eval_target": {"kind": "skill"}}) is None


def test_incomplete_sum_of_parts_never_claims_integration() -> None:
    report = _build_integration_report(
        {"with_skill": 0.9, "sum_of_parts": 0.2, "integration_completeness": {"complete": False}},
        {
            "eval_target": {"kind": "plugin"},
            "skill_workspace": {"staged_skills": ["member"], "sum_of_parts_arm": True},
        },
    )
    assert report is not None
    assert report["verdict"] == "inconclusive"
    assert report["complete"] is False


def test_partial_plugin_payload_is_never_reported_as_a_pass() -> None:
    result = _validation_result_from_payload(
        {
            "best_agent": "codex",
            "execution_status": "succeeded",
            "overall_score": 0.8,
            "verdict": "positive",
            "plugin_provenance": {
                "partial": True,
                "unresolved_skill_refs": ["github::other/repo::skills::member"],
            },
        }
    )

    assert result is not None
    assert result.passed is False
    assert result.metadata["execution_status"] == "skipped"
    assert result.metadata["skip_reason"].startswith("INCOMPLETE:")
