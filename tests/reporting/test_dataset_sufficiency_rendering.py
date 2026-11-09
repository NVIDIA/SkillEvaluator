# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pytest
from rich.console import Console

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, CLIReporter
from skillevaluator.tier3.dataset_sufficiency import assess_dataset_sufficiency
from skillevaluator.tier3.result_display import render_result


def _expected_summary(status: str) -> str:
    if status == "sufficient":
        return (
            "Dataset size: sufficient — 12 cases (policy needs 10), "
            "2 negative cases (needs 1) (policy dataset-sufficiency-v1)"
        )
    return (
        "Dataset size: provisional — 4 cases (policy needs 10), "
        "0 negative cases (needs 1); pass-rate interval ±0.40 is wider than ±0.20 "
        "(policy dataset-sufficiency-v1)"
    )


@pytest.fixture(params=("provisional", "sufficient"))
def sufficiency(request):
    if request.param == "sufficient":
        return assess_dataset_sufficiency(
            distinct_cases=12,
            negative_cases=2,
            pass_rate_interval={"lower": 0.45, "upper": 0.55},
            paired_cases=12,
        )
    return assess_dataset_sufficiency(
        distinct_cases=4,
        negative_cases=0,
        pass_rate_interval={"lower": 0.1, "upper": 0.9},
        paired_cases=4,
    )


def test_tier3_evaluate_final_summary_renders_dataset_size(sufficiency):
    output = render_result({"execution_status": "succeeded", "agents": {}, "dataset_sufficiency": sufficiency})

    expected = _expected_summary(sufficiency["status"])
    assert sufficiency["coverage_summary"] == expected
    assert expected in output


def test_validate_agent_eval_section_renders_dataset_size(sufficiency):
    stream = StringIO()
    CLIReporter._print_agent_eval_tables(
        {
            "verdict": "pass",
            "dataset_sufficiency": sufficiency,
            "dataset_summary": {"total_tasks": 12},
        },
        Console(file=stream, force_terminal=False),
    )

    expected = _expected_summary(sufficiency["status"])
    assert sufficiency["coverage_summary"] == expected
    assert expected in " ".join(stream.getvalue().split())


def test_benchmark_renders_dataset_size(sufficiency):
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = {
        "verdict": "pass",
        "overall_score": 0.8,
        "dataset_summary": {"total_tasks": 12, "negative_tasks": 2},
        "dataset_sufficiency": sufficiency,
        "agents": {},
    }

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])

    expected = _expected_summary(sufficiency["status"])
    assert sufficiency["coverage_summary"] == expected
    assert expected in benchmark
