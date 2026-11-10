# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pytest
from rich.console import Console

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter
from skillevaluator.tier3.dataset_sufficiency import assess_dataset_sufficiency
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
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


def test_benchmark_renders_recorded_negative_count_when_unclassified_cases_remain():
    sufficiency = assess_dataset_sufficiency(
        distinct_cases=13,
        negative_cases=2,
        pass_rate_interval={"lower": 0.45, "upper": 0.55},
        paired_cases=13,
    )
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = {
        "verdict": "pass",
        "dataset_summary": {
            "total_tasks": 13,
            "positive_tasks": 10,
            "negative_tasks": 2,
            "unclassified_tasks": 1,
            "source": "dataset",
        },
        "dataset_sufficiency": sufficiency,
        "agents": {},
    }
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])

    assert sufficiency["status"] == "sufficient"
    assert "2 negative cases (needs 1)" in sufficiency["coverage_summary"]
    assert sufficiency["coverage_summary"] in benchmark


def test_agent_eval_keeps_sufficient_recorded_negative_count_with_unclassified_cases(monkeypatch):
    from skillevaluator.evaluation import tier3_report
    from skillevaluator.tier3 import dataset_sufficiency as sufficiency_module

    observed = []
    assess = sufficiency_module.assess_dataset_sufficiency
    monkeypatch.setattr(
        sufficiency_module,
        "assess_dataset_sufficiency",
        lambda **kwargs: observed.append(kwargs) or assess(**kwargs),
    )
    monkeypatch.setattr(
        tier3_report,
        "_dataset_summary",
        lambda *_args: {
            "total_tasks": 13,
            "positive_tasks": 10,
            "negative_tasks": 2,
            "unclassified_tasks": 1,
            "source": "dataset",
        },
    )

    payload = build_agent_eval_payload(
        "demo",
        {"codex": {"execution_status": "succeeded", "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8)}},
        use_llm_judge=False,
    )

    assert payload is not None
    assert observed[0]["negative_cases"] == 2


@pytest.mark.parametrize(
    "dataset_summary",
    [
        {"total_tasks": 0, "negative_tasks": 0, "source": "dataset"},
        {"total_tasks": 12, "negative_tasks": 0, "source": "unavailable"},
    ],
)
def test_missing_dataset_tasks_omit_sufficiency_from_benchmark_and_html(dataset_summary):
    sufficiency = assess_dataset_sufficiency(
        distinct_cases=12,
        negative_cases=0,
        pass_rate_interval={"lower": 0.45, "upper": 0.55},
        paired_cases=12,
    )
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = {
        "verdict": "pass",
        "overall_score": 0.8,
        "dataset_summary": dataset_summary,
        "dataset_sufficiency": sufficiency,
        "agents": {},
    }

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])
    html = HTMLReporter(include_timestamp=False, report_scope="tier3").render_all([result])

    assert "Tasks: not recorded (legacy or non-live result)" in benchmark
    assert sufficiency["coverage_summary"] not in benchmark
    assert "t3-dataset-sufficiency" not in html


@pytest.mark.parametrize("paired_cases", ["unknown", float("inf"), "12"])
def test_invalid_paired_case_counts_are_unavailable(paired_cases):
    scores = dict.fromkeys(DEFAULT_METRICS, 0.8)
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "with_skill": scores,
                "without_skill": scores,
                "pass_at_k": {
                    "with_skill": {"rate_interval": {"lower": 0.45, "upper": 0.55}},
                    "lift": {"paired_comparison": {"paired_cases": paired_cases}},
                },
            }
        },
        dataset=[{"id": "case-1", "prompt": "Question", "expected_skill": "demo"}],
        use_llm_judge=False,
    )

    assert payload is not None
    sufficiency = payload["agents"]["codex"]["dataset_sufficiency"]
    assert sufficiency["checks"][2]["observed"]["paired_cases"] is None
    assert sufficiency["checks"][2]["met"] is None


def test_agent_eval_reports_unclassified_negative_count_as_unknown():
    scores = dict.fromkeys(DEFAULT_METRICS, 0.8)
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "with_skill": scores,
                "without_skill": scores,
                "pass_at_k": {
                    "with_skill": {"rate_interval": {"lower": 0.45, "upper": 0.55}},
                    "lift": {"paired_comparison": {"paired_cases": 1}},
                },
            }
        },
        dataset=[{"id": "case-1", "prompt": "Question"}],
        use_llm_judge=False,
    )

    assert payload is not None
    sufficiency = payload["agents"]["codex"]["dataset_sufficiency"]
    assert sufficiency["checks"][0]["met"] is None
    assert sufficiency["checks"][0]["observed"]["negative_cases"] is None
    assert "negative cases not recorded" in sufficiency["coverage_summary"]
