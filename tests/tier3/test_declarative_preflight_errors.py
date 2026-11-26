# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regex compiler limits should produce diagnostics during dataset preflight."""

from __future__ import annotations

import json
import sys

import pytest

from skillevaluator.tier3.dataset_coverage import summarize_dataset
from skillevaluator.tier3.eval_core.declarative_checks import validate_declarative_checks
from skillevaluator.tier3.evals_spec import validate_skillevaluators


@pytest.fixture(params=["nesting", "repetition", "decimal_conversion"])
def unsupported_pattern(request):
    if request.param == "nesting":
        depth = sys.getrecursionlimit() * 2
        yield "(" * depth + "a" + ")" * depth
    elif request.param == "repetition":
        yield "a{4294967296}"
    else:
        previous = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(sys.int_info.str_digits_check_threshold)
            yield "a{" + "9" * (sys.int_info.str_digits_check_threshold + 1) + "}"
        finally:
            sys.set_int_max_str_digits(previous)


def test_preflight_reports_compiler_limits_and_continues(unsupported_pattern):
    errors = validate_declarative_checks(
        [
            {"type": "output_matches", "pattern": unsupported_pattern},
            {"type": "file_exists", "path": "../outside"},
            {"type": "output_matches", "pattern": r"(?P<word>ready)\s+(?P=word)"},
        ]
    )

    assert len(errors) == 2
    assert errors[0].startswith("checks[0].pattern is not a valid regular expression: ")
    assert errors[1].startswith("checks[1].path must be relative")


def test_dataset_validation_retains_later_entry_diagnostics(tmp_path, unsupported_pattern):
    evals = tmp_path / "evals"
    evals.mkdir()
    entries = [
        {
            "id": "limited-pattern",
            "prompt": "Return ready",
            "expected_output": "ready",
            "checks": [{"type": "output_matches", "pattern": unsupported_pattern}],
        },
        {
            "id": "later-entry",
            "prompt": "Create a report",
            "expected_output": "Report created",
            "checks": [{"type": "file_exists", "path": "../outside"}],
        },
    ]
    (evals / "evals.json").write_text(json.dumps({"skill_name": "demo", "evals": entries}), encoding="utf-8")

    errors = [result for result in validate_skillevaluators(tmp_path) if result.status == "error"]

    assert any(
        result.path == "entry limited-pattern" and "not a valid regular expression" in result.message
        for result in errors
    )
    assert any(
        result.path == "entry later-entry" and "cannot traverse parent directories" in result.message
        for result in errors
    )


def test_coverage_does_not_count_a_pattern_that_cannot_compile(unsupported_pattern):
    coverage = summarize_dataset(
        [
            {"checks": [{"type": "output_matches", "pattern": unsupported_pattern}]},
            {"checks": [{"type": "output_contains", "text": "ready"}]},
        ]
    )

    assert coverage["case_count"] == 2
    assert coverage["cases_with_expected_behaviors"] == 1
