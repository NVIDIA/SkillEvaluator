# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the packaged calculator reference script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CALC = Path(__file__).resolve().parents[1] / "src/skillevaluator/tier3/reference_skills/calculator/scripts/calc.py"


def _calc(expression: str, timeout: float = 3) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CALC), expression],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("-40 celsius to fahrenheit", "-40 celsius = -40 fahrenheit"),
        ("-4 fahrenheit to celsius", "-4 fahrenheit = -20 celsius"),
        (".5 kg to lbs", "0.5 kg = 1.102 lbs"),
        ("-10% of 50", "Result: -5"),
        ("12.5% of -80", "Result: -10"),
        ("100 celsius to fahrenheit", "100 celsius = 212 fahrenheit"),
        ("15% of 200", "Result: 30"),
    ],
)
def test_conversions_and_percentages_accept_signed_numbers(expression: str, expected: str) -> None:
    result = _calc(expression)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "Result: 14"),
        ("(100 - 25) / 3", "Result: 25"),
        ("2 ** 10", "Result: 1024"),
        ("2**64", "Result: 18446744073709551616"),
        ("2**-2", "Result: 0.25"),
        ("5 miles to km", "5 miles = 8.047 km"),
        ("10 kg to lbs", "10 kg = 22.05 lbs"),
    ],
)
def test_documented_examples_and_exact_powers_are_unchanged(expression: str, expected: str) -> None:
    result = _calc(expression)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    "expression",
    ["10% of 50 + 5", "5 km to miles please", "1.2.3 km to miles"],
)
def test_input_beyond_a_conversion_or_percentage_is_rejected(expression: str) -> None:
    result = _calc(expression)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("Error: ")
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "expression",
    ["9**9**9", "10**10**10", "10.0**400", "1e308*10", "2**4000", "2**(10**309)", "2**(10**999)"],
)
def test_oversized_results_are_rejected_quickly(expression: str) -> None:
    result = _calc(expression)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Error: result is too large"


def test_non_real_results_are_rejected() -> None:
    result = _calc("(-8)**(1/3)")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Error: result is not a real number"


def test_boolean_constants_are_rejected() -> None:
    result = _calc("True+True")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Error: Unsupported value: bool"


def test_division_by_zero_message_is_unchanged() -> None:
    result = _calc("1/0")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Error: division by zero"
