# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent quality failures remain visible alongside structured findings."""

import html
import re

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import BenchmarkReporter, HTMLReporter

_GATE_MESSAGE = "Quality score 67.8/100 is below the required 70/100 in SKILL.md"


def _quality_result() -> ValidationResult:
    result = ValidationResult(validator_name="QUALITY")
    result.fail_gate(_GATE_MESSAGE)
    return result


def _add_finding(result: ValidationResult, severity: Severity, message: str) -> None:
    result.add_finding(
        Finding(
            category="QUALITY",
            severity=severity,
            check_name="quality_efficiency",
            message=message,
            file_path="SKILL.md",
        )
    )


def _blocking_section(output: str) -> str:
    return output.split("## Blocking Findings\n\n", 1)[1].split("\n## ", 1)[0]


@pytest.mark.parametrize("with_findings", [False, True])
def test_html_shows_gate_without_repeating_finding_errors(with_findings: bool) -> None:
    result = _quality_result()
    if with_findings:
        _add_finding(result, Severity.MEDIUM, "Advisory quality observation")
        _add_finding(result, Severity.HIGH, "Blocking quality finding")

    output = HTMLReporter().render(result)
    messages = [
        html.unescape(message)
        for message in re.findall(r'<span class="finding-message">(.*?)</span>', output, re.DOTALL)
    ]

    expected = [_GATE_MESSAGE]
    if with_findings:
        expected.extend(["Advisory quality observation", "Blocking quality finding"])
    assert messages == expected


def test_html_escapes_gate_error_text() -> None:
    result = ValidationResult(validator_name="QUALITY")
    result.fail_gate("Quality score is below <minimum> & requires review")

    output = HTMLReporter().render(result)

    assert '<span class="finding-message">Quality score is below &lt;minimum&gt; &amp; requires review</span>' in output


@pytest.mark.parametrize("with_findings", [False, True])
def test_benchmark_lists_gate_before_findings(with_findings: bool) -> None:
    result = _quality_result()
    if with_findings:
        _add_finding(result, Severity.MEDIUM, "Advisory quality observation")

    output = BenchmarkReporter(include_timestamp=False).render(result)
    blocking = _blocking_section(output)

    assert blocking.startswith(f"- **GATE** QUALITY: {_GATE_MESSAGE}\n")
    assert blocking.count(_GATE_MESSAGE) == 1
    if with_findings:
        assert blocking.index(_GATE_MESSAGE) < blocking.index("Advisory quality observation")


def test_benchmark_prioritizes_gates_within_blocking_findings_limit() -> None:
    result = _quality_result()
    result.fail_gate("Second required quality check failed")
    result.fail_gate("Third required quality check failed")
    _add_finding(result, Severity.CRITICAL, "Critical quality finding")

    output = BenchmarkReporter(include_timestamp=False, max_findings_shown=2).render(result)
    blocking = _blocking_section(output)

    assert len([line for line in blocking.splitlines() if line.startswith("- ")]) == 2
    assert _GATE_MESSAGE in blocking
    assert "Second required quality check failed" in blocking
    assert "Third required quality check failed" not in blocking
    assert "Critical quality finding" not in blocking
    assert "Critical quality finding" in output


def test_benchmark_shares_limit_between_gates_and_findings() -> None:
    finding_result = ValidationResult(validator_name="QUALITY")
    _add_finding(finding_result, Severity.HIGH, "High quality finding")
    _add_finding(finding_result, Severity.CRITICAL, "Critical quality finding")

    output = BenchmarkReporter(include_timestamp=False, max_findings_shown=2).render_all(
        [finding_result, _quality_result()]
    )
    blocking = _blocking_section(output)

    assert len([line for line in blocking.splitlines() if line.startswith("- ")]) == 2
    assert blocking.index(_GATE_MESSAGE) < blocking.index("Critical quality finding")
    assert "High quality finding" not in blocking


@pytest.mark.parametrize(
    "private_path",
    ["/Users/private/catalog/SKILL.md", "/home/private/catalog/SKILL.md", r"C:\Users\private\catalog\SKILL.md"],
)
def test_benchmark_sanitizes_gate_messages_for_publication(private_path: str) -> None:
    private_environment = "private-evaluation-runner"
    result = ValidationResult(validator_name="QUALITY <threshold>")
    result.fail_gate(
        f"Quality score is below the required minimum in {private_path}; "
        f"review <details> [policy](https://example.invalid) on {private_environment}"
    )
    tier3 = ValidationResult(validator_name="AGENT_EVAL")
    tier3.metadata["agent_eval"] = {"summary": {"environment": private_environment}}

    output = BenchmarkReporter(include_timestamp=False).render_all([result, tier3])
    blocking = _blocking_section(output)

    assert "Quality score is below the required minimum" in blocking
    assert "SKILL.md" in blocking
    assert private_path not in output
    assert private_environment not in output
    assert "Isolated sandbox" in blocking
    assert "QUALITY &lt;threshold&gt;" in blocking
    assert "&lt;details&gt;" in blocking
    assert r"\[policy\](https&#58;//example.invalid)" in blocking
