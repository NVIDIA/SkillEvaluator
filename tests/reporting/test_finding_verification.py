# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification rationale stays visible without changing finding gates or exports."""

from __future__ import annotations

import copy
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import CLIReporter, HTMLReporter, JSONReporter, MarkdownReporter
from skillevaluator.reporting.base import passes_required_gate
from skillevaluator.validators.policy import ValidationPolicy, apply_policy
from skillevaluator.validators.security import SecurityValidator

_SGR = re.compile(r"\x1b\[[0-9;]*m")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _text(fragment: str) -> str:
    parser = _Text()
    parser.feed(fragment)
    return " ".join(" ".join(parser.parts).split())


def _finding(**overrides) -> Finding:
    fields = {
        "category": "SECURITY",
        "severity": Severity.HIGH,
        "check_name": "example-check",
        "message": "Review the artifact operation",
        "file_path": "SKILL.md",
        "line_number": 10,
        "line_content": "build artifact",
        "suggestion": "Review the operation and its context.",
    }
    return Finding(**(fields | overrides))


def _result(finding: Finding) -> ValidationResult:
    result = ValidationResult(validator_name="SECURITY", validator_description="Security review")
    result.add_finding(finding)
    return result


def _render(format_name: str, result: ValidationResult) -> tuple[str, str]:
    if format_name == "html":
        output = HTMLReporter(include_timestamp=False, explicit_skill_name="example-skill").render(result)
        blocks = re.findall(r'<div class="llm-verification">(.*?)</div>', output, re.DOTALL)
        return output, _text(" ".join(blocks))
    if format_name == "markdown":
        output = MarkdownReporter(include_timestamp=False, include_details=False).render(result)
        return output, _text(output)
    output = CLIReporter().render(result)
    # Rich adds SGR styling in terminal/color environments. Preserve every other
    # control byte for the safety assertion instead of stripping arbitrary ANSI.
    plain = _SGR.sub("", output)
    assert _CONTROL.search(plain) is None, repr(plain)
    return output, " ".join(plain.split())


@pytest.mark.parametrize("format_name", ["html", "markdown", "cli"])
@pytest.mark.parametrize(
    ("verdict", "confidence", "severity", "passed", "verdict_label", "effect"),
    [
        ("false_positive", "medium", Severity.HIGH, False, "False positive", "Reported severity: HIGH"),
        ("uncertain", "medium", Severity.HIGH, False, "Uncertain", "Reported severity: HIGH"),
        ("true_positive", "high", Severity.HIGH, False, "True positive", "Reported severity: HIGH"),
        (
            "false_positive",
            "high",
            Severity.INFO,
            True,
            "False positive",
            "Verifier recorded a downgrade to INFO; reported severity: INFO",
        ),
    ],
)
def test_verified_finding_shows_reasoning_and_actual_gate(
    tmp_path: Path,
    format_name: str,
    verdict: str,
    confidence: str,
    severity: Severity,
    passed: bool,
    verdict_label: str,
    effect: str,
) -> None:
    """Dropping metadata in a reporter or lowering the confidence threshold breaks this."""
    finding = _finding()
    result = _result(finding)
    reasoning = "The reference points to generated build artifacts."
    with patch("skillevaluator.inference.FindingVerifier") as verifier:
        verifier.return_value.verify.return_value = {
            0: {"verdict": verdict, "confidence": confidence, "rationale": reasoning}
        }
        SecurityValidator(verify_llm=True)._verify_findings_with_llm(result, tmp_path)

    before = copy.deepcopy(result)
    exported_before = JSONReporter(include_timestamp=False).render(result)
    output, visible = _render(format_name, result)

    assert "LLM verification:" in visible
    assert verdict_label in visible
    assert f"{confidence} confidence" in visible
    assert reasoning in visible
    assert effect in visible
    if not passed:
        assert "only high-confidence false positives are downgraded to INFO" in visible
    assert finding.severity == severity
    assert result.passed is passed
    assert passes_required_gate(result) is passed
    assert result == before
    assert JSONReporter(include_timestamp=False).render(result) == exported_before
    assert json.loads(exported_before)["findings"][0]["metadata"]["llm_reasoning"] == reasoning
    assert finding.message in output
    if format_name == "markdown":
        assert finding.suggestion in MarkdownReporter(include_timestamp=False).render(result)
    else:
        assert finding.suggestion in output
    if format_name == "html":
        # Both the skill and validator views show the explanation, including advisory INFO cards.
        assert output.count('<div class="llm-verification">') == 2


@pytest.mark.parametrize("format_name", ["html", "markdown", "cli"])
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"llm_reasoning": "Unverified metadata"},
        {"llm_verdict": ["false_positive"], "llm_confidence": "medium", "llm_reasoning": "Invalid verdict"},
        {"llm_verdict": "unexpected", "llm_confidence": "medium", "llm_reasoning": "Invalid verdict"},
        {"llm_verdict": "false_positive", "llm_confidence": {}, "llm_reasoning": "Invalid confidence"},
        {"llm_verdict": "false_positive", "llm_confidence": "certain", "llm_reasoning": "Invalid confidence"},
        {"llm_verdict": "false_positive", "llm_confidence": "medium", "llm_reasoning": {"text": "Invalid reason"}},
    ],
)
def test_missing_or_invalid_verification_metadata_adds_no_verdict(format_name: str, metadata: dict) -> None:
    result = _result(_finding(metadata=metadata))

    output, visible = _render(format_name, result)

    assert "LLM verification:" not in visible
    assert '<div class="llm-verification">' not in output
    assert "Severity retained" not in visible
    assert not result.passed


@pytest.mark.parametrize("format_name", ["html", "markdown", "cli"])
def test_verification_reasoning_is_inert_and_bounded(format_name: str) -> None:
    prefix = '<script>alert("review")</script> [link=https://example.test]text[/link] | @team'
    reason = prefix + "\x1b[31m\n" + "x" * 10_000 + "UNBOUNDED_TAIL"
    finding = _finding(metadata={"llm_verdict": "false_positive", "llm_confidence": "medium", "llm_reasoning": reason})

    output, visible = _render(format_name, _result(finding))

    assert "LLM verification:" in visible
    assert '<script>alert("review")</script>' in visible
    assert "UNBOUNDED_TAIL" not in visible
    assert "[reasoning truncated]" in visible
    assert len(visible) < 12_000
    assert "\x1b" not in visible
    assert finding.metadata["llm_reasoning"] == reason
    if format_name == "html":
        blocks = re.findall(r'<div class="llm-verification">(.*?)</div>', output, re.DOTALL)
        assert blocks
        assert all("<script>" not in block and "&lt;script&gt;" in block for block in blocks)
    elif format_name == "markdown":
        assert "&#91;link=" in output
        assert "&#124;" in output
        assert "&#64;team" in output
        assert len([line for line in output.splitlines() if line.startswith("|")]) == 3
    else:
        assert "[link=https://example.test]text[/link]" in visible


def test_html_keeps_occurrence_specific_verification_reasoning() -> None:
    result = _result(
        _finding(
            metadata={
                "llm_verdict": "false_positive",
                "llm_confidence": "medium",
                "llm_reasoning": "First occurrence uses an artifact reference.",
            }
        )
    )
    result.add_finding(
        _finding(
            line_number=20,
            metadata={
                "llm_verdict": "uncertain",
                "llm_confidence": "medium",
                "llm_reasoning": "Second occurrence lacks enough context.",
            },
        )
    )

    output, visible = _render("html", result)

    assert output.count('<div class="llm-verification">') == 4
    assert "First occurrence uses an artifact reference." in visible
    assert "Second occurrence lacks enough context." in visible
    assert "False positive" in visible
    assert "Uncertain" in visible


@pytest.mark.parametrize("format_name", ["html", "markdown", "cli"])
def test_policy_can_restore_blocking_severity_after_verifier_downgrade(tmp_path: Path, format_name: str) -> None:
    """Policy remapping must not hide the verifier's recorded downgrade or the final gate."""
    finding = _finding()
    result = _result(finding)
    with patch("skillevaluator.inference.FindingVerifier") as verifier:
        verifier.return_value.verify.return_value = {
            0: {"verdict": "false_positive", "confidence": "high", "rationale": "A documented placeholder."}
        }
        SecurityValidator(verify_llm=True)._verify_findings_with_llm(result, tmp_path)
    assert result.passed
    apply_policy([result], ValidationPolicy(severity_overrides={"SECURITY.example-check": Severity.HIGH}))

    before = copy.deepcopy(result)
    _, visible = _render(format_name, result)

    assert "Verifier recorded a downgrade to INFO; reported severity: HIGH" in visible
    assert "no verification downgrade recorded" not in visible
    assert "high confidence" in visible
    assert finding.metadata["downgraded"] is True
    assert finding.severity == Severity.HIGH
    assert not passes_required_gate(result)
    assert result == before


@pytest.mark.parametrize("force_color", [False, True])
def test_cli_verification_is_visible_with_or_without_color(monkeypatch: pytest.MonkeyPatch, force_color: bool) -> None:
    """Rich's own styling must not obscure semantic assertions or permit terminal controls."""
    monkeypatch.setenv("TERM", "xterm-256color" if force_color else "dumb")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("TTY_COMPATIBLE", raising=False)
    if force_color:
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    hostile_sgr = "\x1b[38;2;1;2;3m"
    reasoning = (
        '<script>alert("review")</script> [link=https://example.test]text[/link] '
        f"{hostile_sgr}\x1b[2J\x1b]8;;https://example.test\x07click\x1b]8;;\x07"
    )
    result = _result(
        _finding(metadata={"llm_verdict": "false_positive", "llm_confidence": "medium", "llm_reasoning": reasoning})
    )

    output, visible = _render("cli", result)

    # Check unmodified output: normalizing SGR would otherwise hide a leaked input style.
    assert hostile_sgr not in output
    assert ("\x1b[" in output) is force_color
    assert "False positive (medium confidence)" in visible
    assert "only high-confidence false positives are downgraded to INFO" in visible
    assert '<script>alert("review")</script>' in visible
    assert "[link=https://example.test]text[/link]" in visible
    assert "[38;2;1;2;3m" in visible
    assert "\x1b[2J" not in output
    assert "\x1b]8;" not in output
    assert not passes_required_gate(result)
