# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only unambiguous finding identities may receive model verdicts."""

from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.inference.finding_verifier import FindingVerifier
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.validators.security import SecurityValidator


def _verdict(index: str, verdict: str = "false_positive") -> str:
    return f'{{"index": {index}, "verdict": "{verdict}", "confidence": "high", "rationale": "Documentation example."}}'


@pytest.mark.parametrize("index", ["true", "false", "0.9", "1.9", "0.0", '"0"', '" 0 "', "null", "-1", "1e400"])
def test_non_integer_or_negative_identity_does_not_alias_a_finding(index: str) -> None:
    valid = _verdict("2", "true_positive")

    verdicts = FindingVerifier._parse_verdicts(_verdict(index) + "\n" + valid)

    assert list(verdicts) == [2]
    assert verdicts[2]["verdict"] == "true_positive"


@pytest.mark.parametrize(
    "reply",
    [
        _verdict("0") + "\n" + _verdict("0", "true_positive"),
        _verdict("0", "true_positive") + "\n" + _verdict("0"),
        "\n".join([_verdict("0"), _verdict("0", "uncertain"), _verdict("0")]),
        _verdict("0") + "\n" + _verdict("0"),
    ],
    ids=["false-then-true", "true-then-false", "third-does-not-restore", "same-verdict"],
)
def test_repeated_identity_is_withheld_without_dropping_other_findings(reply: str) -> None:
    verdicts = FindingVerifier._parse_verdicts(reply + "\n" + _verdict("1"))

    assert list(verdicts) == [1]
    assert verdicts[1]["verdict"] == "false_positive"


@pytest.mark.parametrize(
    "reply",
    [
        _verdict('1, "index": 0'),
        _verdict('1, "inde\\u0078": 0'),
        _verdict("0").replace('"verdict":', '"verdict": "true_positive", "verdict":'),
        _verdict("0").replace('"confidence":', '"confidence": "low", "confidence":'),
    ],
    ids=["index", "escaped-index", "verdict", "confidence"],
)
def test_duplicate_members_cannot_choose_the_effective_verdict(reply: str) -> None:
    assert FindingVerifier._parse_verdicts(reply + "\n" + _verdict("2")) == {
        2: {
            "index": 2,
            "verdict": "false_positive",
            "confidence": "high",
            "rationale": "Documentation example.",
        }
    }


def test_unique_verdicts_and_complete_truncated_lines_keep_their_identity() -> None:
    reply = "Some explanation\n```json\n" + _verdict("1", "true_positive") + "\n" + _verdict("0")
    reply += '\n{"index": 2, "verdict": "false_pos'

    verifier = FindingVerifier()
    verdicts = verifier._parse_truncated_response(reply)

    assert list(verdicts) == [1, 0]
    assert verdicts[1]["index"] == 1
    assert verdicts[0]["index"] == 0


@pytest.mark.parametrize(
    "reply",
    [
        _verdict("false"),
        _verdict("0.9"),
        _verdict("0", "true_positive") + "\n" + _verdict("0"),
        _verdict("0").replace('"verdict":', '"verdict": "true_positive", "verdict":'),
    ],
    ids=["boolean-index", "fractional-index", "repeated-index", "duplicate-member"],
)
def test_ambiguous_verification_keeps_finding_severity_and_reports_partial_coverage(tmp_path: Path, reply: str) -> None:
    (tmp_path / "SKILL.md").write_text("Example documentation.\n", encoding="utf-8")
    result = ValidationResult()
    for index in range(2):
        result.add_structured_finding(
            Finding(
                category="PII",
                severity=Severity.HIGH,
                check_name=f"finding-{index}",
                message=f"Finding {index}",
                file_path="SKILL.md",
            ),
            is_error=True,
        )
    verifier = FindingVerifier()
    with (
        patch.object(verifier, "completions", return_value=reply + "\n" + _verdict("1")),
        patch("skillevaluator.inference.FindingVerifier", return_value=verifier),
    ):
        SecurityValidator(verify_llm=True)._verify_findings_with_llm(result, tmp_path)

    assert result.findings[0].severity == Severity.HIGH
    assert "llm_verdict" not in result.findings[0].metadata
    assert result.findings[1].severity == Severity.INFO
    assert result.findings[1].metadata["llm_verdict"] == "false_positive"
    assert result.metadata["incomplete_scans"] == ["llm-verification"]
    assert any("1 of 2 findings" in message for message in result.messages)
