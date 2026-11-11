# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only, bounded presentation of second-pass finding verification."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from skillevaluator.models import Finding, Severity

_VERDICTS = {
    "true_positive": "True positive",
    "false_positive": "False positive",
    "uncertain": "Uncertain",
}
_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
_REASONING_MAX_CHARS = 1024


@dataclass(frozen=True)
class FindingVerification:
    """Inert text for a finding's recorded verification, never a gate decision."""

    verdict: str
    confidence: str
    reasoning: str
    severity_effect: str


def finding_verification(finding: Finding | dict) -> FindingVerification | None:
    """Project recognized verifier metadata without changing findings or exports.

    Bound reasoning before normalization; renderers still escape it for their
    output format. Drop malformed metadata rather than presenting it as a verdict.
    """
    if isinstance(finding, dict):
        metadata = finding.get("metadata")
        severity = finding.get("severity")
    elif isinstance(finding, Finding):
        metadata = finding.metadata
        severity = finding.severity
    else:
        return None
    if not isinstance(metadata, dict):
        return None
    verdict = metadata.get("llm_verdict")
    confidence = metadata.get("llm_confidence")
    reasoning = metadata.get("llm_reasoning")
    if (
        not isinstance(verdict, str)
        or verdict not in _VERDICTS
        or not isinstance(confidence, str)
        or confidence not in _CONFIDENCE_LEVELS
        or not isinstance(reasoning, str)
    ):
        return None

    bounded = reasoning[:_REASONING_MAX_CHARS]
    bounded = " ".join(
        "".join(" " if unicodedata.category(character).startswith("C") else character for character in bounded).split()
    )
    if len(reasoning) > _REASONING_MAX_CHARS:
        bounded += "… [reasoning truncated]"

    severity_value = severity.value if isinstance(severity, Severity) else severity
    if not isinstance(severity_value, str) or severity_value not in {item.value for item in Severity}:
        return None
    if verdict == "false_positive" and confidence == "high":
        if metadata.get("downgraded") is True:
            effect = f"Verifier recorded a downgrade to INFO; reported severity: {severity_value.upper()}."
        else:
            effect = f"Reported severity: {severity_value.upper()}; no verification downgrade recorded."
    else:
        effect = (
            f"Reported severity: {severity_value.upper()}; "
            "only high-confidence false positives are downgraded to INFO by verification."
        )
    return FindingVerification(_VERDICTS[verdict], confidence, bounded, effect)
