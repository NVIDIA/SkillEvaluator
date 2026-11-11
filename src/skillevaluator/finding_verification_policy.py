# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared downgrade rule for finding verification and its report explanation."""

from skillevaluator.models import Severity

DOWNGRADE_VERDICT = "false_positive"
DOWNGRADE_CONFIDENCE = "high"
DOWNGRADE_SEVERITY = Severity.INFO


def should_downgrade(verdict: str, confidence: str) -> bool:
    """Return whether a verified finding meets the downgrade threshold."""
    return verdict == DOWNGRADE_VERDICT and confidence == DOWNGRADE_CONFIDENCE


def downgrade_policy_explanation() -> str:
    """Describe the same rule the producer applies, without inferring an action."""
    verdict_label = DOWNGRADE_VERDICT.replace("_", " ")
    return (
        f"only {DOWNGRADE_CONFIDENCE}-confidence {verdict_label}s "
        f"are downgraded to {DOWNGRADE_SEVERITY.value.upper()} by verification."
    )
