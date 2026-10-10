# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result metadata for Tier 2 deduplication checks that were skipped or refused unsafe input.

Reporters read these keys: ``skipped`` and ``execution_status`` show a check
as skipped rather than passed, ``optional`` keeps a skip from counting as an
incomplete requested run, and ``security_failure`` keeps a refusal of unsafe
input blocking where findings are otherwise advisory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult


def mark_advisory_skip(result: ValidationResult, reason: str, **extra: object) -> ValidationResult:
    """Record that an advisory check did not run, with ``reason`` as its warning.

    Every reporter then shows an optional skip, not a pass or a failure.
    ``extra`` adds check-specific metadata, such as the limit that was exceeded.
    """
    result.add_warning(reason)
    result.metadata.update(
        {
            "advisory_tier2": True,
            "skipped": True,
            "execution_status": "skipped",
            "skip_reason": reason,
            "optional": True,
            **extra,
        }
    )
    return result


def mark_security_failure(result: ValidationResult) -> ValidationResult:
    """Keep a check that refused unsafe input blocking: failed and not optional."""
    result.passed = False
    result.metadata.update({"security_failure": True, "execution_status": "failed", "optional": False})
    return result


__all__ = ["mark_advisory_skip", "mark_security_failure"]
