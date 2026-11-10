# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping

SKILL_LIFT_DEFINITION = (
    "Skill Lift compares Overall on the metrics and cases both arms scored, so it can differ from subtracting the "
    "displayed Overall scores."
)


def pass_at_k_summary(pass_at_k: object, attempt_policy: object = None) -> str | None:
    if not isinstance(pass_at_k, Mapping):
        return None

    with_skill = pass_at_k.get("with_skill")
    without_skill = pass_at_k.get("without_skill")
    with_skill = with_skill if isinstance(with_skill, Mapping) else {}
    without_skill = without_skill if isinstance(without_skill, Mapping) else {}
    policy = attempt_policy if isinstance(attempt_policy, Mapping) else {}

    k = next(
        (
            value
            for value in (with_skill.get("k"), without_skill.get("k"), pass_at_k.get("k"), policy.get("max_attempts"))
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ),
        None,
    )
    if k is None:
        return None

    threshold = next(
        (
            value
            for value in (
                with_skill.get("pass_threshold"),
                without_skill.get("pass_threshold"),
                policy.get("pass_threshold"),
            )
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        ),
        None,
    )
    threshold_text = f"Overall ≥ {threshold:.2f}" if threshold is not None else "the recorded Overall threshold"

    def rate_value(value: object) -> float | None:
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
            return float(value)
        return None

    with_rate = rate_value(with_skill.get("rate"))
    without_rate = rate_value(without_skill.get("rate"))
    with_rate_text = f"{with_rate * 100:.1f}%" if with_rate is not None else "N/A"
    without_rate_text = f"{without_rate * 100:.1f}%" if without_rate is not None else "N/A"

    lift = pass_at_k.get("lift")
    lift = lift if isinstance(lift, Mapping) else {}
    arm_delta = with_rate - without_rate if with_rate is not None and without_rate is not None else None
    delta_text = f"{arm_delta * 100:+.1f} percentage points" if arm_delta is not None else "N/A"

    shared_clause = ""
    if lift.get("basis") == "shared_metrics":
        shared_with = rate_value(lift.get("with_skill"))
        shared_without = rate_value(lift.get("without_skill"))
        if shared_with is not None and shared_without is not None:
            shared_delta = (shared_with - shared_without) * 100
            shared_clause = (
                f" Shared-metric pass-rate difference {shared_delta:+.1f} percentage points "
                f"(with skill {shared_with * 100:.1f}%, without skill {shared_without * 100:.1f}%)."
            )
        else:
            shared_clause = " Shared-metric pass-rate difference N/A."

    return (
        f"Pass@{k} (supporting reliability: share of cases with a passing attempt, "
        f"attempt passes at {threshold_text}): with skill {with_rate_text}, "
        f"without skill {without_rate_text}; Pass-rate difference {delta_text}.{shared_clause}"
    )
