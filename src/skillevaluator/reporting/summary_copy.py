# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping

SKILL_LIFT_DEFINITION = "Skill Lift = with-skill Overall − without-skill Overall"  # noqa: RUF001


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

    def rate_value(condition: Mapping[str, object]) -> float | None:
        value = condition.get("rate")
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
            return float(value)
        return None

    with_rate = rate_value(with_skill)
    without_rate = rate_value(without_skill)
    with_rate_text = f"{with_rate * 100:.1f}%" if with_rate is not None else "N/A"
    without_rate_text = f"{without_rate * 100:.1f}%" if without_rate is not None else "N/A"

    lift = pass_at_k.get("lift")
    delta = lift.get("delta") if isinstance(lift, Mapping) else None
    if (
        with_rate is not None
        and without_rate is not None
        and isinstance(delta, int | float)
        and not isinstance(delta, bool)
        and math.isfinite(delta)
        and -1 <= delta <= 1
    ):
        delta_text = f"{delta * 100:+.1f} percentage points"
    else:
        delta_text = "N/A"

    return (
        f"Pass@{k} (supporting reliability: share of cases with a passing attempt, "
        f"attempt passes at {threshold_text}): with skill {with_rate_text}, "
        f"without skill {without_rate_text}; Pass-rate difference {delta_text}."
    )
