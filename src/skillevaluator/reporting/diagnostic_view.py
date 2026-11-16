# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only display groups for persisted dataset expectation diagnostics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

__all__ = ["build_dataset_diagnostics_view"]

_STATUS_ORDER = ("unmet", "not_checked", "unscored", "not_applicable", "met")
_MEMBER_FIELDS = {"expectation_id", "evidence", "unmet_checks", "inapplicable_checks"}


def _records(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _status_groups(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    statuses = list(_STATUS_ORDER)
    for judgment in judgments:
        if judgment.get("status") not in statuses:
            statuses.append(judgment.get("status"))

    result = []
    for status in statuses:
        members = [judgment for judgment in judgments if judgment.get("status") == status]
        if not members:
            continue
        groups: list[dict[str, Any]] = []
        causes: list[dict[str, Any]] = []
        for judgment in members:
            # Compare complete structured causes, never fragments of prose. The
            # requirement, excerpts and rubric checks stay on their own member;
            # any other differing association prevents a shared explanation.
            cause = {key: value for key, value in judgment.items() if key not in _MEMBER_FIELDS}
            if cause in causes:
                groups[causes.index(cause)]["judgments"].append(judgment)
            else:
                causes.append(cause)
                groups.append(
                    {
                        "reason": judgment.get("reason"),
                        "next_step": judgment.get("next_step"),
                        "judgments": [judgment],
                    }
                )
        result.append({"status": status, "count": len(members), "groups": groups})
    return result


def build_dataset_diagnostics_view(case_diagnostics: object) -> list[dict[str, Any]]:
    """Add exact explanation groups inside each retained attempt and status.

    No records are scored, combined across attempts, or newly bounded here.
    Original fields, member associations and omission counts are copied intact;
    canonical report data is never mutated or replaced by the display groups.
    """
    cases = deepcopy(_records(case_diagnostics))
    for case in cases:
        for attempt in _records(case.get("attempts")):
            attempt["status_groups"] = _status_groups(_records(attempt.get("judgments")))
    return cases
