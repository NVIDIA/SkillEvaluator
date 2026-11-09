# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

_EMPTY_REFERENCE_VALUES = frozenset({"none", "null", "n/a", "-"})


def normalize_optional_reference(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        if not value:
            return None
        value = str(value)
    normalized = value.strip()
    if not normalized or normalized.casefold() in _EMPTY_REFERENCE_VALUES:
        return None
    return normalized


def normalize_expected_behaviors(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    behaviors = []
    for behavior in value:
        if isinstance(behavior, str):
            normalized = behavior.strip(" \t\n\r\f\v")
            if normalized and normalized.casefold() not in _EMPTY_REFERENCE_VALUES:
                behaviors.append(normalized)
        elif behavior:
            behaviors.append(behavior)
    return behaviors
