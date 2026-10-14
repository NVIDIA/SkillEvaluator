# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 synthetic dataset creation and live agent evaluation."""

from importlib import import_module
from typing import Any

from skillevaluator.tier3.change_detection import Tier3RunDecision, tier3_run_decision

__all__ = [
    "Tier3RunDecision",
    "compare_results",
    "create_dataset",
    "doctor",
    "evaluate",
    "tier3_run_decision",
    "validate_evals",
    "view_results",
]


def __getattr__(name: str) -> Any:
    # CLI registration and help must work without the optional Harbor runtime.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("skillevaluator.tier3.commands"), name)
    globals()[name] = value
    return value
