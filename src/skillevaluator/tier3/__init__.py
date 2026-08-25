# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 synthetic dataset creation and live agent evaluation.

Command exports are loaded on first access so base-install utilities below the
``tier3`` package can be imported without importing Harbor and the other Tier 3
extras.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "compare_results",
    "create_dataset",
    "doctor",
    "evaluate",
    "validate_evals",
    "view_results",
]


def __getattr__(name: str) -> Any:
    """Load public command exports without making Harbor an eager dependency."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("skillevaluator.tier3.commands"), name)


def __dir__() -> list[str]:
    """Expose lazy command exports to interactive callers and documentation."""
    return sorted((*globals(), *__all__))
