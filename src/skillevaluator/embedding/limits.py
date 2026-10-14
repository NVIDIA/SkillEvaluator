# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated resource limits for inter-skill similarity scans."""

from skillevaluator.constants import SIMILARITY_MAX_ENTRIES


def validate_max_entries(max_entries: int) -> None:
    if type(max_entries) is not int or not 1 <= max_entries <= SIMILARITY_MAX_ENTRIES:
        raise ValueError(f"max_entries (--max-entries) must be an integer from 1 to {SIMILARITY_MAX_ENTRIES}")


def validate_max_scalar_comparisons(max_scalar_comparisons: int) -> None:
    if type(max_scalar_comparisons) is not int or max_scalar_comparisons < 1:
        raise ValueError("max_scalar_comparisons (--max-scalar-comparisons) must be a positive integer")
