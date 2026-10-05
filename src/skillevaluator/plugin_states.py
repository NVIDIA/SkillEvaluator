# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State vocabularies of plugin evaluation: dependency resolution and Tier 3 component coverage.

The producers (:mod:`skillevaluator.plugin_dependencies`,
:mod:`skillevaluator.plugin_components`, and Tier 3 staging) and the reports
(:mod:`skillevaluator.reporting.plugin_sections`) share these names. This module
imports nothing, so a report can use them without loading the validators that
produce the states.
"""

from __future__ import annotations

# How a declared bundle-reference dependency resolved (see skillevaluator.plugin_dependencies).
DEPENDENCY_STATES: tuple[str, ...] = ("provided", "referenced", "missing", "external", "unresolved")

# Tier 3 coverage states. Staging assigns one of COVERAGE_STATES to each component;
# after the run the native load census ("loaded") and runtime evidence ("exercised")
# can raise a row by rank, never lower it. A row in one of these ranked states counts
# as evaluated.
COVERAGE_STATES: tuple[str, ...] = ("staged", "not_staged", "unsupported", "unavailable", "invalid")
COVERAGE_STATE_RANK: dict[str, int] = {"staged": 1, "loaded": 2, "exercised": 3}
EVALUATED_COVERAGE_STATES: frozenset[str] = frozenset(COVERAGE_STATE_RANK)
