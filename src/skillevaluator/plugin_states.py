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
# Set by the native load census when the harness reported that a staged component
# did not load. It is not evaluated, but it was staged, so reports count it apart
# from the components that were never staged.
NOT_LOADED_STATE = "not_loaded"
# A skill or rule declared by reference is a coverage row named by its ref
# (``gitlab::<group>/<repo>::skills::release-notes``). When the ref resolved to a
# local member, the row also records under this key the name staging gave that
# member (``release-notes``), the name the load census and activation labels use.
COVERAGE_MEMBER_KEY = "member"
