# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central report-artifact naming for Skill Evaluator.

All generated report basenames are derived here so the final ``skillevaluator``
identity lives in exactly one place (and so an edition-aware rename, if ever
needed, is a one-line change). The benchmark card has a fixed filename to match
Skill Evaluator.
"""

from __future__ import annotations

#: Prefix for all generated report basenames.
REPORT_PREFIX = "skillevaluator"

#: Default report basename (full validate run).
DEFAULT_REPORT_BASENAME = f"{REPORT_PREFIX}-output"

#: Fixed filename for the publication-ready benchmark card.
BENCHMARK_FILENAME = "BENCHMARK.md"


def report_basename(kind: str | None = None) -> str:
    """Return the report basename for a report *kind* (e.g. ``"quality"``).

    ``report_basename()`` returns the default full-run basename.
    """
    if not kind:
        return DEFAULT_REPORT_BASENAME
    return f"{REPORT_PREFIX}-{kind}"
