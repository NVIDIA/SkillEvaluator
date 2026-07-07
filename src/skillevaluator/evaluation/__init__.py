# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Tier 3 evaluation service boundary.

``EvaluationService`` is the single entry point that both the CLI and the API
call for live agent evaluation, dataset creation, and result discovery. It wraps
the native in-process Harbor engine (``tier3``) so neither surface starts a
separate CLI process. Shared option models live in :mod:`skillevaluator.evaluation.options`.
"""

from skillevaluator.evaluation.dimension_judge import (
    DimensionJudge,
    compute_dimensions,
    compute_dimensions_deterministic,
)
from skillevaluator.evaluation.insights_judge import InsightsJudge, build_insights
from skillevaluator.evaluation.options import DatasetOptions, EvaluationOptions
from skillevaluator.evaluation.service import EvaluationService

__all__ = [
    "DatasetOptions",
    "DimensionJudge",
    "EvaluationOptions",
    "EvaluationService",
    "InsightsJudge",
    "build_insights",
    "compute_dimensions",
    "compute_dimensions_deterministic",
]
