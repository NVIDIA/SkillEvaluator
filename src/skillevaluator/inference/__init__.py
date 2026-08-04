# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference subsystem for SkillEvaluator.

Provides the unified :class:`LLMClient` and concrete implementations
for LLM-powered analysis tasks.
"""

from skillevaluator.inference.client import LLMClient
from skillevaluator.inference.finding_verifier import FindingVerifier
from skillevaluator.inference.types import (
    LLMClientError,
    LLMConfigError,
    LLMVerdict,
    parse_bounded_llm_verdict,
    require_bounded_llm_string,
    require_bounded_llm_string_list,
    validate_tier2_llm_prompt,
    validate_tier2_llm_prompt_batch,
    validate_tier2_llm_similarity_score,
)

__all__ = [
    "FindingVerifier",
    "LLMClient",
    "LLMClientError",
    "LLMConfigError",
    "LLMVerdict",
    "parse_bounded_llm_verdict",
    "require_bounded_llm_string",
    "require_bounded_llm_string_list",
    "validate_tier2_llm_prompt",
    "validate_tier2_llm_prompt_batch",
    "validate_tier2_llm_similarity_score",
]
