# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types for the SkillEvaluator inference subsystem."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from skillevaluator.constants import (
    TIER2_LLM_MAX_CALLS,
    TIER2_LLM_MAX_INPUT_SCALAR_CHARS,
    TIER2_LLM_MAX_PROMPT_CHARS,
    TIER2_LLM_MAX_REFERENCE_ITEMS,
    TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
    TIER2_LLM_MAX_TOTAL_PROMPT_CHARS,
)


class LLMClientError(Exception):
    """Raised when an LLM operation fails (missing key, bad response, etc.)."""


LLMConfigError = LLMClientError


@dataclass
class LLMVerdict:
    """Structured result from LLM verification of a content cluster."""

    verdict: str
    confidence: float
    reasoning: str
    suggestion: str


def require_bounded_llm_string(
    value: object,
    label: str,
    *,
    max_chars: int = TIER2_LLM_MAX_INPUT_SCALAR_CHARS,
) -> str:
    """Require a real, bounded string before prompt or report rendering."""
    if type(value) is not str:
        raise LLMClientError(f"{label} must be a string")
    if len(value) > max_chars:
        raise LLMClientError(f"{label} exceeds the {max_chars}-character limit")
    return value


def require_bounded_llm_string_list(
    value: object,
    label: str,
    *,
    max_items: int = TIER2_LLM_MAX_REFERENCE_ITEMS,
    max_chars: int = TIER2_LLM_MAX_INPUT_SCALAR_CHARS,
) -> list[str]:
    """Validate an LLM input list before copying or stringifying it."""
    if type(value) is not list:
        raise LLMClientError(f"{label} must be a list")
    if len(value) > max_items:
        raise LLMClientError(f"{label} exceeds the {max_items}-item limit")
    result: list[str] = []
    total_chars = 0
    for index, item in enumerate(value):
        bounded = require_bounded_llm_string(item, f"{label}[{index}]", max_chars=max_chars)
        total_chars += len(bounded)
        if total_chars > TIER2_LLM_MAX_PROMPT_CHARS:
            raise LLMClientError(f"{label} text exceeds the {TIER2_LLM_MAX_PROMPT_CHARS}-character prompt-input limit")
        result.append(bounded)
    return result


def validate_tier2_llm_prompt(prompt: object, *, context: str) -> str:
    """Validate one fully rendered Tier 2 user prompt before an LLM call."""
    return require_bounded_llm_string(
        prompt,
        f"{context} prompt",
        max_chars=TIER2_LLM_MAX_PROMPT_CHARS,
    )


def validate_tier2_llm_similarity_score(value: object, *, context: str) -> float:
    """Require a finite similarity score within [0, 1] before rendering."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LLMClientError(f"{context} similarity score must be a finite number within [0, 1]")
    try:
        score = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise LLMClientError(f"{context} similarity score must be a finite number within [0, 1]") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise LLMClientError(f"{context} similarity score must be a finite number within [0, 1]")
    return score


def validate_tier2_llm_prompt_batch(
    prompts: Collection[str],
    *,
    context: str,
    max_calls: int = TIER2_LLM_MAX_CALLS,
    max_total_chars: int = TIER2_LLM_MAX_TOTAL_PROMPT_CHARS,
) -> int:
    """Validate call count and cumulative rendered prompt characters."""
    if len(prompts) > max_calls:
        raise LLMClientError(f"{context} candidate/call count exceeds the {max_calls}-call limit")
    total_chars = 0
    for prompt in prompts:
        total_chars += len(validate_tier2_llm_prompt(prompt, context=context))
        if total_chars > max_total_chars:
            raise LLMClientError(f"{context} aggregate prompts exceed the {max_total_chars}-character limit")
    return total_chars


def parse_bounded_llm_verdict(
    data: object,
    *,
    valid_verdicts: Collection[str],
    context: str,
) -> LLMVerdict:
    """Normalize a parsed Tier 2 response into finite, bounded scalars."""
    if not isinstance(data, Mapping):
        raise LLMClientError(f"{context} response must be a JSON object")

    verdict = require_bounded_llm_string(
        data.get("verdict", ""),
        f"{context} verdict",
        max_chars=TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
    )
    if verdict not in valid_verdicts:
        raise LLMClientError(f"LLM returned unknown verdict '{verdict}'. Expected one of: {set(valid_verdicts)}")

    raw_confidence = data.get("confidence", 0.0)
    if not isinstance(raw_confidence, (int, float)) or isinstance(raw_confidence, bool):
        raise LLMClientError(f"{context} confidence must be a finite number within [0, 1]")
    try:
        confidence = float(raw_confidence)
    except (OverflowError, TypeError, ValueError) as exc:
        raise LLMClientError(f"{context} confidence must be a finite number within [0, 1]") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise LLMClientError(f"{context} confidence must be a finite number within [0, 1]")

    reasoning = require_bounded_llm_string(
        data.get("reasoning", ""),
        f"{context} reasoning",
        max_chars=TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
    )
    suggestion = require_bounded_llm_string(
        data.get("suggestion", ""),
        f"{context} suggestion",
        max_chars=TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
    )
    if "recommendation" in data:
        require_bounded_llm_string(
            data["recommendation"],
            f"{context} recommendation",
            max_chars=TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
        )

    return LLMVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        suggestion=suggestion,
    )
