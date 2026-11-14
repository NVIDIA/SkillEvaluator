# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types for the SkillEvaluator inference subsystem."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from skillevaluator.constants import TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS


class LLMClientError(Exception):
    """Raised when an LLM operation fails (missing key, bad response, etc.)."""


class EmptyLLMResponseError(LLMClientError):
    """Raised when a successful provider response contains no model text."""


class LLMClientRefusedError(LLMClientError):
    """Raised when the provider declines a request, such as a safety-classifier refusal."""


class LLMClientTruncatedError(LLMClientError):
    """Raised when a response stops at the output-token limit.

    ``content`` keeps the partial text for callers that can salvage it, and
    ``model_used`` names the model that wrote it when the client knows it.
    """

    def __init__(self, message: str, content: str = "", model_used: str | None = None) -> None:
        super().__init__(message)
        self.content = content
        self.model_used = model_used


LLMConfigError = LLMClientError


@dataclass
class LLMVerdict:
    """Structured result from LLM verification of a content cluster."""

    verdict: str
    confidence: float
    reasoning: str
    suggestion: str
    # The model that answered, as the client reported it: the configured model,
    # or a server-side fallback model that answered in its place. ``None`` when
    # the client named no model.
    model: str | None = None


def require_bounded_llm_string(value: object, label: str, *, max_chars: int) -> str:
    """Return ``value`` when it is a string of at most ``max_chars`` characters.

    Nothing is coerced: a number, list, mapping or ``None`` is an error, so a
    malformed reply never reaches a finding or a report as a non-string.
    """
    if type(value) is not str:
        raise LLMClientError(f"{label} must be a string")
    if len(value) > max_chars:
        raise LLMClientError(f"{label} exceeds the {max_chars}-character limit")
    return value


def _require_unit_interval(value: object, label: str) -> float:
    """Require a JSON number (not a bool or a string) that is finite and within [0, 1]."""
    message = f"{label} must be a finite number within [0, 1]"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LLMClientError(message)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise LLMClientError(message) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise LLMClientError(message)
    return number


def parse_bounded_llm_verdict(
    data: object,
    *,
    valid_verdicts: Collection[str],
    context: str,
    model: object = None,
) -> LLMVerdict:
    """Turn a parsed Tier 2 LLM reply into a verdict with checked, bounded fields.

    The reply must be a JSON object whose ``verdict`` is one of
    ``valid_verdicts`` and whose ``confidence`` is a number from 0 to 1. The
    ``rationale`` (or a legacy ``reasoning``) and ``suggestion`` fields must be
    strings of at most ``TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS`` characters. A
    missing confidence reads as 0 and a missing text field as empty. Any other
    reply raises :class:`LLMClientError`.

    ``model`` names the model that answered and is kept only when it is a
    string. It is stored as given; reports pass it through
    :func:`~skillevaluator.inference.diagnostics.safe_llm_model`.
    """
    if not isinstance(data, Mapping):
        raise LLMClientError(f"{context} response must be a JSON object")

    verdict = require_bounded_llm_string(
        data.get("verdict", ""),
        f"{context} verdict",
        max_chars=TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
    )
    if verdict not in valid_verdicts:
        raise LLMClientError(f"LLM returned unknown verdict '{verdict}'. Expected one of: {set(valid_verdicts)}")

    confidence = _require_unit_interval(data.get("confidence", 0.0), f"{context} confidence")
    reasoning = require_bounded_llm_string(
        data.get("rationale", data.get("reasoning", "")),
        f"{context} rationale",
        max_chars=TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
    )
    suggestion = require_bounded_llm_string(
        data.get("suggestion", ""),
        f"{context} suggestion",
        max_chars=TIER2_LLM_MAX_RESPONSE_SCALAR_CHARS,
    )
    return LLMVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        suggestion=suggestion,
        model=model if isinstance(model, str) else None,
    )
