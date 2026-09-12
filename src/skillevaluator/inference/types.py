# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types for the SkillEvaluator inference subsystem."""

from __future__ import annotations

from dataclasses import dataclass


class LLMClientError(Exception):
    """Raised when an LLM operation fails (missing key, bad response, etc.)."""


class EmptyLLMResponseError(LLMClientError):
    """Raised when a successful provider response contains no model text."""


LLMConfigError = LLMClientError


@dataclass
class LLMVerdict:
    """Structured result from LLM verification of a content cluster."""

    verdict: str
    confidence: float
    reasoning: str
    suggestion: str
