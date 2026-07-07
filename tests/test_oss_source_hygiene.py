# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for product-neutral public source and telemetry naming."""

from __future__ import annotations

import tomllib
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from skillevaluator import telemetry
from skillevaluator.evaluation.insights_judge import _SYSTEM_PROMPT as INSIGHTS_SYSTEM_PROMPT
from skillevaluator.inference.finding_verifier import FindingVerifier
from skillevaluator.validators.secrets import SecretsValidator

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_generation_uses_a_product_neutral_browser_login_example() -> None:
    source = (REPO_ROOT / "src/skillevaluator/tier3/generate_dataset.py").read_text(encoding="utf-8")

    assert "browser_login.py" in source


def test_llm_judge_prompts_are_product_neutral() -> None:
    assert "NVIDIA" not in INSIGHTS_SYSTEM_PROMPT
    assert "NVIDIA" not in FindingVerifier._SYSTEM_PROMPT


def test_harbor_scoring_gap_uses_skillevaluator_error_type() -> None:
    span = object()
    with (
        patch.object(telemetry, "trace_span", return_value=nullcontext(span)),
        patch.object(telemetry, "set_span_attributes"),
        patch.object(telemetry, "mark_span_error") as mark_span_error,
        patch.object(telemetry, "counter"),
    ):
        telemetry.record_harbor_scoring_gap(
            skill_name="sample-skill",
            agent="codex",
            variant="with-skill",
            expected_scored_attempts=1,
            actual_scored_attempts=0,
        )

    assert mark_span_error.call_args.kwargs["error_type"] == "SkillEvaluatorHarborNoScoredTrials"


def test_nvidia_secret_rules_cover_public_keys_only() -> None:
    rules = tomllib.loads(SecretsValidator._NVIDIA_RULES)["rules"]

    assert {rule["id"] for rule in rules} == {"nvidia-api-key", "nvidia-ngc-api-key"}
    assert {rule["description"] for rule in rules} == {"NVIDIA API Key", "NVIDIA NGC API Key"}
