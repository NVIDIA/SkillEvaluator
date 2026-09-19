# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Harbor task.toml verifier environment forwarding for retry settings."""

from __future__ import annotations

from skillevaluator.tier3.harbor.adapter import (
    _VERIFIER_PROVIDER_ENV_VARS,
    _verifier_env_block,
    _verifier_env_vars,
)


def test_verifier_provider_env_vars_includes_retry_settings() -> None:
    """Verify _VERIFIER_PROVIDER_ENV_VARS allowlist contains all retry configuration keys."""
    expected_retry_vars = {
        "SKILL_EVAL_LLM_MAX_RETRIES",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY",
    }
    assert expected_retry_vars.issubset(_VERIFIER_PROVIDER_ENV_VARS)


def test_verifier_env_block_forwards_retry_settings() -> None:
    """Verify _verifier_env_block generates task.toml env lines for staged retry variables."""
    runtime_env = {
        "SKILL_EVAL_LLM_MAX_RETRIES": "5",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "2.0",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY": "40.0",
        "UNRELATED_HOST_VAR": "secret",
    }
    env_vars = _verifier_env_vars(runtime_env)
    assert "SKILL_EVAL_LLM_MAX_RETRIES" in env_vars
    assert "SKILL_EVAL_LLM_RETRY_BASE_DELAY" in env_vars
    assert "SKILL_EVAL_LLM_RETRY_MAX_DELAY" in env_vars
    assert "UNRELATED_HOST_VAR" not in env_vars

    env_block = _verifier_env_block(runtime_env, indent="    ")
    assert '    SKILL_EVAL_LLM_MAX_RETRIES = "${SKILL_EVAL_LLM_MAX_RETRIES}"' in env_block
    assert '    SKILL_EVAL_LLM_RETRY_BASE_DELAY = "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"' in env_block
    assert '    SKILL_EVAL_LLM_RETRY_MAX_DELAY = "${SKILL_EVAL_LLM_RETRY_MAX_DELAY}"' in env_block
    assert "UNRELATED_HOST_VAR" not in env_block
