# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for secret-pattern false positives in the security check.

The ``sk-``/``nvapi-`` key detectors must only match at a token boundary.
Before this fix the patterns were unanchored, so ``sk-`` matched inside
ordinary hyphenated words (``task-granularity`` -> ``sk-granularity``,
``Mask-conditioned`` -> ``sk-conditioned``). Skill docs that legitimately use
those words produced false-positive ``secret_leak`` / ``secret_exposure``
findings and collapsed the Security score (observed 0.22 on the
cupynumeric-migration-readiness skill).
"""

from __future__ import annotations

import pytest

from skillevaluator.tier3.eval_core.checks import check_security

# Words drawn from the cupynumeric-migration-readiness reference docs that
# previously tripped the unanchored ``sk-`` detector.
BENIGN_WORDS = [
    "task-granularity",
    "task-granularity-rule",
    "task-parallel",
    "task-conditioned",
    "Mask-conditioned",
    "disk-allocation",
]


def _fixture_secret(*parts: str) -> str:
    """Build committed fake secrets from pieces so static scanners do not flag them."""
    return "".join(parts)


REAL_SECRETS = [
    _fixture_secret("sk-", "abcdefgh", "12345678"),
    "export NVIDIA_API_KEY=" + _fixture_secret("sk-", "abcdefgh", "12345678"),
    _fixture_secret("nvapi-", "abcdefgh", "12345678"),
    # Key glued directly onto a word char with no separator: still caught via
    # the strong real-key signature (>=20 alnum with lower+upper+digit).
    "logx" + _fixture_secret("sk-", "Ab1Cd2Ef3", "Gh4Ij5Kl6", "Mn7Op8"),
    "token" + _fixture_secret("nvapi-", "Ab1Cd2Ef3", "Gh4Ij5Kl6", "Mn7Op8"),
    # Glued AWS access key ID (uppercase + digit, no lowercase, >=16 body).
    "data" + _fixture_secret("AKIA", "IOSFODNN7", "EXAMPLE"),
]

# Glued runs that look ID/hash-like must NOT be treated as secrets: lowercase
# hex IDs (no uppercase) and short tokens fail the real-key signature. The
# all-caps cases guard the glued-AKIA body (needs a digit; rejects plain words).
GLUED_NON_SECRETS = [
    "task-3f9a2b1c8d7e6f5a4b3c2d1e",
    "disk-deadbeefcafebabe12345678",
    "task-a1b2c3d4e5f6a7b8c9d0e1f2",
    "wordAKIANOTAREALKEYALLLETTERS",
    "shortAKIASHORT12",
]


def _read_call(observation: str) -> dict:
    return {
        "action": "Read",
        "action_input": {"file_path": "references/partitioning-and-balance.md"},
        "observation": observation,
    }


@pytest.mark.parametrize("word", BENIGN_WORDS)
def test_benign_hyphenated_words_in_tool_output_are_not_secrets(word):
    result = check_security(
        agent_text="Analysis complete.",
        tool_calls=[_read_call(f"The {word} pattern affects scaling.")],
    )

    assert result["passed"] is True
    assert result["score"] == 1.0
    assert not any(f["type"] == "secret_exposure" for f in result["findings"])


@pytest.mark.parametrize("word", BENIGN_WORDS)
def test_benign_hyphenated_words_in_agent_response_are_not_secrets(word):
    result = check_security(
        agent_text=f"The cuPyNumeric {word} rule governs partitioning.",
        tool_calls=[],
    )

    assert result["passed"] is True
    assert not any(f["type"] == "secret_leak" for f in result["findings"])


@pytest.mark.parametrize("secret", REAL_SECRETS)
def test_real_key_in_tool_output_is_still_flagged(secret):
    result = check_security(
        agent_text="Done.",
        tool_calls=[_read_call(f"config value: {secret}")],
    )

    assert result["passed"] is False
    assert result["score"] == 0.0
    [finding] = [f for f in result["findings"] if f["type"] == "secret_exposure"]
    assert finding["evidence"] == "[redacted secret exposure]"
    assert secret not in finding["evidence"]


def test_real_key_in_agent_response_is_still_flagged():
    result = check_security(
        agent_text="Use sk-abcdefgh12345678 to authenticate.",
        tool_calls=[],
    )

    assert result["passed"] is False
    assert any(f["type"] == "secret_leak" for f in result["findings"])


@pytest.mark.parametrize("token", GLUED_NON_SECRETS)
def test_glued_id_or_hash_tokens_are_not_secrets(token):
    result = check_security(
        agent_text="Done.",
        tool_calls=[_read_call(f"trial id: {token} completed")],
    )

    assert result["passed"] is True
    assert not any(f["type"] == "secret_exposure" for f in result["findings"])
