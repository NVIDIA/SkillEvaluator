# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for secret-pattern false positives in the standalone Harbor
verifier (``src/skillevaluator/tier3/harbor/templates/eval.py``).

This template runs inside the Harbor sandbox with no package import and is
the copy that actually computed the displayed Security score. Its
``_SECRET_PATTERNS`` are duplicated from ``skillevaluator.tier3.eval_core.checks`` by design
(zero-dependency), so it needs its own regression coverage: the ``sk-``/
``nvapi-`` detectors must only match real keys at a token boundary, not
substrings of ordinary hyphenated words like ``task-granularity``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "skillevaluator"
    / "tier3"
    / "harbor"
    / "templates"
    / "eval.py"
)


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()

BENIGN_WORDS = [
    "task-granularity",
    "task-parallel",
    "task-conditioned",
    "Mask-conditioned",
    "disk-allocation",
]

REAL_SECRETS = [
    "sk-abcdefgh12345678",
    "nvapi-abcdefgh12345678",
    # Key glued onto a word char with no separator -> strong real-key signature.
    "logxsk-Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8",
    # Glued AWS access key ID (uppercase + digit, >=16 body).
    "dataAKIAIOSFODNN7EXAMPLE",
]

GLUED_NON_SECRETS = [
    "task-3f9a2b1c8d7e6f5a4b3c2d1e",
    "disk-deadbeefcafebabe12345678",
    "wordAKIANOTAREALKEYALLLETTERS",
]


def _traj(agent_message: str) -> dict:
    return {"steps": [{"source": "agent", "message": agent_message}]}


def _read_call(observation: str) -> dict:
    return {
        "action": "Read",
        "action_input": {"file_path": "references/partitioning-and-balance.md"},
        "observation": observation,
    }


@pytest.mark.parametrize("word", BENIGN_WORDS)
def test_template_benign_words_in_observation_are_not_secrets(word):
    result = eval_template.check_security(
        _traj("Analysis complete."), [_read_call(f"The {word} pattern matters.")]
    )

    assert result["score"] == 1.0
    assert not any(f["type"] == "secret_exposure" for f in result["findings"])


@pytest.mark.parametrize("word", BENIGN_WORDS)
def test_template_benign_words_in_agent_response_are_not_secrets(word):
    result = eval_template.check_security(_traj(f"Finding: {word} rule."), [])

    assert not any(f["type"] == "secret_leak" for f in result["findings"])


@pytest.mark.parametrize("secret", REAL_SECRETS)
def test_template_real_key_in_observation_is_flagged(secret):
    result = eval_template.check_security(
        _traj("Done."), [_read_call(f"config: {secret}")]
    )

    assert result["score"] == 0.0
    assert any(f["type"] == "secret_exposure" for f in result["findings"])


@pytest.mark.parametrize("token", GLUED_NON_SECRETS)
def test_template_glued_id_or_hash_tokens_are_not_secrets(token):
    result = eval_template.check_security(
        _traj("Done."), [_read_call(f"trial id: {token}")]
    )

    assert result["score"] == 1.0
    assert not any(f["type"] == "secret_exposure" for f in result["findings"])


# Every security constant duplicated between the standalone verifier and
# skillevaluator.tier3.eval_core.checks. The verifier must copy these (zero-dependency,
# runs in-sandbox), so silent drift between the copies is the root-cause class
# behind the unanchored-regex bug -- guard all of them, not just secrets.
_SHARED_SECURITY_CONSTANTS = [
    "_SECRET_PATTERNS",
    "_DESTRUCTIVE_PATTERNS",
    "_UNAUTHORIZED_PATHS",
    "_SENSITIVE_WRITE_PATHS",
    "_PROMPT_INJECTION_PATTERNS",
    "_EXECUTION_TOOL_HINTS",
    "_READ_TOOL_HINTS",
    "_WRITE_TOOL_HINTS",
    "WASTE_INDICATORS",
]


def _normalize(value):
    """Make compiled patterns / tuples / sequences comparable across modules."""
    if hasattr(value, "pattern"):  # compiled regex
        return ("re", value.pattern, value.flags)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(item) for item in value)
    return value


@pytest.mark.parametrize("name", _SHARED_SECURITY_CONSTANTS)
def test_security_constants_stay_in_sync_with_eval_core(name):
    from skillevaluator.tier3.eval_core import checks as eval_core_checks

    assert hasattr(eval_template, name), f"template missing {name}"
    assert hasattr(eval_core_checks, name), f"eval_core.checks missing {name}"
    assert _normalize(getattr(eval_template, name)) == _normalize(
        getattr(eval_core_checks, name)
    ), f"{name} drifted between templates/eval.py and eval_core/checks.py"


@pytest.mark.parametrize(
    "line",
    [
        "plain text with task-granularity is unchanged",
        "token sk-Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8",
        "catalog nvapi-Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8",
        "cursor crsr_deadbeefcafebabe",
        "openshift sha256~abcdefghijklmnop",
        (
            "jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJhZG1pbiI6dHJ1ZX0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "runtime opaque-secret-value",
    ],
)
def test_template_log_redaction_matches_eval_core(line):
    from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line

    extra_secret_values = ["opaque-secret-value"]

    assert eval_template.redact_secrets_in_log_line(
        line,
        extra_secret_values=extra_secret_values,
    ) == redact_secrets_in_log_line(
        line,
        extra_secret_values=extra_secret_values,
    )
