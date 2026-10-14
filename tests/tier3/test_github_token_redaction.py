# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GitHub tokens are masked in Tier 3 evidence excerpts on the host and in the Harbor verifier."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import atif_helpers, secret_redaction

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval_github_tokens", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()

REDACTORS = pytest.mark.parametrize(
    "redact",
    [atif_helpers._redact_evidence_text, eval_template._redact_evidence_text],
    ids=["host", "template"],
)


def _fixture_secret(*parts: str) -> str:
    """Build committed fake secrets from pieces so static scanners do not flag them."""
    return "".join(parts)


_CLASSIC_BODY = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


@REDACTORS
@pytest.mark.parametrize("prefix", ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"])
def test_classic_github_tokens_are_masked(redact, prefix):
    token = _fixture_secret(prefix, _CLASSIC_BODY)

    assert redact(f"token={token} done") == f"token={prefix}<redacted> done"


@REDACTORS
def test_fine_grained_github_pat_is_masked(redact):
    token = _fixture_secret(
        "github_pat_", "11ABCDEFG0123456789_", "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUV"
    )

    assert redact(f"GH_TOKEN={token}\n") == "GH_TOKEN=github_pat_<redacted>"


@REDACTORS
@pytest.mark.parametrize(
    "text", ["ghp_short", "see ghp_ and gho_ prefixes", "github_pat_short", "xghp_" + _CLASSIC_BODY]
)
def test_non_token_github_prefixes_are_untouched(redact, text):
    assert redact(text) == text


def _near_miss_text(copies: int) -> str:
    # Each run is one alphanumeric or underscore too long to end on a word boundary
    # within the 255-character limit, so every match attempt backtracks and fails.
    return ("ghp_" + "a" * 300 + " ") * copies + (" github_pat_" + "_" * 300) * copies


def _best_elapsed(pattern, text: str, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        pattern.sub("", text)
        best = min(best, time.perf_counter() - started)
    return best


@pytest.mark.parametrize(
    "pattern",
    [
        secret_redaction.LOG_GITHUB_TOKEN_RE,
        secret_redaction.LOG_GITHUB_PAT_RE,
        eval_template.LOG_GITHUB_TOKEN_RE,
        eval_template.LOG_GITHUB_PAT_RE,
    ],
    ids=["host-classic", "host-fine-grained", "template-classic", "template-fine-grained"],
)
def test_github_token_patterns_scale_linearly(pattern):
    small = _best_elapsed(pattern, _near_miss_text(250))
    large = _best_elapsed(pattern, _near_miss_text(2_000))

    # 8x the input takes about 8x the time when matching is linear, and 64x when it is quadratic.
    assert large < 24 * small
