# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GitHub, GitLab, and Slack tokens are redacted from logs and artifacts, and nothing else is.

``redact_sensitive_text`` masks artifacts and reports, ``redact_secrets_in_log_line``
masks Tier 3 logs, progress, and judge evidence, and the Harbor verifier has a
standalone copy of the latter. All of them use one set of token patterns, which
``skillevaluator.utils.redaction`` defines; the verifier's copy is pinned to it in
test_harbor_template_secret_patterns.py.
"""

from __future__ import annotations

import re
import re._parser
import subprocess
import sys

import pytest
from tests.conftest import load_harbor_eval_template

from skillevaluator.tier3.eval_core import atif_helpers, secret_redaction
from skillevaluator.utils import redaction

eval_template = load_harbor_eval_template("harbor_template_eval_prefixed_tokens")

# The redactors that write text out, and the judge-evidence wrappers built on the log redactor.
_BASE_REDACTORS = {
    "artifacts": redaction.redact_sensitive_text,
    "logs": secret_redaction.redact_secrets_in_log_line,
    "verifier-logs": eval_template.redact_secrets_in_log_line,
}
_REDACTORS = {
    **_BASE_REDACTORS,
    "judge-evidence": atif_helpers._redact_evidence_text,
    "verifier-judge-evidence": eval_template._redact_evidence_text,
}
REDACTORS = pytest.mark.parametrize("redact", list(_REDACTORS.values()), ids=list(_REDACTORS))
BASE_REDACTORS = pytest.mark.parametrize("redact", list(_BASE_REDACTORS.values()), ids=list(_BASE_REDACTORS))


def _fixture_secret(*parts: str) -> str:
    """Build committed fake secrets from pieces so static scanners do not flag them."""
    return "".join(parts)


_CLASSIC_BODY = _fixture_secret("A1b2C3d4E5", "f6G7h8I9j0", "K1l2M3n4O5", "p6Q7r8")
_SLACK_BODY = _fixture_secret("1234567890", "12-", "9876543210", "98-", "AbCdEfGhIj", "KlMnOpQrSt")
_TOKENS = {
    "github-personal": ("ghp_", _CLASSIC_BODY),
    "github-oauth": ("gho_", _CLASSIC_BODY),
    "github-user-to-server": ("ghu_", _CLASSIC_BODY),
    "github-server-to-server": ("ghs_", _CLASSIC_BODY),
    "github-refresh": ("ghr_", _CLASSIC_BODY),
    "github-fine-grained": (
        "github_pat_",
        _fixture_secret("11ABCDEFG0", "123456789_", "abcdefghij", "klmnopqrst", "uvwxyz0123", "456789ABCD"),
    ),
    "gitlab-personal": ("glpat-", _fixture_secret("aB3dE6gH9j", "K2mN5pQ8sT")),
    "gitlab-personal-long": ("glpat-", _fixture_secret("aB3dE6gH9j", "K2m-N5p_Q8", "sTuV0wX1yZ")),
    "slack-app": ("xoxa-", _fixture_secret("2-", _SLACK_BODY)),
    "slack-bot": ("xoxb-", _SLACK_BODY),
    "slack-user": ("xoxp-", _SLACK_BODY),
    "slack-refresh": ("xoxr-", _SLACK_BODY),
    "slack-session": ("xoxs-", _SLACK_BODY),
}


@REDACTORS
@pytest.mark.parametrize(("prefix", "body"), list(_TOKENS.values()), ids=list(_TOKENS))
def test_every_token_shape_is_redacted_and_keeps_its_prefix(redact, prefix: str, body: str) -> None:
    token = prefix + body

    assert redact(f"pushed with {token} to origin") == f"pushed with {prefix}<redacted> to origin"
    assert redact(f"https://{token}@git.example.com/o/r.git") == f"https://{prefix}<redacted>@git.example.com/o/r.git"


# Text shaped like the start of a token that is not one: a bare or short prefix, a
# prefix glued to a word, a type letter Slack does not issue ("xoxo" is also a word),
# an over-long GitHub body (GitHub tokens end at a word boundary within 255
# characters), and hyphenated words.
NEAR_MISSES = [
    "see the ghp_, github_pat_, glpat- and xoxb- prefixes",
    "ghp_" + _CLASSIC_BODY[:35],
    "xghp_" + _CLASSIC_BODY,
    "ghp_" + "a" * 300,
    "github_pat_" + "short",
    "glpat-" + "a" * 19,
    "xglpat-" + "a" * 20,
    "a glpat-style-token-name",
    "xoxb-" + "123456789",
    "uxoxb-" + "1234567890ab",
    "xoxo-hugs-and-kisses-from-me",
    "xoxz-" + "1234567890ab",
    "tune task-granularity and Mask-conditioned kernels",
]


@REDACTORS
@pytest.mark.parametrize("text", NEAR_MISSES)
def test_text_that_only_resembles_a_token_is_kept(redact, text: str) -> None:
    assert redact(text) == text


@BASE_REDACTORS
def test_long_text_is_redacted_piece_by_piece(redact) -> None:
    # Every near miss beside every token, repeated to over 256 KiB: the result is the
    # redaction of one piece, repeated, so no match reaches across pieces.
    piece = " ".join([*NEAR_MISSES, *(prefix + body for prefix, body in _TOKENS.values())]) + "\n"
    copies = (256 * 1024) // len(piece) + 1

    redacted = redact(piece * copies)

    assert redacted == redact(piece) * copies
    assert redacted.count("<redacted>") == len(_TOKENS) * copies


# Every copy of the token patterns: the shared ones and the Harbor verifier's.
_ALL_TOKEN_PATTERNS = {
    "github": redaction.GITHUB_TOKEN_RE,
    "github-fine-grained": redaction.GITHUB_PAT_RE,
    "gitlab": redaction.GITLAB_PAT_RE,
    "slack": redaction.SLACK_TOKEN_RE,
    "verifier-github": eval_template.LOG_GITHUB_TOKEN_RE,
    "verifier-github-fine-grained": eval_template.LOG_GITHUB_PAT_RE,
    "verifier-gitlab": eval_template.LOG_GITLAB_PAT_RE,
    "verifier-slack": eval_template.LOG_SLACK_TOKEN_RE,
}
# The longest token any pattern matches: "github_pat_" and a 255-character body.
_LONGEST_TOKEN_CHARS = len("github_pat_") + 255


@pytest.mark.parametrize("pattern", list(_ALL_TOKEN_PATTERNS.values()), ids=list(_ALL_TOKEN_PATTERNS))
def test_token_patterns_read_a_bounded_window(pattern: re.Pattern[str]) -> None:
    """A match attempt reads a bounded number of characters, so a scan is linear in the text.

    Without lookahead an attempt reads no further than the longest match it can
    make, plus the one character ``\\b`` looks at, so scanning n characters costs
    at most n times that bound, whatever the text holds.
    """
    assert "(?=" not in pattern.pattern
    assert "(?!" not in pattern.pattern
    _shortest, longest = re._parser.parse(pattern.pattern, pattern.flags).getwidth()
    assert longest <= _LONGEST_TOKEN_CHARS


def test_log_redaction_uses_the_artifact_token_patterns() -> None:
    assert (
        secret_redaction.LOG_GITHUB_TOKEN_RE,
        secret_redaction.LOG_GITHUB_PAT_RE,
        secret_redaction.LOG_GITLAB_PAT_RE,
        secret_redaction.LOG_SLACK_TOKEN_RE,
    ) == redaction.PREFIXED_TOKEN_PATTERNS


def test_the_eval_core_redactor_loads_without_the_utils_package() -> None:
    """eval_core loads on its own: the token patterns live with the log redactor, not under skillevaluator.utils,
    whose package __init__ loads the tool runner and pydantic."""
    code = (
        "import sys\n"
        "import skillevaluator.tier3.eval_core.checks\n"
        "print(sorted(name for name in ('skillevaluator.utils', 'pydantic') if name in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)

    assert result.stdout.strip() == "[]"
