# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GitHub, GitLab, Slack, Hugging Face, and npm tokens and AWS access keys are redacted from logs and
artifacts, and nothing else is.

``redact_sensitive_text`` masks artifacts and reports, ``redact_secrets_in_log_line``
masks Tier 3 logs, progress, and judge evidence, and the Harbor verifier has a
standalone copy of the latter. All of them use one set of token patterns, which
``skillevaluator.utils.redaction`` defines; the verifier's copy is pinned to it in
test_harbor_template_secret_patterns.py. The artifact redactor masks more on top:
a URL's whole userinfo, and a token shape glued into a longer word.
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
_ARTIFACTS = redaction.redact_sensitive_text
_BASE_REDACTORS = {
    "artifacts": _ARTIFACTS,
    "logs": secret_redaction.redact_secrets_in_log_line,
    "verifier-logs": eval_template.redact_secrets_in_log_line,
}
_REDACTORS = {
    **_BASE_REDACTORS,
    "judge-evidence": atif_helpers._redact_evidence_text,
    "verifier-judge-evidence": eval_template._redact_evidence_text,
}
# Every redactor but the artifact one: the log redactor and the wrappers built on it.
_LOG_REDACTORS = {name: redact for name, redact in _REDACTORS.items() if redact is not _ARTIFACTS}
REDACTORS = pytest.mark.parametrize("redact", list(_REDACTORS.values()), ids=list(_REDACTORS))
BASE_REDACTORS = pytest.mark.parametrize("redact", list(_BASE_REDACTORS.values()), ids=list(_BASE_REDACTORS))
LOG_REDACTORS = pytest.mark.parametrize("redact", list(_LOG_REDACTORS.values()), ids=list(_LOG_REDACTORS))


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
    "slack-rotating-refresh": ("xoxe-", _fixture_secret("1-", _SLACK_BODY)),
    "hugging-face": ("hf_", _fixture_secret("AbCdEfGhIj", "KlMnOpQrSt", "UvWxYz0123", "4567")),
    "npm": ("npm_", _fixture_secret("AbCdEfGhIj", "KlMnOpQrSt", "UvWxYz0123", "456789")),
}


@REDACTORS
@pytest.mark.parametrize(("prefix", "body"), list(_TOKENS.values()), ids=list(_TOKENS))
def test_every_token_shape_is_redacted_and_keeps_its_prefix(redact, prefix: str, body: str) -> None:
    token = prefix + body

    assert redact(f"pushed with {token} to origin") == f"pushed with {prefix}<redacted> to origin"


@LOG_REDACTORS
@pytest.mark.parametrize(("prefix", "body"), list(_TOKENS.values()), ids=list(_TOKENS))
def test_a_token_in_url_userinfo_keeps_its_prefix_in_logs(redact, prefix: str, body: str) -> None:
    token = prefix + body

    assert redact(f"https://{token}@git.example.com/o/r.git") == f"https://{prefix}<redacted>@git.example.com/o/r.git"


@pytest.mark.parametrize(("prefix", "body"), list(_TOKENS.values()), ids=list(_TOKENS))
def test_artifacts_redact_the_whole_url_userinfo(prefix: str, body: str) -> None:
    redacted = _ARTIFACTS(f"https://{prefix + body}@git.example.com/o/r.git")

    # The userinfo goes whole, so not even the token's prefix is left.
    assert redacted == "https://<redacted>@git.example.com/o/r.git"
    assert body not in redacted


# Text shaped like the start of a token that is not one, for every redactor: a bare or
# short prefix, a type letter Slack does not issue ("xoxo" is also a word), a Hugging
# Face prefix glued to a word, and hyphenated words.
NEAR_MISSES = [
    "see the ghp_, github_pat_, glpat- and xoxb- prefixes",
    "ghp_" + _CLASSIC_BODY[:35],
    "github_pat_" + "short",
    "glpat-" + "a" * 19,
    "a glpat-style-token-name",
    "xoxb-" + "123456789",
    "xoxo-hugs-and-kisses-from-me",
    "xoxz-" + "1234567890ab",
    "tune task-granularity and Mask-conditioned kernels",
    "npm_config_cache=/tmp/npm-cache npm_" + "a" * 35,
    "hf_" + "a" * 29,
    "xhf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz01234567",
]


# A token shape glued into a longer word, or a GitHub body past 255 characters (GitHub
# tokens end at a word boundary within 255 characters), as ``(text, the token text
# artifacts must not keep)``. The log redactors keep the text; the artifact redactor
# also masks a token inside a name ("quality_ghp_..."), so it redacts these too.
GLUED_NEAR_MISSES = {
    "glued-github": ("xghp_" + _CLASSIC_BODY, _CLASSIC_BODY),
    "over-long-github": ("ghp_" + "a" * 300, "ghp_" + "a" * 36),
    "glued-gitlab": ("xglpat-" + "a" * 20, "glpat-" + "a" * 20),
    "glued-slack": ("uxoxb-" + "1234567890ab", "xoxb-1234567890ab"),
}
_GLUED_TEXTS = [text for text, _token in GLUED_NEAR_MISSES.values()]


@REDACTORS
@pytest.mark.parametrize("text", NEAR_MISSES)
def test_text_that_only_resembles_a_token_is_kept(redact, text: str) -> None:
    assert redact(text) == text


@LOG_REDACTORS
@pytest.mark.parametrize("text", _GLUED_TEXTS, ids=list(GLUED_NEAR_MISSES))
def test_logs_keep_a_token_shape_glued_into_a_word(redact, text: str) -> None:
    assert redact(text) == text


@pytest.mark.parametrize(("text", "token"), list(GLUED_NEAR_MISSES.values()), ids=list(GLUED_NEAR_MISSES))
def test_artifacts_redact_a_token_shape_glued_into_a_word(text: str, token: str) -> None:
    redacted = _ARTIFACTS(text)

    assert token not in redacted
    assert "<redacted>" in redacted


_AWS_KEYS = {
    "long-term": _fixture_secret("AKIA", "IOSFODNN", "7EXAMPLE"),
    "temporary": _fixture_secret("ASIA", "IOSFODNN", "7EXAMPLE"),
}


@REDACTORS
@pytest.mark.parametrize("key", list(_AWS_KEYS.values()), ids=list(_AWS_KEYS))
def test_aws_access_keys_are_redacted(redact, key: str) -> None:
    assert redact(f"signed with {key} for the upload") == "signed with aws-access-key-<redacted> for the upload"


@BASE_REDACTORS
def test_long_text_is_redacted_piece_by_piece(redact) -> None:
    # Every near miss beside every token, repeated to over 256 KiB: the result is the
    # redaction of one piece, repeated, so no match reaches across pieces.
    piece = " ".join([*NEAR_MISSES, *_GLUED_TEXTS, *(prefix + body for prefix, body in _TOKENS.values())]) + "\n"
    copies = (256 * 1024) // len(piece) + 1

    redacted = redact(piece * copies)

    assert redacted == redact(piece) * copies
    # Artifacts also redact each token shape glued into a word.
    per_piece = len(_TOKENS) + (len(_GLUED_TEXTS) if redact is _ARTIFACTS else 0)
    assert redacted.count("<redacted>") == per_piece * copies


# Every copy of the token patterns: the shared ones and the Harbor verifier's.
_ALL_TOKEN_PATTERNS = {
    "github": redaction.GITHUB_TOKEN_RE,
    "github-fine-grained": redaction.GITHUB_PAT_RE,
    "gitlab": redaction.GITLAB_PAT_RE,
    "slack": redaction.SLACK_TOKEN_RE,
    "hugging-face": redaction.HUGGING_FACE_TOKEN_RE,
    "npm": redaction.NPM_TOKEN_RE,
    "verifier-github": eval_template.LOG_GITHUB_TOKEN_RE,
    "verifier-github-fine-grained": eval_template.LOG_GITHUB_PAT_RE,
    "verifier-gitlab": eval_template.LOG_GITLAB_PAT_RE,
    "verifier-slack": eval_template.LOG_SLACK_TOKEN_RE,
    "verifier-hugging-face": eval_template.LOG_HUGGING_FACE_TOKEN_RE,
    "verifier-npm": eval_template.LOG_NPM_TOKEN_RE,
    "one-pass": redaction.PREFIXED_TOKEN_RE,
    "verifier-one-pass": eval_template.LOG_PREFIXED_TOKEN_RE,
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
        secret_redaction.LOG_HUGGING_FACE_TOKEN_RE,
        secret_redaction.LOG_NPM_TOKEN_RE,
    ) == redaction.PREFIXED_TOKEN_PATTERNS
    assert secret_redaction.LOG_PREFIXED_TOKEN_RE is redaction.PREFIXED_TOKEN_RE
    assert secret_redaction.LOG_AWS_ACCESS_KEY_RE is redaction.AWS_ACCESS_KEY_RE


def test_one_pass_redacts_what_each_pattern_redacts_in_turn() -> None:
    """The redactors read the text once, with the alternation of every token pattern."""
    text = " ".join([*NEAR_MISSES, *_GLUED_TEXTS, *(prefix + body for prefix, body in _TOKENS.values())])
    in_turn = text
    for pattern in redaction.PREFIXED_TOKEN_PATTERNS:
        in_turn = pattern.sub(r"\g<prefix><redacted>", in_turn)

    assert redaction.PREFIXED_TOKEN_RE.sub(secret_redaction.keep_token_prefix, text) == in_turn
    assert eval_template.LOG_PREFIXED_TOKEN_RE.sub(eval_template.keep_token_prefix, text) == in_turn


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
