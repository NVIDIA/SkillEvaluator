# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for secret-pattern false positives in Harbor log redaction.

The ``sk-``/``nvapi-`` key shapes must only match at a token boundary. Before
this fix the patterns were unanchored, so ``sk-`` matched inside ordinary
hyphenated words (``task-granularity`` -> ``sk-granularity``) and mangled
normal log lines into ``task-<redacted>``.
"""

from __future__ import annotations

from skillevaluator.tier3.harbor.secret_redaction import redact_secrets_in_log_line


def _fixture_secret(*parts: str) -> str:
    """Build committed fake secrets from pieces so static scanners do not flag them."""
    return "".join(parts)


def test_benign_hyphenated_words_are_not_redacted():
    line = "tune task-granularity and task-parallel for Mask-conditioned kernels"

    assert redact_secrets_in_log_line(line) == line


def test_real_sk_key_is_still_redacted():
    secret = _fixture_secret("sk-", "abcdefgh", "ijklmnop")
    line = f"export NVIDIA_API_KEY={secret}"

    red = redact_secrets_in_log_line(line)

    assert secret not in red
    assert "sk-<redacted>" in red


def test_real_nvapi_key_is_still_redacted():
    secret = _fixture_secret("nvapi-", "abcdefgh", "ijklmnop")
    line = f"key {secret} in config"

    red = redact_secrets_in_log_line(line)

    assert secret not in red
    assert "nvapi-<redacted>" in red


def test_key_glued_to_word_char_is_still_redacted():
    # No separator before the key, but the body is a strong real-key signature
    # (>=20 alnum with lower+upper+digit), so it must still be redacted.
    secret = _fixture_secret("sk-", "Ab1Cd2Ef3", "Gh4Ij5Kl6", "Mn7Op8")
    line = f"log{secret} continues"

    red = redact_secrets_in_log_line(line)

    assert secret not in red
    assert "sk-<redacted>" in red


def test_glued_id_or_hash_is_not_redacted():
    # Lowercase hex IDs/hashes glued onto a word must not be mangled.
    line = "trial task-3f9a2b1c8d7e6f5a4b3c2d1e finished"

    assert redact_secrets_in_log_line(line) == line
