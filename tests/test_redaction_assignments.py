# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensitive-assignment redaction: representative outputs and linear-time matching."""

from __future__ import annotations

import time

import pytest

from skillevaluator.utils.redaction import redact_sensitive_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param('api_key = "abc def"', "api_key=<redacted>", id="quoted-double"),
        pytest.param("API_KEY: 'abc def'", "API_KEY:<redacted>", id="quoted-single-colon"),
        pytest.param(
            'password="it\'s a secret" user=bob',
            "password=<redacted> user=bob",
            id="mixed-quotes-single-inside-double",
        ),
        pytest.param("token='say \"hi\"' next", "token=<redacted> next", id="mixed-quotes-double-inside-single"),
        pytest.param(
            "name=\"bob\" token=\"t1\" note='ok' secret='s2'",
            "name=\"bob\" token=<redacted> note='ok' secret=<redacted>",
            id="quoted-sensitive-and-plain-keys",
        ),
        pytest.param(
            'client_secret: "s3cr3t", region: us-east-1',
            "client_secret:<redacted>, region: us-east-1",
            id="quoted-then-colon",
        ),
        pytest.param("password: hunter2, user: bob", "password:<redacted>, user: bob", id="colon"),
        pytest.param("db.password : hunter2; retries: 3", "db.password:<redacted>; retries: 3", id="colon-dotted-key"),
        pytest.param(
            "token=a; secret=b, password=c",
            "token=<redacted>; secret=<redacted>, password=<redacted>",
            id="equals-several",
        ),
        pytest.param("--api-key=abc123 --verbose", "--api-key=<redacted> --verbose", id="equals-cli-flag"),
        pytest.param("..secret=abc", "..secret=<redacted>", id="equals-leading-dots"),
        pytest.param("étoken.secret=abc", "étoken.secret=<redacted>", id="equals-key-after-non-ascii-word"),
        pytest.param(
            "token_count=42 total_tokens: 100 max_tokens=5",
            "token_count=42 total_tokens: 100 max_tokens=5",
            id="token-counts-kept",
        ),
        pytest.param("password:\n  hunter2\nuser: bob", "password:<redacted>\nuser: bob", id="multiline-colon-value"),
        pytest.param(
            "export GITHUB_TOKEN=ghp_example\nexport PATH=/usr/bin\nsession_token: 'abc'\nprivate-key = \"xyz\"",
            "export GITHUB_TOKEN=<redacted>\nexport PATH=/usr/bin\nsession_token:<redacted>\nprivate-key=<redacted>",
            id="multiline-mixed",
        ),
    ],
)
def test_redact_sensitive_text_redacts_assignments(source: str, expected: str) -> None:
    assert redact_sensitive_text(source) == expected


# Inputs shaped to make the former key pattern, ``\b[a-z0-9_.-]*(?:word)[a-z0-9_.-]*``,
# backtrack: a word boundary at every offset of one long key-character run, many
# sensitive words in one run, and runs followed by a value that never matches. Each
# took several seconds with that pattern and takes milliseconds now.
_ADVERSARIAL_INPUTS = {
    "boundary-at-every-offset": "a-" * 5_000,
    "dotted-boundary-at-every-offset": "x." * 5_000,
    "repeated-word-without-separator": "token" * 10_000,
    "dashed-repeated-word-without-separator": "token-" * 400,
    "unterminated-quote": "token-" * 300 + "token='" + "x" * 3_000,
    "blank-colon-value": "token" * 5_000 + ":" + "\n" * 25_000 + ",",
    "blank-equals-value": "token" * 5_000 + "=" + " " * 25_000 + ",",
}


@pytest.mark.parametrize("source", list(_ADVERSARIAL_INPUTS.values()), ids=list(_ADVERSARIAL_INPUTS))
def test_redact_sensitive_text_is_fast_on_adversarial_assignments(source: str) -> None:
    started = time.perf_counter()
    redacted = redact_sensitive_text(source)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"redaction took {elapsed:.2f}s on a {len(source)}-character input"
    assert redacted == source


def test_redact_sensitive_text_redacts_value_after_long_key_run() -> None:
    prefix = "a-" * 5_000

    started = time.perf_counter()
    redacted = redact_sensitive_text(f"{prefix}token=s3cr3t\n{prefix}secret: 's3cr3t'")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"redaction took {elapsed:.2f}s"
    assert redacted == f"{prefix}token=<redacted>\n{prefix}secret:<redacted>"
