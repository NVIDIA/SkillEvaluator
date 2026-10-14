# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential redaction: representative outputs and linear-time matching."""

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


# Synthetic fixtures are assembled from split literals so secret scanners do not flag them.
_JWT = "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0." + "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(_JWT, "jwt-<redacted>", id="start-of-text"),
        pytest.param(f"x {_JWT} y", "x jwt-<redacted> y", id="after-space"),
        pytest.param(f'"{_JWT}"', '"jwt-<redacted>"', id="quoted"),
        pytest.param(f"jwt={_JWT}", "jwt=jwt-<redacted>", id="after-equals"),
        pytest.param(f"Bearer {_JWT}", "Bearer jwt-<redacted>", id="after-bearer"),
        pytest.param(f"x-{_JWT}", "x-jwt-<redacted>", id="glued-to-dash"),
        pytest.param(f"abc-def-{_JWT}", "abc-def-jwt-<redacted>", id="glued-to-dashed-run"),
        pytest.param(f"eyJ-eyJ-{_JWT}", "jwt-<redacted>", id="run-starts-with-eyJ"),
        pytest.param(f"{_JWT}.{_JWT}", "jwt-<redacted>.jwt-<redacted>", id="two-dotted"),
        pytest.param(f"a_{_JWT}", f"a_{_JWT}", id="glued-to-underscore-kept"),
        pytest.param(f"x{_JWT}", f"x{_JWT}", id="glued-to-letter-kept"),
        pytest.param(f"é{_JWT}", f"é{_JWT}", id="glued-to-non-ascii-letter-kept"),
    ],
)
def test_redact_sensitive_text_redacts_jwt_by_position(source: str, expected: str) -> None:
    assert redact_sensitive_text(source) == expected


_BEGIN = "-----BEGIN "
_END = "-----END "
_PEM_BODY = "synthetic-key-material"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            f"{_BEGIN}X-Y PRIVATE KEY-----\n{_PEM_BODY}\n{_END}X-Y PRIVATE KEY-----",
            "private-key-<redacted>",
            id="dashed-label-word",
        ),
        pytest.param(
            f"{_BEGIN}A-B-C PRIVATE KEY 1-----\n{_PEM_BODY}\n{_END}A-B-C PRIVATE KEY 1-----",
            "private-key-<redacted>",
            id="words-before-and-after",
        ),
        pytest.param(
            f"{_BEGIN}PRIVATE KEYS PRIVATE KEY-----\n{_PEM_BODY}\n{_END}PRIVATE KEYS PRIVATE KEY-----",
            "private-key-<redacted>",
            id="later-private-key-pair",
        ),
        pytest.param(
            f"{_BEGIN}PRIVATE KEY-----{_PEM_BODY}{_END}PRIVATE KEY-----",
            "private-key-<redacted>",
            id="single-line",
        ),
        pytest.param(
            f"{_BEGIN}A{_BEGIN}RSA PRIVATE KEY-----\n{_PEM_BODY}\n{_END}RSA PRIVATE KEY-----",
            f"{_BEGIN}Aprivate-key-<redacted>",
            id="glued-to-earlier-header",
        ),
        pytest.param(
            f"{_END}CERTIFICATE-----\n{_BEGIN}PRIVATE KEY-----\n{_PEM_BODY}",
            "-----END CERTIFICATE-----\nprivate-key-<redacted>",
            id="truncated-after-earlier-end",
        ),
        pytest.param(f"{_BEGIN}PRIVATE KEY-----END x", "private-key-<redacted>", id="truncated-header-runs-into-end"),
        pytest.param(
            f"{_BEGIN}PRIVATE KEY{_BEGIN}PRIVATE KEY-----\n{_PEM_BODY}",
            "private-key-<redacted>",
            id="truncated-overlapping-headers",
        ),
    ],
)
def test_redact_sensitive_text_redacts_private_key_labels(source: str, expected: str) -> None:
    assert redact_sensitive_text(source) == expected


# Inputs shaped to make the former patterns backtrack. For the assignment key pattern,
# ``\b[a-z0-9_.-]*(?:word)[a-z0-9_.-]*``: a word boundary at every offset of one long
# key-character run, many sensitive words in one run, and runs followed by a value
# that never matches. For the JWT pattern: an ``eyJ`` start after every "-" of one
# run. For the private-key patterns: header labels that ran on across later headers,
# many "PRIVATE KEY" pairs in one label, and many headers before one final "-----END ".
# Each took seconds with the former patterns and takes milliseconds now.
_ADVERSARIAL_INPUTS = {
    "boundary-at-every-offset": "a-" * 5_000,
    "dotted-boundary-at-every-offset": "x." * 5_000,
    "repeated-word-without-separator": "token" * 10_000,
    "dashed-repeated-word-without-separator": "token-" * 400,
    "unterminated-quote": "token-" * 300 + "token='" + "x" * 3_000,
    "blank-colon-value": "token" * 5_000 + ":" + "\n" * 25_000 + ",",
    "blank-equals-value": "token" * 5_000 + "=" + " " * 25_000 + ",",
    "jwt-start-after-every-dash": "eyJ-" * 32_768,
    "pem-label-across-headers": f"{_BEGIN}A" * 10_923,
    "pem-repeated-private-key-headers": f"{_BEGIN}PRIVATE KEY-----" * 4_855,
    "pem-repeated-private-key-pairs": _BEGIN + "PRIVATE KEY " * 5_000,
    "pem-headers-before-final-end": f"{_BEGIN}PRIVATE KEY-----\n" * 7_000 + f"{_END}x",
    "pem-label-word-ending-in-dash": f"{_BEGIN}PRIVATE KEY X-" * 2_622,
}
# Every other adversarial input must come back unchanged.
_ADVERSARIAL_OUTPUTS = {
    "pem-repeated-private-key-headers": "private-key-<redacted>",
    "pem-label-word-ending-in-dash": "private-key-<redacted>",
}


@pytest.mark.parametrize(("name", "source"), list(_ADVERSARIAL_INPUTS.items()), ids=list(_ADVERSARIAL_INPUTS))
def test_redact_sensitive_text_is_fast_on_adversarial_inputs(name: str, source: str) -> None:
    started = time.perf_counter()
    redacted = redact_sensitive_text(source)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"redaction took {elapsed:.2f}s on a {len(source)}-character input"
    assert redacted == _ADVERSARIAL_OUTPUTS.get(name, source)


def test_redact_sensitive_text_redacts_value_after_long_key_run() -> None:
    prefix = "a-" * 5_000

    started = time.perf_counter()
    redacted = redact_sensitive_text(f"{prefix}token=s3cr3t\n{prefix}secret: 's3cr3t'")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"redaction took {elapsed:.2f}s"
    assert redacted == f"{prefix}token=<redacted>\n{prefix}secret:<redacted>"
