# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL helpers shared by the MCP, hook, and endpoint policies (no network)."""

from __future__ import annotations

import pytest

from skillevaluator.validators.url_policy import (
    MAX_REPORT_CHARS,
    UrlCredentials,
    UserinfoRule,
    has_secret_shape,
    report_text,
    safe_url,
    url_credentials,
)

_TOKEN = "ghp_" + "0123456789abcdefghij0123456789abcdef"


@pytest.mark.parametrize(
    ("url", "rule", "expected"),
    [
        ("https://user:secret@h.example/x", "any", True),
        ("https://${USER}:${TOKEN}@h.example/x", "any", True),
        ("https://deploy@h.example/x", "any", True),
        ("https://admin:hunter2@h.example:99999/x", "any", True),
        ("https://h.example/x", "any", False),
        ("https://user:secret@h.example/x", "literal", True),
        ("https://${USER}:${TOKEN}@h.example/x", "literal", False),
        ("https://$USER@h.example/x", "literal", False),
        ("https://user:${TOKEN}@h.example/x", "literal", True),
        ("https://${USER}:secret@h.example/x", "literal", True),
        ("https://deploy@h.example/x", "literal", True),
        ("https://h.example/x", "literal", False),
        ("https://user:secret@h.example/x", "secret", True),
        ("https://x-access-token:${GITHUB_TOKEN}@github.com/org/repo.git", "secret", False),
        ("https://deploy@h.example/x", "secret", False),
        (f"https://{_TOKEN}@h.example/x", "secret", True),
        ("ssh://git@github.com/org/repo.git", "secret", False),
    ],
)
def test_url_credentials_reads_the_userinfo_as_written(url: str, rule: UserinfoRule, expected: bool) -> None:
    assert url_credentials(url, userinfo_rule=rule).userinfo is expected


@pytest.mark.parametrize(
    ("query", "keys"),
    [
        ("api_key=literal", ("api_key",)),
        (f"q={_TOKEN}", ("q",)),
        ("token=a&page=2&token=b", ("token",)),
        ("api_key=${API_KEY}", ()),
        ("api_key=", ()),
        ("page=2", ()),
    ],
)
def test_url_credentials_reads_credential_names_and_secret_shaped_values(query: str, keys: tuple[str, ...]) -> None:
    for rule in ("any", "literal", "secret"):
        assert url_credentials(f"https://h.example/x?{query}", userinfo_rule=rule).query_keys == keys


def test_url_credentials_is_false_when_the_url_carries_none() -> None:
    assert not url_credentials("https://h.example/x?page=2", userinfo_rule="any")
    assert not UrlCredentials()
    assert UrlCredentials(query_keys=("token",))


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("https://admin:hunter2@hooks.example.com/x?token=abc#frag", "https://hooks.example.com/x"),
        ("https:admin:hunter2@evil.example/mcp", "https://evil.example/mcp"),
        ("https://evil.net\\.example.com/x", "https://evil.net/.example.com/x"),
        ("https://[::1]:8443/mcp", "https://[::1]:8443/mcp"),
        ("https://admin:hunter2@h.example:99999/x", "https://h.example:99999/x"),
        (f"https://hooks.example.com/notify/{_TOKEN}/x", "https://hooks.example.com/notify/<redacted>/x"),
        (f"https://h.example:bad/{_TOKEN}", "https://h.example:bad/<redacted>"),
    ],
)
def test_safe_url_shows_where_a_client_connects_without_credentials(url: str, shown: str) -> None:
    assert safe_url(url) == shown


def test_safe_url_is_bounded() -> None:
    assert len(safe_url("https://h.example/" + "a" * 10_000)) <= MAX_REPORT_CHARS


def test_report_text_redacts_every_known_secret_shape() -> None:
    text = report_text(
        f"gh auth login --with-token {_TOKEN}; curl https://user:pw@h.example/x -H 'X: xoxb-1234567890abc'"
    )

    assert _TOKEN not in text and "xoxb-1234567890abc" not in text and "user:pw" not in text
    assert text.startswith("gh auth login --with-token <redacted>;")


@pytest.mark.parametrize("prefix", ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"])
def test_every_github_token_prefix_has_a_secret_shape(prefix: str) -> None:
    token = prefix + "0123456789abcdefghij0123456789abcdef"
    assert has_secret_shape(token)
    assert url_credentials(f"https://h.example/mcp?q={token}", userinfo_rule="literal").query_keys == ("q",)


def test_fine_grained_github_token_has_a_secret_shape() -> None:
    assert has_secret_shape("github_pat_" + "11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz")
    assert not has_secret_shape("github_pat_short")
