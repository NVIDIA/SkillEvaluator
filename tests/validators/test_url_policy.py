# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL helpers shared by the MCP, hook, and endpoint policies (no network)."""

from __future__ import annotations

import pytest

from skillevaluator.validators.url_policy import UrlCredentials, url_credentials

_TOKEN = "ghp_" + "0123456789abcdefghij0123456789abcdef"


@pytest.mark.parametrize(
    ("url", "any_userinfo", "expected"),
    [
        ("https://user:secret@h.example/x", True, True),
        ("https://${USER}:${TOKEN}@h.example/x", True, True),
        ("https://deploy@h.example/x", True, True),
        ("https://admin:hunter2@h.example:99999/x", True, True),
        ("https://h.example/x", True, False),
        ("https://user:secret@h.example/x", False, True),
        ("https://x-access-token:${GITHUB_TOKEN}@github.com/org/repo.git", False, False),
        ("https://deploy@h.example/x", False, False),
        (f"https://{_TOKEN}@h.example/x", False, True),
        ("ssh://git@github.com/org/repo.git", False, False),
    ],
)
def test_url_credentials_reads_the_userinfo_as_written(url: str, any_userinfo: bool, expected: bool) -> None:
    assert url_credentials(url, any_userinfo=any_userinfo).userinfo is expected


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
    for any_userinfo in (True, False):
        assert url_credentials(f"https://h.example/x?{query}", any_userinfo=any_userinfo).query_keys == keys


def test_url_credentials_is_false_when_the_url_carries_none() -> None:
    assert not url_credentials("https://h.example/x?page=2", any_userinfo=True)
    assert not UrlCredentials()
    assert UrlCredentials(query_keys=("token",))
