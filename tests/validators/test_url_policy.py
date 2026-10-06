# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL helpers shared by the MCP, hook, and endpoint policies (no network)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.url_policy import (
    MAX_REPORT_CHARS,
    UrlCredentials,
    UserinfoRule,
    has_secret_shape,
    is_env_reference,
    redact_secrets,
    report_text,
    safe_url,
    url_ambiguities,
    url_credentials,
)

_TOKEN = "ghp_" + "0123456789abcdefghij0123456789abcdef"
# A long random user name, as a token used as the user name of a URL is.
_HEX_USER = "3f2a9c1be47d8a05f6e2b9c4d1a7e3f0b8c6d2a1"


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
        ("https://${USER:-}:${PW:-}@h.example/x", "literal", False),
        ("https://${env:USER}@h.example/x", "literal", False),
        ("https://user:${PW:-hunter2}@h.example/x", "literal", True),
        ("https://user:secret@h.example/x", "secret", True),
        ("https://x-access-token:${GITHUB_TOKEN}@github.com/org/repo.git", "secret", False),
        ("https://x-access-token:${GITHUB_TOKEN:-}@github.com/org/repo.git", "secret", False),
        ("https://x-access-token:${GITHUB_TOKEN:-hunter2}@github.com/org/repo.git", "secret", True),
        ("https://deploy@h.example/x", "secret", False),
        ("https://AdminUser1@h.example/x", "secret", False),
        (f"https://{_TOKEN}@h.example/x", "secret", True),
        (f"https://{_HEX_USER}@h.example/x", "secret", True),
        # The raw text holds a backslash in the userinfo, which WHATWG clients read as '/'.
        (f"https://{_HEX_USER}\\@h.example/x", "secret", True),
        ("https://169.254.169.254\\@h.example/x", "secret", False),
        ("https://deploy:31337\\@h.example/x", "secret", True),
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
        ("api_key=${API_KEY:-}", ()),
        ("api_key=${API_KEY:-literal}", ("api_key",)),
        ("api_key=", ()),
        ("page=2", ()),
        # 'key' and 'sig' name a credential in a URL query; a setting is not a credential.
        ("key=primary&sig=v2", ("key", "sig")),
        ("token=true", ()),
        ("password_policy=strict", ()),
    ],
)
def test_url_credentials_reads_credential_names_and_secret_shaped_values(query: str, keys: tuple[str, ...]) -> None:
    for rule in ("any", "literal", "secret"):
        assert url_credentials(f"https://h.example/x?{query}", userinfo_rule=rule).query_keys == keys


@pytest.mark.parametrize(
    ("url", "query_keys", "fragment_keys"),
    [
        ("https://h.example/#/cb?access_token=abc", (), ("access_token",)),
        ("https://h.example/x?page=2#access_token=abc&token_type=bearer", (), ("access_token",)),
        ("https://h.example/x?api_key=literal#section-2", ("api_key",), ()),
        ("https://h.example/x#L10-L20", (), ()),
    ],
)
def test_url_credentials_reads_the_query_only_up_to_the_fragment(
    url: str, query_keys: tuple[str, ...], fragment_keys: tuple[str, ...]
) -> None:
    """Regression: the query was read from the first '?' even inside the fragment ('#/cb?access_token=...')."""
    credentials = url_credentials(url, userinfo_rule="literal")
    assert (credentials.query_keys, credentials.fragment_keys) == (query_keys, fragment_keys)
    assert bool(credentials) is bool(query_keys or fragment_keys)


@pytest.mark.parametrize("query", [_TOKEN, f"{_TOKEN}=1", f"page=2&{_TOKEN}"])
def test_url_credentials_flags_a_query_key_shaped_like_a_secret(query: str) -> None:
    """Regression: only query values were checked, so a bare '?ghp_...' component carried no credential."""
    for rule in ("any", "literal", "secret"):
        assert url_credentials(f"https://h.example/sse?{query}", userinfo_rule=rule).query_keys == (_TOKEN,)


@pytest.mark.parametrize(
    ("value", "reference"),
    [
        ("$TOKEN", True),
        ("${TOKEN}", True),
        ("${TOKEN:-}", True),
        ("${env:TOKEN}", True),
        ("${TOKEN:-literal}", False),
        ("Bearer ${TOKEN}", False),
        ("literal", False),
    ],
)
def test_env_reference_forms(value: str, reference: bool) -> None:
    """A reference with no default (or an empty one) carries no value; a default ships with the plugin."""
    assert is_env_reference(value) is reference


@pytest.mark.parametrize(
    ("url", "ambiguous"),
    [
        ("https://h.example/a b", False),
        ("https://h.example/x?q=a b", False),
        ("https://h.example .com/x", True),
        ("https:// h.example/x", True),
        ("https://h.example/a\u00a0b", True),
        ("https://h.example/a\tb", True),
        ("mailto:a b@h.example", True),
    ],
)
def test_a_plain_space_after_the_host_is_not_ambiguous(url: str, ambiguous: bool) -> None:
    """Clients percent-encode a plain space in the path; whitespace before it, or any other kind, still counts."""
    assert bool(url_ambiguities(url)) is ambiguous


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
        (f"https://hooks.example.com/notify/{_TOKEN}/x", "https://hooks.example.com/notify/ghp_<redacted>/x"),
        (f"https://h.example:bad/{_TOKEN}", "https://h.example:bad/ghp_<redacted>"),
    ],
)
def test_safe_url_shows_where_a_client_connects_without_credentials(url: str, shown: str) -> None:
    assert safe_url(url) == shown


_BACKSLASH_USERINFO = "3f2a9c1be47d8a05f6e2b9c4d1a7e3f0b8c6d2a1"


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        (f"https://{_BACKSLASH_USERINFO}\\@api.example.com/mcp", "https://api.example.com/mcp"),
        ("https://deploy:31337\\@evil.example/mcp", "https://evil.example/mcp"),
        (f"https://{_BACKSLASH_USERINFO}\\@cdn.example.com/server.mcpb", "https://cdn.example.com/server.mcpb"),
        ("https://user:pass@host\\@evil.example/x", "https://evil.example/x"),
    ],
)
def test_safe_url_never_shows_userinfo_that_a_backslash_turns_into_the_host(url: str, shown: str) -> None:
    """Regression: WHATWG reads '\\' as '/', so the userinfo before '\\@' was shown as the host."""
    assert safe_url(url) == shown


def test_safe_url_is_bounded() -> None:
    assert len(safe_url("https://h.example/" + "a" * 10_000)) <= MAX_REPORT_CHARS


def test_report_text_redacts_every_known_secret_shape() -> None:
    text = report_text(
        f"gh auth login --with-token {_TOKEN}; curl https://user:pw@h.example/x -H 'X: xoxb-1234567890abc'"
    )

    assert _TOKEN not in text and "xoxb-1234567890abc" not in text and "user:pw" not in text
    assert text.startswith("gh auth login --with-token ghp_<redacted>;")


_KEY_BODY = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW"
# Assembled at run time so secret scanners don't read the fixtures below as real key blocks.
_PRIVATE_KEY = "PRIVATE " + "KEY"


@pytest.mark.parametrize("separator", ["\\n", "\n"])
def test_report_text_redacts_a_whole_private_key(separator: str) -> None:
    """Regression: only the BEGIN line was redacted, so the key body and its END line were shown."""
    key = (
        f"-----BEGIN OPENSSH {_PRIVATE_KEY}-----{separator}{_KEY_BODY}{separator}-----END OPENSSH {_PRIVATE_KEY}-----"
    )

    text = report_text(f"printf '%s' '{key}' > ~/.ssh/id")

    assert text == "printf '%s' 'private-key-<redacted>' > ~/.ssh/id"


def test_report_text_withholds_everything_after_a_private_key_it_cannot_read_as_a_block() -> None:
    # 'XPRIVATE KEY' has the secret shape of a private-key header but is not a PEM label redaction reads.
    text = report_text(f"echo '-----BEGIN X{_PRIVATE_KEY}----- {_KEY_BODY} -----END X{_PRIVATE_KEY}-----' > k")

    assert text == "echo 'private-key-<redacted>"


def test_report_text_redacts_a_whole_jwt() -> None:
    """Regression: the JWT's header and payload were redacted, but its signature was shown."""
    signature = "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    jwt = f"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.{signature}"

    text = report_text(f"notify --session {jwt} --quiet")

    assert signature not in text
    assert text == "notify --session jwt-<redacted> --quiet"


def test_report_text_never_shows_a_token_cut_at_the_redaction_window() -> None:
    """Regression: a token cut by the redaction window was too short to match its pattern, and the redacted
    JWT before it shrank the text enough to bring the cut token into view."""
    jwt = f"eyJ{'A' * 150}.eyJ{'A' * 150}.{'B' * 20}"
    head = f"sh -c 'notify {jwt}' "
    cut_inside_token = 23  # 'ghp_' and 19 of the token's 36 characters fall inside the window
    pad = "x" * (2 * MAX_REPORT_CHARS - len(head) - len(" --token ") - cut_inside_token)
    command = f"{head}{pad} --token {_TOKEN}"
    assert command[: 2 * MAX_REPORT_CHARS].endswith(_TOKEN[:cut_inside_token])

    text = report_text(command)

    assert _TOKEN[4:8] not in text
    assert text.startswith("sh -c 'notify jwt-<redacted>' xxx") and text.endswith("...<truncated>")


def test_report_text_marks_a_text_cut_by_the_redaction_window_as_truncated() -> None:
    text = report_text("a " * MAX_REPORT_CHARS + "tail")

    assert len(text) <= MAX_REPORT_CHARS and text.endswith("...<truncated>")


@pytest.mark.parametrize(
    "value",
    [
        "task-550e8400e29b41d4a716446655440000",
        "disk-0123456789abcdef0123",
        "risk-ABCDEFGHIJKLMNOPQRSTUV",
        "x_hf_" + "a" * 34,
        "pnpm_" + "A" * 36,
        "npm_config_cache",
    ],
)
def test_ids_that_contain_a_short_token_prefix_have_no_secret_shape(value: str) -> None:
    """Regression: 'sk-' had no left boundary, so 'task-<hex>' and 'disk-<hex>' ids read as API keys."""
    assert not has_secret_shape(value)
    assert url_credentials(f"https://h.example/mcp?session={value}", userinfo_rule="literal").query_keys == ()


@pytest.mark.parametrize(
    "token",
    [
        "sk-" + "abcdefghijklmnop1234",
        "xoxb-" + "1234567890-abcdefghij",
        "xoxe-1-" + "My0xLTEtMTIzNDU2Nzg5MC0xMjM0",
        "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "npm_" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        "AKIA" + "ABCDEFGHIJ012345",
        "ASIA" + "ABCDEFGHIJ012345",
    ],
)
def test_every_canonical_token_shape_is_a_secret_and_is_redacted(token: str) -> None:
    assert has_secret_shape(token) and has_secret_shape(f"key={token}")
    assert token not in report_text(f"run --flag {token} --next")


@pytest.mark.parametrize("prefix", ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"])
def test_every_github_token_prefix_has_a_secret_shape(prefix: str) -> None:
    token = prefix + "0123456789abcdefghij0123456789abcdef"
    assert has_secret_shape(token)
    assert url_credentials(f"https://h.example/mcp?q={token}", userinfo_rule="literal").query_keys == ("q",)


def test_fine_grained_github_token_has_a_secret_shape() -> None:
    assert has_secret_shape("github_pat_" + "11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz")
    assert not has_secret_shape("github_pat_short")


# --------------------------------------------------------------------------- #
# redact_secrets on whole plugin lines                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("${MCP_URL:-https://u:pw@host/mcp?token=x}", "${MCP_URL:-https://host/mcp}"),
        ("https://h/${A:-x y}?token=t z", "https://h/${A:-x y} z"),
        ("a://x${B:-b://y?token=t}", "a://x${B:-b://y}"),
        ("https://h/?u=https://x:pw@z/w?v=1 next", "https://h/ next"),
        ("see https://h/a?q=1#frag and git+ssh://h/r.git?ref=v1", "see https://h/a#frag and git+ssh://h/r.git"),
        ("--api-key=pw1 TOKEN=pw2 --auth $Y", "--api-key=<redacted> TOKEN=<redacted> --auth $Y"),
    ],
)
def test_redact_secrets_drops_userinfo_and_queries_of_urls_anywhere_in_a_line(text: str, shown: str) -> None:
    assert redact_secrets(text) == shown


# (prefix, run, repeats, suffix): the text, and what redact_secrets shows of it. Each text took 10 s to minutes to
# redact, because a pattern was tried at every word, URL, slash, or credential word of the run and read to its end.
_LONG_RUNS = [
    (("https://user:pw@h/", "-ab", 33_000, ""), ("https://h/", "-ab", 33_000, "")),
    (("", "-eyJ", 16_384, " --token hunter2"), ("", "-eyJ", 16_384, " --token <redacted>")),
    (("", "a.b", 33_000, " https://u:pw@h/"), ("", "a.b", 33_000, " https://h/")),
    (("", "a://", 25_000, "?token=hunter2"), ("", "a://", 25_000, "")),
    (("", "x://y", 20_000, "?sig=hunter2"), ("", "x://y", 20_000, "")),
    (("", "a://${x}", 12_000, "?token=hunter2"), ("", "a://${x}", 12_000, "")),
    (("", "/", 100_000, "u:pw@h"), ("", "/", 100_000, "h")),
    (("a:", "/", 100_000, "u:pw@h?token=hunter2"), ("a:", "/", 100_000, "h")),
    (("", "--auth", 16_000, "=hunter2"), ("", "--auth", 16_000, "=<redacted>")),
    (("", "TOKEN", 20_000, "=hunter2"), ("", "TOKEN", 20_000, "=<redacted>")),
]


def test_redact_secrets_takes_linear_time_on_long_runs() -> None:
    """Regression (CWE-1333): SecurityValidator redacts a SkillSpector snippet or a PII line whole, so one crafted
    100 KB line stalled Tier 1 for minutes. Runs in a subprocess so that a slow redaction fails the timeout instead
    of hanging the suite; the limit is generous because the redactions take well under a second."""
    code = textwrap.dedent(
        """
        import json, sys
        from skillevaluator.validators.url_policy import redact_secrets

        def build(prefix, run, repeats, suffix):
            return prefix + run * repeats + suffix

        for text, expected in json.load(sys.stdin):
            shown = redact_secrets(build(*text))
            print("ok" if shown == build(*expected) else json.dumps([text[1], shown[:60], shown[-60:]]))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], input=json.dumps(_LONG_RUNS), capture_output=True, text=True, timeout=60
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["ok"] * len(_LONG_RUNS), completed.stdout


# --------------------------------------------------------------------------- #
# One credential rule for MCP server URLs and HTTP hook URLs                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("url", "flagged"),
    [
        ("https://api.example.com/x?key=rawFAKEvalue0123", True),
        ("https://api.example.com/x?sig=v2", True),
        ("https://api.example.com/x?key=${API_KEY}", False),
        ("https://api.example.com/x?token=true", False),
        ("https://api.example.com/x?password_policy=strict", False),
        ("https://api.example.com/x?q=sk-" + "a1B2" * 5, True),
        (f"https://api.example.com/x?{_TOKEN}", True),
        ("https://api.example.com/x#/cb?access_token=abc123", True),
        ("https://api.example.com/x?page=2", False),
        ("https://admin:hunter2@api.example.com/x", True),
        ("https://user:${PW}@api.example.com/x", True),
        ("https://admin:hunter2\\@api.example.com/x", True),
        (f"https://{_HEX_USER}\\@api.example.com/x", True),
        ("https://169.254.169.254\\@api.example.com/x", False),
    ],
)
def test_mcp_and_hook_urls_get_the_same_credential_verdict(tmp_path: Path, url: str, flagged: bool) -> None:
    """The same URL is an inline credential for an MCP server and an HTTP hook, or for neither.

    The one documented difference is user information made only of references
    (``https://${USER}:${PW}@host``): an HTTP hook's client sends any userinfo
    as Basic auth with every request, so it counts there.
    """
    root = tmp_path / "demo"
    files = {
        ".claude-plugin/plugin.json": {"name": "demo", "version": "1.0.0", "description": "URL credential probe"},
        ".mcp.json": {"mcpServers": {"remote": {"type": "http", "url": url}}},
        "hooks/hooks.json": {"hooks": {"PostToolUse": [{"matcher": "Write", "hooks": [{"type": "http", "url": url}]}]}},
    }
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(content), encoding="utf-8")

    checks = {finding.check_name for finding in PluginSchemaValidator().validate(root).findings}

    assert ("mcp_url_inline_secret" in checks, "plugin_hook_inline_secret" in checks) == (flagged, flagged)
