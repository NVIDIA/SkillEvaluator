# SPDX-License-Identifier: Apache-2.0
"""Operator plugin policy is bounded, explicit and bound to report provenance."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from skillevaluator.validators import policy as policy_module
from skillevaluator.validators.policy import ValidationPolicy, load_policy_file


def test_default_policy_digest_and_summary_stay_compatible():
    policy = ValidationPolicy()
    payload = {
        "profile": "default",
        "audience": "all",
        "author_email_regex": None,
        "severity_overrides": {},
        "metadata_id_requirement": "ignore",
    }
    expected = (
        "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    assert policy.digest == expected
    assert set(policy.to_dict()) == {
        "profile",
        "audience",
        "digest",
        "author_email_regex",
        "severity_overrides",
        "metadata_id_requirement",
        "source",
    }


def test_allowlists_are_typed_redacted_and_order_independent():
    first = ValidationPolicy(
        mcp_allowed_private_hosts=("127.0.0.1", "*.corp.example"),
        hook_allowed_urls=("https://hooks.example.test/events",),
    )
    second = ValidationPolicy(
        mcp_allowed_private_hosts=("*.corp.example", "127.0.0.1"),
        hook_allowed_urls=("https://hooks.example.test/events",),
    )
    assert first.digest == second.digest != ValidationPolicy().digest
    summary = first.to_dict()
    assert summary["mcp_allowed_private_host_count"] == 2
    assert summary["hook_allowed_url_count"] == 1
    assert "corp.example" not in json.dumps(summary) and "hooks.example.test" not in json.dumps(summary)
    assert first.resolve_endpoints is False


def test_existing_host_patterns_and_ip_notation_remain_supported():
    entries = ("localhost", "*.corp.example", "10.0.0.0/8", "::1", "[::1]")
    assert ValidationPolicy(mcp_allowed_private_hosts=entries).mcp_allowed_private_hosts == entries


def test_allowlist_entries_are_stripped_and_deduplicated():
    policy = ValidationPolicy(
        mcp_allowed_private_hosts=(" localhost ", "localhost", "*.corp.example "),
        hook_allowed_urls=(" https://hooks.example.test/ ", "https://hooks.example.test/"),
    )
    assert policy.mcp_allowed_private_hosts == ("localhost", "*.corp.example")
    assert policy.hook_allowed_urls == ("https://hooks.example.test/",)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("localhost", "must be a list of at most 64 strings"),
        ({"host": "localhost"}, "must be a list of at most 64 strings"),
        ([True], "nonempty strings of at most 255 characters"),
        ([1], "nonempty strings of at most 255 characters"),
        ([""], "nonempty strings of at most 255 characters"),
        (["x" * 256], "nonempty strings of at most 255 characters"),
        (["a"] * 65, "must be a list of at most 64 strings"),
        (["local\nlocalhost"], "control or invisible"),
        (["local\u200bhost"], "control or invisible"),
        (["localhost\u0000"], "control or invisible"),
        (["169.254.169.254"], "cloud metadata"),
        (["metadata.google.internal"], "cloud metadata"),
        (["https://localhost"], "invalid URL prefix"),
        (["localhost/path"], "invalid IP address or network"),
        (["localhost:9000"], "invalid IP address or network"),
        (["*"], "invalid host wildcard"),
        (["\uff0a.corp.example"], "invalid host wildcard"),
        (["user@localhost"], "invalid host pattern"),
        (["a?b"], "invalid host pattern"),
        (["a#b"], "invalid host pattern"),
        (["a\\b"], "invalid host pattern"),
        (["%2A"], "invalid host pattern"),
        (["a%20b"], "invalid host pattern"),
        (["example.com%3a8080"], "invalid host pattern"),
        (["user%40localhost"], "invalid host pattern"),
        (["0.0.0.0%2F0"], "invalid host pattern"),
        (["user\uff20localhost"], "invalid host pattern"),
        (["a\uff1fb"], "invalid host pattern"),
        (["a\uff03b"], "invalid host pattern"),
        (["a\uff3cb"], "invalid host pattern"),
        (["\uff05\uff12\uff41"], "invalid host pattern"),
        (["example.com\uff1a8080"], "invalid IP address or network"),
        (["corp\uff0fpath"], "invalid IP address or network"),
        (["\uff11\uff10.\uff10.\uff10.\uff10\uff0f\uff19\uff19"], "invalid IP address or network"),
        ([".corp.example"], r"starts with '\.'.*\*\.suffix"),
        (["\uff0ecorp.example"], r"starts with '\.'"),
        (["example..com"], "empty label"),
        (["-"], "invalid host label"),
        (["*..corp.example"], "empty label"),
    ],
)
def test_invalid_mcp_host_entries_fail_closed(value, match):
    with pytest.raises(ValueError, match=match):
        ValidationPolicy(mcp_allowed_private_hosts=value)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (["https://user:secret@hooks.example.test"], "invalid URL prefix"),
        (["https://hooks.example.test/path?token=secret"], "invalid URL prefix"),
        (["https://hooks.example.test/#fragment"], "invalid URL prefix"),
        # Only the URL-ambiguity check rejects a percent-encoded host.
        (["https://hooks%2Eexample.test/"], "invalid URL prefix"),
        (["https://169.254.169.254/latest"], "cloud metadata"),
        (["ftp://hooks.example.test"], "HTTP or HTTPS"),
        (["x" * 2049], "nonempty strings of at most 2048 characters"),
        (["https://hooks.example.test/\u202e"], "control or invisible"),
        (["http://*/"], "invalid host wildcard"),
        (["http://*.*/"], "invalid host wildcard"),
        (["https://*.hooks.example.test/events"], "URL prefix host must be exact"),
        (["https://\uff0a.hooks.example.test/events"], "invalid host wildcard"),
        (["https://.hooks.example.test/"], r"starts with '\.'"),
        (["https://hooks..example.test/"], "empty label"),
    ],
)
def test_invalid_hook_url_prefixes_fail_closed(value, match):
    with pytest.raises(ValueError, match=match):
        ValidationPolicy(hook_allowed_urls=value)


def test_hook_allowlist_accepts_exact_url_prefixes_and_wildcard_host_patterns():
    entries = ("https://hooks.example.test/events", "*.hooks.example.test")
    assert ValidationPolicy(hook_allowed_urls=entries).hook_allowed_urls == entries


@pytest.mark.parametrize("value", ["true", 1, [], {}])
def test_endpoint_opt_in_requires_boolean(value):
    with pytest.raises(ValueError, match=r"endpoints\.resolve"):
        ValidationPolicy(resolve_endpoints=value)


def test_custom_overlay_preserves_omitted_keys_and_allows_explicit_clearing(tmp_path, monkeypatch):
    base = ValidationPolicy(
        mcp_allowed_private_hosts=("localhost",),
        hook_allowed_urls=("https://hooks.example.test/",),
        resolve_endpoints=True,
    )
    monkeypatch.setattr(policy_module, "load_profile", lambda _name: base)
    path = tmp_path / "policy.yml"
    path.write_text("mcp: {}\nhooks: {}\nendpoints: {}\n")
    preserved = load_policy_file(path)
    assert preserved.mcp_allowed_private_hosts == base.mcp_allowed_private_hosts
    assert preserved.hook_allowed_urls == base.hook_allowed_urls and preserved.resolve_endpoints is True
    path.write_text("mcp:\n  allowed_private_hosts: []\nhooks:\n  allowed_urls: null\nendpoints:\n  resolve: false\n")
    cleared = load_policy_file(path)
    assert cleared.mcp_allowed_private_hosts == cleared.hook_allowed_urls == ()
    assert cleared.resolve_endpoints is False


def test_null_block_keeps_the_base_value_and_null_resolve_clears_it(tmp_path, monkeypatch):
    base = ValidationPolicy(mcp_allowed_private_hosts=("localhost",), resolve_endpoints=True)
    monkeypatch.setattr(policy_module, "load_profile", lambda _name: base)
    path = tmp_path / "policy.yml"
    path.write_text("mcp: null\nendpoints:\n  resolve: null\n", encoding="utf-8")
    loaded = load_policy_file(path)
    assert loaded.mcp_allowed_private_hosts == ("localhost",)
    assert loaded.resolve_endpoints is False


@pytest.mark.parametrize(
    "document", ["mcp: []", "hooks: 3", "endpoints:\n  unexpected: true", "mcp:\n  allow_all: true"]
)
def test_plugin_policy_blocks_reject_unknown_shapes(tmp_path, document):
    path = tmp_path / "policy.yml"
    path.write_text(document)
    with pytest.raises(ValueError):
        load_policy_file(path)


def test_hook_endpoint_grants_do_not_apply_to_mcp_targets():
    from skillevaluator.plugin_dependency_audit import collect_endpoint_targets

    mcp = SimpleNamespace(name="server", file=".mcp.json", config={"url": "https://192.168.1.10/mcp"})
    hook = SimpleNamespace(id="hook", file="hooks.json", url="https://192.168.1.10/events")
    targets = collect_endpoint_targets(
        [mcp],
        [hook],
        allowed_hosts=("corp.example",),
        hook_allowed_urls=("https://192.168.1.10/events",),
    )
    assert targets[0].allowed_hosts == ("corp.example",)
    assert targets[1].allowed_hosts == ("corp.example", "192.168.1.10")
