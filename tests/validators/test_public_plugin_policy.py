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


@pytest.mark.parametrize(
    "value",
    [
        "localhost",
        {"host": "localhost"},
        [True],
        [1],
        [""],
        ["x" * 256],
        ["a"] * 65,
        ["local\nlocalhost"],
        ["169.254.169.254"],
        ["metadata.google.internal"],
        ["https://localhost"],
        ["localhost/path"],
        ["localhost:9000"],
        ["*"],
    ],
)
def test_invalid_mcp_host_entries_fail_closed(value):
    with pytest.raises(ValueError):
        ValidationPolicy(mcp_allowed_private_hosts=value)


@pytest.mark.parametrize(
    "value",
    [
        ["https://user:secret@hooks.example.test"],
        ["https://hooks.example.test/path?token=secret"],
        ["https://hooks.example.test/#fragment"],
        ["https://169.254.169.254/latest"],
        ["ftp://hooks.example.test"],
        ["x" * 2049],
        ["https://hooks.example.test/\u202e"],
    ],
)
def test_invalid_hook_url_prefixes_fail_closed(value):
    with pytest.raises(ValueError):
        ValidationPolicy(hook_allowed_urls=value)


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


@pytest.mark.parametrize(
    "document", ["mcp: []", "hooks: 3", "endpoints:\n  unexpected: true", "mcp:\n  allow_all: true"]
)
def test_plugin_policy_blocks_reject_unknown_shapes(tmp_path, document):
    path = tmp_path / "policy.yml"
    path.write_text(document)
    with pytest.raises(ValueError):
        load_policy_file(path)


def test_policy_reader_is_bounded_and_keeps_legacy_last_key_wins(tmp_path):
    path = tmp_path / "policy.yml"
    path.write_text("profile: before\nprofile: after\n")
    assert load_policy_file(path).profile == "after"
    path.write_text("#" + "x" * (1 << 20))
    with pytest.raises(ValueError, match="Invalid policy YAML"):
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


@pytest.mark.parametrize(
    "document,reason",
    [
        ("#" + "x" * (1 << 20), "the file exceeds the 1 MiB read bound"),
        ("profile: default\nseverity_overrides: [\n", "found '<stream end>' at line 3, column 1"),
        ("a: " + "[" * 500 + "]" * 500, "nesting depth exceeds"),
    ],
    ids=["size-bound", "parse-location", "complexity-limit"],
)
def test_policy_yaml_errors_keep_their_reason(tmp_path, document, reason):
    path = tmp_path / "policy.yml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_policy_file(path)
    message = str(exc_info.value)
    assert message.startswith(f"Invalid policy YAML in {path}: ")
    assert reason in message
