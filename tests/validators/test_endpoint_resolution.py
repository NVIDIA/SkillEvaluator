# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in DNS and redirect checks for MCP and HTTP hook endpoints (no real network)."""

from __future__ import annotations

import json
import socket
import ssl
from pathlib import Path

import pytest

from skillevaluator.models.result import Severity
from skillevaluator.validators import endpoint_resolution as er
from skillevaluator.validators.endpoint_resolution import EndpointChecker, EndpointTarget, HeadResult
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.policy import ValidationPolicy, load_policy_file


class _FakeNetwork:
    def __init__(self, dns: dict[str, list[str] | Exception], heads: dict[str, HeadResult] | None = None) -> None:
        self.dns = dns
        self.heads = heads or {}
        self.resolved: list[str] = []
        self.requests: list[tuple[str, str, int, str, str]] = []

    def resolve(self, host: str, _port: int, timeout: float) -> list[str]:
        assert timeout <= er.DNS_TIMEOUT_SECONDS
        self.resolved.append(host)
        answer = self.dns[host]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def head(self, scheme: str, host: str, port: int, address: str, path: str, timeout: float) -> HeadResult:
        assert timeout <= er.HEAD_TIMEOUT_SECONDS
        self.requests.append((scheme, host, port, address, path))
        return self.heads.get(host, HeadResult(200, None))


def _target(url: str, *, kind: str = "mcp", allowed: tuple[str, ...] = ()) -> EndpointTarget:
    return EndpointTarget(url=url, kind=kind, name="srv", file_path="plugin.json", allowed_hosts=allowed)


def _check(network: _FakeNetwork, *targets: EndpointTarget):
    return EndpointChecker(resolver=network.resolve, head=network.head).check(targets)


def test_public_name_resolving_to_private_address_is_flagged_and_not_contacted() -> None:
    network = _FakeNetwork({"mcp.example.com": ["10.0.0.5"]})
    summary, findings = _check(network, _target("https://mcp.example.com/mcp"))
    [finding] = findings
    assert finding.check_name == "endpoint_resolves_private"
    assert finding.severity == Severity.MEDIUM
    assert finding.metadata == {"mcp_server": "srv"}
    assert network.requests == []
    [row] = summary["endpoints"]
    assert row["status"] == "private"
    assert row["addresses"] == ["10.0.0.5"]
    assert "not contacted" in row["head"]["skipped"]


def test_name_resolving_to_metadata_is_high_and_never_allowlisted() -> None:
    network = _FakeNetwork({"meta.example.com": ["169.254.169.254"]})
    _summary, findings = _check(network, _target("https://meta.example.com/", allowed=("meta.example.com",)))
    assert [(f.check_name, f.severity) for f in findings] == [("endpoint_resolves_metadata", Severity.HIGH)]


def test_allowlisted_private_resolution_is_quiet() -> None:
    network = _FakeNetwork({"mcp.corp.example": ["10.1.2.3"]})
    _summary, findings = _check(network, _target("https://mcp.corp.example/", allowed=("*.corp.example",)))
    assert findings == []


@pytest.mark.parametrize("url", ["https://${API_HOST}/mcp", "https://$API_HOST/mcp", "https://${API_HOST}:8443/"])
def test_unexpanded_variable_hosts_are_skipped_without_dns(url: str) -> None:
    network = _FakeNetwork({})
    summary, findings = _check(network, _target(url))
    assert findings == []
    assert network.resolved == []
    assert network.requests == []
    [row] = summary["endpoints"]
    assert row["status"] == "skipped"
    assert row["reason"] == "host contains an unexpanded variable"


@pytest.mark.parametrize("answers", [["10.1.2.3", "127.0.0.1"], ["127.0.0.1", "10.1.2.3"]])
def test_every_resolved_address_must_be_allowlisted_regardless_of_answer_order(answers: list[str]) -> None:
    cidr = _target("https://mcp.example.com/", allowed=("10.0.0.0/8",))
    [finding] = _check(_FakeNetwork({"mcp.example.com": answers}), cidr)[1]
    assert finding.check_name == "endpoint_resolves_private"
    assert "127.0.0.1" in finding.message
    both = _target("https://mcp.example.com/", allowed=("10.0.0.0/8", "127.0.0.0/8"))
    assert _check(_FakeNetwork({"mcp.example.com": answers}), both)[1] == []
    redirecting = _FakeNetwork(
        {"mcp.example.com": ["93.184.216.34"], "hop.example.com": answers},
        {"mcp.example.com": HeadResult(302, "https://hop.example.com/x")},
    )
    assert [f.check_name for f in _check(redirecting, cidr)[1]] == ["endpoint_redirect_private"]


def test_public_endpoint_gets_one_credential_free_head_without_query() -> None:
    network = _FakeNetwork({"mcp.example.com": ["93.184.216.34"]})
    summary, findings = _check(network, _target("https://user:pw@mcp.example.com:8443/mcp?token=secret#frag"))
    assert findings == []
    assert network.requests == [("https", "mcp.example.com", 8443, "93.184.216.34", "/mcp")]
    row = summary["endpoints"][0]
    assert row["status"] == "public"
    assert row["head"] == {"status": 200, "location": None}
    assert "secret" not in json.dumps(summary)
    assert "pw" not in row["url"]


@pytest.mark.parametrize(
    ("location", "dns", "check", "severity"),
    [
        ("http://169.254.169.254/latest/meta-data", {}, "endpoint_redirect_metadata", Severity.HIGH),
        (
            "https://internal.example.com/x",
            {"internal.example.com": ["192.168.1.10"]},
            "endpoint_redirect_private",
            Severity.MEDIUM,
        ),
        ("/relative/path", {}, None, None),
    ],
)
def test_redirect_target_is_classified_but_never_followed(
    location: str, dns: dict, check: str | None, severity: Severity | None
) -> None:
    network = _FakeNetwork(
        {"mcp.example.com": ["93.184.216.34"], **dns}, {"mcp.example.com": HeadResult(302, location)}
    )
    summary, findings = _check(network, _target("https://mcp.example.com/mcp"))
    assert len(network.requests) == 1
    checks = [(f.check_name, f.severity) for f in findings if f.check_name != "endpoint_redirect_insecure_scheme"]
    assert checks == ([(check, severity)] if check else [])
    assert summary["endpoints"][0]["redirect"]["url"]


def test_https_to_http_redirect_downgrade_is_flagged() -> None:
    network = _FakeNetwork(
        {"mcp.example.com": ["93.184.216.34"], "cdn.example.com": ["93.184.216.35"]},
        {"mcp.example.com": HeadResult(301, "http://cdn.example.com/mcp")},
    )
    _summary, findings = _check(network, _target("https://mcp.example.com/mcp"))
    assert [f.check_name for f in findings] == ["endpoint_redirect_insecure_scheme"]


def test_dns_failure_is_low_and_head_failure_is_recorded() -> None:
    network = _FakeNetwork({"gone.example.com": socket.gaierror("nxdomain"), "down.example.com": ["93.184.216.34"]})

    def failing_head(*_args: object) -> HeadResult:
        raise OSError("connection refused")

    checker = EndpointChecker(resolver=network.resolve, head=failing_head)
    summary, findings = checker.check(
        [_target("https://gone.example.com/"), _target("https://down.example.com/", kind="hook")]
    )
    assert [(f.check_name, f.severity) for f in findings] == [("endpoint_resolution_failed", Severity.LOW)]
    statuses = {row["url"]: row["status"] for row in summary["endpoints"]}
    assert statuses == {"https://gone.example.com/": "unresolved", "https://down.example.com/": "head_failed"}


def test_static_non_public_hosts_are_never_resolved() -> None:
    network = _FakeNetwork({})
    summary, findings = _check(network, _target("https://127.0.0.1:9/x"), _target("https://localhost/x"))
    assert findings == []
    assert network.resolved == []
    assert {row["status"] for row in summary["endpoints"]} == {"static_non_public"}


def test_time_budget_skips_remaining_endpoints() -> None:
    now = [0.0]
    network = _FakeNetwork({"a.example.com": ["93.184.216.34"], "b.example.com": ["93.184.216.34"]})

    def resolve(host: str, port: int, timeout: float) -> list[str]:
        now[0] += 1.0
        return network.resolve(host, port, timeout)

    def head(*args: object) -> HeadResult:
        now[0] += 20.0
        return network.head(*args)  # type: ignore[arg-type]

    checker = EndpointChecker(resolver=resolve, head=head, budget=10.0, clock=lambda: now[0])
    summary, findings = checker.check([_target("https://a.example.com/"), _target("https://b.example.com/")])
    assert [row["status"] for row in summary["endpoints"]] == ["public", "skipped"]
    assert summary["incomplete"] is True
    assert [f.check_name for f in findings] == ["endpoint_resolution_incomplete"]


def test_a_redirect_target_needs_time_only_when_it_must_be_resolved() -> None:
    now = [0.0]
    network = _FakeNetwork({"mcp.example.com": ["93.184.216.34"], "next.example.com": ["93.184.216.35"]})

    def checker(location: str) -> EndpointChecker:
        def head(*_args: object) -> HeadResult:
            now[0] += 100.0  # the HEAD uses up the whole budget
            return HeadResult(302, location)

        now[0] = 0.0
        return EndpointChecker(resolver=network.resolve, head=head, budget=10.0, clock=lambda: now[0])

    summary, _findings = checker("https://next.example.com/x").check([_target("https://mcp.example.com/mcp")])
    assert summary["endpoints"][0]["redirect"]["classification"] == "skipped"
    assert "next.example.com" not in network.resolved
    assert summary["incomplete"] is True

    summary, findings = checker("http://169.254.169.254/latest").check([_target("https://mcp.example.com/mcp")])
    assert summary["endpoints"][0]["redirect"]["classification"] == "metadata"
    assert "endpoint_redirect_metadata" in {finding.check_name for finding in findings}


def test_classify_host_resolves_only_public_looking_names() -> None:
    lookups: list[tuple[str, int]] = []

    def resolve(host: str, port: int) -> list[str]:
        lookups.append((host, port))
        return ["93.184.216.34", "10.0.0.5"]

    loopback = er.classify_host("127.0.0.1", 443, (), resolve=resolve)
    assert (loopback.classification, loopback.blocked) == ("private", ("private", "loopback", "127.0.0.1"))
    assert er.classify_host("127.0.0.1", 443, ("127.0.0.0/8",), resolve=resolve).blocked is None
    assert lookups == []

    resolved = er.classify_host("mcp.example.com", 8443, (), resolve=resolve)
    assert lookups == [("mcp.example.com", 8443)]
    assert resolved.static is None
    assert resolved.addresses == ("93.184.216.34", "10.0.0.5")
    assert (resolved.classification, resolved.blocked) == ("private", ("private", "private (RFC 1918)", "10.0.0.5"))
    assert er.classify_host("mcp.example.com", 443, ("10.0.0.0/8",), resolve=resolve).blocked is None
    assert er.classify_host("mcp.example.com", 443, ()).classification == "public"  # no resolver: never looked up


def test_dns_resolver_looks_names_up_through_the_module_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int, float]] = []

    def resolve(host: str, port: int, timeout: float) -> list[str]:
        seen.append((host, port, timeout))
        return ["93.184.216.34"]

    monkeypatch.setattr(er, "_resolve", resolve)
    assert er.classify_host("h.example", 443, (), resolve=er.dns_resolver(1.5)).addresses == ("93.184.216.34",)
    assert seen == [("h.example", 443, 1.5)]


# --------------------------------------------------------------------------- #
# Plugin schema integration                                                   #
# --------------------------------------------------------------------------- #
def _plugin(root: Path) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "mcpServers": {"remote": {"url": "https://mcp.example.com/mcp", "type": "http"}},
                "hooks": {
                    "PostToolUse": [
                        {"matcher": "Write", "hooks": [{"type": "http", "url": "https://hooks.example.com/h"}]}
                    ]
                },
            }
        )
    )
    return root


@pytest.fixture
def fake_network(monkeypatch: pytest.MonkeyPatch) -> _FakeNetwork:
    network = _FakeNetwork({"mcp.example.com": ["10.9.8.7"], "hooks.example.com": ["169.254.169.254"]})
    monkeypatch.setattr(er, "_resolve", network.resolve)
    monkeypatch.setattr(er, "_head", network.head)
    return network


def test_default_validation_is_network_free(tmp_path: Path, fake_network: _FakeNetwork) -> None:
    result = PluginSchemaValidator().validate(_plugin(tmp_path))
    assert fake_network.resolved == []
    assert "endpoint_resolution" not in result.metadata["plugin"]


def test_resolve_endpoints_flag_checks_mcp_and_hook_urls(tmp_path: Path, fake_network: _FakeNetwork) -> None:
    result = PluginSchemaValidator(resolve_endpoints=True).validate(_plugin(tmp_path))
    assert sorted(fake_network.resolved) == ["hooks.example.com", "mcp.example.com"]
    by_check = {f.check_name: f for f in result.findings if f.check_name.startswith("endpoint_")}
    assert by_check["endpoint_resolves_private"].category == "MCP_DECLARATION"
    assert by_check["endpoint_resolves_private"].metadata["mcp_server"] == "remote"
    hook = by_check["endpoint_resolves_metadata"]
    assert hook.category == "PLUGIN_SCHEMA"
    assert hook.metadata["plugin_component"] == {"type": "hook", "name": "inline"}
    assert not result.passed
    summary = result.metadata["plugin"]["endpoint_resolution"]
    assert summary["enabled"] is True
    assert {row["kind"] for row in summary["endpoints"]} == {"mcp", "hook"}


def test_policy_key_enables_resolution(tmp_path: Path, fake_network: _FakeNetwork) -> None:
    overlay = tmp_path / "policy.yaml"
    overlay.write_text("endpoints:\n  resolve: true\nmcp:\n  allowed_private_hosts: ['mcp.example.com']\n")
    policy = load_policy_file(overlay)
    assert policy.resolve_endpoints is True
    result = PluginSchemaValidator(policy=policy).validate(_plugin(tmp_path / "plugin"))
    checks = {f.check_name for f in result.findings}
    assert "endpoint_resolves_private" not in checks  # allowlisted by mcp.allowed_private_hosts
    assert "endpoint_resolves_metadata" in checks


def test_resolve_endpoints_policy_default_is_off() -> None:
    assert ValidationPolicy().resolve_endpoints is False


def test_head_never_negotiates_below_tls_1_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The redirect check's HTTPS HEAD refuses TLS 1.0 and 1.1 whatever the local OpenSSL defaults allow."""
    seen: list[ssl.TLSVersion] = []

    class _Stop(Exception):
        pass

    class _RecordingContext(ssl.SSLContext):
        def wrap_socket(self, *_args: object, **_kwargs: object) -> ssl.SSLSocket:
            seen.append(self.minimum_version)
            raise _Stop

    def _permissive_context(*_args: object, **_kwargs: object) -> ssl.SSLContext:
        context = _RecordingContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
        return context

    left, right = socket.socketpair()
    monkeypatch.setattr(er.ssl, "create_default_context", _permissive_context)
    monkeypatch.setattr(er.socket, "create_connection", lambda *_args, **_kwargs: left)
    try:
        with pytest.raises(_Stop):
            er._head("https", "hooks.example.com", 443, "93.184.216.34", "/", 1.0)
    finally:
        right.close()

    assert seen == [ssl.TLSVersion.TLSv1_2]
