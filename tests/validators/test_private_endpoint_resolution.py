# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint checks use fixtures and injected transports exclusively."""

from __future__ import annotations

import importlib
import importlib.util
import socket
import ssl
from unittest.mock import Mock

import pytest

MODULE = "skillevaluator.validators.endpoint_resolution"


def test_private_endpoint_api_exists():
    assert importlib.util.find_spec(MODULE) is not None


@pytest.fixture
def er():
    return importlib.import_module(MODULE)


def target(er, url="https://public.example/path?token=secret#frag", **kwargs):
    return er.EndpointTarget(url, "mcp", "server", "manifest.json", **kwargs)


def test_all_dns_runs_before_one_credential_free_head_per_public_target(er):
    calls = []

    def resolve(host, port, timeout):
        calls.append(("dns", host))
        return ["8.8.8.8", "1.1.1.1"]

    def head(scheme, host, port, address, path, timeout):
        calls.append(("head", host, address, path))
        return er.HeadResult(200, None)

    summary, findings = er.EndpointChecker(resolver=resolve, head=head).check(
        [
            target(er, "https://name:password@one.example/path?secret=value#x"),
            target(er, "https://two.example/other"),
        ]
    )
    assert calls == [
        ("dns", "one.example"),
        ("dns", "two.example"),
        ("head", "one.example", "8.8.8.8", "/path"),
        ("head", "two.example", "8.8.8.8", "/other"),
    ]
    assert not summary.get("incomplete")
    assert "password" not in str(summary)
    assert not findings


@pytest.mark.parametrize(
    "answers",
    [["8.8.8.8", "169.254.169.254"], ["8.8.8.8", "10.0.0.1"], ["8.8.8.8", "not-an-address"], [], ["8.8.8.8"] * 65],
)
def test_unsafe_empty_invalid_or_overcap_dns_is_never_contacted(er, answers):
    head = Mock(side_effect=AssertionError("must not contact"))
    summary, _ = er.EndpointChecker(resolver=lambda *_: answers, head=head).check([target(er)])
    assert not head.called
    assert summary["endpoints"][0]["status"] != "public"
    if not answers or "not-an-address" in answers or len(answers) > er.MAX_ADDRESSES:
        assert summary["incomplete"]


@pytest.mark.parametrize("url", ["https://169.254.169.254/", "https://metadata.google.internal/", "http://127.0.0.1/"])
def test_static_nonpublic_is_never_resolved_or_contacted_even_allowlisted(er, url):
    transport = Mock(side_effect=AssertionError("no network"))
    summary, _ = er.EndpointChecker(resolver=transport, head=transport).check(
        [target(er, url, allowed_hosts=("169.254.169.254", "metadata.google.internal", "127.0.0.1"))]
    )
    assert not transport.called
    assert summary["endpoints"][0]["classification"] in {"metadata", "private"}


def test_every_dns_answer_uses_allowlist_and_metadata_never_does(er):
    head = Mock(side_effect=AssertionError("no private network request"))
    summary, findings = er.EndpointChecker(resolver=lambda *_: ["10.0.0.1", "169.254.169.254"], head=head).check(
        [target(er, allowed_hosts=("public.example", "10.0.0.0/8", "169.254.0.0/16"))]
    )
    assert not head.called
    assert any(f.check_name == "endpoint_resolves_metadata" for f in findings)
    assert summary["endpoints"][0]["classification"] == "metadata"


def test_redirect_is_classified_without_a_second_request(er):
    calls = []

    def head(*args):
        calls.append(args)
        return er.HeadResult(302, "http://169.254.169.254/latest?token=secret")

    summary, findings = er.EndpointChecker(resolver=lambda *_: ["8.8.8.8"], head=head).check([target(er)])
    assert len(calls) == 1
    assert summary["endpoints"][0]["redirect"]["classification"] == "metadata"
    assert {f.check_name for f in findings} >= {"endpoint_redirect_metadata", "endpoint_redirect_insecure_scheme"}
    assert "secret" not in str(summary)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "https://", "https://$HOST/path", "https://public.example:bad/"])
def test_unsupported_or_unusable_targets_are_explicitly_incomplete(er, url):
    transport = Mock(side_effect=AssertionError("no network"))
    summary, findings = er.EndpointChecker(resolver=transport, head=transport).check([target(er, url)])
    assert summary["incomplete"]
    assert not transport.called
    assert any(f.check_name == "endpoint_resolution_incomplete" for f in findings)


def test_one_unresolved_target_makes_an_otherwise_successful_check_incomplete(er):
    def resolve(host, *_):
        if host == "bad.example":
            raise socket.gaierror("secret diagnostics")
        return ["8.8.8.8"]

    summary, findings = er.EndpointChecker(resolver=resolve, head=lambda *_: er.HeadResult(200, None)).check(
        [target(er, "https://good.example/"), target(er, "https://bad.example/")]
    )
    assert summary["incomplete"]
    assert "secret diagnostics" not in str(summary) + str(findings)


def test_dns_and_head_receive_only_remaining_total_budget(er):
    now = [0.0]
    timeouts = []

    def resolve(*args):
        timeouts.append(args[-1])
        now[0] += 2.0
        return ["8.8.8.8"]

    def head(*args):
        timeouts.append(args[-1])
        now[0] += 1.0
        return er.HeadResult(200, None)

    summary, _ = er.EndpointChecker(resolver=resolve, head=head, budget=3, clock=lambda: now[0]).check([target(er)])
    assert timeouts == [3.0, 1.0]
    assert not summary.get("incomplete")


def test_dns_budget_exhaustion_prevents_head_and_is_incomplete(er):
    now = [0.0]

    def resolve(*_):
        now[0] += 5
        return ["8.8.8.8"]

    head = Mock(side_effect=AssertionError("budget gone"))
    summary, _ = er.EndpointChecker(resolver=resolve, head=head, budget=2, clock=lambda: now[0]).check([target(er)])
    assert summary["incomplete"]
    assert not head.called


def test_failed_head_or_redirect_resolution_is_incomplete(er):
    summary, _ = er.EndpointChecker(
        resolver=lambda *_: ["8.8.8.8"], head=lambda *_: er.HeadResult(None, None, "credential=value")
    ).check([target(er)])
    assert summary["incomplete"]
    assert "credential=value" not in str(summary)

    def resolve(host, *_):
        return [] if host == "missing.example" else ["8.8.8.8"]

    summary, _ = er.EndpointChecker(
        resolver=resolve, head=lambda *_: er.HeadResult(302, "https://missing.example/")
    ).check([target(er)])
    assert summary["incomplete"]


def test_endpoint_cap_preserves_attribution_for_unchecked_targets(er, monkeypatch):
    monkeypatch.setattr(er, "MAX_ENDPOINTS", 1)
    summary, _ = er.EndpointChecker(resolver=lambda *_: ["8.8.8.8"], head=lambda *_: er.HeadResult(200, None)).check(
        [target(er), target(er, "https://second.example/")]
    )
    assert summary["incomplete"]
    assert summary["endpoints"][1]["name"] == "server"


def test_duplicate_urls_keep_distinct_component_attribution(er):
    summary, _ = er.EndpointChecker(resolver=lambda *_: ["8.8.8.8"], head=lambda *_: er.HeadResult(200, None)).check(
        [target(er), er.EndpointTarget("https://public.example/path?token=secret#frag", "mcp", "other", "other.json")]
    )
    assert {row["name"] for row in summary["endpoints"]} == {"server", "other"}


def test_raw_head_pins_address_checks_tls_and_ignores_ambient_auth(er, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://user:password@proxy.invalid")
    monkeypatch.setenv("NETRC", "/fake/netrc")
    sock = Mock()
    sock.fileno.return_value = 7
    sock.family = socket.AF_INET
    connection = Mock()
    connection.getresponse.return_value.status = 200
    connection.getresponse.return_value.getheader.return_value = None
    context = Mock()
    context.wrap_socket.return_value = sock
    monkeypatch.setattr(er.socket, "create_connection", Mock(return_value=sock))
    monkeypatch.setattr(er.ssl, "create_default_context", Mock(return_value=context))
    monkeypatch.setattr(er.http.client, "HTTPSConnection", Mock(return_value=connection))
    result = er._head("https", "public.example", 443, "8.8.8.8", "/path", 1.0)
    assert result.status == 200
    assert er.socket.create_connection.call_args.args[0] == ("8.8.8.8", 443)
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    assert context.wrap_socket.call_args.kwargs["server_hostname"] == "public.example"
    args, kwargs = connection.request.call_args
    assert args == ("HEAD", "/path")
    assert not any("auth" in key.lower() for key in kwargs["headers"])


def test_nonstring_dns_answers_are_not_public_evidence(er):
    head = Mock(side_effect=AssertionError("invalid evidence"))
    summary, _ = er.EndpointChecker(resolver=lambda *_: [134744072], head=head).check([target(er)])
    assert summary["incomplete"]
    assert not head.called


def test_invalid_location_header_type_is_incomplete(er):
    summary, _ = er.EndpointChecker(resolver=lambda *_: ["8.8.8.8"], head=lambda *_: er.HeadResult(200, 0)).check(
        [target(er)]
    )
    assert summary["incomplete"]


def test_redirect_dns_must_also_finish_within_the_total_budget(er):
    now = [0.0]

    def resolve(host, *_):
        if host == "redirect.example":
            now[0] = 4.0
        return ["8.8.8.8"]

    summary, _ = er.EndpointChecker(
        resolver=resolve,
        head=lambda *_: er.HeadResult(302, "https://redirect.example/"),
        budget=3,
        clock=lambda: now[0],
    ).check([target(er)])
    assert summary["incomplete"]
