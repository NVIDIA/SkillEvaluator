# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint checks (check 9): proof bug L14.

Fixtures follow check-09 p04 (credential query keys), p05 (no scheme), p07b
(a HEAD that never completes), e06 (a backslash URL), and e09 (classifier
probes: a space in the path, the Tencent metadata IP), in the Claude Code and
Codex formats. The network is always a fake: no DNS or HTTP leaves the test.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators import endpoint_resolution as er
from skillevaluator.validators.endpoint_resolution import EndpointChecker, EndpointTarget, HeadResult
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_CODEX_MANIFEST = {
    "name": "demo",
    "version": "1.0.0",
    "description": "Endpoint probe (Codex)",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "Endpoint probe",
        "shortDescription": "Endpoint probe",
        "longDescription": "A plugin used to test MCP endpoint checks.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}


def _plugin(root: Path, servers: dict[str, dict], *, codex: bool = False, hooks: dict | None = None) -> Path:
    manifest = (
        (".codex-plugin/plugin.json", _CODEX_MANIFEST)
        if codex
        else (".claude-plugin/plugin.json", {"name": "demo", "version": "1.0.0", "description": "Endpoint probe"})
    )
    files = [manifest, (".mcp.json", {"mcpServers": servers})]
    if hooks is not None:
        files.append(("hooks/hooks.json", hooks))
    for rel, content in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(content), encoding="utf-8")
    return root


def _by_server(result: ValidationResult) -> dict[str, dict[str, Severity]]:
    found: dict[str, dict[str, Severity]] = {}
    for finding in result.findings:
        server = finding.metadata.get("mcp_server")
        if server:
            found.setdefault(server, {})[finding.check_name] = finding.severity
    return found


def _http(url: str) -> dict:
    return {"type": "http", "url": url}


# --------------------------------------------------------------------------- #
# Credential query keys, a space in the path, a missing scheme, metadata IPs  #
# --------------------------------------------------------------------------- #
_STATIC = {
    "probe-key": _http("https://mcp.example.com/mcp?key=P28-FAKE-KEY-6"),
    "probe-auth": _http("https://mcp.example.com/mcp?auth=P28-FAKE-AUTH-7"),
    "probe-sig": _http("https://mcp.example.com/mcp?sig=P28-FAKE-SIG-8"),
    "ok-env-key": _http("https://mcp.example.com/mcp?key=${MCP_KEY}"),
    "space-in-path": _http("https://mcp.example.com/a b"),
    "no-scheme": _http("mcp.example.com/mcp"),
    "tencent-metadata-ip": _http("https://169.254.0.23/meta-data/"),
    "azure-host-ip": _http("http://168.63.129.16/machine?comp=goalstate"),
    "aws-instance-data-name": _http("https://instance-data/latest/meta-data/"),
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_static_endpoint_rules(tmp_path: Path, codex: bool) -> None:
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", _STATIC, codex=codex))
    findings = _by_server(result)

    for name in ("probe-key", "probe-auth", "probe-sig"):
        assert findings[name] == {"mcp_url_inline_secret": Severity.CRITICAL}, (name, findings.get(name))
    # Claude Code expands the reference; Codex sends '${MCP_KEY}' literally (the H8 rule).
    assert findings.get("ok-env-key") == ({"mcp_url_env_not_expanded": Severity.HIGH} if codex else None)
    assert "space-in-path" not in findings
    assert findings["no-scheme"] == {"mcp_url_scheme_missing": Severity.HIGH}
    assert findings["tencent-metadata-ip"] == {"mcp_endpoint_metadata": Severity.HIGH}
    assert findings["azure-host-ip"]["mcp_endpoint_metadata"] == Severity.HIGH
    assert findings["aws-instance-data-name"] == {"mcp_endpoint_metadata": Severity.HIGH}
    [missing] = [f for f in result.findings if f.check_name == "mcp_url_scheme_missing"]
    assert "dangerous" not in missing.message
    assert not any("P28-FAKE" in f.message for f in result.findings)


def test_hook_urls_get_the_same_credential_keys_and_path_rule(tmp_path: Path) -> None:
    hooks = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Write",
                    "hooks": [
                        {"type": "http", "url": "https://hooks.example.com/ingest?key=P28-FAKE-HOOKKEY-1"},
                        {"type": "http", "url": "https://hooks.example.com/in gest"},
                    ],
                }
            ]
        }
    }
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", {}, hooks=hooks))
    checks = [(f.check_name, f.message) for f in result.findings if f.check_name.startswith("plugin_hook_")]

    assert any(check == "plugin_hook_inline_secret" for check, _message in checks)
    assert not any(check == "plugin_hook_http_url_invalid" for check, _message in checks), checks


def test_backslash_url_messages_name_the_host_clients_reach(tmp_path: Path) -> None:
    """check-09 e06: the metadata finding showed 'https://pub.example.com/' and a userinfo CRITICAL fired."""
    servers = {"backslash-at": _http("https://169.254.169.254\\@pub.example.com/")}
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", servers))

    assert _by_server(result)["backslash-at"] == {
        "mcp_url_malformed_authority": Severity.HIGH,
        "mcp_endpoint_metadata": Severity.HIGH,
    }
    [metadata] = [f for f in result.findings if f.check_name == "mcp_endpoint_metadata"]
    assert "169.254.169.254" in metadata.message
    assert "'https://pub.example.com/'" not in metadata.message


def test_a_literal_password_hidden_behind_a_backslash_is_still_found(tmp_path: Path) -> None:
    servers = {"hidden": _http("https://user:FAKEPASS1234567\\@mcp.example.com/")}
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", servers))

    assert _by_server(result)["hidden"]["mcp_url_inline_secret"] == Severity.CRITICAL
    assert not any("FAKEPASS1234567" in f.message for f in result.findings)


# --------------------------------------------------------------------------- #
# --resolve-endpoints: a failed HEAD or redirect lookup is INCOMPLETE          #
# --------------------------------------------------------------------------- #
def _target(url: str) -> EndpointTarget:
    return EndpointTarget(url=url, kind="mcp", name="srv", file_path="plugin.json")


def test_a_head_that_never_completes_makes_the_check_incomplete() -> None:
    """check-09 p07b '/trickle': the row said head_failed, with no finding and no INCOMPLETE."""

    def head(*_args: object) -> HeadResult:
        raise TimeoutError("no complete response within 5.0s")

    checker = EndpointChecker(resolver=lambda *_args: ["93.184.216.34"], head=head)
    summary, findings = checker.check([_target("https://trickle.example.com/mcp")])

    assert [(f.check_name, f.severity) for f in findings] == [
        ("endpoint_head_failed", Severity.LOW),
        ("endpoint_resolution_incomplete", Severity.MEDIUM),
    ]
    assert summary["incomplete"] is True
    assert any("HEAD request failed" in reason for reason in summary["incomplete_reasons"])
    assert findings[0].metadata == {"mcp_server": "srv"}


def test_a_redirect_target_that_does_not_resolve_makes_the_check_incomplete() -> None:
    def resolve(host: str, _port: int, _timeout: float) -> list[str]:
        if host == "mcp.example.com":
            return ["93.184.216.34"]
        raise socket.gaierror("nxdomain")

    checker = EndpointChecker(resolver=resolve, head=lambda *_args: HeadResult(302, "https://hidden.example.net/next"))
    summary, findings = checker.check([_target("https://mcp.example.com/mcp")])

    assert [(f.check_name, f.severity) for f in findings] == [
        ("endpoint_redirect_unresolved", Severity.LOW),
        ("endpoint_resolution_incomplete", Severity.MEDIUM),
    ]
    assert summary["endpoints"][0]["redirect"]["classification"] == "unresolved"
    assert any("redirect target" in reason for reason in summary["incomplete_reasons"])


def test_validate_reports_a_failed_head_as_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(er, "_resolve", lambda *_args: ["93.184.216.34"])

    def head(*_args: object) -> HeadResult:
        raise TimeoutError("no complete response within 5.0s")

    monkeypatch.setattr(er, "_head", head)
    root = _plugin(tmp_path / "p", {"remote": _http("https://mcp.example.com/mcp")})

    [result] = run_validation(root, checks="schema", content_type=CONTENT_TYPE_PLUGIN, resolve_endpoints=True)

    assert "endpoint-resolution" in result.incomplete_scans
    assert "endpoint_head_failed" in {f.check_name for f in result.findings}


def test_a_name_resolving_to_a_newly_classified_metadata_ip_is_high() -> None:
    checker = EndpointChecker(resolver=lambda *_args: ["168.63.129.16"], head=lambda *_args: pytest.fail("contacted"))
    _summary, findings = checker.check([_target("https://wire.example.com/")])

    assert [(f.check_name, f.severity) for f in findings] == [("endpoint_resolves_metadata", Severity.HIGH)]
