# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plaintext loopback MCP URLs and OAuth metadata URLs (check 9): proof bug M23.

check-09 p01 and the audit re-test check-09-codex-urls: ``http://localhost``
MCP servers were HIGH and failed Tier 1, while the same hook URL is exempt, and
``oauth.authServerMetadataUrl`` (which Claude Code fetches) was never checked.
Both the Claude Code and the Codex formats are tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.tier1.commands import run_validation
from skillevaluator.tier3.plugin_eval import _reject_unsafe_mcp_declaration
from skillevaluator.validators import endpoint_resolution as er
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_CODEX_MANIFEST = {
    "name": "demo",
    "version": "1.0.0",
    "description": "URL probe (Codex)",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "URL probe",
        "shortDescription": "URL probe",
        "longDescription": "A plugin used to test MCP URL checks.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}


def _plugin(root: Path, servers: dict[str, dict], *, codex: bool = False) -> Path:
    manifest = (
        (".codex-plugin/plugin.json", _CODEX_MANIFEST)
        if codex
        else (".claude-plugin/plugin.json", {"name": "demo", "version": "1.0.0", "description": "URL probe"})
    )
    for rel, content in (manifest, (".mcp.json", {"mcpServers": servers})):
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


_P01 = {
    "plain-public": {"type": "http", "url": "http://mcp.example.com/mcp"},
    "plain-ws": {"type": "http", "url": "ws://mcp.example.com/ws"},
    "plain-localhost": {"type": "http", "url": "http://localhost:8080/mcp"},
    "plain-loopback-ip": {"type": "http", "url": "http://127.0.0.1:8080/mcp"},
    "plain-loopback-v6": {"type": "http", "url": "http://[::1]:8080/mcp"},
    "tls-public": {"type": "http", "url": "https://mcp.example.com/mcp"},
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_plaintext_loopback_follows_the_hook_rule(tmp_path: Path, codex: bool) -> None:
    findings = _by_server(PluginSchemaValidator().validate(_plugin(tmp_path / "p", _P01, codex=codex)))

    for name in ("plain-localhost", "plain-loopback-ip", "plain-loopback-v6"):
        assert findings[name] == {"mcp_endpoint_private": Severity.MEDIUM}, (name, findings[name])
    assert findings["plain-public"] == {"mcp_url_insecure_scheme": Severity.HIGH}
    assert findings["plain-ws"] == {"mcp_url_insecure_scheme": Severity.HIGH}
    assert "tls-public" not in findings


def test_a_loopback_only_plugin_passes_tier1_and_tier3_staging(tmp_path: Path) -> None:
    servers = {"desktop-app": {"type": "http", "url": "http://127.0.0.1:8765/mcp"}}
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", servers))

    assert result.passed, [f"{f.check_name}: {f.message}" for f in result.findings]
    _reject_unsafe_mcp_declaration("desktop-app", servers["desktop-app"])


def _oauth_server(metadata_url: str) -> dict:
    return {
        "type": "http",
        "url": "https://api.example.com/mcp",
        "oauth": {"clientId": "demo", "authServerMetadataUrl": metadata_url},
    }


def test_oauth_metadata_url_gets_the_url_policy(tmp_path: Path) -> None:
    servers = {
        "oauth-metadata": _oauth_server("http://169.254.169.254/latest/meta-data/"),
        "oauth-plaintext": _oauth_server("http://auth.example.com/.well-known/oauth-authorization-server"),
        "oauth-secret": _oauth_server("https://auth.example.com/.well-known/x?token=FAKEOAUTHTOKEN123456"),
        "oauth-clean": _oauth_server("https://auth.example.com/.well-known/oauth-authorization-server"),
    }
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", servers))
    findings = _by_server(result)

    assert findings["oauth-metadata"]["mcp_endpoint_metadata"] == Severity.HIGH
    assert findings["oauth-plaintext"] == {"mcp_url_insecure_scheme": Severity.HIGH}
    assert findings["oauth-secret"] == {"mcp_url_inline_secret": Severity.CRITICAL}
    assert "oauth-clean" not in findings
    messages = [f.message for f in result.findings if f.metadata.get("mcp_server") == "oauth-metadata"]
    assert all("oauth.authServerMetadataUrl" in message for message in messages)
    assert not any("FAKEOAUTHTOKEN" in f.message for f in result.findings)


def test_resolve_endpoints_resolves_the_oauth_metadata_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resolved: list[str] = []

    def _resolve(host: str, _port: int, _timeout: float) -> list[str]:
        resolved.append(host)
        return ["169.254.169.254"] if host == "auth.example.com" else ["10.9.9.9"]

    monkeypatch.setattr(er, "_resolve", _resolve)
    monkeypatch.setattr(er, "_head", lambda *_args: pytest.fail("a non-public host must never be contacted"))
    servers = {"oauth": _oauth_server("https://auth.example.com/.well-known/oauth-authorization-server")}

    [result] = run_validation(
        _plugin(tmp_path / "p", servers), checks="schema", content_type=CONTENT_TYPE_PLUGIN, resolve_endpoints=True
    )

    assert sorted(resolved) == ["api.example.com", "auth.example.com"]
    assert "endpoint_resolves_metadata" in {f.check_name for f in result.findings}
