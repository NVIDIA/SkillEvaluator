# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenClaw inline MCP servers: dialect normalization and the shared static policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.constants import (
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_OPENCLAW_MANIFEST_TYPE,
)
from skillevaluator.models.result import Severity
from skillevaluator.plugin_formats import OPENCLAW_PROFILE
from skillevaluator.plugin_mcp import McpCollection, collect_mcp_declarations
from skillevaluator.plugin_paths import PluginRootReader
from skillevaluator.validators.mcp_static import openclaw_ssl_verification_enabled, validate_mcp_server_declaration

MANIFEST = "openclaw.plugin.json"


def _collect(root: Path, servers: dict[str, Any], files: dict[str, str] | None = None) -> McpCollection:
    for rel, text in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    manifest = {"id": "demo", "configSchema": {}, "mcpServers": servers}
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return collect_mcp_declarations(
        PluginRootReader(root), manifest, contained=True, manifest_rel=MANIFEST, profile=OPENCLAW_PROFILE
    )


def _one(collection: McpCollection):
    [declaration] = collection.declarations
    return declaration


def _checks(findings, severity: Severity | None = None) -> set[str]:
    return {f.check_name for f in findings if severity is None or f.severity == severity}


@pytest.mark.parametrize(
    ("server", "transport"),
    [
        ({"url": "https://mcp.example.com/sse"}, "sse"),
        ({"url": "https://mcp.example.com/mcp", "type": "http"}, "http"),
        ({"url": "https://mcp.example.com/mcp", "type": "streamable-http"}, "http"),
        ({"url": "https://mcp.example.com/mcp", "transport": "streamable-http"}, "http"),
        # The canonical transport wins over the legacy type.
        ({"url": "https://mcp.example.com/sse", "transport": "sse", "type": "http"}, "sse"),
        ({"url": "https://mcp.example.com/mcp", "transport": "streamable-http", "type": "sse"}, "http"),
        # A command wins over URL transports.
        ({"command": "node", "transport": "sse"}, "stdio"),
    ],
)
def test_transport_precedence_and_sse_default(tmp_path: Path, server: dict, transport: str) -> None:
    collection = _collect(tmp_path, {"remote": server})
    declaration = _one(collection)
    assert declaration.config["transport"] == transport
    assert "type" not in declaration.config
    assert declaration.transport == transport
    assert "mcp_transport_invalid" not in _checks(collection.findings)


def test_invalid_canonical_transport_is_not_treated_as_legacy_http(tmp_path: Path) -> None:
    collection = _collect(tmp_path, {"remote": {"url": "https://x.example/mcp", "transport": "ws", "type": "http"}})
    assert _one(collection).config["transport"] == "ws"
    assert "mcp_transport_invalid" in _checks(collection.findings, Severity.HIGH)


@pytest.mark.parametrize(
    ("server", "cwd"),
    [
        ({"command": "node", "args": ["server.js"]}, "."),
        ({"command": "node", "cwd": ""}, "."),
        ({"command": "node", "cwd": "", "workingDirectory": "srv"}, "."),
        ({"command": "node", "workingDirectory": "srv"}, "srv"),
        ({"command": "node", "cwd": "srv", "workingDirectory": "other"}, "srv"),
        ({"command": "node", "cwd": 7, "workingDirectory": "srv"}, "srv"),
    ],
)
def test_stdio_launches_keep_their_plugin_root_working_directory(tmp_path: Path, server: dict, cwd: str) -> None:
    assert _one(_collect(tmp_path, {"local": server})).config["cwd"] == cwd


@pytest.mark.parametrize("server", [{"command": "node", "args": ["index.js"]}, {"command": "node", "cwd": ""}])
def test_root_default_working_directory_cannot_hide_unscanned_server_code(tmp_path: Path, server: dict) -> None:
    server = {**server, "args": ["node_modules/srv/index.js"]}
    collection = _collect(tmp_path, {"local": server}, {"node_modules/srv/index.js": "x"})
    assert "plugin_component_path_unscanned" in _checks(collection.server_findings, Severity.HIGH)


@pytest.mark.parametrize("field", ["cwd", "workingDirectory"])
def test_bare_relative_working_directory_is_checked(tmp_path: Path, field: str) -> None:
    collection = _collect(tmp_path, {"local": {"command": "node", "args": ["index.js"], field: "node_modules/srv"}})
    assert "plugin_component_path_unscanned" in _checks(collection.server_findings, Severity.HIGH)


@pytest.mark.parametrize(
    "url",
    ["https://${HOST}/mcp", "https://${HOST:-mcp.example.com}/mcp", "https://mcp.example.com/${PATH_PART}"],
)
def test_url_environment_references_are_verbatim_and_high(tmp_path: Path, url: str) -> None:
    collection = _collect(tmp_path, {"remote": {"url": url}})
    [finding] = [f for f in collection.server_findings if f.check_name == "mcp_url_env_not_expanded"]
    assert finding.severity == Severity.HIGH
    assert "OpenClaw" in finding.message and "Codex" not in finding.message
    assert "an OpenClaw plugin" in (finding.suggestion or "")


def test_other_clients_keep_their_url_environment_rules() -> None:
    url = "https://${HOST:-mcp.example.com}/mcp"
    claude = validate_mcp_server_declaration("s", {"url": url}, "f", manifest_type=PLUGIN_CONTAINED_MANIFEST_TYPE)
    codex = validate_mcp_server_declaration("s", {"url": url}, "f", manifest_type=PLUGIN_CODEX_MANIFEST_TYPE)
    assert "mcp_url_env_not_expanded" not in _checks(claude)
    [finding] = [f for f in codex if f.check_name == "mcp_url_env_not_expanded"]
    assert "Codex" in finding.message and "OpenClaw" not in finding.message


def test_url_expansion_rejection_still_checks_inline_secrets() -> None:
    # Assembled at run time so the full-history secret scan never sees a literal credential.
    sample = "sk-live-" + "0123456789abcdefghij" + "klmnop"
    url = f"https://${{HOST}}/mcp?api_key={sample}"
    findings = validate_mcp_server_declaration("s", {"url": url}, "f", manifest_type=PLUGIN_OPENCLAW_MANIFEST_TYPE)
    assert "mcp_url_env_not_expanded" in _checks(findings)
    assert any(f.severity in {Severity.HIGH, Severity.CRITICAL} and "secret" in f.check_name for f in findings)
    assert all(sample not in f.message for f in findings)


@pytest.mark.parametrize(
    ("options", "enabled"),
    [
        ({}, True),
        ({"sslVerify": True}, True),
        ({"ssl_verify": True}, True),
        ({"sslVerify": False}, False),
        ({"ssl_verify": False}, False),
        # The canonical key overrides the legacy alias in both directions.
        ({"sslVerify": True, "ssl_verify": False}, True),
        ({"sslVerify": False, "ssl_verify": True}, False),
        # A non-boolean value is not a decision; the next source applies.
        ({"sslVerify": "false", "ssl_verify": False}, False),
        ({"sslVerify": None, "ssl_verify": True}, True),
    ],
)
def test_effective_tls_policy(options: dict, enabled: bool) -> None:
    assert openclaw_ssl_verification_enabled(options) is enabled
    findings = validate_mcp_server_declaration(
        "s", {"url": "https://x.example/mcp", **options}, "f", manifest_type=PLUGIN_OPENCLAW_MANIFEST_TYPE
    )
    assert ("mcp_insecure_tls_config" in _checks(findings, Severity.CRITICAL)) is not enabled


def test_tls_options_are_openclaw_specific() -> None:
    findings = validate_mcp_server_declaration(
        "s", {"url": "https://x.example/mcp", "sslVerify": False}, "f", manifest_type=PLUGIN_CONTAINED_MANIFEST_TYPE
    )
    assert "mcp_insecure_tls_config" not in _checks(findings)


@pytest.mark.parametrize(
    ("options", "unapplied"),
    [
        ({}, ()),
        ({"sslVerify": True}, ()),
        ({"ssl_verify": True}, ()),
        ({"sslVerify": False}, ("sslVerify",)),
        ({"ssl_verify": False}, ("ssl_verify",)),
        ({"sslVerify": True, "ssl_verify": False}, ()),
        ({"auth": {"type": "oauth"}, "toolFilter": ["read"]}, ("auth", "toolFilter")),
        ({"enabled": False, "headers": {"X-Trace": "1"}}, ()),
    ],
)
def test_host_only_options_are_recorded_as_unapplied(tmp_path: Path, options: dict, unapplied: tuple) -> None:
    declaration = _one(_collect(tmp_path, {"remote": {"url": "https://x.example/mcp", **options}}))
    assert declaration.unapplied == unapplied


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("enterprise", "idpClientSecret"),
        ("enterprise", "resourceClientSecret"),
        ("auth", "token"),
        ("auth", "clientSecret"),
    ],
)
def test_auth_inline_credentials_block_without_echoing_values(section: str, key: str) -> None:
    # Assembled at run time so the full-history secret scan never sees a literal credential.
    sample = "s3cr3t-Value-" + "0123456789abcdef" + "ABCDEF"
    config = {"url": "https://x.example/mcp", section: {key: sample, "type": "bearer"}}
    findings = validate_mcp_server_declaration("s", config, "f", manifest_type=PLUGIN_OPENCLAW_MANIFEST_TYPE)
    blocking = [f for f in findings if f.severity in {Severity.HIGH, Severity.CRITICAL}]
    assert blocking
    assert all(sample not in f.message for f in findings)


@pytest.mark.parametrize(
    ("section", "block"),
    [
        ("enterprise", {"idpClientSecret": "${IDP_CLIENT_SECRET}"}),
        ("auth", {"type": "oauth"}),
        ("auth", {"type": "bearer", "token": "${MCP_TOKEN}"}),
    ],
)
def test_auth_credential_references_are_allowed(section: str, block: dict) -> None:
    config = {"url": "https://x.example/mcp", section: block}
    findings = validate_mcp_server_declaration("s", config, "f", manifest_type=PLUGIN_OPENCLAW_MANIFEST_TYPE)
    assert not [f for f in findings if f.severity in {Severity.HIGH, Severity.CRITICAL}]
