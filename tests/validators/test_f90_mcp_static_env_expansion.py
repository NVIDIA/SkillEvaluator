# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment expansion in MCP configs (checks 4 and 9): proof bug H8.

Claude Code expands ``${VAR}`` and ``${VAR:-default}`` in an MCP server's url,
env, args, and headers; its own docs use ``"${API_BASE_URL:-https://api.example.com}/mcp"``.
Codex does not expand them in plugin MCP URLs. Cursor documents ``${env:NAME}``.
The fixtures follow check-09 e11, skeptic probes x4 to x8, and the audit
re-test check-09-codex-urls.
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
    "description": "Env expansion probe (Codex)",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "Env probe",
        "shortDescription": "Env probe",
        "longDescription": "A plugin used to test MCP URL expansion.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}
_MANIFESTS = {
    "claude": (".claude-plugin/plugin.json", {"name": "demo", "version": "1.0.0", "description": "Env probe"}),
    "codex": (".codex-plugin/plugin.json", _CODEX_MANIFEST),
    "cursor": (".cursor-plugin/plugin.json", {"name": "demo", "version": "1.0.0", "description": "Env probe"}),
}
_MCP_FILE = {"claude": ".mcp.json", "codex": ".mcp.json", "cursor": "mcp.json"}


def _plugin(root: Path, servers: dict[str, dict], fmt: str = "claude") -> Path:
    manifest_rel, manifest = _MANIFESTS[fmt]
    for rel, content in {manifest_rel: manifest, _MCP_FILE[fmt]: {"mcpServers": servers}}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(content), encoding="utf-8")
    return root


def _validate(root: Path) -> ValidationResult:
    return PluginSchemaValidator().validate(root)


def _by_server(result: ValidationResult) -> dict[str, dict[str, Severity]]:
    found: dict[str, dict[str, Severity]] = {}
    for finding in result.findings:
        server = finding.metadata.get("mcp_server")
        if server:
            found.setdefault(server, {})[finding.check_name] = finding.severity
    return found


_DOCS_FORMS = {
    # The Claude Code MCP docs' own example.
    "doc-example": {"type": "http", "url": "${API_BASE_URL:-https://api.example.com}/mcp"},
    "path-default": {"type": "http", "url": "https://example.com${MCP_PATH:-/mcp}"},
    "whole-url-var": {"type": "http", "url": "${MCP_SERVER_URL}"},
    "host-var": {"type": "http", "url": "https://${MCP_HOST}/mcp"},
    "path-var": {"type": "http", "url": "https://api.example.com/${TENANT}/mcp"},
}


def test_claude_documented_expansion_forms_pass(tmp_path: Path) -> None:
    """check-09 e11: the docs example was CRITICAL mcp_url_dangerous_scheme and failed Tier 1."""
    result = _validate(_plugin(tmp_path / "p", _DOCS_FORMS))

    findings = _by_server(result)
    assert "doc-example" not in findings
    assert "path-default" not in findings
    assert "path-var" not in findings
    assert findings["whole-url-var"] == {"mcp_url_env_unchecked": Severity.LOW}
    assert findings["host-var"] == {"mcp_url_env_unchecked": Severity.LOW}
    assert result.passed, [f"{f.check_name}: {f.message}" for f in result.findings]


def test_claude_default_is_checked_like_any_url(tmp_path: Path) -> None:
    servers = {
        "host-default-metadata": {"type": "http", "url": "https://${MCP_HOST:-169.254.169.254}/latest/meta-data/"},
        "plaintext-default": {"type": "http", "url": "${MCP_URL:-http://mcp.example.com/mcp}"},
    }
    findings = _by_server(_validate(_plugin(tmp_path / "p", servers)))

    assert findings["host-default-metadata"] == {"mcp_endpoint_metadata": Severity.HIGH}
    assert findings["plaintext-default"] == {"mcp_url_insecure_scheme": Severity.HIGH}


def test_codex_gets_a_not_expanded_finding_not_a_dangerous_scheme(tmp_path: Path) -> None:
    """Codex 0.142.5 reads '${VAR:-default}' literally ("relative URL without a base") and never connects."""
    result = _validate(_plugin(tmp_path / "p", {"doc-example": _DOCS_FORMS["doc-example"]}, "codex"))

    assert _by_server(result) == {"doc-example": {"mcp_url_env_not_expanded": Severity.HIGH}}
    [finding] = [f for f in result.findings if f.check_name == "mcp_url_env_not_expanded"]
    assert "Codex" in finding.message and "dangerous" not in finding.message


@pytest.mark.parametrize("fmt", ["claude", "codex"])
def test_empty_default_references_are_not_inline_secrets(tmp_path: Path, fmt: str) -> None:
    """skeptic x4: '${GITHUB_TOKEN:-}' in env or after --token was CRITICAL."""
    servers = {
        "env-default-empty": {
            "command": "npx",
            "args": ["-y", "@scope/server@1.2.3"],
            "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN:-}"},
        },
        "arg-default-empty": {"command": "npx", "args": ["-y", "@scope/server@1.2.3", "--token", "${GITHUB_TOKEN:-}"]},
        "arg-literal-default": {"command": "npx", "args": ["-y", "@scope/server@1.2.3", "--region", "${REGION:-us-1}"]},
        "auth-header": {
            "type": "http",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer ${API_KEY}", "X-Api-Token": "Bearer ${API_TOKEN:-}"},
        },
    }
    findings = _by_server(_validate(_plugin(tmp_path / "p", servers, fmt)))

    blocking = {
        name: checks
        for name, checks in findings.items()
        if any(severity in (Severity.CRITICAL, Severity.HIGH) for severity in checks.values())
    }
    assert blocking == {}


def test_a_secret_shipped_as_a_default_is_still_flagged(tmp_path: Path) -> None:
    servers = {
        "default-secret": {
            "command": "npx",
            "args": ["-y", "@scope/server@1.2.3"],
            "env": {"API_KEY": "${API_KEY:-sk-FAKE0000000000000000000000}"},
        },
        "default-arg-secret": {
            "command": "npx",
            "args": ["-y", "@scope/server@1.2.3", "--token", "${TOKEN:-rawFAKEvalue0123}"],
        },
    }
    findings = _by_server(_validate(_plugin(tmp_path / "p", servers)))

    assert findings["default-secret"]["mcp_inline_secret"] == Severity.CRITICAL
    assert findings["default-arg-secret"]["mcp_command_inline_secret"] == Severity.CRITICAL


def test_cursor_env_interpolation_is_a_reference_marked_unverified(tmp_path: Path) -> None:
    """skeptic x8: Cursor '${env:NAME}' was CRITICAL; whether Cursor expands it is not proven."""
    servers = {
        "gh": {"command": "npx", "args": ["-y", "@scope/server@1.2.3"], "env": {"GITHUB_TOKEN": "${env:GITHUB_TOKEN}"}},
        "api": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ${env:API_TOKEN}"}},
        "argref": {"command": "npx", "args": ["-y", "@scope/server@1.2.3", "--api-key", "${env:API_KEY}"]},
    }
    findings = _by_server(_validate(_plugin(tmp_path / "p", servers, "cursor")))

    for name in servers:
        assert findings[name].get("mcp_env_reference_unverified") == Severity.LOW, findings[name]
        assert Severity.CRITICAL not in findings[name].values(), findings[name]


def test_tier3_staging_accepts_the_docs_example() -> None:
    _reject_unsafe_mcp_declaration("api", {"type": "http", "url": "${API_BASE_URL:-https://api.example.com}/mcp"})


def test_resolve_endpoints_checks_the_default_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """check-09 e11 run-resolve: every expansion URL was skipped without a lookup."""
    resolved: list[str] = []

    def _resolve(host: str, _port: int, _timeout: float) -> list[str]:
        resolved.append(host)
        return ["10.2.3.4"]

    monkeypatch.setattr(er, "_resolve", _resolve)
    monkeypatch.setattr(er, "_head", lambda *_args: pytest.fail("a non-public host must never be contacted"))
    root = _plugin(tmp_path / "p", {"doc-example": _DOCS_FORMS["doc-example"]})

    [result] = run_validation(root, checks="schema", content_type=CONTENT_TYPE_PLUGIN, resolve_endpoints=True)

    assert resolved == ["api.example.com"]
    assert "endpoint_resolves_private" in {f.check_name for f in result.findings}
