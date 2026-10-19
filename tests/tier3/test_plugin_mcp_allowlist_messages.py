# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A plugin's MCP servers pass the operator's MCP security policy, and the refusal says how to allow them.

The policy (main's validate_mcp_server_declarations) guards every MCP server Tier 3 stages: a skill's
evals/environment/mcp_servers.toml, the plugin servers the wrapper stages as plugin_mcp_servers.toml, and
the plugin servers a native adapter writes into its harness config. A plugin user sees the error with the
plugin's servers named as such and the two host settings, with an example value, that approve them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.tier3.harbor.adapter import (
    PLUGIN_MCP_SERVERS_FILENAME,
    PLUGIN_MCP_SOURCE_LABEL,
    _load_mcp_servers,
    _native_plugin_mcp_declarations,
    validate_mcp_server_declarations,
)


def _staged_plugin_servers(tmp_path: Path, toml: str) -> Path:
    package = tmp_path / "demo-plugin-plugin-eval"
    environment = package / "evals" / "environment"
    environment.mkdir(parents=True)
    (environment / PLUGIN_MCP_SERVERS_FILENAME).write_text(toml, encoding="utf-8")
    return package


STDIO_WITH_TOKEN = """
[[mcp_servers]]
name = "tracker"
command = "npx"
args = ["-y", "@example/tracker-mcp@1.4.2", "--token", "${TRACKER_TOKEN}"]
transport = "stdio"
"""

REMOTE_WITH_KEY = """
[[mcp_servers]]
name = "docs"
url = "https://mcp.example.test/sse?key=${DOCS_KEY}"
transport = "sse"
"""


def test_a_plugin_server_env_reference_names_both_settings_with_an_example(tmp_path: Path) -> None:
    package = _staged_plugin_servers(tmp_path, STDIO_WITH_TOKEN)

    with pytest.raises(ValueError) as raised:
        _load_mcp_servers(package, PLUGIN_MCP_SERVERS_FILENAME, allowed_runtime_env={})

    message = str(raised.value)
    assert message.startswith(f"MCP server 'tracker' in {PLUGIN_MCP_SOURCE_LABEL} references environment variable")
    assert "SKILLEVALUATOR_ALLOWED_MCP_SECRETS (and SKILLEVALUATOR_ALLOWED_MCP_HOSTS for remote servers)" in message
    assert "for example SKILLEVALUATOR_ALLOWED_MCP_SECRETS=TRACKER_TOKEN in the environment that runs" in message
    assert "comma-separated" in message
    # The plugin user never sees the generated staging path.
    assert str(package) not in message


def test_the_named_settings_approve_the_plugin_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLEVALUATOR_ALLOWED_MCP_SECRETS", "OTHER_TOKEN,TRACKER_TOKEN")
    package = _staged_plugin_servers(tmp_path, STDIO_WITH_TOKEN)

    [server] = _load_mcp_servers(package, PLUGIN_MCP_SERVERS_FILENAME, allowed_runtime_env={})

    assert server["name"] == "tracker"


def test_a_remote_plugin_server_names_its_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _staged_plugin_servers(tmp_path, REMOTE_WITH_KEY)

    with pytest.raises(ValueError) as unapproved_secret:
        _load_mcp_servers(package, PLUGIN_MCP_SERVERS_FILENAME, allowed_runtime_env={})
    assert "SKILLEVALUATOR_ALLOWED_MCP_SECRETS=DOCS_KEY and SKILLEVALUATOR_ALLOWED_MCP_HOSTS=mcp.example.test" in str(
        unapproved_secret.value
    )

    monkeypatch.setenv("SKILLEVALUATOR_ALLOWED_MCP_SECRETS", "DOCS_KEY")
    with pytest.raises(ValueError) as unapproved_host:
        _load_mcp_servers(package, PLUGIN_MCP_SERVERS_FILENAME, allowed_runtime_env={})
    message = str(unapproved_host.value)
    assert f"in {PLUGIN_MCP_SOURCE_LABEL} sends secret '${{DOCS_KEY}}' to unapproved host 'mcp.example.test'" in message
    assert "for example SKILLEVALUATOR_ALLOWED_MCP_HOSTS=mcp.example.test" in message

    monkeypatch.setenv("SKILLEVALUATOR_ALLOWED_MCP_HOSTS", "mcp.example.test")
    assert [server["name"] for server in _load_mcp_servers(package, PLUGIN_MCP_SERVERS_FILENAME)] == ["docs"]


def _native_plugin(agent: str, *, copies_plugin_tree: bool) -> SimpleNamespace:
    source = SimpleNamespace(
        mcp_servers=[{"name": "tracker", "command": "npx", "args": ["-y", "@example/tracker-mcp@1.4.2"]}],
        plugin_file_mcp_servers=[],
        mcp_declared={"tracker": {"env": {"TRACKER_API_TOKEN": "${TRACKER_TOKEN}"}}},
    )
    return SimpleNamespace(source=source, adapter=SimpleNamespace(agent=agent, copies_plugin_tree=copies_plugin_tree))


def test_native_claude_code_plugin_servers_pass_the_same_policy() -> None:
    """Claude Code applies a server's declared env, so its native staging is checked with it."""
    declarations = _native_plugin_mcp_declarations(_native_plugin("claude-code", copies_plugin_tree=True))

    assert declarations == [
        {
            "name": "tracker",
            "command": "npx",
            "args": ["-y", "@example/tracker-mcp@1.4.2"],
            "env": {"TRACKER_API_TOKEN": "${TRACKER_TOKEN}"},
        }
    ]
    with pytest.raises(ValueError, match="for example SKILLEVALUATOR_ALLOWED_MCP_SECRETS=TRACKER_TOKEN"):
        validate_mcp_server_declarations(declarations, allowed_runtime_env={}, source_label=PLUGIN_MCP_SOURCE_LABEL)


def test_a_native_adapter_that_does_not_apply_declared_env_is_checked_on_what_it_stages() -> None:
    declarations = _native_plugin_mcp_declarations(_native_plugin("codex", copies_plugin_tree=False))

    assert declarations == [{"name": "tracker", "command": "npx", "args": ["-y", "@example/tracker-mcp@1.4.2"]}]
    assert validate_mcp_server_declarations(declarations, allowed_runtime_env={}) == declarations
