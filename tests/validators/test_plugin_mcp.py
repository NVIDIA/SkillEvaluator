# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP declaration collection: merge order, launch paths, and reported references (plugin_mcp)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.plugin_mcp import McpCollection, collect_mcp_declarations
from skillevaluator.plugin_paths import PluginRootReader

_MANIFEST = ".claude-plugin/plugin.json"


def _collect(root: Path, mcp_servers: Any, files: dict[str, Any] | None = None) -> McpCollection:
    (root / ".claude-plugin").mkdir(exist_ok=True)
    for rel, payload in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {"name": "demo"} if mcp_servers is None else {"name": "demo", "mcpServers": mcp_servers}
    return collect_mcp_declarations(PluginRootReader(root), manifest, contained=True, manifest_rel=_MANIFEST)


_VALID = {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/server.js"]}


@pytest.mark.parametrize("later", ["oops", {"command": 5}, {"command": ""}])
@pytest.mark.parametrize("declared", ["./extra.json", ["./extra.json"]])
def test_invalid_later_config_file_server_does_not_replace_earlier(tmp_path: Path, later: Any, declared: Any) -> None:
    """Regression: Claude Code drops a config-file server that fails its schema before merging, so the earlier
    valid server runs; the collection marked the invalid one effective and said the later one is kept."""
    files = {".mcp.json": {"mcpServers": {"srv": _VALID}}, "extra.json": {"mcpServers": {"srv": later}}}
    collection = _collect(tmp_path, declared, files)
    [effective] = collection.effective
    assert (effective.file, effective.config) == (".mcp.json", _VALID)
    assert not [f for f in collection.findings if f.check_name == "mcp_server_duplicate_name"]


def test_valid_later_config_file_server_replaces_earlier(tmp_path: Path) -> None:
    later = {"url": "https://example.com/mcp"}
    files = {".mcp.json": {"mcpServers": {"srv": _VALID}}, "extra.json": {"mcpServers": {"srv": later}}}
    collection = _collect(tmp_path, "./extra.json", files)
    [effective] = collection.effective
    assert effective.file == "extra.json"
    assert [f.check_name for f in collection.findings].count("mcp_server_duplicate_name") == 1


def test_invalid_later_inline_server_still_replaces_earlier(tmp_path: Path) -> None:
    """An inline map is merged before it is checked, so its server replaces the file's either way."""
    collection = _collect(tmp_path, {"srv": "oops"}, {".mcp.json": {"mcpServers": {"srv": _VALID}}})
    [effective] = collection.effective
    assert (effective.source, effective.config) == ("inline", "oops")


@pytest.mark.parametrize(
    "launch",
    [
        {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}\\node_modules\\evil\\index.js"]},
        {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/node_modules\\evil\\index.js"]},
        {"command": "node", "args": ["$CLAUDE_PLUGIN_ROOT\\node_modules\\evil\\index.js"]},
        {"command": "node", "args": ["--entry=${CLAUDE_PLUGIN_ROOT}\\node_modules\\evil\\index.js"]},
        {"command": "${CLAUDE_PLUGIN_ROOT}\\node_modules\\.bin\\evil.cmd"},
    ],
)
def test_backslash_after_root_variable_names_unscanned_server_code(tmp_path: Path, launch: dict[str, Any]) -> None:
    """Regression: a root variable followed by '\\' (a Windows separator) hid server code in node_modules/."""
    collection = _collect(tmp_path, None, {".mcp.json": {"mcpServers": {"srv": launch}}})
    unscanned = [f for f in collection.server_findings if f.check_name == "plugin_component_path_unscanned"]
    assert len(unscanned) == 1
    assert "node_modules/" in unscanned[0].message


def _password() -> str:
    return "pw-" + "test-value"


@pytest.mark.parametrize(
    "ref",
    [
        "ws://user:{secret}@example.com/srv.mcpb",
        "wss://user:{secret}@example.com/srv.mcpb",
        "ftp://user:{secret}@example.com/servers.json",
        "https:/user:{secret}@example.com/srv.mcpb",
        "https:\\\\user:{secret}@example.com/srv.mcpb",
        "//user:{secret}@example.com/srv.mcpb",
        "./servers.json?token={secret}",
        "./servers.json?token={secret}.json",
    ],
)
def test_mcp_servers_reference_credentials_are_not_reported(tmp_path: Path, ref: str) -> None:
    """Regression: a non-http(s) mcpServers reference was quoted raw in findings, their metadata, and the
    broken-source row, so URL userinfo or a query credential reached the report."""
    secret = _password()
    collection = _collect(tmp_path, ref.format(secret=secret))
    assert collection.findings and collection.broken_sources
    for finding in collection.findings:
        assert secret not in finding.message
        assert secret not in json.dumps(finding.metadata)
        assert secret not in (finding.suggestion or "")
    assert secret not in repr(collection.broken_sources)
    # The broken-source row and its finding still name the same (redacted) reference.
    refs = {finding.metadata.get("plugin_component_ref") for finding in collection.findings}
    assert {raw for raw, _path, _problem in collection.broken_sources} <= refs


def test_plain_mcp_servers_path_is_reported_as_written(tmp_path: Path) -> None:
    collection = _collect(tmp_path, "./missing servers.json")
    assert collection.broken_sources == [("./missing servers.json", "missing servers.json", "missing")]
