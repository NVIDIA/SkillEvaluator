# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H8 at Tier 3: a Codex plugin's ``${VAR:-default}`` MCP URL is refused like Tier 1 reports it.

Claude Code expands ``${VAR:-default}`` in a plugin MCP URL; Codex 0.142.5 does
not (audit re-test: "relative URL without a base"). The MCP static package
reads the URL with the loading client's rules at Tier 1, so a Codex manifest
gets a HIGH ``mcp_url_env_not_expanded``. Tier 3 staging read every declared
server with Claude Code's rules: before the merge it staged that server for a
Codex plugin, and after it only the inventory's findings refused the run, with
a message about a server that is "not staged".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

_SERVER = {"api": {"type": "http", "url": "${API_BASE_URL:-https://api.example.com}/mcp"}}


def _plugin(root: Path, manifest_dir: str) -> Path:
    (root / manifest_dir).mkdir(parents=True)
    manifest = {"name": "demo", "version": "1.0.0", "description": "demo", "mcpServers": _SERVER}
    (root / manifest_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "evals").mkdir()
    (root / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p", "expected_output": "o"}]))
    return root


def test_codex_plugin_with_an_unexpanded_url_default_is_refused(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "codex", ".codex-plugin")

    with pytest.raises(ValueError, match="mcp_url_env_not_expanded") as refused:
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")

    # The staged server itself is refused, with the Codex rule; not as an unstaged server another client loads.
    assert str(refused.value).startswith("Plugin manifest MCP server 'api' failed blocking static validation")
    assert "not staged" not in str(refused.value)


def test_claude_plugin_with_the_same_url_default_still_stages(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "claude", ".claude-plugin")

    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")

    assert package.runnable_mcp_servers == ("api",)
