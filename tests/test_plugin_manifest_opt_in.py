# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root manifest opt-in remains visible when client JSON exceeds parser bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.plugin_discovery import discover_plugin_members
from skillevaluator.plugin_manifest import agent_plugins_opt_in, locate_plugin_manifest

SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"])
@pytest.mark.parametrize("escape_all", [False, True])
def test_overcomplex_unicode_escaped_schema_remains_an_explicit_manifest_error(
    tmp_path: Path, encoding: str, escape_all: bool
) -> None:
    text = json.dumps({"$schema": SCHEMA, "name": "demo", "description": "x" * 65_537})
    escaped_host = "".join(f"\\u{ord(character):04X}" for character in "agent-plugins.org")
    if not escape_all:
        escaped_host = r"\u0061gent-plugins.org"
    text = text.replace("agent-plugins.org", escaped_host).replace('"$schema"', r'"\u0024schema"')
    raw = text.encode(encoding)
    assert json.loads(raw)["$schema"] == SCHEMA
    (tmp_path / "plugin.json").write_bytes(raw)

    assert agent_plugins_opt_in(raw)
    located = locate_plugin_manifest(tmp_path)
    assert located is not None
    assert located.manifest_filename == "plugin.json"
    result = discover_plugin_members(tmp_path)
    assert result is not None
    assert not result.complete
    assert any(finding.check_name == "plugin_manifest_unreadable" for finding in result.findings)


@pytest.mark.parametrize("description", ["Legacy plugin.", "x" * 65_537])
def test_unrelated_root_plugin_json_remains_ignored(tmp_path: Path, description: str) -> None:
    raw = json.dumps({"name": "legacy", "description": description}).encode()
    (tmp_path / "plugin.json").write_bytes(raw)

    assert not agent_plugins_opt_in(raw)
    assert locate_plugin_manifest(tmp_path) is None
    assert discover_plugin_members(tmp_path) is None
