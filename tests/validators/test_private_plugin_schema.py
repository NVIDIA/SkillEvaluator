# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.constants import (
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE as AP,
)
from skillevaluator.constants import (
    PLUGIN_CODEX_MANIFEST_TYPE as CODEX,
)
from skillevaluator.constants import (
    PLUGIN_CONTAINED_MANIFEST_TYPE as CLAUDE,
)
from skillevaluator.constants import (
    PLUGIN_CURSOR_MANIFEST_TYPE as CURSOR,
)
from skillevaluator.plugin_format_schema import validate_manifest_fields
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.validators.private_plugin_schema import validate_inventory_schema


@pytest.mark.parametrize(
    "kind,field,value",
    [
        (CLAUDE, "name", 12),
        (CLAUDE, "commands", [12]),
        (CLAUDE, "author", {"name": 12}),
        (CLAUDE, "lspServers", {"x": {"command": 12}}),
        (CLAUDE, "monitors", [{"name": "x", "command": 12}]),
        (CODEX, "name", "bad name"),
        (CODEX, "version", {}),
        (CURSOR, "name", "Upper Name"),
        (CURSOR, "skills", [12]),
        (AP, "$schema", "https://agent-plugins.org/schemas/9.0.0/plugin.schema.json"),
        (AP, "name", "bad--name"),
    ],
)
def test_private_format_field_rules(kind, field, value):
    data = {
        "name": "demo",
        "version": "1.0.0",
        "description": "Demo",
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
    }
    data[field] = value
    issues = validate_manifest_fields(kind, data)
    assert any(i.field.split(".")[0] == field and i.level == "error" for i in issues)


def test_valid_claude_name_remains_accepted():
    assert not [i for i in validate_manifest_fields(CLAUDE, {"name": "Demo_name"}) if i.level == "error"]


def test_unknown_fields_are_bounded():
    issues = validate_manifest_fields(CURSOR, {"name": "demo", **{f"unknown{i}": i for i in range(200)}})
    assert len(issues) < 45
    assert any("unknown" in i.error for i in issues)


def test_additional_manifest_gets_schema_checks(tmp_path: Path):
    for rel, data in [(".claude-plugin/plugin.json", {"name": "demo"}), (".cursor-plugin/plugin.json", {"name": 12})]:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data))
    inventory = plugin_inventory_for_root(tmp_path)
    assert inventory is not None
    result = validate_inventory_schema(inventory)
    assert not result.passed
    assert any(f.metadata.get("manifest_type") == CURSOR and f.metadata.get("field") == "name" for f in result.findings)


def test_bundle_manifest_uses_existing_model(tmp_path: Path):
    (tmp_path / "agent_plugin.yaml").write_text("name: demo\nauthor: {email: demo@example.com}\n")
    inventory = plugin_inventory_for_root(tmp_path)
    assert inventory is not None
    result = validate_inventory_schema(inventory)
    assert not result.passed
    assert any(f.check_name.startswith("schema:") for f in result.findings)
