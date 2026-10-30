# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from skillevaluator.plugin_formats import (
    PROFILES,
    agent_plugins_schema_version,
    declared_value_replaces_default,
    declares_agent_plugins_schema,
    parse_manifest_text,
    profile_for,
)
from skillevaluator.utils.structured_data import StructuredDataError


@pytest.mark.parametrize("kind", ["claude_plugin_json", "codex_plugin_json", "cursor_plugin_json", "agent_plugins_v1"])
def test_each_private_profile_parses_real_json(kind):
    assert profile_for(kind).manifest_type == kind
    assert parse_manifest_text(kind, '{"name":"demo","skills":["./skills"]}') == {
        "name": "demo",
        "skills": ["./skills"],
    }
    with pytest.raises(StructuredDataError):
        parse_manifest_text(kind, '{"name":NaN}')


def test_yaml_and_unknown_profile_compatibility():
    assert parse_manifest_text("agent_bundle_yaml", "name: demo\nskills: {refs: []}") == {
        "name": "demo",
        "skills": {"refs": []},
    }
    assert profile_for(None) is PROFILES["claude_plugin_json"]
    assert profile_for("unsupported") is PROFILES["claude_plugin_json"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("./custom", True),
        ("custom", False),
        ("../custom", False),
        (["bad", "./custom"], True),
        ([], False),
        ({"x": 1}, False),
    ],
)
def test_codex_skill_replacement_is_explicit(value, expected):
    assert declared_value_replaces_default(profile_for("codex_plugin_json"), "skills", value) is expected


def test_root_plugin_json_opt_in_is_not_general_file_detection():
    assert not declares_agent_plugins_schema({"name": "legacy-root-plugin"})
    assert declares_agent_plugins_schema({"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"})


@pytest.mark.parametrize("kind", ["plugin", "mcp"])
def test_agent_plugins_schema_version_requires_the_exact_url(kind):
    url = f"https://agent-plugins.org/schemas/1.0.0/{kind}.schema.json"
    assert agent_plugins_schema_version(url, kind=kind) == "1.0.0"
    for padded in (url + "\n", url + " ", url + "\r\n", " " + url):
        assert agent_plugins_schema_version(padded, kind=kind) is None
