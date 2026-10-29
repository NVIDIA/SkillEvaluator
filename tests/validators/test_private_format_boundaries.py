# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent compatibility and evidence-boundary tests; no real CLI runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.constants import (
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CURSOR_MANIFEST_TYPE,
)
from skillevaluator.plugin_format_schema import validate_manifest_fields
from skillevaluator.utils.tool_runner import ToolResult
from skillevaluator.validators.claude_plugin_validate import ClaudePluginValidateParity, _child_env


class FakeClaude:
    def __init__(self, reports=(), *, available=True):
        self.is_available = available
        self.reports = list(reports)
        self.calls = []

    def get_install_hint(self):
        return "fixture CLI unavailable"

    def run(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return self.reports.pop(0)


def root(tmp_path: Path) -> Path:
    manifest = tmp_path / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir()
    manifest.write_text('{"name":"fixture"}')
    return tmp_path


def report(*, success=True, errors=None, warnings=None):
    return {
        "success": success,
        "manifest": {
            "file": ".claude-plugin/plugin.json",
            "errors": errors if errors is not None else [],
            "warnings": warnings if warnings is not None else [],
        },
        "contents": [],
    }


@pytest.mark.parametrize("exit_code", [2, -1, True])
def test_json_verdict_requires_a_completed_supported_process_state(tmp_path, exit_code):
    tool = FakeClaude([ToolResult(True, json.dumps(report()), "", exit_code)])
    result = ClaudePluginValidateParity(tool).validate(root(tmp_path), skillevaluator_verdict="passed")
    assert result.is_incomplete
    assert result.metadata["plugin"]["validator_parity"]["status"] == "error"


def test_json_cannot_claim_success_when_manifest_has_errors(tmp_path):
    tool = FakeClaude([ToolResult(True, json.dumps(report(errors=[{"path": "name", "message": "invalid"}])), "", 0)])
    result = ClaudePluginValidateParity(tool).validate(root(tmp_path), skillevaluator_verdict="passed")
    assert result.is_incomplete


@pytest.mark.parametrize("errors", [42, {"message": "invalid"}, "invalid"])
def test_malformed_json_sections_become_incomplete_without_escaping(tmp_path, errors):
    tool = FakeClaude([ToolResult(True, json.dumps(report(errors=errors)), "", 0)])
    result = ClaudePluginValidateParity(tool).validate(root(tmp_path))
    assert result.is_incomplete


def test_missing_optional_cli_is_explicitly_skipped_with_reason(tmp_path):
    tool = FakeClaude(available=False)
    result = ClaudePluginValidateParity(tool).validate(root(tmp_path))
    parity = result.metadata["plugin"]["validator_parity"]
    assert parity["status"] == "skipped" and parity["reason"]
    assert not result.is_incomplete
    assert not tool.calls


def test_nonclaude_manifest_is_not_applicable_without_invoking_cli(tmp_path):
    manifest = tmp_path / ".cursor-plugin" / "plugin.json"
    manifest.parent.mkdir()
    manifest.write_text('{"name":"fixture"}')
    tool = FakeClaude()
    result = ClaudePluginValidateParity(tool).validate(tmp_path)
    assert not result.is_incomplete
    assert result.metadata["plugin"]["validator_parity"]["status"] == "not_applicable"
    assert not tool.calls


def test_old_cli_json_fallback_is_bounded_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "never-forward")
    tool = FakeClaude(
        [
            ToolResult(True, "", "error: unknown option '--json'", 1),
            ToolResult(True, "Validating plugin manifest: .claude-plugin/plugin.json\nValidation passed\n", "", 0),
        ]
    )
    result = ClaudePluginValidateParity(tool).validate(root(tmp_path), skillevaluator_verdict="passed")
    assert not result.is_incomplete
    assert len(tool.calls) == 2
    assert "--json" in tool.calls[0][0] and "--json" not in tool.calls[1][0]
    for _, kwargs in tool.calls:
        assert kwargs["replace_env"] is True
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["CLAUDE_CONFIG_DIR"].startswith(kwargs["env"]["HOME"])
        assert kwargs["cwd"] != tmp_path


def test_warning_only_report_preserves_pass_and_records_strict_failure(tmp_path):
    tool = FakeClaude(
        [ToolResult(True, json.dumps(report(warnings=[{"path": "version", "message": "recommended"}])), "", 0)]
    )
    result = ClaudePluginValidateParity(tool).validate(root(tmp_path), skillevaluator_verdict="passed")
    parity = result.metadata["plugin"]["validator_parity"]
    assert not result.is_incomplete
    assert parity["claude_verdict"] == "passed" and parity["strict_verdict"] == "failed"
    assert parity["agree"] is True


def test_cli_environment_drops_provider_cloud_and_proxy_auth(tmp_path):
    env = _child_env(
        tmp_path,
        environ={
            "PATH": "/bin",
            "HOME": "/real-home",
            "ANTHROPIC_API_KEY": "provider-secret",
            "AWS_SESSION_TOKEN": "cloud-secret",
            "HTTPS_PROXY": "https://user:proxy-password@proxy.example:8080",
            "DOCKER_AUTH_CONFIG": "registry-secret",
        },
    )
    assert env["HOME"] == str(tmp_path)
    assert not {"ANTHROPIC_API_KEY", "AWS_SESSION_TOKEN", "DOCKER_AUTH_CONFIG"} & env.keys()
    assert "proxy-password" not in str(env)


def test_agent_plugins_bad_version_remains_advisory_while_schema_version_blocks():
    data = {
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        "name": "fixture",
        "version": "release-name",
    }
    issues = validate_manifest_fields(PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE, data)
    assert not any(issue.level == "error" and issue.field == "version" for issue in issues)
    data["$schema"] = "https://agent-plugins.org/schemas/2.0.0/plugin.schema.json"
    assert any(
        issue.level == "error" and issue.field == "$schema"
        for issue in validate_manifest_fields(PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE, data)
    )


@pytest.mark.parametrize(
    "manifest_type", [PLUGIN_CONTAINED_MANIFEST_TYPE, PLUGIN_CODEX_MANIFEST_TYPE, PLUGIN_CURSOR_MANIFEST_TYPE]
)
def test_other_client_names_cannot_change_agent_plugins_name_rules(manifest_type):
    data = {"name": "has--repetition", "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"}
    # Agent Plugins' consecutive-hyphen rule is independent of each client's own name rules.
    assert any(
        issue.field == "name" and issue.level == "error"
        for issue in validate_manifest_fields(PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE, data)
    )
    assert isinstance(validate_manifest_fields(manifest_type, data), list)


def test_codex_overlay_checks_only_fields_the_overlay_can_apply():
    issues = validate_manifest_fields(
        PLUGIN_CODEX_MANIFEST_TYPE, {"name": 42, "version": 42, "interface": []}, overlay=True
    )
    assert any(issue.field == "interface" and issue.level == "error" for issue in issues)
    assert not any(issue.field in {"name", "version"} and issue.level == "error" for issue in issues)


def test_extension_validation_bound_is_explicit_instead_of_silently_accepting_suffix():
    data = {
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        "name": "fixture",
        "extensions": {**{f"com.fixture.valid{index}": {} for index in range(64)}, "com.fixture.hidden": []},
    }
    issues = validate_manifest_fields(PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE, data)
    assert any(
        issue.field.startswith("extensions")
        and (
            issue.level == "error"
            or "bound" in issue.message
            or "limit" in issue.message
            or "not checked" in issue.message
        )
        for issue in issues
    )
