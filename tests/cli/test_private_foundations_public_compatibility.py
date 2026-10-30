# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private format/policy foundations must preserve the installed public workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.cli_core import detect_content_type
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.policy import load_policy_file


def test_existing_yaml_duplicate_key_acceptance_is_preserved(tmp_path: Path) -> None:
    (tmp_path / "agent_plugin.yaml").write_text(
        "name: previous-name\nname: current-name\ndescription: Example bundle\n"
        "version: 1.0\nauthor:\n  email: author@example.com\nmcp:\n  - name: example\n    provider: stdio\n",
        encoding="utf-8",
    )

    result = PluginSchemaValidator().validate(tmp_path)

    assert result.passed
    assert result.metadata["plugin"]["name"] == "current-name"


def test_existing_contained_validation_does_not_enable_private_policy(tmp_path: Path) -> None:
    manifest = tmp_path / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "name": "example",
                "mcpServers": {"example": {"command": "npx", "args": ["example-package@latest"]}},
                "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "echo example"}]}]},
            }
        ),
        encoding="utf-8",
    )

    result = PluginSchemaValidator().validate(tmp_path)

    assert result.passed
    assert result.findings == []
    assert result.metadata["manifest_type"] == "claude_plugin_json"


@pytest.mark.parametrize("relative", [".codex-plugin/plugin.json", ".cursor-plugin/plugin.json", "plugin.json"])
def test_private_formats_do_not_change_detection_or_forced_validation(tmp_path: Path, relative: str) -> None:
    manifest = tmp_path / relative
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '{"name":"example","$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"}',
        encoding="utf-8",
    )

    assert detect_content_type(tmp_path) == "unknown"
    assert detect_content_type(manifest) == "unknown"
    result = PluginSchemaValidator().validate(tmp_path)
    assert not result.passed
    assert [finding.check_name for finding in result.findings] == ["manifest_missing"]

    skill = tmp_path / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: Example\n---\n", encoding="utf-8")
    assert detect_content_type(tmp_path) == "skill"


@pytest.mark.parametrize("relative", [".codex-plugin/plugin.json", ".cursor-plugin/plugin.json", "plugin.json"])
def test_forced_private_format_cli_preserves_failure_and_all_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    root = tmp_path / "input"
    manifest = root / relative
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"name":"example"}', encoding="utf-8")
    monkeypatch.setenv("SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE", str(tmp_path / "state" / "key"))
    reports = tmp_path / "reports"
    args = ["validate", str(root), "--type", "plugin", "--tiers", "1", "--checks", "schema", "--no-llm"]
    args.extend(["--no-autopilot", "-o", str(reports)])
    for report_format in ("json", "html", "markdown", "sarif"):
        args.extend(["-r", report_format])

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1, result.output
    files = list(reports.iterdir())
    assert len(files) == 4
    assert len(list(reports.glob("*.html"))) == 1
    assert len(list(reports.glob("*.md"))) == 1
    payload = json.loads(
        next(path for path in files if path.suffix == ".json" and ".sarif." not in path.name).read_text()
    )
    assert payload["overall_passed"] is False
    assert payload["total_errors"] == 1
    assert payload["results"][0]["findings"][0]["check_name"] == "manifest_missing"
    sarif = json.loads(next(reports.glob("*.sarif.json")).read_text())
    assert len(sarif["runs"][0]["results"]) == 1
    assert sarif["runs"][0]["results"][0]["ruleId"].endswith("/manifest_missing")


def test_private_policy_keys_do_not_change_public_policy(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("{}\n", encoding="utf-8")
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        "mcp:\n  allowed_private_hosts: [10.0.0.1]\n"
        "hooks:\n  allowed_urls: [https://example.com/]\nendpoints:\n  resolve: true\n",
        encoding="utf-8",
    )

    expected = load_policy_file(baseline).to_dict()
    actual = load_policy_file(candidate).to_dict()
    expected.pop("source")
    actual.pop("source")
    assert actual == expected
