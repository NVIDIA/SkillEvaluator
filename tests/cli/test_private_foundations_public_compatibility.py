# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public plugin format detection, report output, and policy compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.cli_core import detect_content_type
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.policy import load_policy_file

_SUPPORTED_MANIFESTS = (
    (".codex-plugin/plugin.json", "codex_plugin_json"),
    (".cursor-plugin/plugin.json", "cursor_plugin_json"),
    ("plugin.json", "agent_plugins_v1"),
)


def _write_manifest(path: Path, manifest_type: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    declaration = {"name": "example"}
    if manifest_type == "agent_plugins_v1":
        declaration["$schema"] = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")


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


@pytest.mark.parametrize(("relative", "manifest_type"), _SUPPORTED_MANIFESTS)
def test_supported_manifests_take_precedence_over_bundled_skills(
    tmp_path: Path, relative: str, manifest_type: str
) -> None:
    manifest = tmp_path / relative
    _write_manifest(manifest, manifest_type)

    skill = tmp_path / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: Example\n---\n", encoding="utf-8")

    assert detect_content_type(tmp_path) == "plugin"
    assert detect_content_type(manifest) == "plugin"
    schema = run_validation(manifest, checks="schema", content_type="plugin")[0]
    assert schema.metadata["manifest_type"] == manifest_type


@pytest.mark.parametrize(("relative", "manifest_type"), (*_SUPPORTED_MANIFESTS, ("plugin.json", None)))
def test_forced_plugin_type_emits_all_reports_for_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str, manifest_type: str | None
) -> None:
    root = tmp_path / "input"
    manifest = root / relative
    _write_manifest(manifest, manifest_type)
    if manifest_type is None:
        assert detect_content_type(root) == detect_content_type(manifest) == "unknown"
    monkeypatch.setenv("SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE", str(tmp_path / "state" / "key"))
    reports = tmp_path / "reports"
    args = ["validate", str(root), "--type", "plugin", "--tiers", "1", "--checks", "schema", "--no-llm"]
    args.extend(["--no-autopilot", "-o", str(reports)])
    for report_format in ("json", "html", "markdown", "sarif"):
        args.extend(["-r", report_format])

    result = CliRunner().invoke(cli, args)

    supported = manifest_type is not None
    assert result.exit_code == (0 if supported else 1), result.output
    files = list(reports.iterdir())
    assert len(files) == 4
    assert len(list(reports.glob("*.html"))) == 1
    assert len(list(reports.glob("*.md"))) == 1
    payload = json.loads(
        next(path for path in files if path.suffix == ".json" and ".sarif." not in path.name).read_text()
    )
    assert payload["overall_passed"] is supported
    sarif = json.loads(next(reports.glob("*.sarif.json")).read_text())
    assert len(sarif["runs"]) == 1
    if supported:
        assert payload["plugin_static"]["name"] == "example"
        assert payload["total_errors"] == 0
    else:
        assert "plugin_static" not in payload
        assert payload["total_errors"] == 1
        assert payload["results"][0]["findings"][0]["check_name"] == "manifest_missing_or_unsafe"
        (finding,) = sarif["runs"][0]["results"]
        assert finding["ruleId"].endswith("/manifest_missing_or_unsafe")


def test_plugin_policy_keys_are_explicit_and_redacted_in_the_public_summary(tmp_path: Path) -> None:
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
    assert actual["digest"] != expected["digest"]
    assert actual.pop("mcp_allowed_private_host_count") == 1
    assert actual.pop("hook_allowed_url_count") == 1
    assert actual.pop("resolve_endpoints") is True
    actual.pop("digest")
    expected.pop("digest")
    assert actual == expected
    assert "10.0.0.1" not in json.dumps(actual) and "example.com" not in json.dumps(actual)
