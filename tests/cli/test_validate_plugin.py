# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 dispatch tests for the plugin content type (run_validation)."""

from pathlib import Path

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting.console_ui import final_check_ticker_row
from skillevaluator.tier1.commands import enabled_check_lineup, run_validation
from skillevaluator.validators.policy import ValidationPolicy

_VALID_MANIFEST = """
name: my-bundle
author:
  email: dev@example.com
skills:
  refs:
    - "github::example-org/example-repo::skills::build-infra"
"""


def _make_plugin(tmp_path: Path) -> Path:
    (tmp_path / "agent_plugin.yaml").write_text(_VALID_MANIFEST)
    return tmp_path


def test_run_validation_plugin_reports_bounded_manifest_inventory(tmp_path: Path):
    results = run_validation(_make_plugin(tmp_path), checks="schema", content_type=CONTENT_TYPE_PLUGIN)
    schema = results[0]
    assert schema.validator_name == "Plugin Schema & Component Checks"
    assert schema.metadata["plugin"]["name"] == "my-bundle"
    assert schema.metadata["plugin"]["component_inventory"]["components"]
    assert all(r.passed for r in results)


def test_run_validation_reports_mandatory_schema_progress_for_narrowed_plugin_checks(tmp_path: Path):
    checks = "unicode"
    started = []
    run_validation(_make_plugin(tmp_path), checks=checks, content_type=CONTENT_TYPE_PLUGIN, on_check=started.append)

    assert started == ["schema", "unicode"]
    assert enabled_check_lineup(checks, content_type=CONTENT_TYPE_PLUGIN) == started
    assert enabled_check_lineup(checks) == ["unicode"]


def test_optional_plugin_checks_report_progress_and_outcomes(tmp_path, monkeypatch):
    from skillevaluator.tier1 import plugin_checks

    def fake_endpoint_check(_self, _targets):
        return {"incomplete": True}, []

    def fake_parity_check(_self, *_args, **_kwargs):
        return ValidationResult(validator_name="Claude Plugin Validate Parity")

    monkeypatch.setattr(plugin_checks.EndpointChecker, "check", fake_endpoint_check)
    monkeypatch.setattr(plugin_checks.ClaudePluginValidateParity, "validate", fake_parity_check)
    started = []
    results = run_validation(
        _make_plugin(tmp_path),
        checks="schema",
        content_type=CONTENT_TYPE_PLUGIN,
        policy=ValidationPolicy(resolve_endpoints=True),
        claude_validate=True,
        on_check=started.append,
    )

    assert started == ["schema", "endpoint-resolution", "claude-validate"]
    lineup = enabled_check_lineup(
        "schema", content_type=CONTENT_TYPE_PLUGIN, resolve_endpoints=True, claude_validate=True
    )
    assert lineup == started
    row = final_check_ticker_row(lineup, results)
    assert "".join(text for text, _style in row.segments) == "✓ schema · ✗ endpoint-resolution · ✓ claude-validate"


def test_run_validation_does_not_run_skill_checks_for_references(tmp_path: Path):
    results = run_validation(
        _make_plugin(tmp_path),
        checks="schema,quality,lint",
        content_type=CONTENT_TYPE_PLUGIN,
    )
    names = [r.validator_name or "" for r in results]
    assert any("Plugin Schema" in n for n in names)
    assert not any("Quality" in n for n in names)
    assert not any("Lint" in n or "Script" in n for n in names)


def test_run_validation_runs_quality_and_lint_for_bundled_skills(tmp_path: Path):
    manifest = tmp_path / ".claude-plugin/plugin.json"
    manifest.parent.mkdir()
    manifest.write_text('{"name": "my-bundle"}', encoding="utf-8")
    member = tmp_path / "skills/demo/SKILL.md"
    member.parent.mkdir(parents=True)
    member.write_text(
        "---\nname: demo\ndescription: Use when checking project workflows.\n---\n"
        "# Demo\n\n## Instructions\n\nRead the project and explain its workflow.\n",
        encoding="utf-8",
    )

    results = run_validation(manifest, checks="quality,lint", content_type=CONTENT_TYPE_PLUGIN)

    performed = {
        result.metadata["plugin_check"] for result in results if result.metadata.get("plugin_member") == "skills/demo"
    }
    assert performed == {"quality", "lint"}


def test_run_validation_plugin_does_not_require_catalog_path(tmp_path: Path):
    """Plugin validation must succeed without any catalog (no dedup dependency)."""
    results = run_validation(_make_plugin(tmp_path), checks="schema", content_type=CONTENT_TYPE_PLUGIN)
    assert results
    assert all(r.passed for r in results)
