# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR243 findings 3 and 8: loaded skill severity and stable identity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Severity
from skillevaluator.validators.plugin_schema import PluginSchemaValidator


def _write_json(root: Path, path: str, data: dict) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


def _skill(root: Path, path: str, name: str = "Wrong_Name") -> Path:
    target = root / path / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"---\nname: {name}\ndescription: Demonstrate a portable skill.\n"
        "metadata:\n  author: Example Author <author@example.com>\n---\n# Instructions\nUse this skill.\n",
        encoding="utf-8",
    )
    return target


def _plugin(root: Path, skills: list[str] | None) -> None:
    fields = {"skills": skills} if skills is not None else {}
    _write_json(root, "openclaw.plugin.json", {"id": "demo", "configSchema": {}, **fields})


@pytest.mark.parametrize("declared", [".", "./skills/alpha", "skills/alpha", "./skills", "skills"])
def test_finding3_evaluated_openclaw_member_keeps_blocking_severity(tmp_path: Path, declared: str) -> None:
    _plugin(tmp_path, [declared])
    _skill(tmp_path, "skills/alpha")
    result = PluginSchemaValidator().validate(tmp_path)
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    [member] = [r for r in rows if r["type"] == "skill" and r["support"] == "evaluated"]
    assert member["path"] == "skills/alpha"
    findings = [f for f in result.findings if f.check_name == "frontmatter_field"]
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH
    assert not findings[0].metadata.get("advisory")
    assert "does not load" not in findings[0].message
    assert not result.passed
    assert member["findings"] >= 1


@pytest.mark.parametrize("manifest", [".cursor-plugin/plugin.json", ".codex-plugin/plugin.json"])
def test_finding3_other_explicit_clients_keep_member_errors_blocking(tmp_path: Path, manifest: str) -> None:
    data = {"name": "demo", "skills": "./skills/alpha"}
    if "codex" in manifest:
        data["interface"] = {"displayName": "Demo", "shortDescription": "Demonstrate a plugin"}
    _write_json(tmp_path, manifest, data)
    _skill(tmp_path, "skills/alpha")
    result = PluginSchemaValidator().validate(tmp_path)
    [finding] = [f for f in result.findings if f.check_name == "frontmatter_field"]
    assert finding.severity == Severity.HIGH
    assert not finding.metadata.get("advisory")
    assert not result.passed


@pytest.mark.parametrize("skills", [None, [], ["portable"], ["./skills/alpha"]])
def test_finding3_undeclared_default_skills_remain_advisory(tmp_path: Path, skills: list[str] | None) -> None:
    _plugin(tmp_path, skills)
    _skill(tmp_path, "skills/ignored")
    _skill(tmp_path, "portable/alpha", "alpha")
    _skill(tmp_path, "skills/alpha", "alpha")
    result = PluginSchemaValidator().validate(tmp_path)
    [finding] = [f for f in result.findings if f.check_name == "frontmatter_field"]
    assert finding.severity == Severity.MEDIUM
    assert finding.metadata.get("advisory")
    assert result.passed, result.errors


def test_finding3_declared_terminal_skill_does_not_promote_nested_example(tmp_path: Path) -> None:
    _plugin(tmp_path, ["skills"])
    _skill(tmp_path, "skills/alpha", "alpha")
    _skill(tmp_path, "skills/alpha/templates/example")
    result = PluginSchemaValidator().validate(tmp_path)
    [finding] = [f for f in result.findings if f.check_name == "frontmatter_field"]
    assert finding.metadata.get("advisory")
    assert finding.severity == Severity.MEDIUM
    assert result.passed, result.errors


def test_finding3_inventory_member_at_skills_root_is_checked(tmp_path: Path) -> None:
    _plugin(tmp_path, ["skills"])
    _skill(tmp_path, "skills")
    result = PluginSchemaValidator().validate(tmp_path)
    [finding] = [f for f in result.findings if f.check_name == "frontmatter_field"]
    assert finding.severity == Severity.HIGH
    assert not finding.metadata.get("advisory")
    assert not result.passed


def test_finding3_tier1_cli_rejects_evaluated_member(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    _plugin(root, ["."])
    _skill(root, "skills/alpha")
    result = CliRunner().invoke(
        cli,
        ["validate", str(root), "--tiers", "1", "--checks", "schema", "--output-dir", str(tmp_path / "reports")],
    )
    assert result.exit_code == 1, result.output


@pytest.mark.parametrize("selected_openclaw", [False, True])
def test_finding8_identity_conflict_names_actual_fields_both_directions(
    tmp_path: Path, selected_openclaw: bool
) -> None:
    _write_json(tmp_path, ".claude-plugin/plugin.json", {"name": "release-helper"})
    manifest = _write_json(
        tmp_path,
        "openclaw.plugin.json",
        {"id": "release-helper-oc", "name": "release-helper", "configSchema": {}},
    )
    result = PluginSchemaValidator().validate(manifest if selected_openclaw else tmp_path)
    [finding] = [f for f in result.findings if f.check_name == "plugin_manifest_conflict"]
    assert finding.severity == Severity.MEDIUM
    assert finding.metadata["field"] == "id"
    assert finding.metadata["selected_field"] == ("id" if selected_openclaw else "name")
    assert finding.metadata["additional_field"] == ("name" if selected_openclaw else "id")
    assert "openclaw.plugin.json declares id 'release-helper-oc'" in finding.message
    assert ".claude-plugin/plugin.json declares name 'release-helper'" in finding.message
    assert "OpenClaw id" in finding.suggestion
    assert "name" in finding.suggestion
    declarations = result.metadata["plugin"]["manifest_declarations"]
    assert {row["manifest_filename"]: row["identity_field"] for row in declarations["manifests"]} == {
        "openclaw.plugin.json": "id",
        ".claude-plugin/plugin.json": "name",
    }
    [conflict] = declarations["conflicts"]
    assert conflict["field"] == "id"
    assert conflict["selected_field"] == finding.metadata["selected_field"]
    assert conflict["additional_field"] == finding.metadata["additional_field"]


@pytest.mark.parametrize("selected_openclaw", [False, True])
def test_finding8_display_name_does_not_affect_identity_and_version_stays_version(
    tmp_path: Path, selected_openclaw: bool
) -> None:
    _write_json(tmp_path, ".claude-plugin/plugin.json", {"name": "demo", "version": "1.0.0"})
    manifest = _write_json(
        tmp_path,
        "openclaw.plugin.json",
        {"id": "demo", "name": "Unrelated display label", "version": "2.0.0", "configSchema": {}},
    )
    result = PluginSchemaValidator().validate(manifest if selected_openclaw else tmp_path)
    [finding] = [f for f in result.findings if f.check_name == "plugin_manifest_conflict"]
    assert finding.metadata["field"] == "version"
    assert finding.suggestion == "Keep the plugin version identical in every manifest."


@pytest.mark.parametrize("declared", ["../outside", "/tmp/outside", "missing"])
def test_finding3_declared_unsafe_or_missing_paths_still_fail_closed(tmp_path: Path, declared: str) -> None:
    _plugin(tmp_path, [declared])
    result = PluginSchemaValidator().validate(tmp_path)
    assert not result.passed
    assert any(f.severity.is_error() and "path" in f.check_name for f in result.findings)


def test_finding3_linked_declared_skill_still_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    _plugin(root, ["skills/alpha"])
    outside = _skill(tmp_path, "outside", "alpha").parent
    (root / "skills").mkdir()
    (root / "skills/alpha").symlink_to(outside, target_is_directory=True)
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    assert any(f.severity.is_error() and "unsafe" in f.check_name for f in result.findings)


def test_finding3_member_manifest_replaced_after_discovery_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plugin(tmp_path, ["skills/alpha"])
    target = _skill(tmp_path, "skills/alpha", "alpha")
    report_unscanned = PluginSchemaValidator._report_unscanned_skills

    def replace_after_discovery(self, unscanned, root, result):
        replacement = target.with_name("replacement.md")
        replacement.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        replacement.replace(target)
        return report_unscanned(self, unscanned, root, result)

    monkeypatch.setattr(PluginSchemaValidator, "_report_unscanned_skills", replace_after_discovery)
    result = PluginSchemaValidator().validate(tmp_path)
    assert not result.passed
    assert result.metadata.get("security_failure")
    assert any(f.check_name == "bundled_skill_path_unsafe" and f.severity.is_error() for f in result.findings)
