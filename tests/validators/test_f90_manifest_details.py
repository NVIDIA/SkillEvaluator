# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check-1 details (proof L6): link names, YAML version text, policy-raised findings, the overlay label,
a named manifest that is not selected, and content errors in the selected manifest.

Each fixture is the proof's check-01 example of the same letter: e06, e07, e09, e10, e12, e13, e16, e19, e22.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Severity
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.plugin_sections import manifest_declarations_view
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.policy import ValidationPolicy

_SKIP_LINKS = pytest.mark.skipif(os.name == "nt", reason="POSIX link fixture")
_AP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
_CLI_ARGS = ["--type", "plugin", "--tiers", "1", "--no-llm", "-r", "json"]


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _checks(result) -> dict[str, Severity]:
    return {finding.check_name: finding.severity for finding in result.findings}


@_SKIP_LINKS
def test_symlink_inside_the_root_is_named_as_a_link_e10(tmp_path: Path) -> None:
    root = _write(tmp_path / "plugin", {"real-manifest.json": {"name": "demo"}})
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "plugin.json").symlink_to("../real-manifest.json")

    result = PluginSchemaValidator().validate(root)

    assert _checks(result) == {"manifest_linked": Severity.HIGH}
    assert "inside the plugin root" in result.findings[0].message
    assert result.metadata["security_failure"] is True  # still fails closed


@_SKIP_LINKS
def test_hard_link_is_named_as_one_e22(tmp_path: Path) -> None:
    root = _write(tmp_path / "plugin", {"shared.json": {"name": "demo"}})
    (root / ".claude-plugin").mkdir()
    os.link(root / "shared.json", root / ".claude-plugin" / "plugin.json")

    result = PluginSchemaValidator().validate(root)

    assert _checks(result) == {"manifest_hardlinked": Severity.HIGH}


@_SKIP_LINKS
def test_symlink_that_leaves_the_root_keeps_its_name_e09(tmp_path: Path) -> None:
    _write(tmp_path, {"outside-manifest.json": {"name": "demo"}})
    root = tmp_path / "plugin"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").symlink_to("../../outside-manifest.json")

    result = PluginSchemaValidator().validate(root)

    assert _checks(result) == {"manifest_outside_root": Severity.HIGH}


def test_yaml_numeric_version_keeps_its_text_e06(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "plugin",
        {
            "agent_plugin.yaml": (
                "name: demo-bundle\nversion: 1.10\nauthor:\n  email: dev@example.com\n"
                "skills:\n  refs:\n    - github::acme/tools::skills::lint\n"
            )
        },
    )
    result = PluginSchemaValidator().validate(root)

    rows = result.metadata["plugin"]["manifest_declarations"]["manifests"]
    assert rows[0]["version"] == "1.10"
    assert _checks(result)["schema:version:numeric"] == Severity.LOW
    assert result.passed


def test_yaml_numeric_version_in_an_additional_manifest_keeps_its_text(tmp_path: Path) -> None:
    # Verifier round 2: agent_plugin.yml read 1.10 as 1.1, so two identical files got a false MEDIUM conflict.
    text = "name: demo\nversion: 1.10\nauthor:\n  email: a@b.c\nmcp:\n  - name: x\n    provider: y\n"
    root = _write(tmp_path / "plugin", {"agent_plugin.yaml": text, "agent_plugin.yml": text})
    result = PluginSchemaValidator().validate(root)

    assert "plugin_manifest_conflict" not in _checks(result)
    rows = result.metadata["plugin"]["manifest_declarations"]["manifests"]
    assert [(row["manifest_filename"], row["version"]) for row in rows] == [
        ("agent_plugin.yaml", "1.10"),
        ("agent_plugin.yml", "1.10"),
    ]


def test_policy_raised_finding_removes_the_passed_row_e16(tmp_path: Path) -> None:
    # check-01 p07/e16: a Codex manifest with packaging warnings only, and a policy that raises one to HIGH.
    root = _write(tmp_path / "plugin", {".codex-plugin/plugin.json": {"name": "My_Plugin", "version": "v1"}})
    raised = ValidationPolicy(severity_overrides={"PLUGIN_SCHEMA.schema:interface:missing": Severity.HIGH})

    plain = PluginSchemaValidator().validate(root)
    with_policy = PluginSchemaValidator(policy=raised).validate(root)

    assert any(detail.check_name == "plugin_manifest" for detail in plain.success_details)
    assert not any(detail.check_name == "plugin_manifest" for detail in with_policy.success_details)


def test_policy_lowered_finding_restores_the_passed_row(tmp_path: Path) -> None:
    root = _write(tmp_path / "plugin", {".claude-plugin/plugin.json": {"name": "demo", "version": 1}})
    lowered = ValidationPolicy(severity_overrides={"PLUGIN_SCHEMA.schema:version:type": Severity.MEDIUM})

    result = PluginSchemaValidator(policy=lowered).validate(root)

    assert any(detail.check_name == "plugin_manifest" for detail in result.success_details)


def test_codex_overlay_row_is_labelled_e13(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "plugin",
        {
            "plugin.json": {"$schema": _AP_SCHEMA, "name": "demo-ap", "version": "1.0.0"},
            ".codex-plugin/plugin.json": {"interface": {"displayName": "Demo Codex"}},
        },
    )
    result = PluginSchemaValidator().validate(root)
    view = manifest_declarations_view(result.metadata["plugin"]["manifest_declarations"])

    assert view is not None
    overlay = view["rows"][1]
    assert overlay["overlay"] is True
    assert overlay["status"] == "parsed, Codex overlay"
    assert overlay["name"] == "from the root manifest"
    lines: list[str] = []
    MarkdownReporter._render_manifest_declarations(view, lines)
    assert any("parsed, Codex overlay" in line for line in lines)


def test_named_manifest_that_is_not_selected_is_reported_e12(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "alpha", "version": "1.0.0"},
            ".codex-plugin/plugin.json": {"name": "beta", "version": "2.0.0"},
        },
    )
    runner = CliRunner()
    named = runner.invoke(
        cli,
        [
            "validate",
            str(root / ".codex-plugin" / "plugin.json"),
            *_CLI_ARGS,
            "--checks",
            "schema",
            "-o",
            str(tmp_path / "a"),
        ],
    )
    runner.invoke(
        cli,
        [
            "validate",
            str(root / ".claude-plugin" / "plugin.json"),
            *_CLI_ARGS,
            "--checks",
            "schema",
            "-o",
            str(tmp_path / "b"),
        ],
    )

    def findings(out: str) -> dict[str, dict]:
        report = json.loads(next((tmp_path / out).glob("*.json")).read_text(encoding="utf-8"))
        return {f["check_name"]: f for result in report["results"] for f in result["findings"]}

    assert named.exit_code == 0, named.output
    note = findings("a")["manifest_named_not_selected"]
    assert note["severity"] == "info"
    assert ".codex-plugin/plugin.json" in note["message"]
    assert ".claude-plugin/plugin.json" in note["message"]
    assert "manifest_named_not_selected" not in findings("b")


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        # e19: a Latin-1 byte in the selected manifest.
        (b'{"name": "demo", "description": "caf\xe9"}\n', "encoding"),
        # e07: valid JSON padded past the 1 MiB read bound.
        (
            json.dumps({"name": "demo", "metadata": {f"p{index}": "x" * 60_000 for index in range(20)}}).encode(),
            "size_limit",
        ),
    ],
    ids=["e19-latin1", "e07-oversize"],
)
def test_content_error_in_the_selected_manifest_does_not_stop_the_run(
    tmp_path: Path, content: bytes, reason: str
) -> None:
    root = _write(tmp_path / "plugin", {".claude-plugin/plugin.json": content})

    results = run_validation(root, checks="schema,pii", content_type=CONTENT_TYPE_PLUGIN)

    assert [result.validator_name for result in results] == ["Plugin Schema & Bundle References", "PII Scan"]
    schema = results[0]
    assert "security_failure" not in schema.metadata
    finding = next(f for f in schema.findings if f.check_name == "manifest_unreadable")
    assert finding.severity == Severity.HIGH
    assert finding.metadata["reason"] == reason
    assert "read leniently" in finding.message
    # The selected manifest's fields are validated as well as its components.
    assert "its fields and the components it declares are checked below" in finding.message
    assert schema.metadata["plugin"]["name"] == "demo"
    assert not schema.passed


def test_content_error_in_the_selected_manifest_still_runs_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Claude:
        is_available = True

        def get_install_hint(self) -> str:
            return ""

        def run(self, _args: list[str], **_kwargs: Any) -> ToolResult:
            payload = {"success": True, "manifest": {"file": "plugin.json", "errors": [], "warnings": []}}
            return ToolResult(True, json.dumps(payload), "", 0)

    monkeypatch.setattr(Tools, "claude", Claude())
    root = _write(tmp_path / "plugin", {".claude-plugin/plugin.json": b'{"name": "demo", "description": "caf\xe9"}'})

    results = run_validation(root, checks="schema,claude-validate", content_type=CONTENT_TYPE_PLUGIN)

    assert results[-1].validator_name == "Claude Plugin Validate Parity"
    assert results[-1].metadata["plugin"]["validator_parity"]["status"] == "compared"


def test_one_defect_in_an_overlay_gets_one_finding_e14(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "plugin",
        {
            "plugin.json": {"$schema": _AP_SCHEMA, "name": "demo-ap", "version": "1.0.0"},
            ".codex-plugin/plugin.json": {"interface": {"displayName": "Demo Codex"}, "apps": 5},
        },
    )
    result = PluginSchemaValidator().validate(root)

    # Before: MEDIUM plugin_manifest_additional_invalid and HIGH plugin_component_path_invalid for the same value.
    assert _checks(result) == {"plugin_component_path_invalid": Severity.HIGH}
    rows = result.metadata["plugin"]["manifest_declarations"]["manifests"]
    assert rows[1]["status"] == "invalid"


def test_additional_manifest_type_errors_the_inventory_does_not_check_stay_reported(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "plugin",
        {
            "agent_plugin.yaml": (
                "name: demo\nauthor:\n  email: dev@example.com\nmcp:\n  - name: tracker\n    provider: example-provider\n"
            ),
            ".claude-plugin/plugin.json": {"name": "demo", "settings": 5},
        },
    )
    result = PluginSchemaValidator().validate(root)

    assert _checks(result)["plugin_manifest_additional_invalid"] == Severity.MEDIUM
