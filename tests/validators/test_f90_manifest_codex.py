# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codex manifest severities match what Codex 0.142.5 does (proof H17, M38).

The probes are the skeptic's check-1 Codex manifests, re-run against the real
client after the audit (``audit-retests/check-01-client-oracles``): ``codex
plugin add`` refuses the first group ("missing or invalid plugin.json") and
installs the second. Each probe is the complete n02 manifest with one field
changed. The round-2 probes (interface images, screenshots, and explicit nulls)
were run the same way against Codex 0.142.5 with a one-plugin local marketplace.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Severity
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

# check-01 n02-codex-complete: Codex installs it.
N02 = {
    "name": "demo-codex",
    "version": "1.0.0",
    "description": "Demo Codex plugin",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "Demo Codex",
        "shortDescription": "A demo plugin",
        "longDescription": "A demo plugin used to test manifest validation.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}


def _probe(**changes: object) -> dict:
    manifest = copy.deepcopy(N02)
    for key, value in changes.items():
        if key.startswith("interface__"):
            manifest["interface"][key.removeprefix("interface__")] = value
        else:
            manifest[key] = value
    return manifest


def _validate(root: Path, manifest: dict):
    path = root / ".codex-plugin" / "plugin.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return PluginSchemaValidator().validate(root)


def _manifest_findings(result) -> list:
    """Check 1's own findings (the component path and MCP checks report separately)."""
    return [
        finding
        for finding in result.findings
        if finding.check_name.startswith("schema:") or finding.check_name.startswith("plugin_manifest")
    ]


# Codex refuses these; PR #28 passed the first five with exit 0 (H17).
REFUSED = {
    "cx-caps-str": _probe(interface__capabilities="Interactive"),
    "cx-if-category-num": _probe(interface__category=5),
    "cx-if-displayname-num": _probe(interface__displayName=5),
    "cx-if-shortdesc-list": _probe(interface__shortDescription=["a"]),
    "cx-if-websiteurl-num": _probe(interface__websiteURL=5),
    "cx-apps-num": _probe(apps=5),
    "cx-apps-list": _probe(apps=["./a.app.json"]),
    "cx-description-num": _probe(description=5),
    "cx-interface-str": _probe(interface="x"),
    "cx-keywords-str": _probe(keywords="x"),
    "cx-version-num": _probe(version=123),
    # Codex refuses these too; the first H17 fix still passed them with exit 0 (verifier round 2).
    "cx-if-logo-num": _probe(interface__logo=5),
    "cx-if-logodark-num": _probe(interface__logoDark=5),
    "cx-if-composericon-num": _probe(interface__composerIcon=5),
    "cx-if-logo-list": _probe(interface__logo=["./a.png"]),
    "cx-if-screenshots-str": _probe(interface__screenshots="x"),
    "cx-if-screenshots-numlist": _probe(interface__screenshots=[5]),
    "cx-if-screenshots-obj": _probe(interface__screenshots={"a": "b"}),
    "cx-if-screenshots-null": _probe(interface__screenshots=None),
    "cx-caps-null": _probe(interface__capabilities=None),
    "cx-caps-nulllist": _probe(interface__capabilities=[None]),
    "cx-keywords-null": _probe(keywords=None),
}
# Codex installs these; PR #28 failed them HIGH (M38).
INSTALLED = {
    "cx-homepage-num": _probe(homepage=42),
    "cx-extensions-str": _probe(extensions="x"),
    "cx-hooks-num": _probe(hooks=5),
    "cx-mcp-num": _probe(mcpServers=5),
    "cx-skills-num": _probe(skills=5),
    "cx-name-65": _probe(name="a" * 65),
    "cx-name-leading-hyphen": _probe(name="-abc"),
    "cx-author-str": _probe(author="me"),
    "cx-defaultprompt-str": _probe(interface__defaultPrompt="hi"),
    "cx-if-defaultprompt-num": _probe(interface__defaultPrompt=5),
    # Controls for the round-2 probes: Codex installs each of these.
    "cx-if-logo-str": _probe(interface__logo="./missing.png"),
    "cx-if-logo-null": _probe(interface__logo=None),
    "cx-if-logodark-null": _probe(interface__logoDark=None),
    "cx-if-composericon-null": _probe(interface__composerIcon=None),
    "cx-if-brandcolor-null": _probe(interface__brandColor=None),
    "cx-if-screenshots-list": _probe(interface__screenshots=["./a.png"]),
    "cx-if-screenshots-empty": _probe(interface__screenshots=[]),
    "cx-caps-empty": _probe(interface__capabilities=[]),
    "cx-keywords-empty": _probe(keywords=[]),
    "cx-if-displayname-null": _probe(interface__displayName=None),
    "cx-if-defaultprompt-null": _probe(interface__defaultPrompt=None),
    "cx-description-null": _probe(description=None),
    "cx-version-null": _probe(version=None),
    "cx-apps-null": _probe(apps=None),
    "cx-interface-null": _probe(interface=None),
}


# A number in a component field is reported once, by the component path check (one defect, one finding).
_COMPONENT_PATH_PROBES = {"cx-apps-num"}


@pytest.mark.parametrize("probe", sorted(REFUSED))
def test_manifests_codex_refuses_are_high(tmp_path: Path, probe: str) -> None:
    result = _validate(tmp_path, REFUSED[probe])

    reported = result.findings if probe in _COMPONENT_PATH_PROBES else _manifest_findings(result)
    assert any(finding.severity == Severity.HIGH for finding in reported), result.findings
    assert not result.passed


def test_scalar_apps_value_gets_one_finding(tmp_path: Path) -> None:
    result = _validate(tmp_path, REFUSED["cx-apps-num"])

    about_apps = [f for f in result.findings if f.metadata.get("field") == "apps" or "'apps'" in f.message]
    assert [(f.check_name, f.severity) for f in about_apps] == [("plugin_component_path_invalid", Severity.HIGH)]


@pytest.mark.parametrize("probe", sorted(INSTALLED))
def test_manifests_codex_installs_get_no_high_manifest_finding(tmp_path: Path, probe: str) -> None:
    result = _validate(tmp_path, INSTALLED[probe])

    severities = {finding.check_name: finding.severity for finding in _manifest_findings(result)}
    assert Severity.HIGH not in severities.values(), severities


def test_installed_probes_without_component_values_pass(tmp_path: Path) -> None:
    # The component-value probes (hooks, mcpServers, skills) also get component-path and MCP findings, which
    # belong to other checks; these four have nothing else to report and must pass.
    for probe in ("cx-homepage-num", "cx-extensions-str", "cx-name-65", "cx-name-leading-hyphen"):
        result = _validate(tmp_path / probe, INSTALLED[probe])
        assert result.passed, (probe, result.findings)


def test_complete_codex_manifest_has_no_finding_n02(tmp_path: Path) -> None:
    result = _validate(tmp_path, N02)

    assert _manifest_findings(result) == []
    assert result.passed


def test_capabilities_string_names_the_field_c6(tmp_path: Path) -> None:
    result = _validate(tmp_path, REFUSED["cx-caps-str"])

    finding = next(f for f in result.findings if f.check_name == "schema:interface.capabilities:type")
    assert finding.severity == Severity.HIGH
    assert "missing or invalid plugin.json" in finding.message


def test_codex_unknown_fields_are_medium_with_a_hint_e21(tmp_path: Path) -> None:
    manifest = _probe(mcpServer={"x": {"command": "x"}}, skils="./skills/")
    result = _validate(tmp_path, manifest)

    unknown = [f for f in result.findings if f.check_name == "plugin_manifest_unknown_field"]
    assert sorted(f.metadata["field"] for f in unknown) == ["mcpServer", "skils"]
    assert all(f.severity == Severity.MEDIUM for f in unknown)
    hints = " ".join(f.suggestion for f in unknown)
    assert "Did you mean 'mcpServers'?" in hints
    assert "Did you mean 'skills'?" in hints
    assert result.passed


def test_codex_interface_unknown_key_is_medium(tmp_path: Path) -> None:
    result = _validate(tmp_path, _probe(interface__tagline="x"))

    finding = next(f for f in result.findings if f.check_name == "schema:interface:unknown_key")
    assert finding.severity == Severity.MEDIUM


@pytest.mark.parametrize("version", ["", "  ", "1.0 beta", "..", "1/0"])
def test_codex_refuses_blank_or_unsafe_versions(tmp_path: Path, version: str) -> None:
    result = _validate(tmp_path, _probe(version=version))

    assert any(f.severity == Severity.HIGH and f.metadata.get("field") == "version" for f in result.findings)


def test_codex_name_rules(tmp_path: Path) -> None:
    long_name = _validate(tmp_path / "long", _probe(name="a" * 65))
    spaced = _validate(tmp_path / "spaced", _probe(name="my plugin"))

    assert {f.check_name: f.severity for f in _manifest_findings(long_name)} == {
        "schema:name:too_long": Severity.MEDIUM
    }
    assert "schema:name:pattern" in {f.check_name for f in _manifest_findings(spaced)}
    assert not spaced.passed


def test_codex_overlay_interface_types_are_checked(tmp_path: Path) -> None:
    (tmp_path / "plugin.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
                "name": "demo-ap",
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / ".codex-plugin" / "plugin.json"
    overlay.parent.mkdir()
    overlay.write_text(json.dumps({"interface": {"capabilities": "Interactive"}}), encoding="utf-8")

    result = PluginSchemaValidator().validate(tmp_path)

    invalid = [f for f in result.findings if f.check_name == "plugin_manifest_additional_invalid"]
    assert len(invalid) == 1
    assert "interface.capabilities" in invalid[0].message


def test_codex_overlay_refuses_null_keywords_and_wrong_image_fields(tmp_path: Path) -> None:
    (tmp_path / "plugin.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
                "name": "demo-ap",
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / ".codex-plugin" / "plugin.json"
    overlay.parent.mkdir()
    overlay.write_text(
        json.dumps({"keywords": None, "interface": {"logo": 5, "screenshots": "x"}}),
        encoding="utf-8",
    )

    result = PluginSchemaValidator().validate(tmp_path)

    [invalid] = [f for f in result.findings if f.check_name == "plugin_manifest_additional_invalid"]
    for field in ("'keywords'", "'interface.logo'", "'interface.screenshots'"):
        assert field in invalid.message, field


def test_codex_overlay_accepts_the_installed_controls(tmp_path: Path) -> None:
    (tmp_path / "plugin.json").write_text(
        json.dumps({"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "demo-ap"}),
        encoding="utf-8",
    )
    overlay = tmp_path / ".codex-plugin" / "plugin.json"
    overlay.parent.mkdir()
    overlay.write_text(
        json.dumps({"keywords": [], "interface": {"logo": None, "screenshots": [], "capabilities": []}}),
        encoding="utf-8",
    )

    result = PluginSchemaValidator().validate(tmp_path)

    assert not [f for f in result.findings if f.check_name == "plugin_manifest_additional_invalid"]


def test_validate_cli_fails_c6_and_passes_n02(tmp_path: Path) -> None:
    def run(name: str, manifest: dict) -> int:
        root = tmp_path / name
        path = root / ".codex-plugin" / "plugin.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        args = ["--type", "plugin", "--tiers", "1", "--no-llm", "--checks", "schema", "-r", "json"]
        return CliRunner().invoke(cli, ["validate", str(root), *args, "-o", str(tmp_path / f"{name}-out")]).exit_code

    assert run("c6", REFUSED["cx-caps-str"]) == 1
    assert run("n02", N02) == 0
    assert run("homepage", INSTALLED["cx-homepage-num"]) == 0
    # Verifier round 2: these passed Tier 1 with exit 0, though Codex refuses them.
    assert run("logo", REFUSED["cx-if-logo-num"]) == 1
    assert run("keywords-null", REFUSED["cx-keywords-null"]) == 1
    assert run("screenshots", REFUSED["cx-if-screenshots-str"]) == 1
    assert run("logo-str", INSTALLED["cx-if-logo-str"]) == 0
