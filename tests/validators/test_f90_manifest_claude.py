# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code manifests are checked the way Claude Code 2.1.284 checks them (proof H7, M22, M38, L6).

The fixtures are the proof's check-1 and check-7 examples. The expected errors
are the ones ``claude plugin validate`` printed for the same files: six errors
for g01 (k03), "dependencies: Invalid input" for c7, "lspServers: Invalid
input" for check-7 edge-05, an error for a padded name, and a pass for a
65-character name. The homepage and experimental probes are verifier round-2
probes, run against ``claude plugin validate --json`` 2.1.284.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Severity
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

# check-01 g01 (also k03, where Claude Code reports six errors and one warning).
G01 = {
    "name": "My Plugin!",
    "version": 123,
    "description": ["not", "a", "string"],
    "author": "me",
    "keywords": "not-a-list",
    "homepage": 42,
    "bogusKey": True,
}
# check-07 edge-05: three inline LSP servers that Claude Code rejects.
EDGE05_LSP = {
    "nocommand": {"args": ["--stdio"], "extensionToLanguage": {".py": "python"}},
    "spaced": {
        "command": "typescript-language-server --stdio",
        "extensionToLanguage": {".ts": "typescript", ".tsx": "typescriptreact"},
    },
    "unknownkey": {"command": "gopls", "args": ["serve"], "extensionToLanguage": {".go": "go"}, "autoApprove": True},
}
FULL_METADATA = {
    "version": "1.0.0",
    "description": "A demo plugin",
    "author": {"name": "Example Dev", "email": "dev@example.com"},
}


def _claude(root: Path, manifest: dict) -> Path:
    path = root / ".claude-plugin" / "plugin.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _schema_findings(result, *, severity: Severity | None = None) -> dict[str, list]:
    found: dict[str, list] = {}
    for finding in result.findings:
        if finding.category != "PLUGIN_SCHEMA":
            continue
        if severity is not None and finding.severity != severity:
            continue
        found.setdefault(finding.check_name, []).append(finding)
    return found


def _success_messages(result) -> list[str]:
    return [detail.message for detail in result.success_details]


def test_g01_wrong_types_fail_on_the_six_fields_claude_code_rejects(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_claude(tmp_path, G01))

    high = _schema_findings(result, severity=Severity.HIGH)
    fields = {finding.metadata["field"] for findings in high.values() for finding in findings}
    # Claude Code 2.1.284 (k03): name, version, description, author, homepage, keywords.
    assert fields == {"name", "version", "description", "author", "homepage", "keywords"}
    assert "schema:name:spaces" in high
    assert not result.passed
    assert not any("is valid" in message for message in _success_messages(result))
    unknown = _schema_findings(result)["plugin_manifest_unknown_field"]
    assert [finding.metadata["field"] for finding in unknown] == ["bogusKey"]
    assert unknown[0].severity == Severity.MEDIUM


def test_padded_name_is_refused_like_claude_code_does(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_claude(tmp_path, {"name": "  demo  "}))

    assert not result.passed
    assert "schema:name:spaces" in _schema_findings(result, severity=Severity.HIGH)


def test_dependencies_string_is_refused_c7(tmp_path: Path) -> None:
    manifest = {
        "name": "claude-demo",
        **FULL_METADATA,
        "keywords": ["release", "notes"],
        "dependencies": "not-a-list",
    }
    result = PluginSchemaValidator().validate(_claude(tmp_path, manifest))

    assert not result.passed
    assert "schema:dependencies:type" in _schema_findings(result, severity=Severity.HIGH)


def test_dependency_entries_must_be_plugin_names(tmp_path: Path) -> None:
    manifest = {"name": "demo", "dependencies": ["ok-name", "bad name", {"marketplace": "m"}, 5]}
    result = PluginSchemaValidator().validate(_claude(tmp_path, manifest))

    invalid = _schema_findings(result, severity=Severity.HIGH)["schema:dependencies:invalid"]
    assert len(invalid) == 3
    assert "dependencies[1]" in invalid[0].message


def test_invalid_inline_lsp_servers_fail_edge05(tmp_path: Path) -> None:
    manifest = {"name": "c7-lsp-gaps", **FULL_METADATA, "license": "Apache-2.0", "lspServers": EDGE05_LSP}
    result = PluginSchemaValidator().validate(_claude(tmp_path, manifest))

    invalid = _schema_findings(result, severity=Severity.HIGH)["schema:lspServers:invalid"]
    messages = " ".join(finding.message for finding in invalid)
    assert len(invalid) == 3
    assert "'nocommand' has no 'command'" in messages
    assert "'spaced' 'command' contains a space" in messages
    assert "'unknownkey' has unknown key(s) 'autoApprove'" in messages
    assert not result.passed


def test_valid_inline_lsp_server_and_monitor_pass(tmp_path: Path) -> None:
    manifest = {
        "name": "lsp-ok",
        **FULL_METADATA,
        "lspServers": {
            "ts": {
                "command": "typescript-language-server",
                "args": ["--stdio"],
                "extensionToLanguage": {".ts": "typescript"},
            }
        },
        "experimental": {"monitors": [{"name": "watch", "command": "./watch.sh", "description": "Watch builds"}]},
    }
    result = PluginSchemaValidator().validate(_claude(tmp_path, manifest))

    # The monitor's own risk notes (LOW) come from the monitor checks; the manifest has no schema finding.
    assert not [check for check in _schema_findings(result) if check.startswith(("schema:", "plugin_manifest"))]
    assert result.passed


def test_invalid_inline_monitor_fails(tmp_path: Path) -> None:
    manifest = {"name": "mon", "monitors": [{"name": "watch", "command": "./watch.sh"}, {"name": "watch"}]}
    result = PluginSchemaValidator().validate(_claude(tmp_path, manifest))

    invalid = _schema_findings(result, severity=Severity.HIGH)["schema:monitors:invalid"]
    assert any("'description'" in finding.message for finding in invalid)
    assert any("repeats a monitor name" in finding.message for finding in invalid)


def test_component_field_shapes_are_checked(tmp_path: Path) -> None:
    manifest = {"name": "shapes", "skills": 5, "agents": {"a": 1}, "settings": [], "hooks": 3}
    result = PluginSchemaValidator().validate(_claude(tmp_path, manifest))

    high = _schema_findings(result, severity=Severity.HIGH)
    for check in ("schema:agents:type", "schema:settings:type"):
        assert check in high, check
    # A number in a path field is reported once, by the component path check (one defect, one finding).
    assert "schema:skills:type" not in high
    assert "schema:hooks:type" not in high
    assert {"5", "3"} <= {f.metadata["plugin_component_ref"] for f in high["plugin_component_path_invalid"]}
    assert not result.passed


# Verifier round 2 (L6): one defect in the selected manifest got two HIGH findings, schema:<field>:type and the
# component path finding; PR #28's base reported only the component path finding.
SCALAR_COMPONENTS = {
    "agents": "plugin_component_path_invalid",
    "commands": "plugin_component_path_invalid",
    "hooks": "plugin_component_path_invalid",
    "lspServers": "plugin_component_path_invalid",
    "outputStyles": "plugin_component_path_invalid",
    "skills": "plugin_component_path_invalid",
    "mcpServers": "mcp_servers_not_object",
}


def test_scalar_component_value_gets_one_finding(tmp_path: Path) -> None:
    for field, check_name in SCALAR_COMPONENTS.items():
        result = PluginSchemaValidator().validate(_claude(tmp_path / field, {"name": "demo", field: 5}))

        blocking = [(f.check_name, f.severity) for f in result.findings if f.severity == Severity.HIGH]
        assert blocking == [(check_name, Severity.HIGH)], (field, blocking)
        assert not result.passed


def test_scalar_experimental_monitors_gets_one_finding(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_claude(tmp_path, {"name": "demo", "experimental": {"monitors": 5}}))

    blocking = [f.check_name for f in result.findings if f.severity == Severity.HIGH]
    assert blocking == ["plugin_component_path_invalid"]


# `claude plugin validate --json` (2.1.284) reports "homepage: Invalid input" for each of these, and
# `claude --plugin-dir <probe> plugin list --json` shows the space and port probes disabled for it.
REFUSED_HOMEPAGES = [
    "https://exa mple.com",
    "https://[::1",
    "https://example.com:99999",
    "https://ex%zzample.com",
    "https://exa<mple.com",
    "https://exa>mple.com",
    "https://exa^mple.com",
    "https://exa|mple.com",
    "https://ex%20ample.com",
    "https://ex%25mple.com",
    "https://ex%FFmple.com",
    "https://example.com:8a/",
    "https://1.2.3.256/",
    "https://[zz]/",
    "https://[::1]x/",
    "https://user@/x",
    "https://xn--zz.com/",
    "https://exa\uff1cmple.com",
    "ftp://a b/",
    "foo://exa mple.com",
    "foo://example.com:99999",
    "file://exa mple/x",
]
# ... and accepts each of these.
ACCEPTED_HOMEPAGES = [
    "https://example.com:65535/x",
    "https://[::1]:8080/",
    "https://ex%41mple.com",
    "https://example.com:/",
    "https://exa#mple.com",
    "https://exa\\mple.com",
    "https://user@example.com",
    "https://example.com/%zz",
    "https://ex%C3%A9mple.com",
    "https://ex\u00e4mple.com",
    "https:///x",
    "https://0x7f.1/",
    "https://1.2.3.4./",
    "https://4294967295/",
    "https://[::ffff:1.2.3.4]/",
    "https://example.com\u00a0",
    "foo://ex%zz.com/",
    "mailto:x@y.z",
    "file:///tmp/x",
]


def test_homepages_claude_code_refuses_are_high(tmp_path: Path) -> None:
    for index, homepage in enumerate(REFUSED_HOMEPAGES):
        result = PluginSchemaValidator().validate(
            _claude(tmp_path / str(index), {"name": "demo", "homepage": homepage})
        )

        invalid = _schema_findings(result, severity=Severity.HIGH).get("schema:homepage:invalid_url")
        assert invalid, homepage
        assert not result.passed, homepage


def test_homepages_claude_code_accepts_pass(tmp_path: Path) -> None:
    for index, homepage in enumerate(ACCEPTED_HOMEPAGES):
        result = PluginSchemaValidator().validate(
            _claude(tmp_path / str(index), {"name": "demo", "homepage": homepage})
        )

        assert "schema:homepage:invalid_url" not in _schema_findings(result), homepage
        assert result.passed, (homepage, result.findings)


def test_experimental_component_fields_are_named_like_claude_code_names_them(tmp_path: Path) -> None:
    # Claude Code reports "experimental.themes: Invalid input" for a number and for null.
    for index, value in enumerate((5, None, [5])):
        manifest = {"name": "demo", "experimental": {"themes": value}}
        result = PluginSchemaValidator().validate(_claude(tmp_path / str(index), manifest))

        [finding] = _schema_findings(result, severity=Severity.HIGH)["schema:experimental.themes:type"]
        assert finding.metadata["field"] == "experimental.themes"
        assert "'experimental.themes'" in finding.message
        assert not result.passed
    ok = PluginSchemaValidator().validate(
        _claude(tmp_path / "ok", {"name": "demo", "experimental": {"themes": "./themes/", "evals": ["./evals/"]}})
    )
    assert not [name for name in _schema_findings(ok) if name.startswith("schema:experimental")]


def test_long_name_passes_because_claude_code_has_no_length_limit(tmp_path: Path) -> None:
    # check-01 p05 / skeptic cl-name-65-full: `claude plugin validate --strict` passes this file.
    long_name = "a" * 65
    result = PluginSchemaValidator().validate(_claude(tmp_path / "p65", {"name": long_name, **FULL_METADATA}))
    longer = PluginSchemaValidator().validate(_claude(tmp_path / "p100", {"name": "b" * 100}))

    assert result.passed, result.findings
    assert longer.passed, longer.findings
    assert result.metadata["plugin"]["name"] == long_name


def test_name_problems_get_their_own_messages(tmp_path: Path) -> None:
    # check-01 p03 (missing), p04 (empty), e02 (number): one message each, not one "missing" message for all.
    missing = PluginSchemaValidator().validate(_claude(tmp_path / "m", {"description": "x"}))
    empty = PluginSchemaValidator().validate(_claude(tmp_path / "e", {"name": ""}))
    number = PluginSchemaValidator().validate(_claude(tmp_path / "n", {"name": 42}))

    assert list(_schema_findings(missing)) == ["schema:name:missing"]
    assert list(_schema_findings(empty)) == ["schema:name:empty"]
    assert list(_schema_findings(number)) == ["schema:name:type"]
    assert "must be a string (got number)" in number.findings[0].message
    assert "is empty" in empty.findings[0].message


def test_name_rules_follow_claude_code_load_and_install_rules(tmp_path: Path) -> None:
    hostile = PluginSchemaValidator().validate(_claude(tmp_path / "h", {"name": "<b>x</b>|y", "version": "1.0.0"}))
    upper = PluginSchemaValidator().validate(_claude(tmp_path / "u", {"name": "Demo"}))
    bang = PluginSchemaValidator().validate(_claude(tmp_path / "b", {"name": "demo!"}))

    # Claude Code refuses '/' in a name when it loads the plugin by path.
    assert "schema:name:path_unsafe" in _schema_findings(hostile, severity=Severity.HIGH)
    assert _schema_findings(upper, severity=Severity.LOW).keys() == {"schema:name:not_kebab_case"}
    assert upper.passed
    assert _schema_findings(bang, severity=Severity.MEDIUM).keys() == {"schema:name:not_installable"}
    assert bang.passed


def test_clean_claude_manifest_passes_with_an_accurate_success_row(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_claude(tmp_path, {"name": "demo", "version": "v1"}))

    assert result.passed
    assert _schema_findings(result) == {}
    messages = _success_messages(result)
    assert (
        "Claude Code plugin manifest 'demo' (.claude-plugin/plugin.json) passed the required-field checks" in messages
    )
    assert not any("deferred" in message for message in messages)


def test_additional_claude_manifest_gets_the_same_schema(tmp_path: Path) -> None:
    # agent_plugin.yaml wins precedence, so the Claude manifest is an additional declaration.
    (tmp_path / "agent_plugin.yaml").write_text(
        "name: demo\nauthor:\n  email: dev@example.com\nmcp:\n  - name: tracker\n    provider: example-provider\n",
        encoding="utf-8",
    )
    _claude(tmp_path, {"name": "demo", "version": 7})
    result = PluginSchemaValidator().validate(tmp_path)

    additional = _schema_findings(result)["plugin_manifest_additional_invalid"]
    assert additional[0].severity == Severity.MEDIUM
    assert "'version' must be a string" in additional[0].message
    rows = result.metadata["plugin"]["manifest_declarations"]["manifests"]
    assert [row["status"] for row in rows] == ["selected", "invalid"]


def test_validate_cli_exit_codes_follow_claude_code(tmp_path: Path) -> None:
    runner = CliRunner()
    base = ["--type", "plugin", "--tiers", "1", "--no-llm", "--checks", "schema", "-r", "json"]
    g01 = runner.invoke(cli, ["validate", str(_claude(tmp_path / "g01", G01)), *base, "-o", str(tmp_path / "o1")])
    clean = runner.invoke(
        cli, ["validate", str(_claude(tmp_path / "n01", {"name": "demo"})), *base, "-o", str(tmp_path / "o2")]
    )

    assert g01.exit_code == 1, g01.output
    assert clean.exit_code == 0, clean.output
