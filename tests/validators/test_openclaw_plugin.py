# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenClaw manifest support without claiming Gateway runtime execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.cli_core import detect_content_type, resolve_plugin_path
from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_SKILL = "---\nname: alpha\ndescription: Demonstrate a portable skill.\nmetadata:\n  author: Example Author <author@example.com>\n---\n# Instructions\nUse alpha.\n"


def _plugin(root: Path, **fields) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "openclaw.plugin.json").write_text(
        json.dumps({"id": "demo", "configSchema": {"type": "object"}, **fields}), encoding="utf-8"
    )
    return root


def _skill(root: Path, path: str) -> None:
    target = root / path / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SKILL, encoding="utf-8")


def test_native_manifest_discovery_and_identity(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "package", name="Demo display name")
    for path in (root, root / "openclaw.plugin.json"):
        located = locate_plugin_manifest(path)
        assert located is not None
        assert located.root == root
        assert located.manifest_type == "openclaw_plugin_json"
        assert detect_content_type(path) == CONTENT_TYPE_PLUGIN
        assert resolve_plugin_path(path) == root
    result = PluginSchemaValidator().validate(root)
    assert result.metadata["plugin"]["name"] == "demo"
    assert not any(f.severity.is_error() for f in result.findings)


@pytest.mark.parametrize(
    "fields",
    [{"id": ""}, {"id": 7}, {"configSchema": None}, {"configSchema": []}, {"skills": "skills"}, {"skills": [7]}],
)
def test_invalid_required_fields_and_skill_types(tmp_path: Path, fields: dict) -> None:
    root = _plugin(tmp_path, **fields)
    result = PluginSchemaValidator().validate(root)
    assert any(f.severity.is_error() for f in result.findings)


@pytest.mark.parametrize("skills", [None, [], ["portable"], ["skills"]])
def test_only_declared_skill_roots_are_evaluated(tmp_path: Path, skills: list | None) -> None:
    root = _plugin(tmp_path, **({"skills": skills} if skills is not None else {}))
    _skill(root, "skills/alpha")
    _skill(root, "portable/group/alpha")
    result = PluginSchemaValidator().validate(root)
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    paths = {r["path"] for r in rows if r["type"] == "skill" and r["support"] == "evaluated"}
    assert paths == ({"portable/group/alpha"} if skills == ["portable"] else {"skills/alpha"} if skills else set())
    assert any(r["type"] == "extension" and r["support"] == "unsupported" for r in rows)


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", "missing"])
def test_declared_skill_paths_fail_closed(tmp_path: Path, path: str) -> None:
    result = PluginSchemaValidator().validate(_plugin(tmp_path, skills=[path]))
    assert any(f.severity.is_error() and "path" in f.check_name for f in result.findings)


def test_symlinked_manifest_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "outside.json"
    target.write_text('{"id":"outside","configSchema":{}}', encoding="utf-8")
    root = tmp_path / "plugin"
    root.mkdir()
    (root / "openclaw.plugin.json").symlink_to(target)
    with pytest.raises(PluginManifestPathError):
        locate_plugin_manifest(root)


def test_portable_tier3_package_reports_runtime_as_unsupported(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "plugin", skills=["portable"], name="Demo display name")
    _skill(root, "portable/alpha")
    evals = root / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('[{"id":"one","prompt":"Use alpha","expected_output":"ok"}]', encoding="utf-8")
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not package.skipped
    assert package.plugin_name == "demo"
    assert [p.name for p in package.include_skills] == ["alpha"]
    rows = package.provenance()["component_coverage"]["components"]
    runtime = [r for r in rows if r["type"] == "extension"]
    assert runtime and all(r["state"] == "unsupported" for r in runtime)
    assert "OpenClaw" in runtime[0]["reason"]
    assert not (package.package_path / "index.ts").exists()


def test_cli_accepts_native_manifest(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "plugin", skills=["skills"])
    _skill(root, "skills/alpha")
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(root),
            "--tiers",
            "1",
            "--checks",
            "schema",
            "--report",
            "json",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "openclaw_plugin_json" in "".join(p.read_text() for p in (tmp_path / "reports").glob("*.json"))


@pytest.mark.parametrize("entry", ["../outside.ts", "/tmp/outside.ts", "missing.ts", 7])
def test_runtime_entrypoint_paths_are_checked_without_importing(tmp_path: Path, entry) -> None:
    root = _plugin(tmp_path)
    (root / "package.json").write_text(json.dumps({"openclaw": {"extensions": [entry]}}), encoding="utf-8")
    result = PluginSchemaValidator().validate(root)
    assert any(f.severity.is_error() for f in result.findings)


def test_runtime_entrypoints_are_inventoried_without_execution(tmp_path: Path) -> None:
    root = _plugin(tmp_path)
    (root / "package.json").write_text(json.dumps({"openclaw": {"extensions": ["./index.ts"]}}), encoding="utf-8")
    (root / "index.ts").write_text("throw new Error('must never execute');", encoding="utf-8")
    result = PluginSchemaValidator().validate(root)
    assert not any(f.severity.is_error() for f in result.findings)
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    assert any(r["type"] == "extension" and r["path"] == "index.ts" and r["support"] == "unsupported" for r in rows)


def test_runtime_only_plugin_is_skipped_not_certified(tmp_path: Path) -> None:
    package = prepare_plugin_eval_package(_plugin(tmp_path / "plugin"), stage_root=tmp_path / "stage")
    assert package.skipped
    rows = package.provenance()["component_coverage"]["components"]
    assert any(r["name"] == "OpenClaw runtime" and r["state"] == "unsupported" for r in rows)


def test_existing_manifest_precedence_is_preserved(tmp_path: Path) -> None:
    root = _plugin(tmp_path, name="Different display label")
    vendor = root / ".claude-plugin"
    vendor.mkdir()
    (vendor / "plugin.json").write_text('{"name":"demo"}', encoding="utf-8")
    located = locate_plugin_manifest(root)
    assert located is not None and located.manifest_type == "claude_plugin_json"
    assert [p.manifest_type for p in located.additional] == ["openclaw_plugin_json"]
    result = PluginSchemaValidator().validate(root)
    assert not any(f.check_name == "plugin_manifest_conflict" for f in result.findings)
    assert any(r["name"] == "OpenClaw runtime" for r in result.metadata["plugin"]["component_inventory"]["components"])


def test_inline_mcp_gets_blocking_static_checks(tmp_path: Path) -> None:
    root = _plugin(tmp_path, mcpServers={"bad": {"command": "npx", "args": ["@example/server@latest"]}})
    result = PluginSchemaValidator().validate(root)
    assert any(f.check_name == "mcp_command_floating_version" and f.severity.is_error() for f in result.findings)


def test_terminal_skill_does_not_load_nested_example_skills(tmp_path: Path) -> None:
    root = _plugin(tmp_path, skills=["portable"])
    _skill(root, "portable/alpha")
    _skill(root, "portable/alpha/templates/example")
    result = PluginSchemaValidator().validate(root)
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    assert {r["path"] for r in rows if r["type"] == "skill" and r["support"] == "evaluated"} == {"portable/alpha"}


@pytest.mark.parametrize(
    "server",
    [
        {"command": "node", "args": ["./server.js"]},
        {"command": "node", "args": ["--version"], "workingDirectory": "./data"},
    ],
)
def test_openclaw_stdio_cwd_is_not_silently_dropped(tmp_path: Path, server: dict) -> None:
    root = _plugin(tmp_path / "plugin", skills=["portable"], mcpServers={"local": server})
    _skill(root, "portable/alpha")
    (root / "server.js").write_text("throw new Error('not executed');", encoding="utf-8")
    (root / "data").mkdir()
    evals = root / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('[{"id":"one","prompt":"Use alpha","expected_output":"ok"}]', encoding="utf-8")
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert "local" not in package.runnable_mcp_servers
    rows = package.provenance()["component_coverage"]["components"]
    assert any(r["type"] == "mcp" and r["name"] == "local" and r["state"] == "unsupported" for r in rows)


def test_disabled_openclaw_mcp_server_is_not_staged(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "plugin", mcpServers={"remote": {"url": "https://mcp.example.com/mcp", "enabled": False}})
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not package.runnable_mcp_servers
    assert package.skipped


@pytest.mark.parametrize("fields", [{"id": ""}, {"id": None}, {"configSchema": None}, {"skills": "portable"}])
def test_direct_tier3_rejects_invalid_native_manifest(tmp_path: Path, fields: dict) -> None:
    root = _plugin(tmp_path / "plugin", **fields)
    with pytest.raises(ValueError, match="Invalid OpenClaw manifest"):
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")


@pytest.mark.parametrize(
    "server,transport",
    [
        ({"url": "https://mcp.example.com/mcp", "transport": "streamable-http"}, "http"),
        ({"url": "https://mcp.example.com/mcp"}, "sse"),
        ({"url": "https://mcp.example.com/mcp", "transport": "sse", "type": "http"}, "sse"),
        ({"url": "https://mcp.example.com/mcp", "type": "http"}, "http"),
    ],
)
def test_native_remote_mcp_transport_semantics(tmp_path: Path, server: dict, transport: str) -> None:
    root = _plugin(tmp_path / "plugin", mcpServers={"remote": server})
    evals = root / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('[{"id":"one","prompt":"Use remote","expected_output":"ok"}]', encoding="utf-8")
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not package.skipped
    assert package.runnable_mcp_servers == ("remote",)
    config_path = package.package_path / "evals/environment/plugin_mcp_servers.toml"
    assert f'transport = "{transport}"' in config_path.read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("auth", "oauth"),
        ("oauth", {"clientId": "example"}),
        ("enterprise", {"enabled": True}),
        ("clientCert", "./client.pem"),
        ("client_key", "./client.key"),
    ],
)
def test_openclaw_host_mcp_options_are_explicit_gaps(tmp_path: Path, field: str, value) -> None:
    root = _plugin(tmp_path / "plugin", mcpServers={"remote": {"url": "https://mcp.example.com/mcp", field: value}})
    evals = root / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('[{"id":"one","prompt":"Use remote","expected_output":"ok"}]', encoding="utf-8")
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    row = next(r for r in package.provenance()["component_coverage"]["components"] if r["type"] == "mcp")
    assert row["state"] == "staged"  # transport staged, host configuration not applied
    assert package.provenance()["partial"]
    assert package.mcp_unsupported_config == ("remote",)
    assert field in row["reason"]


@pytest.mark.parametrize("field", ["cwd", "workingDirectory"])
def test_native_mcp_bare_relative_working_directory_is_checked(tmp_path: Path, field: str) -> None:
    root = _plugin(tmp_path, mcpServers={"local": {"command": "node", "args": ["./evil.js"], field: "node_modules"}})
    result = PluginSchemaValidator().validate(root)
    assert any(f.check_name == "plugin_component_path_unscanned" and f.severity.is_error() for f in result.findings)


@pytest.mark.parametrize("path", ["skills/alpha", "custom/group/alpha", "."])
def test_explicit_plugin_root_supports_container_and_single_skill(tmp_path: Path, path: str) -> None:
    root = _plugin(tmp_path / "plugin", skills=["."])
    _skill(root, path)
    evals = tmp_path / "evals.json"
    evals.write_text('[{"id":"one","prompt":"Use alpha","expected_output":"ok"}]', encoding="utf-8")
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage", evals_source=evals)
    assert not package.skipped
    assert package.include_skills == ((root / path).resolve(),)
    rows = package.provenance()["component_coverage"]["components"]
    assert any(r["type"] == "skill" and r["path"] == path and r["state"] == "staged" for r in rows)


def test_terminal_boundaries_and_artifact_named_members_match_all_tiers(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "plugin", skills=["portable"])
    _skill(root, "portable/alpha")
    _skill(root, "portable/alpha/templates/results/example")
    _skill(root, "portable/group/results/alpha")
    result = PluginSchemaValidator().validate(root)
    assert not any(f.check_name == "plugin_skill_in_unscanned_folder" for f in result.findings)
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    assert {r["path"] for r in rows if r["type"] == "skill" and r["support"] == "evaluated"} == {
        "portable/alpha",
        "portable/group/results/alpha",
    }


def test_invalid_canonical_mcp_transport_is_not_treated_as_legacy_http(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(
        _plugin(
            tmp_path, mcpServers={"remote": {"url": "https://mcp.example.com/mcp", "transport": "http", "type": "sse"}}
        )
    )
    assert any(f.check_name == "mcp_transport_invalid" and f.severity.is_error() for f in result.findings)


@pytest.mark.parametrize(
    "options", [{"sslVerify": False}, {"ssl_verify": False}, {"sslVerify": False, "ssl_verify": True}]
)
def test_native_tls_verification_disablement_blocks_before_staging(tmp_path: Path, options: dict) -> None:
    root = _plugin(tmp_path / "plugin", mcpServers={"remote": {"url": "https://mcp.example.com/mcp", **options}})
    result = PluginSchemaValidator().validate(root)
    assert any(f.check_name == "mcp_insecure_tls_config" and f.severity.is_error() for f in result.findings)
    with pytest.raises(ValueError, match="blocking static"):
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_canonical_tls_verification_overrides_legacy_alias(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(
        _plugin(
            tmp_path,
            mcpServers={"remote": {"url": "https://mcp.example.com/mcp", "sslVerify": True, "ssl_verify": False}},
        )
    )
    assert not any(f.check_name == "mcp_insecure_tls_config" for f in result.findings)


@pytest.mark.parametrize("key", ["idpClientSecret", "resourceClientSecret"])
def test_enterprise_inline_credentials_block_without_echoing_values(tmp_path: Path, key: str) -> None:
    secret = "synthetic-enterprise-secret-must-not-be-staged"
    root = _plugin(
        tmp_path / "plugin", mcpServers={"remote": {"url": "https://mcp.example.com/mcp", "enterprise": {key: secret}}}
    )
    result = PluginSchemaValidator().validate(root)
    assert any(f.check_name == "mcp_inline_secret" and f.severity.is_error() for f in result.findings)
    assert secret not in str([f.to_dict() for f in result.findings])
    with pytest.raises(ValueError, match="blocking static"):
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_enterprise_credential_references_are_allowed_but_not_forwarded(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "plugin",
        configSchema={"type": "object", "properties": {"unused": {"default": "unsupported-host-metadata"}}},
        mcpServers={
            "remote": {
                "url": "https://mcp.example.com/mcp",
                "enterprise": {"idpClientSecret": "${IDP_SECRET}", "resourceClientSecret": "${RESOURCE_SECRET}"},
            }
        },
    )
    evals = root / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('[{"id":"one","prompt":"Use remote","expected_output":"ok"}]', encoding="utf-8")
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert package.provenance()["partial"]
    staged_manifest = (package.package_path / "agent_plugin.yaml").read_text()
    assert "enterprise" not in staged_manifest
    assert "unsupported-host-metadata" not in staged_manifest
    assert "configSchema" not in staged_manifest
