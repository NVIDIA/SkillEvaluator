# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR243 F6/F12-F15: native OpenClaw MCP launch and coverage semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.constants import PLUGIN_CODEX_MANIFEST_TYPE, PLUGIN_OPENCLAW_MANIFEST_TYPE
from skillevaluator.evaluation import EvaluationService
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.validators.mcp_static import validate_mcp_server_declaration
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_ENDPOINT = "https://mcp.example.com/mcp"
_ROOT_DEFAULTS = [
    {},
    {"cwd": ""},
    {"cwd": None},
    {"cwd": False},
    {"workingDirectory": ""},
    {"workingDirectory": None},
    {"workingDirectory": False},
    {"cwd": None, "workingDirectory": "."},
    {"cwd": False, "workingDirectory": ""},
    # An empty cwd is still a string: absolutization makes it the root before
    # the runtime chooses cwd over workingDirectory. Do not use the alias here.
    {"cwd": "", "workingDirectory": "safe"},
]


def _plugin(root: Path, server: dict, *, skill: bool = True) -> Path:
    root.mkdir(parents=True)
    manifest = {"id": "mcp-regression", "configSchema": {}, "mcpServers": {"srv": server}}
    if skill:
        manifest["skills"] = ["portable"]
        member = root / "portable/alpha"
        member.mkdir(parents=True)
        (member / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Portable test guidance.\nmetadata:\n"
            "  author: Example <example@example.com>\n---\n# Instructions\nUse alpha.\n",
            encoding="utf-8",
        )
    (root / "openclaw.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    evals = root / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('[{"id":"one","prompt":"Use srv","expected_output":"ok"}]', encoding="utf-8")
    return root


def _mcp_row(package) -> dict:
    return next(row for row in package.provenance()["component_coverage"]["components"] if row["type"] == "mcp")


@pytest.mark.parametrize("cwd", _ROOT_DEFAULTS)
def test_f6_root_default_cannot_hide_unscanned_server_code(tmp_path: Path, cwd: dict) -> None:
    root = _plugin(tmp_path / "plugin", {"command": "node", "args": ["node_modules/srv/index.js"], **cwd})
    findings = PluginSchemaValidator().validate(root).findings
    assert any(f.check_name == "plugin_component_path_unscanned" and f.severity.is_error() for f in findings)
    with pytest.raises(ValueError, match="plugin_component_path_unscanned"):
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize("cwd", _ROOT_DEFAULTS)
def test_f6_root_default_never_stages_stdio_without_cwd(tmp_path: Path, cwd: dict) -> None:
    root = _plugin(tmp_path / "plugin", {"command": "uvx", "args": ["example-mcp==1.2.3"], **cwd})
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert package.runnable_mcp_servers == ()
    assert package.mcp_unsupported_config == ("srv",)
    assert package.provenance()["partial"]
    assert _mcp_row(package)["state"] == "unsupported"
    assert not (package.package_path / "evals/environment/plugin_mcp_servers.toml").exists()


@pytest.mark.parametrize(
    "cwd",
    [
        {"cwd": None, "workingDirectory": "node_modules/srv"},
        {"cwd": False, "workingDirectory": "node_modules/srv"},
        {"cwd": "node_modules/srv", "workingDirectory": "safe"},
    ],
)
def test_f6_cwd_string_precedence_keeps_unscanned_alias_checks(tmp_path: Path, cwd: dict) -> None:
    root = _plugin(tmp_path / "plugin", {"command": "node", "args": ["index.js"], **cwd})
    findings = PluginSchemaValidator().validate(root).findings
    assert any(f.check_name == "plugin_component_path_unscanned" and f.severity.is_error() for f in findings)
    with pytest.raises(ValueError, match="plugin_component_path_unscanned"):
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")


@pytest.mark.parametrize(
    "url",
    ["${MCP_URL:-https://mcp.example.com/mcp}", "https://${MCP_HOST:-mcp.example.com}/mcp", "https://${MCP_HOST}/mcp"],
)
def test_f12_native_url_environment_references_are_verbatim_and_block_staging(tmp_path: Path, url: str) -> None:
    root = _plugin(tmp_path / "plugin", {"url": url, "transport": "streamable-http"})
    findings = PluginSchemaValidator().validate(root).findings
    finding = next((f for f in findings if f.check_name == "mcp_url_env_not_expanded"), None)
    assert finding is not None and finding.severity.is_error()
    assert "OpenClaw" in finding.message and "verbatim" in finding.message
    assert "Codex" not in finding.message
    with pytest.raises(ValueError, match="mcp_url_env_not_expanded"):
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"sslVerify": True},
        {"ssl_verify": True},
        {"sslVerify": True, "ssl_verify": False},
        {"sslVerify": None, "ssl_verify": True},
    ],
)
def test_f13_effective_default_tls_is_not_an_unapplied_gap(tmp_path: Path, options: dict) -> None:
    root = _plugin(tmp_path / "plugin", {"url": _ENDPOINT, **options})
    assert not any(f.severity.is_error() for f in PluginSchemaValidator().validate(root).findings)
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert package.runnable_mcp_servers == ("srv",)
    assert package.mcp_unsupported_config == ()
    assert package.provenance()["partial"] is False
    assert "unapplied" not in _mcp_row(package)["reason"]


@pytest.mark.parametrize(
    "options",
    [
        {"sslVerify": False},
        {"ssl_verify": False},
        {"sslVerify": False, "ssl_verify": True},
        {"sslVerify": None, "ssl_verify": False},
    ],
)
def test_f13_effective_tls_false_still_fails_closed(tmp_path: Path, options: dict) -> None:
    root = _plugin(tmp_path / "plugin", {"url": _ENDPOINT, **options})
    findings = PluginSchemaValidator().validate(root).findings
    assert any(f.check_name == "mcp_insecure_tls_config" and f.severity.is_error() for f in findings)
    with pytest.raises(ValueError, match="mcp_insecure_tls_config"):
        prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize(
    "server",
    [{"command": "uvx", "args": ["example-mcp==1.2.3"]}, {"command": "${CLAUDE_PLUGIN_ROOT}/bin/server"}],
)
def test_f14_stdio_skip_and_coverage_explain_unpreservable_working_directory(tmp_path: Path, server: dict) -> None:
    root = _plugin(tmp_path / "plugin", server, skill=False)
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert package.skipped and package.provenance()["partial"]
    assert package.mcp_unsupported_config == ("srv",)
    for reason in (package.skip_reason, _mcp_row(package)["reason"]):
        assert "OpenClaw stdio" in reason
        assert "plugin-root working directory cannot be preserved" in reason
        assert "${CLAUDE_PLUGIN_ROOT}" not in reason
        assert "--plugin-load native" not in reason


def test_f14_cli_json_retains_stdio_skip_reason_without_running_agents(tmp_path: Path, monkeypatch) -> None:
    root = _plugin(tmp_path / "plugin", {"command": "uvx", "args": ["example-mcp==1.2.3"]}, skill=False)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Unsupported stdio must not start an evaluation agent")

    monkeypatch.setattr(EvaluationService, "evaluate", must_not_run)
    reports = tmp_path / "reports"
    result = CliRunner().invoke(cli, ["tier3", "evaluate-plugin", str(root), "-r", "json", "-o", str(reports)])
    assert result.exit_code == 1, result.output
    assert "INCOMPLETE" in result.output
    assert "plugin-root working directory cannot be preserved" in result.output
    report = json.loads(next(reports.glob("*.json")).read_text())
    assert report["overall_status"] == "incomplete"
    serialized = json.dumps(report)
    assert "OpenClaw stdio" in serialized
    assert "plugin-root working directory cannot be preserved" in serialized
    assert "${CLAUDE_PLUGIN_ROOT}" not in serialized


@pytest.mark.parametrize("enabled", [False, True])
def test_f15_plugin_enabled_flag_is_not_an_operator_switch(tmp_path: Path, enabled: bool) -> None:
    root = _plugin(tmp_path / "plugin", {"url": _ENDPOINT, "transport": "streamable-http", "enabled": enabled})
    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")
    assert package.runnable_mcp_servers == ("srv",)
    assert not package.skipped
    assert package.provenance()["partial"] is False
    row = _mcp_row(package)
    assert row["state"] == "staged"
    config = (package.package_path / "evals/environment/plugin_mcp_servers.toml").read_text()
    assert _ENDPOINT in config
    assert "enabled" not in config


def test_other_clients_keep_their_url_environment_rules() -> None:
    server = {"url": "${MCP_URL:-https://mcp.example.com/mcp}"}
    claude = validate_mcp_server_declaration("srv", server, "plugin.json")
    codex = validate_mcp_server_declaration("srv", server, "plugin.json", manifest_type=PLUGIN_CODEX_MANIFEST_TYPE)
    assert not any(f.severity.is_error() for f in claude)
    assert any(f.check_name == "mcp_url_env_not_expanded" and f.severity.is_error() for f in codex)
    assert any("Codex" in f.message for f in codex)


def test_native_url_expansion_rejection_still_checks_inline_secrets() -> None:
    findings = validate_mcp_server_declaration(
        "srv",
        {"url": "${MCP_URL:-https://user:password@mcp.example.com/mcp}"},
        "openclaw.plugin.json",
        manifest_type=PLUGIN_OPENCLAW_MANIFEST_TYPE,
    )
    assert any(f.check_name == "mcp_url_inline_secret" and f.severity.is_error() for f in findings)
    assert all("password" not in f.message for f in findings)
