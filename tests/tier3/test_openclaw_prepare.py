# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenClaw plugins prepared for the private runtime: portable members only, runtime never certified."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.tier3 import plugin_execution as px
from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.plugin_native import OPENCLAW_RUNTIME_NOT_LOADED, ClaudeCodeAdapter, NativePluginSource
from skillevaluator.tier3.plugin_prepare import (
    OPENCLAW_ROOT_SKILL_REASON,
    PreparationError,
    prepare_runtime_package,
    stage_wrapper_skill,
)

MANIFEST = "openclaw.plugin.json"


def _skill(name: str) -> str:
    return f"---\nname: {name}\ndescription: {name} guidance.\n---\nBody.\n"


def _tree(root: Path, files: dict[str, Any]) -> Path:
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return root


def _openclaw(root: Path, files: dict[str, Any] | None = None, **fields: Any) -> Path:
    manifest = {"id": "release-helper", "configSchema": {"type": "object"}, **fields}
    return _tree(root, {MANIFEST: manifest, **(files or {})})


def _package(root: Path):
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    return prepare_runtime_package(inventory)


def _audit(package) -> dict[str, str]:
    return {row.name: row.reason for row in package.audit}


def test_portable_members_are_staged_and_runtime_is_recorded_unsupported(tmp_path: Path) -> None:
    files = {"custom/ship/SKILL.md": _skill("ship"), "skills/implicit/SKILL.md": _skill("implicit")}
    package = _package(_openclaw(tmp_path, files, name="Release Helper", skills=["custom"]))
    assert package.plugin_name == "release-helper"
    assert [member.path for member in package.skills] == ["custom/ship"]
    assert package.native_source.member_skills == (tmp_path / "custom" / "ship",)
    assert _audit(package)["OpenClaw runtime"] == OPENCLAW_RUNTIME_NOT_LOADED
    assert package.runnable and package.complete
    rows = ClaudeCodeAdapter()._unsupported_rows(package.native_source)
    assert ("extension", "OpenClaw runtime", OPENCLAW_RUNTIME_NOT_LOADED) in rows


@pytest.mark.parametrize("declared_root", [".", "./"])
def test_plugin_root_is_never_a_portable_member(tmp_path: Path, declared_root: str) -> None:
    files = {
        "SKILL.md": _skill("root"),
        ".env": "TOKEN=x",
        "evals/evals.json": [{"id": "one", "prompt": "Use the guidance."}],
        "portable/a/SKILL.md": _skill("a"),
    }
    plugin = _openclaw(tmp_path / "plugin", files, skills=[declared_root, "portable"])
    package = _package(plugin)
    assert [member.path for member in package.skills] == ["portable/a"]
    assert plugin not in package.native_source.member_skills
    assert _audit(package)[plugin.name] == OPENCLAW_ROOT_SKILL_REASON
    wrapper = stage_wrapper_skill(package, tmp_path / "wrapper", evals_source=plugin / "evals" / "evals.json")
    staged = sorted(path.relative_to(wrapper.path).as_posix() for path in wrapper.path.rglob("*") if path.is_file())
    assert staged == ["SKILL.md", "evals/evals.json", "skills/a/SKILL.md"]


@pytest.mark.parametrize("declared_root", [".", "./"])
def test_root_only_plugin_is_not_runnable_and_says_why(tmp_path: Path, declared_root: str) -> None:
    package = _package(_openclaw(tmp_path, {"SKILL.md": _skill("root")}, skills=[declared_root]))
    assert not package.runnable and not package.complete
    assert any("OpenClaw runtime evaluation is unavailable" in reason for reason in package.incomplete_reasons)
    with pytest.raises(px.PluginLoadError, match="OpenClaw runtime evaluation is unavailable"):
        px.evaluate_prepared_plugin(
            package, EvaluationOptions(skill_path=tmp_path), work_dir=tmp_path, evaluate=lambda _o: pytest.fail("ran")
        )


def test_runtime_only_plugin_is_incomplete_never_a_successful_skip(tmp_path: Path) -> None:
    files = {"package.json": {"openclaw": {"extensions": ["./index.js"]}}, "index.js": "export {}"}
    package = _package(_openclaw(tmp_path, files))
    assert not package.runnable
    assert not package.complete
    assert {row.type for row in package.audit} == {"extension"}


@pytest.mark.parametrize(
    "server",
    [
        {"command": "node", "args": ["server.js"]},
        {"command": "node", "args": ["server.js"], "cwd": "srv"},
        {"command": "node", "args": ["server.js"], "workingDirectory": "srv"},
    ],
)
def test_stdio_servers_are_never_started_without_their_working_directory(tmp_path: Path, server: dict) -> None:
    files = {"portable/a/SKILL.md": _skill("a"), "server.js": "x", "srv/server.js": "x"}
    package = _package(_openclaw(tmp_path, files, skills=["portable"], mcpServers={"local": server}))
    assert package.mcp_servers == ()
    assert package.native_source.mcp_servers == () and package.native_source.plugin_file_mcp_servers == ()
    assert "plugin-root working directory cannot be preserved" in _audit(package)["local"]
    assert "OpenClaw stdio MCP server 'local' was not started" in package.incomplete_reasons


@pytest.mark.parametrize("enabled", [False, True])
def test_plugin_enabled_flag_is_not_an_operator_switch(tmp_path: Path, enabled: bool) -> None:
    server = {"url": "https://mcp.example.com/sse", "enabled": enabled, "workingDirectory": "."}
    package = _package(_openclaw(tmp_path, mcpServers={"remote": server}))
    [staged] = package.mcp_servers
    assert (staged["name"], staged["transport"]) == ("remote", "sse")
    assert "cwd" not in staged and "workingDirectory" not in staged
    assert [server["name"] for server in package.native_source.mcp_servers] == ["remote"]
    assert package.runnable and package.complete


def test_unapplied_host_configuration_makes_the_run_incomplete(tmp_path: Path) -> None:
    server = {"url": "https://mcp.example.com/mcp", "transport": "streamable-http", "auth": {"type": "oauth"}}
    package = _package(_openclaw(tmp_path, mcpServers={"remote": server}))
    assert package.runnable
    assert _audit(package)["remote"] == "unapplied configuration: auth"
    assert package.incomplete_reasons == (
        "MCP server 'remote' declares configuration the runtime does not apply: auth",
    )


@pytest.mark.parametrize(
    "fields",
    [{"id": ""}, {"id": None}, {"configSchema": None}, {"skills": "portable"}],
)
def test_invalid_native_manifest_is_refused_before_staging(tmp_path: Path, fields: dict) -> None:
    _openclaw(tmp_path, {"portable/a/SKILL.md": _skill("a")}, **{"skills": ["portable"], **fields})
    with pytest.raises(PreparationError, match="invalid OpenClaw manifest"):
        _package(tmp_path)


def test_host_metadata_never_reaches_the_staged_manifest(tmp_path: Path) -> None:
    fields = {
        "skills": ["portable"],
        "configSchema": {"type": "object", "properties": {"token": {"default": "never-forward"}}},
        "uiHints": {"token": {"sensitive": True}},
        "commands": "./commands",
        "userConfig": {"token": {"type": "string", "default": "never-forward"}},
    }
    package = _package(_openclaw(tmp_path, {"portable/a/SKILL.md": _skill("a")}, **fields))
    assert set(package.native_source.manifest) == {"id", "skills"}
    manifest = ClaudeCodeAdapter().manifest(package.native_source)
    # Member skills are staged under skills/, so the staged manifest declares no paths.
    assert manifest == {"name": "release-helper", "description": ""}
    assert "never-forward" not in json.dumps(manifest)


def test_native_claude_code_loads_only_the_inventoried_member_skills(tmp_path: Path) -> None:
    files = {
        "SKILL.md": _skill("root"),
        "custom/a/SKILL.md": _skill("a"),
        "custom/node_modules/dep/SKILL.md": _skill("dep"),
        "skills/implicit/SKILL.md": _skill("implicit"),
        "commands/deploy.md": "# Deploy\n",
        "agents/helper.md": "---\nname: helper\ndescription: Helper.\n---\nBody.\n",
        "hooks/hooks.json": {"hooks": {}},
        "settings.json": {"agent": "helper"},
        ".mcp.json": {"mcpServers": {}},
        "bin/tool": "#!/bin/sh\n",
        "package.json": {"name": "release-helper"},
    }
    schema = {"type": "object", "properties": {"token": {"default": "never-forward"}}}
    plugin = _openclaw(tmp_path / "plugin", files, skills=[".", "custom"], configSchema=schema)
    package = _package(plugin)
    adapter = ClaudeCodeAdapter()
    staged = adapter.staged_skills(package.native_source)
    assert [(name, rel) for name, rel, _copy in staged] == [("a", "skills/a")]
    assert staged[0][2] is not None and staged[0][2].resolve() == (plugin / "custom" / "a").resolve()
    bundle = adapter.build(package.native_source)
    assert bundle.plugin_tree is None
    assert [dest for _copy, dest in bundle.trees] == [f"{adapter.bundle_dir}/skills/a"]
    assert ("bin", "bin") in {(kind, name) for kind, name, _reason in adapter._unsupported_rows(package.native_source)}
    staging = build_native_task_staging("claude-code", adapter, package.native_source)
    stage_native_bundle(tmp_path / "environment", staging)
    staged_root = tmp_path / "environment" / "skilleval" / "native" / "claude-code" / "plugin"
    staged_files = sorted(path.relative_to(staged_root).as_posix() for path in staged_root.rglob("*") if path.is_file())
    # Only adapter-generated files and the member skill; nothing else from the plugin tree.
    assert staged_files == [".claude-plugin/plugin.json", ".mcp.json", "hooks/hooks.json", "skills/a/SKILL.md"]
    assert json.loads((staged_root / ".mcp.json").read_text(encoding="utf-8")) == {"mcpServers": {}}
    assert json.loads((staged_root / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"] == {}
    assert "never-forward" not in (staged_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")


def test_openclaw_member_skills_with_one_name_fail_closed_natively(tmp_path: Path) -> None:
    files = {"one/a/SKILL.md": _skill("a"), "two/a/SKILL.md": _skill("a")}
    package = _package(_openclaw(tmp_path, files, skills=["one", "two"]))
    with pytest.raises(ValueError, match="already has skills/a"):
        ClaudeCodeAdapter().staged_skills(package.native_source)


# --------------------------------------------------------------------------- #
# Source-client-aware skill projection for native Claude Code                 #
# --------------------------------------------------------------------------- #
_OMIT = object()


def _source(tmp_path: Path, manifest_rel: str, skills: Any) -> NativePluginSource:
    manifest = {"name": "demo"} if skills is _OMIT else {"name": "demo", "skills": skills}
    return NativePluginSource(
        plugin_name="demo",
        description="",
        contained=True,
        plugin_root=tmp_path,
        manifest=manifest,
        manifest_rel=manifest_rel,
    )


@pytest.mark.parametrize(
    ("skills", "expected"),
    [
        ("skills", "./skills"),
        ("skills/alpha", "./skills/alpha"),
        ("./skills", "./skills"),
        (["portable", "./other"], ["./portable", "./other"]),
    ],
)
@pytest.mark.parametrize("manifest_rel", [".cursor-plugin/plugin.json", ".Cursor-Plugin/plugin.json"])
def test_cursor_undotted_roots_become_valid_claude_paths(
    tmp_path: Path, manifest_rel: str, skills: Any, expected: Any
) -> None:
    assert ClaudeCodeAdapter().manifest(_source(tmp_path, manifest_rel, skills))["skills"] == expected


@pytest.mark.parametrize("skills", ["skills", ["portable", "./other"], [".", "portable"], "./"])
def test_openclaw_declares_no_skill_paths_to_claude(tmp_path: Path, skills: Any) -> None:
    _tree(tmp_path, {"skills/implicit/SKILL.md": _skill("implicit"), "portable/a/SKILL.md": _skill("a")})
    source = _source(tmp_path, MANIFEST, skills)
    assert "skills" not in ClaudeCodeAdapter().manifest(source)
    assert ClaudeCodeAdapter()._declared_skill_dirs(source) == []
    assert ClaudeCodeAdapter().staged_skills(source) == []


@pytest.mark.parametrize(
    "skills",
    ["extras", ["extras"], ["./"], ["../escape"], [7], "./", []],
)
def test_codex_values_codex_rejects_never_reach_claude(tmp_path: Path, skills: Any) -> None:
    source = _source(tmp_path, ".codex-plugin/plugin.json", skills)
    assert "skills" not in ClaudeCodeAdapter().manifest(source)
    assert ClaudeCodeAdapter()._declared_skill_dirs(source) == []


@pytest.mark.parametrize(
    ("skills", "expected"),
    [
        ("./skills/core", "./skills/core"),
        (["./skills/core", "extras"], ["./skills/core"]),
        (["./a", "./b"], ["./a", "./b"]),
    ],
)
def test_codex_keeps_only_the_paths_it_accepts_in_their_shape(tmp_path: Path, skills: Any, expected: Any) -> None:
    assert ClaudeCodeAdapter().manifest(_source(tmp_path, ".codex-plugin/plugin.json", skills))["skills"] == expected


def test_codex_ignored_extras_are_not_enabled_in_the_staged_census(tmp_path: Path) -> None:
    _tree(tmp_path, {"skills/core/SKILL.md": _skill("core"), "extras/bonus/SKILL.md": _skill("bonus")})
    source = _source(tmp_path, ".codex-plugin/plugin.json", ["./skills/core", "extras"])
    staged = {rel for _name, rel, _copy in ClaudeCodeAdapter().staged_skills(source)}
    assert staged == {"skills/core"}


@pytest.mark.parametrize("manifest_rel", [".claude-plugin/plugin.json", "plugin.json"])
@pytest.mark.parametrize("skills", ["extras", ["extras"], ["../escape"], [7], None, []])
def test_other_formats_keep_their_declarations_unrepaired(tmp_path: Path, manifest_rel: str, skills: Any) -> None:
    assert ClaudeCodeAdapter().manifest(_source(tmp_path, manifest_rel, skills))["skills"] == skills


def _blocking(inventory) -> list:
    from skillevaluator.models.result import Severity

    return [finding for finding in inventory.findings if finding.severity in {Severity.HIGH, Severity.CRITICAL}]


@pytest.mark.parametrize(
    ("server", "check"),
    [
        ({"url": "https://${HOST}/mcp"}, "mcp_url_env_not_expanded"),
        ({"url": "https://mcp.example.com/mcp", "sslVerify": False}, "mcp_insecure_tls_config"),
        ({"url": "https://mcp.example.com/mcp", "ssl_verify": False}, "mcp_insecure_tls_config"),
    ],
)
def test_blocking_mcp_findings_refuse_preparation_under_the_callers_policy(
    tmp_path: Path, server: dict, check: str
) -> None:
    inventory = plugin_inventory_for_root(_openclaw(tmp_path, mcpServers={"remote": server}))
    assert inventory is not None
    blocking = _blocking(inventory)
    assert check in {finding.check_name for finding in blocking}
    with pytest.raises(PreparationError, match=check):
        prepare_runtime_package(inventory, blocking_findings=blocking)


def test_literal_enterprise_secret_refuses_preparation_under_the_callers_policy(tmp_path: Path) -> None:
    # Assembled at run time so the full-history secret scan never sees a literal credential.
    sample = "s3cr3t-Value-" + "0123456789abcdef" + "ABCDEF"
    server = {"url": "https://mcp.example.com/mcp", "enterprise": {"idpClientSecret": sample}}
    inventory = plugin_inventory_for_root(_openclaw(tmp_path, mcpServers={"remote": server}))
    assert inventory is not None
    blocking = _blocking(inventory)
    assert blocking and all(sample not in finding.message for finding in inventory.findings)
    with pytest.raises(PreparationError, match="plugin policy blocks runtime evaluation"):
        prepare_runtime_package(inventory, blocking_findings=blocking)
