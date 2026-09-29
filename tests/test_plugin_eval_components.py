# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 plugin staging for every mcpServers form, plus report-only component coverage."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest

from skillevaluator.tier3.plugin_eval import PLUGIN_MCP_SERVERS_FILENAME, prepare_plugin_eval_package

_SKIP_SYMLINKS = pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
_PINNED = {"command": "npx", "args": ["-y", "@scope/fs@1.2.3"]}


def _plugin(root: Path, manifest: dict, files: dict[str, str | dict | list] | None = None) -> Path:
    (root / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "demo", **manifest}), encoding="utf-8")
    evals = root / "evals"
    evals.mkdir(exist_ok=True)
    (evals / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p", "expected_output": "o"}]))
    for rel, content in (files or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _staged_servers(package) -> dict[str, dict]:
    toml_path = package.package_path / "evals" / "environment" / PLUGIN_MCP_SERVERS_FILENAME
    return {server["name"]: server for server in tomllib.loads(toml_path.read_text())["mcp_servers"]}


def _prepare(root: Path, tmp_path: Path):
    return prepare_plugin_eval_package(root, stage_root=tmp_path / "stage")


# --------------------------------------------------------------------------- #
# Every mcpServers form is staged                                             #
# --------------------------------------------------------------------------- #
def test_inline_map_is_staged(tmp_path: Path) -> None:
    package = _prepare(_plugin(tmp_path / "p", {"mcpServers": {"fs": _PINNED}}), tmp_path)
    assert _staged_servers(package)["fs"]["args"] == ["-y", "@scope/fs@1.2.3"]


@pytest.mark.parametrize("wrapped", [True, False])
def test_path_ref_is_staged(tmp_path: Path, wrapped: bool) -> None:
    servers = {"remote": {"url": "https://mcp.example.com/mcp", "transport": "http"}}
    root = _plugin(
        tmp_path / "p",
        {"mcpServers": "./config/servers.json"},
        {"config/servers.json": {"mcpServers": servers} if wrapped else servers},
    )
    staged = _staged_servers(_prepare(root, tmp_path))
    assert staged["remote"] == {"name": "remote", "url": "https://mcp.example.com/mcp", "transport": "http"}


def test_array_form_is_staged(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "p",
        {"mcpServers": ["./a.json", {"inline-srv": {"command": "node", "args": ["./s.js"]}}]},
        {"a.json": {"from-file": _PINNED}},
    )
    package = _prepare(root, tmp_path)
    assert set(_staged_servers(package)) == {"from-file", "inline-srv"}
    assert package.runnable_mcp_servers == ("from-file", "inline-srv")


def test_absent_mcp_servers_stages_root_mcp_json(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "p", {}, {".mcp.json": {"mcpServers": {"fs": _PINNED, "prov": {"provider": "pub"}}}})
    package = _prepare(root, tmp_path)
    assert set(_staged_servers(package)) == {"fs"}
    assert package.unresolved_mcp_servers == ("prov",)


def test_later_declaration_replaces_mcp_json_server(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "p",
        {"mcpServers": {"fs": {"command": "node", "args": ["./local.js"]}}},
        {".mcp.json": {"mcpServers": {"fs": _PINNED}}},
    )
    staged = _staged_servers(_prepare(root, tmp_path))
    assert staged["fs"]["command"] == "node"


# --------------------------------------------------------------------------- #
# Fail closed before staging                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("manifest", "files", "match"),
    [
        ({"mcpServers": "../outside.json"}, {}, "plugin_component_path_escape"),
        ({"mcpServers": "/etc/mcp.json"}, {}, "plugin_component_path_escape"),
        ({"mcpServers": "./missing.json"}, {}, "plugin_component_path_missing"),
        ({"mcpServers": "./bad.json"}, {"bad.json": "{nope"}, "mcp_config_file_invalid"),
        ({"mcpServers": 42}, {}, "mcp_servers_not_object"),
        ({"mcpServers": [3]}, {}, "mcp_servers_entry_invalid"),
        ({}, {".mcp.json": {"evil": {"command": "sh", "args": ["-c", "run"]}}}, "mcp_command_dangerous_form"),
        (
            {"mcpServers": "./m.json"},
            {"m.json": {"s": {"command": "claude", "args": ["--dangerously-skip-permissions"]}}},
            "mcp_permission_bypass_flag",
        ),
        ({}, {".mcp.json": {"s": {"command": "srv", "env": {"LD_PRELOAD": "/x.so"}}}}, "mcp_env_code_injection"),
        ({"mcpServers": {"m": {"url": "https://169.254.169.254/"}}}, {}, "mcp_endpoint_metadata"),
    ],
)
def test_unsafe_mcp_sources_fail_closed(tmp_path: Path, manifest: dict, files: dict, match: str) -> None:
    root = _plugin(tmp_path / "p", manifest, files)
    with pytest.raises(ValueError, match=match):
        _prepare(root, tmp_path)
    assert not (tmp_path / "stage").exists() or not any((tmp_path / "stage").rglob(PLUGIN_MCP_SERVERS_FILENAME))


def test_shadowed_unsafe_declaration_still_fails_closed(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "p",
        {"mcpServers": {"fs": _PINNED}},
        {".mcp.json": {"mcpServers": {"fs": {"command": "sh", "args": ["-c", "run"]}}}},
    )
    with pytest.raises(ValueError, match="mcp_command_dangerous_form"):
        _prepare(root, tmp_path)


def test_oversize_mcp_config_fails_closed(tmp_path: Path) -> None:
    from skillevaluator.constants import PLUGIN_CONFIG_MAX_BYTES

    root = _plugin(tmp_path / "p", {"mcpServers": "./big.json"}, {"big.json": " " * (PLUGIN_CONFIG_MAX_BYTES + 1)})
    with pytest.raises(ValueError, match="mcp_config_file_too_large"):
        _prepare(root, tmp_path)


@_SKIP_SYMLINKS
def test_symlinked_mcp_config_fails_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"x": _PINNED}), encoding="utf-8")
    root = _plugin(tmp_path / "p", {"mcpServers": "./cfg/sub/link.json"})
    (root / "cfg" / "sub").mkdir(parents=True)
    (root / "cfg" / "sub" / "link.json").symlink_to(outside)
    with pytest.raises(ValueError, match="plugin_component_path_unsafe"):
        _prepare(root, tmp_path)


@_SKIP_SYMLINKS
def test_symlinked_root_mcp_json_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"x": _PINNED}), encoding="utf-8")
    root = _plugin(tmp_path / "p", {})
    (root / ".mcp.json").symlink_to(outside)
    with pytest.raises(ValueError):
        _prepare(root, tmp_path)


def test_medium_findings_do_not_block_staging(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "p",
        {
            "mcpServers": {
                "u": {"command": "uvx", "args": ["tool"], "autoApprove": ["*"]},
                "l": {"url": "https://localhost/x"},
            }
        },
    )
    assert set(_staged_servers(_prepare(root, tmp_path))) == {"u", "l"}


# --------------------------------------------------------------------------- #
# Report-only provenance: component_coverage / context_cost / mcp_pinning      #
# --------------------------------------------------------------------------- #
def test_provenance_reports_coverage_cost_and_pinning(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "p",
        {
            "mcpServers": ["./srv.mcpb", {"fs": _PINNED, "u": {"command": "uvx", "args": ["tool"]}}],
            "skills": ["./extra/"],
        },
        {
            ".mcp.json": {"prov": {"provider": "public-provider"}},
            "srv.mcpb": "bundle",
            "skills/demo/SKILL.md": "---\nname: demo\ndescription: Demo skill\n---\n# Demo\nBody\n",
            "extra/other/SKILL.md": "---\nname: other\ndescription: Other\n---\nBody\n",
            "rules/style.md": "Be concise.",
            "agents/a.md": "---\ndescription: Agent\n---\nPrompt\n",
            "hooks/hooks.json": {"hooks": {}},
        },
    )
    provenance = _prepare(root, tmp_path).provenance()
    coverage = provenance["component_coverage"]
    states = {(row["type"], row["name"]): row["state"] for row in coverage["components"]}
    assert states[("skill", "demo")] == "staged"
    assert states[("skill", "other")] == "not_staged"
    assert states[("rule", "style.md")] == "staged"
    assert states[("mcp", "fs")] == "staged"
    assert states[("mcp", "prov")] == "unavailable"
    assert states[("mcp", "./srv.mcpb")] == "unsupported"
    assert states[("agent", "a")] == "unsupported"
    assert states[("hook", "hooks/hooks.json")] == "unsupported"
    assert coverage["not_evaluated"] == sum(1 for state in states.values() if state != "staged")
    assert sum(coverage["counts"].values()) == len(coverage["components"])
    for row in coverage["components"]:
        assert set(row) == {"type", "name", "origin", "path", "state", "reason"} and row["reason"]
    assert provenance["mcp_pinning"] == {"total": 3, "pinned": 1, "unpinned": 1, "not_applicable": 1, "ratio": 0.5}
    cost = provenance["context_cost"]
    assert cost["method"] == "static_estimate" and cost["estimator"] == "chars_div_4"
    assert cost["always_on_tokens"] > 0 and cost["on_demand_tokens"] > 0
    assert any("wrapper SKILL.md" in note for note in cost["notes"])
    # Unsupported types are reported, not gated: 'partial' still reflects only the
    # existing provider-only MCP rule.
    assert provenance["partial"] is True


def test_unsupported_components_do_not_mark_run_partial(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "p",
        {"mcpServers": {"fs": _PINNED}},
        {"agents/a.md": "x", "commands/c.md": "y", "settings.json": {"agent": "a"}},
    )
    provenance = _prepare(root, tmp_path).provenance()
    assert provenance["partial"] is False
    assert provenance["component_coverage"]["counts"]["unsupported"] == 3


def test_invalid_declared_component_is_reported_invalid(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "p", {"mcpServers": {"fs": _PINNED}, "agents": "./missing.md"})
    rows = _prepare(root, tmp_path).provenance()["component_coverage"]["components"]
    invalid = [row for row in rows if row["state"] == "invalid"]
    assert invalid == [
        {
            "type": "agent",
            "name": "./missing.md",
            "origin": "declared",
            "path": "missing.md",
            "state": "invalid",
            "reason": "declared path does not exist",
        }
    ]


def test_skipped_package_still_reports_coverage(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "p", {"mcpServers": {"prov": {"provider": "public-provider"}}}, {"agents/a.md": "x"})
    package = _prepare(root, tmp_path)
    assert package.skipped
    states = {row["name"]: row["state"] for row in package.provenance()["component_coverage"]["components"]}
    assert states == {"prov": "unavailable", "a": "unsupported"}


def test_bundle_manifest_root_mcp_json_is_not_staged(tmp_path: Path) -> None:
    root = tmp_path / "b"
    (root / "skills" / "alpha" / "evals").mkdir(parents=True)
    (root / "agent_plugin.yaml").write_text(
        "name: bundle\nauthor:\n  email: a@example.com\nmcp:\n  - name: search\n    provider: public-provider\n",
        encoding="utf-8",
    )
    (root / "skills" / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: A\n---\nA\n")
    (root / "skills" / "alpha" / "evals" / "evals.json").write_text(json.dumps([{"id": "c", "prompt": "p"}]))
    (root / ".mcp.json").write_text(json.dumps({"local": _PINNED}), encoding="utf-8")
    package = _prepare(root, tmp_path)
    assert package.runnable_mcp_servers == ()
    states = {
        row["name"]: (row["state"], row["reason"]) for row in package.provenance()["component_coverage"]["components"]
    }
    assert states["local"][0] == "not_staged"
    assert "'mcp' list" in states["local"][1]
    assert states["search"][0] == "unavailable"
