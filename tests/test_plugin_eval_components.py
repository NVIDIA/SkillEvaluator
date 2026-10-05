# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 plugin staging for every mcpServers form, plus report-only component coverage."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest

from skillevaluator.constants import PLUGIN_CONFIG_MAX_BYTES
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
        {"mcpServers": ["./a.json", {"inline-srv": {"command": "node", "args": ["/opt/mcp/s.js"]}}]},
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
        {"mcpServers": {"fs": {"command": "node", "args": ["/opt/mcp/local.js"]}}},
        {".mcp.json": {"mcpServers": {"fs": _PINNED}}},
    )
    staged = _staged_servers(_prepare(root, tmp_path))
    assert staged["fs"]["command"] == "node"


# --------------------------------------------------------------------------- #
# Servers launched from plugin files cannot start, so they are not runnable    #
# --------------------------------------------------------------------------- #
_SKILL_MD = "---\nname: demo\ndescription: Demo skill\n---\n# Demo\nBody\n"
_PLUGIN_FILE_LAUNCHES = [
    {"command": "${CLAUDE_PLUGIN_ROOT}/servers/db-server", "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/c.json"]},
    {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/server.js"]},
    {"command": "python", "args": ["-m", "srv", "--data=$CLAUDE_PLUGIN_DATA/db"]},
    {"command": "python", "args": ["-m", "srv"], "cwd": "${CLAUDE_PLUGIN_ROOT}"},
    {"command": "node", "args": ["./s.js"]},
    {"command": "srv", "args": ["--config=../shared/config.json"]},
    {"command": "servers/db-server"},
]


def _mcp_row(provenance: dict, name: str) -> dict:
    return next(
        row for row in provenance["component_coverage"]["components"] if row["type"] == "mcp" and row["name"] == name
    )


def test_plugin_root_mcp_json_only_plugin_is_skipped_not_run(tmp_path: Path) -> None:
    # The documented Claude Code layout: the server binary ships in the plugin and
    # .mcp.json launches it through ${CLAUDE_PLUGIN_ROOT}. Tier 3 stages neither the
    # binary nor the variable, so this must stay an honest skip, not a "complete"
    # run of a server that cannot start.
    root = _plugin(
        tmp_path / "p",
        {},
        {
            ".mcp.json": {"mcpServers": {"plugin-database": _PLUGIN_FILE_LAUNCHES[0]}},
            "servers/db-server": "#!/bin/sh\necho hi\n",
            "c.json": "{}",
        },
    )
    package = _prepare(root, tmp_path)
    assert package.skipped
    assert "launched from unstaged plugin files" in package.skip_reason
    assert package.runnable_mcp_servers == ()
    provenance = package.provenance()
    assert provenance["partial"] is True
    assert provenance["mcp_unsupported_config"] == ["plugin-database"]
    assert _mcp_row(provenance, "plugin-database")["state"] == "unsupported"
    assert not (tmp_path / "stage").exists() or not any((tmp_path / "stage").rglob(PLUGIN_MCP_SERVERS_FILENAME))


@pytest.mark.parametrize(
    "config",
    _PLUGIN_FILE_LAUNCHES,
    ids=["root-command", "root-arg", "data-flag", "root-cwd", "dot-arg", "parent-flag", "relative-command"],
)
def test_plugin_file_launch_is_not_staged_and_marks_run_incomplete(tmp_path: Path, config: dict) -> None:
    root = _plugin(
        tmp_path / "p",
        {"mcpServers": {"fs": _PINNED, "local": config}},
        {"skills/demo/SKILL.md": _SKILL_MD},
    )
    package = _prepare(root, tmp_path)
    assert not package.skipped
    assert package.runnable_mcp_servers == ("fs",)
    assert set(_staged_servers(package)) == {"fs"}
    toml_text = (package.package_path / "evals" / "environment" / PLUGIN_MCP_SERVERS_FILENAME).read_text()
    assert "CLAUDE_PLUGIN" not in toml_text
    provenance = package.provenance()
    assert provenance["partial"] is True
    assert provenance["mcp_unsupported_config"] == ["local"]
    row = _mcp_row(provenance, "local")
    assert row["state"] == "unsupported" and "does not stage" in row["reason"]


def test_bundle_manifest_plugin_file_launch_is_not_staged(tmp_path: Path) -> None:
    root = tmp_path / "b"
    (root / "skills" / "alpha" / "evals").mkdir(parents=True)
    (root / "agent_plugin.yaml").write_text(
        "name: bundle\nauthor:\n  email: a@example.com\n"
        "mcp:\n  - name: local\n    command: ${CLAUDE_PLUGIN_ROOT}/bin/srv\n",
        encoding="utf-8",
    )
    (root / "skills" / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: A\n---\nA\n")
    (root / "skills" / "alpha" / "evals" / "evals.json").write_text(json.dumps([{"id": "c", "prompt": "p"}]))
    # agent_plugin.yaml mcp entries take only name and provider, as the Tier 1 schema says (proof M37),
    # so a runnable entry is refused before anything is staged.
    with pytest.raises(ValueError, match="name and provider"):
        _prepare(root, tmp_path)


def test_working_directory_and_absolute_paths_stay_runnable(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path / "p",
        {
            "mcpServers": {
                "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem@1.0.0", ".", "./"]},
                "abs": {"command": "/usr/local/bin/srv", "args": ["--config=/etc/srv.json"]},
            }
        },
    )
    package = _prepare(root, tmp_path)
    assert package.runnable_mcp_servers == ("files", "abs")
    assert package.provenance()["mcp_unsupported_config"] == []


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


_AUTHORED_PLUGIN_MCP = '[[mcp_servers]]\nname = "authored"\ncommand = "sh"\nargs = ["-c", "curl evil"]\n'


@pytest.mark.parametrize(
    ("manifest", "files"),
    [({"mcpServers": {"real": _PINNED}}, {}), ({}, {"skills/demo/SKILL.md": _SKILL_MD})],
    ids=["runnable-servers", "no-runnable-servers"],
)
def test_authored_plugin_mcp_servers_file_fails_closed(tmp_path: Path, manifest: dict, files: dict) -> None:
    # The plugin's own evals/ is the evals source; an authored copy of the generated
    # file would bypass the MCP checks and redaction in the with-plugin arm.
    files = {**files, f"evals/environment/{PLUGIN_MCP_SERVERS_FILENAME}": _AUTHORED_PLUGIN_MCP}
    root = _plugin(tmp_path / "p", manifest, files)
    with pytest.raises(ValueError, match=r"provides environment/plugin_mcp_servers\.toml"):
        _prepare(root, tmp_path)


@_SKIP_SYMLINKS
def test_dangling_plugin_mcp_servers_link_fails_closed(tmp_path: Path) -> None:
    from skillevaluator.tier3.plugin_eval import _write_plugin_mcp_servers_toml

    environment = tmp_path / "evals" / "environment"
    environment.mkdir(parents=True)
    (environment / PLUGIN_MCP_SERVERS_FILENAME).symlink_to(tmp_path / "missing.toml")
    with pytest.raises(ValueError, match=r"provides environment/plugin_mcp_servers\.toml"):
        _write_plugin_mcp_servers_toml(tmp_path / "evals", [])


def test_shared_mcp_servers_file_does_not_replace_plugin_servers(tmp_path: Path) -> None:
    shared = '[[mcp_servers]]\nname = "shared"\nurl = "https://mcp.example.com/mcp"\n'
    root = _plugin(tmp_path / "p", {"mcpServers": {"real": _PINNED}}, {"evals/environment/mcp_servers.toml": shared})
    package = _prepare(root, tmp_path)
    assert set(_staged_servers(package)) == {"real"}
    assert (package.package_path / "evals" / "environment" / "mcp_servers.toml").read_text() == shared


def test_oversize_mcp_config_fails_closed(tmp_path: Path) -> None:
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
    assert cost["method"] == "static_estimate" and cost["estimator"] == "chars_div_4_cjk"
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


def _bundle_plugin(root: Path, manifest_tail: str = "") -> Path:
    (root / "skills" / "alpha" / "evals").mkdir(parents=True)
    (root / "agent_plugin.yaml").write_text(
        f"name: bundle\nauthor:\n  email: a@example.com\n{manifest_tail}", encoding="utf-8"
    )
    (root / "skills" / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: A\n---\nA\n")
    (root / "skills" / "alpha" / "evals" / "evals.json").write_text(json.dumps([{"id": "c", "prompt": "p"}]))
    return root


@pytest.mark.parametrize("mcp_json", ["{not json", " " * (PLUGIN_CONFIG_MAX_BYTES + 1)], ids=["invalid", "oversize"])
@pytest.mark.parametrize(
    "manifest_tail", ["", "mcp:\n  - name: search\n    provider: public-provider\n"], ids=["no-mcp-list", "mcp-list"]
)
def test_bundle_manifest_broken_root_mcp_json_does_not_block(tmp_path: Path, mcp_json: str, manifest_tail: str) -> None:
    # An agent_plugin.yaml plugin never stages its root .mcp.json, so a stale or
    # malformed one is reported (coverage 'invalid'; Tier 1 flags it) but must not
    # block preparation -- with or without an unrelated 'mcp' list in the manifest.
    root = _bundle_plugin(tmp_path / "b", manifest_tail)
    (root / ".mcp.json").write_text(mcp_json, encoding="utf-8")
    package = _prepare(root, tmp_path)
    assert not package.skipped
    assert package.runnable_mcp_servers == ()
    assert _mcp_row(package.provenance(), ".mcp.json")["state"] == "invalid"


def test_staged_root_mcp_json_still_fails_closed(tmp_path: Path) -> None:
    # The same file blocks wherever it IS staged: a contained plugin's root
    # .mcp.json, or an agent_plugin.yaml mcpServers path that names it explicitly.
    contained = _plugin(tmp_path / "c", {}, {".mcp.json": "{not json"})
    with pytest.raises(ValueError, match="mcp_config_file_invalid"):
        _prepare(contained, tmp_path)
    bundle = _bundle_plugin(tmp_path / "b", "mcpServers: ./.mcp.json\n")
    (bundle / ".mcp.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="mcp_config_file_invalid"):
        _prepare(bundle, tmp_path)
