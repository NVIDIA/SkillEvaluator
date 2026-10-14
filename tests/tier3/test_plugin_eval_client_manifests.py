# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 coverage and the dependency audit see every client manifest a plugin ships.

A client manifest over SkillEvaluator's 1 MiB read bound or not UTF-8 is still
loaded by its client (Codex reads any size, Claude Code reads Latin-1), so its
hooks and MCP servers must show in Tier 3 coverage and reach the CVE audit.
Components that another client loads through its own defaults name that client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.validators.dependencies import DependencySecurityValidator

_SKILL = "---\nname: alpha\ndescription: Alpha skill.\n---\nDo alpha.\n"
_EVALS = {"skill_name": "demo", "evals": [{"id": "c1", "prompt": "Do alpha.", "expected_output": "Done."}]}
_HOOKS = {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo checked"}]}]}
_SERVERS = {"hidden-srv": {"command": "npx", "args": ["-y", "@example/hidden-mcp@1.2.3"]}}


def _padded(data: dict[str, Any], size: int = CONTENT_DEDUP_MAX_FILE_BYTES + 4096) -> bytes:
    text = json.dumps(data)
    return (text[:-1] + " " * (size - len(text)) + "}").encode()


def _latin1(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("latin-1")


def _plugin(root: Path, files: dict[str, Any]) -> Path:
    for rel, content in {"skills/alpha/SKILL.md": _SKILL, "evals/evals.json": _EVALS, **files}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _rows(package: Any) -> dict[tuple[str, str], dict[str, Any]]:
    coverage = package.provenance()["component_coverage"]
    return {(row["type"], row["name"]): row for row in coverage["components"]}


_CODEX_MANIFEST = {
    "name": "demo",
    "version": "1.0.0",
    "description": "Demo",
    "author": {"name": "Example"},
    "mcpServers": _SERVERS,
    "hooks": {"hooks": _HOOKS},
}
# Codex reads a manifest of any size; a Latin-1 byte makes it invalid UTF-8.
_UNREADABLE = {
    "oversize-codex": (".codex-plugin/plugin.json", lambda: _padded(_CODEX_MANIFEST)),
    "latin1-codex": (".codex-plugin/plugin.json", lambda: _latin1({**_CODEX_MANIFEST, "description": "caf\xe9"})),
}


@pytest.mark.parametrize("case", sorted(_UNREADABLE))
def test_tier3_coverage_lists_what_an_unreadable_additional_manifest_declares(tmp_path: Path, case: str) -> None:
    rel, content = _UNREADABLE[case]
    plugin = _plugin(tmp_path / "demo", {".claude-plugin/plugin.json": {"name": "demo"}, rel: content()})

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    rows = _rows(package)
    server = rows[("mcp", "hidden-srv")]
    assert server["state"] == "not_staged"
    assert "declared only by the additional manifest .codex-plugin/plugin.json" in server["reason"]
    assert any(kind == "hook" and row["state"] in {"not_staged", "unsupported"} for (kind, _), row in rows.items())
    # The additional manifest's server is never staged into the run.
    assert "hidden-srv" not in package.runnable_mcp_servers


@pytest.mark.parametrize("case", sorted(_UNREADABLE))
def test_dependency_audit_reads_an_unreadable_additional_manifests_mcp_servers(tmp_path: Path, case: str) -> None:
    rel, content = _UNREADABLE[case]
    plugin = _plugin(tmp_path / "demo", {".claude-plugin/plugin.json": {"name": "demo"}, rel: content()})

    declarations = DependencySecurityValidator()._mcp_declarations(plugin)

    assert [(declaration.name, declaration.file) for declaration in declarations] == [
        ("hidden-srv", ".codex-plugin/plugin.json")
    ]


def test_cross_client_coverage_rows_name_the_client_that_loads_them(tmp_path: Path) -> None:
    # A Cursor folder without .claude-plugin: Claude Code --plugin-dir loads .mcp.json from its defaults.
    plugin = _plugin(
        tmp_path / "demo",
        {
            ".cursor-plugin/plugin.json": {"name": "demo", "version": "1.0.0"},
            ".mcp.json": {"mcpServers": _SERVERS},
        },
    )

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    reason = _rows(package)[("mcp", "hidden-srv")]["reason"]
    assert reason.startswith("loaded only by Claude Code (--plugin-dir reads its default locations")
    assert "additional manifest" not in reason
    assert [declaration.name for declaration in DependencySecurityValidator()._mcp_declarations(plugin)] == [
        "hidden-srv"
    ]
