# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`validate --resolve-endpoints` is opt-in and reaches the plugin schema validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.cli import cli
from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators import endpoint_resolution as er


def _plugin(root: Path) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "demo", "mcpServers": {"remote": {"url": "https://mcp.example.com/mcp", "type": "http"}}})
    )
    return root


@pytest.mark.parametrize(("flag", "expected"), [((), False), (("--resolve-endpoints",), True)])
def test_validate_forwards_resolve_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: tuple[str, ...], expected: bool
) -> None:
    captured: list[Any] = []

    def _run_validation(_target: Path, **kwargs: Any) -> list:
        captured.append(kwargs.get("resolve_endpoints"))
        return []

    monkeypatch.setattr(cli_module, "run_validation", _run_validation)
    monkeypatch.setattr(cli_module, "emit_reports", lambda *_args, **_kwargs: True)
    result = CliRunner().invoke(
        cli,
        ["validate", str(_plugin(tmp_path / "p")), "--tiers", "1", "--no-llm", "-o", str(tmp_path / "out"), *flag],
    )
    assert result.exit_code == 0, result.output
    assert captured == [expected]


def test_run_validation_resolves_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resolved: list[str] = []

    def _resolve(host: str, _port: int, _timeout: float) -> list[str]:
        resolved.append(host)
        return ["10.2.3.4"]

    monkeypatch.setattr(er, "_resolve", _resolve)
    monkeypatch.setattr(er, "_head", lambda *_args: pytest.fail("a non-public host must never be contacted"))
    root = _plugin(tmp_path / "p")

    [default] = run_validation(root, checks="schema", content_type=CONTENT_TYPE_PLUGIN)
    assert resolved == []
    assert "endpoint_resolution" not in default.metadata["plugin"]

    [opted_in] = run_validation(root, checks="schema", content_type=CONTENT_TYPE_PLUGIN, resolve_endpoints=True)
    assert resolved == ["mcp.example.com"]
    assert "endpoint_resolves_private" in {f.check_name for f in opted_in.findings}


def test_resolve_endpoints_covers_mcp_servers_of_additional_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an MCP server that only an additional manifest declares was never DNS/redirect-checked."""
    captured: list[er.EndpointTarget] = []

    def _check(_self: er.EndpointChecker, targets: Any) -> tuple[dict[str, Any], list]:
        captured.extend(targets)
        return {"enabled": True, "endpoints": [], "counts": {}}, []

    monkeypatch.setattr(er.EndpointChecker, "check", _check)
    root = tmp_path / "p"
    shared = "https://shared.example.com/mcp"
    files = {
        ".claude-plugin/plugin.json": {"name": "demo"},
        ".mcp.json": {"mcpServers": {"shared": {"type": "http", "url": shared}}},
        "plugin.json": {"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "demo"},
        "mcp.json": {
            "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
            "mcpServers": {
                "shared": {"type": "streamable-http", "url": shared},
                "remote": {"type": "streamable-http", "url": "https://attacker.example.net/mcp"},
            },
        },
    }
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(content))

    [result] = run_validation(root, checks="schema", content_type=CONTENT_TYPE_PLUGIN, resolve_endpoints=True)

    assert [(target.kind, target.name, target.url) for target in captured] == [
        ("mcp", "shared", shared),
        ("mcp", "remote", "https://attacker.example.net/mcp"),
    ]
    assert captured[1].file_path == str(root / "mcp.json")
    assert result.metadata["plugin"]["endpoint_resolution"]["enabled"] is True
