# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the MCP server and cookbook guides true to the code.

Their examples are copied by readers. These tests load the documented examples through
the real loaders, so an example cannot stop working without a test failing. Navigation
and links are checked by ``fern check`` in CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from click.testing import CliRunner

from skillevaluator.cli import cli

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FENCE = re.compile(r"^```(?P<lang>\w*)(?: title=\"(?P<title>[^\"]+)\")?\n(?P<body>.*?)\n```$", re.DOTALL | re.MULTILINE)


def _page_text(stem: str) -> str:
    return (DOCS / f"{stem}.mdx").read_text(encoding="utf-8")


def _fenced_block(stem: str, title: str) -> str:
    for match in FENCE.finditer(_page_text(stem)):
        if match["title"] == title:
            return match["body"]
    raise AssertionError(f"docs/{stem}.mdx has no code block titled {title!r}")


def test_mcp_guide_examples_load_through_the_real_validators(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.adapter import _load_mcp_servers, _validate_and_sanitize_custom_compose

    environment = tmp_path / "my-skill" / "evals" / "environment"
    (environment / "tool-api").mkdir(parents=True)
    (environment / "mcp_servers.toml").write_text(
        _fenced_block("mcp-servers", "evals/environment/mcp_servers.toml") + "\n", encoding="utf-8"
    )
    compose = environment / "docker-compose.yaml"
    compose.write_text(_fenced_block("mcp-servers", "evals/environment/docker-compose.yaml") + "\n", encoding="utf-8")

    servers = _load_mcp_servers(tmp_path / "my-skill", allowed_runtime_env={})
    _validate_and_sanitize_custom_compose(compose, allowed_env=set())
    sanitized = yaml.safe_load(compose.read_text(encoding="utf-8"))

    assert {server["name"]: server["transport"] for server in servers} == {
        "tool-api": "streamable-http",
        "local-tools": "stdio",
    }
    assert sanitized["services"]["main"] == {
        "depends_on": {"tool-api": {"condition": "service_healthy"}},
    }
    assert "healthcheck" in sanitized["services"]["tool-api"]


def test_cookbook_skill_passes_strict_validation(tmp_path: Path) -> None:
    skill = tmp_path / "svc-cli"
    (skill / "evals" / "environment").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: svc-cli\n"
        "description: Read project and usage information from the Service API with the service-cli tool. "
        "Use when the user asks which Service projects they can access.\n---\n\n# Service CLI\n\nUse service-cli.\n",
        encoding="utf-8",
    )
    for relative in ("environment/Dockerfile", "config.yml", "evals.json"):
        block = _fenced_block("cookbook-authenticated-cli", f"svc-cli/evals/{relative}")
        (skill / "evals" / relative).write_text(block + "\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict"])

    assert result.exit_code == 0, result.output
    dataset = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert [case["expected_skill"] for case in dataset["evals"]] == ["svc-cli", "svc-cli", None]
