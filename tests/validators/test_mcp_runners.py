# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One reader for MCP package-runner argv: pinning, the container-image lookup, and the CVE audit agree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Severity
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.mcp_static import (
    RunnerInvocation,
    classify_mcp_pinning,
    mcp_container_image,
    parse_mcp_runner,
    validate_mcp_server_declaration,
)

_DIGEST = "sha256:" + "a" * 64


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"command": "npx", "args": ["-y", "pkg@1.2.3", "--port", "8080"]}, ("npm", "npx", ("pkg@1.2.3",))),
        ({"command": "npx", "args": ["-p", "a@1.0.0", "--package=b", "cmd"]}, ("npm", "npx", ("a@1.0.0", "b"))),
        ({"command": "npx -y pkg"}, ("npm", "npx", ("pkg",))),
        ({"command": "C:\\tools\\bunx.exe", "args": ["pkg"]}, ("npm", "bunx", ("pkg",))),
        ({"command": "pnpm", "args": ["dlx", "pkg"]}, ("npm", "pnpm dlx", ("pkg",))),
        ({"command": "yarn", "args": ["dlx", "pkg"]}, ("npm", "yarn dlx", ("pkg",))),
        ({"command": "npm", "args": ["exec", "--", "pkg"]}, ("npm", "npm exec", ("pkg",))),
        ({"command": "npx", "args": ["-y"]}, ("npm", "npx", ())),
        (
            {"command": "uvx", "args": ["--with", "a==1.0,b", "--with=c", "pkg==2.0"]},
            ("pypi", "uvx", ("pkg==2.0", "a==1.0", "b", "c")),
        ),
        ({"command": "uvx", "args": ["--from", "pkg==2.0", "cmd"]}, ("pypi", "uvx", ("pkg==2.0",))),
        ({"command": "uvx", "args": ["--with", "a"]}, ("pypi", "uvx", ())),
        ({"command": "uv", "args": ["tool", "run", "--with", "a", "pkg"]}, ("pypi", "uv tool run", ("pkg", "a"))),
        ({"command": "pipx", "args": ["run", "--spec", "pkg==2.0", "cmd"]}, ("pypi", "pipx run", ("pkg==2.0",))),
        ({"command": "deno", "args": ["run", "-A", "npm:pkg@1.2.3"]}, ("deno", "deno run", ("npm:pkg@1.2.3",))),
        (
            {"command": "docker", "args": ["run", "-i", "--rm", "-e", "X=1", "img:1.2.3"]},
            ("container", "docker run", ("img:1.2.3",)),
        ),
        ({"command": "podman", "args": ["container", "run", "img"]}, ("container", "podman run", ("img",))),
        ({"command": "nerdctl", "args": ["run"]}, ("container", "nerdctl run", ())),
    ],
)
def test_runner_argv_is_read_the_way_the_runner_reads_it(config: dict[str, Any], expected: tuple) -> None:
    invocation = parse_mcp_runner(config)
    assert invocation is not None
    assert (invocation.ecosystem, invocation.runner, invocation.specs) == expected


@pytest.mark.parametrize(
    "config",
    [
        {"command": "node", "args": ["./server.js"]},
        {"command": "docker", "args": ["compose", "up"]},
        {"command": "pnpm", "args": ["install"]},
        {"command": "uv", "args": ["pip", "install", "pkg"]},
        {"command": "  "},
        {"url": "https://mcp.example.com/mcp"},
        {"provider": "public-provider"},
        "not-an-object",
    ],
)
def test_declarations_that_run_no_package_runner(config: Any) -> None:
    assert parse_mcp_runner(config) is None


def test_npm_specs_include_deno_npm_modules_only() -> None:
    assert RunnerInvocation("deno", "deno run", ("npm:pkg@1.2.3",)).npm_specs == ("pkg@1.2.3",)
    assert RunnerInvocation("deno", "deno run", ("jsr:@std/http@1.0.0",)).npm_specs == ()
    assert RunnerInvocation("npm", "npx", ("a", "b")).npm_specs == ("a", "b")
    assert RunnerInvocation("pypi", "uvx", ("a",)).npm_specs == ()


@pytest.mark.parametrize(
    "config",
    [
        {"command": "docker", "args": ["run", "-i", "--rm", "--name", "x", "ghcr.io/o/img@" + _DIGEST]},
        {"command": "podman run -p 8080:80 img:1.2.3"},
        {"command": "npx", "args": ["-y", "pkg@1.2.3"]},
        {"command": "docker", "args": ["compose", "up"]},
    ],
)
def test_container_image_lookup_reads_the_same_invocation(config: dict[str, Any]) -> None:
    invocation = parse_mcp_runner(config)
    expected = invocation.specs[0] if invocation is not None and invocation.ecosystem == "container" else None
    assert mcp_container_image(config) == expected


# --------------------------------------------------------------------------- #
# uvx --with: every installed requirement must be exact                       #
# --------------------------------------------------------------------------- #
def test_floating_uvx_with_requirement_is_unpinned() -> None:
    """Regression: ``uvx --with foo pkg==1.0`` was pinned although uv installs a floating ``foo``."""
    pin = classify_mcp_pinning({"command": "uvx", "args": ["--with", "foo", "pkg==1.0"]})
    assert pin.status == "unpinned"
    assert pin.detail == "uvx: package 'foo' has no version (resolves to the latest release)"
    findings = validate_mcp_server_declaration("s", {"command": "uvx", "args": ["--with", "foo", "pkg==1.0"]}, "p")
    assert [(f.check_name, f.severity) for f in findings] == [("mcp_unpinned_package", Severity.MEDIUM)]


@pytest.mark.parametrize(
    "args",
    [
        ["--with", "foo==2.0", "pkg==1.0"],
        ["--with", "foo==2.0,bar==3.1", "pkg==1.0"],
        ["--with", "./local-plugin", "pkg==1.0"],
        ["--from", "pkg==1.0", "--with", "foo@2.0.1", "cmd"],
    ],
)
def test_exact_uvx_with_requirements_keep_the_server_pinned(args: list[str]) -> None:
    assert classify_mcp_pinning({"command": "uvx", "args": args}).status == "pinned"


def test_uv_tool_run_reads_with_requirements_too() -> None:
    pin = classify_mcp_pinning({"command": "uv", "args": ["tool", "run", "--with", "foo>=1", "pkg==1.0"]})
    assert (pin.status, pin.detail) == (
        "unpinned",
        "uv tool run: requirement 'foo>=1' is a range or tag, not an exact '==' version",
    )


# --------------------------------------------------------------------------- #
# The audit reads the same specs                                              #
# --------------------------------------------------------------------------- #
class _FakeTool:
    def __init__(self, name: str, responses: list[ToolResult] | None = None, *, available: bool = True) -> None:
        self.name = name
        self.responses = list(responses or [])
        self.available = available
        self.calls: list[dict[str, Any]] = []

    @property
    def is_available(self) -> bool:
        return self.available

    def get_install_hint(self) -> str:
        return f"install {self.name}"

    def run(self, args: list[str], **kwargs: Any) -> ToolResult:
        cwd = kwargs.get("cwd")
        files = {path.name: path.read_text() for path in Path(cwd).iterdir() if path.is_file()} if cwd else {}
        self.calls.append({"args": list(args), "files": files})
        return self.responses.pop(0) if self.responses else ToolResult(True, '{"dependencies": []}', "", 0)


@pytest.fixture
def pip_audit(monkeypatch: pytest.MonkeyPatch) -> _FakeTool:
    for name in ("osv_scanner", "npm", "grype", "trivy", "safety"):
        monkeypatch.setattr(Tools, name, _FakeTool(name, available=False))
    fake = _FakeTool("pip-audit")
    monkeypatch.setattr(Tools, "pip_audit", fake)
    return fake


def _audit(root: Path, servers: dict[str, Any]) -> Any:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "demo"}))
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": servers}))
    [result] = run_validation(root, checks="dependency", content_type=CONTENT_TYPE_PLUGIN)
    return result


def test_audit_and_pinning_read_the_same_uvx_requirements(tmp_path: Path, pip_audit: _FakeTool) -> None:
    server = {"command": "uvx", "args": ["--with", "foo", "--with", "bar==2.0", "pkg==1.0"]}
    result = _audit(tmp_path / "demo", {"srv": server})

    [call] = pip_audit.calls
    assert call["files"] == {"requirements-0.txt": "bar==2.0\npkg==1.0\n"}
    unverified = [
        f.metadata["package_name"] for f in result.findings if f.check_name == "dependency-version-unverified"
    ]
    assert unverified == ["foo"]
    assert classify_mcp_pinning(server).detail == "uvx: package 'foo' has no version (resolves to the latest release)"
