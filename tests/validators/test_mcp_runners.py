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
from skillevaluator.validators import dependency_ecosystems as eco
from skillevaluator.validators.dependencies import _python_runner_declaration
from skillevaluator.validators.mcp_static import (
    RunnerInvocation,
    classify_mcp_pinning,
    exact_npm_version,
    exact_pypi_version,
    mcp_container_image,
    parse_mcp_runner,
    validate_mcp_server_declaration,
)
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

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
# A runner's options end at the package it runs                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("config", "specs"),
    [
        ({"command": "npx", "args": ["-y", "some-mcp@1.2.3", "-p", "3000"]}, ("some-mcp@1.2.3",)),
        ({"command": "npx", "args": ["-p", "a@1.0.0", "cmd", "--package", "b"]}, ("a@1.0.0",)),
        ({"command": "bunx", "args": ["pkg@1.2.3", "--package=other"]}, ("pkg@1.2.3",)),
        ({"command": "pnpx", "args": ["pkg@1.2.3", "-p", "3000"]}, ("pkg@1.2.3",)),
        ({"command": "pnpm", "args": ["dlx", "pkg@1.2.3", "--package", "x"]}, ("pkg@1.2.3",)),
        ({"command": "yarn", "args": ["dlx", "pkg@1.2.3", "-p", "x"]}, ("pkg@1.2.3",)),
        ({"command": "uvx", "args": ["srv==1.0", "--with", "x", "--from", "y"]}, ("srv==1.0",)),
        ({"command": "uv", "args": ["tool", "run", "srv==1.0", "--with=x"]}, ("srv==1.0",)),
        ({"command": "uvx", "args": ["--from", "srv==1.0", "cmd", "--with", "x"]}, ("srv==1.0",)),
        ({"command": "pipx", "args": ["run", "srv==1.0", "--spec", "other"]}, ("srv==1.0",)),
        ({"command": "deno", "args": ["run", "npm:pkg@1.2.3", "-c", "x.json"]}, ("npm:pkg@1.2.3",)),
        ({"command": "docker", "args": ["run", "-i", "img:1.2.3", "--name", "x", "-e", "Y=1"]}, ("img:1.2.3",)),
        ({"command": "podman", "args": ["run", "--rm", "img:1.2.3", "-v", "/a:/b", "other:latest"]}, ("img:1.2.3",)),
        ({"command": "nerdctl", "args": ["container", "run", "img:1.2.3", "-p", "80:80"]}, ("img:1.2.3",)),
    ],
)
def test_arguments_after_the_package_belong_to_the_server(config: dict[str, Any], specs: tuple[str, ...]) -> None:
    """Regression: a server's own "-p 3000" was read as npx's --package, and "--with x" as a uvx extra."""
    invocation = parse_mcp_runner(config)
    assert invocation is not None
    assert invocation.specs == specs


@pytest.mark.parametrize(
    ("args", "specs"),
    [
        (["exec", "some-mcp", "-p", "other@1.0.0"], ("other@1.0.0",)),
        (["exec", "--", "some-mcp", "-p", "3000"], ("some-mcp",)),
    ],
)
def test_npm_exec_reads_its_options_until_the_separator(args: list[str], specs: tuple[str, ...]) -> None:
    invocation = parse_mcp_runner({"command": "npm", "args": args})
    assert invocation is not None
    assert invocation.specs == specs


def test_server_port_flag_keeps_an_exact_npx_package_pinned() -> None:
    pin = classify_mcp_pinning({"command": "npx", "args": ["-y", "some-mcp@1.2.3", "-p", "3000"]})
    assert (pin.status, pin.detail) == ("pinned", "npx: exact version 'some-mcp@1.2.3'")


def _audited_versions(config: dict[str, Any]) -> list[str | None]:
    """The exact version the audit finds for each package the runner installs (``None`` when floating)."""
    invocation = parse_mcp_runner(config)
    assert invocation is not None
    if invocation.ecosystem == "pypi":
        declarations = [_python_runner_declaration(spec) for spec in invocation.specs]
    else:
        declarations = [eco.npm_spec_declaration(spec, "mcp") for spec in invocation.npm_specs]
    return [declaration.exact_version for declaration in declarations if declaration is not None]


@pytest.mark.parametrize(
    "config",
    [
        {"command": "npx", "args": ["-y", "some-mcp@1.2.3", "-p", "3000"]},
        {"command": "npx", "args": ["-y", "some-mcp", "-p", "3000"]},
        {"command": "npx", "args": ["-p", "a@1.0.0", "cmd", "--package", "b"]},
        {"command": "npm", "args": ["exec", "some-mcp", "-p", "other@1.0.0"]},
        {"command": "uvx", "args": ["srv==1.0", "--with", "x"]},
        {"command": "uvx", "args": ["--with", "x", "srv==1.0"]},
        {"command": "pipx", "args": ["run", "srv==1.0", "--spec", "other"]},
    ],
)
def test_runner_is_pinned_exactly_when_the_audit_finds_one_version_per_package(config: dict[str, Any]) -> None:
    versions = _audited_versions(config)
    assert versions
    assert (classify_mcp_pinning(config).status == "pinned") == all(version is not None for version in versions)


# --------------------------------------------------------------------------- #
# One exact-version matcher per ecosystem                                     #
# --------------------------------------------------------------------------- #
def test_npm_equals_pin_is_pinned_like_the_audit_reads_it() -> None:
    """Regression: ``npx -y pkg@=1.2.3`` was unpinned to the pinning check but an exact 1.2.3 to the audit."""
    pin = classify_mcp_pinning({"command": "npx", "args": ["-y", "pkg@=1.2.3"]})
    assert (pin.status, pin.detail) == ("pinned", "npx: exact version 'pkg@=1.2.3'")
    declaration = eco.npm_spec_declaration("pkg@=1.2.3", "mcp")
    assert declaration is not None
    assert (declaration.name, declaration.exact_version) == ("pkg", "1.2.3")


@pytest.mark.parametrize(
    ("version", "exact"),
    [("1.2.3", "1.2.3"), ("=1.2.3", "1.2.3"), ("v1.2.3-beta.1", "1.2.3-beta.1"), ("^1.2.3", None), ("1.2", None)],
)
def test_exact_npm_version(version: str, exact: str | None) -> None:
    assert exact_npm_version(version) == exact


@pytest.mark.parametrize(
    ("requirement", "exact"),
    [
        ("pkg==1.2.3", "1.2.3"),
        ("pkg[extra] == 1.0.post1", "1.0.post1"),
        ("pkg@1.2", "1.2"),
        ("pkg==1.0; python_version >= '3.12'", "1.0"),
        ("pkg===1.0", None),
        ("pkg==1.0.*", None),
        ("pkg==1.0,<2", None),
        ("pkg@latest", None),
        ("pkg @ https://example.invalid/pkg.whl", None),
        ("pkg", None),
    ],
)
def test_exact_pypi_version(requirement: str, exact: str | None) -> None:
    assert exact_pypi_version(requirement) == exact


_NPM_RUNNER_SPECS = (
    *("pkg", "pkg@1.2.3", "pkg@=1.2.3", "pkg@v1.2.3", "pkg@^1.2.3", "pkg@1", "pkg@latest", "pkg@", "@scope/pkg"),
    *("@scope/pkg@1.2.3", "@scope/pkg@=1.2.3-beta.1", "pkg@>=1.2.3", "pkg@==1.2.3"),
)
_PYTHON_RUNNER_SPECS = (
    *("pkg", "pkg==1.0", "pkg[x]==1.0", "pkg===1.0", "pkg==1.0.*", "pkg>=1", "pkg==1.0,<2", "pkg==1.0rc", "pkg@1.2"),
    *("pkg@1.2.3", "pkg@v1.2.3", "pkg@1.2.3-beta.1", "pkg@latest", "pkg==1.0; python_version > '3'", "pkg==v1.0"),
    "pkg @ https://example.invalid/pkg.whl",
)


@pytest.mark.parametrize("spec", _NPM_RUNNER_SPECS)
def test_npm_runner_spec_is_pinned_exactly_when_the_audit_finds_one_version(spec: str) -> None:
    pin = classify_mcp_pinning({"command": "npx", "args": ["-y", spec]})
    declaration = eco.npm_spec_declaration(spec, "mcp")
    assert declaration is not None
    assert (pin.status == "pinned") == (declaration.exact_version is not None)


@pytest.mark.parametrize("spec", _PYTHON_RUNNER_SPECS)
def test_python_runner_spec_is_pinned_exactly_when_the_audit_finds_one_version(spec: str) -> None:
    pin = classify_mcp_pinning({"command": "uvx", "args": [spec]})
    declaration = _python_runner_declaration(spec)
    assert declaration is not None
    assert (pin.status == "pinned") == (declaration.exact_version is not None)


# --------------------------------------------------------------------------- #
# Remote sources are a field, not detail text                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("config", "remote"),
    [
        ({"command": "npx", "args": ["-y", "github:user/repo"]}, True),
        ({"command": "npx", "args": ["-y", "user/repo#" + "b" * 40]}, True),
        ({"command": "uvx", "args": ["--from", "git+https://github.com/o/r", "tool"]}, True),
        ({"command": "deno", "args": ["run", "https://deno.land/x/mod@v1.2.3/mod.ts"]}, True),
        ({"command": "deno", "args": ["run", "npm:pkg"]}, False),
        ({"command": "npx", "args": ["-y", "pkg"]}, False),
        ({"command": "uvx", "args": ["remote module"]}, False),
        ({"command": "docker", "args": ["run", "img"]}, False),
    ],
)
def test_remote_package_source_is_recorded_on_the_classification(config: dict[str, Any], remote: bool) -> None:
    assert classify_mcp_pinning(config).remote is remote


def _hook_checks(root: Path, command: str) -> set[str]:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "demo"}))
    (root / "hooks").mkdir()
    hooks = {"hooks": {"PostToolUse": [{"hooks": [{"type": "command", "command": command}]}]}}
    (root / "hooks" / "hooks.json").write_text(json.dumps(hooks))
    return {finding.check_name for finding in PluginSchemaValidator().validate(root).findings}


def test_hook_remote_code_follows_the_field_not_the_detail_wording(tmp_path: Path) -> None:
    """A package whose detail text merely reads 'remote module' is an unpinned registry package, not remote code."""
    checks = _hook_checks(tmp_path / "words", "uvx 'remote module'")
    assert "plugin_hook_unpinned_package" in checks
    assert "plugin_hook_remote_code" not in checks
    assert "plugin_hook_remote_code" in _hook_checks(tmp_path / "git", "npx -y github:user/repo")


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


@pytest.fixture
def osv_scanner(monkeypatch: pytest.MonkeyPatch, pip_audit: _FakeTool) -> _FakeTool:
    fake = _FakeTool("osv-scanner")
    monkeypatch.setattr(Tools, "osv_scanner", fake)
    return fake


def test_server_arguments_are_never_audited_as_packages(
    tmp_path: Path, pip_audit: _FakeTool, osv_scanner: _FakeTool
) -> None:
    """Regression: "-p 3000" was audited as npm package "3000" and the pinned server package was never audited."""
    servers = {
        "web": {"command": "npx", "args": ["-y", "some-mcp@1.2.3", "-p", "3000"]},
        "py": {"command": "uvx", "args": ["srv==1.0", "--with", "x"]},
    }
    result = _audit(tmp_path / "demo", servers)

    [npm_call] = osv_scanner.calls
    packages = json.loads(npm_call["files"]["package-lock.json"])["packages"]
    assert {key: value for key, value in packages.items() if key} == {"node_modules/some-mcp": {"version": "1.2.3"}}
    [pip_call] = pip_audit.calls
    assert pip_call["files"] == {"requirements-0.txt": "srv==1.0\n"}
    assert not [f for f in result.findings if f.check_name == "dependency-version-unverified"]


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
