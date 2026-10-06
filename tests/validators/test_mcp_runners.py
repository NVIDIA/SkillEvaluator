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
    PypiPin,
    RunnerInvocation,
    classify_mcp_pinning,
    exact_npm_version,
    is_pep440_version,
    mcp_container_image,
    parse_mcp_runner,
    pypi_pin,
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


@pytest.mark.parametrize(
    ("config", "specs"),
    [
        ({"command": "uvx", "args": ["-w", "dep==1.0.0", "server-tool"]}, ("server-tool", "dep==1.0.0")),
        ({"command": "uv", "args": ["tool", "run", "-w", "dep==1.0.0", "server-tool"]}, ("server-tool", "dep==1.0.0")),
        (
            {"command": "uvx", "args": ["-w", "floating-dep", "server-tool==1.0.0"]},
            ("server-tool==1.0.0", "floating-dep"),
        ),
        ({"command": "uvx", "args": ["-b", "constraints.txt", "-C", "k=v", "srv==1.0"]}, ("srv==1.0",)),
    ],
)
def test_uvx_short_options_are_read_like_their_long_forms(config: dict[str, Any], specs: tuple[str, ...]) -> None:
    """Regression: '-w' (--with), '-b', and '-C' were read as switches, so their value became the package."""
    invocation = parse_mcp_runner(config)
    assert invocation is not None and invocation.specs == specs
    assert classify_mcp_pinning(config).status == ("pinned" if specs == ("srv==1.0",) else "unpinned")


def test_uvx_with_requirements_file_is_unpinned() -> None:
    """Regression: the packages of '--with-requirements reqs.txt' were never checked, yet the server was pinned."""
    config = {"command": "uvx", "args": ["--with-requirements", "reqs.txt", "mcp-server==1.0.0"]}

    invocation = parse_mcp_runner(config)
    pin = classify_mcp_pinning(config)

    assert invocation is not None and invocation.requirement_files == ("reqs.txt",)
    assert (pin.status, pin.detail) == (
        "unpinned",
        "uvx: the packages of '--with-requirements reqs.txt' are not read, so they cannot be checked",
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


@pytest.mark.parametrize(
    ("args", "specs"),
    [
        # npm options that take a value: the word after them is not the package.
        (["--globalconfig", "x@1.0.0", "-p", "github:evil/x", "x-cmd"], ("github:evil/x",)),
        (["--node-options", "x@1.0.0", "--package", "some-floating-pkg", "x-cmd"], ("some-floating-pkg",)),
        (["--loglevel", "silent", "pkg@1.2.3", "-p", "3000"], ("pkg@1.2.3",)),
        # An option the reader does not know: npx gives it the next word as its value, npm reads it as a
        # switch, so that word may be the package too.
        (["--some-new-option", "x@1.0.0", "-p", "github:evil/x", "x-cmd"], ("github:evil/x", "x@1.0.0")),
        (["--some-new-switch", "pkg"], ("pkg",)),
        (["--no-yes", "pkg@1.2.3", "-p", "3000"], ("3000", "pkg@1.2.3")),
        # npm switches take no value.
        (["-y", "--prefer-offline", "-q", "pkg@1.2.3", "-p", "3000"], ("pkg@1.2.3",)),
    ],
)
def test_npx_reads_its_options_the_way_npx_does(args: list[str], specs: tuple[str, ...]) -> None:
    """Regression: an npm value option missing from the table ('--globalconfig x@1.0.0') was read as a switch, so
    its value became the package and a later '-p <floating spec>' was never checked or audited."""
    invocation = parse_mcp_runner({"command": "npx", "args": args})
    assert invocation is not None
    assert invocation.specs == specs


@pytest.mark.parametrize(
    "args",
    [
        ["--globalconfig", "x@1.0.0", "-p", "github:evil/x", "x-cmd"],
        ["--node-options", "x@1.0.0", "--package", "some-floating-pkg", "x-cmd"],
        ["--some-new-option", "x@1.0.0", "-p", "github:evil/x", "x-cmd"],
    ],
)
def test_npx_package_after_an_option_value_is_pin_checked(args: list[str]) -> None:
    assert classify_mcp_pinning({"command": "npx", "args": args}).status == "unpinned"


def test_npm_exec_reads_a_value_option_before_the_package() -> None:
    invocation = parse_mcp_runner({"command": "npm", "args": ["exec", "--globalconfig", "x@1.0.0", "x-cmd"]})
    assert invocation is not None and invocation.specs == ("x-cmd",)


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
# A runner named by a path with spaces                                        #
# --------------------------------------------------------------------------- #
_NPX_WITH_SPACES = "C:\\Program Files\\nodejs\\npx.cmd"
_UVX_WITH_SPACES = "/opt/my tools/uvx"
_DOCKER_WITH_SPACES = "C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"command": _NPX_WITH_SPACES, "args": ["-y", "pkg@1.2.3"]}, ("npm", "npx", ("pkg@1.2.3",))),
        ({"command": _UVX_WITH_SPACES, "args": ["--with", "foo", "pkg==1.0"]}, ("pypi", "uvx", ("pkg==1.0", "foo"))),
        ({"command": _DOCKER_WITH_SPACES, "args": ["run", "img:1.2.3"]}, ("container", "docker run", ("img:1.2.3",))),
        ({"command": "/usr/local/bin/npx -y pkg@1.2.3"}, ("npm", "npx", ("pkg@1.2.3",))),
    ],
)
def test_runner_named_by_a_path_with_spaces_is_recognized(config: dict[str, Any], expected: tuple) -> None:
    """Regression: "C:\\Program Files\\nodejs\\npx.cmd" was split into words, so no runner was recognized."""
    invocation = parse_mcp_runner(config)
    assert invocation is not None
    assert (invocation.ecosystem, invocation.runner, invocation.specs) == expected


def test_pinning_reads_a_runner_path_with_spaces() -> None:
    assert classify_mcp_pinning({"command": _NPX_WITH_SPACES, "args": ["-y", "pkg@1.2.3"]}).status == "pinned"
    floating = classify_mcp_pinning({"command": _UVX_WITH_SPACES, "args": ["--with", "foo", "pkg==1.0"]})
    assert (floating.status, floating.detail) == (
        "unpinned",
        "uvx: package 'foo' has no version (resolves to the latest release)",
    )
    compose = classify_mcp_pinning({"command": _DOCKER_WITH_SPACES, "args": ["compose", "up"]})
    assert (compose.status, compose.detail) == ("not_applicable", "docker invocation is not 'run'")
    assert mcp_container_image({"command": _DOCKER_WITH_SPACES, "args": ["run", "-i", "img:1.2.3"]}) == "img:1.2.3"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {"command": "npx -y @modelcontextprotocol/server-filesystem /home/user/src/docker"},
            ("npm", "npx", ("@modelcontextprotocol/server-filesystem",)),
        ),
        ({"command": "npx -y github:evil/npx", "args": ["left-pad@1.3.0"]}, ("npm", "npx", ("github:evil/npx",))),
        ({"command": "docker run --rm -i ghcr.io/acme/docker"}, ("container", "docker run", ("ghcr.io/acme/docker",))),
        (
            {"command": "uvx --from git+https://github.com/evil/uvx", "args": ["pkg==1.0"]},
            ("pypi", "uvx", ("git+https://github.com/evil/uvx",)),
        ),
    ],
)
def test_a_command_line_is_read_by_its_first_word(config: dict[str, Any], expected: tuple) -> None:
    """Regression: a command line whose last path segment named a runner ('.../docker', 'github:evil/npx') was
    read as one program of that name, so pinning, the image lookup, and the audit read the wrong program."""
    invocation = parse_mcp_runner(config)
    assert invocation is not None
    assert (invocation.ecosystem, invocation.runner, invocation.specs) == expected
    assert classify_mcp_pinning(config).status == "unpinned"


def test_a_command_line_that_only_ends_in_a_runner_path_is_not_that_runner() -> None:
    pin = classify_mcp_pinning({"command": "node ./server.js --data /srv/docker"})
    assert (pin.status, pin.detail) == ("not_applicable", "local interpreter, script, or binary ('node')")


# --------------------------------------------------------------------------- #
# A wrapper such as env, nohup, or timeout is looked through                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {"command": "env", "args": ["NODE_ENV=production", "npx", "-y", "some-mcp-server"]},
            ("npm", "npx", ("some-mcp-server",)),
        ),
        ({"command": "/usr/bin/env", "args": ["npx", "-y", "some-mcp-server"]}, ("npm", "npx", ("some-mcp-server",))),
        (
            {"command": "env", "args": ["-i", "PATH=/usr/bin", "uvx", "mcp-server-fetch"]},
            ("pypi", "uvx", ("mcp-server-fetch",)),
        ),
        ({"command": "env -u HOME npx -y some-mcp-server"}, ("npm", "npx", ("some-mcp-server",))),
        ({"command": "env", "args": ["docker", "run", "img"]}, ("container", "docker run", ("img",))),
        ({"command": "nohup", "args": ["npx", "-y", "pkg"]}, ("npm", "npx", ("pkg",))),
        ({"command": "timeout", "args": ["-s", "KILL", "600", "uvx", "pkg"]}, ("pypi", "uvx", ("pkg",))),
        ({"command": "nice", "args": ["-n", "10", "npx", "pkg"]}, ("npm", "npx", ("pkg",))),
        ({"command": "stdbuf", "args": ["-o", "L", "npx", "pkg"]}, ("npm", "npx", ("pkg",))),
        ({"command": "setsid", "args": ["-f", "time", "-p", "npx", "pkg"]}, ("npm", "npx", ("pkg",))),
        ({"command": "sudo", "args": ["-u", "app", "-E", "MODE=1", "npx", "pkg"]}, ("npm", "npx", ("pkg",))),
        ({"command": "doas", "args": ["-u", "app", "env", "nohup", "npx", "pkg"]}, ("npm", "npx", ("pkg",))),
    ],
)
def test_a_wrapped_runner_is_read_through_its_wrapper(config: dict[str, Any], expected: tuple) -> None:
    """Regression: 'env ... npx -y pkg' was a local binary ('env'), so pinning, the image lookup, and the audit
    skipped the package that the wrapped runner installs."""
    invocation = parse_mcp_runner(config)
    assert invocation is not None
    assert (invocation.ecosystem, invocation.runner, invocation.specs) == expected
    assert classify_mcp_pinning(config).status == "unpinned"


@pytest.mark.parametrize("command", ["env", "nohup", "timeout"])
def test_a_wrapper_that_runs_nothing_is_a_local_binary(command: str) -> None:
    pin = classify_mcp_pinning({"command": command, "args": ["60"] if command == "timeout" else []})
    assert (pin.status, pin.detail) == ("not_applicable", f"local interpreter, script, or binary ({command!r})")


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
    ("version", "valid"),
    [
        *[(version, True) for version in ("1.2.3", "v1.2.3", "V1.0", "1.0rc", "1.0c1", "1.0-1", "1.2.3-beta.1")],
        *[(version, True) for version in ("1.0_alpha2", "1!2.0", "1.0.post1.dev2+local.7", "2024.11.25")],
        *[(version, False) for version in ("1.0.*", "latest", "1.0+", "1..0", "", "local-build")],
    ],
)
def test_pep440_versions_match_in_every_spelling_pep_440_accepts(version: str, valid: bool) -> None:
    assert is_pep440_version(version) is valid


@pytest.mark.parametrize(
    ("requirement", "pin"),
    [
        ("pkg==1.2.3", PypiPin("1.2.3", auditable=True)),
        ("pkg[extra] == 1.0.post1", PypiPin("1.0.post1", auditable=True)),
        ("pkg@1.2", PypiPin("1.2", auditable=True)),
        ("pkg==1.0; python_version >= '3.12'", PypiPin("1.0", auditable=True)),
        ("pkg==v1.2.3", PypiPin("v1.2.3", auditable=True)),
        ("pkg==1.0rc", PypiPin("1.0rc", auditable=True)),
        ("pkg==1.0-1", PypiPin("1.0-1", auditable=True)),
        ("pkg@1.2.3-beta.1", PypiPin("1.2.3-beta.1", auditable=True)),
        ("pkg===1.0", PypiPin("1.0", auditable=True)),
        ("pkg===local-build", PypiPin("local-build", auditable=False)),
        ("pkg (==1.0)", PypiPin("1.0", auditable=True)),
        ("pkg[x] (===1.0)", PypiPin("1.0", auditable=True)),
        ("pkg==1.0.*", None),
        ("pkg==1.0,<2", None),
        ("pkg>=1", None),
        ("pkg@latest", None),
        ("pkg==latest", None),
        ("pkg @ https://example.invalid/pkg.whl", None),
        ("pkg", None),
    ],
)
def test_pypi_pin(requirement: str, pin: PypiPin | None) -> None:
    assert pypi_pin(requirement) == pin


_NPM_RUNNER_SPECS = (
    *("pkg", "pkg@1.2.3", "pkg@=1.2.3", "pkg@v1.2.3", "pkg@^1.2.3", "pkg@1", "pkg@latest", "pkg@", "@scope/pkg"),
    *("@scope/pkg@1.2.3", "@scope/pkg@=1.2.3-beta.1", "pkg@>=1.2.3", "pkg@==1.2.3"),
)
_PYTHON_RUNNER_SPECS = (
    *("pkg", "pkg==1.0", "pkg[x]==1.0", "pkg===1.0", "pkg==1.0.*", "pkg>=1", "pkg==1.0,<2", "pkg==1.0rc", "pkg@1.2"),
    *("pkg@1.2.3", "pkg@v1.2.3", "pkg@1.2.3-beta.1", "pkg@latest", "pkg==1.0; python_version > '3'", "pkg==v1.0"),
    *("pkg==1.0-1", "pkg (==1.0)", "pkg[x] (===1.0)", "pkg==latest", "pkg @ https://example.invalid/pkg.whl"),
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


@pytest.mark.parametrize(
    "spec", ["pkg===1.0", "pkg@v1.2.3", "pkg@1.2.3-beta.1", "pkg==1.0rc", "pkg==1.0-1", "pkg==v1.0", "pkg (==1.0)"]
)
def test_every_pep440_spelling_of_an_exact_version_is_pinned(spec: str) -> None:
    """Regression: exact pins in a non-canonical PEP 440 spelling (or with ``===``) were reported unpinned."""
    findings = validate_mcp_server_declaration("s", {"command": "uvx", "args": [spec]}, "p.json")
    assert "mcp_unpinned_package" not in {finding.check_name for finding in findings}
    assert classify_mcp_pinning({"command": "uvx", "args": [spec]}).status == "pinned"


def test_arbitrary_equality_with_a_non_version_is_pinned_but_unverified() -> None:
    assert classify_mcp_pinning({"command": "uvx", "args": ["pkg===local-build"]}).status == "pinned"
    declaration = _python_runner_declaration("pkg===local-build")
    assert declaration is not None
    assert (declaration.name, declaration.exact_version) == ("pkg", None)


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


@pytest.mark.parametrize(
    "module",
    [
        "https://deno.land/x/mod@v1.2.3/mod.ts",
        "https://deno.land/std@0.224.0/http/file_server.ts",
        "https://esm.sh/preact@10.19.2",
    ],
)
def test_deno_remote_module_with_a_version_in_its_path_is_pinned(module: str) -> None:
    assert classify_mcp_pinning({"command": "deno", "args": ["run", "-A", module]}).status == "pinned"


@pytest.mark.parametrize(
    "module",
    [
        "https://evil.example/payload.ts#@1.0.0",
        "https://evil.example/payload.ts?v=@1.0.0",
        "https://evil.example/@1.0.0/../payload.ts",
        "https://evil.example/@1.0.0/%2e%2e/payload.ts",
        "https://evil.example/x@1.0.0/./../payload.ts",
    ],
)
def test_deno_module_version_counts_only_in_the_path_the_server_reads(tmp_path: Path, module: str) -> None:
    """Regression: '@1.0.0' anywhere in the URL, even in the query, the fragment, or a '..' segment that the client
    resolves away, made an arbitrary remote module pinned, and the hook lost its CRITICAL remote-code finding."""
    assert classify_mcp_pinning({"command": "deno", "args": ["run", "-A", module]}).status == "unpinned"
    assert "plugin_hook_remote_code" in _hook_checks(tmp_path, f"deno run -A {module}")


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


def test_audit_reads_runner_paths_with_spaces(tmp_path: Path, pip_audit: _FakeTool, osv_scanner: _FakeTool) -> None:
    servers = {
        "web": {"command": _NPX_WITH_SPACES, "args": ["-y", "lodash@4.17.20"]},
        "py": {"command": _UVX_WITH_SPACES, "args": ["mcp-server-fetch==2024.11.25"]},
    }
    _audit(tmp_path / "demo", servers)

    [npm_call] = osv_scanner.calls
    packages = json.loads(npm_call["files"]["package-lock.json"])["packages"]
    assert packages["node_modules/lodash"] == {"version": "4.17.20"}
    [pip_call] = pip_audit.calls
    assert pip_call["files"] == {"requirements-0.txt": "mcp-server-fetch==2024.11.25\n"}


def test_audit_reads_a_runner_through_its_wrapper(tmp_path: Path, pip_audit: _FakeTool, osv_scanner: _FakeTool) -> None:
    """Regression: 'env NODE_ENV=production npx -y pkg' was a local binary, so its package was never audited."""
    servers = {
        "web": {"command": "env", "args": ["NODE_ENV=production", "npx", "-y", "lodash@4.17.20"]},
        "py": {"command": "nohup", "args": ["uvx", "mcp-server-fetch==2024.11.25"]},
    }
    _audit(tmp_path / "demo", servers)

    [npm_call] = osv_scanner.calls
    packages = json.loads(npm_call["files"]["package-lock.json"])["packages"]
    assert packages["node_modules/lodash"] == {"version": "4.17.20"}
    [pip_call] = pip_audit.calls
    assert pip_call["files"] == {"requirements-0.txt": "mcp-server-fetch==2024.11.25\n"}


def test_audit_reports_a_uvx_requirements_file_as_unverified(tmp_path: Path, pip_audit: _FakeTool) -> None:
    server = {"command": "uvx", "args": ["--with-requirements", "reqs.txt", "mcp-server==1.0.0"]}
    result = _audit(tmp_path / "demo", {"srv": server})

    [call] = pip_audit.calls
    assert call["files"] == {"requirements-0.txt": "mcp-server==1.0.0\n"}
    unverified = [
        f.metadata["declared_constraint"] for f in result.findings if f.check_name == "dependency-version-unverified"
    ]
    assert unverified == ["--with-requirements reqs.txt"]


def test_pip_audit_gets_every_exact_pep440_spelling_as_name_equals_version(
    tmp_path: Path, pip_audit: _FakeTool
) -> None:
    """Regression: these pins were unverified (or unpinned) although each names one release."""
    servers = {
        "v": {"command": "uvx", "args": ["--from", "alpha@v1.2.3", "alpha"]},
        "parens": {"command": "uvx", "args": ["beta (==1.0)"]},
        "arbitrary": {"command": "pipx", "args": ["run", "--spec", "gamma===2.0", "gamma"]},
        "beta-tag": {"command": "uvx", "args": ["delta@1.2.3-beta.1"]},
        "label": {"command": "uvx", "args": ["epsilon===local-build"]},
    }
    result = _audit(tmp_path / "demo", servers)

    [call] = pip_audit.calls
    assert call["files"] == {"requirements-0.txt": "alpha==v1.2.3\nbeta==1.0\ndelta==1.2.3-beta.1\ngamma==2.0\n"}
    unverified = [
        (f.metadata["package_name"], f.metadata["declared_constraint"])
        for f in result.findings
        if f.check_name == "dependency-version-unverified"
    ]
    assert unverified == [("epsilon", "epsilon===local-build")]


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
