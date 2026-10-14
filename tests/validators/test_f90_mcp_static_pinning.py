# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinning (check 11): floating tags at the version position, wrappers, and hooks.

Regression tests for proof bugs H13, M24, and L16. Each test builds a small
plugin from the proof's example and runs the real plugin schema validator (or
the Tier 3 staging gate), in the Claude Code and Codex formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.tier3.plugin_eval import _reject_unsafe_mcp_declaration
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_DIGEST = "sha256:" + "e6cfbb2cfe389f581d4d1284b6787410e00e59e765e794801f9c6e58aaa19ab4"
_CODEX_MANIFEST = {
    "name": "demo",
    "version": "1.0.0",
    "description": "Pinning test plugin (Codex)",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "Pinning test",
        "shortDescription": "Pinning test plugin",
        "longDescription": "A plugin used to test MCP and hook pinning checks.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}


def _plugin(root: Path, files: dict[str, object], *, codex: bool = False) -> Path:
    manifest = (
        {".codex-plugin/plugin.json": _CODEX_MANIFEST}
        if codex
        else {".claude-plugin/plugin.json": {"name": "demo", "version": "1.0.0", "description": "Pinning test"}}
    )
    for rel, content in {**manifest, **files}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _mcp(servers: dict[str, dict], tmp_path: Path, *, codex: bool = False) -> ValidationResult:
    root = _plugin(tmp_path / ("codex" if codex else "claude"), {".mcp.json": {"mcpServers": servers}}, codex=codex)
    return PluginSchemaValidator().validate(root)


def _by_server(result: ValidationResult) -> dict[str, dict[str, Severity]]:
    found: dict[str, dict[str, Severity]] = {}
    for finding in result.findings:
        server = finding.metadata.get("mcp_server")
        if server:
            found.setdefault(server, {})[finding.check_name] = finding.severity
    return found


def _servers(result: ValidationResult) -> dict[str, dict]:
    return {server["name"]: server for server in result.metadata["plugin"]["mcp"]["servers"]}


def _npx(spec: str) -> dict:
    return {"command": "npx", "args": ["-y", spec]}


def _docker(image: str) -> dict:
    return {"command": "docker", "args": ["run", "-i", "--rm", image]}


# --------------------------------------------------------------------------- #
# H13: floating markers only at the version position                          #
# --------------------------------------------------------------------------- #
_H13_PINNED = {
    # check-11 e01
    "nextui": _npx("@nextui-org/mcp@1.0.0"),
    "headless": _npx("@headlessui/mcp@1.0.0"),
    "entry-point": {"command": "python", "args": ["-m", "example_mcp", "--entry", "example_mcp.cli:main"]},
    "email-arg": {
        "command": "node",
        "args": ["${CLAUDE_PLUGIN_ROOT}/server.js", "--notify", "bot@headquarters.example"],
    },
    # skeptic x02: real scope names that start with a marker
    "mastercard": _npx("@mastercard/mcp@1.0.0"),
    "next-scope": _npx("@next/mcp@1.0.0"),
    "maintainer": _npx("@maintainx/mcp@1.0.0"),
    # audit re-test check-11-pinning-probes / skeptic x01: a digest pins the image whatever the tag says
    "latest-digest": _docker(f"ghcr.io/example/mcp:latest@{_DIGEST}"),
    "main-digest": _docker(f"ghcr.io/example/mcp:main@{_DIGEST}"),
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_exact_pins_and_marker_substrings_are_not_floating(tmp_path: Path, codex: bool) -> None:
    result = _mcp(_H13_PINNED, tmp_path, codex=codex)

    findings = _by_server(result)
    assert not any("mcp_command_floating_version" in checks for checks in findings.values()), findings
    assert not any("mcp_unpinned_package" in checks for checks in findings.values()), findings
    servers = _servers(result)
    for name in ("nextui", "headless", "mastercard", "next-scope", "maintainer", "latest-digest", "main-digest"):
        assert servers[name]["pinned"] is True, servers[name]
    assert servers["entry-point"]["pinned"] is None
    assert servers["email-arg"]["pinned"] is None
    assert result.metadata["plugin"]["mcp"]["pinning"]["ratio"] == 1.0


def test_tier3_staging_accepts_an_exact_scoped_pin() -> None:
    """t01 step 3: Tier 3 refused '@nextui-org/mcp@1.0.0' as floating; it is an exact pin."""
    _reject_unsafe_mcp_declaration("nextui", _npx("@nextui-org/mcp@1.0.0"))
    with pytest.raises(ValueError, match="mcp_command_floating_version"):
        _reject_unsafe_mcp_declaration("latest", _npx("@nextui-org/mcp@latest"))


# --------------------------------------------------------------------------- #
# L16: one rule for every moving tag, also inside a version-like image tag     #
# --------------------------------------------------------------------------- #
_MOVING = {
    # check-11 e15
    "npm-next": _npx("@example/a@next"),
    "npm-canary": _npx("@example/b@canary"),
    "npm-beta": _npx("@example/c@beta"),
    "npm-rc": _npx("@example/d@rc"),
    "img-main": _docker("ghcr.io/example/mcp:main"),
    "img-master": _docker("ghcr.io/example/mcp:master"),
    "img-nightly": _docker("ghcr.io/example/mcp:nightly"),
    # skeptic x04: ':1-latest' looked like a version tag
    "one-latest": _docker("ghcr.io/example/mcp:1-latest"),
    "zero-nightly": _docker("ghcr.io/example/mcp:0-nightly"),
    "uvx-latest": {"command": "uvx", "args": ["example-mcp@latest"]},
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_every_moving_tag_is_blocking(tmp_path: Path, codex: bool) -> None:
    findings = _by_server(_mcp(_MOVING, tmp_path, codex=codex))

    for name in _MOVING:
        assert findings.get(name) == {"mcp_command_floating_version": Severity.HIGH}, (name, findings.get(name))


_IMAGE_PRERELEASES = ("1.0.0-beta", "2.1.0-rc", "1.2.3-alpha", "1.4.2-stable", "3.0-dev", "2024.01-preview")


def test_ranges_and_prereleases_keep_their_severity(tmp_path: Path) -> None:
    servers = {
        "range": _npx("@example/a@^1.2.0"),
        "range-beta": _npx("@example/b@^1.0.0-beta"),
        "prerelease": _npx("@example/c@1.2.3-rc.1"),
        "version-tag": _docker("ghcr.io/example/mcp:1.2.3"),
        "short-digest": _docker("ghcr.io/example/mcp@sha256:abc123"),
        # Review round: an exact image pre-release names one release, as '@1.0.0-beta' does for npm.
        **{f"img-{tag}": _docker(f"ghcr.io/example/mcp:{tag}") for tag in _IMAGE_PRERELEASES},
        # ... but 'latest' moves wherever it appears.
        "img-1.2-latest": _docker("ghcr.io/example/mcp:1.2-latest"),
        "img-stable-alpine": _docker("ghcr.io/example/mcp:stable-alpine"),
    }
    result = _mcp(servers, tmp_path)
    findings = _by_server(result)

    assert findings["range"] == {"mcp_unpinned_package": Severity.MEDIUM}
    assert findings["range-beta"] == {"mcp_unpinned_package": Severity.MEDIUM}
    assert "prerelease" not in findings
    assert "version-tag" not in findings
    assert findings["short-digest"] == {"mcp_unpinned_package": Severity.MEDIUM}
    for tag in _IMAGE_PRERELEASES:
        assert f"img-{tag}" not in findings, (tag, findings.get(f"img-{tag}"))
        assert _servers(result)[f"img-{tag}"]["pinned"] is True
    assert findings["img-1.2-latest"] == {"mcp_command_floating_version": Severity.HIGH}
    assert findings["img-stable-alpine"] == {"mcp_command_floating_version": Severity.HIGH}


def test_tier3_stages_an_exact_image_prerelease() -> None:
    _reject_unsafe_mcp_declaration("img-beta", _docker("ghcr.io/example/mcp:1.0.0-beta"))
    with pytest.raises(ValueError, match="mcp_command_floating_version"):
        _reject_unsafe_mcp_declaration("img-latest", _docker("ghcr.io/example/mcp:1-latest"))


# --------------------------------------------------------------------------- #
# Review round: '${NAME:-default}' in command and args, and uvx --with         #
# --------------------------------------------------------------------------- #
_ENV_DEFAULTS = {
    "arg-default-latest": {"command": "npx", "args": ["-y", "${PKG:-@example/mcp@latest}"]},
    "img-default-latest": {"command": "docker", "args": ["run", "-i", "${IMG:-ghcr.io/example/mcp:latest}"]},
    "version-default-latest": {"command": "npx", "args": ["-y", "@example/mcp@${VERSION:-latest}"]},
    "version-default-exact": {"command": "npx", "args": ["-y", "@example/mcp@${VERSION:-1.2.3}"]},
    "no-default": {"command": "npx", "args": ["-y", "${PKG}"]},
}


def test_claude_reads_launch_defaults_like_url_defaults(tmp_path: Path) -> None:
    """Claude Code expands '${NAME:-default}' in command and args, so the shipped default is classified."""
    result = _mcp(_ENV_DEFAULTS, tmp_path)
    findings = _by_server(result)
    servers = _servers(result)

    for name in ("arg-default-latest", "img-default-latest", "version-default-latest"):
        assert findings.get(name) == {"mcp_command_floating_version": Severity.HIGH}, (name, findings.get(name))
    assert "version-default-exact" not in findings
    assert servers["version-default-exact"]["pinned"] is True
    assert findings["no-default"] == {"mcp_unpinned_package": Severity.MEDIUM}
    assert "environment reference" in servers["no-default"]["pin_detail"]
    assert "git/URL" not in servers["arg-default-latest"]["pin_detail"]


def test_codex_runs_launch_references_as_text(tmp_path: Path) -> None:
    """Codex does not expand references, so the spec is the literal text: unpinned, never pinned by a default."""
    findings = _by_server(_mcp(_ENV_DEFAULTS, tmp_path, codex=True))

    for name in _ENV_DEFAULTS:
        assert findings.get(name) == {"mcp_unpinned_package": Severity.MEDIUM}, (name, findings.get(name))


def test_tier3_refuses_a_floating_launch_default() -> None:
    with pytest.raises(ValueError, match="mcp_command_floating_version"):
        _reject_unsafe_mcp_declaration("arg-default-latest", _ENV_DEFAULTS["arg-default-latest"])


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_uvx_floating_extra_beats_an_unversioned_tool(tmp_path: Path, codex: bool) -> None:
    servers = {
        "with-latest-unversioned-tool": {"command": "uvx", "args": ["--with", "requests@latest", "example-mcp"]},
        "with-latest-pinned-tool": {"command": "uvx", "args": ["--with", "requests@latest", "example-mcp==1.0.0"]},
        "with-exact-unversioned-tool": {"command": "uvx", "args": ["--with", "requests==2.32.3", "example-mcp"]},
    }
    result = _mcp(servers, tmp_path, codex=codex)
    findings = _by_server(result)

    assert findings["with-latest-unversioned-tool"] == {"mcp_command_floating_version": Severity.HIGH}
    assert findings["with-latest-pinned-tool"] == {"mcp_command_floating_version": Severity.HIGH}
    assert "--with" in _servers(result)["with-latest-unversioned-tool"]["pin_detail"]
    assert findings["with-exact-unversioned-tool"] == {"mcp_unpinned_package": Severity.MEDIUM}


# --------------------------------------------------------------------------- #
# M24 + L16: runners behind wrappers, other runners, flag placement            #
# --------------------------------------------------------------------------- #
_WRAPPED = {
    # check-11 e02
    "win-cmd": {"command": "cmd", "args": ["/c", "npx", "-y", "@example/mcp-server"]},
    "env-prefix": {"command": "env", "args": ["NODE_ENV=production", "npx", "-y", "@example/mcp-server"]},
    "bun-x": {"command": "bun", "args": ["x", "@example/mcp-server"]},
    "uv-run-with": {"command": "uv", "args": ["run", "--with", "example-mcp", "example-mcp"]},
    # skeptic x05 / audit re-test: pnpm --package before dlx, go run, dnx
    "pnpm-package-first": {"command": "pnpm", "args": ["--package=@example/mcp-server", "dlx", "example-mcp"]},
    "go-run-major": {"command": "go", "args": ["run", "github.com/example/mcp@v1"]},
    "dnx": {"command": "dnx", "args": ["Example.Mcp", "--yes"]},
    # skeptic x07: a global flag before 'exec'
    "npm-global-flag": {"command": "npm", "args": ["--yes", "exec", "@example/mcp-server"]},
    # check-11 e03: the '--with' extra is installed on every run too
    "uvx-with": {"command": "uvx", "args": ["--with", "requests", "example-mcp==1.2.3"]},
    # skeptic x03: Windows launchers, also a full path with a space
    "npx-cmd": {"command": "npx.cmd", "args": ["-y", "@example/mcp-server"]},
    "npx-fullpath": {"command": "C:\\Program Files\\nodejs\\npx.cmd", "args": ["-y", "@example/mcp-server"]},
    "uvx-exe": {"command": "uvx.exe", "args": ["example-mcp"]},
    # other wrappers
    "timeout": {"command": "timeout", "args": ["30", "npx", "-y", "@example/mcp-server"]},
    "cmd-string": {"command": "cmd", "args": ["/c", "npx -y @example/mcp-server"]},
}


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_runners_behind_wrappers_are_classified(tmp_path: Path, codex: bool) -> None:
    result = _mcp(_WRAPPED, tmp_path, codex=codex)

    findings = _by_server(result)
    servers = _servers(result)
    for name in _WRAPPED:
        assert findings.get(name) == {"mcp_unpinned_package": Severity.MEDIUM}, (name, findings.get(name))
        assert servers[name]["pinned"] is False, servers[name]
    assert "--with" in servers["uvx-with"]["pin_detail"]


def test_value_flags_and_pinned_wrappers_stay_pinned(tmp_path: Path) -> None:
    servers = {
        # skeptic x07: '--loglevel warn' is a flag value, not the package
        "npx-loglevel-pinned": {"command": "npx", "args": ["--yes", "--loglevel", "warn", "@example/mcp-server@1.2.3"]},
        "cmd-pinned": {"command": "cmd", "args": ["/c", "npx", "-y", "@example/mcp-server@1.2.3"]},
        "uvx-with-pinned": {"command": "uvx", "args": ["--with", "requests==2.32.3", "example-mcp==1.2.3"]},
        "go-run-exact": {"command": "go", "args": ["run", "github.com/example/mcp@v1.2.3"]},
        "uv-run-project": {"command": "uv", "args": ["run", "example-mcp"]},
    }
    result = _mcp(servers, tmp_path)

    assert _by_server(result) == {}
    pinned = {name: row["pinned"] for name, row in _servers(result).items()}
    assert pinned == {
        "npx-loglevel-pinned": True,
        "cmd-pinned": True,
        "uvx-with-pinned": True,
        "go-run-exact": True,
        "uv-run-project": None,
    }


# --------------------------------------------------------------------------- #
# L16: hooks get the same rules as MCP and LSP servers                         #
# --------------------------------------------------------------------------- #
def _hook_file(*commands: str) -> dict:
    return {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": command}]} for command in commands]}}


def _hook_findings(result: ValidationResult) -> list[tuple[str, Severity, str]]:
    return [
        (finding.check_name, finding.severity, finding.message)
        for finding in result.findings
        if finding.check_name in {"plugin_hook_unpinned_package", "plugin_hook_command_floating_version"}
    ]


@pytest.mark.parametrize("codex", [False, True], ids=["claude", "codex"])
def test_hook_latest_is_high_and_docker_is_classified(tmp_path: Path, codex: bool) -> None:
    """check-11 e06: hook '@latest' was only MEDIUM and an untagged docker image had no finding."""
    hooks = _hook_file(
        "npx -y @example/hook-tool@latest",
        "docker run --rm ghcr.io/example/hook-image",
        "env CI=1 npx -y @example/hook-tool",
    )
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", {"hooks/hooks.json": hooks}, codex=codex))

    found = _hook_findings(result)
    floating = [message for check, severity, message in found if check == "plugin_hook_command_floating_version"]
    unpinned = [message for check, severity, message in found if check == "plugin_hook_unpinned_package"]
    assert all(severity == Severity.HIGH for check, severity, _ in found if check.endswith("floating_version"))
    assert len(floating) == 1 and "@example/hook-tool@latest" in floating[0]
    assert len(unpinned) == 2
    assert any("ghcr.io/example/hook-image" in message for message in unpinned)


def test_hook_wrappers_are_looked_through(tmp_path: Path) -> None:
    """skeptic x06: bash -c, timeout, npx.cmd, uvx --with, and cd && npx."""
    hooks = _hook_file(
        "bash -c 'npx -y @example/hook-a'",
        "timeout 30 npx -y @example/hook-b",
        "npx.cmd -y @example/hook-c",
        "uvx --with requests example-hook==1.0.0",
        'cd "${CLAUDE_PLUGIN_ROOT}" && npx -y @example/hook-e',
    )
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", {"hooks/hooks.json": hooks}))

    messages = [message for check, _severity, message in _hook_findings(result)]
    for package in ("@example/hook-a", "@example/hook-b", "@example/hook-c", "'requests'", "@example/hook-e"):
        assert any(package in message for message in messages), (package, messages)


def test_pinned_hook_runs_pass(tmp_path: Path) -> None:
    hooks = _hook_file(
        "npx -y @example/hook-tool@1.2.3",
        f"docker run --rm ghcr.io/example/hook-image@{_DIGEST}",
        "bash -c 'npx -y @nextui-org/hook@1.0.0'",
    )
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", {"hooks/hooks.json": hooks}))

    assert _hook_findings(result) == []


def test_lsp_floating_rule_reads_the_version_position(tmp_path: Path) -> None:
    lsp = {
        "ts-latest": {"command": "npx", "args": ["-y", "typescript-language-server@latest", "--stdio"]},
        "scoped-pinned": {"command": "npx", "args": ["-y", "@nextui-org/lsp@1.0.0", "--stdio"]},
    }
    for server in lsp.values():
        server["extensionToLanguage"] = {".ts": "typescript"}
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", {".lsp.json": lsp}))

    floating = [f for f in result.findings if f.check_name == "plugin_lsp_command_floating_version"]
    assert [f.metadata["plugin_component"]["name"] for f in floating] == ["ts-latest"]
    assert floating[0].severity == Severity.HIGH
