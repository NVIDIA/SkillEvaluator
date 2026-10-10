# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No report format copies a secret (checks 4, 9, 10): proof bug H9, Tier 1 half.

The fixtures follow check-04 edge-10 (a ``ghp_`` token in MCP args), check-10
p03 (a token in a hook script, found by the PII scan), and skeptic x01 (a
password inside ``${VAR:-https://user:pw@host}``). The real ``validate`` command
writes every report format, and each one must hold zero copies of each fake
secret while the findings still fire. All values are fake and built at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.plugin_eval import _reject_unsafe_mcp_declaration
from skillevaluator.validators.mcp_static import redacted_url

# Fake values with real token shapes, assembled so the source never holds a token literal.
_GHP_ARG = "ghp" + "_" + "FAKE" + "0" * 32
_GHP_SCRIPT = "ghp" + "_" + "FAKEscriptTOKEN" + "1" * 21
_URL_PASSWORD = "AUDITFAKEPASS" + "99"
_QUERY_TOKEN = "AUDITFAKETOKEN" + "123456"
_GIT_PASSWORD = "FAKEGITPASS" + "123456"
_CMD_TOKEN = "FAKECMDTOKEN" + "123456"
_PATH_DEFAULT_TOKEN = "LEAKPROBE" + "7782"
# Values with no known token shape: in program text, in a container image spec, and one the PII scan reads as a
# Bitcoin address (check-04 pos-38/edge-42, check-10 s03-e3).
_PROGRAM_TOKEN = "FAKEpyTok3n" + "0123456789abcdef"
_IMAGE_PASSWORD = "FAKEimgPW" + "0123456789"
_ADDRESS_SHAPED_TOKEN = "1" + "FAKEbtcShapedTokenAbcdefgh"
_SECRETS = (
    _GHP_ARG,
    _GHP_SCRIPT,
    _URL_PASSWORD,
    _QUERY_TOKEN,
    _GIT_PASSWORD,
    _CMD_TOKEN,
    _PATH_DEFAULT_TOKEN,
    _PROGRAM_TOKEN,
    _IMAGE_PASSWORD,
    _ADDRESS_SHAPED_TOKEN,
)

_CODEX_MANIFEST = {
    "name": "leaky",
    "version": "1.0.0",
    "description": "Secret echo probe (Codex)",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "Secret echo probe",
        "shortDescription": "Secret echo probe",
        "longDescription": "A plugin used to test that reports never copy secrets.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}


def _servers() -> dict[str, dict]:
    return {
        # check-04 edge-10
        "leaky-arg": {"command": "npx", "args": ["-y", "@scope/server@1.2.3", _GHP_ARG]},
        # skeptic x01 / audit re-test check-09-codex-urls
        "pw-in-default": {
            "type": "http",
            "url": f"${{MCP_URL:-https://audituser:{_URL_PASSWORD}@api.example.com/mcp}}",
        },
        "query-token": {"type": "http", "url": f"https://api.example.com/mcp?token={_QUERY_TOKEN}"},
        # review round: a query inside a path-position default ('http' so a finding shows the URL)
        "query-in-path-default": {
            "type": "http",
            "url": f"http://api.example.com${{MCP_PATH:-/mcp?access_token={_PATH_DEFAULT_TOKEN}}}",
        },
        # a package spec that is a URL with a credential in it
        "git-spec": {"command": "npx", "args": ["-y", f"git+https://bot:{_GIT_PASSWORD}@github.com/example/mcp.git"]},
        # a whole command line with a credential flag and a shell operator
        "cmd-line": {"command": f"npx -y @scope/server@1.2.3 --token {_CMD_TOKEN} | tee log"},
        # check-04 pos-38: a credential assigned inside an inline program
        "py-env": {"command": "python3", "args": ["-c", f"import os; os.environ['API_KEY']='{_PROGRAM_TOKEN}'"]},
        # check-04 edge-42: user:password@ in a container image spec
        "docker-cred": {
            "command": "docker",
            "args": ["run", "-i", f"bob:{_IMAGE_PASSWORD}@registry.example.com/i:latest"],
        },
    }


def _write(root: Path, files: dict[str, object]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _plugin(root: Path, fmt: str) -> Path:
    manifest = (
        {".codex-plugin/plugin.json": _CODEX_MANIFEST}
        if fmt == "codex"
        else {".claude-plugin/plugin.json": {"name": "leaky", "version": "1.0.0", "description": "Secret echo probe"}}
    )
    hooks = {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "sh ./scripts/post-status.sh"}]}]}}
    return _write(
        root,
        {
            **manifest,
            ".mcp.json": {"mcpServers": _servers()},
            "hooks/hooks.json": hooks,
            # check-10 p03: a token in a shipped script, found by the PII scan
            "scripts/post-status.sh": f'#!/bin/sh\nTOKEN="{_GHP_SCRIPT}"\necho "status posted" >&2\n',
            # check-10 s03-e3: a credential the PII scan reports as a Bitcoin address
            "scripts/config.yaml": f'service:\n  api_token: "{_ADDRESS_SHAPED_TOKEN}"\n',
        },
    )


def _run(root: Path, out: Path) -> tuple[int, str]:
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(root),
            "--type",
            "plugin",
            "--tiers",
            "1",
            "--no-llm",
            "--checks",
            "schema,pii",
            "-r",
            "cli,json,markdown,html,sarif",
            "-o",
            str(out),
        ],
    )
    return result.exit_code, result.output


@pytest.mark.parametrize("fmt", ["claude", "codex"])
def test_no_report_format_copies_a_secret(tmp_path: Path, fmt: str) -> None:
    out = tmp_path / "out"
    exit_code, cli_output = _run(_plugin(tmp_path / "plugin", fmt), out)

    assert exit_code == 1, cli_output
    reports = {path.name: path.read_text(encoding="utf-8") for path in out.iterdir() if path.is_file()}
    suffixes = {name.rsplit(".", 1)[-1] for name in reports}
    assert {"json", "md", "html"} <= suffixes and any(name.endswith(".sarif.json") for name in reports)
    assert "BENCHMARK.md" in reports
    copies = {
        (name, secret): text.count(secret)
        for name, text in {**reports, "cli": cli_output}.items()
        for secret in _SECRETS
        if secret in text
    }
    assert copies == {}

    # The findings still fire; only their text is redacted.
    [data] = [json.loads(text) for name, text in reports.items() if name.endswith(".json") and "sarif" not in name]
    checks = {finding["check_name"] for result in data["results"] for finding in result.get("findings", [])}
    assert {"mcp_command_inline_secret", "mcp_url_inline_secret", "github_tokens"} <= checks


def test_pii_finding_keeps_a_redacted_value(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(_plugin(tmp_path / "plugin", "claude"), out)

    [report] = [path for path in out.glob("*.json") if not path.name.endswith(".sarif.json")]
    findings = [
        finding
        for result in json.loads(report.read_text())["results"]
        for finding in result.get("findings", [])
        if finding["check_name"] == "github_tokens"
    ]
    assert findings
    for finding in findings:
        assert finding["metadata"]["matched_value"].startswith("ghp_…[redacted")
        assert finding["metadata"]["value_redacted"] is True
        assert "<redacted>" in finding["line_content"]


def test_tier3_refusal_does_not_echo_the_secret() -> None:
    with pytest.raises(ValueError, match="mcp_command_inline_secret") as raised:
        _reject_unsafe_mcp_declaration("leaky-arg", _servers()["leaky-arg"])
    assert _GHP_ARG not in str(raised.value)


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        (
            f"${{MCP_URL:-https://audituser:{_URL_PASSWORD}@api.example.com/mcp}}",
            "${MCP_URL:-https://api.example.com/mcp}",
        ),
        # Expanded, '/x' follows the default's '?', so it is part of the query and is not shown either.
        (f"${{A:-https://api.example.com/mcp?token={_QUERY_TOKEN}}}/x", "${A:-https://api.example.com/mcp}"),
        (f"//bot:{_URL_PASSWORD}@host.example/x", "//host.example/x"),
        (f"https://u:{_URL_PASSWORD}@host.example/x?token={_QUERY_TOKEN}#f", "https://host.example/x"),
        # Review round: what is hidden is decided on the expanded URL, so a query in a path default, after a
        # reference, or user information split over references never shows.
        (f"https://api.example.com${{B:-/x?token={_QUERY_TOKEN}}}", "https://api.example.com${B:-/x}"),
        (
            f"http://api.example.com${{MCP_PATH:-/mcp?access_token={_PATH_DEFAULT_TOKEN}}}",
            "http://api.example.com${MCP_PATH:-/mcp}",
        ),
        (f"${{A:-https://h.example/mcp?x=1}}&token={_QUERY_TOKEN}", "${A:-https://h.example/mcp}"),
        (f"${{A:-https://h.example}}/mcp?token={_QUERY_TOKEN}", "${A:-https://h.example}/mcp"),
        (f"${{BASE}}/mcp?token={_QUERY_TOKEN}", "${BASE}/mcp"),
        (f"https://${{U:-user}}:${{PW:-{_URL_PASSWORD}}}@api.example.com/mcp", "https://api.example.com/mcp"),
        (f"https://${{H:-u:{_URL_PASSWORD}@api.example.com}}/mcp", "https://${H:-api.example.com}/mcp"),
        (f"https://api.example.com/mcp#${{F:-{_QUERY_TOKEN}}}", "https://api.example.com/mcp"),
        ("https://${HOST}/mcp", "https://${HOST}/mcp"),
    ],
)
def test_redacted_url_strips_userinfo_anywhere(url: str, shown: str) -> None:
    assert redacted_url(url) == shown


def test_redaction_keeps_specs_that_only_look_like_userinfo() -> None:
    """A digest after a tag, an npm alias, or a git ref is not user information and stays readable."""
    from skillevaluator.validators.mcp_static import redact_secrets

    assert redacted_url(f"${{X:-https:user:{_URL_PASSWORD}@host.example/mcp}}/x") == "${X:-https:host.example/mcp}/x"
    assert redact_secrets("ghcr.io/example/mcp:1.2.3@sha256:abc") == "ghcr.io/example/mcp:1.2.3@sha256:abc"
    assert redact_secrets("npm:pkg@1.2.3") == "npm:pkg@1.2.3"
    assert redact_secrets(f"git+https://bot:{_GIT_PASSWORD}@github.com/x/y.git#v1.2.3") == (
        "git+https://github.com/x/y.git#v1.2.3"
    )
