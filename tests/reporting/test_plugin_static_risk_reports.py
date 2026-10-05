# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 plugin static-risk sections (hooks, privileges, parity, CVEs, endpoints) in every report."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
from _plugin_fixtures import HOSTILE, element_text, tier1_plugin_result

from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.plugin_sections import (
    cve_summary_view,
    endpoint_resolution_view,
    hook_risk_view,
    privileges_view,
    static_risk_view,
    tier1_plugin_view,
    validator_parity_view,
)

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult

STATIC_RISK: dict[str, Any] = {
    "hook_risk": {
        "hooks": [
            {
                "id": "hooks/hooks.json#SessionStart[0].hooks[0]",
                "source": "hooks/hooks.json",
                "file": "hooks/hooks.json",
                "event": "SessionStart",
                "matcher": None,
                "handler_type": "command",
                "target": "echo ok",
                "risk_flags": ["context_injection"],
            },
            {
                "id": "hooks/hooks.json#PreToolUse[0].hooks[0]",
                "source": "hooks/hooks.json",
                "file": "hooks/hooks.json",
                "event": "PreToolUse",
                "matcher": "Bash",
                "handler_type": "command",
                "target": HOSTILE,
                "risk_flags": ["auto_approve", "remote_code"],
            },
            "not-a-hook",
        ],
        "counts": {"total": 2, "flagged": 2, "by_flag": {"auto_approve": 1, "context_injection": 1, "remote_code": 1}},
    },
    "privileges": {
        "components": [
            {
                "type": "agent",
                "name": "reviewer",
                "path": "agents/reviewer.md",
                "flags": ["inherits_all_tools"],
                "tools": None,
                "inherits_all_tools": True,
            },
            {
                "type": "command",
                "name": "ship",
                "path": "commands/ship.md",
                "flags": ["unrestricted_bash"],
                "allowed_tools": ["Bash"],
                "model": "opus",
                "model_invocable": True,
            },
        ],
        "counts": {"agents": 1, "commands": 1},
    },
    "validator_parity": {
        "status": "compared",
        "claude_verdict": "failed",
        "skillevaluator_verdict": "passed",
        "agree": False,
        "error_count": 1,
        "warning_count": 0,
        "errors": [".claude-plugin/plugin.json: name: Invalid input"],
        "warnings": [],
    },
    "cve_summary": {
        "method": "declared_exact_pins",
        "ecosystems": {
            "python": {"status": "not_found"},
            "npm": {
                "status": "audited",
                "audited": 3,
                "unverified": 1,
                "scanners": ["osv-scanner"],
                "vulnerabilities": {"high": 2, "low": 1},
            },
            "container": {
                "status": "incomplete",
                "audited": 0,
                "unverified": 0,
                "scanners": [],
                "vulnerabilities": {},
                "errors": ["no container vulnerability scanner is installed"],
            },
        },
    },
    "endpoint_resolution": {
        "enabled": True,
        "endpoints": [
            {
                "kind": "mcp",
                "name": "remote",
                "url": "https://mcp.example.com/mcp",
                "status": "private",
                "addresses": ["10.0.0.5"],
                "head": {"skipped": "not contacted"},
            },
            {
                "kind": "hook",
                "name": "h#1",
                "url": "https://hooks.example.com/h",
                "status": "public",
                "addresses": ["93.184.216.34"],
                "head": {"status": 302, "location": "https://in.example.com/"},
                "redirect": {"url": "https://in.example.com/", "classification": "private"},
            },
        ],
        "counts": {"private": 1, "public": 1},
    },
}


def _result(extra: dict[str, Any] | None = None) -> ValidationResult:
    result = tier1_plugin_result()
    result.metadata["plugin"].update(deepcopy(STATIC_RISK if extra is None else extra))
    return result


def test_views_are_none_without_keys_and_tolerate_malformed_values() -> None:
    assert static_risk_view({"name": "p"}) is None
    assert tier1_plugin_view({"name": "p", "manifest_type": "x"})["static_risk"] is None
    assert hook_risk_view({"hooks": "nope"}) is None
    assert privileges_view({"components": [1, 2]}) is None
    assert cve_summary_view({"ecosystems": {"npm": {"status": "not_found"}}}) is None
    assert validator_parity_view([]) is None
    assert endpoint_resolution_view({"endpoints": None, "enabled": True})["rows"] == []


def test_views_order_flagged_rows_first() -> None:
    view = static_risk_view(STATIC_RISK)
    assert view is not None
    assert [row["event"] for row in view["hooks"]["rows"]] == ["PreToolUse", "SessionStart"]
    assert view["hooks"]["rows"][0]["flags"] == ["auto-approves", "runs remote code"]
    assert [row["name"] for row in view["privileges"]["rows"]] == ["ship", "reviewer"]
    assert view["privileges"]["flagged"] == 1
    assert [row["ecosystem"] for row in view["cve"]["rows"]] == ["npm", "container"]
    assert view["cve"]["incomplete"] is True
    assert view["cve"]["rows"][0]["severity_label"] == "2 high, 1 low"
    assert view["parity"]["agreement"] == "disagree"


def test_json_report_carries_the_raw_blocks() -> None:
    data = json.loads(JSONReporter(include_timestamp=False).render_all([_result()]))
    for key in ("hook_risk", "privileges", "validator_parity", "cve_summary", "endpoint_resolution"):
        assert key in data["plugin"]


def test_markdown_renders_static_risk_sections() -> None:
    markdown = MarkdownReporter(include_timestamp=False).render_all([_result()])
    assert "### Subagent, command, and skill privileges (1 agents, 1 commands, 0 skills; 1 flagged)" in markdown
    assert "| command | ship | Bash | opus | — | user and model | unrestricted Bash |" in markdown
    assert "### Hook risk (2 handlers; 2 flagged)" in markdown
    assert HOSTILE not in markdown
    assert "### Dependency CVE audit" in markdown
    assert "| npm | Audited | osv-scanner | 3 | 1 | 2 high, 1 low |" in markdown
    assert "container audit incomplete:** no container vulnerability scanner is installed" in markdown
    assert "**claude plugin validate:** failed" in markdown
    assert "### Endpoint DNS and redirect checks (2 endpoints)" in markdown


def test_html_renders_and_escapes_static_risk_sections() -> None:
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all([_result()])
    assert HOSTILE not in html
    hooks = element_text(html, "plugin-hook-risk") or ""
    assert "Hook risk (2 handlers) 2 flagged" in hooks
    assert "auto-approves" in hooks
    privileges = element_text(html, "plugin-privileges") or ""
    assert "ship" in privileges and "unrestricted Bash" in privileges
    cve = element_text(html, "plugin-cve-summary") or ""
    assert "INCOMPLETE" in cve and "osv-scanner" in cve
    parity = element_text(html, "plugin-validator-parity") or ""
    assert "disagree" in parity and "Invalid input" in parity
    endpoints = element_text(html, "plugin-endpoint-resolution") or ""
    assert "10.0.0.5" in endpoints and "https://in.example.com/ (private)" in endpoints


def test_html_without_static_risk_has_no_new_sections() -> None:
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all([_result({})])
    for section in ("plugin-hook-risk", "plugin-privileges", "plugin-cve-summary", "plugin-validator-parity"):
        assert element_text(html, section) is None


def test_cli_prints_static_risk_summary() -> None:
    output = CLIReporter().render_all([_result()])
    plain = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())
    assert "Plugin privileges: 1 subagent(s), 1 command(s), 0 skill(s); 1 flagged" in plain
    assert "command ship: unrestricted Bash" in plain
    assert "Plugin hooks: 2 handler(s); 2 flagged" in plain
    assert "Dependency CVE audit: npm Audited (3 audited, 1 unverified, 2 high, 1 low)" in plain
    assert "claude plugin validate: failed (1 errors, 0 warnings); SkillEvaluator passed (disagree)" in plain
    assert "Endpoint DNS/redirect checks: 2 endpoint(s) (private=1, public=1)" in plain


@pytest.mark.parametrize("matcher", ["/x", "mcp__memory__.*", "Bash", "[/bold]"])
def test_cli_prints_plugin_controlled_hook_matchers_literally(matcher: str) -> None:
    # "[/x]" used to raise rich.errors.MarkupError and "[mcp__memory__.*]" was
    # swallowed as a style tag: the matcher's brackets must be escaped too.
    risk = deepcopy(STATIC_RISK)
    risk["hook_risk"]["hooks"][1]["matcher"] = matcher

    output = CLIReporter().render_all([_result(risk)])

    plain = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())
    assert f"- PreToolUse [{matcher}] command: auto-approves, runs remote code" in plain
    assert "- SessionStart [(all)] command:" in plain


def test_endpoint_redirect_to_metadata_or_private_is_not_a_green_row() -> None:
    endpoints = [
        {
            "kind": "mcp",
            "name": name,
            "url": f"https://{name}.example.com/mcp",
            "status": "public",
            "addresses": ["93.184.216.34"],
            "head": {"status": 302},
            "redirect": redirect,
        }
        for name, redirect in (
            ("meta", {"url": "http://169.254.169.254/latest", "classification": "metadata", "downgrade": True}),
            ("priv", {"url": "https://in.example.com/", "classification": "private"}),
            ("down", {"url": "http://docs.example.com/", "classification": "public", "downgrade": True}),
            ("fine", {"url": "https://docs.example.com/", "classification": "public"}),
        )
    ]
    block = {"enabled": True, "endpoints": endpoints, "counts": {"public": 4}}
    view = endpoint_resolution_view(block)

    assert view is not None
    assert {row["name"]: row["status_class"] for row in view["rows"]} == {
        "meta": "fail",
        "priv": "warn",
        "down": "warn",
        "fine": "ok",
    }
    assert view["redirects_flagged"] == 3
    output = CLIReporter().render_all([_result({"endpoint_resolution": block})])
    plain = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())
    assert "Endpoint DNS/redirect checks: 4 endpoint(s) (public=4); 3 flagged redirect(s)" in plain
