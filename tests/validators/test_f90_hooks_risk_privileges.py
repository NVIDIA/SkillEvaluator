# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M18 and L11: subagent, command, and skill privileges read the way the real clients read them.

Fixtures follow the proof's ``check-06`` examples: ``fmt-codex-commands-skills``
and ``edge-07-codex-folder-agents-claude-view`` (M18), ``edge-01``, ``edge-04``,
``edge-06``, and ``xref-01-skill-allowed-tools-list`` (L11).
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from skillevaluator.models.result import Severity
from skillevaluator.reporting.cli import print_plugin_tier1_static
from skillevaluator.reporting.plugin_sections import privileges_view
from skillevaluator.tier3.plugin_native import _bypass_frontmatter_refusal
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.schema import SchemaValidator

if TYPE_CHECKING:
    from skillevaluator.models.result import Finding, ValidationResult

_CODEX_MANIFEST = {
    "name": "c06-codex",
    "version": "1.0.0",
    "description": "Check 6 Codex example.",
    "author": {"name": "Check Six"},
    "skills": "./skills/",
    "interface": {
        "displayName": "Check Six",
        "shortDescription": "Check 6 example",
        "longDescription": "A Check 6 example plugin.",
        "developerName": "Check Six",
        "category": "Developer Tools",
        "capabilities": [],
        "websiteURL": "https://example.com/",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Use the check six plugin."],
    },
}
_SKILL = (
    "---\nname: shell\ndescription: Answer a small question. Use when the user asks a short question.\n"
    "allowed-tools: Bash\nmetadata:\n  author: Check Six <check6@example.com>\n---\n# Shell\n\n"
    "## Instructions\n\nAnswer the question in one sentence.\n\n## Examples\n\n"
    'Input: "hello". Output: "a short answer".\n'
)


def _write(root: Path, files: dict[str, str | dict]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content))
    return root


def _md(frontmatter: str, body: str = "Do the task, then report what you did.") -> str:
    return f"---\n{frontmatter}\n---\n{body}\n"


def _validate(root: Path) -> ValidationResult:
    return PluginSchemaValidator().validate(root)


def _grants(result: ValidationResult, prefix: str = "plugin_") -> list[Finding]:
    return [
        finding
        for finding in result.findings
        if finding.check_name.startswith(prefix) and finding.check_name.split("_")[1] in {"agent", "command", "skill"}
    ]


# --------------------------------------------------------------------------- #
# M18                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("manifest_dir", [".codex-plugin", ".claude-plugin"])
def test_allowed_tools_bash_is_claude_only_on_a_codex_plugin(tmp_path: Path, manifest_dir: str) -> None:
    manifest = _CODEX_MANIFEST if manifest_dir == ".codex-plugin" else {"name": "c06-claude", "version": "1.0.0"}
    plugin = _write(
        tmp_path / "plugin",
        {
            f"{manifest_dir}/plugin.json": manifest,
            "skills/shell/SKILL.md": _SKILL,
            "commands/any.md": _md("description: Run any shell command.\nallowed-tools: Bash"),
            "commands/status.md": _md("description: Git status.\nallowed-tools: Bash(git status:*)"),
        },
    )

    result = _validate(plugin)
    grants = {finding.check_name: finding for finding in _grants(result)}

    assert set(grants) == {"plugin_skill_unrestricted_bash", "plugin_command_unrestricted_bash"}
    if manifest_dir == ".codex-plugin":
        for finding in grants.values():
            assert finding.severity == Severity.MEDIUM
            assert "Codex ignores allowed-tools" in finding.message
            assert "Claude Code" in finding.message
        assert result.passed
        rows = {row["name"]: row for row in result.metadata["plugin"]["privileges"]["components"]}
        assert "claude_only_grant" in rows["shell"]["flags"]
    else:
        assert {finding.severity for finding in grants.values()} == {Severity.HIGH}
        assert all("Codex" not in finding.message for finding in grants.values())


def test_claude_view_of_a_codex_folder_names_claude_code_and_does_not_fail(tmp_path: Path) -> None:
    plugin = _write(
        tmp_path / "plugin",
        {
            ".codex-plugin/plugin.json": _CODEX_MANIFEST,
            "skills/alpha/SKILL.md": _SKILL.replace("name: shell", "name: alpha").replace("allowed-tools: Bash\n", ""),
            "agents/bypass.md": _md(
                "name: bypass\ndescription: Would be risky in Claude Code.\ntools: Bash\n"
                "permissionMode: bypassPermissions"
            ),
        },
    )

    result = _validate(plugin)
    bypass = next(finding for finding in result.findings if finding.check_name == "plugin_agent_bypass_permissions")

    assert bypass.severity == Severity.MEDIUM
    assert bypass.message.startswith("loaded only by Claude Code (--plugin-dir")
    assert bypass.metadata["loaded_by"].startswith("Claude Code")
    assert result.passed


def test_codex_only_command_hidden_by_a_claude_command_map_has_no_effect(tmp_path: Path) -> None:
    """``check-06/pos-07-command-map-override``: the map replaces commands/ in Claude Code, but Codex reads it."""
    plugin = _write(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {
                "name": "c06-map",
                "version": "1.0.0",
                "description": "Check 6 example plugin c06-map.",
                "commands": {"deploy": {"source": "./cmds/deploy.md", "description": "Deploy"}},
            },
            "cmds/deploy.md": _md("description: Deploy.\nallowed-tools: Bash(git status:*)"),
            "commands/ignored.md": _md("description: Not loaded by Claude Code.\nallowed-tools: Bash"),
        },
    )

    result = _validate(plugin)
    ignored = [finding for finding in _grants(result) if str(finding.file_path).endswith("commands/ignored.md")]
    rows = {row["name"]: row for row in result.metadata["plugin"]["privileges"]["components"]}
    view = privileges_view(result.metadata["plugin"]["privileges"])

    assert [finding.severity for finding in ignored] == [Severity.LOW]
    message = ignored[0].message
    assert message.startswith("loaded only by Codex")
    assert "has no effect" in message
    assert "Claude Code does not load this command" in message
    assert "if Claude Code loads this command" not in message
    assert result.passed
    assert "inert_grant" in rows["ignored"]["flags"]
    assert "claude_only_grant" not in rows["ignored"]["flags"]
    assert result.metadata["plugin"]["privileges"]["counts"]["flagged"] == 0
    assert view is not None
    ignored_row = next(row for row in view["rows"] if row["name"] == "ignored")
    assert not ignored_row["risky"]
    assert "no effect in either client" in ignored_row["flags"]


# --------------------------------------------------------------------------- #
# L11                                                                         #
# --------------------------------------------------------------------------- #
def _claude(root: Path, files: dict) -> Path:
    return _write(root, {".claude-plugin/plugin.json": {"name": "c06", "version": "1.0.0"}, **files})


def test_deny_matching_follows_claude_code(tmp_path: Path) -> None:
    plugin = _claude(
        tmp_path / "plugin",
        {
            "agents/deny-lowercase.md": _md("name: deny-lowercase\ndescription: d\ntools: Bash\ndisallowedTools: bash"),
            "agents/deny-star.md": _md('name: deny-star\ndescription: d\ntools: Bash\ndisallowedTools: "Bash(*)"'),
            "agents/deny-scoped.md": _md(
                'name: deny-scoped\ndescription: d\ntools: Bash\ndisallowedTools: "Bash(rm *)"'
            ),
        },
    )

    flagged = {
        finding.metadata["plugin_component"]["name"]
        for finding in _validate(plugin).findings
        if finding.check_name == "plugin_agent_unrestricted_bash"
    }

    # Tool names are case-sensitive ('bash' denies nothing); 'Bash(*)' removes the whole tool.
    assert flagged == {"deny-lowercase", "deny-scoped"}


def test_bypass_spelling_agrees_between_tier1_and_tier3(tmp_path: Path) -> None:
    frontmatter = "name: mode-case\ndescription: d\ntools: Read\npermissionMode: BypassPermissions"
    plugin = _claude(tmp_path / "plugin", {"agents/mode-case.md": _md(frontmatter)})

    checks = {finding.check_name: finding.severity for finding in _validate(plugin).findings}

    assert checks.get("plugin_agent_bypass_permissions") == Severity.HIGH
    assert _bypass_frontmatter_refusal({"permissionMode": "BypassPermissions"}, where="agent") is not None


def test_string_true_disables_model_invocation(tmp_path: Path) -> None:
    plugin = _claude(
        tmp_path / "plugin",
        {"commands/dmi-string.md": _md('description: d\ndisable-model-invocation: "true"\nallowed-tools: Bash')},
    )

    result = _validate(plugin)
    row = next(row for row in result.metadata["plugin"]["privileges"]["components"] if row["name"] == "dmi-string")
    finding = next(finding for finding in result.findings if finding.check_name == "plugin_command_unrestricted_bash")

    assert row["model_invocable"] is False
    assert "only the user can invoke it" in finding.message


def test_star_in_allowed_tools_pre_approves_nothing(tmp_path: Path) -> None:
    plugin = _claude(
        tmp_path / "plugin",
        {
            "commands/all.md": _md('description: d\nallowed-tools: "*"'),
            "commands/mcp.md": _md('description: d\nallowed-tools: "mcp__*"'),
        },
    )

    found = {
        finding.metadata["plugin_component"]["name"]: finding.severity
        for finding in _validate(plugin).findings
        if finding.check_name == "plugin_command_wildcard_tools"
    }

    assert found == {"all": Severity.LOW, "mcp": Severity.MEDIUM}


def test_privilege_headers_count_skills(tmp_path: Path) -> None:
    plugin = _claude(
        tmp_path / "plugin",
        {"skills/shell/SKILL.md": _SKILL, "commands/any.md": _md("description: d\nallowed-tools: Bash")},
    )
    view = privileges_view(_validate(plugin).metadata["plugin"]["privileges"])
    output = StringIO()

    print_plugin_tier1_static({"privileges": view}, Console(file=output, width=200, color_system=None))

    assert view is not None
    assert view["skills"] == 1
    assert "0 subagent(s), 1 command(s), 1 skill(s); 2 flagged" in output.getvalue()


def test_yaml_list_allowed_tools_is_valid_skill_frontmatter(tmp_path: Path) -> None:
    skill = _write(
        tmp_path / "skills" / "lister",
        {
            "SKILL.md": "---\nname: lister\ndescription: Answer a small question. Use when the user asks a short "
            "question.\nallowed-tools:\n  - Read\n  - Grep\nmetadata:\n  author: Check Six <check6@example.com>\n"
            "---\n# Lister\n\nAnswer the question in one sentence.\n"
        },
    )

    result = SchemaValidator().validate(skill)

    assert not [finding for finding in result.findings if finding.check_name == "frontmatter_field"]


def test_other_cross_client_findings_name_the_client_and_keep_their_severity(tmp_path: Path) -> None:
    plugin = _write(
        tmp_path / "plugin",
        {
            ".codex-plugin/plugin.json": _CODEX_MANIFEST,
            ".lsp.json": {"py": {"command": "bash", "args": ["-c", "pyright-langserver --stdio"]}},
        },
    )

    lsp = next(
        finding for finding in _validate(plugin).findings if finding.check_name == "plugin_lsp_command_dangerous_form"
    )

    # Codex loads no LSP servers; Claude Code --plugin-dir would start this one, so it stays CRITICAL.
    assert lsp.severity == Severity.CRITICAL
    assert lsp.message.startswith("loaded only by Claude Code (--plugin-dir")
