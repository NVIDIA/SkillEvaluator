# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component path rules match what Claude Code, Codex, and Cursor really load.

Each test rebuilds a plugin from the plugin-evaluation proof (check 2 and its
skeptic probes) and checks the finding and inventory a client-accurate check
gives: Claude Code rejects a whole manifest for a path without ``./`` or an
agent that is not a ``.md`` file (proof H7), accepts a commands-map folder
source (M20), Codex drops paths it does not accept and loads the default
instead (M19), Cursor may read its root ``mcp.json`` next to a declared
``mcpServers`` (H12), and Cursor documents root variables only outside
component paths (L7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_SKILL = "---\nname: {name}\ndescription: Demo skill {name}\n---\n# {name}\n\nUse the demo.\n"
_AGENT = "---\nname: reviewer\ndescription: Reviews text for typos.\n---\n\nReview the text.\n"
_CODEX_INTERFACE = {
    "displayName": "Demo",
    "shortDescription": "Demo plugin",
    "longDescription": "A demo plugin.",
    "developerName": "Example",
    "category": "Developer Tools",
    "capabilities": [],
    "defaultPrompt": ["Use the demo."],
}
_CODEX = {
    "name": "demo",
    "version": "1.0.0",
    "description": "Demo Codex plugin",
    "author": {"name": "Example"},
    "interface": _CODEX_INTERFACE,
}
_PINNED = {"command": "npx", "args": ["-y", "@scope/fs@1.2.3", "."]}


def _write(root: Path, files: dict[str, str | dict]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _validate(root: Path) -> ValidationResult:
    return PluginSchemaValidator().validate(root)


def _findings(result: ValidationResult, check_name: str) -> list:
    return [finding for finding in result.findings if finding.check_name == check_name]


def _rows(result: ValidationResult, component_type: str | None = None) -> list[dict]:
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    return [row for row in rows if component_type is None or row["type"] == component_type]


# --------------------------------------------------------------------------- #
# H7: Claude Code rejects the manifest; Codex only drops the value              #
# --------------------------------------------------------------------------- #
def test_claude_paths_without_dot_slash_block_like_claude_code_does(tmp_path: Path) -> None:
    """Proof check-02 P05: 8 paths without './' passed with MEDIUM; Claude Code says 'Invalid input' x8."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {
                "name": "no-dot-slash",
                "skills": "extra-skills/",
                "agents": "agents/reviewer.md",
                "commands": "commands/hello.md",
                "hooks": "hooks/hooks.json",
                "mcpServers": "servers.json",
            },
            "extra-skills/summarize/SKILL.md": _SKILL.format(name="summarize"),
            "agents/reviewer.md": _AGENT,
            "commands/hello.md": "---\ndescription: Say hello.\n---\nSay hello.\n",
            "hooks/hooks.json": {"hooks": {}},
            "servers.json": {"mcpServers": {"fs": _PINNED}},
        },
    )

    result = _validate(root)
    style = _findings(result, "plugin_component_path_style")
    assert len(style) == 5
    assert {finding.severity for finding in style} == {Severity.HIGH}
    assert all("rejects the whole manifest" in finding.message for finding in style)
    assert not result.passed


def test_codex_path_without_dot_slash_stays_medium(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".codex-plugin/plugin.json": {**_CODEX, "skills": "my-skills/"},
            "my-skills/beta/SKILL.md": _SKILL.format(name="beta"),
            "skills/alpha/SKILL.md": _SKILL.format(name="alpha"),
        },
    )

    [style] = _findings(_validate(root), "plugin_component_path_style")
    assert style.severity == Severity.MEDIUM
    assert "loads the default location instead" in style.message


@pytest.mark.parametrize(
    "agents",
    ["./agents/", "./agents/reviewer.txt", ["./agents/reviewer.md", "./agents/notes.txt"]],
    ids=["folder", "not-md", "list-with-not-md"],
)
def test_claude_agents_must_be_markdown_files(tmp_path: Path, agents: object) -> None:
    """Proof check-02 E07 and skeptic X04: Claude Code's schema takes only '.md' agent file paths."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "agents-kind", "agents": agents},
            "agents/reviewer.md": _AGENT,
            "agents/reviewer.txt": _AGENT,
            "agents/notes.txt": _AGENT,
        },
    )

    result = _validate(root)
    [invalid] = _findings(result, "plugin_component_path_invalid")
    assert invalid.severity == Severity.HIGH
    assert "Markdown (.md)" in invalid.message
    assert any(row.get("problem") == "invalid" for row in _rows(result, "agent"))
    assert not result.passed


def test_claude_hooks_path_must_be_json(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "hooks-kind", "hooks": "./hooks/hooks.yaml"},
            "hooks/hooks.yaml": "hooks: {}\n",
        },
    )

    [invalid] = _findings(_validate(root), "plugin_component_path_invalid")
    assert "JSON (.json)" in invalid.message


def test_claude_markdown_agent_list_passes(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "agents-ok", "agents": ["./agents/reviewer.md"]},
            "agents/reviewer.md": _AGENT,
        },
    )

    result = _validate(root)
    assert not _findings(result, "plugin_component_path_invalid")
    assert [row["name"] for row in _rows(result, "agent")] == ["reviewer"]


# --------------------------------------------------------------------------- #
# M20: a commands-map folder source is valid (Claude Code loads it)            #
# --------------------------------------------------------------------------- #
def test_commands_map_folder_source_is_loaded_not_invalid(tmp_path: Path) -> None:
    """Proof check-02 E10: '{"hi": {"source": "./cmds/"}}' was a false HIGH; Claude Code loads 'hi'."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {
                "name": "commands-map-folder",
                "commands": {"hi": {"source": "./cmds/", "description": "Say hi."}},
            },
            "cmds/hi.md": "---\ndescription: Say hi.\nallowed-tools: Bash\n---\nSay hi.\n",
        },
    )

    result = _validate(root)
    assert not _findings(result, "plugin_component_path_invalid")
    [row] = _rows(result, "command")
    # The row names the command file, so Tier 3 stages the file, not the folder.
    assert (row["name"], row["path"]) == ("hi", "cmds/hi.md")
    assert "problem" not in row
    # The folder's command file is read, so its grants are checked like any other command.
    assert _findings(result, "plugin_command_unrestricted_bash")


_E10 = {
    ".claude-plugin/plugin.json": {
        "name": "commands-map-folder",
        "version": "1.0.0",
        "commands": {"hi": {"source": "./cmds/", "description": "Say hi."}},
    },
    "cmds/hi.md": "---\ndescription: Say hi.\n---\n\nSay hi to the user.\n",
    "skills/greet/SKILL.md": _SKILL.format(name="greet"),
    "evals/evals.json": {
        "skill_name": "greet",
        "evals": [{"id": "c1", "prompt": "Say hello.", "expected_output": "Hello."}],
    },
}


@pytest.mark.parametrize("agent", ["claude-code", "opencode", "codex"])
def test_commands_map_folder_source_stages_natively(tmp_path: Path, agent: str) -> None:
    """Proof check-02 E10 at Tier 3: native staging read the folder 'cmds' as a file and refused the whole run."""
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    root = _write(tmp_path / "p", _E10)

    package = prepare_plugin_eval_package(
        root,
        stage_root=tmp_path / "stage",
        evals_source=root / "evals",
        plugin_load="native",
        agents=agent,
        env_mode="docker",
    )
    rows = package.provenance()["component_coverage"]["components"]
    [command] = [row for row in rows if row["type"] == "command"]
    assert (command["name"], command["path"]) == ("hi", "cmds/hi.md")
    assert command["state"] != "invalid"


def test_commands_map_folder_source_loads_each_markdown_file(tmp_path: Path) -> None:
    """Claude Code 2.1.284 loads every .md file directly in a folder source, named after the file."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "cmds", "commands": {"hi": {"source": "./cmds/"}}},
            "cmds/a.md": "---\ndescription: A.\n---\nA.\n",
            "cmds/b.md": "---\ndescription: B.\nallowed-tools: Bash\n---\nB.\n",
            "cmds/notes.txt": "not a command\n",
            "cmds/sub/c.md": "---\ndescription: C.\n---\nC.\n",
        },
    )

    result = _validate(root)
    assert [(row["name"], row["path"]) for row in _rows(result, "command")] == [
        ("a", "cmds/a.md"),
        ("b", "cmds/b.md"),
    ]
    # b.md is not named after the map key, and its Bash grant is still checked.
    assert _findings(result, "plugin_command_unrestricted_bash")


def test_commands_map_folder_without_markdown_files_is_an_invalid_row(tmp_path: Path) -> None:
    """Claude Code loads no command from a folder source without .md files; the row must not look evaluated."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "cmds", "commands": {"hi": {"source": "./cmds/"}}},
            "cmds/notes.txt": "not a command\n",
        },
    )

    result = _validate(root)
    [finding] = _findings(result, "plugin_command_folder_empty")
    assert finding.severity == Severity.MEDIUM
    [row] = _rows(result, "command")
    assert (row["name"], row["path"], row["problem"], row["findings"]) == ("hi", "cmds", "invalid", 1)


# --------------------------------------------------------------------------- #
# M19: Codex drops a path it does not accept; it is neither listed nor staged  #
# --------------------------------------------------------------------------- #
def test_codex_dropped_paths_are_not_inventoried(tmp_path: Path) -> None:
    """Proof check-02 P06: Codex 0.142.5 loads only 'alpha'; 'beta' and the other dropped values were listed."""
    root = _write(
        tmp_path / "p",
        {
            ".codex-plugin/plugin.json": {
                **_CODEX,
                "skills": "my-skills/",
                "mcpServers": "servers.json",
                "hooks": "hooks/custom.json",
                "commands": "custom-commands/",
                "apps": "apps.json",
            },
            "my-skills/beta/SKILL.md": _SKILL.format(name="beta"),
            "skills/alpha/SKILL.md": _SKILL.format(name="alpha"),
            ".mcp.json": {"mcpServers": {"default-server": _PINNED}},
            "servers.json": {"mcpServers": {"declared-server": _PINNED}},
            "hooks/hooks.json": {"hooks": {}},
            "hooks/custom.json": {"hooks": {}},
            "commands/default-cmd.md": "---\ndescription: d\n---\nbody\n",
            "custom-commands/custom-cmd.md": "---\ndescription: d\n---\nbody\n",
            ".app.json": {"apps": {"default-app": {"id": "a"}}},
            "apps.json": {"apps": {"declared-app": {"id": "b"}}},
        },
    )

    result = _validate(root)
    names = {(row["type"], row["name"]) for row in _rows(result)}
    assert {("skill", "alpha"), ("mcp", "default-server"), ("command", "default-cmd"), ("app", "default-app")} <= names
    assert (
        not {("skill", "beta"), ("mcp", "declared-server"), ("command", "custom-cmd"), ("app", "declared-app")} & names
    )
    assert not [row for row in _rows(result, "hook") if row["path"] == "hooks/custom.json"]
    assert len(_findings(result, "plugin_component_path_style")) == 5


def test_codex_list_keeps_accepted_entries_and_drops_the_rest(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".codex-plugin/plugin.json": {**_CODEX, "skills": ["./kept/", "dropped/"]},
            "kept/one/SKILL.md": _SKILL.format(name="one"),
            "dropped/two/SKILL.md": _SKILL.format(name="two"),
        },
    )

    paths = {row["path"] for row in _rows(_validate(root), "skill")}
    assert "kept/one" in paths
    assert "dropped/two" not in paths


def test_codex_dropped_skill_path_is_not_staged_at_tier3(tmp_path: Path) -> None:
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    root = _write(
        tmp_path / "p",
        {
            ".codex-plugin/plugin.json": {**_CODEX, "skills": "my-skills/"},
            "my-skills/beta/SKILL.md": _SKILL.format(name="beta"),
            "skills/alpha/SKILL.md": _SKILL.format(name="alpha"),
        },
    )
    evals = _write(
        tmp_path / "evals",
        {"evals.json": {"skill_name": "probe", "evals": [{"id": "c1", "prompt": "Hi.", "expected_output": "Hi."}]}},
    )

    package = prepare_plugin_eval_package(root, stage_root=tmp_path / "stage", evals_source=evals)
    rows = package.provenance()["component_coverage"]["components"]
    staged = {row["name"] for row in rows if row["type"] == "skill" and row["state"] == "staged"}
    assert staged == {"alpha"}


# --------------------------------------------------------------------------- #
# H12: Cursor's root mcp.json stays checked next to a declared mcpServers      #
# --------------------------------------------------------------------------- #
def _cursor_mcp(tmp_path: Path, *, declared: bool) -> Path:
    manifest: dict = {"name": "cursor-mcp", "version": "1.0.0", "description": "Cursor MCP probe"}
    if declared:
        manifest["mcpServers"] = "./servers.json"
    return _write(
        tmp_path / ("declared" if declared else "root-only"),
        {
            ".cursor-plugin/plugin.json": manifest,
            "servers.json": {"mcpServers": {"declared-server": {"command": "node", "args": ["./server.js"]}}},
            "mcp.json": {"mcpServers": {"root-mcp-json": {"command": "bash", "args": ["-c", "echo hi"]}}},
            "skills/greet/SKILL.md": _SKILL.format(name="greet"),
        },
    )


@pytest.mark.parametrize("declared", [True, False], ids=["X02-declared", "X03-root-only"])
def test_cursor_root_mcp_json_is_checked_even_when_mcp_servers_is_declared(tmp_path: Path, declared: bool) -> None:
    """Skeptic X02 gave exit 0 and no finding for the root 'bash -c' server; X03 (no declaration) gave CRITICAL."""
    result = _validate(_cursor_mcp(tmp_path, declared=declared))

    dangerous = _findings(result, "mcp_command_dangerous_form")
    assert [finding.metadata.get("mcp_server") for finding in dangerous] == ["root-mcp-json"]
    assert dangerous[0].severity == Severity.CRITICAL
    rows = {row["name"]: row for row in _rows(result, "mcp")}
    assert rows["root-mcp-json"]["path"] == "mcp.json"
    assert rows["root-mcp-json"]["findings"] >= 1
    # The merge is a reading of the Cursor loader, not a Cursor run: say so only when both files are present.
    notes = _findings(result, "mcp_root_config_also_checked")
    assert [finding.severity for finding in notes] == ([Severity.LOW] if declared else [])


# --------------------------------------------------------------------------- #
# L7: Cursor root variables in component paths are flagged, still resolved     #
# --------------------------------------------------------------------------- #
def test_cursor_root_variable_in_component_path_is_flagged(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".cursor-plugin/plugin.json": {"name": "demo", "skills": "${CURSOR_PLUGIN_ROOT}/my-skills"},
            "my-skills/good/SKILL.md": _SKILL.format(name="good"),
        },
    )

    result = _validate(root)
    [flag] = _findings(result, "plugin_component_path_root_variable")
    assert flag.severity == Severity.MEDIUM
    assert "${CURSOR_PLUGIN_ROOT}" in flag.message
    # The files it names are still checked, in case the client does expand it.
    assert any(row["path"] == "my-skills/good" for row in _rows(result, "skill"))
    assert not _findings(result, "plugin_component_path_invalid")


def test_cursor_relative_component_path_is_not_flagged(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".cursor-plugin/plugin.json": {"name": "demo", "skills": "my-skills/"},
            "my-skills/good/SKILL.md": _SKILL.format(name="good"),
        },
    )

    result = _validate(root)
    assert not _findings(result, "plugin_component_path_root_variable")
    assert not _findings(result, "plugin_component_path_style")
