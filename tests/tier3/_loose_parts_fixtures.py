# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake Claude Code plugin with every component type, for the loose-parts arm tests.

``shipit`` has two member skills, an always-on rule, a glob-scoped rule, an
agent-requested rule, an MCP server that launches a plugin script with env, a
subagent, a command, two hooks (one through ``${CLAUDE_PLUGIN_ROOT}``), an
output style, a ``.lsp.json``, a plugin ``settings.json`` and a ``bin/``. Nothing
here starts Harbor, Docker, or an agent.
"""

from __future__ import annotations

import json
from pathlib import Path

PLUGIN = "shipit"
GUARD_SCRIPT = (
    "#!/bin/sh\n# Refuse git push during release prep.\ncase \"$(cat)\" in *'git push'*) exit 2;; esac\nexit 0\n"
)
SERVER_SCRIPT = "#!/usr/bin/env node\n// reltools stdio server (fake)\nconsole.log('reltools');\n"
AGENT_TEXT = (
    "---\nname: release-reviewer\ndescription: Reviews release notes against the policy. Read-only.\n"
    "tools: Read, Grep\n---\n\nYou review release notes. Do not edit files.\n"
)
COMMAND_TEXT = (
    '---\ndescription: Run the pre-release checklist\nargument-hint: "[path]"\n---\n\n'
    "Check the release notes at $ARGUMENTS.\n"
)
ALWAYS_RULE = "# Release policy\n\n- Never call `publish_release`.\n- Notes end with `Signed-off-by: shipit`.\n"
GLOB_RULE = '---\nglobs: "RELEASE*.md, CHANGELOG.md"\n---\n\nRelease files use ISO dates.\n'
REQUESTED_RULE = (
    "---\ndescription: Use when asked about semver edge cases\nalwaysApply: false\n---\n\nPrefer minor bumps.\n"
)
HOOKS = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": 'sh "${CLAUDE_PLUGIN_ROOT}/hooks/guard-push.sh"'}],
            }
        ],
        "PostToolUse": [
            {"matcher": "Write|Edit", "hooks": [{"type": "command", "command": "echo wrote >> /tmp/shipit.log"}]}
        ],
    }
}
MCP = {
    "mcpServers": {
        "reltools": {
            "command": "node",
            "args": ["${CLAUDE_PLUGIN_ROOT}/servers/reltools.js", "--strict"],
            "env": {"RELTOOLS_MODE": "strict", "RELTOOLS_TOKEN": "${RELTOOLS_TOKEN}"},
        }
    }
}


def write(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def make_plugin(root: Path, *, name: str = PLUGIN) -> Path:
    """Write the full ``shipit`` plugin under ``root/<name>`` and return its directory."""
    plugin = root / name
    write(
        plugin / ".claude-plugin" / "plugin.json",
        json.dumps({"name": name, "version": "1.0.0", "description": "Prepare a release: changelog and version."}),
    )
    write(
        plugin / "skills" / "changelog-collect" / "SKILL.md",
        "---\nname: changelog-collect\ndescription: Collect the changes since the last tag.\n---\n\nList changes.\n",
    )
    write(
        plugin / "skills" / "version-bump" / "SKILL.md",
        "---\nname: version-bump\ndescription: Compute the next semver version.\n---\n\nRun scripts/bump.sh.\n",
    )
    write(plugin / "skills" / "version-bump" / "scripts" / "bump.sh", "#!/bin/sh\necho 1.2.4\n", mode=0o755)
    write(plugin / "skills" / "version-bump" / "evals" / "evals.json", '{"secret": "expected output"}')
    write(plugin / "rules" / "release-policy.md", ALWAYS_RULE)
    write(plugin / "rules" / "release-files.md", GLOB_RULE)
    write(plugin / "rules" / "semver-help.mdc", REQUESTED_RULE)
    write(plugin / ".mcp.json", json.dumps(MCP))
    write(plugin / "servers" / "reltools.js", SERVER_SCRIPT, mode=0o755)
    write(plugin / "servers" / "README.md", "Not referenced by any config.\n")
    write(plugin / "agents" / "release-reviewer.md", AGENT_TEXT)
    write(plugin / "commands" / "release-check.md", COMMAND_TEXT)
    write(plugin / "hooks" / "hooks.json", json.dumps(HOOKS))
    write(plugin / "hooks" / "guard-push.sh", GUARD_SCRIPT, mode=0o755)
    write(plugin / "output-styles" / "terse.md", "---\nname: terse\ndescription: Terse answers\n---\nBe terse.\n")
    write(plugin / ".lsp.json", json.dumps({"go": {"command": "gopls", "extensionToLanguage": {".go": "go"}}}))
    write(
        plugin / "settings.json",
        json.dumps({"agent": "release-reviewer", "subagentStatusLine": {"type": "command", "command": "echo x"}}),
    )
    write(plugin / "bin" / "shipit-notify", "#!/bin/sh\necho notified\n", mode=0o755)
    write(
        plugin / "evals" / "evals.json",
        json.dumps(
            {
                "skill_name": name,
                "evals": [
                    {
                        "id": "case-1",
                        "prompt": "Prepare the release notes.",
                        "expected_output": "Release notes.",
                        "assertions": ["Notes written"],
                        "expected_skill": "changelog-collect",
                    }
                ],
            }
        ),
    )
    return plugin


def make_skills_only_plugin(root: Path, *, name: str = "bare-kit") -> Path:
    """A plugin whose only loadable parts are skills (plus glue): loose parts equal the skills alone."""
    plugin = root / name
    write(plugin / ".claude-plugin" / "plugin.json", json.dumps({"name": name, "description": "Skills only."}))
    write(
        plugin / "skills" / "alpha" / "SKILL.md",
        "---\nname: alpha\ndescription: Alpha skill.\n---\n\nAlpha.\n",
    )
    write(plugin / "output-styles" / "terse.md", "---\nname: terse\ndescription: Terse\n---\nBe terse.\n")
    write(
        plugin / "evals" / "evals.json",
        json.dumps(
            {
                "skill_name": name,
                "evals": [{"id": "c1", "prompt": "Do alpha.", "expected_output": "x", "assertions": ["x"]}],
            }
        ),
    )
    return plugin


def native_source(plugin: Path, stage_root: Path):
    """The native plugin snapshot, built by the real Tier 3 package preparation."""
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    package = prepare_plugin_eval_package(plugin, stage_root=stage_root, plugin_load="native")
    assert package.native_source is not None
    return package
