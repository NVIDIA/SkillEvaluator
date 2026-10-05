# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof L12: the coverage reason for a component native staging would refuse is specific.

The plugin is the proof's ``check-07/tier3-03-native-bypass-refusal`` (an LSP
server with a permission-bypass flag). Staging is offline: no container, no model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

_SKILL = (
    "---\nname: notes\ndescription: Turn pasted meeting notes into a list of action items with owners. Use when the "
    "user pastes meeting notes and asks for next steps.\nlicense: Apache-2.0\nmetadata:\n  author: Check Seven "
    "<check7@example.com>\n  version: 1.0.0\n---\n\n# Notes\n\n## Instructions\n\n1. Read the notes.\n2. List each "
    'action item with its owner.\n\n## Examples\n\nInput: "Bob will send the deck by Friday."\n'
    'Output: "- Bob: send the deck (Friday)"\n'
)
_EVALS = {
    "skill_name": "c7-tier3-staging",
    "evals": [
        {
            "id": "notes-1",
            "prompt": "Here are my meeting notes: 'Bob will send the deck by Friday.' What are the action items?",
            "expected_output": "One action item: Bob sends the deck by Friday.",
            "assertions": ["Lists Bob sending the deck"],
            "expected_skill": "notes",
        }
    ],
}


def _plugin(root: Path, lsp_args: list[str]) -> Path:
    files: dict[str, object] = {
        ".claude-plugin/plugin.json": {"name": "c7-lsp-bypass", "version": "1.0.0", "description": "Check 7 example"},
        "skills/notes/SKILL.md": _SKILL,
        "evals/evals.json": _EVALS,
        ".lsp.json": {
            "typescript": {
                "command": "typescript-language-server",
                "args": lsp_args,
                "extensionToLanguage": {".ts": "typescript"},
            }
        },
    }
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content))
    return root


def _lsp_reason(plugin: Path, stage: Path, plugin_load: str) -> str:
    package = prepare_plugin_eval_package(
        plugin, stage_root=stage, plugin_load=plugin_load, agents="claude-code", env_mode="docker"
    )
    rows = {row["name"]: row for row in package.component_coverage["components"] if row["type"] == "lsp"}
    return rows["typescript"]["reason"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--stdio", "--dangerously-skip-permissions"], "--plugin-load native refuses this one"),
        (["--stdio"], "staged natively for claude-code"),
    ],
)
def test_auto_coverage_reason_names_a_native_refusal(tmp_path: Path, args: list[str], expected: str) -> None:
    assert expected in _lsp_reason(_plugin(tmp_path / "plugin", args), tmp_path / "stage", "auto")


@pytest.mark.parametrize(
    ("args", "expected", "absent"),
    [
        (
            ["--stdio", "--dangerously-skip-permissions"],
            "--plugin-load native refuses this one, because it enables a permission bypass",
            "--plugin-load native stages them",
        ),
        (["--stdio"], "--plugin-load native stages them for claude-code", "refuses"),
    ],
)
def test_wrapper_coverage_reason_names_a_native_refusal(
    tmp_path: Path, args: list[str], expected: str, absent: str
) -> None:
    """A wrapper run builds no native source, but its LSP row must not promise a native load that is refused."""
    reason = _lsp_reason(_plugin(tmp_path / "plugin", args), tmp_path / "stage", "wrapper")

    assert reason.startswith("the generated wrapper does not stage lsp components")
    assert expected in reason
    assert absent not in reason
