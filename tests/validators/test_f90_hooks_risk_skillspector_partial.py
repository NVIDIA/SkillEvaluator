# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H10: a benign hooks file or a mention of an unshipped file must not make Tier 1 INCOMPLETE.

SkillSpector 2.11.2 reports ``analysis_completeness.status: partial`` for any
plugin ``hooks/hooks.json`` with a handler (``opaque_content`` from its
``bundled_execution_surface`` analyzer) and for a skill that names a file it does
not ship (``reference_missing``). The fixtures are the real 2.11.2 JSON for the
proof's ``check-10/n03-claude-benign-hooks`` plugin and the bundled skill of
``check-10/e09-benign-path-reference-incomplete`` (local paths replaced).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.plugin_tree import plugin_tree_scope
from skillevaluator.validators.security import SecurityValidator

FIXTURES = Path(__file__).parents[1] / "fixtures"
HOOKS_REPORT = json.loads((FIXTURES / "skillspector-2.11.2-plugin-hooks-opaque-no-llm.json").read_text())
REFERENCE_REPORT = json.loads((FIXTURES / "skillspector-2.11.2-reference-missing-no-llm.json").read_text())

HOOKS_JSON = {
    "hooks": {
        "PostToolUse": [
            {
                "matcher": "Write",
                "hooks": [{"type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/hooks/scripts/fmt.sh"}],
            }
        ]
    }
}
CODEX_MANIFEST = {
    "name": "x07-benign-hooks",
    "version": "1.0.0",
    "description": "benign Codex plugin with a hook",
    "author": {"name": "Example"},
    "interface": {"displayName": "X07", "shortDescription": "fixture"},
}


def _plugin(root: Path, *, manifest: str = "claude", hooks: str | None = None) -> Path:
    """The n03 plugin (Claude) or its Codex twin (proof x07): a clean agent and one harmless hook."""
    if manifest == "claude":
        (root / ".claude-plugin").mkdir(parents=True)
        (root / ".claude-plugin" / "plugin.json").write_text(
            json.dumps({"name": "n03-benign-hooks", "version": "1.0.0", "description": "benign hooks"})
        )
    else:
        (root / ".codex-plugin").mkdir(parents=True)
        (root / ".codex-plugin" / "plugin.json").write_text(json.dumps(CODEX_MANIFEST))
    (root / "agents").mkdir()
    (root / "agents" / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews a diff for style problems.\ntools: Read, Grep\n---\n"
        "You review code and list style problems.\n"
    )
    (root / "hooks" / "scripts").mkdir(parents=True)
    (root / "hooks" / "hooks.json").write_text(hooks if hooks is not None else json.dumps(HOOKS_JSON))
    (root / "hooks" / "scripts" / "fmt.sh").write_text('#!/bin/sh\nset -eu\necho "formatted" >&2\n')
    return root


def _scan(target: Path, report: dict, *, plugin_root: Path | None = None):
    result_json = json.dumps(report)
    with (
        patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector"),
        patch.object(
            Tools.skillspector,
            "run",
            return_value=ToolResult(success=True, stdout=result_json, stderr="", exit_code=0),
        ),
    ):
        validator = SecurityValidator(use_llm=False)
        if plugin_root is None:
            return validator._run_skillspector(target)
        with plugin_tree_scope(plugin_root, []):
            return validator._run_skillspector(target)


@pytest.mark.parametrize("manifest", ["claude", "codex"])
def test_benign_plugin_hooks_file_is_not_incomplete(tmp_path: Path, manifest: str) -> None:
    plugin = _plugin(tmp_path / "plugin", manifest=manifest)

    result = _scan(plugin, HOOKS_REPORT, plugin_root=plugin)

    assert result.status == "passed", result.errors
    assert not result.is_incomplete
    assert any("hooks/hooks.json" in warning and "hook risk check" in warning for warning in result.warnings)
    # SkillSpector's own LOW note about the hook surface is still reported.
    assert [finding.severity.value for finding in result.findings] == ["low"]


def test_opaque_file_the_hook_model_did_not_read_stays_incomplete(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin")
    report = copy.deepcopy(HOOKS_REPORT)
    report["analysis_completeness"]["ledger_exceptions"][0]["path"] = "config/hooks.json"

    result = _scan(plugin, report, plugin_root=plugin)

    assert result.is_incomplete
    assert any("incomplete analysis" in error for error in result.errors)


def test_unparsable_hooks_file_stays_incomplete(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", hooks="{not json")

    result = _scan(plugin, HOOKS_REPORT, plugin_root=plugin)

    assert result.is_incomplete


def test_opaque_hooks_outside_a_plugin_scan_stays_incomplete(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin")

    result = _scan(plugin, HOOKS_REPORT)

    assert result.is_incomplete


def test_other_partial_reason_next_to_a_covered_one_stays_incomplete(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin")
    report = copy.deepcopy(HOOKS_REPORT)
    report["analysis_completeness"]["ledger_exceptions"].append(
        {
            "outcome": "partial",
            "phase": "reference_resolution",
            "reason_code": "reference_unresolved",
            "message": "A reference could not be resolved.",
            "path": "agents/reviewer.md",
            "start_line": 3,
            "end_line": 3,
            "fatal": False,
        }
    )

    result = _scan(plugin, report, plugin_root=plugin)

    assert result.is_incomplete


def test_reference_to_an_unshipped_file_is_a_note_not_incomplete(tmp_path: Path) -> None:
    skill = tmp_path / "helper"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: helper\ndescription: Formats changelog entries.\n---\n# Helper\n\n1. Read `CHANGELOG.md`.\n"
    )

    result = _scan(skill, REFERENCE_REPORT)

    assert result.status == "passed", result.errors
    assert any("reference_missing" in warning and "SKILL.md:14" in warning for warning in result.warnings)
