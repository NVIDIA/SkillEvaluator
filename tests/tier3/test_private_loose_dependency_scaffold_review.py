# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource parity normalizes generated scaffolding without hiding authored content."""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.tier3.loose_parts import LOOSE_BUNDLE_DIR, LooseParts, build_loose_parts
from skillevaluator.tier3.plugin_native import NativePluginSource


def _stage_relocated_member(tmp_path: Path, *, authored_skills_directory: bool = False):
    plugin = tmp_path / "plugin"
    metadata = plugin / ".claude-plugin"
    metadata.mkdir(parents=True)
    manifest = {"name": "relocated", "description": "A relocated member fixture"}
    (metadata / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    member = plugin / "other" / "a"
    member.mkdir(parents=True)
    (member / "SKILL.md").write_text("---\nname: a\ndescription: A relocated skill\n---\nUse a.\n", encoding="utf-8")
    (plugin / "run.js").write_text("console.log('ready');\n", encoding="utf-8")
    if authored_skills_directory:
        (plugin / "skills").mkdir(mode=0o755)
        (plugin / "skills").chmod(0o755)
    source = NativePluginSource(
        plugin_name="relocated",
        description="A relocated member fixture",
        contained=True,
        plugin_root=plugin,
        manifest=manifest,
        manifest_rel=".claude-plugin/plugin.json",
        member_skills=(member,),
        plugin_file_mcp_servers=({"name": "tool", "command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/run.js"]},),
    )
    parts = build_loose_parts(source)
    assert parts.error is None, parts.error
    tasks = {}
    for arm in ("with_skill", "loose_parts", "sum_of_parts"):
        task = tmp_path / "tasks" / arm / "case-1"
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    return parts, tasks


def _assert_equal(parts: LooseParts, tasks: dict[str, list[Path]]) -> None:
    report = parts.content_report_for_tasks(tasks)
    assert report["equal_across_arms"] is True, report["mismatches"]


def test_relocated_member_generated_parent_does_not_create_a_resource_mismatch(tmp_path: Path) -> None:
    parts, tasks = _stage_relocated_member(tmp_path)
    full = tasks["with_skill"][0] / "environment/skilleval/native/claude-code/plugin"
    loose = tasks["loose_parts"][0] / "environment/skilleval" / LOOSE_BUNDLE_DIR / "root"
    assert (full / "skills/a/SKILL.md").is_file()
    assert not (loose / "skills").exists()
    _assert_equal(parts, tasks)


def test_unexpected_file_in_generated_parent_still_breaks_resource_parity(tmp_path: Path) -> None:
    parts, tasks = _stage_relocated_member(tmp_path)
    _assert_equal(parts, tasks)
    full = tasks["with_skill"][0] / "environment/skilleval/native/claude-code/plugin"
    (full / "skills/unexpected.js").write_text("console.log('unexpected');\n", encoding="utf-8")

    report = parts.content_report_for_tasks(tasks)

    assert report["equal_across_arms"] is False, report
    assert "script:." in report["mismatches"]


def test_authored_empty_parent_directory_mode_still_breaks_resource_parity(tmp_path: Path) -> None:
    parts, tasks = _stage_relocated_member(tmp_path, authored_skills_directory=True)
    _assert_equal(parts, tasks)
    loose = tasks["loose_parts"][0] / "environment/skilleval" / LOOSE_BUNDLE_DIR / "root"
    assert list((loose / "skills").iterdir()) == []
    (loose / "skills").chmod(0o700)

    report = parts.content_report_for_tasks(tasks)

    assert report["equal_across_arms"] is False, report
    assert "script:." in report["mismatches"]
