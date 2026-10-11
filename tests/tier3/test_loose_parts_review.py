# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent review of the loose-parts arm: the gaps between "same files" and "same capability".

Each test pins one way the full-plugin arm and the loose-parts arm could carry
the same bytes and still not be compared fairly:

* the verifier must treat a staged plugin command the same in both arms
  (Claude Code runs it through its ``Skill`` tool as ``<plugin>:<command>`` in
  the full arm and as ``<command>`` in the loose arm);
* Claude Code ignores ``permissionMode``, ``hooks`` and ``mcpServers`` in a
  plugin subagent but honours them in a user subagent, so a byte-identical copy
  of such a subagent is not the same capability;
* Claude Code writes the plugin root into a plugin skill, subagent or command
  text for ``${CLAUDE_PLUGIN_ROOT}``; the loose copy keeps the variable, so the
  run records which pieces carry one.

Nothing here starts Harbor, Docker, or an agent, and no model is called.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _loose_parts_fixtures import AGENT_TEXT, COMMAND_TEXT, make_plugin, native_source, write
from test_native_name_matching import _run_verifier, verifier

from skillevaluator.tier3.eval_core import checks as host_checks
from skillevaluator.tier3.loose_parts import build_loose_parts, loose_parts_task_staging


def _task_entry(tmp_path: Path, plugin: Path, *, loose: bool) -> dict[str, Any]:
    """The entry.json the real emitter writes for the full-plugin or the loose-parts arm."""
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
    from skillevaluator.tier3.harbor.native_staging import build_native_task_staging
    from skillevaluator.tier3.plugin_native import ClaudeCodeAdapter

    package = native_source(plugin, tmp_path / "stage")
    if loose:
        staging = loose_parts_task_staging("claude-code", package.native_source)
    else:
        staging = build_native_task_staging("claude-code", ClaudeCodeAdapter(), package.native_source)
    tasks = generate_harbor_tasks(
        package.package_path,
        tmp_path / ("loose" if loose else "full"),
        with_skill=not loose,
        workspace_skill_paths=list(package.include_skills),
        workspace_mode="group",
        native_plugin=staging,
    )
    return json.loads((Path(tasks[0]) / "tests" / "entry.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Plugin commands are command activations in both arms                        #
# --------------------------------------------------------------------------- #
def test_the_loose_arm_tells_the_verifier_its_staged_commands(tmp_path: Path) -> None:
    entry = _task_entry(tmp_path, make_plugin(tmp_path / "plugins"), loose=True)

    # Bare names, as Claude Code reports a user command; no plugin prefix to strip.
    assert entry["native_plugin_commands"] == ["release-check"]
    assert "native_plugin_prefix" not in entry


@pytest.mark.parametrize("module", [verifier, host_checks], ids=["harbor-template", "host-checks"])
def test_both_verifier_copies_split_a_bare_staged_command(module: Any) -> None:
    kept, commands = module.split_native_command_calls(
        ["changelog-collect", "release-check"], "", ["release-check"], ["changelog-collect", "version-bump"]
    )
    assert (kept, commands) == (["changelog-collect"], ["release-check"])
    # A staged skill that shares a command's name stays a skill; no listed command changes nothing.
    assert module.split_native_command_calls(["alpha"], "", ["alpha"], ["alpha"]) == (["alpha"], [])
    assert module.split_native_command_calls(["release-check"], "", [], ["alpha"]) == (["release-check"], [])


def test_the_verifier_scores_a_plugin_command_the_same_in_the_full_and_loose_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    case = {"expected_skill": "changelog-collect", "should_trigger": True, "skill_workspace_mode": "isolated"}
    full_entry = {**_task_entry(tmp_path / "f", plugin, loose=False), **case}
    loose_entry = {**_task_entry(tmp_path / "l", plugin, loose=True), **case}

    # The same behaviour in each arm's own spelling: the expected skill, then the plugin's command.
    full = _run_verifier(
        tmp_path / "full-run", monkeypatch, full_entry, ["shipit:changelog-collect", "shipit:release-check"]
    )
    loose = _run_verifier(tmp_path / "loose-run", monkeypatch, loose_entry, ["changelog-collect", "release-check"])

    assert full["skill_efficiency"]["routing"]["score"] == 1.0, full["skill_efficiency"]["routing"]
    assert loose["skill_efficiency"]["routing"]["score"] == 1.0, loose["skill_efficiency"]["routing"]
    assert loose["skill_execution"]["activation"]["score"] == full["skill_execution"]["activation"]["score"]
    assert loose["_native_plugin_commands"] == ["release-check"]


# --------------------------------------------------------------------------- #
# Subagent fields Claude Code ignores for a plugin but honours for a user      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field",
    [
        "permissionMode: acceptEdits",
        "mcpServers:\n  - reltools",
        "hooks:\n  Stop:\n    - hooks:\n        - type: command\n          command: echo stop",
    ],
    ids=["permissionMode", "mcpServers", "hooks"],
)
def test_a_subagent_field_only_a_user_agent_honours_is_not_the_same_capability(tmp_path: Path, field: str) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    write(
        plugin / "agents" / "release-reviewer.md", AGENT_TEXT.replace("tools: Read, Grep\n", f"tools: Read\n{field}\n")
    )
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)

    report = parts.content_report()
    info = parts.info(content=report)

    assert report["equal_across_arms"] is False
    assert "agent:release-reviewer" in report["mismatches"]
    reason = report["capability_differences"]["agent:release-reviewer"]
    assert "ignored for plugin agents" in reason
    assert field.split(":", 1)[0] in reason
    assert info["equal_across_arms"] is False
    assert info["capability_differences"] == report["capability_differences"]


def test_the_emitted_tasks_hashes_keep_the_subagent_difference(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
    from skillevaluator.tier3.harbor.native_staging import build_native_task_staging
    from skillevaluator.tier3.plugin_native import ClaudeCodeAdapter

    plugin = make_plugin(tmp_path / "plugins")
    write(
        plugin / "agents" / "release-reviewer.md", AGENT_TEXT.replace("tools: Read, Grep\n", "permissionMode: plan\n")
    )
    package = native_source(plugin, tmp_path / "stage")
    parts = build_loose_parts(package.native_source)
    common = {"workspace_skill_paths": list(package.include_skills), "workspace_mode": "group"}
    full = generate_harbor_tasks(
        package.package_path,
        tmp_path / "full",
        with_skill=True,
        native_plugin=build_native_task_staging("claude-code", ClaudeCodeAdapter(), package.native_source),
        **common,
    )
    loose = generate_harbor_tasks(
        package.package_path, tmp_path / "loose", with_skill=False, native_plugin=parts.task_staging(), **common
    )

    report = parts.content_report_for_tasks({"with_skill": full, "loose_parts": loose})

    # The bytes match, but the capability does not: the run must not call it equal.
    assert (
        report["content_hashes"]["with_skill"]["agent:release-reviewer"]
        == (report["content_hashes"]["loose_parts"]["agent:release-reviewer"])
    )
    assert report["equal_across_arms"] is False
    assert report["mismatches"] == ["agent:release-reviewer"]
    assert "permissionMode" in report["capability_differences"]["agent:release-reviewer"]


def test_a_plain_subagent_keeps_the_arms_equal(tmp_path: Path) -> None:
    parts = build_loose_parts(native_source(make_plugin(tmp_path / "plugins"), tmp_path / "stage").native_source)

    report = parts.content_report()

    assert report["equal_across_arms"] is True, report["mismatches"]
    assert "capability_differences" not in report


# --------------------------------------------------------------------------- #
# ${CLAUDE_PLUGIN_ROOT} in skill, subagent and command text                    #
# --------------------------------------------------------------------------- #
def test_plugin_root_text_in_a_skill_agent_or_command_is_recorded(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    write(
        plugin / "commands" / "release-check.md",
        COMMAND_TEXT + "\nRead ${CLAUDE_PLUGIN_ROOT}/rules/release-policy.md\n",
    )
    write(
        plugin / "skills" / "changelog-collect" / "SKILL.md",
        "---\nname: changelog-collect\ndescription: Collect changes.\n---\n\nRun ${CLAUDE_PLUGIN_ROOT}/hooks/guard-push.sh\n",
    )
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)

    info = parts.info(content=parts.content_report())

    # Recorded, not a mismatch: the bytes are the plugin's own, but the model reads a different text.
    assert sorted(info["plugin_root_in_text"]) == ["command:release-check", "skill:changelog-collect"]
    assert "CLAUDE_PLUGIN_ROOT" in info["plugin_root_in_text"]["command:release-check"]
    assert info["equal_across_arms"] is True


def test_a_plugin_without_plugin_root_text_records_none(tmp_path: Path) -> None:
    parts = build_loose_parts(native_source(make_plugin(tmp_path / "plugins"), tmp_path / "stage").native_source)

    assert parts.info(content=parts.content_report())["plugin_root_in_text"] == {}
