# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package


def _skill(root: Path, name: str = "demo") -> Path:
    skill = root / "skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Public test skill\n---\n# {name}\n\nUse this skill.\n",
        encoding="utf-8",
    )
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps({"skill_name": name, "evals": [{"id": "case-1", "prompt": "Do it", "expected_output": "Done"}]}),
        encoding="utf-8",
    )
    return skill


def test_contained_plugin_stages_member_skill_and_combined_dataset(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "public-plugin", "skills": "./skills"}), encoding="utf-8")
    member = _skill(plugin)

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert not package.skipped
    assert package.include_skills == (member.resolve(),)
    assert package.package_path is not None
    assert (package.package_path / "SKILL.md").is_file()
    entries = json.loads((package.package_path / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert entries[0]["id"] == "demo-case-1"
    assert entries[0]["plugin_eval_source_skill"] == "demo"


def test_remote_only_public_bundle_is_honestly_skipped(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "agent_plugin.yaml").write_text(
        """
name: remote-only
author: {email: dev@example.com}
skills:
  refs: [github::other/repository::skills::remote]
""",
        encoding="utf-8",
    )
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
    assert package.skipped
    assert package.package_path is None
    assert package.unresolved_skill_refs == ("github::other/repository::skills::remote",)


def test_same_repo_public_ref_resolves_without_remote_fetch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugins" / "bundle"
    plugin.mkdir(parents=True)
    member = _skill(repo, "local")
    (plugin / "agent_plugin.yaml").write_text(
        """
name: bundle
author: {email: dev@example.com}
skills:
  refs: [github::example/repo::skills::local]
""",
        encoding="utf-8",
    )
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", repo_root=repo)
    assert package.include_skills == (member.resolve(),)
    assert package.unresolved_skill_refs == ()


def test_contained_mcp_secret_is_rejected_before_toml_write(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "name": "unsafe",
                "mcpServers": {"server": {"command": "server", "env": {"API_KEY": "literal-secret"}}},
            }
        ),
        encoding="utf-8",
    )
    evals = plugin / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps({"evals": [{"id": "case", "prompt": "Use server", "expected_output": "done"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="static safety validation"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_contained_mcp_shell_command_is_rejected_before_execution(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"name": "unsafe", "mcpServers": {"server": {"command": "sh", "args": ["-c", "run"]}}}),
        encoding="utf-8",
    )
    evals = plugin / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="mcp_command_dangerous_form"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_plugin_evals_symlink_escape_is_rejected(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "p", "mcpServers": {"x": {"command": "server"}}}), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evals.json").write_text("[]", encoding="utf-8")
    try:
        (plugin / "evals").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="outside"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_symlinked_member_skill_outside_plugin_is_not_staged(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "p", "skills": "./skills"}), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: outside\ndescription: outside\n---\n", encoding="utf-8")
    skills = plugin / "skills"
    skills.mkdir()
    try:
        (skills / "outside").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
    assert package.skipped
    assert package.include_skills == ()
