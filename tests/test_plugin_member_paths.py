# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded searches for plugin skills that bundled-skill discovery prunes."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from skillevaluator import plugin_member_paths
from skillevaluator.models.result import Severity
from skillevaluator.plugin_discovery import discover_plugin_members
from skillevaluator.plugin_formats import CLAUDE_PROFILE, CODEX_PROFILE
from skillevaluator.plugin_member_paths import find_dependency_folder_skills, find_unscanned_plugin_skill_manifests
from skillevaluator.plugin_paths import scan_excluded_name
from skillevaluator.utils.secure_fs import MAX_SECURE_DIRECTORY_DEPTH

SKILL = "---\nname: demo\ndescription: A real fixture.\n---\nDo the task.\n"


def write_tree(root: Path, files: dict) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def checks(result) -> dict[str, Severity]:
    return {finding.check_name: finding.severity for finding in result.findings}


def case_insensitive(root: Path, alias: str, canonical: str) -> bool:
    """Whether ``root/alias`` (already created) is also reachable as ``root/canonical``."""
    return (root / canonical).exists()


def test_unscanned_skill_walk_reports_its_entry_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin_member_paths, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 24)
    write_tree(
        tmp_path,
        {
            CODEX_PROFILE.manifest_path: {"name": "demo", "skills": "./packs"},
            "packs/alpha/SKILL.md": SKILL,
            "packs/alpha/versions/v1/SKILL.md": SKILL,
            "packs/omega/SKILL.md": SKILL,
            **{f"packs/omega/results/run-{index:02}.txt": "x" for index in range(30)},
        },
    )
    assert find_unscanned_plugin_skill_manifests(tmp_path, "packs") == ([], False)

    result = discover_plugin_members(tmp_path)
    assert result is not None
    assert checks(result)["plugin_skill_scan_incomplete"] == Severity.LOW
    assert not result.complete


def test_unscanned_skill_walk_finds_skills_within_its_budget(tmp_path):
    write_tree(
        tmp_path,
        {
            CODEX_PROFILE.manifest_path: {"name": "demo", "skills": "./packs"},
            "packs/alpha/SKILL.md": SKILL,
            "packs/alpha/versions/v1/SKILL.md": SKILL,
        },
    )
    assert find_unscanned_plugin_skill_manifests(tmp_path, "packs") == (
        [PurePosixPath("packs/alpha/versions/v1/SKILL.md")],
        True,
    )
    result = discover_plugin_members(tmp_path)
    assert result is not None
    assert "plugin_skill_scan_incomplete" not in checks(result)


def test_unscanned_skill_walk_reports_its_depth_limit(tmp_path):
    deep = "/".join(["d"] * MAX_SECURE_DIRECTORY_DEPTH)
    (tmp_path / "packs" / "alpha" / "evals").joinpath(*deep.split("/")).mkdir(parents=True)
    found, complete = find_unscanned_plugin_skill_manifests(tmp_path, "packs")
    assert found == []
    assert not complete


def test_scan_excluded_name_matches_only_spellings_the_filesystem_aliases(tmp_path):
    (tmp_path / "EVALS").mkdir()
    (tmp_path / "plain").mkdir()
    assert scan_excluded_name("evals", None) == "evals"
    assert scan_excluded_name("EVALS", None) is None
    assert scan_excluded_name("plain", tmp_path / "plain") is None
    expected = "evals" if case_insensitive(tmp_path, "EVALS", "evals") else None
    assert scan_excluded_name("EVALS", tmp_path / "EVALS") == expected


@pytest.mark.parametrize(("folder", "canonical"), [("EVALS", "evals"), ("Results", "results")])
def test_cased_artifact_folder_in_a_declared_skills_folder(tmp_path, folder, canonical):
    write_tree(
        tmp_path,
        {
            CODEX_PROFILE.manifest_path: {"name": "demo", "skills": "./custom"},
            "custom/top/SKILL.md": SKILL,
            f"custom/top/{folder}/inner/SKILL.md": SKILL,
        },
    )
    result = discover_plugin_members(tmp_path)
    assert result is not None
    skills = {m.path for m in result.components if m.type == "skill"}
    if case_insensitive(tmp_path / "custom" / "top", folder, canonical):
        # Discovery prunes the cased spelling like the excluded name, so the skill is reported.
        assert find_unscanned_plugin_skill_manifests(tmp_path, "custom") == (
            [PurePosixPath(f"custom/top/{folder}/inner/SKILL.md")],
            True,
        )
        assert checks(result)["plugin_skill_in_unscanned_folder"] == Severity.HIGH
        assert not result.complete
    else:
        # A distinct folder on a case-sensitive filesystem is discovered and scanned like any other.
        assert f"custom/top/{folder}/inner" in skills
        assert "plugin_skill_in_unscanned_folder" not in checks(result)


def test_cased_dependency_folder_in_a_declared_skills_folder(tmp_path):
    write_tree(
        tmp_path,
        {
            CODEX_PROFILE.manifest_path: {"name": "demo", "skills": "./custom"},
            "custom/top/SKILL.md": SKILL,
            "custom/top/Node_Modules/pkg/SKILL.md": SKILL,
        },
    )
    found, complete = find_dependency_folder_skills(tmp_path, "custom")
    result = discover_plugin_members(tmp_path)
    assert result is not None
    if case_insensitive(tmp_path / "custom" / "top", "Node_Modules", "node_modules"):
        assert (found, complete) == ([PurePosixPath("custom/top/Node_Modules/pkg/SKILL.md")], True)
        finding = next(f for f in result.findings if f.check_name == "plugin_skill_in_unscanned_folder")
        assert "'Node_Modules/'" in finding.message
        assert not result.complete
    else:
        assert (found, complete) == ([], True)
        assert "custom/top/Node_Modules/pkg" in {m.path for m in result.components if m.type == "skill"}


def test_cased_artifact_folder_below_a_component_folder(tmp_path):
    write_tree(
        tmp_path,
        {
            CLAUDE_PROFILE.manifest_path: {"name": "demo"},
            "agents/EVALS/helper.md": "---\nname: helper\ndescription: Helps.\n---\nHelp.\n",
        },
    )
    result = discover_plugin_members(tmp_path)
    assert result is not None
    assert [m.path for m in result.components if m.type == "agent"] == ["agents/EVALS/helper.md"]
    if case_insensitive(tmp_path / "agents", "EVALS", "evals"):
        assert checks(result)["plugin_component_path_unscanned"] == Severity.HIGH
        assert not result.complete
    else:
        assert "plugin_component_path_unscanned" not in checks(result)
