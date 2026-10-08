# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M38: a plugin name longer than 64 characters passes Tier 1, so Tier 2 and Tier 3 must take it too.

The manifest package stopped failing 65-character Claude Code and Codex
plugin names at Tier 1 (Claude Code sets no limit, Codex loads them; check-01
e21 and the Codex oracle ``cx-name-65``). Tier 3 staging and the Tier 2
plugin profile still capped the plugin name at the 64-character skill-name
limit, so such a plugin passed Tier 1 and then crashed both. The generated
wrapper ``SKILL.md`` is a skill, so its name is cut to 64 characters with a
short hash, and the report-only signals still know it is the wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.constants import NAME_MAX_LENGTH
from skillevaluator.deduplication.plugin.profile import load_plugin_profile
from skillevaluator.tier3.harbor.runner import _plugin_signals_context
from skillevaluator.tier3.plugin_eval import _wrapper_skill_name, prepare_plugin_eval_package
from skillevaluator.validators.frontmatter_parser import parse_frontmatter

LONG_NAME = "release-kit-" + "x" * 60  # 72 characters


def _plugin(root: Path, name: str = LONG_NAME) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    manifest = {"name": name, "version": "1.0.0", "description": "Release helpers."}
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "skills" / "notes").mkdir(parents=True)
    (root / "skills" / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: Write release notes.\n---\n\n# Notes\n\nWrite them.\n", encoding="utf-8"
    )
    (root / "evals").mkdir()
    (root / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p", "expected_output": "o"}]))
    return root


def test_tier3_stages_a_long_plugin_name_with_a_skill_sized_wrapper(tmp_path: Path) -> None:
    package = prepare_plugin_eval_package(_plugin(tmp_path / "plugin"), stage_root=tmp_path / "stage")

    assert package.plugin_name == LONG_NAME
    parsed, _ = parse_frontmatter(package.package_path / "SKILL.md")
    wrapper = parsed.yaml_data["name"]
    assert len(wrapper) <= NAME_MAX_LENGTH
    assert wrapper.startswith("release-kit-xxx")
    assert wrapper == _wrapper_skill_name(LONG_NAME)


def test_short_plugin_names_keep_their_wrapper_name() -> None:
    assert _wrapper_skill_name("release-kit") == "release-kit"
    assert _wrapper_skill_name("a" * 64) == "a" * 64
    assert _wrapper_skill_name("a" * 70) != _wrapper_skill_name("a" * 71)


def test_signals_treat_the_cut_wrapper_name_as_the_wrapper(tmp_path: Path) -> None:
    package = prepare_plugin_eval_package(_plugin(tmp_path / "plugin"), stage_root=tmp_path / "stage")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    context = _plugin_signals_context(
        skill_path=package.package_path,
        evaluator_skill_path=package.package_path,
        workspace_skills=[],
        run_dir=run_dir,
        baseline_has_members=False,
    )

    assert _wrapper_skill_name(LONG_NAME) in context.wrapper_skills
    # The namespace Claude Code gives the plugin's skills is still the full name.
    assert LONG_NAME in context.plugin_names


def test_tier2_profile_loads_a_long_plugin_name(tmp_path: Path) -> None:
    profile = load_plugin_profile(_plugin(tmp_path / "plugin"))

    assert profile.name == LONG_NAME
