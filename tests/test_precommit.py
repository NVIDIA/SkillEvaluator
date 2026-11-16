# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.precommit import (
    CHECKS,
    collect_skill_roots,
    run_precommit,
    skill_root_for,
)

REPO = Path(__file__).resolve().parents[1]

_VALID_SKILL_MD = """\
---
name: {name}
description: A {name} skill used to test pre-commit path mapping. Use when checking offline Tier 1 hooks.
metadata:
  author: Test Author <author@example.com>
---

# {title}

## Instructions

1. Run the documented command.

## Examples

Example of {name}.
"""


def _write_skill(parent: Path, name: str, *, body: str | None = None) -> Path:
    skill = parent / name
    skill.mkdir(parents=True)
    text = body if body is not None else _VALID_SKILL_MD.format(name=name, title=name.replace("-", " ").title())
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    return skill


def test_skill_root_maps_nested_skill_not_outer(tmp_path: Path) -> None:
    outer = _write_skill(tmp_path, "outer")
    inner = _write_skill(outer, "inner")
    script = inner / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("print(0)\n", encoding="utf-8")

    assert skill_root_for(script, cwd=tmp_path) == inner.resolve()
    assert skill_root_for(inner / "SKILL.md", cwd=tmp_path) == inner.resolve()
    assert skill_root_for(outer / "SKILL.md", cwd=tmp_path) == outer.resolve()


def test_skill_root_ignores_files_outside_a_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "sample")
    readme = tmp_path / "README.md"
    readme.write_text("notes\n", encoding="utf-8")
    stray = tmp_path / "docs" / "guide.md"
    stray.parent.mkdir()
    stray.write_text("guide\n", encoding="utf-8")

    assert skill_root_for(readme, cwd=tmp_path) is None
    assert skill_root_for(stray, cwd=tmp_path) is None
    assert skill_root_for("README.md", cwd=tmp_path) is None


def test_skill_root_skips_deleted_skill(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    kept = _write_skill(skills, "kept")
    deleted = skills / "gone" / "scripts" / "run.py"

    assert skill_root_for(deleted, cwd=tmp_path) is None
    assert skill_root_for(skills / "gone" / "SKILL.md", cwd=tmp_path) is None
    assert skill_root_for(kept / "SKILL.md", cwd=tmp_path) == kept.resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="backslash is a separator only on Windows")
def test_skill_root_accepts_windows_style_separators(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path / "skills", "win-skill")
    script = skill / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("print(0)\n", encoding="utf-8")

    mapped = skill_root_for(r"skills\win-skill\scripts\run.py", cwd=tmp_path)
    assert mapped == skill.resolve()
    assert skill_root_for(r"skills\win-skill\SKILL.md", cwd=tmp_path) == skill.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="backslash is a valid filename character only on POSIX")
def test_skill_root_keeps_literal_backslash_in_posix_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill = _write_skill(tmp_path, "literal\\skill", body="# Broken\n\nNo frontmatter.\n")
    assert skill.name == "literal\\skill"
    # A same-named slash path must not be found instead.
    assert not (tmp_path / "literal" / "skill").exists()

    relative = "literal\\skill/SKILL.md"
    assert skill_root_for(relative, cwd=tmp_path) == skill.resolve()

    code, lines = run_precommit("schema", [relative], cwd=tmp_path)
    assert code == 1
    assert len(lines) == 1
    assert lines[0].startswith("FAIL  literal\\skill  schema")

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["precommit-run", "schema", relative])
    assert result.exit_code == 1, result.output
    assert "FAIL  literal\\skill  schema" in result.output


def test_collect_skill_roots_deduplicates_and_preserves_order(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    first = _write_skill(skills, "alpha")
    second = _write_skill(skills, "beta")
    (first / "scripts").mkdir()
    (first / "scripts" / "a.py").write_text("print(1)\n", encoding="utf-8")
    (first / "references").mkdir()
    (first / "references" / "note.md").write_text("note\n", encoding="utf-8")

    roots = collect_skill_roots(
        [
            first / "scripts" / "a.py",
            first / "SKILL.md",
            second / "SKILL.md",
            first / "references" / "note.md",
            tmp_path / "README.md",
        ],
        cwd=tmp_path,
    )
    assert roots == [first.resolve(), second.resolve()]


def test_precommit_quality_exit_codes(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, "sample-skill")
    paths = [str(skill / "SKILL.md")]

    passed_code, passed_lines = run_precommit("quality", paths, min_score=0, cwd=tmp_path)
    assert passed_code == 0
    assert len(passed_lines) == 1
    assert passed_lines[0].startswith("PASS  sample-skill  quality  ")
    assert "/100" in passed_lines[0]

    failed_code, failed_lines = run_precommit("quality", paths, min_score=100, cwd=tmp_path)
    assert failed_code == 1
    assert failed_lines[0].startswith("FAIL  sample-skill  quality  ")
    assert "(min 100)" in failed_lines[0]


def test_precommit_schema_exit_codes(tmp_path: Path) -> None:
    valid = _write_skill(tmp_path, "valid-skill")
    invalid = _write_skill(
        tmp_path,
        "broken-skill",
        body="# Broken\n\nNo frontmatter.\n",
    )

    code, lines = run_precommit("schema", [str(valid / "SKILL.md")], cwd=tmp_path)
    assert code == 0
    assert lines == ["PASS  valid-skill  schema"]

    code, lines = run_precommit("schema", [str(invalid / "SKILL.md")], cwd=tmp_path)
    assert code == 1
    assert lines[0].startswith("FAIL  broken-skill  schema  ")


def test_precommit_lint_scripts_is_advisory_pass(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, "lint-skill")
    script = skill / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("print(1)\n", encoding="utf-8")

    code, lines = run_precommit("lint-scripts", [str(script)], cwd=tmp_path)
    assert code == 0
    assert lines == ["PASS  lint-skill  lint-scripts"]


def test_precommit_ignores_unrelated_and_deleted_paths(tmp_path: Path) -> None:
    code, lines = run_precommit(
        "schema",
        ["README.md", "skills/missing/SKILL.md", "docs/notes.md"],
        cwd=tmp_path,
    )
    assert code == 0
    assert lines == []


def test_precommit_does_not_write_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    skill = _write_skill(tmp_path, "sample-skill")
    code, _lines = run_precommit("quality", [str(skill / "SKILL.md")], min_score=0, cwd=tmp_path)
    assert code == 0
    assert not (tmp_path / "reports").exists()
    assert list(tmp_path.glob("skillevaluator-output*")) == []


def test_precommit_cli_exit_codes(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, "cli-skill")
    runner = CliRunner()

    passed = runner.invoke(
        cli,
        ["precommit-run", "--min-score", "70", "quality", str(skill / "SKILL.md")],
    )
    assert passed.exit_code == 0, passed.output
    assert "PASS  cli-skill  quality" in passed.output

    failed = runner.invoke(
        cli,
        ["precommit-run", "--min-score", "100", "quality", str(skill / "SKILL.md")],
    )
    assert failed.exit_code == 1, failed.output
    assert "FAIL  cli-skill  quality" in failed.output
    assert "(min 100)" in failed.output

    # pre-commit appends hook args after the entry, then filenames.
    interleaved = runner.invoke(
        cli,
        ["precommit-run", "quality", "--min-score", "100", str(skill / "SKILL.md")],
    )
    assert interleaved.exit_code == 1, interleaved.output
    assert "(min 100)" in interleaved.output


def test_precommit_run_is_hidden_from_main_help() -> None:
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0
    assert "precommit-run" not in help_result.output

    command_help = runner.invoke(cli, ["precommit-run", "--help"])
    assert command_help.exit_code == 0
    assert "quality" in command_help.output
    assert "schema" in command_help.output
    assert "lint-scripts" in command_help.output


def test_precommit_hooks_yaml_entries_resolve_to_commands() -> None:
    hooks_path = REPO / ".pre-commit-hooks.yaml"
    hooks = yaml.safe_load(hooks_path.read_text(encoding="utf-8"))
    assert isinstance(hooks, list)
    assert hooks, "expected at least one hook"

    runner = CliRunner()
    ids = []
    for hook in hooks:
        assert hook["language"] == "python"
        assert hook.get("files")
        ids.append(hook["id"])
        parts = shlex.split(hook["entry"])
        assert parts[0] == "skillevaluator"
        assert parts[1:]
        result = runner.invoke(cli, [*parts[1:], "--help"])
        assert result.exit_code == 0, f"{hook['id']}: {result.output}"
        command_name = parts[1]
        assert command_name in cli.commands
        if len(parts) > 2:
            check = parts[2]
            assert check in CHECKS

    assert ids == [
        "skillevaluator-quality",
        "skillevaluator-schema",
        "skillevaluator-lint-scripts",
    ]
