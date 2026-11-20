# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What Bandit scans must not depend on how the skill or a folder above it is named.

Bandit drops a file when an ``--exclude`` entry is a substring of its whole
path, so the bare names the check used to pass ("test", "evals", "results",
"versions", ".git") hid every file of a skill such as ``webapp-testing``.
"""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.code_risk import CodeRiskValidator

RISKY = "import subprocess\n\n\ndef run(command):\n    subprocess.call(command, shell=True)\n"
LIVE_FILES = {"scripts/run.py", "scripts/contest.py", "pkg,one/mod.py"}
TEST_FILES = {"scripts/test_run.py", "scripts/run_test.py", "tests/helper.py", "pkg,one/test/helper.py"}
ARTIFACT_COPIES = {"evals/results/run.py", "versions/v1/run.py"}

needs_bandit = pytest.mark.skipif(not Tools.bandit.is_available, reason="Bandit is not installed")


@pytest.fixture
def plain_root() -> Iterator[Path]:
    """A directory whose own path holds none of the excluded names, unlike pytest's ``tmp_path``."""
    with tempfile.TemporaryDirectory(prefix="se_") as tmp:
        root = Path(tmp).resolve()
        if any(name in str(root) for name in ("test", "evals", "results", "versions", ".git")):
            pytest.skip(f"{root} already contains an excluded name")
        yield root


def _skill(path: Path) -> Path:
    for relative in sorted(LIVE_FILES | TEST_FILES | ARTIFACT_COPIES):
        file = path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(RISKY, encoding="utf-8")
    (path / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when checking a web app.\n---\n", encoding="utf-8"
    )
    return path


def _flagged_files(skill: Path, **options: bool) -> set[str]:
    result = CodeRiskValidator(use_semgrep=False, **options).validate(skill)
    assert not result.is_incomplete
    root = skill.resolve()
    return {Path(finding.file_path).relative_to(root).as_posix() for finding in result.findings}


@needs_bandit
@pytest.mark.parametrize(
    "name", ["webapp-checks", "webapp-testing", "unit-conversions", "search-results", "latest-evals"]
)
def test_skill_name_does_not_decide_what_bandit_scans(plain_root: Path, name: str) -> None:
    # Tests and artifact copies are skipped, and a file whose name only
    # contains "test" (contest.py) is not a test.
    assert _flagged_files(_skill(plain_root / "skills" / name)) == LIVE_FILES


@needs_bandit
@pytest.mark.parametrize("above", ["skills-test", ".github", "results/2026"])
def test_folder_above_the_skill_does_not_hide_it(plain_root: Path, above: str) -> None:
    assert _flagged_files(_skill(plain_root / above / "skills" / "webapp-checks")) == LIVE_FILES


@needs_bandit
def test_tests_are_scanned_when_they_are_not_excluded(plain_root: Path) -> None:
    skill = _skill(plain_root / "skills" / "webapp-testing")

    assert _flagged_files(skill, exclude_tests=False) == LIVE_FILES | TEST_FILES


@needs_bandit
@pytest.mark.parametrize("name", ["a,b", "a[1]", "a*b?c"])
def test_commas_and_pattern_characters_in_a_name_keep_the_same_files(plain_root: Path, name: str) -> None:
    # Bandit splits --exclude at commas and reads each entry as a pattern.
    assert _flagged_files(_skill(plain_root / "skills" / name)) == LIVE_FILES


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_exclude_entries_are_paths_below_the_skill(mock_run, tmp_path: Path) -> None:
    mock_run.return_value = ToolResult(success=True, stdout=json.dumps({"results": []}), stderr="", exit_code=0)
    skill = _skill(tmp_path / "webapp-testing")

    CodeRiskValidator()._run_bandit(skill)

    args = mock_run.call_args.args[0]
    entries = args[args.index("--exclude") + 1].split(",")
    assert len(entries) == 6
    assert all(Path(entry).is_absolute() and entry.startswith(str(skill.resolve())) for entry in entries)


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_exclude_is_passed_empty_when_there_is_nothing_to_skip(mock_run, tmp_path: Path) -> None:
    # Without the option Bandit falls back to its own list, which has ".git" in it.
    mock_run.return_value = ToolResult(success=True, stdout=json.dumps({"results": []}), stderr="", exit_code=0)
    (tmp_path / "run.py").write_text(RISKY, encoding="utf-8")

    CodeRiskValidator()._run_bandit(tmp_path)

    args = mock_run.call_args.args[0]
    assert args[args.index("--exclude") + 1] == ""
