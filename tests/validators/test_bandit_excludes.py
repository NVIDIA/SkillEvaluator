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


@needs_bandit
def test_parse_errors_in_test_files_do_not_leave_the_scan_incomplete(plain_root: Path) -> None:
    # Test files outside tests/ are scanned and left out of the report, so
    # one that Bandit cannot parse must not count either.
    skill = _skill(plain_root / "skills" / "webapp-checks")
    (skill / "scripts" / "test_broken.py").write_text("def broken(:\n", encoding="utf-8")

    assert _flagged_files(skill) == LIVE_FILES


@needs_bandit
def test_skill_passes_when_only_its_test_files_have_findings(plain_root: Path) -> None:
    skill = plain_root / "skills" / "webapp-checks"
    (skill / "scripts").mkdir(parents=True)
    (skill / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (skill / "scripts" / "test_run.py").write_text(RISKY, encoding="utf-8")

    result = CodeRiskValidator(use_semgrep=False).validate(skill)

    assert result.passed and not result.is_incomplete
    assert result.findings == []


def _exclude_entries(mock_run, skill: Path) -> list[str]:
    mock_run.return_value = ToolResult(success=True, stdout=json.dumps({"results": []}), stderr="", exit_code=0)
    CodeRiskValidator()._run_bandit(skill)
    args = mock_run.call_args.args[0]
    return args[args.index("--exclude") + 1].split(",")


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_exclude_entries_are_folders_below_the_skill(mock_run, tmp_path: Path) -> None:
    skill = _skill(tmp_path / "webapp-testing")

    entries = _exclude_entries(mock_run, skill)

    root = skill.resolve()
    # A plain folder goes in as it is, and Bandit skips everything below it.
    # The comma in "pkg,one" cannot, because Bandit splits the option there.
    assert sorted(entries) == sorted(
        [
            str(root / "evals"),
            str(root / "tests"),
            str(root / "versions"),
            str(root / "pkg?one" / "test" / "*"),
        ]
    )


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_exclude_does_not_grow_with_the_number_of_test_files(mock_run, tmp_path: Path) -> None:
    # One entry per test file would not fit a Windows command line for a
    # skill that ships a few hundred of them.
    skill = _skill(tmp_path / "webapp-testing")
    before = _exclude_entries(mock_run, skill)
    for index in range(300):
        (skill / "scripts" / f"test_case_{index}.py").write_text(RISKY, encoding="utf-8")

    assert _exclude_entries(mock_run, skill) == before


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_exclude_is_passed_empty_when_there_is_nothing_to_skip(mock_run, tmp_path: Path) -> None:
    # Without the option Bandit falls back to its own list, which has ".git" in it.
    mock_run.return_value = ToolResult(success=True, stdout=json.dumps({"results": []}), stderr="", exit_code=0)
    (tmp_path / "run.py").write_text(RISKY, encoding="utf-8")

    CodeRiskValidator()._run_bandit(tmp_path)

    args = mock_run.call_args.args[0]
    assert args[args.index("--exclude") + 1] == ""
