# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What Bandit scans must not depend on how the skill or a folder above it is named.

Bandit drops a file when an ``--exclude`` entry is a substring of its whole
path, so the bare names the check used to pass ("test", "evals", "results",
"versions", ".git") hid every file of a skill such as ``webapp-testing``.
"""

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.constants import SCAN_EXCLUDED_DIRS
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
@pytest.mark.parametrize(
    "name",
    [
        "a,b",
        "a[1]",
        pytest.param("a*b?c", marks=pytest.mark.skipif(os.name == "nt", reason="Windows has no * or ? in a name")),
    ],
)
def test_commas_and_pattern_characters_in_a_name_keep_the_same_files(plain_root: Path, name: str) -> None:
    # Bandit splits --exclude at commas and reads each entry as a pattern.
    assert _flagged_files(_skill(plain_root / "skills" / name)) == LIVE_FILES


@needs_bandit
def test_parse_errors_in_test_files_do_not_leave_the_scan_incomplete(plain_root: Path) -> None:
    skill = _skill(plain_root / "skills" / "webapp-checks")
    (skill / "scripts" / "test_broken.py").write_text("def broken(:\n", encoding="utf-8")

    assert _flagged_files(skill) == LIVE_FILES


@needs_bandit
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a named pipe")
def test_test_files_are_not_opened(plain_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Bandit waits for ever on a named pipe that it opens. Without the exclude
    # entries that only ends when the scan is stopped, sooner here than usual.
    run = Tools.bandit.run
    monkeypatch.setattr(Tools.bandit, "run", lambda args, timeout: run(args, timeout=min(timeout, 20)))
    skill = _skill(plain_root / "skills" / "webapp-checks")
    os.mkfifo(plain_root / "pipe")
    os.mkfifo(skill / "scripts" / "pipe_test.py")
    (skill / "scripts" / "test_pipe.py").symlink_to(plain_root / "pipe")

    assert _flagged_files(skill) == LIVE_FILES


@needs_bandit
def test_folder_named_like_a_test_file_keeps_its_other_files(plain_root: Path) -> None:
    # A pattern for test_*.py would also match test_tools/launch.py, so here
    # those test files are read and left out of the report, parse errors too.
    skill = _skill(plain_root / "skills" / "webapp-checks")
    (skill / "test_tools").mkdir()
    (skill / "test_tools" / "launch.py").write_text(RISKY, encoding="utf-8")
    (skill / "test_tools" / "test_launch.py").write_text(RISKY, encoding="utf-8")
    (skill / "scripts" / "test_broken.py").write_text("def broken(:\n", encoding="utf-8")

    assert _flagged_files(skill) == LIVE_FILES | {"test_tools/launch.py"}


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
def test_exclude_entries_are_anchored_at_the_skill(mock_run, tmp_path: Path) -> None:
    skill = tmp_path / "webapp-testing"
    skill.mkdir()
    (skill / "run.py").write_text(RISKY, encoding="utf-8")

    entries = _exclude_entries(mock_run, skill)

    # Two patterns for each folder name, one right below the skill and one
    # for any depth, and three for test files. No entry is a bare name that
    # could match above the skill.
    assert len(entries) == 2 * (len(SCAN_EXCLUDED_DIRS) + 2) + 3
    assert all(entry.startswith(str(skill.resolve()) + os.sep) for entry in entries)
    assert str(skill.resolve() / "tests" / "*") in entries
    assert str(skill.resolve() / "*" / "tests" / "*") in entries
    assert {str(skill.resolve() / name) for name in ("test_*.py", "*/test_*.py", "*_test.py")} <= set(entries)


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_exclude_does_not_grow_with_the_tree(mock_run, tmp_path: Path) -> None:
    # One entry per test file or per tests/ folder would not fit a Windows
    # command line for a skill that vendors a large library.
    skill = _skill(tmp_path / "webapp-testing")
    before = _exclude_entries(mock_run, skill)
    for index in range(300):
        package = skill / "vendor" / f"package_{index}"
        (package / "tests").mkdir(parents=True)
        (package / "tests" / "helper.py").write_text(RISKY, encoding="utf-8")
        (package / f"test_case_{index}.py").write_text(RISKY, encoding="utf-8")

    assert _exclude_entries(mock_run, skill) == before


@needs_bandit
def test_differently_cased_artifact_folder_follows_the_filesystem(plain_root: Path) -> None:
    # "Evals" is the evals folder where the filesystem ignores case, and a
    # folder of its own where it does not.
    skill = _skill(plain_root / "skills" / "webapp-checks")
    (skill / "Evals").mkdir(exist_ok=True)
    (skill / "Evals" / "copy.py").write_text(RISKY, encoding="utf-8")
    same_folder = (skill / "Evals").samefile(skill / "evals")

    assert _flagged_files(skill) == (LIVE_FILES if same_folder else LIVE_FILES | {"Evals/copy.py"})
