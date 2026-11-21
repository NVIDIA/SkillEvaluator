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
from skillevaluator.validators import code_risk
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
@pytest.mark.parametrize("other_code", [set(), {"test_tools/launch.py"}], ids=["alone", "beside-a-test_tools-folder"])
def test_test_files_are_not_opened(plain_root: Path, monkeypatch: pytest.MonkeyPatch, other_code: set[str]) -> None:
    # Bandit waits for ever on a named pipe that it opens. Without the exclude
    # entries that only ends when the scan is stopped, sooner here than usual.
    # The pattern for test_*.py also hides test_tools/launch.py, and scanning
    # that file must not bring the test files back.
    run = Tools.bandit.run
    monkeypatch.setattr(Tools.bandit, "run", lambda args, timeout: run(args, timeout=min(timeout, 20)))
    skill = _skill(plain_root / "skills" / "webapp-checks")
    os.mkfifo(plain_root / "pipe")
    os.mkfifo(skill / "scripts" / "pipe_test.py")
    (skill / "scripts" / "test_pipe.py").symlink_to(plain_root / "pipe")
    for relative in other_code:
        (skill / relative).parent.mkdir()
        (skill / relative).write_text(RISKY, encoding="utf-8")
        os.mkfifo((skill / relative).with_name("test_pipe.py"))

    assert _flagged_files(skill) == LIVE_FILES | other_code


@needs_bandit
def test_folder_named_like_a_test_file_keeps_its_other_files(plain_root: Path) -> None:
    # Bandit's "*" also matches a path separator, so the pattern for
    # test_*.py hides test_tools/launch.py too. That file is still reported,
    # and the test files beside it are not. The folder above the skill is
    # one that Bandit's own exclude list would hide.
    skill = _skill(plain_root / ".github" / "skills" / "webapp-checks")
    (skill / "test_tools").mkdir()
    (skill / "test_tools" / "launch.py").write_text(RISKY, encoding="utf-8")
    (skill / "test_tools" / "test_launch.py").write_text(RISKY, encoding="utf-8")
    (skill / "scripts" / "test_broken.py").write_text("def broken(:\n", encoding="utf-8")

    assert _flagged_files(skill) == LIVE_FILES | {"test_tools/launch.py"}


@needs_bandit
@pytest.mark.skipif(os.name == "nt", reason="Windows has no * in a name")
def test_folder_named_like_an_exclude_entry_keeps_the_same_files(plain_root: Path) -> None:
    # Bandit adds its own "*" to an entry that is the path of a directory, and
    # the entry then no longer hides the test files. The same files are
    # reported as without those folders.
    skill = _skill(plain_root / "skills" / "webapp-checks")
    for folder in ("*_test.py", "tests/*"):
        (skill / folder).mkdir()
        (skill / folder / "launch.py").write_text(RISKY, encoding="utf-8")

    assert _flagged_files(skill) == LIVE_FILES | {"*_test.py/launch.py"}


@needs_bandit
def test_folder_that_cannot_be_read_does_not_stop_the_scan(plain_root: Path) -> None:
    # Bandit checks whether an exclude entry is a directory, and so does this
    # check. Below a folder that cannot be opened that question has no
    # answer, which must not end the scan.
    skill = _skill(plain_root / "skills" / "webapp-checks")
    (skill / "tests").chmod(0o000)
    try:
        assert _flagged_files(skill) == LIVE_FILES
    finally:
        (skill / "tests").chmod(0o700)


@needs_bandit
def test_skill_passes_when_only_its_test_files_have_findings(plain_root: Path) -> None:
    skill = plain_root / "skills" / "webapp-checks"
    (skill / "scripts").mkdir(parents=True)
    (skill / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (skill / "scripts" / "test_run.py").write_text(RISKY, encoding="utf-8")

    result = CodeRiskValidator(use_semgrep=False).validate(skill)

    assert result.passed and not result.is_incomplete
    assert result.findings == []


CLEAN = ToolResult(success=True, stdout=json.dumps({"results": [], "errors": []}), stderr="", exit_code=0)


def _bandit_runs(mock_run, skill: Path, **options: bool) -> list[list[str]]:
    """The arguments of each Bandit run that a scan of *skill* makes."""
    if mock_run.return_value is not CLEAN and not mock_run.side_effect:
        mock_run.return_value = CLEAN
    mock_run.reset_mock()
    CodeRiskValidator(**options)._run_bandit(skill)
    return [call.args[0] for call in mock_run.call_args_list]


def _named_files(args: list[str]) -> list[str]:
    """The files that one Bandit run is given by name."""
    return [] if args[0] == "-r" else args[: args.index("-f")]


def _exclude_entries(mock_run, skill: Path) -> list[str]:
    args = _bandit_runs(mock_run, skill)[0]
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


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_one_run_when_the_patterns_hide_nothing_else(mock_run, tmp_path: Path) -> None:
    (run,) = _bandit_runs(mock_run, _skill(tmp_path / "webapp-testing"))

    assert run[:2] == ["-r", str((tmp_path / "webapp-testing").resolve())]


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_file_that_a_pattern_hides_is_passed_by_name(mock_run, tmp_path: Path) -> None:
    skill = _skill(tmp_path / "webapp-testing")
    (skill / "test_tools").mkdir()
    for name in ("launch.py", "test_launch.py", "window.pyw"):
        (skill / "test_tools" / name).write_text(RISKY, encoding="utf-8")

    whole_tree, by_name = _bandit_runs(mock_run, skill)

    # The first run still hides every test file. The second names the one
    # file hidden with them and excludes nothing. window.pyw does not end in
    # ".py", so the first run reads it.
    assert whole_tree[:2] == ["-r", str(skill.resolve())]
    assert str(skill.resolve() / "test_*.py") in whole_tree[whole_tree.index("--exclude") + 1].split(",")
    assert _named_files(by_name) == [str(skill.resolve() / "test_tools" / "launch.py")]
    assert by_name[1:] == [*whole_tree[2:6], "--exclude", "", *whole_tree[8:]]
    # With the test files in the scan no pattern hides that file.
    assert len(_bandit_runs(mock_run, skill, exclude_tests=False)) == 1


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_files_passed_by_name_keep_each_command_line_bounded(mock_run, tmp_path: Path) -> None:
    # Windows allows 32,767 characters for a command line. A name with
    # letters outside ASCII takes more room than it has characters.
    skill = _skill(tmp_path / "webapp-testing")
    (skill / "test_tools").mkdir()
    hidden = [skill.resolve() / "test_tools" / f"module_{index:03d}_{'ü' * 60}.py" for index in range(600)]
    for file in hidden:
        file.write_text(RISKY, encoding="utf-8")

    whole_tree, *by_name = _bandit_runs(mock_run, skill)

    assert whole_tree[0] == "-r" and len(by_name) > 2
    assert all(sum(len(os.fsencode(arg)) + 3 for arg in args) < 32_767 - 2_000 for args in by_name)
    assert [name for args in by_name for name in _named_files(args)] == [str(file) for file in hidden]


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_runs_of_one_scan_share_its_time(mock_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # One run had 120 seconds, and so does a scan that takes several.
    skill = _skill(tmp_path / "webapp-testing")
    (skill / "test_tools").mkdir()
    for name in ("first.py", "second.py", "third.py"):
        (skill / "test_tools" / name).write_text(RISKY, encoding="utf-8")
    monkeypatch.setattr(code_risk, "_BANDIT_NAMED_FILES_BYTES", 1)
    clock = iter([0.0, 0.0, 50.0, 100.0, 150.0])
    monkeypatch.setattr(code_risk, "monotonic", lambda: next(clock))
    mock_run.return_value = CLEAN

    result = CodeRiskValidator()._run_bandit(skill)

    # The fourth run is not started, and the scan is not reported as clean.
    assert [call.kwargs["timeout"] for call in mock_run.call_args_list] == [120, 70, 20]
    assert result.is_incomplete and not result.success_details
    assert any("timed out after 120 seconds" in warning for warning in result.warnings)


@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_scan_errors_of_all_runs_are_written_out_three_at_most(mock_run, tmp_path: Path) -> None:
    skill = _skill(tmp_path / "webapp-testing")
    (skill / "test_tools").mkdir()
    (skill / "test_tools" / "launch.py").write_text(RISKY, encoding="utf-8")
    errors = [
        {"filename": f"file_{index}.py", "reason": "syntax error while parsing AST from file"} for index in range(2)
    ]
    mock_run.return_value = ToolResult(
        success=True, stdout=json.dumps({"results": [], "errors": errors}), stderr="", exit_code=0
    )

    result = CodeRiskValidator()._run_bandit(skill)

    assert mock_run.call_count == 2
    assert sum("Bandit scan error" in error for error in result.errors) == 3
    assert result.is_incomplete and not result.success_details


@pytest.mark.skipif(os.name == "nt", reason="Windows has no * in a name")
@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_bandit_does_not_walk_the_tree_when_an_exclude_entry_is_a_folder(mock_run, tmp_path: Path) -> None:
    skill = _skill(tmp_path / "webapp-testing")
    (skill / "tests" / "*").mkdir()

    (by_name,) = _bandit_runs(mock_run, skill)

    assert _named_files(by_name) == sorted(str(skill.resolve() / relative) for relative in LIVE_FILES)
    # A skill with nothing left to scan needs no run at all.
    for relative in LIVE_FILES:
        (skill / relative).unlink()
    assert _bandit_runs(mock_run, skill) == []
    result = CodeRiskValidator()._run_bandit(skill)
    assert not result.is_incomplete and [success.check_name for success in result.success_details] == ["bandit"]


def _finding(file: Path) -> dict:
    return {
        "issue_severity": "HIGH",
        "issue_confidence": "HIGH",
        "issue_text": "subprocess call with shell=True identified, security issue.",
        "test_id": "B602",
        "test_name": "subprocess_popen_with_shell_equals_true",
        "filename": str(file),
        "line_number": 5,
    }


def _report(*files: Path) -> ToolResult:
    findings = [_finding(file) for file in files]
    return ToolResult(
        success=True, stdout=json.dumps({"results": findings, "errors": []}), stderr="", exit_code=1 if findings else 0
    )


@pytest.mark.parametrize(
    ("whole_tree", "by_name", "flagged", "incomplete", "successes"),
    [
        ([], [], [], False, 1),
        (["scripts/run.py"], [], ["scripts/run.py"], False, 0),
        ([], ["test_tools/launch.py"], ["test_tools/launch.py"], False, 0),
        (["scripts/run.py"], ["test_tools/launch.py"], ["scripts/run.py", "test_tools/launch.py"], False, 0),
        (["scripts/run.py"], None, ["scripts/run.py"], True, 0),
        ([], None, [], True, 0),
        (None, [], [], True, 0),
    ],
    ids=[
        "both-clean",
        "finding-in-the-tree",
        "finding-in-a-named-file",
        "findings-in-both",
        "second-run-fails-after-a-finding",
        "second-run-fails-after-a-clean-run",
        "first-run-fails",
    ],
)
@patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
@patch.object(Tools.bandit, "run")
def test_two_runs_make_one_report(
    mock_run,
    tmp_path: Path,
    whole_tree: list[str] | None,
    by_name: list[str] | None,
    flagged: list[str],
    incomplete: bool,
    successes: int,
) -> None:
    skill = _skill(tmp_path / "webapp-testing").resolve()
    (skill / "test_tools").mkdir()
    (skill / "test_tools" / "launch.py").write_text(RISKY, encoding="utf-8")
    timed_out = ToolResult(success=False, stdout="", stderr="", exit_code=-1, error_message="bandit timed out")
    mock_run.side_effect = [
        timed_out if files is None else _report(*(skill / relative for relative in files))
        for files in (whole_tree, by_name)
    ]

    result = CodeRiskValidator()._run_bandit(skill)

    assert [Path(finding.file_path).relative_to(skill).as_posix() for finding in result.findings] == flagged
    assert result.is_incomplete is incomplete
    assert len(result.success_details) == successes
    assert sum("Bandit found" in message for message in result.messages) == (1 if flagged else 0)
    # A run after one that failed would not make the scan complete.
    assert mock_run.call_count == (1 if whole_tree is None else 2)


@needs_bandit
def test_differently_cased_artifact_folder_follows_the_filesystem(plain_root: Path) -> None:
    # "Evals" is the evals folder where the filesystem ignores case, and a
    # folder of its own where it does not.
    skill = _skill(plain_root / "skills" / "webapp-checks")
    (skill / "Evals").mkdir(exist_ok=True)
    (skill / "Evals" / "copy.py").write_text(RISKY, encoding="utf-8")
    same_folder = (skill / "Evals").samefile(skill / "evals")

    assert _flagged_files(skill) == (LIVE_FILES if same_folder else LIVE_FILES | {"Evals/copy.py"})
