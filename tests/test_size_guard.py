# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from scripts.ci import size_guard

GIT = ["git", "-c", "user.name=CI Test", "-c", "user.email=ci-test@example.com"]
LINE, HELPER, TEST, ASYNC = "x = 1\n", "y = 2\n" * 5, "def test_a():\n    pass\n", "async def test_b():\n    pass\n"
BASELINE = json.dumps({size_guard.RATIO: 0.1, "files": {"src/big.py": 20, "uv.lock": 1}})
START = {"src/big.py": LINE * 20, "src/old.py": LINE * 30, "tests/helper.py": HELPER, "tests/test_big.py": TEST}
SMALL = {"src/old.py": LINE * 32, "tests/test_big.py": TEST + TEST.replace("test_a", "test_c")}
BIG = {
    "src/new.py": LINE * 151,
    "tests/test_big.py": None,
    "tests/unit/test_new.py": "".join(TEST.replace("test_a", f"test_{i}") for i in range(4)) + ASYNC + LINE * 446,
}
BIG_SIZES = ("+151 -0, net +151", "+456 -2, net +454", "+5 -1, net +4")
BIG_OVER = {
    "G1": "net lines under src/: 151 (limit 150); net lines under tests/: 454 (limit 400)",
    "G2": "new test files: 1 (limit 0); new test functions: 5 (limit 3)",
    "G3": "net test lines for 151 added source lines: 454 (limit 453)",
    "G5": "new files under src/: 1 (limit 0)",
}
MIXED = {  # Two baseline files grown (one renamed), a file moved from tests/ to src/, a large fixture, a binary file.
    "src/big.py": None,
    "src/bigger.py": LINE * 21,
    "uv.lock": LINE * 2,
    "tests/helper.py": None,
    "src/helper.txt": HELPER,
    "tests/test_big.py": TEST + LINE * 5,
    "tests/fixtures/a`b.json": TEST + LINE * 99,
    "tests/fixtures/blob.bin": "\0" * 300,
}
MIXED_OVER = {
    "G3": "baseline files that grew: 2 (limit 0); net test lines for 6 added source lines: 101 (limit 100)",
    "G4": "lines in new `tests/fixtures/a\\u0060b.json`: 101 (limit 100)",
    "G5": "new files under src/: 1 (limit 0)",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run([*GIT, "-C", str(repo), *args], check=True, capture_output=True)


def _commit(repo: Path, files: dict[str, str | None]) -> None:
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).unlink() if content is None else (repo / name).write_bytes(content.encode())
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "fix(cli)!: the commit subject")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet", "-b", "trunk")
    _git(tmp_path, "config", "diff.renames", "false")  # The script has to ask for rename detection itself,
    _git(tmp_path, "config", "color.ui", "always")  # and for plain output.
    _commit(tmp_path, {**START, "uv.lock": LINE, size_guard.BASELINE: BASELINE})
    _commit(tmp_path, {"src/later.py": LINE * 500})  # The base branch moves on; this is not in the pull request.
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "trunk")
    _git(tmp_path, "switch", "--quiet", "-c", "pr", "HEAD~1")
    return tmp_path


@pytest.mark.parametrize(
    "case",
    [  # title, labels, changes, lines under src/, lines under tests/, `def test_` lines, what is over its limit
        ("fix: keep it small", "", SMALL, ("+2 -0, net +2", "+2 -0, net +2", "+1 -0, net +1"), {}),
        ("", "bug, size-exception", BIG, BIG_SIZES, BIG_OVER),  # No title: the commit subject, a fix.
        ("Add a module", "", BIG, BIG_SIZES, {"G3": BIG_OVER["G3"]}),  # Not a fix: G1, G2 and G5 do not apply.
        (" Fix: move a module", "", MIXED, ("+6 -0, net +6", "+106 -5, net +101", "+0 -0, net +0"), MIXED_OVER),
    ],
)
def test_report(repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, case: tuple) -> None:
    title, labels, changes, sizes, over = case
    _commit(repo, changes)
    monkeypatch.setenv("PR_TITLE", title)
    monkeypatch.setenv("PR_LABELS", labels)
    assert size_guard.main(["--repo", str(repo / "src")]) == 0
    report = capsys.readouterr().out.splitlines()
    assert [line.split(": ")[1] for line in report[2:5]] == list(sizes)
    verdict = ": excepted: " if labels else ": over: "
    assert {line[2:4]: line.split(verdict)[1] for line in report[-7:-2] if not line.endswith(": ok")} == over
    assert size_guard.main(["--repo", str(repo), "--enforce"]) == int(bool(over) and not labels)
    assert [size_guard.main(["--repo", str(repo), "--base", "no-ref", *flag]) for flag in ([], ["--enforce"])] == [0, 2]


def test_update_baseline_only_lowers(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    update, target = ["--repo", str(repo), "--update-baseline"], repo / size_guard.BASELINE
    names, before = ("src/big.py", "src/huge.py", "tests/test_big.py"), target.read_text(encoding="utf-8")
    _commit(repo, {**dict.fromkeys((*names, "CHANGELOG.md"), LINE * 1501), "src/notes.txt": LINE * 1500})
    assert size_guard.main(update) == 1  # A higher ratio, a higher count, two new files. Not CHANGELOG.md, not 1,500.
    assert capsys.readouterr().err.strip().endswith(f": {size_guard.RATIO}, {', '.join(names)}")
    assert target.read_text(encoding="utf-8") == before
    assert (size_guard.main([*update, "--allow-raise"]), size_guard.main(update)) == (0, 0)
    recorded = {"version": 1, size_guard.RATIO: 0.5, "files": dict.fromkeys(names, 1501)}
    assert json.loads(target.read_text(encoding="utf-8")) == recorded


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"tests/unit/widget_test.py": TEST}, ["`tests/unit/widget_test.py`"]),
        ({"tests/helper.py": None, "tests/helper_test.py": HELPER}, ["`tests/helper_test.py`"]),
        ({"tests/test_big.py": None, "tests/big_test.py": TEST}, []),
    ],
)
def test_new_test_module_paths(repo: Path, changes: dict[str, str | None], expected: list[str]) -> None:
    _commit(repo, changes)
    found = size_guard.measure(repo, "origin/main", "HEAD")
    assert found["new_test_files"] == expected


def test_deleted_functions_do_not_offset_added_functions(repo: Path) -> None:
    tests = "".join(TEST.replace("test_a", f"test_{name}") for name in ("one", "two", "three", "four"))
    _commit(repo, {"tests/test_big.py": tests})
    found = size_guard.measure(repo, "origin/main", "HEAD")
    assert found["test_defs"] == (4, 1)
    assert size_guard.judge(found, fix=True)["G2"] == ["new test functions: 4 (limit 3)"]


def test_existing_functions_stay_paired_on_module_rename(repo: Path) -> None:
    existing = LINE * 25 + TEST + "class TestChecks:\n    def test_a(self):\n        pass\n"
    _commit(repo, {"tests/test_big.py": existing})
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    renamed = existing.replace("def test_a():", "def test_a(value=None):").replace("test_a(self)", "test_b(self)")
    _commit(repo, {"tests/test_big.py": None, "tests/big_test.py": renamed})
    found = size_guard.measure(repo, "origin/main", "HEAD")
    assert found["new_test_files"] == []
    assert found["new_test_functions"] == 1
    assert found["test_defs"] == (2, 2)


def test_cross_scope_rename_preserves_baseline_growth_and_source_budget(repo: Path) -> None:
    content = "".join(f"constant_{i} = {i}\n" for i in range(30))
    baseline = json.dumps({"files": {"scripts/big.py": 30}})
    _commit(repo, {"scripts/big.py": content, size_guard.BASELINE: baseline})
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _commit(repo, {"scripts/big.py": None, "src/relocated.py": content + LINE})
    found = size_guard.measure(repo, "origin/main", "HEAD")
    assert found["grown"] == {"`src/relocated.py`": 1}
    assert found["src/"] == (31, 0)
    assert found["new_src_files"] == ["`src/relocated.py`"]
