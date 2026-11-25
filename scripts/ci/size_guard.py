#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the size of a pull request against the line budgets in CONTRIBUTING.md.

Prints a Markdown report and exits 0. Only ``--enforce`` changes that: 1 when a rule
is over its limit, 2 when git could not compare the refs. CI does not pass it yet.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE = "config/size_baseline.json"
RATIO = "python_tests_to_source_ratio"
BIG_FILE = 1500  # A tracked file with more lines is recorded in the baseline.
FIX_SRC, FIX_TESTS, FIX_TEST_FUNCTIONS = 150, 400, 3  # G1 and G2: net lines and test functions a fix may add.
TEST_RATIO, TEST_FLOOR = 3, 100  # G3: net test lines per added source line, once past the floor.
FIXTURE_LINES = 100  # G4
RULES = {
    "G1": "fix line budget",
    "G2": "fix extends existing tests",
    "G3": "big files and test ratio",
    "G4": "no large fixtures",
    "G5": "fix adds no source file",
}
FIX_ONLY = {"G1", "G2", "G5"}
# The conventional type of a title: ``fix(scope)!: x`` is ``fix``. A title without a prefix is ``other``.
_TITLE = re.compile(r"(\w+)(\([^)]*\))?!?:")
# One ``--numstat -z`` record. A rename has an empty path, then the old and the new path.
_NUMSTAT = re.compile(r"(\d+|-)\t(\d+|-)\t(?:\0([^\0]*)\0)?([^\0]+)\0")
_TEST_FILE = re.compile(r"tests/(?:[^/]+/)*(?:test_[^/]*|[^/]*_test)\.py")
_TEST_DEF = re.compile(r"^([+-])\s*(?:async\s+)?def test_", re.MULTILINE)
# Git grep terminates each path with NUL and the following line count with LF.
_GREP_COUNT = re.compile(r"([^\0]+)\0([0-9]+)\n")


def _git_bytes(repo: Path, *args: str) -> bytes:
    # stderr is not captured, so a failing git command explains itself in the CI log.
    command = ["git", "-C", str(repo), *args]
    return subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout


def _git(repo: Path, *args: str) -> str:
    return _git_bytes(repo, *args).decode("utf-8", errors="replace")


def _baseline(repo: Path) -> dict:
    path = repo / BASELINE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _name(path: str) -> str:
    """A path as inert Markdown on one line: file names come from the pull request and may hold anything."""
    return "`" + json.dumps(path)[1:-1].replace("`", "\\u0060") + "`"


def _unittest_bases(tree: ast.Module) -> set[str]:
    """Names for unittest.TestCase that are imported directly by this module."""
    bases = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            bases.update(f"{alias.asname or alias.name}.TestCase" for alias in statement.names if alias.name == "unittest")
        elif isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module == "unittest":
            bases.update(alias.asname or alias.name for alias in statement.names if alias.name == "TestCase")
    return bases


def _test_functions(repo: Path, blob: str) -> dict[tuple[str, ...], ast.FunctionDef | ast.AsyncFunctionDef]:
    source = _git_bytes(repo, "cat-file", "blob", blob)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # A syntactically invalid module has no collectable test functions.
        return {}

    unittest_bases = _unittest_bases(tree)
    names: dict[tuple[str, ...], ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        if isinstance(node, ast.ClassDef):
            # Match pytest's Test* naming rule for plain classes.
            if not node.name.startswith("Test") and not any(ast.unparse(base) in unittest_bases for base in node.bases):
                return
            scope = (*scope, node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                names[(*scope, node.name)] = node
            return
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return names


def _new_tests(repo: Path, span: str) -> tuple[list[str], int]:
    # Raw records give both blob IDs and pair test-module renames, while moves
    # into tests/ are additions. Compare names within each paired module.
    diff = _git(repo, "diff", "--raw", "--no-abbrev", "-z", "-M", span, "--", "tests/")
    records = iter(diff.split("\0")[:-1])
    modules = []
    added: Counter[tuple[tuple[str, ...], str]] = Counter()
    removed: Counter[tuple[tuple[str, ...], str]] = Counter()
    for header, before_path in zip(records, records, strict=True):
        before_mode, mode, before_blob, blob, status = header.split()
        path = next(records) if status.startswith("R") else before_path
        before_module = before_mode != ":000000" and _TEST_FILE.fullmatch(before_path)
        after_module = mode != "000000" and _TEST_FILE.fullmatch(path)
        if not before_module and not after_module:
            continue
        if not before_module:
            modules.append(_name(path))
        previous = _test_functions(repo, before_blob) if before_module and before_mode.startswith(":100") else {}
        current = _test_functions(repo, blob) if after_module and mode.startswith("100") else {}
        added.update((name, ast.dump(current[name])) for name in current.keys() - previous.keys())
        removed.update((name, ast.dump(previous[name])) for name in previous.keys() - current.keys())
    # A removal can offset one otherwise-new definition with the same qualified
    # name and AST in another module. A copy leaves no removal to offset it.
    return modules, (added - removed).total()


def measure(repo: Path, base: str, head: str) -> dict:
    """Everything the report shows, read from the diff between the merge base of ``base`` and ``head``."""
    span = f"{base}...{head}"
    rows, new = {}, []  # Lines per path, created paths.
    # Renames are paired inside one part only, so a file moved into src/ or tests/ counts as new there.
    for part in ("src/", "tests/"):
        numstat = _git(repo, "diff", "--numstat", "-z", "-M", span, "--", part)
        for added, removed, _, path in _NUMSTAT.findall(numstat):
            rows[path] = (0, 0) if added == "-" else (int(added), int(removed))  # A binary file counts as 0.
        new += _git(repo, "diff", "--name-only", "-z", "-M", "--diff-filter=A", span, "--", part).split("\0")[:-1]
    # Baseline growth follows a rename anywhere in the tree.
    baseline_files = _baseline(repo).get("files", {})
    all_numstat = _git(repo, "diff", "--numstat", "-z", "-M", span, "--")
    grown = {
        _name(path): int(added) - int(removed)
        for added, removed, old, path in _NUMSTAT.findall(all_numstat)
        if (old or path) in baseline_files and added != "-" and int(added) > int(removed)
    }
    new_test_files, new_test_functions = _new_tests(repo, span)
    defs = _TEST_DEF.findall(_git(repo, "diff", "-U0", "-M", "--no-color", "--no-ext-diff", span, "--", "tests/*.py"))
    found: dict = {
        "new_test_files": new_test_files,
        "new_test_functions": new_test_functions,
        "test_defs": (defs.count("+"), defs.count("-")),
        "new_src_files": [_name(path) for path in new if path.startswith("src/")],
        "grown": grown,
        "fixtures": {
            _name(path): rows[path][0] for path in new if path.startswith("tests/") and not path.endswith(".py")
        },
    }
    for root in ("src/", "tests/"):
        inside = [row for path, row in rows.items() if path.startswith(root)]
        found[root] = (sum(row[0] for row in inside), sum(row[1] for row in inside))
    return found


def judge(found: dict, fix: bool) -> dict[str, list[str]]:
    """What is over its limit, per rule. The ``FIX_ONLY`` rules apply to ``fix`` titles only."""
    (src_added, src_removed), (tests_added, tests_removed) = found["src/"], found["tests/"]
    growth = tests_added - tests_removed
    checks = [  # rule, what the number counts, number, limit
        ("G1", "net lines under src/", src_added - src_removed, FIX_SRC),
        ("G1", "net lines under tests/", growth, FIX_TESTS),
        ("G2", "new test files", len(found["new_test_files"]), 0),
        ("G2", "new test functions", found["new_test_functions"], FIX_TEST_FUNCTIONS),
        ("G3", "baseline files that grew", len(found["grown"]), 0),
        ("G3", f"net test lines for {src_added} added source lines", growth, max(TEST_FLOOR, TEST_RATIO * src_added)),
        *(("G4", f"lines in new {path}", lines, FIXTURE_LINES) for path, lines in found["fixtures"].items()),
        ("G5", "new files under src/", len(found["new_src_files"]), 0),
    ]
    over: dict[str, list[str]] = {rule: [] for rule in RULES}
    for rule, what, number, limit in checks:
        if number > limit and (fix or rule not in FIX_ONLY):
            over[rule].append(f"{what}: {number} (limit {limit})")
    return over


def render(found: dict, kind: str, over: dict[str, list[str]], excepted: bool) -> str:
    """The Markdown report: the measurements, then one line per rule."""
    sizes = {"`src/`": found["src/"], "`tests/`": found["tests/"], "`def test_` lines": found["test_defs"]}
    grown = ", ".join(f"{path} +{net}" for path, net in found["grown"].items())
    verdict = "excepted: " if excepted else "over: "
    lines = [
        f"### Pull request size (report-only), `{kind}` title",
        "",
        *(f"- {what}: +{added} -{removed}, net {added - removed:+}" for what, (added, removed) in sizes.items()),
        f"- New test files: {', '.join(found['new_test_files']) or 'none'}",
        f"- New test functions: {found['new_test_functions']}",
        f"- New files under `src/`: {', '.join(found['new_src_files']) or 'none'}",
        f"- Baseline files that grew: {grown or 'none'}",
        f"- Added test lines against added source lines: {found['tests/'][0]} / {found['src/'][0]}",
        "",
    ]
    for rule, description in RULES.items():
        if kind != "fix" and rule in FIX_ONLY:
            status = "n/a (fix titles only)"
        elif over[rule]:
            status = verdict + "; ".join(over[rule])
        else:
            status = "ok"
        lines.append(f"- {rule} {description}: {status}")
    lines.extend(
        ["", 'See "Keeping Changes Small" in CONTRIBUTING.md; the `size-exception` label marks an agreed exception.']
    )
    return "\n".join(lines)


def update_baseline(repo: Path, ref: str, allow_raise: bool) -> int:
    """Rewrite the baseline from ``ref``. Without ``allow_raise`` no number may go up and no file may be added."""
    # CHANGELOG.md is not recorded: every user-visible change has to add an entry to it.
    records = _git(repo, "grep", "-I", "-c", "-z", "--no-color", "", ref, "--", ":!CHANGELOG.md")
    counts = {path[len(ref) + 1 :]: int(count) for path, count in _GREP_COUNT.findall(records)}
    python = {path: count for path, count in counts.items() if path.endswith(".py")}
    src, tests = (sum(n for path, n in python.items() if path.startswith(root)) for root in ("src/", "tests/"))
    files = {path: count for path, count in sorted(counts.items()) if count > BIG_FILE}
    new = {"version": 1, RATIO: round(tests / max(src, 1), 2), "files": files}
    old = _baseline(repo)
    before = {RATIO: old.get(RATIO, new[RATIO]), **old.get("files", {})}
    raised = [key for key, value in {RATIO: new[RATIO], **files}.items() if value > before.get(key, 0)]
    if raised and not allow_raise:
        print(f"refusing to raise or add (pass --allow-raise to accept): {', '.join(raised)}", file=sys.stderr)
        return 1
    (repo / BASELINE).write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="any directory inside the Git repository")
    parser.add_argument("--base", default="origin/main", help="base ref; the diff starts at its merge base with --head")
    parser.add_argument("--head", default="HEAD", help="head ref")
    parser.add_argument("--title", default=os.environ.get("PR_TITLE"), help="default: $PR_TITLE, else the head subject")
    parser.add_argument("--labels", default=os.environ.get("PR_LABELS", ""), help="comma-separated; default $PR_LABELS")
    parser.add_argument("--enforce", action="store_true", help="exit 1 when a rule is over its limit")
    parser.add_argument("--update-baseline", action="store_true", help=f"rewrite {BASELINE} from --head")
    parser.add_argument("--allow-raise", action="store_true", help="let --update-baseline raise a number or add a file")
    args = parser.parse_args(argv)
    try:
        repo = Path(_git(args.repo, "rev-parse", "--show-toplevel").strip())
        if args.update_baseline:
            return update_baseline(repo, args.head, args.allow_raise)
        title = _TITLE.match((args.title or _git(repo, "log", "-1", "--format=%s", args.head)).strip())
        found = measure(repo, args.base, args.head)
    except subprocess.CalledProcessError as error:  # git has printed the reason to stderr.
        print(f"Pull request size not measured: `git {error.cmd[3]}` failed. Fetch the full history or pass `--base`.")
        return 2 * args.enforce
    kind = title.group(1).lower() if title else "other"
    excepted = "size-exception" in {label.strip() for label in args.labels.split(",")}
    over = judge(found, kind == "fix")
    print(render(found, kind, over, excepted))
    return int(args.enforce and not excepted and any(over.values()))


if __name__ == "__main__":
    raise SystemExit(main())
