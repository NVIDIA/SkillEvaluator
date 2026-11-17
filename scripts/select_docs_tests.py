#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List the test files that read documentation, for the docs-only CI lane.

A docs-only pull request skips the platform matrix, so the tests that pin
documentation text have to run somewhere else. This prints every test file that
names a path the change classifier treats as documentation, or that imports a
test helper which does.

The match is on file text and is generous on purpose. An extra test file costs
seconds; a missed one lets a broken page reach ``main``. Two things are not
followed: a ``conftest.py`` fixture that reads documentation, and a helper
imported by a computed name.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from scripts.classify_ci_changes import DOC_FILES, DOC_PREFIXES

# The same patterns pytest collects (``python_files`` in pyproject.toml).
_TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


def doc_reference_pattern() -> re.Pattern[str]:
    """Match the ways a test names a path the classifier calls documentation.

    The patterns come from the classifier's own lists, so a path added there is
    covered here without a second edit.
    """
    alternatives = sorted({re.escape(PurePosixPath(path.decode()).name) for path in DOC_FILES})
    for directory in sorted({PurePosixPath(prefix.decode()).name for prefix in DOC_PREFIXES}):
        name = re.escape(directory)
        # ``"docs/page.mdx"`` as one string, or ``root / "docs" / "page.mdx"``.
        alternatives.append(rf"(?<![\w.-]){name}/")
        alternatives.append(rf"""["']{name}["']""")
    return re.compile("|".join(alternatives))


def _is_test_file(path: Path) -> bool:
    return any(fnmatch(path.name, pattern) for pattern in _TEST_FILE_PATTERNS)


def _imported_names(source: str) -> set[str] | None:
    """Every name an import statement mentions, or ``None`` if the file does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
            names.update(alias.name for alias in node.names)
    return names


def select_docs_tests(root: Path) -> list[str]:
    """Return the test files under ``root/tests`` that read documentation, as sorted POSIX paths."""
    pattern = doc_reference_pattern()
    reading: set[Path] = set()
    imports: dict[Path, set[str]] = {}
    for path in sorted((root / "tests").rglob("*.py")):
        if path.name == "conftest.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        names = _imported_names(source)
        if names is None or pattern.search(source):
            # A file that does not parse cannot be ruled out, so it is selected.
            reading.add(path)
        else:
            imports[path] = names

    # A test can read documentation through a helper, and that helper through
    # another one, so follow imports until nothing new turns up.
    while True:
        # Collected test modules can also supply helpers. A package's reader
        # lives in __init__.py but is imported by the package directory's name.
        helper_names = {path.stem for path in reading}
        helper_names.update(path.parent.name for path in reading if path.name == "__init__.py")
        found = {path for path, names in imports.items() if names & helper_names}
        if not found:
            break
        reading |= found
        for path in found:
            del imports[path]

    return sorted(path.relative_to(root).as_posix() for path in reading if _is_test_file(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root to inspect")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the selected test files, one per line."""
    args = _parser().parse_args(argv)
    selected = select_docs_tests(args.root)
    if not selected:
        # pytest with no paths runs the whole suite, so an empty answer must stop the lane.
        print("no documentation tests found; refusing to run an empty selection", file=sys.stderr)
        return 1
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
