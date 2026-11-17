#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify pull-request changes for fail-closed CI routing."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DOC_PREFIXES = (b"docs/",)
# The root changelog and readme are documentation too: nearly every docs pull
# request adds a changelog line, and that line used to send it down the full
# lane. The docs lane runs the tests that read these files, so a broken entry is
# still caught on the pull request.
DOC_FILES = frozenset({b"fern/docs.yml", b"CHANGELOG.md", b"README.md"})
KNOWN_STATUSES = frozenset(b"ACDMRTUXB")

# The Tier 3 platform jobs are the macOS sandbox tests and the native Windows
# local-mode check. They exist for operating-system behaviour, and that code
# lives under these source paths. The Linux jobs run the same test files, so a
# change anywhere else can only break those tests in a way Linux already shows.
SOURCE_ROOT = b"src/skillevaluator"
TESTS_ROOT = b"tests"
TIER3_PLATFORM_SOURCE = (b"src/skillevaluator/tier3", b"src/skillevaluator/utils")
# The test files the two jobs run. tests/test_ci_workflows.py checks this list
# against the jobs' commands in ci.yml.
TIER3_PLATFORM_TESTS = frozenset(
    {
        b"tests/embedding/test_catalog.py",
        b"tests/test_harbor_local_mode.py",
        b"tests/test_harbor_local_sandbox.py",
        b"tests/test_harbor_runtime_preflight.py",
        b"tests/test_local_mode_contract.py",
        b"tests/test_tier3_progress.py",
        b"tests/test_tier3_result_display.py",
        b"tests/validators/test_rubric_eval.py",
    }
)
# Shared by every test: fixtures the jobs' tests read, and anything pytest loads for them.
TIER3_PLATFORM_TEST_SUPPORT = b"tests/fixtures"


def _is_documentation(path: bytes) -> bool:
    return path.startswith(DOC_PREFIXES) or path in DOC_FILES


def is_docs_only(paths: Sequence[bytes]) -> bool:
    """Return whether every changed path is documentation the docs lane covers."""
    return bool(paths) and all(_is_documentation(path) for path in paths)


def _is_under(path: bytes, root: bytes) -> bool:
    """Whether ``path`` is ``root`` or inside it. A directory swapped for a file or link reports its bare name."""
    return path == root or path.startswith(root + b"/")


def _is_test_module(path: bytes) -> bool:
    name = path.rsplit(b"/", 1)[-1]
    return (name.startswith(b"test_") and name.endswith(b".py")) or name.endswith(b"_test.py")


def _touches_tier3_platform(path: bytes) -> bool:
    if _is_documentation(path):
        return False
    if path != SOURCE_ROOT and _is_under(path, SOURCE_ROOT):
        # ``tier3`` also matches top-level modules such as ``tier3_environments.py``.
        return any(path.startswith(prefix) for prefix in TIER3_PLATFORM_SOURCE)
    if path != TESTS_ROOT and _is_under(path, TESTS_ROOT):
        if _is_under(path, TIER3_PLATFORM_TEST_SUPPORT):
            return True
        if _is_test_module(path):
            return path in TIER3_PLATFORM_TESTS
        # A conftest, a helper module or other test data: it may be loaded by any test.
        return path.endswith(b".py")
    # Dependencies, CI configuration, scripts and anything else unrecognized.
    return True


def is_tier3_platform_related(paths: Sequence[bytes]) -> bool:
    """Return whether a change can alter the result of the Tier 3 platform jobs.

    The answer is ``True`` unless every changed path is known not to matter, so
    an unrecognized path, or no paths at all, runs the jobs.
    """
    return not paths or any(_touches_tier3_platform(path) for path in paths)


def parse_name_status_z(payload: bytes) -> list[bytes]:
    """Parse ``git diff --name-status -z`` without losing rename sources."""
    if not payload:
        return []
    if not payload.endswith(b"\0"):
        raise ValueError("incomplete Git status record: missing NUL terminator")

    fields = payload.split(b"\0")
    fields.pop()
    paths: list[bytes] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status or status[0] not in KNOWN_STATUSES:
            raise ValueError(f"unrecognized Git status record: {status!r}")

        code = status[:1]
        if code in {b"R", b"C"}:
            if len(status) == 1 or not status[1:].isdigit():
                raise ValueError(f"unrecognized Git status record: {status!r}")
            path_count = 2
        else:
            if len(status) != 1:
                raise ValueError(f"unrecognized Git status record: {status!r}")
            path_count = 1

        if index + path_count > len(fields):
            raise ValueError(f"incomplete Git status record: {status!r}")
        record_paths = fields[index : index + path_count]
        if any(not path for path in record_paths):
            raise ValueError(f"incomplete Git status record: {status!r}")
        paths.extend(record_paths)
        index += path_count
    return paths


def _validate_revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", value) is None:
        raise ValueError(f"invalid Git revision: {value!r}")
    return value


def changed_paths(repo: Path, base: str, head: str) -> list[bytes]:
    """Return every path changed from the merge base through ``head``."""
    base = _validate_revision(base)
    head = _validate_revision(head)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--find-copies-harder",
            f"{base}...{head}",
            "--",
        ],
        check=True,
        capture_output=True,
    )
    return parse_name_status_z(result.stdout)


def _write_result(docs_only: bool, tier3_platform: bool) -> None:
    lines = "".join(
        f"{name}={'true' if value else 'false'}\n"
        for name, value in (("docs_only", docs_only), ("tier3_platform", tier3_platform))
    )
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(lines)
    print(lines, end="")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git repository to inspect")
    parser.add_argument("--base", required=True, help="pull-request base commit SHA")
    parser.add_argument("--head", required=True, help="pull-request head commit SHA")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Classify one pull-request diff and emit a GitHub Actions output."""
    args = _parser().parse_args(argv)
    try:
        paths = changed_paths(args.repo, args.base, args.head)
        if not paths:
            raise ValueError("no changed paths found")
        docs_only = is_docs_only(paths)
        tier3_platform = is_tier3_platform_related(paths)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"change classification failed; falling back to full CI: {error}", file=sys.stderr)
        docs_only = False
        tier3_platform = True

    _write_result(docs_only, tier3_platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
