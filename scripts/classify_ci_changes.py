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

DOC_PREFIXES = (b"docs/", b"fern/")
KNOWN_STATUSES = frozenset(b"ACDMRTUXB")


def is_docs_only(paths: Sequence[bytes]) -> bool:
    """Return whether every changed path belongs to published documentation."""
    return bool(paths) and all(path.startswith(DOC_PREFIXES) for path in paths)


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


def _write_result(docs_only: bool) -> None:
    line = f"docs_only={'true' if docs_only else 'false'}"
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"{line}\n")
    print(line)


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
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"change classification failed; falling back to full CI: {error}", file=sys.stderr)
        docs_only = False

    _write_result(docs_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
