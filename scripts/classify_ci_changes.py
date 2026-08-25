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
SKILL_FILENAME = b"SKILL.md"
TIER3_EVIDENCE_FILENAMES = (b"skill-card.md", b"BENCHMARK.md")


def is_docs_only(paths: Sequence[bytes]) -> bool:
    """Return whether every changed path belongs to published documentation."""
    return bool(paths) and all(path.startswith(DOC_PREFIXES) for path in paths)


def _is_skill_file(path: bytes) -> bool:
    return path == SKILL_FILENAME or path.endswith(b"/" + SKILL_FILENAME)


def _split_frontmatter(content: bytes) -> tuple[bytes, bytes] | None:
    """Split a Markdown file into YAML frontmatter and body, if present.

    This intentionally validates only the delimiter shape.  The Tier 1 schema
    validator remains responsible for validating the YAML itself; CI routing
    must stay dependency-free because it runs before the project is installed.
    """
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip(b"\r\n") != b"---":
        return None

    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip(b"\r\n") in {b"---", b"..."}:
            return b"".join(lines[: index + 1]), b"".join(lines[index + 1 :])
    return None


def _metadata_section(frontmatter: bytes) -> tuple[bytes, bytes, bytes] | None:
    """Return the immutable prefix/suffix around a top-level ``metadata`` key.

    The PR classifier must run before dependencies are installed, so it keeps a
    deliberately narrow YAML shape instead of parsing arbitrary YAML.  Anything
    outside this conventional top-level metadata block is treated as behavioral
    and therefore falls back to a full Tier 3 run.
    """
    lines = frontmatter.splitlines(keepends=True)
    for index, line in enumerate(lines[1:-1], start=1):
        if not line.startswith(b"metadata:"):
            continue
        value = line[len(b"metadata:") :].strip()
        end = index + 1
        if not value or value.startswith(b"#"):
            while end < len(lines) - 1:
                candidate = lines[end]
                if candidate.startswith((b" ", b"\t", b"\r", b"\n", b"#")):
                    end += 1
                    continue
                break
        return b"".join(lines[:index]), b"".join(lines[index:end]), b"".join(lines[end:])
    return None


def _is_metadata_only_change(previous: bytes, current: bytes) -> bool:
    previous_parts = _split_frontmatter(previous)
    current_parts = _split_frontmatter(current)
    if previous_parts is None or current_parts is None:
        return False
    previous_frontmatter, previous_body = previous_parts
    current_frontmatter, current_body = current_parts
    if previous_body != current_body:
        return False

    previous_metadata = _metadata_section(previous_frontmatter)
    current_metadata = _metadata_section(current_frontmatter)
    if previous_metadata is None and current_metadata is None:
        return False
    if previous_metadata is None:
        current_prefix, current_block, current_suffix = current_metadata
        return current_prefix + current_suffix == previous_frontmatter and bool(current_block)
    if current_metadata is None:
        previous_prefix, previous_block, previous_suffix = previous_metadata
        return previous_prefix + previous_suffix == current_frontmatter and bool(previous_block)
    previous_prefix, previous_block, previous_suffix = previous_metadata
    current_prefix, current_block, current_suffix = current_metadata
    return (
        previous_prefix == current_prefix
        and previous_suffix == current_suffix
        and previous_block != current_block
    )


def _merge_base(repo: Path, base: str, head: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", base, head],
        check=True,
        capture_output=True,
    )
    merge_base = result.stdout.strip().decode("ascii")
    return _validate_revision(merge_base)


def _revision_file(repo: Path, revision: str, path: bytes) -> bytes:
    """Read ``path`` from a Git revision without touching the worktree."""
    path_text = os.fsdecode(path)
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path_text}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _has_tier3_evidence(repo: Path, revision: str, skill_path: bytes) -> bool:
    skill_parent = skill_path.rsplit(b"/", 1)[0] if b"/" in skill_path else b""
    for filename in TIER3_EVIDENCE_FILENAMES:
        evidence_path = filename if not skill_parent else skill_parent + b"/" + filename
        try:
            _revision_file(repo, revision, evidence_path)
        except subprocess.CalledProcessError:
            continue
        return True
    return False


def is_metadata_only(repo: Path, base: str, head: str, paths: Sequence[bytes]) -> bool:
    """Return whether a diff changes only existing skills' frontmatter.

    Tier 3 is expensive and need not run after metadata-only edits, but this is
    safe only when the affected skill already has a generated card or benchmark
    from an earlier evaluation.  Evidence is read from the merge-base revision
    so a pull request cannot qualify itself by adding a new artifact.
    """
    if not paths or not all(_is_skill_file(path) for path in paths):
        return False

    merge_base = _merge_base(repo, base, head)
    for path in paths:
        if not _has_tier3_evidence(repo, merge_base, path):
            return False

        try:
            previous = _revision_file(repo, merge_base, path)
            current = _revision_file(repo, head, path)
        except subprocess.CalledProcessError:
            return False
        if not _is_metadata_only_change(previous, current):
            return False
    return True


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


def _write_result(docs_only: bool, metadata_only: bool) -> None:
    lines = (
        f"docs_only={'true' if docs_only else 'false'}",
        f"metadata_only={'true' if metadata_only else 'false'}",
    )
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.writelines(f"{line}\n" for line in lines)
    print(*lines, sep="\n")


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
        metadata_only = is_metadata_only(args.repo, args.base, args.head, paths)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"change classification failed; falling back to full CI: {error}", file=sys.stderr)
        docs_only = False
        metadata_only = False

    _write_result(docs_only, metadata_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
