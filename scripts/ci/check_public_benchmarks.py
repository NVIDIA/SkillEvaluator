#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate generated BENCHMARK.md files before publishing them."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_MARKERS = (
    "# Skill Benchmark:",
    "Overall verdict:",
    "## Evaluation Metadata",
    "- Evaluation date:",
    "- Evaluator version:",
    "- Agents:",
    "- Tasks:",
    "- Dataset digest:",
    "- Attempts per task:",
    "- Environment:",
    "- Tier 3 evidence:",
    "## Results at a Glance",
    "## Tier Status",
    "## Freshness",
)

LINE_RULES = (
    (
        "retired product identity",
        re.compile(r"\b[a-z]*[\s_-]*skills[\s_-]*eval\b", flags=re.IGNORECASE),
    ),
    (
        "internal environment identity",
        re.compile(
            r"(?:^\s*-\s*Environment:\s*`?astra`?\s*$|\bastra[\s_-]+sandbox\b)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "internal validation profile",
        re.compile(
            r"^\s*-\s*(?:Skill\s+Evaluator\s+)?Profile:\s*`?(?:internal|external)`?\s*$",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "absolute macOS user path",
        re.compile(r"/Users/[^/\s`]+/"),
    ),
    (
        "absolute Linux home path",
        re.compile(r"/home/[^/\s`]+/"),
    ),
    (
        "absolute Windows user path",
        re.compile(r"[A-Za-z]:\\Users\\[^\\\s`]+\\"),
    ),
    (
        "legacy ambiguous uplift cell",
        re.compile(r"\b\d+%\s+\([+-]\d+%\)"),
    ),
    (
        "legacy Num score column",
        re.compile(r"\|\s*Dimension\s*\|\s*Num\s*\|", flags=re.IGNORECASE),
    ),
)

_AGENT_MODEL_STATES = re.compile(
    r"^[^,]+ \((?:`[^`]+`|model not recorded)\)"
    r"(?:,\s*[^,]+ \((?:`[^`]+`|model not recorded)\))*$",
    flags=re.IGNORECASE,
)
_OVERALL_PASS = re.compile(
    r"^\s*>\s*.*Overall verdict:\s*PASS\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
_METADATA_FIELD_RULES = (
    (
        "Evaluation date",
        re.compile(r"(?:\d{4}-\d{2}-\d{2}|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Evaluator version",
        re.compile(r"(?:`[^`\s][^`]*`|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Tasks",
        re.compile(r"(?:[1-9]\d*\s+evaluation tasks?(?:\s+\(.*\))?|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Dataset digest",
        re.compile(r"(?:`[^`\s][^`]*`(?:\s+\([^)]*\))?|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Attempts per task",
        re.compile(r"(?:[1-9]\d*|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Environment",
        re.compile(r"(?:`[^`\s][^`]*`|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Tier 3 evidence",
        re.compile(r"(?:required for publication|optional by policy)", flags=re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class Offender:
    path: Path
    line: int
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}"


def benchmark_files(roots: list[Path]) -> list[Path]:
    """Return unique BENCHMARK.md files below the requested roots."""
    found: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"input path does not exist: {resolved}")
        if resolved.is_file() and resolved.name == "BENCHMARK.md":
            found.add(resolved)
        elif resolved.is_dir():
            found.update(path.resolve() for path in resolved.rglob("BENCHMARK.md") if path.is_file())
    return sorted(found)


def scan_file(path: Path) -> list[Offender]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [Offender(path, 1, f"unreadable file ({type(error).__name__})")]

    offenders: list[Offender] = []

    for marker in REQUIRED_MARKERS:
        if marker not in text:
            offenders.append(Offender(path, 1, f"missing required section: {marker}"))

    for line_number, line in enumerate(text.splitlines(), 1):
        for reason, pattern in LINE_RULES:
            if pattern.search(line):
                offenders.append(Offender(path, line_number, reason))

    _check_metadata_semantics(path, text, offenders)
    _check_verdict_tier_consistency(path, text, offenders)
    return offenders


def _check_metadata_semantics(path: Path, text: str, offenders: list[Offender]) -> None:
    metadata_lines = _metadata_section_lines(text)
    if not metadata_lines:
        return

    for field, pattern in _METADATA_FIELD_RULES:
        matches = _metadata_field_matches(metadata_lines, field)
        marker = f"- {field}:"
        if not matches:
            offenders.append(Offender(path, metadata_lines[0][0], f"missing metadata field: {marker}"))
            continue
        line_number, value = matches[0]
        if len(matches) > 1 or not pattern.fullmatch(value):
            offenders.append(Offender(path, line_number, f"invalid metadata field: {marker}"))

    agent_matches = _metadata_field_matches(metadata_lines, "Agents")
    if not agent_matches:
        offenders.append(Offender(path, metadata_lines[0][0], "missing metadata field: - Agents:"))
        return
    line_number, value = agent_matches[0]
    lowered = value.lower()
    if len(agent_matches) > 1:
        offenders.append(Offender(path, line_number, "agent model identity not recorded"))
    elif lowered.startswith("not recorded"):
        return
    elif lowered.startswith("requested but not run"):
        if "model not recorded" not in lowered:
            offenders.append(Offender(path, line_number, "agent model identity not recorded"))
    elif not _AGENT_MODEL_STATES.fullmatch(value):
        offenders.append(Offender(path, line_number, "agent model identity not recorded"))


def _metadata_section_lines(text: str) -> list[tuple[int, str]]:
    """Return line-numbered content from the first Evaluation Metadata section."""
    lines = text.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"\s*##\s+Evaluation Metadata\s*", line, flags=re.IGNORECASE)
        ),
        None,
    )
    if start is None:
        return []

    section: list[tuple[int, str]] = []
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if re.match(r"^\s*##\s+", line):
            break
        section.append((index + 1, line))
    return section


def _metadata_field_matches(
    metadata_lines: list[tuple[int, str]],
    field: str,
) -> list[tuple[int, str]]:
    pattern = re.compile(rf"^\s*-\s*{re.escape(field)}:\s*(?P<value>.*)$", flags=re.IGNORECASE)
    return [
        (line_number, match.group("value").strip())
        for line_number, line in metadata_lines
        if (match := pattern.fullmatch(line))
    ]


def _metadata_field_value(text: str, field: str) -> str | None:
    matches = _metadata_field_matches(_metadata_section_lines(text), field)
    return matches[0][1] if len(matches) == 1 else None


def _tier3_status(text: str) -> tuple[int, str] | None:
    for line_number, line in enumerate(text.splitlines(), 1):
        if not re.match(r"^\|\s*Tier\s*3\s*\|", line, flags=re.IGNORECASE):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            return line_number, ""
        status = re.sub(r"[*_`]", "", cells[2]).strip().upper()
        return line_number, status
    return None


def _check_verdict_tier_consistency(path: Path, text: str, offenders: list[Offender]) -> None:
    tier3_row = _tier3_status(text)
    if tier3_row is None:
        offenders.append(Offender(path, 1, "missing Tier 3 status row"))
        return
    if not _OVERALL_PASS.search(text):
        return

    line_number, tier3_status = tier3_row
    tier3_complete = tier3_status == "PASS"
    tier3_optional = (_metadata_field_value(text, "Tier 3 evidence") or "").lower() == "optional by policy"
    if not tier3_complete and not tier3_optional:
        offenders.append(Offender(path, line_number, "publication PASS without completed Tier 3 evidence"))


def find_offenders(roots: list[Path]) -> tuple[list[Path], list[Offender]]:
    files: set[Path] = set()
    offenders: list[Offender] = []
    for root in roots:
        try:
            files.update(benchmark_files([root]))
        except FileNotFoundError:
            offenders.append(Offender(root.expanduser().resolve(), 1, "input path does not exist"))

    sorted_files = sorted(files)
    offenders.extend(offender for path in sorted_files for offender in scan_file(path))
    return sorted_files, offenders


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="BENCHMARK.md file or directory tree to scan (default: current directory)",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Fail when no BENCHMARK.md files are found",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    files, offenders = find_offenders(args.paths)

    if offenders:
        print("Public benchmark scan FAILED:")
        for offender in offenders:
            print(f"  {offender}")
        return 1
    if args.require_files and not files:
        print("Public benchmark scan FAILED: no BENCHMARK.md files found.")
        return 1

    print(f"Public benchmark scan passed ({len(files)} file(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
