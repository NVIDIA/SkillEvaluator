#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lint committed BENCHMARK.md fixtures for structure and known leak patterns.

A clean scan means no configured pattern matched; it does not prove that a card
is safe to publish. Publication still requires the repository's broader review
and security controls.
"""

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
        "validation profile metadata",
        re.compile(
            r"^\s*-\s*(?:Skill\s+Evaluator\s+)?Profile\s*:",
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

_AGENT_MODEL_STATE = re.compile(
    r"^[^,]+ \((?:`[^`,]+`|model not recorded)\)$",
    flags=re.IGNORECASE,
)
_RECORDED_AGENT_MODEL_STATE = re.compile(r"^[^,]+ \(`[^`,]+`\)$")
# A verdict line states a verdict and nothing else. Every card's policy section
# restates the rule as a bullet ("- Overall verdict: PASS only when every
# configured dimension passes ..."), so a list item is never a verdict, and a
# stated verdict either ends its line or introduces the wording that explains
# it. Both detectors below are built from this one tail, so they cannot
# disagree about what a verdict line is; only the blockquote marker differs.
_VERDICT_LINE_START = r"^(?!\s*(?:[-*+]|\d+[.)])\s)\s*"
_VERDICT_PASS_TAIL = r"\s*.*Overall verdict:\s*PASS\b(?:\*\*)?\s*(?:$|[\u2014:\-(])"
_OVERALL_PASS = re.compile(
    rf"{_VERDICT_LINE_START}>{_VERDICT_PASS_TAIL}",
    flags=re.IGNORECASE | re.MULTILINE,
)
# Each field carries the predicate that accepts its recorded value, because the
# container reference needs a length rule no single pattern can express.
_METADATA_FIELD_RULES = (
    (
        "Evaluation date",
        re.compile(r"(?:\d{4}-\d{2}-\d{2}|not recorded\b.*)", flags=re.IGNORECASE).fullmatch,
    ),
    (
        "Evaluator version",
        re.compile(r"(?:`[^`\s][^`]*`|not recorded\b.*)", flags=re.IGNORECASE).fullmatch,
    ),
    (
        "Tasks",
        re.compile(r"(?:[1-9]\d*\s+evaluation tasks?(?:\s+\(.*\))?|not recorded\b.*)", flags=re.IGNORECASE).fullmatch,
    ),
    (
        "Dataset digest",
        re.compile(r"(?:`[^`\s][^`]*`(?:\s+\([^)]*\))?|not recorded\b.*)", flags=re.IGNORECASE).fullmatch,
    ),
    (
        "Attempts per task",
        re.compile(r"(?:[1-9]\d*|not recorded\b.*)", flags=re.IGNORECASE).fullmatch,
    ),
    (
        "Environment",
        re.compile(r"(?:`[^`\s][^`]*`|not recorded\b.*)", flags=re.IGNORECASE).fullmatch,
    ),
    (
        "Tier 3 evidence",
        re.compile(r"(?:required for publication|optional by policy)", flags=re.IGNORECASE).fullmatch,
    ),
)
_PASS_METADATA_FIELD_RULES = (
    ("Evaluation date", re.compile(r"\d{4}-\d{2}-\d{2}").fullmatch),
    ("Evaluator version", re.compile(r"`[^`\s][^`]*`").fullmatch),
    ("Tasks", re.compile(r"[1-9]\d*\s+evaluation tasks?(?:\s+\(.*\))?", flags=re.IGNORECASE).fullmatch),
    (
        "Dataset digest",
        re.compile(
            r"`sha256:[0-9a-f]{64}`\s+\(skill-evaluator-dataset-snapshot/1\)",
            flags=re.IGNORECASE,
        ).fullmatch,
    ),
    ("Attempts per task", re.compile(r"[1-9]\d*").fullmatch),
    ("Environment", re.compile(r"`[^`\s][^`]*`").fullmatch),
)


# Evaluated-source provenance is opt-in. A card published before the identity
# contract existed cannot carry these fields, and the orchestration side has to
# supply them first, so the default scan stays byte-compatible with the previous
# behaviour and CI enables --require-source-provenance once the pipeline does.
_SOURCE_PROVENANCE_MARKERS = (
    "- Evaluated source:",
    "- Evaluated source revision:",
    "- Evaluator container revision:",
)
_SOURCE_REPOSITORY_VALUE = r"`[A-Za-z0-9][A-Za-z0-9._-]{0,38}/[A-Za-z0-9][A-Za-z0-9._-]{0,99}`"
# Mirrors ``skillevaluator.source_identity``: a full Git object id (SHA-1 or
# SHA-256), or a digest whose width matches the algorithm it names. A short
# prefix and an under-length digest are both rejected as ambiguous.
_SOURCE_REVISION_VALUE = r"`(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64}|sha384:[0-9a-f]{96}|sha512:[0-9a-f]{128})`"
_NOT_RECORDED_VALUE = re.compile(r"not recorded\b.*")
# Mirrors ``skillevaluator.source_identity`` character for character, including
# the path length, so the two sides cannot drift: an OCI reference is bounded
# component by component rather than as a whole string, because one cap over the
# whole reference counts the ``@sha256:`` suffix against the repository path and
# silently discards an ordinary name pinned by digest. The bound measures the
# path once the registry host has been split off it, which is where the
# reference grammar's RepositoryNameTotalLengthMax applies.
_CONTAINER_PATH_MAX = 255
# A first component is a registry host only where the reference grammar says so:
# it is ``localhost``, it carries a dot, or it carries a port. Anything else
# begins the path, whose components are lower case, while a host label is
# matched in either case because DNS is case-insensitive.
_CONTAINER_HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_CONTAINER_REFERENCE_PATTERN = (
    r"(?P<name>"
    rf"(?:(?:localhost|{_CONTAINER_HOST_LABEL}(?:\.{_CONTAINER_HOST_LABEL})+)(?::[0-9]+)?/"
    rf"|{_CONTAINER_HOST_LABEL}:[0-9]+/)?"
    r"(?P<path>"
    r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*"
    r")"
    r")"
    r"(?::(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}))?"
    r"(?:@(?P<digest>sha256:[0-9a-f]{64}|sha384:[0-9a-f]{96}|sha512:[0-9a-f]{128}))?"
)
_CONTAINER_REFERENCE = re.compile(_CONTAINER_REFERENCE_PATTERN)
# Anchors included, so this is the same pattern text as ``_SOURCE_COMMIT`` and
# the drift test can compare the two strings rather than trusting the eye.
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _code_span_value(value: str) -> str | None:
    """Return what a metadata field renders inside its code span, or ``None``."""
    if len(value) > 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return None


def _valid_container_reference(reference: str, *, require_digest: bool) -> bool:
    if _GIT_OBJECT_ID.fullmatch(reference):
        # An evaluator run from a source checkout names its implementation
        # revision instead of an image, and that revision is already immutable.
        return True
    match = _CONTAINER_REFERENCE.fullmatch(reference)
    if match is None or len(match["path"]) > _CONTAINER_PATH_MAX:
        return False
    # A bare name identifies a repository rather than the build that ran, so it
    # never records a revision on its own.
    return bool(match["digest"]) if require_digest else bool(match["tag"] or match["digest"])


def _valid_container_revision(value: str) -> bool:
    """Accept a reference that names a revision, or the recorded absence."""
    if _NOT_RECORDED_VALUE.fullmatch(value):
        return True
    reference = _code_span_value(value)
    return reference is not None and _valid_container_reference(reference, require_digest=False)


_SOURCE_METADATA_FIELD_RULES = (
    ("Evaluated source", re.compile(rf"(?:{_SOURCE_REPOSITORY_VALUE}|not recorded\b.*)").fullmatch),
    ("Evaluated source revision", re.compile(rf"(?:{_SOURCE_REVISION_VALUE}|not recorded\b.*)").fullmatch),
    ("Evaluator container revision", _valid_container_revision),
)

# A published PASS must name the source it evaluated, whatever the Tier 3 policy
# says, so an "optional by policy" card cannot publish without the identity.
# A hand-authored backfill card may state its verdict without a blockquote, and
# the rollout runbook points this flag at exactly those trees, so the source
# check matches the verdict line either way. Making the marker optional is what
# makes the list-item exclusion load-bearing here: the policy section restates
# the rule as a bullet on every card, so without it an INCOMPLETE or FAIL card
# reads as a published PASS and is charged for provenance it never claimed.
_PUBLISHED_PASS = re.compile(
    rf"{_VERDICT_LINE_START}>?{_VERDICT_PASS_TAIL}",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _valid_pass_container_revision(value: str) -> bool:
    """Require the digest of the build that ran, on the same grammar as above.

    A mutable tag can be repointed to a different build after the card is
    published, which leaves the reader unable to recover what actually ran, so a
    PASS has to pin the evaluator by digest or by a full implementation revision.
    """
    reference = _code_span_value(value)
    return reference is not None and _valid_container_reference(reference, require_digest=True)


_PASS_SOURCE_PROVENANCE_RULES = (
    ("Evaluated source", re.compile(_SOURCE_REPOSITORY_VALUE).fullmatch),
    ("Evaluated source revision", re.compile(_SOURCE_REVISION_VALUE).fullmatch),
    ("Evaluator container revision", _valid_pass_container_revision),
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


def scan_file(path: Path, *, require_source_provenance: bool = False) -> list[Offender]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [Offender(path, 1, f"unreadable file ({type(error).__name__})")]

    offenders: list[Offender] = []

    markers = REQUIRED_MARKERS + (_SOURCE_PROVENANCE_MARKERS if require_source_provenance else ())
    for marker in markers:
        if marker not in text:
            offenders.append(Offender(path, 1, f"missing required section: {marker}"))

    for line_number, line in enumerate(text.splitlines(), 1):
        for reason, pattern in LINE_RULES:
            if pattern.search(line):
                offenders.append(Offender(path, line_number, reason))

    _check_metadata_semantics(path, text, offenders, require_source_provenance=require_source_provenance)
    _check_verdict_tier_consistency(path, text, offenders)
    if require_source_provenance and _PUBLISHED_PASS.search(text):
        # Independent of the Tier 3 row: a card publishing a PASS must name the
        # source it evaluated, including one skipped under an optional policy.
        _check_source_provenance(path, text, offenders)
    return offenders


def _check_metadata_semantics(
    path: Path,
    text: str,
    offenders: list[Offender],
    *,
    require_source_provenance: bool = False,
) -> None:
    metadata_lines = _metadata_section_lines(text)
    if not metadata_lines:
        return

    rules = _METADATA_FIELD_RULES
    if require_source_provenance:
        rules += _SOURCE_METADATA_FIELD_RULES
    for field, is_valid in rules:
        matches = _metadata_field_matches(metadata_lines, field)
        marker = f"- {field}:"
        if not matches:
            offenders.append(Offender(path, metadata_lines[0][0], f"missing metadata field: {marker}"))
            continue
        line_number, value = matches[0]
        if len(matches) > 1 or not is_valid(value):
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
    elif not _valid_agent_model_states(value):
        offenders.append(Offender(path, line_number, "agent model identity not recorded"))


def _valid_agent_model_states(value: str) -> bool:
    """Validate a comma-delimited agent list without a backtracking list regex."""
    agents = [agent.strip() for agent in value.split(",")]
    return bool(agents) and all(_AGENT_MODEL_STATE.fullmatch(agent) for agent in agents)


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
    if tier3_complete:
        _check_pass_provenance(path, text, offenders)
    if not tier3_complete and not tier3_optional:
        offenders.append(Offender(path, line_number, "publication PASS without completed Tier 3 evidence"))


def _check_source_provenance(path: Path, text: str, offenders: list[Offender]) -> None:
    """Reject a published PASS that does not record the source it evaluated."""
    metadata_lines = _metadata_section_lines(text)
    fallback_line = metadata_lines[0][0] if metadata_lines else 1
    for field, is_valid in _PASS_SOURCE_PROVENANCE_RULES:
        matches = _metadata_field_matches(metadata_lines, field)
        line_number = matches[0][0] if matches else fallback_line
        if len(matches) != 1 or not is_valid(matches[0][1]):
            offenders.append(Offender(path, line_number, f"publication PASS without recorded {field.lower()}"))


def _check_pass_provenance(path: Path, text: str, offenders: list[Offender]) -> None:
    """Reject PASS cards that replace required provenance with legacy placeholders."""
    metadata_lines = _metadata_section_lines(text)
    fallback_line = metadata_lines[0][0] if metadata_lines else 1
    for field, is_valid in _PASS_METADATA_FIELD_RULES:
        matches = _metadata_field_matches(metadata_lines, field)
        line_number = matches[0][0] if matches else fallback_line
        if len(matches) != 1 or not is_valid(matches[0][1]):
            offenders.append(Offender(path, line_number, f"publication PASS without recorded {field.lower()}"))

    agent_matches = _metadata_field_matches(metadata_lines, "Agents")
    line_number = agent_matches[0][0] if agent_matches else fallback_line
    if len(agent_matches) != 1:
        return
    value = agent_matches[0][1]
    if (value.lower().startswith("not recorded") or _valid_agent_model_states(value)) and not (
        _valid_recorded_agent_models(value)
    ):
        offenders.append(Offender(path, line_number, "publication PASS without recorded agent model identity"))


def _valid_recorded_agent_models(value: str) -> bool:
    agents = [agent.strip() for agent in value.split(",")]
    return bool(agents) and all(_RECORDED_AGENT_MODEL_STATE.fullmatch(agent) for agent in agents)


def find_offenders(
    roots: list[Path],
    *,
    require_source_provenance: bool = False,
) -> tuple[list[Path], list[Offender]]:
    files: set[Path] = set()
    offenders: list[Offender] = []
    for root in roots:
        try:
            files.update(benchmark_files([root]))
        except FileNotFoundError:
            offenders.append(Offender(root.expanduser().resolve(), 1, "input path does not exist"))

    sorted_files = sorted(files)
    offenders.extend(
        offender
        for path in sorted_files
        for offender in scan_file(path, require_source_provenance=require_source_provenance)
    )
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
    parser.add_argument(
        "--require-source-provenance",
        action="store_true",
        help="Fail PASS cards that do not record an evaluated source repository and revision",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    files, offenders = find_offenders(
        args.paths,
        require_source_provenance=args.require_source_provenance,
    )

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
