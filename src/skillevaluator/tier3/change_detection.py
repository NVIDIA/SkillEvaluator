# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a skill change needs a fresh Tier 3 live evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SKILL_MANIFEST = "SKILL.md"
TIER3_EVIDENCE_FILES = ("skill-card.md", "BENCHMARK.md")


@dataclass(frozen=True)
class Tier3RunDecision:
    """The fail-closed Tier 3 decision for one current/baseline skill pair."""

    should_run: bool
    reason_code: str
    evidence_file: str | None = None

    @property
    def should_skip(self) -> bool:
        """Return whether a fresh Tier 3 live evaluation may be skipped."""
        return not self.should_run

    def to_dict(self) -> dict[str, str | bool | None]:
        """Serialize the decision for CI logs and generated reports."""
        return {
            "should_run": self.should_run,
            "reason_code": self.reason_code,
            "evidence_file": self.evidence_file,
        }


def _skill_root(path: Path) -> Path:
    return path.parent if path.name == SKILL_MANIFEST else path


def _read_manifest(skill_root: Path) -> bytes | None:
    try:
        return (skill_root / SKILL_MANIFEST).read_bytes()
    except OSError:
        return None


def _split_frontmatter(content: bytes) -> tuple[bytes, bytes] | None:
    """Return frontmatter YAML and Markdown body without normalizing either."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip(b"\r\n") != b"---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip(b"\r\n") == b"---":
            return b"".join(lines[1:index]), b"".join(lines[index + 1 :])
    return None


def _frontmatter_mapping(frontmatter: bytes) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(frontmatter.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _existing_evidence(skill_root: Path) -> str | None:
    for filename in TIER3_EVIDENCE_FILES:
        if (skill_root / filename).is_file():
            return filename
    return None


def tier3_run_decision(skill_path: Path, previous_skill_path: Path) -> Tier3RunDecision:
    """Return whether ``skill_path`` needs a new Tier 3 live evaluation.

    A fresh run is skipped only when the Markdown body and all frontmatter
    fields except ``metadata`` are unchanged from ``previous_skill_path``.  A
    generated ``skill-card.md`` or ``BENCHMARK.md`` must already exist with the
    previous skill.  Any unreadable path, invalid frontmatter, missing evidence,
    or behavioral change fails closed and requires Tier 3.
    """
    skill_root = _skill_root(skill_path)
    previous_root = _skill_root(previous_skill_path)
    evidence_file = _existing_evidence(previous_root)
    if evidence_file is None:
        return Tier3RunDecision(True, "previous_tier3_evidence_missing")

    current_manifest = _read_manifest(skill_root)
    previous_manifest = _read_manifest(previous_root)
    if current_manifest is None or previous_manifest is None:
        return Tier3RunDecision(True, "skill_manifest_unreadable", evidence_file)

    current_parts = _split_frontmatter(current_manifest)
    previous_parts = _split_frontmatter(previous_manifest)
    if current_parts is None or previous_parts is None:
        return Tier3RunDecision(True, "skill_frontmatter_invalid", evidence_file)

    current_frontmatter, current_body = current_parts
    previous_frontmatter, previous_body = previous_parts
    if current_body != previous_body:
        return Tier3RunDecision(True, "skill_body_changed", evidence_file)

    current_data = _frontmatter_mapping(current_frontmatter)
    previous_data = _frontmatter_mapping(previous_frontmatter)
    if current_data is None or previous_data is None:
        return Tier3RunDecision(True, "skill_frontmatter_invalid", evidence_file)

    current_non_metadata = {key: value for key, value in current_data.items() if key != "metadata"}
    previous_non_metadata = {key: value for key, value in previous_data.items() if key != "metadata"}
    if current_non_metadata != previous_non_metadata:
        return Tier3RunDecision(True, "skill_frontmatter_changed", evidence_file)

    if current_data.get("metadata") != previous_data.get("metadata"):
        return Tier3RunDecision(False, "metadata_only_change", evidence_file)
    return Tier3RunDecision(False, "skill_unchanged", evidence_file)


__all__ = ["Tier3RunDecision", "tier3_run_decision"]
