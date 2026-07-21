# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic normalization for public plugin dependency references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DuplicateRefGroup:
    """References that identify the same plugin dependency."""

    canonical_id: str
    occurrences: list[Any]


def normalize_ref(ref: Any) -> str | None:
    """Normalize string and selector refs without fetching remote content."""
    if isinstance(ref, str):
        value = ref.strip()
        if not value:
            return None
        segments = [segment.strip() for segment in value.split("::")]
        if len(segments) == 4 and all(segments):
            source, repo, dependency_type, name = segments
            return f"{source.lower()}::{repo.lower().removesuffix('.git')}::{dependency_type.lower()}::{name}"
        return value

    if isinstance(ref, dict):
        source, repo, ref_path = ref.get("source"), ref.get("repo"), ref.get("path")
    else:
        source, repo, ref_path = (
            getattr(ref, "source", None),
            getattr(ref, "repo", None),
            getattr(ref, "path", None),
        )
    if not all(isinstance(value, str) for value in (source, repo, ref_path)):
        return None
    source = source.strip().lower()
    repo = repo.strip().removesuffix(".git").lower()
    segments = [segment for segment in ref_path.strip().split("/") if segment]
    if not source or not repo or not segments:
        return None
    dependency_type = segments[0].lower()
    name = "/".join(segments[1:])
    return f"{source}::{repo}::{dependency_type}::{name}"


def find_duplicate_refs(refs: list[Any] | None) -> list[DuplicateRefGroup]:
    """Return duplicate groups in stable first-appearance order."""
    groups: dict[str, list[Any]] = {}
    order: list[str] = []
    for ref in refs or []:
        canonical = normalize_ref(ref)
        if canonical is None:
            continue
        if canonical not in groups:
            groups[canonical] = []
            order.append(canonical)
        groups[canonical].append(ref)
    return [DuplicateRefGroup(value, groups[value]) for value in order if len(groups[value]) > 1]
