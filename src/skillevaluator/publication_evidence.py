# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned producer identity for publication-certifying validator results.

The marker records that a result passed through a built-in Tier 1 or Tier 2
command wrapper. It is intentionally a provenance contract for trusted local
or CI artifacts, not a cryptographic signature for hostile result files.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult


PUBLICATION_EVIDENCE_SCHEMA_VERSION = 1
_PUBLICATION_EVIDENCE_FIELDS = frozenset({"schema_version", "producer", "tier", "check_id"})
_RECOGNIZED_CHECKS_BY_TIER = {
    1: frozenset(
        {
            "schema",
            "version",
            "security",
            "pii",
            "license",
            "code-integrity",
            "dependency",
            "unicode",
            "quality",
            "lint",
            "rubric",
        }
    ),
    2: frozenset({"similarity", "context-optimization"}),
}
_PRODUCER_BY_TIER = {tier: f"skillevaluator.tier{tier}" for tier in _RECOGNIZED_CHECKS_BY_TIER}


@dataclass(frozen=True)
class PublicationEvidenceIdentity:
    """Canonical identity of one built-in publication evidence producer."""

    schema_version: int
    producer: str
    tier: int
    check_id: str


def publication_evidence_identity(value: object) -> PublicationEvidenceIdentity | None:
    """Parse a marker only when every field exactly matches the built-in contract."""
    if not isinstance(value, dict) or set(value) != _PUBLICATION_EVIDENCE_FIELDS:
        return None
    schema_version = value.get("schema_version")
    producer = value.get("producer")
    tier = value.get("tier")
    check_id = value.get("check_id")
    if (
        type(schema_version) is not int
        or schema_version != PUBLICATION_EVIDENCE_SCHEMA_VERSION
        or type(tier) is not int
        or tier not in _RECOGNIZED_CHECKS_BY_TIER
        or not isinstance(producer, str)
        or producer != _PRODUCER_BY_TIER[tier]
        or not isinstance(check_id, str)
        or check_id not in _RECOGNIZED_CHECKS_BY_TIER[tier]
    ):
        return None
    return PublicationEvidenceIdentity(schema_version, producer, tier, check_id)


def publication_evidence_dict(value: object) -> dict[str, object] | None:
    """Project an untrusted marker to the four canonical fields safe for reports."""
    identity = publication_evidence_identity(value)
    return asdict(identity) if identity is not None else None


def result_publication_evidence(result: ValidationResult) -> PublicationEvidenceIdentity | None:
    """Return one result's recognized built-in producer identity, if present."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if result.validator_name == "AGENT_EVAL" or isinstance(metadata.get("agent_eval"), dict):
        # A live Tier 3 result cannot double as static or semantic evidence,
        # even if an imported artifact carries a syntactically valid marker.
        return None
    return publication_evidence_identity(metadata.get("publication_evidence"))


def result_publication_evidence_dict(result: ValidationResult) -> dict[str, object] | None:
    """Return a fresh canonical producer marker safe for report serialization."""
    identity = result_publication_evidence(result)
    return asdict(identity) if identity is not None else None


def result_has_publication_evidence(result: ValidationResult, *, tier: int | None = None) -> bool:
    """Return whether a result came through a recognized built-in tier wrapper."""
    identity = result_publication_evidence(result)
    return bool(identity is not None and (tier is None or identity.tier == tier))


def stamp_publication_evidence(
    results: Iterable[ValidationResult],
    *,
    tier: int,
    check_id: str,
) -> None:
    """Stamp results from one trusted built-in command wrapper."""
    candidate = {
        "schema_version": PUBLICATION_EVIDENCE_SCHEMA_VERSION,
        "producer": _PRODUCER_BY_TIER.get(tier),
        "tier": tier,
        "check_id": check_id,
    }
    marker = publication_evidence_dict(candidate)
    if marker is None:
        raise ValueError(f"tier {tier} check {check_id!r} is not a recognized publication check")
    for result in results:
        if not isinstance(result.metadata, dict):
            result.metadata = {}
        result.metadata["publication_evidence"] = dict(marker)


__all__ = [
    "PUBLICATION_EVIDENCE_SCHEMA_VERSION",
    "PublicationEvidenceIdentity",
    "publication_evidence_dict",
    "publication_evidence_identity",
    "result_has_publication_evidence",
    "result_publication_evidence",
    "result_publication_evidence_dict",
    "stamp_publication_evidence",
]
