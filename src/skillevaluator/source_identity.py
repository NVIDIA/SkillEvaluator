# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluated-source identity carried from the orchestration input.

A published benchmark card has to say which source tree was evaluated, separately
from the evaluator build that evaluated it. That identity is supplied by the
orchestration input and carried through the evaluation-result contract unchanged;
it is never inferred from repository state while rendering, because the tree that
renders a card is the evaluator checkout rather than the evaluated skill's source.

The field patterns live here, outside both ``evaluation`` and ``reporting``, so
the producer, the renderer and the publication gate all validate against exactly
one definition of the identity.

Validation is what makes the identity safe to publish, so the card renders these
values verbatim rather than escaping them. Escaping would corrupt them: Markdown
inline escaping rewrites ``_`` to ``\\_`` and ``@`` to ``&#64;``, which turns
``org/nv_examples`` and ``ghcr.io/x@sha256:...`` into strings that no longer name
the thing they identify. Every character below is inert inside a Markdown code
span -- backtick, backslash, angle brackets and whitespace are all excluded -- so
a validated value cannot break out of the span or inject markup.
"""

from __future__ import annotations

import re
from typing import Final

# Forge limits: GitHub owners are <=39 characters and repository names <=100.
_SOURCE_REPOSITORY: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,38}/[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_SOURCE_COMMIT: Final = re.compile(r"^[0-9a-f]{7,64}$")
# A digest names its algorithm, so the algorithm half is an allowlist rather
# than a charset: "totally-fake:0000..." must not read as a canonical digest.
_SOURCE_CONTENT_DIGEST: Final = re.compile(r"^(?:sha256|sha384|sha512):[0-9a-f]{32,128}$")
# Admits a full OCI reference so a container revision can be pinned by digest.
_EVALUATOR_CONTAINER_REVISION: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")

EVALUATED_SOURCE_FIELDS: Final = (
    ("repository", _SOURCE_REPOSITORY),
    ("commit", _SOURCE_COMMIT),
    ("content_digest", _SOURCE_CONTENT_DIGEST),
    ("evaluator_container_revision", _EVALUATOR_CONTAINER_REVISION),
)

_CASE_FOLDED: Final = frozenset({"commit", "content_digest"})


def normalized_evaluated_source(value: object) -> dict[str, str] | None:
    """Return the validated evaluated-source identity, or ``None``.

    Each field is accepted only in its canonical shape, so a published card can
    never carry an unverified or Markdown-injecting provenance value. A field
    that does not validate is dropped rather than guessed at, which keeps the
    card honest about what the orchestration input actually recorded.
    """
    if not isinstance(value, dict):
        return None

    normalized: dict[str, str] = {}
    for field_name, pattern in EVALUATED_SOURCE_FIELDS:
        raw = value.get(field_name)
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if field_name in _CASE_FOLDED:
            # Hex revisions are case-insensitive, so fold them to one spelling
            # rather than dropping an otherwise valid uppercase digest.
            candidate = candidate.lower()
        if candidate and pattern.fullmatch(candidate):
            normalized[field_name] = candidate
    return normalized or None


def evaluated_source_revision(source: dict[str, str] | None) -> str:
    """Return the immutable revision of the evaluated source.

    Issue #72 accepts either an evaluated source commit SHA *or* a canonical
    digest of the evaluated skill content, so a run with no upstream commit
    still records an immutable identity. The card labels this ``Evaluated
    source revision`` rather than ``commit`` because it may legitimately be
    either.
    """
    if not source:
        return ""
    return str(source.get("commit") or source.get("content_digest") or "")


class EvaluatedSourceConflict(ValueError):
    """Two orchestration inputs disagree about what was evaluated."""


def resolve_evaluated_source(
    explicit: object,
    fallback: object,
) -> dict[str, str] | None:
    """Merge the two orchestration channels into one identity.

    Fields are resolved individually rather than whole-dict, so a mistyped
    field in one channel cannot silently discard a good identity in the other.

    Publication has to fail closed when the recorded provenance "conflicts with
    the orchestration input", so a field the two channels both supply and
    disagree about raises rather than being resolved by precedence: a card that
    quietly picked one of two contradictory source revisions would be exactly
    the unverifiable provenance this contract exists to prevent.
    """
    primary = normalized_evaluated_source(explicit) or {}
    secondary = normalized_evaluated_source(fallback) or {}

    conflicts = sorted(field for field in primary.keys() & secondary.keys() if primary[field] != secondary[field])
    if conflicts:
        detail = ", ".join(f"{field}: {secondary[field]!r} vs {primary[field]!r}" for field in conflicts)
        raise EvaluatedSourceConflict(f"conflicting evaluated source identity ({detail})")

    merged = {**secondary, **primary}
    return merged or None
