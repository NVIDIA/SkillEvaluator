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
from collections.abc import Iterable
from typing import Final

# Forge limits: GitHub owners are <=39 characters and repository names <=100.
_SOURCE_REPOSITORY: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,38}/[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
# A Git object id is a full SHA-1 (40 hex) or SHA-256 (64 hex) name. A short
# prefix is ambiguous -- it can grow a second match as the tree grows -- so it
# must not render as the immutable revision a reader is asked to trust.
_SOURCE_COMMIT: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
# A digest names its algorithm, so the algorithm half is an allowlist rather
# than a charset ("totally-fake:0000..." must not read as a canonical digest),
# and each algorithm admits only its own exact width, so "sha512:" followed by
# 32 hex characters cannot pass as a canonical sha512 digest.
_SOURCE_CONTENT_DIGEST: Final = re.compile(r"^(?:sha256:[0-9a-f]{64}|sha384:[0-9a-f]{96}|sha512:[0-9a-f]{128})$")
# An OCI reference is bounded component by component rather than as a whole.
# One cap over the whole string counts the ``@sha256:`` suffix against the
# repository path, which silently discards an ordinary name pinned by digest,
# and the OCI grammar already bounds each component on its own terms. The bound
# is that grammar's RepositoryNameTotalLengthMax, which measures the path once
# the registry host has been split off it, so a long host name cannot spend the
# budget a path is entitled to.
_CONTAINER_PATH_MAX: Final = 255
# A registry host label. Host names are matched in either case because DNS is
# case-insensitive, unlike the path, which the grammar admits in lower case only.
_CONTAINER_HOST_LABEL: Final = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
# name: an optional registry host followed by the slash-separated path; path:
# that path on its own, which is what the length bound above measures; tag: an
# optional mutable label of at most 128 characters; digest: the allowlist and
# exact widths used for the content digest above, so an under-length or
# invented algorithm cannot pass.
#
# A first component is a registry host only where the reference grammar says so:
# it is ``localhost``, it carries a dot, or it carries a port. Anything else
# begins the path and is held to the path's rules. Reading any single first
# label as a host instead made the grammar asymmetric, accepting
# ``NVIDIA/skillevaluator@sha256:...`` while refusing the same repository under
# a registry, ``ghcr.io/NVIDIA/skillevaluator@sha256:...``; both are uppercase
# path components, so both are refused.
_CONTAINER_REFERENCE_PATTERN: Final = (
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
_CONTAINER_REFERENCE: Final = re.compile(_CONTAINER_REFERENCE_PATTERN)


def is_container_revision(value: str) -> bool:
    """Return whether the value names the evaluator build that produced a card.

    The path length is checked in Python rather than in the pattern because a
    regex cannot bound one alternation-heavy group without either duplicating
    the grammar or capping the reference as a whole, which is the bug this
    replaces.
    """
    if _SOURCE_COMMIT.fullmatch(value):
        # An evaluator run from a source checkout has no image to name, so its
        # own implementation revision stands in, as the publication gate has
        # always allowed for a published PASS.
        return True
    reference = _CONTAINER_REFERENCE.fullmatch(value)
    if reference is None or len(reference["path"]) > _CONTAINER_PATH_MAX:
        return False
    # A bare name identifies a repository, not a revision: it cannot tell a
    # reader which build ran, so it is refused here rather than published as an
    # identity that does not identify anything.
    return bool(reference["tag"] or reference["digest"])


# Each field carries the predicate that accepts its canonical shape, because
# the container reference needs a length rule the pattern cannot express.
EVALUATED_SOURCE_FIELDS: Final = (
    ("repository", _SOURCE_REPOSITORY.fullmatch),
    ("commit", _SOURCE_COMMIT.fullmatch),
    ("content_digest", _SOURCE_CONTENT_DIGEST.fullmatch),
    ("evaluator_container_revision", is_container_revision),
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
    for field_name, is_canonical in EVALUATED_SOURCE_FIELDS:
        raw = value.get(field_name)
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if field_name in _CASE_FOLDED:
            # Hex revisions are case-insensitive, so fold them to one spelling
            # rather than dropping an otherwise valid uppercase digest.
            candidate = candidate.lower()
        if candidate and is_canonical(candidate):
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


def merge_evaluated_sources(candidates: Iterable[object]) -> dict[str, str] | None:
    """Merge every populated carrier of the identity into one, or raise.

    A card can carry the identity in more than two places -- the Tier 3 payload,
    that payload's summary, and the metadata of any validation result -- and the
    order those are visited is incidental to how the run was scheduled. Reducing
    them by "first valid wins" would therefore let result ordering decide which
    source a published card claims to describe, which is precisely the
    unverifiable provenance this contract exists to prevent.

    Every carrier is normalized and folded together field by field instead, so a
    disagreement anywhere among them fails closed rather than being silently
    resolved by position.
    """
    merged: dict[str, str] = {}
    for candidate in candidates:
        normalized = normalized_evaluated_source(candidate)
        if not normalized:
            continue
        conflicts = sorted(field for field in normalized.keys() & merged.keys() if normalized[field] != merged[field])
        if conflicts:
            detail = ", ".join(f"{field}: {merged[field]!r} vs {normalized[field]!r}" for field in conflicts)
            raise EvaluatedSourceConflict(f"conflicting evaluated source identity ({detail})")
        merged.update(normalized)
    return merged or None


def evaluated_source_carriers(metadata: object) -> tuple[object, ...]:
    """Return every place one validation result can carry the identity.

    A result records the identity in its own metadata, and a Tier 3 result also
    carries the agent-eval payload, which the producer stamps at both its top
    level and inside its summary. Listing the three places once, here, keeps the
    card, the machine-readable report and the CLI reading the same set: a
    carrier honoured by one report and ignored by another is how a run ends up
    publishing two different answers about what it evaluated.

    A level that is missing or is not a mapping yields a ``None`` entry rather
    than being skipped, because ``merge_evaluated_sources`` already drops
    whatever does not validate and a caller should not have to know the shape of
    each nesting level. A metadata value that is not a mapping carries nothing,
    so it yields no entries at all.
    """
    if not isinstance(metadata, dict):
        return ()
    payload = metadata.get("agent_eval")
    payload = payload if isinstance(payload, dict) else {}
    summary = payload.get("summary")
    return (
        metadata.get("evaluated_source"),
        payload.get("evaluated_source"),
        summary.get("evaluated_source") if isinstance(summary, dict) else None,
    )


def recorded_evaluated_source(metadatas: Iterable[object]) -> dict[str, str] | None:
    """Return the single identity a run recorded, or raise if it recorded two.

    Every carrier of every result is folded, rather than one nested agent-eval
    payload being selected and the rest of them ignored. Selecting one would let
    result ordering decide what a report claims to describe: a second agent-eval
    result naming a different source tree would be shadowed by whichever result
    the run happened to produce first, and reversing the list would publish the
    other identity instead of failing closed.

    Conflicts raise ``EvaluatedSourceConflict``, so a run that cannot say what it
    evaluated publishes nothing rather than a guess.
    """
    return merge_evaluated_sources(
        carrier for metadata in metadatas for carrier in evaluated_source_carriers(metadata)
    )


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
    return merge_evaluated_sources((fallback, explicit))
