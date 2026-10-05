# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding registry for pre-computed similarity indexes.

Manages building, caching, and querying a collection of content
embeddings for pairwise duplicate detection.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from skillevaluator.constants import (
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_WORKFLOWS,
    PLUGIN_CATALOG_MAX_MEMBER_CHARS,
    PLUGIN_CATALOG_MAX_MEMBERS,
    PLUGIN_MANIFEST_RELATIVE_PATHS,
    SIMILARITY_CRITICAL_THRESHOLD,
    SIMILARITY_DEFAULT_MAX_ENTRIES,
    SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    SIMILARITY_HIGH_THRESHOLD,
    SIMILARITY_LOW_THRESHOLD,
    SIMILARITY_MAX_ENTRIES,
    SIMILARITY_MEDIUM_THRESHOLD,
)
from skillevaluator.embedding.client import (
    EmbeddingClient,
    SimilarityConfigError,
    normalize_embedding_vector,
    unit_vector_similarity,
    validate_embedding_vector,
    validate_similarity_threshold,
)
from skillevaluator.embedding.extractor import (
    MAX_MANIFEST_BYTES,
    ContentEntry,
    ExtractionBudget,
    discover_and_extract,
)
from skillevaluator.embedding.limits import validate_max_entries, validate_max_scalar_comparisons
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Severity
from skillevaluator.utils.path_security import canonicalize_trusted_root_alias
from skillevaluator.utils.secure_fs import (
    SecurePathError,
    read_bounded,
    stat_is_link_or_reparse,
    windows_final_path,
)
from skillevaluator.utils.structured_data import StructuredDataLimitError, preflight_json_structure
from skillevaluator.utils.tier2_paths import is_link_or_reparse, safe_path_label

logger = get_logger(__name__)

# Version 1 catalogs hold skill/rules/workflows entries only. Version 2 adds a
# ``plugins`` list for plugin entries. Skill-only catalogs are still written as
# version 1 so older readers keep loading them; both versions load here.
SKILL_CATALOG_SCHEMA_VERSION = 1
PLUGIN_CATALOG_SCHEMA_VERSION = 2
SUPPORTED_CATALOG_SCHEMA_VERSIONS = frozenset({SKILL_CATALOG_SCHEMA_VERSION, PLUGIN_CATALOG_SCHEMA_VERSION})
MAX_CATALOG_BYTES = 32 * 1024 * 1024
MAX_CATALOG_ENTRIES = SIMILARITY_MAX_ENTRIES
MAX_VECTOR_DIMENSION = 65_536
MAX_CATALOG_TEXT_LENGTH = 16_384
MAX_CATALOG_VECTOR_VALUES = 1_000_000
EMBEDDING_BATCH_SIZE = 64
MAX_DESCRIPTION_EMBEDDING_TEXT_CHARS = 16_384
MAX_FULL_BODY_EMBEDDING_TEXT_BYTES = MAX_MANIFEST_BYTES
MAX_SIMILARITY_MATCHES = 1_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ENDPOINT_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CATALOG_CONTENT_TYPES = {
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_WORKFLOWS,
}
_CATALOG_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "provider",
        "model",
        "mode",
        "endpoint_fingerprint",
        "vector_dimension",
        "created_at",
        "entries",
    }
)
_CATALOG_ENTRY_FIELDS = frozenset(
    {
        "id",
        "name",
        "description",
        "path",
        "content_type",
        "content_fingerprint",
        "embedding",
    }
)
_CATALOG_ROOT_FIELDS_V2 = _CATALOG_ROOT_FIELDS | {"plugins"}
_CATALOG_PLUGIN_FIELDS = frozenset(
    {
        "id",
        "name",
        "description",
        "path",
        "manifest",
        "source_fingerprint",
        "members",
        "embedding",
    }
)
MAX_CATALOG_PLUGIN_MEMBERS = PLUGIN_CATALOG_MAX_MEMBERS
MAX_CATALOG_PLUGIN_MEMBER_CHARS = PLUGIN_CATALOG_MAX_MEMBER_CHARS
MAX_CATALOG_MEMBER_NAMES = 262_144
_PLUGIN_CATALOG_MANIFESTS = frozenset(PLUGIN_MANIFEST_RELATIVE_PATHS)


@dataclass
class RegistryEntry:
    """A single item in the embedding index."""

    name: str
    description: str
    path: str
    content_type: str
    embedding: list[float] = field(default_factory=list)
    entry_id: str = ""
    content_fingerprint: str = ""


@dataclass
class PluginRegistryEntry:
    """One plugin in a version 2 local catalog.

    ``source_fingerprint`` is the SHA-256 of the plugin manifest bytes: a
    credential-free source identity used to exclude the plugin under test.
    ``members`` holds casefolded member skill names for overlap scoring.
    """

    name: str
    description: str
    path: str
    manifest: str
    source_fingerprint: str
    members: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    entry_id: str = ""


@dataclass
class PluginCatalogBuild:
    """Counts from :meth:`EmbeddingRegistry.build_plugin_catalog`.

    ``skipped_plugins`` names plugins without a description (only their bundled
    skills were indexed); ``invalid_plugins`` holds ``(relative path, reason)``
    for plugins whose manifest could not supply a profile (nothing was indexed).
    """

    plugins: int = 0
    skills: int = 0
    skipped_plugins: list[str] = field(default_factory=list)
    invalid_plugins: list[tuple[str, str]] = field(default_factory=list)


def classify(score: float) -> tuple[str, Severity]:
    """Map a cosine similarity score to a classification tier and severity.

    The four fixed tiers are checked in descending order; the caller's
    --threshold flag controls which tiers are *reported*, not how they
    are classified.
    """
    if score >= SIMILARITY_CRITICAL_THRESHOLD:
        return "EXACT_DUPLICATE", Severity.CRITICAL
    if score >= SIMILARITY_HIGH_THRESHOLD:
        return "HIGH_SIMILARITY", Severity.HIGH
    if score >= SIMILARITY_MEDIUM_THRESHOLD:
        return "SIMILAR", Severity.MEDIUM
    if score >= SIMILARITY_LOW_THRESHOLD:
        return "LOOSELY_RELATED", Severity.LOW
    return "DISTINCT", Severity.INFO


@dataclass
class SimilarityMatch:
    """A pair of content items whose similarity exceeds the threshold."""

    entry_a: str
    entry_b: str
    score: float
    path_a: str
    path_b: str
    classification: str
    severity: Severity

    @classmethod
    def from_score(
        cls,
        *,
        name_a: str,
        name_b: str,
        path_a: str,
        path_b: str,
        score: float,
    ) -> SimilarityMatch:
        """Build a match from a raw cosine similarity score.

        Automatically classifies the score into the appropriate tier.
        """
        classification, severity = classify(score)
        return cls(
            entry_a=name_a,
            entry_b=name_b,
            score=score,
            path_a=path_a,
            path_b=path_b,
            classification=classification,
            severity=severity,
        )


class EmbeddingRegistry:
    """Builds and queries an in-memory embedding index.

    Supports two workflows:
    1. Live scan: build_from_directory() discovers content, embeds it, stores.
    2. Cached: load_cache() / save_cache() for pre-computed indexes.
    """

    def __init__(
        self,
        client: EmbeddingClient,
        *,
        full_body: bool = False,
        max_entries: int = SIMILARITY_DEFAULT_MAX_ENTRIES,
        max_scalar_comparisons: int = SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    ) -> None:
        validate_max_entries(max_entries)
        validate_max_scalar_comparisons(max_scalar_comparisons)
        self._client = client
        self._full_body = full_body
        self._max_entries = max_entries
        self._max_pairwise_comparisons = max_entries * (max_entries - 1) // 2
        self._max_scalar_comparisons = max_scalar_comparisons
        self._entries: dict[str, RegistryEntry] = {}
        self._plugin_entries: dict[str, PluginRegistryEntry] = {}
        self._vector_dimension: int | None = None

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def plugin_size(self) -> int:
        """Number of plugin entries (version 2 catalogs only)."""
        return len(self._plugin_entries)

    @property
    def skill_entries(self) -> tuple[RegistryEntry, ...]:
        return tuple(entry for entry in self._entries.values() if entry.content_type == CONTENT_TYPE_SKILL)

    @property
    def plugin_entries(self) -> tuple[PluginRegistryEntry, ...]:
        return tuple(self._plugin_entries.values())

    def build_plugin_catalog(self, root: Path) -> PluginCatalogBuild:
        """Index plugins at or below ``root`` plus their bundled skills.

        Each plugin with a description contributes one plugin entry (name and
        description embedding, member skill names, manifest source fingerprint).
        Every bundled ``skills/**/SKILL.md`` whose frontmatter has a name and
        description contributes one skill entry. A plugin whose manifest cannot
        supply a profile is left out and recorded by relative path in
        ``invalid_plugins``; unsafe inputs still fail the whole build.
        Discovery, manifest reads, and skill reads use the secure bounded
        helpers; ``max_entries`` bounds the selected plugin and skill manifests,
        and one aggregate byte budget bounds everything read before any
        embedding request.
        """
        from skillevaluator.deduplication.plugin.profile import (
            PluginProfileError,
            PluginSkillLimitError,
            discover_plugin_roots,
            load_plugin_profile,
        )

        if self._full_body:
            raise ValueError("Plugin catalogs use description embeddings; --full-body is not supported for plugins")
        limit = min(self._max_entries, MAX_CATALOG_ENTRIES)
        plugin_roots = discover_plugin_roots(root, max_plugins=limit)
        build = PluginCatalogBuild()
        if not plugin_roots:
            return build
        root_absolute = Path(os.path.abspath(os.fspath(root)))  # noqa: PTH100 - lexical, no-follow
        budget = ExtractionBudget(max_entries=limit)
        pending_skills: list[tuple[RegistryEntry, str]] = []
        pending_plugins: list[tuple[PluginRegistryEntry, str]] = []
        selected = 0
        for plugin_root in plugin_roots:
            selected += 1
            plugin_absolute = Path(os.path.abspath(os.fspath(plugin_root)))  # noqa: PTH100
            try:
                plugin_path = plugin_absolute.relative_to(root_absolute).as_posix() or "."
            except ValueError as exc:
                raise ValueError(f"Discovered plugin path escapes scan root: {plugin_root.name}") from exc
            try:
                profile = load_plugin_profile(plugin_root, max_skills=max(0, limit - selected), budget=budget)
            except PluginSkillLimitError as exc:
                raise ValueError(
                    f"Collection entry limit exceeded ({limit}) before embedding; "
                    "increase --max-entries within its supported range to scan the complete collection"
                ) from exc
            except PluginProfileError as exc:
                build.invalid_plugins.append((plugin_path, str(exc)))
                continue
            selected += len(profile.bundled_skills)
            if profile.description is None:
                build.skipped_plugins.append(profile.name)
            else:
                pending_plugins.append(
                    (
                        PluginRegistryEntry(
                            entry_id=f"{CONTENT_TYPE_PLUGIN}:{plugin_path}",
                            name=profile.name,
                            description=profile.description,
                            path=plugin_path,
                            manifest=profile.manifest,
                            source_fingerprint=profile.source_fingerprint,
                            members=list(profile.members),
                        ),
                        profile.embedding_text,
                    )
                )
            for skill in profile.bundled_skills:
                if skill.entry is None:
                    continue
                skill_path = (PurePosixPath(plugin_path) / "skills" / skill.rel).as_posix()
                pending_skills.append(
                    (
                        RegistryEntry(
                            entry_id=f"{CONTENT_TYPE_SKILL}:{skill_path}",
                            name=skill.entry.name,
                            description=skill.entry.description,
                            path=skill_path,
                            content_type=CONTENT_TYPE_SKILL,
                            content_fingerprint=_fingerprint(skill.entry.embedding_text),
                        ),
                        skill.entry.embedding_text,
                    )
                )

        texts = [text for _entry, text in pending_skills] + [text for _entry, text in pending_plugins]
        for text in texts:
            _validate_embedding_text(text, full_body=False)
        vectors = self._embed_texts(texts)
        for (entry, _text), vector in zip(pending_skills, vectors[: len(pending_skills)], strict=True):
            entry.embedding = vector
        for (plugin, _text), vector in zip(pending_plugins, vectors[len(pending_skills) :], strict=True):
            plugin.embedding = vector
        self._entries.update((entry.entry_id, entry) for entry, _text in pending_skills)
        self._plugin_entries.update((plugin.entry_id, plugin) for plugin, _text in pending_plugins)
        build.plugins = len(pending_plugins)
        build.skills = len(pending_skills)
        logger.debug(
            "Indexed %d plugin and %d bundled skill entries from %s",
            build.plugins,
            build.skills,
            safe_path_label(root),
        )
        return build

    def _embed_texts(
        self,
        texts: list[str],
        *,
        full_body: bool = False,
        on_first_batch: Callable[[int], None] | None = None,
    ) -> list[list[float]]:
        """Embed texts in bounded batches, validating each vector against the registry width.

        Full-body texts are embedded one at a time with chunked pooling.
        ``on_first_batch`` receives the vector width once the first batch is
        validated, before any further request, so it can refuse the workload.
        """
        vector_dimension = self._vector_dimension
        vectors: list[list[float]] = []
        batch_size = 1 if full_body else EMBEDDING_BATCH_SIZE
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_vectors = [self._client.embed_chunked(batch[0])] if full_body else self._client.embed(batch)
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"Embedding provider returned {len(batch_vectors)} vectors for {len(batch)} entries in batch"
                )
            for vector in batch_vectors:
                vector_dimension = _validate_vector(vector, vector_dimension)
                vectors.append(vector)
            if start == 0 and on_first_batch is not None:
                on_first_batch(vector_dimension or 0)
        self._vector_dimension = vector_dimension
        return vectors

    def _prepare_catalog(
        self,
        entries: Sequence[RegistryEntry | PluginRegistryEntry],
        *,
        comparisons: int,
    ) -> tuple[int, list[list[float]]]:
        """Check the comparison work, then return the vector width and one unit vector per entry.

        ``comparisons`` is the number of vector pairs the caller will score.
        Each catalog vector is validated against the width and normalized once.
        """
        vector_dimension = _registry_vector_dimension(entries, self._vector_dimension)
        _validate_scalar_work(comparisons, vector_dimension, self._max_scalar_comparisons)
        return vector_dimension, _normalized_registry_vectors(entries, vector_dimension)

    def score_plugin_text(self, text: str) -> list[tuple[PluginRegistryEntry, float]]:
        """Embed one plugin description text and score it against every catalog plugin entry.

        The comparison work (catalog plugins x dimensions) is bounded by
        ``max_scalar_comparisons``. Callers apply thresholds and self-exclusion.
        """
        entries = list(self._plugin_entries.values())
        if not entries:
            return []
        vector_dimension, unit_vectors = self._prepare_catalog(entries, comparisons=len(entries))
        _validate_embedding_text(text, full_body=False)
        query_vector = _normalized_vector(self._client.embed_single(text), vector_dimension or None)
        return [
            (entry, unit_vector_similarity(query_vector, unit_vector))
            for entry, unit_vector in zip(entries, unit_vectors, strict=True)
        ]

    def score_skill_entries(self, targets: list[ContentEntry]) -> list[list[tuple[RegistryEntry, float]]]:
        """Batch-embed target skills and score each against every catalog skill entry.

        Each target is one catalog query, so the per-query work bound
        (catalog skills x dimensions) matches :meth:`query_entry`.
        """
        catalog_entries = list(self.skill_entries)
        if not targets or not catalog_entries:
            return [[] for _target in targets]
        vector_dimension, unit_vectors = self._prepare_catalog(catalog_entries, comparisons=len(catalog_entries))
        texts = [target.embedding_text for target in targets]
        for text in texts:
            _validate_embedding_text(text, full_body=False)
        query_vectors = [_normalized_vector(vector, vector_dimension or None) for vector in self._embed_texts(texts)]
        return [
            [
                (catalog_entry, unit_vector_similarity(query_vector, unit_vector))
                for catalog_entry, unit_vector in zip(catalog_entries, unit_vectors, strict=True)
            ]
            for query_vector in query_vectors
        ]

    def build_from_directory(
        self,
        root: Path,
        content_type: str,
        *,
        minimum_entries: int = 1,
        for_pairwise_scan: bool = False,
    ) -> int:
        """Discover content items, embed them, and populate the index.

        Set ``for_pairwise_scan`` when the caller will compare all entries.
        The first validated response then checks the pairwise work budget
        before requesting the remaining embeddings. Index-only builds retain
        their independent capacity for single-target queries.

        Returns:
            Number of entries successfully indexed.
        """
        content_entries = discover_and_extract(root, content_type, max_entries=self._max_entries)
        if not content_entries:
            logger.debug("No content entries found in %s", safe_path_label(root))
            return 0
        if len(content_entries) < minimum_entries:
            raise ValueError(
                "Collection similarity requires at least 2 skills; for one skill use context-optimization-check"
            )
        if len(content_entries) > MAX_CATALOG_ENTRIES:
            raise ValueError(f"Catalog entry limit exceeded ({MAX_CATALOG_ENTRIES})")

        texts = [entry.full_text if self._full_body else entry.embedding_text for entry in content_entries]
        for text in texts:
            _validate_embedding_text(text, full_body=self._full_body)
        resolved_root = root.resolve(strict=True)
        pending_entries: list[RegistryEntry] = []
        for entry in content_entries:
            resolved_entry = Path(entry.path).resolve(strict=True)
            try:
                relative_path = resolved_entry.relative_to(resolved_root).as_posix() or "."
            except ValueError as exc:
                raise ValueError(f"Discovered content path escapes scan root: {entry.path}") from exc
            entry_id = f"{entry.content_type}:{relative_path}"
            pending_entries.append(
                RegistryEntry(
                    name=entry.name,
                    description=entry.description,
                    path=relative_path,
                    content_type=entry.content_type,
                    entry_id=entry_id,
                    content_fingerprint=_fingerprint(entry.full_text if self._full_body else entry.embedding_text),
                )
            )

        entry_count = len(self._entries.keys() | {entry.entry_id for entry in pending_entries})
        comparison_count = entry_count * (entry_count - 1) // 2

        def check_pairwise_work(vector_dimension: int) -> None:
            _validate_scalar_work(comparison_count, vector_dimension, self._max_scalar_comparisons)

        vectors = self._embed_texts(
            texts,
            full_body=self._full_body,
            on_first_batch=check_pairwise_work if for_pairwise_scan else None,
        )
        for entry, vector in zip(pending_entries, vectors, strict=True):
            entry.embedding = vector
        self._entries.update((entry.entry_id, entry) for entry in pending_entries)

        logger.debug("Indexed %d entries from %s", len(self._entries), safe_path_label(root))
        return len(self._entries)

    def find_duplicates(self, threshold: float) -> list[SimilarityMatch]:
        """Pairwise comparison of all indexed entries.

        Only pairs with cosine similarity >= threshold are returned,
        sorted by score descending.
        """
        validate_similarity_threshold(threshold, context="Similarity")
        entries = list(self._entries.values())
        comparison_count = len(entries) * (len(entries) - 1) // 2
        if comparison_count > self._max_pairwise_comparisons:
            raise ValueError(
                f"Pairwise comparison limit exceeded ({self._max_pairwise_comparisons}); "
                "increase --max-entries within its supported range to compare the complete collection"
            )
        _, unit_vectors = self._prepare_catalog(entries, comparisons=comparison_count)
        matches: list[SimilarityMatch] = []

        for (a, unit_a), (b, unit_b) in combinations(zip(entries, unit_vectors, strict=True), 2):
            score = unit_vector_similarity(unit_a, unit_b)
            if score >= threshold:
                if len(matches) >= MAX_SIMILARITY_MATCHES:
                    raise ValueError(f"Similarity match limit exceeded ({MAX_SIMILARITY_MATCHES})")
                matches.append(
                    SimilarityMatch.from_score(
                        name_a=a.name,
                        name_b=b.name,
                        path_a=a.path,
                        path_b=b.path,
                        score=score,
                    )
                )

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def query(self, text: str, threshold: float) -> list[SimilarityMatch]:
        """Compare a single text against all indexed entries.

        Useful for checking a new item against the existing registry.
        """
        validate_similarity_threshold(threshold, context="Similarity")
        entries = list(self._entries.values())
        vector_dimension, unit_vectors = self._prepare_catalog(entries, comparisons=len(entries))
        _validate_embedding_text(text, full_body=self._full_body)
        vector = self._client.embed_chunked(text) if self._full_body else self._client.embed_single(text)
        query_vector = _normalized_vector(vector, vector_dimension or self._vector_dimension)

        matches: list[SimilarityMatch] = []
        for entry, unit_vector in zip(entries, unit_vectors, strict=True):
            score = unit_vector_similarity(query_vector, unit_vector)
            if score >= threshold:
                if len(matches) >= MAX_SIMILARITY_MATCHES:
                    raise ValueError(f"Similarity match limit exceeded ({MAX_SIMILARITY_MATCHES})")
                matches.append(
                    SimilarityMatch.from_score(
                        name_a="(query)",
                        name_b=entry.name,
                        path_a="",
                        path_b=entry.path,
                        score=score,
                    )
                )

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def query_entry(self, entry: ContentEntry, threshold: float) -> list[SimilarityMatch]:
        """Compare one extracted target entry against every catalog entry."""
        validate_similarity_threshold(threshold, context="Similarity")
        catalog_entries = list(self._entries.values())
        vector_dimension, unit_vectors = self._prepare_catalog(catalog_entries, comparisons=len(catalog_entries))
        text = entry.full_text if self._full_body else entry.embedding_text
        _validate_embedding_text(text, full_body=self._full_body)
        vector = self._client.embed_chunked(text) if self._full_body else self._client.embed_single(text)
        query_vector = _normalized_vector(vector, vector_dimension or self._vector_dimension)
        target_path = Path(entry.path).name or "."

        matches: list[SimilarityMatch] = []
        for catalog_entry, unit_vector in zip(catalog_entries, unit_vectors, strict=True):
            score = unit_vector_similarity(query_vector, unit_vector)
            if score >= threshold:
                if len(matches) >= MAX_SIMILARITY_MATCHES:
                    raise ValueError(f"Similarity match limit exceeded ({MAX_SIMILARITY_MATCHES})")
                matches.append(
                    SimilarityMatch.from_score(
                        name_a=entry.name,
                        name_b=catalog_entry.name,
                        path_a=target_path,
                        path_b=catalog_entry.path,
                        score=score,
                    )
                )
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches

    # ------------------------------------------------------------------
    # Catalog persistence
    # ------------------------------------------------------------------

    def save_catalog(self, catalog_path: Path) -> None:
        """Persist a validated, versioned local embedding catalog.

        Catalogs without plugin entries keep schema version 1 so older readers
        still load them; catalogs with plugin entries use schema version 2.
        """
        if not self._entries and not self._plugin_entries:
            raise ValueError("Cannot save an empty catalog")
        if len(self._entries) + len(self._plugin_entries) > MAX_CATALOG_ENTRIES:
            raise ValueError(f"Catalog entry limit exceeded ({MAX_CATALOG_ENTRIES})")

        entries: list[dict[str, object]] = []
        vector_dimension: int | None = None
        for key, entry in sorted(self._entries.items()):
            vector_dimension = _validate_vector(entry.embedding, vector_dimension)
            entry_id = entry.entry_id or key
            _validate_catalog_identity(entry_id, entry.path, entry.content_type)
            fingerprint = entry.content_fingerprint
            if not _SHA256_PATTERN.fullmatch(fingerprint):
                raise ValueError(f"Catalog entry '{entry_id}' has an invalid content fingerprint")
            _validate_text_fields(entry_id, entry.name, entry.description)
            entries.append(
                {
                    "id": entry_id,
                    "name": entry.name,
                    "description": entry.description,
                    "path": entry.path,
                    "content_type": entry.content_type,
                    "content_fingerprint": fingerprint,
                    "embedding": entry.embedding,
                }
            )

        plugins: list[dict[str, object]] = []
        member_names = 0
        for key, plugin in sorted(self._plugin_entries.items()):
            vector_dimension = _validate_vector(plugin.embedding, vector_dimension)
            entry_id = plugin.entry_id or key
            members = _validate_plugin_entry_fields(
                entry_id,
                path=plugin.path,
                name=plugin.name,
                description=plugin.description,
                manifest=plugin.manifest,
                source_fingerprint=plugin.source_fingerprint,
                members=plugin.members,
            )
            member_names += len(members)
            plugins.append(
                {
                    "id": entry_id,
                    "name": plugin.name,
                    "description": plugin.description,
                    "path": plugin.path,
                    "manifest": plugin.manifest,
                    "source_fingerprint": plugin.source_fingerprint,
                    "members": members,
                    "embedding": plugin.embedding,
                }
            )
        if member_names > MAX_CATALOG_MEMBER_NAMES:
            raise ValueError(f"Catalog plugin member name limit exceeded ({MAX_CATALOG_MEMBER_NAMES})")

        if vector_dimension is None:
            raise ValueError("Cannot save a catalog without embedding vectors")
        if (len(entries) + len(plugins)) * vector_dimension > MAX_CATALOG_VECTOR_VALUES:
            raise ValueError(f"Catalog vector scalar limit exceeded ({MAX_CATALOG_VECTOR_VALUES})")
        data: dict[str, object] = {
            "schema_version": PLUGIN_CATALOG_SCHEMA_VERSION if plugins else SKILL_CATALOG_SCHEMA_VERSION,
            "provider": _client_provider(self._client),
            "model": self._client.model,
            "mode": "full-body" if self._full_body else "description",
            "endpoint_fingerprint": _client_endpoint_fingerprint(self._client),
            "vector_dimension": vector_dimension,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "entries": entries,
        }
        if plugins:
            data["plugins"] = plugins
        serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
        if len(serialized.encode("utf-8")) > MAX_CATALOG_BYTES:
            raise ValueError(f"Catalog size limit exceeded ({MAX_CATALOG_BYTES} bytes)")
        _write_catalog_atomically(catalog_path, serialized.encode("utf-8"))
        logger.debug(
            "Saved local catalog to %s (%d entries, %d plugins)",
            safe_path_label(catalog_path),
            len(self._entries),
            len(self._plugin_entries),
        )

    def load_catalog(self, catalog_path: Path) -> None:
        """Load and validate a versioned local embedding catalog.

        Schema version 1 (skill/rules/workflows entries only) and version 2
        (adds a ``plugins`` list) are both accepted.
        """
        try:
            serialized = _read_catalog_text(catalog_path)
            preflight_json_structure(
                serialized,
                max_depth=100,
                max_tokens=(MAX_CATALOG_VECTOR_VALUES * 2)
                + (MAX_CATALOG_ENTRIES * 20)
                + (MAX_CATALOG_MEMBER_NAMES * 2),
                max_collection_items=MAX_VECTOR_DIMENSION,
                # Leave room for duplicate/unknown keys so the schema layer can
                # report them precisely while still bounding object materialization.
                max_mapping_items=2
                * max(len(_CATALOG_ROOT_FIELDS_V2), len(_CATALOG_ENTRY_FIELDS), len(_CATALOG_PLUGIN_FIELDS)),
                max_string_chars=MAX_CATALOG_TEXT_LENGTH,
            )
            raw = json.loads(
                serialized,
                object_pairs_hook=_catalog_object,
                parse_constant=_reject_json_constant,
            )
        except _DuplicateCatalogKeyError as exc:
            raise ValueError(f"Catalog JSON contains duplicate key: {exc.key}") from exc
        except (UnicodeError, json.JSONDecodeError, StructuredDataLimitError, RecursionError) as exc:
            raise ValueError(f"Malformed catalog JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Catalog root must be a JSON object")

        schema_version = raw.get("schema_version")
        if type(schema_version) is not int or schema_version not in SUPPORTED_CATALOG_SCHEMA_VERSIONS:
            supported = ", ".join(str(version) for version in sorted(SUPPORTED_CATALOG_SCHEMA_VERSIONS))
            raise ValueError(f"Unsupported catalog schema version: {schema_version!r}; expected one of {supported}")
        root_fields = (
            _CATALOG_ROOT_FIELDS_V2 if schema_version >= PLUGIN_CATALOG_SCHEMA_VERSION else _CATALOG_ROOT_FIELDS
        )
        _validate_exact_fields(raw, root_fields, "Catalog root")

        provider = _validate_catalog_string(raw.get("provider"), "Catalog provider")
        expected_provider = _client_provider(self._client)
        if provider != expected_provider:
            raise ValueError(f"Catalog provider mismatch: {provider!r}; expected {expected_provider!r}")
        model = _validate_catalog_string(raw.get("model"), "Catalog model")
        if model != self._client.model:
            raise ValueError(f"Catalog model mismatch: {model!r}; expected {self._client.model!r}")
        expected_mode = "full-body" if self._full_body else "description"
        mode = _validate_catalog_string(raw.get("mode"), "Catalog mode")
        if mode != expected_mode:
            raise ValueError(f"Catalog mode mismatch: {mode!r}; expected {expected_mode!r}")
        endpoint_fingerprint = _validate_catalog_string(
            raw.get("endpoint_fingerprint"),
            "Catalog endpoint fingerprint",
        )
        if not _ENDPOINT_FINGERPRINT_PATTERN.fullmatch(endpoint_fingerprint):
            raise ValueError("Catalog endpoint fingerprint is invalid")
        expected_endpoint_fingerprint = _client_endpoint_fingerprint(self._client)
        if endpoint_fingerprint != expected_endpoint_fingerprint:
            raise ValueError("Catalog embedding endpoint mismatch")
        created_at = _validate_catalog_string(raw.get("created_at"), "Catalog created_at")
        try:
            parsed_created_at = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ValueError("Catalog created_at must be an ISO-8601 timestamp") from exc
        if parsed_created_at.tzinfo is None:
            raise ValueError("Catalog created_at must include a timezone")

        vector_dimension = raw.get("vector_dimension")
        if type(vector_dimension) is not int or vector_dimension <= 0 or vector_dimension > MAX_VECTOR_DIMENSION:
            raise ValueError(f"Invalid catalog vector dimension: {vector_dimension!r}")
        raw_entries = raw.get("entries")
        raw_plugins = raw.get("plugins", [])
        if not isinstance(raw_entries, list):
            raise ValueError("Catalog entries must be a list")
        if not isinstance(raw_plugins, list):
            raise ValueError("Catalog plugins must be a list")
        if schema_version == SKILL_CATALOG_SCHEMA_VERSION and not raw_entries:
            raise ValueError("Catalog entries must be a non-empty list")
        if not raw_entries and not raw_plugins:
            raise ValueError("Catalog must contain at least one entry or plugin")
        total_entries = len(raw_entries) + len(raw_plugins)
        if total_entries > MAX_CATALOG_ENTRIES:
            raise ValueError(f"Catalog entry limit exceeded ({MAX_CATALOG_ENTRIES})")
        if total_entries * vector_dimension > MAX_CATALOG_VECTOR_VALUES:
            raise ValueError(f"Catalog vector scalar limit exceeded ({MAX_CATALOG_VECTOR_VALUES})")

        loaded: dict[str, RegistryEntry] = {}
        for index, entry_data in enumerate(raw_entries):
            if not isinstance(entry_data, dict):
                raise ValueError(f"Catalog entry {index} must be an object")
            _validate_exact_fields(entry_data, _CATALOG_ENTRY_FIELDS, f"Catalog entry {index}")
            entry_id = entry_data["id"]
            name = entry_data["name"]
            description = entry_data["description"]
            path = entry_data["path"]
            content_type = entry_data["content_type"]
            fingerprint = entry_data["content_fingerprint"]
            _validate_catalog_identity(entry_id, path, content_type)
            _validate_text_fields(entry_id, name, description)
            if not isinstance(fingerprint, str) or not _SHA256_PATTERN.fullmatch(fingerprint):
                raise ValueError(f"Catalog entry '{entry_id}' has an invalid content fingerprint")
            embedding = entry_data["embedding"]
            _validate_vector(embedding, vector_dimension)
            if entry_id in loaded:
                raise ValueError(f"Catalog contains duplicate entry id: {entry_id}")
            loaded[entry_id] = RegistryEntry(
                entry_id=entry_id,
                name=name,
                description=description,
                path=path,
                content_type=content_type,
                content_fingerprint=fingerprint,
                embedding=embedding,
            )

        loaded_plugins: dict[str, PluginRegistryEntry] = {}
        member_names = 0
        for index, plugin_data in enumerate(raw_plugins):
            if not isinstance(plugin_data, dict):
                raise ValueError(f"Catalog plugin {index} must be an object")
            _validate_exact_fields(plugin_data, _CATALOG_PLUGIN_FIELDS, f"Catalog plugin {index}")
            entry_id = plugin_data["id"]
            members = _validate_plugin_entry_fields(
                entry_id,
                path=plugin_data["path"],
                name=plugin_data["name"],
                description=plugin_data["description"],
                manifest=plugin_data["manifest"],
                source_fingerprint=plugin_data["source_fingerprint"],
                members=plugin_data["members"],
            )
            member_names += len(members)
            if member_names > MAX_CATALOG_MEMBER_NAMES:
                raise ValueError(f"Catalog plugin member name limit exceeded ({MAX_CATALOG_MEMBER_NAMES})")
            embedding = plugin_data["embedding"]
            _validate_vector(embedding, vector_dimension)
            if entry_id in loaded_plugins:
                raise ValueError(f"Catalog contains duplicate plugin id: {entry_id}")
            loaded_plugins[entry_id] = PluginRegistryEntry(
                entry_id=entry_id,
                name=plugin_data["name"],
                description=plugin_data["description"],
                path=plugin_data["path"],
                manifest=plugin_data["manifest"],
                source_fingerprint=plugin_data["source_fingerprint"],
                members=members,
                embedding=embedding,
            )

        self._entries = loaded
        self._plugin_entries = loaded_plugins
        self._vector_dimension = vector_dimension
        logger.debug(
            "Loaded %d entries and %d plugins from local catalog %s",
            len(self._entries),
            len(self._plugin_entries),
            safe_path_label(catalog_path),
        )

    def save_cache(self, cache_path: Path) -> None:
        """Deprecated compatibility alias for :meth:`save_catalog`."""
        self.save_catalog(cache_path)

    def load_cache(self, cache_path: Path) -> None:
        """Deprecated compatibility alias for :meth:`load_catalog`."""
        self.load_catalog(cache_path)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _client_provider(client: EmbeddingClient) -> str:
    config = client._resolved_config()
    provider = getattr(config, "provider", None)
    if not isinstance(provider, str) or not provider:
        raise ValueError("Embedding provider metadata is unavailable")
    return provider


def _client_endpoint_fingerprint(client: EmbeddingClient) -> str:
    """Return a credential-free identity for the configured embedding endpoint."""
    config = client._resolved_config()
    provider = getattr(config, "provider", None)
    if not isinstance(provider, str) or not provider:
        raise ValueError("Embedding provider metadata is unavailable")
    base_url = getattr(config, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        identity = f"provider:{provider}"
    else:
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Embedding endpoint metadata is invalid") from exc
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Embedding endpoint metadata must be an HTTP(S) URL")
        scheme = parsed.scheme.lower()
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
        authority = host if port is None or default_port else f"{host}:{port}"
        path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
        identity = f"{scheme}://{authority}{path}"
    return f"sha256:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _validate_embedding_text(text: object, *, full_body: bool) -> None:
    if not isinstance(text, str):
        raise ValueError("Embedding text must be a string")
    if full_body:
        try:
            encoded_size = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("Full-body embedding text contains invalid Unicode") from exc
        if encoded_size > MAX_FULL_BODY_EMBEDDING_TEXT_BYTES:
            raise ValueError(
                f"Full-body embedding text byte limit exceeded ({MAX_FULL_BODY_EMBEDDING_TEXT_BYTES}) before embedding"
            )
        return
    if len(text) > MAX_DESCRIPTION_EMBEDDING_TEXT_CHARS:
        raise ValueError(
            "Description embedding text character limit exceeded "
            f"({MAX_DESCRIPTION_EMBEDDING_TEXT_CHARS}) before embedding"
        )


def _validate_vector(vector: object, expected_dimension: int | None) -> int:
    if isinstance(vector, list) and len(vector) > MAX_VECTOR_DIMENSION:
        raise ValueError(f"Catalog vector dimension exceeds {MAX_VECTOR_DIMENSION}")
    try:
        return validate_embedding_vector(
            vector,
            expected_dimension,
            context="Catalog embedding",
        )
    except SimilarityConfigError as exc:
        raise ValueError(str(exc)) from exc


def _normalized_vector(vector: object, expected_dimension: int | None) -> list[float]:
    if isinstance(vector, list) and len(vector) > MAX_VECTOR_DIMENSION:
        raise ValueError(f"Catalog vector dimension exceeds {MAX_VECTOR_DIMENSION}")
    try:
        return normalize_embedding_vector(
            vector,
            expected_dimension,
            context="Catalog embedding",
        )
    except SimilarityConfigError as exc:
        raise ValueError(str(exc)) from exc


def _registry_vector_dimension(
    entries: Sequence[RegistryEntry | PluginRegistryEntry],
    expected_dimension: int | None,
) -> int:
    """Return the width for work-budget checks without retaining normalized copies."""
    if expected_dimension is not None or not entries:
        return expected_dimension or 0
    return _validate_vector(entries[0].embedding, None)


def _normalized_registry_vectors(
    entries: Sequence[RegistryEntry | PluginRegistryEntry],
    dimension: int,
) -> list[list[float]]:
    """Validate every vector against the registry width and normalize each once."""
    return [_normalized_vector(entry.embedding, dimension or None) for entry in entries]


def _validate_scalar_work(comparison_count: int, vector_dimension: int, max_scalar_comparisons: int) -> None:
    scalar_work = comparison_count * vector_dimension
    if scalar_work > max_scalar_comparisons:
        raise ValueError(
            f"Scalar comparison work limit exceeded ({max_scalar_comparisons}); requested {scalar_work}. "
            "Increase --max-scalar-comparisons to allow this comparison workload"
        )


def _validate_catalog_relative_path(path: object) -> str:
    path = _validate_catalog_string(path, "Catalog path")
    if "\\" in path or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise ValueError(f"Catalog path must be relative: {path!r}")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != path:
        raise ValueError(f"Catalog path must be relative and normalized: {path!r}")
    return path


def _validate_catalog_identity(entry_id: object, path: object, content_type: object) -> None:
    path = _validate_catalog_relative_path(path)
    content_type = _validate_catalog_string(content_type, "Catalog content type")
    entry_id = _validate_catalog_string(entry_id, "Catalog entry id")
    if content_type not in _CATALOG_CONTENT_TYPES:
        raise ValueError(f"Catalog content type is unsupported: {content_type!r}")
    expected_id = f"{content_type}:{path}"
    if entry_id != expected_id:
        raise ValueError(f"Catalog entry id must match its relative path: expected {expected_id!r}")


def _validate_plugin_entry_fields(
    entry_id: object,
    *,
    path: object,
    name: object,
    description: object,
    manifest: object,
    source_fingerprint: object,
    members: object,
) -> list[str]:
    """Validate one version 2 plugin entry and return its canonical member list."""
    entry_id = _validate_catalog_string(entry_id, "Catalog plugin id")
    path = _validate_catalog_relative_path(path)
    expected_id = f"{CONTENT_TYPE_PLUGIN}:{path}"
    if entry_id != expected_id:
        raise ValueError(f"Catalog plugin id must match its relative path: expected {expected_id!r}")
    _validate_text_fields(entry_id, name, description)
    if manifest not in _PLUGIN_CATALOG_MANIFESTS:
        raise ValueError(f"Catalog plugin '{entry_id}' has an unsupported manifest: {manifest!r}")
    if not isinstance(source_fingerprint, str) or not _SHA256_PATTERN.fullmatch(source_fingerprint):
        raise ValueError(f"Catalog plugin '{entry_id}' has an invalid source fingerprint")
    if type(members) is not list:
        raise ValueError(f"Catalog plugin '{entry_id}' members must be a list")
    if len(members) > MAX_CATALOG_PLUGIN_MEMBERS:
        raise ValueError(f"Catalog plugin '{entry_id}' members exceed the {MAX_CATALOG_PLUGIN_MEMBERS}-item limit")
    validated: list[str] = []
    for member in members:
        value = _validate_catalog_string(member, f"Catalog plugin '{entry_id}' member")
        if len(value) > MAX_CATALOG_PLUGIN_MEMBER_CHARS or value != value.strip().casefold():
            raise ValueError(f"Catalog plugin '{entry_id}' members must be bounded, trimmed, casefolded names")
        validated.append(value)
    if validated != sorted(set(validated)):
        raise ValueError(f"Catalog plugin '{entry_id}' members must be unique and sorted")
    return validated


def _validate_text_fields(entry_id: str, name: object, description: object) -> None:
    _validate_catalog_string(name, f"Catalog entry '{entry_id}' name")
    _validate_catalog_string(description, f"Catalog entry '{entry_id}' description")


def _validate_catalog_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_CATALOG_TEXT_LENGTH:
        raise ValueError(f"{label} must be a non-empty bounded string")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value):
        raise ValueError(f"{label} contains unsafe control or surrogate characters")
    return value


def _validate_exact_fields(data: dict[str, object], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - data.keys())
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    unknown = sorted(data.keys() - expected)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


class _DuplicateCatalogKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _catalog_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateCatalogKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Catalog JSON contains non-finite number: {value}")


_CATALOG_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_CATALOG_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOCTTY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOINHERIT", 0)
)
_CATALOG_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOINHERIT", 0)
)


def _catalog_absolute_path(catalog_path: Path) -> Path:
    if not catalog_path.name or catalog_path.name in {".", ".."}:
        raise ValueError(f"Catalog path must name a file: {catalog_path}")
    absolute = Path(os.path.abspath(os.fspath(catalog_path)))  # noqa: PTH100
    return canonicalize_trusted_root_alias(absolute)


def _supports_posix_catalog_io() -> bool:
    return bool(
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and os.mkdir in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
    )


def _open_posix_catalog_parent(catalog_path: Path, *, create: bool) -> tuple[Path, int, str]:
    if not _supports_posix_catalog_io():
        raise ValueError("This platform cannot guarantee secure descriptor-relative catalog I/O")
    absolute = _catalog_absolute_path(catalog_path)
    try:
        descriptor = os.open(absolute.anchor, _CATALOG_DIRECTORY_FLAGS)
    except OSError as exc:
        raise ValueError(f"Unable to securely open catalog filesystem root: {exc}") from exc
    try:
        root_metadata = os.fstat(descriptor)
        if stat_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("Catalog filesystem root is not a regular directory")
        for component in absolute.parent.parts[1:]:
            try:
                child = os.open(component, _CATALOG_DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError as exc:
                if not create:
                    raise ValueError(f"Catalog does not exist: {catalog_path}") from exc
                try:
                    os.mkdir(component, 0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as mkdir_exc:
                    raise ValueError(f"Unable to create catalog parent directory: {mkdir_exc}") from mkdir_exc
                try:
                    child = os.open(component, _CATALOG_DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError as open_exc:
                    raise ValueError(
                        "Catalog path contains a symlink, reparse point, or non-directory component"
                    ) from open_exc
            except OSError as exc:
                raise ValueError("Catalog path contains a symlink, reparse point, or non-directory component") from exc
            try:
                child_metadata = os.fstat(child)
                if stat_is_link_or_reparse(child_metadata) or not stat.S_ISDIR(child_metadata.st_mode):
                    raise ValueError("Catalog path contains a symlink, reparse point, or non-directory component")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return absolute, descriptor, absolute.name
    except BaseException:
        os.close(descriptor)
        raise


def _inspect_posix_catalog_file(
    parent_descriptor: int,
    name: str,
    catalog_path: Path,
    *,
    missing_ok: bool,
) -> os.stat_result | None:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        if missing_ok:
            return None
        raise ValueError(f"Catalog does not exist: {catalog_path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to inspect catalog: {exc}") from exc
    if stat_is_link_or_reparse(metadata):
        raise ValueError(f"Catalog path is a symlink or reparse point: {catalog_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Catalog path is not a regular file: {catalog_path}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"Catalog path is hard-linked (link count > 1): {catalog_path}")
    return metadata


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise OSError("Short write while saving catalog")
        written += count


def _write_catalog_atomically_posix(catalog_path: Path, payload: bytes) -> None:
    _absolute, parent_descriptor, name = _open_posix_catalog_parent(catalog_path, create=True)
    descriptor = -1
    temporary_name: str | None = None
    opened_metadata: os.stat_result | None = None
    try:
        _inspect_posix_catalog_file(parent_descriptor, name, catalog_path, missing_ok=True)
        for _attempt in range(128):
            candidate = f".{name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(candidate, _CATALOG_WRITE_FLAGS, 0o600, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            except OSError as exc:
                raise ValueError(f"Unable to create catalog temporary file: {exc}") from exc
            temporary_name = candidate
            break
        if descriptor < 0 or temporary_name is None:
            raise ValueError("Unable to allocate a unique catalog temporary file")

        opened_metadata = os.fstat(descriptor)
        _require_regular_single_link(opened_metadata, "Catalog temporary path is not a regular file")
        current_metadata = os.stat(temporary_name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_regular_single_link(
            current_metadata, "Catalog temporary path changed during creation", same_as=opened_metadata
        )

        _write_all(descriptor, payload)
        os.fsync(descriptor)
        current_metadata = os.stat(temporary_name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_regular_single_link(
            current_metadata, "Catalog temporary path changed while saving", same_as=opened_metadata
        )
        _inspect_posix_catalog_file(parent_descriptor, name, catalog_path, missing_ok=True)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        published = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_regular_single_link(
            published, "Catalog publication changed during atomic replacement", same_as=opened_metadata
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _validate_windows_catalog_parent(catalog_path: Path, *, create: bool) -> Path:
    absolute = _catalog_absolute_path(catalog_path)
    current = Path(absolute.anchor)
    try:
        root_metadata = current.lstat()
    except OSError as exc:
        raise ValueError(f"Unable to inspect catalog filesystem root: {exc}") from exc
    if stat_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("Catalog filesystem root is not a regular directory")

    for component in absolute.parent.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            if not create:
                raise ValueError(f"Catalog does not exist: {catalog_path}") from exc
            try:
                current.mkdir(mode=0o777)
            except FileExistsError:
                pass
            except OSError as mkdir_exc:
                raise ValueError(f"Unable to create catalog parent directory: {mkdir_exc}") from mkdir_exc
            try:
                metadata = current.lstat()
            except OSError as inspect_exc:
                raise ValueError(f"Unable to inspect catalog parent directory: {inspect_exc}") from inspect_exc
        except OSError as exc:
            raise ValueError(f"Unable to inspect catalog parent directory: {exc}") from exc
        if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Catalog path contains a symlink, reparse point, or non-directory component: {current}")
    return absolute


def _inspect_windows_catalog_file(
    catalog_path: Path,
    *,
    missing_ok: bool,
) -> os.stat_result | None:
    try:
        metadata = catalog_path.lstat()
    except FileNotFoundError as exc:
        if missing_ok:
            return None
        raise ValueError(f"Catalog does not exist: {catalog_path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to inspect catalog: {exc}") from exc
    if stat_is_link_or_reparse(metadata):
        raise ValueError(f"Catalog path is a symlink or reparse point: {catalog_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Catalog path is not a regular file: {catalog_path}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"Catalog path is hard-linked (link count > 1): {catalog_path}")
    return metadata


def _verify_windows_open_path(descriptor: int, catalog_path: Path) -> None:
    expected = os.path.normcase(os.fspath(catalog_path.absolute()))
    actual = os.path.normcase(os.fspath(windows_final_path(descriptor).absolute()))
    if actual != expected:
        raise ValueError("Opened catalog handle resolves through a reparse point or unexpected path")


def _write_catalog_atomically_windows(catalog_path: Path, payload: bytes) -> None:
    absolute = _validate_windows_catalog_parent(catalog_path, create=True)
    _inspect_windows_catalog_file(absolute, missing_ok=True)
    descriptor = -1
    temporary_path: Path | None = None
    opened_metadata: os.stat_result | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{absolute.name}.",
            suffix=".tmp",
            dir=absolute.parent,
        )
        temporary_path = Path(temporary_name)
        opened_metadata = os.fstat(descriptor)
        _require_regular_single_link(opened_metadata, "Catalog temporary path is not a regular file")
        current_metadata = temporary_path.lstat()
        _require_regular_single_link(
            current_metadata, "Catalog temporary path changed during creation", same_as=opened_metadata
        )

        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        _validate_windows_catalog_parent(absolute, create=False)
        current_metadata = temporary_path.lstat()
        _require_regular_single_link(
            current_metadata, "Catalog temporary path changed while saving", same_as=opened_metadata
        )
        _inspect_windows_catalog_file(absolute, missing_ok=True)
        temporary_path.replace(absolute)
        temporary_path = None
        published = absolute.lstat()
        _require_regular_single_link(
            published, "Catalog publication changed during atomic replacement", same_as=opened_metadata
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_catalog_atomically(catalog_path: Path, payload: bytes) -> None:
    if os.name == "posix":
        _write_catalog_atomically_posix(catalog_path, payload)
        return
    if os.name == "nt":
        _write_catalog_atomically_windows(catalog_path, payload)
        return
    raise ValueError("This platform cannot guarantee secure catalog writes")


def _read_bounded_catalog_descriptor(descriptor: int, catalog_path: Path) -> str:
    opened_metadata = os.fstat(descriptor)
    _require_regular_single_link(opened_metadata, f"Catalog path is not a regular file: {catalog_path}")
    if opened_metadata.st_size > MAX_CATALOG_BYTES:
        raise ValueError(f"Catalog size limit exceeded ({MAX_CATALOG_BYTES} bytes)")
    try:
        raw = read_bounded(descriptor, MAX_CATALOG_BYTES)
    except SecurePathError as exc:
        raise ValueError(f"Catalog size limit exceeded ({MAX_CATALOG_BYTES} bytes)") from exc
    return raw.decode("utf-8")


def _read_catalog_text_posix(catalog_path: Path) -> str:
    _absolute, parent_descriptor, name = _open_posix_catalog_parent(catalog_path, create=False)
    descriptor = -1
    try:
        before = _inspect_posix_catalog_file(parent_descriptor, name, catalog_path, missing_ok=False)
        if before is None:
            raise ValueError(f"Catalog does not exist: {catalog_path}")
        try:
            descriptor = os.open(name, _CATALOG_READ_FLAGS, dir_fd=parent_descriptor)
        except FileNotFoundError as exc:
            raise ValueError("Catalog changed while being opened") from exc
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise ValueError(f"Catalog path is a symlink or reparse point: {catalog_path}") from exc
            raise ValueError(f"Unable to open catalog: {exc}") from exc
        opened_metadata = os.fstat(descriptor)
        _require_regular_single_link(
            opened_metadata, "Catalog changed or is not a regular file while being opened", same_as=before
        )
        serialized = _read_bounded_catalog_descriptor(descriptor, catalog_path)
        after = _inspect_posix_catalog_file(parent_descriptor, name, catalog_path, missing_ok=False)
        if after is None or not os.path.samestat(opened_metadata, after):
            raise ValueError("Catalog changed or became a symlink/reparse point while being read")
        return serialized
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _read_catalog_text_windows(catalog_path: Path) -> str:
    absolute = _validate_windows_catalog_parent(catalog_path, create=False)
    before = _inspect_windows_catalog_file(absolute, missing_ok=False)
    if before is None:
        raise ValueError(f"Catalog does not exist: {catalog_path}")
    try:
        descriptor = os.open(absolute, _CATALOG_READ_FLAGS)
    except FileNotFoundError as exc:
        raise ValueError("Catalog changed while being opened") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK} or is_link_or_reparse(absolute):
            raise ValueError(f"Catalog path is a symlink or reparse point: {catalog_path}") from exc
        raise ValueError(f"Unable to open catalog: {exc}") from exc

    try:
        opened_metadata = os.fstat(descriptor)
        _require_regular_single_link(
            opened_metadata, "Catalog changed or is not a regular file while being opened", same_as=before
        )
        _verify_windows_open_path(descriptor, absolute)
        serialized = _read_bounded_catalog_descriptor(descriptor, catalog_path)
        _validate_windows_catalog_parent(absolute, create=False)
        after = _inspect_windows_catalog_file(absolute, missing_ok=False)
        if after is None or not os.path.samestat(opened_metadata, after):
            raise ValueError("Catalog changed or became a symlink/reparse point while being read")
        return serialized
    finally:
        os.close(descriptor)


def _read_catalog_text(catalog_path: Path) -> str:
    if os.name == "posix":
        return _read_catalog_text_posix(catalog_path)
    if os.name == "nt":
        return _read_catalog_text_windows(catalog_path)
    raise ValueError("This platform cannot guarantee secure catalog reads")


def _require_regular_single_link(
    metadata: os.stat_result,
    message: str,
    *,
    same_as: os.stat_result | None = None,
) -> None:
    """Raise ``ValueError(message)`` unless ``metadata`` is a regular, single-link file that is not a link.

    With ``same_as``, the file must also keep that snapshot's identity.
    """
    if (
        stat_is_link_or_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or getattr(metadata, "st_nlink", 1) != 1
        or (same_as is not None and not os.path.samestat(same_as, metadata))
    ):
        raise ValueError(message)
