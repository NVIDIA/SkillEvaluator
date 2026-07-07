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
import math
import os
import re
import secrets
import stat
import tempfile
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from skillevaluator.constants import (
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_WORKFLOWS,
    SIMILARITY_CRITICAL_THRESHOLD,
    SIMILARITY_HIGH_THRESHOLD,
    SIMILARITY_LOW_THRESHOLD,
    SIMILARITY_MEDIUM_THRESHOLD,
)
from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError, validate_embedding_vector
from skillevaluator.embedding.extractor import (
    MAX_COLLECTION_ENTRIES,
    MAX_MANIFEST_BYTES,
    ContentEntry,
    discover_and_extract,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Severity
from skillevaluator.utils.path_security import canonicalize_trusted_root_alias
from skillevaluator.utils.tier2_paths import safe_path_label

logger = get_logger(__name__)

CATALOG_SCHEMA_VERSION = 1
MAX_CATALOG_BYTES = 32 * 1024 * 1024
MAX_CATALOG_ENTRIES = 5_000
MAX_VECTOR_DIMENSION = 65_536
MAX_CATALOG_TEXT_LENGTH = 16_384
EMBEDDING_BATCH_SIZE = 64
MAX_DESCRIPTION_EMBEDDING_TEXT_CHARS = 16_384
MAX_FULL_BODY_EMBEDDING_TEXT_BYTES = MAX_MANIFEST_BYTES
MAX_PAIRWISE_COMPARISONS = MAX_COLLECTION_ENTRIES * (MAX_COLLECTION_ENTRIES - 1) // 2
MAX_SCALAR_COMPARISONS = 25_000_000
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

    def __init__(self, client: EmbeddingClient, *, full_body: bool = False) -> None:
        self._client = client
        self._full_body = full_body
        self._entries: dict[str, RegistryEntry] = {}
        self._vector_dimension: int | None = None

    @property
    def size(self) -> int:
        return len(self._entries)

    def build_from_directory(
        self,
        root: Path,
        content_type: str,
        *,
        minimum_entries: int = 1,
    ) -> int:
        """Discover content items, embed them, and populate the index.

        Returns:
            Number of entries successfully indexed.
        """
        content_entries = discover_and_extract(root, content_type)
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
        if self._full_body:
            vectors = [self._client.embed_chunked(text) for text in texts]
        else:
            vectors = []
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                vectors.extend(self._client.embed(texts[start : start + EMBEDDING_BATCH_SIZE]))
        if len(vectors) != len(content_entries):
            raise ValueError(f"Embedding provider returned {len(vectors)} vectors for {len(content_entries)} entries")

        resolved_root = root.resolve(strict=True)
        for entry, vector in zip(content_entries, vectors, strict=True):
            vector_dimension = _validate_vector(vector, self._vector_dimension)
            if self._vector_dimension is None:
                self._vector_dimension = vector_dimension
            resolved_entry = Path(entry.path).resolve(strict=True)
            try:
                relative_path = resolved_entry.relative_to(resolved_root).as_posix() or "."
            except ValueError as exc:
                raise ValueError(f"Discovered content path escapes scan root: {entry.path}") from exc
            entry_id = f"{entry.content_type}:{relative_path}"
            self._entries[entry_id] = RegistryEntry(
                name=entry.name,
                description=entry.description,
                path=relative_path,
                content_type=entry.content_type,
                embedding=vector,
                entry_id=entry_id,
                content_fingerprint=_fingerprint(entry.full_text if self._full_body else entry.embedding_text),
            )

        logger.debug("Indexed %d entries from %s", len(self._entries), safe_path_label(root))
        return len(self._entries)

    def find_duplicates(self, threshold: float) -> list[SimilarityMatch]:
        """Pairwise comparison of all indexed entries.

        Only pairs with cosine similarity >= threshold are returned,
        sorted by score descending.
        """
        _validate_threshold(threshold)
        entries = list(self._entries.values())
        comparison_count = len(entries) * (len(entries) - 1) // 2
        if comparison_count > MAX_PAIRWISE_COMPARISONS:
            raise ValueError(f"Pairwise comparison limit exceeded ({MAX_PAIRWISE_COMPARISONS})")
        vector_dimension = _validate_registry_vectors(entries, self._vector_dimension)
        _validate_scalar_work(comparison_count, vector_dimension)
        matches: list[SimilarityMatch] = []

        for a, b in combinations(entries, 2):
            score = EmbeddingClient.cosine_similarity(a.embedding, b.embedding)
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
        _validate_threshold(threshold)
        entries = list(self._entries.values())
        vector_dimension = _validate_registry_vectors(entries, self._vector_dimension)
        _validate_scalar_work(len(entries), vector_dimension)
        _validate_embedding_text(text, full_body=self._full_body)
        vector = self._client.embed_chunked(text) if self._full_body else self._client.embed_single(text)
        _validate_vector(vector, vector_dimension or self._vector_dimension)

        matches: list[SimilarityMatch] = []
        for entry in entries:
            score = EmbeddingClient.cosine_similarity(vector, entry.embedding)
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
        _validate_threshold(threshold)
        catalog_entries = list(self._entries.values())
        vector_dimension = _validate_registry_vectors(catalog_entries, self._vector_dimension)
        _validate_scalar_work(len(catalog_entries), vector_dimension)
        text = entry.full_text if self._full_body else entry.embedding_text
        _validate_embedding_text(text, full_body=self._full_body)
        vector = self._client.embed_chunked(text) if self._full_body else self._client.embed_single(text)
        _validate_vector(vector, vector_dimension or self._vector_dimension)
        target_path = Path(entry.path).name or "."

        matches: list[SimilarityMatch] = []
        for catalog_entry in catalog_entries:
            score = EmbeddingClient.cosine_similarity(vector, catalog_entry.embedding)
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
        """Persist a validated, versioned local embedding catalog."""
        if not self._entries:
            raise ValueError("Cannot save an empty catalog")
        if len(self._entries) > MAX_CATALOG_ENTRIES:
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

        if vector_dimension is None:
            raise ValueError("Cannot save a catalog without embedding vectors")
        data = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "provider": _client_provider(self._client),
            "model": self._client.model,
            "mode": "full-body" if self._full_body else "description",
            "endpoint_fingerprint": _client_endpoint_fingerprint(self._client),
            "vector_dimension": vector_dimension,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "entries": entries,
        }
        serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
        if len(serialized.encode("utf-8")) > MAX_CATALOG_BYTES:
            raise ValueError(f"Catalog size limit exceeded ({MAX_CATALOG_BYTES} bytes)")
        _write_catalog_atomically(catalog_path, serialized.encode("utf-8"))
        logger.debug("Saved local catalog to %s (%d entries)", safe_path_label(catalog_path), len(self._entries))

    def load_catalog(self, catalog_path: Path) -> None:
        """Load and validate a versioned local embedding catalog."""
        try:
            serialized = _read_catalog_text(catalog_path)
            raw = json.loads(
                serialized,
                object_pairs_hook=_catalog_object,
                parse_constant=_reject_json_constant,
            )
        except _DuplicateCatalogKeyError as exc:
            raise ValueError(f"Catalog JSON contains duplicate key: {exc.key}") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed catalog JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Catalog root must be a JSON object")
        _validate_exact_fields(raw, _CATALOG_ROOT_FIELDS, "Catalog root")

        schema_version = raw.get("schema_version")
        if type(schema_version) is not int or schema_version != CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported catalog schema version: {schema_version!r}; expected {CATALOG_SCHEMA_VERSION}"
            )
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
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("Catalog entries must be a non-empty list")
        if len(raw_entries) > MAX_CATALOG_ENTRIES:
            raise ValueError(f"Catalog entry limit exceeded ({MAX_CATALOG_ENTRIES})")

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

        self._entries = loaded
        self._vector_dimension = vector_dimension
        logger.debug("Loaded %d entries from local catalog %s", len(self._entries), safe_path_label(catalog_path))

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


def _validate_threshold(threshold: float) -> None:
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("Similarity threshold must be finite and within [0, 1]")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold must be finite and within [0, 1]")


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


def _validate_registry_vectors(entries: list[RegistryEntry], expected_dimension: int | None) -> int:
    dimension = expected_dimension
    for entry in entries:
        dimension = _validate_vector(entry.embedding, dimension)
    return dimension or 0


def _validate_scalar_work(comparison_count: int, vector_dimension: int) -> None:
    scalar_work = comparison_count * vector_dimension
    if scalar_work > MAX_SCALAR_COMPARISONS:
        raise ValueError(f"Scalar comparison work limit exceeded ({MAX_SCALAR_COMPARISONS}); requested {scalar_work}")


def _validate_catalog_identity(entry_id: object, path: object, content_type: object) -> None:
    path = _validate_catalog_string(path, "Catalog path")
    content_type = _validate_catalog_string(content_type, "Catalog content type")
    entry_id = _validate_catalog_string(entry_id, "Catalog entry id")
    if "\\" in path or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise ValueError(f"Catalog path must be relative: {path!r}")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != path:
        raise ValueError(f"Catalog path must be relative and normalized: {path!r}")
    if content_type not in _CATALOG_CONTENT_TYPES:
        raise ValueError(f"Catalog content type is unsupported: {content_type!r}")
    expected_id = f"{content_type}:{path}"
    if entry_id != expected_id:
        raise ValueError(f"Catalog entry id must match its relative path: expected {expected_id!r}")


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
        if _stat_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
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
                if _stat_is_link_or_reparse(child_metadata) or not stat.S_ISDIR(child_metadata.st_mode):
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
    if _stat_is_link_or_reparse(metadata):
        raise ValueError(f"Catalog path is a symlink or reparse point: {catalog_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Catalog path is not a regular file: {catalog_path}")
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
        if _stat_is_link_or_reparse(opened_metadata) or not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError("Catalog temporary path is not a regular file")
        current_metadata = os.stat(temporary_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _stat_is_link_or_reparse(current_metadata) or not os.path.samestat(opened_metadata, current_metadata):
            raise ValueError("Catalog temporary path changed during creation")

        _write_all(descriptor, payload)
        os.fsync(descriptor)
        current_metadata = os.stat(temporary_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _stat_is_link_or_reparse(current_metadata) or not os.path.samestat(opened_metadata, current_metadata):
            raise ValueError("Catalog temporary path changed while saving")
        _inspect_posix_catalog_file(parent_descriptor, name, catalog_path, missing_ok=True)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        published = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _stat_is_link_or_reparse(published) or not os.path.samestat(opened_metadata, published):
            raise ValueError("Catalog publication changed during atomic replacement")
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
    if _stat_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
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
        if _stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
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
    if _stat_is_link_or_reparse(metadata):
        raise ValueError(f"Catalog path is a symlink or reparse point: {catalog_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Catalog path is not a regular file: {catalog_path}")
    return metadata


def _windows_final_path(descriptor: int) -> Path:
    if os.name != "nt":
        raise OSError("Windows handle verification is unavailable on this platform")

    import ctypes
    import msvcrt
    from ctypes import wintypes

    get_final_path = ctypes.windll.kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_final_path(msvcrt.get_osfhandle(descriptor), buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), "Cannot resolve opened Windows catalog handle")
    path = buffer.value
    if path.startswith("\\\\?\\UNC\\"):
        path = "\\\\" + path[8:]
    elif path.startswith("\\\\?\\"):
        path = path[4:]
    return Path(path)


def _verify_windows_open_path(descriptor: int, catalog_path: Path) -> None:
    expected = os.path.normcase(os.fspath(catalog_path.absolute()))
    actual = os.path.normcase(os.fspath(_windows_final_path(descriptor).absolute()))
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
        if _stat_is_link_or_reparse(opened_metadata) or not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError("Catalog temporary path is not a regular file")
        current_metadata = temporary_path.lstat()
        if _stat_is_link_or_reparse(current_metadata) or not os.path.samestat(opened_metadata, current_metadata):
            raise ValueError("Catalog temporary path changed during creation")

        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        _validate_windows_catalog_parent(absolute, create=False)
        current_metadata = temporary_path.lstat()
        if _stat_is_link_or_reparse(current_metadata) or not os.path.samestat(opened_metadata, current_metadata):
            raise ValueError("Catalog temporary path changed while saving")
        _inspect_windows_catalog_file(absolute, missing_ok=True)
        temporary_path.replace(absolute)
        temporary_path = None
        published = absolute.lstat()
        if _stat_is_link_or_reparse(published) or not os.path.samestat(opened_metadata, published):
            raise ValueError("Catalog publication changed during atomic replacement")
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
    if _stat_is_link_or_reparse(opened_metadata) or not stat.S_ISREG(opened_metadata.st_mode):
        raise ValueError(f"Catalog path is not a regular file: {catalog_path}")
    if opened_metadata.st_size > MAX_CATALOG_BYTES:
        raise ValueError(f"Catalog size limit exceeded ({MAX_CATALOG_BYTES} bytes)")

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(65_536, MAX_CATALOG_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_CATALOG_BYTES:
            raise ValueError(f"Catalog size limit exceeded ({MAX_CATALOG_BYTES} bytes)")
    return b"".join(chunks).decode("utf-8")


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
        if (
            _stat_is_link_or_reparse(opened_metadata)
            or not stat.S_ISREG(opened_metadata.st_mode)
            or not os.path.samestat(before, opened_metadata)
        ):
            raise ValueError("Catalog changed or is not a regular file while being opened")
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
        if exc.errno in {errno.ELOOP, errno.EMLINK} or _is_link_or_reparse(absolute):
            raise ValueError(f"Catalog path is a symlink or reparse point: {catalog_path}") from exc
        raise ValueError(f"Unable to open catalog: {exc}") from exc

    try:
        opened_metadata = os.fstat(descriptor)
        if (
            _stat_is_link_or_reparse(opened_metadata)
            or not stat.S_ISREG(opened_metadata.st_mode)
            or not os.path.samestat(before, opened_metadata)
        ):
            raise ValueError("Catalog changed or is not a regular file while being opened")
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


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return _stat_is_link_or_reparse(metadata)


def _stat_is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
