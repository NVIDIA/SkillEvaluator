# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in read-only v1/v2 catalog projection for private bundle comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from skillevaluator.constants import SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS
from skillevaluator.embedding import registry as core
from skillevaluator.embedding.client import normalize_embedding_vector, unit_vector_similarity
from skillevaluator.embedding.limits import validate_max_scalar_comparisons
from skillevaluator.plugin_manifest import canonical_manifest_relative
from skillevaluator.utils.structured_data import preflight_json_structure

from .profile import MAX_PLUGIN_MEMBER_CHARS, MAX_PLUGIN_MEMBERS

MAX_CATALOG_VECTOR_VALUES = 1_000_000
MAX_CATALOG_MEMBER_NAMES = 262_144

_PLUGIN_FIELDS = frozenset(
    {"id", "name", "description", "path", "manifest", "source_fingerprint", "members", "embedding"}
)


@dataclass(frozen=True)
class PluginRegistryEntry:
    entry_id: str
    name: str
    description: str
    path: str
    manifest: str
    source_fingerprint: str
    members: tuple[str, ...]
    embedding: list[float]


@dataclass(frozen=True)
class LocalCatalog:
    """Validated local entries; public registry acceptance and persistence stay v1."""

    skills: tuple[core.RegistryEntry, ...]
    plugins: tuple[PluginRegistryEntry, ...]
    vector_dimension: int
    client: object
    max_scalar_comparisons: int

    @classmethod
    def load(cls, path: Path, *, client, max_scalar_comparisons=SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS):
        validate_max_scalar_comparisons(max_scalar_comparisons)
        text = core._read_catalog_text(path)
        preflight_json_structure(
            text,
            max_tokens=2 * MAX_CATALOG_VECTOR_VALUES + 20 * core.MAX_CATALOG_ENTRIES + 2 * MAX_CATALOG_MEMBER_NAMES,
            max_collection_items=core.MAX_VECTOR_DIMENSION,
            max_mapping_items=2 * len(_PLUGIN_FIELDS),
            max_string_chars=core.MAX_CATALOG_TEXT_LENGTH,
        )
        try:
            raw = json.loads(text, object_pairs_hook=core._catalog_object, parse_constant=core._reject_json_constant)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Malformed local catalog JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("Catalog root must be an object")
        version = raw.get("schema_version")
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported private catalog schema version")
        core._validate_exact_fields(
            raw, core._CATALOG_ROOT_FIELDS | ({"plugins"} if version == 2 else set()), "Catalog root"
        )
        for field, expected in (
            ("provider", core._client_provider(client)),
            ("model", client.model),
            ("mode", "description"),
            ("endpoint_fingerprint", core._client_endpoint_fingerprint(client)),
        ):
            value = core._validate_catalog_string(raw[field], f"Catalog {field}")
            if value != expected:
                raise ValueError(f"Catalog {field} mismatch")
        created = core._validate_catalog_string(raw["created_at"], "Catalog created_at")
        if datetime.fromisoformat(created).tzinfo is None:
            raise ValueError("Catalog timestamp requires a timezone")
        dimension = raw["vector_dimension"]
        if type(dimension) is not int or not 1 <= dimension <= core.MAX_VECTOR_DIMENSION:
            raise ValueError("Invalid catalog vector dimension")
        entries, plugins = raw["entries"], raw.get("plugins", [])
        if not isinstance(entries, list) or not isinstance(plugins, list) or (not entries and not plugins):
            raise ValueError("Catalog entries must be non-empty lists")
        if len(entries) + len(plugins) > core.MAX_CATALOG_ENTRIES:
            raise ValueError("Catalog entry limit exceeded")
        if (len(entries) + len(plugins)) * dimension > MAX_CATALOG_VECTOR_VALUES:
            raise ValueError("Catalog vector scalar limit exceeded")
        skills = []
        bundles = []
        member_names = 0
        ids = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Catalog entry must be an object")
            core._validate_exact_fields(entry, core._CATALOG_ENTRY_FIELDS, "Catalog entry")
            core._validate_catalog_identity(entry["id"], entry["path"], entry["content_type"])
            core._validate_text_fields(entry["id"], entry["name"], entry["description"])
            _fingerprint(entry["content_fingerprint"])
            _admit_id(ids, entry["id"])
            core._validate_vector(entry["embedding"], dimension)
            if entry["content_type"] == "skill":
                skills.append(
                    core.RegistryEntry(
                        entry["name"],
                        entry["description"],
                        entry["path"],
                        "skill",
                        entry["embedding"],
                        entry["id"],
                        entry["content_fingerprint"],
                    )
                )
        for entry in plugins:
            if not isinstance(entry, dict):
                raise ValueError("Catalog bundle must be an object")
            core._validate_exact_fields(entry, _PLUGIN_FIELDS, "Catalog bundle")
            # Reuse the standard path/id validator with a skill prefix; only the private kind differs.
            path_value = core._validate_catalog_string(entry["path"], "Catalog path")
            core._validate_catalog_identity(f"skill:{path_value}", path_value, "skill")
            if entry["id"] != f"plugin:{path_value}":
                raise ValueError("Catalog bundle id/path mismatch")
            _admit_id(ids, entry["id"])
            core._validate_text_fields(entry["id"], entry["name"], entry["description"])
            manifest = core._validate_catalog_string(entry["manifest"], "Catalog manifest")
            if canonical_manifest_relative(manifest) is None:
                raise ValueError("Unsupported catalog bundle manifest")
            _fingerprint(entry["source_fingerprint"])
            members = entry["members"]
            if not isinstance(members, list) or len(members) > MAX_PLUGIN_MEMBERS:
                raise ValueError("Catalog bundle member limit exceeded")
            if any(
                not isinstance(m, str) or not m or len(m) > MAX_PLUGIN_MEMBER_CHARS or m != m.strip().casefold()
                for m in members
            ) or len(set(members)) != len(members):
                raise ValueError("Invalid catalog bundle members")
            member_names += len(members)
            if member_names > MAX_CATALOG_MEMBER_NAMES:
                raise ValueError("Catalog aggregate member limit exceeded")
            for member in members:
                core._validate_catalog_string(member, "Catalog member")
            core._validate_vector(entry["embedding"], dimension)
            bundles.append(
                PluginRegistryEntry(
                    entry["id"],
                    entry["name"],
                    entry["description"],
                    path_value,
                    manifest,
                    entry["source_fingerprint"],
                    tuple(members),
                    entry["embedding"],
                )
            )
        return cls(tuple(skills), tuple(bundles), dimension, client, max_scalar_comparisons)

    def score(self, texts: list[str], entries) -> list[list[tuple[object, float]]]:
        """Preflight the complete workload, normalize each catalog vector once, and batch targets."""
        if not texts or not entries:
            return [[] for _ in texts]
        core._validate_scalar_work(len(texts) * len(entries), self.vector_dimension, self.max_scalar_comparisons)
        units = [normalize_embedding_vector(e.embedding, self.vector_dimension) for e in entries]
        for text in texts:
            core._validate_embedding_text(text, full_body=False)
        vectors = []
        for start in range(0, len(texts), core.EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + core.EMBEDDING_BATCH_SIZE]
            response = self.client.embed(batch)
            if len(response) != len(batch):
                raise ValueError("Embedding provider returned an invalid vector count")
            vectors.extend(normalize_embedding_vector(v, self.vector_dimension) for v in response)
        return [
            [(entry, unit_vector_similarity(vector, unit)) for entry, unit in zip(entries, units, strict=True)]
            for vector in vectors
        ]


def _fingerprint(value):
    if not isinstance(value, str) or core._SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("Invalid catalog fingerprint")


def _admit_id(ids, value):
    if value in ids:
        raise ValueError("Duplicate catalog entry id")
    ids.add(value)
