# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in v1/v2 catalog construction, persistence and private bundle comparisons."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    SCAN_ARTIFACT_DIRS,
    SCAN_EXCLUDED_DIRS,
    SIMILARITY_DEFAULT_MAX_ENTRIES,
    SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    SIMILARITY_MAX_DISCOVERED_PATHS,
)
from skillevaluator.embedding import registry as core
from skillevaluator.embedding.client import normalize_embedding_vector, unit_vector_similarity
from skillevaluator.embedding.limits import validate_max_entries, validate_max_scalar_comparisons
from skillevaluator.plugin_manifest import canonical_manifest_relative, locate_plugin_manifest, manifest_root_for
from skillevaluator.utils.secure_fs import SecurePathError, discover_secure_files
from skillevaluator.utils.structured_data import preflight_json_structure

from .profile import MAX_PLUGIN_MEMBER_CHARS, MAX_PLUGIN_MEMBERS, load_plugin_profile

MAX_CATALOG_VECTOR_VALUES = 1_000_000
MAX_CATALOG_MEMBER_NAMES = 262_144
MAX_CATALOG_SOURCE_BYTES = CONTENT_DEDUP_MAX_TOTAL_BYTES

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
    entries: tuple[core.RegistryEntry, ...] = ()
    skipped_plugins: tuple[tuple[str, str], ...] = ()

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
        return cls._from_payload(raw, client=client, max_scalar_comparisons=max_scalar_comparisons)

    @classmethod
    def _from_payload(cls, raw, *, client, max_scalar_comparisons):
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
        all_entries = []
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
            projected = core.RegistryEntry(
                entry["name"],
                entry["description"],
                entry["path"],
                entry["content_type"],
                entry["embedding"],
                entry["id"],
                entry["content_fingerprint"],
            )
            all_entries.append(projected)
            if entry["content_type"] == "skill":
                skills.append(projected)
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
        return cls(tuple(skills), tuple(bundles), dimension, client, max_scalar_comparisons, tuple(all_entries))

    @classmethod
    def build(
        cls,
        root: Path,
        *,
        client,
        max_entries=SIMILARITY_DEFAULT_MAX_ENTRIES,
        max_scalar_comparisons=SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    ):
        """Build a description catalog from one bundle or a bounded local collection.

        Profiles come from the shared inventory. Source discovery, byte/entry
        admission and all input text checks finish before any embedding request.
        Invalid profiles remain visible in ``skipped_plugins``; unsafe inputs
        refuse the build. The first validated response bounds total vector work
        before any later batch. No public command invokes this private API.
        """
        validate_max_entries(max_entries)
        validate_max_scalar_comparisons(max_scalar_comparisons)
        roots = _catalog_roots(root, max_entries=max_entries)
        pending_skills = []
        pending_plugins = []
        skipped = []
        for candidate in roots:
            prefix = candidate.relative_to(root).as_posix()
            try:
                profile = load_plugin_profile(candidate)
            except SecurePathError:
                raise
            except ValueError:
                skipped.append((prefix, "Bundle profile was invalid or incomplete"))
                continue
            if profile.description is not None:
                pending_plugins.append(
                    PluginRegistryEntry(
                        f"plugin:{prefix}",
                        profile.name,
                        profile.description,
                        prefix,
                        profile.manifest,
                        profile.source_fingerprint,
                        profile.members,
                        [],
                    )
                )
            else:
                skipped.append((prefix, "Bundle description is absent; only comparable skills were indexed"))
            for skill in profile.bundled_skills:
                if skill.entry is None:
                    continue
                relative = (PurePosixPath(prefix) / skill.root_relative).as_posix()
                text = skill.entry.embedding_text
                pending_skills.append(
                    core.RegistryEntry(
                        skill.entry.name,
                        skill.entry.description,
                        relative,
                        "skill",
                        [],
                        f"skill:{relative}",
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                )
            if len(pending_skills) + len(pending_plugins) > max_entries:
                raise ValueError("Catalog entry limit exceeded before embedding")
        pending = [*pending_skills, *pending_plugins]
        if not pending:
            raise ValueError("No comparable local bundle or skill entries found")
        if sum(len(entry.members) for entry in pending_plugins) > MAX_CATALOG_MEMBER_NAMES:
            raise ValueError("Catalog aggregate member limit exceeded before embedding")
        for entry in pending:
            core._validate_text_fields(entry.entry_id, entry.name, entry.description)
            core._validate_catalog_identity(f"skill:{entry.path}", entry.path, "skill")
        for entry in pending_plugins:
            for member in entry.members:
                core._validate_catalog_string(member, "Catalog member")
        texts = [f"{entry.name}: {entry.description}" for entry in pending]
        for text in texts:
            core._validate_embedding_text(text, full_body=False)
        dimension = None
        vectors = []
        for start in range(0, len(texts), core.EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + core.EMBEDDING_BATCH_SIZE]
            response = client.embed(batch)
            if len(response) != len(batch):
                raise ValueError("Embedding provider returned an invalid vector count")
            for vector in response:
                dimension = core._validate_vector(vector, dimension)
            if len(pending) * dimension > MAX_CATALOG_VECTOR_VALUES:
                raise ValueError("Catalog vector scalar limit exceeded before later embedding batches")
            vectors.extend(response)
        skills = tuple(
            replace(entry, embedding=vector)
            for entry, vector in zip(pending_skills, vectors[: len(pending_skills)], strict=True)
        )
        plugins = tuple(
            replace(entry, embedding=vector)
            for entry, vector in zip(pending_plugins, vectors[len(pending_skills) :], strict=True)
        )
        catalog = cls(skills, plugins, dimension, client, max_scalar_comparisons, skills, tuple(skipped))
        # Reuse the load-time schema checks before returning a constructed index.
        cls._from_payload(catalog._payload(), client=client, max_scalar_comparisons=max_scalar_comparisons)
        return catalog

    def _payload(self):
        entries = []
        for entry in sorted(self.entries or self.skills, key=lambda value: value.entry_id):
            data = asdict(entry)
            data["id"] = data.pop("entry_id")
            entries.append(data)
        plugins = []
        for entry in sorted(self.plugins, key=lambda value: value.entry_id):
            data = asdict(entry)
            data["id"] = data.pop("entry_id")
            data["members"] = list(data["members"])
            plugins.append(data)
        data = {
            "schema_version": 2 if plugins else 1,
            "provider": core._client_provider(self.client),
            "model": self.client.model,
            "mode": "description",
            "endpoint_fingerprint": core._client_endpoint_fingerprint(self.client),
            "vector_dimension": self.vector_dimension,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "entries": entries,
        }
        if plugins:
            data["plugins"] = plugins
        return data

    def save(self, path: Path) -> None:
        """Validate the complete index before an atomic, no-follow local write."""
        data = self._payload()
        self._from_payload(data, client=self.client, max_scalar_comparisons=self.max_scalar_comparisons)
        serialized = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(serialized) > core.MAX_CATALOG_BYTES:
            raise ValueError("Catalog byte limit exceeded")
        core._write_catalog_atomically(path, serialized)

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


def _catalog_roots(root: Path, *, max_entries: int) -> list[Path]:
    # A single traversal bounds all candidate source bytes before profile reads.
    # Include artifact-named folders because a client can explicitly load them.
    files = discover_secure_files(
        root,
        selected=lambda _relative: True,
        excluded_dirs=SCAN_EXCLUDED_DIRS - SCAN_ARTIFACT_DIRS,
        max_paths=SIMILARITY_MAX_DISCOVERED_PATHS,
        allow_context_alias=False,
        refuse_selected_dirs=False,
    )
    total = 0
    candidates = set()
    for file in files:
        if file.metadata.st_size > CONTENT_DEDUP_MAX_FILE_BYTES:
            raise ValueError("Catalog source per-file byte limit exceeded")
        total += file.metadata.st_size
        if total > MAX_CATALOG_SOURCE_BYTES:
            raise ValueError("Catalog source byte limit exceeded before profile reads")
        parent = manifest_root_for(file.relative_path)
        if parent is not None:
            candidates.add(root / parent)
    selected = locate_plugin_manifest(root)
    if selected is not None:
        return [root]
    roots = []
    for candidate in sorted(candidates):
        if locate_plugin_manifest(candidate) is not None:
            roots.append(candidate)
        if len(roots) > max_entries:
            raise ValueError("Catalog entry limit exceeded before profile reads")
    return roots
