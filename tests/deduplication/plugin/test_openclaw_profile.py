# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenClaw bundles in the private local-catalog comparisons (Checks A, B, C-intra and C-inter)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skillevaluator.deduplication.plugin import commands
from skillevaluator.deduplication.plugin.catalog import LocalCatalog
from skillevaluator.deduplication.plugin.catalog_checks import check_catalog_skills
from skillevaluator.deduplication.plugin.profile import PLUGIN_NAME_MAX_REPORT_CHARS, load_plugin_profile
from skillevaluator.utils.secure_fs import SecurePathError

MANIFEST = "openclaw.plugin.json"


@pytest.fixture(autouse=True)
def forbid_live_providers(monkeypatch):
    from skillevaluator.embedding.client import EmbeddingClient
    from skillevaluator.inference.client import LLMClient

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Live provider calls are forbidden in these tests")

    monkeypatch.setattr(EmbeddingClient, "embed", forbidden)
    monkeypatch.setattr(EmbeddingClient, "embed_chunked", forbidden)
    monkeypatch.setattr(LLMClient, "extract_json_from_response", forbidden)


class FakeEmbedding:
    model = "fixture-model"

    def __init__(self):
        self.calls: list[list[str]] = []

    def _resolved_config(self):
        return SimpleNamespace(provider="openai", base_url=None)

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_single(self, text):
        return self.embed([text])[0]


def _skill(name: str) -> str:
    return f"---\nname: {name}\ndescription: {name} guidance for releases.\n---\nBody.\n"


def _tree(root: Path, files: dict[str, Any]) -> Path:
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return root


_OMIT = object()


def _openclaw(root: Path, files: dict[str, Any] | None = None, **fields: Any) -> Path:
    manifest = {"id": "release-helper", "configSchema": {"type": "object"}, **fields}
    manifest = {key: value for key, value in manifest.items() if value is not _OMIT}
    return _tree(root, {MANIFEST: manifest, **(files or {})})


def _catalog(tmp_path: Path, *, skills=(), plugins=()) -> LocalCatalog:
    from skillevaluator.embedding.registry import _client_endpoint_fingerprint

    client = FakeEmbedding()
    payload = {
        "schema_version": 2,
        "provider": "openai",
        "model": client.model,
        "mode": "description",
        "endpoint_fingerprint": _client_endpoint_fingerprint(client),
        "vector_dimension": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "entries": list(skills),
        "plugins": list(plugins),
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return LocalCatalog.load(path, client=client)


def _skill_entry(path: str, fingerprint: str) -> dict[str, Any]:
    return {
        "id": f"skill:{path}",
        "name": "catalog-skill",
        "description": "Catalog skill",
        "path": path,
        "content_type": "skill",
        "content_fingerprint": fingerprint,
        "embedding": [1.0, 0.0],
    }


def _fingerprint(profile, root_relative: str) -> str:
    [skill] = [skill for skill in profile.bundled_skills if skill.root_relative == root_relative]
    return hashlib.sha256(skill.entry.embedding_text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Profile identity and members                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("display_name", [_OMIT, "Readable Display Name"])
def test_profile_identity_is_the_manifest_id(tmp_path: Path, display_name: Any) -> None:
    root = _openclaw(tmp_path / "folder-name", name=display_name)
    profile = load_plugin_profile(root)
    assert profile.name == "release-helper"
    assert profile.manifest == MANIFEST


@pytest.mark.parametrize("identity", [_OMIT, None, "", "   ", 17, [], "x" * (PLUGIN_NAME_MAX_REPORT_CHARS + 1)])
def test_profile_requires_a_bounded_nonempty_id_without_folder_fallback(tmp_path: Path, identity: Any) -> None:
    root = _openclaw(tmp_path / "folder-name", id=identity, name="Display Name")
    with pytest.raises(ValueError) as caught:
        load_plugin_profile(root)
    assert not isinstance(caught.value, SecurePathError)


@pytest.mark.parametrize("schema", [_OMIT, None, [], "schema", False])
def test_profile_requires_a_config_schema_object(tmp_path: Path, schema: Any) -> None:
    with pytest.raises(ValueError, match="configSchema"):
        load_plugin_profile(_openclaw(tmp_path, configSchema=schema))


def test_profile_members_come_only_from_declared_roots(tmp_path: Path) -> None:
    files = {
        "skills/implicit/SKILL.md": _skill("implicit"),
        "custom/a/SKILL.md": _skill("alpha"),
        "custom/group/b/SKILL.md": _skill("beta"),
        "SKILL.md": _skill("root-skill"),
    }
    profile = load_plugin_profile(_openclaw(tmp_path, files, skills=["custom", "."]))
    assert [skill.root_relative for skill in profile.bundled_skills] == [".", "custom/a", "custom/group/b"]
    assert profile.members == ("alpha", "beta", "root-skill")


@pytest.mark.parametrize("fields", [{}, {"skills": []}])
def test_omitted_or_empty_skills_have_no_members(tmp_path: Path, fields: dict) -> None:
    profile = load_plugin_profile(_openclaw(tmp_path, {"skills/implicit/SKILL.md": _skill("implicit")}, **fields))
    assert profile.bundled_skills == ()
    assert profile.members == ()


def test_catalog_paths_are_relative_to_every_declared_root(tmp_path: Path) -> None:
    files = {"custom/a/SKILL.md": _skill("alpha"), "direct/SKILL.md": _skill("direct")}
    profile = load_plugin_profile(_openclaw(tmp_path, files, skills=["./custom", "custom/a", "direct"]))
    paths = {skill.root_relative: skill.catalog_relative_paths for skill in profile.bundled_skills}
    assert paths == {"custom/a": (".", "a"), "direct": (".",)}


# --------------------------------------------------------------------------- #
# C-inter self identity                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("root", "relative"), [("custom", "a"), ("custom/a", "."), (".", "custom/a")])
def test_skill_only_catalog_built_from_a_declared_root_is_self(tmp_path: Path, root: str, relative: str) -> None:
    plugin = _openclaw(tmp_path / "plugin", {"custom/a/SKILL.md": _skill("alpha")}, skills=[root])
    profile = load_plugin_profile(plugin)
    catalog = _catalog(tmp_path, skills=[_skill_entry(relative, _fingerprint(profile, "custom/a"))])
    result = check_catalog_skills(profile, catalog, threshold=0.5)
    assert not result.findings
    assert result.metadata["plugin"]["catalog_skill_similarity"]["status"] == "compared"
    [success] = [row for row in result.success_details if row.check_name == "skill_catalog_comparison"]
    assert success.metadata["self_matches_excluded"] == 1
    assert success.metadata["excluded_catalog_paths"] == [relative]


def test_same_content_elsewhere_or_changed_content_at_self_path_is_compared(tmp_path: Path) -> None:
    plugin = _openclaw(tmp_path / "plugin", {"custom/a/SKILL.md": _skill("alpha")}, skills=["custom"])
    profile = load_plugin_profile(plugin)
    fingerprint = _fingerprint(profile, "custom/a")
    catalog = _catalog(tmp_path, skills=[_skill_entry("other/a", fingerprint), _skill_entry("a", "0" * 64)])
    result = check_catalog_skills(profile, catalog, threshold=0.5)
    assert sorted(finding.metadata["catalog_path"] for finding in result.findings) == ["a", "other/a"]


# --------------------------------------------------------------------------- #
# Composed scan                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fields",
    [{"id": _OMIT}, {"id": ""}, {"configSchema": _OMIT}, {"configSchema": "schema"}],
)
def test_malformed_native_identity_is_an_incomplete_comparison_not_a_security_failure(
    tmp_path: Path, monkeypatch, fields: dict
) -> None:
    monkeypatch.setattr(commands, "run_plugin_skill_context_dedup", lambda *_a, **_k: pytest.fail("no paid work"))
    plugin = _openclaw(tmp_path, {"custom/a/SKILL.md": _skill("alpha")}, skills=["custom"], **fields)
    results = commands.run_plugin_dedup_scan(plugin)
    assert len(results) == 4
    assert not any(result.metadata.get("security_failure") for result in results)
    assert results[1].errors
    assert all(
        result.metadata.get("skipped") and result.metadata.get("optional_skip") is False for result in results[2:]
    )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_linked_declared_root_remains_a_security_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(commands, "run_plugin_skill_context_dedup", lambda *_a, **_k: pytest.fail("no paid work"))
    outside = _tree(tmp_path / "outside", {"a/SKILL.md": _skill("alpha")})
    plugin = _openclaw(tmp_path / "plugin", skills=["linked"])
    try:
        (plugin / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    results = commands.run_plugin_dedup_scan(plugin)
    assert results[1].metadata.get("security_failure")


def test_context_comparison_uses_the_declared_roots(tmp_path: Path, monkeypatch) -> None:
    seen: list[list[str]] = []

    def context(profile, **_kwargs):
        seen.append([skill.root_relative for skill in profile.bundled_skills])
        return []

    monkeypatch.setattr(commands, "run_plugin_skill_context_dedup", context)
    files = {"skills/implicit/SKILL.md": _skill("implicit"), "portable/deploy/SKILL.md": _skill("deploy")}
    commands.run_plugin_dedup_scan(_openclaw(tmp_path, files, skills=["portable"]))
    assert seen == [["portable/deploy"]]


@pytest.mark.parametrize("skills", ["./custom", None, {}, [1]])
def test_malformed_skill_root_declarations_never_reach_comparisons(tmp_path: Path, monkeypatch, skills: Any) -> None:
    monkeypatch.setattr(commands, "run_plugin_skill_context_dedup", lambda *_a, **_k: pytest.fail("no paid work"))
    plugin = _openclaw(tmp_path, {"custom/a/SKILL.md": _skill("alpha")}, skills=skills)
    with pytest.raises(ValueError) as caught:
        load_plugin_profile(plugin)
    assert not isinstance(caught.value, SecurePathError)
    results = commands.run_plugin_dedup_scan(plugin)
    assert not any(result.metadata.get("security_failure") for result in results)
    assert results[1].errors
    assert all(
        result.metadata.get("skipped") and result.metadata.get("optional_skip") is False for result in results[2:]
    )


def test_unnamed_root_skill_is_labelled_like_its_inventory_row(tmp_path: Path) -> None:
    plugin = _openclaw(tmp_path / "root-folder", {"SKILL.md": "---\ndescription: No name.\n---\nBody.\n"}, skills=["."])
    profile = load_plugin_profile(plugin)
    assert profile.members == ("root-folder",)


def test_built_catalog_over_openclaw_plugins_records_declared_paths_and_excludes_self(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    files = {"custom/a/SKILL.md": _skill("alpha"), "skills/implicit/SKILL.md": _skill("implicit")}
    plugin = _openclaw(collection / "release", files, description="Release guidance.", skills=["custom"])
    _openclaw(collection / "other", {"portable/b/SKILL.md": _skill("beta")}, id="other", skills=["portable"])
    catalog = LocalCatalog.build(collection, client=FakeEmbedding())
    assert sorted(entry.path for entry in catalog.skills) == ["other/portable/b", "release/custom/a"]
    assert [entry.path for entry in catalog.plugins] == ["release"]
    result = check_catalog_skills(load_plugin_profile(plugin), catalog, threshold=0.5)
    assert [finding.metadata["catalog_path"] for finding in result.findings] == ["other/portable/b"]
    [success] = [row for row in result.success_details if row.check_name == "skill_catalog_comparison"]
    assert success.metadata["excluded_catalog_paths"] == ["release/custom/a"]
