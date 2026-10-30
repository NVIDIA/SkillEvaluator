# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline selection and frozen-source checks for group-workspace decoys."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.tier3.decoys import load_decoy_manifest, select_decoys, write_decoy_manifest


def _skill(root: Path, name: str, *, skill_name: str | None = None) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {skill_name or path.name}\ndescription: Help with {path.name}.\n---\n\n# {path.name}\n",
        encoding="utf-8",
    )
    return path


def _client(vectors: list[list[float]] | None = None) -> EmbeddingClient:
    client = MagicMock(spec=EmbeddingClient)
    client.model = "test-model"
    client._resolved_config.return_value = SimpleNamespace(provider="nv_build", base_url="https://example.com/v1")
    client.embed.return_value = vectors or [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
    client.embed_single.return_value = [1.0, 0.0]
    return client


def _catalog(root: Path, path: Path, client: EmbeddingClient) -> None:
    registry = EmbeddingRegistry(client)
    registry.build_from_directory(root, "skill")
    registry.save_catalog(path)


@pytest.fixture
def selection(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "catalog-skills"
    skills = [_skill(root, name) for name in ("api-debugger", "log-triage", "release-planner")]
    target = _skill(tmp_path, "target")
    catalog = tmp_path / "catalog.json"
    client = _client()
    _catalog(root, catalog, client)
    return SimpleNamespace(root=root, target=target, catalog=catalog, client=client, skills=skills)


def _select(selection: SimpleNamespace, **kwargs: object) -> dict:
    return select_decoys(
        selection.target,
        catalog_path=selection.catalog,
        catalog_root=selection.root,
        client=selection.client,
        count=kwargs.pop("count", 2),
        **kwargs,
    )


def test_selects_ranked_skills_and_records_complete_provenance(selection: SimpleNamespace) -> None:
    manifest = _select(selection)
    assert [item["path"] for item in manifest["selected"]] == ["api-debugger", "log-triage"]
    assert [item["score"] for item in manifest["selected"]] == [1.0, 0.8]
    assert manifest["catalog"]["sha256"] == hashlib.sha256(selection.catalog.read_bytes()).hexdigest()
    assert manifest["catalog"]["model"] == "test-model"
    assert manifest["catalog"]["mode"] == "description"
    assert (
        manifest["target"]["manifest_sha256"]
        == hashlib.sha256((selection.target / "SKILL.md").read_bytes()).hexdigest()
    )
    assert all(len(item["source_digest"]) == 64 for item in manifest["selected"])


def test_ties_and_reordered_catalog_are_repeatable(selection: SimpleNamespace) -> None:
    raw = json.loads(selection.catalog.read_text())
    for item in raw["entries"]:
        item["embedding"] = [1.0, 0.0]
    selection.catalog.write_text(json.dumps(raw))
    first = _select(selection)
    raw["entries"].reverse()
    selection.catalog.write_text(json.dumps(raw))
    second = _select(selection)
    assert first["selected"] == second["selected"]
    assert [item["path"] for item in first["selected"]] == ["api-debugger", "log-triage"]


def test_explicit_exclusion_is_respected(selection: SimpleNamespace) -> None:
    manifest = _select(selection, exclude=("api-debugger",))
    assert [item["path"] for item in manifest["selected"]] == ["log-triage", "release-planner"]
    assert manifest["exclude"] == ["api-debugger"]


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 1001])
def test_rejects_invalid_count_before_model_calls(selection: SimpleNamespace, count: object) -> None:
    with pytest.raises(ValueError, match="count"):
        _select(selection, count=count)
    selection.client.embed_single.assert_not_called()


def test_insufficient_candidates_never_silently_reduces_count(selection: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="Requested 4 decoys but only 3"):
        _select(selection, count=4)


def test_unknown_exclusion_is_a_configuration_error(selection: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="absent from catalog"):
        _select(selection, exclude=("typo",))
    selection.client.embed_single.assert_not_called()


def test_stale_catalog_description_fails(selection: SimpleNamespace) -> None:
    path = selection.skills[0] / "SKILL.md"
    path.write_text(path.read_text().replace("Help with", "Changed description for"))
    with pytest.raises(ValueError, match="stale"):
        _select(selection)


def test_excludes_target_basename_metadata_identity_and_copied_instructions(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    target = _skill(root, "target")
    _skill(root, "a-collision", skill_name="target")
    clone = _skill(root, "b-clone")
    (clone / "SKILL.md").write_text((target / "SKILL.md").read_text().replace("name: target", "name: b-clone"))
    _skill(root, "c-good")
    client = _client([[1.0, 0.0]] * 4)
    catalog = tmp_path / "catalog.json"
    _catalog(root, catalog, client)
    manifest = select_decoys(target, catalog_path=catalog, catalog_root=root, count=1, client=client)
    assert manifest["selected"][0]["path"] == "c-good"


def test_deduplicates_directory_basename_across_nested_catalog(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _skill(root, "a/shared", skill_name="one")
    _skill(root, "b/shared", skill_name="two")
    _skill(root, "c-other")
    target = _skill(tmp_path, "target")
    client = _client([[1.0, 0.0]] * 3)
    catalog = tmp_path / "catalog.json"
    _catalog(root, catalog, client)
    manifest = select_decoys(target, catalog_path=catalog, catalog_root=root, count=2, client=client)
    assert [item["path"] for item in manifest["selected"]] == ["a/shared", "c-other"]


def test_round_trip_load_requires_no_catalog_or_model(selection: SimpleNamespace, tmp_path: Path) -> None:
    manifest = _select(selection)
    path = tmp_path / "decoys.json"
    write_decoy_manifest(manifest, path)
    selection.catalog.unlink()
    selection.client.reset_mock()
    paths, loaded = load_decoy_manifest(selection.target, path)
    assert paths == selection.skills[:2]
    assert loaded == manifest
    selection.client.embed_single.assert_not_called()


@pytest.mark.parametrize("filename", ["SKILL.md", "scripts/helper.py", "evals/evals.json"])
def test_source_changes_invalidate_frozen_selection(
    selection: SimpleNamespace,
    tmp_path: Path,
    filename: str,
) -> None:
    path = tmp_path / "decoys.json"
    write_decoy_manifest(_select(selection), path)
    changed = selection.skills[0] / filename
    changed.parent.mkdir(exist_ok=True)
    with changed.open("a") as stream:
        stream.write("\nchanged\n")
    with pytest.raises(ValueError, match="changed"):
        load_decoy_manifest(selection.target, path)


def test_target_instruction_changes_invalidate_selection(selection: SimpleNamespace, tmp_path: Path) -> None:
    path = tmp_path / "decoys.json"
    write_decoy_manifest(_select(selection), path)
    with (selection.target / "SKILL.md").open("a") as stream:
        stream.write("\nNew instructions.\n")
    with pytest.raises(ValueError, match="target changed"):
        load_decoy_manifest(selection.target, path)


def test_manifest_can_live_in_target_evals_without_self_invalidating(selection: SimpleNamespace) -> None:
    path = selection.target / "evals" / "decoys.json"
    write_decoy_manifest(_select(selection), path)
    assert load_decoy_manifest(selection.target, path)[0] == selection.skills[:2]


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../api-debugger", "a\\b", "a//b"])
def test_rejects_escaping_or_noncanonical_manifest_paths(
    selection: SimpleNamespace,
    tmp_path: Path,
    relative: str,
) -> None:
    manifest = _select(selection)
    manifest["selected"][0]["path"] = relative
    path = tmp_path / "decoys.json"
    write_decoy_manifest(manifest, path)
    with pytest.raises(ValueError, match="relative"):
        load_decoy_manifest(selection.target, path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(schema_version=True),
        lambda data: data.update(strategy="unknown"),
        lambda data: data.update(source_digest_algorithm="unknown"),
        lambda data: data.update(unexpected=True),
        lambda data: data.update(count=3),
        lambda data: data["selected"][0].update(score=True),
        lambda data: data["selected"][0].update(score=10**1000),
        lambda data: data["selected"][0].update(source_digest="invalid"),
        lambda data: data["selected"][0].update(name="wrong-name"),
    ],
)
def test_rejects_malformed_manifests(selection: SimpleNamespace, tmp_path: Path, mutate: object) -> None:
    manifest = _select(selection)
    mutate(manifest)
    path = tmp_path / "decoys.json"
    write_decoy_manifest(manifest, path)
    with pytest.raises(ValueError):
        load_decoy_manifest(selection.target, path)


def test_rejects_duplicate_selection(selection: SimpleNamespace, tmp_path: Path) -> None:
    manifest = _select(selection)
    manifest["selected"][1] = copy.deepcopy(manifest["selected"][0])
    path = tmp_path / "decoys.json"
    write_decoy_manifest(manifest, path)
    with pytest.raises(ValueError, match="duplicate"):
        load_decoy_manifest(selection.target, path)


def test_rejects_duplicate_json_keys(selection: SimpleNamespace, tmp_path: Path) -> None:
    path = tmp_path / "decoys.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError):
        load_decoy_manifest(selection.target, path)


def test_rejects_unsafe_decoy_sources(selection: SimpleNamespace, tmp_path: Path) -> None:
    (selection.skills[0] / "linked").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="unsafe or unstable"):
        _select(selection)


def test_rejects_replaced_skill_directory_symlink(selection: SimpleNamespace, tmp_path: Path) -> None:
    path = tmp_path / "decoys.json"
    write_decoy_manifest(_select(selection), path)
    skill = selection.skills[0]
    moved = tmp_path / "moved"
    skill.rename(moved)
    skill.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_decoy_manifest(selection.target, path)


def test_writer_does_not_follow_output_symlinks(selection: SimpleNamespace, tmp_path: Path) -> None:
    destination = tmp_path / "existing.json"
    destination.write_text("keep me")
    link = tmp_path / "decoys.json"
    link.symlink_to(destination)
    with pytest.raises(ValueError, match="symlink"):
        write_decoy_manifest(_select(selection), link)
    assert destination.read_text() == "keep me"


@pytest.mark.parametrize("relative", ["references/renamed.md", "nested/package/SKILL.md"])
def test_excludes_target_payload_in_other_runtime_files(selection: SimpleNamespace, relative: str) -> None:
    copied = selection.skills[0] / relative
    copied.parent.mkdir(parents=True)
    copied.write_bytes((selection.target / "SKILL.md").read_bytes())
    manifest = _select(selection)
    assert [item["path"] for item in manifest["selected"]] == ["log-triage", "release-planner"]


def test_rejects_mixed_content_catalog_before_model_calls(selection: SimpleNamespace) -> None:
    raw = json.loads(selection.catalog.read_text())
    rule = copy.deepcopy(raw["entries"][0])
    rule.update(id="rules:api-debugger", content_type="rules")
    raw["entries"].append(rule)
    selection.catalog.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="skill-only"):
        _select(selection)
    selection.client.embed_single.assert_not_called()
