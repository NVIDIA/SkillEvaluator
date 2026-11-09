# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline selection and frozen-source checks for group-workspace decoys."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.tier3.decoy_identity import DECOY_SOURCE_DIGEST_ALGORITHM, decoy_source_digest
from skillevaluator.tier3.decoys import (
    _catalog_root_record,
    load_decoy_manifest,
    select_decoys,
    write_decoy_manifest,
)


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


def _selected_record(path: Path, root: Path) -> dict:
    digest = decoy_source_digest(path)
    assert digest is not None
    return {
        "path": path.relative_to(root).as_posix(),
        "name": path.name,
        "skill_name": path.name,
        "score": 1.0,
        "source_digest": digest,
    }


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
    assert manifest["source_digest_algorithm"] == DECOY_SOURCE_DIGEST_ALGORITHM


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


def test_excludes_skills_nested_inside_the_target(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    target = _skill(root, "a-target")
    _skill(root, "a-target/companions/rollback-helper")
    _skill(root, "z-other")
    client = _client([[1.0, 0.0]] * 3)
    catalog = tmp_path / "catalog.json"
    _catalog(root, catalog, client)

    manifest = select_decoys(target, catalog_path=catalog, catalog_root=root, count=1, client=client)

    assert manifest["selected"][0]["path"] == "z-other"


def test_loader_rejects_a_hand_edited_nested_target_selection(selection: SimpleNamespace, tmp_path: Path) -> None:
    manifest = _select(selection, count=1)
    nested = _skill(selection.target, "companions/rollback-helper")
    manifest["catalog_root"] = "."
    manifest["selected"] = [_selected_record(nested, selection.target.parent)]
    path = tmp_path / "nested.json"
    write_decoy_manifest(manifest, path)

    with pytest.raises(ValueError, match="nested inside the target"):
        load_decoy_manifest(selection.target, path)


def test_portable_manifest_uses_the_evaluated_checkout(tmp_path: Path) -> None:
    original = tmp_path / "checkout-a"
    root = original / "skills"
    target = _skill(root, "target")
    neighbor = _skill(root, "neighbor")
    client = _client([[1.0, 0.0]] * 2)
    catalog = original / "catalog.json"
    _catalog(root, catalog, client)
    manifest = select_decoys(target, catalog_path=catalog, catalog_root=root, count=1, client=client)
    assert manifest["catalog_root"] == "."
    saved = target / "evals" / "decoys.json"
    write_decoy_manifest(manifest, saved)
    copied = tmp_path / "checkout-b"
    shutil.copytree(original, copied)
    copied_target = copied / "skills" / "target"
    copied_neighbor = copied / "skills" / "neighbor"
    copied_manifest = copied_target / "evals" / "decoys.json"
    client.reset_mock()

    # The original still exists, but this checkout must read its own sources.
    assert load_decoy_manifest(copied_target, copied_manifest)[0] == [copied_neighbor]
    original_text = (neighbor / "SKILL.md").read_text(encoding="utf-8")
    (copied_neighbor / "SKILL.md").write_text(original_text + "\nChanged only in checkout B.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Selected decoy source changed"):
        load_decoy_manifest(copied_target, copied_manifest)
    (copied_neighbor / "SKILL.md").write_text(original_text, encoding="utf-8")
    shutil.rmtree(original)

    assert load_decoy_manifest(copied_target, copied_manifest)[0] == [copied_neighbor]
    client.embed.assert_not_called()
    client.embed_single.assert_not_called()


def test_collection_root_can_be_relative_to_a_nested_targets_parent(tmp_path: Path) -> None:
    root = tmp_path / "collection-a"
    target = _skill(root, "category/target")
    _skill(root, "neighbor")
    client = _client([[1.0, 0.0]] * 2)
    catalog = tmp_path / "catalog.json"
    _catalog(root, catalog, client)
    manifest = select_decoys(target, catalog_path=catalog, catalog_root=root, count=1, client=client)
    assert manifest["catalog_root"] == ".."
    write_decoy_manifest(manifest, target / "evals" / "decoys.json")
    relocated = tmp_path / "collection-b"
    shutil.copytree(root, relocated)
    shutil.rmtree(root)

    assert load_decoy_manifest(
        relocated / "category" / "target", relocated / "category" / "target" / "evals" / "decoys.json"
    )[0] == [relocated / "neighbor"]


def _case_insensitive(path: Path) -> bool:
    probe = path / "CaseProbe"
    probe.mkdir()
    try:
        return (path / "caseprobe").exists()
    finally:
        probe.rmdir()


@pytest.mark.parametrize(
    ("root_spelling", "target_spelling", "target_name", "expected"),
    [
        ("Skills", "skills/target", "target", "."),
        ("skills", "Skills/target", "target", "."),
        ("Skills", "skills/category/target", "category/target", ".."),
    ],
)
def test_case_variant_spellings_keep_a_portable_root(
    tmp_path: Path, root_spelling: str, target_spelling: str, target_name: str, expected: str
) -> None:
    if not _case_insensitive(tmp_path):
        pytest.skip("requires a case-insensitive filesystem")
    original = tmp_path / "checkout-a"
    root = original / "skills"
    _skill(root, target_name)
    _skill(root, "neighbor")
    client = _client([[1.0, 0.0]] * 2)
    catalog = original / "catalog.json"
    _catalog(root, catalog, client)
    target = original.joinpath(*target_spelling.split("/"))
    manifest = select_decoys(
        target, catalog_path=catalog, catalog_root=original / root_spelling, count=1, client=client
    )
    assert manifest["catalog_root"] == expected
    write_decoy_manifest(manifest, target / "evals" / "decoys.json")
    copied = tmp_path / "checkout-b"
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    copied_target = copied.joinpath(*target_spelling.split("/"))

    # The manifest no longer names checkout-a, so it loads from the copy alone.
    loaded = load_decoy_manifest(copied_target, copied_target / "evals" / "decoys.json")[0]
    assert len(loaded) == 1
    assert loaded[0].samefile(copied / "skills" / "neighbor")


def test_catalog_root_spellings_are_matched_by_file_identity(tmp_path: Path) -> None:
    # A linked spelling takes the same identity match as a case variant on a
    # case-insensitive filesystem, so this also runs on case-sensitive CI.
    root = tmp_path / "skills"
    target = _skill(root, "category/target")
    alias = tmp_path / "skills-alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available")

    assert _catalog_root_record(alias, root / "target") == "."
    assert _catalog_root_record(alias, target) == ".."
    assert _catalog_root_record(alias / "category", root / "target") == "category"
    assert _catalog_root_record(tmp_path / "elsewhere", root / "target") == str(tmp_path / "elsewhere")


def test_unrelated_external_collection_keeps_an_absolute_root(tmp_path: Path) -> None:
    target = _skill(tmp_path / "targets", "target")
    root = tmp_path / "external-collection"
    neighbor = _skill(root, "neighbor")
    client = _client([[1.0, 0.0]])
    catalog = tmp_path / "catalog.json"
    _catalog(root, catalog, client)
    manifest = select_decoys(target, catalog_path=catalog, catalog_root=root, count=1, client=client)
    assert manifest["catalog_root"] == str(root)
    saved = target / "evals" / "decoys.json"
    write_decoy_manifest(manifest, saved)

    assert load_decoy_manifest(target, saved)[0] == [neighbor]


def test_loader_accepts_an_explicit_absolute_catalog_root(selection: SimpleNamespace, tmp_path: Path) -> None:
    manifest = _select(selection)
    manifest["catalog_root"] = str(selection.root)
    path = tmp_path / "absolute.json"
    write_decoy_manifest(manifest, path)

    assert load_decoy_manifest(selection.target, path)[0] == selection.skills[:2]


@pytest.mark.parametrize("root_value", ["linked", "linked/../catalog-skills"])
def test_relative_catalog_root_cannot_hide_a_symlink(
    selection: SimpleNamespace,
    tmp_path: Path,
    root_value: str,
) -> None:
    manifest = _select(selection)
    (tmp_path / "linked").symlink_to(selection.root, target_is_directory=True)
    manifest["catalog_root"] = root_value
    path = tmp_path / "linked-root.json"
    write_decoy_manifest(manifest, path)

    with pytest.raises(ValueError, match="symlink"):
        load_decoy_manifest(selection.target, path)


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


@pytest.mark.parametrize("filename", ["SKILL.md", "scripts/helper.py", "references/guide.md"])
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


@pytest.mark.parametrize("filename", ["evals.json", "config.yml", "decoys.json"])
def test_evaluator_only_changes_preserve_frozen_selection(
    selection: SimpleNamespace,
    tmp_path: Path,
    filename: str,
) -> None:
    path = tmp_path / "decoys.json"
    write_decoy_manifest(_select(selection), path)
    changed = selection.skills[0] / "evals" / filename
    changed.parent.mkdir()
    changed.write_text("changed evaluator input\n", encoding="utf-8")

    assert load_decoy_manifest(selection.target, path)[0] == selection.skills[:2]


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


def test_mutually_selected_skills_keep_valid_manifests_after_evaluator_updates(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    alpha = _skill(root, "alpha")
    beta = _skill(root, "beta")
    gamma = _skill(root, "gamma")
    client = _client([[1.0, 0.0]] * 3)
    catalog = tmp_path / "catalog.json"
    _catalog(root, catalog, client)
    alpha_manifest = alpha / "evals" / "decoys.json"
    beta_manifest = beta / "evals" / "decoys.json"

    def save(target: Path, path: Path) -> None:
        write_decoy_manifest(
            select_decoys(target, catalog_path=catalog, catalog_root=root, count=2, client=client), path
        )

    save(alpha, alpha_manifest)
    save(beta, beta_manifest)
    for target in (alpha, beta):
        (target / "evals" / "config.yml").write_text(
            f"schema_version: 1\nskill_workspace:\n  mode: group\n  decoy_manifest: {target.name}/evals/decoys.json\n",
            encoding="utf-8",
        )
    assert load_decoy_manifest(alpha, alpha_manifest)[0] == [beta, gamma]
    assert load_decoy_manifest(beta, beta_manifest)[0] == [alpha, gamma]
    save(alpha, alpha_manifest)
    assert load_decoy_manifest(alpha, alpha_manifest)[0] == [beta, gamma]
    assert load_decoy_manifest(beta, beta_manifest)[0] == [alpha, gamma]


def test_old_full_source_digest_requires_regeneration(selection: SimpleNamespace, tmp_path: Path) -> None:
    manifest = _select(selection)
    manifest["source_digest_algorithm"] = "skill-evaluator-source-tree/2"
    path = tmp_path / "old-manifest.json"
    write_decoy_manifest(manifest, path)

    with pytest.raises(ValueError, match="source digest algorithm; regenerate the selection"):
        load_decoy_manifest(selection.target, path)


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


def test_loader_rejects_a_renamed_target_alias_with_a_correct_source_digest(
    selection: SimpleNamespace,
    tmp_path: Path,
) -> None:
    manifest = _select(selection, count=1)
    decoy = selection.skills[0]
    target_body = (selection.target / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[2]
    (decoy / "SKILL.md").write_text(
        f"---\nname: {decoy.name}\ndescription: A differently named skill.\n---{target_body}", encoding="utf-8"
    )
    manifest["selected"] = [_selected_record(decoy, selection.root)]
    path = tmp_path / "alias.json"
    write_decoy_manifest(manifest, path)

    with pytest.raises(ValueError, match="target alias or duplicate skill identity"):
        load_decoy_manifest(selection.target, path)


def test_loader_rejects_target_payload_with_a_correct_source_digest(
    selection: SimpleNamespace,
    tmp_path: Path,
) -> None:
    manifest = _select(selection, count=1)
    decoy = selection.skills[0]
    payload = decoy / "references" / "renamed.md"
    payload.parent.mkdir()
    payload.write_bytes((selection.target / "SKILL.md").read_bytes())
    manifest["selected"] = [_selected_record(decoy, selection.root)]
    path = tmp_path / "payload.json"
    write_decoy_manifest(manifest, path)

    with pytest.raises(ValueError, match="Selected decoy contains target instructions"):
        load_decoy_manifest(selection.target, path)


def test_rejects_mixed_content_catalog_before_model_calls(selection: SimpleNamespace) -> None:
    raw = json.loads(selection.catalog.read_text())
    rule = copy.deepcopy(raw["entries"][0])
    rule.update(id="rules:api-debugger", content_type="rules")
    raw["entries"].append(rule)
    selection.catalog.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="skill-only"):
        _select(selection)
    selection.client.embed_single.assert_not_called()


# Decoy selection and group-mode loading read SKILL.md frontmatter as similarity-check does.
_STRING_FIELDS_REQUIRED = r"requires a skill with a root SKILL\.md whose frontmatter has a string name and description"
NON_STRING_FRONTMATTER = [
    pytest.param("name: yes\ndescription: Help with target.", id="boolean-name"),
    pytest.param("name: 2048\ndescription: Help with target.", id="integer-name"),
    pytest.param("name: target\ndescription: [Draft]", id="list-description"),
]


def _write_manifest(skill: Path, frontmatter: str) -> None:
    (skill / "SKILL.md").write_bytes(f"---\n{frontmatter}\n---\n\n# {skill.name}\n".encode())


def _alias_chain(name: str, *, levels: int = 18) -> str:
    """Return frontmatter whose description is a list of 2**levels items built from chained aliases."""
    lines = ["seed: &a0 safe"]
    lines += [f"a{level}: &a{level} [*a{level - 1}, *a{level - 1}]" for level in range(1, levels + 1)]
    return "\n".join([*lines, f"name: {name}", f"description: *a{levels}"])


@pytest.mark.parametrize("frontmatter", NON_STRING_FRONTMATTER)
def test_target_name_or_description_that_is_not_a_string_is_rejected(
    selection: SimpleNamespace, frontmatter: str
) -> None:
    _write_manifest(selection.target, frontmatter)

    with pytest.raises(ValueError, match=_STRING_FIELDS_REQUIRED):
        _select(selection)
    selection.client.embed_single.assert_not_called()


def test_target_frontmatter_over_a_structure_limit_fails_before_model_calls(selection: SimpleNamespace) -> None:
    _write_manifest(selection.target, _alias_chain("target"))

    with pytest.raises(ValueError, match=r"^Manifest structured-data complexity limit exceeded: target/SKILL\.md$"):
        _select(selection)
    selection.client.embed_single.assert_not_called()


@pytest.mark.parametrize(
    ("changed", "frontmatter", "match"),
    [
        pytest.param("target", "name: yes\ndescription: Help with target.", _STRING_FIELDS_REQUIRED, id="target"),
        pytest.param(
            "selected", "name: yes\ndescription: Help with api-debugger.", _STRING_FIELDS_REQUIRED, id="selected"
        ),
        pytest.param(
            "selected",
            _alias_chain("api-debugger"),
            r"^Manifest structured-data complexity limit exceeded: api-debugger/SKILL\.md$",
            id="selected-over-limit",
        ),
    ],
)
def test_loader_reads_frontmatter_as_selection_does(
    selection: SimpleNamespace, tmp_path: Path, changed: str, frontmatter: str, match: str
) -> None:
    path = tmp_path / "decoys.json"
    write_decoy_manifest(_select(selection), path)
    _write_manifest(selection.target if changed == "target" else selection.skills[0], frontmatter)

    with pytest.raises(ValueError, match=match):
        load_decoy_manifest(selection.target, path)
