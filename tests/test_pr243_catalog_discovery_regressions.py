# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for PR243 review findings 1, 9, 10 and 11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.constants import CONTENT_DEDUP_MAX_DISCOVERED_PATHS
from skillevaluator.deduplication.plugin.catalog_checks import check_catalog_skills
from skillevaluator.deduplication.plugin.profile import load_plugin_profile
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.models.result import ValidationResult
from skillevaluator.plugin_skills import discover_openclaw_skill_manifests
from skillevaluator.tier2 import commands
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.similarity import SimilarityValidator


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    monkeypatch.delenv("SKILL_EVAL_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("SKILL_EVAL_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(EmbeddingClient, "embed", lambda _self, texts: [[1.0, 0.0] for _text in texts])
    monkeypatch.setattr(EmbeddingClient, "embed_single", lambda _self, _text: [1.0, 0.0])


def _plugin(root: Path, roots: list[str], *, native: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if native:
        data = {"id": root.name, "description": "Deploy applications", "configSchema": {}, "skills": roots}
        (root / "openclaw.plugin.json").write_text(json.dumps(data), encoding="utf-8")
    else:
        (root / ".claude-plugin").mkdir()
        (root / ".claude-plugin/plugin.json").write_text(
            json.dumps({"name": root.name, "description": "Deploy applications"}), encoding="utf-8"
        )
    return root


def _skill(plugin: Path, relative: str, name: str = "deploy-app") -> Path:
    folder = plugin / relative
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Deploy applications\n---\nInstructions\n", encoding="utf-8"
    )
    return folder


def _loaded(catalog: Path) -> EmbeddingRegistry:
    registry = EmbeddingRegistry(EmbeddingClient())
    registry.load_catalog(catalog)
    return registry


def _excluded(result: ValidationResult) -> list[str]:
    [success] = [item for item in result.success_details if item.check_name == "skill_catalog_comparison"]
    return success.metadata["excluded_catalog_paths"]


@pytest.mark.parametrize(
    ("root", "relative", "native"),
    [
        ("./skills", "skills/deploy-app", True),
        ("portable", "portable/deploy-app", True),
        (".", ".", True),
        ("skills", "skills/deploy-app", False),
    ],
)
@pytest.mark.parametrize("collection", [True, False])
def test_saved_plugin_catalog_paths_exist_and_exclude_self(
    tmp_path: Path, root: str, relative: str, native: bool, collection: bool
) -> None:
    plugin = _plugin(tmp_path / "collection/one", [root], native=native)
    folder = _skill(plugin, relative)
    scan_root = plugin.parent if collection else plugin
    catalog = tmp_path / "catalog.json"
    saved = SimilarityValidator(content_type="plugin", save_catalog_path=catalog).validate(scan_root)
    assert saved.passed, saved.errors
    registry = _loaded(catalog)
    [entry] = registry.skill_entries
    assert scan_root / entry.path == folder
    result = check_catalog_skills(load_plugin_profile(plugin), registry, threshold=0.5)
    assert not result.findings
    assert _excluded(result) == [folder.relative_to(scan_root).as_posix()]


@pytest.mark.parametrize(
    ("root", "relative"), [("./skills", "deploy-app"), ("portable", "team/deploy-app"), (".", ".")]
)
def test_skill_only_catalog_excludes_declared_root_relative_self(tmp_path: Path, root: str, relative: str) -> None:
    plugin = _plugin(tmp_path / "plugin", [root])
    declared_root = plugin / root
    _skill(declared_root, relative)
    registry = EmbeddingRegistry(EmbeddingClient())
    assert registry.build_from_directory(declared_root, "skill") == 1
    catalog = tmp_path / "skills.json"
    registry.save_catalog(catalog)
    result = check_catalog_skills(load_plugin_profile(plugin), _loaded(catalog), threshold=0.5)
    assert not result.findings
    assert _excluded(result) == [relative]


def test_skill_only_same_content_elsewhere_still_compared(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", ["portable"])
    _skill(plugin, "portable/deploy-app")
    _skill(tmp_path / "other", "unrelated/deploy-app")
    registry = EmbeddingRegistry(EmbeddingClient())
    registry.build_from_directory(tmp_path / "other", "skill")
    result = check_catalog_skills(load_plugin_profile(plugin), registry, threshold=0.5)
    assert len(result.findings) == 1
    assert _excluded(result) == []


@pytest.mark.parametrize("unsafe_implicit", [False, True])
def test_c_intra_uses_declared_custom_root_and_runs_catalog_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_implicit: bool
) -> None:
    plugin = _plugin(tmp_path / "plugin", ["portable"])
    selected = _skill(plugin, "portable/deploy-app")
    implicit = _skill(plugin, "skills/legacy", name="legacy")
    if unsafe_implicit:
        (implicit / "assets").mkdir()
        (implicit / "assets/link").symlink_to(tmp_path / "absent")
    captured = []

    def context(_root, **kwargs):
        captured.extend(kwargs["skill_dirs"])
        return [ValidationResult(validator_name="Context Deduplication")]

    monkeypatch.setattr(commands, "run_plugin_skill_context_dedup", context)
    results = commands.run_plugin_dedup_scan(plugin)
    assert captured == [selected]
    assert len(results) == 4


@pytest.mark.parametrize("relative", ["portable/deploy-app", "."])
def test_c_intra_labels_custom_and_root_skills_relative_to_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    plugin = _plugin(tmp_path / "plugin", ["." if relative == "." else "portable"])
    skill = _skill(plugin, relative)
    visited = []

    class OfflineValidator:
        def __init__(self, **_kwargs):
            pass

        def validate(self, path):
            visited.append(path)
            result = ValidationResult(validator_name="Context Deduplication")
            result.add_success("offline", "checked")
            return result

    monkeypatch.setattr(commands, "IntraSkillValidator", OfflineValidator)
    [result] = commands.run_plugin_skill_context_dedup(plugin)
    assert result.passed
    assert visited == [skill]
    assert result.success_details[0].check_name == f"[{relative}] offline"


def test_missing_declared_roots_do_not_abort_profile_or_catalog(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "collection/optional", ["skills", "optional-skills"])
    _skill(plugin, "skills/deploy-app")
    peer = _plugin(tmp_path / "collection/peer", ["node_modules/@tloncorp/tlon-skill"])
    assert load_plugin_profile(plugin).members == ("deploy-app",)
    assert load_plugin_profile(peer).members == ()
    tier1 = PluginSchemaValidator().validate(plugin)
    assert any(f.check_name == "plugin_component_path_missing" for f in tier1.findings)
    catalog = tmp_path / "catalog.json"
    result = SimilarityValidator(content_type="plugin", save_catalog_path=catalog).validate(plugin.parent)
    assert result.passed, result.errors
    registry = _loaded(catalog)
    assert registry.plugin_size == 2
    results = commands.run_plugin_catalog_checks(plugin, catalog=catalog)
    assert len(results) == 2
    assert not any(item.metadata.get("security_failure") for item in results)


@pytest.mark.parametrize(
    "kind", ["dangling-link", "linked-parent", "file", "hardlink", "file-parent", "hardlink-parent"]
)
def test_unsafe_declared_roots_are_not_skipped_as_missing(tmp_path: Path, kind: str) -> None:
    plugin = _plugin(tmp_path / "plugin", ["optional"])
    if kind == "linked-parent":
        (plugin / "optional").symlink_to(tmp_path / "absent", target_is_directory=True)
        _plugin(plugin, ["optional/missing"])
    elif kind == "dangling-link":
        (plugin / "optional").symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        (plugin / "optional").write_text("not a directory", encoding="utf-8")
        if kind in {"hardlink", "hardlink-parent"}:
            (plugin / "alias").hardlink_to(plugin / "optional")
        if kind in {"file-parent", "hardlink-parent"}:
            _plugin(plugin, ["optional/missing"])
    with pytest.raises(SecurePathError):
        load_plugin_profile(plugin)


def test_tier2_uses_same_4096_path_budget_as_shared_discovery(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", ["skills"])
    _skill(plugin, "skills/deploy-app")
    assets = plugin / "skills/shared-assets"
    assets.mkdir()
    for number in range(CONTENT_DEDUP_MAX_DISCOVERED_PATHS + 1):
        (assets / f"asset-{number:04d}").touch()
    with pytest.raises(SecurePathError) as shared:
        discover_openclaw_skill_manifests(plugin, {"skills": ["skills"]})
    assert shared.value.code == "path_count_limit"
    tier1 = PluginSchemaValidator().validate(plugin)
    assert any("OpenClaw skill discovery failed closed" in f.message for f in tier1.findings)
    inventory = tier1.metadata["plugin"]["component_inventory"]["components"]
    assert not any(row["type"] == "skill" for row in inventory)
    with pytest.raises(SecurePathError) as profile:
        load_plugin_profile(plugin, max_skills=32)
    assert profile.value.code == "path_count_limit"
    assert profile.value.metadata["limit"] == CONTENT_DEDUP_MAX_DISCOVERED_PATHS


@pytest.mark.parametrize("max_paths", [2, 3])
def test_budget_errors_report_overall_limit_not_remaining_allowance(tmp_path: Path, max_paths: int) -> None:
    plugin = _plugin(tmp_path / "plugin", ["one", "two"])
    _skill(plugin, "one")
    _skill(plugin, "two")
    (plugin / "one/notes.txt").touch()
    (plugin / "two/notes.txt").touch()
    with pytest.raises(SecurePathError) as failure:
        discover_openclaw_skill_manifests(plugin, {"skills": ["one", "two"]}, max_paths=max_paths)
    assert failure.value.code == "path_count_limit"
    assert f"path limit of {max_paths} entries" in str(failure.value)
    assert failure.value.metadata["limit"] == max_paths


def test_native_c_intra_rejects_legacy_inventory_and_ignores_implicit_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _plugin(tmp_path / "plugin", [])
    implicit = _skill(plugin, "skills/legacy")

    def unexpected_validation(*_args, **_kwargs):
        pytest.fail("undeclared native skill was validated")

    monkeypatch.setattr(commands, "IntraSkillValidator", unexpected_validation)
    [result] = commands.run_plugin_skill_context_dedup(plugin, skill_dirs=[implicit])
    assert result.passed
    assert result.success_details[0].message == "No bundled skills to deduplicate"


def test_overlapping_native_roots_keep_all_skill_only_catalog_identities(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", ["portable", "portable/team", "./portable"])
    _skill(plugin, "portable/team/deploy-app")
    [skill] = load_plugin_profile(plugin).bundled_skills
    assert skill.catalog_relative_paths == ("deploy-app", "team/deploy-app")


def test_declared_root_direct_skill_excludes_dot_in_skill_only_catalog(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", ["portable/direct"])
    skill_root = _skill(plugin, "portable/direct")
    registry = EmbeddingRegistry(EmbeddingClient())
    registry.build_from_directory(skill_root, "skill")
    catalog = tmp_path / "catalog.json"
    registry.save_catalog(catalog)
    result = check_catalog_skills(load_plugin_profile(plugin), _loaded(catalog), threshold=0.5)
    assert not result.findings
    assert _excluded(result) == ["."]


def test_skill_only_same_path_with_changed_content_is_not_self(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", ["portable"])
    folder = _skill(plugin, "portable/deploy-app")
    registry = EmbeddingRegistry(EmbeddingClient())
    registry.build_from_directory(plugin / "portable", "skill")
    (folder / "SKILL.md").write_text(
        "---\nname: deploy-app\ndescription: Different deployment instructions\n---\n", encoding="utf-8"
    )
    result = check_catalog_skills(load_plugin_profile(plugin), registry, threshold=0.5)
    assert len(result.findings) == 1
    assert _excluded(result) == []
