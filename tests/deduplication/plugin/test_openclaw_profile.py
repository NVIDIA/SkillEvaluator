# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, bounded Tier 2 profiles of explicitly declared OpenClaw skills."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import skillevaluator.deduplication.plugin.profile as profile_module
from skillevaluator.constants import PLUGIN_NAME_MAX_REPORT_CHARS
from skillevaluator.deduplication.plugin.profile import (
    MAX_PLUGIN_MEMBERS,
    PluginProfileError,
    PluginSkillLimitError,
    discover_plugin_roots,
    load_plugin_profile,
)
from skillevaluator.embedding import extractor
from skillevaluator.embedding.extractor import CollectionLimitError, ExtractionBudget
from skillevaluator.plugin_skills import discover_openclaw_skill_manifests
from skillevaluator.utils.helpers import find_bundled_plugin_skill_manifests
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot


def _plugin(root: Path, **fields: object) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"id": "stable-plugin-id", "configSchema": {}, "description": "Portable plugin skills", **fields}
    (root / "openclaw.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _skill(root: Path, relative: str, *, name: str = "portable-skill", valid: bool = True) -> Path:
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "SKILL.md"
    manifest.write_text(
        f"---\nname: {name if valid else '[invalid, name]'}\ndescription: Portable skill description\n---\nInstructions\n",
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize("display_name", [None, "Readable Display Name"])
def test_profile_identity_is_manifest_id_not_display_name_or_folder(tmp_path: Path, display_name: str | None) -> None:
    fields = {} if display_name is None else {"name": display_name}
    plugin = _plugin(tmp_path / "unrelated-folder", **fields)

    profile = load_plugin_profile(plugin)

    assert profile.name == "stable-plugin-id"
    assert profile.embedding_text == "stable-plugin-id: Portable plugin skills"
    assert profile.manifest == "openclaw.plugin.json"
    assert profile.source_fingerprint == hashlib.sha256((plugin / profile.manifest).read_bytes()).hexdigest()
    assert profile.members == ()
    assert profile.bundled_skills == ()


@pytest.mark.parametrize("identity", [None, "", "   ", 17, [], "x" * (PLUGIN_NAME_MAX_REPORT_CHARS + 1)])
def test_profile_requires_bounded_nonempty_id_even_with_display_name(tmp_path: Path, identity: object) -> None:
    plugin = _plugin(tmp_path / "plugin", id=identity, name="not-an-identity")

    with pytest.raises(PluginProfileError, match="Invalid plugin name"):
        load_plugin_profile(plugin)


def test_profile_does_not_fall_back_to_folder_when_id_is_omitted(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin")
    (plugin / "openclaw.plugin.json").write_text('{"name": "Display", "configSchema": {}}', encoding="utf-8")

    with pytest.raises(PluginProfileError, match="Invalid plugin name"):
        load_plugin_profile(plugin)


@pytest.mark.parametrize("schema", [None, [], "schema", False])
def test_profile_requires_config_schema_object(tmp_path: Path, schema: object) -> None:
    plugin = _plugin(tmp_path / "plugin", configSchema=schema)

    with pytest.raises(PluginProfileError, match="configSchema must be an object"):
        load_plugin_profile(plugin)


def test_profile_loads_custom_direct_immediate_and_nested_roots(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["./custom/direct", "other-skills"])
    direct = _skill(plugin, "custom/direct", name="direct")
    immediate = _skill(plugin, "other-skills/immediate", name="immediate")
    nested = _skill(plugin, "other-skills/team/nested", name="nested")
    _skill(plugin, "skills/implicit", name="implicit")
    _skill(plugin, "unrelated/ignored", name="ignored")

    profile = load_plugin_profile(plugin)

    assert profile.members == ("direct", "immediate", "nested")
    assert [skill.root_relative for skill in profile.bundled_skills] == [
        "custom/direct",
        "other-skills/immediate",
        "other-skills/team/nested",
    ]
    assert {skill.path for skill in profile.bundled_skills} == {direct.parent, immediate.parent, nested.parent}
    assert {skill.entry.path for skill in profile.bundled_skills if skill.entry} == {
        str(direct.parent),
        str(immediate.parent),
        str(nested.parent),
    }


def test_profile_stops_descending_at_skill_boundaries(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom/direct", "collection"])
    _skill(plugin, "custom/direct", name="direct")
    _skill(plugin, "custom/direct/nested", name="not-a-separate-skill")
    _skill(plugin, "collection/team/skill", name="collected")
    _skill(plugin, "collection/team/skill/templates/example", name="not-collected")
    # Even an unsafe descendant is not traversed below the skill boundary.
    (plugin / "collection/team/skill/templates/escape").symlink_to(tmp_path / "missing", target_is_directory=True)

    profile = load_plugin_profile(plugin)

    assert profile.members == ("collected", "direct")
    assert len(profile.bundled_skills) == 2


def test_recursive_descent_skips_hidden_and_node_modules_children_only(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom/roots"])
    _skill(plugin, "custom/roots/visible/skill", name="visible")
    for relative in (
        "custom/roots/.hidden/skill",
        "custom/roots/node_modules/package/skill",
        "custom/roots/visible/.nested/skill",
        "custom/roots/visible/node_modules/skill",
    ):
        _skill(plugin, relative, name="not-loaded")
        # A deeper unsafe entry proves excluded child trees are not enumerated.
        (plugin / relative / "escape").symlink_to(tmp_path / "missing", target_is_directory=True)

    profile = load_plugin_profile(plugin)

    assert profile.members == ("visible",)
    assert [skill.root_relative for skill in profile.bundled_skills] == ["custom/roots/visible/skill"]


@pytest.mark.parametrize("root", [".hidden", "node_modules", "custom/.hidden", "custom/node_modules"])
def test_explicit_hidden_or_node_modules_root_is_not_excluded(tmp_path: Path, root: str) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=[root])
    _skill(plugin, f"{root}/nested/skill", name="explicitly-loaded")
    _skill(plugin, f"{root}/.hidden/skill", name="still-excluded-child")
    _skill(plugin, f"{root}/node_modules/skill", name="also-excluded-child")

    profile = load_plugin_profile(plugin)

    assert profile.members == ("explicitly-loaded",)
    assert [skill.root_relative for skill in profile.bundled_skills] == [f"{root}/nested/skill"]


def test_openclaw_does_not_apply_codex_artifact_folder_exclusions(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom"])
    for relative, name in (
        ("results/skill", "results-skill"),
        ("evals/skill", "evals-skill"),
        ("versions/v1", "version"),
    ):
        _skill(plugin, f"custom/{relative}", name=name)
    _skill(plugin, "custom/results/skill/template/nested", name="template-not-loaded")

    profile = load_plugin_profile(plugin)
    discovered = discover_openclaw_skill_manifests(plugin, {"skills": ["custom"]})

    assert profile.members == ("evals-skill", "results-skill", "version")
    assert [file.relative_path.as_posix() for file in discovered] == [
        "custom/evals/skill/SKILL.md",
        "custom/results/skill/SKILL.md",
        "custom/versions/v1/SKILL.md",
    ]
    assert all(file.root == plugin and file.path == plugin / file.relative_path for file in discovered)
    with SecureRoot(plugin) as secure_root:
        assert all("description:" in secure_root.read_file_text(file, 1024) for file in discovered)


@pytest.mark.parametrize("fields", [{}, {"skills": []}])
def test_omitted_or_empty_skills_do_not_load_implicit_or_supplied_legacy_inventory(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    plugin = _plugin(tmp_path / "plugin", **fields)
    _skill(plugin, "skills/implicit")
    _skill(plugin, ".", name="root-is-not-implicit")
    legacy_inventory = find_bundled_plugin_skill_manifests(plugin)
    assert legacy_inventory

    profile = load_plugin_profile(plugin, skill_manifests=legacy_inventory)

    assert profile.members == ()
    assert profile.bundled_skills == ()


def test_declared_plugin_root_skill_has_correct_root_relative_identity(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["."])
    _skill(plugin, ".", name="root-skill")
    _skill(plugin, "skills/not-selected")

    profile = load_plugin_profile(plugin)

    [skill] = profile.bundled_skills
    assert skill.root_relative == "."
    assert skill.path == plugin
    assert skill.entry is not None
    assert skill.entry.path == str(plugin)
    assert profile.members == ("root-skill",)


def test_profile_keeps_correct_custom_root_for_skipped_malformed_skill(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom/direct"])
    _skill(plugin, "custom/direct", valid=False)
    budget = ExtractionBudget()

    profile = load_plugin_profile(plugin, budget=budget)

    [skill] = profile.bundled_skills
    assert skill.entry is None
    assert skill.skip_reason
    assert skill.root_relative == "custom/direct"
    assert skill.path == plugin / "custom/direct"
    assert profile.members == ("direct",)
    assert budget.entry_count == 1


def test_relative_plugin_root_and_overlapping_roots_are_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom", "custom/team", "./custom"])
    _skill(plugin, "custom/team/skill", name="single")
    monkeypatch.chdir(tmp_path)
    budget = ExtractionBudget()

    profile = load_plugin_profile(Path("plugin"), budget=budget)

    [skill] = profile.bundled_skills
    assert skill.root_relative == "custom/team/skill"
    assert skill.path == Path("plugin/custom/team/skill")
    assert skill.entry is not None
    # Catalog entry paths use native filesystem separators; manifest-relative
    # identities above remain POSIX on every platform.
    assert skill.entry.path == str(Path("plugin") / "custom" / "team" / "skill")
    assert budget.entry_count == 1


@pytest.mark.parametrize("roots", [None, "./skills", {}, [1], [""]])
def test_invalid_skill_root_declarations_fail_closed(tmp_path: Path, roots: object) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=roots)

    with pytest.raises(ValueError):
        load_plugin_profile(plugin)


@pytest.mark.parametrize(
    "root",
    [
        "../outside",
        "./safe/../outside",
        "/tmp/absolute",
        "~/skills",
        "C:\\skills",
        "./C:/skills",
        "skills:stream",
        "${PLUGIN_ROOT}/skills",
        "bad\x00path",
        "x" * 4097,
    ],
)
def test_unsafe_declared_paths_are_rejected_before_skill_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str
) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=[root])
    monkeypatch.setattr(profile_module, "extract_skill_manifest", lambda *_args, **_kwargs: pytest.fail("unsafe read"))

    with pytest.raises(SecurePathError, match="Unsafe OpenClaw skills directory"):
        load_plugin_profile(plugin)


@pytest.mark.parametrize("kind", ["root", "ancestor", "child", "manifest", "hardlink"])
def test_declared_skill_discovery_rejects_linked_paths(tmp_path: Path, kind: str) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom/deep/root"])
    outside = tmp_path / "outside"
    _skill(outside, "real")
    custom = plugin / "custom/deep"
    custom.mkdir(parents=True)
    if kind == "root":
        (custom / "root").symlink_to(outside / "real", target_is_directory=True)
    elif kind == "ancestor":
        (custom / "root").symlink_to(outside, target_is_directory=True)
        _plugin(plugin, skills=["custom/deep/root/real"])
    else:
        skill_root = custom / "root"
        skill_root.mkdir()
        if kind == "child":
            (skill_root / "linked").symlink_to(outside / "real", target_is_directory=True)
        elif kind == "manifest":
            (skill_root / "SKILL.md").symlink_to(outside / "real/SKILL.md")
        else:
            (skill_root / "SKILL.md").hardlink_to(outside / "real/SKILL.md")

    with pytest.raises(SecurePathError):
        load_plugin_profile(plugin)


def test_skill_limit_is_enforced_before_reading_any_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom"])
    for name in ("one", "two"):
        _skill(plugin, f"custom/{name}", name=name)
    monkeypatch.setattr(profile_module, "extract_skill_manifest", lambda *_args, **_kwargs: pytest.fail("skill read"))

    with pytest.raises(PluginSkillLimitError) as failure:
        load_plugin_profile(plugin, max_skills=1)

    assert (failure.value.actual, failure.value.limit) == (2, 1)


def test_declared_root_count_is_bounded(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom"] * (MAX_PLUGIN_MEMBERS + 1))

    with pytest.raises(PluginProfileError, match="item limit"):
        load_plugin_profile(plugin)


def test_discovery_path_budget_is_shared_across_declared_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom/one", "custom/two"])
    for name in ("one", "two"):
        _skill(plugin, f"custom/{name}", name=name)
        (plugin / f"custom/{name}/notes.txt").write_text("notes", encoding="utf-8")
    monkeypatch.setattr(profile_module, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 3)

    with pytest.raises(SecurePathError) as failure:
        load_plugin_profile(plugin)

    assert failure.value.code == "path_count_limit"


def test_discovery_directory_depth_is_bounded(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom"])
    _skill(plugin, "custom/a/b/c", name="too-deep")

    with pytest.raises(SecurePathError) as failure:
        discover_openclaw_skill_manifests(plugin, {"skills": ["custom"]}, max_depth=2)

    assert failure.value.code == "directory_depth_limit"


@pytest.mark.parametrize(
    "limits",
    [
        {"max_roots": -1},
        {"max_roots": True},
        {"max_paths": 0},
        {"max_paths": True},
        {"max_depth": 0},
        {"max_depth": 65},
    ],
)
def test_shared_discovery_rejects_invalid_limits(tmp_path: Path, limits: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        discover_openclaw_skill_manifests(tmp_path, {}, **limits)


def test_shared_discovery_bounds_roots_without_profile_exceptions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1-item limit") as failure:
        discover_openclaw_skill_manifests(tmp_path, {"skills": ["one", "two"]}, max_roots=1)
    assert type(failure.value) is ValueError


def test_manifest_and_malformed_skill_bytes_share_collection_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom"])
    skill = _skill(plugin, "custom", valid=False)
    manifest_bytes = (plugin / "openclaw.plugin.json").stat().st_size
    monkeypatch.setattr(extractor, "MAX_COLLECTION_BYTES", manifest_bytes + skill.stat().st_size - 1)

    with pytest.raises(CollectionLimitError, match="total byte limit"):
        load_plugin_profile(plugin, budget=ExtractionBudget())


def test_collection_entry_budget_counts_declared_skills(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom"])
    _skill(plugin, "custom/one", name="one")
    _skill(plugin, "custom/two", name="two")

    with pytest.raises(CollectionLimitError, match="entry limit"):
        load_plugin_profile(plugin, budget=ExtractionBudget(max_entries=1))


def test_discovered_skill_inode_is_checked_again_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _plugin(tmp_path / "plugin", skills=["custom"])
    manifest = _skill(plugin, "custom", name="original")
    original_read = SecureRoot.read_file_text
    swapped = False

    def replace_before_read(secure_root: SecureRoot, file, max_bytes: int) -> str:
        nonlocal swapped
        if file.relative_path.name == "SKILL.md" and not swapped:
            replacement = manifest.with_name("replacement.md")
            replacement.write_text("---\nname: replacement\ndescription: changed\n---\n", encoding="utf-8")
            replacement.replace(manifest)
            swapped = True
        return original_read(secure_root, file, max_bytes)

    monkeypatch.setattr(SecureRoot, "read_file_text", replace_before_read)

    with pytest.raises(SecurePathError):
        load_plugin_profile(plugin)
    assert swapped


def test_nested_openclaw_plugins_are_discovered_with_collection_limit(tmp_path: Path) -> None:
    first = _plugin(tmp_path / "collection/team/one")
    second = _plugin(tmp_path / "collection/team/two")

    assert discover_plugin_roots(tmp_path / "collection", max_plugins=2) == [first, second]
    with pytest.raises(ValueError, match="entry limit"):
        discover_plugin_roots(tmp_path / "collection", max_plugins=1)
