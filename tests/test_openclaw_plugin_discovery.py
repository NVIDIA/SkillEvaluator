# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenClaw manifests through the private shared inventory and manifest schema checks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from skillevaluator.constants import CONTENT_DEDUP_MAX_DISCOVERED_PATHS, PLUGIN_OPENCLAW_MANIFEST_TYPE
from skillevaluator.models.result import Severity
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.plugin_manifest import locate_plugin_manifest, manifest_relative_path, manifest_root_for
from skillevaluator.plugin_skills import discover_openclaw_skill_manifests
from skillevaluator.utils.secure_fs import MAX_SECURE_DIRECTORY_DEPTH, SecurePathError
from skillevaluator.validators.private_plugin_schema import validate_inventory_schema

MANIFEST = "openclaw.plugin.json"


def _skill(name: str) -> str:
    return f"---\nname: {name}\ndescription: {name} guidance.\n---\nBody.\n"


def _tree(root: Path, files: dict[str, Any]) -> Path:
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return root


def _openclaw(root: Path, files: dict[str, Any] | None = None, **fields: Any) -> Path:
    manifest = {"id": "release-helper", "configSchema": {"type": "object"}, **fields}
    return _tree(root, {MANIFEST: manifest, **(files or {})})


def _inventory(root: Path):
    result = plugin_inventory_for_root(root)
    assert result is not None
    return result


def _members(inventory, component_type: str = "skill") -> list[str]:
    return sorted(
        member.path or ""
        for member in inventory.selected_components
        if member.type == component_type and member.support == "evaluated" and not member.problem
    )


def _checks(findings) -> set[str]:
    return {finding.check_name for finding in findings}


# --------------------------------------------------------------------------- #
# Manifest discovery and identity                                             #
# --------------------------------------------------------------------------- #
def test_root_manifest_is_located_and_names_its_root(tmp_path: Path) -> None:
    _openclaw(tmp_path)
    located = locate_plugin_manifest(tmp_path)
    assert located is not None
    assert (located.manifest_type, located.manifest_filename) == (PLUGIN_OPENCLAW_MANIFEST_TYPE, MANIFEST)
    assert manifest_relative_path(tmp_path / "OpenClaw.Plugin.json") == Path("OpenClaw.Plugin.json")
    assert manifest_root_for(tmp_path / MANIFEST) == tmp_path


def test_existing_manifest_precedence_is_preserved(tmp_path: Path) -> None:
    _openclaw(tmp_path, {".claude-plugin/plugin.json": {"name": "release-helper"}})
    located = locate_plugin_manifest(tmp_path)
    assert located is not None
    assert located.manifest_filename == ".claude-plugin/plugin.json"
    assert [extra.manifest_filename for extra in located.additional] == [MANIFEST]


@pytest.mark.parametrize("display_name", [None, "Readable Display Name"])
def test_identity_is_the_manifest_id_not_the_display_name(tmp_path: Path, display_name: str | None) -> None:
    fields = {} if display_name is None else {"name": display_name}
    result = validate_inventory_schema(_inventory(_openclaw(tmp_path, **fields)))
    [row] = result.metadata["plugin"]["manifest_declarations"]
    assert (row["name"], row["identity_field"]) == ("release-helper", "id")
    assert not [finding for finding in result.findings if finding.check_name.startswith("schema:")]


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"id": None}, "id"),
        ({"id": ""}, "id"),
        ({"id": 7}, "id"),
        ({"configSchema": None}, "configSchema"),
        ({"configSchema": []}, "configSchema"),
        ({"skills": "portable"}, "skills"),
        ({"skills": ["portable", 7]}, "skills"),
        ({"channels": [""]}, "channels"),
        ({"description": 7}, "description"),
    ],
)
def test_invalid_native_contract_is_a_blocking_schema_finding(tmp_path: Path, fields: dict, field: str) -> None:
    manifest = {"id": "release-helper", "configSchema": {"type": "object"}, **fields}
    manifest = {key: value for key, value in manifest.items() if value is not None}
    _tree(tmp_path, {MANIFEST: manifest, "portable/SKILL.md": _skill("portable")})
    result = validate_inventory_schema(_inventory(tmp_path))
    matching = [finding for finding in result.findings if finding.check_name.startswith(f"schema:{field}:")]
    assert matching
    assert all(finding.severity == Severity.HIGH for finding in matching)


def test_unknown_openclaw_metadata_is_retained_without_findings(tmp_path: Path) -> None:
    _openclaw(
        tmp_path,
        channels=["slack"],
        providers=["acme"],
        uiHints={"title": "Release helper"},
        requiresPlugins=["memory-core"],
    )
    result = validate_inventory_schema(_inventory(tmp_path))
    assert not [finding for finding in result.findings if finding.check_name.startswith("schema:")]


@pytest.mark.parametrize("selected_openclaw", [False, True])
def test_identity_conflict_names_the_actual_fields_in_both_directions(tmp_path: Path, selected_openclaw: bool) -> None:
    _openclaw(tmp_path, {".codex-plugin/plugin.json": {"name": "other-name", "version": "1.0.0"}})
    # Codex outranks openclaw.plugin.json; naming the native manifest selects it directly.
    inventory = _inventory(tmp_path / MANIFEST if selected_openclaw else tmp_path)
    assert (inventory.selected_manifest.manifest_type == PLUGIN_OPENCLAW_MANIFEST_TYPE) is selected_openclaw
    result = validate_inventory_schema(inventory)
    [conflict] = [finding for finding in result.findings if finding.check_name == "plugin_manifest_conflict"]
    assert "id 'release-helper'" in conflict.message
    assert "name 'other-name'" in conflict.message
    assert conflict.message.index("declares") < conflict.message.index("selected")
    assert conflict.suggestion == "Keep the OpenClaw id equal to the plugin name in the other manifests."
    assert conflict.metadata["identity_field"] == ("name" if selected_openclaw else "id")


def test_version_only_conflict_is_not_reported_as_an_identity_conflict(tmp_path: Path) -> None:
    _openclaw(tmp_path, {".claude-plugin/plugin.json": {"version": "2.0.0"}}, version="1.0.0")
    result = validate_inventory_schema(_inventory(tmp_path))
    [conflict] = [finding for finding in result.findings if finding.check_name == "plugin_manifest_conflict"]
    assert conflict.message == "Client manifests declare different names or versions."
    assert conflict.suggestion is None


def test_matching_identity_with_a_different_version_stays_a_version_conflict(tmp_path: Path) -> None:
    _openclaw(
        tmp_path,
        {".codex-plugin/plugin.json": {"name": "release-helper", "version": "2.0.0"}},
        name="Release Helper",
        version="1.0.0",
    )
    result = validate_inventory_schema(_inventory(tmp_path / MANIFEST))
    [conflict] = [finding for finding in result.findings if finding.check_name == "plugin_manifest_conflict"]
    assert conflict.message == "Client manifests declare different names or versions."


def test_display_name_never_conflicts_with_another_manifest_name(tmp_path: Path) -> None:
    _openclaw(
        tmp_path,
        {".codex-plugin/plugin.json": {"name": "release-helper", "version": "1.0.0"}},
        name="Release Helper",
    )
    result = validate_inventory_schema(_inventory(tmp_path))
    assert "plugin_manifest_conflict" not in _checks(result.findings)


# --------------------------------------------------------------------------- #
# Declared skill roots                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("skills", "expected"),
    [(None, []), ([], []), (["portable"], ["portable/own"]), (["skills"], ["skills/default"])],
)
def test_only_declared_skill_roots_are_members(tmp_path: Path, skills: list | None, expected: list[str]) -> None:
    fields = {} if skills is None else {"skills": skills}
    files = {"skills/default/SKILL.md": _skill("default"), "portable/own/SKILL.md": _skill("own")}
    inventory = _inventory(_openclaw(tmp_path, files, **fields))
    assert _members(inventory) == expected
    if not expected:
        # Another client's default folder stays static-only cross-client inventory.
        [default] = [member for member in inventory.components if member.path == "skills/default"]
        assert default.support == "static_only"
        assert default.loaded_by


def test_declared_roots_are_recursive_and_stop_at_skill_boundaries(tmp_path: Path) -> None:
    files = {
        "custom/a/SKILL.md": _skill("a"),
        "custom/a/nested/SKILL.md": _skill("nested"),
        "custom/b/c/SKILL.md": _skill("c"),
        "custom/.hidden/x/SKILL.md": _skill("hidden"),
        "custom/node_modules/y/SKILL.md": _skill("dependency"),
        # OpenClaw applies no evaluation-artifact folder exclusions.
        "custom/evals/z/SKILL.md": _skill("z"),
        "direct/SKILL.md": _skill("direct"),
    }
    inventory = _inventory(_openclaw(tmp_path, files, skills=["custom", "./direct"]))
    assert _members(inventory) == ["custom/a", "custom/b/c", "custom/evals/z", "direct"]
    assert inventory.complete


@pytest.mark.parametrize("root", [".hidden", "node_modules", "custom/.hidden", "custom/node_modules"])
def test_explicitly_declared_hidden_or_dependency_root_is_searched(tmp_path: Path, root: str) -> None:
    inventory = _inventory(_openclaw(tmp_path, {f"{root}/x/SKILL.md": _skill("x")}, skills=[root]))
    assert _members(inventory) == [f"{root}/x"]


def test_root_skill_selected_through_dot_is_inventoried(tmp_path: Path) -> None:
    inventory = _inventory(_openclaw(tmp_path, {"SKILL.md": _skill("root")}, skills=["."]))
    [member] = inventory.selected_skills
    assert (member.path, member.name) == (".", tmp_path.name)
    assert member.frontmatter["name"] == "root"


def test_overlapping_roots_are_deduplicated(tmp_path: Path) -> None:
    files = {"custom/a/SKILL.md": _skill("a")}
    inventory = _inventory(_openclaw(tmp_path, files, skills=["custom", "custom/a", "./custom"]))
    assert _members(inventory) == ["custom/a"]


@pytest.mark.parametrize(("declared", "problem"), [("../outside", "escape"), ("/tmp/outside", "escape")])
def test_escaping_declared_roots_fail_closed(tmp_path: Path, declared: str, problem: str) -> None:
    inventory = _inventory(_openclaw(tmp_path, skills=[declared]))
    broken = [member for member in inventory.selected_components if member.type == "skill" and member.problem]
    assert [member.problem for member in broken] == [problem]
    assert not inventory.complete


def test_missing_declared_root_is_reported(tmp_path: Path) -> None:
    inventory = _inventory(_openclaw(tmp_path, skills=["missing"]))
    [broken] = [member for member in inventory.selected_components if member.type == "skill"]
    assert broken.problem == "missing"
    assert any(finding.severity == Severity.HIGH for finding in inventory.findings)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
@pytest.mark.parametrize("kind", ["root", "child", "manifest"])
def test_linked_declared_skill_paths_fail_closed(tmp_path: Path, kind: str) -> None:
    plugin = tmp_path / "plugin"
    outside = _tree(tmp_path / "outside", {"x/SKILL.md": _skill("x"), "SKILL.md": _skill("outside")})
    _openclaw(plugin, {"custom/keep/SKILL.md": _skill("keep")}, skills=["custom"])
    try:
        if kind == "root":
            (plugin / "linked").symlink_to(outside, target_is_directory=True)
            _openclaw(plugin, skills=["linked"])
        elif kind == "child":
            (plugin / "custom" / "x").symlink_to(outside / "x", target_is_directory=True)
        else:
            (plugin / "custom" / "y").mkdir()
            (plugin / "custom" / "y" / "SKILL.md").symlink_to(outside / "SKILL.md")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    inventory = _inventory(plugin)
    assert any(member.problem == "unsafe" for member in inventory.selected_components if member.type == "skill")
    assert not inventory.complete


def test_shared_discovery_rejects_invalid_limits_and_declarations(tmp_path: Path) -> None:
    for limits in (
        {"max_roots": -1},
        {"max_paths": 0},
        {"max_depth": 0},
        {"max_depth": MAX_SECURE_DIRECTORY_DEPTH + 1},
    ):
        with pytest.raises(ValueError):
            discover_openclaw_skill_manifests(tmp_path, {"skills": []}, **limits)
    for roots in (None, "./skills", {}, [1]):
        with pytest.raises(ValueError):
            discover_openclaw_skill_manifests(tmp_path, {"skills": roots})
    with pytest.raises(ValueError, match="2-item limit"):
        discover_openclaw_skill_manifests(tmp_path, {"skills": ["a", "b", "c"]}, max_roots=2)
    with pytest.raises(SecurePathError):
        discover_openclaw_skill_manifests(tmp_path, {"skills": ["../escape"]})


def test_shared_discovery_path_budget_spans_all_declared_roots(tmp_path: Path) -> None:
    _tree(tmp_path, {f"{root}/{index}/notes.txt": "x" for root in ("a", "b") for index in range(3)})
    with pytest.raises(SecurePathError) as caught:
        discover_openclaw_skill_manifests(tmp_path, {"skills": ["a", "b"]}, max_paths=4)
    assert caught.value.code == "path_count_limit"
    assert caught.value.metadata["limit"] == 4
    assert len(discover_openclaw_skill_manifests(tmp_path, {"skills": ["a", "b"]})) == 0
    assert CONTENT_DEDUP_MAX_DISCOVERED_PATHS > 4


def test_shared_discovery_depth_is_bounded(tmp_path: Path) -> None:
    deep = PurePosixPath(*(f"d{index}" for index in range(4)))
    _tree(tmp_path, {f"{deep}/SKILL.md": _skill("deep")})
    with pytest.raises(SecurePathError) as caught:
        discover_openclaw_skill_manifests(tmp_path, {"skills": ["d0"]}, max_depth=2)
    assert caught.value.code == "directory_depth_limit"
    [found] = discover_openclaw_skill_manifests(tmp_path, {"skills": ["d0"]})
    assert found.relative_path.parent.as_posix() == deep.as_posix()


def test_missing_declared_roots_are_skipped_by_shared_discovery(tmp_path: Path) -> None:
    _tree(tmp_path, {"present/SKILL.md": _skill("present")})
    found = discover_openclaw_skill_manifests(tmp_path, {"skills": ["missing", "present"]})
    assert [manifest.relative_path.parent.as_posix() for manifest in found] == ["present"]
    _tree(tmp_path, {"file": "x"})
    with pytest.raises(SecurePathError):
        discover_openclaw_skill_manifests(tmp_path, {"skills": ["file"]})


# --------------------------------------------------------------------------- #
# package.json#openclaw runtime entrypoints                                   #
# --------------------------------------------------------------------------- #
def _runtime_rows(inventory) -> dict[str, str | None]:
    return {
        member.name: member.path
        for member in inventory.selected_components
        if member.type == "extension" and not member.problem
    }


def _package(root: Path, openclaw: Any, files: dict[str, Any] | None = None, **fields: Any) -> Path:
    return _openclaw(
        root, {"package.json": {"name": "release-helper", "openclaw": openclaw}, **(files or {})}, **fields
    )


def test_runtime_row_is_kept_without_package_entrypoints(tmp_path: Path) -> None:
    inventory = _inventory(_openclaw(tmp_path, {"portable/SKILL.md": _skill("portable")}, skills=["portable"]))
    [runtime] = [member for member in inventory.selected_components if member.type == "extension"]
    assert (runtime.name, runtime.path, runtime.support) == ("OpenClaw runtime", MANIFEST, "unsupported")


def test_runtime_entrypoints_are_inventoried_without_importing_them(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("OpenClaw entrypoints must never be executed or fetched")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    _package(
        tmp_path,
        {"extensions": ["./src/index.ts"], "setupEntry": "./src/setup.ts"},
        {"src/index.ts": "export {}", "src/setup.ts": "export {}"},
    )
    inventory = _inventory(tmp_path)
    assert _runtime_rows(inventory) == {
        "OpenClaw runtime": MANIFEST,
        "openclaw.extensions: ./src/index.ts": "src/index.ts",
        "openclaw.setupEntry: ./src/setup.ts": "src/setup.ts",
    }
    assert all(
        member.support == "unsupported" for member in inventory.selected_components if member.type == "extension"
    )
    assert inventory.complete


@pytest.mark.parametrize("with_setup", [False, True])
def test_published_paired_runtime_does_not_require_typescript_source(tmp_path: Path, with_setup: bool) -> None:
    metadata: dict[str, Any] = {"extensions": ["./src/index.ts"], "runtimeExtensions": ["./dist/index.js"]}
    files = {"dist/index.js": "export {}"}
    if with_setup:
        metadata.update(setupEntry="./src/setup.ts", runtimeSetupEntry="./dist/setup.js")
        files["dist/setup.js"] = "export {}"
    inventory = _inventory(_package(tmp_path, metadata, files))
    assert not [finding for finding in inventory.findings if finding.severity in {Severity.HIGH, Severity.CRITICAL}]
    assert _runtime_rows(inventory)["openclaw.runtimeExtensions: ./dist/index.js"] == "dist/index.js"
    assert inventory.complete


@pytest.mark.parametrize("built", ["dist/index.js", "dist/index.cjs", "dist/src/index.mjs", "src/index.js"])
@pytest.mark.parametrize("field", ["extensions", "setupEntry"])
def test_inferred_built_runtime_excuses_a_missing_source(tmp_path: Path, built: str, field: str) -> None:
    value: Any = ["./src/index.ts"] if field == "extensions" else "./src/index.ts"
    inventory = _inventory(_package(tmp_path, {field: value}, {built: "export {}"}))
    assert not [finding for finding in inventory.findings if finding.severity in {Severity.HIGH, Severity.CRITICAL}]
    runtime_field = "runtimeExtensions" if field == "extensions" else "runtimeSetupEntry"
    assert _runtime_rows(inventory)[f"openclaw.{runtime_field} (inferred): {built}"] == built


@pytest.mark.parametrize("field", ["extensions", "setupEntry"])
def test_missing_source_without_a_valid_runtime_is_blocking(tmp_path: Path, field: str) -> None:
    value: Any = ["./src/index.ts"] if field == "extensions" else "./src/index.ts"
    inventory = _inventory(_package(tmp_path, {field: value}))
    assert any(finding.severity == Severity.HIGH for finding in inventory.findings)
    assert any(member.problem == "missing" for member in inventory.selected_components if member.type == "extension")


def test_missing_javascript_source_has_no_inferred_fallback(tmp_path: Path) -> None:
    inventory = _inventory(_package(tmp_path, {"extensions": ["./index.js"]}, {"dist/index.js": "export {}"}))
    assert any(member.problem == "missing" for member in inventory.selected_components if member.type == "extension")


def test_explicit_runtime_must_exist_even_when_inferred_output_exists(tmp_path: Path) -> None:
    metadata = {"extensions": ["./src/index.ts"], "runtimeExtensions": ["./dist/missing.js"]}
    inventory = _inventory(_package(tmp_path, metadata, {"src/index.ts": "x", "dist/index.js": "x"}))
    broken = [member for member in inventory.selected_components if member.type == "extension" and member.problem]
    assert [member.problem for member in broken] == ["missing"]


@pytest.mark.parametrize("source", ["../outside.ts", "/tmp/outside.ts", 7, ""])
def test_runtime_never_bypasses_source_path_policy(tmp_path: Path, source: Any) -> None:
    metadata = {"extensions": [source], "runtimeExtensions": ["./dist/index.js"]}
    inventory = _inventory(_package(tmp_path, metadata, {"dist/index.js": "x"}))
    assert any(finding.severity == Severity.HIGH for finding in inventory.findings)
    assert not inventory.complete


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("entrypoints", "package.json#openclaw must be an object"),
        ({"extensions": "./src/index.ts"}, "openclaw.extensions must be an array"),
        ({"setupEntry": ["./src/setup.ts"]}, "openclaw.setupEntry must be a string"),
        (
            {"extensions": ["./a.ts", "./b.ts"], "runtimeExtensions": ["./dist/a.js"]},
            "openclaw.runtimeExtensions length must match openclaw.extensions length",
        ),
        ({"runtimeSetupEntry": "./dist/setup.js"}, "openclaw.runtimeSetupEntry requires openclaw.setupEntry"),
    ],
)
def test_runtime_metadata_shape_and_pairing_are_not_relaxed(tmp_path: Path, metadata: Any, message: str) -> None:
    files = dict.fromkeys(("a.ts", "b.ts", "dist/a.js", "dist/setup.js", "src/index.ts", "src/setup.ts"), "x")
    inventory = _inventory(_package(tmp_path, metadata, files))
    invalid = [finding for finding in inventory.findings if finding.check_name == "plugin_openclaw_package_invalid"]
    assert [finding.message for finding in invalid] == [message]
    assert all(finding.severity == Severity.HIGH for finding in invalid)


def test_runtime_entries_stay_paired_by_index(tmp_path: Path) -> None:
    metadata = {"extensions": ["./src/a.ts", "./src/b.ts"], "runtimeExtensions": ["./dist/a.js", "./dist/b.js"]}
    inventory = _inventory(_package(tmp_path, metadata, {"dist/a.js": "x", "src/b.ts": "x"}))
    # b's explicit runtime is missing; its existing source does not excuse that.
    broken = [member for member in inventory.selected_components if member.type == "extension" and member.problem]
    assert [(member.name, member.problem) for member in broken] == [("./dist/b.js", "missing")]


def test_empty_runtime_array_allows_upstream_inference(tmp_path: Path) -> None:
    metadata = {"extensions": ["./src/index.ts"], "runtimeExtensions": []}
    inventory = _inventory(_package(tmp_path, metadata, {"dist/index.js": "x"}))
    assert "openclaw.runtimeExtensions (inferred): dist/index.js" in _runtime_rows(inventory)
    assert inventory.complete


@pytest.mark.parametrize("skills", ["./skills", None, {}, "skills"])
def test_non_array_skills_declaration_reads_no_root(tmp_path: Path, skills: Any) -> None:
    manifest = {"id": "release-helper", "configSchema": {"type": "object"}, "skills": skills}
    _tree(tmp_path, {MANIFEST: manifest, "skills/a/SKILL.md": _skill("a")})
    inventory = _inventory(tmp_path)
    assert _members(inventory) == []
    [broken] = [member for member in inventory.selected_components if member.type == "skill"]
    assert broken.problem == "invalid"
    assert "plugin_component_path_invalid" in _checks(inventory.findings)
    assert not inventory.complete
    result = validate_inventory_schema(inventory)
    assert any(finding.check_name.startswith("schema:skills:") for finding in result.findings)
