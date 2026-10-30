# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin manifest lookup, Claude Code marketplace lookup, and dependency states."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from skillevaluator.constants import CONTENT_DEDUP_MAX_DISCOVERED_PATHS
from skillevaluator.plugin_dependencies import CAUSE_ERROR, CAUSE_LINK, CAUSE_UNSAFE
from skillevaluator.plugin_manifest import (
    PluginManifestPathError,
    classify_claude_dependency,
    find_claude_marketplace,
    locate_plugin_manifest,
)

requires_symlinks = pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")


def _claude_plugin(root: Path) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "demo"}), encoding="utf-8")
    (root / "CLAUDE.md").write_text("# Demo\n", encoding="utf-8")
    return root


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("symlinks are unavailable")


@requires_symlinks
@pytest.mark.parametrize(
    ("link", "target"),
    [("AGENTS.md", "CLAUDE.md"), ("LICENSE", "../LICENSE"), ("docs/alias.md", "../CLAUDE.md")],
)
def test_unrelated_link_does_not_fail_the_manifest_lookup(tmp_path: Path, link: str, target: str) -> None:
    root = _claude_plugin(tmp_path / "plugin")
    _symlink(root / link, Path(target))

    located = locate_plugin_manifest(root)

    assert located is not None
    assert located.manifest_filename == ".claude-plugin/plugin.json"


def test_wide_folder_does_not_fail_the_manifest_lookup(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "plugin")
    (root / "assets").mkdir()
    for index in range(CONTENT_DEDUP_MAX_DISCOVERED_PATHS + 4):
        (root / "assets" / f"icon-{index}.svg").touch()

    located = locate_plugin_manifest(root)

    assert located is not None
    assert located.manifest_filename == ".claude-plugin/plugin.json"


@requires_symlinks
def test_linked_manifest_or_vendor_folder_still_fails_closed(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "plugin")
    (root / "real.json").write_text(json.dumps({"name": "demo"}), encoding="utf-8")
    _symlink(root / ".codex-plugin" / "plugin.json", Path("../real.json"))
    with pytest.raises(PluginManifestPathError, match="symlink"):
        locate_plugin_manifest(root)

    other = _claude_plugin(tmp_path / "other")
    (tmp_path / "vendor").mkdir()
    _symlink(other / ".Cursor-Plugin", tmp_path / "vendor", directory=True)
    with pytest.raises(PluginManifestPathError, match="link"):
        locate_plugin_manifest(other)


def test_manifest_name_that_is_a_folder_or_hard_link_fails_closed(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "plugin")
    (root / "agent_plugin.yaml").mkdir()
    with pytest.raises(PluginManifestPathError, match="not a regular file"):
        locate_plugin_manifest(root)

    other = _claude_plugin(tmp_path / "other")
    try:
        (tmp_path / "copy.json").hardlink_to(other / ".claude-plugin" / "plugin.json")
    except OSError:
        pytest.skip("hardlinks are unavailable")
    with pytest.raises(PluginManifestPathError, match="hard-linked"):
        locate_plugin_manifest(other)


def _marketplace(root: Path, plugins: list[dict], **extra) -> Path:
    (root / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    body = {"name": "mk", "plugins": [{"name": "a", "source": "./plugins/a"}, *plugins], **extra}
    (root / ".claude-plugin" / "marketplace.json").write_text(json.dumps(body), encoding="utf-8")
    plugin = root / "plugins" / "a"
    plugin.mkdir(parents=True, exist_ok=True)
    return plugin


def _classify(root: Path, name: str, marketplace_name: str | None = None):
    plugin = root / "plugins" / "a"
    marketplace = find_claude_marketplace(plugin.resolve(), plugin_name="a", stop=root.resolve())
    return classify_claude_dependency(name, marketplace_name, marketplace)


def test_marketplace_lookup_finds_the_marketplace_that_lists_the_plugin(tmp_path: Path) -> None:
    plugin = _marketplace(
        tmp_path,
        [{"name": "b", "source": "./plugins/b"}, {"name": "remote", "source": {"source": "github"}}],
        allowCrossMarketplaceDependenciesOn=["other", 3],
    )

    marketplace = find_claude_marketplace(plugin.resolve(), plugin_name="a", stop=tmp_path.resolve())

    assert marketplace is not None
    assert marketplace.name == "mk"
    assert marketplace.root == tmp_path.resolve()
    assert marketplace.plugins == {"a": "plugins/a", "b": "plugins/b", "remote": ""}
    assert marketplace.allowed_marketplaces == frozenset({"other"})


def test_marketplace_that_does_not_list_the_plugin_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".claude-plugin").mkdir()
    body = {"name": "mk", "plugins": [{"name": "b", "source": "./plugins/b"}]}
    (tmp_path / ".claude-plugin" / "marketplace.json").write_text(json.dumps(body), encoding="utf-8")
    plugin = tmp_path / "plugins" / "a"
    plugin.mkdir(parents=True)

    assert find_claude_marketplace(plugin.resolve(), plugin_name="a", stop=tmp_path.resolve()) is None
    row = classify_claude_dependency("b", None, None)
    assert (row.state, row.cause) == ("unresolved", None)


def test_listed_dependencies_are_referenced_or_missing(tmp_path: Path) -> None:
    _marketplace(tmp_path, [{"name": "b", "source": "./plugins/b"}, {"name": "gone", "source": "./plugins/gone"}])
    (tmp_path / "plugins" / "b").mkdir()

    referenced = _classify(tmp_path, "b")
    absent_folder = _classify(tmp_path, "gone")
    unlisted = _classify(tmp_path, "unlisted")

    assert (referenced.state, referenced.path) == ("referenced", "plugins/b")
    assert (absent_folder.state, absent_folder.path) == ("missing", "plugins/gone")
    assert "does not exist" in absent_folder.reason
    assert (unlisted.state, unlisted.path) == ("missing", None)


def test_cross_marketplace_dependencies_are_external(tmp_path: Path) -> None:
    _marketplace(tmp_path, [], allowCrossMarketplaceDependenciesOn=["allowed"])

    allowed = _classify(tmp_path, "x", "allowed")
    blocked = _classify(tmp_path, "x", "blocked")

    assert allowed.state == blocked.state == "external"
    assert "which marketplace 'mk' allows" in allowed.reason
    assert "does not list it in allowCrossMarketplaceDependenciesOn" in blocked.reason
    assert classify_claude_dependency("x", "other", None).state == "external"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
@pytest.mark.parametrize("source", ["./plugins/linked", "./via/deep"])
def test_linked_source_folder_is_unresolved_not_missing(tmp_path: Path, source: str) -> None:
    _marketplace(tmp_path, [{"name": "b", "source": source}])
    (tmp_path / "vendor" / "deep").mkdir(parents=True)
    try:
        (tmp_path / "plugins" / "linked").symlink_to(tmp_path / "vendor" / "deep", target_is_directory=True)
        (tmp_path / "via").symlink_to(tmp_path / "vendor", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    row = _classify(tmp_path, "b")

    assert (row.state, row.path, row.cause) == ("unresolved", None, CAUSE_LINK)
    assert "passes through a link" in row.reason


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permission bits")
def test_uninspectable_source_folder_is_unresolved_not_missing(tmp_path: Path) -> None:
    _marketplace(tmp_path, [{"name": "b", "source": "./locked/b"}])
    locked = tmp_path / "locked"
    (locked / "b").mkdir(parents=True)
    locked.chmod(0)
    try:
        row = _classify(tmp_path, "b")
    finally:
        locked.chmod(0o755)

    assert (row.state, row.path, row.cause) == ("unresolved", None, CAUSE_ERROR)
    assert "cannot be inspected" in row.reason


def test_nul_in_a_source_folder_is_unresolved_instead_of_raising(tmp_path: Path) -> None:
    _marketplace(tmp_path, [{"name": "b", "source": "./plugins/b\x00x"}])

    row = _classify(tmp_path, "b")

    assert (row.state, row.path, row.cause) == ("unresolved", None, CAUSE_UNSAFE)
