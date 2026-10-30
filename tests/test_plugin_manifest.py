# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code marketplace lookup and dependency states."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from skillevaluator.plugin_dependencies import CAUSE_ERROR, CAUSE_LINK, CAUSE_UNSAFE
from skillevaluator.plugin_manifest import classify_claude_dependency, find_claude_marketplace


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
