# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrapper skill and bounded dataset preparation for a private plugin evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.tier3.plugin_prepare import PreparationError, prepare_runtime_package, stage_wrapper_skill


def _skill(name: str) -> str:
    return f"---\nname: {name}\ndescription: {name} skill.\n---\nBody.\n"


def _tree(root: Path, files: dict) -> Path:
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return root


def _package(root: Path):
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    return prepare_runtime_package(inventory)


def _plugin(tmp_path: Path, files: dict) -> Path:
    return _tree(tmp_path / "plugin", {".claude-plugin/plugin.json": {"name": "Demo Plugin"}, **files})


def _entries(wrapper) -> list[dict]:
    return json.loads((wrapper.path / "evals" / "evals.json").read_text(encoding="utf-8"))


def test_member_datasets_combine_with_deterministic_unique_ids(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path,
        {
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"id": "x", "prompt": "p1"}, {"prompt": "p2"}],
            "skills/b/SKILL.md": _skill("b"),
            "skills/b/evals/evals.json": [{"id": "x", "prompt": "p3"}],
        },
    )
    first = stage_wrapper_skill(_package(root), tmp_path / "out1")
    second = stage_wrapper_skill(_package(root), tmp_path / "out2")
    ids = [entry["id"] for entry in _entries(first)]
    assert ids == ["a-x", "a-case-002", "b-x"] == [entry["id"] for entry in _entries(second)]
    assert first.name == second.name == "plugin-demo-plugin"
    assert {entry["plugin_eval_source_skill"] for entry in _entries(first)} == {"a", "b"}
    assert first.case_sources == {
        "a-x": "skills/a/evals/evals.json",
        "a-case-002": "skills/a/evals/evals.json",
        "b-x": "skills/b/evals/evals.json",
    }


def test_explicit_eval_source_wins_and_is_never_agent_visible(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path,
        {
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"prompt": "m"}],
            "evals/answers.txt": "secret",
        },
    )
    source = _tree(tmp_path / "ds", {"evals.json": [{"id": "only", "prompt": "q"}], "files/in put.txt": "x"})
    wrapper = stage_wrapper_skill(_package(root), tmp_path / "out", evals_source=source)
    assert [entry["id"] for entry in _entries(wrapper)] == ["only"]
    assert (wrapper.path / "evals" / "files" / "in put.txt").read_text() == "x"
    visible = [p for p in (wrapper.path / "skills").rglob("*") if p.is_file()]
    assert visible and not any("evals" in p.relative_to(wrapper.path).parts[2:] for p in visible)


def test_colliding_fixtures_from_two_members_are_refused_and_rolled_back(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path,
        {
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"prompt": "1"}],
            "skills/a/evals/files/data.txt": "one",
            "skills/b/SKILL.md": _skill("b"),
            "skills/b/evals/evals.json": [{"prompt": "2"}],
            "skills/b/evals/files/data.txt": "two",
        },
    )
    out = tmp_path / "out"
    with pytest.raises(PreparationError, match="fixture collision"):
        stage_wrapper_skill(_package(root), out)
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out")]


def test_no_dataset_is_refused(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"skills/a/SKILL.md": _skill("a")})
    with pytest.raises(PreparationError, match="no evaluation dataset"):
        stage_wrapper_skill(_package(root), tmp_path / "out")


def test_malformed_and_oversized_datasets_are_refused(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"skills/a/SKILL.md": _skill("a"), "skills/a/evals/evals.json": "{not json"})
    with pytest.raises(PreparationError, match="dataset"):
        stage_wrapper_skill(_package(root), tmp_path / "out")
    many = _tree(tmp_path / "many", {"evals.json": [{"prompt": str(i)} for i in range(5)]})
    with pytest.raises(PreparationError, match="limit"):
        stage_wrapper_skill(_package(root), tmp_path / "out2", evals_source=many, max_cases=4)


def test_linked_member_dataset_is_refused(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"skills/a/SKILL.md": _skill("a")})
    outside = _tree(tmp_path / "outside", {"evals.json": [{"prompt": "x"}]})
    (root / "skills/a/evals").mkdir()
    (root / "skills/a/evals/evals.json").symlink_to(outside / "evals.json")
    with pytest.raises(PreparationError, match="linked"):
        stage_wrapper_skill(_package(root), tmp_path / "out")


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"skills/a/SKILL.md": _skill("a"), "skills/a/evals/evals.json": [{"prompt": "1"}]})
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(PreparationError, match="exists"):
        stage_wrapper_skill(_package(root), out)
