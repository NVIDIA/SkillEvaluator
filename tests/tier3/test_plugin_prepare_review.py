# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real source-tree and task-input regressions for private plugin preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.tier3.harbor.adapter import _stage_task_inputs, generate_harbor_tasks
from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.plugin_native import HARNESS_ADAPTERS, ClaudeCodeAdapter
from skillevaluator.tier3.plugin_prepare import (
    PreparationError,
    SourceIdentity,
    prepare_runtime_package,
    stage_wrapper_skill,
    verify_source_identity,
)


def _tree(root: Path, files: dict) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _skill(name: str) -> str:
    return f"---\nname: {name}\ndescription: {name} skill\n---\nSkill body.\n"


def _inventory(root: Path):
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    return inventory


def _package(root: Path):
    return prepare_runtime_package(_inventory(root))


@pytest.mark.parametrize("layout", ["evals/evals.json", "eval/dataset.json"])
def test_relative_root_never_copies_member_evaluation_material(tmp_path: Path, monkeypatch, layout: str) -> None:
    _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "skills/a/SKILL.md": _skill("a"),
            f"skills/a/{layout}": [{"prompt": "task", "expected_output": "SECRET-ANSWER"}],
        },
    )
    monkeypatch.chdir(tmp_path)
    wrapper = stage_wrapper_skill(_package(Path("plugin")), tmp_path / "out")
    assert sorted(
        p.relative_to(wrapper.path / "skills").as_posix() for p in (wrapper.path / "skills").rglob("*") if p.is_file()
    ) == ["a/SKILL.md"]
    tasks = generate_harbor_tasks(wrapper.path, tmp_path / "tasks")
    assert len(tasks) == 1
    visible = tasks[0] / "environment/skills"
    assert not any("SECRET-ANSWER" in p.read_text() for p in visible.rglob("*") if p.is_file())


def _inputs(wrapper, entry: dict, destination: Path) -> dict[str, bytes]:
    _stage_task_inputs(
        destination,
        input_files_dir=wrapper.path / "evals/files",
        entry=entry,
        source_skill_path=wrapper.path,
        evals_dir=wrapper.path / "evals",
    )
    return {
        p.relative_to(destination / "input").as_posix(): p.read_bytes()
        for p in (destination / "input").rglob("*")
        if p.is_file()
    }


def test_combined_cases_keep_member_fixture_scope_and_input_paths(tmp_path: Path) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"id": "one", "prompt": "a", "files": ["data/report.csv"]}],
            "skills/a/evals/data/report.csv": "a",
            "skills/b/SKILL.md": _skill("b"),
            "skills/b/evals/evals.json": [{"id": "two", "prompt": "b"}],
            "skills/b/evals/files/b.txt": "b",
            "skills/c/SKILL.md": _skill("c"),
            "skills/c/evals/evals.json": [{"id": "three", "prompt": "c", "files": ["files"]}],
            "skills/c/evals/files/c.txt": "c",
            "skills/d/SKILL.md": _skill("d"),
            "skills/d/evals/evals.json": [{"id": "four", "prompt": "d"}],
        },
    )
    wrapper = stage_wrapper_skill(_package(root), tmp_path / "out")
    entries = json.loads((wrapper.path / "evals/evals.json").read_text())
    expected = {"a-one": {"data/report.csv": b"a"}, "b-two": {"b.txt": b"b"}, "c-three": {"c.txt": b"c"}, "d-four": {}}
    assert {entry["id"]: _inputs(wrapper, entry, tmp_path / entry["id"]) for entry in entries} == expected
    tasks = generate_harbor_tasks(wrapper.path, tmp_path / "tasks")
    assert len(tasks) == 4
    assert sorted(
        sorted(
            p.relative_to(task / "environment/input").as_posix()
            for p in (task / "environment/input").rglob("*")
            if p.is_file()
        )
        for task in tasks
    ) == [[], ["b.txt"], ["c.txt"], ["data/report.csv"]]


def test_explicit_dataset_preserves_non_files_fixture_paths(tmp_path: Path) -> None:
    root = _tree(
        tmp_path / "plugin", {".claude-plugin/plugin.json": {"name": "demo"}, "skills/a/SKILL.md": _skill("a")}
    )
    source = _tree(
        tmp_path / "data", {"evals.json": [{"prompt": "task", "files": ["data/input.txt"]}], "data/input.txt": "input"}
    )
    wrapper = stage_wrapper_skill(_package(root), tmp_path / "out", evals_source=source)
    [entry] = json.loads((wrapper.path / "evals/evals.json").read_text())
    assert _inputs(wrapper, entry, tmp_path / "task") == {"data/input.txt": b"input"}


@pytest.mark.parametrize("ref", ["missing.txt", "../outside.txt", "evals.json", "results/secret.txt"])
@pytest.mark.parametrize("explicit", [False, True])
def test_invalid_fixture_reference_is_refused_before_task_generation(tmp_path: Path, ref: str, explicit: bool) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"prompt": "task", "files": [ref]}],
            "skills/a/evals/results/secret.txt": "secret",
        },
    )
    source = root / "skills/a/evals" if explicit else None
    with pytest.raises(PreparationError, match=r"(fixture|input)"):
        stage_wrapper_skill(_package(root), tmp_path / "out", evals_source=source)
    assert not (tmp_path / "out").exists()


def test_prepared_mcp_env_and_headers_reach_real_native_bundle(tmp_path: Path) -> None:
    servers = {
        "stdio": {"command": "npx", "args": ["example-server"], "env": {"API_BASE": "${API_BASE}", "MODE": "ci"}},
        "http": {"url": "https://example.invalid/mcp", "headers": {"Authorization": "Bearer ${MCP_TOKEN}"}},
    }
    root = _tree(
        tmp_path / "plugin", {".claude-plugin/plugin.json": {"name": "demo"}, ".mcp.json": {"mcpServers": servers}}
    )
    source = _package(root).native_source
    staging = build_native_task_staging("claude-code", ClaudeCodeAdapter(), source)
    stage_native_bundle(tmp_path / "environment", staging)
    actual = json.loads((tmp_path / "environment/skilleval/native/claude-code/plugin/.mcp.json").read_text())[
        "mcpServers"
    ]
    assert actual["stdio"]["env"] == servers["stdio"]["env"]
    assert actual["http"]["headers"] == servers["http"]["headers"]


@pytest.mark.parametrize(
    ("manifest_dir", "prefix"),
    [
        (".claude-plugin", "${CLAUDE_PLUGIN_ROOT}"),
        (".codex-plugin", "${PLUGIN_ROOT}"),
        (".cursor-plugin", "${CURSOR_PLUGIN_ROOT}"),
    ],
)
@pytest.mark.parametrize("agent", ["codex", "opencode"])
def test_plugin_file_mcp_is_only_projected_for_a_tree_copying_adapter(
    tmp_path: Path, manifest_dir: str, prefix: str, agent: str
) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            f"{manifest_dir}/plugin.json": {"name": "demo"},
            ("mcp.json" if manifest_dir == ".cursor-plugin" else ".mcp.json"): {
                "mcpServers": {"local": {"command": "node", "args": [prefix + "/server.js"]}}
            },
            "server.js": "// fixture",
        },
    )
    source = _package(root).native_source
    assert source.mcp_servers == ()
    assert source.plugin_file_mcp_servers[0]["args"] == ["${CLAUDE_PLUGIN_ROOT}/server.js"]
    bundle = HARNESS_ADAPTERS[agent].build(source)
    assert "${CLAUDE_PLUGIN_ROOT}" not in "\n".join(bundle.generated.values())
    assert not any(row["type"] == "mcp" for row in bundle.declared)
    assert "launches from plugin files" in bundle.generated["native/setup.sh"]
    claude = ClaudeCodeAdapter().build(source)
    assert "${CLAUDE_PLUGIN_ROOT}/server.js" in claude.generated["native/claude-code/plugin/.mcp.json"]


@pytest.mark.parametrize(
    "changed", ["skills/a/SKILL.md", "hooks/hooks.json", "scripts/tool.py", "skills/a/evals/evals.json"]
)
def test_source_tree_identity_guards_analysis_and_staging(tmp_path: Path, changed: str) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "skills/a/SKILL.md": _skill("a"),
            "hooks/hooks.json": {"hooks": {}},
            "scripts/tool.py": "pass\n",
            "skills/a/evals/evals.json": [{"prompt": "task"}],
        },
    )
    inventory = _inventory(root)
    expected = SourceIdentity.of(inventory)
    package = prepare_runtime_package(inventory, expected=expected)
    target = root / changed
    target.write_text(target.read_text() + "\n ")
    with pytest.raises(PreparationError, match="changed after analysis"):
        prepare_runtime_package(inventory, expected=expected)
    with pytest.raises(PreparationError, match="between preparation and staging"):
        verify_source_identity(package, inventory)
    with pytest.raises(PreparationError, match="changed"):
        stage_wrapper_skill(package, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_generated_results_do_not_invalidate_source_tree_identity(tmp_path: Path) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"prompt": "task"}],
        },
    )
    (root / "evals").mkdir()
    inventory = _inventory(root)
    package = prepare_runtime_package(inventory)
    _tree(root, {"evals/results/report.json": {"generated": True}, "BENCHMARK.md": "generated"})
    verify_source_identity(package, inventory)
    stage_wrapper_skill(package, tmp_path / "out")


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS trusted root aliases")
@pytest.mark.parametrize("layout", ["evals/evals.json", "eval/dataset.json"])
def test_macos_unresolved_tmp_alias_excludes_evaluation_material(tmp_path: Path, layout: str) -> None:
    import tempfile

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="plugin-prepare-alias-") as temporary:
        root = _tree(
            Path(temporary) / "plugin",
            {
                ".claude-plugin/plugin.json": {"name": "demo"},
                "skills/a/SKILL.md": _skill("a"),
                f"skills/a/{layout}": [{"prompt": "task", "expected_output": "SECRET-ANSWER"}],
            },
        )
        assert root != root.resolve()
        wrapper = stage_wrapper_skill(_package(root), tmp_path / "out")
        assert [p.name for p in (wrapper.path / "skills/a").iterdir()] == ["SKILL.md"]


def test_case_selection_preserves_nested_and_empty_directories_without_other_members(tmp_path: Path) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"prompt": "task", "files": ["data"]}],
            "skills/a/evals/data/nested/input.txt": "input",
            "skills/b/SKILL.md": _skill("b"),
            "skills/b/evals/evals.json": [{"prompt": "task", "files": []}],
            "skills/b/evals/files/hidden.txt": "not selected",
        },
    )
    (root / "skills/a/evals/data/empty").mkdir()
    wrapper = stage_wrapper_skill(_package(root), tmp_path / "out")
    entries = json.loads((wrapper.path / "evals/evals.json").read_text())
    assert _inputs(wrapper, entries[0], tmp_path / "a") == {"data/nested/input.txt": b"input"}
    assert (tmp_path / "a/input/data/empty").is_dir()
    assert _inputs(wrapper, entries[1], tmp_path / "b") == {}
    assert not (wrapper.path / "evals/files/hidden.txt").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_explicit_dataset_refuses_linked_fixture_paths(tmp_path: Path, kind: str) -> None:
    root = _tree(
        tmp_path / "plugin", {".claude-plugin/plugin.json": {"name": "demo"}, "skills/a/SKILL.md": _skill("a")}
    )
    source = _tree(tmp_path / "source", {"evals.json": [{"prompt": "task", "files": ["data"]}]})
    outside = _tree(tmp_path / "outside", {"input.txt": "outside"})
    (source / "data").symlink_to(outside if kind == "directory" else outside / "input.txt")
    with pytest.raises(PreparationError, match="input fixtures"):
        stage_wrapper_skill(_package(root), tmp_path / "out", evals_source=source)
    assert not (tmp_path / "out").exists()


def test_source_mutation_during_wrapper_copy_refuses_publication(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.tier3.harbor import secure_copy

    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "skills/a/SKILL.md": _skill("a"),
            "skills/a/evals/evals.json": [{"prompt": "task"}],
        },
    )
    package = _package(root)
    copy = secure_copy.copytree_secure

    def mutate_after_copy(source, destination, **kwargs):
        result = copy(source, destination, **kwargs)
        (root / "skills/a/SKILL.md").write_text(_skill("changed"))
        return result

    monkeypatch.setattr(secure_copy, "copytree_secure", mutate_after_copy)
    with pytest.raises(PreparationError, match="between preparation and staging"):
        stage_wrapper_skill(package, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))


@pytest.mark.parametrize("launch", ["./server.js", "--file=./server.js", "server.js", "--file=server.js"])
def test_plugin_relative_mcp_launches_are_anchored_to_copied_tree(tmp_path: Path, launch: str) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            ".mcp.json": {"mcpServers": {"local": {"command": "node", "args": [launch]}}},
            "server.js": "// fixture",
        },
    )
    package = _package(root)
    assert package.native_source.mcp_servers == ()
    expected = ("--file=" if launch.startswith("--") else "") + "${CLAUDE_PLUGIN_ROOT}/server.js"
    assert package.native_source.plugin_file_mcp_servers[0]["args"] == [expected]
    assert any(row.name == "local" and "requires plugin tree" in row.reason for row in package.audit)


def test_mcp_cwd_is_retained_as_unsupported_instead_of_launching_in_task_cwd(tmp_path: Path) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".codex-plugin/plugin.json": {"name": "demo"},
            ".mcp.json": {"mcpServers": {"local": {"command": "node", "args": ["server.js"], "cwd": "./scripts"}}},
            "scripts/server.js": "// fixture",
        },
    )
    package = _package(root)
    assert package.mcp_servers[0]["cwd"] == "./scripts"
    assert package.native_source.mcp_servers == package.native_source.plugin_file_mcp_servers == ()
    assert ("mcp", "local") in package.native_source.other
    assert any(row.name == "local" and "cwd" in row.reason for row in package.audit)
    for adapter in HARNESS_ADAPTERS.values():
        bundle = adapter.build(package.native_source)
        assert not any(row["type"] == "mcp" for row in bundle.declared)


@pytest.mark.parametrize("agent", ["codex", "opencode"])
def test_actual_setup_reports_plugin_file_mcp_not_loaded(tmp_path: Path, agent: str) -> None:
    import os
    import subprocess

    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            ".mcp.json": {"mcpServers": {"local": {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/server.js"]}}},
            "server.js": "// fixture; never launched",
        },
    )
    source = _package(root).native_source
    staging = build_native_task_staging(agent, HARNESS_ADAPTERS[agent], source)
    environment = tmp_path / "environment"
    stage_native_bundle(environment, staging)
    bundle = environment / "skilleval"
    logs, home = tmp_path / "logs", tmp_path / "home"
    home.mkdir()
    script = (
        (bundle / "native/setup.sh")
        .read_text()
        .replace("/skilleval/", str(bundle) + "/")
        .replace("/logs/agent", str(logs))
    )
    completed = subprocess.run(
        ["sh", "-c", script],
        env={"PATH": os.environ["PATH"], "HOME": str(home), "CODEX_HOME": str(home / ".codex")},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    census = json.loads((logs / "skilleval-load-census.json").read_text())
    assert not any(row["type"] == "mcp" for row in census["listed"])
    assert any(row["type"] == "mcp" and row["name"] == "local" for row in census["not_loaded"])


def test_prepared_mcp_literal_secret_still_refuses_native_projection(tmp_path: Path) -> None:
    root = _tree(
        tmp_path / "plugin",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            ".mcp.json": {
                "mcpServers": {
                    "remote": {
                        "url": "https://example.invalid/mcp",
                        "headers": {"Authorization": "Bearer sk-abcdefghijklmnopqrstuvwxyz1234567890"},
                    }
                }
            },
        },
    )
    source = _package(root).native_source
    with pytest.raises(ValueError, match="literal secret"):
        ClaudeCodeAdapter().build(source)


def test_oversized_explicit_fixture_is_refused_before_engine_copy(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES
    from skillevaluator.tier3.harbor import adapter

    root = _tree(
        tmp_path / "plugin", {".claude-plugin/plugin.json": {"name": "demo"}, "skills/a/SKILL.md": _skill("a")}
    )
    source = _tree(tmp_path / "source", {"evals.json": [{"prompt": "task", "files": ["large.bin"]}]})
    with (source / "large.bin").open("wb") as output:
        output.truncate(CONTENT_DEDUP_MAX_FILE_BYTES + 1)

    def unexpected_copy(*args, **kwargs):
        pytest.fail("oversized fixture must be refused before the engine copies it")

    monkeypatch.setattr(adapter, "_stage_task_inputs", unexpected_copy)
    with pytest.raises(PreparationError, match="input fixtures"):
        stage_wrapper_skill(_package(root), tmp_path / "out", evals_source=source)
    assert not (tmp_path / "out").exists()
