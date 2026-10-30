# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR 243 findings 4/5: published entrypoints and foreign Claude skill paths.

Offline fixtures only: no native JavaScript, harness, Docker, or model runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.tier3.plugin_native import ClaudeCodeAdapter, NativePluginSource
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_SKILL = """---
name: alpha
description: Demonstrate a portable skill.
metadata:
  author: Example Author <author@example.com>
---
# Instructions
Use alpha.
"""


def _write(root: Path, rel: str, content: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _plugin(root: Path, metadata: dict | None = None, *, skill_root: str = "skills") -> Path:
    _write(root, "openclaw.plugin.json", json.dumps({"id": "demo", "configSchema": {}, "skills": [skill_root]}))
    _write(root, "skills/alpha/SKILL.md", _SKILL)
    _write(root, "evals/evals.json", '[{"id":"one","prompt":"Use alpha","expected_output":"ok"}]')
    if metadata is not None:
        _write(root, "package.json", json.dumps({"openclaw": metadata}))
    return root


def _errors(result) -> list:
    return [finding for finding in result.findings if finding.severity.is_error()]


def _runtime_rows(result) -> list[dict]:
    return [row for row in result.metadata["plugin"]["component_inventory"]["components"] if row["type"] == "extension"]


@pytest.mark.parametrize("with_setup", [False, True])
def test_published_paired_runtime_does_not_require_typescript_source(tmp_path: Path, with_setup: bool) -> None:
    metadata = {"extensions": ["./index.ts"], "runtimeExtensions": ["./dist/index.js"]}
    if with_setup:
        metadata.update({"setupEntry": "./setup-entry.ts", "runtimeSetupEntry": "./dist/setup-entry.js"})
    root = _plugin(tmp_path, metadata)
    _write(root, "dist/index.js", "throw new Error('must not execute');")
    if with_setup:
        _write(root, "dist/setup-entry.js", "throw new Error('must not execute');")
    result = PluginSchemaValidator().validate(root)
    assert result.passed, [(f.check_name, f.message) for f in _errors(result)]
    rows = _runtime_rows(result)
    expected = {"openclaw.plugin.json", "index.ts", "dist/index.js"}
    if with_setup:
        expected.update({"setup-entry.ts", "dist/setup-entry.js"})
    assert {row["path"] for row in rows} == expected
    assert all(row["support"] == "unsupported" and "problem" not in row for row in rows)
    assert not (root / "index.ts").exists()


@pytest.mark.parametrize("field", ["extensions", "setupEntry"])
@pytest.mark.parametrize(
    "source,runtime",
    [
        ("./index.ts", "dist/index.js"),
        ("./src/nested/index.tsx", "dist/nested/index.mjs"),
        ("src/nested/index.mts", "dist/src/nested/index.mjs"),
        ("src/nested/index.cts", "dist/nested/index.cjs"),
        ("./index.TS", "index.js"),
    ],
)
def test_published_inferred_runtime_does_not_require_source(
    tmp_path: Path, field: str, source: str, runtime: str
) -> None:
    root = _plugin(tmp_path, {field: [source] if field == "extensions" else source})
    _write(root, runtime, "throw new Error('must not execute');")
    result = PluginSchemaValidator().validate(root)
    assert result.passed, [(f.check_name, f.message) for f in _errors(result)]
    rows = _runtime_rows(result)
    assert {row["path"] for row in rows} == {"openclaw.plugin.json", source.removeprefix("./"), runtime}
    assert all(row["support"] == "unsupported" and "problem" not in row for row in rows)


@pytest.mark.parametrize("field", ["extensions", "setupEntry"])
@pytest.mark.parametrize("paired", [False, True])
def test_missing_source_without_valid_runtime_is_still_blocking(tmp_path: Path, field: str, paired: bool) -> None:
    runtime_field = "runtimeExtensions" if field == "extensions" else "runtimeSetupEntry"
    metadata = {field: ["index.ts"] if field == "extensions" else "index.ts"}
    if paired:
        metadata[runtime_field] = ["dist/index.js"] if field == "extensions" else "dist/index.js"
    result = PluginSchemaValidator().validate(_plugin(tmp_path, metadata))
    assert not result.passed
    assert any(f.check_name == "plugin_component_path_missing" for f in _errors(result))


@pytest.mark.parametrize("source", ["index.ts", "index.js"])
def test_existing_local_source_remains_supported_without_runtime(tmp_path: Path, source: str) -> None:
    root = _plugin(tmp_path, {"extensions": [source]})
    _write(root, source, "throw new Error('must not execute');")
    result = PluginSchemaValidator().validate(root)
    assert result.passed
    assert any(row["path"] == source and "problem" not in row for row in _runtime_rows(result))


@pytest.mark.parametrize("source", ["../outside.ts", "/tmp/outside.ts", "x" * 4097, 7, ""])
def test_paired_runtime_never_bypasses_source_path_policy(tmp_path: Path, source) -> None:
    root = _plugin(tmp_path, {"extensions": [source], "runtimeExtensions": ["dist/index.js"]})
    _write(root, "dist/index.js", "// runtime")
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    assert any(
        f.check_name in {"plugin_component_path_escape", "plugin_component_path_invalid"} for f in _errors(result)
    )


@pytest.mark.parametrize("target", ["source", "explicit_runtime", "inferred_runtime"])
@pytest.mark.parametrize("kind", ["link", "parent_link", "hardlink", "fifo", "directory"])
def test_runtime_fallback_never_hides_unsafe_paths(tmp_path: Path, target: str, kind: str) -> None:
    metadata = {"extensions": ["src/index.ts"]}
    if target != "inferred_runtime":
        metadata["runtimeExtensions"] = ["dist/index.js"]
    root = _plugin(tmp_path / "plugin", metadata)
    path = root / ("src/index.ts" if target == "source" else "dist/index.js")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = _write(outside, path.name, "// outside")
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "parent_link":
        path.parent.rmdir()
        path.parent.symlink_to(outside, target_is_directory=True)
    elif kind == "link":
        path.symlink_to(outside_file)
    elif kind == "hardlink":
        os.link(outside_file, path)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    if target == "source":
        _write(root, "dist/index.js", "// valid runtime does not excuse unsafe source")
    # A later valid inferred candidate must not excuse an earlier unsafe one.
    if target == "inferred_runtime":
        _write(root, "src/index.js", "// later candidate")
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    check = "plugin_component_path_invalid" if kind == "directory" else "plugin_component_path_unsafe"
    assert any(f.check_name in {check, "unsafe_plugin_filesystem"} for f in _errors(result))


@pytest.mark.parametrize("runtime", ["../outside.js", "/tmp/outside.js", "x" * 4097, 7, ""])
def test_explicit_runtime_must_be_valid_even_when_inferred_output_exists(tmp_path: Path, runtime) -> None:
    root = _plugin(tmp_path, {"extensions": ["index.ts"], "runtimeExtensions": [runtime]})
    _write(root, "dist/index.js", "// do not substitute for an explicit invalid runtime")
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    assert any(
        f.check_name in {"plugin_component_path_escape", "plugin_component_path_invalid"} for f in _errors(result)
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"extensions": "index.ts"},
        {"extensions": ["index.ts"], "runtimeExtensions": "dist/index.js"},
        {"setupEntry": ["index.ts"]},
        {"setupEntry": "index.ts", "runtimeSetupEntry": ["dist/index.js"]},
        {"extensions": ["index.ts", "other.ts"], "runtimeExtensions": ["dist/index.js"]},
        {"runtimeSetupEntry": "dist/index.js"},
    ],
)
def test_runtime_metadata_shape_and_pairing_are_not_silently_relaxed(tmp_path: Path, metadata: dict) -> None:
    root = _plugin(tmp_path, metadata)
    for rel in ("index.ts", "other.ts", "dist/index.js"):
        _write(root, rel, "// existing files cannot excuse invalid declarations")
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    assert any(f.check_name == "plugin_openclaw_package_invalid" for f in _errors(result))


@pytest.mark.parametrize("plugin_load", ["native", "auto"])
@pytest.mark.parametrize("skill_root", ["skills", "skills/alpha", "./skills", "./skills/alpha"])
def test_openclaw_skills_stage_as_valid_claude_paths(tmp_path: Path, plugin_load: str, skill_root: str) -> None:
    root = _plugin(tmp_path / "plugin", skill_root=skill_root)
    package = prepare_plugin_eval_package(
        root, stage_root=tmp_path / "stage", plugin_load=plugin_load, agents="claude-code", env_mode="docker"
    )
    assert not package.skipped
    source = package.native_source
    assert source is not None
    adapter = ClaudeCodeAdapter()
    staging = build_native_task_staging("claude-code", adapter, source)
    environment = tmp_path / "task/environment"
    environment.mkdir(parents=True)
    stage_native_bundle(environment, staging)
    staged_root = environment / "skilleval/native/claude-code/plugin"
    staged_manifest = json.loads((staged_root / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    assert staged_manifest["skills"] == ["./" + skill_root.removeprefix("./")]
    assert (staged_root / "skills/alpha/SKILL.md").read_text(encoding="utf-8") == _SKILL
    assert source.manifest["skills"] == [skill_root]
    assert adapter.staged_skills(source) == [("alpha", "skills/alpha", None)]
    result = PluginSchemaValidator().validate(staged_root)
    assert result.passed, [(f.check_name, f.message) for f in _errors(result)]


@pytest.mark.parametrize(
    "manifest_rel",
    [".claude-plugin/plugin.json", ".cursor-plugin/plugin.json", ".codex-plugin/plugin.json", "plugin.json"],
)
@pytest.mark.parametrize("skills", ["./portable/alpha", ["./portable/alpha", "./other"]])
def test_existing_format_skill_declarations_are_preserved(tmp_path: Path, manifest_rel: str, skills) -> None:
    source = NativePluginSource(
        plugin_name="demo",
        description="Demo",
        contained=True,
        plugin_root=tmp_path,
        manifest={"skills": skills},
        manifest_rel=manifest_rel,
    )
    assert ClaudeCodeAdapter().manifest(source)["skills"] == skills


def test_foreign_scalar_skill_path_is_normalized_without_rewriting_other_fields(tmp_path: Path) -> None:
    source = NativePluginSource(
        plugin_name="demo",
        description="Demo",
        contained=True,
        plugin_root=tmp_path,
        manifest={"skills": "portable/alpha", "commands": ["./commands/alpha.md"], "version": "1.0.0"},
        manifest_rel=".cursor-plugin/plugin.json",
    )
    manifest = ClaudeCodeAdapter().manifest(source)
    assert manifest["skills"] == "./portable/alpha"
    assert manifest["commands"] == source.manifest["commands"]
    assert manifest["version"] == "1.0.0"


def test_runtime_entries_stay_paired_by_index(tmp_path: Path) -> None:
    root = _plugin(
        tmp_path,
        {"extensions": ["one.ts", "two.ts"], "runtimeExtensions": ["dist/one.js", "dist/two.js"]},
    )
    _write(root, "dist/one.js", "// only the first pair exists")
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    rows = _runtime_rows(result)
    assert any(row["path"] == "one.ts" and "problem" not in row for row in rows)
    assert any(row["path"] == "two.ts" and row.get("problem") == "missing" for row in rows)
    assert any(row["path"] == "dist/two.js" and row.get("problem") == "missing" for row in rows)


def test_empty_runtime_array_allows_upstream_inference(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"extensions": ["index.ts"], "runtimeExtensions": []})
    _write(root, "dist/index.js", "// inferred")
    assert PluginSchemaValidator().validate(root).passed


def test_missing_non_typescript_source_has_no_inferred_dist_fallback(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"extensions": ["index.js"]})
    _write(root, "dist/index.js", "// not an upstream candidate for index.js")
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    assert any(f.check_name == "plugin_component_path_missing" for f in _errors(result))


def test_existing_source_does_not_excuse_missing_explicit_runtime(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"extensions": ["index.ts"], "runtimeExtensions": ["dist/missing.js"]})
    _write(root, "index.ts", "// local source")
    _write(root, "dist/index.js", "// inferred output is not the explicit runtime")
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    assert any(row["path"] == "dist/missing.js" and row.get("problem") == "missing" for row in _runtime_rows(result))


@pytest.mark.parametrize("manifest_rel", [".claude-plugin/plugin.json", "openclaw.plugin.json"])
@pytest.mark.parametrize("skills", [["../escape"], ["/tmp/absolute"], [7], []])
def test_path_conversion_does_not_repair_or_drop_invalid_declarations(
    tmp_path: Path, manifest_rel: str, skills: list
) -> None:
    source = NativePluginSource(
        plugin_name="demo",
        description="Demo",
        contained=True,
        plugin_root=tmp_path,
        manifest={"skills": skills},
        manifest_rel=manifest_rel,
    )
    assert ClaudeCodeAdapter().manifest(source)["skills"] == skills


def test_native_claude_declaration_is_not_translated_as_foreign(tmp_path: Path) -> None:
    source = NativePluginSource(
        plugin_name="demo",
        description="Demo",
        contained=True,
        plugin_root=tmp_path,
        manifest={"skills": ["undotted"]},
        manifest_rel=".claude-plugin/plugin.json",
    )
    assert ClaudeCodeAdapter().manifest(source)["skills"] == ["undotted"]
